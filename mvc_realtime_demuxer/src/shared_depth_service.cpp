#ifdef SYLC_NATIVE_RENDERER

#include "shared_depth_service.h"

#include "depth_engine.h"
#include "depth_stabilizer.h"
#include "cut_gate.h"
#include "parallel_chunks.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <future>
#include <limits>
#include <map>

namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kLeaderLease = std::chrono::milliseconds(350);
constexpr int kHistogramBins = 32;
constexpr size_t kMaxWarmIdleServices = 1;
std::atomic<uint64_t> g_service_instance{0};

std::wstring service_key(const std::wstring& model, const std::wstring& ort,
                          int width, int height) {
    return model + L"\x1f" + ort + L"\x1f" + std::to_wstring(width) +
           L"x" + std::to_wstring(height);
}

// Process-wide ownership with two explicit guarantees:
//   * at most one unattached healthy session stays warm;
//   * every evicted/renderer reference is released by `reaper_`, never by a
//     presentation thread that may be holding NativeRenderer::Impl::mtx.
// Active services are never evicted. A failed active service remains visible to
// every attached surface long enough for the UI to observe its error, then is
// removed as soon as the last client detaches.
class ServiceRegistry {
public:
    using Factory = std::function<std::shared_ptr<SharedDepthService>()>;

    ServiceRegistry() : reaper_(&ServiceRegistry::reaper_main, this) {}

    ~ServiceRegistry() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (auto& item : entries_)
                retired_.push_back(std::move(item.second.service));
            entries_.clear();
            stopping_ = true;
        }
        cv_.notify_one();
        if (reaper_.joinable()) reaper_.join();
    }

    std::shared_ptr<SharedDepthService> acquire_attached(
            const std::wstring& key, uint64_t client_id,
            const Factory& factory) {
        bool notify_reaper = false;
        std::shared_ptr<SharedDepthService> service;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = entries_.find(key);
            if (it != entries_.end() && it->second.service->failed() &&
                it->second.service->client_count() == 0) {
                // Never return a poisoned idle key. The old worker/session is
                // released asynchronously while this call installs a clean one.
                retired_.push_back(std::move(it->second.service));
                entries_.erase(it);
                notify_reaper = true;
                it = entries_.end();
            }
            if (it != entries_.end()) {
                it->second.touch = ++touch_;
                service = it->second.service;
            } else {
                service = factory();
                entries_.emplace(key, Entry{service, ++touch_});
            }
            // Atomic with lookup/insertion: detach+trim cannot observe a zero-
            // client gap and evict this service before the caller owns it.
            service->attach(client_id);
        }
        if (notify_reaper) cv_.notify_one();
        return service;
    }

    void client_became_idle(SharedDepthService* service) {
        bool notify_reaper = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (auto& item : entries_) {
                if (item.second.service.get() == service) {
                    item.second.touch = ++touch_;
                    break;
                }
            }
            const size_t before = retired_.size();
            trim_idle_locked();
            notify_reaper = retired_.size() != before;
        }
        if (notify_reaper) cv_.notify_one();
    }

    void defer_release(std::shared_ptr<SharedDepthService>&& service) {
        if (!service) return;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            retired_.push_back(std::move(service));
        }
        cv_.notify_one();
    }

    void stats(size_t& services, size_t& active, size_t& idle) const {
        std::lock_guard<std::mutex> lk(mtx_);
        services = entries_.size();
        active = 0;
        idle = 0;
        for (const auto& item : entries_) {
            if (item.second.service->client_count() > 0) ++active;
            else ++idle;
        }
    }

private:
    struct Entry {
        std::shared_ptr<SharedDepthService> service;
        uint64_t touch = 0;
    };

    void trim_idle_locked() {
        Entry* warmest = nullptr;
        for (auto& item : entries_) {
            Entry& entry = item.second;
            if (entry.service->client_count() == 0 && !entry.service->failed() &&
                (!warmest || entry.touch > warmest->touch)) {
                warmest = &entry;
            }
        }

        size_t healthy_idle_kept = 0;
        for (auto it = entries_.begin(); it != entries_.end();) {
            Entry& entry = it->second;
            const bool idle = entry.service->client_count() == 0;
            const bool keep_warm = idle && !entry.service->failed() &&
                &entry == warmest && healthy_idle_kept < kMaxWarmIdleServices;
            if (keep_warm) ++healthy_idle_kept;
            if (idle && !keep_warm) {
                retired_.push_back(std::move(entry.service));
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void reaper_main() {
        for (;;) {
            std::vector<std::shared_ptr<SharedDepthService>> batch;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [this] {
                    return stopping_ || !retired_.empty();
                });
                if (retired_.empty() && stopping_) break;
                batch.swap(retired_);
            }
            // SharedDepthService::~SharedDepthService() signals and joins the
            // ORT worker here, outside both registry and renderer mutexes.
            batch.clear();
        }
    }

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::map<std::wstring, Entry> entries_;
    std::vector<std::shared_ptr<SharedDepthService>> retired_;
    uint64_t touch_ = 0;
    bool stopping_ = false;
    std::thread reaper_;
};

ServiceRegistry& service_registry() {
    static ServiceRegistry registry;
    return registry;
}

int64_t steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now().time_since_epoch()).count();
}

float clamp01(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

void normalize_da3_confidence(const std::vector<float>& raw,
                              std::vector<float>& normalized) {
    // DA3's expp1 confidence is 1 + exp(logit). The exported Base head
    // measures an excess range around 0..0.12; a 0.01 soft knee preserves
    // its ranking while turning it into a useful, conservative [0..1]
    // fusion weight. Legacy mono-output graphs use an all-ones buffer.
    bool has_signal = false;
    for (float c : raw) {
        if (std::isfinite(c) && std::abs(c - 1.0f) > 1.0e-5f) {
            has_signal = true;
            break;
        }
    }
    if (!has_signal) {
        std::fill(normalized.begin(), normalized.end(), 1.0f);
        return;
    }
    for (size_t i = 0; i < raw.size(); ++i) {
        const float c = raw[i];
        const float excess =
            std::isfinite(c) ? std::max(0.0f, c - 1.0f) : 0.0f;
        normalized[i] = excess / (excess + 0.01f);
    }
}

struct LetterboxCandidate {
    int top = 0;
    int bottom = 0;
    bool valid = false;
};

LetterboxCandidate detect_horizontal_letterbox(
        const std::vector<float>& luma, int width, int height) {
    LetterboxCandidate result;
    if (width < 32 || height < 32 ||
        luma.size() != static_cast<size_t>(width) * height) {
        return result;
    }

    // Encoded video black is normally exact/near-exact after the limited-range
    // YUV conversion. Compression noise is tolerated, but a dark movie row is
    // not accepted unless almost every sampled pixel is black and the row mean
    // is extremely low.
    auto black_row = [&](int y) {
        const float* row = luma.data() + static_cast<size_t>(y) * width;
        double sum = 0.0;
        int black = 0;
        for (int x = 0; x < width; ++x) {
            const float v = row[x];
            sum += v;
            if (v <= 0.055f) ++black;
        }
        const float mean = static_cast<float>(sum / width);
        return mean <= 0.030f &&
               black >= static_cast<int>(std::ceil(width * 0.985));
    };

    const int max_band = static_cast<int>(height * 0.36f);
    int top = 0;
    while (top < max_band && black_row(top)) ++top;
    int bottom = 0;
    while (bottom < max_band && black_row(height - 1 - bottom)) ++bottom;

    const int min_band = std::max(2, static_cast<int>(height * 0.015f));
    const int symmetry_tolerance =
        std::max(2, static_cast<int>(height * 0.015f));
    if (top < min_band || bottom < min_band ||
        std::abs(top - bottom) > symmetry_tolerance ||
        top + bottom >= static_cast<int>(height * 0.55f)) {
        return result;
    }

    // A fade/full-black frame otherwise looks like one enormous matte. Demand
    // visible content immediately inside at least one of the two boundaries.
    int visible = 0;
    int samples = 0;
    const int probe_rows = std::min(4, height - top - bottom);
    for (int r = 0; r < probe_rows; ++r) {
        const int ys[2] = {top + r, height - bottom - 1 - r};
        for (int k = 0; k < 2; ++k) {
            const float* row =
                luma.data() + static_cast<size_t>(ys[k]) * width;
            for (int x = 0; x < width; ++x) {
                if (row[x] >= 0.075f) ++visible;
                ++samples;
            }
        }
    }
    if (samples == 0 || visible < static_cast<int>(samples * 0.02f))
        return result;

    result.top = top;
    result.bottom = bottom;
    result.valid = true;
    return result;
}

float bilinear_sample(const std::vector<float>& image, int width, int height,
                      float x, float y) {
    x = std::max(0.0f, std::min(static_cast<float>(width - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(height - 1), y));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(width - 1, x0 + 1);
    const int y1 = std::min(height - 1, y0 + 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float a = image[static_cast<size_t>(y0) * width + x0] * (1.0f - fx) +
                    image[static_cast<size_t>(y0) * width + x1] * fx;
    const float b = image[static_cast<size_t>(y1) * width + x0] * (1.0f - fx) +
                    image[static_cast<size_t>(y1) * width + x1] * fx;
    return a * (1.0f - fy) + b * fy;
}

// Worker-side thread budget for the parallel CPU stages. SYLC_FLOW_THREADS
// overrides (1 = sequential rollback); the default leaves plenty of cores
// for the decoder and the GUI thread.
int flow_threads_from_env() {
    char* env = nullptr;
    size_t len = 0;
    int value = 0;
    if (_dupenv_s(&env, &len, "SYLC_FLOW_THREADS") == 0 && env) {
        value = std::atoi(env);
        std::free(env);
    }
    if (value >= 1) return std::min(value, 16);
    const int hc = static_cast<int>(std::thread::hardware_concurrency());
    // Cap raised 8 -> 12 -> 16 (2026-08-02): under TensorRT the CPU stages
    // ARE the cycle, and the author measured a further real gain at 16 on the
    // 16-core reference box (SYLC_FLOW_THREADS=16). With the persistent chunk
    // pool a wider budget no longer pays a spawn tax. hc/2 keeps small
    // machines conservative.
    return std::max(2, std::min(16, hc / 2));
}

// Round 6b — dedicated inference thread. The worker submits the prepped
// temporal window for map N and only waits for map N-1's result AFTER that
// submission: the GPU infers N while the CPU runs boundary/refine/step/
// geometry for N-1. The result QUEUE is bounded at 2 (steady state holds one
// in-flight map plus, transiently, one cut-retry): wait_result() extracts by
// sequence so a retry can overtake an already-finished later map without
// dropping it.
struct InferPipe {
    explicit InferPipe(DepthEngine& engine) : engine_(engine) {}

    ~InferPipe() { stop(); }

    void start() {
        thread_ = std::thread([this] { run(); });
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    // Copies `input` into the job slot. The slot is free by construction when
    // called in the worker's strict submit/wait alternation; the wait below is
    // a safety net, not a hot path.
    void submit(const std::vector<float>& input, uint64_t seq) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&] { return stop_ || !job_fresh_; });
        if (stop_) return;
        job_input_ = input;
        job_seq_ = seq;
        job_fresh_ = true;
        cv_.notify_all();
    }

    struct Result {
        std::vector<float> raw, confidence;
        std::string error;
        bool failed = false;
        uint64_t seq = 0;
        double infer_ms = 0.0;
    };

    // Waits until the result for `seq` exists and extracts it (by sequence,
    // not queue order). Returns false only on stop.
    bool wait_result(uint64_t seq, Result& out) {
        std::unique_lock<std::mutex> lk(mtx_);
        for (;;) {
            for (auto it = results_.begin(); it != results_.end(); ++it) {
                if (it->seq == seq) {
                    out = std::move(*it);
                    results_.erase(it);
                    cv_.notify_all();   // a waiting producer may now push
                    return true;
                }
            }
            if (stop_) return false;
            cv_.wait(lk);
        }
    }

private:
    void run() {
        std::vector<float> input;
        for (;;) {
            uint64_t seq = 0;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || job_fresh_; });
                if (stop_) return;
                input.swap(job_input_);
                seq = job_seq_;
                job_fresh_ = false;
            }
            cv_.notify_all();           // job slot free: a submit may proceed
            Result result;
            result.seq = seq;
            result.raw.resize(out_n_);
            result.confidence.assign(out_n_, 1.0f);
            const auto t0 = Clock::now();
            result.failed = !engine_.infer(
                input.data(), result.raw.data(), result.error,
                result.confidence.data());
            result.infer_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - t0).count();
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || results_.size() < 2; });
                if (stop_) return;
                results_.push_back(std::move(result));
            }
            cv_.notify_all();
        }
    }

    DepthEngine& engine_;
    std::thread thread_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<float> job_input_;
    uint64_t job_seq_ = 0;
    bool job_fresh_ = false;
    std::vector<Result> results_;
    bool stop_ = false;

public:
    size_t out_n_ = 0;                 // set once before start(): grid n
};

}  // namespace

namespace synth3d_aspect {

HorizontalBars detect_horizontal_letterbox(
        const std::vector<float>& luma, int width, int height) {
    const LetterboxCandidate candidate =
        ::detect_horizontal_letterbox(luma, width, height);
    HorizontalBars result;
    result.top = candidate.top;
    result.bottom = candidate.bottom;
    result.valid = candidate.valid;
    return result;
}

}  // namespace synth3d_aspect

namespace synth3d_flow {

// A deliberately compact CPU flow estimator. Historically inference was the
// dominant cost; under TensorRT (~18 ms) these two calls per cycle became
// the cycle's largest term (measured 68-108 ms sequential on real film), so
// the four O(n) phases run on `max_threads` threads. REQUIRED to be
// bit-exact vs max_threads=1: rows/nodes/candidates are independent, and
// the global-search minimum is reduced chunk-in-order so the sequential
// iteration-order winner is always selected (pinned by
// test_flow_parallel.py).
// Flow is source->destination displacement stored at destination coordinates.
void fuse_bidirectional(const std::vector<float>& prev_luma,
                        const std::vector<float>& cur_luma,
                        const DenseFlow& forward,
                        const DenseFlow& future,
                        int width, int height,
                        float video_time_scale,
                        std::vector<float>& out_x,
                        std::vector<float>& out_y,
                        std::vector<float>& out_reliability,
                        std::vector<float>& out_motion,
                        double& motion_sum, float& mean_flow,
                        int max_threads,
                        const DenseFlow* codec_hint) {
    const size_t n = static_cast<size_t>(width) * height;
    const bool has_hint = codec_hint != nullptr &&
                          codec_hint->x.size() == n &&
                          codec_hint->y.size() == n &&
                          codec_hint->quality.size() == n;
    out_x.resize(n);
    out_y.resize(n);
    out_reliability.resize(n);
    out_motion.resize(n);
    const int chunks = parallel_chunk_count(height, max_threads);
    std::vector<double> chunk_motion(chunks, 0.0);
    std::vector<double> chunk_mag(chunks, 0.0);
    parallel_chunks(height, max_threads,
                    [&](int chunk_id, int y_begin, int y_end) {
    double local_motion = 0.0;
    double local_mag = 0.0;
    for (int y = y_begin; y < y_end; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const float ax = forward.x[i], ay = forward.y[i];
            const float bx = -future.x[i], by = -future.y[i];
            const float qa = clamp01(forward.quality[i]);
            const float qb = clamp01(future.quality[i]);
            auto residual = [&](float fx, float fy) {
                const float sx = static_cast<float>(x) - fx;
                const float sy = static_cast<float>(y) - fy;
                if (sx < 0.0f || sy < 0.0f ||
                    sx > static_cast<float>(width - 1) ||
                    sy > static_cast<float>(height - 1))
                    return 1.0f;               // hors cadre: candidat invalide
                return std::abs(cur_luma[i] -
                                bilinear_sample(prev_luma, width, height,
                                                sx, sy));
            };
            const float ra = residual(ax, ay);
            const float rb = residual(bx, by);
            // Égalité -> causal (déterministe, et le prior de causalité est
            // le bon: à résidu égal, l'estimateur direct l'emporte).
            const bool pick_b = rb < ra;
            float px = pick_b ? bx : ax;
            float py = pick_b ? by : ay;
            float r = pick_b ? rb : ra;
            float q_sel = pick_b ? qb : qa;
            // Phase 1 (04/08) : candidat DÉCODEUR, même arbitrage. Qualité
            // 0 = bloc intra/absent, le candidat n'existe pas — jamais
            // adopté même si son résidu serait meilleur (un vecteur non
            // déclaré n'est pas une observation). Strictement < : à
            // égalité, causal puis futur gardent la main.
            if (has_hint) {
                const float qh = clamp01(codec_hint->quality[i]);
                if (qh > 0.0f) {
                    const float hx = codec_hint->x[i];
                    const float hy = codec_hint->y[i];
                    const float rh = residual(hx, hy);
                    if (rh < r) {
                        px = hx;
                        py = hy;
                        r = rh;
                        q_sel = qh;
                    }
                }
            }
            // Accord symétrique des deux directions: mouvement localement
            // linéaire confirmé par deux estimateurs indépendants.
            const float sym_err = std::sqrt(
                (forward.x[i] + future.x[i]) * (forward.x[i] + future.x[i]) +
                (forward.y[i] + future.y[i]) * (forward.y[i] + future.y[i]));
            const float sym = std::exp(-0.35f * sym_err) *
                              std::sqrt(qa * qb);
            // Auto-confiance résiduelle du candidat retenu: un transport qui
            // reproduit la cible est fiable même si l'autre direction
            // désaccorde (dernière frame d'un plan: le futur appartient au
            // plan suivant).
            const float self_conf = q_sel * std::exp(-40.0f * r);
            const float rel = clamp01(std::max(sym, self_conf));
            out_x[i] = px;
            out_y[i] = py;
            out_reliability[i] = rel;
            // Même forme que l'ancien passage FB: résidu aligné + pénalité
            // d'incertitude pré-échelonnée en temps source.
            out_motion[i] = clamp01(
                r + (1.0f - rel) * 0.12f * video_time_scale);
            local_motion += out_motion[i];
            local_mag += std::sqrt(px * px + py * py);
        }
    }
    chunk_motion[chunk_id] = local_motion;
    chunk_mag[chunk_id] = local_mag;
    });
    motion_sum = 0.0;
    double mag_sum = 0.0;
    for (int c = 0; c < chunks; ++c) {
        motion_sum += chunk_motion[c];
        mag_sum += chunk_mag[c];
    }
    mean_flow = static_cast<float>(mag_sum / static_cast<double>(n));
}

void rasterize_motion_hints(const int16_t* mv_xy, const uint8_t* valid,
                            int blocks_w, int blocks_h,
                            int source_width, int source_height,
                            int grid_width, int grid_height,
                            float time_scale,
                            std::vector<float>& out_x,
                            std::vector<float>& out_y,
                            std::vector<float>& out_quality) {
    const size_t n = static_cast<size_t>(grid_width) * grid_height;
    out_x.assign(n, 0.0f);
    out_y.assign(n, 0.0f);
    out_quality.assign(n, 0.0f);
    if (mv_xy == nullptr || blocks_w <= 0 || blocks_h <= 0 ||
        source_width <= 0 || source_height <= 0 ||
        grid_width <= 0 || grid_height <= 0) {
        return;
    }
    // Quart-de-pel source -> texels grille, étirement anisotrope (le même
    // que le prep). time_scale ramène « par frame d'affichage » à « par
    // observation consommée ».
    const float sx = 0.25f * time_scale *
                     static_cast<float>(grid_width) / source_width;
    const float sy = 0.25f * time_scale *
                     static_cast<float>(grid_height) / source_height;
    for (int gy = 0; gy < grid_height; ++gy) {
        const int by = std::min(
            blocks_h - 1,
            static_cast<int>((gy + 0.5f) * blocks_h / grid_height));
        for (int gx = 0; gx < grid_width; ++gx) {
            const int bx = std::min(
                blocks_w - 1,
                static_cast<int>((gx + 0.5f) * blocks_w / grid_width));
            const size_t b = static_cast<size_t>(by) * blocks_w + bx;
            if (valid != nullptr && valid[b] == 0) continue;
            const size_t i = static_cast<size_t>(gy) * grid_width + gx;
            out_x[i] = mv_xy[2 * b] * sx;
            out_y[i] = mv_xy[2 * b + 1] * sy;
            out_quality[i] = 1.0f;
        }
    }
}

double mean_divergence(const std::vector<float>& flow_x,
                       const std::vector<float>& flow_y,
                       int width, int height) {
    const size_t n = static_cast<size_t>(width) * height;
    if (width < 3 || height < 3 || flow_x.size() != n || flow_y.size() != n)
        return 0.0;
    double sum = 0.0;
    long count = 0;
    for (int y = 1; y < height - 1; y += 2) {
        const size_t row = static_cast<size_t>(y) * width;
        for (int x = 1; x < width - 1; x += 2) {
            const double dfx = 0.5 * (flow_x[row + x + 1] -
                                      flow_x[row + x - 1]);
            const double dfy = 0.5 * (flow_y[row + width + x] -
                                      flow_y[row - width + x]);
            sum += dfx + dfy;
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}

void expand_boundary_motion(std::vector<float>& motion,
                            const std::vector<float>& boundary,
                            const std::vector<float>& flow_x,
                            const std::vector<float>& flow_y,
                            int width, int height,
                            std::vector<float>& scratch,
                            int max_threads,
                            bool directional) {
    const size_t n = static_cast<size_t>(width) * height;
    if (width < 3 || height < 3 || motion.size() != n ||
        boundary.size() != n) {
        return;
    }
    const bool has_flow = directional &&
                          flow_x.size() == n && flow_y.size() == n;
    scratch = motion;
    // Sondes anti-traînée : amont/aval sur les deux axes, distances 2 et 3
    // (le 3×3 isotrope couvre déjà la distance 1).
    static const int kProbes[8][2] = {
        {-2, 0}, {2, 0}, {0, -2}, {0, 2},
        {-3, 0}, {3, 0}, {0, -3}, {0, 3}};
    parallel_chunks(height - 2, max_threads,
                    [&](int, int begin, int end) {
    for (int y = 1 + begin; y < 1 + end; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            if (boundary[i] < 0.20f) continue;
            float local_max = 0.0f;
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox)
                    local_max = std::max(
                        local_max,
                        scratch[static_cast<size_t>(y + oy) * width +
                                x + ox]);
            if (has_flow) {
                for (int p = 0; p < 8; ++p) {
                    const int px = x + kProbes[p][0];
                    const int py = y + kProbes[p][1];
                    if (px < 0 || px >= width || py < 0 || py >= height)
                        continue;
                    const size_t j = static_cast<size_t>(py) * width + px;
                    const float mj = scratch[j];
                    if (mj <= local_max) continue;
                    // Le flot DE LA SONDE doit avancer de moi vers elle
                    // (projeté >= 0.75 texel) : je suis alors sa traînée.
                    const float advance =
                        flow_x[j] * kProbes[p][0] + flow_y[j] * kProbes[p][1];
                    const float span = static_cast<float>(
                        kProbes[p][0] * kProbes[p][0] +
                        kProbes[p][1] * kProbes[p][1]);
                    if (advance * advance >= 0.5625f * span &&
                        advance > 0.0f) {
                        local_max = mj;
                    }
                }
            }
            motion[i] = local_max;
        }
    }
    });
}

DenseFlow estimate_flow(const std::vector<float>& source_full,
                        const std::vector<float>& destination_full,
                        int full_width, int full_height,
                        int max_threads) {
    constexpr int kDownscale = 4;
    constexpr int kGlobalRadius = 12;
    // 8 coarse pixels = 32 pixels on the 756 grid. The former 64-pixel
    // lattice could not represent independent head/neck/background motion.
    constexpr int kGridStep = 8;
    constexpr int kLocalRadius = 3;
    constexpr int kPatchRadius = 3;
    const int width = full_width / kDownscale;
    const int height = full_height / kDownscale;
    const size_t coarse_n = static_cast<size_t>(width) * height;
    std::vector<float> source(coarse_n, 0.0f);
    std::vector<float> destination(coarse_n, 0.0f);

    parallel_chunks(height, max_threads, [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width; ++x) {
                double source_sum = 0.0;
                double destination_sum = 0.0;
                for (int yy = 0; yy < kDownscale; ++yy) {
                    const int fy = y * kDownscale + yy;
                    for (int xx = 0; xx < kDownscale; ++xx) {
                        const int fx = x * kDownscale + xx;
                        const size_t fi =
                            static_cast<size_t>(fy) * full_width + fx;
                        source_sum += source_full[fi];
                        destination_sum += destination_full[fi];
                    }
                }
                const size_t ci = static_cast<size_t>(y) * width + x;
                source[ci] = static_cast<float>(source_sum / 16.0);
                destination[ci] = static_cast<float>(destination_sum / 16.0);
            }
        }
    });

    // First recover the dominant camera translation. Sampling every third
    // coarse pixel keeps this bounded while remaining robust to foregrounds.
    int global_dx = 0;
    int global_dy = 0;
    {
        // Parallel over candidate dy rows; each chunk keeps its FIRST
        // minimum (strict <, same scan order as the sequential loop), and
        // the chunks are reduced in ascending order below -- the winner is
        // therefore identical to the sequential iteration-order winner.
        const int total_dy = 2 * kGlobalRadius + 1;
        const int chunks = parallel_chunk_count(total_dy, max_threads);
        std::vector<double> chunk_best(chunks,
            std::numeric_limits<double>::infinity());
        std::vector<int> chunk_dx(chunks, 0);
        std::vector<int> chunk_dy(chunks, 0);
        parallel_chunks(total_dy, max_threads,
                        [&](int chunk_id, int begin, int end) {
            double best = std::numeric_limits<double>::infinity();
            int best_dx = 0;
            int best_dy = 0;
            for (int dyi = begin; dyi < end; ++dyi) {
                const int dy = dyi - kGlobalRadius;
                for (int dx = -kGlobalRadius; dx <= kGlobalRadius; ++dx) {
                    double cost = 0.0;
                    int count = 0;
                    for (int y = kGlobalRadius; y < height - kGlobalRadius;
                         y += 3) {
                        const int sy = y - dy;
                        if (sy < 1 || sy >= height - 1) continue;
                        for (int x = kGlobalRadius;
                             x < width - kGlobalRadius; x += 3) {
                            const int sx = x - dx;
                            if (sx < 1 || sx >= width - 1) continue;
                            const float dst =
                                destination[static_cast<size_t>(y) * width + x];
                            const float src =
                                source[static_cast<size_t>(sy) * width + sx];
                            cost += std::min(0.20f, std::abs(dst - src));
                            ++count;
                        }
                    }
                    if (count > 0) cost /= count;
                    // Repeated textures can have several photometrically
                    // identical matches. A tiny minimum-motion prior resolves
                    // only those ties; genuine translation still wins by its
                    // much larger SAD gain.
                    cost += 0.000005 * static_cast<double>(dx * dx + dy * dy);
                    if (cost < best) {
                        best = cost;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }
            chunk_best[chunk_id] = best;
            chunk_dx[chunk_id] = best_dx;
            chunk_dy[chunk_id] = best_dy;
        });
        double global_best = std::numeric_limits<double>::infinity();
        for (int c = 0; c < chunks; ++c) {
            if (chunk_best[c] < global_best) {
                global_best = chunk_best[c];
                global_dx = chunk_dx[c];
                global_dy = chunk_dy[c];
            }
        }
    }

    const int grid_width = (width - 1 + kGridStep - 1) / kGridStep + 1;
    const int grid_height = (height - 1 + kGridStep - 1) / kGridStep + 1;
    const size_t grid_n = static_cast<size_t>(grid_width) * grid_height;
    std::vector<float> node_x(grid_n, static_cast<float>(global_dx));
    std::vector<float> node_y(grid_n, static_cast<float>(global_dy));
    std::vector<float> node_quality(grid_n, 0.0f);

    parallel_chunks(grid_height, max_threads,
                    [&](int, int gy_begin, int gy_end) {
    for (int gy = gy_begin; gy < gy_end; ++gy) {
        const int cy = std::min(height - 1, gy * kGridStep);
        for (int gx = 0; gx < grid_width; ++gx) {
            const int cx = std::min(width - 1, gx * kGridStep);
            double destination_sum = 0.0;
            double destination_sq = 0.0;
            int destination_count = 0;
            for (int py = -kPatchRadius; py <= kPatchRadius; ++py) {
                const int y = cy + py;
                if (y < 0 || y >= height) continue;
                for (int px = -kPatchRadius; px <= kPatchRadius; ++px) {
                    const int x = cx + px;
                    if (x < 0 || x >= width) continue;
                    const float v = destination[static_cast<size_t>(y) * width + x];
                    destination_sum += v;
                    destination_sq += static_cast<double>(v) * v;
                    ++destination_count;
                }
            }
            const double destination_mean =
                destination_count ? destination_sum / destination_count : 0.0;
            const double destination_variance = destination_count
                ? std::max(0.0, destination_sq / destination_count -
                                    destination_mean * destination_mean)
                : 0.0;
            const float texture = clamp01(
                static_cast<float>(std::sqrt(destination_variance) * 16.0));

            float best = std::numeric_limits<float>::infinity();
            float second = std::numeric_limits<float>::infinity();
            int best_dx = global_dx;
            int best_dy = global_dy;
            for (int dy = global_dy - kLocalRadius;
                 dy <= global_dy + kLocalRadius; ++dy) {
                for (int dx = global_dx - kLocalRadius;
                     dx <= global_dx + kLocalRadius; ++dx) {
                    double source_sum = 0.0;
                    double cost = 0.0;
                    int count = 0;
                    for (int py = -kPatchRadius; py <= kPatchRadius; ++py) {
                        const int y = cy + py;
                        const int sy = y - dy;
                        if (y < 0 || y >= height || sy < 0 || sy >= height)
                            continue;
                        for (int px = -kPatchRadius; px <= kPatchRadius; ++px) {
                            const int x = cx + px;
                            const int sx = x - dx;
                            if (x < 0 || x >= width || sx < 0 || sx >= width)
                                continue;
                            source_sum +=
                                source[static_cast<size_t>(sy) * width + sx];
                            ++count;
                        }
                    }
                    if (count < 16) continue;
                    const double source_mean = source_sum / count;
                    for (int py = -kPatchRadius; py <= kPatchRadius; ++py) {
                        const int y = cy + py;
                        const int sy = y - dy;
                        if (y < 0 || y >= height || sy < 0 || sy >= height)
                            continue;
                        for (int px = -kPatchRadius; px <= kPatchRadius; ++px) {
                            const int x = cx + px;
                            const int sx = x - dx;
                            if (x < 0 || x >= width || sx < 0 || sx >= width)
                                continue;
                            const float a =
                                destination[static_cast<size_t>(y) * width + x] -
                                static_cast<float>(destination_mean);
                            const float b =
                                source[static_cast<size_t>(sy) * width + sx] -
                                static_cast<float>(source_mean);
                            cost += std::min(0.20f, std::abs(a - b));
                        }
                    }
                    const int ddx = dx - global_dx;
                    const int ddy = dy - global_dy;
                    const float normalized =
                        static_cast<float>(cost / count) +
                        0.00002f * static_cast<float>(ddx * ddx + ddy * ddy);
                    if (normalized < best) {
                        second = best;
                        best = normalized;
                        best_dx = dx;
                        best_dy = dy;
                    } else if (normalized < second) {
                        second = normalized;
                    }
                }
            }
            const size_t gi = static_cast<size_t>(gy) * grid_width + gx;
            const float uniqueness = std::isfinite(second)
                ? clamp01((second - best) / (second + 0.002f) * 8.0f) : 0.0f;
            const float photometric = std::exp(-18.0f * std::min(best, 1.0f));
            // Textureless patches use the stable global camera motion rather
            // than an arbitrary local winner.
            if (texture < 0.12f) {
                node_x[gi] = static_cast<float>(global_dx);
                node_y[gi] = static_cast<float>(global_dy);
                node_quality[gi] = 0.50f * photometric;
            } else {
                node_x[gi] = static_cast<float>(best_dx);
                node_y[gi] = static_cast<float>(best_dy);
                node_quality[gi] = photometric *
                    clamp01(0.25f + 0.45f * texture + 0.30f * uniqueness);
            }
        }
    }
    });

    // Smooth only local deviations from the global transform. The confidence
    // weighting preserves coherent object motion but rejects isolated matches.
    std::vector<float> smooth_x(grid_n);
    std::vector<float> smooth_y(grid_n);
    for (int gy = 0; gy < grid_height; ++gy) {
        for (int gx = 0; gx < grid_width; ++gx) {
            double sum_x = 0.0, sum_y = 0.0, sum_w = 0.0;
            for (int oy = -1; oy <= 1; ++oy) {
                const int yy = std::max(0, std::min(grid_height - 1, gy + oy));
                for (int ox = -1; ox <= 1; ++ox) {
                    const int xx = std::max(0, std::min(grid_width - 1, gx + ox));
                    const size_t ni = static_cast<size_t>(yy) * grid_width + xx;
                    const double spatial = (ox == 0 && oy == 0) ? 2.0 : 1.0;
                    const double weight = spatial * (0.20 + node_quality[ni]);
                    sum_x += weight * node_x[ni];
                    sum_y += weight * node_y[ni];
                    sum_w += weight;
                }
            }
            const size_t gi = static_cast<size_t>(gy) * grid_width + gx;
            smooth_x[gi] = static_cast<float>(sum_x / sum_w);
            smooth_y[gi] = static_cast<float>(sum_y / sum_w);
        }
    }

    DenseFlow result;
    const size_t full_n = static_cast<size_t>(full_width) * full_height;
    result.x.resize(full_n);
    result.y.resize(full_n);
    result.quality.resize(full_n);
    parallel_chunks(full_height, max_threads,
                    [&](int, int y_begin, int y_end) {
    for (int y = y_begin; y < y_end; ++y) {
        const float cy = static_cast<float>(y) / kDownscale;
        const int gy0 = std::min(grid_height - 1,
                                 static_cast<int>(cy) / kGridStep);
        const int gy1 = std::min(grid_height - 1, gy0 + 1);
        const float y0 = static_cast<float>(
            std::min(height - 1, gy0 * kGridStep));
        const float y1 = static_cast<float>(
            std::min(height - 1, gy1 * kGridStep));
        const float ty = y1 > y0 ? clamp01((cy - y0) / (y1 - y0)) : 0.0f;
        for (int x = 0; x < full_width; ++x) {
            const float cx = static_cast<float>(x) / kDownscale;
            const int gx0 = std::min(grid_width - 1,
                                     static_cast<int>(cx) / kGridStep);
            const int gx1 = std::min(grid_width - 1, gx0 + 1);
            const float x0 = static_cast<float>(
                std::min(width - 1, gx0 * kGridStep));
            const float x1 = static_cast<float>(
                std::min(width - 1, gx1 * kGridStep));
            const float tx = x1 > x0 ? clamp01((cx - x0) / (x1 - x0)) : 0.0f;
            const size_t i00 = static_cast<size_t>(gy0) * grid_width + gx0;
            const size_t i10 = static_cast<size_t>(gy0) * grid_width + gx1;
            const size_t i01 = static_cast<size_t>(gy1) * grid_width + gx0;
            const size_t i11 = static_cast<size_t>(gy1) * grid_width + gx1;
            auto interpolate = [&](const std::vector<float>& values) {
                const float top = values[i00] * (1.0f - tx) + values[i10] * tx;
                const float bottom =
                    values[i01] * (1.0f - tx) + values[i11] * tx;
                return top * (1.0f - ty) + bottom * ty;
            };
            const size_t fi = static_cast<size_t>(y) * full_width + x;
            result.x[fi] = interpolate(smooth_x) * kDownscale;
            result.y[fi] = interpolate(smooth_y) * kDownscale;
            result.quality[fi] = clamp01(interpolate(node_quality));
        }
    }
    });
    return result;
}

}  // namespace synth3d_flow

namespace synth3d_surface {

void compute_boundary(const std::vector<float>& depth,
                      const std::vector<float>& luma,
                      int width, int height,
                      std::vector<float>& boundary,
                      std::vector<float>& scratch,
                      int max_threads) {
    const size_t n = static_cast<size_t>(width) * height;
    boundary.assign(n, 0.0f);
    scratch.assign(n, 0.0f);
    double mean = 0.0, m2 = 0.0, count = 0.0;
    for (float value : depth) {
        count += 1.0;
        const double delta = static_cast<double>(value) - mean;
        mean += delta / count;
        m2 += delta * (static_cast<double>(value) - mean);
    }
    const float sigma = static_cast<float>(
        std::sqrt(std::max(0.0, m2 / std::max(1.0, count))));
    const float depth_scale = std::max(1.0e-6f, 0.32f * sigma);

    // Both passes are row-parallel with per-row writes only (pass 2 reads
    // scratch rows y±1, all fully written after the first join): bitwise
    // identical at any thread count. Measured serial cost was one of the
    // residual terms of the round-6 cycle shave.
    parallel_chunks(height - 2, max_threads, [&](int, int begin, int end) {
    for (int y = 1 + begin; y < 1 + end; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const float depth_gradient = std::max(
                std::abs(depth[i + 1] - depth[i - 1]),
                std::abs(depth[i + width] - depth[i - width]));
            const float luma_gradient = std::max(
                std::abs(luma[i + 1] - luma[i - 1]),
                std::abs(luma[i + width] - luma[i - width]));
            const float depth_edge = clamp01(
                (depth_gradient / depth_scale - 0.18f) / 0.82f);
            const float image_support = clamp01(
                (luma_gradient - 0.018f) / 0.12f);
            // A model-only edge remains protected, but an aligned image edge
            // raises confidence that this is a real surface boundary rather
            // than texture/noise inside one object.
            // Squaring cleanly separates ordinary within-surface variation
            // from a true layer discontinuity. A strong model-only break is
            // still guarded (0.45), while moderate uncorroborated ripples no
            // longer disable every spatial cleanup tap around them.
            scratch[i] = clamp01(
                depth_edge * depth_edge *
                (0.45f + 0.55f * image_support));
        }
    }
    });

    // Form a narrow three-pixel guard band. Temporal decisions inside this
    // band are layer-aware; there is deliberately no blur/average here.
    parallel_chunks(height - 2, max_threads, [&](int, int begin, int end) {
    for (int y = 1 + begin; y < 1 + end; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float value = 0.0f;
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox)
                    value = std::max(
                        value,
                        scratch[static_cast<size_t>(y + oy) * width + x + ox]);
            boundary[static_cast<size_t>(y) * width + x] = value;
        }
    }
    });
}

void refine_observation(std::vector<float>& depth,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<float>& scratch,
                        int max_threads) {
    const size_t n = static_cast<size_t>(width) * height;
    if (width < 3 || height < 3 || depth.size() != n || luma.size() != n ||
        boundary.size() != n ||
        (!confidence.empty() && confidence.size() != n)) {
        return;
    }
    scratch = depth;
    auto smooth01 = [](float edge0, float edge1, float value) {
        float t = (value - edge0) / std::max(1.0e-6f, edge1 - edge0);
        t = clamp01(t);
        return t * t * (3.0f - 2.0f * t);
    };

    // A compact cross+diagonal joint filter. It is deliberately disabled on
    // the three-pixel boundary guard produced above, so it calms texture-shaped
    // depth noise and stair-stepped gradients without averaging foreground and
    // background into an impossible semi-depth at silhouettes.
    parallel_chunks(height - 2, max_threads,
                    [&](int, int row_begin, int row_end) {
        for (int yy = row_begin; yy < row_end; ++yy) {
            const int y = yy + 1;
            for (int x = 1; x < width - 1; ++x) {
                const size_t i = static_cast<size_t>(y) * width + x;
                const float center_guard = 1.0f -
                    smooth01(0.08f, 0.38f, boundary[i]);
                if (center_guard <= 0.0f) continue;

                float sum = 1.6f * depth[i];
                float weight_sum = 1.6f;
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        if (ox == 0 && oy == 0) continue;
                        const size_t j = static_cast<size_t>(y + oy) * width +
                                         static_cast<size_t>(x + ox);
                        const float luma_guard = 1.0f - smooth01(
                            0.018f, 0.105f, std::abs(luma[j] - luma[i]));
                        const float surface_guard = 1.0f - smooth01(
                            0.10f, 0.42f,
                            std::max(boundary[i], boundary[j]));
                        const float spatial = (ox != 0 && oy != 0) ? 0.52f : 0.82f;
                        const float w = spatial * luma_guard * surface_guard;
                        sum += w * depth[j];
                        weight_sum += w;
                    }
                }
                const float filtered = sum / std::max(1.0e-6f, weight_sum);
                const float conf = confidence.empty()
                    ? 1.0f : clamp01(confidence[i]);
                // Uncertain smooth regions benefit most; trusted detail is
                // touched conservatively. Even at confidence zero the maximum
                // blend is 0.46, so this pass can never replace the observation.
                const float blend = center_guard * (0.28f + 0.18f * (1.0f - conf));
                scratch[i] = depth[i] + blend * (filtered - depth[i]);
            }
        }
    });
    depth.swap(scratch);
}

namespace {
inline float median5(float a, float b, float c, float d, float e) {
    float v[5] = {a, b, c, d, e};
    std::sort(v, v + 5);
    return v[2];
}
}  // namespace

void realign_contours(std::vector<uint16_t>& depth_q16,
                      const std::vector<float>& luma,
                      int width, int height,
                      std::vector<uint16_t>& scratch,
                      int max_threads) {
    const size_t n = static_cast<size_t>(width) * height;
    if (width < 3 || height < 3 || depth_q16.size() != n ||
        luma.size() != n) {
        return;
    }
    constexpr int kReach = 4;       // image-edge search radius (texels)
    constexpr int kRepOff = 2;      // representatives at anchor +2..+6
    constexpr int kRepEnd = 6;
    constexpr float kWide = 0.7f;   // weight of the ±2-scale gradient
    constexpr float kEdgeMin = 0.12f;   // decisive luma edge (bi-scale)
    constexpr float kUnique = 0.55f;    // 2nd edge above this fraction = ambiguous
    constexpr float kSep = 0.10f;       // real layer separation (nearness)
    constexpr float kMove = 0.06f;      // minimum own-side error to act
    constexpr float kLumaMargin = 0.03f;
    constexpr float kAgreeLuma = 0.06f; // trench fill: luma continuity bound
    constexpr int kMajorityLines = 3;   // of 5 lines across (see phase 2)
    // Floating-edge veto (04/08 v2, "the deformed cheek"): a depth cliff
    // that already sits on image evidence is already anchored — only a
    // cliff floating on image-FLAT ground (the model's invention: a halo
    // over uniform sky) may be moved. Without this, a soft shot lets a
    // dominant INTERNAL shading edge (jaw shadow -> lit cheek) masquerade
    // as the layer edge: the lit strip's luma resembles the background
    // more than the adjacent shadow, the luma vote is fooled, and the
    // face is carved toward the far layer. Luma-value proximity is not
    // layer membership on smooth-shaded surfaces; the presence of image
    // support under the cliff is the structural discriminator.
    constexpr float kFlatGround = 0.035f;

    // v3 (04/08, "the actress's wavy cheek"): per-texel decisions on a
    // grainy, defocused silhouette flickered line-to-line (measured 30/48,
    // 17/48, even 1/48 lines corrected), each flip being a FULL layer jump
    // — the wavy contour. Three additions, one shared principle (decide on
    // robust statistics, commit only coherent decisions):
    //   1. the guide luma is smoothed 1-2-1 ACROSS scan lines, so sensor
    //      grain cannot flip marginal gates between neighbouring lines;
    //   2. gradients are BI-SCALE (max of the ±1 diff and 0.7x the ±2
    //      diff): a defocused silhouette spreads over 3-5 texels and is
    //      invisible to the fine diff alone (uniqueness widens its
    //      exclusion to ±3 when the wide scale wins — a soft edge's own
    //      shoulders are not "second edges");
    //   3. representatives anchor BEYOND the contested interval
    //      [min(e,q*), max(e,q*)] — image edge to floating cliff — so no
    //      halo/ramp width can ever poison the medians; a defocused ramp
    //      then splits at its own luma midpoint (sub-texel-correct), and
    //      when both rims agree on ONE layer the interval is a hallucinated
    //      trench, refilled from the agreed layer (luma-continuity gated).
    // Phase 2 commits a proposal only when >= 3 of the 5 surrounding scan
    // lines propose within ±1 texel: an isolated line's full-layer jump IS
    // the artifact, so it is never applied.
    std::vector<float> guide(n, 0.0f);
    std::vector<uint8_t> prop_mask(n, uint8_t(0));
    std::vector<uint16_t> prop_value(n, uint16_t(0));

    auto pass = [&](bool vertical) {
        const int along_n = vertical ? height : width;
        const int across_n = vertical ? width : height;
        if (along_n < 2 * (kReach + kRepEnd) + 1) return;
        scratch.assign(depth_q16.begin(), depth_q16.end());
        const uint16_t* in = scratch.data();

        // Transverse 1-2-1 smoothing of the decision guide (border rows
        // replicate). The published luma is untouched.
        parallel_chunks(height, max_threads, [&](int, int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    size_t a, b;
                    if (vertical) {
                        a = static_cast<size_t>(y) * width +
                            std::max(0, x - 1);
                        b = static_cast<size_t>(y) * width +
                            std::min(width - 1, x + 1);
                    } else {
                        a = static_cast<size_t>(std::max(0, y - 1)) * width + x;
                        b = static_cast<size_t>(std::min(height - 1, y + 1)) *
                            width + x;
                    }
                    guide[i] = 0.25f * (luma[a] + 2.0f * luma[i] + luma[b]);
                }
            }
        });
        std::fill(prop_mask.begin(), prop_mask.end(), uint8_t(0));

        // ---- phase 1: per-texel proposals --------------------------------
        parallel_chunks(across_n, max_threads,
                        [&](int, int begin, int end) {
        for (int across = begin; across < end; ++across) {
            auto at = [&](int along) {
                return vertical
                    ? static_cast<size_t>(along) * width + across
                    : static_cast<size_t>(across) * width + along;
            };
            // Bi-scale gradient with the winning scale reported (fine=false
            // means the wide scale won).
            auto grad = [&](int j, bool& fine_won) {
                const float fine = std::abs(guide[at(j + 1)] -
                                            guide[at(j - 1)]);
                float wide = 0.0f;
                if (j >= 2 && j + 2 <= along_n - 1)
                    wide = kWide * std::abs(guide[at(j + 2)] -
                                            guide[at(j - 2)]);
                fine_won = fine >= wide;
                return std::max(fine, wide);
            };
            for (int p = kReach + kRepEnd;
                 p < along_n - (kReach + kRepEnd); ++p) {
                const int j0 = std::max(1, p - kReach);
                const int j1 = std::min(along_n - 2, p + kReach);
                // Cheap gate: no real layer separation nearby = nothing to
                // re-anchor (covers the vast flat majority of the grid).
                float d_min = 1.0f, d_max = 0.0f;
                for (int j = j0; j <= j1; ++j) {
                    const float dj = in[at(j)] / 65535.0f;
                    d_min = std::min(d_min, dj);
                    d_max = std::max(d_max, dj);
                }
                if (d_max - d_min < kSep) continue;
                // Dominant image edge, then its uniqueness: a second strong
                // edge in the window (textured background, filament) makes
                // THE silhouette position ambiguous — do nothing rather
                // than re-anchor onto the wrong edge.
                float g_max = 0.0f;
                int e = -1;
                bool e_fine = true;
                for (int j = j0; j <= j1; ++j) {
                    bool fine_won = true;
                    const float g = grad(j, fine_won);
                    if (g > g_max) { g_max = g; e = j; e_fine = fine_won; }
                }
                if (g_max < kEdgeMin || e < 0) continue;
                const int exclude = e_fine ? 2 : 3;
                float g_second = 0.0f;
                for (int j = j0; j <= j1; ++j) {
                    if (std::abs(j - e) <= exclude) continue;
                    bool fine_won = true;
                    g_second = std::max(g_second, grad(j, fine_won));
                }
                if (g_second > kUnique * g_max) continue;
                // Floating-edge veto: among the significant depth cliffs in
                // the window, the one FARTHEST from the image edge is the
                // cliff this correction would erase (the halo's outer rim /
                // the invented mid-face cliff of an eaten arm). It may only
                // be erased if the image is flat beneath it — a supported
                // cliff is a real, already-anchored silhouette (possibly
                // weaker than a nearby shading edge; see the deformed-cheek
                // report): leave it alone.
                int q_star = -1;
                {
                    float cd_max = 0.0f;
                    for (int j = j0; j <= j1; ++j) {
                        cd_max = std::max(
                            cd_max,
                            std::abs(static_cast<float>(in[at(j + 1)]) -
                                     static_cast<float>(in[at(j - 1)])) /
                                65535.0f);
                    }
                    if (cd_max < kSep) continue;
                    int best_distance = -1;
                    for (int j = j0; j <= j1; ++j) {
                        const float cd = std::abs(
                            static_cast<float>(in[at(j + 1)]) -
                            static_cast<float>(in[at(j - 1)])) / 65535.0f;
                        if (cd < 0.5f * cd_max) continue;
                        const int distance = std::abs(j - e);
                        if (distance > best_distance) {
                            best_distance = distance;
                            q_star = j;
                        }
                    }
                    bool fine_won = true;
                    if (q_star >= 0 &&
                        grad(q_star, fine_won) >= kFlatGround) {
                        continue;
                    }
                }
                // The contested interval runs from the image edge to the
                // floating cliff; only texels inside it may move, and the
                // side representatives anchor BEYOND it, so no halo or
                // defocus-ramp width can poison the medians.
                const int lo_anchor = std::min(e, q_star);
                const int hi_anchor = std::max(e, q_star);
                if (p < lo_anchor || p > hi_anchor) continue;
                if (lo_anchor - kRepEnd < 0 ||
                    hi_anchor + kRepEnd > along_n - 1) {
                    continue;
                }
                float rep_lo_d, rep_hi_d, rep_lo_l, rep_hi_l;
                {
                    float dm[5], dp[5], lm[5], lp[5];
                    for (int k = 0; k < 5; ++k) {
                        const size_t lo = at(lo_anchor - kRepOff - k);
                        const size_t hi = at(hi_anchor + kRepOff + k);
                        dm[k] = in[lo] / 65535.0f;
                        dp[k] = in[hi] / 65535.0f;
                        lm[k] = guide[lo];
                        lp[k] = guide[hi];
                    }
                    rep_lo_d = median5(dm[0], dm[1], dm[2], dm[3], dm[4]);
                    rep_hi_d = median5(dp[0], dp[1], dp[2], dp[3], dp[4]);
                    rep_lo_l = median5(lm[0], lm[1], lm[2], lm[3], lm[4]);
                    rep_hi_l = median5(lp[0], lp[1], lp[2], lp[3], lp[4]);
                }
                const float y = guide[at(p)];
                const float d = in[at(p)] / 65535.0f;
                float proposal;
                if (std::abs(rep_lo_d - rep_hi_d) >= kSep) {
                    // Layer-edge mode. Which side of the image edge does
                    // this texel live on? The edge texel itself is decided
                    // by its own luma (a step between two texels maximizes
                    // the centered diff on BOTH, and the argmax may land
                    // one texel short) — a defocused ramp thereby splits
                    // at its luma midpoint.
                    int side;
                    if (p < e) side = -1;
                    else if (p > e) side = +1;
                    else if (std::abs(y - rep_lo_l) + kLumaMargin <
                             std::abs(y - rep_hi_l)) side = -1;
                    else if (std::abs(y - rep_hi_l) + kLumaMargin <
                             std::abs(y - rep_lo_l)) side = +1;
                    else continue;   // genuinely mixed edge texel
                    const float rep_own_d = side < 0 ? rep_lo_d : rep_hi_d;
                    const float rep_oth_d = side < 0 ? rep_hi_d : rep_lo_d;
                    const float rep_own_l = side < 0 ? rep_lo_l : rep_hi_l;
                    const float rep_oth_l = side < 0 ? rep_hi_l : rep_lo_l;
                    // Misassignment test: current depth clearly belongs to
                    // the OTHER side while the luma votes for its own side.
                    // Both must hold — the luma vote is what protects a
                    // bright filament with legitimate near depth.
                    const float err_own = std::abs(d - rep_own_d);
                    const float err_oth = std::abs(d - rep_oth_d);
                    if (err_own <= kMove || err_oth >= 0.5f * err_own)
                        continue;
                    if (std::abs(y - rep_own_l) + kLumaMargin >=
                        std::abs(y - rep_oth_l)) {
                        continue;
                    }
                    proposal = rep_own_d;
                } else {
                    // Agreement mode: both rims are the SAME layer — the
                    // interval is a hallucinated trench (no image support
                    // on its floating rim, luma continuous with the rims):
                    // refill it from the agreed layer.
                    const float agreed = 0.5f * (rep_lo_d + rep_hi_d);
                    if (std::abs(d - agreed) <= 2.0f * kMove) continue;
                    if (std::abs(y - 0.5f * (rep_lo_l + rep_hi_l)) >
                        kAgreeLuma) {
                        continue;
                    }
                    proposal = agreed;
                }
                prop_mask[at(p)] = 1;
                prop_value[at(p)] = static_cast<uint16_t>(
                    clamp01(proposal) * 65535.0f + 0.5f);
            }
        }
        });

        // ---- phase 2: transverse majority --------------------------------
        // Commit only decisions shared by >= kMajorityLines of the 5 scan
        // lines around the texel (within ±1 texel along the scan): an
        // isolated line's full-layer jump is exactly the wavy-contour
        // artifact and must never reach the output.
        parallel_chunks(across_n, max_threads,
                        [&](int, int begin, int end) {
        for (int across = begin; across < end; ++across) {
            for (int p = 0; p < along_n; ++p) {
                const size_t ip = vertical
                    ? static_cast<size_t>(p) * width + across
                    : static_cast<size_t>(across) * width + p;
                if (!prop_mask[ip]) continue;
                int lines = 0;
                for (int da = -2; da <= 2; ++da) {
                    const int ac = across + da;
                    if (ac < 0 || ac >= across_n) continue;
                    bool any = false;
                    for (int dl = -1; dl <= 1 && !any; ++dl) {
                        const int al = p + dl;
                        if (al < 0 || al >= along_n) continue;
                        const size_t j = vertical
                            ? static_cast<size_t>(al) * width + ac
                            : static_cast<size_t>(ac) * width + al;
                        if (prop_mask[j]) any = true;
                    }
                    if (any) ++lines;
                }
                if (lines >= kMajorityLines)
                    depth_q16[ip] = prop_value[ip];
            }
        }
        });
    };
    pass(false);
    pass(true);
}

void build_geometry_map(const std::vector<uint16_t>& depth_q16,
                        const std::vector<float>& rgb_chw,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<uint16_t>& geometry_rgba16,
                        std::vector<float>& scratch,
                        int max_threads,
                        double* local_ms,
                        double* propagation_ms) {
    if (local_ms) *local_ms = 0.0;
    if (propagation_ms) *propagation_ms = 0.0;
    const size_t n = static_cast<size_t>(width) * height;
    geometry_rgba16.assign(n * 4, uint16_t(0));
    if (width <= 0 || height <= 0 || depth_q16.size() != n ||
        luma.size() != n ||
        (!confidence.empty() && confidence.size() != n) ||
        (!boundary.empty() && boundary.size() != n)) {
        return;
    }
    const auto local_start = Clock::now();
    const bool have_rgb = rgb_chw.size() == 3 * n;
    scratch.assign(n, 0.0f);
    auto smooth01 = [](float edge0, float edge1, float value) {
        float t = (value - edge0) / std::max(1.0e-6f, edge1 - edge0);
        t = clamp01(t);
        return t * t * (3.0f - 2.0f * t);
    };
    auto depth_at = [&](size_t i) {
        return static_cast<float>(depth_q16[i]) / 65535.0f;
    };
    auto rgb_at = [&](int channel, size_t i) {
        if (!have_rgb) return luma[i];
        if (channel == 0) return clamp01(rgb_chw[i] * 0.229f + 0.485f);
        if (channel == 1) return clamp01(rgb_chw[n + i] * 0.224f + 0.456f);
        return clamp01(rgb_chw[2 * n + i] * 0.225f + 0.406f);
    };

    // Detect disagreement between what the image says is a contour and what
    // the depth field says is a layer boundary. This is the characteristic
    // signal of an arm/hand being eaten by the background. A smaller penalty
    // also covers unsupported model-only cliffs and low-confidence edges.
    parallel_chunks(height, max_threads,
                    [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = static_cast<size_t>(y) * width + x;
                const size_t xl = static_cast<size_t>(y) * width +
                                  std::max(0, x - 1);
                const size_t xr = static_cast<size_t>(y) * width +
                                  std::min(width - 1, x + 1);
                const size_t yu = static_cast<size_t>(std::max(0, y - 1)) *
                                  width + x;
                const size_t yd = static_cast<size_t>(std::min(height - 1, y + 1)) *
                                  width + x;
                const float dg = std::max(
                    std::abs(depth_at(xr) - depth_at(xl)),
                    std::abs(depth_at(yd) - depth_at(yu)));
                const float lg = std::max(
                    std::abs(luma[xr] - luma[xl]),
                    std::abs(luma[yd] - luma[yu]));
                const float image_edge = smooth01(0.022f, 0.135f, lg);
                const float depth_edge = smooth01(0.030f, 0.155f, dg);
                const float conf = confidence.empty()
                    ? 1.0f : clamp01(confidence[i]);
                const float unsupported_image = image_edge * (1.0f - depth_edge);
                const float unsupported_depth = depth_edge * (1.0f - image_edge);
                const float low_confidence = (1.0f - conf) *
                    (0.12f + 0.48f * image_edge);
                const float boundary_uncertainty = boundary.empty()
                    ? 0.0f : (1.0f - conf) * 0.30f * clamp01(boundary[i]);
                // A texture edge on a confident flat wall/screen is ordinary
                // appearance, not missing geometry. Require confidence loss
                // before the expensive foreground-ownership path treats an
                // image-only edge as a torn silhouette.
                scratch[i] = clamp01(std::max(
                    unsupported_image * (0.08f + 0.92f * (1.0f - conf)),
                    std::max(0.62f * unsupported_depth,
                             std::max(low_confidence, boundary_uncertainty))));
            }
        }
    });

    // Dilate the uncertainty once into scratch. Separating this pass avoids
    // re-running a 5x5 maximum inside every ownership-search iteration.
    std::vector<float> dilated(n, 0.0f);
    parallel_chunks(height, max_threads,
                    [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width; ++x) {
                float value = 0.0f;
                for (int oy = -2; oy <= 2; ++oy) {
                    const int yy = std::max(0, std::min(height - 1, y + oy));
                    for (int ox = -2; ox <= 2; ++ox) {
                        const int xx = std::max(0, std::min(width - 1, x + ox));
                        value = std::max(
                            value,
                            scratch[static_cast<size_t>(yy) * width + xx]);
                    }
                }
                dilated[static_cast<size_t>(y) * width + x] = value;
            }
        }
    });
    scratch.swap(dilated);
    std::vector<float> foreground_anchors(n, 0.0f);

    parallel_chunks(height, max_threads,
                    [&](int, int y_begin, int y_end) {
        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = static_cast<size_t>(y) * width + x;
                const float dilated_uncertainty = scratch[i];

                const float center_depth = depth_at(i);
                float candidate_depth = center_depth;
                float repair = 0.0f;
                float foreground_anchor = 0.0f;
                if (dilated_uncertainty > 0.14f) {
                    const float cr = rgb_at(0, i);
                    const float cg = rgb_at(1, i);
                    const float cb = rgb_at(2, i);
                    float near_max = center_depth;
                    float local_min = center_depth;
                    float anchor_support = 0.0f;
                    for (int oy = -3; oy <= 3; ++oy) {
                        const int yy = std::max(0, std::min(height - 1, y + oy));
                        for (int ox = -3; ox <= 3; ++ox) {
                            if (ox == 0 && oy == 0) continue;
                            const int xx = std::max(0, std::min(width - 1, x + ox));
                            const size_t j = static_cast<size_t>(yy) * width + xx;
                            const float dj = depth_at(j);
                            local_min = std::min(local_min, dj);
                            const float color_delta = std::max(
                                std::abs(rgb_at(0, j) - cr),
                                std::max(std::abs(rgb_at(1, j) - cg),
                                         std::abs(rgb_at(2, j) - cb)));
                            const float affinity = 1.0f -
                                smooth01(0.035f, 0.18f, color_delta);
                            if (affinity >= 0.38f) {
                                near_max = std::max(near_max, dj);
                                if (std::abs(dj - center_depth) <= 0.065f) {
                                    const float distance2 =
                                        static_cast<float>(ox * ox + oy * oy);
                                    anchor_support += affinity /
                                        (1.0f + 0.18f * distance2);
                                }
                            }
                        }
                    }
                    foreground_anchor =
                        smooth01(0.040f, 0.16f, center_depth - local_min) *
                        smooth01(0.55f, 2.2f, anchor_support);

                    float weighted_depth = 0.0f;
                    float support = 0.0f;
                    int support_count = 0;
                    for (int oy = -3; oy <= 3; ++oy) {
                        const int yy = std::max(0, std::min(height - 1, y + oy));
                        for (int ox = -3; ox <= 3; ++ox) {
                            if (ox == 0 && oy == 0) continue;
                            const int xx = std::max(0, std::min(width - 1, x + ox));
                            const size_t j = static_cast<size_t>(yy) * width + xx;
                            const float dj = depth_at(j);
                            if (dj < near_max - 0.065f ||
                                dj <= center_depth + 0.025f) continue;
                            const float color_delta = std::max(
                                std::abs(rgb_at(0, j) - cr),
                                std::max(std::abs(rgb_at(1, j) - cg),
                                         std::abs(rgb_at(2, j) - cb)));
                            const float affinity = 1.0f -
                                smooth01(0.035f, 0.18f, color_delta);
                            const float distance2 = static_cast<float>(ox * ox + oy * oy);
                            const float spatial = 1.0f / (1.0f + 0.18f * distance2);
                            const float weight = affinity * spatial;
                            if (weight <= 0.05f) continue;
                            weighted_depth += weight * dj;
                            support += weight;
                            ++support_count;
                        }
                    }
                    if (support_count >= 2 && support > 0.55f) {
                        candidate_depth = weighted_depth / support;
                        const float support_t = smooth01(0.55f, 2.4f, support);
                        const float delta_t = smooth01(
                            0.025f, 0.16f, candidate_depth - center_depth);
                        repair = clamp01(
                            1.15f * dilated_uncertainty * support_t * delta_t);
                    }
                }

                const float safety = std::max(
                    std::max(0.12f, 1.0f - 0.82f * dilated_uncertainty),
                    0.90f * foreground_anchor);
                foreground_anchors[i] = foreground_anchor;
                const size_t out = 4 * i;
                geometry_rgba16[out + 0] = depth_q16[i];
                geometry_rgba16[out + 1] = static_cast<uint16_t>(
                    clamp01(candidate_depth) * 65535.0f + 0.5f);
                geometry_rgba16[out + 2] = static_cast<uint16_t>(
                    clamp01(safety) * 65535.0f + 0.5f);
                geometry_rgba16[out + 3] = static_cast<uint16_t>(
                    clamp01(repair) * 65535.0f + 0.5f);
            }
        }
    });

    if (local_ms) {
        *local_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - local_start).count();
    }
    const auto propagation_start = Clock::now();
    // Short color-geodesic ownership propagation. A square inference grid can
    // stretch an eight-source-pixel foreshortened arm to 10-17 grid texels;
    // one fixed-radius search cannot cross it. Six one-texel hops, following
    // only RGB-compatible neighbours and decaying each hop, extend a supported
    // near layer through the limb while naturally stopping at the screen.
    std::vector<size_t> frontier, next_frontier;
    frontier.reserve(n / 32);
    next_frontier.reserve(n / 32);
    for (size_t i = 0; i < n; ++i)
        if (foreground_anchors[i] > 0.08f ||
            geometry_rgba16[4 * i + 3] > uint16_t(5243))
            frontier.push_back(i);
    std::vector<uint8_t> queued(n, uint8_t(0));
    auto color_delta_between = [&](size_t a, size_t b) {
        if (!have_rgb) return std::abs(luma[a] - luma[b]);
        return std::max(
            std::abs(rgb_chw[a] - rgb_chw[b]) * 0.229f,
            std::max(
                std::abs(rgb_chw[n + a] - rgb_chw[n + b]) * 0.224f,
                std::abs(rgb_chw[2 * n + a] - rgb_chw[2 * n + b]) * 0.225f));
    };
    constexpr int kOwnershipHops = 6;
    for (int hop = 0; hop < kOwnershipHops && !frontier.empty(); ++hop) {
        std::fill(queued.begin(), queued.end(), uint8_t(0));
        next_frontier.clear();
        for (size_t source : frontier) {
            const int sy = static_cast<int>(source / width);
            const int sx = static_cast<int>(source % width);
            const size_t source_out = 4 * source;
            const float source_repair =
                geometry_rgba16[source_out + 3] / 65535.0f;
            const float source_strength = std::max(
                source_repair, foreground_anchors[source]);
            if (source_strength <= 0.08f) continue;
            const float source_depth =
                geometry_rgba16[source_out + 1] / 65535.0f;
            for (int oy = -1; oy <= 1; ++oy) {
                const int y = std::max(0, std::min(height - 1, sy + oy));
                for (int ox = -1; ox <= 1; ++ox) {
                    if (ox == 0 && oy == 0) continue;
                    const int x = std::max(0, std::min(width - 1, sx + ox));
                    const size_t target = static_cast<size_t>(y) * width + x;
                    const float affinity = 1.0f - smooth01(
                        0.030f, 0.16f, color_delta_between(source, target));
                    const float candidate_strength =
                        0.93f * source_strength * affinity;
                    if (candidate_strength <= 0.08f) continue;
                    const float raw_depth = depth_at(target);
                    if (source_depth <= raw_depth + 0.025f) continue;
                    const size_t target_out = 4 * target;
                    const float current_owned =
                        geometry_rgba16[target_out + 1] / 65535.0f;
                    const float current_repair =
                        geometry_rgba16[target_out + 3] / 65535.0f;
                    const float candidate_score = source_depth +
                        0.08f * candidate_strength;
                    const float current_score = current_owned +
                        0.08f * current_repair;
                    if (candidate_score <= current_score) continue;
                    geometry_rgba16[target_out + 1] = static_cast<uint16_t>(
                        clamp01(source_depth) * 65535.0f + 0.5f);
                    geometry_rgba16[target_out + 3] = static_cast<uint16_t>(
                        clamp01(candidate_strength) * 65535.0f + 0.5f);
                    if (!queued[target]) {
                        queued[target] = uint8_t(1);
                        next_frontier.push_back(target);
                    }
                }
            }
        }
        frontier.swap(next_frontier);
    }
    if (propagation_ms) {
        *propagation_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - propagation_start).count();
    }
}

}  // namespace synth3d_surface

std::shared_ptr<SharedDepthService> SharedDepthService::acquire_attached(
        const std::wstring& model_path, const std::wstring& ort_dir,
        int width, int height, uint64_t client_id) {
    if (width <= 0) width = kDefaultSide;
    if (height <= 0) height = width;
    const std::wstring key = service_key(model_path, ort_dir, width, height);
    return service_registry().acquire_attached(
        key, client_id, [&] {
            auto service = std::shared_ptr<SharedDepthService>(
                new SharedDepthService(model_path, ort_dir, width, height));
            service->start_worker();
            return service;
        });
}

void SharedDepthService::detach_and_release(
        std::shared_ptr<SharedDepthService>& service, uint64_t client_id) {
    if (!service) return;
    service->detach(client_id);
    service_registry().defer_release(std::move(service));
}

void SharedDepthService::debug_registry_stats(
        size_t& services, size_t& active, size_t& idle) {
    service_registry().stats(services, active, idle);
}

SharedDepthService::SharedDepthService(std::wstring model_path, std::wstring ort_dir,
                                       int width, int height)
    : model_path_(std::move(model_path)), ort_dir_(std::move(ort_dir)),
      width_(width), height_(height),
      instance_id_(g_service_instance.fetch_add(1, std::memory_order_relaxed) + 1) {
    const size_t n = static_cast<size_t>(width_) * height_;
    input_mailbox_.assign(3 * n, 0.0f);
}

SharedDepthService::~SharedDepthService() {
    state_.store(State::Stopping, std::memory_order_release);
    stop_.store(true, std::memory_order_release);
    input_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void SharedDepthService::start_worker() {
    worker_ = std::thread(&SharedDepthService::worker_main, this);
}

void SharedDepthService::attach(uint64_t client_id) {
    const int previous = clients_.fetch_add(1, std::memory_order_acq_rel);
    if (previous != 0) return;

    // First surface of a new playback session: never expose the final map from
    // the previous title while the fresh inference is warming up.
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        input_fresh_ = false;
        input_capture_time_ = {};
        input_video_time_ms_ = -1.0;
        input_source_width_ = 0;
        input_source_height_ = 0;
        leader_id_ = client_id;
        leader_seen_ = Clock::now();
    }
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        latest_.reset();
        ++output_sequence_;
        output_time_ = {};
        output_video_ms_ = -1.0;
    }
    last_snap_ms_.store(-1, std::memory_order_release);
    last_snap_video_ms_.store(-1.0, std::memory_order_release);
    suggested_convergence_.store(0.5f, std::memory_order_release);
    reset_stabilizer_.store(true, std::memory_order_release);
    crop_top_.store(0, std::memory_order_release);
    crop_bottom_.store(0, std::memory_order_release);
    crop_source_width_.store(0, std::memory_order_release);
    crop_source_height_.store(0, std::memory_order_release);
    crop_confidence_.store(0.0f, std::memory_order_release);
    crop_ready_.store(false, std::memory_order_release);
}

void SharedDepthService::detach(uint64_t client_id) {
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        if (leader_id_ == client_id) {
            leader_id_ = 0;
            leader_seen_ = {};
        }
    }
    bool became_idle = false;
    int old = clients_.load(std::memory_order_acquire);
    while (old > 0) {
        if (clients_.compare_exchange_weak(
                old, old - 1, std::memory_order_acq_rel)) {
            became_idle = old == 1;
            break;
        }
    }
    input_cv_.notify_all();
    if (became_idle) service_registry().client_became_idle(this);
}

bool SharedDepthService::running() const {
    return state_.load(std::memory_order_acquire) == State::Running;
}

bool SharedDepthService::failed() const {
    return state_.load(std::memory_order_acquire) == State::Error;
}

int SharedDepthService::client_count() const {
    return clients_.load(std::memory_order_acquire);
}

bool SharedDepthService::wants_input(uint64_t client_id,
                                     double video_time_ms) {
    if (!running() || clients_.load(std::memory_order_acquire) <= 0) return false;
    const auto now = Clock::now();
    std::lock_guard<std::mutex> lk(input_mtx_);
    if (input_fresh_) return false;  // worker already owes us this map

    const bool lease_expired =
        leader_id_ == 0 || leader_seen_.time_since_epoch().count() == 0 ||
        now - leader_seen_ > kLeaderLease;
    if (leader_id_ != client_id && !lease_expired) return false;

    leader_id_ = client_id;
    leader_seen_ = now;
    const double last_pts =
        last_submitted_video_ms_.load(std::memory_order_acquire);
    if (video_time_ms >= 0.0 && last_pts >= 0.0 &&
        std::abs(video_time_ms - last_pts) <= kPtsJitterMs) {
        duplicate_pts_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    grants_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void SharedDepthService::note_plate_state(
        bool requested, bool armed, bool live, double cut_in) {
    plate_requested_.store(requested, std::memory_order_relaxed);
    plate_armed_.store(armed, std::memory_order_relaxed);
    plate_live_.store(live, std::memory_order_relaxed);
    if (requested && armed && !live) {
        plate_held_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    // Only meaningful while the advisory still describes a real, recent cut:
    // the sentinel (-1e9) and long-expired values must not count.
    if (live && cut_in > -500.0 && cut_in < 300.0) {
        plate_post_.fetch_add(1, std::memory_order_relaxed);
        plate_cut_in_.store(static_cast<float>(cut_in),
                            std::memory_order_relaxed);
    }
}

void SharedDepthService::note_drain_miss(bool ring_empty) {
    if (ring_empty) drain_empty_.fetch_add(1, std::memory_order_relaxed);
    else drain_stalled_.fetch_add(1, std::memory_order_relaxed);
}

void SharedDepthService::note_pass(uint64_t client_id) {
    {
        // Same mutex wants_input() already takes once per pass, so this adds no
        // new contention profile.
        std::lock_guard<std::mutex> lk(input_mtx_);
        if (leader_id_ != client_id) return;
    }
    const int64_t now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count();
    const int64_t prev = last_pass_ns_.exchange(now, std::memory_order_acq_rel);
    if (prev == 0) return;                       // first pass has no interval
    const int64_t dt_us = (now - prev) / 1000;
    // A paused or backgrounded surface can leave an arbitrarily long gap that
    // would swamp the average and hide the very jitter this is here to show.
    if (dt_us <= 0 || dt_us > 500000) return;
    pass_us_sum_.fetch_add(static_cast<uint64_t>(dt_us),
                           std::memory_order_relaxed);
    pass_count_.fetch_add(1, std::memory_order_relaxed);
}

bool SharedDepthService::submit(
        uint64_t client_id, std::vector<float>& chw,
        std::vector<float>& full_frame_luma,
        double video_time_ms,
        std::chrono::steady_clock::time_point capture_time,
        int source_width, int source_height) {
    const size_t pixels = static_cast<size_t>(width_) * height_;
    if (chw.size() != 3 * pixels ||
        (!full_frame_luma.empty() && full_frame_luma.size() != pixels) ||
        !running()) return false;
    if (capture_time.time_since_epoch().count() == 0)
        capture_time = Clock::now();
    if (!std::isfinite(video_time_ms) || video_time_ms < 0.0)
        video_time_ms = -1.0;
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        if (leader_id_ != client_id) return false;
        // Copies can already be in the renderer's asynchronous staging ring
        // when wants_input() first observes a repeated pause PTS. Reject them
        // here as the authoritative second gate, before any temporal state or
        // counters advance.
        const double last_pts =
            last_submitted_video_ms_.load(std::memory_order_acquire);
        if (video_time_ms >= 0.0 && last_pts >= 0.0 &&
            std::abs(video_time_ms - last_pts) <= kPtsJitterMs) {
            duplicate_pts_.fetch_add(1, std::memory_order_relaxed);
            leader_seen_ = Clock::now();
            return false;
        }
        // Latest-only means OVERWRITE: refusing while a frame sat unconsumed
        // turned submit jitter into skipped source intervals (measured
        // round 6b: source_ms alternating 42/83 ms and ~20% of the pipeline's
        // throughput lost). The worker always wants the FRESHEST frame; an
        // overwritten one was stale by definition.
        input_mailbox_.swap(chw);
        input_full_frame_luma_.swap(full_frame_luma);
        input_capture_time_ = capture_time;
        input_video_time_ms_ = video_time_ms;
        input_source_width_ = std::max(0, source_width);
        input_source_height_ = std::max(0, source_height);
        input_fresh_ = true;
        if (video_time_ms >= 0.0)
            last_submitted_video_ms_.store(
                video_time_ms, std::memory_order_release);
        leader_seen_ = Clock::now();
        submits_.fetch_add(1, std::memory_order_relaxed);
        // Wall time from the GPU copy being issued to it reaching the mailbox.
        // One render pass is the designed cost (the drain is deliberately
        // non-blocking and picks up the previous pass's copy); materially more
        // means copies land late, and every miss costs a whole source interval.
        const int64_t lat_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                leader_seen_ - capture_time).count();
        if (lat_us >= 0 && lat_us <= 500000) {
            taplat_us_sum_.fetch_add(static_cast<uint64_t>(lat_us),
                                     std::memory_order_relaxed);
            taplat_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    input_cv_.notify_one();
    return true;
}

std::shared_ptr<const SharedDepthService::GeometryFrame> SharedDepthService::snapshot(
        uint64_t after_sequence, uint64_t& sequence,
        double& source_video_ms) const {
    std::lock_guard<std::mutex> lk(output_mtx_);
    sequence = output_sequence_;
    source_video_ms = output_video_ms_;
    if (!latest_ || output_sequence_ == after_sequence) return {};
    return latest_;
}

void SharedDepthService::notify_seek() {
    reset_stabilizer_.store(true, std::memory_order_release);
    // A seek may legitimately land on a PTS seen before the jump. The new
    // epoch must be allowed to submit it once.
    last_submitted_video_ms_.store(-1.0, std::memory_order_release);
    // Advisory events belong to the pre-seek timeline. The NONE sentinel,
    // never -1.0: a small negative is real hold-window data.
    lookahead_cut_ms_.store(kLookaheadNone, std::memory_order_release);
    lookahead_storm_ms_.store(kLookaheadNone, std::memory_order_release);
    // Cut boundaries too: after a seek the media clock jumps, and a stale
    // boundary from the pre-seek timeline could pin the cross-shot gate
    // (backward seek) or fake one (forward seek across a cut).
    std::lock_guard<std::mutex> lk(cut_pts_mtx_);
    cut_pts_.clear();
}

void SharedDepthService::set_lookahead_advisory(double cut_in_ms,
                                                double storm_in_ms,
                                                double cut_pts_ms) {
    static const bool enabled = []() {
        char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
        size_t len = 0;
        bool on = true;
        if (_dupenv_s(&env, &len, "SYLC_LOOKAHEAD") == 0 && env) {
            on = env[0] != '0';   // SYLC_LOOKAHEAD=0 = rollback
            free(env);
        }
        return on;
    }();
    if (!enabled) return;
    lookahead_cut_ms_.store(cut_in_ms, std::memory_order_release);
    lookahead_storm_ms_.store(storm_in_ms, std::memory_order_release);
    lookahead_set_ms_.store(steady_now_ms(), std::memory_order_release);
    if (cut_pts_ms >= 0.0) note_cut_pts(cut_pts_ms);
}

void SharedDepthService::note_cut_pts(double pts_ms) {
    if (!(pts_ms >= 0.0) || !std::isfinite(pts_ms)) return;
    std::lock_guard<std::mutex> lk(cut_pts_mtx_);
    // The same cut arrives from two independent observers (scout ahead of
    // presentation, worker snap behind it) whose media clocks agree only to
    // rounding + one observation: merge anything within half a frame. 21 ms
    // covers 24 fps; at higher rates two REAL cuts are never that close.
    for (double& c : cut_pts_) {
        if (std::abs(c - pts_ms) < 21.0) {
            // Keep the EARLIER boundary: the scout dates the cut at the new
            // shot's first frame, the worker snap can only be at/after it.
            c = std::min(c, pts_ms);
            return;
        }
    }
    if (cut_pts_.size() >= kMaxCutBoundaries)
        cut_pts_.erase(cut_pts_.begin());
    cut_pts_.push_back(pts_ms);
    std::sort(cut_pts_.begin(), cut_pts_.end());
}

void SharedDepthService::note_motion_hints(
        double pts_ms, double frame_ms, int blocks_w, int blocks_h,
        int source_width, int source_height,
        std::vector<int16_t>&& mv_xy, std::vector<uint8_t>&& valid) {
    static const bool enabled = []() {
        char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
        size_t len = 0;
        bool on = true;
        if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_MV_HINTS") == 0 && env) {
            on = env[0] != '0';   // SYLC_SYNTH3D_MV_HINTS=0 = rollback
            free(env);
        }
        return on;
    }();
    if (!enabled) return;
    if (!(pts_ms >= 0.0) || blocks_w <= 0 || blocks_h <= 0 ||
        mv_xy.size() != 2 * static_cast<size_t>(blocks_w) * blocks_h) {
        return;
    }
    std::lock_guard<std::mutex> lk(hints_mtx_);
    // Every presentation surface forwards the decoder packet, while all
    // surfaces using the same graph attach to this ONE process-wide service.
    // Repeated packets for the same source frame must therefore replace their
    // existing slot instead of shrinking the six-frame ring to 6 / clients.
    // Keep the epsilon deliberately tiny: the 21 ms lookup tolerance absorbs
    // clock rounding, but using it here would merge real adjacent frames at
    // 50/60/120 fps.
    constexpr double kSamePtsEpsilonMs = 0.01;
    MotionHints* slot = nullptr;
    for (MotionHints& hint : hints_) {
        if (hint.pts_ms >= 0.0 &&
            std::abs(hint.pts_ms - pts_ms) <= kSamePtsEpsilonMs) {
            slot = &hint;
            break;
        }
    }
    if (!slot) {
        slot = &hints_[hint_write_];
        hint_write_ = (hint_write_ + 1) % kHintRing;
    }
    slot->pts_ms = pts_ms;
    slot->frame_ms = frame_ms > 1.0 ? frame_ms : 41.7;
    slot->blocks_w = blocks_w;
    slot->blocks_h = blocks_h;
    slot->source_width = source_width;
    slot->source_height = source_height;
    slot->mv_xy = std::move(mv_xy);
    slot->valid = std::move(valid);
}

bool SharedDepthService::fetch_motion_hints(
        double pts_ms, MotionHints& out) const {
    if (!(pts_ms >= 0.0)) return false;
    std::lock_guard<std::mutex> lk(hints_mtx_);
    for (const MotionHints& h : hints_) {
        // Même clef média que la garde cross-shot : la moitié d'une frame
        // absorbe les arrondis des différentes horloges.
        if (h.pts_ms >= 0.0 && std::abs(h.pts_ms - pts_ms) < 21.0) {
            out = h;
            return true;
        }
    }
    return false;
}

bool SharedDepthService::cross_shot(double map_video_ms,
                                    double presented_video_ms) const {
    std::lock_guard<std::mutex> lk(cut_pts_mtx_);
    bool gated = false;
    for (size_t i = 0; i < cut_pts_.size();) {
        const double c = cut_pts_[i];
        // Prune by MEDIA distance, not wall time: a boundary 4 s behind the
        // playhead can no longer separate any live map from any presented
        // frame (the worker republishes within ~2 frames of real time).
        if (presented_video_ms >= 0.0 && presented_video_ms - c > 4000.0) {
            cut_pts_.erase(cut_pts_.begin() + static_cast<ptrdiff_t>(i));
            continue;
        }
        if (cross_shot_gate(map_video_ms, presented_video_ms, c))
            gated = true;
        ++i;
    }
    if (gated) gate_frames_.fetch_add(1, std::memory_order_relaxed);
    return gated;
}

double SharedDepthService::latest_presented_cut(
        double presented_video_ms) const {
    if (!std::isfinite(presented_video_ms) || presented_video_ms < 0.0)
        return -1.0;
    std::lock_guard<std::mutex> lk(cut_pts_mtx_);
    double latest = -1.0;
    for (double c : cut_pts_) {
        if (presented_video_ms >= c - kPtsJitterMs)
            latest = std::max(latest, c);
    }
    return latest;
}

int64_t SharedDepthService::last_snap_steady_ms() const {
    return last_snap_ms_.load(std::memory_order_acquire);
}

double SharedDepthService::last_snap_video_ms() const {
    return last_snap_video_ms_.load(std::memory_order_acquire);
}

void SharedDepthService::set_error(const std::string& message) {
    {
        std::lock_guard<std::mutex> lk(meta_mtx_);
        error_ = message;
        for (char& c : error_)
            if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    }
    state_.store(State::Error, std::memory_order_release);
}

std::string SharedDepthService::status() const {
    const State s = state_.load(std::memory_order_acquire);
    const char* state_name =
        s == State::Running ? "running" :
        s == State::Error ? "error" :
        s == State::Stopping ? "off" : "init";
    std::string provider, error;
    {
        std::lock_guard<std::mutex> lk(meta_mtx_);
        provider = provider_;
        error = error_;
    }
    long long age_ms = -1;
    {
        std::lock_guard<std::mutex> lk(output_mtx_);
        if (output_time_.time_since_epoch().count() != 0)
            age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - output_time_).count();
    }
    size_t mv_slots = 0;
    {
        std::lock_guard<std::mutex> lk(hints_mtx_);
        for (const MotionHints& hint : hints_)
            if (hint.pts_ms >= 0.0) ++mv_slots;
    }
    char buf[1536] = {};
    std::snprintf(
        buf, sizeof(buf),
        "state=%s provider=%s views=%d side=%d fps=%.1f "
        "flow_ms=%.1f infer_ms=%.1f "
        "stab_ms=%.1f obs_ms=%.1f guard_ms=%.1f reproj_ms=%.1f "
        "step_ms=%.1f realign_ms=%.1f owner_ms=%.1f "
        "owner_local_ms=%.1f owner_prop_ms=%.1f pack_ms=%.1f owner_gpu=%d "
        "inwait_ms=%.1f reswait_ms=%.1f joinwait_ms=%.1f "
        "cycle_ms=%.1f pass_ms=%.1f taplat_ms=%.1f "
        "grants=%llu submits=%llu duppts=%llu "
        "dempty=%llu dstall=%llu "
        "plateheld=%llu platepost=%llu platecut=%.0f "
        "plate_req=%d plate_arm=%d plate_live=%d "
        "la_stale=%llu la_seen=%llu gate=%llu "
        "source_ms=%.1f update_ms=%.1f age_ms=%lld clients=%d cuts=%llu "
        "cut_bnd=%llu cut_src=%llu cut_depth=%llu depthres=%.3f "
        "depthbase=%.3f "
        "mvslots=%zu "
        "motion=%.3f flow=%.2f alpha=%.3f conf=%.3f stable=%.3f "
        "history=%.2f scene=%.3f crop=%d:%d:%d:%d crop_conf=%.2f "
        "crop_ready=%d grid=%dx%d instance=%llu err=%s",
        state_name, provider.empty() ? "none" : provider.c_str(),
        temporal_views_.load(std::memory_order_acquire),
        width_,
        static_cast<double>(fps_.load(std::memory_order_acquire)),
        static_cast<double>(flow_ms_.load(std::memory_order_acquire)),
        static_cast<double>(infer_ms_.load(std::memory_order_acquire)),
        static_cast<double>(stab_ms_.load(std::memory_order_acquire)),
        static_cast<double>(obs_ms_.load(std::memory_order_acquire)),
        static_cast<double>(guard_ms_.load(std::memory_order_acquire)),
        static_cast<double>(reproj_ms_.load(std::memory_order_acquire)),
        static_cast<double>(step_ms_.load(std::memory_order_acquire)),
        static_cast<double>(realign_ms_.load(std::memory_order_acquire)),
        static_cast<double>(owner_ms_.load(std::memory_order_acquire)),
        static_cast<double>(owner_local_ms_.load(std::memory_order_acquire)),
        static_cast<double>(owner_prop_ms_.load(std::memory_order_acquire)),
        static_cast<double>(pack_ms_.load(std::memory_order_acquire)),
        gpu_owner_.load(std::memory_order_acquire) ? 1 : 0,
        static_cast<double>(inwait_ms_.load(std::memory_order_acquire)),
        static_cast<double>(reswait_ms_.load(std::memory_order_acquire)),
        static_cast<double>(joinwait_ms_.load(std::memory_order_acquire)),
        static_cast<double>(cycle_ms_.load(std::memory_order_acquire)),
        static_cast<double>(pass_ms_.load(std::memory_order_acquire)),
        static_cast<double>(taplat_ms_.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(grants_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(submits_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            duplicate_pts_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            drain_empty_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            drain_stalled_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            plate_held_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            plate_post_.load(std::memory_order_relaxed)),
        static_cast<double>(plate_cut_in_.load(std::memory_order_relaxed)),
        plate_requested_.load(std::memory_order_relaxed) ? 1 : 0,
        plate_armed_.load(std::memory_order_relaxed) ? 1 : 0,
        plate_live_.load(std::memory_order_relaxed) ? 1 : 0,
        static_cast<unsigned long long>(
            la_stale_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            la_seen_.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(
            gate_frames_.load(std::memory_order_relaxed)),
        static_cast<double>(source_dt_ms_.load(std::memory_order_acquire)),
        static_cast<double>(update_dt_ms_.load(std::memory_order_acquire)), age_ms,
        clients_.load(std::memory_order_acquire),
        static_cast<unsigned long long>(cuts_.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            cut_boundary_.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            cut_source_.load(std::memory_order_acquire)),
        static_cast<unsigned long long>(
            cut_depth_.load(std::memory_order_acquire)),
        static_cast<double>(depth_residual_.load(std::memory_order_acquire)),
        static_cast<double>(depth_baseline_.load(std::memory_order_acquire)),
        mv_slots,
        static_cast<double>(motion_.load(std::memory_order_acquire)),
        static_cast<double>(flow_.load(std::memory_order_acquire)),
        static_cast<double>(effective_alpha_.load(std::memory_order_acquire)),
        static_cast<double>(confidence_.load(std::memory_order_acquire)),
        static_cast<double>(stability_.load(std::memory_order_acquire)),
        static_cast<double>(history_support_.load(std::memory_order_acquire)),
        static_cast<double>(scene_change_.load(std::memory_order_acquire)),
        crop_top_.load(std::memory_order_acquire),
        crop_bottom_.load(std::memory_order_acquire),
        crop_source_width_.load(std::memory_order_acquire),
        crop_source_height_.load(std::memory_order_acquire),
        static_cast<double>(crop_confidence_.load(std::memory_order_acquire)),
        crop_ready_.load(std::memory_order_acquire) ? 1 : 0,
        width_, height_,
        static_cast<unsigned long long>(instance_id_),
        error.empty() ? "-" : error.c_str());
    return buf;
}

void SharedDepthService::worker_main() {
    if (model_path_.empty()) {
        set_error("no model configured");
        return;
    }

    DepthEngine engine;
    DepthConfig cfg;
    cfg.model_path = model_path_;
    cfg.ort_dir = ort_dir_;
    cfg.side = width_;         // compatibility field for square-era callers
    cfg.width = width_;
    cfg.height = height_;      // the grid this service was keyed on
    cfg.invert_output = true;  // DA3 true depth is converted to inverse depth
    std::string error;
    if (!engine.init(cfg, error)) {
        set_error(error);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(meta_mtx_);
        provider_ = engine.provider();
        error_.clear();
    }
    state_.store(State::Running, std::memory_order_release);

    // Local copy of the immutable instance grid: every buffer below, and every
    // parallel/flow/tile bound, is derived from it.
    const int width = width_;
    const int height = height_;
    const size_t n = static_cast<size_t>(width) * height;
    const size_t frame_values = 3 * n;
    const int input_views = std::max(1, engine.input_views());
    temporal_views_.store(input_views, std::memory_order_release);

    // Round 6b — two-stage pipeline. Everything the S-stage (post/step/
    // geometry/publish) of map N-1 still needs after map N's prep has begun
    // lives in a per-map context; the two contexts alternate. A context is
    // recycled exactly one iteration after its S-stage completed, and the
    // pending map's flow task is joined at the TOP of each iteration, before
    // any buffer it reads (the recycled context's luma) can be overwritten.
    struct MapCtx {
        std::vector<float> input;              // CHW RGB (geometry + retry)
        std::vector<float> luma;
        std::vector<float> motion;
        std::vector<float> flow_reliability;
        std::vector<float> publish_flow_x, publish_flow_y;
        // Round 7 « flot biface » : la direction causale (prev->cur) part à
        // la prep de CE map ; la direction future (next->cur) ne peut partir
        // qu'à la prep du map SUIVANT ; la fusion se fait dans l'étage S.
        std::vector<float> fwd_x, fwd_y, fwd_q;
        std::vector<float> fut_x, fut_y, fut_q;
        // Phase 1 (04/08) : candidat DÉCODEUR rasterisé à la prep (grille),
        // troisième entrée de la fusion à l'étage S. Vide = pas d'indice.
        std::vector<float> hint_x, hint_y, hint_q;
        std::future<void> flow_task;           // direction causale
        std::future<void> future_task;         // direction future
        double flow_task_ms = 0.0;
        double future_task_ms = 0.0;
        bool flow_ran = false;
        std::array<float, kHistogramBins> histogram{};
        double motion_sum = 0.0;
        float mean_flow = 0.0f;
        float histogram_distance = 0.0f;
        bool source_cut = false;
        // This observation CROSSED a recorded cut boundary (authoritative
        // scout/worker knowledge): the S-stage must guarantee the snap even
        // when both content detectors are blind (similar compositions).
        bool boundary_cut = false;
        bool have_prev = false;
        bool valid_video_time = false;
        double video_time_ms = -1.0;
        float video_time_scale = 1.0f;
        float source_dt_ms = DepthStabilizer::kReferenceDtMs;
        uint64_t seq = 0;
    };
    // Anneau de TROIS contextes : la fusion du map N (dans S) transporte vers
    // la luma de N-1, qui doit donc survivre jusqu'à l'itération N+1 incluse.
    std::array<MapCtx, 3> ctx_ring;
    for (auto& c : ctx_ring) {
        c.input.assign(3 * n, 0.0f);
        c.luma.assign(n, 0.0f);
        c.motion.assign(n, 0.0f);
        c.flow_reliability.assign(n, 0.0f);
    }

    std::vector<float> temporal_input(
        static_cast<size_t>(input_views) * frame_values, 0.0f);
    int temporal_history = 0;
    std::vector<float> confidence(n, 1.0f);
    std::vector<float> full_frame_luma;
    std::vector<float> surface_boundary(n, 0.0f);
    std::vector<float> surface_scratch(n, 0.0f);
    std::vector<uint16_t> q16(n, 0);
    std::vector<uint16_t> realign_scratch;
    bool have_previous_image = false;
    DepthStabilizer stabilizer(n);
    // Confirms the source-histogram scene-cut signal over two consecutive
    // frames before it may suppress flow/re-prime temporal history or drive
    // step()'s internal scene-cut OR -- collapses single-frame flash/fade
    // spikes that the raw histogram threshold alone would mistake for a cut.
    // The depth-residual cut computed inside stabilizer.step() is NOT routed
    // through this gate: it stays instantaneous, exactly as before.
    CutGate cut_gate;
    // Keep the gate's own default from drifting off the stabilizer's OR-threshold:
    // a confirmed cut in the gap between them would re-prime the temporal window
    // without snapping the EMA or stamping the ramp.
    cut_gate.histogram_threshold = stabilizer.scene_cut_threshold;

    int fps_count = 0;
    auto fps_start = Clock::now();
    double flow_ms_accum = 0.0;
    double infer_ms_accum = 0.0;
    double stab_ms_accum = 0.0;
    double obs_ms_accum = 0.0;
    double guard_ms_accum = 0.0;
    double reproj_ms_accum = 0.0;
    double step_ms_accum = 0.0;
    double realign_ms_accum = 0.0;
    double owner_ms_accum = 0.0;
    double owner_local_ms_accum = 0.0;
    double owner_prop_ms_accum = 0.0;
    double pack_ms_accum = 0.0;
    // Blocked-time accumulators (see the atomics in the header). Declared here
    // so finish_map's [&] capture reaches them exactly like the stage timers.
    double inwait_ms_accum = 0.0;
    double reswait_ms_accum = 0.0;
    double joinwait_ms_accum = 0.0;
    double cycle_ms_accum = 0.0;
    // Previous snapshot of the client-side cumulative counters. They are
    // written from the render thread, so the window value is a delta rather
    // than an accumulate-and-clear like the stage timers above.
    uint64_t pass_count_seen = 0;
    uint64_t pass_us_seen = 0;
    uint64_t taplat_count_seen = 0;
    uint64_t taplat_us_seen = 0;
    auto last_cycle_top = Clock::now();
    bool have_cycle_top = false;
    // Two deliberately separate clocks:
    //   video time is the media PTS carried with these exact pixels. It drives
    //   ALL content-temporal behavior (motion + EMA/tone/history/snap);
    //   update time is sampled when the worker actually updates a map. It is
    //   compute observability only and never changes geometry.
    //
    // A source frame can wait in the mailbox while the previous inference
    // runs. Measuring both at worker pickup was the old hidden coupling: a
    // provider change altered perceived motion even when the video frames
    // being compared were unchanged.
    auto last_capture_time = Clock::now();  // fallback when a source has no PTS
    auto last_update_time = Clock::now();
    double last_video_time_ms = -1.0;
    bool have_capture_time = false;
    bool have_video_time = false;
    bool have_update_time = false;
    const int flow_threads = flow_threads_from_env();
    stabilizer.worker_threads = flow_threads;
    {
        // SYLC_SYNTH3D_DEPTHCUT_ADAPT=0 restores the fixed 0.35 depth-cut
        // threshold (disables the ambient-baseline outlier gate).
        char* env = nullptr;   // house idiom (MSVC-safe)
        size_t len = 0;
        if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_DEPTHCUT_ADAPT") == 0 && env) {
            if (env[0] == '0') stabilizer.cut_baseline_gain = 0.0f;
            std::free(env);
        }
    }
    int pending_crop_top = 0;
    int pending_crop_bottom = 0;
    int pending_crop_count = 0;
    int pending_no_crop_count = 0;

    // Pipeline switch: SYLC_DEPTH_PIPELINE=0 keeps ONE code path but waits
    // for each inference right after submitting it (sequential timing) —
    // the behavioural rollback for the round-6b overlap.
    bool pipeline_enabled = true;
    {
        char* env = nullptr;
        size_t len = 0;
        if (_dupenv_s(&env, &len, "SYLC_DEPTH_PIPELINE") == 0 && env) {
            if (std::atoi(env) == 0) pipeline_enabled = false;
            std::free(env);
        }
    }
    InferPipe pipe(engine);
    pipe.out_n_ = n;
    pipe.start();
    uint64_t next_seq = 1;
    bool have_pending = false;          // the PRIOR ctx awaits its S-stage
    int flip = 0;

    // ------------------------------------------------------------------ S
    // Finish one map: consume its inference result, run boundary/refine,
    // fuse the two flow directions (round 7), reproject+step, the (rare,
    // views>1) cut retry, status stores, geometry build and publication.
    // `prev_luma` is the luma of the map BEFORE `done` (the transport
    // target); `next_ctx` is the freshly prepped following map (nullptr in
    // sequential mode) — its histogram carries the zero-latency cut verdict.
    // `reprime_input` is the freshest consumed frame (re-primes the LIVE
    // temporal window after a depth-cut retry).
    // Returns false on a fatal engine error.
    auto finish_map = [&](MapCtx& done,
                          const std::vector<float>& prev_luma,
                          const MapCtx* next_ctx,
                          const std::vector<float>& reprime_input) -> bool {
        InferPipe::Result result;
        const auto reswait_start = Clock::now();
        if (!pipe.wait_result(done.seq, result)) return false;   // stopping
        // WALL time spent blocked here. result.infer_ms is the engine's own
        // measurement, so a queued/late inference was previously invisible.
        reswait_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - reswait_start).count();
        if (result.failed) {
            set_error(result.error);
            return false;
        }
        infer_ms_accum += result.infer_ms;
        const auto joinwait_start = Clock::now();
        if (done.flow_task.valid()) {
            done.flow_task.get();
            flow_ms_accum += done.flow_task_ms;
        }
        if (done.future_task.valid()) {
            done.future_task.get();
            flow_ms_accum += done.future_task_ms;
        }
        // Same asymmetry as above: flow_task_ms is the task's own duration, so
        // time the worker sat waiting on the pool was never counted.
        joinwait_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - joinwait_start).count();
        std::vector<float>& raw = result.raw;
        std::vector<float>& raw_confidence = result.confidence;
        auto& luma = done.luma;
        auto& motion = done.motion;
        auto& input = done.input;
        auto& flow_reliability = done.flow_reliability;
        auto& publish_flow_x = done.publish_flow_x;
        auto& publish_flow_y = done.publish_flow_y;
        const bool valid_video_time = done.valid_video_time;
        const double video_time_ms = done.video_time_ms;
        const float video_time_scale = done.video_time_scale;
        const bool source_cut = done.source_cut;
        const float histogram_distance = done.histogram_distance;
        double motion_sum = done.motion_sum;
        float mean_flow = done.mean_flow;

        const auto obs_start = Clock::now();
        normalize_da3_confidence(raw_confidence, confidence);
        synth3d_surface::compute_boundary(
            raw, luma, width, height, surface_boundary, surface_scratch,
            flow_threads);
        synth3d_surface::refine_observation(
            raw, luma, confidence, surface_boundary,
            width, height, surface_scratch, flow_threads);
        obs_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - obs_start).count();

        // Round 7 — fusion biface. Le transport publié (reproject, plaque),
        // la fiabilité et le masque de mouvement sortent de la fusion des
        // deux directions ; sans direction future (mode séquentiel, drain),
        // un futur miroir à qualité nulle fait dégénérer la fusion en causal
        // pur avec l'auto-confiance résiduelle comme fiabilité.
        if (done.flow_ran && done.fwd_q.size() == n) {
            synth3d_flow::DenseFlow forward;
            forward.x = std::move(done.fwd_x);
            forward.y = std::move(done.fwd_y);
            forward.quality = std::move(done.fwd_q);
            synth3d_flow::DenseFlow future;
            if (done.fut_q.size() == n) {
                future.x = std::move(done.fut_x);
                future.y = std::move(done.fut_y);
                future.quality = std::move(done.fut_q);
            } else {
                future.x.resize(n);
                future.y.resize(n);
                future.quality.assign(n, 0.0f);
                for (size_t k = 0; k < n; ++k) {
                    future.x[k] = -forward.x[k];
                    future.y[k] = -forward.y[k];
                }
            }
            const auto fuse_start = Clock::now();
            // Phase 1 : le candidat décodeur entre dans le même arbitrage
            // par résidu que le futur — rejeté là où il est faux, adopté là
            // où l'encodeur voyait mieux que notre block matching (oracle :
            // ×2 en motion/fast/looming).
            synth3d_flow::DenseFlow hint;
            synth3d_flow::DenseFlow* hint_ptr = nullptr;
            if (done.hint_q.size() == n) {
                hint.x = std::move(done.hint_x);
                hint.y = std::move(done.hint_y);
                hint.quality = std::move(done.hint_q);
                hint_ptr = &hint;
            }
            synth3d_flow::fuse_bidirectional(
                prev_luma, luma, forward, future, width, height,
                done.video_time_scale, publish_flow_x, publish_flow_y,
                flow_reliability, motion, motion_sum, mean_flow,
                flow_threads, hint_ptr);
            flow_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - fuse_start).count();
        }

        // Round 7 — verdict de cut à latence nulle : un vrai cut à `done` =
        // pic de distance (prev, done) ET calme (done, next) — le flash, lui,
        // repart aussi haut vers next et reste rejeté en une frame.
        bool cut_ahead = false;
        if (next_ctx != nullptr && done.have_prev) {
            float d_next = 0.0f;
            for (int b = 0; b < kHistogramBins; ++b)
                d_next += std::abs(next_ctx->histogram[
                    static_cast<size_t>(b)] -
                    done.histogram[static_cast<size_t>(b)]);
            d_next *= 0.5f;
            cut_ahead =
                histogram_distance >= stabilizer.scene_cut_threshold &&
                d_next < 0.5f * stabilizer.scene_cut_threshold;
        }
        const bool source_cut_step = source_cut || cut_ahead;

        const auto guard_start = Clock::now();
        if (done.have_prev) {
            // Motion at a silhouette is often registered one or two pixels on
            // only one side. Expand it along the guard band so the complete
            // neck/shoulder boundary chooses the short-memory path together.
            // Phase 2 (04/08) : la portée directionnelle anti-traînée ne
            // s'active qu'avec un transport fusionné disponible.
            static const bool trail_erode = []() {
                char* env = nullptr;   // house idiom (MSVC-safe)
                size_t len = 0;
                bool on = true;
                if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_TRAIL_ERODE") == 0 &&
                    env) {
                    on = env[0] != '0';   // =0 : expansion 3x3 historique
                    free(env);
                }
                return on;
            }();
            synth3d_flow::expand_boundary_motion(
                motion, surface_boundary, publish_flow_x, publish_flow_y,
                width, height, surface_scratch, flow_threads,
                trail_erode && publish_flow_x.size() == n &&
                    publish_flow_y.size() == n);
        }
        guard_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - guard_start).count();
        const auto stab_start = Clock::now();
        float update_dt_ms = DepthStabilizer::kReferenceDtMs;
        if (have_update_time)
            update_dt_ms = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    stab_start - last_update_time).count());
        last_update_time = stab_start;
        have_update_time = true;
        // The stabilizer clocks belong to the map being STEPPED, not to the
        // most recently consumed frame (they may differ by one map now).
        stabilizer.set_source_dt_ms(done.source_dt_ms);
        // Compute cadence is exposed in status, but DepthStabilizer keeps it
        // out of state evolution; source_dt_ms (video PTS) owns that math.
        stabilizer.set_update_dt_ms(update_dt_ms);
        update_dt_ms_.store(
            stabilizer.update_dt_ms(), std::memory_order_release);
        source_dt_ms_.store(
            stabilizer.source_dt_ms(), std::memory_order_release);
        // Reproject moved OUT of the flow task (round 6b): the stabilizer
        // state must only be transported between step(N-1) and step(N); in
        // the pipelined order the flow task can run while the previous map
        // is still unstepped. On a look-ahead-confirmed cut the transport is
        // cross-shot by construction — skipped, step() snaps right below.
        const auto reproj_start = Clock::now();
        if (!cut_ahead &&
            publish_flow_x.size() == n && publish_flow_y.size() == n)
            stabilizer.reproject(
                publish_flow_x.data(), publish_flow_y.data(),
                flow_reliability.data(), width, height);
        reproj_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - reproj_start).count();
        const auto step_start = Clock::now();
        // LOOK-AHEAD ADVISORY (two-filter scout, spec 2026-08-03): a motion
        // storm was OBSERVED in the decoded future within ~2.5 frames of the
        // presented position. The oracle proved storms have no content ramp
        // (lead=0 even at full res) so the reactive path pays its first 1-2
        // frames at +170% residual — close that window by raising the motion
        // floor NOW: the existing motion gate then runs its fast-adapt path
        // from the storm's first frame. `motion` is rewritten every cycle by
        // the biface fusion, so the in-place floor cannot leak forward.
        {
            const double la_storm =
                lookahead_storm_ms_.load(std::memory_order_acquire);
            const int64_t la_set =
                lookahead_set_ms_.load(std::memory_order_acquire);
            // -200..0 = the scout's hold window (sentinel -1e9 far below):
            // the storm's OPENING frames — exactly the +170% residual window
            // — stay floored even when the pump's tick quantization jumps
            // the delay past zero between two refreshes.
            if (la_storm > -200.0 && la_set >= 0 &&
                steady_now_ms() - la_set < 500 &&
                la_storm <= 2.5 * std::max(1.0f, stabilizer.source_dt_ms()) &&
                done.have_prev) {
                for (auto& mv : motion) mv = std::max(mv, 0.35f);
            }
        }
        // Phase 2 (04/08) — plancher LOOMING : l'expansion (marche face
        // caméra, dolly, zoom) est lue DANS la divergence du transport
        // fusionné — plus besoin d'un avis externe pour ce régime. Comme
        // l'avis tempête, un plancher de mouvement gradué déclenche le
        // chemin mémoire-courte du stabilizer sur tout le champ pendant que
        // l'image entière se déforme (l'oracle a montré que la translation
        // pure y sous-estime le mouvement réel). Seuil = DIV_LOOM de
        // l'oracle (0.0035/texel à cadence source), rampe pleine à 2x.
        // SYLC_SYNTH3D_LOOM_FLOOR=0 = rollback.
        {
            static const bool loom_on = []() {
                char* env = nullptr;   // house idiom (MSVC-safe)
                size_t len = 0;
                bool on = true;
                if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_LOOM_FLOOR") == 0 &&
                    env) {
                    on = env[0] != '0';
                    free(env);
                }
                return on;
            }();
            if (loom_on && done.have_prev &&
                publish_flow_x.size() == n && publish_flow_y.size() == n) {
                const double div = synth3d_flow::mean_divergence(
                    publish_flow_x, publish_flow_y, width, height) *
                    (41.7 / std::max(4.0f, done.source_dt_ms));
                const float loom = clamp01(
                    (static_cast<float>(std::abs(div)) - 0.0035f) / 0.0035f);
                if (loom > 0.0f) {
                    const float floor_value = 0.30f * loom;
                    for (auto& mv : motion)
                        mv = std::max(mv, floor_value);
                }
            }
        }
        // A crossed boundary is authoritative: raise the signal to the
        // stabilizer's own threshold so its internal OR fires even when the
        // two shots' histograms AND depth statistics are deceptively similar
        // (shot/reverse-shot faces) — the case where blending would imprint
        // the old shot's contours into the new one for the EMA's half-life.
        float scene_signal = source_cut_step ? histogram_distance : 0.0f;
        if (done.boundary_cut)
            scene_signal = boundary_scene_signal(
                histogram_distance, stabilizer.scene_cut_threshold);
        bool cut = stabilizer.step(
            raw.data(), q16.data(),
            done.have_prev ? motion.data() : nullptr,
            scene_signal, confidence.data(),
            surface_boundary.data());
        // Snapshot the cause before the optional temporal-window retry resets
        // the stabilizer. Previously that retry made a real geometry snap
        // disappear from `cuts`, preventing any evidence-based threshold work.
        const bool detected_depth_cut = stabilizer.last_depth_cut();
        const bool detected_scene_cut = stabilizer.last_scene_cut();
        const float detected_depth_residual =
            stabilizer.last_depth_residual();
        depth_baseline_.store(stabilizer.residual_baseline(),
                              std::memory_order_release);
        depth_residual_.store(detected_depth_residual,
                              std::memory_order_release);
        stab_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - stab_start).count();
        step_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - step_start).count();
        if (cut && input_views > 1 && !source_cut_step) {
            // The depth residual found a cut that the source histogram missed.
            // Re-run this rare transition with a clean temporal window so the
            // published map cannot contain cross-shot attention ghosts.
            // Pipelined adaptation: the retry goes through the infer pipe with
            // its own sequence (one bubble on a rare event); the LIVE temporal
            // window is re-primed with the freshest consumed frame (also
            // post-cut content — the cut is at THIS map).
            std::vector<float> retry_window(
                static_cast<size_t>(input_views) * frame_values);
            for (int view = 0; view < input_views; ++view)
                std::copy(
                    input.begin(), input.end(),
                    retry_window.begin() +
                        static_cast<size_t>(view) * frame_values);
            const uint64_t retry_seq = next_seq++;
            pipe.submit(retry_window, retry_seq);
            InferPipe::Result retry;
            if (!pipe.wait_result(retry_seq, retry)) return false;
            if (retry.failed) {
                set_error(retry.error);
                return false;
            }
            infer_ms_accum += retry.infer_ms;
            raw = std::move(retry.raw);
            raw_confidence = std::move(retry.confidence);
            normalize_da3_confidence(raw_confidence, confidence);
            synth3d_surface::compute_boundary(
                raw, luma, width, height,
                surface_boundary, surface_scratch, flow_threads);
            synth3d_surface::refine_observation(
                raw, luma, confidence, surface_boundary,
                width, height, surface_scratch, flow_threads);
            stabilizer.reset();
            const auto retry_stab_start = Clock::now();
            cut = stabilizer.step(
                raw.data(), q16.data(), nullptr, 0.0f, confidence.data(),
                surface_boundary.data());
            stab_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - retry_stab_start).count();
            step_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - retry_stab_start).count();
            // reset() unprimes the stabilizer, so this step() always takes the
            // priming path and returns cut=false -- yet the EMA and tone range
            // were both just fully replaced, a real geometry teleport.
            // Record both the ramp event and its honest diagnostic cause.
            cuts_.fetch_add(1, std::memory_order_acq_rel);
            cut_depth_.fetch_add(1, std::memory_order_acq_rel);
            last_snap_ms_.store(steady_now_ms(), std::memory_order_release);
            last_snap_video_ms_.store(
                valid_video_time ? video_time_ms : -1.0,
                std::memory_order_release);
            // Depth-residual retry = a real cut at THIS map's observation:
            // record its boundary for the cross-shot gate too.
            if (valid_video_time) note_cut_pts(video_time_ms);
            for (int view = 0; view < input_views; ++view)
                std::copy(
                    reprime_input.begin(), reprime_input.end(),
                    temporal_input.begin() +
                        static_cast<size_t>(view) * frame_values);
            temporal_history = 1;
        }

        // Velocity-normalized in SOURCE time so the status-line motion=
        // reading describes the video, not the provider's compute cadence.
        motion_.store(
            done.have_prev
                ? static_cast<float>(motion_sum / n) / video_time_scale : 0.0f,
            std::memory_order_release);
        flow_.store(mean_flow, std::memory_order_release);
        effective_alpha_.store(stabilizer.last_effective_alpha(),
                               std::memory_order_release);
        scene_change_.store(histogram_distance, std::memory_order_release);
        confidence_.store(stabilizer.last_confidence(),
                          std::memory_order_release);
        stability_.store(stabilizer.last_stability(),
                         std::memory_order_release);
        history_support_.store(stabilizer.last_history_support(),
                               std::memory_order_release);
        suggested_convergence_.store(stabilizer.suggested_convergence(),
                                     std::memory_order_release);
        if (cut) {
            cuts_.fetch_add(1, std::memory_order_acq_rel);
            if (done.boundary_cut)
                cut_boundary_.fetch_add(1, std::memory_order_acq_rel);
            else if (detected_scene_cut || source_cut_step)
                cut_source_.fetch_add(1, std::memory_order_acq_rel);
            else if (detected_depth_cut)
                cut_depth_.fetch_add(1, std::memory_order_acq_rel);
            last_snap_ms_.store(steady_now_ms(), std::memory_order_release);
            last_snap_video_ms_.store(
                valid_video_time ? video_time_ms : -1.0,
                std::memory_order_release);
            // Cross-shot boundary from the worker's own detection: covers a
            // cut the scout missed (dedupes with the scout's pts otherwise).
            // A boundary-driven cut is already in the list — re-noting it at
            // this observation's pts (up to a frame later) would just add a
            // second, later-dated boundary for the same cut.
            if (valid_video_time && !done.boundary_cut)
                note_cut_pts(video_time_ms);
        }

        {
            std::vector<uint16_t> ownership_geometry;
            const auto geometry_start = Clock::now();
            // Contour re-anchoring (04/08): snap the stabilized map's layer
            // edges onto the image edges before ownership analysis, so the
            // near-halo a monocular model paints over the background beside
            // a silhouette never reaches the warp (or blocks the plate from
            // learning the true background there). See realign_contours.
            static const bool realign_on = []() {
                char* env = nullptr;   // house idiom (MSVC-safe)
                size_t len = 0;
                bool on = true;
                if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_REALIGN") == 0 &&
                    env) {
                    on = env[0] != '0';   // SYLC_SYNTH3D_REALIGN=0 = rollback
                    free(env);
                }
                return on;
            }();
            if (realign_on) {
                const auto realign_start = Clock::now();
                synth3d_surface::realign_contours(
                    q16, luma, width, height, realign_scratch, flow_threads);
                realign_ms_accum += std::chrono::duration<double, std::milli>(
                    Clock::now() - realign_start).count();
            }
            const bool gpu_ownership =
                gpu_ownership_enabled_.load(std::memory_order_acquire);
            auto published = std::make_shared<GeometryFrame>();
            published->gpu_ownership = gpu_ownership;
            gpu_owner_.store(gpu_ownership, std::memory_order_release);
            const bool has_flow = publish_flow_x.size() == n &&
                                  publish_flow_y.size() == n;
            const auto pack_start = Clock::now();
            double pack_elapsed_ms = 0.0;
            if (gpu_ownership) {
                // The renderer consumes these filterable UNORM planes without
                // a GPU->CPU round trip. Keeping transport separate preserves
                // the existing plate path while ownership/safety is completed
                // by DirectCompute on the renderer's own D3D11 device.
                published->surface_rgba16.resize(4 * n);
                published->rgb_rgba16.resize(4 * n);
                published->transport_rgba16.resize(4 * n);
                parallel_chunks(static_cast<int>(n), flow_threads,
                                [&](int, int i_begin, int i_end) {
                for (size_t i = static_cast<size_t>(i_begin);
                     i < static_cast<size_t>(i_end); ++i) {
                    const size_t out = 4 * i;
                    published->surface_rgba16[out + 0] = q16[i];
                    published->surface_rgba16[out + 1] = static_cast<uint16_t>(
                        clamp01(luma[i]) * 65535.0f + 0.5f);
                    published->surface_rgba16[out + 2] = static_cast<uint16_t>(
                        clamp01(confidence[i]) * 65535.0f + 0.5f);
                    published->surface_rgba16[out + 3] = static_cast<uint16_t>(
                        clamp01(surface_boundary[i]) * 65535.0f + 0.5f);
                    published->rgb_rgba16[out + 0] = static_cast<uint16_t>(
                        clamp01(input[i] * 0.229f + 0.485f) * 65535.0f + 0.5f);
                    published->rgb_rgba16[out + 1] = static_cast<uint16_t>(
                        clamp01(input[n + i] * 0.224f + 0.456f) * 65535.0f + 0.5f);
                    published->rgb_rgba16[out + 2] = static_cast<uint16_t>(
                        clamp01(input[2 * n + i] * 0.225f + 0.406f) * 65535.0f + 0.5f);
                    published->rgb_rgba16[out + 3] = 0;
                    const float fx = has_flow ? publish_flow_x[i] : 0.0f;
                    const float fy = has_flow ? publish_flow_y[i] : 0.0f;
                    const float rel = has_flow ? flow_reliability[i] : 1.0f;
                    published->transport_rgba16[out + 0] = static_cast<uint16_t>(
                        clamp01(fx / 128.0f + 0.5f) * 65535.0f + 0.5f);
                    published->transport_rgba16[out + 1] = static_cast<uint16_t>(
                        clamp01(fy / 128.0f + 0.5f) * 65535.0f + 0.5f);
                    published->transport_rgba16[out + 2] = static_cast<uint16_t>(
                        clamp01(rel) * 65535.0f + 0.5f);
                    published->transport_rgba16[out + 3] = 0;
                }
                });
                pack_elapsed_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - pack_start).count();
            } else {
                const auto owner_start = Clock::now();
                double owner_local_ms = 0.0;
                double owner_prop_ms = 0.0;
                synth3d_surface::build_geometry_map(
                    q16, input, luma, confidence, surface_boundary,
                    width, height, ownership_geometry,
                    surface_scratch, flow_threads,
                    &owner_local_ms, &owner_prop_ms);
                owner_ms_accum += std::chrono::duration<double, std::milli>(
                    Clock::now() - owner_start).count();
                owner_local_ms_accum += owner_local_ms;
                owner_prop_ms_accum += owner_prop_ms;
                published->geometry.resize(kGeometryChannels * n);
                const auto cpu_pack_start = Clock::now();
                parallel_chunks(static_cast<int>(n), flow_threads,
                                [&](int, int i_begin, int i_end) {
                for (size_t i = static_cast<size_t>(i_begin);
                     i < static_cast<size_t>(i_end); ++i) {
                    const float raw_depth = ownership_geometry[4 * i + 0] / 65535.0f;
                    const float owned_depth = ownership_geometry[4 * i + 1] / 65535.0f;
                    const float safety = ownership_geometry[4 * i + 2] / 65535.0f;
                    const float repair = ownership_geometry[4 * i + 3] / 65535.0f;
                    const size_t out = kGeometryChannels * i;
                    published->geometry[out + 0] = static_cast<uint16_t>(
                        clamp01(raw_depth + repair * (owned_depth - raw_depth)) *
                        65535.0f + 0.5f);
                    published->geometry[out + 1] = static_cast<uint16_t>(
                        clamp01(safety + 0.72f * repair) * 65535.0f + 0.5f);
                    const float fx = has_flow ? publish_flow_x[i] : 0.0f;
                    const float fy = has_flow ? publish_flow_y[i] : 0.0f;
                    const float rel = has_flow ? flow_reliability[i] : 1.0f;
                    published->geometry[out + 2] = static_cast<uint16_t>(
                        clamp01(fx / 128.0f + 0.5f) * 65535.0f + 0.5f);
                    published->geometry[out + 3] = static_cast<uint16_t>(
                        clamp01(fy / 128.0f + 0.5f) * 65535.0f + 0.5f);
                    published->geometry[out + 4] = static_cast<uint16_t>(
                        clamp01(rel) * 65535.0f + 0.5f);
                    published->geometry[out + 5] = 0;
                }
                });
                pack_elapsed_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - cpu_pack_start).count();
            }
            pack_ms_accum += pack_elapsed_ms;
            stab_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - geometry_start).count();
            std::lock_guard<std::mutex> lk(output_mtx_);
            latest_ = std::move(published);
            ++output_sequence_;
            output_time_ = Clock::now();
            // Shot identity (04/08): the map travels with the media PTS of
            // the observation it was computed from. This is what lets the
            // renderer test cross-shot state instead of racing deadlines.
            output_video_ms_ =
                valid_video_time ? video_time_ms : -1.0;
        }

        ++fps_count;
        const auto now = Clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 2.0) {
            fps_.store(static_cast<float>(fps_count / elapsed),
                       std::memory_order_release);
            flow_ms_.store(static_cast<float>(flow_ms_accum / fps_count),
                           std::memory_order_release);
            infer_ms_.store(static_cast<float>(infer_ms_accum / fps_count),
                            std::memory_order_release);
            stab_ms_.store(static_cast<float>(stab_ms_accum / fps_count),
                          std::memory_order_release);
            obs_ms_.store(static_cast<float>(obs_ms_accum / fps_count),
                          std::memory_order_release);
            guard_ms_.store(static_cast<float>(guard_ms_accum / fps_count),
                            std::memory_order_release);
            reproj_ms_.store(static_cast<float>(reproj_ms_accum / fps_count),
                             std::memory_order_release);
            step_ms_.store(static_cast<float>(step_ms_accum / fps_count),
                           std::memory_order_release);
            realign_ms_.store(static_cast<float>(realign_ms_accum / fps_count),
                              std::memory_order_release);
            owner_ms_.store(static_cast<float>(owner_ms_accum / fps_count),
                            std::memory_order_release);
            owner_local_ms_.store(
                static_cast<float>(owner_local_ms_accum / fps_count),
                std::memory_order_release);
            owner_prop_ms_.store(
                static_cast<float>(owner_prop_ms_accum / fps_count),
                std::memory_order_release);
            pack_ms_.store(static_cast<float>(pack_ms_accum / fps_count),
                           std::memory_order_release);
            inwait_ms_.store(static_cast<float>(inwait_ms_accum / fps_count),
                             std::memory_order_release);
            reswait_ms_.store(static_cast<float>(reswait_ms_accum / fps_count),
                              std::memory_order_release);
            joinwait_ms_.store(
                static_cast<float>(joinwait_ms_accum / fps_count),
                std::memory_order_release);
            cycle_ms_.store(static_cast<float>(cycle_ms_accum / fps_count),
                            std::memory_order_release);
            // Client-side counters: cumulative and written from other threads,
            // so the window value is a delta against the previous snapshot.
            // Single reader (this worker), hence no lock.
            const uint64_t pc = pass_count_.load(std::memory_order_relaxed);
            const uint64_t ps = pass_us_sum_.load(std::memory_order_relaxed);
            if (pc > pass_count_seen) {
                pass_ms_.store(
                    static_cast<float>((ps - pass_us_seen) /
                                       static_cast<double>(pc - pass_count_seen)
                                       * 1e-3),
                    std::memory_order_release);
            }
            pass_count_seen = pc;
            pass_us_seen = ps;

            const uint64_t tc = taplat_count_.load(std::memory_order_relaxed);
            const uint64_t ts = taplat_us_sum_.load(std::memory_order_relaxed);
            if (tc > taplat_count_seen) {
                taplat_ms_.store(
                    static_cast<float>((ts - taplat_us_seen) /
                                       static_cast<double>(tc - taplat_count_seen)
                                       * 1e-3),
                    std::memory_order_release);
            }
            taplat_count_seen = tc;
            taplat_us_seen = ts;

            fps_count = 0;
            flow_ms_accum = 0.0;
            infer_ms_accum = 0.0;
            stab_ms_accum = 0.0;
            obs_ms_accum = 0.0;
            guard_ms_accum = 0.0;
            reproj_ms_accum = 0.0;
            step_ms_accum = 0.0;
            realign_ms_accum = 0.0;
            owner_ms_accum = 0.0;
            owner_local_ms_accum = 0.0;
            owner_prop_ms_accum = 0.0;
            pack_ms_accum = 0.0;
            inwait_ms_accum = 0.0;
            reswait_ms_accum = 0.0;
            joinwait_ms_accum = 0.0;
            cycle_ms_accum = 0.0;
            fps_start = now;
        }
        return true;
    };

    while (!stop_.load(std::memory_order_acquire)) {
        MapCtx& cur = ctx_ring[static_cast<size_t>(flip)];
        MapCtx& prior = ctx_ring[static_cast<size_t>((flip + 2) % 3)];
        MapCtx& grand = ctx_ring[static_cast<size_t>((flip + 1) % 3)];

        // Recyclage sûr : les tâches de flot lancées pour CE contexte il y a
        // trois itérations sont normalement jointes dans son étage S ; un
        // drain (seek) a pu les laisser vivantes. Jamais d'écriture sous une
        // lecture concurrente.
        const auto recycle_start = Clock::now();
        if (cur.flow_task.valid()) cur.flow_task.get();
        if (cur.future_task.valid()) cur.future_task.get();
        joinwait_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - recycle_start).count();

        Clock::time_point capture_time{};
        double video_time_ms = -1.0;
        int source_width = 0;
        int source_height = 0;
        const auto inwait_start = Clock::now();
        {
            std::unique_lock<std::mutex> lk(input_mtx_);
            input_cv_.wait_for(lk, std::chrono::milliseconds(100), [this] {
                return stop_.load(std::memory_order_acquire) || input_fresh_;
            });
            if (stop_.load(std::memory_order_acquire)) break;
            // A timeout produced no map: it must not pollute the per-map
            // averages, so nothing is accumulated on this path.
            if (!input_fresh_) continue;
            cur.input.swap(input_mailbox_);
            full_frame_luma.swap(input_full_frame_luma_);
            capture_time = input_capture_time_;
            video_time_ms = input_video_time_ms_;
            source_width = input_source_width_;
            source_height = input_source_height_;
            input_fresh_ = false;
        }

        const auto stage1_start = Clock::now();
        // Idle time attributable to THIS map: the worker had nothing to chew on
        // until a submission landed. Large here == the engine is out of phase
        // with the frame grid, not short of compute.
        inwait_ms_accum += std::chrono::duration<double, std::milli>(
            stage1_start - inwait_start).count();
        // Consume-to-consume: the true production period, and the yardstick the
        // stage+wait breakdown must add up to.
        if (have_cycle_top)
            cycle_ms_accum += std::chrono::duration<double, std::milli>(
                stage1_start - last_cycle_top).count();
        last_cycle_top = stage1_start;
        have_cycle_top = true;

        const bool reset = reset_stabilizer_.exchange(false, std::memory_order_acq_rel);
        if (reset) {
            // A seek moved the content: the in-flight PRE-seek map must not
            // be published at the new position. Its result is consumed and
            // dropped so the pipe stays in lock-step; its flow tasks are
            // joined so the drained context can be recycled safely.
            if (have_pending) {
                if (prior.flow_task.valid()) prior.flow_task.get();
                if (prior.future_task.valid()) prior.future_task.get();
                InferPipe::Result discard;
                if (!pipe.wait_result(prior.seq, discard)) break;
                have_pending = false;
            }
            stabilizer.reset();
            have_previous_image = false;
            have_capture_time = false;
            have_video_time = false;
            have_update_time = false;
            temporal_history = 0;
            pending_crop_top = 0;
            pending_crop_bottom = 0;
            pending_crop_count = 0;
            pending_no_crop_count = 0;
            crop_top_.store(0, std::memory_order_release);
            crop_bottom_.store(0, std::memory_order_release);
            crop_source_width_.store(0, std::memory_order_release);
            crop_source_height_.store(0, std::memory_order_release);
            crop_confidence_.store(0.0f, std::memory_order_release);
            crop_ready_.store(false, std::memory_order_release);
            last_snap_ms_.store(steady_now_ms(), std::memory_order_release);
            last_snap_video_ms_.store(
                std::isfinite(video_time_ms) && video_time_ms >= 0.0
                    ? video_time_ms : -1.0,
                std::memory_order_release);
        }

        if (capture_time.time_since_epoch().count() == 0)
            capture_time = Clock::now();

        // Publish the coded frame dimensions even when no encoded matte exists.
        // Pre-cropped cinema masters (for example 1920x800) otherwise report
        // crop=0:0:0:0 forever and the player cannot select a rectangular graph.
        crop_source_width_.store(
            std::max(0, source_width), std::memory_order_release);
        crop_source_height_.store(
            std::max(0, source_height), std::memory_order_release);

        float source_dt_ms = DepthStabilizer::kReferenceDtMs;
        const bool valid_video_time =
            std::isfinite(video_time_ms) && video_time_ms >= 0.0;
        const double video_delta_ms =
            valid_video_time && have_video_time
                ? video_time_ms - last_video_time_ms : -1.0;
        if (valid_video_time && have_video_time) {
            // A valid PTS pair owns the content clock completely. Duplicate
            // frames (delta == 0) and an unannounced backward discontinuity
            // must never leak presentation/compute wall time into geometry.
            // The stabilizer's 4 ms floor is the closest representable
            // no-progress interval; a real seek resets the history separately.
            source_dt_ms = video_delta_ms > 0.0
                ? static_cast<float>(video_delta_ms) : 4.0f;
        } else if (have_capture_time) {
            // Sources without trustworthy PTS retain the old presentation
            // capture fallback. It is still distinct from worker compute time.
            source_dt_ms = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    capture_time - last_capture_time).count());
        }
        // Cross-shot boundary consumption (04/08 round 2): does THIS
        // observation cross a recorded cut boundary? Same math as the
        // renderer gate, applied to (previous observation, this observation).
        // Consumed exactly once by construction: next cycle the previous pts
        // is already at/after the boundary. Checked BEFORE last_video_time_ms
        // is overwritten below.
        cur.boundary_cut = false;
        if (valid_video_time && have_video_time) {
            std::lock_guard<std::mutex> lk(cut_pts_mtx_);
            for (double c : cut_pts_) {
                if (cross_shot_gate(last_video_time_ms, video_time_ms, c)) {
                    cur.boundary_cut = true;
                    break;
                }
            }
        }
        last_capture_time = capture_time;
        have_capture_time = true;
        last_video_time_ms = valid_video_time ? video_time_ms : -1.0;
        have_video_time = valid_video_time;
        // The stabilizer is clocked at the S-stage on the map actually being
        // stepped; the prep stage only RECORDS this map's clocks. The clamp is
        // mirrored here so flow gating uses the same scale step() will see.
        cur.source_dt_ms = std::max(4.0f, std::min(500.0f, source_dt_ms));
        cur.valid_video_time = valid_video_time;
        cur.video_time_ms = video_time_ms;
        cur.seq = next_seq++;
        cur.have_prev = have_previous_image;
        cur.publish_flow_x.clear();
        cur.publish_flow_y.clear();
        cur.fwd_q.clear();
        cur.fut_q.clear();
        cur.hint_q.clear();
        cur.flow_ran = false;
        cur.flow_task_ms = 0.0;
        cur.future_task_ms = 0.0;
        cur.mean_flow = 0.0f;
        const float video_time_scale =
            cur.source_dt_ms / DepthStabilizer::kReferenceDtMs;
        cur.video_time_scale = video_time_scale;
        // Phase 1 (04/08) : indice décodeur pour CETTE observation, clef
        // pts (même horloge média que la garde cross-shot). L'échelle
        // temporelle ramène « par frame d'affichage » à « par observation »
        // (le worker peut consommer une frame sur deux).
        if (valid_video_time) {
            MotionHints hints;
            if (fetch_motion_hints(video_time_ms, hints)) {
                synth3d_flow::rasterize_motion_hints(
                    hints.mv_xy.data(),
                    hints.valid.empty() ? nullptr : hints.valid.data(),
                    hints.blocks_w, hints.blocks_h,
                    hints.source_width > 0 ? hints.source_width
                                           : std::max(1, source_width),
                    hints.source_height > 0 ? hints.source_height
                                            : std::max(1, source_height),
                    width, height,
                    static_cast<float>(cur.source_dt_ms / hints.frame_ms),
                    cur.hint_x, cur.hint_y, cur.hint_q);
            }
        }

        // Reference bindings keep the transplanted blocks below verbatim:
        // this map's buffers live in `cur`, the previous map's luma in `prior`
        // (unread on the very first map — the motion branch is guarded).
        auto& input = cur.input;
        auto& luma = cur.luma;
        auto& motion = cur.motion;
        const std::vector<float>& previous_luma = prior.luma;

        std::array<float, kHistogramBins> histogram{};
        double motion_sum = 0.0;
        {
            // Per-pixel writes are independent; the histogram is a count
            // accumulation (integers stored in float, all << 2^24), so the
            // per-chunk bins reduce to EXACTLY the sequential totals. The
            // motion sum is a diagnostic/threshold input where chunk-order
            // rounding (last bit) is immaterial.
            const int total = static_cast<int>(n);
            const int chunks = parallel_chunk_count(total, flow_threads);
            std::vector<std::array<float, kHistogramBins>> chunk_hist(
                static_cast<size_t>(chunks));
            std::vector<double> chunk_motion(static_cast<size_t>(chunks), 0.0);
            parallel_chunks(total, flow_threads,
                            [&](int chunk_id, int begin, int end) {
                auto& hist = chunk_hist[static_cast<size_t>(chunk_id)];
                hist.fill(0.0f);
                double local_motion = 0.0;
                for (int ii = begin; ii < end; ++ii) {
                    const size_t i = static_cast<size_t>(ii);
                    // Undo ImageNet normalization and derive BT.709 luma. The
                    // result remains in [0..1] because the prep shader already
                    // saturates RGB.
                    const float r = input[i] * 0.229f + 0.485f;
                    const float g = input[n + i] * 0.224f + 0.456f;
                    const float b = input[2 * n + i] * 0.225f + 0.406f;
                    const float y = clamp01(
                        0.2126f * r + 0.7152f * g + 0.0722f * b);
                    luma[i] = y;
                    const int bin = std::min(kHistogramBins - 1,
                                             static_cast<int>(y * kHistogramBins));
                    hist[bin] += 1.0f;
                    if (have_previous_image) {
                        const float d = std::abs(y - previous_luma[i]);
                        motion[i] = clamp01(d);
                        local_motion += d;
                    } else {
                        motion[i] = 0.0f;
                    }
                }
                chunk_motion[static_cast<size_t>(chunk_id)] = local_motion;
            });
            for (int c = 0; c < chunks; ++c) {
                motion_sum += chunk_motion[static_cast<size_t>(c)];
                for (int bin = 0; bin < kHistogramBins; ++bin)
                    histogram[static_cast<size_t>(bin)] +=
                        chunk_hist[static_cast<size_t>(c)]
                                  [static_cast<size_t>(bin)];
            }
        }

        // Detect only encoded, symmetric horizontal mattes. The decision is
        // consolidated across several inference observations so a dark shot,
        // fade or transient subtitle cannot select a new model shape.
        if (source_width > 0 && source_height > 0) {
            // RGB is sampled from the active ROI for inference. The prep
            // target's alpha channel independently carries full-frame luma so
            // a rectangular service can still see an IMAX opening (or a new
            // matte) outside its current crop without another GPU pass/readback.
            const std::vector<float>& aspect_luma =
                full_frame_luma.size() == n ? full_frame_luma : luma;
            const LetterboxCandidate crop =
                detect_horizontal_letterbox(aspect_luma, width, height);
            if (crop.valid) {
                pending_no_crop_count = 0;
                const int tolerance = std::max(2, height / 100);
                if (std::abs(crop.top - pending_crop_top) <= tolerance &&
                    std::abs(crop.bottom - pending_crop_bottom) <= tolerance) {
                    ++pending_crop_count;
                } else {
                    pending_crop_top = crop.top;
                    pending_crop_bottom = crop.bottom;
                    pending_crop_count = 1;
                    crop_ready_.store(false, std::memory_order_release);
                }
                crop_confidence_.store(
                    std::min(1.0f, pending_crop_count / 8.0f),
                    std::memory_order_release);
                if (pending_crop_count >= 8) {
                    const int source_top = static_cast<int>(std::lround(
                        static_cast<double>(pending_crop_top) *
                        source_height / height));
                    const int source_bottom = static_cast<int>(std::lround(
                        static_cast<double>(pending_crop_bottom) *
                        source_height / height));
                    crop_top_.store(source_top, std::memory_order_release);
                    crop_bottom_.store(source_bottom, std::memory_order_release);
                    crop_source_width_.store(
                        source_width, std::memory_order_release);
                    crop_source_height_.store(
                        source_height, std::memory_order_release);
                    crop_ready_.store(true, std::memory_order_release);
                }
            } else {
                pending_crop_count = 0;
                pending_crop_top = 0;
                pending_crop_bottom = 0;
                ++pending_no_crop_count;
                crop_confidence_.store(0.0f, std::memory_order_release);
                if (pending_no_crop_count >= 8) {
                    crop_top_.store(0, std::memory_order_release);
                    crop_bottom_.store(0, std::memory_order_release);
                    crop_ready_.store(true, std::memory_order_release);
                } else {
                    crop_ready_.store(false, std::memory_order_release);
                }
            }
        }
        for (float& v : histogram) v /= static_cast<float>(n);
        cur.histogram = histogram;

        float histogram_distance = 0.0f;
        if (have_previous_image) {
            for (int i = 0; i < kHistogramBins; ++i)
                histogram_distance += std::abs(
                    histogram[i] - prior.histogram[i]);
            histogram_distance *= 0.5f;  // total-variation distance [0..1]
        }

        // Gated (confirmed) verdict: an unconfirmed single-frame spike reads
        // as "not a cut" here so it cannot suppress flow/re-prime temporal
        // history, and (below) cannot drive step()'s internal scene-cut OR
        // either. The residual cut lives entirely inside step()'s own OR and
        // is never routed through this gate. depth_cut=true — the "an
        // authoritative cut already happened this cycle" path — is taken
        // when this observation crossed a recorded boundary (04/08 round 2):
        // it confirms instantly (a clean hard cut produces ONE exceedance,
        // which alone would never confirm here) and re-arms, so flow is
        // suppressed and the temporal window re-primed on the exact first
        // post-cut observation.
        const bool source_cut =
            cut_gate.update(cur.boundary_cut, histogram_distance);
        cur.source_cut = source_cut;
        cur.histogram_distance = histogram_distance;
        cur.motion_sum = motion_sum;
        const float direct_motion_mean = have_previous_image
            ? static_cast<float>(motion_sum / n) : 0.0f;
        float local_motion_peak = 0.0f;
        if (have_previous_image) {
            constexpr int kMotionTile = 24;
            for (int ty = 0; ty < height; ty += kMotionTile) {
                for (int tx = 0; tx < width; tx += kMotionTile) {
                    double tile_sum = 0.0;
                    int tile_count = 0;
                    const int y1 = std::min(height, ty + kMotionTile);
                    const int x1 = std::min(width, tx + kMotionTile);
                    for (int y = ty; y < y1; ++y) {
                        for (int x = tx; x < x1; ++x) {
                            tile_sum +=
                                motion[static_cast<size_t>(y) * width + x];
                            ++tile_count;
                        }
                    }
                    if (tile_count)
                        local_motion_peak = std::max(
                            local_motion_peak,
                            static_cast<float>(tile_sum / tile_count));
                }
            }
        }
        // Below this level the scene reads as static -- the two images are
        // effectively the same after decoder/compression noise. Keeping the
        // local EMA in place is both more accurate and avoids paying for an
        // ambiguous forward/backward flow pair entirely (estimate_flow and
        // the reproject are both skipped this cycle; `motion[]` keeps
        // the raw per-pixel diffs already written above, and `mean_flow`
        // stays at its 0.0 default so the status line reports flow=0.00).
        // The thresholds are per-REFERENCE-interval displacements; divide the
        // measured per-frame means by SOURCE time, never by worker/inference
        // cadence. The same two video observations must make the same flow
        // decision under DirectML and TensorRT.
        // Round 5a: the fused flow of THIS cycle also ships to the GPU with
        // the published map (background-plate transport); empty publish
        // vectors mean "identity transport".
        // Round 6/6b: the flow work runs on std::async concurrently with the
        // GPU inference and the previous map's post/step stage.
        // Round 7 « flot biface » : la direction ARRIÈRE (FB) est remplacée
        // par la direction FUTURE (next->cur), calculable seulement à la
        // prep du map suivant — les deux directions fusionnent dans l'étage
        // S (fuse_bidirectional) qui produit transport, fiabilité et masque
        // de mouvement. Oracle 2026-08-03 : -43 % de résidu en statique,
        // -90 % en rapide/post-cut vs le causal seul.
        // flow_ms reports the stages' OWN wall time: under overlap
        // flow_ms+infer_ms may legitimately exceed update_ms.
        const bool run_flow = have_previous_image && !source_cut &&
            (direct_motion_mean / video_time_scale >= 0.008f ||
             local_motion_peak / video_time_scale >= 0.025f);
        cur.flow_ran = run_flow;
        if (run_flow) {
            cur.flow_task = std::async(std::launch::async, [&]() {
                const auto flow_task_start = Clock::now();
                synth3d_flow::DenseFlow forward = synth3d_flow::estimate_flow(
                    previous_luma, luma, width, height, flow_threads);
                cur.fwd_x = std::move(forward.x);
                cur.fwd_y = std::move(forward.y);
                cur.fwd_q = std::move(forward.quality);
                cur.flow_task_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - flow_task_start).count();
            });
        }
        // La direction FUTURE du map en attente : sa « next » est la frame
        // tout juste préparée. Jointe dans son étage S ci-dessous.
        if (have_pending && prior.flow_ran) {
            prior.future_task = std::async(std::launch::async, [&]() {
                const auto t0 = Clock::now();
                synth3d_flow::DenseFlow future = synth3d_flow::estimate_flow(
                    cur.luma, prior.luma, width, height, flow_threads);
                prior.fut_x = std::move(future.x);
                prior.fut_y = std::move(future.y);
                prior.fut_q = std::move(future.quality);
                prior.future_task_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - t0).count();
            });
        }
        if (input_views == 1) {
            std::copy(input.begin(), input.end(), temporal_input.begin());
            temporal_history = 1;
        } else if (temporal_history == 0 || source_cut) {
            // Prime every temporal slot with the current image. This avoids
            // black padding on startup and cross-shot attention on hard cuts.
            for (int view = 0; view < input_views; ++view)
                std::copy(
                    input.begin(), input.end(),
                    temporal_input.begin() +
                        static_cast<size_t>(view) * frame_values);
            temporal_history = 1;
        } else {
            std::move(
                temporal_input.begin() + frame_values,
                temporal_input.end(), temporal_input.begin());
            std::copy(
                input.begin(), input.end(),
                temporal_input.end() - static_cast<ptrdiff_t>(frame_values));
            temporal_history = std::min(input_views, temporal_history + 1);
        }

        const auto stage1_end = Clock::now();
        flow_ms_accum += std::chrono::duration<double, std::milli>(
            stage1_end - stage1_start).count();

        // GPU starts on THIS map now; the S-stage below finishes the PREVIOUS
        // one while it runs. This ordering IS the round-6b pipeline.
        pipe.submit(temporal_input, cur.seq);

        have_previous_image = true;

        if (pipeline_enabled) {
            if (have_pending) {
                // S(prior) : sa cible de transport est la luma de `grand`
                // (le map d'avant lui) et son FUTUR est `cur`, tout juste
                // préparé — le K=1 du flot biface est structurellement
                // gratuit dans ce pipeline.
                if (!finish_map(prior, grand.luma, &cur, cur.input)) break;
            }
            have_pending = true;
        } else {
            // Rollback timing (SYLC_DEPTH_PIPELINE=0): same code path, but the
            // freshly submitted inference is consumed immediately (no future
            // direction, no look-ahead cut verdict).
            if (!finish_map(cur, prior.luma, nullptr, cur.input)) break;
            have_pending = false;
        }
        flip = (flip + 1) % 3;
    }
    // The pipe thread must be parked BEFORE the engine goes away: its loop
    // calls engine.infer() and an explicit stop() here beats relying on
    // destruction order while a session teardown races a late job.
    pipe.stop();
    engine.shutdown();
}

#endif  // SYLC_NATIVE_RENDERER
