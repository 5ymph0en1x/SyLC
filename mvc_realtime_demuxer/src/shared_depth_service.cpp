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
#include <limits>
#include <map>

namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kLeaderLease = std::chrono::milliseconds(350);
constexpr int kHistogramBins = 32;

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

// Strong ownership is intentional: disabling the last renderer must never run
// an ORT destructor (or wait for CreateSession) on the presentation thread.
std::map<std::wstring, std::shared_ptr<SharedDepthService>>& registry() {
    static std::map<std::wstring, std::shared_ptr<SharedDepthService>> r;
    return r;
}

std::wstring service_key(const std::wstring& model, const std::wstring& ort,
                          int width, int height) {
    return model + L"\x1f" + ort + L"\x1f" + std::to_wstring(width) +
           L"x" + std::to_wstring(height);
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
    return std::max(2, std::min(8, hc / 2));
}

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

namespace {

void compute_surface_boundary(const std::vector<float>& depth,
                              const std::vector<float>& luma,
                              int width, int height,
                              std::vector<float>& boundary,
                              std::vector<float>& scratch) {
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
    const float depth_scale = std::max(1.0e-6f, 0.22f * sigma);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const float depth_gradient = std::max(
                std::abs(depth[i + 1] - depth[i - 1]),
                std::abs(depth[i + width] - depth[i - width]));
            const float luma_gradient = std::max(
                std::abs(luma[i + 1] - luma[i - 1]),
                std::abs(luma[i + width] - luma[i - width]));
            const float depth_edge = clamp01(
                (depth_gradient / depth_scale - 0.08f) / 0.72f);
            const float image_support = clamp01(
                (luma_gradient - 0.018f) / 0.12f);
            // A model-only edge remains protected, but an aligned image edge
            // raises confidence that this is a real surface boundary rather
            // than texture/noise inside one object.
            scratch[i] = clamp01(
                depth_edge * (0.45f + 0.55f * image_support));
        }
    }

    // Form a narrow three-pixel guard band. Temporal decisions inside this
    // band are layer-aware; there is deliberately no blur/average here.
    for (int y = 1; y < height - 1; ++y) {
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
}

}  // namespace

std::shared_ptr<SharedDepthService> SharedDepthService::acquire(
        const std::wstring& model_path, const std::wstring& ort_dir,
        int width, int height) {
    if (width <= 0) width = kDefaultSide;
    if (height <= 0) height = width;
    const std::wstring key = service_key(model_path, ort_dir, width, height);
    std::lock_guard<std::mutex> lk(registry_mutex());
    auto& r = registry();
    auto it = r.find(key);
    if (it != r.end()) return it->second;

    auto service = std::shared_ptr<SharedDepthService>(
        new SharedDepthService(model_path, ort_dir, width, height));
    r.emplace(key, service);
    service->start_worker();
    return service;
}

SharedDepthService::SharedDepthService(std::wstring model_path, std::wstring ort_dir,
                                       int width, int height)
    : model_path_(std::move(model_path)), ort_dir_(std::move(ort_dir)),
      width_(width), height_(height) {
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
    }
    last_snap_ms_.store(-1, std::memory_order_release);
    last_snap_video_ms_.store(-1.0, std::memory_order_release);
    reset_stabilizer_.store(true, std::memory_order_release);
    crop_top_.store(0, std::memory_order_release);
    crop_bottom_.store(0, std::memory_order_release);
    crop_source_width_.store(0, std::memory_order_release);
    crop_source_height_.store(0, std::memory_order_release);
    crop_confidence_.store(0.0f, std::memory_order_release);
}

void SharedDepthService::detach(uint64_t client_id) {
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        if (leader_id_ == client_id) {
            leader_id_ = 0;
            leader_seen_ = {};
        }
    }
    int old = clients_.load(std::memory_order_acquire);
    while (old > 0 && !clients_.compare_exchange_weak(
               old, old - 1, std::memory_order_acq_rel)) {}
    input_cv_.notify_all();
}

bool SharedDepthService::running() const {
    return state_.load(std::memory_order_acquire) == State::Running;
}

bool SharedDepthService::wants_input(uint64_t client_id) {
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
    return true;
}

bool SharedDepthService::submit(
        uint64_t client_id, std::vector<float>& chw,
        double video_time_ms,
        std::chrono::steady_clock::time_point capture_time,
        int source_width, int source_height) {
    const size_t expected = 3ull * static_cast<size_t>(width_) * height_;
    if (chw.size() != expected || !running()) return false;
    if (capture_time.time_since_epoch().count() == 0)
        capture_time = Clock::now();
    if (!std::isfinite(video_time_ms) || video_time_ms < 0.0)
        video_time_ms = -1.0;
    {
        std::lock_guard<std::mutex> lk(input_mtx_);
        if (leader_id_ != client_id || input_fresh_) return false;
        input_mailbox_.swap(chw);
        input_capture_time_ = capture_time;
        input_video_time_ms_ = video_time_ms;
        input_source_width_ = std::max(0, source_width);
        input_source_height_ = std::max(0, source_height);
        input_fresh_ = true;
        leader_seen_ = Clock::now();
    }
    input_cv_.notify_one();
    return true;
}

std::shared_ptr<const SharedDepthService::DepthMap> SharedDepthService::snapshot(
        uint64_t after_sequence, uint64_t& sequence) const {
    std::lock_guard<std::mutex> lk(output_mtx_);
    sequence = output_sequence_;
    if (!latest_ || output_sequence_ == after_sequence) return {};
    return latest_;
}

void SharedDepthService::notify_seek() {
    reset_stabilizer_.store(true, std::memory_order_release);
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
    char buf[512] = {};
    std::snprintf(
        buf, sizeof(buf),
        "state=%s provider=%s views=%d side=%d fps=%.1f "
        "flow_ms=%.1f infer_ms=%.1f "
        "stab_ms=%.1f source_ms=%.1f update_ms=%.1f age_ms=%lld clients=%d cuts=%llu "
        "motion=%.3f flow=%.2f alpha=%.3f conf=%.3f stable=%.3f "
        "history=%.2f scene=%.3f crop=%d:%d:%d:%d crop_conf=%.2f "
        "grid=%dx%d err=%s",
        state_name, provider.empty() ? "none" : provider.c_str(),
        temporal_views_.load(std::memory_order_acquire),
        width_,
        static_cast<double>(fps_.load(std::memory_order_acquire)),
        static_cast<double>(flow_ms_.load(std::memory_order_acquire)),
        static_cast<double>(infer_ms_.load(std::memory_order_acquire)),
        static_cast<double>(stab_ms_.load(std::memory_order_acquire)),
        static_cast<double>(source_dt_ms_.load(std::memory_order_acquire)),
        static_cast<double>(update_dt_ms_.load(std::memory_order_acquire)), age_ms,
        clients_.load(std::memory_order_acquire),
        static_cast<unsigned long long>(cuts_.load(std::memory_order_acquire)),
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
        width_, height_,
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
    std::vector<float> input(3 * n, 0.0f);
    std::vector<float> temporal_input(
        static_cast<size_t>(input_views) * frame_values, 0.0f);
    int temporal_history = 0;
    std::vector<float> raw(n, 0.0f);
    std::vector<float> raw_confidence(n, 1.0f);
    std::vector<float> confidence(n, 1.0f);
    std::vector<float> luma(n, 0.0f);
    std::vector<float> previous_luma(n, 0.0f);
    std::vector<float> motion(n, 0.0f);
    std::vector<float> flow_reliability(n, 0.0f);
    std::vector<float> surface_boundary(n, 0.0f);
    std::vector<float> surface_scratch(n, 0.0f);
    std::vector<uint16_t> q16(n, 0);
    std::array<float, kHistogramBins> previous_hist{};
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
    int pending_crop_top = 0;
    int pending_crop_bottom = 0;
    int pending_crop_count = 0;

    while (!stop_.load(std::memory_order_acquire)) {
        Clock::time_point capture_time{};
        double video_time_ms = -1.0;
        int source_width = 0;
        int source_height = 0;
        {
            std::unique_lock<std::mutex> lk(input_mtx_);
            input_cv_.wait_for(lk, std::chrono::milliseconds(100), [this] {
                return stop_.load(std::memory_order_acquire) || input_fresh_;
            });
            if (stop_.load(std::memory_order_acquire)) break;
            if (!input_fresh_) continue;
            input.swap(input_mailbox_);
            capture_time = input_capture_time_;
            video_time_ms = input_video_time_ms_;
            source_width = input_source_width_;
            source_height = input_source_height_;
            input_fresh_ = false;
        }

        const auto stage1_start = Clock::now();

        const bool reset = reset_stabilizer_.exchange(false, std::memory_order_acq_rel);
        if (reset) {
            stabilizer.reset();
            have_previous_image = false;
            have_capture_time = false;
            have_video_time = false;
            have_update_time = false;
            temporal_history = 0;
            pending_crop_top = 0;
            pending_crop_bottom = 0;
            pending_crop_count = 0;
            crop_top_.store(0, std::memory_order_release);
            crop_bottom_.store(0, std::memory_order_release);
            crop_source_width_.store(0, std::memory_order_release);
            crop_source_height_.store(0, std::memory_order_release);
            crop_confidence_.store(0.0f, std::memory_order_release);
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
        last_capture_time = capture_time;
        have_capture_time = true;
        last_video_time_ms = valid_video_time ? video_time_ms : -1.0;
        have_video_time = valid_video_time;
        stabilizer.set_source_dt_ms(source_dt_ms);
        // Read back the clamped value so flow gating, uncertainty composition,
        // status and the stabilizer use one identical source-time scale.
        source_dt_ms_.store(
            stabilizer.source_dt_ms(), std::memory_order_release);
        const float video_time_scale =
            stabilizer.source_dt_ms() / DepthStabilizer::kReferenceDtMs;

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
            const LetterboxCandidate crop =
                detect_horizontal_letterbox(luma, width, height);
            if (crop.valid) {
                const int tolerance = std::max(2, height / 100);
                if (std::abs(crop.top - pending_crop_top) <= tolerance &&
                    std::abs(crop.bottom - pending_crop_bottom) <= tolerance) {
                    ++pending_crop_count;
                } else {
                    pending_crop_top = crop.top;
                    pending_crop_bottom = crop.bottom;
                    pending_crop_count = 1;
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
                }
            } else if (pending_crop_count < 8) {
                pending_crop_count = 0;
                pending_crop_top = 0;
                pending_crop_bottom = 0;
                crop_confidence_.store(0.0f, std::memory_order_release);
            }
        }
        for (float& v : histogram) v /= static_cast<float>(n);

        float histogram_distance = 0.0f;
        if (have_previous_image) {
            for (int i = 0; i < kHistogramBins; ++i)
                histogram_distance += std::abs(histogram[i] - previous_hist[i]);
            histogram_distance *= 0.5f;  // total-variation distance [0..1]
        }

        // Gated (confirmed) verdict: an unconfirmed single-frame spike reads
        // as "not a cut" here so it cannot suppress flow/re-prime temporal
        // history, and (below) cannot drive step()'s internal scene-cut OR
        // either. depth_cut is always false on this call: the residual cut
        // lives entirely inside step()'s own OR and is never routed through
        // this gate -- CutGate's depth_cut=true path exists for callers who
        // want an immediate verdict and is exercised only by its unit tests
        // (test_cut_gate.py), not by this integration.
        const bool source_cut = cut_gate.update(false, histogram_distance);
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
        float mean_flow = 0.0f;
        // Below this level the scene reads as static -- the two images are
        // effectively the same after decoder/compression noise. Keeping the
        // local EMA in place is both more accurate and avoids paying for an
        // ambiguous forward/backward flow pair entirely (estimate_flow +
        // stabilizer.reproject are both skipped this cycle; `motion[]` keeps
        // the raw per-pixel diffs already written above, and `mean_flow`
        // stays at its 0.0 default so the status line reports flow=0.00).
        // Task 6 perf: widened from 0.003 to 0.008 -- Task 1's measurement
        // showed flow+prep at only ~3-4ms (~3% of the ~130-150ms cycle, all
        // spent on the always-run luma/histogram/motion pass above, not on
        // estimate_flow itself) even when the estimator did run, so this is
        // a genuinely static-content classifier, not a perf-desperate one.
        // `have_previous_image` guards the very first frame after a
        // reset/cut (previous_luma not yet meaningful): flow is already
        // unconditionally skipped upstream of this threshold in that case.
        // The thresholds are per-REFERENCE-interval displacements; divide the
        // measured per-frame means by SOURCE time, never by worker/inference
        // cadence. The same two video observations must make the same flow
        // decision under DirectML and TensorRT.
        if (have_previous_image && !source_cut &&
            (direct_motion_mean / video_time_scale >= 0.008f ||
             local_motion_peak / video_time_scale >= 0.025f)) {
            synth3d_flow::DenseFlow forward = synth3d_flow::estimate_flow(
                previous_luma, luma, width, height, flow_threads);
            synth3d_flow::DenseFlow backward = synth3d_flow::estimate_flow(
                luma, previous_luma, width, height, flow_threads);
            double aligned_motion_sum = 0.0;
            double flow_sum = 0.0;
            // FB-consistency: per-pixel writes are independent; the two
            // diagnostic sums are reduced per chunk then in chunk order
            // (their last-bit rounding vs the sequential order is
            // immaterial -- they only feed the status line).
            const int fb_chunks = parallel_chunk_count(height, flow_threads);
            std::vector<double> fb_motion(fb_chunks, 0.0);
            std::vector<double> fb_flow(fb_chunks, 0.0);
            parallel_chunks(height, flow_threads,
                            [&](int chunk_id, int y_begin, int y_end) {
            double chunk_motion = 0.0;
            double chunk_flow = 0.0;
            for (int y = y_begin; y < y_end; ++y) {
                for (int x = 0; x < width; ++x) {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    const float previous_x = static_cast<float>(x) - forward.x[i];
                    const float previous_y = static_cast<float>(y) - forward.y[i];
                    const float reverse_x = bilinear_sample(
                        backward.x, width, height, previous_x, previous_y);
                    const float reverse_y = bilinear_sample(
                        backward.y, width, height, previous_x, previous_y);
                    const float fb_error = std::sqrt(
                        (forward.x[i] + reverse_x) *
                            (forward.x[i] + reverse_x) +
                        (forward.y[i] + reverse_y) *
                            (forward.y[i] + reverse_y));
                    const float consistency = std::exp(-0.35f * fb_error);
                    const float reverse_quality = bilinear_sample(
                        backward.quality, width, height, previous_x, previous_y);
                    const bool in_bounds =
                        previous_x >= 0.0f && previous_y >= 0.0f &&
                        previous_x <= static_cast<float>(width - 1) &&
                        previous_y <= static_cast<float>(height - 1);
                    const float reliability = in_bounds
                        ? clamp01(std::sqrt(
                              forward.quality[i] * reverse_quality) *
                                  consistency)
                        : 0.0f;
                    flow_reliability[i] = reliability;
                    const float aligned_previous = bilinear_sample(
                        previous_luma, width, height, previous_x, previous_y);
                    const float residual = std::abs(luma[i] - aligned_previous);
                    // Occlusions and failed correspondences must accept the new
                    // observation rapidly instead of dragging stale geometry.
                    // Only the photometric residual is a per-source-interval
                    // displacement. Correspondence uncertainty is dimensionless,
                    // so pre-scale its contribution here: step() divides the sum
                    // by video_time_scale and the penalty therefore remains
                    // cadence-invariant instead of being amplified at high fps.
                    motion[i] = clamp01(
                        residual +
                        (1.0f - reliability) * 0.12f * video_time_scale);
                    chunk_motion += motion[i];
                    chunk_flow += std::sqrt(
                        forward.x[i] * forward.x[i] +
                        forward.y[i] * forward.y[i]);
                }
            }
            fb_motion[chunk_id] = chunk_motion;
            fb_flow[chunk_id] = chunk_flow;
            });
            for (int c = 0; c < fb_chunks; ++c) {
                aligned_motion_sum += fb_motion[c];
                flow_sum += fb_flow[c];
            }
            motion_sum = aligned_motion_sum;
            mean_flow = static_cast<float>(flow_sum / n);
            stabilizer.reproject(
                forward.x.data(), forward.y.data(),
                flow_reliability.data(), width, height);
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

        const auto infer_start = Clock::now();
        if (!engine.infer(
                temporal_input.data(), raw.data(), error,
                raw_confidence.data())) {
            set_error(error);
            break;
        }
        infer_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - infer_start).count();
        normalize_da3_confidence(raw_confidence, confidence);
        compute_surface_boundary(
            raw, luma, width, height, surface_boundary, surface_scratch);
        if (have_previous_image) {
            // Motion at a silhouette is often registered one or two pixels on
            // only one side. Expand it along the guard band so the complete
            // neck/shoulder boundary chooses the short-memory path together.
            surface_scratch = motion;
            for (int y = 1; y < height - 1; ++y) {
                for (int x = 1; x < width - 1; ++x) {
                    const size_t i = static_cast<size_t>(y) * width + x;
                    if (surface_boundary[i] < 0.20f) continue;
                    float local_max = 0.0f;
                    for (int oy = -1; oy <= 1; ++oy)
                        for (int ox = -1; ox <= 1; ++ox)
                            local_max = std::max(
                                local_max,
                                surface_scratch[
                                    static_cast<size_t>(y + oy) * width +
                                    x + ox]);
                    motion[i] = local_max;
                }
            }
        }
        const auto stab_start = Clock::now();
        float update_dt_ms = DepthStabilizer::kReferenceDtMs;
        if (have_update_time)
            update_dt_ms = static_cast<float>(
                std::chrono::duration<double, std::milli>(
                    stab_start - last_update_time).count());
        last_update_time = stab_start;
        have_update_time = true;
        // Compute cadence is exposed in status, but DepthStabilizer keeps it
        // out of state evolution; source_dt_ms (video PTS) owns that math.
        stabilizer.set_update_dt_ms(update_dt_ms);
        update_dt_ms_.store(
            stabilizer.update_dt_ms(), std::memory_order_release);
        bool cut = stabilizer.step(
            raw.data(), q16.data(),
            have_previous_image ? motion.data() : nullptr,
            source_cut ? histogram_distance : 0.0f, confidence.data(),
            surface_boundary.data());
        stab_ms_accum += std::chrono::duration<double, std::milli>(
            Clock::now() - stab_start).count();
        if (cut && input_views > 1 && !source_cut) {
            // The depth residual found a cut that the source histogram missed.
            // Re-run this rare transition with a clean temporal window so the
            // published map cannot contain cross-shot attention ghosts.
            for (int view = 0; view < input_views; ++view)
                std::copy(
                    input.begin(), input.end(),
                    temporal_input.begin() +
                        static_cast<size_t>(view) * frame_values);
            const auto retry_infer_start = Clock::now();
            if (!engine.infer(
                    temporal_input.data(), raw.data(), error,
                    raw_confidence.data())) {
                set_error(error);
                break;
            }
            infer_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - retry_infer_start).count();
            normalize_da3_confidence(raw_confidence, confidence);
            compute_surface_boundary(
                raw, luma, width, height,
                surface_boundary, surface_scratch);
            stabilizer.reset();
            const auto retry_stab_start = Clock::now();
            cut = stabilizer.step(
                raw.data(), q16.data(), nullptr, 0.0f, confidence.data(),
                surface_boundary.data());
            stab_ms_accum += std::chrono::duration<double, std::milli>(
                Clock::now() - retry_stab_start).count();
            // reset() unprimes the stabilizer, so this step() always takes the
            // priming path and returns cut=false -- yet the EMA and tone range
            // were both just fully replaced, a real geometry teleport
            // functionally identical to a snap. Record it unconditionally so
            // the ramp actually engages here; deliberately NOT mirrored into
            // cuts_ (that diagnostic counter's pre-existing blind spot on
            // this exact branch is out of scope for this fix).
            last_snap_ms_.store(steady_now_ms(), std::memory_order_release);
            last_snap_video_ms_.store(
                valid_video_time ? video_time_ms : -1.0,
                std::memory_order_release);
            temporal_history = 1;
        }

        // Velocity-normalized in SOURCE time so the status-line motion=
        // reading describes the video, not the provider's compute cadence.
        motion_.store(
            have_previous_image
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
        if (cut) {
            cuts_.fetch_add(1, std::memory_order_acq_rel);
            last_snap_ms_.store(steady_now_ms(), std::memory_order_release);
            last_snap_video_ms_.store(
                valid_video_time ? video_time_ms : -1.0,
                std::memory_order_release);
        }

        {
            auto published = std::make_shared<DepthMap>(q16);
            std::lock_guard<std::mutex> lk(output_mtx_);
            latest_ = std::move(published);
            ++output_sequence_;
            output_time_ = Clock::now();
        }

        previous_luma.swap(luma);
        previous_hist = histogram;
        have_previous_image = true;

        ++fps_count;
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 2.0) {
            fps_.store(static_cast<float>(fps_count / elapsed), std::memory_order_release);
            flow_ms_.store(static_cast<float>(flow_ms_accum / fps_count),
                           std::memory_order_release);
            infer_ms_.store(static_cast<float>(infer_ms_accum / fps_count),
                            std::memory_order_release);
            stab_ms_.store(static_cast<float>(stab_ms_accum / fps_count),
                          std::memory_order_release);
            fps_count = 0;
            flow_ms_accum = 0.0;
            infer_ms_accum = 0.0;
            stab_ms_accum = 0.0;
            fps_start = now;
        }
    }
    engine.shutdown();
}

#endif  // SYLC_NATIVE_RENDERER
