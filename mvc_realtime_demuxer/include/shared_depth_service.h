// SharedDepthService — one ONNX depth session for every Synth3D renderer.
//
// A SyLC presentation can own up to four D3D11 renderers (embedded preview,
// frame-pack output, two projector eyes). Their GPU resources necessarily stay
// device-local, but depth inference must not be duplicated: all surfaces show
// the same source frame and therefore need the same temporally-stabilized map.
//
// This process-wide service owns the ORT session, temporal state and latest
// immutable q16 map. Renderers attach by model/runtime/grid key. One renderer
// holds a short renewable "input leader" lease and performs the width()*height()
// readback;
// every renderer consumes the same published map on its own D3D device. If the
// leader stops presenting, another surface takes over after a bounded lease
// timeout.
//
// The registry keeps at most one unattached service warm. Older idle services
// and failed services are retired by a dedicated reaper thread, so interactive
// disable remains non-blocking even while ORT is building/destroying a session
// without letting preset/aspect switches retain every model for process life.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>   // _dupenv_s/free: SYLC_LOOKAHEAD_DECAY probe (header-inline)
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "depth_engine.h"   // kDefaultDepthSide (the grid this service defaults to)

// CPU optical flow used by the service worker (defined in the .cpp).
// Exposed here so the python test binding can drive the exact production
// code path: the parallel implementation is required to be BIT-EXACT vs
// max_threads=1 (independent rows/nodes/candidates; the global-search min
// is reduced chunk-in-order so the sequential winner always wins).
namespace synth3d_flow {
struct DenseFlow {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> quality;
};
// Round 7 « flot biface » : fusion par pixel du transport CAUSAL
// (forward = prev->cur) et du candidat FUTUR négué (future = next->cur, donc
// -future estime prev->cur sous mouvement localement linéaire), départagés
// par résidu photométrique contre la VRAIE cible du transport (prev). Oracle
// 2026-08-03 (Oblivion) : -43 % de résidu en statique, -90 % en mouvement
// rapide/post-cut. La fiabilité est le max de l'accord symétrique des deux
// directions et de l'auto-confiance résiduelle du candidat retenu — la
// seconde évite de jeter l'historique de la DERNIÈRE frame d'un plan (le
// futur y appartient au plan suivant et désaccorde toujours).
// Sorties par pixel : flot fusionné, fiabilité, masque de mouvement (résidu
// aligné + pénalité d'incertitude, même forme que l'ancien passage FB).
// Bit-exact à tout nombre de threads (écritures par pixel, sommes réduites
// par chunk en ordre).
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
                        int max_threads);

DenseFlow estimate_flow(const std::vector<float>& source_full,
                        const std::vector<float>& destination_full,
                        int full_width, int full_height,
                        int max_threads = 1);
}  // namespace synth3d_flow

namespace synth3d_aspect {
struct HorizontalBars {
    int top = 0;
    int bottom = 0;
    bool valid = false;
};
HorizontalBars detect_horizontal_letterbox(
    const std::vector<float>& luma, int width, int height);
}  // namespace synth3d_aspect

// Spatial refinement used between raw DA3 inference and temporal fusion.
// The production worker and Python regression binding share this exact path.
namespace synth3d_surface {
void compute_boundary(const std::vector<float>& depth,
                      const std::vector<float>& luma,
                      int width, int height,
                      std::vector<float>& boundary,
                      std::vector<float>& scratch,
                      int max_threads = 1);
void refine_observation(std::vector<float>& depth,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<float>& scratch,
                        int max_threads = 1);
// Build the packed RGBA16 ownership analysis used by the worker:
//   R = stabilized depth, G = conservative foreground-owned depth,
//   B = stereo safety, A = ownership repair strength.
// The RGB guide is the worker's ImageNet-normalized CHW model input. Repairs
// only ever propagate a supported NEAR layer into an ambiguous image contour;
// the safety channel remains available to flatten anything still unresolved.
// The worker then precomposes this analysis into the renderer's RG16 map.
void build_geometry_map(const std::vector<uint16_t>& depth_q16,
                        const std::vector<float>& rgb_chw,
                        const std::vector<float>& luma,
                        const std::vector<float>& confidence,
                        const std::vector<float>& boundary,
                        int width, int height,
                        std::vector<uint16_t>& geometry_rgba16,
                        std::vector<float>& scratch,
                        int max_threads = 1);
}  // namespace synth3d_surface

class SharedDepthService {
public:
    static constexpr int kDefaultSide = kDefaultDepthSide;
    // Six uint16 values per inference-grid pixel. Channels 0-1 are the
    // historical pair precomposed from the four CPU ownership channels
    // (effective depth, effective stereo safety) and feed the RG16 geometry
    // texture. Channels 2-5 are the round-5a transport block for the GPU
    // background plate: flow x/y in grid pixels quantized as (v/128)+0.5,
    // flow reliability, and one reserved slot — a skipped-flow (static)
    // cycle publishes the identity transport at full reliability.
    using GeometryMap = std::vector<uint16_t>;
    static constexpr size_t kGeometryChannels = 6;

    // Both dimensions are part of the cache key. `height <= 0` preserves the
    // historical square call and resolves to width x width.
    // Acquisition and client attachment are one registry transaction. Keeping
    // them atomic prevents an idle trim from evicting a service in the narrow
    // gap between returning it and attach().
    static std::shared_ptr<SharedDepthService> acquire_attached(
        const std::wstring& model_path, const std::wstring& ort_dir,
        int width, int height, uint64_t client_id);

    // Detach and hand the renderer's owning reference to the registry reaper.
    // The caller's shared_ptr is empty on return; the last ORT/thread teardown
    // can therefore never run on the renderer/presentation thread.
    static void detach_and_release(
        std::shared_ptr<SharedDepthService>& service, uint64_t client_id);

    ~SharedDepthService();
    SharedDepthService(const SharedDepthService&) = delete;
    SharedDepthService& operator=(const SharedDepthService&) = delete;

    void attach(uint64_t client_id);
    void detach(uint64_t client_id);

    // Renderer-side leader protocol. wants_input() is cheap and must be called
    // before drawing/copying the width()*height() prep target. submit() swaps the CHW
    // buffer, exact media PTS and capture fallback into the service. Keeping
    // both clocks beside the pixels is essential: PTS drives content-temporal
    // behavior; worker/capture wall time is only compute observability/fallback.
    bool wants_input(uint64_t client_id);
    // Diagnostic: a granted tap that ended without a submit. `ring_empty` tells
    // the two causes apart (nothing pending vs a copy still in flight).
    void note_drain_miss(bool ring_empty);
    bool submit(uint64_t client_id, std::vector<float>& chw,
                std::vector<float>& full_frame_luma,
                double video_time_ms,
                std::chrono::steady_clock::time_point capture_time,
                int source_width = 0, int source_height = 0);
    // Compatibility path for non-renderer callers: no PTS, capture "now".
    bool submit(uint64_t client_id, std::vector<float>& chw) {
        std::vector<float> no_full_frame_luma;
        return submit(
            client_id, chw, no_full_frame_luma, -1.0,
            std::chrono::steady_clock::now(), 0, 0);
    }

    // The inference grid this service was created with. side() remains as a
    // compatibility alias for the horizontal/long-edge dimension.
    int side() const { return width_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Returns an immutable latest map only when its publication sequence is
    // newer than `after_sequence`; null means "keep the local GPU texture".
    std::shared_ptr<const GeometryMap> snapshot(
        uint64_t after_sequence, uint64_t& sequence) const;

    void notify_seek();
    bool running() const;
    bool failed() const;
    int client_count() const;
    uint64_t instance_id() const { return instance_id_; }
    std::string status() const;

    // Narrow diagnostics used by native regression tests and support logs.
    static void debug_registry_stats(
        size_t& services, size_t& active, size_t& idle);

    // steady_clock ms timestamp of the last effective geometry snap (a
    // confirmed scene cut OR a consumed seek/reset), or -1 if none has
    // happened yet. Read by Synth3D::process() to drive the post-cut/seek
    // ease-out ramp (Task 5); never blocks, safe to call from any thread.
    int64_t last_snap_steady_ms() const;
    // Media PTS of the source observation that caused the last effective snap,
    // or -1 when the timed frame path is unavailable.
    double last_snap_video_ms() const;
    // Per-shot zero-parallax suggestion from the stabilizer (normalized
    // nearness, already smoothed in video time and snapped on cuts). Read by
    // Synth3D::process() when auto-convergence is enabled; never blocks.
    float suggested_convergence() const {
        return suggested_convergence_.load(std::memory_order_acquire);
    }

    // Look-ahead advisory (two-filter scout, spec 2026-08-03): delays in ms
    // from the PRESENTED position to the next cut / motion-storm onset seen
    // in the DECODED future, or <0 for none. Refreshed by the player each UI
    // tick; the worker only honours values fresher than ~500ms so a stalled
    // pump can never pin an imminence. SYLC_LOOKAHEAD=0 disables intake.
    void set_lookahead_advisory(double cut_in_ms, double storm_in_ms);

    // Freshness-checked delay to the next OBSERVED cut (<0 = none/stale).
    // First consumer: Synth3D::process()'s PRE-cut disparity ease-down — the
    // scout dates a cut at the first frame of the NEW shot (T1), so a
    // positive value means "the presented frame still belongs to the dying
    // shot": glide its depth to flat so it meets the post-snap ramp's flat
    // start and the cut carries no stereo jump (author rule 2026-08-03:
    // always anticipate the cut, never merely react at T1).
    // "No event" sentinel. MUST be far outside the hold window: small
    // negative delays are REAL data ("the cut landed ~1 frame ago, hold the
    // flat"), so -1.0 would collide with them and flatten depth forever.
    static constexpr double kLookaheadNone = -1.0e9;

    // Dead-reckoning of the pump-quantized advisory (2026-08-03: the
    // one-frame wavy tear at every cut). The player refreshes the advisory
    // from a 10 Hz QTimer (SyLC_3D_Player.py _playback_timer setInterval(100))
    // while process() presents at the source cadence (41.7 ms at 24 fps), so
    // the stored delay is FROZEN for up to ~100 ms = 2.4 presented frames.
    // At the cut frame T1 the frozen value still reads +0..100 ms ("cut
    // ahead"): the ease-down computes t2 = stale/300 and re-opens up to 55%
    // of the disparity budget, warping the NEW shot's first frame with the
    // OLD shot's map — and cb.temporal_fill (old-shot plate) only cuts at
    // <= 0, so the plate keeps filling the holes with old-shot pixels.
    // Subtracting the advisory's steady-clock age reconstructs the true
    // delay to within pump/position jitter (video time advances at wall
    // rate during playback; while paused the pump keeps re-stamping every
    // 100 ms, bounding the decay error to one pump period). The sentinel
    // never decays. Pure and static so the test binding pins the exact
    // production math (_synth3d_advisory_decay_test).
    static double advisory_decay(double value, int64_t age_ms) {
        if (value <= 0.5 * kLookaheadNone) return kLookaheadNone;
        return value - static_cast<double>(age_ms);
    }

    double lookahead_cut_in_ms() const {
        const int64_t set = lookahead_set_ms_.load(std::memory_order_acquire);
        if (set < 0) return kLookaheadNone;
        const int64_t now =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        const int64_t age = now - set;
        if (age > 500) return kLookaheadNone;
        const double raw = lookahead_cut_ms_.load(std::memory_order_acquire);
        const double corrected = advisory_decay(raw, age);
        // la_seen: the advisory's raw delay flipped positive -> negative, i.e.
        // one cut travelled through the pump. exchange() claims the transition
        // ONCE process-wide, so the count is invariant to the number of
        // reading surfaces AND to pump/present phase (>= one negative store is
        // guaranteed before the scout's 120 ms purge at 10 Hz; the sentinel is
        // excluded so a seek purge does not fake a cut). THIS is the counter
        // two runs may be compared on (~= cuts the scout reported).
        const double prev = la_prev_raw_.exchange(
            raw, std::memory_order_relaxed);
        if (prev > 0.0 && raw <= 0.0 && raw > 0.5 * kLookaheadNone)
            la_seen_.fetch_add(1, std::memory_order_relaxed);
        // la_stale: reads inside [cut landed .. next pump tick] while the RAW
        // value still said "cut ahead" -- the presents the FROZEN advisory
        // rendered with stale positive disparity. The per-read predicate uses
        // only raw and age (pump-written), so the flag cannot change it; but
        // the per-cut count is a small integer (0-3) cut from a lattice
        // (41.7 ms frame grid x 100 ms pump x per-run position/present clock
        // bias): simulated per-run means span 0.47..2.04 per cut with a
        // geometric floor ~0.5. It is therefore NOT comparable across runs --
        // normalize on la_seen, and treat a value below ~0.5/cut as a
        // measurement artifact (counter reset mid-window: service instance
        // changed; or missed reads), not as a flag effect.
        if (raw > 0.0 && corrected <= 0.0)
            la_stale_.fetch_add(1, std::memory_order_relaxed);
        static const bool decay_on = []() {
            char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
            size_t len = 0;
            bool on = true;
            if (_dupenv_s(&env, &len, "SYLC_LOOKAHEAD_DECAY") == 0 && env) {
                on = env[0] != '0';   // SYLC_LOOKAHEAD_DECAY=0 = frozen advisory
                free(env);
            }
            return on;
        }();
        return decay_on ? corrected : raw;
    }

private:
    enum class State : int { Init = 0, Running = 1, Error = 2, Stopping = 3 };

    SharedDepthService(std::wstring model_path, std::wstring ort_dir,
                       int width, int height);
    void start_worker();
    void worker_main();
    void set_error(const std::string& message);

    std::wstring model_path_;
    std::wstring ort_dir_;
    const int width_;
    const int height_;
    const uint64_t instance_id_;

    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<State> state_{State::Init};
    std::atomic<int> clients_{0};
    std::atomic<bool> reset_stabilizer_{true};

    mutable std::mutex input_mtx_;
    std::condition_variable input_cv_;
    std::vector<float> input_mailbox_;
    std::vector<float> input_full_frame_luma_;
    std::chrono::steady_clock::time_point input_capture_time_{};
    double input_video_time_ms_ = -1.0;
    int input_source_width_ = 0;
    int input_source_height_ = 0;
    bool input_fresh_ = false;
    uint64_t leader_id_ = 0;
    std::chrono::steady_clock::time_point leader_seen_{};

    mutable std::mutex output_mtx_;
    std::shared_ptr<const GeometryMap> latest_;
    uint64_t output_sequence_ = 0;
    std::chrono::steady_clock::time_point output_time_{};

    mutable std::mutex meta_mtx_;
    std::string provider_ = "none";
    std::string error_;
    std::atomic<float> fps_{0.0f};
    std::atomic<float> motion_{0.0f};
    // Per-cycle worker stage costs, averaged over the same 2s window as fps_.
    std::atomic<float> flow_ms_{0.0f};   // flow estimation + reprojection + prep
    std::atomic<float> infer_ms_{0.0f};  // engine.infer()
    std::atomic<float> stab_ms_{0.0f};   // stabilizer.step()
    std::atomic<float> source_dt_ms_{120.0f}; // source VIDEO-PTS interval
    std::atomic<float> update_dt_ms_{120.0f}; // COMPUTE map-update interval
    // --- wait instrumentation (diagnostic only; no behaviour depends on it) --
    // flow_ms/infer_ms/stab_ms time WORK. A worker cycle can also be spent
    // BLOCKED, and that time was invisible: measured stages summed to ~31 ms
    // inside a measured 51 ms cycle on the edge264 path. These four name the
    // gap. Averaged over the same 2 s window as the stage timers.
    std::atomic<float> inwait_ms_{0.0f};   // blocked waiting for a submission
    std::atomic<float> reswait_ms_{0.0f};  // blocked in InferPipe::wait_result
    std::atomic<float> joinwait_ms_{0.0f}; // blocked joining flow/future tasks
    std::atomic<float> cycle_ms_{0.0f};    // whole iteration, loop top to top
    // Tap accounting, SERVICE-side so it survives leader election: only one of
    // the N client surfaces ever feeds, and it is not necessarily the one whose
    // status the player polls. grants-submits == presented frames that were
    // granted a tap but produced no submission (the readback had not landed).
    std::atomic<uint64_t> grants_{0};
    std::atomic<uint64_t> submits_{0};
    // Why a granted tap produced nothing: the ring held no pending copy at all
    // (empty) versus it held one the GPU had not finished (stalled). The two
    // point at completely different fixes, so they are counted apart.
    std::atomic<uint64_t> drain_empty_{0};
    std::atomic<uint64_t> drain_stalled_{0};
    std::atomic<float> effective_alpha_{0.0f};
    std::atomic<float> scene_change_{0.0f};
    std::atomic<float> confidence_{1.0f};
    std::atomic<float> flow_{0.0f};
    std::atomic<float> stability_{0.0f};
    std::atomic<float> history_support_{1.0f};
    std::atomic<float> suggested_convergence_{0.5f};
    std::atomic<int> temporal_views_{1};
    // Stable encoded-letterbox observation, expressed in SOURCE pixels.
    // Zero/confidence 0 means no trustworthy horizontal matte was found.
    std::atomic<int> crop_top_{0};
    std::atomic<int> crop_bottom_{0};
    std::atomic<int> crop_source_width_{0};
    std::atomic<int> crop_source_height_{0};
    std::atomic<float> crop_confidence_{0.0f};
    // True after eight agreeing observations of either a matte or no matte.
    // Unlike crop_confidence, this can certify a transition back to 16:9.
    std::atomic<bool> crop_ready_{false};
    std::atomic<uint64_t> cuts_{0};
    std::atomic<int64_t> last_snap_ms_{-1};
    std::atomic<double> last_snap_video_ms_{-1.0};
    // Look-ahead advisory intake (see set_lookahead_advisory).
    std::atomic<double> lookahead_cut_ms_{kLookaheadNone};
    std::atomic<double> lookahead_storm_ms_{kLookaheadNone};
    std::atomic<int64_t> lookahead_set_ms_{-1};
    // Advisory reads that were stale-positive across a landed cut (see
    // lookahead_cut_in_ms; phase-sensitive, never compare across runs).
    // mutable: counted from the const accessor on the present path.
    mutable std::atomic<uint64_t> la_stale_{0};
    // Advisory raw-delay sign transitions = cuts that travelled through the
    // pump. Phase- and surface-count-invariant (claimed once via exchange);
    // the honest cross-run witness/denominator for la_stale.
    mutable std::atomic<double> la_prev_raw_{kLookaheadNone};
    mutable std::atomic<uint64_t> la_seen_{0};
};
