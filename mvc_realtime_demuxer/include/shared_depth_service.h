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
// Services are cached for the process lifetime. That deliberately makes an
// interactive disable non-blocking even while ORT is building a session, and
// makes re-enable instant. Workers are joined when the module/process tears
// down, never while a renderer mutex is serving video.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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

class SharedDepthService {
public:
    static constexpr int kDefaultSide = kDefaultDepthSide;
    using DepthMap = std::vector<uint16_t>;

    // Both dimensions are part of the cache key. `height <= 0` preserves the
    // historical square call and resolves to width x width.
    static std::shared_ptr<SharedDepthService> acquire(
        const std::wstring& model_path, const std::wstring& ort_dir,
        int width, int height = 0);

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
    bool submit(uint64_t client_id, std::vector<float>& chw,
                double video_time_ms,
                std::chrono::steady_clock::time_point capture_time,
                int source_width = 0, int source_height = 0);
    // Compatibility path for non-renderer callers: no PTS, capture "now".
    bool submit(uint64_t client_id, std::vector<float>& chw) {
        return submit(
            client_id, chw, -1.0, std::chrono::steady_clock::now(), 0, 0);
    }

    // The inference grid this service was created with. side() remains as a
    // compatibility alias for the horizontal/long-edge dimension.
    int side() const { return width_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Returns an immutable latest map only when its publication sequence is
    // newer than `after_sequence`; null means "keep the local GPU texture".
    std::shared_ptr<const DepthMap> snapshot(
        uint64_t after_sequence, uint64_t& sequence) const;

    void notify_seek();
    bool running() const;
    std::string status() const;

    // steady_clock ms timestamp of the last effective geometry snap (a
    // confirmed scene cut OR a consumed seek/reset), or -1 if none has
    // happened yet. Read by Synth3D::process() to drive the post-cut/seek
    // ease-out ramp (Task 5); never blocks, safe to call from any thread.
    int64_t last_snap_steady_ms() const;
    // Media PTS of the source observation that caused the last effective snap,
    // or -1 when the timed frame path is unavailable.
    double last_snap_video_ms() const;

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

    std::thread worker_;
    std::atomic<bool> stop_{false};
    std::atomic<State> state_{State::Init};
    std::atomic<int> clients_{0};
    std::atomic<bool> reset_stabilizer_{true};

    mutable std::mutex input_mtx_;
    std::condition_variable input_cv_;
    std::vector<float> input_mailbox_;
    std::chrono::steady_clock::time_point input_capture_time_{};
    double input_video_time_ms_ = -1.0;
    int input_source_width_ = 0;
    int input_source_height_ = 0;
    bool input_fresh_ = false;
    uint64_t leader_id_ = 0;
    std::chrono::steady_clock::time_point leader_seen_{};

    mutable std::mutex output_mtx_;
    std::shared_ptr<const DepthMap> latest_;
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
    std::atomic<float> effective_alpha_{0.0f};
    std::atomic<float> scene_change_{0.0f};
    std::atomic<float> confidence_{1.0f};
    std::atomic<float> flow_{0.0f};
    std::atomic<float> stability_{0.0f};
    std::atomic<float> history_support_{1.0f};
    std::atomic<int> temporal_views_{1};
    // Stable encoded-letterbox observation, expressed in SOURCE pixels.
    // Zero/confidence 0 means no trustworthy horizontal matte was found.
    std::atomic<int> crop_top_{0};
    std::atomic<int> crop_bottom_{0};
    std::atomic<int> crop_source_width_{0};
    std::atomic<int> crop_source_height_{0};
    std::atomic<float> crop_confidence_{0.0f};
    std::atomic<uint64_t> cuts_{0};
    std::atomic<int64_t> last_snap_ms_{-1};
    std::atomic<double> last_snap_video_ms_{-1.0};
};
