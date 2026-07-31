// Synth3D: the GPU half of SyLC's real-time 2D->3D "synth3d" mode
// (see .superpowers/sdd/2026-07-28-2d-to-3d-realtime, Task 4).
//
// Pipeline, per displayed frame:
//   1. depth prep   — the source Y/U/V planes are converted to RGB and rendered
//                     into a width x height RGBA32F target (the attached service's
//                     inference grid: 756 for the quality preset, 518 below it).
//   2. readback     — that target is CopyResource'd into a 3-deep STAGING ring and
//                     mapped with D3D11_MAP_FLAG_DO_NOT_WAIT. A slot the GPU has
//                     not finished writing is simply retried on a later frame, so
//                     the presenter thread NEVER blocks on the GPU.
//   3. inference    — a process-wide SharedDepthService owns ONE DepthEngine +
//                     DepthStabilizer for all renderer surfaces. A renewable leader
//                     lease selects which surface performs readback; every D3D device
//                     consumes the same immutable stabilized uint16 nearness map.
//   4. warp         — the nearness map is uploaded to an R16_UNORM texture and two
//                     multiple-render-target passes (luma 2-MRT, chroma 4-MRT) produce
//                     a synthesized stereo pair in six dedicated textures. depth_view
//                     swaps those passes for a false-color visualization of the depth.
//
// THREADING. Every D3D method here runs under NativeRenderer::Impl::mtx. The shared
// worker never touches D3D or a renderer mutex. SharedDepthService is process-cached:
// detaching the last surface cannot destroy ORT or wait for CreateSession on the
// presentation thread, and a subsequent enable reuses the warm session.
//
// Windows-only (d3d11.h); compiled into the module inside the BUILD_NATIVE_RENDERER
// block, like native_renderer.cpp / sbs_nv12_packer.cpp.
#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "shared_depth_service.h"

struct Synth3DParams {
    bool  enabled = false;
    float strength_pct = 1.5f;    // max disparity, % of image width
    float convergence = 0.5f;     // 0..1 in normalized nearness
    bool  depth_view = false;
    bool  diagnostics = false;    // depth + disocclusion overlay on warped image
    std::wstring model_path, ort_dir;
    // Backward-compatible square grid. Explicit positive grid_width/grid_height
    // take precedence.
    int   side = SharedDepthService::kDefaultSide;
    int   grid_width = 0;
    int   grid_height = 0;
    // Encoded horizontal mattes, normalized to the decoded source height.
    // The prep pass crops them; the warp holds them at convergence.
    float crop_top = 0.0f;
    float crop_bottom = 0.0f;
};

// Owned by NativeRenderer::Impl; ALL D3D methods are called under Impl::mtx on
// the immediate context. The inference thread never touches D3D.
class Synth3D {
public:
    static constexpr int kNumOut = 6;                    // Y_L,U_L,V_L,Y_R,U_R,V_R

    Synth3D();
    ~Synth3D();
    Synth3D(const Synth3D&) = delete;                    // owns a thread + D3D resources
    Synth3D& operator=(const Synth3D&) = delete;

    // Attach to (or keep) the process-wide inference service. Session creation is
    // asynchronous; start() never waits for ORT.
    bool start(ID3D11Device* dev, const Synth3DParams& p, std::string& err);
    void set_params(const Synth3DParams& p);           // strength/convergence/view
    // Per displayed frame, under mtx. src[3] = Y/U/V SRVs of the SOURCE eye,
    // y_w/y_h = luma plane size, c_w/c_h = chroma. Returns false on failure
    // (caller falls back to the untouched source SRVs).
    bool process(ID3D11DeviceContext* ctx,
                 ID3D11ShaderResourceView* const src[3],
                 uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                 DXGI_FORMAT plane_fmt, float plane_scale,
                 int matrix_sel, int transfer_sel, double video_time_ms);
    // 6 SRVs (Y_L,U_L,V_L,Y_R,U_R,V_R) valid after a successful process().
    ID3D11ShaderResourceView* const* output_srvs() const;
    bool outputs_valid() const;
    // The grid every depth resource here is sized for. side() is retained as a
    // compatibility alias for the horizontal/long-edge dimension.
    int  side() const { return grid_width_; }
    int  grid_width() const { return grid_width_; }
    int  grid_height() const { return grid_height_; }
    std::string status() const;                        // cf. format in Interfaces
    // Debug bypass. `count` is the number of uint16 elements available at q16
    // (0 with nullptr, which disarms). REFUSED -- false, nothing armed, the
    // previous state kept -- unless count == grid_width()*grid_height(): the buffer belongs to
    // the caller, so only the caller knows its length, and validating it here
    // is what keeps a wrong-sized map from being read past its end.
    bool set_test_depth(const uint16_t* q16_or_null, size_t count);
    bool read_plane(ID3D11DeviceContext* ctx, int slot,
                    std::vector<uint8_t>& out, uint32_t& w, uint32_t& h,
                    uint32_t& bpp, std::string& err);  // debug staging readback
    // Seek: re-prime the temporal filter on the next inference instead of blending the
    // EMA across the discontinuity — the same mechanism as the scene-cut snap and as a
    // frame-size change. Cheap and lock-free; safe to call whether or not a worker runs.
    void notify_seek();
    // Post-cut/seek ease-out ramp duration in ms (default 300). Right after a
    // confirmed cut or a consumed seek, process() scales cb.max_disp by
    // t*(2-t) where t = clamp((now - last_snap)/ramp_ms, 0, 1), so the new
    // shot starts flat and inflates instead of teleporting the stereo
    // geometry in one frame. 0 disables the ramp (full disparity always).
    // Debug-only setter (bound as synth3d_set_ramp_ms); never applied to the
    // test-depth bypass path (golden/warp tests keep their full geometry).
    void set_ramp_ms(float ramp_ms);
    // Detach from the shared service (interactive disable; never blocks).
    void request_stop();
    void join_worker();                 // compatibility no-op (service owns worker)
    void stop();

private:
    static constexpr int kRing = 3;                    // readback staging ring depth

    template <typename T> using CP = Microsoft::WRL::ComPtr<T>;

    // --- renderer thread only (called under NativeRenderer::Impl::mtx) ----------
    bool ensure_pipeline(std::string& err);            // shaders/sampler/raster/cbuffer
    bool ensure_prep(std::string& err);                // grid prep RT + staging ring
    bool ensure_depth(std::string& err);               // grid R16_UNORM nearness tex
    bool ensure_warp(uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                     DXGI_FORMAT fmt, std::string& err);   // the six output textures
    void push_readback(ID3D11DeviceContext* ctx,
                       double video_time_ms);           // CopyResource into the ring
    void drain_readback(ID3D11DeviceContext* ctx);     // DO_NOT_WAIT map -> mailbox
    bool upload_depth(ID3D11DeviceContext* ctx);       // mailbox/test -> depthTex_
    void release_depth_grid();                         // drop grid-sized resources only
    void release_gpu();                                // drop every D3D resource

    CP<ID3D11Device>          device_;
    CP<ID3D11VertexShader>    vs_;
    CP<ID3D11PixelShader>     psPrep_, psWarpLuma_, psWarpChroma_;
    CP<ID3D11PixelShader>     psViewLuma_, psViewChroma_;
    CP<ID3D11SamplerState>    sampler_;                // LINEAR / CLAMP
    CP<ID3D11RasterizerState> raster_;                 // CULL_NONE (fullscreen triangle)
    CP<ID3D11Buffer>          cb_;                     // b0: SynthCB (64 bytes)

    CP<ID3D11Texture2D>       prepTex_;                // grid RGBA32F, RTV+SRV
    CP<ID3D11RenderTargetView> prepRtv_;
    CP<ID3D11Texture2D>       stag_[kRing];            // grid RGBA32F STAGING (READ)
    uint64_t                  stag_seq_[kRing] = {0, 0, 0};   // 0 = free, else write order
    // Fallback capture wall time plus exact media PTS of the prep image copied
    // into each slot. Both travel with the pixels through readback/mailbox;
    // worker timing is deliberately a third, compute-only clock.
    std::chrono::steady_clock::time_point stag_capture_time_[kRing]{};
    double                    stag_video_time_ms_[kRing] = {-1.0, -1.0, -1.0};
    int                       stag_write_ = 0;
    uint64_t                  seq_ctr_ = 0;

    CP<ID3D11Texture2D>        depthTex_;              // grid R16_UNORM, DYNAMIC
    CP<ID3D11ShaderResourceView> depthSrv_;

    CP<ID3D11Texture2D>          warpTex_[kNumOut];
    CP<ID3D11RenderTargetView>   warpRtv_[kNumOut];
    CP<ID3D11ShaderResourceView> warpSrv_[kNumOut];
    ID3D11ShaderResourceView*    out_srv_[kNumOut] = {nullptr, nullptr, nullptr,
                                                      nullptr, nullptr, nullptr};

    CP<ID3D11Texture2D> readTex_;                      // debug read_plane staging
    uint32_t            read_w_ = 0, read_h_ = 0;
    DXGI_FORMAT         read_fmt_ = DXGI_FORMAT_UNKNOWN;

    uint32_t    y_w_ = 0, y_h_ = 0, c_w_ = 0, c_h_ = 0;
    DXGI_FORMAT plane_fmt_ = DXGI_FORMAT_UNKNOWN;
    bool        outputs_valid_ = false;
    bool        depth_valid_   = false;                // a nearness map has been uploaded
    bool        depth_dirty_   = false;                // test depth changed -> re-upload

    std::vector<uint16_t> test_depth_;                 // armed test bypass (grid)
    bool                  test_armed_ = false;
    std::vector<float>    in_scratch_;                 // readback -> CHW, off-lock
    int                   grid_width_ = SharedDepthService::kDefaultSide;
    int                   grid_height_ = SharedDepthService::kDefaultSide;
    std::shared_ptr<SharedDepthService> depth_service_;
    uint64_t              client_id_ = 0;
    uint64_t              depth_sequence_ = 0;
    bool                  service_attached_ = false;
    float                 ramp_ms_ = 300.f;          // post-cut/seek ease-out duration

    Synth3DParams params_;                             // Impl::mtx-guarded (renderer side)
};
