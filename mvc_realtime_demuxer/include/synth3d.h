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
//                     consumes the same immutable stabilized geometry map.
//   4. warp         — effective depth/safety are uploaded to an RG16_UNORM texture and two
//                     multiple-render-target passes (luma 2-MRT, chroma 4-MRT) produce
//                     a synthesized stereo pair in six dedicated textures. depth_view
//                     swaps those passes for a false-color visualization of the depth.
//
// THREADING. Every D3D method here runs under NativeRenderer::Impl::mtx. The shared
// worker never touches D3D or a renderer mutex. SharedDepthService keeps one idle
// session warm and sends older/failed sessions to a reaper thread: detaching the
// last surface cannot destroy ORT or wait for CreateSession on presentation.
//
// Windows-only (d3d11.h); compiled into the module inside the BUILD_NATIVE_RENDERER
// block, like native_renderer.cpp / sbs_nv12_packer.cpp.
#pragma once
#include <d3d11.h>
#include <wrl/client.h>
#include <atomic>
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
    // When true, process() replaces `convergence` with the shared service's
    // per-shot suggestion (stabilizer percentile, video-time smoothed). The
    // manual value above remains the fallback before the first map arrives.
    bool  auto_convergence = false;
    // Round 5a: temporal background plate. Disocclusion holes prefer
    // flow-transported, previously SEEN background over the stretch fallback.
    // Off by default (author visual gate); off is byte-identical to before.
    bool  temporal_fill = false;
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
    // Debug-only packed geometry injection. Every pointer addresses `count`
    // uint16 values. Null optional channels default to depth / fully safe / no
    // repair, preserving set_test_depth's historical behavior.
    bool set_test_geometry(const uint16_t* depth, const uint16_t* owned,
                           const uint16_t* safety,
                           const uint16_t* ownership, size_t count);
    // Prototype surface for an externally precomputed human alpha matte.
    // The matte is deliberately independent of the depth grid: normalized UV
    // sampling accepts a full-resolution matte or any aspect-matched proxy.
    // mode 1 protects stereo safety/ownership; mode 2 additionally performs
    // alpha-aware foreground decontamination in the narrow transition band.
    // Passing nullptr disarms the matte and restores the historical shader path.
    bool set_test_matte(const uint8_t* alpha_or_null, uint32_t width,
                        uint32_t height, size_t count, int mode);
    bool read_plane(ID3D11DeviceContext* ctx, int slot,
                    std::vector<uint8_t>& out, uint32_t& w, uint32_t& h,
                    uint32_t& bpp, std::string& err);  // debug staging readback
    // Debug/tests: read the current background plate (grid RGBA16_UNORM:
    // Y,U,V,confidence). Fails when temporal_fill never armed the plate.
    bool read_plate(ID3D11DeviceContext* ctx, std::vector<uint8_t>& out,
                    uint32_t& w, uint32_t& h, std::string& err);
    // Seek: re-prime the temporal filter on the next inference instead of blending the
    // EMA across the discontinuity — the same mechanism as the scene-cut snap and as a
    // frame-size change. Cheap and lock-free; safe to call whether or not a worker runs.
    void notify_seek();
    // Look-ahead advisory (two-filter scout): forwards the presented-relative
    // delays to the depth service. Lock-free, no-op without a service.
    void set_lookahead_advisory(double cut_in_ms, double storm_in_ms);
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
    bool ensure_depth(std::string& err);               // grid RG16_UNORM geometry tex
    bool ensure_matte(std::string& err);               // optional RG8 alpha/distance tex
    bool ensure_warp(uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                     DXGI_FORMAT fmt, std::string& err);   // the six output textures
    void push_readback(ID3D11DeviceContext* ctx,
                       double video_time_ms);           // CopyResource into the ring
    void drain_readback(ID3D11DeviceContext* ctx);     // DO_NOT_WAIT map -> mailbox
    bool upload_depth(ID3D11DeviceContext* ctx,
                      std::string& err);               // mailbox/test -> depthTex_
    bool upload_matte(ID3D11DeviceContext* ctx,
                      std::string& err);               // precomputed alpha -> matteTex_
    void set_local_error(const std::string& err);      // renderer-local GPU failure
    void release_depth_grid();                         // drop grid-sized resources only
    void release_gpu();                                // drop every D3D resource

    CP<ID3D11Device>          device_;
    CP<ID3D11VertexShader>    vs_;
    CP<ID3D11PixelShader>     psPrep_, psWarpLuma_, psWarpChroma_;
    CP<ID3D11PixelShader>     psViewLuma_, psViewChroma_;
    CP<ID3D11PixelShader>     psPlateAccum_;           // round 5a background plate
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

    CP<ID3D11Texture2D>        depthTex_;              // grid RG16_UNORM, DYNAMIC
    CP<ID3D11ShaderResourceView> depthSrv_;

    // Round 5a — temporal background plate (all grid-sized):
    // transport = flow x/y + reliability from the published map; plate =
    // ping-pong accumulation of flow-transported background YUV+confidence.
    CP<ID3D11Texture2D>          transportTex_;        // grid RGBA16_UNORM, DYNAMIC
    CP<ID3D11ShaderResourceView> transportSrv_;
    CP<ID3D11Texture2D>          plateTex_[2];         // grid RGBA16_UNORM, RTV+SRV
    CP<ID3D11RenderTargetView>   plateRtv_[2];
    CP<ID3D11ShaderResourceView> plateSrv_[2];
    CP<ID3D11Texture2D>          plateReadTex_;        // debug read_plate staging
    int     plate_read_ = 0;
    bool    plate_valid_ = false;      // false => clear both before next accum
    bool    map_refreshed_ = false;    // upload_depth uploaded a NEW map this call
    int64_t plate_snap_seen_ = -1;     // service snap stamp consumed by the purge

    CP<ID3D11Texture2D>          matteTex_;            // arbitrary-grid RG8_UNORM, DYNAMIC
    CP<ID3D11ShaderResourceView> matteSrv_;

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

    std::vector<uint8_t> test_matte_;                  // packed RG alpha/boundary-distance
    uint32_t              matte_width_ = 0;
    uint32_t              matte_height_ = 0;
    uint32_t              matte_texture_width_ = 0;
    uint32_t              matte_texture_height_ = 0;
    int                   matte_mode_ = 0;             // 0=off, 1=guard, 2=contour
    bool                  matte_armed_ = false;
    bool                  matte_dirty_ = false;

    std::vector<uint16_t> test_geometry_;              // armed packed test bypass (2*grid)
    bool                  test_armed_ = false;
    std::vector<float>    in_scratch_;                 // readback -> CHW, off-lock
    std::vector<float>    aspect_luma_scratch_;        // full-frame luma from prep alpha
    int                   grid_width_ = SharedDepthService::kDefaultSide;
    int                   grid_height_ = SharedDepthService::kDefaultSide;
    std::shared_ptr<SharedDepthService> depth_service_;
    uint64_t              client_id_ = 0;
    uint64_t              depth_sequence_ = 0;
    bool                  service_attached_ = false;
    float                 ramp_ms_ = 300.f;          // post-cut/seek ease-out duration
    std::string           local_error_;              // GPU/resource failure for status/UI

    Synth3DParams params_;                             // Impl::mtx-guarded (renderer side)
};
