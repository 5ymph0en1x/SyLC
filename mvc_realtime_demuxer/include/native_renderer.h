// Native D3D11 renderer for SyLC 3D Player (Tokyo #3).
//
// STAGE S1: owns a Win32 HWND-bound, flip-model, FP16 scRGB(HDR) swapchain and
// presents (clears to black) on demand. No decode, no shader, no upload yet —
// this stage only proves the HDR swapchain comes up and presents, and that
// fullscreen toggling preserves the HDR flip-model path.
//
// THREADING CONTRACT (see native_renderer/NATIVE_RENDERER_DESIGN.md §0/§5):
// there is intentionally NO internal present loop. Every method must be called
// from the single owning thread (later: the decode/present thread). Presentation
// is "on arrival", driven by the decoder's existing audio-locked pacing — never
// by a renderer-side clock — to avoid double-pacing the velvet liquid scheduler.
//
// The whole header is gated on SYLC_NATIVE_RENDERER so the module builds
// unchanged when the renderer is disabled (e.g. non-Windows or option OFF).
#pragma once
#ifdef SYLC_NATIVE_RENDERER

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "depth_engine.h"   // kDefaultDepthSide (default synth3d inference grid)

namespace sylc {

class NativeRenderer {
public:
    NativeRenderer();
    ~NativeRenderer();

    NativeRenderer(const NativeRenderer&) = delete;
    NativeRenderer& operator=(const NativeRenderer&) = delete;

    // Create the D3D11 device + flip-model FP16 scRGB swapchain on an existing
    // Win32 window. `hwnd` is the HWND passed from Python as an integer (e.g.
    // int(widget.winId())). `width`/`height` are the initial backbuffer size in
    // physical pixels. Returns false on failure (see last_error()).
    // hdr selects the swapchain FORMAT (which determines how DWM interprets the
    // values — format matters, not just SetColorSpace1):
    //   hdr=false (SDR display): R8G8B8A8_UNORM + default G22/sRGB. DWM treats the
    //     values as gamma-encoded, so the shader's gamma-domain output displays
    //     correctly with NO EOTF (set output_gamma=0). This matches the Qt path on
    //     an SDR display.
    //   hdr=true (HDR display): R16G16B16A16_FLOAT + scRGB G10 (linear). DWM treats
    //     the values as LINEAR, so the shader output must be linearized first
    //     (output_gamma ~2.4) and scaled by the SDR white level.
    bool initialize(uint64_t hwnd, uint32_t width, uint32_t height, bool hdr = false);

    // Flip-model requires ResizeBuffers on window resize (a viewport change is
    // not enough). Releases the RTV, resizes, recreates the RTV. No-op for a
    // degenerate (minimized) size.
    bool resize(uint32_t width, uint32_t height);

    // --- S2: frame upload + shaded draw -------------------------------------
    // Set the shader uniforms (cbuffer b0). stereo_mode: 0=2D,1=framepack,2=SBS,
    // 3=TAB. subtitle_rect is normalized (x,y,w,h). sdr_white = SDRWhiteNits/80.
    // subtitle_disparity: stereoscopic subtitle depth — horizontal disparity
    // normalized to EYE width; > 0 = crossed (overlay floats in FRONT of the
    // screen), 0 = screen depth. Each eye view is shifted by half, in opposite
    // directions. Ignored in 2D mode.
    void set_uniforms(int stereo_mode, int subtitle_enabled,
                      float rect_x, float rect_y, float rect_w, float rect_h,
                      float sdr_white_level, float output_gamma = 0.0f,
                      float subtitle_disparity = 0.0f);

    // Upload one R8 plane. plane_index: 0=Y_L,1=U_L,2=V_L,3=Y_R,4=U_R,5=V_R.
    // The texture is (re)created to (width,height) on first use / size change, so
    // no padding is needed (texture matches source exactly). src_stride is the
    // source row stride in bytes; the copy honors the GPU's mapped RowPitch.
    bool upload_plane(int plane_index, const uint8_t* data,
                      uint32_t width, uint32_t height, uint32_t src_stride);

    // Upload one 16-bit plane into an R16_UNORM texture (10-bit HEVC: yuv420p10le
    // stores the value in the LOW bits). Same plane_index mapping as upload_plane;
    // the texture is (re)created when the FORMAT or dimensions change. src_stride is
    // the source row stride in BYTES (tight = width*2). Pair with set_plane_scale()
    // so the shader rescales the sample back to [0,1] before YUV->RGB.
    bool upload_plane16(int plane_index, const uint16_t* data,
                        uint32_t width, uint32_t height, uint32_t src_stride);

    // Media presentation timestamp of the planes uploaded for the next
    // present, in milliseconds. Negative/non-finite means unavailable and
    // activates the renderer-side steady-clock fallback. This value travels
    // with the exact readback pixels; it is never reconstructed from inference
    // or GUI cadence.
    void set_video_time_ms(double video_time_ms);

    // Per-sample scale multiplied into every Y/U/V texel BEFORE the YUV->RGB math
    // (stored in the cbuffer). 1.0 = 8-bit R8 (identity); 65535/1023 ~= 64.06 maps a
    // 10-bit value stored low in an R16_UNORM texel back to [0,1]. Takes effect on
    // the next present. 8-bit uploads must set this to 1.0.
    void set_plane_scale(float scale);

    // HDR10/PQ color selectors (HEVC). Stored in the cbuffer (c3.y/c3.z). Both 0 (DEFAULT)
    // is the legacy path — BYTE-IDENTICAL for MVC/H.264 and every existing source.
    //   yuv_matrix_sel: 0 = BT.601 limited (legacy), 1 = BT.709 limited, 2 = BT.2020nc limited.
    //   transfer_sel:   0 = legacy gamma/sdr_white, 1 = PQ -> scRGB absolute (HDR display),
    //                   2 = PQ -> tone-mapped SDR (SDR display fallback).
    // Mirrors set_plane_scale: mutates only these two cbuffer fields and re-uploads, so a
    // per-frame color-param change takes effect without re-specifying every other uniform.
    // Takes effect on the next present; serialized by the same mutex as the sibling setters.
    void set_color_params(int yuv_matrix_sel, int transfer_sel);

    // C2: display-aspect override for the packed-frame formats. > 0 forces the display
    // aspect (width/height) used by the letterbox/pillarbox geometry instead of deriving
    // it from the uploaded eye dimensions. Needed for half-SBS/half-TAB, where each eye is
    // squeezed (e.g. 960x1080) yet must display at the ORIGINAL 2D aspect (the packed
    // frame's own W/H). 0.0f = derive from planes (default; full formats, MVC, 2D).
    // 2D, SBS and TAB main-window canvases all preserve this source aspect; only the
    // physical FramePack canvas remains fixed at 1920/2205. Takes effect on the next
    // present; serialized by the same mutex as the sibling setters.
    void set_source_aspect(float aspect);

    // --- synth3d (2D->3D): AI depth + DIBR warp ------------------------------
    // Turns the uploaded 2D source into a synthesized stereo pair, in place: the six
    // plane SRVs the display draw (and cast_encode) consume are substituted with six
    // warp outputs produced on the GPU from a monocular depth map. The depth itself is
    // inferred by a process-wide worker (ONNX Runtime) that NEVER touches D3D or a
    // renderer mutex. Surfaces using the same model share that worker and its latest
    // immutable depth map, so a slow or failed model only lowers the depth refresh
    // rate — decode and present never wait for it (spec Sec.7).
    //
    // set_synth3d(true, ...): create the GPU pipeline and attach to the shared service.
    //   The ORT session is built on its worker (it takes seconds), so this returns
    //   immediately. strength_pct = max disparity as a % of image width; convergence =
    //   the normalized nearness placed at zero parallax (0..1); depth_view replaces the
    //   warp with a false-color visualization of the depth map; diagnostics overlays
    //   depth and the continuous disocclusion mask on the live stereo result.
    //   model_path/ort_dir may be empty when a debug test depth drives the warp.
    //   side preserves the square API. Positive grid_width/grid_height override
    //   it for a fixed rectangular model and are part of the shared-service key.
    //   crop_top/crop_bottom are normalized source-image mattes removed before
    //   inference; the warp maps the rectangular depth back into that same ROI.
    // set_synth3d(false): detach and return to untouched source planes on the next
    //   present. The service remains warm and is never joined under a renderer mutex.
    // synth3d_status(): one parseable line beginning with
    //   "state=... provider=... side=... fps=... source_ms=... update_ms=... "
    //   "age_ms=... clients=... cuts=... motion=... alpha=... scene=... "
    //   "crop=top:bottom:sourceW:sourceH crop_conf=... grid=WxH err=...".
    //   source_ms is media-PTS spacing; update_ms is compute/map cadence.
    //   state is the shared engine's state;
    //   a debug test depth drives the warp independently of it. side is 0 when
    //   nothing is attached (the running line also carries views= before side=).
    bool set_synth3d(bool enabled, float strength_pct = 1.5f, float convergence = 0.5f,
                     bool depth_view = false,
                      const std::wstring& model_path = std::wstring(),
                      const std::wstring& ort_dir = std::wstring(),
                      bool diagnostics = false,
                      int side = kDefaultDepthSide,
                      int grid_width = 0, int grid_height = 0,
                      float crop_top = 0.0f, float crop_bottom = 0.0f);
    std::string synth3d_status() const;
    // Debug/tests. set_test_depth(nullptr, 0) returns to the engine; otherwise q16
    // points at a synth3d_grid_width()*synth3d_grid_height() map that bypasses inference
    // entirely, and `count` is how many uint16 elements that buffer actually holds.
    // Returns false (arming nothing) when count does not match the live grid --
    // the grid follows the depth preset, so a caller CAN hold a map of the wrong
    // size, and neither over-reading it nor cropping it is acceptable. True when
    // no pipeline exists yet: that stays the historic silent no-op, and nothing
    // is read. read_plane stages one warp output back to the CPU (slot 0..5 =
    // Y_L,U_L,V_L,Y_R,U_R,V_R); it MAPs blocking, never on the playback path.
    bool synth3d_set_test_depth(const uint16_t* q16_or_null, size_t count);
    // The live inference grid every depth-side resource is sized for (and the
    // element count per side that set_test_depth demands); 0 before any synth3d
    // pipeline has been created.
    int synth3d_side() const;
    int synth3d_grid_width() const;
    int synth3d_grid_height() const;
    bool synth3d_read_plane(int slot, std::vector<uint8_t>& out, uint32_t& w,
                            uint32_t& h, uint32_t& bpp, std::string& err);
    // Seek: re-prime the temporal depth filter on the next inference rather than let it
    // blend the EMA across the discontinuity (the same mechanism as its scene-cut snap).
    // Cheap, and a no-op when no synth3d pipeline exists.
    void synth3d_notify_seek();
    // Debug: override the post-cut/seek ease-out ramp duration (ms, default
    // 300). 0 disables the ramp (full disparity always). No-op when no
    // synth3d pipeline exists yet.
    void synth3d_set_ramp_ms(float ramp_ms);

    // Upload the RGBA8 subtitle overlay (texture slot t0). Straight alpha.
    bool upload_subtitle(const uint8_t* data,
                         uint32_t width, uint32_t height, uint32_t src_stride);

    // Forget the current frame (present() falls back to clearing black).
    void clear_frame();

    // S3: seek/pause gate. While paused, present() holds the last presented frame
    // (does no GPU work) so a seek can refeed/realloc without racing the present
    // path. All public runtime methods are serialized by an internal mutex, so the
    // presenter thread (present/upload) and the GUI thread (resize/pause) are safe
    // to call concurrently.
    void pause();
    void resume();
    bool is_paused() const { return paused_; }

    // Clear the current back buffer to opaque black and Present. Interval 1 is
    // the timing-authority default; interval 0 is used only for a simultaneous
    // secondary preview so a second swapchain cannot block frame delivery.
    // If a frame has been uploaded and the pipeline is ready, draws the shaded,
    // aspect-correct quad before presenting; otherwise presents black (S1).
    bool present(uint32_t sync_interval = 1);

    // True if the scRGB HDR color space was accepted by the swapchain/output.
    bool is_hdr() const { return hdr_enabled_; }

    // Human-readable backend description and last error, for smoke tests.
    std::string backend_info() const { return backend_info_; }
    std::string last_error() const { return last_error_; }

    // Release all D3D resources. Idempotent.
    void shutdown();

    // --- SyLC Cast (PC sender): NV12-SBS pack + NVENC HEVC encode -----------
    // A cast pipeline (Task-2 SbsNv12Packer + Task-1 NvencEncoder) bound to THIS
    // renderer's D3D11 device. It consumes the SAME six YUV plane SRVs uploaded by
    // set_yuv_frame (srv[1..6] = L Y/U/V, R Y/U/V) — never RGB — packs them into an
    // NV12 3840x1080 side-by-side texture, and encodes that to HEVC. The cast path
    // never calls present(); it only needs the device/context/srv. The three stateful
    // methods (cast_start / cast_encode / cast_stop) are serialized by the same internal
    // mutex as the render/upload path; cast_available() is a lock-free NVENC probe that
    // takes no lock and touches no shared state.
    //
    // cast_available(): true if NVENC (nvEncodeAPI64.dll) is present. Safe on a host
    //   with no NVIDIA GPU (returns false; no device needed).
    // cast_start(mode,fps,bitrate): build the packer + encoder on impl_->device.
    //   mode = "lossless" (LosslessWired, bit-exact) or anything else (CbrLowLatency).
    //   bitrate_bps applies to CBR only (<=0 keeps the encoder default). false on error.
    // cast_encode(pts_ms,force_idr): pack the last-uploaded planes and encode ONE
    //   frame; returns the HEVC Annex-B packets (empty on error, see last_error()).
    // cast_reconfigure(mode,bitrate): hot-change the RUNNING encoder to a new bitrate/mode
    //   mid-stream without tearing down the packer/texture. A same-mode bitrate change goes
    //   through NvEncReconfigureEncoder (seamless, no session teardown); a mode switch the
    //   driver won't reconfigure cleanly falls back to an encoder-only reopen (the packer
    //   and its NV12 texture stay; only the NVENC session cycles + re-registers). Either
    //   way the next encoded frame is an IDR. false on error (see last_error()).
    // cast_stop(): flush + tear down the pipeline. Idempotent.
    bool cast_available() const;
    bool cast_start(const std::string& mode, int fps, int64_t bitrate_bps,
                    bool main10 = false);
    std::vector<std::vector<uint8_t>> cast_encode(int64_t pts_ms, bool force_idr);
    bool cast_reconfigure(const std::string& mode, int64_t bitrate_bps);
    void cast_stop();

private:
    // Plane texture pixel format. Kept free of DXGI/Windows types so this public
    // header needs no d3d headers; mapped to DXGI_FORMAT in the .cpp.
    //   R8    = 8-bit luma/chroma plane (R8_UNORM)
    //   R16   = 16-bit (10-bit HEVC) luma/chroma plane (R16_UNORM)
    //   RGBA8 = straight-alpha subtitle overlay (R8G8B8A8_UNORM)
    enum class TexFormat { R8, R16, RGBA8 };

    bool create_rtv_for_backbuffer();
    void release_backbuffer_views();
    // ResizeBuffers + RTV recreate assuming the impl_->mtx is ALREADY held (used by
    // both the public resize() and the present()-time self-heal so a locked present
    // never re-locks the non-recursive mutex -> deadlock).
    bool resize_backbuffer_locked(uint32_t width, uint32_t height);
    bool create_pipeline();                    // shaders, input layout, sampler, cbuffer, vbuffer
    bool ensure_texture(int slot, uint32_t w, uint32_t h, TexFormat fmt);

    struct Impl;            // holds the ComPtr<> members (kept out of this header)
    Impl* impl_ = nullptr;

    bool        hdr_enabled_   = false;
    bool        pipeline_ready_ = false;
    bool        has_frame_     = false;        // a left-Y plane has been uploaded
    bool        paused_        = false;        // seek/pause gate (guarded by impl_->mtx)
    uint64_t    hwnd_   = 0;                   // owning HWND (for present()-time client-size self-heal)
    uint32_t    width_  = 0;                   // backbuffer size (physical px)
    uint32_t    height_ = 0;
    uint32_t    src_w_  = 0;                   // decoded left-Y size (for 2D aspect)
    uint32_t    src_h_  = 0;
    int         stereo_mode_ = 0;
    float       plane_scale_ = 1.0f;           // cbuffer plane_scale (guarded by impl_->mtx)
    double      video_time_ms_ = -1.0;          // current source-frame PTS (mtx-guarded)
    int         yuv_matrix_sel_ = 0;           // cbuffer yuv_matrix_sel (mtx-guarded), 0=legacy BT.601
    int         transfer_sel_   = 0;           // cbuffer transfer_sel (mtx-guarded), 0=legacy
    float       aspect_ = 0.0f;                 // C2 display-aspect override, 0=derive (mtx-guarded)
    std::string backend_info_;
    std::string last_error_;
};

} // namespace sylc

#endif // SYLC_NATIVE_RENDERER
