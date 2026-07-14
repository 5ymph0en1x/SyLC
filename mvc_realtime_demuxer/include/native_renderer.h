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

#include <cstdint>
#include <string>

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

    // Clear the current back buffer to opaque black and Present (interval 1,
    // matching the current ~1-frame latency the audio offset is tuned against).
    // If a frame has been uploaded and the pipeline is ready, draws the shaded,
    // aspect-correct quad before presenting; otherwise presents black (S1).
    bool present();

    // True if the scRGB HDR color space was accepted by the swapchain/output.
    bool is_hdr() const { return hdr_enabled_; }

    // Human-readable backend description and last error, for smoke tests.
    std::string backend_info() const { return backend_info_; }
    std::string last_error() const { return last_error_; }

    // Release all D3D resources. Idempotent.
    void shutdown();

private:
    bool create_rtv_for_backbuffer();
    void release_backbuffer_views();
    bool create_pipeline();                    // shaders, input layout, sampler, cbuffer, vbuffer
    bool ensure_texture(int slot, uint32_t w, uint32_t h, bool rgba);

    struct Impl;            // holds the ComPtr<> members (kept out of this header)
    Impl* impl_ = nullptr;

    bool        hdr_enabled_   = false;
    bool        pipeline_ready_ = false;
    bool        has_frame_     = false;        // a left-Y plane has been uploaded
    bool        paused_        = false;        // seek/pause gate (guarded by impl_->mtx)
    uint32_t    width_  = 0;                   // backbuffer size
    uint32_t    height_ = 0;
    uint32_t    src_w_  = 0;                   // decoded left-Y size (for 2D aspect)
    uint32_t    src_h_  = 0;
    int         stereo_mode_ = 0;
    std::string backend_info_;
    std::string last_error_;
};

} // namespace sylc

#endif // SYLC_NATIVE_RENDERER
