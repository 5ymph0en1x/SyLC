// Native D3D11 renderer — STAGES S1 (swapchain/present) + S2 (shaded YUV draw).
// See native_renderer.h and native_renderer/NATIVE_RENDERER_DESIGN.md.
#ifdef SYLC_NATIVE_RENDERER

#include "native_renderer.h"
#include "nvenc_encoder.h"             // SyLC Cast: Task-1 NVENC HEVC encoder wrapper
#include "sbs_nv12_packer.h"           // SyLC Cast: Task-2 YUV -> NV12 SBS packer
#include "shader_resource_ids.h"
#include "shader_resources.h"
#include "synth3d.h"                   // synth3d (2D->3D): depth prep + DIBR warp passes

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>

using Microsoft::WRL::ComPtr;

namespace sylc {

static std::string d3d_error(const char* operation, HRESULT hr,
                             ID3D11Device* device = nullptr) {
    char buf[192] = {};
    const HRESULT removed = device ? device->GetDeviceRemovedReason() : S_OK;
    if (device && FAILED(removed)) {
        std::snprintf(buf, sizeof(buf), "%s failed (hr=0x%08lX, removed=0x%08lX)",
                      operation, static_cast<unsigned long>(hr),
                      static_cast<unsigned long>(removed));
    } else {
        std::snprintf(buf, sizeof(buf), "%s failed (hr=0x%08lX)",
                      operation, static_cast<unsigned long>(hr));
    }
    return std::string(buf);
}

// Texture slots match the HLSL register bindings:
//   t0 = subtitle (RGBA8), t1..t3 = Y/U/V left, t4..t6 = Y/U/V right,
//   t7 = stereoscopic playback HUD (RGBA8). Appending the HUD keeps every
// existing YUV/synth3d/cast slot stable.
static constexpr int kNumTex = 8;

// cbuffer 'buf' (register b0) — layout must match the HLSL packoffsets:
//   c0.x int stereo_mode, c0.y int subtitle_enabled, c0.z float subtitle_disparity,
//   c1 float4 subtitle_rect, c2.x float sdr_white_level, c3.x float plane_scale,
//   c4 HUD flags/depth/opacity, c5 float4 HUD rect. 96 bytes total (6 x 16).
struct FrameCB {
    int   stereo_mode;        // c0.x (offset 0)
    int   subtitle_enabled;   // c0.y (offset 4)
    float subtitle_disparity; // c0.z (offset 8)   normalized eye-width; >0 = pop-out
    int   _pad0;              // c0.w
    float subtitle_rect[4];   // c1   (offset 16)
    float sdr_white_level;    // c2.x (offset 32)
    float output_gamma;       // c2.y (offset 36)  EOTF exponent; <=0 disables
    float fp_vfill;           // c2.z (offset 40)  FramePack: eye vertical fill of a 1080 slot
    float fp_hfill;           // c2.w (offset 44)  FramePack: eye horizontal fill
    float plane_scale;        // c3.x (offset 48)  per-sample scale before YUV->RGB (10-bit R16)
    int   yuv_matrix_sel;     // c3.y (offset 52)  0=BT.601 legacy, 1=BT.709, 2=BT.2020nc
    int   transfer_sel;       // c3.z (offset 56)  0=legacy, 1=PQ->scRGB abs, 2=PQ->tonemap SDR
    float _pad1;              // c3.w (offset 60)
    int   hud_enabled;        // c4.x (offset 64)
    float hud_disparity;      // c4.y (offset 68) normalized eye-width; >0 = pop-out
    float hud_opacity;        // c4.z (offset 72)
    float _pad2;              // c4.w (offset 76)
    float hud_rect[4];        // c5   (offset 80)
};
static_assert(sizeof(FrameCB) == 96, "cbuffer must be 96 bytes");

// SyLC Cast pipeline: the Task-2 NV12-SBS packer + the Task-1 NVENC encoder, both
// bound to the renderer's D3D11 device. Held by value (both are non-copyable, unique
// resource owners) inside a struct the Impl owns via unique_ptr; created on
// cast_start(), torn down on cast_stop()/shutdown(). File-local — the header only
// forward-declares nothing of it (opaque behind Impl), exactly like the ComPtr members.
struct CastPipeline {
    sylc::SbsNv12Packer packer;   // 6 YUV plane SRVs -> NV12/P010 3840x1080 SBS texture
    sylc::NvencEncoder  enc;      // NV12/P010 texture -> HEVC NAL bytes
    void*               regInput = nullptr;   // NVENC-registered handle for the NV12 tex
    uint32_t            fps      = 24;         // frame rate from cast_start (for reconfigure)
    bool                main10   = false;      // Main10/P010 HDR session (for reconfigure)
    bool                active   = false;
};

struct NativeRenderer::Impl {
    ComPtr<ID3D11Device>           device;
    ComPtr<ID3D11DeviceContext>    context;
    ComPtr<IDXGISwapChain1>        swapchain;
    ComPtr<ID3D11RenderTargetView> rtv;

    // Pipeline (S2)
    ComPtr<ID3D11VertexShader>     vs;
    ComPtr<ID3D11PixelShader>      ps;
    ComPtr<ID3D11InputLayout>      input_layout;
    ComPtr<ID3D11Buffer>           vbuffer;
    ComPtr<ID3D11Buffer>           cbuffer;
    ComPtr<ID3D11SamplerState>     sampler;
    ComPtr<ID3D11RasterizerState>  raster;

    // Textures + SRVs
    ComPtr<ID3D11Texture2D>          tex[kNumTex];
    ComPtr<ID3D11ShaderResourceView> srv[kNumTex];
    uint32_t   tex_w[kNumTex] = {0};
    uint32_t   tex_h[kNumTex] = {0};
    TexFormat  tex_fmt[kNumTex] = { TexFormat::R8, TexFormat::R8, TexFormat::R8,
                                    TexFormat::R8, TexFormat::R8, TexFormat::R8,
                                    TexFormat::R8, TexFormat::R8 };

    // SyLC Cast: the offscreen NV12-SBS pack + NVENC encode pipeline. Null until
    // cast_start() builds it on `device`; reset by cast_stop()/shutdown(). Held by
    // unique_ptr so the non-copyable packer/encoder live at a stable address.
    std::unique_ptr<CastPipeline> cast;

    // synth3d (2D->3D): the depth-prep / inference / DIBR-warp pipeline. Null until
    // set_synth3d(true) builds it on `device`; inference is supplied by the
    // process-wide SharedDepthService while warp textures stay device-local. Its
    // six warp outputs are substituted for srv[1..6] in present()/cast_encode().
    std::unique_ptr<Synth3D> synth3d;
    Synth3DParams            synth3d_params;

    // Last-written constant-buffer contents. set_uniforms rebuilds this fully each
    // call; set_plane_scale mutates only .plane_scale and re-uploads — so a
    // per-frame plane_scale change takes effect without the caller re-specifying
    // every other uniform.
    FrameCB cb = {};

    // Serializes all GPU/context/swapchain access so the presenter thread
    // (present/upload) and the GUI thread (resize/pause/shutdown) never touch the
    // D3D11 immediate context simultaneously. Non-recursive: no locked public
    // method calls another locked public method (setup paths take no lock).
    std::mutex mtx;
};

NativeRenderer::NativeRenderer() : impl_(new Impl()) {}

NativeRenderer::~NativeRenderer() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

// ---------------------------------------------------------------------------
// S1: device + swapchain
// ---------------------------------------------------------------------------
bool NativeRenderer::initialize(uint64_t hwnd, uint32_t width, uint32_t height, bool hdr) {
    last_error_.clear();
    if (!hwnd)  { last_error_ = "initialize: null HWND"; return false; }
    if (!impl_) { last_error_ = "initialize: no impl";  return false; }

    width_  = width  ? width  : 1u;
    height_ = height ? height : 1u;
    hwnd_   = hwnd;   // remembered so present() can self-heal a drifted backbuffer size
    aspect_ = 0.0f;   // C2: fresh session derives display aspect from planes until overridden

    const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    const D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL got = D3D_FEATURE_LEVEL_11_0;

    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
        want, static_cast<UINT>(sizeof(want) / sizeof(want[0])),
        D3D11_SDK_VERSION, &impl_->device, &got, &impl_->context);
    if (FAILED(hr)) {
        hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            &want[1], 1, D3D11_SDK_VERSION, &impl_->device, &got, &impl_->context);
        if (FAILED(hr)) { last_error_ = d3d_error("D3D11CreateDevice", hr); return false; }
    }

    ComPtr<IDXGIDevice> dxgiDevice;
    if (FAILED(impl_->device.As(&dxgiDevice))) { last_error_ = "QI IDXGIDevice failed"; return false; }

    // DXGI queues three frames ahead by default. For a player whose video clock
    // is mpv's, that buys nothing -- the frames are already paced upstream --
    // and costs up to three refreshes of latency plus an irregular Present()
    // block once the queue saturates, which lands squarely on the p99 frame
    // time that Z2A_VALIDATION.md calls the deciding number. One frame of
    // queue keeps the GPU fed while making the block predictable.
    // Device-scoped on purpose: SetMaximumFrameLatency on IDXGISwapChain2
    // requires the waitable-object flag, which would mean the caller must wait
    // before rendering -- a second pacing authority alongside the presenter.
    unsigned frame_latency = 0;
    ComPtr<IDXGIDevice1> dxgiDevice1;
    if (SUCCEEDED(dxgiDevice.As(&dxgiDevice1)) && dxgiDevice1) {
        if (SUCCEEDED(dxgiDevice1->SetMaximumFrameLatency(1))) frame_latency = 1;
    }

    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDevice->GetAdapter(&adapter))) { last_error_ = "GetAdapter failed"; return false; }
    ComPtr<IDXGIFactory2> factory;
    if (FAILED(adapter->GetParent(IID_PPV_ARGS(&factory)))) { last_error_ = "GetParent IDXGIFactory2 failed"; return false; }

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width              = width_;
    sd.Height             = height_;
    // Format determines DWM's interpretation: FP16 -> scRGB linear (HDR);
    // R8G8B8A8_UNORM -> default sRGB/gamma (SDR, displays gamma-domain output as-is).
    sd.Format             = hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count   = 1;
    sd.BufferUsage        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    // Three buffers under the flip model: with two, the GPU cannot start the
    // next frame until the scanned-out buffer is released, so a frame that
    // arrives just after a vsync waits a whole refresh. The third buffer costs
    // one backbuffer of VRAM (~16 MB at 1080p FP16) and removes that stall;
    // the queue depth stays bounded by the frame latency set above.
    sd.BufferCount        = 3;
    sd.Scaling            = DXGI_SCALING_STRETCH;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode          = DXGI_ALPHA_MODE_IGNORE;

    const HWND win = reinterpret_cast<HWND>(static_cast<uintptr_t>(hwnd));
    hr = factory->CreateSwapChainForHwnd(impl_->device.Get(), win, &sd, nullptr, nullptr, &impl_->swapchain);
    if (FAILED(hr)) {
        last_error_ = d3d_error("CreateSwapChainForHwnd", hr, impl_->device.Get());
        return false;
    }
    factory->MakeWindowAssociation(win, DXGI_MWA_NO_ALT_ENTER);

    // Color space. The shader output is GAMMA-ENCODED (BT.601 matrix, no
    // linearization). Forcing scRGB linear (G10) makes the compositor treat it as
    // linear -> washed out. Leaving the DXGI default (G22/gamma) displays the
    // gamma-encoded output with correct contrast — matching the Qt renderer.
    hdr_enabled_ = false;
    const char* cs_name = "SDR-8bit-G22(gamma)";
    if (hdr) {
        ComPtr<IDXGISwapChain3> sc3;
        if (SUCCEEDED(impl_->swapchain.As(&sc3))) {
            const DXGI_COLOR_SPACE_TYPE cs = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709; // scRGB linear
            UINT support = 0;
            if (SUCCEEDED(sc3->CheckColorSpaceSupport(cs, &support)) &&
                (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
                if (SUCCEEDED(sc3->SetColorSpace1(cs))) { hdr_enabled_ = true; cs_name = "HDR-FP16-scRGB-G10(linear)"; }
                else cs_name = "FP16-scRGB-set-failed";
            } else cs_name = "FP16-scRGB-unsupported";
        }
    }

    if (!create_rtv_for_backbuffer()) return false;
    if (!create_pipeline()) return false;   // sets last_error_ on failure

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "D3D11 flip-model | FL=0x%04x | %ux%u | %s | buffers=%u | latency=%s | pipeline=%s",
        static_cast<unsigned>(got), width_, height_, cs_name,
        static_cast<unsigned>(sd.BufferCount),
        frame_latency ? "1" : "driver-default",
        pipeline_ready_ ? "ready" : "FAILED");
    backend_info_ = buf;
    return true;
}

bool NativeRenderer::create_rtv_for_backbuffer() {
    ComPtr<ID3D11Texture2D> backbuffer;
    if (FAILED(impl_->swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer)))) {
        last_error_ = "GetBuffer(0) failed"; return false;
    }
    if (FAILED(impl_->device->CreateRenderTargetView(backbuffer.Get(), nullptr, &impl_->rtv))) {
        last_error_ = "CreateRenderTargetView failed"; return false;
    }
    return true;
}

void NativeRenderer::release_backbuffer_views() { if (impl_) impl_->rtv.Reset(); }

// Assumes impl_->mtx is HELD. Releases the RTV, ResizeBuffers to (width,height),
// recreates the RTV. Callers: resize() (locks then calls) and present()'s self-heal
// (already locked). No-op-success on a degenerate size.
bool NativeRenderer::resize_backbuffer_locked(uint32_t width, uint32_t height) {
    if (!impl_ || !impl_->swapchain) { last_error_ = "resize before initialize"; return false; }
    if (width == 0 || height == 0) return true;
    if (width == width_ && height == height_ && impl_->rtv) return true; // already correct
    release_backbuffer_views();
    const HRESULT hr = impl_->swapchain->ResizeBuffers(
        0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    if (FAILED(hr)) {
        last_error_ = d3d_error("ResizeBuffers", hr, impl_->device.Get());
        return false;
    }
    width_ = width; height_ = height;
    return create_rtv_for_backbuffer();
}

bool NativeRenderer::resize(uint32_t width, uint32_t height) {
    if (!impl_ || !impl_->swapchain) { last_error_ = "resize before initialize"; return false; }
    if (width == 0 || height == 0) return true;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return resize_backbuffer_locked(width, height);
}

// ---------------------------------------------------------------------------
// S2: pipeline, textures, uniforms, upload, shaded draw
// ---------------------------------------------------------------------------
bool NativeRenderer::create_pipeline() {
    pipeline_ready_ = false;

    ShaderBytecode vsBytecode;
    ShaderBytecode psBytecode;
    if (!load_shader_bytecode(IDR_SYLC_NATIVE_VS, vsBytecode, last_error_))
        return false;
    if (!load_shader_bytecode(IDR_SYLC_NATIVE_PS, psBytecode, last_error_))
        return false;

    if (FAILED(impl_->device->CreateVertexShader(vsBytecode.data, vsBytecode.size,
                                                 nullptr, &impl_->vs))) {
        last_error_ = "CreateVertexShader failed"; return false;
    }
    if (FAILED(impl_->device->CreatePixelShader(psBytecode.data, psBytecode.size,
                                                nullptr, &impl_->ps))) {
        last_error_ = "CreatePixelShader failed"; return false;
    }

    // Input layout matches the VS SPIRV_Cross_Input: position@TEXCOORD0, texCoord@TEXCOORD1.
    const D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 8,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    if (FAILED(impl_->device->CreateInputLayout(layout, 2, vsBytecode.data,
                                                vsBytecode.size, &impl_->input_layout))) {
        last_error_ = "CreateInputLayout failed"; return false;
    }

    // Fullscreen-quad triangle strip: position.xy, texcoord.xy (texcoord NOT flipped
    // here — the shader applies y_flipped). Identical to the Qt renderer's vertices.
    const float verts[16] = {
        -1.f, -1.f,  0.f, 0.f,   // bottom-left
         1.f, -1.f,  1.f, 0.f,   // bottom-right
        -1.f,  1.f,  0.f, 1.f,   // top-left
         1.f,  1.f,  1.f, 1.f,   // top-right
    };
    D3D11_BUFFER_DESC vbd = {};
    vbd.ByteWidth = sizeof(verts);
    vbd.Usage     = D3D11_USAGE_IMMUTABLE;
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA vinit = {}; vinit.pSysMem = verts;
    if (FAILED(impl_->device->CreateBuffer(&vbd, &vinit, &impl_->vbuffer))) {
        last_error_ = "CreateBuffer(vertex) failed"; return false;
    }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth      = sizeof(FrameCB);
    cbd.Usage          = D3D11_USAGE_DEFAULT;
    cbd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(impl_->device->CreateBuffer(&cbd, nullptr, &impl_->cbuffer))) {
        last_error_ = "CreateBuffer(constant) failed"; return false;
    }
    // Default uniforms (2D, no subtitle, SDR white = 1.0, no EOTF).
    set_uniforms(0, 0, 0.f, 0.f, 1.f, 1.f, 1.0f, 0.0f);

    D3D11_SAMPLER_DESC samp = {};
    samp.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samp.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.MinLOD   = 0.f;
    samp.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(impl_->device->CreateSamplerState(&samp, &impl_->sampler))) {
        last_error_ = "CreateSamplerState failed"; return false;
    }

    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode        = D3D11_FILL_SOLID;
    rs.CullMode        = D3D11_CULL_NONE;
    rs.DepthClipEnable = TRUE;
    if (FAILED(impl_->device->CreateRasterizerState(&rs, &impl_->raster))) {
        last_error_ = "CreateRasterizerState failed"; return false;
    }

    // RGBA overlay slots are always valid: transparent 1x1 textures keep their
    // SRVs non-null even while subtitles/HUD are disabled.
    static const uint8_t kTransparent[4] = { 0, 0, 0, 0 };
    if (!upload_subtitle(kTransparent, 1, 1, 4)) return false;
    if (!upload_hud(kTransparent, 1, 1, 4)) return false;

    pipeline_ready_ = true;
    return true;
}

bool NativeRenderer::ensure_texture(int slot, uint32_t w, uint32_t h, TexFormat fmt) {
    if (slot < 0 || slot >= kNumTex) { last_error_ = "ensure_texture: bad slot"; return false; }
    if (w == 0 || h == 0) { last_error_ = "ensure_texture: zero size"; return false; }
    if (impl_->tex[slot] && impl_->tex_w[slot] == w && impl_->tex_h[slot] == h &&
        impl_->tex_fmt[slot] == fmt) {
        return true; // reuse
    }
    // Recreate when the FORMAT or the dimensions change (e.g. R8 8-bit -> R16
    // 10-bit on a codec switch, not just a resolution change).
    impl_->srv[slot].Reset();
    impl_->tex[slot].Reset();

    DXGI_FORMAT dxfmt = DXGI_FORMAT_R8_UNORM;
    switch (fmt) {
        case TexFormat::R16:   dxfmt = DXGI_FORMAT_R16_UNORM;      break;
        case TexFormat::RGBA8: dxfmt = DXGI_FORMAT_R8G8B8A8_UNORM; break;
        case TexFormat::R8:    default: dxfmt = DXGI_FORMAT_R8_UNORM; break;
    }

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = w;
    td.Height           = h;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = dxfmt;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(impl_->device->CreateTexture2D(&td, nullptr, &impl_->tex[slot]))) {
        last_error_ = "CreateTexture2D failed"; return false;
    }
    if (FAILED(impl_->device->CreateShaderResourceView(impl_->tex[slot].Get(), nullptr, &impl_->srv[slot]))) {
        last_error_ = "CreateShaderResourceView failed"; return false;
    }
    impl_->tex_w[slot] = w; impl_->tex_h[slot] = h; impl_->tex_fmt[slot] = fmt;

    // Init-clear to limited-range black + neutral chroma (subtitle/RGBA -> 0).
    // Prevents garbage on first present/resize. For R16 the 10-bit values live in
    // the low bits: Y black = 64, chroma neutral = 512 (== 8-bit 16/128 << 2).
    const bool isY = (slot == 1 || slot == 4);
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (SUCCEEDED(impl_->context->Map(impl_->tex[slot].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        auto* dst = static_cast<uint8_t*>(m.pData);
        if (fmt == TexFormat::R16) {
            const uint16_t init16 = isY ? 64u : 512u;
            for (uint32_t r = 0; r < h; ++r) {
                auto* row = reinterpret_cast<uint16_t*>(dst + r * m.RowPitch);
                for (uint32_t x = 0; x < w; ++x) row[x] = init16;
            }
        } else {
            const uint8_t  init     = isY ? 16 : (fmt == TexFormat::RGBA8 ? 0 : 128);
            const uint32_t rowBytes = (fmt == TexFormat::RGBA8) ? (w * 4) : w;
            for (uint32_t r = 0; r < h; ++r) std::memset(dst + r * m.RowPitch, init, rowBytes);
        }
        impl_->context->Unmap(impl_->tex[slot].Get(), 0);
    }
    return true;
}

// Free helper that touches only public D3D types (not the private Impl).
static HRESULT upload_to_tex(ID3D11DeviceContext* ctx, ID3D11Texture2D* tex,
                             const uint8_t* data, uint32_t w, uint32_t h,
                             uint32_t srcStride, uint32_t bytesPerPixel) {
    D3D11_MAPPED_SUBRESOURCE m = {};
    const HRESULT hr = ctx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    if (FAILED(hr)) return hr;
    auto* dst = static_cast<uint8_t*>(m.pData);
    const uint32_t rowBytes = w * bytesPerPixel;
    for (uint32_t r = 0; r < h; ++r)
        std::memcpy(dst + r * m.RowPitch, data + r * srcStride, rowBytes);
    ctx->Unmap(tex, 0);
    return S_OK;
}

bool NativeRenderer::upload_plane(int plane_index, const uint8_t* data,
                                  uint32_t width, uint32_t height, uint32_t src_stride) {
    if (!impl_ || !impl_->context) { last_error_ = "upload_plane before initialize"; return false; }
    if (plane_index < 0 || plane_index > 5) { last_error_ = "upload_plane: bad index"; return false; }
    if (!data) { last_error_ = "upload_plane: null data"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    const int slot = plane_index + 1; // 0->t1 .. 5->t6
    if (!ensure_texture(slot, width, height, TexFormat::R8)) return false;
    const HRESULT hr = upload_to_tex(
        impl_->context.Get(), impl_->tex[slot].Get(),
        data, width, height, src_stride, 1);
    if (FAILED(hr)) {
        last_error_ = d3d_error("Map(plane)", hr, impl_->device.Get());
        return false;
    }
    if (plane_index == 0) { src_w_ = width; src_h_ = height; has_frame_ = true; }
    return true;
}

bool NativeRenderer::upload_plane16(int plane_index, const uint16_t* data,
                                    uint32_t width, uint32_t height, uint32_t src_stride) {
    if (!impl_ || !impl_->context) { last_error_ = "upload_plane16 before initialize"; return false; }
    if (plane_index < 0 || plane_index > 5) { last_error_ = "upload_plane16: bad index"; return false; }
    if (!data) { last_error_ = "upload_plane16: null data"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    const int slot = plane_index + 1; // 0->t1 .. 5->t6
    if (!ensure_texture(slot, width, height, TexFormat::R16)) return false;
    // src_stride is in BYTES; upload_to_tex copies width*2 bytes per row.
    const HRESULT hr = upload_to_tex(
        impl_->context.Get(), impl_->tex[slot].Get(),
        reinterpret_cast<const uint8_t*>(data), width, height, src_stride, 2);
    if (FAILED(hr)) {
        last_error_ = d3d_error("Map(plane16)", hr, impl_->device.Get());
        return false;
    }
    if (plane_index == 0) { src_w_ = width; src_h_ = height; has_frame_ = true; }
    return true;
}

bool NativeRenderer::upload_subtitle(const uint8_t* data, uint32_t width, uint32_t height,
                                     uint32_t src_stride) {
    if (!impl_ || !impl_->context) { last_error_ = "upload_subtitle before initialize"; return false; }
    if (!data) { last_error_ = "upload_subtitle: null data"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!ensure_texture(0, width, height, TexFormat::RGBA8)) return false;
    const HRESULT hr = upload_to_tex(
        impl_->context.Get(), impl_->tex[0].Get(),
        data, width, height, src_stride, 4);
    if (FAILED(hr)) {
        last_error_ = d3d_error("Map(subtitle)", hr, impl_->device.Get());
        return false;
    }
    return true;
}

bool NativeRenderer::upload_hud(const uint8_t* data, uint32_t width, uint32_t height,
                                uint32_t src_stride) {
    if (!impl_ || !impl_->context) { last_error_ = "upload_hud before initialize"; return false; }
    if (!data) { last_error_ = "upload_hud: null data"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!ensure_texture(7, width, height, TexFormat::RGBA8)) return false;
    const HRESULT hr = upload_to_tex(
        impl_->context.Get(), impl_->tex[7].Get(),
        data, width, height, src_stride, 4);
    if (FAILED(hr)) {
        last_error_ = d3d_error("Map(hud)", hr, impl_->device.Get());
        return false;
    }
    return true;
}

// Store the per-sample plane scale in the cbuffer. set_uniforms rewrites the full
// cbuffer (including this value from plane_scale_) each frame, so this only needs
// to re-upload when the value actually changes — the plane upload calls this after
// set_uniforms has run, so the change takes effect on the same present.
void NativeRenderer::set_plane_scale(float scale) {
    if (!impl_ || !impl_->cbuffer || !impl_->context) { plane_scale_ = scale; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (plane_scale_ == scale && impl_->cb.plane_scale == scale) return; // no GPU write needed
    plane_scale_ = scale;
    impl_->cb.plane_scale = scale;
    impl_->context->UpdateSubresource(impl_->cbuffer.Get(), 0, nullptr, &impl_->cb, 0, 0);
}

// HDR10/PQ color selectors. Mirrors set_plane_scale: mutate only the two cbuffer fields
// (c3.y/c3.z) and re-upload, short-circuiting when unchanged. set_uniforms rewrites the
// full cbuffer (carrying yuv_matrix_sel_/transfer_sel_) each frame, and the widget forwards
// this AFTER set_uniforms, so a per-frame change lands on the same present. Defaults 0/0.
void NativeRenderer::set_color_params(int yuv_matrix_sel, int transfer_sel) {
    if (!impl_ || !impl_->cbuffer || !impl_->context) {
        yuv_matrix_sel_ = yuv_matrix_sel; transfer_sel_ = transfer_sel; return;
    }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (yuv_matrix_sel_ == yuv_matrix_sel && transfer_sel_ == transfer_sel &&
        impl_->cb.yuv_matrix_sel == yuv_matrix_sel && impl_->cb.transfer_sel == transfer_sel) {
        return; // no GPU write needed
    }
    yuv_matrix_sel_ = yuv_matrix_sel;
    transfer_sel_   = transfer_sel;
    impl_->cb.yuv_matrix_sel = yuv_matrix_sel;
    impl_->cb.transfer_sel   = transfer_sel;
    impl_->context->UpdateSubresource(impl_->cbuffer.Get(), 0, nullptr, &impl_->cb, 0, 0);
}

// C2: display-aspect override. It only affects the CPU-side viewport/letterbox math in
// set_uniforms (framepack per-eye fit) and present (every main-window layout), not the
// cbuffer, so there is no GPU write here — just a mutex-guarded store, mirroring how
// set_plane_scale serializes against the geometry readers.
void NativeRenderer::set_source_aspect(float aspect) {
    if (!impl_) { aspect_ = aspect; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    aspect_ = aspect;
}

void NativeRenderer::set_video_time_ms(double video_time_ms) {
    const double value =
        std::isfinite(video_time_ms) && video_time_ms >= 0.0
            ? video_time_ms : -1.0;
    if (!impl_) { video_time_ms_ = value; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    video_time_ms_ = value;
}

void NativeRenderer::set_synth3d_output_eye(int eye) {
    const int selected = eye == 1 ? 1 : 0;
    if (!impl_) { synth3d_output_eye_ = selected; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    synth3d_output_eye_ = selected;
}

void NativeRenderer::set_uniforms(int stereo_mode, int subtitle_enabled,
                                  float rx, float ry, float rw, float rh,
                                  float sdr_white, float output_gamma,
                                  float subtitle_disparity) {
    if (!impl_ || !impl_->cbuffer || !impl_->context) { stereo_mode_ = stereo_mode; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    stereo_mode_ = stereo_mode;
    FrameCB cb = {};
    // HUD state has an independent setter/cadence. Carry it across the
    // per-frame subtitle/color rewrite instead of blanking it every frame.
    cb.hud_enabled    = impl_->cb.hud_enabled;
    cb.hud_disparity  = impl_->cb.hud_disparity;
    cb.hud_opacity    = impl_->cb.hud_opacity;
    std::copy_n(impl_->cb.hud_rect, 4, cb.hud_rect);
    cb.stereo_mode        = stereo_mode;
    cb.subtitle_enabled   = subtitle_enabled;
    cb.subtitle_disparity = subtitle_disparity;
    cb.subtitle_rect[0] = rx; cb.subtitle_rect[1] = ry;
    cb.subtitle_rect[2] = rw; cb.subtitle_rect[3] = rh;
    cb.sdr_white_level  = sdr_white;
    cb.output_gamma     = output_gamma;
    // Carry the current plane_scale (set by the last set_yuv_frame/16) so a full
    // cbuffer rewrite here never clobbers it. set_plane_scale mutates just this
    // field afterward when the plane upload changes it.
    cb.plane_scale      = plane_scale_;
    // Same carry for the HDR10/PQ selectors (set_color_params mutates just these two).
    cb.yuv_matrix_sel   = yuv_matrix_sel_;
    cb.transfer_sel     = transfer_sel_;
    // FramePack letterbox: fit the decoded eye (src_w_ x src_h_) into a 1920x1080
    // slot preserving aspect — a non-16:9 eye (e.g. Full-SBS 1920x1012) gets black
    // bars instead of a vertical stretch. 1.0/1.0 = fills the slot (16:9 / MVC).
    float vfill = 1.0f, hfill = 1.0f;
    if (src_w_ > 0 && src_h_ > 0) {
        // C2: use the display-aspect override when set (half-SBS/half-TAB), else derive
        // the per-eye aspect from the uploaded plane dimensions.
        float eye = aspect_ > 0.0f ? aspect_ : float(src_w_) / float(src_h_);
        const float slot = 1920.0f / 1080.0f;
        if (eye >= slot) { hfill = 1.0f; vfill = slot / eye; }
        else             { vfill = 1.0f; hfill = eye / slot; }
    }
    cb.fp_vfill = vfill;
    cb.fp_hfill = hfill;
    impl_->cb = cb;   // cache so set_plane_scale can re-upload with the rest intact
    impl_->context->UpdateSubresource(impl_->cbuffer.Get(), 0, nullptr, &impl_->cb, 0, 0);
}

void NativeRenderer::set_hud_state(bool enabled,
                                   float rx, float ry, float rw, float rh,
                                   float disparity, float opacity) {
    if (!impl_ || !impl_->cbuffer || !impl_->context) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->cb.hud_enabled = enabled ? 1 : 0;
    impl_->cb.hud_disparity = std::isfinite(disparity) ? disparity : 0.0f;
    impl_->cb.hud_opacity = std::clamp(opacity, 0.0f, 1.0f);
    impl_->cb.hud_rect[0] = rx;
    impl_->cb.hud_rect[1] = ry;
    // windows.h defines max as a macro in this translation unit; explicit
    // conditionals keep this header-order independent.
    impl_->cb.hud_rect[2] = rw > 0.0f ? rw : 0.0f;
    impl_->cb.hud_rect[3] = rh > 0.0f ? rh : 0.0f;
    impl_->context->UpdateSubresource(impl_->cbuffer.Get(), 0, nullptr, &impl_->cb, 0, 0);
}

void NativeRenderer::clear_frame() {
    // C2: reset the display-aspect override so a subsequent full/MVC/2D source derives its
    // aspect from planes again (the widget/player also re-sets it per frame).
    if (impl_) {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        has_frame_ = false;
        aspect_ = 0.0f;
        video_time_ms_ = -1.0;
    } else {
        has_frame_ = false;
        aspect_ = 0.0f;
        video_time_ms_ = -1.0;
    }
}

void NativeRenderer::pause() {
    if (!impl_) { paused_ = true; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    paused_ = true;
}

void NativeRenderer::resume() {
    if (!impl_) { paused_ = false; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    paused_ = false;
}

bool NativeRenderer::present(uint32_t sync_interval) {
    if (!impl_ || !impl_->swapchain || !impl_->context) { last_error_ = "present before initialize"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (paused_) return true;   // seek/pause gate: hold last frame, no GPU work

    // Self-heal a drifted backbuffer. A resize event can be MISSED (fake-fullscreen
    // SetWindowPos with SWP_NOZORDER, a DPI change, an off-thread SetWindowPos) leaving
    // the swapchain a few px off the true client area — DXGI then stretches (blur) or the
    // aspect-fit viewport leaves an uncovered edge column. GetClientRect is in PHYSICAL
    // pixels, so syncing to it every present ALSO corrects a widget that resized with
    // logical (DPI-unscaled) sizes. Best-effort: the rtv guard below covers any failure.
    if (hwnd_) {
        RECT rc = {};
        if (GetClientRect(reinterpret_cast<HWND>(static_cast<uintptr_t>(hwnd_)), &rc)) {
            const uint32_t cw = static_cast<uint32_t>(rc.right - rc.left);
            const uint32_t ch = static_cast<uint32_t>(rc.bottom - rc.top);
            if (cw > 0 && ch > 0 && (cw != width_ || ch != height_) &&
                !resize_backbuffer_locked(cw, ch))
                return false;
        }
    }

    if (!impl_->rtv && !create_rtv_for_backbuffer()) return false;

    ID3D11DeviceContext* ctx = impl_->context.Get();

    // synth3d (2D->3D): run the depth-prep / readback / warp passes BEFORE any display
    // state is set. process() binds its own render targets, viewport, shaders, sampler,
    // constant buffer and resources — every one of which the display path below re-sets
    // — so the only thing it must leave clean is the OM stage, which it unbinds itself.
    // A false return (no depth yet, a creation failure) leaves the untouched source
    // SRVs bound: the 2D picture keeps playing, it never waits for depth.
    bool synth_ok = false;
    if (has_frame_ && pipeline_ready_ && impl_->synth3d && impl_->synth3d_params.enabled) {
        ID3D11ShaderResourceView* src[3] = { impl_->srv[1].Get(), impl_->srv[2].Get(),
                                             impl_->srv[3].Get() };
        if (src[0] && src[1] && src[2]) {
            synth_ok = impl_->synth3d->process(
                ctx, src, impl_->tex_w[1], impl_->tex_h[1], impl_->tex_w[2], impl_->tex_h[2],
                impl_->tex_fmt[1] == TexFormat::R16 ? DXGI_FORMAT_R16_UNORM
                                                    : DXGI_FORMAT_R8_UNORM,
                impl_->cb.plane_scale, impl_->cb.yuv_matrix_sel,
                impl_->cb.transfer_sel, video_time_ms_);
        }
    }

    ID3D11RenderTargetView* rtv = impl_->rtv.Get();
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    const float black[4] = { 0.f, 0.f, 0.f, 1.f };
    ctx->ClearRenderTargetView(rtv, black);

    if (has_frame_ && pipeline_ready_) {
        // Aspect-preserving, PIXEL-SNAPPED viewport (pillarbox/letterbox). Integer
        // TopLeftX/Width/TopLeftY/Height are required for EXACT pixel coverage: the old
        // fractional viewport, combined with D3D11's top-left fill rule and the black
        // clear, left a 1px UNCOVERED column on the right (or row on the bottom) whenever
        // the aspect mismatch was a sub-pixel fraction — the reported edge band. Snapping
        // to the pixel grid removes the sliver; snapping a <=1px total-bar (sub-pixel
        // aspect mismatch, e.g. a 1921-wide window on 16:9 content) to FULL FILL removes
        // the band entirely; any genuine >=2px letterbox/pillarbox is kept and centered
        // symmetrically (an odd leftover puts the extra pixel on the right/bottom).
        // FramePack has a physical 1920x2205 transport canvas. Every main-window
        // layout (2D/SBS/TAB), however, inherits the SOURCE display aspect: a
        // synthesized 2.39:1 movie remains 2.39:1 instead of being forced into
        // 16:9 merely because its two eyes are arranged side-by-side or stacked.
        const float source_aspect =
            aspect_ > 0.0f ? aspect_
                           : (src_h_ ? float(src_w_) / float(src_h_)
                                     : 1920.0f / 1080.0f);
        const float target_aspect =
            stereo_mode_ == 1 ? 1920.0f / 2205.0f : source_aspect;

        D3D11_VIEWPORT vp = {};
        vp.MinDepth = 0.f; vp.MaxDepth = 1.f;
        const uint32_t ow = width_, oh = height_;
        uint32_t vw = ow, vh = oh, vx = 0, vy = 0;
        if (ow > 0 && oh > 0 && target_aspect > 0.0f) {
            const float out_aspect = float(ow) / float(oh);
            if (out_aspect > target_aspect) {            // wider -> pillarbox (bars left/right)
                vh = oh;
                long wi = std::lround(float(oh) * target_aspect);   // ideal content width
                if (wi < 1) wi = 1;
                vw = (wi >= long(ow) - 1) ? ow : static_cast<uint32_t>(wi); // sub-px -> fill
                if (vw > ow) vw = ow;
                vx = (ow - vw) / 2;                      // integer center (extra px -> right)
            } else {                                     // taller -> letterbox (bars top/bottom)
                vw = ow;
                long hi = std::lround(float(ow) / target_aspect);   // ideal content height
                if (hi < 1) hi = 1;
                vh = (hi >= long(oh) - 1) ? oh : static_cast<uint32_t>(hi); // sub-px -> fill
                if (vh > oh) vh = oh;
                vy = (oh - vh) / 2;                      // integer center (extra px -> bottom)
            }
        }
        vp.TopLeftX = float(vx); vp.Width  = float(vw);
        vp.TopLeftY = float(vy); vp.Height = float(vh);
        ctx->RSSetViewports(1, &vp);
        ctx->RSSetState(impl_->raster.Get());

        const UINT stride = 16, offset = 0;
        ID3D11Buffer* vb = impl_->vbuffer.Get();
        ctx->IASetInputLayout(impl_->input_layout.Get());
        ctx->IASetVertexBuffers(0, 1, &vb, &stride, &offset);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        ctx->VSSetShader(impl_->vs.Get(), nullptr, 0);
        ctx->PSSetShader(impl_->ps.Get(), nullptr, 0);

        ID3D11Buffer* cb = impl_->cbuffer.Get();
        ctx->PSSetConstantBuffers(0, 1, &cb);

        ID3D11ShaderResourceView* srvs[kNumTex];
        ID3D11SamplerState*       samps[kNumTex];
        for (int i = 0; i < kNumTex; ++i) {
            srvs[i]  = impl_->srv[i].Get();         // may be null for unused stereo slots (not sampled)
            samps[i] = impl_->sampler.Get();
        }
        // synth3d: swap the six plane slots (t1..t6) for the synthesized stereo pair.
        // t0 (subtitle) and t7 (HUD) are untouched — both overlays are
        // composited independently after the eye image is selected.
        if (synth_ok) {
            ID3D11ShaderResourceView* const* o = impl_->synth3d->output_srvs();
            for (int i = 0; i < 6; ++i) srvs[1 + i] = o[i];
            // Dual Projector uses two stereo_mode=0 windows.  That display
            // shader samples t1..t3 only, so the right window must explicitly
            // expose Synth3D's right trio there.  Source-plane routing cannot
            // solve this: the successful synth pass replaces all six SRVs.
            if (stereo_mode_ == 0 && synth3d_output_eye_ == 1) {
                srvs[1] = o[3];
                srvs[2] = o[4];
                srvs[3] = o[5];
            }
        } else if (impl_->synth3d && impl_->synth3d_params.enabled) {
            // Depth may be warming up, a newly-created presentation surface may
            // not have uploaded the shared snapshot yet, or a warp pass may fail.
            // The widget deliberately omits the duplicate source-right upload
            // while synthesis is enabled. Alias L into R for this transitional
            // frame so FramePack/SBS/TAB degrade to honest mono, never an
            // uninitialised green eye.
            srvs[4] = srvs[1];
            srvs[5] = srvs[2];
            srvs[6] = srvs[3];
        }
        ctx->PSSetShaderResources(0, kNumTex, srvs);
        ctx->PSSetSamplers(0, kNumTex, samps);

        ctx->Draw(4, 0);
    }

    // DXGI accepts 0..4. The player uses 1 for the framepack timing authority
    // and 0 for the simultaneous main-window preview.
    const UINT interval = static_cast<UINT>(sync_interval > 4 ? 4 : sync_interval);
    const HRESULT hr = impl_->swapchain->Present(interval, 0);
    if (FAILED(hr)) {
        last_error_ = d3d_error("Present", hr, impl_->device.Get());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// SyLC Cast (PC sender): offscreen NV12-SBS pack + NVENC HEVC encode.
// Consumes the SAME six YUV plane SRVs the render path uploads (srv[1..6]);
// never touches the swapchain / present(). See native_renderer.h.
// ---------------------------------------------------------------------------
bool NativeRenderer::cast_available() const {
    // Pure NVENC probe (LoadLibrary + CreateInstance). Safe with no NVIDIA GPU.
    return sylc::NvencEncoder::available();
}

bool NativeRenderer::cast_start(const std::string& mode, int fps, int64_t bitrate_bps,
                                bool main10) {
    last_error_.clear();
    if (!impl_ || !impl_->device) { last_error_ = "cast_start: renderer not initialized"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->cast && impl_->cast->active) { last_error_ = "cast_start: already active"; return false; }

    auto cp = std::make_unique<CastPipeline>();
    sylc::NvencConfig cfg;
    cfg.width  = 3840;
    cfg.height = 1080;
    cfg.fps    = (fps > 0) ? static_cast<uint32_t>(fps) : 24u;
    cfg.mode   = (mode == "lossless") ? sylc::CastMode::LosslessWired
                                      : sylc::CastMode::CbrLowLatency;
    if (bitrate_bps > 0) cfg.cbrBitrateBps = static_cast<uint32_t>(bitrate_bps);
    cfg.main10 = main10;
    cp->fps    = cfg.fps;   // remembered for cast_reconfigure (its signature carries no fps)
    cp->main10 = main10;

    std::string err;
    if (!cp->packer.init(impl_->device.Get(), err, main10)) {
        last_error_ = "cast_start: packer.init: " + err; return false;
    }
    if (!cp->enc.open(impl_->device.Get(), cfg, err)) {
        last_error_ = "cast_start: enc.open: " + err;
        cp->packer.shutdown();
        return false;
    }
    // Zero-copy: register the packer's NV12 output texture as the NVENC input surface.
    if (!cp->enc.registerInput(cp->packer.outputTexture(), &cp->regInput, err)) {
        last_error_ = "cast_start: registerInput: " + err;
        cp->enc.close();
        cp->packer.shutdown();
        return false;
    }
    cp->active = true;
    impl_->cast = std::move(cp);
    return true;
}

std::vector<std::vector<uint8_t>> NativeRenderer::cast_encode(int64_t pts_ms, bool force_idr) {
    std::vector<std::vector<uint8_t>> pkts;
    if (!impl_) { return pkts; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->cast || !impl_->cast->active) { last_error_ = "cast_encode: not started"; return pkts; }
    CastPipeline& c = *impl_->cast;

    // The six YUV plane SRVs uploaded by set_yuv_frame: srv[1..3]=L Y/U/V, srv[4..6]=R Y/U/V.
    ID3D11ShaderResourceView* srv6[6] = {
        impl_->srv[1].Get(), impl_->srv[2].Get(), impl_->srv[3].Get(),
        impl_->srv[4].Get(), impl_->srv[5].Get(), impl_->srv[6].Get(),
    };
    // synth3d: cast exactly what the PC shows — the synthesized pair, not the 2D
    // source. No process() here: this frame's present() already produced it (and
    // running the warp twice per frame would double the GPU cost for nothing).
    if (impl_->synth3d && impl_->synth3d_params.enabled && impl_->synth3d->outputs_valid()) {
        ID3D11ShaderResourceView* const* o = impl_->synth3d->output_srvs();
        for (int i = 0; i < 6; ++i) srv6[i] = o[i];
    } else if (impl_->synth3d && impl_->synth3d_params.enabled) {
        // Same warm-up/failure contract as present(): send mono duplicated to
        // the headset rather than failing on the intentionally absent R upload.
        srv6[3] = srv6[0];
        srv6[4] = srv6[1];
        srv6[5] = srv6[2];
    }
    for (int i = 0; i < 6; ++i) {
        if (!srv6[i]) { last_error_ = "cast_encode: a YUV plane SRV is null (call set_yuv_frame first)"; return pkts; }
    }

    // Subtitle burn-in: mirror the DISPLAY's current subtitle state into the
    // pack, so the cast shows exactly what the PC shows -- same RGBA overlay
    // (srv[0], always valid: 1x1 transparent when none), same enable flag,
    // same rect, same stereoscopic half-per-eye disparity. Toggling subtitles
    // on the PC therefore toggles them on the headset, live.
    sylc::CastSubtitle sub;
    sub.srv       = impl_->srv[0].Get();
    sub.enabled   = impl_->cb.subtitle_enabled;
    sub.disparity = impl_->cb.subtitle_disparity;
    sub.rect[0] = impl_->cb.subtitle_rect[0]; sub.rect[1] = impl_->cb.subtitle_rect[1];
    sub.rect[2] = impl_->cb.subtitle_rect[2]; sub.rect[3] = impl_->cb.subtitle_rect[3];

    // plane_scale mirrors the DISPLAY's per-source value (1.0 for 8-bit and HW
    // P010 planes, 65535/1023 for SW-decoded 10-bit) so the pack writes
    // correctly-aligned NV12/P010 regardless of how the planes were uploaded.
    c.packer.pack(impl_->context.Get(), srv6, sub, impl_->cb.plane_scale);
    std::string err;
    if (!c.enc.encode(c.regInput, pts_ms, force_idr, pkts, err)) {
        last_error_ = "cast_encode: " + err;   // pkts left as whatever encode appended (empty on error)
    }
    return pkts;
}

bool NativeRenderer::cast_reconfigure(const std::string& mode, int64_t bitrate_bps) {
    last_error_.clear();
    if (!impl_ || !impl_->device) { last_error_ = "cast_reconfigure: renderer not initialized"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);   // serialized with cast_encode/cast_stop
    if (!impl_->cast || !impl_->cast->active) { last_error_ = "cast_reconfigure: not started"; return false; }
    CastPipeline& c = *impl_->cast;

    // Cast is always the fixed 3840x1080 SBS surface; only bitrate/mode change here. Keep
    // the fps cast_start ran with (the signature carries none) so the CBR VBV stays sized.
    sylc::NvencConfig cfg;
    cfg.width  = 3840;
    cfg.height = 1080;
    cfg.fps    = c.fps;
    cfg.main10 = c.main10;   // the packer texture is fixed; bit depth must not drift
    cfg.mode   = (mode == "lossless") ? sylc::CastMode::LosslessWired
                                      : sylc::CastMode::CbrLowLatency;
    if (bitrate_bps > 0) cfg.cbrBitrateBps = static_cast<uint32_t>(bitrate_bps);

    std::string err;
    // 1) Seamless path: NvEncReconfigureEncoder (no teardown; next frame forced to IDR).
    //    This is the common Wi-Fi fallback — a same-mode CBR bitrate step.
    if (c.enc.reconfigure(cfg, err)) return true;

    // 2) The driver rejected the change (e.g. a lossless<->cbr mode switch alters the GOP
    //    structure / tuning, which NvEncReconfigureEncoder does not support). Fall back to
    //    an ENCODER-ONLY reopen: close the NVENC session, reopen it with the new config,
    //    and re-register the SAME packer output texture. The packer and its NV12 texture
    //    are untouched -> a brief hiccup + a fresh IDR, acceptable for a rare mode switch.
    // This is a notable, rare event (a seamless hot-reconfigure would have been silent), so
    // log it: absence of this line for a transition means it took the seamless path.
    std::fprintf(stderr, "[cast] reconfigure -> '%s': seamless NvEncReconfigureEncoder "
                         "rejected (%s); cycling the NVENC session\n",
                 mode.c_str(), err.c_str());
    c.enc.close();
    c.regInput = nullptr;
    if (!c.enc.open(impl_->device.Get(), cfg, err)) {
        last_error_ = "cast_reconfigure: reopen enc.open: " + err;
        c.active = false;   // pipeline unusable until a clean cast_stop() tears it down
        return false;
    }
    if (!c.enc.registerInput(c.packer.outputTexture(), &c.regInput, err)) {
        last_error_ = "cast_reconfigure: reopen registerInput: " + err;
        c.active = false;
        return false;
    }
    return true;
}

void NativeRenderer::cast_stop() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->cast) return;   // idempotent
    CastPipeline& c = *impl_->cast;
    // No enc.flush(): Task-1 forces frameIntervalP=1 (synchronous 1:1), so every frame's
    // bitstream is emitted DURING its cast_encode() call and nothing is ever buffered.
    // Task-1's flush() sends an EOS picture then nvEncLockBitstream on the single output
    // buffer, which BUSY-WAITS forever when no frame is pending (proven hanging on the
    // RTX 4090) -> calling it here would spin cast_stop indefinitely. Draining is
    // unnecessary for a 1:1 encoder, so we close the session directly.
    c.enc.close();
    c.packer.shutdown();
    impl_->cast.reset();
}

// ---------------------------------------------------------------------------
// synth3d (2D->3D): AI depth + DIBR warp. See native_renderer.h.
// Every method below takes impl_->mtx, so the GPU passes in present() and the
// enable/parameter/debug calls from the GUI thread are serialized. Inference lives
// in a process-cached service and never touches this renderer's D3D state.
// ---------------------------------------------------------------------------
bool NativeRenderer::set_synth3d(bool enabled, float strength_pct, float convergence,
                                  bool depth_view, const std::wstring& model_path,
                                  const std::wstring& ort_dir, bool diagnostics,
                                  int side, int grid_width, int grid_height,
                                  float crop_top, float crop_bottom,
                                  bool auto_convergence, bool temporal_fill,
                                  bool stereo_lab, bool comfort_enabled,
                                  float comfort_soft_pct,
                                  float comfort_hard_pct) {
    last_error_.clear();
    if (!impl_) { last_error_ = "set_synth3d: no impl"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->device) { last_error_ = "set_synth3d before initialize"; return false; }

    if (!enabled) {
        impl_->synth3d_params.enabled = false;
        // Detach only. The bounded registry either keeps this one warm or hands
        // its final reference to the reaper, so disabling never waits for ORT
        // under this renderer mutex.
        if (impl_->synth3d) impl_->synth3d->request_stop();
        return true;
    }

    Synth3DParams p;
    p.enabled      = true;
    p.strength_pct = strength_pct;
    p.convergence  = convergence;
    p.auto_convergence = auto_convergence;
    p.temporal_fill = temporal_fill;
    // Stereo Lab owns only separate final textures. Its explicit false value
    // (or SYLC_STEREO_LAB=0) remains the exact raw v5.2.1c rollback.
    p.stereo_lab = stereo_lab;
    if (comfort_enabled &&
        (!std::isfinite(comfort_soft_pct) ||
         !std::isfinite(comfort_hard_pct) ||
         comfort_soft_pct < 0.0f ||
         comfort_hard_pct <= comfort_soft_pct)) {
        last_error_ = "set_synth3d: invalid calibrated comfort envelope";
        return false;
    }
    p.comfort_enabled = false;
    p.comfort_soft_pct = 0.0f;
    p.comfort_hard_pct = 0.0f;
    p.depth_view   = depth_view;
    p.diagnostics  = diagnostics;
    p.model_path   = model_path;
    p.ort_dir      = ort_dir;
    p.side         = side > 0 ? side : kDefaultDepthSide;
    if (grid_width > 0 && grid_height > 0) {
        p.grid_width = grid_width;
        p.grid_height = grid_height;
    }
    p.crop_top = std::clamp(crop_top, 0.0f, 1.0f);
    p.crop_bottom = std::clamp(crop_bottom, 0.0f, 1.0f);
    if (p.crop_top + p.crop_bottom >= 0.95f) {
        p.crop_top = 0.0f;
        p.crop_bottom = 0.0f;
    }

    if (!impl_->synth3d) impl_->synth3d = std::make_unique<Synth3D>();
    std::string err;
    if (!impl_->synth3d->start(impl_->device.Get(), p, err)) {
        last_error_ = "set_synth3d: " + err;
        impl_->synth3d_params.enabled = false;
        return false;
    }
    impl_->synth3d_params = p;
    return true;
}

std::string NativeRenderer::synth3d_status() const {
    // BYTE-IDENTICAL to the off line Synth3D::status() returns (synth3d.cpp):
    // nothing is attached, so there is no inference grid and side= reads 0.
    // Any field added to one of these two lines MUST be added to the other.
    static const char* kOff =
        "state=off provider=none side=0 fps=0.0 flow_ms=0.0 infer_ms=0.0 "
        "stab_ms=0.0 "
        "source_ms=120.0 update_ms=120.0 age_ms=-1 clients=0 cuts=0 "
        "motion=0.000 alpha=0.000 stable=0.000 history=1.00 scene=0.000 "
        "crop=0:0:0:0 crop_conf=0.00 crop_ready=0 grid=0x0 instance=0 err=- "
        "lab=off lab_mean=0.000 lab_p95=0.000 lab_asym=0.000 lab_px=0.0 "
        "pair=off pair_grid=0x0";
    if (!impl_) return kOff;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d || !impl_->synth3d_params.enabled) return kOff;
    return impl_->synth3d->status();
}

void NativeRenderer::synth3d_notify_seek() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return;   // no pipeline yet: nothing to re-prime
    impl_->synth3d->notify_seek();
}

void NativeRenderer::synth3d_set_ramp_ms(float ramp_ms) {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return;   // no pipeline yet: nothing to configure
    impl_->synth3d->set_ramp_ms(ramp_ms);
}

bool NativeRenderer::synth3d_set_test_depth(const uint16_t* q16_or_null,
                                            size_t count) {
    if (!impl_) return true;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    // No pipeline: nothing to arm and nothing read from q16 -- the historic
    // no-op, reported as success. The size check below happens under THIS lock,
    // together with the copy, so the grid cannot change between them.
    if (!impl_->synth3d) return true;
    return impl_->synth3d->set_test_depth(q16_or_null, count);
}

bool NativeRenderer::synth3d_set_test_geometry(const uint16_t* depth,
                                               const uint16_t* owned,
                                               const uint16_t* safety,
                                               const uint16_t* ownership,
                                               size_t count) {
    if (!impl_) return true;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return true;
    return impl_->synth3d->set_test_geometry(
        depth, owned, safety, ownership, count);
}

bool NativeRenderer::synth3d_set_test_matte(const uint8_t* alpha_or_null,
                                            uint32_t width, uint32_t height,
                                            size_t count, int mode,
                                            const uint8_t* reliability_or_null,
                                            size_t reliability_count) {
    if (!impl_) return true;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return true;
    return impl_->synth3d->set_test_matte(
        alpha_or_null, width, height, count, mode,
        reliability_or_null, reliability_count);
}

int NativeRenderer::synth3d_side() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->synth3d ? impl_->synth3d->side() : 0;
}

int NativeRenderer::synth3d_grid_width() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->synth3d ? impl_->synth3d->grid_width() : 0;
}

int NativeRenderer::synth3d_grid_height() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return impl_->synth3d ? impl_->synth3d->grid_height() : 0;
}

bool NativeRenderer::synth3d_read_plane(int slot, std::vector<uint8_t>& out, uint32_t& w,
                                        uint32_t& h, uint32_t& bpp, std::string& err) {
    if (!impl_ || !impl_->context) { err = "synth3d_read_plane: not initialized"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) { err = "synth3d_read_plane: synth3d not enabled"; return false; }
    return impl_->synth3d->read_plane(impl_->context.Get(), slot, out, w, h, bpp, err);
}

bool NativeRenderer::synth3d_set_lookahead(double cut_in_ms, double storm_in_ms,
                                           double cut_pts_ms) {
    if (!impl_) return false;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return false;
    impl_->synth3d->set_lookahead_advisory(cut_in_ms, storm_in_ms, cut_pts_ms);
    return true;
}

bool NativeRenderer::synth3d_set_motion_hints(
        double pts_ms, double frame_ms, int blocks_w, int blocks_h,
        int source_width, int source_height,
        std::vector<int16_t>&& mv_xy, std::vector<uint8_t>&& valid) {
    if (!impl_) return false;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) return false;
    impl_->synth3d->set_motion_hints(
        pts_ms, frame_ms, blocks_w, blocks_h, source_width, source_height,
        std::move(mv_xy), std::move(valid));
    return true;
}

bool NativeRenderer::synth3d_read_plate(std::vector<uint8_t>& out, uint32_t& w,
                                        uint32_t& h, std::string& err) {
    if (!impl_ || !impl_->context) { err = "synth3d_read_plate: not initialized"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (!impl_->synth3d) { err = "synth3d_read_plate: synth3d not enabled"; return false; }
    return impl_->synth3d->read_plate(impl_->context.Get(), out, w, h, err);
}

void NativeRenderer::shutdown() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    // Detach from the process-wide service before releasing device-local resources.
    // stop() is non-blocking: the cached service owns the ORT worker.
    if (impl_->synth3d) impl_->synth3d->stop();
    if (impl_->context) impl_->context->ClearState();
    // SyLC Cast: tear down the NVENC session + packer BEFORE releasing the device they
    // were created on. Inlined (not cast_stop(), which re-locks the non-recursive mutex).
    // No enc.flush() — see cast_stop(): frameIntervalP=1 buffers nothing, and Task-1's
    // flush() busy-waits on the EOS bitstream lock when nothing is pending. Close directly.
    if (impl_->cast) {
        impl_->cast->enc.close();
        impl_->cast->packer.shutdown();
        impl_->cast.reset();
    }
    // synth3d: the worker is already joined (above); destroy the object so its D3D
    // resources are released BEFORE the device they were created on.
    impl_->synth3d.reset();
    impl_->synth3d_params = Synth3DParams{};
    for (int i = 0; i < kNumTex; ++i) { impl_->srv[i].Reset(); impl_->tex[i].Reset(); }
    impl_->raster.Reset(); impl_->sampler.Reset(); impl_->cbuffer.Reset();
    impl_->vbuffer.Reset(); impl_->input_layout.Reset();
    impl_->ps.Reset(); impl_->vs.Reset();
    impl_->rtv.Reset(); impl_->swapchain.Reset();
    impl_->context.Reset(); impl_->device.Reset();
    hdr_enabled_ = false; pipeline_ready_ = false; has_frame_ = false;
    video_time_ms_ = -1.0;
    hwnd_ = 0;   // forget the window; a later initialize() rebinds to the current HWND
}

} // namespace sylc

#endif // SYLC_NATIVE_RENDERER
