// Native D3D11 renderer — STAGES S1 (swapchain/present) + S2 (shaded YUV draw).
// See native_renderer.h and native_renderer/NATIVE_RENDERER_DESIGN.md.
#ifdef SYLC_NATIVE_RENDERER

#include "native_renderer.h"
#include "native_renderer_shaders.h"   // kVertexHLSL / kFragmentHLSL (exact, embedded)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <mutex>

using Microsoft::WRL::ComPtr;

namespace sylc {

// Texture slots match the HLSL register bindings:
//   t0 = subtitle (RGBA8), t1..t3 = Y/U/V left, t4..t6 = Y/U/V right.
static constexpr int kNumTex = 7;

// cbuffer 'buf' (register b0) — layout must match the HLSL packoffsets:
//   c0.x int stereo_mode, c0.y int subtitle_enabled, c0.z float subtitle_disparity,
//   c1 float4 subtitle_rect, c2.x float sdr_white_level, c3.x float plane_scale.
//   64 bytes total (4 x 16).
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
};
// The two HDR selectors reuse the former c3.yz padding, so the cbuffer stays 64 bytes and
// the ABI is unchanged. Both 0 (DEFAULT) is byte-identical to the pre-HDR shader.
static_assert(sizeof(FrameCB) == 64, "cbuffer must be 64 bytes");

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
                                    TexFormat::R8 };

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
        if (FAILED(hr)) { last_error_ = "D3D11CreateDevice failed"; return false; }
    }

    ComPtr<IDXGIDevice> dxgiDevice;
    if (FAILED(impl_->device.As(&dxgiDevice))) { last_error_ = "QI IDXGIDevice failed"; return false; }
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
    sd.BufferCount        = 2;
    sd.Scaling            = DXGI_SCALING_STRETCH;
    sd.SwapEffect         = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode          = DXGI_ALPHA_MODE_IGNORE;

    const HWND win = reinterpret_cast<HWND>(static_cast<uintptr_t>(hwnd));
    hr = factory->CreateSwapChainForHwnd(impl_->device.Get(), win, &sd, nullptr, nullptr, &impl_->swapchain);
    if (FAILED(hr)) { last_error_ = "CreateSwapChainForHwnd failed"; return false; }
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
        "D3D11 flip-model | FL=0x%04x | %ux%u | %s | pipeline=%s",
        static_cast<unsigned>(got), width_, height_,
        cs_name, pipeline_ready_ ? "ready" : "FAILED");
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
    if (FAILED(impl_->swapchain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0))) {
        last_error_ = "ResizeBuffers failed"; return false;
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

    auto compile = [&](const char* src, const char* target, ComPtr<ID3DBlob>& out) -> bool {
        ComPtr<ID3DBlob> err;
        const UINT cflags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS;
        HRESULT hr = D3DCompile(src, std::strlen(src), "yuv_framepack", nullptr, nullptr,
                                "main", target, cflags, 0, &out, &err);
        if (FAILED(hr)) {
            last_error_ = std::string("D3DCompile(") + target + ") failed";
            if (err) { last_error_ += ": "; last_error_ += static_cast<const char*>(err->GetBufferPointer()); }
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> vsBlob, psBlob;
    if (!compile(kVertexHLSL,   "vs_5_0", vsBlob)) return false;
    if (!compile(kFragmentHLSL, "ps_5_0", psBlob)) return false;

    if (FAILED(impl_->device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                                 nullptr, &impl_->vs))) {
        last_error_ = "CreateVertexShader failed"; return false;
    }
    if (FAILED(impl_->device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(),
                                                nullptr, &impl_->ps))) {
        last_error_ = "CreatePixelShader failed"; return false;
    }

    // Input layout matches the VS SPIRV_Cross_Input: position@TEXCOORD0, texCoord@TEXCOORD1.
    const D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32_FLOAT, 0, 8,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    if (FAILED(impl_->device->CreateInputLayout(layout, 2, vsBlob->GetBufferPointer(),
                                                vsBlob->GetBufferSize(), &impl_->input_layout))) {
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

    // Slot t0 (subtitle) always valid: a 1x1 transparent texture so the SRV is
    // never null even when subtitles are disabled.
    static const uint8_t kTransparent[4] = { 0, 0, 0, 0 };
    if (!upload_subtitle(kTransparent, 1, 1, 4)) return false;

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
static bool upload_to_tex(ID3D11DeviceContext* ctx, ID3D11Texture2D* tex,
                          const uint8_t* data, uint32_t w, uint32_t h,
                          uint32_t srcStride, uint32_t bytesPerPixel) {
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
    auto* dst = static_cast<uint8_t*>(m.pData);
    const uint32_t rowBytes = w * bytesPerPixel;
    for (uint32_t r = 0; r < h; ++r)
        std::memcpy(dst + r * m.RowPitch, data + r * srcStride, rowBytes);
    ctx->Unmap(tex, 0);
    return true;
}

bool NativeRenderer::upload_plane(int plane_index, const uint8_t* data,
                                  uint32_t width, uint32_t height, uint32_t src_stride) {
    if (!impl_ || !impl_->context) { last_error_ = "upload_plane before initialize"; return false; }
    if (plane_index < 0 || plane_index > 5) { last_error_ = "upload_plane: bad index"; return false; }
    if (!data) { last_error_ = "upload_plane: null data"; return false; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    const int slot = plane_index + 1; // 0->t1 .. 5->t6
    if (!ensure_texture(slot, width, height, TexFormat::R8)) return false;
    if (!upload_to_tex(impl_->context.Get(), impl_->tex[slot].Get(), data, width, height, src_stride, 1)) {
        last_error_ = "Map(plane) failed"; return false;
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
    if (!upload_to_tex(impl_->context.Get(), impl_->tex[slot].Get(),
                       reinterpret_cast<const uint8_t*>(data), width, height, src_stride, 2)) {
        last_error_ = "Map(plane16) failed"; return false;
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
    if (!upload_to_tex(impl_->context.Get(), impl_->tex[0].Get(), data, width, height, src_stride, 4)) {
        last_error_ = "Map(subtitle) failed"; return false;
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
// set_uniforms (framepack per-eye fit) and present (2D target aspect), not the cbuffer,
// so there is no GPU write here — just a mutex-guarded store, mirroring how set_plane_scale
// serializes against the geometry readers.
void NativeRenderer::set_source_aspect(float aspect) {
    if (!impl_) { aspect_ = aspect; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    aspect_ = aspect;
}

void NativeRenderer::set_uniforms(int stereo_mode, int subtitle_enabled,
                                  float rx, float ry, float rw, float rh,
                                  float sdr_white, float output_gamma,
                                  float subtitle_disparity) {
    if (!impl_ || !impl_->cbuffer || !impl_->context) { stereo_mode_ = stereo_mode; return; }
    std::lock_guard<std::mutex> lk(impl_->mtx);
    stereo_mode_ = stereo_mode;
    FrameCB cb = {};
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

void NativeRenderer::clear_frame() {
    // C2: reset the display-aspect override so a subsequent full/MVC/2D source derives its
    // aspect from planes again (the widget/player also re-sets it per frame).
    if (impl_) { std::lock_guard<std::mutex> lk(impl_->mtx); has_frame_ = false; aspect_ = 0.0f; }
    else { has_frame_ = false; aspect_ = 0.0f; }
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

bool NativeRenderer::present() {
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
            if (cw > 0 && ch > 0 && (cw != width_ || ch != height_))
                resize_backbuffer_locked(cw, ch);
        }
    }

    if (!impl_->rtv && !create_rtv_for_backbuffer()) return false;

    ID3D11DeviceContext* ctx = impl_->context.Get();
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
        float target_aspect;
        if (stereo_mode_ == 1)       target_aspect = 1920.0f / 2205.0f;
        // C2: 2D uses the display-aspect override when set (half-SBS/half-TAB base eye is
        // squeezed), else derives it from the uploaded plane dimensions.
        else if (stereo_mode_ == 0)  target_aspect = (aspect_ > 0.0f ? aspect_
                                                       : (src_h_ ? float(src_w_) / float(src_h_) : 1920.0f / 1080.0f));
        else                         target_aspect = 1920.0f / 1080.0f;

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
        ctx->PSSetShaderResources(0, kNumTex, srvs);
        ctx->PSSetSamplers(0, kNumTex, samps);

        ctx->Draw(4, 0);
    }

    if (FAILED(impl_->swapchain->Present(1, 0))) { last_error_ = "Present failed"; return false; }
    return true;
}

void NativeRenderer::shutdown() {
    if (!impl_) return;
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->context) impl_->context->ClearState();
    for (int i = 0; i < kNumTex; ++i) { impl_->srv[i].Reset(); impl_->tex[i].Reset(); }
    impl_->raster.Reset(); impl_->sampler.Reset(); impl_->cbuffer.Reset();
    impl_->vbuffer.Reset(); impl_->input_layout.Reset();
    impl_->ps.Reset(); impl_->vs.Reset();
    impl_->rtv.Reset(); impl_->swapchain.Reset();
    impl_->context.Reset(); impl_->device.Reset();
    hdr_enabled_ = false; pipeline_ready_ = false; has_frame_ = false;
    hwnd_ = 0;   // forget the window; a later initialize() rebinds to the current HWND
}

} // namespace sylc

#endif // SYLC_NATIVE_RENDERER
