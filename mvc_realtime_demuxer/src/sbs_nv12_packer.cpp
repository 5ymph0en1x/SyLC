// YUV -> NV12 side-by-side packer -- implementation. See sbs_nv12_packer.h.
//
// Renders the renderer's six already-uploaded R8_UNORM YUV plane SRVs (L: Y/U/V,
// R: Y/U/V) into one DXGI_FORMAT_NV12 3840x1080 texture: left eye in the left 1920
// columns, right eye in the right 1920. Two fullscreen-triangle passes with POINT
// sampling (no interpolation -> lossless):
//   * luma   pass -> an R8_UNORM   RTV of the NV12 luma   plane (3840x1080)
//   * chroma pass -> an R8G8_UNORM RTV of the NV12 chroma plane (1920x540, U,V)
// The NV12 output is what Task 1's NvencEncoder registers + encodes zero-copy.
//
// Shader-embed idiom mirrors native_renderer.cpp exactly: the HLSL is hand-authored in
// shaders/yuv_to_nv12_sbs.hlsl, embedded into the committed sbs_nv12_packer_shaders.h by
// native_renderer/gen_shader_header.py (SINGLE source of truth -- re-run the generator if
// the .hlsl changes; do NOT hand-edit the generated header), and compiled at RUNTIME by
// d3dcompiler with the same flags. No RGB anywhere in this path.
//
// Gated on SYLC_NATIVE_RENDERER (like native_renderer.cpp): it is only added to the
// build inside the Windows-only BUILD_NATIVE_RENDERER block, which links d3d11 +
// d3dcompiler and defines this macro.
#ifdef SYLC_NATIVE_RENDERER

#include "sbs_nv12_packer.h"
#include "sbs_nv12_packer_shaders.h"   // kSbsPackHLSL, generated from shaders/yuv_to_nv12_sbs.hlsl

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace sylc {

// SBS output geometry. NV12 requires even dimensions; both are even. NV12's chroma
// plane is half-res in both axes, so its R8G8 view is 1920x540.
static constexpr uint32_t kSbsWidth  = 3840;
static constexpr uint32_t kSbsHeight = 1080;
static constexpr uint32_t kChromaW   = kSbsWidth  / 2;   // 1920
static constexpr uint32_t kChromaH   = kSbsHeight / 2;   // 540
static constexpr int      kNumSrv    = 6;

// kSbsPackHLSL (the VS + PS_Luma + PS_Chroma source) is #included above from the
// generated sbs_nv12_packer_shaders.h and compiled once per entry point in init().

// cbuffer b0 (SubCB in the HLSL): subtitle burn-in parameters. 32 bytes (2 x 16).
struct SubCB {
    int   enabled;      // c0.x
    float disparity;    // c0.y  normalized eye width; > 0 = pop-out
    float planeScale;   // c0.z  video-sample scale (1.0 = 8-bit/HW-P010 passthrough)
    float _pad0;        // c0.w
    float rect[4];      // c1    normalized (x, y, w, h) in per-eye space
};
static_assert(sizeof(SubCB) == 32, "SubCB must be 32 bytes");

// Opaque state (kept out of the public header, exactly like NativeRenderer).
struct SbsNv12Packer::Impl {
    ComPtr<ID3D11Device>           device;
    ComPtr<ID3D11Texture2D>        nv12;        // the NV12 3840x1080 SBS output
    ComPtr<ID3D11RenderTargetView> lumaRtv;     // R8   view -> Y  plane (3840x1080)
    ComPtr<ID3D11RenderTargetView> chromaRtv;   // R8G8 view -> UV plane (1920x540)
    ComPtr<ID3D11VertexShader>     vs;
    ComPtr<ID3D11PixelShader>      psLuma;
    ComPtr<ID3D11PixelShader>      psChroma;
    ComPtr<ID3D11SamplerState>     sampler;     // POINT / CLAMP (lossless video copy)
    ComPtr<ID3D11SamplerState>     subSampler;  // LINEAR / CLAMP (subtitle overlay scaling)
    ComPtr<ID3D11Buffer>           subCb;       // b0: SubCB (subtitle burn-in params)
    ComPtr<ID3D11RasterizerState>  raster;      // CULL_NONE (fullscreen triangle)
    bool tenBit = false;                        // P010 output (Main10 HDR cast)
};

SbsNv12Packer::SbsNv12Packer() : impl_(new Impl()) {}

SbsNv12Packer::~SbsNv12Packer() {
    shutdown();
    delete impl_;
    impl_ = nullptr;
}

bool SbsNv12Packer::init(ID3D11Device* dev, std::string& err, bool tenBit) {
    err.clear();
    if (!impl_)      { err = "init: no impl";              return false; }
    if (!dev)        { err = "init: null device";          return false; }
    if (impl_->nv12) { err = "init: already initialized";  return false; }
    impl_->device = dev;
    impl_->tenBit = tenBit;

    // --- Compile VS + both PS from the embedded HLSL (runtime d3dcompiler) --------
    // Same source string compiled once per entry point; unused functions/textures for
    // a given entry point are simply not referenced (standard D3DCompile behavior).
    auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& out) -> bool {
        ComPtr<ID3DBlob> errBlob;
        const UINT cflags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS;
        HRESULT hr = D3DCompile(kSbsPackHLSL, std::strlen(kSbsPackHLSL), "yuv_to_nv12_sbs",
                                nullptr, nullptr, entry, target, cflags, 0, &out, &errBlob);
        if (FAILED(hr)) {
            err = std::string("D3DCompile(") + entry + ") failed";
            if (errBlob) { err += ": "; err += static_cast<const char*>(errBlob->GetBufferPointer()); }
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> vsBlob, psLumaBlob, psChromaBlob;
    if (!compile("VS",        "vs_5_0", vsBlob))       return false;
    if (!compile("PS_Luma",   "ps_5_0", psLumaBlob))   return false;
    if (!compile("PS_Chroma", "ps_5_0", psChromaBlob)) return false;

    if (FAILED(impl_->device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                                 nullptr, &impl_->vs))) {
        err = "CreateVertexShader failed"; return false;
    }
    if (FAILED(impl_->device->CreatePixelShader(psLumaBlob->GetBufferPointer(), psLumaBlob->GetBufferSize(),
                                                nullptr, &impl_->psLuma))) {
        err = "CreatePixelShader(luma) failed"; return false;
    }
    if (FAILED(impl_->device->CreatePixelShader(psChromaBlob->GetBufferPointer(), psChromaBlob->GetBufferSize(),
                                                nullptr, &impl_->psChroma))) {
        err = "CreatePixelShader(chroma) failed"; return false;
    }

    // --- NV12 output texture: bindable as RTV (we render into it) and SRV ---------
    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = kSbsWidth;
    td.Height           = kSbsHeight;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = tenBit ? DXGI_FORMAT_P010 : DXGI_FORMAT_NV12;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DEFAULT;
    td.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(impl_->device->CreateTexture2D(&td, nullptr, &impl_->nv12))) {
        err = "CreateTexture2D(NV12) failed"; return false;
    }

    // Luma plane RTV: an R8_UNORM view addresses NV12's Y plane at full 3840x1080.
    D3D11_RENDER_TARGET_VIEW_DESC lr = {};
    lr.Format             = tenBit ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
    lr.ViewDimension      = D3D11_RTV_DIMENSION_TEXTURE2D;
    lr.Texture2D.MipSlice = 0;
    if (FAILED(impl_->device->CreateRenderTargetView(impl_->nv12.Get(), &lr, &impl_->lumaRtv))) {
        err = "CreateRenderTargetView(luma R8) failed"; return false;
    }
    // Chroma plane RTV: an R8G8_UNORM view addresses NV12's interleaved UV plane,
    // which is half-res in both axes -> 1920x540.
    D3D11_RENDER_TARGET_VIEW_DESC cr = {};
    cr.Format             = tenBit ? DXGI_FORMAT_R16G16_UNORM : DXGI_FORMAT_R8G8_UNORM;
    cr.ViewDimension      = D3D11_RTV_DIMENSION_TEXTURE2D;
    cr.Texture2D.MipSlice = 0;
    if (FAILED(impl_->device->CreateRenderTargetView(impl_->nv12.Get(), &cr, &impl_->chromaRtv))) {
        err = "CreateRenderTargetView(chroma R8G8) failed"; return false;
    }

    // POINT + CLAMP sampler: no interpolation keeps the copy lossless; the shader uses
    // SampleLevel(...,0) so only the point filter matters.
    D3D11_SAMPLER_DESC samp = {};
    samp.Filter   = D3D11_FILTER_MIN_MAG_MIP_POINT;
    samp.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samp.MinLOD   = 0.f;
    samp.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(impl_->device->CreateSamplerState(&samp, &impl_->sampler))) {
        err = "CreateSamplerState failed"; return false;
    }

    // LINEAR + CLAMP sampler for the subtitle overlay (s1): the overlay texture
    // maps onto the eye at an arbitrary scale, so bilinear keeps its edges clean.
    samp.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    if (FAILED(impl_->device->CreateSamplerState(&samp, &impl_->subSampler))) {
        err = "CreateSamplerState(subtitle) failed"; return false;
    }

    // b0: the subtitle burn-in cbuffer, refreshed per pack() via UpdateSubresource.
    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(SubCB);
    cbd.Usage     = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(impl_->device->CreateBuffer(&cbd, nullptr, &impl_->subCb))) {
        err = "CreateBuffer(SubCB) failed"; return false;
    }

    // CULL_NONE so the fullscreen triangle is never back-face culled regardless of winding.
    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode        = D3D11_FILL_SOLID;
    rs.CullMode        = D3D11_CULL_NONE;
    rs.DepthClipEnable = TRUE;
    if (FAILED(impl_->device->CreateRasterizerState(&rs, &impl_->raster))) {
        err = "CreateRasterizerState failed"; return false;
    }
    return true;
}

ID3D11Texture2D* SbsNv12Packer::outputTexture() const {
    return impl_ ? impl_->nv12.Get() : nullptr;
}

void SbsNv12Packer::pack(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* const yuvSrv[6]) {
    pack(ctx, yuvSrv, CastSubtitle{}, 1.0f);   // no subtitle: identical to the old API
}

void SbsNv12Packer::pack(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* const yuvSrv[6],
                         const CastSubtitle& sub, float planeScale) {
    if (!impl_ || !ctx) return;
    // Call only after init() == true. Guard EVERY resource pack() binds so a partially
    // failed init() (whose false a caller ignored) can't bind null state + emit garbage.
    if (!impl_->nv12 || !impl_->vs || !impl_->psLuma || !impl_->psChroma ||
        !impl_->lumaRtv || !impl_->chromaRtv || !impl_->sampler ||
        !impl_->subSampler || !impl_->subCb || !impl_->raster) return;

    // Subtitle burn-in parameters for both passes. An unset SRV forces
    // enabled=0 (sampling an unbound SRV would read zeros, but there is no
    // reason to even evaluate the branch then).
    SubCB cb = {};
    cb.enabled   = (sub.enabled && sub.srv) ? 1 : 0;
    cb.disparity = sub.disparity;
    cb.planeScale = (planeScale > 0.f) ? planeScale : 1.0f;
    cb.rect[0] = sub.rect[0]; cb.rect[1] = sub.rect[1];
    cb.rect[2] = sub.rect[2]; cb.rect[3] = sub.rect[3];
    ctx->UpdateSubresource(impl_->subCb.Get(), 0, nullptr, &cb, 0, 0);

    // Shared state for both passes: fullscreen triangle from SV_VertexID (no vertex
    // buffer / input layout), the six YUV SRVs on t0..t5 + the subtitle on t6,
    // the POINT sampler on s0 + the LINEAR subtitle sampler on s1, SubCB on b0.
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(impl_->vs.Get(), nullptr, 0);
    ctx->RSSetState(impl_->raster.Get());
    ID3D11SamplerState* samps[2] = { impl_->sampler.Get(), impl_->subSampler.Get() };
    ctx->PSSetSamplers(0, 2, samps);
    ID3D11Buffer* cbuf = impl_->subCb.Get();
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, kNumSrv, yuvSrv);  // yuvSrv already ID3D11SRV* const*
    ID3D11ShaderResourceView* subSrv = cb.enabled ? sub.srv : nullptr;
    ctx->PSSetShaderResources(kNumSrv, 1, &subSrv);  // t6

    // Pass 1 -- luma: R8 view of the Y plane, full 3840x1080.
    {
        ID3D11RenderTargetView* rtv = impl_->lumaRtv.Get();
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        D3D11_VIEWPORT vp = {};
        vp.Width = float(kSbsWidth); vp.Height = float(kSbsHeight);
        vp.MinDepth = 0.f; vp.MaxDepth = 1.f;
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetShader(impl_->psLuma.Get(), nullptr, 0);
        ctx->Draw(3, 0);   // the draw overwrites the whole plane -> no clear needed
    }

    // Pass 2 -- chroma: R8G8 view of the interleaved UV plane, 1920x540.
    {
        ID3D11RenderTargetView* rtv = impl_->chromaRtv.Get();
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        D3D11_VIEWPORT vp = {};
        vp.Width = float(kChromaW); vp.Height = float(kChromaH);
        vp.MinDepth = 0.f; vp.MaxDepth = 1.f;
        ctx->RSSetViewports(1, &vp);
        ctx->PSSetShader(impl_->psChroma.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }

    // Release the NV12 from the OM stage so the caller can hand it to NVENC
    // (nvEncMapInputResource) without a write-target hazard on the same texture.
    ID3D11RenderTargetView* nullRtv = nullptr;
    ctx->OMSetRenderTargets(1, &nullRtv, nullptr);
}

void SbsNv12Packer::shutdown() {
    if (!impl_) return;
    impl_->raster.Reset();
    impl_->subCb.Reset();
    impl_->subSampler.Reset();
    impl_->sampler.Reset();
    impl_->psChroma.Reset();
    impl_->psLuma.Reset();
    impl_->vs.Reset();
    impl_->chromaRtv.Reset();
    impl_->lumaRtv.Reset();
    impl_->nv12.Reset();
    impl_->device.Reset();
}

} // namespace sylc

#endif // SYLC_NATIVE_RENDERER
