// Synth3D — device-local GPU passes backed by the process-wide depth service.
// See include/synth3d.h for the pipeline and the threading contract.
//
// The hand-authored shaders/synth3d.hlsl is compiled once per entry point by FXC
// during the CMake build and embedded as RCDATA in the PYD.
//
// Gated on SYLC_NATIVE_RENDERER like native_renderer.cpp: it is only added to the
// build inside the Windows-only BUILD_NATIVE_RENDERER block.
#ifdef SYLC_NATIVE_RENDERER

#include "shader_resource_ids.h"
#include "shader_resources.h"
#include "synth3d.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>   // _dupenv_s / free for the SYLC_SYNTH3D_FLUSH probe
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {

// cbuffer b0 (SynthCB in the HLSL) — 96 bytes (6 x 16). Layout must match the
// cbuffer declaration in shaders/synth3d.hlsl.
struct SynthCB {
    float max_disp;        // c0.x  disparity budget, fraction of image WIDTH
    float convergence;     // c0.y  nearness at zero parallax
    float plane_scale;     // c0.z  same semantics as the display shader
    float inv_w;           // c0.w  1 / LUMA plane width (both passes)
    int   yuv_matrix_sel;  // c1.x
    int   transfer_sel;    // c1.y
    int   diagnostics;     // c1.z
    float edge_strength;   // c1.w: luma guidance sensitivity
    float depth_texel[2];  // c2.xy: 1 / inference width,height
    float matte_texel[2];  // c2.zw: 1 / optional alpha-matte width,height
    float crop_top;        // c3.x
    float crop_bottom;     // c3.y
    float inv_h;           // c3.z: 1 / LUMA plane height
    int   matte_mode;      // c3.w: 0=off, 1=guard, 2=alpha-aware contour
    int   temporal_fill;   // c4.x: round 5a background plate on/off
    float plate_ceiling;   // c4.y: nearness ceiling for plate refresh
    float far_snap_on;     // c4.z: v4 sub-texel background reclaim (1=on)
    float pad4;            // c4.w
    float comfort_soft_disp; // c5.x: untouched knee, fraction of image width
    float comfort_hard_disp; // c5.y: asymptotic envelope, image-width fraction
    int   comfort_enabled;   // c5.z
    float pad5;              // c5.w
};
static_assert(sizeof(SynthCB) == 96, "SynthCB must be 96 bytes");

// ImageNet normalization — the SAME constants DepthEngine's input contract expects
// and that python_bindings.cpp's depth_infer_test applies (Task 2).
constexpr float kMean[3]   = {0.485f, 0.456f, 0.406f};
constexpr float kInvStd[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

std::atomic<uint64_t> gSynthClientId{0};

void set_viewport(ID3D11DeviceContext* ctx, uint32_t w, uint32_t h) {
    D3D11_VIEWPORT vp = {};
    vp.Width    = static_cast<float>(w);
    vp.Height   = static_cast<float>(h);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ctx->RSSetViewports(1, &vp);
}

}  // namespace

Synth3D::Synth3D()
    : client_id_(gSynthClientId.fetch_add(1, std::memory_order_relaxed) + 1) {}

Synth3D::~Synth3D() { stop(); }

// ---------------------------------------------------------------------------
// Resource creation (renderer thread, under NativeRenderer::Impl::mtx)
// ---------------------------------------------------------------------------
bool Synth3D::ensure_pipeline(std::string& err) {
    if (vs_) return true;
    if (!device_) { err = "synth3d: no device"; return false; }

    auto make_ps = [&](int resource_id, const char* entry,
                       ComPtr<ID3D11PixelShader>& out) -> bool {
        sylc::ShaderBytecode bytecode;
        if (!sylc::load_shader_bytecode(resource_id, bytecode, err)) return false;
        if (FAILED(device_->CreatePixelShader(bytecode.data, bytecode.size,
                                              nullptr, &out))) {
            err = std::string("CreatePixelShader(") + entry + ") failed";
            return false;
        }
        return true;
    };

    sylc::ShaderBytecode vsBytecode;
    if (!sylc::load_shader_bytecode(
            IDR_SYLC_SYNTH3D_VS_FULL, vsBytecode, err)) return false;
    if (FAILED(device_->CreateVertexShader(vsBytecode.data, vsBytecode.size,
                                           nullptr, &vs_))) {
        err = "CreateVertexShader(VS_Full) failed"; return false;
    }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_DEPTH_PREP,
                 "PS_DepthPrep", psPrep_)) { vs_.Reset(); return false; }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_WARP_LUMA,
                 "PS_WarpLuma", psWarpLuma_)) { vs_.Reset(); return false; }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_WARP_CHROMA,
                 "PS_WarpChroma", psWarpChroma_)) { vs_.Reset(); return false; }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_DEPTH_VIEW_LUMA,
                 "PS_DepthViewLuma", psViewLuma_)) { vs_.Reset(); return false; }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_DEPTH_VIEW_CHROMA,
                 "PS_DepthViewChroma", psViewChroma_)) { vs_.Reset(); return false; }
    if (!make_ps(IDR_SYLC_SYNTH3D_PS_PLATE_ACCUM,
                 "PS_PlateAccum", psPlateAccum_)) { vs_.Reset(); return false; }

    // LINEAR + CLAMP: the prep pass downscales to the inference grid and the warp resamples at
    // fractional offsets, so bilinear is wanted in both (unlike the lossless packer).
    D3D11_SAMPLER_DESC sd = {};
    sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MinLOD   = 0.0f;
    sd.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(device_->CreateSamplerState(&sd, &sampler_))) {
        err = "synth3d: CreateSamplerState failed"; vs_.Reset(); return false;
    }

    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode        = D3D11_FILL_SOLID;
    rs.CullMode        = D3D11_CULL_NONE;   // fullscreen triangle, any winding
    rs.DepthClipEnable = TRUE;
    if (FAILED(device_->CreateRasterizerState(&rs, &raster_))) {
        err = "synth3d: CreateRasterizerState failed"; vs_.Reset(); return false;
    }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(SynthCB);
    cbd.Usage     = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device_->CreateBuffer(&cbd, nullptr, &cb_))) {
        err = "synth3d: CreateBuffer(SynthCB) failed"; vs_.Reset(); return false;
    }
    return true;
}

bool Synth3D::ensure_provenance_shader(std::string& err) {
    if (psProvenance_) return true;
    if (!device_) {
        err = "synth3d: no device for provenance sidecar";
        return false;
    }
    sylc::ShaderBytecode bytecode;
    if (!sylc::load_shader_bytecode(
            IDR_SYLC_SYNTH3D_PS_PROVENANCE, bytecode, err)) {
        return false;
    }
    if (FAILED(device_->CreatePixelShader(
            bytecode.data, bytecode.size, nullptr, &psProvenance_))) {
        err = "CreatePixelShader(PS_WarpProvenance) failed";
        return false;
    }
    return true;
}

bool Synth3D::ensure_prep(std::string& err) {
    if (prepTex_) return true;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = static_cast<UINT>(grid_width_);
    td.Height           = static_cast<UINT>(grid_height_);
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DEFAULT;
    td.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device_->CreateTexture2D(&td, nullptr, &prepTex_))) {
        err = "synth3d: CreateTexture2D(prep) failed"; return false;
    }
    if (FAILED(device_->CreateRenderTargetView(prepTex_.Get(), nullptr, &prepRtv_))) {
        err = "synth3d: CreateRenderTargetView(prep) failed"; prepTex_.Reset(); return false;
    }

    // Staging ring: the prep target is CopyResource'd here, then mapped with
    // DO_NOT_WAIT one frame (or more) later, so the presenter thread never waits
    // on the GPU. kRing slots bound how stale a drained frame can be.
    D3D11_TEXTURE2D_DESC sd = td;
    sd.Usage          = D3D11_USAGE_STAGING;
    sd.BindFlags      = 0;
    sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    for (int i = 0; i < kRing; ++i) {
        if (FAILED(device_->CreateTexture2D(&sd, nullptr, &stag_[i]))) {
            err = "synth3d: CreateTexture2D(staging) failed";
            for (int k = 0; k < kRing; ++k) stag_[k].Reset();
            prepRtv_.Reset(); prepTex_.Reset();
            return false;
        }
        stag_seq_[i] = 0;
        stag_capture_time_[i] = {};
        stag_video_time_ms_[i] = -1.0;
    }
    stag_write_ = 0;
    seq_ctr_    = 0;
    return true;
}

bool Synth3D::ensure_depth(std::string& err) {
    if (depthTex_) return true;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = static_cast<UINT>(grid_width_);
    td.Height           = static_cast<UINT>(grid_height_);
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R16G16_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device_->CreateTexture2D(&td, nullptr, &depthTex_))) {
        err = "synth3d: CreateTexture2D(depth) failed"; return false;
    }
    if (FAILED(device_->CreateShaderResourceView(depthTex_.Get(), nullptr, &depthSrv_))) {
        err = "synth3d: CreateShaderResourceView(depth) failed"; depthTex_.Reset(); return false;
    }

    // Round 5a: transport (flow x/y + reliability from the published map) and
    // the two ping-pong plate targets. Same grid, filterable UNORM. Their
    // creation is unconditional (cheap) so a mid-session temporal_fill toggle
    // needs no resource churn; the passes themselves are gated on the flag.
    D3D11_TEXTURE2D_DESC tt = td;
    tt.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
    if (FAILED(device_->CreateTexture2D(&tt, nullptr, &transportTex_))) {
        err = "synth3d: CreateTexture2D(transport) failed"; return false;
    }
    if (FAILED(device_->CreateShaderResourceView(
            transportTex_.Get(), nullptr, &transportSrv_))) {
        err = "synth3d: CreateShaderResourceView(transport) failed";
        transportTex_.Reset(); return false;
    }
    D3D11_TEXTURE2D_DESC pd = td;
    pd.Format         = DXGI_FORMAT_R16G16B16A16_UNORM;
    pd.Usage          = D3D11_USAGE_DEFAULT;
    pd.BindFlags      = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    pd.CPUAccessFlags = 0;
    for (int i = 0; i < 2; ++i) {
        if (FAILED(device_->CreateTexture2D(&pd, nullptr, &plateTex_[i])) ||
            FAILED(device_->CreateRenderTargetView(
                plateTex_[i].Get(), nullptr, &plateRtv_[i])) ||
            FAILED(device_->CreateShaderResourceView(
                plateTex_[i].Get(), nullptr, &plateSrv_[i]))) {
            err = "synth3d: plate texture creation failed";
            plateTex_[0].Reset(); plateTex_[1].Reset();
            plateRtv_[0].Reset(); plateRtv_[1].Reset();
            plateSrv_[0].Reset(); plateSrv_[1].Reset();
            return false;
        }
    }
    plate_read_ = 0;
    plate_valid_ = false;

    depth_valid_ = false;   // nothing uploaded yet: process() stays a 2D passthrough
    return true;
}

bool Synth3D::ensure_matte(std::string& err) {
    if (!matte_armed_) return true;
    if (matteTex_ && matte_texture_width_ == matte_width_ &&
        matte_texture_height_ == matte_height_) {
        return true;
    }

    matteSrv_.Reset();
    matteTex_.Reset();
    matte_texture_width_ = matte_texture_height_ = 0;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = matte_width_;
    td.Height           = matte_height_;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(device_->CreateTexture2D(&td, nullptr, &matteTex_))) {
        err = "synth3d: CreateTexture2D(human matte) failed";
        return false;
    }
    if (FAILED(device_->CreateShaderResourceView(
            matteTex_.Get(), nullptr, &matteSrv_))) {
        err = "synth3d: CreateShaderResourceView(human matte) failed";
        matteTex_.Reset();
        return false;
    }
    matte_texture_width_ = matte_width_;
    matte_texture_height_ = matte_height_;
    matte_dirty_ = true;
    return true;
}

bool Synth3D::ensure_warp(uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                          DXGI_FORMAT fmt, std::string& err) {
    if (warpTex_[0] && provenanceTex_[0] &&
        y_w_ == y_w && y_h_ == y_h && c_w_ == c_w && c_h_ == c_h &&
        plane_fmt_ == fmt) {
        return true;
    }
    for (int i = 0; i < kNumOut; ++i) {
        warpSrv_[i].Reset(); warpRtv_[i].Reset(); warpTex_[i].Reset();
        out_srv_[i] = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        provenanceSrv_[i].Reset();
        provenanceRtv_[i].Reset();
        provenanceTex_[i].Reset();
    }
    outputs_valid_ = false;
    readTex_.Reset(); read_w_ = read_h_ = 0; read_fmt_ = DXGI_FORMAT_UNKNOWN;

    // Slots 0/3 are the luma planes (full size), 1/2/4/5 the chroma planes.
    for (int i = 0; i < kNumOut; ++i) {
        const bool isLuma = (i == 0 || i == 3);
        D3D11_TEXTURE2D_DESC td = {};
        td.Width            = isLuma ? y_w : c_w;
        td.Height           = isLuma ? y_h : c_h;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = fmt;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_DEFAULT;   // the plane textures are DYNAMIC and
        td.BindFlags        = D3D11_BIND_RENDER_TARGET |   // therefore NOT RTV-capable;
                              D3D11_BIND_SHADER_RESOURCE;  // these are dedicated outputs.
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &warpTex_[i])) ||
            FAILED(device_->CreateRenderTargetView(warpTex_[i].Get(), nullptr, &warpRtv_[i])) ||
            FAILED(device_->CreateShaderResourceView(warpTex_[i].Get(), nullptr, &warpSrv_[i]))) {
            err = "synth3d: warp output texture/view creation failed";
            for (int k = 0; k < kNumOut; ++k) {
                warpSrv_[k].Reset(); warpRtv_[k].Reset(); warpTex_[k].Reset();
                out_srv_[k] = nullptr;
            }
            return false;
        }
        out_srv_[i] = warpSrv_[i].Get();
    }

    // The full-resolution luma pass is the sole visibility solver.  Persist
    // its compact decision in two integer maps so 4:2:0 chroma cannot choose
    // a different layer merely because it runs on a half-resolution lattice.
    D3D11_TEXTURE2D_DESC provenance_desc = {};
    provenance_desc.Width = y_w;
    provenance_desc.Height = y_h;
    provenance_desc.MipLevels = 1;
    provenance_desc.ArraySize = 1;
    provenance_desc.Format = DXGI_FORMAT_R32_UINT;
    provenance_desc.SampleDesc.Count = 1;
    provenance_desc.Usage = D3D11_USAGE_DEFAULT;
    provenance_desc.BindFlags = D3D11_BIND_RENDER_TARGET |
                                D3D11_BIND_SHADER_RESOURCE;
    for (int i = 0; i < 2; ++i) {
        if (FAILED(device_->CreateTexture2D(
                &provenance_desc, nullptr, &provenanceTex_[i])) ||
            FAILED(device_->CreateRenderTargetView(
                provenanceTex_[i].Get(), nullptr, &provenanceRtv_[i])) ||
            FAILED(device_->CreateShaderResourceView(
                provenanceTex_[i].Get(), nullptr, &provenanceSrv_[i]))) {
            err = "synth3d: provenance texture/view creation failed";
            for (int k = 0; k < kNumOut; ++k) {
                warpSrv_[k].Reset(); warpRtv_[k].Reset(); warpTex_[k].Reset();
                out_srv_[k] = nullptr;
            }
            for (int k = 0; k < 2; ++k) {
                provenanceSrv_[k].Reset();
                provenanceRtv_[k].Reset();
                provenanceTex_[k].Reset();
            }
            return false;
        }
    }
    y_w_ = y_w; y_h_ = y_h; c_w_ = c_w; c_h_ = c_h; plane_fmt_ = fmt;
    // A geometry/format change means a new source: re-prime the shared temporal
    // filter on the next inference, exactly like a seek.
    if (depth_service_) depth_service_->notify_seek();
    return true;
}

void Synth3D::release_depth_grid() {
    // Everything sized from the inference grid. Dropped when the attached service changes
    // grid so ensure_prep/ensure_depth rebuild at the new resolution instead
    // of returning the previous preset's textures on their "already exists"
    // fast path. The warp outputs follow the SOURCE geometry, not the grid,
    // and are deliberately left alone.
    for (int i = 0; i < kRing; ++i) {
        stag_[i].Reset();
        stag_seq_[i] = 0;
        stag_capture_time_[i] = {};
        stag_video_time_ms_[i] = -1.0;
    }
    depthSrv_.Reset(); depthTex_.Reset();
    transportSrv_.Reset(); transportTex_.Reset();
    for (int i = 0; i < 2; ++i) {
        plateSrv_[i].Reset();
        plateRtv_[i].Reset();
        plateTex_[i].Reset();
    }
    plateReadTex_.Reset();
    plate_read_ = 0;
    plate_valid_ = false;
    plate_cut_seen_ms_ = -1.0;
    prepRtv_.Reset(); prepTex_.Reset();
    stag_write_  = 0;
    seq_ctr_     = 0;
    depth_valid_ = false;
    depth_dirty_ = test_armed_;
}

void Synth3D::release_gpu() {
    lab_outputs_valid_ = false;
    if (stereo_lab_) stereo_lab_->release();
    for (int i = 0; i < kNumOut; ++i) {
        warpSrv_[i].Reset(); warpRtv_[i].Reset(); warpTex_[i].Reset();
        out_srv_[i] = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        provenanceSrv_[i].Reset();
        provenanceRtv_[i].Reset();
        provenanceTex_[i].Reset();
    }
    release_depth_grid();
    matteSrv_.Reset(); matteTex_.Reset();
    matte_texture_width_ = matte_texture_height_ = 0;
    matte_dirty_ = matte_armed_;
    readTex_.Reset(); read_w_ = read_h_ = 0; read_fmt_ = DXGI_FORMAT_UNKNOWN;
    cb_.Reset(); raster_.Reset(); sampler_.Reset();
    psProvenance_.Reset();
    psViewChroma_.Reset(); psViewLuma_.Reset();
    psWarpChroma_.Reset(); psWarpLuma_.Reset(); psPrep_.Reset();
    vs_.Reset();
    y_w_ = y_h_ = c_w_ = c_h_ = 0;
    plane_fmt_ = DXGI_FORMAT_UNKNOWN;
    outputs_valid_ = false;
}

// ---------------------------------------------------------------------------
// Readback ring (never blocks the presenter thread)
// ---------------------------------------------------------------------------
void Synth3D::push_readback(
        ID3D11DeviceContext* ctx, double video_time_ms) {
    const int w = stag_write_;
    if (!stag_[w]) return;
    // Overwriting a slot that was never drained simply DROPS that older request —
    // which is what bounds staleness to kRing frames.
    // Timestamp the exact source observation represented by this GPU copy.
    // The worker may not dequeue it until a previous inference has finished;
    // retaining this instant is what separates video time from compute time.
    stag_capture_time_[w] = std::chrono::steady_clock::now();
    stag_video_time_ms_[w] =
        std::isfinite(video_time_ms) && video_time_ms >= 0.0
            ? video_time_ms : -1.0;
    ctx->CopyResource(stag_[w].Get(), prepTex_.Get());
    stag_seq_[w] = ++seq_ctr_;
    stag_write_  = (w + 1) % kRing;

    // Submit the copy NOW. Without this the driver batches it, and a frame
    // carrying little GPU work never accumulates enough to flush on its own:
    // the copy was still unmappable a WHOLE frame later, drain_readback found
    // nothing, no submission happened at all, and the worker lost 41.7 ms.
    // Measured 1080p H.264 (edge264, light frames) vs the same film in 4K HEVC
    // (avcodec, heavy frames -- which flushed implicitly and never missed):
    //   without: 436/1991 granted taps stalled (22%), cycle 50 ms, 19.5 fps
    //   with:      1/1897 (priming only),          cycle 41.7 ms, 24.0 fps
    // The cost is the lost batching: present p50 +0.2 ms against a 41.7 ms
    // budget. 4K is unaffected (it was already source-capped).
    // SYLC_SYNTH3D_FLUSH=0 restores the batched behaviour.
    static const bool force_flush = []() {
        char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
        size_t len = 0;
        bool on = true;
        if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_FLUSH") == 0 && env) {
            on = env[0] != '0';   // SYLC_SYNTH3D_FLUSH=0 = rollback
            free(env);
        }
        return on;
    }();
    if (force_flush) ctx->Flush();
}

void Synth3D::drain_readback(ID3D11DeviceContext* ctx) {
    if (!depth_service_) return;

    // NEWEST ready slot wins. The model must see the most recent frame we actually
    // have: draining oldest-first would hand it a map up to kRing frames old, and
    // stale depth on a moving picture reads as the depth lagging the image. Walk the
    // pending slots newest -> oldest and take the first whose copy has landed.
    int order[kRing];
    int n = 0;
    for (int i = 0; i < kRing; ++i)
        if (stag_seq_[i] != 0) order[n++] = i;
    if (n == 0) { depth_service_->note_drain_miss(true); return; }
    for (int i = 1; i < n; ++i) {                 // insertion sort by seq DESC (n <= 3)
        const int k = order[i];
        int j = i - 1;
        while (j >= 0 && stag_seq_[order[j]] < stag_seq_[k]) { order[j + 1] = order[j]; --j; }
        order[j + 1] = k;
    }

    D3D11_MAPPED_SUBRESOURCE m = {};
    int best = -1;
    for (int i = 0; i < n; ++i) {
        const HRESULT hr = ctx->Map(stag_[order[i]].Get(), 0, D3D11_MAP_READ,
                                    D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
        if (SUCCEEDED(hr)) { best = order[i]; break; }
        if (hr != DXGI_ERROR_WAS_STILL_DRAWING) {
            stag_seq_[order[i]] = 0;  // broken: drop
            stag_capture_time_[order[i]] = {};
            stag_video_time_ms_[order[i]] = -1.0;
        }
    }
    if (best < 0) {
        // Nothing has landed yet; retry next frame. NEVER blocks -- but this
        // frame produced NO submission, so the worker keeps waiting.
        depth_service_->note_drain_miss(false);
        return;
    }

    // RGBA32F -> CHW float32, ImageNet-normalized. Written into a renderer-owned
    // scratch buffer so the mailbox mutex is held only for an O(1) vector swap.
    const size_t plane = static_cast<size_t>(grid_width_) * grid_height_;
    if (in_scratch_.size() != 3 * plane) in_scratch_.assign(3 * plane, 0.0f);
    if (aspect_luma_scratch_.size() != plane)
        aspect_luma_scratch_.assign(plane, 0.0f);
    float* dst = in_scratch_.data();
    float* aspect_dst = aspect_luma_scratch_.data();
    const auto* base = static_cast<const uint8_t*>(m.pData);
    for (int row = 0; row < grid_height_; ++row) {
        const float* s = reinterpret_cast<const float*>(base + static_cast<size_t>(row) * m.RowPitch);
        float* d = dst + static_cast<size_t>(row) * grid_width_;
        for (int x = 0; x < grid_width_; ++x) {
            d[x]               = (s[x * 4 + 0] - kMean[0]) * kInvStd[0];
            d[plane + x]       = (s[x * 4 + 1] - kMean[1]) * kInvStd[1];
            d[2 * plane + x]   = (s[x * 4 + 2] - kMean[2]) * kInvStd[2];
            aspect_dst[static_cast<size_t>(row) * grid_width_ + x] =
                s[x * 4 + 3];
        }
    }
    ctx->Unmap(stag_[best].Get(), 0);
    // Retire this slot AND every older one: they can only be staler than what we took.
    const uint64_t used = stag_seq_[best];
    const auto capture_time = stag_capture_time_[best];
    const double video_time_ms = stag_video_time_ms_[best];
    for (int i = 0; i < kRing; ++i) {
        if (stag_seq_[i] <= used) {
            stag_seq_[i] = 0;
            stag_capture_time_[i] = {};
            stag_video_time_ms_[i] = -1.0;
        }
    }

    // submit() swaps the vector and its source timestamp into the single shared
    // mailbox. If leadership moved while this GPU copy was in flight, the stale
    // result is discarded.
    depth_service_->submit(
        client_id_, in_scratch_, aspect_luma_scratch_,
        video_time_ms, capture_time,
        static_cast<int>(y_w_), static_cast<int>(y_h_));
}

bool Synth3D::upload_depth(ID3D11DeviceContext* ctx, std::string& err) {
    const uint16_t* geometry = nullptr;
    // Declared at function scope: geometry aliases this map's storage, so the
    // reference must outlive the Map+memcpy loop below. Scoping it to the
    // else-block dropped the last renderer-side reference before the copy;
    // the worker publishing the next map then freed the buffer mid-copy
    // (use-after-free on the GUI thread, ~5x likelier under TensorRT's
    // higher publication rate).
    std::shared_ptr<const SharedDepthService::GeometryMap> snapshot;
    if (test_armed_) {
        // The debug bypass wins over the engine: re-upload only when it changed.
        if (!depth_dirty_) return true;
        if (test_geometry_.size() !=
            SharedDepthService::kGeometryChannels *
            static_cast<size_t>(grid_width_) * grid_height_) {
            err = "synth3d: test geometry size no longer matches the live grid";
            return false;
        }
        geometry = test_geometry_.data();
    } else {
        if (!depth_service_) return true;
        uint64_t sequence = depth_sequence_;
        double source_video_ms = -1.0;
        snapshot = depth_service_->snapshot(
            depth_sequence_, sequence, source_video_ms);
        if (!snapshot) return true;                // keep the local GPU texture
        // Shot identity of the map this surface is about to warp with:
        // adopted together with the pixels (cross-shot gate, 04/08).
        depth_video_ms_ = source_video_ms;
        if (snapshot->size() !=
            SharedDepthService::kGeometryChannels *
            static_cast<size_t>(grid_width_) * grid_height_) {
            err = "synth3d: shared geometry map size does not match the live grid";
            return false;
        }
        geometry = snapshot->data();
        depth_sequence_ = sequence;
    }

    // Six interleaved channels per texel: 0-1 feed the RG16 geometry texture,
    // 2-5 the RGBA16 transport texture (round 5a). Strided copies replace the
    // historical row memcpy, whose layout only matched the 2-channel era.
    constexpr size_t kCh = SharedDepthService::kGeometryChannels;
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(depthTex_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        err = "synth3d: Map(depth texture) failed";
        return false;
    }
    auto* out = static_cast<uint8_t*>(m.pData);
    for (int row = 0; row < grid_height_; ++row) {
        auto* dst = reinterpret_cast<uint16_t*>(
            out + static_cast<size_t>(row) * m.RowPitch);
        const uint16_t* src_row = geometry +
            kCh * static_cast<size_t>(row) * grid_width_;
        for (int x = 0; x < grid_width_; ++x) {
            dst[2 * x + 0] = src_row[kCh * x + 0];
            dst[2 * x + 1] = src_row[kCh * x + 1];
        }
    }
    ctx->Unmap(depthTex_.Get(), 0);
    if (transportTex_) {
        D3D11_MAPPED_SUBRESOURCE mt = {};
        if (FAILED(ctx->Map(transportTex_.Get(), 0,
                            D3D11_MAP_WRITE_DISCARD, 0, &mt))) {
            err = "synth3d: Map(transport texture) failed";
            return false;
        }
        auto* tout = static_cast<uint8_t*>(mt.pData);
        for (int row = 0; row < grid_height_; ++row) {
            auto* dst = reinterpret_cast<uint16_t*>(
                tout + static_cast<size_t>(row) * mt.RowPitch);
            const uint16_t* src_row = geometry +
                kCh * static_cast<size_t>(row) * grid_width_;
            for (int x = 0; x < grid_width_; ++x) {
                dst[4 * x + 0] = src_row[kCh * x + 2];
                dst[4 * x + 1] = src_row[kCh * x + 3];
                dst[4 * x + 2] = src_row[kCh * x + 4];
                dst[4 * x + 3] = src_row[kCh * x + 5];
            }
        }
        ctx->Unmap(transportTex_.Get(), 0);
    }
    depth_valid_ = true;
    depth_dirty_ = false;
    map_refreshed_ = true;   // a NEW map landed: the plate may accumulate once
    return true;
}

bool Synth3D::upload_matte(ID3D11DeviceContext* ctx, std::string& err) {
    if (!matte_armed_ || !matte_dirty_) return true;
    const size_t expected = 4 * static_cast<size_t>(matte_width_) * matte_height_;
    if (!matteTex_ || test_matte_.size() != expected) {
        err = "synth3d: human matte storage does not match its texture";
        return false;
    }

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(matteTex_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
        err = "synth3d: Map(human matte texture) failed";
        return false;
    }
    auto* out = static_cast<uint8_t*>(m.pData);
    for (uint32_t row = 0; row < matte_height_; ++row) {
        std::memcpy(out + static_cast<size_t>(row) * m.RowPitch,
                    test_matte_.data() + 4 * static_cast<size_t>(row) * matte_width_,
                    4 * matte_width_);
    }
    ctx->Unmap(matteTex_.Get(), 0);
    matte_dirty_ = false;
    return true;
}

void Synth3D::set_local_error(const std::string& err) {
    local_error_ = err.empty() ? "synth3d: unknown renderer failure" : err;
    for (char& c : local_error_)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
}

// ---------------------------------------------------------------------------
// Per-frame GPU work
// ---------------------------------------------------------------------------
bool Synth3D::process(ID3D11DeviceContext* ctx,
                      ID3D11ShaderResourceView* const src[3],
                      uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                      DXGI_FORMAT plane_fmt, float plane_scale,
                      int matrix_sel, int transfer_sel,
                      double video_time_ms) {
    outputs_valid_ = false;
    lab_outputs_valid_ = false;
    map_refreshed_ = false;
    if (!ctx || !src || !src[0] || !src[1] || !src[2]) return false;
    if (!y_w || !y_h || !c_w || !c_h) return false;

    // Timed here, before any early return below can skip it: the question this
    // answers is how often the renderer comes back at all, not how often it
    // completes a warp.
    if (!test_armed_ && depth_service_) depth_service_->note_pass(client_id_);

    std::string err;
    if (!ensure_pipeline(err) || !ensure_prep(err) || !ensure_depth(err) ||
        !ensure_matte(err) ||
        !ensure_warp(y_w, y_h, c_w, c_h, plane_fmt, err)) {
        set_local_error(err);
        return false;
    }

    // The final Stereo Lab pass uses through t11. Any of its corrected outputs
    // may also have been bound by the previous present()/cast. Clear the whole
    // shared range before raw warp or Lab textures return to the OM stage.
    ID3D11ShaderResourceView* const kNoSrv[12] = {};
    ctx->PSSetShaderResources(0, 12, kNoSrv);

    SynthCB cb = {};
    cb.max_disp       = params_.strength_pct * 0.01f;   // % of width -> fraction
    cb.convergence    = params_.convergence;
    // Per-shot auto-convergence: the service's smoothed stabilizer suggestion
    // replaces the manual plane. The test-depth bypass keeps the manual value
    // (golden/warp tests assert on exact geometry), and so does the window
    // before the first map (suggestion primes at 0.5 = the manual default).
    if (params_.auto_convergence && !test_armed_ && depth_service_) {
        const float suggested = depth_service_->suggested_convergence();
        if (suggested >= 0.0f && suggested <= 1.0f)
            cb.convergence = suggested;
    }
    cb.plane_scale    = (plane_scale > 0.0f) ? plane_scale : 1.0f;
    cb.inv_w          = 1.0f / static_cast<float>(y_w);
    cb.yuv_matrix_sel = matrix_sel;
    cb.transfer_sel   = transfer_sel;
    cb.diagnostics    = params_.diagnostics ? 1 : 0;
    cb.edge_strength  = 28.0f;
    cb.depth_texel[0] = 1.0f / static_cast<float>(grid_width_);
    cb.depth_texel[1] = 1.0f / static_cast<float>(grid_height_);
    cb.matte_texel[0] = matte_armed_ ?
        1.0f / static_cast<float>(matte_width_) : 1.0f;
    cb.matte_texel[1] = matte_armed_ ?
        1.0f / static_cast<float>(matte_height_) : 1.0f;
    cb.crop_top = (std::max)(0.0f, (std::min)(0.45f, params_.crop_top));
    cb.crop_bottom = (std::max)(0.0f, (std::min)(0.45f, params_.crop_bottom));
    cb.inv_h = 1.0f / static_cast<float>(y_h);
    cb.matte_mode = matte_armed_ ? matte_mode_ : 0;
    const bool plate_on = params_.temporal_fill &&
                          plateTex_[0] && plateTex_[1] && psPlateAccum_;
    cb.temporal_fill = plate_on ? 1 : 0;
    cb.plate_ceiling = 0.45f;
    // v4 sub-texel background reclaim in guided_nearness (see the shader):
    // the CPU realign fixed the grid scale, this closes the pixel band
    // inside the edge texel. SYLC_SYNTH3D_FAR_SNAP=0 = rollback.
    static const bool far_snap_on = []() {
        char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
        size_t len = 0;
        bool on = true;
        if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_FAR_SNAP") == 0 && env) {
            on = env[0] != '0';
            free(env);
        }
        return on;
    }();
    cb.far_snap_on = far_snap_on ? 1.0f : 0.0f;
    cb.comfort_soft_disp = params_.comfort_soft_pct * 0.01f;
    cb.comfort_hard_disp = params_.comfort_hard_pct * 0.01f;
    cb.comfort_enabled = (params_.comfort_enabled &&
                          cb.comfort_soft_disp >= 0.0f &&
                          cb.comfort_hard_disp > cb.comfort_soft_disp) ? 1 : 0;
    if (cb.crop_top + cb.crop_bottom > 0.55f) {
        cb.crop_top = 0.0f;
        cb.crop_bottom = 0.0f;
    }

    // Post-cut/seek ease-out ramp: a fresh snap teleports the stereo geometry
    // in one frame (100-300ms for the eyes to re-fuse), which reads as a jolt.
    // Scale the disparity budget down right after the snap and let it inflate
    // back to full over ramp_ms_. The test-depth bypass keeps its full
    // geometry unconditionally -- golden/warp tests assert on it directly.
    double cut_in_diag = SharedDepthService::kLookaheadNone;
    if (!test_armed_ && depth_service_) {
        const double snap_video = depth_service_->last_snap_video_ms();
        const bool have_video_clock =
            std::isfinite(video_time_ms) && video_time_ms >= 0.0 &&
            std::isfinite(snap_video) && snap_video >= 0.0;
        const int64_t snap_steady = depth_service_->last_snap_steady_ms();
        if ((have_video_clock || snap_steady >= 0) && ramp_ms_ > 0.0f) {
            double elapsed_ms = 0.0;
            if (have_video_clock) {
                elapsed_ms = video_time_ms - snap_video;
            } else {
                const int64_t now_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                elapsed_ms = static_cast<double>(now_ms - snap_steady);
            }
            float t = static_cast<float>(elapsed_ms / ramp_ms_);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);   // windows.h min/max macros: no std::
            const float scale = t * (2.0f - t);   // ease-out
            cb.max_disp *= scale;
        }
        // PRE-cut ease-down (author rule 2026-08-03: « il faut toujours
        // anticiper la coupe d'au moins 1 image »). The scout observed a cut
        // in the DECODED future, dated at the new shot's first frame; while
        // the presented frame still belongs to the dying shot, glide its
        // disparity toward flat with the MIRROR of the post-snap ease. At the
        // cut the two ramps meet at zero: depth is continuous THROUGH the
        // cut instead of jumping from full (T0) to flat (T1). The advisory
        // self-purges at T1 and the post-snap ramp takes over seamlessly.
        // A NEGATIVE delay (down to the scout's hold window, ~120ms) means
        // the cut just landed: HOLD the flat until the post-snap ramp has
        // certainly taken over — without it, a worker snap registering one
        // frame late let T1 render at full disparity with a possibly
        // cross-shot map (the author's "red/cyan residue overflowing onto
        // the frame after the cut").
        const double cut_in = depth_service_->lookahead_cut_in_ms();
        cut_in_diag = cut_in;
        if (cut_in > -150.0 && cut_in < static_cast<double>(ramp_ms_) &&
            ramp_ms_ > 0.0f) {
            // -150..0 = the scout's hold window (the sentinel is -1e9, far
            // below): keep FLAT through it. 0..ramp = the mirrored ease-down.
            float t2 = static_cast<float>(
                (cut_in > 0.0 ? cut_in : 0.0) / ramp_ms_);
            t2 = t2 > 1.0f ? 1.0f : t2;
            cb.max_disp *= t2 * (2.0f - t2);
            // Through the hold the presented frame belongs to the NEW shot
            // while the published map/plate may still be the OLD shot's:
            // never sample the plate there (its background IS old-shot
            // pixels), and the zero budget makes diagnostic_yuv paint
            // neutral instead of the stale map's color flats.
            //
            // NO pre-cut lead here, and that is deliberate — it was tried on
            // 2026-08-03 and reverted the same evening on author A/B. Leading
            // by one source frame kills the plate on the dying shot's LAST
            // frame, where it is same-shot and entirely legitimate: plate_trust
            // collapses to 0 and those disocclusions fall back to the stretch,
            // which reads worse than the remembered background it replaced.
            // The anticipation rule belongs to the DISPARITY (which does glide
            // down over ramp_ms_ before the cut); the plate is not a continuous
            // effect but a memory, and memory of the CURRENT shot stays valid
            // right up to its last frame. The real exposure is on the other
            // side of the cut — see the plate purge below.
            if (cut_in <= 0.0)
                cb.temporal_fill = 0;
        }
    }
    ctx->UpdateSubresource(cb_.Get(), 0, nullptr, &cb, 0, 0);

    // Shared state for every pass: fullscreen triangle from SV_VertexID (no vertex
    // buffer / input layout). present() re-sets every one of these before its own
    // draw, so nothing here needs saving/restoring except the OM stage (below).
    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs_.Get(), nullptr, 0);
    ctx->RSSetState(raster_.Get());
    ID3D11SamplerState* smp = sampler_.Get();
    ctx->PSSetSamplers(0, 1, &smp);
    ID3D11Buffer* cbuf = cb_.Get();
    ctx->PSSetConstantBuffers(0, 1, &cbuf);
    ctx->PSSetShaderResources(0, 3, src);

    // (1) depth prep + non-blocking readback. Skipped whenever the worker cannot use
    // the result — a test depth is armed (inference bypassed) or the engine is not
    // running (still loading, or failed) — so a dead engine costs zero GPU/PCIe.
    if (!test_armed_ && depth_service_ &&
        depth_service_->wants_input(client_id_, video_time_ms)) {
        ID3D11RenderTargetView* rtv = prepRtv_.Get();
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        set_viewport(
            ctx, static_cast<uint32_t>(grid_width_),
            static_cast<uint32_t>(grid_height_));
        ctx->PSSetShader(psPrep_.Get(), nullptr, 0);
        ctx->Draw(3, 0);                 // the draw covers the whole RT -> no clear
        push_readback(ctx, video_time_ms);
    }
    // Retire any copies that were already in flight when a repeated pause PTS
    // closed the capture gate. submit() performs the authoritative duplicate
    // check, so none can advance the worker; draining prevents them resurfacing
    // as stale observations when playback resumes.
    if (!test_armed_ && depth_service_) {
        bool staging_pending = false;
        for (int i = 0; i < kRing; ++i)
            staging_pending = staging_pending || stag_seq_[i] != 0;
        if (staging_pending) drain_readback(ctx);
    }

    // (2) publish the freshest nearness map (test bypass wins over the engine).
    if (!upload_depth(ctx, err) || !upload_matte(ctx, err)) {
        set_local_error(err);
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        return false;
    }
    // Resource setup/upload succeeded. A previous transient renderer error may
    // now be retried explicitly (or after a driver recovery) without leaving a
    // stale red UI state behind.
    local_error_.clear();

    if (!depth_valid_) {
        // No depth yet (engine still loading, no test depth): leave the caller on the
        // untouched source SRVs. Playback continues in 2D — it never waits for depth.
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        return false;
    }

    // A temporal plate is device-local memory and therefore needs its OWN
    // shot identity. Waiting for the depth worker's snap is too late: with the
    // readback/inference pipeline, T0 of the new shot can be presented one or
    // two frames before that snap. Consume the absolute decoder/scout boundary
    // on the presentation clock and invalidate both ping-pong targets exactly
    // at T0, independently on every renderer surface.
    double reached_cut_pts = -1.0;
    bool plate_cut_crossed = false;
    if (depth_service_ && std::isfinite(video_time_ms) && video_time_ms >= 0.0) {
        reached_cut_pts = depth_service_->latest_presented_cut(video_time_ms);
        if (reached_cut_pts >= 0.0 &&
            (plate_cut_seen_ms_ < 0.0 ||
             reached_cut_pts > plate_cut_seen_ms_ +
                                   SharedDepthService::kPtsJitterMs)) {
            plate_cut_seen_ms_ = reached_cut_pts;
            plate_valid_ = false;
            plate_cut_crossed = true;
            // Even if a post-cut map is already available, do not sample a
            // plate on the boundary frame. It may be rebuilt from T0 below,
            // but becomes visible only on a later same-shot frame.
            cb.temporal_fill = 0;
            if (stereo_lab_) stereo_lab_->reset_epoch();
        }
    }

    // Cross-shot gate (04/08) — the STATE test that replaces the deadline
    // race. With the readback + pipeline lag, the map uploaded above can
    // still belong to the shot BEFORE a cut that the presented frame is
    // already past; warping the new shot with the old shot's geometry paints
    // the old silhouettes as disocclusion masks (the red/cyan contours that
    // survive every hard cut). While a recorded boundary c satisfies
    // depth_video_ms_ < c <= video_time_ms, present FLAT: zero disparity
    // budget (nothing is warped, diagnostics paint neutral) and no plate.
    // The gate releases when the first map observed at/after the cut lands —
    // by state, never by timer. Evaluated HERE because only upload_depth
    // just decided which map this frame actually warps with; the cb was
    // already uploaded for the prep pass, so a triggered gate re-uploads it
    // (80 bytes). The temporal ramps above stay: they ease comfort and are
    // the only cover on untimed clocks. SYLC_SYNTH3D_SHOT_GATE=0 = rollback.
    bool shot_gate = false;
    if (!test_armed_ && depth_service_ &&
        std::isfinite(video_time_ms) && video_time_ms >= 0.0) {
        static const bool gate_enabled = []() {
            char* env = nullptr;   // house idiom (MSVC-safe, cf. SYLC_FLOW_THREADS)
            size_t len = 0;
            bool on = true;
            if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_SHOT_GATE") == 0 && env) {
                on = env[0] != '0';   // SYLC_SYNTH3D_SHOT_GATE=0 = rollback
                free(env);
            }
            return on;
        }();
        if (gate_enabled) {
            shot_gate = depth_service_->cross_shot(
                depth_video_ms_, video_time_ms);
            if (shot_gate) {
                cb.max_disp = 0.0f;
                cb.temporal_fill = 0;
            }
        }
    }
    // Plate-specific state gate. This remains active even when the disparity
    // shot gate is rolled back through its environment flag: no temporal RGB
    // may be accumulated or sampled until the uploaded map is from the new
    // shot. Test geometry is explicitly current-by-construction.
    const bool plate_map_gate =
        plate_on && !test_armed_ && reached_cut_pts >= 0.0 &&
        (!(depth_video_ms_ >= 0.0) ||
         depth_video_ms_ < reached_cut_pts - SharedDepthService::kPtsJitterMs);
    if (plate_map_gate)
        cb.temporal_fill = 0;
    if (plate_cut_crossed || shot_gate || plate_map_gate)
        ctx->UpdateSubresource(cb_.Get(), 0, nullptr, &cb, 0, 0);
    // Instrument only the FINAL decision, after the PTS identity and map-state
    // gates. Otherwise a cut-driven purge could be falsely reported as a live
    // old-shot plate merely because the relative advisory was still positive.
    if (depth_service_)
        depth_service_->note_plate_state(
            params_.temporal_fill, plate_on, cb.temporal_fill != 0,
            cut_in_diag);

    // (2b) round 5a — temporal background plate. The accum pass runs once per
    // NEW map (the transport is map-to-map displacement; per-frame reruns
    // would re-apply it), except under a test depth whose transport is the
    // identity and whose tests drive accumulation by present() calls.
    if (plate_on) {
        const int64_t snap = depth_service_
            ? depth_service_->last_snap_steady_ms() : -1;
        if (snap != plate_snap_seen_) {
            plate_snap_seen_ = snap;
            plate_valid_ = false;   // cut/seek: never fill from another shot
        }
        if (!plate_valid_) {
            const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            ctx->ClearRenderTargetView(plateRtv_[0].Get(), zero);
            ctx->ClearRenderTargetView(plateRtv_[1].Get(), zero);
            plate_valid_ = true;
        }
        // Never accumulate THROUGH the cross-shot gate: the fresh map is by
        // definition pre-cut geometry while the frame pixels are post-cut —
        // the mix is old-shot transport over new-shot content. The plate is
        // already purged by the cut-PTS identity above; skipping also keeps it
        // clean until a map from the new shot is available.
        if ((map_refreshed_ && !shot_gate && !plate_map_gate) || test_armed_) {
            const int write = 1 - plate_read_;
            ID3D11ShaderResourceView* plate_in[4] = {
                depthSrv_.Get(), nullptr,
                plateSrv_[plate_read_].Get(), transportSrv_.Get()
            };
            ctx->PSSetShaderResources(3, 4, plate_in);
            ID3D11RenderTargetView* rtv = plateRtv_[write].Get();
            ctx->OMSetRenderTargets(1, &rtv, nullptr);
            set_viewport(
                ctx, static_cast<uint32_t>(grid_width_),
                static_cast<uint32_t>(grid_height_));
            ctx->PSSetShader(psPlateAccum_.Get(), nullptr, 0);
            ctx->Draw(3, 0);
            ctx->OMSetRenderTargets(0, nullptr, nullptr);
            // The freshly written plate becomes the read side for the warp
            // below AND for the next accumulation. Unbind t5 first: it will
            // be rebound as this pass's RT on the next new map.
            ID3D11ShaderResourceView* none[2] = {nullptr, nullptr};
            ctx->PSSetShaderResources(5, 2, none);
            plate_read_ = write;
        }
    }

    // (3) The two image-forming passes below are the exact v5.2.1c shaders.
    // Their six outputs remain the immutable raw stereo reference.
    ID3D11ShaderResourceView* geometry_matte_plate[3] = {
        depthSrv_.Get(), matte_armed_ ? matteSrv_.Get() : nullptr,
        plate_on ? plateSrv_[plate_read_].Get() : nullptr
    };
    ctx->PSSetShaderResources(3, 3, geometry_matte_plate);
    if (params_.depth_view) {
        ID3D11RenderTargetView* rtvs[2] = {
            warpRtv_[0].Get(), warpRtv_[3].Get()
        };
        ctx->OMSetRenderTargets(2, rtvs, nullptr);
        set_viewport(ctx, y_w, y_h);
        ctx->PSSetShader(psViewLuma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    } else {
        ID3D11RenderTargetView* rtvs[2] = {
            warpRtv_[0].Get(), warpRtv_[3].Get()
        };
        ctx->OMSetRenderTargets(2, rtvs, nullptr);
        set_viewport(ctx, y_w, y_h);
        ctx->PSSetShader(psWarpLuma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }
    {
        ID3D11RenderTargetView* rtvs[4] = { warpRtv_[1].Get(), warpRtv_[2].Get(),
                                            warpRtv_[4].Get(), warpRtv_[5].Get() };
        ctx->OMSetRenderTargets(4, rtvs, nullptr);
        set_viewport(ctx, c_w, c_h);
        if (params_.depth_view) {
            ctx->PSSetShader(psViewChroma_.Get(), nullptr, 0);
        } else {
            ctx->PSSetShader(psWarpChroma_.Get(), nullptr, 0);
        }
        ctx->Draw(3, 0);
    }

    // Release the OM stage so the optional final Lab can consume the raw warp.
    ctx->OMSetRenderTargets(0, nullptr, nullptr);

    // Stereo Lab is deliberately downstream and additive: the six warp
    // textures above remain immutable golden/raw outputs. A Lab allocation or
    // shader failure falls back to them for this and every later frame instead
    // of taking synthesis or playback down.
    for (int i = 0; i < kNumOut; ++i)
        out_srv_[i] = warpSrv_[i].Get();
    lab_outputs_valid_ = false;
    if (!params_.depth_view && params_.stereo_lab) {
        if (!stereo_lab_) stereo_lab_ = std::make_unique<StereoLab>();
        ID3D11ShaderResourceView* raw[kNumOut] = {
            warpSrv_[0].Get(), warpSrv_[1].Get(), warpSrv_[2].Get(),
            warpSrv_[3].Get(), warpSrv_[4].Get(), warpSrv_[5].Get()
        };
        std::string lab_err;
        const bool provenance_ready = ensure_provenance_shader(lab_err);
        if (provenance_ready) {
            // Metadata-only sidecar: it re-evaluates the c inverse warp but
            // writes no colour plane. Thus the raw outputs above stay
            // byte-exact while the Lab receives current-frame ownership.
            ID3D11RenderTargetView* provenance_rtvs[2] = {
                provenanceRtv_[0].Get(), provenanceRtv_[1].Get()
            };
            ctx->OMSetRenderTargets(2, provenance_rtvs, nullptr);
            set_viewport(ctx, y_w, y_h);
            ctx->PSSetShader(psProvenance_.Get(), nullptr, 0);
            ctx->Draw(3, 0);
            ctx->OMSetRenderTargets(0, nullptr, nullptr);
        }
        ID3D11ShaderResourceView* provenance[2] = {
            provenance_ready ? provenanceSrv_[0].Get() : nullptr,
            provenance_ready ? provenanceSrv_[1].Get() : nullptr
        };
        lab_outputs_valid_ = stereo_lab_->process(
            device_.Get(), ctx, vs_.Get(), sampler_.Get(), raw, src,
            provenance, y_w, y_h, c_w, c_h, plane_fmt,
            cb.plane_scale, cb.max_disp, cb.comfort_enabled != 0,
            cb.comfort_soft_disp, cb.comfort_hard_disp, true, lab_err);
        if (lab_outputs_valid_) {
            ID3D11ShaderResourceView* const* corrected =
                stereo_lab_->output_srvs();
            for (int i = 0; i < kNumOut; ++i) out_srv_[i] = corrected[i];
        }
    }
    outputs_valid_ = true;
    return true;
}

ID3D11ShaderResourceView* const* Synth3D::output_srvs() const { return out_srv_; }
bool Synth3D::outputs_valid() const { return outputs_valid_; }

bool Synth3D::set_test_depth(const uint16_t* q16_or_null, size_t count) {
    return set_test_geometry(
        q16_or_null, nullptr, nullptr, nullptr, count);
}

bool Synth3D::set_test_geometry(const uint16_t* depth,
                                const uint16_t* owned,
                                const uint16_t* safety,
                                const uint16_t* ownership,
                                size_t count) {
    if (!depth) { test_armed_ = false; return true; }
    const size_t n = static_cast<size_t>(grid_width_) * grid_height_;
    // A map that is not exactly this grid is refused rather than read to n
    // elements (over-read on a smaller buffer) or cropped to n (a silently
    // wrong warp on a larger one). The grid moves with the depth preset, so
    // this mismatch is reachable from a plain caller, not just from a bug.
    if (count != n) return false;
    test_geometry_.resize(SharedDepthService::kGeometryChannels * n);
    for (size_t i = 0; i < n; ++i) {
        const size_t out = SharedDepthService::kGeometryChannels * i;
        const float raw_depth = depth[i] / 65535.0f;
        const float owned_depth = owned ? owned[i] / 65535.0f : raw_depth;
        const float safe = safety ? safety[i] / 65535.0f : 1.0f;
        const float repair = ownership ? ownership[i] / 65535.0f : 0.0f;
        const float effective_depth = raw_depth +
            repair * (owned_depth - raw_depth);
        const float effective_safety = (std::min)(
            1.0f, safe + 0.72f * repair);
        test_geometry_[out + 0] = static_cast<uint16_t>(
            effective_depth * 65535.0f + 0.5f);
        test_geometry_[out + 1] = static_cast<uint16_t>(
            effective_safety * 65535.0f + 0.5f);
        // Identity transport at full reliability: the test bench is a static
        // camera by construction, so the plate transports in place.
        test_geometry_[out + 2] = 32768;
        test_geometry_[out + 3] = 32768;
        test_geometry_[out + 4] = 65535;
        test_geometry_[out + 5] = 0;
    }
    test_armed_  = true;
    depth_dirty_ = true;
    return true;
}

bool Synth3D::set_test_matte(const uint8_t* alpha_or_null,
                             uint32_t width, uint32_t height,
                             size_t count, int mode,
                             const uint8_t* reliability_or_null,
                             size_t reliability_count) {
    if (!alpha_or_null) {
        test_matte_.clear();
        matte_width_ = matte_height_ = 0;
        matte_mode_ = 0;
        matte_armed_ = false;
        matte_dirty_ = false;
        return true;
    }
    if (mode < 1 || mode > 2 || width == 0 || height == 0 ||
        width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION) {
        return false;
    }
    const size_t expected = static_cast<size_t>(width) * height;
    if (count != expected) return false;
    if (reliability_or_null && reliability_count != expected) return false;

    // R keeps network alpha, G the horizontal boundary distance, and B the
    // CURRENT-frame local registration reliability. The latter is deliberately
    // independent of the scalar tracker score: a globally good transport can
    // still have no correspondence in the newly uncovered strip behind a
    // moving actor. A is reserved for a future matte-side diagnostic.
    test_matte_.assign(4 * expected, 0);
    std::vector<uint16_t> distance(width, 255);
    std::vector<uint8_t> effective_alpha(width, 0);
    for (uint32_t row = 0; row < height; ++row) {
        const uint8_t* src = alpha_or_null + static_cast<size_t>(row) * width;
        const size_t row_offset = static_cast<size_t>(row) * width;
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t reliability = reliability_or_null
                ? reliability_or_null[row_offset + x] : 255u;
            effective_alpha[x] = static_cast<uint8_t>(
                (static_cast<uint32_t>(src[x]) * reliability + 127u) / 255u);
        }
        int last = -65536;
        for (uint32_t x = 0; x < width; ++x) {
            const int a = effective_alpha[x];
            const bool fractional = a > 4 && a < 251;
            const bool left_step = x > 0 && std::abs(
                a - static_cast<int>(effective_alpha[x - 1])) > 10;
            const bool right_step = x + 1 < width &&
                std::abs(a - static_cast<int>(effective_alpha[x + 1])) > 10;
            if (fractional || left_step || right_step) last = static_cast<int>(x);
            distance[x] = static_cast<uint16_t>((std::min)(
                255, static_cast<int>(x) - last));
        }
        int next = 65536;
        for (uint32_t rx = width; rx-- > 0;) {
            const int a = effective_alpha[rx];
            const bool fractional = a > 4 && a < 251;
            const bool left_step = rx > 0 &&
                std::abs(a - static_cast<int>(effective_alpha[rx - 1])) > 10;
            const bool right_step = rx + 1 < width &&
                std::abs(a - static_cast<int>(effective_alpha[rx + 1])) > 10;
            if (fractional || left_step || right_step) next = static_cast<int>(rx);
            distance[rx] = static_cast<uint16_t>((std::min)(
                static_cast<int>(distance[rx]),
                (std::min)(255, next - static_cast<int>(rx))));
        }
        for (uint32_t x = 0; x < width; ++x) {
            const size_t source_index = static_cast<size_t>(row) * width + x;
            const size_t out = 4 * source_index;
            test_matte_[out] = src[x];
            test_matte_[out + 1] = static_cast<uint8_t>(distance[x]);
            test_matte_[out + 2] = reliability_or_null
                ? reliability_or_null[source_index] : 255u;
            test_matte_[out + 3] = 0u;
        }
    }
    matte_width_ = width;
    matte_height_ = height;
    matte_mode_ = mode;
    matte_armed_ = true;
    matte_dirty_ = true;
    return true;
}

bool Synth3D::read_plate(ID3D11DeviceContext* ctx, std::vector<uint8_t>& out,
                         uint32_t& w, uint32_t& h, std::string& err) {
    out.clear(); w = 0; h = 0;
    if (!ctx || !device_) { err = "read_plate: synth3d not started"; return false; }
    if (!plateTex_[plate_read_] || !plate_valid_) {
        err = "read_plate: no plate accumulated (temporal_fill off?)";
        return false;
    }
    const uint32_t pw = static_cast<uint32_t>(grid_width_);
    const uint32_t ph = static_cast<uint32_t>(grid_height_);
    if (!plateReadTex_) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width            = pw;
        td.Height           = ph;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = DXGI_FORMAT_R16G16B16A16_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_STAGING;
        td.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &plateReadTex_))) {
            err = "read_plate: CreateTexture2D(staging) failed"; return false;
        }
    }
    ctx->CopyResource(plateReadTex_.Get(), plateTex_[plate_read_].Get());
    D3D11_MAPPED_SUBRESOURCE m = {};
    // DEBUG PATH ONLY: blocking map, same contract as read_plane.
    if (FAILED(ctx->Map(plateReadTex_.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        err = "read_plate: Map failed"; return false;
    }
    out.resize(static_cast<size_t>(pw) * ph * 8u);
    const auto* base = static_cast<const uint8_t*>(m.pData);
    for (uint32_t r = 0; r < ph; ++r)
        std::memcpy(out.data() + static_cast<size_t>(r) * pw * 8u,
                    base + static_cast<size_t>(r) * m.RowPitch, pw * 8u);
    ctx->Unmap(plateReadTex_.Get(), 0);
    w = pw; h = ph;
    return true;
}

bool Synth3D::read_plane(ID3D11DeviceContext* ctx, int slot, std::vector<uint8_t>& out,
                         uint32_t& w, uint32_t& h, uint32_t& bpp, std::string& err) {
    out.clear(); w = 0; h = 0; bpp = 0;
    if (!ctx || !device_) { err = "read_plane: synth3d not started"; return false; }
    if (slot < 0 || slot >= kNumOut || !warpTex_[slot]) { err = "read_plane: bad slot"; return false; }
    const bool isLuma = (slot == 0 || slot == 3);
    const uint32_t pw = isLuma ? y_w_ : c_w_;
    const uint32_t ph = isLuma ? y_h_ : c_h_;
    const uint32_t pb = (plane_fmt_ == DXGI_FORMAT_R16_UNORM) ? 2u : 1u;
    if (!pw || !ph) { err = "read_plane: no output produced yet"; return false; }

    if (!readTex_ || read_w_ != pw || read_h_ != ph || read_fmt_ != plane_fmt_) {
        readTex_.Reset();
        D3D11_TEXTURE2D_DESC td = {};
        td.Width            = pw;
        td.Height           = ph;
        td.MipLevels        = 1;
        td.ArraySize        = 1;
        td.Format           = plane_fmt_;
        td.SampleDesc.Count = 1;
        td.Usage            = D3D11_USAGE_STAGING;
        td.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;
        if (FAILED(device_->CreateTexture2D(&td, nullptr, &readTex_))) {
            err = "read_plane: CreateTexture2D(staging) failed"; return false;
        }
        read_w_ = pw; read_h_ = ph; read_fmt_ = plane_fmt_;
    }

    ID3D11Texture2D* final_texture =
        lab_outputs_valid_ && stereo_lab_
            ? stereo_lab_->output_texture(slot) : warpTex_[slot].Get();
    if (!final_texture) { err = "read_plane: final output unavailable"; return false; }
    ctx->CopyResource(readTex_.Get(), final_texture);
    D3D11_MAPPED_SUBRESOURCE m = {};
    // DEBUG PATH ONLY (tests / probes): a BLOCKING map is acceptable here because this
    // is never called from the playback path — unlike drain_readback's DO_NOT_WAIT.
    if (FAILED(ctx->Map(readTex_.Get(), 0, D3D11_MAP_READ, 0, &m))) {
        err = "read_plane: Map failed"; return false;
    }
    out.resize(static_cast<size_t>(pw) * ph * pb);
    const auto* base = static_cast<const uint8_t*>(m.pData);
    for (uint32_t r = 0; r < ph; ++r)
        std::memcpy(out.data() + static_cast<size_t>(r) * pw * pb,
                    base + static_cast<size_t>(r) * m.RowPitch,
                    static_cast<size_t>(pw) * pb);
    ctx->Unmap(readTex_.Get(), 0);
    w = pw; h = ph; bpp = pb;
    return true;
}

// ---------------------------------------------------------------------------
// Lifecycle + status
// ---------------------------------------------------------------------------
void Synth3D::set_params(const Synth3DParams& p) {
    // Live warp parameters only. A model/runtime change is handled by start()
    // because it selects a different shared service key.
    params_.enabled     = p.enabled;
    params_.strength_pct = p.strength_pct;
    params_.convergence = p.convergence;
    params_.auto_convergence = p.auto_convergence;
    params_.temporal_fill = p.temporal_fill;
    params_.stereo_lab = p.stereo_lab;
    params_.comfort_enabled = p.comfort_enabled;
    params_.comfort_soft_pct = p.comfort_soft_pct;
    params_.comfort_hard_pct = p.comfort_hard_pct;
    params_.depth_view  = p.depth_view;
    params_.diagnostics = p.diagnostics;
    params_.crop_top     = p.crop_top;
    params_.crop_bottom  = p.crop_bottom;
}

bool Synth3D::start(ID3D11Device* dev, const Synth3DParams& p, std::string& err) {
    err.clear();
    local_error_.clear();
    if (!dev) { err = "synth3d start: null device"; return false; }
    if (device_ && device_.Get() != dev) {
        // The renderer was re-initialized on a new device: detach this surface
        // and drop only its device-local resources. The shared model stays warm.
        request_stop();
        release_gpu();
    }
    device_ = dev;

    // The grid is part of the source identity. Explicit rectangular dimensions
    // win; side preserves every square-era caller unchanged.
    const int requested_width =
        (p.grid_width > 0 && p.grid_height > 0)
            ? p.grid_width
            : (p.side > 0 ? p.side : SharedDepthService::kDefaultSide);
    const int requested_height =
        (p.grid_width > 0 && p.grid_height > 0)
            ? p.grid_height
            : requested_width;
    const bool same_source =
        p.model_path == params_.model_path && p.ort_dir == params_.ort_dir &&
        requested_width == grid_width_ && requested_height == grid_height_;
    const bool source_is_live =
        (service_attached_ && depth_service_) ||
        (p.model_path.empty() && !depth_service_);
    if (params_.enabled && same_source && source_is_live) {
        set_params(p);
        return true;
    }

    request_stop();
    params_ = p;

    if (!p.model_path.empty()) {
        depth_service_ = SharedDepthService::acquire_attached(
            p.model_path, p.ort_dir, requested_width, requested_height,
            client_id_);
        service_attached_ = true;
    }
    // An attached service is the authority on the grid (acquire() normalizes
    // it); without one the requested value stands so the test bypass and the
    // status line still describe a definite resolution.
    const int new_width =
        depth_service_ ? depth_service_->width() : requested_width;
    const int new_height =
        depth_service_ ? depth_service_->height() : requested_height;
    if (new_width != grid_width_ || new_height != grid_height_) {
        grid_width_ = new_width;
        grid_height_ = new_height;
        release_depth_grid();
        test_armed_ = false;   // the armed map belongs to the previous grid
    }
    params_.side = grid_width_;
    params_.grid_width = grid_width_;
    params_.grid_height = grid_height_;
    const size_t n = static_cast<size_t>(grid_width_) * grid_height_;
    in_scratch_.assign(3 * n, 0.0f);
    aspect_luma_scratch_.assign(n, 0.0f);
    depth_sequence_ = 0;
    depth_video_ms_ = -1.0;
    depth_valid_ = false;
    depth_dirty_ = test_armed_;
    // An empty path is valid for renderer tests: synth3d_set_test_depth()
    // supplies the map and no ORT service is needed.
    return true;
}

void Synth3D::request_stop() {
    if (depth_service_ && service_attached_)
        SharedDepthService::detach_and_release(depth_service_, client_id_);
    service_attached_ = false;
    // acquire_attached() and service_attached_ are assigned together, so the
    // branch above always moves a non-null service to the reaper.
    outputs_valid_ = false;
    depth_valid_ = false;
    depth_sequence_ = 0;
    depth_video_ms_ = -1.0;
    plate_valid_ = false;
    plate_snap_seen_ = -1;
    plate_cut_seen_ms_ = -1.0;
    lab_outputs_valid_ = false;
    if (stereo_lab_) stereo_lab_->reset_epoch();
    params_.enabled = false;
    local_error_.clear();
}

void Synth3D::join_worker() {
    // Kept for the NativeRenderer lifecycle contract. SharedDepthService owns
    // the worker and its registry reaper owns teardown, so a renderer has
    // nothing to join.
}

void Synth3D::stop() {
    request_stop();
}

void Synth3D::notify_seek() {
    // The service snap arrives only when its worker consumes the first
    // post-seek observation. Purge renderer-local temporal RGB immediately;
    // a seek is a shot discontinuity even before that asynchronous roundtrip.
    plate_valid_ = false;
    plate_cut_seen_ms_ = -1.0;
    lab_outputs_valid_ = false;
    if (stereo_lab_) stereo_lab_->reset_epoch();
    if (depth_service_) depth_service_->notify_seek();
}

void Synth3D::set_lookahead_advisory(double cut_in_ms, double storm_in_ms,
                                     double cut_pts_ms) {
    if (depth_service_)
        depth_service_->set_lookahead_advisory(
            cut_in_ms, storm_in_ms, cut_pts_ms);
}

void Synth3D::set_motion_hints(double pts_ms, double frame_ms,
                               int blocks_w, int blocks_h,
                               int source_width, int source_height,
                               std::vector<int16_t>&& mv_xy,
                               std::vector<uint8_t>&& valid) {
    if (depth_service_)
        depth_service_->note_motion_hints(
            pts_ms, frame_ms, blocks_w, blocks_h,
            source_width, source_height,
            std::move(mv_xy), std::move(valid));
}

void Synth3D::set_ramp_ms(float ramp_ms) {
    ramp_ms_ = ramp_ms;
}

std::string Synth3D::status() const {
    const std::string lab_suffix =
        params_.stereo_lab && stereo_lab_
            ? stereo_lab_->status_suffix()
            : " lab=off lab_mean=0.000 lab_p95=0.000 "
              "lab_asym=0.000 lab_px=0.0 lab_edge_px=0.0 "
              "lab_edge_p95=0.000 comfort_hit_pct=0.0 "
              "comfort_loss_mean_px=0.00 comfort_loss_p95_px=0.00 "
              "pair=off pair_grid=0x0";
    char comfort_buf[160] = {};
    const bool comfort_valid = params_.comfort_enabled &&
        params_.comfort_soft_pct >= 0.0f &&
        params_.comfort_hard_pct > params_.comfort_soft_pct;
    std::snprintf(
        comfort_buf, sizeof(comfort_buf),
        " comfort=%s comfort_soft_pct=%.4f comfort_hard_pct=%.4f",
        comfort_valid ? "calibrated" : "off",
        comfort_valid ? params_.comfort_soft_pct : 0.0f,
        comfort_valid ? params_.comfort_hard_pct : 0.0f);
    const std::string render_mode =
        params_.stereo_lab && lab_outputs_valid_ ? "v521c+lab" : "v521c";
    const std::string final_suffix =
        " render=" + render_mode + lab_suffix + comfort_buf;
    if (!local_error_.empty()) {
        char buf[640] = {};
        std::snprintf(
            buf, sizeof(buf),
            "state=error provider=renderer side=%d fps=0.0 "
            "flow_ms=0.0 infer_ms=0.0 stab_ms=0.0 source_ms=120.0 "
            "update_ms=120.0 age_ms=-1 clients=%d cuts=0 motion=0.000 "
            "alpha=0.000 stable=0.000 history=1.00 scene=0.000 "
            "crop=0:0:0:0 crop_conf=0.00 crop_ready=0 grid=%dx%d "
            "instance=%llu err=%s",
            grid_width_, depth_service_ ? depth_service_->client_count() : 0,
            grid_width_, grid_height_,
            static_cast<unsigned long long>(
                depth_service_ ? depth_service_->instance_id() : 0),
            local_error_.c_str());
        return std::string(buf) + final_suffix;
    }
    // Tap accounting lives in the SHARED service, not here: leader election
    // means only one of the N surfaces ever feeds it, and it is not the one the
    // player polls -- per-client counters read 0 on the surface being read.
    if (depth_service_) return depth_service_->status() + final_suffix;
    if (test_armed_) {
        // The bypass has no service, but it does drive a real grid: report the
        // one its map and textures are sized for. Every other field is the
        // "no worker" constant set, as in the off line below.
        char buf[512] = {};
        std::snprintf(
            buf, sizeof(buf),
            "state=running provider=test side=%d fps=0.0 "
            "flow_ms=0.0 infer_ms=0.0 "
            "stab_ms=0.0 source_ms=120.0 update_ms=120.0 age_ms=0 clients=1 "
            "cuts=0 motion=0.000 alpha=1.000 stable=0.000 history=1.00 "
            "scene=0.000 crop=0:0:0:0 crop_conf=0.00 crop_ready=0 "
            "grid=%dx%d instance=0 err=-",
            grid_width_, grid_width_, grid_height_);
        return std::string(buf) + final_suffix;
    }
    // Byte-identical to NativeRenderer::synth3d_status()'s kOff -- nothing is
    // attached, so there is no grid to report and side= reads 0. Any field
    // added to one of these two lines MUST be added to the other.
    return std::string(
           "state=off provider=none side=0 fps=0.0 flow_ms=0.0 infer_ms=0.0 "
           "stab_ms=0.0 source_ms=120.0 update_ms=120.0 age_ms=-1 clients=0 "
           "cuts=0 motion=0.000 alpha=0.000 stable=0.000 history=1.00 "
           "scene=0.000 crop=0:0:0:0 crop_conf=0.00 crop_ready=0 "
           "grid=0x0 instance=0 err=-") + final_suffix;
}

#endif // SYLC_NATIVE_RENDERER
