// Synth3D — device-local GPU passes backed by the process-wide depth service.
// See include/synth3d.h for the pipeline and the threading contract.
//
// Shader-embed idiom mirrors sbs_nv12_packer.cpp exactly: the HLSL is hand-authored
// in shaders/synth3d.hlsl, embedded into the committed src/synth3d_shaders.h by
// native_renderer/gen_shader_header.py (SINGLE source of truth — re-run the generator
// if the .hlsl changes; do NOT hand-edit the generated header), and compiled at
// RUNTIME by d3dcompiler, once per entry point.
//
// Gated on SYLC_NATIVE_RENDERER like native_renderer.cpp: it is only added to the
// build inside the Windows-only BUILD_NATIVE_RENDERER block.
#ifdef SYLC_NATIVE_RENDERER

#include "synth3d.h"
#include "synth3d_shaders.h"   // sylc::kSynth3dHLSL, generated from shaders/synth3d.hlsl

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3dcompiler.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {

// cbuffer b0 (SynthCB in the HLSL) — 64 bytes (4 x 16). Layout must match the
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
    float _pad[2];         // c2.zw
    float crop_top;        // c3.x
    float crop_bottom;     // c3.y
    float _pad2[2];        // c3.zw
};
static_assert(sizeof(SynthCB) == 64, "SynthCB must be 64 bytes");

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

    auto compile = [&](const char* entry, const char* target, ComPtr<ID3DBlob>& out) -> bool {
        ComPtr<ID3DBlob> eb;
        const UINT cflags = D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_ENABLE_STRICTNESS;
        const HRESULT hr = D3DCompile(sylc::kSynth3dHLSL, std::strlen(sylc::kSynth3dHLSL),
                                      "synth3d", nullptr, nullptr, entry, target,
                                      cflags, 0, &out, &eb);
        if (FAILED(hr)) {
            err = std::string("D3DCompile(") + entry + ") failed";
            if (eb) { err += ": "; err += static_cast<const char*>(eb->GetBufferPointer()); }
            return false;
        }
        return true;
    };
    auto make_ps = [&](const char* entry, ComPtr<ID3D11PixelShader>& out) -> bool {
        ComPtr<ID3DBlob> blob;
        if (!compile(entry, "ps_5_0", blob)) return false;
        if (FAILED(device_->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(),
                                              nullptr, &out))) {
            err = std::string("CreatePixelShader(") + entry + ") failed";
            return false;
        }
        return true;
    };

    ComPtr<ID3DBlob> vsBlob;
    if (!compile("VS_Full", "vs_5_0", vsBlob)) return false;
    if (FAILED(device_->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                           nullptr, &vs_))) {
        err = "CreateVertexShader(VS_Full) failed"; return false;
    }
    if (!make_ps("PS_DepthPrep",      psPrep_))       { vs_.Reset(); return false; }
    if (!make_ps("PS_WarpLuma",       psWarpLuma_))   { vs_.Reset(); return false; }
    if (!make_ps("PS_WarpChroma",     psWarpChroma_)) { vs_.Reset(); return false; }
    if (!make_ps("PS_DepthViewLuma",  psViewLuma_))   { vs_.Reset(); return false; }
    if (!make_ps("PS_DepthViewChroma",psViewChroma_)) { vs_.Reset(); return false; }

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
    td.Format           = DXGI_FORMAT_R16_UNORM;
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
    depth_valid_ = false;   // nothing uploaded yet: process() stays a 2D passthrough
    return true;
}

bool Synth3D::ensure_warp(uint32_t y_w, uint32_t y_h, uint32_t c_w, uint32_t c_h,
                          DXGI_FORMAT fmt, std::string& err) {
    if (warpTex_[0] && y_w_ == y_w && y_h_ == y_h && c_w_ == c_w && c_h_ == c_h &&
        plane_fmt_ == fmt) {
        return true;
    }
    for (int i = 0; i < kNumOut; ++i) {
        warpSrv_[i].Reset(); warpRtv_[i].Reset(); warpTex_[i].Reset();
        out_srv_[i] = nullptr;
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
    prepRtv_.Reset(); prepTex_.Reset();
    stag_write_  = 0;
    seq_ctr_     = 0;
    depth_valid_ = false;
    depth_dirty_ = test_armed_;
}

void Synth3D::release_gpu() {
    for (int i = 0; i < kNumOut; ++i) {
        warpSrv_[i].Reset(); warpRtv_[i].Reset(); warpTex_[i].Reset();
        out_srv_[i] = nullptr;
    }
    release_depth_grid();
    readTex_.Reset(); read_w_ = read_h_ = 0; read_fmt_ = DXGI_FORMAT_UNKNOWN;
    cb_.Reset(); raster_.Reset(); sampler_.Reset();
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
    if (n == 0) return;
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
    if (best < 0) return;   // nothing has landed yet; retry next frame. NEVER blocks.

    // RGBA32F -> CHW float32, ImageNet-normalized. Written into a renderer-owned
    // scratch buffer so the mailbox mutex is held only for an O(1) vector swap.
    const size_t plane = static_cast<size_t>(grid_width_) * grid_height_;
    if (in_scratch_.size() != 3 * plane) in_scratch_.assign(3 * plane, 0.0f);
    float* dst = in_scratch_.data();
    const auto* base = static_cast<const uint8_t*>(m.pData);
    for (int row = 0; row < grid_height_; ++row) {
        const float* s = reinterpret_cast<const float*>(base + static_cast<size_t>(row) * m.RowPitch);
        float* d = dst + static_cast<size_t>(row) * grid_width_;
        for (int x = 0; x < grid_width_; ++x) {
            d[x]               = (s[x * 4 + 0] - kMean[0]) * kInvStd[0];
            d[plane + x]       = (s[x * 4 + 1] - kMean[1]) * kInvStd[1];
            d[2 * plane + x]   = (s[x * 4 + 2] - kMean[2]) * kInvStd[2];
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
        client_id_, in_scratch_, video_time_ms, capture_time,
        static_cast<int>(y_w_), static_cast<int>(y_h_));
}

bool Synth3D::upload_depth(ID3D11DeviceContext* ctx) {
    const uint16_t* q16 = nullptr;
    // Declared at function scope: q16 aliases this map's storage, so the
    // reference must outlive the Map+memcpy loop below. Scoping it to the
    // else-block dropped the last renderer-side reference before the copy;
    // the worker publishing the next map then freed the buffer mid-copy
    // (use-after-free on the GUI thread, ~5x likelier under TensorRT's
    // higher publication rate).
    std::shared_ptr<const SharedDepthService::DepthMap> snapshot;
    if (test_armed_) {
        // The debug bypass wins over the engine: re-upload only when it changed.
        if (!depth_dirty_) return true;
        if (test_depth_.size() !=
            static_cast<size_t>(grid_width_) * grid_height_) return false;
        q16 = test_depth_.data();
    } else {
        if (!depth_service_) return true;
        uint64_t sequence = depth_sequence_;
        snapshot = depth_service_->snapshot(depth_sequence_, sequence);
        if (!snapshot) return true;                // keep the local GPU texture
        if (snapshot->size() !=
            static_cast<size_t>(grid_width_) * grid_height_) return false;
        q16 = snapshot->data();
        depth_sequence_ = sequence;
    }

    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(depthTex_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return false;
    auto* out = static_cast<uint8_t*>(m.pData);
    for (int row = 0; row < grid_height_; ++row)
        std::memcpy(out + static_cast<size_t>(row) * m.RowPitch,
                    q16 + static_cast<size_t>(row) * grid_width_,
                    static_cast<size_t>(grid_width_) * sizeof(uint16_t));
    ctx->Unmap(depthTex_.Get(), 0);
    depth_valid_ = true;
    depth_dirty_ = false;
    return true;
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
    if (!ctx || !src || !src[0] || !src[1] || !src[2]) return false;
    if (!y_w || !y_h || !c_w || !c_h) return false;

    std::string err;
    if (!ensure_pipeline(err)) return false;
    if (!ensure_prep(err))     return false;
    if (!ensure_depth(err))    return false;
    if (!ensure_warp(y_w, y_h, c_w, c_h, plane_fmt, err)) return false;

    // The six warp textures were bound as PS resources by the PREVIOUS present()
    // (and by cast_encode's packer); binding them as render targets now would be a
    // read/write hazard. Unbind t0..t6 — present() re-binds all seven itself.
    ID3D11ShaderResourceView* const kNoSrv[7] = {};
    ctx->PSSetShaderResources(0, 7, kNoSrv);

    SynthCB cb = {};
    cb.max_disp       = params_.strength_pct * 0.01f;   // % of width -> fraction
    cb.convergence    = params_.convergence;
    cb.plane_scale    = (plane_scale > 0.0f) ? plane_scale : 1.0f;
    cb.inv_w          = 1.0f / static_cast<float>(y_w);
    cb.yuv_matrix_sel = matrix_sel;
    cb.transfer_sel   = transfer_sel;
    cb.diagnostics    = params_.diagnostics ? 1 : 0;
    cb.edge_strength  = 28.0f;
    cb.depth_texel[0] = 1.0f / static_cast<float>(grid_width_);
    cb.depth_texel[1] = 1.0f / static_cast<float>(grid_height_);
    cb.crop_top = (std::max)(0.0f, (std::min)(0.45f, params_.crop_top));
    cb.crop_bottom = (std::max)(0.0f, (std::min)(0.45f, params_.crop_bottom));
    if (cb.crop_top + cb.crop_bottom > 0.55f) {
        cb.crop_top = 0.0f;
        cb.crop_bottom = 0.0f;
    }

    // Post-cut/seek ease-out ramp: a fresh snap teleports the stereo geometry
    // in one frame (100-300ms for the eyes to re-fuse), which reads as a jolt.
    // Scale the disparity budget down right after the snap and let it inflate
    // back to full over ramp_ms_. The test-depth bypass keeps its full
    // geometry unconditionally -- golden/warp tests assert on it directly.
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
        depth_service_->wants_input(client_id_)) {
        ID3D11RenderTargetView* rtv = prepRtv_.Get();
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        set_viewport(
            ctx, static_cast<uint32_t>(grid_width_),
            static_cast<uint32_t>(grid_height_));
        ctx->PSSetShader(psPrep_.Get(), nullptr, 0);
        ctx->Draw(3, 0);                 // the draw covers the whole RT -> no clear
        push_readback(ctx, video_time_ms);
        drain_readback(ctx);
    }

    // (2) publish the freshest nearness map (test bypass wins over the engine).
    upload_depth(ctx);

    if (!depth_valid_) {
        // No depth yet (engine still loading, no test depth): leave the caller on the
        // untouched source SRVs. Playback continues in 2D — it never waits for depth.
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        return false;
    }

    // (3) luma pass (2 MRT: Y_L,Y_R) then chroma pass (4 MRT: U_L,V_L,U_R,V_R).
    ID3D11ShaderResourceView* depth = depthSrv_.Get();
    ctx->PSSetShaderResources(3, 1, &depth);
    {
        ID3D11RenderTargetView* rtvs[2] = { warpRtv_[0].Get(), warpRtv_[3].Get() };
        ctx->OMSetRenderTargets(2, rtvs, nullptr);
        set_viewport(ctx, y_w, y_h);
        ctx->PSSetShader(params_.depth_view ? psViewLuma_.Get() : psWarpLuma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }
    {
        ID3D11RenderTargetView* rtvs[4] = { warpRtv_[1].Get(), warpRtv_[2].Get(),
                                            warpRtv_[4].Get(), warpRtv_[5].Get() };
        ctx->OMSetRenderTargets(4, rtvs, nullptr);
        set_viewport(ctx, c_w, c_h);
        ctx->PSSetShader(params_.depth_view ? psViewChroma_.Get() : psWarpChroma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }

    // Release the OM stage so present()'s draw can bind these same textures as SRVs.
    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    outputs_valid_ = true;
    return true;
}

ID3D11ShaderResourceView* const* Synth3D::output_srvs() const { return out_srv_; }
bool Synth3D::outputs_valid() const { return outputs_valid_; }

bool Synth3D::set_test_depth(const uint16_t* q16_or_null, size_t count) {
    if (!q16_or_null) { test_armed_ = false; return true; }
    const size_t n = static_cast<size_t>(grid_width_) * grid_height_;
    // A map that is not exactly this grid is refused rather than read to n
    // elements (over-read on a smaller buffer) or cropped to n (a silently
    // wrong warp on a larger one). The grid moves with the depth preset, so
    // this mismatch is reachable from a plain caller, not just from a bug.
    if (count != n) return false;
    test_depth_.assign(q16_or_null, q16_or_null + n);
    test_armed_  = true;
    depth_dirty_ = true;
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

    ctx->CopyResource(readTex_.Get(), warpTex_[slot].Get());
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
    params_.depth_view  = p.depth_view;
    params_.diagnostics = p.diagnostics;
    params_.crop_top     = p.crop_top;
    params_.crop_bottom  = p.crop_bottom;
}

bool Synth3D::start(ID3D11Device* dev, const Synth3DParams& p, std::string& err) {
    err.clear();
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
        depth_service_ = SharedDepthService::acquire(
            p.model_path, p.ort_dir, requested_width, requested_height);
        depth_service_->attach(client_id_);
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
    depth_sequence_ = 0;
    depth_valid_ = false;
    depth_dirty_ = test_armed_;
    // An empty path is valid for renderer tests: synth3d_set_test_depth()
    // supplies the map and no ORT service is needed.
    return true;
}

void Synth3D::request_stop() {
    if (depth_service_ && service_attached_)
        depth_service_->detach(client_id_);
    service_attached_ = false;
    depth_service_.reset();  // registry owns the warm session; this never joins
    outputs_valid_ = false;
    depth_valid_ = false;
    depth_sequence_ = 0;
    params_.enabled = false;
}

void Synth3D::join_worker() {
    // Kept for the NativeRenderer lifecycle contract. SharedDepthService owns
    // and process-caches the worker, so a renderer has nothing to join.
}

void Synth3D::stop() {
    request_stop();
}

void Synth3D::notify_seek() {
    if (depth_service_) depth_service_->notify_seek();
}

void Synth3D::set_ramp_ms(float ramp_ms) {
    ramp_ms_ = ramp_ms;
}

std::string Synth3D::status() const {
    if (depth_service_) return depth_service_->status();
    if (test_armed_) {
        // The bypass has no service, but it does drive a real grid: report the
        // one its map and textures are sized for. Every other field is the
        // "no worker" constant set, as in the off line below.
        char buf[256] = {};
        std::snprintf(
            buf, sizeof(buf),
            "state=running provider=test side=%d fps=0.0 "
            "flow_ms=0.0 infer_ms=0.0 "
            "stab_ms=0.0 source_ms=120.0 update_ms=120.0 age_ms=0 clients=1 "
            "cuts=0 motion=0.000 alpha=1.000 stable=0.000 history=1.00 "
            "scene=0.000 crop=0:0:0:0 crop_conf=0.00 grid=%dx%d err=-",
            grid_width_, grid_width_, grid_height_);
        return buf;
    }
    // Byte-identical to NativeRenderer::synth3d_status()'s kOff -- nothing is
    // attached, so there is no grid to report and side= reads 0. Any field
    // added to one of these two lines MUST be added to the other.
    return "state=off provider=none side=0 fps=0.0 flow_ms=0.0 infer_ms=0.0 "
           "stab_ms=0.0 source_ms=120.0 update_ms=120.0 age_ms=-1 clients=0 "
           "cuts=0 motion=0.000 alpha=0.000 stable=0.000 history=1.00 "
           "scene=0.000 crop=0:0:0:0 crop_conf=0.00 grid=0x0 err=-";
}

#endif // SYLC_NATIVE_RENDERER
