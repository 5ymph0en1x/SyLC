// SyLC Stereo Lab — additive final binocular-coherence pass.
#ifdef SYLC_NATIVE_RENDERER

#include "shader_resource_ids.h"
#include "shader_resources.h"
#include "stereo_lab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kMetricWidth = 96;
constexpr uint32_t kMetricHeight = 54;
constexpr uint64_t kMetricInterval = 15;
constexpr uint32_t kPairFieldMaxWidth = 192;

struct LabCB {
    float inv_w;
    float inv_h;
    float plane_scale;
    float lab_gain;
    float max_disp;
    float comfort_soft_disp;
    float comfort_hard_disp;
    float comfort_enabled;
    float pair_inv_w;
    float pair_inv_h;
    float pair_field_enabled;
    float pair_padding;
};
static_assert(sizeof(LabCB) == 48, "LabCB must be three D3D constant registers");

void set_viewport(ID3D11DeviceContext* ctx, uint32_t width, uint32_t height) {
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(width);
    vp.Height = static_cast<float>(height);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    ctx->RSSetViewports(1, &vp);
}

}  // namespace

bool StereoLab::environment_enabled() {
    static const bool enabled = []() {
        char* env = nullptr;
        size_t len = 0;
        bool value = true;
        if (_dupenv_s(&env, &len, "SYLC_STEREO_LAB") == 0 && env) {
            value = env[0] != '0';
            free(env);
        }
        return value;
    }();
    return enabled;
}

float StereoLab::environment_gain() {
    static const float gain = []() {
        char* env = nullptr;
        size_t len = 0;
        float value = 0.72f;
        if (_dupenv_s(&env, &len, "SYLC_STEREO_LAB_GAIN") == 0 && env) {
            char* end = nullptr;
            const float parsed = std::strtof(env, &end);
            if (end != env && std::isfinite(parsed))
                value = (std::max)(0.0f, (std::min)(1.0f, parsed));
            free(env);
        }
        return value;
    }();
    return gain;
}

bool StereoLab::ensure_pipeline(ID3D11Device* device, std::string& err) {
    if (ps_pair_field_ && ps_luma_) return true;
    if (!device) { err = "stereo_lab: null device"; return false; }

    auto make_ps = [&](int resource_id, const char* entry,
                       Microsoft::WRL::ComPtr<ID3D11PixelShader>& out) {
        sylc::ShaderBytecode bytecode;
        if (!sylc::load_shader_bytecode(resource_id, bytecode, err)) return false;
        if (FAILED(device->CreatePixelShader(
                bytecode.data, bytecode.size, nullptr, &out))) {
            err = std::string("stereo_lab: CreatePixelShader(") + entry + ") failed";
            return false;
        }
        return true;
    };

    if (!make_ps(IDR_SYLC_STEREO_LAB_PS_PAIR_FIELD,
                 "PS_PairField", ps_pair_field_) ||
        !make_ps(IDR_SYLC_STEREO_LAB_PS_LUMA,
                 "PS_LabLuma", ps_luma_) ||
        !make_ps(IDR_SYLC_STEREO_LAB_PS_CHROMA,
                 "PS_LabChroma", ps_chroma_) ||
        !make_ps(IDR_SYLC_STEREO_LAB_PS_METRICS,
                 "PS_LabMetrics", ps_metrics_)) {
        return false;
    }

    D3D11_BUFFER_DESC cbd = {};
    cbd.ByteWidth = sizeof(LabCB);
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(device->CreateBuffer(&cbd, nullptr, &cb_))) {
        err = "stereo_lab: CreateBuffer(LabCB) failed";
        return false;
    }
    return true;
}

bool StereoLab::ensure_resources(ID3D11Device* device,
                                 uint32_t y_w, uint32_t y_h,
                                 uint32_t c_w, uint32_t c_h,
                                 DXGI_FORMAT plane_fmt, std::string& err) {
    if (out_tex_[0] && guard_tex_ && pair_tex_ &&
        y_w_ == y_w && y_h_ == y_h &&
        c_w_ == c_w && c_h_ == c_h && plane_fmt_ == plane_fmt) {
        return true;
    }

    for (int i = 0; i < kNumOut; ++i) {
        out_srv_[i] = nullptr;
        out_srv_owned_[i].Reset();
        out_rtv_[i].Reset();
        out_tex_[i].Reset();
    }
    guard_srv_.Reset(); guard_rtv_.Reset(); guard_tex_.Reset();
    pair_srv_.Reset(); pair_rtv_.Reset(); pair_tex_.Reset();
    metric_rtv_.Reset(); metric_tex_.Reset(); metric_read_tex_.Reset();
    metric_pending_ = false;
    outputs_valid_ = false;

    for (int i = 0; i < kNumOut; ++i) {
        const bool luma = i == 0 || i == 3;
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = luma ? y_w : c_w;
        td.Height = luma ? y_h : c_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = plane_fmt;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        if (FAILED(device->CreateTexture2D(&td, nullptr, &out_tex_[i])) ||
            FAILED(device->CreateRenderTargetView(
                out_tex_[i].Get(), nullptr, &out_rtv_[i])) ||
            FAILED(device->CreateShaderResourceView(
                out_tex_[i].Get(), nullptr, &out_srv_owned_[i]))) {
            err = "stereo_lab: corrected plane texture/view creation failed";
            return false;
        }
        out_srv_[i] = out_srv_owned_[i].Get();
    }

    D3D11_TEXTURE2D_DESC guard_desc = {};
    guard_desc.Width = y_w;
    guard_desc.Height = y_h;
    guard_desc.MipLevels = 1;
    guard_desc.ArraySize = 1;
    guard_desc.Format = DXGI_FORMAT_R8G8_UNORM;
    guard_desc.SampleDesc.Count = 1;
    guard_desc.Usage = D3D11_USAGE_DEFAULT;
    guard_desc.BindFlags = D3D11_BIND_RENDER_TARGET |
                           D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(device->CreateTexture2D(&guard_desc, nullptr, &guard_tex_)) ||
        FAILED(device->CreateRenderTargetView(
            guard_tex_.Get(), nullptr, &guard_rtv_)) ||
        FAILED(device->CreateShaderResourceView(
            guard_tex_.Get(), nullptr, &guard_srv_))) {
        err = "stereo_lab: guard map texture/view creation failed";
        return false;
    }

    pair_w_ = (std::max)(1u, (std::min)(kPairFieldMaxWidth, y_w));
    pair_h_ = (std::max)(1u, static_cast<uint32_t>(
        (static_cast<uint64_t>(y_h) * pair_w_ + y_w / 2u) /
        (std::max)(1u, y_w)));
    D3D11_TEXTURE2D_DESC pair_desc = guard_desc;
    pair_desc.Width = pair_w_;
    pair_desc.Height = pair_h_;
    pair_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (FAILED(device->CreateTexture2D(&pair_desc, nullptr, &pair_tex_)) ||
        FAILED(device->CreateRenderTargetView(
            pair_tex_.Get(), nullptr, &pair_rtv_)) ||
        FAILED(device->CreateShaderResourceView(
            pair_tex_.Get(), nullptr, &pair_srv_))) {
        err = "stereo_lab: sparse pair field texture/view creation failed";
        return false;
    }

    D3D11_TEXTURE2D_DESC metric_desc = guard_desc;
    metric_desc.Width = kMetricWidth;
    metric_desc.Height = kMetricHeight;
    metric_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    metric_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    if (FAILED(device->CreateTexture2D(
            &metric_desc, nullptr, &metric_tex_)) ||
        FAILED(device->CreateRenderTargetView(
            metric_tex_.Get(), nullptr, &metric_rtv_))) {
        err = "stereo_lab: metric texture/view creation failed";
        return false;
    }
    metric_desc.Usage = D3D11_USAGE_STAGING;
    metric_desc.BindFlags = 0;
    metric_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(device->CreateTexture2D(
            &metric_desc, nullptr, &metric_read_tex_))) {
        err = "stereo_lab: metric staging creation failed";
        return false;
    }

    y_w_ = y_w; y_h_ = y_h; c_w_ = c_w; c_h_ = c_h;
    plane_fmt_ = plane_fmt;
    reset_epoch();
    return true;
}

void StereoLab::drain_metrics(ID3D11DeviceContext* ctx) {
    if (!metric_pending_ || !metric_read_tex_) return;
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    const HRESULT hr = ctx->Map(metric_read_tex_.Get(), 0, D3D11_MAP_READ,
                                D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) return;
    if (FAILED(hr)) {
        metric_pending_ = false;
        return;
    }

    const size_t count = static_cast<size_t>(kMetricWidth) * kMetricHeight;
    std::vector<uint8_t> samples(count * 4);
    for (uint32_t y = 0; y < kMetricHeight; ++y) {
        const auto* row = static_cast<const uint8_t*>(mapped.pData) +
                          static_cast<size_t>(y) * mapped.RowPitch;
        std::memcpy(samples.data() + static_cast<size_t>(y) * kMetricWidth * 4,
                    row, static_cast<size_t>(kMetricWidth) * 4);
    }
    ctx->Unmap(metric_read_tex_.Get(), 0);
    metric_pending_ = false;

    if (pending_epoch_ != epoch_) return;
    const StereoLabMetricSummary summary =
        reduce_metric_rgba8(samples.data(), count,
                            pending_comfort_scale_px_);
    metric_mean_ = summary.mean;
    metric_p95_ = summary.p95_active;
    metric_asym_ = summary.asym;
    metric_coverage_ = summary.coverage;
    comfort_mean_loss_px_ = summary.comfort_mean_loss_px;
    comfort_p95_loss_px_ = summary.comfort_p95_loss_px;
    comfort_coverage_ = summary.comfort_coverage;
    edge_veto_p95_ = summary.edge_veto_p95;
    edge_veto_coverage_ = summary.edge_veto_coverage;
}

StereoLabMetricSummary StereoLab::reduce_metric_rg8(
        const uint8_t* rg, size_t sample_count) {
    StereoLabMetricSummary result;
    if (!rg || sample_count == 0) return result;

    double sum = 0.0;
    double asym = 0.0;
    std::vector<uint8_t> active_peaks;
    active_peaks.reserve(sample_count / 8);
    for (size_t i = 0; i < sample_count; ++i) {
        const uint8_t left = rg[2 * i];
        const uint8_t right = rg[2 * i + 1];
        const uint8_t peak = (std::max)(left, right);
        sum += static_cast<double>(left + right) / (2.0 * 255.0);
        asym += std::abs(static_cast<int>(left) - static_cast<int>(right)) /
                255.0;
        // Coverage and intensity deliberately share the same definition of
        // "active".  A sparse 3% correction must report its own intensity,
        // not the 95th percentile of the 97% untouched background.
        if (peak >= 13) active_peaks.push_back(peak);  // >= 5% convergence
    }

    result.mean = static_cast<float>(sum / sample_count);
    result.asym = static_cast<float>(asym / sample_count);
    result.coverage = 100.0f * static_cast<float>(active_peaks.size()) /
                      static_cast<float>(sample_count);
    if (!active_peaks.empty()) {
        const size_t p95_index = static_cast<size_t>(
            0.95 * static_cast<double>(active_peaks.size() - 1));
        std::nth_element(active_peaks.begin(),
                         active_peaks.begin() + p95_index,
                         active_peaks.end());
        result.p95_active = active_peaks[p95_index] / 255.0f;
    }
    return result;
}

StereoLabMetricSummary StereoLab::reduce_metric_rgba8(
        const uint8_t* rgba, size_t sample_count,
        float comfort_scale_px) {
    StereoLabMetricSummary result;
    if (!rgba || sample_count == 0) return result;

    double guard_sum = 0.0;
    double guard_asym = 0.0;
    double comfort_sum = 0.0;
    std::vector<uint8_t> guard_active;
    std::vector<uint8_t> comfort_active;
    std::vector<uint8_t> edge_active;
    guard_active.reserve(sample_count / 8);
    comfort_active.reserve(sample_count / 8);
    edge_active.reserve(sample_count / 16);

    for (size_t i = 0; i < sample_count; ++i) {
        const uint8_t left = rgba[4 * i];
        const uint8_t right = rgba[4 * i + 1];
        const uint8_t comfort = rgba[4 * i + 2];
        const uint8_t edge = rgba[4 * i + 3];
        const uint8_t peak = (std::max)(left, right);
        guard_sum += static_cast<double>(left + right) / (2.0 * 255.0);
        guard_asym += std::abs(static_cast<int>(left) -
                               static_cast<int>(right)) / 255.0;
        comfort_sum += comfort / 255.0;
        if (peak >= 13) guard_active.push_back(peak);
        // One UNORM step is below a quarter pixel even at the maximum 3%
        // Full-HD strength: sensitive enough without counting float noise.
        if (comfort > 0) comfort_active.push_back(comfort);
        if (edge >= 13) edge_active.push_back(edge);
    }

    result.mean = static_cast<float>(guard_sum / sample_count);
    result.asym = static_cast<float>(guard_asym / sample_count);
    result.coverage = 100.0f * static_cast<float>(guard_active.size()) /
                      static_cast<float>(sample_count);
    result.comfort_mean_loss_px = static_cast<float>(
        comfort_sum / sample_count) * comfort_scale_px;
    result.comfort_coverage =
        100.0f * static_cast<float>(comfort_active.size()) /
        static_cast<float>(sample_count);
    result.edge_veto_coverage =
        100.0f * static_cast<float>(edge_active.size()) /
        static_cast<float>(sample_count);

    auto active_p95 = [](std::vector<uint8_t>& values) {
        if (values.empty()) return 0.0f;
        const size_t index = static_cast<size_t>(
            0.95 * static_cast<double>(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + index, values.end());
        return values[index] / 255.0f;
    };
    result.p95_active = active_p95(guard_active);
    result.comfort_p95_loss_px = active_p95(comfort_active) *
                                 comfort_scale_px;
    result.edge_veto_p95 = active_p95(edge_active);
    return result;
}

void StereoLab::sample_metrics(ID3D11DeviceContext* ctx) {
    if (!metric_tex_ || !metric_read_tex_ || metric_pending_) return;
    ctx->CopyResource(metric_read_tex_.Get(), metric_tex_.Get());
    pending_epoch_ = epoch_;
    pending_comfort_scale_px_ = comfort_scale_px_;
    metric_pending_ = true;
}

bool StereoLab::process(ID3D11Device* device, ID3D11DeviceContext* ctx,
                        ID3D11VertexShader* fullscreen_vs,
                        ID3D11SamplerState* linear_sampler,
                        ID3D11ShaderResourceView* const raw[6],
                        ID3D11ShaderResourceView* const source[3],
                        ID3D11ShaderResourceView* const provenance[2],
                        uint32_t y_w, uint32_t y_h,
                        uint32_t c_w, uint32_t c_h,
                        DXGI_FORMAT plane_fmt, float plane_scale,
                        float max_disp, bool comfort_enabled,
                        float comfort_soft_disp, float comfort_hard_disp,
                        bool requested, std::string& err) {
    outputs_valid_ = false;
    err.clear();
    if (!requested || !environment_enabled() || environment_gain() <= 0.0f) {
        mode_ = "off";
        return false;
    }
    // T0/seek and a user-selected zero strength remain the established raw
    // path, avoiding a redundant resample and preserving exact cut identity.
    if (std::abs(max_disp) < 0.5f / static_cast<float>((std::max)(1u, y_w))) {
        mode_ = "identity";
        metric_mean_ = metric_p95_ = metric_asym_ = metric_coverage_ = 0.0f;
        comfort_mean_loss_px_ = comfort_p95_loss_px_ =
            comfort_coverage_ = edge_veto_p95_ = edge_veto_coverage_ = 0.0f;
        return false;
    }
    if (failed_) {
        mode_ = "bypass";
        err = error_;
        return false;
    }
    if (!device || !ctx || !fullscreen_vs || !linear_sampler ||
        !raw || !source || !provenance || !raw[0] || !raw[5] ||
        !source[0] || !source[1] || !source[2] ||
        !provenance[0] || !provenance[1]) {
        mode_ = "bypass";
        error_ = err = "stereo_lab: incomplete process inputs";
        failed_ = true;
        return false;
    }
    if (device_ && device_.Get() != device) release();
    device_ = device;
    if (!ensure_pipeline(device, err) ||
        !ensure_resources(device, y_w, y_h, c_w, c_h, plane_fmt, err)) {
        error_ = err;
        failed_ = true;
        mode_ = "bypass";
        return false;
    }

    LabCB cb = {};
    cb.inv_w = 1.0f / static_cast<float>(y_w);
    cb.inv_h = 1.0f / static_cast<float>(y_h);
    cb.plane_scale = plane_scale > 0.0f ? plane_scale : 1.0f;
    cb.lab_gain = environment_gain();
    cb.max_disp = std::abs(max_disp);
    cb.comfort_soft_disp = comfort_soft_disp;
    cb.comfort_hard_disp = comfort_hard_disp;
    cb.comfort_enabled = comfort_enabled ? 1.0f : 0.0f;
    cb.pair_inv_w = 1.0f / static_cast<float>(pair_w_);
    cb.pair_inv_h = 1.0f / static_cast<float>(pair_h_);
    cb.pair_field_enabled = 1.0f;
    comfort_scale_px_ = cb.max_disp * static_cast<float>(y_w);
    ctx->UpdateSubresource(cb_.Get(), 0, nullptr, &cb, 0, 0);

    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(fullscreen_vs, nullptr, 0);
    ctx->PSSetSamplers(0, 1, &linear_sampler);
    ID3D11Buffer* cbuffer = cb_.Get();
    ctx->PSSetConstantBuffers(0, 1, &cbuffer);

    ID3D11ShaderResourceView* inputs[13] = {
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
        provenance[0], provenance[1], source[0], source[1], source[2],
        nullptr, nullptr
    };
    ctx->PSSetShaderResources(0, 13, inputs);
    {
        // Rebuild the cyclopean topology from this frame only.  It is a
        // source-space active set, never a history surface, so a cut cannot
        // carry pair state into the following shot.
        ID3D11RenderTargetView* rtv = pair_rtv_.Get();
        ctx->OMSetRenderTargets(1, &rtv, nullptr);
        set_viewport(ctx, pair_w_, pair_h_);
        ctx->PSSetShader(ps_pair_field_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
        ctx->OMSetRenderTargets(0, nullptr, nullptr);
        ID3D11ShaderResourceView* pair = pair_srv_.Get();
        ctx->PSSetShaderResources(12, 1, &pair);
    }
    {
        ID3D11RenderTargetView* rtvs[3] = {
            out_rtv_[0].Get(), out_rtv_[3].Get(), guard_rtv_.Get()
        };
        ctx->OMSetRenderTargets(3, rtvs, nullptr);
        set_viewport(ctx, y_w, y_h);
        ctx->PSSetShader(ps_luma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }
    {
        ID3D11RenderTargetView* rtvs[4] = {
            out_rtv_[1].Get(), out_rtv_[2].Get(),
            out_rtv_[4].Get(), out_rtv_[5].Get()
        };
        ctx->OMSetRenderTargets(4, rtvs, nullptr);
        set_viewport(ctx, c_w, c_h);
        ID3D11ShaderResourceView* guard = guard_srv_.Get();
        ctx->PSSetShaderResources(11, 1, &guard);
        ctx->PSSetShader(ps_chroma_.Get(), nullptr, 0);
        ctx->Draw(3, 0);
    }

    ++frame_counter_;
    if (frame_counter_ % kMetricInterval == 0) {
        drain_metrics(ctx);
        if (!metric_pending_) {
            ID3D11RenderTargetView* rtv = metric_rtv_.Get();
            ctx->OMSetRenderTargets(1, &rtv, nullptr);
            set_viewport(ctx, kMetricWidth, kMetricHeight);
            ctx->PSSetShader(ps_metrics_.Get(), nullptr, 0);
            ctx->Draw(3, 0);
            ctx->OMSetRenderTargets(0, nullptr, nullptr);
            sample_metrics(ctx);
        }
    }

    ctx->OMSetRenderTargets(0, nullptr, nullptr);
    ID3D11ShaderResourceView* const none[13] = {};
    ctx->PSSetShaderResources(0, 13, none);
    outputs_valid_ = true;
    mode_ = "active";
    error_.clear();
    return true;
}

ID3D11Texture2D* StereoLab::output_texture(int slot) const {
    return slot >= 0 && slot < kNumOut ? out_tex_[slot].Get() : nullptr;
}

void StereoLab::reset_epoch() {
    ++epoch_;
    metric_mean_ = metric_p95_ = metric_asym_ = metric_coverage_ = 0.0f;
    comfort_mean_loss_px_ = comfort_p95_loss_px_ = comfort_coverage_ = 0.0f;
    edge_veto_p95_ = edge_veto_coverage_ = 0.0f;
}

void StereoLab::release() {
    for (int i = 0; i < kNumOut; ++i) {
        out_srv_[i] = nullptr;
        out_srv_owned_[i].Reset(); out_rtv_[i].Reset(); out_tex_[i].Reset();
    }
    guard_srv_.Reset(); guard_rtv_.Reset(); guard_tex_.Reset();
    pair_srv_.Reset(); pair_rtv_.Reset(); pair_tex_.Reset();
    metric_rtv_.Reset(); metric_tex_.Reset(); metric_read_tex_.Reset();
    cb_.Reset(); ps_metrics_.Reset(); ps_chroma_.Reset(); ps_luma_.Reset();
    ps_pair_field_.Reset();
    device_.Reset();
    y_w_ = y_h_ = c_w_ = c_h_ = 0;
    pair_w_ = pair_h_ = 0;
    plane_fmt_ = DXGI_FORMAT_UNKNOWN;
    metric_pending_ = false;
    outputs_valid_ = false;
    failed_ = false;
    error_.clear();
    mode_ = "off";
    reset_epoch();
}

std::string StereoLab::status_suffix() const {
    char buf[448] = {};
    const bool pair_active = mode_ == "active" && pair_tex_;
    std::snprintf(buf, sizeof(buf),
                  " lab=%s lab_mean=%.3f lab_p95=%.3f "
                  "lab_asym=%.3f lab_px=%.1f "
                  "lab_edge_px=%.1f lab_edge_p95=%.3f "
                  "comfort_hit_pct=%.1f comfort_loss_mean_px=%.2f "
                  "comfort_loss_p95_px=%.2f pair=%s pair_grid=%ux%u",
                  mode_.c_str(), metric_mean_, metric_p95_,
                  metric_asym_, metric_coverage_,
                  edge_veto_coverage_, edge_veto_p95_,
                  comfort_coverage_, comfort_mean_loss_px_,
                  comfort_p95_loss_px_,
                  pair_active ? "sparse-source" : "off",
                  pair_active ? pair_w_ : 0u,
                  pair_active ? pair_h_ : 0u);
    return buf;
}

#endif  // SYLC_NATIVE_RENDERER
