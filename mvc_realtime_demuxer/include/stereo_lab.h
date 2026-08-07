// SyLC Stereo Lab: device-local, post-warp binocular coherence guard.
// The established Synth3D textures are immutable inputs; corrected textures
// are separate and can be bypassed without rebuilding or changing the warp.
#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstddef>
#include <string>

struct StereoLabMetricSummary {
    float mean = 0.0f;        // whole-frame correction energy
    float p95_active = 0.0f;  // intensity percentile, active population only
    float asym = 0.0f;        // whole-frame left/right guard imbalance
    float coverage = 0.0f;    // percentage of samples at >= 5% correction
    float comfort_mean_loss_px = 0.0f;
    float comfort_p95_loss_px = 0.0f;
    float comfort_coverage = 0.0f;
    float edge_veto_p95 = 0.0f;
    float edge_veto_coverage = 0.0f;
};

class StereoLab {
public:
    static constexpr int kNumOut = 6;

    StereoLab() = default;
    ~StereoLab() = default;
    StereoLab(const StereoLab&) = delete;
    StereoLab& operator=(const StereoLab&) = delete;

    bool process(ID3D11Device* device, ID3D11DeviceContext* ctx,
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
                 bool requested, std::string& err);

    ID3D11ShaderResourceView* const* output_srvs() const { return out_srv_; }
    ID3D11Texture2D* output_texture(int slot) const;
    bool outputs_valid() const { return outputs_valid_; }
    void reset_epoch();
    void release();
    std::string status_suffix() const;

    // Shared by the asynchronous readback and its deterministic regression
    // test. ``rg`` contains one uint8 left/right guard pair per sample.
    static StereoLabMetricSummary reduce_metric_rg8(
        const uint8_t* rg, size_t sample_count);
    static StereoLabMetricSummary reduce_metric_rgba8(
        const uint8_t* rgba, size_t sample_count,
        float comfort_scale_px);

private:
    template <typename T> using CP = Microsoft::WRL::ComPtr<T>;

    bool ensure_pipeline(ID3D11Device* device, std::string& err);
    bool ensure_resources(ID3D11Device* device,
                          uint32_t y_w, uint32_t y_h,
                          uint32_t c_w, uint32_t c_h,
                          DXGI_FORMAT plane_fmt, std::string& err);
    void sample_metrics(ID3D11DeviceContext* ctx);
    void drain_metrics(ID3D11DeviceContext* ctx);
    static bool environment_enabled();
    static float environment_gain();

    CP<ID3D11Device> device_;
    CP<ID3D11PixelShader> ps_pair_field_, ps_luma_, ps_chroma_, ps_metrics_;
    CP<ID3D11Buffer> cb_;
    CP<ID3D11Texture2D> out_tex_[kNumOut];
    CP<ID3D11RenderTargetView> out_rtv_[kNumOut];
    CP<ID3D11ShaderResourceView> out_srv_owned_[kNumOut];
    ID3D11ShaderResourceView* out_srv_[kNumOut] = {
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    CP<ID3D11Texture2D> guard_tex_;          // full-luma R8G8_UNORM
    CP<ID3D11RenderTargetView> guard_rtv_;
    CP<ID3D11ShaderResourceView> guard_srv_;
    // Per-frame cyclopean active set in source coordinates.  RGBA stores
    // correction potential, priority, contour and occlusion respectively.
    CP<ID3D11Texture2D> pair_tex_;
    CP<ID3D11RenderTargetView> pair_rtv_;
    CP<ID3D11ShaderResourceView> pair_srv_;
    CP<ID3D11Texture2D> metric_tex_;         // 96x54 R8G8B8A8_UNORM
    CP<ID3D11RenderTargetView> metric_rtv_;
    CP<ID3D11Texture2D> metric_read_tex_;    // non-blocking staging

    uint32_t y_w_ = 0, y_h_ = 0, c_w_ = 0, c_h_ = 0;
    uint32_t pair_w_ = 0, pair_h_ = 0;
    DXGI_FORMAT plane_fmt_ = DXGI_FORMAT_UNKNOWN;
    uint64_t frame_counter_ = 0;
    uint64_t epoch_ = 1;
    uint64_t pending_epoch_ = 0;
    bool metric_pending_ = false;
    bool outputs_valid_ = false;
    bool failed_ = false;
    std::string error_;
    std::string mode_ = "off";
    float metric_mean_ = 0.0f;
    float metric_p95_ = 0.0f;
    float metric_asym_ = 0.0f;
    float metric_coverage_ = 0.0f;
    float comfort_scale_px_ = 0.0f;
    float pending_comfort_scale_px_ = 0.0f;
    float comfort_mean_loss_px_ = 0.0f;
    float comfort_p95_loss_px_ = 0.0f;
    float comfort_coverage_ = 0.0f;
    float edge_veto_p95_ = 0.0f;
    float edge_veto_coverage_ = 0.0f;
};
