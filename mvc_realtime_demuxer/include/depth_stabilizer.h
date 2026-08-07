// DepthStabilizer: temporal stabilization for the synth3d (2D->3D) depth
// stream (see .superpowers/sdd/2026-07-28-2d-to-3d-realtime, Task 3).
//
// DepthEngine::infer() produces a per-frame monocular depth map that is
// scale/shift-ambiguous frame to frame (the model has no notion of a
// consistent metric scale), which reads as flicker/pumping once warped into
// stereo. DepthStabilizer fits the new frame onto the running EMA (least
// squares scale+shift), blends it in, and snaps (re-primes) on a scene cut
// instead of blending across it. It also renormalizes to the 2nd/98th
// percentile of the EMA and quantizes to uint16 for the warp shader.
//
// Pure math, no ORT/D3D — runs on the synth3d inference thread (Task 4);
// per spec Sec.7 ("playback never dies for 3D") this file cannot itself
// fail: NaN/inf-free callers are the DepthEngine contract.
//
// NOT thread-safe: single owner thread, same idiom as DepthEngine.
#pragma once

#include <cstdint>
#include <vector>

#include "parallel_select.h"

class DepthStabilizer {
public:
    explicit DepthStabilizer(size_t n);
    void reset();                                  // seek / first frame
    size_t size() const { return ema_.size(); }    // n, for binding-side validation

    // Two clocks exist and must not be conflated:
    //   source_dt_ms -- elapsed VIDEO time (PTS) between the two source
    //     observations. It drives every content-temporal rule: motion velocity,
    //     EMA/tone rates, stability, snap confirmation and history span.
    //   update_dt_ms -- elapsed COMPUTE wall time between stabilized-map
    //     updates. It is retained only for diagnostics/observability; changing
    //     backend speed must not change the geometry produced for the same
    //     video observations.
    //
    // Every constant remains calibrated at kReferenceDtMs. set_dt_ms() is the
    // backward-compatible single-clock API and sets BOTH intervals to the
    // same effective value; the live service uses the two explicit setters.
    // Compute time keeps the historical [20, 500] ms clamp. Video time accepts
    // [4, 500] ms so high-refresh material is not silently read as 50 fps.
    // NaN/<=0 falls back to the reference. Defaults preserve the historical
    // bitwise behavior.
    static constexpr float kReferenceDtMs = 120.0f;
    void set_dt_ms(float dt_ms);
    void set_update_dt_ms(float dt_ms);
    void set_source_dt_ms(float dt_ms);
    float dt_ms() const { return source_dt_ms_; }  // legacy temporal alias
    float update_dt_ms() const { return update_dt_ms_; }
    float source_dt_ms() const { return source_dt_ms_; }
    size_t history_count() const { return history_count_; }

    // Thread budget for reproject()'s transport passes and step()'s pixel
    // loop (both data-parallel per pixel). 1 = sequential. Outputs are
    // bitwise identical at any value (per-pixel writes only; the diagnostic
    // sums are reduced chunk-in-order and may differ in the last bit).
    int worker_threads = 1;

    // Transport the established geometry into the current frame before
    // step(). flow is previous->current displacement, expressed in full-grid
    // pixels at each current destination; reliability is [0..1].
    void reproject(const float* flow_x, const float* flow_y,
                   const float* reliability, size_t width, size_t height);

    // raw model output in, normalized quantized nearness out.
    //
    // `motion` is an optional per-pixel [0..1] luma-change map, already aligned
    // to the inference grid. It does not alter the scale/shift fit: it only
    // controls how quickly each pixel accepts the fitted observation. Static
    // areas therefore remain calm while moving contours do not drag an old
    // depth edge behind them. `scene_change` is an independent source-image
    // histogram distance [0..1]; it complements the depth residual because a
    // monocular model can occasionally map two unrelated shots onto deceptively
    // similar relative-depth distributions.
    //
    // Returns true if this frame was detected as a scene cut (EMA snapped).
    bool step(const float* raw, uint16_t* out_q16,
              const float* motion = nullptr, float scene_change = 0.0f,
              const float* confidence = nullptr,
              const float* surface_boundary = nullptr);

    float alpha = 0.2f;            // EMA blend
    float alpha_static = 0.12f;    // adaptive EMA floor (motion == 0)
    // Keep the fast path below 0.5: without optical-flow reprojection, a
    // single noisy inference must never overturn a previously stable contour.
    float alpha_motion = 0.45f;    // adaptive EMA ceiling (strong motion)
    float motion_low = 0.015f;     // luma delta below this is treated as static
    float motion_high = 0.12f;     // luma delta above this accepts quickly
    float tone_alpha = 0.06f;      // slow per-shot depth-range adaptation
    float depth_contrast = 0.82f;  // soft depth-budget compression about 0.5
    // Gentle monotonic S-curve before depth_contrast: separates middle planes
    // while the existing headroom still protects the near/far extremes.
    float depth_scurve = 0.12f;
    float confidence_floor = 0.12f;// never completely freeze uncertain pixels
    // Once a smooth, confident surface has remained stable, tiny fitted-depth
    // changes are model jitter rather than useful geometry. Reduce (never zero)
    // their local alpha; motion/boundaries bypass this deadband immediately.
    float local_stability_low = 0.62f;
    float local_stability_high = 0.86f;
    float local_jitter_low = 0.006f;
    float local_jitter_high = 0.040f;
    float local_jitter_alpha_scale = 0.28f;
    // Robust multi-timescale memory. In locally calm regions, the fitted DA3
    // observation is selected from a short reprojected history instead of
    // averaging mutually exclusive foreground/background depths into a ramp.
    // Moving or newly-occluded surface boundaries immediately collapse back
    // to the current observation.
    bool temporal_surface_memory = true;
    float temporal_stable_low = 0.45f;
    float temporal_stable_high = 0.75f;
    float boundary_fast_motion = 0.035f;
    // Two-cycle same-sign outlier confirmation ("snap"). A per-pixel EMA
    // time-averages DA3's independent per-inference edge placement jitter
    // (+/-1-2 texels on an otherwise static contour) into a soft spatial
    // ramp, which the stereo warp turns into a rubber-band contour. A pixel
    // whose fitted observation differs from the running EMA by more than
    // snap_frac * (tone_hi_ - tone_lo_) is a candidate outlier; only once
    // the SAME sign repeats on the very next cycle does it snap straight to
    // the fitted value (bypassing alpha/confidence entirely -- a confirmed
    // outlier is the new edge position, not noise to be damped). A single,
    // unconfirmed outlier cycle still blends normally, so isolated one-frame
    // noise never snaps.
    float snap_frac = 0.18f;
    // Video-time spans of the two map-counted mechanisms above/below, both
    // expressed at the reference cadence (2 cycles / 1 slot per map at
    // 120 ms). At faster cadences the snap needs more same-sign cycles
    // (>= 2 always) and the history ring advances less than once per map.
    float snap_confirm_ms = 240.0f;
    float history_commit_ms = 120.0f;
    // Threshold on mean|a*raw+b - ema| normalized by max(stddev(ema),
    // stddev(raw)) -- dimensionally consistent in both directions (a
    // structured ema normalizes by its own spread; a flat ema meeting
    // structured raw normalizes by raw's spread instead). Two degenerate
    // cases bypass the ratio entirely: flat ema + flat raw never cuts
    // (nothing to compare), flat ema + structured raw always cuts (the
    // fit trivially reproduces a constant ema, so the ratio is
    // provably 0/anything regardless of cut_threshold -- see step()).
    float cut_threshold = 0.35f;
    // Adaptive gate on the depth-cut residual (2026-08-06). The ambient
    // normalized residual is content-dependent: measured live medians run
    // ~0.02-0.07 on calm shots but 0.16-0.22 during fast motion, against the
    // fixed 0.35 threshold above -- on a busy film the ambient peaks crossed
    // it about once per second (331 depth cuts in 433 s), and every false
    // fire hard-resets the temporal state: visible depth pumping. A depth cut
    // must now ALSO be an outlier against the content's own recent level:
    //     normalized > cut_baseline_gain * running_baseline.
    // Calm content keeps the absolute threshold (gain*baseline sits far below
    // 0.35); busy content raises the effective bar to where only a genuine
    // affine-fit failure -- a real cut -- reaches. The scene/histogram
    // detector and the boundary scout are unaffected and still OR in. A slow
    // dissolve raises the baseline as it progresses, which correctly KEEPS the
    // blend path for it. 0 disables the gate (pre-2026-08-06 behavior;
    // service maps SYLC_SYNTH3D_DEPTHCUT_ADAPT=0 to it).
    float cut_baseline_gain = 3.0f;
    float scene_cut_threshold = 0.42f;
    // Auto-convergence: per-shot zero-parallax suggestion. The percentile of
    // the SAME confidence-filtered stabilized sample the tone range uses,
    // expressed in the SAME normalized nearness space the warp samples (the
    // s-curve/contrast transform is monotonic, so transforming the percentile
    // value equals the percentile of the transformed map). 0.55 places most
    // of an ordinary scene slightly behind the screen (broadcast comfort).
    // Smoothed in VIDEO time with a per-unit-time cap (anti-breathing, same
    // pattern as the tone range) and snapped on cut/prime -- the existing
    // post-snap disparity ramp then hides the zero-plane teleport.
    float auto_convergence_percentile = 0.55f;
    float convergence_alpha = 0.08f;

    // Last-step observability. These values feed the renderer's diagnostics
    // line and are intentionally read-only to callers.
    float last_motion() const { return last_motion_; }
    float last_effective_alpha() const { return last_effective_alpha_; }
    float last_scene_change() const { return last_scene_change_; }
    float last_confidence() const { return last_confidence_; }
    bool last_cut() const { return last_cut_; }
    bool last_depth_cut() const { return last_depth_cut_; }
    bool last_scene_cut() const { return last_scene_cut_; }
    // -1 denotes the explicit flat-reference/structured-input degeneracy,
    // which is a depth cut but has no meaningful normalized ratio.
    float last_depth_residual() const { return last_depth_residual_; }
    float residual_baseline() const { return residual_baseline_; }
    uint64_t cut_count() const { return cut_count_; }
    uint64_t snap_count() const { return snap_count_; }
    float last_stability() const { return last_stability_; }
    float last_history_support() const { return last_history_support_; }
    float suggested_convergence() const { return auto_convergence_; }

    // least-squares closed form: argmin_{a,b} sum (a*cur+b - ref)^2
    static void fit_scale_shift(const float* ref, const float* cur, size_t n,
                                float& a, float& b);

private:
    static constexpr size_t kHistorySlots = 5;
    void clear_history();
    void commit_history(const float* values, const float* confidence);

    std::vector<float> ema_;
    bool primed_ = false;
    std::vector<float> tmp_;
    sylc_select::Scratch select_scratch_;  // histogram/offset reuse, no per-map alloc
    float last_motion_ = 0.0f;
    float last_effective_alpha_ = 0.0f;
    float last_scene_change_ = 0.0f;
    float last_confidence_ = 1.0f;
    bool last_cut_ = false;
    bool last_depth_cut_ = false;
    bool last_scene_cut_ = false;
    float last_depth_residual_ = 0.0f;
    float residual_baseline_ = 0.0f;       // ambient level between cuts
    bool residual_baseline_primed_ = false;
    uint64_t cut_count_ = 0;
    bool tone_primed_ = false;
    float tone_lo_ = 0.0f;
    float tone_hi_ = 1.0f;
    bool convergence_primed_ = false;
    float auto_convergence_ = 0.5f;
    // 0 = no pending outlier; otherwise sign = outlier direction and
    // magnitude = consecutive same-sign candidate cycles seen so far (the
    // snap fires when the count reaches ceil(snap_confirm_ms / source_dt_ms_)).
    std::vector<int8_t> snap_pending_;
    float update_dt_ms_ = kReferenceDtMs;
    float source_dt_ms_ = kReferenceDtMs;
    float history_elapsed_ms_ = 0.0f;
    uint64_t snap_count_ = 0;
    // Five aligned observations are enough for a robust median at temporal
    // dilations 1/2/4 without turning the live path into a long-latency batch.
    std::vector<float> history_values_;
    std::vector<float> history_weights_;
    std::vector<float> history_slot_tmp_;
    std::vector<float> history_weight_tmp_;
    std::vector<float> stability_;
    std::vector<float> stability_tmp_;
    size_t history_head_ = 0;
    size_t history_count_ = 0;
    float last_stability_ = 0.0f;
    float last_history_support_ = 1.0f;
};
