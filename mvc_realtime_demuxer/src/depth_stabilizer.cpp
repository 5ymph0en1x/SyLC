// DepthStabilizer implementation. See depth_stabilizer.h.
#include "depth_stabilizer.h"

#include "parallel_chunks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

DepthStabilizer::DepthStabilizer(size_t n)
    : ema_(n),
      tmp_(n),
      snap_pending_(n, 0),
      history_values_(kHistorySlots * n, 0.0f),
      history_weights_(kHistorySlots * n, 0.0f),
      // Pixel-major full-ring scratch: all five temporal values for one pixel
      // are adjacent. Both reproject() and step() consume history per pixel,
      // so this avoids ten independent full-grid memory streams.
      history_slot_tmp_(kHistorySlots * n, 0.0f),
      history_weight_tmp_(kHistorySlots * n, 0.0f),
      stability_(n, 0.0f),
      stability_tmp_(n, 0.0f) {}

void DepthStabilizer::clear_history() {
    history_head_ = 0;
    history_count_ = 0;
    history_elapsed_ms_ = 0.0f;
    std::fill(history_weights_.begin(), history_weights_.end(), 0.0f);
    std::fill(stability_.begin(), stability_.end(), 0.0f);
    last_stability_ = 0.0f;
    last_history_support_ = 1.0f;
}

void DepthStabilizer::set_dt_ms(float dt_ms) {
    set_update_dt_ms(dt_ms);
    // Preserve the historical one-clock contract exactly, including its
    // 20 ms floor. The explicit video-time setter may go down to 4 ms.
    source_dt_ms_ = update_dt_ms_;
}

void DepthStabilizer::set_update_dt_ms(float dt_ms) {
    if (!(dt_ms > 0.0f)) dt_ms = kReferenceDtMs;   // NaN/zero/negative guard
    update_dt_ms_ = std::max(20.0f, std::min(500.0f, dt_ms));
}

void DepthStabilizer::set_source_dt_ms(float dt_ms) {
    if (!(dt_ms > 0.0f)) dt_ms = kReferenceDtMs;   // NaN/zero/negative guard
    source_dt_ms_ = std::max(4.0f, std::min(500.0f, dt_ms));
}

void DepthStabilizer::commit_history(const float* values,
                                     const float* confidence) {
    const size_t n = ema_.size();
    for (size_t i = 0; i < n; ++i) {
        const size_t hi = i * kHistorySlots + history_head_;
        history_values_[hi] = values[i];
        history_weights_[hi] = confidence
            ? std::max(0.0f, std::min(1.0f, confidence[i])) : 1.0f;
    }
    history_head_ = (history_head_ + 1) % kHistorySlots;
    history_count_ = std::min(kHistorySlots, history_count_ + 1);
}

void DepthStabilizer::reset() {
    primed_ = false;
    last_motion_ = 0.0f;
    last_effective_alpha_ = 0.0f;
    last_scene_change_ = 0.0f;
    last_confidence_ = 1.0f;
    last_cut_ = false;
    last_depth_cut_ = false;
    last_scene_cut_ = false;
    last_depth_residual_ = 0.0f;
    residual_baseline_ = 0.0f;
    residual_baseline_primed_ = false;
    tone_primed_ = false;
    convergence_primed_ = false;
    auto_convergence_ = 0.5f;
    std::fill(snap_pending_.begin(), snap_pending_.end(), int8_t(0));
    clear_history();
}

void DepthStabilizer::reproject(const float* flow_x, const float* flow_y,
                                const float* reliability,
                                size_t width, size_t height) {
    if (!primed_ || !flow_x || !flow_y || width * height != ema_.size())
        return;
    const size_t n = ema_.size();
    tmp_.resize(n);
    auto sample = [&](const float* values, float x, float y) {
        x = std::max(0.0f, std::min(static_cast<float>(width - 1), x));
        y = std::max(0.0f, std::min(static_cast<float>(height - 1), y));
        const size_t x0 = static_cast<size_t>(x);
        const size_t y0 = static_cast<size_t>(y);
        const size_t x1 = std::min(width - 1, x0 + 1);
        const size_t y1 = std::min(height - 1, y0 + 1);
        const float fx = x - static_cast<float>(x0);
        const float fy = y - static_cast<float>(y0);
        const float a = values[y0 * width + x0] * (1.0f - fx) +
                        values[y0 * width + x1] * fx;
        const float b = values[y1 * width + x0] * (1.0f - fx) +
                        values[y1 * width + x1] * fx;
        return a * (1.0f - fy) + b * fy;
    };
    static const bool fused_reproject = []() {
        char* env = nullptr;
        size_t len = 0;
        bool on = true;
        if (_dupenv_s(&env, &len, "SYLC_SYNTH3D_REPROJECT_FUSED") == 0 && env) {
            on = env[0] != '0';
            free(env);
        }
        return on;
    }();

    // One parallel region for every transported plane. In the fused path the
    // clamped source coordinate, four indices and bilinear weights are formed
    // ONCE per destination and reused for EMA, history values/weights and
    // stability. The historical path remains available for A/B rollback.
    if (fused_reproject) {
        parallel_chunks(static_cast<int>(height), worker_threads,
                        [&](int, int y_begin, int y_end) {
        for (size_t y = static_cast<size_t>(y_begin);
             y < static_cast<size_t>(y_end); ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t i = y * width + x;
                const float trust = reliability
                    ? std::max(0.0f, std::min(1.0f, reliability[i])) : 1.0f;
                const float sx = std::max(0.0f, std::min(
                    static_cast<float>(width - 1),
                    static_cast<float>(x) - flow_x[i]));
                const float sy = std::max(0.0f, std::min(
                    static_cast<float>(height - 1),
                    static_cast<float>(y) - flow_y[i]));
                const size_t x0 = static_cast<size_t>(sx);
                const size_t y0 = static_cast<size_t>(sy);
                const size_t x1 = std::min(width - 1, x0 + 1);
                const size_t y1 = std::min(height - 1, y0 + 1);
                const float fx = sx - static_cast<float>(x0);
                const float fy = sy - static_cast<float>(y0);
                const size_t i00 = y0 * width + x0;
                const size_t i10 = y0 * width + x1;
                const size_t i01 = y1 * width + x0;
                const size_t i11 = y1 * width + x1;
                auto sample_indices = [&](const float* values) {
                    const float a = values[i00] * (1.0f - fx) +
                                    values[i10] * fx;
                    const float b = values[i01] * (1.0f - fx) +
                                    values[i11] * fx;
                    return a * (1.0f - fy) + b * fy;
                };
                const float transported = sample_indices(ema_.data());
                tmp_[i] = ema_[i] * (1.0f - trust) + transported * trust;
                for (size_t slot = 0; slot < history_count_; ++slot) {
                    const size_t si = i * kHistorySlots + slot;
                    auto sample_slot = [&](const std::vector<float>& values) {
                        const float a =
                            values[i00 * kHistorySlots + slot] * (1.0f - fx) +
                            values[i10 * kHistorySlots + slot] * fx;
                        const float b =
                            values[i01 * kHistorySlots + slot] * (1.0f - fx) +
                            values[i11 * kHistorySlots + slot] * fx;
                        return a * (1.0f - fy) + b * fy;
                    };
                    history_slot_tmp_[si] = sample_slot(history_values_);
                    history_weight_tmp_[si] =
                        sample_slot(history_weights_) * trust;
                }
                stability_tmp_[i] = sample_indices(stability_.data()) * trust;
            }
        }
        });
    } else {
        parallel_chunks(static_cast<int>(height), worker_threads,
                        [&](int, int y_begin, int y_end) {
        for (size_t y = static_cast<size_t>(y_begin);
             y < static_cast<size_t>(y_end); ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t i = y * width + x;
                const float trust = reliability
                    ? std::max(0.0f, std::min(1.0f, reliability[i])) : 1.0f;
                const float source_x = static_cast<float>(x) - flow_x[i];
                const float source_y = static_cast<float>(y) - flow_y[i];
                const float transported = sample(ema_.data(), source_x, source_y);
                tmp_[i] = ema_[i] * (1.0f - trust) + transported * trust;
                for (size_t slot = 0; slot < history_count_; ++slot) {
                    const size_t si = i * kHistorySlots + slot;
                    auto sample_slot = [&](const std::vector<float>& values) {
                        float sx = std::max(0.0f, std::min(
                            static_cast<float>(width - 1), source_x));
                        float sy = std::max(0.0f, std::min(
                            static_cast<float>(height - 1), source_y));
                        const size_t x0 = static_cast<size_t>(sx);
                        const size_t y0 = static_cast<size_t>(sy);
                        const size_t x1 = std::min(width - 1, x0 + 1);
                        const size_t y1 = std::min(height - 1, y0 + 1);
                        const float fx = sx - static_cast<float>(x0);
                        const float fy = sy - static_cast<float>(y0);
                        const float aa =
                            values[(y0 * width + x0) * kHistorySlots + slot] *
                                (1.0f - fx) +
                            values[(y0 * width + x1) * kHistorySlots + slot] * fx;
                        const float bb =
                            values[(y1 * width + x0) * kHistorySlots + slot] *
                                (1.0f - fx) +
                            values[(y1 * width + x1) * kHistorySlots + slot] * fx;
                        return aa * (1.0f - fy) + bb * fy;
                    };
                    history_slot_tmp_[si] = sample_slot(history_values_);
                    history_weight_tmp_[si] =
                        sample_slot(history_weights_) * trust;
                }
                stability_tmp_[i] = sample(
                    stability_.data(), source_x, source_y) * trust;
            }
        }
        });
    }
    ema_.swap(tmp_);
    history_values_.swap(history_slot_tmp_);
    history_weights_.swap(history_weight_tmp_);
    stability_.swap(stability_tmp_);
    // The flow transport just moved the established geometry; any pending
    // outlier signs referred to positions that no longer correspond to the
    // same content, so a static-scene confirmation must not carry across a
    // reprojection.
    std::fill(snap_pending_.begin(), snap_pending_.end(), int8_t(0));
}

void DepthStabilizer::fit_scale_shift(const float* ref, const float* cur, size_t n,
                                       float& a, float& b) {
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(cur[i]);
        const double y = static_cast<double>(ref[i]);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    const double dn = static_cast<double>(n);
    const double denom = dn * sxx - sx * sx;
    if (std::abs(denom) < 1e-9) {
        a = 1.0f;
        b = 0.0f;
        return;
    }
    const double da = (dn * sxy - sx * sy) / denom;
    const double db = (sy - da * sx) / dn;
    a = static_cast<float>(da);
    b = static_cast<float>(db);
}

bool DepthStabilizer::step(const float* raw, uint16_t* out_q16,
                           const float* motion, float scene_change,
                           const float* confidence,
                           const float* surface_boundary) {
    const size_t n = ema_.size();
    bool cut = false;
    // Video-time scale of this source observation relative to the calibration
    // cadence. video_ts == 1 takes the exact historical arithmetic (no pow,
    // bitwise back-compat); other cadences convert every per-observation rate r
    // into 1-(1-r)^video_ts so half-lives stay video-time true. Per-pixel rates
    // (alpha after its modifiers, the motion-driven stability decay) go
    // through small LUTs -- two transcendentals per LUT entry per step
    // instead of two per pixel.
    // All state evolution belongs to VIDEO time. update_dt_ms_ measures how
    // long the provider/worker took and is deliberately absent from the math.
    const float video_ts = source_dt_ms_ / kReferenceDtMs;
    const bool unit_time = std::fabs(video_ts - 1.0f) < 1e-6f;
    constexpr int kAlphaLut = 128;
    constexpr int kDecayLut = 64;
    float alpha_lut[kAlphaLut + 1] = {};
    float decay_lut[kDecayLut + 1] = {};
    if (!unit_time) {
        for (int k = 0; k <= kAlphaLut; ++k)
            alpha_lut[k] = 1.0f - std::pow(
                1.0f - static_cast<float>(k) / kAlphaLut, video_ts);
        for (int k = 0; k <= kDecayLut; ++k)
            decay_lut[k] = std::pow(
                1.0f - 0.78f * static_cast<float>(k) / kDecayLut, video_ts);
    }
    auto norm_alpha = [&](float al) {
        if (unit_time) return al;
        al = std::max(0.0f, std::min(1.0f, al));
        const float p = al * kAlphaLut;
        const int k = std::min(kAlphaLut - 1, static_cast<int>(p));
        const float f = p - static_cast<float>(k);
        return alpha_lut[k] + (alpha_lut[k + 1] - alpha_lut[k]) * f;
    };
    auto motion_decay = [&](float motion_t) {
        if (unit_time) return 1.0f - 0.78f * motion_t;
        const float p = motion_t * kDecayLut;
        const int k = std::min(kDecayLut - 1, static_cast<int>(p));
        const float f = p - static_cast<float>(k);
        return decay_lut[k] + (decay_lut[k + 1] - decay_lut[k]) * f;
    };
    const float rise_eff =
        unit_time ? 0.18f
                  : 1.0f - std::pow(1.0f - 0.18f, video_ts);
    const float boundary_collapse =
        unit_time ? 0.20f : std::pow(0.20f, video_ts);
    last_scene_change_ = std::max(0.0f, std::min(1.0f, scene_change));
    last_motion_ = 0.0f;
    last_effective_alpha_ = 0.0f;
    last_history_support_ = 1.0f;
    last_depth_cut_ = false;
    last_scene_cut_ = false;
    last_depth_residual_ = 0.0f;
    // Accumulate the diagnostic confidence in a pass that already visits the
    // full grid below.  This used to be its own O(n) traversal before both the
    // priming loop and the confidence-weighted affine fit.
    double confidence_sum = 0.0;
    if (!confidence) last_confidence_ = 1.0f;

    if (!primed_) {
        std::copy(raw, raw + n, ema_.begin());
        primed_ = true;
        last_effective_alpha_ = 1.0f;
        std::fill(snap_pending_.begin(), snap_pending_.end(), int8_t(0));
        clear_history();
        for (size_t i = 0; i < n; ++i) {
            const float conf = confidence
                ? std::max(0.0f, std::min(1.0f, confidence[i])) : 1.0f;
            confidence_sum += conf;
            stability_[i] = 0.50f * conf;
        }
        if (confidence)
            last_confidence_ = static_cast<float>(confidence_sum / n);
        commit_history(raw, confidence);
        double stability_sum = 0.0;
        for (float value : stability_) stability_sum += value;
        last_stability_ = static_cast<float>(stability_sum / n);
    } else {
        float a = 1.0f, b = 0.0f;
        if (confidence) {
            // Confidence-weighted affine alignment. DA3's uncertain sky,
            // reflections and soft boundaries must not decide the scale of an
            // entire shot. A small floor keeps the fit well-conditioned.
            double sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (size_t i = 0; i < n; ++i) {
                confidence_sum += std::max(
                    0.0, std::min(1.0, static_cast<double>(confidence[i])));
                const double w = std::max(
                    static_cast<double>(confidence_floor),
                    std::min(1.0, static_cast<double>(confidence[i])));
                const double x = raw[i], y = ema_[i];
                sw += w; sx += w * x; sy += w * y;
                sxx += w * x * x; sxy += w * x * y;
            }
            const double denom = sw * sxx - sx * sx;
            if (sw > 1e-9 && std::abs(denom) > 1e-9) {
                a = static_cast<float>((sw * sxy - sx * sy) / denom);
                b = static_cast<float>((sy - static_cast<double>(a) * sx) / sw);
            }
            last_confidence_ = static_cast<float>(confidence_sum / n);
        } else {
            fit_scale_shift(ema_.data(), raw, n, a, b);
        }

        // Fused single-pass moment accumulation (Task 6 perf; fix round 1:
        // Welford, not E[x^2]-E[x]^2). The residual sum and the raw
        // first/second moments of ema_/raw were previously four independent
        // O(n) loops (mean_ema, then var_ema depending on it, then mean_raw,
        // then var_raw depending on it) plus a fifth for the residual -- all
        // touching the same two arrays. Collapsed here into one pass using
        // Welford's online algorithm for both streams: equally single-pass,
        // but numerically stable by construction. E[x^2]-E[x]^2 was tried
        // first and rejected on review: in the near-flat regime this
        // stabilizer explicitly special-cases (test_flat_scene_stays_stable,
        // N~268k, jitter 1e-8, true variance ~3e-17) that formula's own
        // rounding noise floor sits 2-3 orders of magnitude ABOVE the true
        // signal -- the computed variance there was noise, correctly
        // classified only by the coarse kStddevEps=1e-6 gate's incidental
        // margin, silently erodable by a larger grid or bigger-scale model
        // output. Welford's m2 is a sum of products of same-sign deltas
        // (d1*d2, both converging toward the same running mean) and cannot
        // go negative beyond negligible rounding, so sqrt() below needs no
        // defensive clamp.
        double sum_abs_residual = 0.0;
        double count = 0.0;
        double mean_ema = 0.0, m2_ema = 0.0;
        double mean_raw = 0.0, m2_raw = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double ema_i = static_cast<double>(ema_[i]);
            const double raw_i = static_cast<double>(raw[i]);
            const double fitted = static_cast<double>(a) * raw_i + static_cast<double>(b);
            sum_abs_residual += std::abs(fitted - ema_i);

            count += 1.0;
            const double d1e = ema_i - mean_ema;
            mean_ema += d1e / count;
            const double d2e = ema_i - mean_ema;
            m2_ema += d1e * d2e;

            const double d1r = raw_i - mean_raw;
            mean_raw += d1r / count;
            const double d2r = raw_i - mean_raw;
            m2_raw += d1r * d2r;
        }
        const double dn = static_cast<double>(n);
        const double mean_residual = sum_abs_residual / dn;

        // Population variance (/n, matching the pre-fusion two-pass code --
        // not Welford's more common sample form, which would divide by
        // count-1).
        const double var_ema = m2_ema / dn;
        const double stddev_ema = std::sqrt(var_ema);

        const double var_raw = m2_raw / dn;
        const double stddev_raw = std::sqrt(var_raw);

        constexpr double kStddevEps = 1e-6;
        const bool ema_flat = stddev_ema < kStddevEps;
        const bool raw_flat = stddev_raw < kStddevEps;
        if (ema_flat && raw_flat) {
            // Both sides flat (e.g. black holding black): no spread on
            // either side to normalize a residual against, and nothing
            // changed. No cut; blend normally below.
            cut = false;
            last_depth_residual_ = 0.0f;
        } else if (ema_flat && !raw_flat) {
            // Degenerate (constant) reference meeting a structured frame.
            // The OLS fit of a structured `raw` onto a *constant* `ema`
            // always resolves to a=0, b=mean(ema) -- which reproduces ema
            // exactly (mean_residual == 0 mathematically, not just
            // numerically) no matter how different raw's structure is. The
            // residual test below is therefore provably blind in this one
            // direction, so it is handled as an explicit case: a flat
            // reference carries no information to reject real incoming
            // structure, so treat it as a cut.
            cut = true;
            last_depth_residual_ = -1.0f;
        } else {
            // Normalize by whichever side actually has spread to compare
            // against (dimensionally consistent: model units / model units)
            // instead of an absolute-residual fallback.
            const double denom = std::max(stddev_ema, stddev_raw);
            const double normalized = mean_residual / denom;
            last_depth_residual_ = static_cast<float>(normalized);
            cut = normalized > static_cast<double>(cut_threshold);
            // Adaptive gate (see cut_baseline_gain in the header): on content
            // whose ambient residual rides high, an absolute crossing only
            // counts as a cut if it is also an outlier against the recent
            // baseline.
            if (cut && cut_baseline_gain > 0.0f && residual_baseline_primed_ &&
                normalized <= static_cast<double>(cut_baseline_gain) *
                              static_cast<double>(residual_baseline_))
                cut = false;
            if (!cut) {
                // Track the ambient level in VIDEO time (same discipline as
                // the tone range: identical half-life whether one second
                // arrives as 8 maps or 19). Cut frames are excluded -- the
                // baseline describes the regime BETWEEN cuts, and the
                // post-snap residual collapse re-enters it naturally.
                constexpr float kResidualBaselineAlpha = 0.05f;
                if (!residual_baseline_primed_) {
                    residual_baseline_ = static_cast<float>(normalized);
                    residual_baseline_primed_ = true;
                } else {
                    const float alpha_eff = unit_time
                        ? kResidualBaselineAlpha
                        : 1.0f - std::pow(1.0f - kResidualBaselineAlpha,
                                          video_ts);
                    residual_baseline_ += alpha_eff *
                        (static_cast<float>(normalized) - residual_baseline_);
                }
            }
        }

        last_depth_cut_ = cut;

        // The source-image signal is independent of the model's arbitrary
        // scale/shift. Requiring either detector (rather than both) prevents a
        // hard edit from being blended merely because two shots happen to have
        // similar depth statistics.
        last_scene_cut_ = last_scene_change_ >= scene_cut_threshold;
        cut = cut || last_scene_cut_;

        if (cut) {
            std::copy(raw, raw + n, ema_.begin());
            last_effective_alpha_ = 1.0f;
            ++cut_count_;
            std::fill(snap_pending_.begin(), snap_pending_.end(), int8_t(0));
            clear_history();
            for (size_t i = 0; i < n; ++i) {
                const float conf = confidence
                    ? std::max(0.0f, std::min(1.0f, confidence[i])) : 1.0f;
                stability_[i] = 0.50f * conf;
            }
            commit_history(raw, confidence);
            double stability_sum = 0.0;
            for (float value : stability_) stability_sum += value;
            last_stability_ = static_cast<float>(stability_sum / n);
        } else {
            double motion_sum = 0.0;
            double alpha_sum = 0.0;
            double stability_sum = 0.0;
            double support_sum = 0.0;
            const float span = std::max(1e-6f, motion_high - motion_low);
            // Outlier-snap threshold uses the PREVIOUS cycle's tone range:
            // the tone_lo_/tone_hi_ update runs after this loop (below), so
            // at this point they still hold the last committed reference
            // range -- exactly the "established" range a static-edge
            // confirmation should be judged against. Before that range has
            // ever been established, skip snapping entirely.
            const bool snap_active = tone_primed_;
            const float snap_threshold =
                snap_frac * std::max(1e-6f, tone_hi_ - tone_lo_);
            // The confirmation is a ~snap_confirm_ms VIDEO-time window, not
            // "2 maps": at the reference cadence this is exactly the
            // historical 2 cycles; at TensorRT cadence it takes more
            // same-sign cycles so a static contour still gets the same
            // amount of real time to prove the edge moved. Never below 2 --
            // a single unconfirmed outlier must never snap.
            const int snap_required = std::max(2, static_cast<int>(
                std::ceil(snap_confirm_ms / source_dt_ms_)));
            // Insert the current affine-aligned observation before selection.
            // It is initially the newest candidate (age 0); the ring pointer is
            // committed only after every pixel has read the same history view.
            const size_t current_slot = history_head_;
            const size_t available_history =
                std::min(kHistorySlots, history_count_ + 1);
            // Data-parallel blend: the insertion write and every read the
            // selection makes for pixel i touch history/state at index i
            // only, so chunks over the flat index are independent and the
            // per-pixel outputs (ema_, stability_, snap_pending_, history)
            // are bitwise identical at any thread count. The four running
            // sums are diagnostics; they reduce chunk-in-order below.
            const int blend_chunks = parallel_chunk_count(
                static_cast<int>(n), worker_threads);
            std::vector<double> chunk_motion(blend_chunks, 0.0);
            std::vector<double> chunk_alpha(blend_chunks, 0.0);
            std::vector<double> chunk_stability(blend_chunks, 0.0);
            std::vector<double> chunk_support(blend_chunks, 0.0);
            std::vector<uint64_t> chunk_snaps(blend_chunks, 0);
            parallel_chunks(static_cast<int>(n), worker_threads,
                            [&](int chunk_id, int i_begin, int i_end) {
            double local_motion = 0.0;
            double local_alpha = 0.0;
            double local_stability = 0.0;
            double local_support = 0.0;
            uint64_t local_snaps = 0;
            for (size_t i = static_cast<size_t>(i_begin);
                 i < static_cast<size_t>(i_end); ++i) {
                const size_t current_hi = i * kHistorySlots + current_slot;
                history_values_[current_hi] = a * raw[i] + b;
                history_weights_[current_hi] = confidence
                    ? std::max(0.0f, std::min(1.0f, confidence[i])) : 1.0f;
                float al = alpha;
                float mv = 0.0f;
                float motion_t = 0.0f;
                if (motion) {
                    mv = std::max(0.0f, std::min(1.0f, motion[i]));
                    // Velocity normalization belongs to SOURCE time, not map
                    // update time. A frame may wait in the mailbox while the
                    // previous inference runs; provider speed must not change
                    // the perceived velocity of those two source images.
                    if (std::fabs(video_ts - 1.0f) >= 1e-6f)
                        mv = std::min(1.0f, mv / video_ts);
                    motion_t = (mv - motion_low) / span;
                    motion_t = std::max(0.0f, std::min(1.0f, motion_t));
                    // smoothstep avoids a visible boundary between "static"
                    // and "moving" classifications in the disparity field.
                    motion_t = motion_t * motion_t * (3.0f - 2.0f * motion_t);
                    al = alpha_static +
                         (alpha_motion - alpha_static) * motion_t;
                    local_motion += mv;
                }
                al = std::max(0.0f, std::min(1.0f, al));
                const float conf = confidence
                    ? std::max(0.0f, std::min(1.0f, confidence[i])) : 1.0f;
                const float boundary = surface_boundary
                    ? std::max(0.0f, std::min(1.0f, surface_boundary[i])) : 0.0f;

                // Calm evidence builds memory slowly; real local motion or an
                // articulated boundary destroys it quickly. This decision is
                // per pixel -- a mostly static shot can therefore keep a long
                // wall history while a person's neck uses the current frame.
                const float stable_target = 0.25f + 0.75f * conf;
                if (motion_t > 0.0f) {
                    stability_[i] *= motion_decay(motion_t);
                } else {
                    stability_[i] +=
                        rise_eff * (stable_target - stability_[i]);
                }
                const bool moving_boundary =
                    boundary >= 0.35f && mv >= boundary_fast_motion;
                if (moving_boundary)
                    stability_[i] *= boundary_collapse;
                stability_[i] =
                    std::max(0.0f, std::min(1.0f, stability_[i]));

                // Low-confidence observations gently defer to the transported
                // history instead of injecting unstable geometry.
                al *= confidence_floor + (1.0f - confidence_floor) * conf;
                const float fitted = a * raw[i] + b;
                float target = fitted;
                size_t desired_history = 1;
                // Smooth surface interiors are already handled efficiently by
                // the established EMA. Spend the robust multi-frame selector
                // where it adds information: real depth boundaries and
                // uncertain DA3 regions. This keeps the live path fast at any
                // inference grid.
                const bool needs_surface_history =
                    boundary >= 0.12f || conf < 0.55f;
                if (temporal_surface_memory && !moving_boundary &&
                    needs_surface_history) {
                    if (stability_[i] >= temporal_stable_high)
                        desired_history = kHistorySlots;
                    else if (stability_[i] >= temporal_stable_low)
                        desired_history = 3;
                }
                desired_history =
                    std::min(desired_history, available_history);

                if (desired_history > 1) {
                    float values[kHistorySlots] = {};
                    float weights[kHistorySlots] = {};
                    size_t candidates = 0;
                    float total_weight = 0.0f;
                    for (size_t age = 0; age < desired_history; ++age) {
                        const size_t slot =
                            (current_slot + kHistorySlots - age) %
                            kHistorySlots;
                        const size_t hi = i * kHistorySlots + slot;
                        // Dilated history remains influential but never
                        // outweighs all newer evidence on its own.
                        const float recency =
                            age == 0 ? 1.0f :
                            age == 1 ? 0.94f :
                            age == 2 ? 0.88f :
                            age == 3 ? 0.80f : 0.72f;
                        const float weight =
                            history_weights_[hi] * recency;
                        if (weight < 0.02f) continue;
                        values[candidates] = history_values_[hi];
                        weights[candidates] = weight;
                        total_weight += weight;
                        ++candidates;
                    }
                    // Weighted median: unlike an arithmetic mean it can select
                    // foreground OR background at a jittering contour, never
                    // an impossible semi-depth between the two surfaces.
                    for (size_t p = 1; p < candidates; ++p) {
                        const float value = values[p];
                        const float weight = weights[p];
                        size_t q = p;
                        while (q > 0 && values[q - 1] > value) {
                            values[q] = values[q - 1];
                            weights[q] = weights[q - 1];
                            --q;
                        }
                        values[q] = value;
                        weights[q] = weight;
                    }
                    if (candidates >= 2 && total_weight > 1e-6f) {
                        float accumulated = 0.0f;
                        for (size_t p = 0; p < candidates; ++p) {
                            accumulated += weights[p];
                            if (accumulated >= 0.5f * total_weight) {
                                target = values[p];
                                break;
                            }
                        }
                        desired_history = candidates;
                    } else {
                        desired_history = 1;
                    }
                }

                if (boundary >= 0.35f && desired_history >= 3) {
                    // The robust selector has chosen an actual layer at a
                    // calm edge. Commit that discrete decision decisively;
                    // feeding it through the ordinary 0.12 EMA would recreate
                    // the very semi-depth ramp the median removed.
                    al = std::max(al, 0.22f + 0.42f * conf);
                }
                if (moving_boundary) {
                    // Do not drag an old layer across an articulated contour.
                    // High confidence adopts the current DA3 edge quickly;
                    // low confidence still retains a modest safety blend.
                    target = fitted;
                    al = std::max(al, 0.18f + 0.64f * conf);
                    snap_pending_[i] = 0;
                }

                // Local calm-surface deadband. DA3 can move a perfectly still
                // wall/face by a few percent of the established range from one
                // inference to the next. Once that pixel has accumulated stable,
                // confident evidence, slow only those sub-threshold changes.
                // Boundaries and motion bypass this immediately, so this cannot
                // leave a stale silhouette or articulated feature behind.
                if (!moving_boundary && boundary < 0.12f && tone_primed_ &&
                    history_count_ >= 2) {
                    const float established =
                        std::max(1.0e-6f, tone_hi_ - tone_lo_);
                    const float jitter =
                        std::abs(target - ema_[i]) / established;
                    float stable_t = (stability_[i] - local_stability_low) /
                        std::max(1.0e-6f,
                                 local_stability_high - local_stability_low);
                    stable_t = std::max(0.0f, std::min(1.0f, stable_t));
                    stable_t = stable_t * stable_t * (3.0f - 2.0f * stable_t);
                    float jitter_t = (jitter - local_jitter_low) /
                        std::max(1.0e-6f,
                                 local_jitter_high - local_jitter_low);
                    jitter_t = std::max(0.0f, std::min(1.0f, jitter_t));
                    jitter_t = jitter_t * jitter_t * (3.0f - 2.0f * jitter_t);
                    const float calm = stable_t * (1.0f - motion_t) * conf;
                    const float scale = std::max(
                        0.0f, std::min(1.0f, local_jitter_alpha_scale));
                    al *= 1.0f - calm * (1.0f - jitter_t) * (1.0f - scale);
                }

                bool snapped = false;
                if (snap_active && !moving_boundary) {
                    const float diff = target - ema_[i];
                    if (std::abs(diff) > snap_threshold) {
                        const bool positive = diff > 0.0f;
                        const int8_t pending = snap_pending_[i];
                        const int confirm_count = (pending != 0 &&
                                                   (pending > 0) == positive)
                            ? std::abs(pending) + 1 : 1;
                        if (confirm_count >= snap_required) {
                            // Confirmed same-sign for the full VIDEO-time
                            // window: adopt the fitted observation outright.
                            // This bypasses alpha/confidence by design -- a
                            // confirmed outlier IS the new edge position,
                            // not noise to be damped.
                            ema_[i] = target;
                            ++local_snaps;
                            snapped = true;
                            snap_pending_[i] = 0;
                        } else {
                            const int capped = std::min(confirm_count, 100);
                            snap_pending_[i] = positive
                                ? static_cast<int8_t>(capped)
                                : static_cast<int8_t>(-capped);
                        }
                    } else {
                        snap_pending_[i] = 0;
                    }
                }
                // Report and blend with the time-true alpha so a faster
                // provider does not shorten the blend half-life.
                al = norm_alpha(al);
                if (!snapped)
                    ema_[i] = (1.0f - al) * ema_[i] + al * target;
                local_alpha += al;
                local_stability += stability_[i];
                local_support += static_cast<double>(desired_history);
            }
            chunk_motion[chunk_id] = local_motion;
            chunk_alpha[chunk_id] = local_alpha;
            chunk_stability[chunk_id] = local_stability;
            chunk_support[chunk_id] = local_support;
            chunk_snaps[chunk_id] = local_snaps;
            });
            for (int c = 0; c < blend_chunks; ++c) {
                motion_sum += chunk_motion[c];
                alpha_sum += chunk_alpha[c];
                stability_sum += chunk_stability[c];
                support_sum += chunk_support[c];
                snap_count_ += chunk_snaps[c];
            }
            // The ring must span ~kHistorySlots * history_commit_ms of VIDEO
            // time at any cadence: advance the head only when the commit
            // interval has elapsed (exactly once per map at the reference
            // cadence). When it does not advance, the next step simply
            // overwrites the current slot -- the newest candidate is always
            // the current observation either way.
            history_elapsed_ms_ += source_dt_ms_;
            if (history_elapsed_ms_ >= history_commit_ms) {
                history_head_ = (history_head_ + 1) % kHistorySlots;
                history_count_ = std::min(kHistorySlots, history_count_ + 1);
                history_elapsed_ms_ = std::min(
                    history_elapsed_ms_ - history_commit_ms,
                    history_commit_ms);
            }
            last_motion_ = motion ? static_cast<float>(motion_sum / n) : 0.0f;
            last_effective_alpha_ = static_cast<float>(alpha_sum / n);
            last_stability_ = static_cast<float>(stability_sum / n);
            last_history_support_ = static_cast<float>(support_sum / n);
        }
    }
    last_cut_ = cut;

    // Confidence-aware percentile targets. Low-confidence regions still keep
    // their depth but cannot stretch the stereo budget for the whole picture.
    //
    // The three percentiles (tone lo/hi here, auto-convergence a few blocks
    // down) all select from the same multiset. A k-th order statistic is a
    // value, not a reduction: any correct algorithm returns the same value, so
    // counting selection is bitwise nth_element's answer at any worker count
    // -- unlike the affine/Welford reductions above, which stay serial because
    // float association order is part of THEIR result. Counting does ~3 passes
    // of work where introselect does ~1, so it only wins once enough workers
    // share it: below the measured crossover the sequential path costs less.
    // Both paths produce the identical value (pinned by the bench selftest and
    // by the cross-path bit-exact sweep), so the branch is invisible in the
    // output.
    constexpr int kSelectParallelThreshold = 4;
    const bool parallel_sel = worker_threads >= kSelectParallelThreshold;
    tmp_.clear();
    if (confidence) {
        if (parallel_sel) {
            // Order-preserving parallel filter: identical sequence at any
            // worker count (see parallel_select.h).
            sylc_select::filter_ge(ema_.data(), confidence,
                                   static_cast<int>(n), 0.30f, worker_threads,
                                   tmp_, select_scratch_);
        } else {
            tmp_.reserve(n);
            for (size_t i = 0; i < n; ++i)
                if (confidence[i] >= 0.30f) tmp_.push_back(ema_[i]);
        }
    }
    if (tmp_.size() < std::max<size_t>(64, n / 20))
        tmp_.assign(ema_.begin(), ema_.end());
    const size_t tone_n = tmp_.size();
    const size_t lo_i = static_cast<size_t>(0.02 * static_cast<double>(tone_n - 1));
    const size_t hi_i = static_cast<size_t>(0.98 * static_cast<double>(tone_n - 1));
    const float conv_p = std::max(0.0f, std::min(
        1.0f, auto_convergence_percentile));
    const size_t conv_i = static_cast<size_t>(
        static_cast<double>(conv_p) * static_cast<double>(tone_n - 1));
    float sel_vals[3];
    if (parallel_sel) {
        const size_t sel_ranks[3] = {lo_i, hi_i, conv_i};
        sylc_select::select_ranks(tmp_.data(), static_cast<int>(tone_n),
                                  worker_threads, sel_ranks, 3, sel_vals,
                                  select_scratch_);
    } else {
        std::nth_element(tmp_.begin(),
                         tmp_.begin() + static_cast<ptrdiff_t>(lo_i),
                         tmp_.end());
        sel_vals[0] = tmp_[lo_i];
        std::nth_element(tmp_.begin(),
                         tmp_.begin() + static_cast<ptrdiff_t>(hi_i),
                         tmp_.end());
        sel_vals[1] = tmp_[hi_i];
        std::nth_element(tmp_.begin(),
                         tmp_.begin() + static_cast<ptrdiff_t>(conv_i),
                         tmp_.end());
        sel_vals[2] = tmp_[conv_i];
    }
    const float target_lo = sel_vals[0];
    const float target_hi = sel_vals[1];

    if (!tone_primed_ || cut) {
        tone_lo_ = target_lo;
        tone_hi_ = target_hi;
        tone_primed_ = true;
    } else {
        // Expand promptly to avoid clipping newly entering geometry; contract
        // slowly so a subject entering/leaving cannot make the whole scene
        // "breathe". Limit each update relative to the established range.
        const float established = std::max(1e-6f, tone_hi_ - tone_lo_);
        // Per-unit-time cap: the same VIDEO second may move the range
        // by the same amount whether it arrives as 8 maps or 19.
        const float max_step =
            established * 0.04f * std::min(video_ts, 4.0f);
        const float expand_eff =
            unit_time ? 0.20f
                      : 1.0f - std::pow(0.80f, video_ts);
        const float tone_eff =
            unit_time ? tone_alpha
                      : 1.0f - std::pow(1.0f - tone_alpha, video_ts);
        const float lo_rate = target_lo < tone_lo_ ? expand_eff : tone_eff;
        const float hi_rate = target_hi > tone_hi_ ? expand_eff : tone_eff;
        const float dlo = std::max(-max_step, std::min(
            max_step, lo_rate * (target_lo - tone_lo_)));
        const float dhi = std::max(-max_step, std::min(
            max_step, hi_rate * (target_hi - tone_hi_)));
        tone_lo_ += dlo;
        tone_hi_ += dhi;
    }

    const float range = tone_hi_ - tone_lo_;
    // Shared by the quantization below AND the auto-convergence suggestion:
    // both must live in the exact normalized nearness space the warp samples.
    auto normalize_tone = [&](float v) {
        float t = (v - tone_lo_) / range;
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
        // Monotonic S-curve: a little more separation between middle
        // planes, continuously, without changing ordering or consuming
        // the near/far headroom applied below.
        const float smooth = t * t * (3.0f - 2.0f * t);
        const float curve = std::max(0.0f, std::min(1.0f, depth_scurve));
        t += curve * (smooth - t);
        // Reserve headroom at both comfort extremes. Strength remains the
        // user's maximum budget; ordinary scenes no longer hit it merely
        // because their own percentiles were stretched to 0 and 1.
        t = 0.5f + depth_contrast * (t - 0.5f);
        return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    };

    // Auto-convergence suggestion, selected above in the same fused pass as
    // the tone percentiles (same confidence-filtered sample). The transform is
    // monotonic, so normalize_tone(percentile(sample)) IS the percentile of
    // the normalized map.
    {
        float target = 0.5f;
        if (range >= 1e-6f && tone_n > 0) {
            target = normalize_tone(sel_vals[2]);
        }
        if (!convergence_primed_ || cut) {
            auto_convergence_ = target;
            convergence_primed_ = true;
        } else {
            // Same video-time discipline as the tone range: identical
            // half-life whether the same second arrives as 8 maps or 19,
            // and a hard per-unit-time cap so the zero-plane cannot breathe.
            const float conv_eff =
                unit_time ? convergence_alpha
                          : 1.0f - std::pow(1.0f - convergence_alpha, video_ts);
            const float max_step = 0.05f * std::min(video_ts, 4.0f);
            const float delta = std::max(-max_step, std::min(
                max_step, conv_eff * (target - auto_convergence_)));
            auto_convergence_ += delta;
        }
        auto_convergence_ = std::max(0.0f, std::min(1.0f, auto_convergence_));
    }

    if (range < 1e-6f) {
        std::fill(out_q16, out_q16 + n, static_cast<uint16_t>(32768));
    } else {
        // Pure per-pixel write: no accumulation, no cross-pixel dependency, and
        // normalize_tone() reads only values that are already final. Each pixel
        // therefore takes the same operations in the same order whatever thread
        // runs it, so this is BITWISE identical to the sequential loop at any
        // worker_threads -- the guarantee the header makes for per-pixel
        // outputs, unlike the diagnostic reductions above.
        // It is not free work either: normalize_tone divides once per pixel.
        parallel_chunks(static_cast<int>(n), worker_threads,
                        [&](int, int i_begin, int i_end) {
            for (size_t i = static_cast<size_t>(i_begin);
                 i < static_cast<size_t>(i_end); ++i)
                out_q16[i] = static_cast<uint16_t>(
                    normalize_tone(ema_[i]) * 65535.0f + 0.5f);
        });
    }

    return cut;
}
