// synth3d.hlsl — depth prep, DIBR warp, depth view. Entry points:
//   VS_Full / PS_DepthPrep / PS_WarpLuma / PS_WarpChroma / PS_WarpProvenance
//   PS_DepthViewLuma / PS_DepthViewChroma
//
// Source of truth. CMake tracks this file, compiles every entry point with FXC,
// and embeds the resulting DXBC as RCDATA in mvc_demuxer_cpp.pyd.
Texture2D<float> SrcY : register(t0);
Texture2D<float> SrcU : register(t1);
Texture2D<float> SrcV : register(t2);
// RG16_UNORM geometry grid. The worker has already combined the conservative
// foreground-ownership proposal with the stabilized depth:
//   R effective depth, G effective stereo safety.
// This keeps the richer four-channel CPU analysis without doubling the
// texture bandwidth of every warp probe.
Texture2D<float2> Geometry : register(t3);
// Optional precomputed human matte (MatAnyone/MatAnyone 2 prototype). R is
// alpha; G is the CPU-computed horizontal distance to its nearest boundary in
// matte pixels, capped at 255. The distance field turns the many multi-scale
// alpha probes of the first prototype into one texture fetch.
Texture2D<float2> HumanMatte : register(t4);
// Round 5a — temporal background plate (grid resolution). RGB = flow-
// transported background YUV in SOURCE plane space (sampled raw from
// SrcY/U/V, so exact colors — no round trip through the prep RGB transform);
// A = accumulated confidence. Bound in the warp passes when temporal_fill=1.
Texture2D<float4> Plate : register(t5);
// Map-to-map transport for the accum pass: RG = flow in grid pixels
// quantized (v/128)+0.5, B = flow reliability, A = reserved.
Texture2D<float4> Transport : register(t6);
// Experimental one-frame reveal evidence. Future YUV uses the same source
// format/plane_scale as SrcY/U/V. LookaheadFlow is sampled at the visible
// background donor: RG=current->future motion in inference-grid pixels,
// B=NVOFA reliability, A reserved. These bindings are null and the cbuffer
// gate is zero unless SYLC_SYNTH3D_LOOKAHEAD=1 armed the current PTS.
Texture2D<float> FutureY : register(t7);
Texture2D<float> FutureU : register(t8);
Texture2D<float> FutureV : register(t9);
Texture2D<float4> LookaheadFlow : register(t10);
SamplerState linSmp : register(s0);

cbuffer SynthCB : register(b0) {
    float max_disp;      // c0.x  disparity budget, fraction of image WIDTH
    float convergence;   // c0.y  nearness at zero parallax (0..1)
    float plane_scale;   // c0.z  same semantics as the display shader
    float inv_w;         // c0.w  1 / luma plane width
    int   yuv_matrix_sel;// c1.x  0=BT.601, 1=BT.709, 2=BT.2020nc
    int   transfer_sel;  // c1.y  0=SDR, 2=PQ tonemap (for DepthPrep only)
    int   diagnostics;   // c1.z  depth/disocclusion overlay
    float edge_strength; // c1.w  joint-bilateral luma sensitivity
    float2 depth_texel;  // c2.xy 1 / depth-map dimensions
    float2 matte_texel;  // c2.zw 1 / alpha-matte dimensions
    float crop_top;      // c3.x  encoded top matte, normalized source height
    float crop_bottom;   // c3.y  encoded bottom matte
    float inv_h;         // c3.z  1 / luma plane height
    int matte_mode;      // c3.w  0=off, 1=guard, 2=alpha-aware contour
    int temporal_fill;   // c4.x  round 5a background plate on/off
    float plate_ceiling; // c4.y  nearness ceiling for plate refresh
    float far_snap_on;   // c4.z  v4 sub-texel background reclaim (1=on)
    float _pad4;         // c4.w
    // c5 is consumed by the downstream Stereo Lab through C++ parameters;
    // keeping its native layout explicit makes c6 unambiguous here.
    float comfort_soft_disp; // c5.x
    float comfort_hard_disp; // c5.y
    int comfort_enabled;     // c5.z
    float _pad5;             // c5.w
    int lookahead_fill;      // c6.x  future evidence valid for this exact PTS
    float lookahead_min_conf;// c6.y  conservative acceptance knee
    float lookahead_strength;// c6.z  maximum reveal blend
    float _pad6;             // c6.w
};

struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VSOut VS_Full(uint id : SV_VertexID) {
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uv;
    return o;
}

// ---- edge-aware upsampling -------------------------------------------------
// A plain bilinear sample blurs a depth edge from the inference grid over
// foreground/background pixels when the source is 1080p/4K. A joint-bilateral
// cross keeps the depth on the same side as the full-resolution luma edge when
// that edge lands ON a depth-grid texel boundary -- but a boundary that falls
// BETWEEN two texels (~5 image px apart at 4K on the 756 grid, ~7.4 on 518)
// still shows through: Nearness itself is sampled
// with a linear filter, so the CENTER sample n0 is already a blended,
// neither-side value there, and no amount of re-averaging nearby texels
// removes that contour halo (round 1's known residual, the third mechanism).
//
// v3: guided_nearness builds two candidates from one coherent 3x3 set of
// exact depth-texel samples. A is a joint depth/luma mean for clean gradients;
// B is an observed layer selected by guide luma and can never collapse to the
// bilinear semi-depth n0. Diagonal and one-texel-thin structures participate,
// while snapping is gated by actual depth spread plus image-edge support.
float source_luma(float2 uv) {
    return saturate(SrcY.SampleLevel(linSmp, uv, 0) * plane_scale);
}

float content_vscale() {
    return max(1e-5, 1.0 - crop_top - crop_bottom);
}

bool in_active_content(float2 uv) {
    return uv.y >= crop_top && uv.y <= (1.0 - crop_bottom);
}

float2 source_to_depth_uv(float2 uv) {
    return float2(uv.x, (uv.y - crop_top) / content_vscale());
}

float2 depth_to_source_uv(float2 uv) {
    return float2(uv.x, crop_top + uv.y * content_vscale());
}

float human_alpha(float2 uv) {
    if (matte_mode == 0 || !in_active_content(uv)) return 0.0;
    return HumanMatte.SampleLevel(linSmp, saturate(uv), 0).r;
}

// Fractional coverage catches soft hair/fur/cloth pixels; the symmetric
// gradient also catches a binary matte after bilinear resampling. This is a
// narrow ownership signal, not a blanket "all humans are flat" mask.
float human_boundary_score(float2 uv) {
    if (matte_mode == 0 || !in_active_content(uv)) return 0.0;
    float2 matte = HumanMatte.SampleLevel(linSmp, saturate(uv), 0);
    float a = matte.r;
    float distance_px = matte.g * 255.0;
    float wide_px = matte_mode >= 2 ? 7.0 : 4.5;
    // Horizontal inverse mapping may skip the literal alpha transition by up
    // to half the user's disparity budget. Scale that reach into the matte's
    // own pixels so full- and half-resolution mattes behave identically.
    float reach_px = max(
        wide_px,
        (matte_mode >= 2 ? 0.60 : 0.53) * max_disp /
            max(matte_texel.x, 1e-6));
    float fractional = 4.0 * a * (1.0 - a);
    float proximity = 1.0 - smoothstep(
        max(0.0, 0.72 * reach_px), reach_px + 1.5, distance_px);
    return saturate(max(fractional, 0.86 * proximity));
}

float effective_nearness(float2 geometry) {
    return geometry.r;
}

float median3(float a, float b, float c) {
    return max(min(a, b), min(max(a, b), c));
}

// Reject narrow FAR cracks while retaining supported foreground filaments.
// Nearness is deliberately max-combined across horizontal/vertical consensus:
// a two-texel model hole inside a face must not outrank the surrounding face,
// whereas a real silhouette still owns the side on which it has two samples.
float robust_grid_nearness(float2 depth_uv) {
    float2 center = (floor(depth_uv / depth_texel) + 0.5) * depth_texel;
    float c = effective_nearness(Geometry.SampleLevel(linSmp, center, 0));
    float xl1 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center - float2(depth_texel.x, 0.0)), 0));
    float xr1 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center + float2(depth_texel.x, 0.0)), 0));
    float yu1 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center - float2(0.0, depth_texel.y)), 0));
    float yd1 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center + float2(0.0, depth_texel.y)), 0));
    float xl2 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center - float2(2.0 * depth_texel.x, 0.0)), 0));
    float xr2 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center + float2(2.0 * depth_texel.x, 0.0)), 0));
    float yu2 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center - float2(0.0, 2.0 * depth_texel.y)), 0));
    float yd2 = effective_nearness(Geometry.SampleLevel(
        linSmp, saturate(center + float2(0.0, 2.0 * depth_texel.y)), 0));
    return max(max(median3(xl1, c, xr1), median3(yu1, c, yd1)),
               max(median3(xl2, c, xr2), median3(yu2, c, yd2)));
}

float ridge_axis(float center, float a, float b) {
    float da = center - a;
    float db = center - b;
    return da * db > 0.0 ? min(abs(da), abs(db)) : 0.0;
}

// Full-resolution ridge evidence. Unlike an ordinary one-sided silhouette,
// a utensil/hair/wire is brighter or darker than BOTH sides on at least one
// axis. Such a valid source preimage must never be replaced by hole filling.
float fine_structure_score(float2 uv) {
    float y0 = source_luma(uv);
    float2 sx = float2(1.5 * inv_w, 0.0);
    float2 sy = float2(0.0, 1.5 * inv_h);
    float horizontal = ridge_axis(
        y0, source_luma(saturate(uv - sx)), source_luma(saturate(uv + sx)));
    float vertical = ridge_axis(
        y0, source_luma(saturate(uv - sy)), source_luma(saturate(uv + sy)));
    return smoothstep(0.030, 0.13, max(horizontal, vertical));
}

float stereo_safety(float2 uv) {
    if (!in_active_content(uv)) return 1.0;
    float2 geometry = Geometry.SampleLevel(
        linSmp, saturate(source_to_depth_uv(uv)), 0);
    return geometry.g;
}

float guided_nearness(float2 uv) {
    // Encoded mattes remain at the convergence plane: zero disparity, no
    // warped matte/content boundary and no depth lookup outside the ROI.
    if (!in_active_content(uv)) return convergence;
    float2 depth_uv = saturate(source_to_depth_uv(uv));
    float y0 = source_luma(uv);
    float n0 = robust_grid_nearness(depth_uv);

    // One coherent 3x3 exact-texel neighbourhood supplies BOTH candidates:
    // A is a joint depth/luma mean for smooth gradients; B is a real sampled
    // layer, never the bilinear semi-depth at a contour. Reusing the same taps
    // costs roughly the old cross+four-corner implementation while observing
    // diagonal and one-texel-thin structures that the four-corner set missed.
    float2 center = (floor(depth_uv / depth_texel) + 0.5) * depth_texel;
    float matte_a0 = human_alpha(uv);
    float sum = 1.25 * n0;
    float weights = 1.25;
    float min_n = n0, max_n = n0;
    float edge_mag = 0.0;
    float best_score = 1e9;
    float B = n0;
    [unroll] for (int oy = -1; oy <= 1; ++oy) {
        [unroll] for (int ox = -1; ox <= 1; ++ox) {
            float2 tc = saturate(center + float2(ox, oy) * depth_texel);
            float nq = effective_nearness(
                Geometry.SampleLevel(linSmp, tc, 0));
            float2 source_tc = depth_to_source_uv(tc);
            float dl = abs(source_luma(source_tc) - y0);
            float dd = abs(nq - n0);
            float dm = matte_mode != 0 ?
                abs(human_alpha(source_tc) - matte_a0) : 0.0;
            float spatial = (ox == 0 && oy == 0) ? 1.0 :
                            ((ox != 0 && oy != 0) ? 0.48 : 0.76);
            // Alpha is an explicit layer-ownership observation. It therefore
            // rejects opposite-side depth taps more strongly than luma, whose
            // contrast may be weak or misleading at hair and fuzzy clothing.
            float matte_reject = matte_mode >= 2 ? 13.0 : 8.0;
            float w = spatial * exp2(
                -edge_strength * dl - 16.0 * dd - matte_reject * dm);
            sum += w * nq;
            weights += w;
            min_n = min(min_n, nq);
            max_n = max(max_n, nq);
            edge_mag = max(edge_mag, dl);

            // When two layers have near-identical guide luma, a tiny near-
            // surface tie break prevents the background from cutting a bright
            // hole into a foreground hair/hand. A real luma match (>~1 code
            // value) still dominates this deliberately small bias.
            float score = dl + (matte_mode >= 2 ? 0.16 : 0.10) * dm -
                          0.010 * nq +
                          0.0015 * (ox * ox + oy * oy);
            if (score < best_score) {
                best_score = score;
                B = nq;
            }
        }
    }
    float A = sum / max(weights, 1e-5);
    float depth_spread = max_n - min_n;
    float depth_edge = smoothstep(0.035, 0.17, depth_spread);
    float guide_edge = smoothstep(0.025, 0.16, edge_mag);
    // A minority NEAR sample can be a hair, finger or utensil and is allowed
    // to win through guide agreement. A minority FAR sample is a likely model
    // crack; it needs spatial consensus and therefore remains in robust n0/A.
    float near_or_consensus = smoothstep(-0.035, 0.005, B - n0);
    float snap = depth_edge * guide_edge * near_or_consensus;
    float result = lerp(A, B, snap);

    // v4 (04/08) — sub-texel background reclaim. The CPU contour re-anchor
    // (realign_contours) fixes the map at GRID scale; what remains is the
    // pixel band inside the edge texel's footprint: background pixels
    // carrying the foreground's depth (the residual fringe on cap brims and
    // profiles). No local color model can arbitrate — the edge texel's own
    // CENTER luma is already background-like — so the CPU cure transposes
    // one scale down: representatives anchored BEYOND the contested texel
    // (±2 texels, epipolar direction, clean at grid scale by the CPU pass),
    // and the pixel's own full-resolution luma votes between them. Guards:
    // a real layer separation between the reps, reclaim pulls toward FAR
    // only, full-res ridges (hair/wire) and the human matte veto it.
    // SYLC_SYNTH3D_FAR_SNAP=0 disables (cb flag).
    float gate = depth_edge * guide_edge;
    if (far_snap_on > 0.5 && gate > 0.05) {
        float2 lc = saturate(center - float2(2.0 * depth_texel.x, 0.0));
        float2 rc = saturate(center + float2(2.0 * depth_texel.x, 0.0));
        float nl = effective_nearness(Geometry.SampleLevel(linSmp, lc, 0));
        float nr = effective_nearness(Geometry.SampleLevel(linSmp, rc, 0));
        if (abs(nl - nr) >= 0.10) {
            // Layer coherence — the CPU pass's anchored-median principle in
            // its cheapest sufficient form: the far side must present TWO
            // agreeing taps at +2 and +3 texels. A 1-2 texel model crack
            // (possibly luma-aligned with a facial feature) never can; a
            // genuine background layer always does.
            float dir = nl < nr ? -1.0 : 1.0;
            float n2 = min(nl, nr);
            float2 c3 = saturate(
                center + float2(dir * 3.0 * depth_texel.x, 0.0));
            float n3 = effective_nearness(
                Geometry.SampleLevel(linSmp, c3, 0));
            if (abs(n2 - n3) < 0.10) {
                float y_far = source_luma(depth_to_source_uv(c3));
                float y_near = source_luma(depth_to_source_uv(
                    dir < 0.0 ? rc : lc));
                float vote = smoothstep(0.02, 0.08,
                                        abs(y0 - y_near) - abs(y0 - y_far));
                float human_guard = 1.0 -
                    smoothstep(0.20, 0.60, matte_a0);
                float reclaim = gate * vote * human_guard *
                                (1.0 - fine_structure_score(uv)) *
                                smoothstep(0.02, 0.06, result - n2);
                result = lerp(result, n2, reclaim);
            }
        }
    }
    return result;
}

// ---- round 5a: temporal background plate ----------------------------------
// The plate lives in depth-grid space (same ROI as Geometry). Trust rises
// smoothly with accumulated confidence so a young plate defers to the
// stretch fallback instead of flashing half-remembered content.
float4 plate_sample(float2 source_uv) {
    return Plate.SampleLevel(
        linSmp, saturate(source_to_depth_uv(source_uv)), 0);
}

float plate_trust(float2 source_uv) {
    if (temporal_fill == 0 || !in_active_content(source_uv)) return 0.0;
    return smoothstep(0.30, 0.80, plate_sample(source_uv).a);
}

// PS_PlateAccum — grid-resolution ping-pong accumulation. i.uv IS depth-grid
// space (the RT is the plate). Refresh where the CURRENT texel shows stable,
// clearly-background content (nearness under the ceiling, safety high — the
// safety channel already collapses at contested contours); elsewhere carry
// the flow-transported memory so a passing foreground never contaminates it.
float4 PS_PlateAccum(VSOut i) : SV_Target {
    float4 transport = Transport.SampleLevel(linSmp, i.uv, 0);
    float2 flow_px = (transport.rg - 0.5) * 128.0;
    float rel = transport.b;
    float2 prev_uv = i.uv - flow_px * depth_texel;
    bool oob = any(prev_uv != saturate(prev_uv));
    float4 prev = Plate.SampleLevel(linSmp, saturate(prev_uv), 0);
    // Unreliable transport decays the memory rather than trusting a wrong
    // displacement; leaving the frame discards it outright.
    float carried = oob ? 0.0 : prev.a * lerp(0.55, 1.0, rel);

    float2 src_uv = depth_to_source_uv(i.uv);
    float3 current = float3(SrcY.SampleLevel(linSmp, src_uv, 0),
                            SrcU.SampleLevel(linSmp, src_uv, 0),
                            SrcV.SampleLevel(linSmp, src_uv, 0));
    float2 geometry = Geometry.SampleLevel(linSmp, i.uv, 0);
    float bg = (1.0 - smoothstep(plate_ceiling - 0.06, plate_ceiling + 0.04,
                                 geometry.r))
             * smoothstep(0.45, 0.80, geometry.g);

    float refresh = 0.30 * bg;
    float3 color = (carried <= 0.001) ? current
                                      : lerp(prev.rgb, current, refresh);
    // Foreground overhead (bg=0): the memory persists almost untouched —
    // that persistence IS the feature. Background in view: confidence rises.
    float alpha = saturate(carried * (bg > 0.001 ? 1.0 : 0.995) + refresh);
    return float4(color, alpha);
}

// ---- diagnostic palette (shared by full depth-view and live overlay) -------
float3 falseColor(float n) {
    float t = 0.125 + 0.75 * saturate(n);
    return saturate(float3(1.5 - abs(4.0 * t - 3.0),
                           1.5 - abs(4.0 * t - 2.0),
                           1.5 - abs(4.0 * t - 1.0)));
}
float3 rgb_to_yuv709_full(float3 c) {
    float y = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    return float3(y * (219.0 / 255.0) + 16.0 / 255.0,
                  (c.b - y) / 1.8556 * (224.0 / 255.0) + 0.5,
                  (c.r - y) / 1.5748 * (224.0 / 255.0) + 0.5);
}
float3 diagnostic_yuv(float nearness, float fill, float plate_used) {
    // Depth is a quiet translucent turbo ramp. Pixels reconstructed from the
    // background turn progressively warm, making the disocclusion mask visible
    // without replacing the movie with a debug-only view. Round 5a: holes
    // served from the temporal plate shift toward cyan so the author's A/B
    // can tell remembered background apart from the stretch fallback.
    float3 depth_rgb = falseColor(nearness);
    float3 fill_rgb = lerp(float3(1.0, 0.12, 0.02), float3(0.10, 0.85, 1.0),
                           saturate(plate_used));
    float3 mixed = lerp(depth_rgb, fill_rgb, saturate(fill));
    // Zero disparity budget = NO active 3D knowledge (pre-first-map, and the
    // flat hold through a cut): fade the overlay to neutral gray so a stale
    // cross-shot map can never paint its color flats onto the first frames
    // of the new shot (author report 2026-08-03 — the residue was invisible
    // in the movie, max_disp=0 warps nothing, but diagnostics showed it).
    float knowledge = saturate(max_disp / 0.0005);
    return rgb_to_yuv709_full(lerp(float3(0.5, 0.5, 0.5), mixed, knowledge));
}

// ---- signed disparity (image-width fraction) at a source-space uv ----
float comfortable_depth_offset(float nearness) {
    float d = nearness - convergence;
    float span = d >= 0.0 ? max(0.08, 1.0 - convergence)
                          : max(0.08, convergence);
    float z = saturate(abs(d) / span);
    // Crossed (near) disparity receives a slightly earlier knee than far
    // disparity. The curve is identity through ordinary mid-depths and only
    // compresses the extreme tail; at z=1 it retains ~89% of the user's
    // requested maximum, so QUALITY keeps its impact without hard clipping.
    float knee = d >= 0.0 ? 0.68 : 0.76;
    float over = max(0.0, z - knee);
    float softened = z <= knee ? z : knee + over / (1.0 + 1.60 * over);
    return (d < 0.0 ? -1.0 : 1.0) * softened * span;
}

float disp_for_nearness(float2 uv, float nearness) {
    float safety = stereo_safety(uv);
    float boundary = 0.0;
    if (matte_mode != 0) {
        boundary = human_boundary_score(uv);
        // Compute proximity once and reuse it for both the established safety
        // response and the explicit human-contour disparity budget.
        float floor_scale = matte_mode >= 2 ? 0.035 : 0.24;
        safety *= lerp(1.0, floor_scale, boundary);
    }
    // Perceptual safety valve. It is vastly less objectionable to flatten an
    // unresolved contour locally than to present different, torn anatomy to
    // the two eyes. The smooth response prevents a visible disparity seam.
    const float safe_budget = lerp(
        0.08, 1.0, smoothstep(0.22, 0.88, safety));
    float matte_budget = 1.0;
    if (matte_mode != 0) {
        matte_budget = lerp(
            1.0, matte_mode >= 2 ? 0.08 : 0.36, boundary);
    }
    return max_disp * safe_budget * matte_budget *
           comfortable_depth_offset(nearness);
}

float disp_at(float2 uv) {
    return disp_for_nearness(uv, guided_nearness(uv));
}

struct WarpInfo {
    float2 uv;         // blended coordinate for diagnostics/depth lookup
    float2 destination_uv;
    float2 base_uv;    // inverse-map solution before hole filling
    float2 background_uv;
    float fill;       // continuous confidence that background pull is required
    float nearness;
    float background_dir;
    float matte_boundary;
};

// Backward DIBR: find src x solving x_s = x_d - eyeSign * disp(x_s)/2.
// eyeSign follows the subtitle convention (VERIFY in Task 5): +1 = left eye.
// Three fixed-point iterations are followed by a continuous residual mask.
// Across a depth edge, the farther probe supplies a background coordinate and
// smoothstep blends toward it over roughly 1–6 luma pixels. This avoids both
// the hard seam of the old binary branch and the double image of an indiscriminate
// blur: only pixels with an unsatisfied inverse mapping receive the pull.
//
// inv_w is ALWAYS the LUMA plane's 1/width, including in the chroma pass: the
// disocclusion probe (+-8 texels) is a search radius in DISPARITY space, which
// is defined against the full-resolution image, and warp_src works in
// normalized uv so the identical function serves both plane resolutions.
WarpInfo warp_info(float2 uvd, float eyeSign) {
    float xs = uvd.x;
    [unroll] for (int i = 0; i < 3; ++i)
        xs = uvd.x - eyeSign * 0.5 * disp_at(float2(xs, uvd.y));
    float base_x = saturate(xs);
    float base_n = guided_nearness(float2(base_x, uvd.y));
    float base_forward = base_x + eyeSign * 0.5 *
                         disp_for_nearness(float2(base_x, uvd.y), base_n);
    float visible_error = abs(base_forward - uvd.x);
    float visible_score = visible_error - 3.0 * inv_w * base_n;
    float initial_n = base_n;
    bool occluder_selected = false;

    // One shared epipolar search serves two distinct decisions:
    //   1. candidates are forward-projected and the closest valid layer wins;
    //      this retains hair, fingers and other structures too thin for a
    //      fixed-point inverse iteration to land on by chance;
    //   2. supported side-pairs identify the immediately following layer for
    //      a hole -- never an isolated far outlier.
    float probe = clamp(max_disp, 8.0 * inv_w, 64.0 * inv_w);
    float background_x = base_x;
    float background_dir = eyeSign;
    float min_probe_n = base_n, max_probe_n = base_n;
    float coarse_n[6];
    float coarse_x[6];
    // The inner +/-0.18 probes land near the half-disparity of ordinary
    // foregrounds and catch one-pixel structures skipped by the old +/-0.25.
    const float offsets[6] = {-0.5, -0.25, -0.18, 0.18, 0.25, 0.5};
    [unroll] for (int k = 0; k < 6; ++k) {
        float qx = saturate(uvd.x + offsets[k] * probe);
        float nq = guided_nearness(float2(qx, uvd.y));
        const bool fine_probe = (k == 2 || k == 3);
        if (fine_probe) {
            // The first fine tap only has to notice the filament. Once it
            // does, solve its exact source coordinate from that layer so one
            // source line cannot be accepted by a 7-pixel destination band.
            qx = saturate(uvd.x - eyeSign * 0.5 *
                          disp_for_nearness(float2(qx, uvd.y), nq));
            nq = guided_nearness(float2(qx, uvd.y));
        }
        coarse_n[k] = nq;
        coarse_x[k] = qx;
        float projected_x = qx + eyeSign * 0.5 *
                            disp_for_nearness(float2(qx, uvd.y), nq);
        float candidate_error = abs(projected_x - uvd.x);
        float candidate_score = candidate_error - 3.0 * inv_w * nq;
        float candidate_limit = fine_probe ? 1.25 * inv_w : 3.5 * inv_w;
        if (candidate_error <= candidate_limit &&
            candidate_score < visible_score) {
            occluder_selected = nq > initial_n + 0.025;
            visible_score = candidate_score;
            visible_error = candidate_error;
            base_x = qx;
            base_n = nq;
        }
        min_probe_n = min(min_probe_n, nq);
        max_probe_n = max(max_probe_n, nq);
    }

    // Background requires a coherent pair on one side. A lone far outlier is
    // never evidence, and when several layers exist the closest supported
    // layer behind the local front wins instead of the globally farthest one.
    float front_n = max(base_n, max_probe_n);
    float left_n = 0.5 * (coarse_n[0] + coarse_n[1]);
    float right_n = 0.5 * (coarse_n[4] + coarse_n[5]);
    float left_support = 1.0 - smoothstep(
        0.025, 0.085, abs(coarse_n[0] - coarse_n[1]));
    float right_support = 1.0 - smoothstep(
        0.025, 0.085, abs(coarse_n[4] - coarse_n[5]));
    float left_gap = front_n - left_n;
    float right_gap = front_n - right_n;
    float left_evidence = left_support * smoothstep(0.025, 0.11, left_gap);
    float right_evidence = right_support * smoothstep(0.025, 0.11, right_gap);
    float layer_evidence = 0.0;
    float chosen_background_n = -1.0;
    if (left_evidence > 0.01) {
        background_x = coarse_x[1];
        background_dir = -1.0;
        chosen_background_n = left_n;
        layer_evidence = left_evidence;
    }
    if (right_evidence > 0.01 && right_n > chosen_background_n) {
        background_x = coarse_x[4];
        background_dir = 1.0;
        chosen_background_n = right_n;
        layer_evidence = right_evidence;
    }

    // Gradient-aware parallax visibility. The forward map is
    // xd(xs)=xs+eyeSign*disp(xs)/2. Its Jacobian expands above 1 in
    // disocclusions and folds toward/below 0 at competing foreground samples.
    const float grad_step = 1.5;
    float dl = disp_at(float2(saturate(base_x - grad_step * inv_w), uvd.y));
    float dr = disp_at(float2(saturate(base_x + grad_step * inv_w), uvd.y));
    float ddisp_dx = (dr - dl) / max(2.0 * grad_step * inv_w, 1e-6);
    float jacobian = 1.0 + eyeSign * 0.5 * ddisp_dx;
    float residual_mask = smoothstep(1.0 * inv_w, 6.0 * inv_w,
                                     visible_error);
    float edge_mask = smoothstep(0.025, 0.18, max_probe_n - min_probe_n);
    float expansion_mask = smoothstep(1.04, 1.55, jacobian);
    float fold_mask = 1.0 - smoothstep(0.05, 0.55, jacobian);
    float visibility_mask = max(expansion_mask, fold_mask);
    float fill = max(residual_mask, visibility_mask) *
                 (0.25 + 0.75 * edge_mask);
    // If the forward check found a nearer valid preimage, it is the visible
    // occluder, not a hole. Pulling background here is precisely what erased
    // one-texel hair and wires in the old purely backward solution.
    if (occluder_selected) fill = 0.0;
    // No supported rear layer means no hallucinated background. Full-source
    // ridges get an additional veto so a valid utensil/hair preimage survives
    // even at the edge of a genuine disocclusion.
    fill *= layer_evidence;
    float structure_validity = 1.0 - smoothstep(
        1.0 * inv_w, 2.5 * inv_w, visible_error);
    fill *= 1.0 - 0.96 * structure_validity *
            fine_structure_score(float2(base_x, uvd.y));
    if (matte_mode != 0) {
        float2 human_uv = float2(base_x, uvd.y);
        float alpha = human_alpha(human_uv);
        float boundary = human_boundary_score(human_uv);
        // A coherent alpha-covered preimage owns the contour even when the
        // monocular depth erodes it. A genuinely uncovered destination has a
        // large forward residual, so the gate releases and background fill is
        // still allowed behind the moving person.
        float visible_human = max(alpha, boundary) *
            (1.0 - smoothstep(1.5 * inv_w, 5.0 * inv_w, visible_error));
        float veto = matte_mode >= 2 ? 0.985 : 0.90;
        fill *= 1.0 - veto * visible_human;
    }
    // Four-to-eight-pixel feather in disparity space: the Jacobian supplies
    // accurate localization while this soft knee prevents a binary seam.
    fill = smoothstep(0.08, 0.92, saturate(fill));
    background_x = saturate(background_x);
    float blended_x = lerp(base_x, background_x, fill);

    WarpInfo o;
    o.uv = float2(blended_x, uvd.y);
    o.destination_uv = uvd;
    o.base_uv = float2(base_x, uvd.y);
    o.background_uv = float2(background_x, uvd.y);
    o.fill = fill;
    o.nearness = guided_nearness(o.uv);
    o.background_dir = background_dir;
    o.matte_boundary = human_boundary_score(o.base_uv);
    return o;
}

// Sidecar metadata for Stereo Lab. With lookahead_fill=0 the luma/chroma path
// remains the byte-exact v5.2.1c image-forming reference. This independent pass
// evaluates the SAME inverse warp and exports only its source ownership, so
// reconnecting the Lab cannot change one raw c pixel.
//
// R32_UINT layout:
//   0..15  inverse-map base x (UNORM16)
//  16..23  signed background delta in luma pixels, biased by 128
//  24..30  fill confidence (UNORM7)
//       31 background direction (0=left, 1=right)
uint pack_warp_provenance(WarpInfo w) {
    uint base_q = (uint)round(saturate(w.base_uv.x) * 65535.0);
    float delta_px = (w.background_uv.x - w.base_uv.x) /
                     max(inv_w, 1e-8);
    int delta_i = (int)round(clamp(delta_px, -127.0, 127.0));
    uint delta_q = (uint)(delta_i + 128);
    uint fill_q = (uint)round(saturate(w.fill) * 127.0);
    uint dir_q = w.background_dir >= 0.0 ? 1u : 0u;
    return base_q | (delta_q << 16) | (fill_q << 24) | (dir_q << 31);
}

float matte_background_direction(float2 uv, float fallback_dir) {
    float step_x = max(matte_texel.x, inv_w);
    float left = human_alpha(saturate(uv - float2(2.0 * step_x, 0.0)));
    float right = human_alpha(saturate(uv + float2(2.0 * step_x, 0.0)));
    return abs(left - right) > 0.015 ? (right < left ? 1.0 : -1.0) : fallback_dir;
}

float estimate_matte_background_luma(float2 uv, float dir, float fallback) {
    float sum = 0.0, weights = 0.0;
    const float steps[4] = {2.0, 4.0, 7.0, 11.0};
    [unroll] for (int k = 0; k < 4; ++k) {
        float2 q = saturate(uv + float2(dir * steps[k] * inv_w, 0.0));
        float outside = 1.0 - human_alpha(q);
        float weight = outside * outside * (1.0 - 0.12 * k);
        sum += weight * SrcY.SampleLevel(linSmp, q, 0);
        weights += weight;
    }
    return weights > 0.08 ? sum / weights : fallback;
}

float2 estimate_matte_background_chroma(float2 uv, float dir, float2 fallback) {
    float2 sum = 0.0;
    float weights = 0.0;
    const float steps[4] = {2.0, 4.0, 7.0, 11.0};
    [unroll] for (int k = 0; k < 4; ++k) {
        float2 q = saturate(uv + float2(dir * steps[k] * inv_w, 0.0));
        float outside = 1.0 - human_alpha(q);
        float weight = outside * outside * (1.0 - 0.12 * k);
        sum += weight * float2(
            SrcU.SampleLevel(linSmp, q, 0),
            SrcV.SampleLevel(linSmp, q, 0));
        weights += weight;
    }
    return weights > 0.08 ? sum / weights : fallback;
}

// Ask t+1 for a pixel only after the ordinary inverse warp has already found
// a disocclusion. Motion is measured on the visible background donor because
// no current-frame flow exists inside the hole itself; extending that local
// motion to the adjacent destination is the same contract as the CPU A/B
// reference. The photo-consistency guard makes a still-visible foreground a
// rejection, not a temporal smear. Return RGB=raw future YUV, A=blend trust.
float4 lookahead_candidate(WarpInfo w, float3 donor_yuv) {
    if (lookahead_fill == 0 || w.fill <= 0.001)
        return 0.0;
    float4 flow = LookaheadFlow.SampleLevel(
        linSmp, saturate(w.background_uv), 0);
    float2 motion_uv = flow.xy * depth_texel;
    float2 future_uv = w.destination_uv + motion_uv;
    float inside = (future_uv.x >= 0.0 && future_uv.x <= 1.0 &&
                    future_uv.y >= 0.0 && future_uv.y <= 1.0 &&
                    in_active_content(future_uv)) ? 1.0 : 0.0;
    // Gross vectors are almost always a cost outlier/cut. Fast pans validated
    // on Oblivion remain far below this 16%-of-frame ceiling.
    float plausible_motion = 1.0 - smoothstep(
        0.12, 0.16, length(motion_uv));
    float3 candidate = float3(
        FutureY.SampleLevel(linSmp, saturate(future_uv), 0),
        FutureU.SampleLevel(linSmp, saturate(future_uv), 0),
        FutureV.SampleLevel(linSmp, saturate(future_uv), 0));
    float luma_delta = abs(candidate.x - donor_yuv.x) * plane_scale;
    float chroma_delta = length(candidate.yz - donor_yuv.yz) * plane_scale;
    // exp(-delta/0.12), expressed as exp2 for Shader Model 5. The same sigma
    // and 0.24 knee produced 43.8% safe hole coverage on the moving real-film
    // benchmark; below the knee the established spatial/plate fill is exact.
    float colour_conf = exp2(-12.02 * (luma_delta + 0.45 * chroma_delta));
    float confidence = saturate(flow.z) * colour_conf * inside * plausible_motion;
    float trust = smoothstep(
        lookahead_min_conf, 0.80, confidence) * lookahead_strength;
    return float4(candidate, saturate(trust));
}

float reconstruct_luma(WarpInfo w) {
    float primary = SrcY.SampleLevel(linSmp, w.base_uv, 0);
    float background = primary;
    float2 step_uv = float2(w.background_dir * inv_w, 0.0);
    if (w.fill > 0.001) {
        background =
            0.58 * SrcY.SampleLevel(linSmp, w.background_uv, 0) +
            0.28 * SrcY.SampleLevel(linSmp,
                saturate(w.background_uv + 1.5 * step_uv), 0) +
            0.14 * SrcY.SampleLevel(linSmp,
                saturate(w.background_uv + 3.5 * step_uv), 0);
    }
    // Round 5a: a trusted temporal plate replaces the stretch estimate with
    // background that was actually SEEN — the fill weight itself and every
    // veto upstream (fine structures, human contour) remain untouched.
    if (w.fill > 0.001) {
        float trust = plate_trust(w.destination_uv);
        if (trust > 0.001)
            background = lerp(background,
                              plate_sample(w.destination_uv).r, trust);
    }
    if (w.fill > 0.001) {
        float3 donor = float3(
            SrcY.SampleLevel(linSmp, w.background_uv, 0),
            SrcU.SampleLevel(linSmp, w.background_uv, 0),
            SrcV.SampleLevel(linSmp, w.background_uv, 0));
        float4 reveal = lookahead_candidate(w, donor);
        background = lerp(background, reveal.x, reveal.a);
    }
    float historical = lerp(primary, background, w.fill);
    if (matte_mode < 2 || w.matte_boundary <= 0.001) return historical;

    float alpha = human_alpha(w.base_uv);
    float fractional = smoothstep(0.015, 0.22, alpha) *
                       (1.0 - smoothstep(0.82, 0.995, alpha));
    float blend = w.matte_boundary * fractional;
    if (blend <= 0.001) return historical;

    float dir = matte_background_direction(w.base_uv, w.background_dir);
    float source_bg = estimate_matte_background_luma(
        w.base_uv, dir, background);
    float destination_bg = estimate_matte_background_luma(
        w.destination_uv, dir, background);
    float channel_max = 1.0 / max(plane_scale, 1.0);
    float foreground = clamp(
        (primary - (1.0 - alpha) * source_bg) / max(alpha, 0.06),
        0.0, channel_max);
    float layered = lerp(destination_bg, foreground, alpha);
    layered = lerp(layered, background, w.fill);
    return lerp(historical, layered, blend);
}

float2 reconstruct_chroma(WarpInfo w) {
    float2 primary = float2(
        SrcU.SampleLevel(linSmp, w.base_uv, 0),
        SrcV.SampleLevel(linSmp, w.base_uv, 0));
    float2 background = primary;
    // One chroma texel spans two luma texels in 4:2:0, but UV coordinates are
    // normalized identically. Move deeper into the chosen background layer.
    float2 step_uv = float2(w.background_dir * 2.0 * inv_w, 0.0);
    if (w.fill > 0.001) {
        background =
            0.58 * float2(
                SrcU.SampleLevel(linSmp, w.background_uv, 0),
                SrcV.SampleLevel(linSmp, w.background_uv, 0)) +
            0.28 * float2(
                SrcU.SampleLevel(linSmp,
                    saturate(w.background_uv + 1.0 * step_uv), 0),
                SrcV.SampleLevel(linSmp,
                    saturate(w.background_uv + 1.0 * step_uv), 0)) +
            0.14 * float2(
                SrcU.SampleLevel(linSmp,
                    saturate(w.background_uv + 2.0 * step_uv), 0),
                SrcV.SampleLevel(linSmp,
                    saturate(w.background_uv + 2.0 * step_uv), 0));
    }
    // Round 5a: same plate substitution as the luma pass (channels G/B).
    if (w.fill > 0.001) {
        float trust = plate_trust(w.destination_uv);
        if (trust > 0.001)
            background = lerp(background,
                              plate_sample(w.destination_uv).gb, trust);
    }
    if (w.fill > 0.001) {
        float3 donor = float3(
            SrcY.SampleLevel(linSmp, w.background_uv, 0),
            SrcU.SampleLevel(linSmp, w.background_uv, 0),
            SrcV.SampleLevel(linSmp, w.background_uv, 0));
        float4 reveal = lookahead_candidate(w, donor);
        background = lerp(background, reveal.yz, reveal.a);
    }
    float2 historical = lerp(primary, background, w.fill);
    if (matte_mode < 2 || w.matte_boundary <= 0.001) return historical;

    float alpha = human_alpha(w.base_uv);
    float fractional = smoothstep(0.015, 0.22, alpha) *
                       (1.0 - smoothstep(0.82, 0.995, alpha));
    float blend = w.matte_boundary * fractional;
    if (blend <= 0.001) return historical;

    float dir = matte_background_direction(w.base_uv, w.background_dir);
    float2 source_bg = estimate_matte_background_chroma(
        w.base_uv, dir, background);
    float2 destination_bg = estimate_matte_background_chroma(
        w.destination_uv, dir, background);
    float channel_max = 1.0 / max(plane_scale, 1.0);
    float2 foreground = clamp(
        (primary - (1.0 - alpha) * source_bg) / max(alpha, 0.06),
        0.0, channel_max);
    float2 layered = lerp(destination_bg, foreground, alpha);
    layered = lerp(layered, background, w.fill);
    return lerp(historical, layered, blend);
}

struct Eyes2 { float L : SV_Target0; float R : SV_Target1; };
Eyes2 PS_WarpLuma(VSOut i) {
    WarpInfo wl = warp_info(i.uv, +1.0);
    WarpInfo wr = warp_info(i.uv, -1.0);
    Eyes2 o;
    o.L = reconstruct_luma(wl);
    o.R = reconstruct_luma(wr);
    // The diagnostic palette describes synthesized content. Encoded mattes
    // are deliberately outside the depth ROI and must remain byte-identical
    // black; tinting them made a correctly detected crop look like a miss.
    if (diagnostics != 0 && in_active_content(i.uv)) {
        float pl = plate_trust(wl.destination_uv);
        float pr = plate_trust(wr.destination_uv);
        float3 dl = diagnostic_yuv(wl.nearness, wl.fill, pl);
        float3 dr = diagnostic_yuv(wr.nearness, wr.fill, pr);
        o.L = lerp(o.L, dl.x / plane_scale, 0.18 + 0.57 * wl.fill);
        o.R = lerp(o.R, dr.x / plane_scale, 0.18 + 0.57 * wr.fill);
    }
    return o;
}

struct Provenance2 {
    uint L : SV_Target0;
    uint R : SV_Target1;
};
Provenance2 PS_WarpProvenance(VSOut i) {
    Provenance2 o;
    o.L = pack_warp_provenance(warp_info(i.uv, +1.0));
    o.R = pack_warp_provenance(warp_info(i.uv, -1.0));
    return o;
}

struct Eyes4 { float UL : SV_Target0; float VL : SV_Target1;
               float UR : SV_Target2; float VR : SV_Target3; };
Eyes4 PS_WarpChroma(VSOut i) {
    WarpInfo wl = warp_info(i.uv, +1.0);
    WarpInfo wr = warp_info(i.uv, -1.0);
    Eyes4 o;
    float2 cl = reconstruct_chroma(wl);
    float2 cr = reconstruct_chroma(wr);
    o.UL = cl.x; o.VL = cl.y;
    o.UR = cr.x; o.VR = cr.y;
    if (diagnostics != 0 && in_active_content(i.uv)) {
        float pl = plate_trust(wl.destination_uv);
        float pr = plate_trust(wr.destination_uv);
        float3 dl = diagnostic_yuv(wl.nearness, wl.fill, pl);
        float3 dr = diagnostic_yuv(wr.nearness, wr.fill, pr);
        float al = 0.18 + 0.57 * wl.fill, ar = 0.18 + 0.57 * wr.fill;
        o.UL = lerp(o.UL, dl.y / plane_scale, al);
        o.VL = lerp(o.VL, dl.z / plane_scale, al);
        o.UR = lerp(o.UR, dr.y / plane_scale, ar);
        o.VR = lerp(o.VR, dr.z / plane_scale, ar);
    }
    return o;
}

// ---- depth prep: YUV -> normalized RGB for the model (RGBA32F side x side) ----
float3 yuv_to_rgb_prep(float y, float u, float v) {
    y = (y * plane_scale - 16.0 / 255.0) * 1.16438353;
    u = u * plane_scale - 0.5; v = v * plane_scale - 0.5;
    float3 rgb;
    if (yuv_matrix_sel == 2)      rgb = float3(y + 1.4746 * v,
        y - 0.16455 * u - 0.57135 * v, y + 1.8814 * u);
    else if (yuv_matrix_sel == 1) rgb = float3(y + 1.5748 * v,
        y - 0.18732 * u - 0.46812 * v, y + 1.8556 * u);
    else                          rgb = float3(y + 1.402 * v,
        y - 0.344136 * u - 0.714136 * v, y + 1.772 * u);
    if (transfer_sel == 2) rgb = rgb / (1.0 + rgb);      // cheap PQ-ish rolloff
    return saturate(rgb);
}
float4 PS_DepthPrep(VSOut i) : SV_Target {
    float2 source_uv = depth_to_source_uv(i.uv);
    float3 rgb = yuv_to_rgb_prep(SrcY.SampleLevel(linSmp, source_uv, 0),
                                 SrcU.SampleLevel(linSmp, source_uv, 0),
                                 SrcV.SampleLevel(linSmp, source_uv, 0));
    // RGB follows the active crop and feeds the depth graph. Alpha remains an
    // uncropped, limited-range-normalized luma probe. It rides through the
    // existing staging readback for free, allowing the aspect detector to see
    // a Scope/IMAX transition even while a rectangular graph is active.
    float full_luma = saturate(
        (SrcY.SampleLevel(linSmp, i.uv, 0) * plane_scale - 16.0 / 255.0) *
        1.16438353);
    return float4(rgb, full_luma); // ImageNet mean/std applied to RGB CPU-side
}

// ---- depth view: turbo-ish false color, written straight as YUV planes ----
Eyes2 PS_DepthViewLuma(VSOut i) {
    float n = guided_nearness(i.uv);
    Eyes2 o; o.L = o.R = rgb_to_yuv709_full(falseColor(n)).x / plane_scale;
    return o;
}
Eyes4 PS_DepthViewChroma(VSOut i) {
    float n = guided_nearness(i.uv);
    float3 yuv = rgb_to_yuv709_full(falseColor(n));
    Eyes4 o;
    o.UL = o.UR = yuv.y / plane_scale; o.VL = o.VR = yuv.z / plane_scale;
    return o;
}
