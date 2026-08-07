// SyLC Stereo Lab — final binocular coherence guard.
//
// The existing Synth3D warp remains the source of truth and writes RawY/U/V
// plus one packed full-luma provenance map per eye.  This post-pass only acts
// where two non-occluded pixels claim to be the same cyclopean source point
// but fail the reciprocal left/right mapping test.  In those pixels it moves
// both views locally toward the untouched centre view.  A separate sparse
// branch handles disocclusions: it only replaces foreground-contaminated fill
// when the conjugate eye certifies one source-space background node, or when
// the untouched centre view supplies an equally background-like retreat.

Texture2D<float> RawYL : register(t0);
Texture2D<float> RawUL : register(t1);
Texture2D<float> RawVL : register(t2);
Texture2D<float> RawYR : register(t3);
Texture2D<float> RawUR : register(t4);
Texture2D<float> RawVR : register(t5);
Texture2D<uint> ProvenanceL : register(t6);
Texture2D<uint> ProvenanceR : register(t7);
Texture2D<float> SourceY : register(t8);
Texture2D<float> SourceU : register(t9);
Texture2D<float> SourceV : register(t10);
Texture2D<float2> GuardMap : register(t11);
Texture2D<float4> PairField : register(t12);
SamplerState linSmp : register(s0);

cbuffer LabCB : register(b0) {
    float inv_w;       // 1 / full-resolution luma width
    float inv_h;       // 1 / full-resolution luma height
    float plane_scale; // R16 low-bit content normalization, 1 for R8
    float lab_gain;    // maximum local convergence toward the centre view
    float max_disp;    // current artistic disparity ceiling, width fraction
    float comfort_soft_disp;
    float comfort_hard_disp;
    float comfort_enabled;
    float pair_inv_w;       // reciprocal sparse-field width
    float pair_inv_h;       // reciprocal sparse-field height
    float pair_field_enabled;
    float pair_padding;
};

struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

struct Provenance {
    float base_x;
    float background_x;
    float fill;
    float background_dir;
};

uint point_load_u32(Texture2D<uint> tex, float2 uv) {
    uint width, height;
    tex.GetDimensions(width, height);
    uint2 p = min((uint2)(saturate(uv) * float2(width, height)),
                  uint2(width - 1, height - 1));
    return tex.Load(int3(p, 0));
}

float point_load_f32(Texture2D<float> tex, float2 uv) {
    uint width, height;
    tex.GetDimensions(width, height);
    uint2 p = min((uint2)(saturate(uv) * float2(width, height)),
                  uint2(width - 1, height - 1));
    return tex.Load(int3(p, 0));
}

float2 point_load_f2(Texture2D<float2> tex, float2 uv) {
    uint width, height;
    tex.GetDimensions(width, height);
    uint2 p = min((uint2)(saturate(uv) * float2(width, height)),
                  uint2(width - 1, height - 1));
    return tex.Load(int3(p, 0));
}

float2 chroma_footprint_guard(Texture2D<float2> tex, float2 uv) {
    uint width, height;
    tex.GetDimensions(width, height);
    uint2 hi = min((uint2)(saturate(uv) * float2(width, height)),
                   uint2(width - 1, height - 1));
    uint2 lo = hi - min(hi, uint2(1, 1));
    return 0.25 * (tex.Load(int3(lo, 0)) +
                   tex.Load(int3(uint2(hi.x, lo.y), 0)) +
                   tex.Load(int3(uint2(lo.x, hi.y), 0)) +
                   tex.Load(int3(hi, 0)));
}

Provenance decode_provenance(uint packed) {
    Provenance p;
    p.base_x = (float)(packed & 0xffffu) / 65535.0;
    int delta_px = (int)((packed >> 16) & 0xffu) - 128;
    p.background_x = saturate(p.base_x + (float)delta_px * inv_w);
    p.fill = (float)((packed >> 24) & 0x7fu) / 127.0;
    p.background_dir = (packed & 0x80000000u) != 0u ? 1.0 : -1.0;
    return p;
}

float ridge_axis(float centre, float a, float b) {
    float da = centre - a;
    float db = centre - b;
    return da * db > 0.0 ? min(abs(da), abs(db)) : 0.0;
}

// A line/hair/utensil that differs from BOTH sides is a valid fine source
// structure, not binocular noise.  The Lab must never flatten it merely
// because the opposite inverse warp skipped its sub-grid depth support.
float fine_structure_score(float2 source_uv) {
    float y0 = SourceY.SampleLevel(linSmp, source_uv, 0) * plane_scale;
    float2 sx = float2(1.5 * inv_w, 0.0);
    float2 sy = float2(0.0, 1.5 * inv_h);
    float horizontal = ridge_axis(
        y0,
        SourceY.SampleLevel(linSmp, saturate(source_uv - sx), 0) * plane_scale,
        SourceY.SampleLevel(linSmp, saturate(source_uv + sx), 0) * plane_scale);
    float vertical = ridge_axis(
        y0,
        SourceY.SampleLevel(linSmp, saturate(source_uv - sy), 0) * plane_scale,
        SourceY.SampleLevel(linSmp, saturate(source_uv + sy), 0) * plane_scale);
    return smoothstep(0.025, 0.11, max(horizontal, vertical));
}

// A valid Lab correction may move its source lookup by several pixels.  A
// bilinear lookup is harmless inside one surface, but crossing a coherent
// one-sided silhouette creates the exact dark/grey fringe reported in motion:
// the sample becomes a mixture of foreground and background.  Measure the
// luma path itself and recognize a single coherent transition.  Texture and
// thin ridges accumulate detour variation and are handled by the established
// fine-structure veto instead of being mistaken for a silhouette.
float source_path_edge_score(float2 source_uv, float destination_x) {
    float delta = destination_x - source_uv.x;
    if (abs(delta) < 0.75 * inv_w) return 0.0;

    float2 p0 = source_uv;
    float2 p1 = float2(source_uv.x + 0.25 * delta, source_uv.y);
    float2 p2 = float2(source_uv.x + 0.50 * delta, source_uv.y);
    float2 p3 = float2(source_uv.x + 0.75 * delta, source_uv.y);
    float2 p4 = float2(destination_x, source_uv.y);
    float y0 = point_load_f32(SourceY, p0) * plane_scale;
    float y1 = point_load_f32(SourceY, p1) * plane_scale;
    float y2 = point_load_f32(SourceY, p2) * plane_scale;
    float y3 = point_load_f32(SourceY, p3) * plane_scale;
    float y4 = point_load_f32(SourceY, p4) * plane_scale;
    float endpoint = abs(y4 - y0);
    float travel = abs(y1 - y0) + abs(y2 - y1) +
                   abs(y3 - y2) + abs(y4 - y3);
    float detour = max(0.0, travel - endpoint);
    float coherent = 1.0 - smoothstep(6.0 / 255.0, 30.0 / 255.0, detour);
    float contrast = smoothstep(14.0 / 255.0, 48.0 / 255.0, endpoint);
    return saturate(coherent * contrast);
}

// Source-space contour energy used by the sparse topology.  This is not a
// blur or a correction coefficient: it only marks a node for exact evaluation
// by the full-resolution reciprocal guard below.
float source_contour_score(float2 source_uv) {
    float2 px = float2(inv_w, inv_h);
    float left = point_load_f32(SourceY,
        saturate(source_uv - float2(px.x, 0.0))) * plane_scale;
    float right = point_load_f32(SourceY,
        saturate(source_uv + float2(px.x, 0.0))) * plane_scale;
    float up = point_load_f32(SourceY,
        saturate(source_uv - float2(0.0, px.y))) * plane_scale;
    float down = point_load_f32(SourceY,
        saturate(source_uv + float2(0.0, px.y))) * plane_scale;
    return smoothstep(10.0 / 255.0, 42.0 / 255.0,
                      max(abs(right - left), abs(down - up)));
}

// Provenance maps destination -> source.  Three fixed-point steps recover the
// destination projection of a cyclopean source node without building a dense
// inverse warp.  The final support value also exposes holes/disocclusions.
Provenance project_source_node(Texture2D<uint> provenance,
                               float2 source_uv,
                               out float2 destination_uv) {
    float x = source_uv.x;
    Provenance support;
    [unroll]
    for (int iteration = 0; iteration < 3; ++iteration) {
        support = decode_provenance(point_load_u32(
            provenance, float2(x, source_uv.y)));
        x = saturate(x + source_uv.x - support.base_x);
    }
    destination_uv = float2(x, source_uv.y);
    return decode_provenance(point_load_u32(provenance, destination_uv));
}

// One independent cyclopean source node.  R estimates useful correction;
// G is active-set priority; B and A preserve the reasons for refinement so
// contour and occlusion cells survive even when their correction is vetoed.
float4 pair_field_node(float2 source_uv) {
    float2 left_uv, right_uv;
    Provenance left = project_source_node(
        ProvenanceL, source_uv, left_uv);
    Provenance right = project_source_node(
        ProvenanceR, source_uv, right_uv);

    float source_support_error = max(
        abs(left.base_x - source_uv.x),
        abs(right.base_x - source_uv.x)) / max(inv_w, 1e-8);
    float symmetry_error = 2.0 * abs(
        0.5 * (left_uv.x + right_uv.x) - source_uv.x) /
        max(inv_w, 1e-8);
    float residual = max(source_support_error, symmetry_error);
    float residual_priority = smoothstep(0.85, 3.40, residual);

    float separation = min(abs(left_uv.x - source_uv.x),
                           abs(right_uv.x - source_uv.x)) /
                       max(inv_w, 1e-8);
    float stereo_relevance = smoothstep(0.75, 2.50, separation);
    float occlusion = smoothstep(0.025, 0.14,
                                 max(left.fill, right.fill));
    float contour = source_contour_score(source_uv);
    float correction = saturate(
        lab_gain * residual_priority * stereo_relevance *
        (1.0 - occlusion) * (1.0 - 0.995 * contour));
    float priority = max(residual_priority, max(contour, occlusion));
    return float4(correction, priority, contour, occlusion);
}

// Five probes cover each sparse source cell.  Component-wise maxima are
// conservative upper bounds: a thin contour, a one-eye hole or a residual at
// any probe activates the cell, after which the dense pass makes the exact
// pairwise decision.  At Full HD this is a 192x108 field (~1% of luma).
float4 PS_PairField(VSOut i) : SV_Target {
    float2 half_cell = 0.45 * float2(pair_inv_w, pair_inv_h);
    float4 node = pair_field_node(i.uv);
    node = max(node, pair_field_node(saturate(
        i.uv + float2(half_cell.x, 0.0))));
    node = max(node, pair_field_node(saturate(
        i.uv - float2(half_cell.x, 0.0))));
    node = max(node, pair_field_node(saturate(
        i.uv + float2(0.0, half_cell.y))));
    node = max(node, pair_field_node(saturate(
        i.uv - float2(0.0, half_cell.y))));
    return node;
}

// For a symmetric stereo pair, a source point s observed at x in one eye must
// appear at x' = 2s-x in the other.  Following the other provenance back must
// recover both s and x.  This is stronger than comparing two depth values at
// the same screen coordinate, which would incorrectly punish real occlusions.
float reciprocal_guard(float2 destination_uv,
                       Texture2D<uint> current_provenance,
                       Texture2D<uint> other_provenance,
                       Texture2D<float> current_luma,
                       Texture2D<float> other_luma,
                       out float edge_protection,
                       out float canonical_source_x) {
    edge_protection = 0.0;
    Provenance current = decode_provenance(
        point_load_u32(current_provenance, destination_uv));
    canonical_source_x = current.base_x;
    float current_separation_px = abs(destination_uv.x - current.base_x) /
                                  max(inv_w, 1e-8);
    // Sparse active-set gate: below this displacement the pair cannot create
    // a visible stereo disagreement.  Reject it before the opposite-eye
    // provenance, photometry and contour taps.  On ordinary mid-depth regions
    // this removes most of the expensive pair work without coarsening edges.
    if (current_separation_px < 0.60) return 0.0;

    // The sparse field is indexed by the claimed cyclopean source point, not
    // by either eye's screen coordinate.  An exactly zero priority means the
    // five probes of this source cell found neither contour, occlusion nor a
    // residual capable of reaching the correction knee.  Only then can the
    // expensive opposite-eye and path analysis be skipped safely.
    if (pair_field_enabled > 0.5) {
        float4 sparse_node = PairField.SampleLevel(
            linSmp, float2(current.base_x, destination_uv.y), 0);
        if (sparse_node.g <= 0.0) return 0.0;
    }
    float other_x = 2.0 * current.base_x - destination_uv.x;
    if (other_x <= 0.0 || other_x >= 1.0) return 0.0;

    float2 other_uv = float2(other_x, destination_uv.y);
    Provenance other = decode_provenance(
        point_load_u32(other_provenance, other_uv));

    // Background revealed to only one eye has no reciprocal correspondence.
    // Its established disocclusion reconstruction must not be flattened.
    float occlusion_veto = 1.0 - smoothstep(
        0.025, 0.14, max(current.fill, other.fill));
    if (occlusion_veto <= 0.001) return 0.0;

    float source_error_px = abs(other.base_x - current.base_x) /
                            max(inv_w, 1e-8);
    float back_x = 2.0 * other.base_x - other_x;
    float reciprocal_error_px = abs(back_x - destination_uv.x) /
                                max(inv_w, 1e-8);
    float geometry_error = max(source_error_px, reciprocal_error_px);

    // Ignore subpixel quantization and start correcting only once fusion can
    // genuinely disagree.  The soft upper knee avoids a binary contour.
    float geometry_guard = smoothstep(0.85, 3.40, geometry_error);
    float other_separation_px = abs(other_x - other.base_x) /
                                max(inv_w, 1e-8);
    // A real cyclopean node is supported by both projections.  Using the
    // weaker member prevents one eye from unilaterally authorizing a
    // correction that its conjugate pixel does not geometrically support.
    float pair_separation_px = min(current_separation_px,
                                   other_separation_px);
    float stereo_relevance = smoothstep(0.75, 2.50, pair_separation_px);
    // The remaining terms require two photometric samples, three
    // fine-structure probes and two path scans.  If geometry cannot reach the
    // corrected_sample activation threshold even at unit confidence, no later
    // factor can revive the node: prune it now with bit-identical output.
    if (geometry_guard <= 0.001 || stereo_relevance <= 0.001 ||
        lab_gain * geometry_guard * stereo_relevance *
            occlusion_veto <= 0.001)
        return 0.0;

    float y_current = point_load_f32(current_luma, destination_uv);
    float y_other = other_luma.SampleLevel(linSmp, other_uv, 0);
    float photo_error = abs(y_current - y_other) * max(plane_scale, 1.0);
    float photo_guard = smoothstep(2.0 / 255.0, 18.0 / 255.0, photo_error);
    canonical_source_x = 0.5 * (current.base_x + other.base_x);
    float pair_structure = max(
        fine_structure_score(float2(current.base_x, destination_uv.y)),
        fine_structure_score(float2(other.base_x, destination_uv.y)));
    pair_structure = max(pair_structure, fine_structure_score(
        float2(canonical_source_x, destination_uv.y)));
    float structure_veto = 1.0 - 0.995 * pair_structure;

    float raw_guard = saturate(
        lab_gain * geometry_guard * stereo_relevance *
        occlusion_veto * structure_veto *
        (0.72 + 0.28 * photo_guard));
    float current_path_edge = source_path_edge_score(
        float2(current.base_x, destination_uv.y), destination_uv.x);
    float other_path_edge = source_path_edge_score(
        float2(other.base_x, destination_uv.y), other_x);
    // The pair is indivisible: a clean silhouette seen by either projection
    // protects both.  This removes the last unilateral L/R decision from Lab.
    float path_edge = max(current_path_edge, other_path_edge);
    edge_protection = raw_guard * path_edge;
    // Keep a tiny continuous residual instead of a binary switch.  The
    // corrected_sample entrance knee below makes this an exact visual no-op
    // on a protected high-contrast silhouette.
    return raw_guard * (1.0 - 0.995 * path_edge);
}

float canonical_source_owner(float2 destination_uv,
                             Texture2D<uint> current_provenance,
                             Texture2D<uint> other_provenance) {
    Provenance current = decode_provenance(
        point_load_u32(current_provenance, destination_uv));
    float other_x = 2.0 * current.base_x - destination_uv.x;
    if (other_x <= 0.0 || other_x >= 1.0) return current.base_x;
    Provenance other = decode_provenance(point_load_u32(
        other_provenance, float2(other_x, destination_uv.y)));
    return 0.5 * (current.base_x + other.base_x);
}

float edge_aware_source_sample(Texture2D<float> source, float2 uv) {
    uint width, height;
    source.GetDimensions(width, height);
    float dx = 1.0 / max(1.0, (float)width);
    float linear_value = source.SampleLevel(linSmp, uv, 0);
    float point_value = point_load_f32(source, uv);
    float left = point_load_f32(source, saturate(uv - float2(dx, 0.0)));
    float right = point_load_f32(source, saturate(uv + float2(dx, 0.0)));
    float local_contrast = max(abs(point_value - left),
                               abs(point_value - right)) * plane_scale;
    float edge = smoothstep(10.0 / 255.0, 40.0 / 255.0, local_contrast);
    // Preserve subpixel smoothness inside a surface, but never manufacture a
    // third luma/chroma value at a strong layer boundary.
    return lerp(linear_value, point_value, edge);
}

float corrected_sample(float raw, Texture2D<float> source,
                       Provenance p, float2 destination_uv, float guard,
                       float canonical_source_x) {
    // Exact no-op for the overwhelming majority of the image.  Point-loading
    // raw avoids introducing even a half-texel resample in untouched pixels.
    if (guard <= 0.001) return raw;
    // Both conjugate pixels converge on one shared SOURCE owner. Blending
    // toward each eye's destination coordinate would select two different
    // centre-view pixels and is perceived as a local L/R patchwork.
    float corrected_x = lerp(p.base_x, canonical_source_x, guard);
    float centre = edge_aware_source_sample(
        source, float2(corrected_x, destination_uv.y));
    // Once the guard is established, use the geometrically attenuated sample;
    // feather only the low-confidence entrance to prevent a visible switch.
    float confidence = smoothstep(0.015, 0.11, guard);
    return lerp(raw, centre, confidence);
}

struct OcclusionSolution {
    float source_x;
    float strength;
    float certified;
};

// Solve a hole in source coordinates.  The current eye proposes the rear
// source carried in its packed provenance; the opposite eye must observe that
// source without filling before it becomes a certified cyclopean background.
// If it cannot, a zero-disparity retreat is permitted only when the centre-view
// pixel itself is photometrically background-like.  In both cases the raw fill
// must first be demonstrably closer to the foreground than to the rear layer.
// This prevents a good temporal plate from being overwritten merely because a
// geometric hole exists.
OcclusionSolution solve_cyclopean_occlusion(
        float2 destination_uv,
        Provenance current,
        Texture2D<uint> other_provenance,
        Texture2D<float> current_luma) {
    OcclusionSolution solution;
    solution.source_x = destination_uv.x;
    solution.strength = 0.0;
    solution.certified = 0.0;

    float fill_strength = smoothstep(0.08, 0.78, current.fill);
    float rear_span_px = abs(current.background_x - current.base_x) /
                         max(inv_w, 1e-8);
    if (fill_strength <= 0.001 || rear_span_px < 0.75)
        return solution;

    float sparse_occlusion = max(
        PairField.SampleLevel(linSmp,
            float2(current.base_x, destination_uv.y), 0).a,
        PairField.SampleLevel(linSmp,
            float2(current.background_x, destination_uv.y), 0).a);
    if (pair_field_enabled > 0.5 && sparse_occlusion <= 0.0)
        return solution;
    float sparse_gate = pair_field_enabled > 0.5
        ? smoothstep(0.0, 0.10, sparse_occlusion) : 1.0;

    float rear_x = current.background_x;
    float other_x = 2.0 * rear_x - destination_uv.x;
    float support = 0.0;
    float canonical_x = rear_x;
    if (other_x > 0.0 && other_x < 1.0) {
        Provenance other = decode_provenance(point_load_u32(
            other_provenance, float2(other_x, destination_uv.y)));
        float source_error_px = abs(other.base_x - rear_x) /
                                max(inv_w, 1e-8);
        float visible_other = 1.0 - smoothstep(0.025, 0.14, other.fill);
        support = (1.0 - smoothstep(1.25, 4.25, source_error_px)) *
                  visible_other;
        canonical_x = lerp(rear_x, other.base_x, 0.5 * support);
    }

    float raw_y = point_load_f32(current_luma, destination_uv) * plane_scale;
    float base_y = point_load_f32(SourceY,
        float2(current.base_x, destination_uv.y)) * plane_scale;
    float rear_y = point_load_f32(SourceY,
        float2(canonical_x, destination_uv.y)) * plane_scale;
    float destination_y = point_load_f32(SourceY, destination_uv) * plane_scale;
    float layer_contrast = abs(base_y - rear_y);
    float foreground_bias = abs(raw_y - rear_y) - abs(raw_y - base_y);
    float contamination = smoothstep(4.0 / 255.0, 34.0 / 255.0,
                                     foreground_bias) *
                          smoothstep(12.0 / 255.0, 48.0 / 255.0,
                                     layer_contrast);

    // Uncertified fallback is intentionally conditional: the centre pixel is
    // allowed to collapse disparity only when it resembles the rear layer
    // more than the foreground.  Otherwise the established temporal plate/raw
    // warp remains untouched.
    float destination_background = smoothstep(
        4.0 / 255.0, 30.0 / 255.0,
        abs(destination_y - base_y) - abs(destination_y - rear_y));
    float repair_evidence = max(support,
        (1.0 - support) * destination_background);
    solution.source_x = lerp(destination_uv.x, canonical_x, support);
    solution.certified = support;
    solution.strength = saturate(
        fill_strength * sparse_gate * contamination * repair_evidence *
        lerp(0.52, 0.92, support));
    return solution;
}

float cyclopean_source_sample(Texture2D<float> source, float2 uv) {
    float linear_value = source.SampleLevel(linSmp, uv, 0);
    float point_value = point_load_f32(source, uv);
    // All planes obey the luma silhouette. Point ownership at that boundary
    // prevents the repair itself from manufacturing a grey/chroma mixture.
    float edge = source_contour_score(uv);
    return lerp(linear_value, point_value, edge);
}

float provenance_locked_chroma(float raw, Texture2D<float> source,
                               Texture2D<float> raw_luma,
                               Texture2D<float> other_luma,
                               Texture2D<uint> other_provenance,
                               Provenance p, float2 destination_uv) {
    // The c reference intentionally solves its 4:2:0 warp independently. At
    // a depth contour that can assign colour to a different layer than luma,
    // which looks like a small patch borrowed from the other eye. The Lab
    // corrects only those visible, displaced contour samples using the luma
    // provenance sidecar; flat interiors and true disocclusions stay raw.
    float separation_px = abs(destination_uv.x - p.base_x) /
                          max(inv_w, 1e-8);
    float visible = 1.0 - smoothstep(0.02, 0.14, p.fill);
    float contour = source_contour_score(
        float2(p.base_x, destination_uv.y));
    float output_y = point_load_f32(raw_luma, destination_uv) * plane_scale;
    float owner_y = point_load_f32(
        SourceY, float2(p.base_x, destination_uv.y)) * plane_scale;
    // Only transfer the luma owner when the final luma actually exhibits that
    // source layer. At a feathered or reconstructed boundary, forcing a crisp
    // chroma owner would create the very colour/luma split we are removing.
    float owner_evidence = 1.0 - smoothstep(
        2.0 / 255.0, 18.0 / 255.0, abs(output_y - owner_y));
    // Surface ownership is binocular even though disocclusion content is not.
    // The conjugate eye only CERTIFIES this lock; it never moves the current
    // eye's already validated source coordinate.  This removes threshold
    // crossings that used to recolour just one eye at a 4:2:0 phase edge.
    float other_x = 2.0 * p.base_x - destination_uv.x;
    if (other_x <= 0.0 || other_x >= 1.0) return raw;
    float2 other_uv = float2(other_x, destination_uv.y);
    Provenance other = decode_provenance(
        point_load_u32(other_provenance, other_uv));
    float source_error_px = abs(other.base_x - p.base_x) /
                            max(inv_w, 1e-8);
    float reciprocal_support = 1.0 - smoothstep(
        0.85, 3.40, source_error_px);
    float other_visible = 1.0 - smoothstep(0.02, 0.14, other.fill);
    float other_output_y = other_luma.SampleLevel(
        linSmp, other_uv, 0) * plane_scale;
    float other_owner_y = point_load_f32(
        SourceY, float2(other.base_x, destination_uv.y)) * plane_scale;
    float other_owner_evidence = 1.0 - smoothstep(
        2.0 / 255.0, 18.0 / 255.0,
        abs(other_output_y - other_owner_y));
    float paired_evidence = min(owner_evidence, other_owner_evidence) *
                            reciprocal_support * other_visible;
    float lock = smoothstep(0.65, 1.75, separation_px) * visible *
                 smoothstep(0.08, 0.52, contour) * paired_evidence;
    if (lock <= 0.001) return raw;
    float owned = cyclopean_source_sample(
        source, float2(p.base_x, destination_uv.y));
    float difference = abs(raw - owned) * max(plane_scale, 1.0);
    float visible_need = smoothstep(1.5 / 255.0, 12.0 / 255.0,
                                    difference);
    return lerp(raw, owned, lock * visible_need);
}

float occlusion_corrected_sample(float raw, Texture2D<float> source,
                                 float2 destination_uv,
                                 OcclusionSolution solution) {
    if (solution.strength <= 0.001) return raw;
    float repaired = cyclopean_source_sample(
        source, float2(solution.source_x, destination_uv.y));
    float difference = abs(raw - repaired) * max(plane_scale, 1.0);
    float visible_need = smoothstep(2.0 / 255.0, 18.0 / 255.0,
                                    difference);
    return lerp(raw, repaired, solution.strength * visible_need);
}

struct LabLumaOut {
    float L : SV_Target0;
    float R : SV_Target1;
    float2 guard : SV_Target2;
};

LabLumaOut PS_LabLuma(VSOut i) {
    LabLumaOut o;
    float raw_l = point_load_f32(RawYL, i.uv);
    float raw_r = point_load_f32(RawYR, i.uv);
    float edge_l = 0.0, edge_r = 0.0;
    float source_l = i.uv.x, source_r = i.uv.x;
    float guard_l = reciprocal_guard(
        i.uv, ProvenanceL, ProvenanceR, RawYL, RawYR, edge_l, source_l);
    float guard_r = reciprocal_guard(
        i.uv, ProvenanceR, ProvenanceL, RawYR, RawYL, edge_r, source_r);
    Provenance pl = decode_provenance(point_load_u32(ProvenanceL, i.uv));
    Provenance pr = decode_provenance(point_load_u32(ProvenanceR, i.uv));
    o.L = corrected_sample(raw_l, SourceY, pl, i.uv, guard_l, source_l);
    o.R = corrected_sample(raw_r, SourceY, pr, i.uv, guard_r, source_r);
    OcclusionSolution occlusion_l = solve_cyclopean_occlusion(
        i.uv, pl, ProvenanceR, RawYL);
    OcclusionSolution occlusion_r = solve_cyclopean_occlusion(
        i.uv, pr, ProvenanceL, RawYR);
    o.L = occlusion_corrected_sample(
        o.L, SourceY, i.uv, occlusion_l);
    o.R = occlusion_corrected_sample(
        o.R, SourceY, i.uv, occlusion_r);
    o.guard = float2(guard_l, guard_r);
    return o;
}

// Recover the disparity before the rational comfort envelope from the final
// non-occluded provenance displacement.  This is a passive shadow estimate:
// it never feeds the warp or any later frame.  Backward DIBR shifts each eye
// by half the full binocular disparity, hence the factor two.
float comfort_shadow_loss(Provenance p, float2 destination_uv) {
    if (comfort_enabled < 0.5 || p.fill > 0.025 ||
        comfort_hard_disp <= comfort_soft_disp || max_disp <= inv_w)
        return 0.0;
    float applied = 2.0 * abs(destination_uv.x - p.base_x);
    if (applied <= comfort_soft_disp || applied >= comfort_hard_disp)
        return 0.0;
    float span = comfort_hard_disp - comfort_soft_disp;
    float compressed = min(applied - comfort_soft_disp, span * 0.999);
    float raw = comfort_soft_disp + compressed * span /
                max(span - compressed, 1e-7);
    raw = min(raw, abs(max_disp));
    return saturate(max(0.0, raw - applied) / max(abs(max_disp), inv_w));
}

struct LabChromaOut {
    float UL : SV_Target0;
    float VL : SV_Target1;
    float UR : SV_Target2;
    float VR : SV_Target3;
};

LabChromaOut PS_LabChroma(VSOut i) {
    LabChromaOut o;
    // One chroma sample represents exactly a 2x2 luma cell. Average those four
    // decisions only; generic bilinear filtering can borrow guards from the
    // next cell, while a single point can disagree with the cell's luma mean.
    float2 guard = chroma_footprint_guard(GuardMap, i.uv);
    Provenance pl = decode_provenance(point_load_u32(ProvenanceL, i.uv));
    Provenance pr = decode_provenance(point_load_u32(ProvenanceR, i.uv));
    OcclusionSolution occlusion_l = solve_cyclopean_occlusion(
        i.uv, pl, ProvenanceR, RawYL);
    OcclusionSolution occlusion_r = solve_cyclopean_occlusion(
        i.uv, pr, ProvenanceL, RawYR);
    // The dense luma pass already computed/vetoed the expensive guard. Chroma
    // only recovers its shared source owner here; it must not independently
    // redo photometric or contour decisions on the 4:2:0 lattice.
    float source_l = canonical_source_owner(
        i.uv, ProvenanceL, ProvenanceR);
    float source_r = canonical_source_owner(
        i.uv, ProvenanceR, ProvenanceL);
    float raw_ul = provenance_locked_chroma(
        point_load_f32(RawUL, i.uv), SourceU, RawYL, RawYR,
        ProvenanceR, pl, i.uv);
    float raw_vl = provenance_locked_chroma(
        point_load_f32(RawVL, i.uv), SourceV, RawYL, RawYR,
        ProvenanceR, pl, i.uv);
    float raw_ur = provenance_locked_chroma(
        point_load_f32(RawUR, i.uv), SourceU, RawYR, RawYL,
        ProvenanceL, pr, i.uv);
    float raw_vr = provenance_locked_chroma(
        point_load_f32(RawVR, i.uv), SourceV, RawYR, RawYL,
        ProvenanceL, pr, i.uv);
    o.UL = corrected_sample(
        raw_ul, SourceU, pl, i.uv, guard.x, source_l);
    o.VL = corrected_sample(
        raw_vl, SourceV, pl, i.uv, guard.x, source_l);
    o.UR = corrected_sample(
        raw_ur, SourceU, pr, i.uv, guard.y, source_r);
    o.VR = corrected_sample(
        raw_vr, SourceV, pr, i.uv, guard.y, source_r);
    o.UL = occlusion_corrected_sample(
        o.UL, SourceU, i.uv, occlusion_l);
    o.VL = occlusion_corrected_sample(
        o.VL, SourceV, i.uv, occlusion_l);
    o.UR = occlusion_corrected_sample(
        o.UR, SourceU, i.uv, occlusion_r);
    o.VR = occlusion_corrected_sample(
        o.VR, SourceV, i.uv, occlusion_r);
    return o;
}

// Tiny 96x54 monitoring pass.  RG retains the established Lab guards, B is
// the passive comfort disparity loss normalized by max_disp, and A records
// Lab corrections vetoed at coherent source silhouettes.  The readback is
// still only 20 KiB and remains asynchronous.
float4 PS_LabMetrics(VSOut i) : SV_Target {
    const float2 footprint = float2(0.5 / 96.0, 0.5 / 54.0);
    float2 sum = GuardMap.SampleLevel(linSmp, i.uv, 0);
    sum += GuardMap.SampleLevel(linSmp, saturate(i.uv + float2( footprint.x, 0)), 0);
    sum += GuardMap.SampleLevel(linSmp, saturate(i.uv + float2(-footprint.x, 0)), 0);
    sum += GuardMap.SampleLevel(linSmp, saturate(i.uv + float2(0,  footprint.y)), 0);
    sum += GuardMap.SampleLevel(linSmp, saturate(i.uv + float2(0, -footprint.y)), 0);
    Provenance pl = decode_provenance(point_load_u32(ProvenanceL, i.uv));
    Provenance pr = decode_provenance(point_load_u32(ProvenanceR, i.uv));
    float comfort_loss = max(comfort_shadow_loss(pl, i.uv),
                             comfort_shadow_loss(pr, i.uv));
    float edge_l = 0.0, edge_r = 0.0;
    float source_l = i.uv.x, source_r = i.uv.x;
    reciprocal_guard(i.uv, ProvenanceL, ProvenanceR,
                     RawYL, RawYR, edge_l, source_l);
    reciprocal_guard(i.uv, ProvenanceR, ProvenanceL,
                     RawYR, RawYL, edge_r, source_r);
    return float4(sum * 0.2, comfort_loss, max(edge_l, edge_r));
}
