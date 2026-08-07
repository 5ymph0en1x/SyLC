// DirectCompute ownership/safety construction for the Synth3D inference grid.
// Inputs are filter-free RGBA16_UNORM publications from SharedDepthService:
//   Surface: depth, luma, confidence, boundary
//   RGB:     linear RGB, unused
// The five entry points deliberately use gather/ping-pong propagation so no
// unordered cross-thread writes or device-specific atomics affect the result.

cbuffer OwnerCB : register(b0) {
    uint OwnerWidth;
    uint OwnerHeight;
    uint OwnerPad0;
    uint OwnerPad1;
};

Texture2D<float4> OwnerSurface : register(t0);
Texture2D<float4> OwnerRgb : register(t1);
Texture2D<float> OwnerScalar : register(t2);
Texture2D<float4> OwnerState : register(t3);

RWTexture2D<float> OwnerScalarOut : register(u0);
RWTexture2D<float4> OwnerStateOut : register(u1);
RWTexture2D<float2> OwnerGeometryOut : register(u2);

float owner_smooth(float edge0, float edge1, float value) {
    float t = saturate((value - edge0) / max(1.0e-6, edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

int2 owner_clamp(int2 p) {
    return clamp(p, int2(0, 0), int2(OwnerWidth - 1, OwnerHeight - 1));
}

[numthreads(16, 16, 1)]
void CS_OwnerUncertainty(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OwnerWidth || tid.y >= OwnerHeight) return;
    int2 p = int2(tid.xy);
    float4 center = OwnerSurface.Load(int3(p, 0));
    float dx = abs(
        OwnerSurface.Load(int3(owner_clamp(p + int2(1, 0)), 0)).r -
        OwnerSurface.Load(int3(owner_clamp(p - int2(1, 0)), 0)).r);
    float dy = abs(
        OwnerSurface.Load(int3(owner_clamp(p + int2(0, 1)), 0)).r -
        OwnerSurface.Load(int3(owner_clamp(p - int2(0, 1)), 0)).r);
    float lx = abs(
        OwnerSurface.Load(int3(owner_clamp(p + int2(1, 0)), 0)).g -
        OwnerSurface.Load(int3(owner_clamp(p - int2(1, 0)), 0)).g);
    float ly = abs(
        OwnerSurface.Load(int3(owner_clamp(p + int2(0, 1)), 0)).g -
        OwnerSurface.Load(int3(owner_clamp(p - int2(0, 1)), 0)).g);
    float imageEdge = owner_smooth(0.022, 0.135, max(lx, ly));
    float depthEdge = owner_smooth(0.030, 0.155, max(dx, dy));
    float conf = saturate(center.b);
    float unsupportedImage = imageEdge * (1.0 - depthEdge);
    float unsupportedDepth = depthEdge * (1.0 - imageEdge);
    float lowConfidence = (1.0 - conf) * (0.12 + 0.48 * imageEdge);
    float boundaryUncertainty = (1.0 - conf) * 0.30 * saturate(center.a);
    OwnerScalarOut[p] = saturate(max(
        unsupportedImage * (0.08 + 0.92 * (1.0 - conf)),
        max(0.62 * unsupportedDepth, max(lowConfidence, boundaryUncertainty))));
}

[numthreads(16, 16, 1)]
void CS_OwnerDilate(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OwnerWidth || tid.y >= OwnerHeight) return;
    int2 p = int2(tid.xy);
    float value = 0.0;
    [unroll] for (int oy = -2; oy <= 2; ++oy) {
        [unroll] for (int ox = -2; ox <= 2; ++ox) {
            value = max(value, OwnerScalar.Load(
                int3(owner_clamp(p + int2(ox, oy)), 0)));
        }
    }
    OwnerScalarOut[p] = value;
}

[numthreads(16, 16, 1)]
void CS_OwnerLocal(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OwnerWidth || tid.y >= OwnerHeight) return;
    int2 p = int2(tid.xy);
    float4 surface = OwnerSurface.Load(int3(p, 0));
    float centerDepth = surface.r;
    float candidateDepth = centerDepth;
    float repair = 0.0;
    float foregroundAnchor = 0.0;
    float uncertainty = OwnerScalar.Load(int3(p, 0));

    if (uncertainty > 0.14) {
        float3 centerRgb = OwnerRgb.Load(int3(p, 0)).rgb;
        float nearMax = centerDepth;
        float localMin = centerDepth;
        float anchorSupport = 0.0;
        [unroll] for (int oy = -3; oy <= 3; ++oy) {
            [unroll] for (int ox = -3; ox <= 3; ++ox) {
                if (ox == 0 && oy == 0) continue;
                int2 q = owner_clamp(p + int2(ox, oy));
                float dj = OwnerSurface.Load(int3(q, 0)).r;
                localMin = min(localMin, dj);
                float3 rgb = OwnerRgb.Load(int3(q, 0)).rgb;
                float colorDelta = max(abs(rgb.r - centerRgb.r),
                    max(abs(rgb.g - centerRgb.g), abs(rgb.b - centerRgb.b)));
                float affinity = 1.0 - owner_smooth(0.035, 0.18, colorDelta);
                if (affinity >= 0.38) {
                    nearMax = max(nearMax, dj);
                    if (abs(dj - centerDepth) <= 0.065) {
                        float distance2 = float(ox * ox + oy * oy);
                        anchorSupport += affinity / (1.0 + 0.18 * distance2);
                    }
                }
            }
        }
        foregroundAnchor = owner_smooth(0.040, 0.16, centerDepth - localMin) *
                           owner_smooth(0.55, 2.2, anchorSupport);

        float weightedDepth = 0.0;
        float support = 0.0;
        int supportCount = 0;
        [unroll] for (int oy2 = -3; oy2 <= 3; ++oy2) {
            [unroll] for (int ox2 = -3; ox2 <= 3; ++ox2) {
                if (ox2 == 0 && oy2 == 0) continue;
                int2 q = owner_clamp(p + int2(ox2, oy2));
                float dj = OwnerSurface.Load(int3(q, 0)).r;
                if (dj < nearMax - 0.065 || dj <= centerDepth + 0.025) continue;
                float3 rgb = OwnerRgb.Load(int3(q, 0)).rgb;
                float colorDelta = max(abs(rgb.r - centerRgb.r),
                    max(abs(rgb.g - centerRgb.g), abs(rgb.b - centerRgb.b)));
                float affinity = 1.0 - owner_smooth(0.035, 0.18, colorDelta);
                float distance2 = float(ox2 * ox2 + oy2 * oy2);
                float weight = affinity / (1.0 + 0.18 * distance2);
                if (weight <= 0.05) continue;
                weightedDepth += weight * dj;
                support += weight;
                ++supportCount;
            }
        }
        if (supportCount >= 2 && support > 0.55) {
            candidateDepth = weightedDepth / support;
            repair = saturate(1.15 * uncertainty *
                owner_smooth(0.55, 2.4, support) *
                owner_smooth(0.025, 0.16, candidateDepth - centerDepth));
        }
    }

    float safety = max(max(0.12, 1.0 - 0.82 * uncertainty),
                       0.90 * foregroundAnchor);
    float strength = max(repair, foregroundAnchor);
    OwnerStateOut[p] = float4(saturate(candidateDepth), saturate(safety),
                              saturate(repair), strength > 0.08 ? strength : 0.0);
}

[numthreads(16, 16, 1)]
void CS_OwnerPropagate(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OwnerWidth || tid.y >= OwnerHeight) return;
    int2 p = int2(tid.xy);
    float rawDepth = OwnerSurface.Load(int3(p, 0)).r;
    float3 targetRgb = OwnerRgb.Load(int3(p, 0)).rgb;
    float4 current = OwnerState.Load(int3(p, 0));
    float bestOwned = current.r;
    float bestRepair = current.b;
    float bestScore = bestOwned + 0.08 * bestRepair;
    float nextFrontier = 0.0;

    [unroll] for (int oy = -1; oy <= 1; ++oy) {
        [unroll] for (int ox = -1; ox <= 1; ++ox) {
            if (ox == 0 && oy == 0) continue;
            int2 q = owner_clamp(p + int2(ox, oy));
            float4 source = OwnerState.Load(int3(q, 0));
            if (source.a <= 0.08) continue;
            float3 sourceRgb = OwnerRgb.Load(int3(q, 0)).rgb;
            float colorDelta = max(abs(sourceRgb.r - targetRgb.r),
                max(abs(sourceRgb.g - targetRgb.g), abs(sourceRgb.b - targetRgb.b)));
            float affinity = 1.0 - owner_smooth(0.030, 0.16, colorDelta);
            float candidateStrength = 0.93 * source.a * affinity;
            if (candidateStrength <= 0.08 || source.r <= rawDepth + 0.025) continue;
            float candidateScore = source.r + 0.08 * candidateStrength;
            if (candidateScore > bestScore) {
                bestScore = candidateScore;
                bestOwned = source.r;
                bestRepair = candidateStrength;
                nextFrontier = candidateStrength;
            }
        }
    }
    OwnerStateOut[p] = float4(bestOwned, current.g, bestRepair, nextFrontier);
}

[numthreads(16, 16, 1)]
void CS_OwnerCompose(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OwnerWidth || tid.y >= OwnerHeight) return;
    int2 p = int2(tid.xy);
    float rawDepth = OwnerSurface.Load(int3(p, 0)).r;
    float4 state = OwnerState.Load(int3(p, 0));
    OwnerGeometryOut[p] = saturate(float2(
        rawDepth + state.b * (state.r - rawDepth),
        state.g + 0.72 * state.b));
}
