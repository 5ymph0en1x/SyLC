static const float _182[6] = { -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f };

cbuffer buf : register(b0)
{
    int _401_stereo_mode : packoffset(c0);
    int _401_subtitle_enabled : packoffset(c0.y);
    // Stereoscopic subtitle depth: horizontal disparity normalized to eye width.
    // > 0 = crossed (left-eye copy shifted right, right-eye left) = in FRONT of
    // the screen; 0 = screen depth (flat). Each eye is shifted by half.
    float _401_subtitle_disparity : packoffset(c0.z);
    float4 _401_subtitle_rect : packoffset(c1);
    float _401_sdr_white_level : packoffset(c2);
    float _401_output_gamma : packoffset(c2.y);
    float _401_fp_vfill : packoffset(c2.z);
    float _401_fp_hfill : packoffset(c2.w);
    // Per-sample scale applied to each Y/U/V texel BEFORE YUV->RGB. 1.0 = 8-bit R8
    // (identity); 65535/1023 ~= 64.06 rescales a 10-bit value stored low in an
    // R16_UNORM texel back to [0,1].
    float _401_plane_scale : packoffset(c3);
    // HDR10/PQ color path (HEVC). Both 0 (DEFAULT) reproduces the pre-HDR shader
    // BYTE-FOR-BYTE (legacy BT.601 limited matrix + gamma/sdr_white output). These two
    // ints reuse the free c3.y/c3.z padding, so the cbuffer stays 64 bytes.
    //   yuv_matrix_sel: 0 = BT.601 limited (legacy), 1 = BT.709 limited, 2 = BT.2020nc limited.
    //   transfer_sel:   0 = legacy, 1 = PQ -> scRGB absolute (HDR), 2 = PQ -> tone-mapped SDR.
    int _401_yuv_matrix_sel : packoffset(c3.y);
    int _401_transfer_sel : packoffset(c3.z);
};

Texture2D<float4> texSubtitle : register(t0);
SamplerState _texSubtitle_sampler : register(s0);
Texture2D<float4> texY_L : register(t1);
SamplerState _texY_L_sampler : register(s1);
Texture2D<float4> texU_L : register(t2);
SamplerState _texU_L_sampler : register(s2);
Texture2D<float4> texV_L : register(t3);
SamplerState _texV_L_sampler : register(s3);
Texture2D<float4> texY_R : register(t4);
SamplerState _texY_R_sampler : register(s4);
Texture2D<float4> texU_R : register(t5);
SamplerState _texU_R_sampler : register(s5);
Texture2D<float4> texV_R : register(t6);
SamplerState _texV_R_sampler : register(s6);

static float2 v_texCoord;
static float4 fragColor;

struct SPIRV_Cross_Input
{
    float2 v_texCoord : TEXCOORD0;
};

struct SPIRV_Cross_Output
{
    float4 fragColor : SV_Target0;
};

// --- HDR10 (SMPTE ST 2084 PQ + BT.2020) analytic color path -----------------
// PQ EOTF, per channel: non-linear Ep in [0,1] -> LINEAR light in nits (0..10000).
// Constants are the exact ST 2084 values (mirrored bit-for-bit by tests/hevc/test_pq_math.py).
float3 pq_eotf_nits(float3 Ep)
{
    const float m1 = 0.1593017578125f;
    const float m2 = 78.84375f;
    const float pq_c1 = 0.8359375f;
    const float pq_c2 = 18.8515625f;
    const float pq_c3 = 18.6875f;
    float3 p = pow(max(Ep, 0.0f.xxx), (1.0f / m2).xxx);
    float3 num = max(p - pq_c1.xxx, 0.0f.xxx);
    float3 den = pq_c2.xxx - (pq_c3.xxx * p);
    return 10000.0f.xxx * pow(num / den, (1.0f / m1).xxx);
}

// BT.2020 -> BT.709 gamut in LINEAR light (mul(M, v) dots each row with the 2020 vector).
static const float3x3 kBt2020to709 = {
     1.6605f, -0.5876f, -0.0728f,
    -0.1246f,  1.1329f, -0.0083f,
    -0.0182f, -0.1006f,  1.1187f
};

float3 YUVtoRGB(inout float y, inout float u, inout float v)
{
    y = (y - 0.062745101749897003173828125f) * 1.16438353061676025390625f;
    u -= 0.5f;
    v -= 0.5f;
    float r, g, b;
    if (_401_yuv_matrix_sel == 2)
    {
        // BT.2020 non-constant luminance (limited-range expansion already applied above).
        r = y + (1.4746f * v);
        g = (y - (0.16455f * u)) - (0.57135f * v);
        b = y + (1.8814f * u);
    }
    else if (_401_yuv_matrix_sel == 1)
    {
        // BT.709 limited.
        r = y + (1.5748f * v);
        g = (y - (0.18732f * u)) - (0.46812f * v);
        b = y + (1.8556f * u);
    }
    else
    {
        // Legacy BT.601 limited (DEFAULT) — identical coefficients/associativity to the
        // pre-HDR shader, so the 0/0 path is byte-for-byte the original.
        r = y + (1.401999950408935546875f * v);
        g = (y - (0.3441359996795654296875f * u)) - (0.71413600444793701171875f * v);
        b = y + (1.77199995517730712890625f * u);
    }
    float3 rgb = float3(r, g, b);
    if (_401_transfer_sel == 0)
    {
        // Legacy transfer: gamma-domain RGB clamped to [0,1] (unchanged).
        return clamp(rgb, 0.0f.xxx, 1.0f.xxx);
    }
    // PQ content: rgb are the ST 2084-encoded R'G'B' in [0,1]. Linearize (EOTF) to
    // BT.2020 nits, then convert gamut to BT.709 linear light.
    float3 lin709 = mul(kBt2020to709, pq_eotf_nits(rgb));
    if (_401_transfer_sel == 1)
    {
        // HDR display (scRGB FP16, 1.0 = 80 nits): ABSOLUTE. NO negative clamp
        // (out-of-gamut is legal in scRGB); frag_main skips legacy gamma/sdr_white.
        return lin709 / 80.0f;
    }
    // transfer_sel == 2: SDR-display fallback. Clamp negatives, Reinhard tone-map to
    // [0,1) LINEAR, then re-encode with the inverse display gamma (pow 1/2.2). The SDR
    // swapchain is R8G8B8A8 interpreted as G22 (gamma 2.2): legacy SDR video reaches it
    // already gamma-encoded and passes straight through, so our linear tone-mapped value
    // must occupy that same gamma domain — hence the pow(mapped, 1/2.2) encode here.
    // frag_main skips legacy gamma/sdr_white for transfer_sel != 0.
    lin709 = max(lin709, 0.0f.xxx);
    float3 l = lin709 / 250.0f;
    float3 mapped = l / (1.0f.xxx + l);
    return pow(mapped, (1.0f / 2.2f).xxx);
}

float3 sampleYUV(Texture2D<float4> texY, SamplerState _texY_sampler, Texture2D<float4> texU, SamplerState _texU_sampler, Texture2D<float4> texV, SamplerState _texV_sampler, inout float2 uv)
{
    // HALF-TEXEL clamp from the REAL luma dimensions (was a magic 0.001/0.999 that
    // skipped the outer ~2 texels at 1920 wide). With the CLAMP sampler this pins the
    // sample to the outermost real texel CENTER: full edge fidelity, no wrap, no black.
    uint _syw, _syh; texY.GetDimensions(_syw, _syh);
    float2 _syht = float2(0.5f, 0.5f) / max(float2(float(_syw), float(_syh)), float2(1.0f, 1.0f));
    uv = clamp(uv, _syht, float2(1.0f, 1.0f) - _syht);
    float y = texY.Sample(_texY_sampler, uv).x * _401_plane_scale;
    float u = texU.Sample(_texU_sampler, uv).x * _401_plane_scale;
    float v = texV.Sample(_texV_sampler, uv).x * _401_plane_scale;
    float param = y;
    float param_1 = u;
    float param_2 = v;
    float3 _396 = YUVtoRGB(param, param_1, param_2);
    return _396;
}

float spline36(inout float x)
{
    x = abs(x);
    if (x < 1.0f)
    {
        return (((((1.18181812763214111328125f * x) - 2.1674640178680419921875f) * x) - 0.01435406692326068878173828125f) * x) + 1.0f;
    }
    else
    {
        if (x < 2.0f)
        {
            float t = x - 1.0f;
            return (((((-0.545454561710357666015625f) * t) + 1.2918660640716552734375f) * t) - 0.746411502361297607421875f) * t;
        }
        else
        {
            if (x < 3.0f)
            {
                float t_1 = x - 2.0f;
                return ((((0.0909090936183929443359375f * t_1) - 0.21531100571155548095703125f) * t_1) + 0.12440191209316253662109375f) * t_1;
            }
        }
    }
    return 0.0f;
}

float3 sampleYUV_Spline36_H(Texture2D<float4> texY, SamplerState _texY_sampler, Texture2D<float4> texU, SamplerState _texU_sampler, Texture2D<float4> texV, SamplerState _texV_sampler, float2 uv)
{
    float pixelW = 0.0005208333604969084262847900390625f;
    float srcX = uv.x * 1920.0f;
    float centerX = srcX;
    float sumY = 0.0f;
    float sumU = 0.0f;
    float sumV = 0.0f;
    float sumW = 0.0f;
    // HALF-TEXEL clamp bounds from the REAL luma dimensions (was magic 0.001/0.999).
    uint _shw, _shh; texY.GetDimensions(_shw, _shh);
    float _shtx = 0.5f / max(float(_shw), 1.0f);
    float _shty = 0.5f / max(float(_shh), 1.0f);
    for (int i = 0; i < 6; i++)
    {
        float offset = _182[i];
        float param = offset;
        float _206 = spline36(param);
        float weight = _206;
        float sampleX = (centerX + offset) * pixelW;
        sampleX = clamp(sampleX, _shtx, 1.0f - _shtx);
        float2 sampleUV = float2(sampleX, clamp(uv.y, _shty, 1.0f - _shty));
        sumY += (texY.Sample(_texY_sampler, sampleUV).x * weight);
        sumU += (texU.Sample(_texU_sampler, sampleUV).x * weight);
        sumV += (texV.Sample(_texV_sampler, sampleUV).x * weight);
        sumW += weight;
    }
    if (sumW > 0.0f)
    {
        sumY /= sumW;
        sumU /= sumW;
        sumV /= sumW;
    }
    sumY *= _401_plane_scale;
    sumU *= _401_plane_scale;
    sumV *= _401_plane_scale;
    float param_1 = sumY;
    float param_2 = sumU;
    float param_3 = sumV;
    float3 _273 = YUVtoRGB(param_1, param_2, param_3);
    return _273;
}

float3 sampleYUV_Spline36_V(Texture2D<float4> texY, SamplerState _texY_sampler, Texture2D<float4> texU, SamplerState _texU_sampler, Texture2D<float4> texV, SamplerState _texV_sampler, float2 uv)
{
    float pixelH = 0.000925925909541547298431396484375f;
    float srcY = uv.y * 1080.0f;
    float centerY = srcY;
    float sumY = 0.0f;
    float sumU = 0.0f;
    float sumV = 0.0f;
    float sumW = 0.0f;
    // HALF-TEXEL clamp bounds from the REAL luma dimensions (was magic 0.001/0.999).
    uint _svw, _svh; texY.GetDimensions(_svw, _svh);
    float _svtx = 0.5f / max(float(_svw), 1.0f);
    float _svty = 0.5f / max(float(_svh), 1.0f);
    for (int i = 0; i < 6; i++)
    {
        float offset = _182[i];
        float param = offset;
        float _305 = spline36(param);
        float weight = _305;
        float sampleY = (centerY + offset) * pixelH;
        sampleY = clamp(sampleY, _svty, 1.0f - _svty);
        float2 sampleUV = float2(clamp(uv.x, _svtx, 1.0f - _svtx), sampleY);
        sumY += (texY.Sample(_texY_sampler, sampleUV).x * weight);
        sumU += (texU.Sample(_texU_sampler, sampleUV).x * weight);
        sumV += (texV.Sample(_texV_sampler, sampleUV).x * weight);
        sumW += weight;
    }
    if (sumW > 0.0f)
    {
        sumY /= sumW;
        sumU /= sumW;
        sumV /= sumW;
    }
    sumY *= _401_plane_scale;
    sumU *= _401_plane_scale;
    sumV *= _401_plane_scale;
    float param_1 = sumY;
    float param_2 = sumU;
    float param_3 = sumV;
    float3 _368 = YUVtoRGB(param_1, param_2, param_3);
    return _368;
}

float4 sampleSubtitle(float2 videoUV, float eyeShift)
{
    if (_401_subtitle_enabled == 0)
    {
        return 0.0f.xxxx;
    }
    // eyeShift = displayed horizontal shift for THIS eye (normalized eye width).
    // Shifting the displayed overlay by +s means sampling the rect at (x - s).
    float x = videoUV.x - eyeShift;
    float sx = _401_subtitle_rect.x;
    float sy = _401_subtitle_rect.y;
    float sw = _401_subtitle_rect.z;
    float sh = _401_subtitle_rect.w;
    bool _429 = x >= sx;
    bool _438;
    if (_429)
    {
        _438 = x < (sx + sw);
    }
    else
    {
        _438 = _429;
    }
    bool _445;
    if (_438)
    {
        _445 = videoUV.y >= sy;
    }
    else
    {
        _445 = _438;
    }
    bool _454;
    if (_445)
    {
        _454 = videoUV.y < (sy + sh);
    }
    else
    {
        _454 = _445;
    }
    if (_454)
    {
        float2 subUV = float2((x - sx) / sw, (videoUV.y - sy) / sh);
        return texSubtitle.Sample(_texSubtitle_sampler, subUV);
    }
    return 0.0f.xxxx;
}

float3 blendSubtitle(float3 videoRGB, float4 subtitleRGBA)
{
    return lerp(videoRGB, subtitleRGBA.xyz, subtitleRGBA.w.xxx);
}

void frag_main()
{
    float y_flipped = 1.0f - v_texCoord.y;
    float3 rgb;
    float2 videoUV;
    // Which eye this pixel belongs to, for stereoscopic subtitle depth:
    // +1 = left eye, -1 = right eye, 0 = mono/2D (no disparity applied).
    float eyeSign = 0.0f;
    if (_401_stereo_mode == 0)
    {
        float2 uv = float2(v_texCoord.x, y_flipped);
        float2 param = uv;
        float3 _510 = sampleYUV(texY_L, _texY_L_sampler, texU_L, _texU_L_sampler, texV_L, _texV_L_sampler, param);
        rgb = _510;
        videoUV = uv;
    }
    else
    {
        if (_401_stereo_mode == 1)
        {
            // FramePack: top eye + 45px gap + bottom eye. Each eye is letterboxed
            // into its 1080-tall slot using fp_vfill/fp_hfill (derived from the source
            // aspect) so a non-16:9 eye (e.g. Full-SBS 1920x1012) keeps its native
            // resolution with black bars instead of a vertical stretch. A 16:9 eye
            // (MVC) has fp_vfill=fp_hfill=1 and fills the slot exactly (unchanged).
            float vbar = (1.0f - _401_fp_vfill) * 0.5f;
            float hbar = (1.0f - _401_fp_hfill) * 0.5f;
            if (y_flipped < 0.4897958934307098388671875f)
            {
                float ly = y_flipped / 0.4897958934307098388671875f;
                if (ly < vbar || ly > (1.0f - vbar) || v_texCoord.x < hbar || v_texCoord.x > (1.0f - hbar))
                {
                    rgb = 0.0f.xxx;
                    videoUV = (-1.0f).xx;
                }
                else
                {
                    float2 uv_1 = float2((v_texCoord.x - hbar) / _401_fp_hfill, (ly - vbar) / _401_fp_vfill);
                    float2 param_1 = uv_1;
                    rgb = sampleYUV(texY_L, _texY_L_sampler, texU_L, _texU_L_sampler, texV_L, _texV_L_sampler, param_1);
                    videoUV = uv_1;
                    eyeSign = 1.0f;
                }
            }
            else
            {
                if (y_flipped > 0.5102040767669677734375f)
                {
                    float ly2 = (y_flipped - 0.5102040767669677734375f) / 0.4897958934307098388671875f;
                    if (ly2 < vbar || ly2 > (1.0f - vbar) || v_texCoord.x < hbar || v_texCoord.x > (1.0f - hbar))
                    {
                        rgb = 0.0f.xxx;
                        videoUV = (-1.0f).xx;
                    }
                    else
                    {
                        float2 uv_2 = float2((v_texCoord.x - hbar) / _401_fp_hfill, (ly2 - vbar) / _401_fp_vfill);
                        float2 param_2 = uv_2;
                        rgb = sampleYUV(texY_R, _texY_R_sampler, texU_R, _texU_R_sampler, texV_R, _texV_R_sampler, param_2);
                        videoUV = uv_2;
                        eyeSign = -1.0f;
                    }
                }
                else
                {
                    rgb = 0.0f.xxx;
                    videoUV = (-1.0f).xx;
                }
            }
        }
        else
        {
            if (_401_stereo_mode == 2)
            {
                if (v_texCoord.x < 0.5f)
                {
                    float2 uv_3 = float2(v_texCoord.x * 2.0f, y_flipped);
                    float2 param_3 = uv_3;
                    rgb = sampleYUV_Spline36_H(texY_L, _texY_L_sampler, texU_L, _texU_L_sampler, texV_L, _texV_L_sampler, param_3);
                    videoUV = uv_3;
                    eyeSign = 1.0f;
                }
                else
                {
                    float2 uv_4 = float2((v_texCoord.x - 0.5f) * 2.0f, y_flipped);
                    float2 param_4 = uv_4;
                    rgb = sampleYUV_Spline36_H(texY_R, _texY_R_sampler, texU_R, _texU_R_sampler, texV_R, _texV_R_sampler, param_4);
                    videoUV = uv_4;
                    eyeSign = -1.0f;
                }
            }
            else
            {
                if (_401_stereo_mode == 3)
                {
                    if (y_flipped < 0.5f)
                    {
                        float2 uv_5 = float2(v_texCoord.x, y_flipped * 2.0f);
                        float2 param_5 = uv_5;
                        rgb = sampleYUV_Spline36_V(texY_L, _texY_L_sampler, texU_L, _texU_L_sampler, texV_L, _texV_L_sampler, param_5);
                        videoUV = uv_5;
                        eyeSign = 1.0f;
                    }
                    else
                    {
                        float2 uv_6 = float2(v_texCoord.x, (y_flipped - 0.5f) * 2.0f);
                        float2 param_6 = uv_6;
                        rgb = sampleYUV_Spline36_V(texY_R, _texY_R_sampler, texU_R, _texU_R_sampler, texV_R, _texV_R_sampler, param_6);
                        videoUV = uv_6;
                        eyeSign = -1.0f;
                    }
                }
            }
        }
    }
    float2 param_7 = videoUV;
    // Each eye gets half the disparity, in opposite directions (crossed when > 0).
    float4 subtitle = sampleSubtitle(param_7, eyeSign * _401_subtitle_disparity * 0.5f);
    float3 param_8 = rgb;
    float4 param_9 = subtitle;
    rgb = blendSubtitle(param_8, param_9);
    // Legacy output encode. transfer_sel == 0 (all existing content) runs it UNCHANGED:
    //   Optional EOTF: linearize the gamma-domain RGB before scaling into the (linear)
    //   scRGB FP16 buffer. output_gamma <= 0 disables it (raw passthrough).
    // For the PQ paths (transfer_sel != 0) it is SKIPPED — YUVtoRGB already produced the
    // final absolute-scRGB (1) or gamma-encoded tone-mapped SDR (2) value.
    if (_401_transfer_sel == 0)
    {
        if (_401_output_gamma > 0.0f) { rgb = pow(max(rgb, 0.0f.xxx), _401_output_gamma.xxx); }
        rgb *= _401_sdr_white_level;
    }
    fragColor = float4(rgb, 1.0f);
}

SPIRV_Cross_Output main(SPIRV_Cross_Input stage_input)
{
    v_texCoord = stage_input.v_texCoord;
    frag_main();
    SPIRV_Cross_Output stage_output;
    stage_output.fragColor = fragColor;
    return stage_output;
}
