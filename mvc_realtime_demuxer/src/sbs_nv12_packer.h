// sbs_nv12_packer.h — packs 6 planar YUV SRVs (L:Y,U,V  R:Y,U,V, all R8_UNORM)
//   into one NV12 3840x1080 SBS texture. Opaque Impl (no D3D types leaked beyond fwd-decls).
//
// Second consumer of the renderer's already-uploaded per-plane YUV textures: it renders
// them into ONE NV12 3840x1080 side-by-side texture (left eye in the left 1920 columns,
// right eye in the right 1920) that Task 1's NvencEncoder registers + encodes zero-copy.
// The path stays in YUV/NV12 end to end — never RGB — so the cast is lossless. The NV12
// texture, its luma/chroma render-target views, and the two pixel shaders all live behind
// an opaque Impl, exactly like NativeRenderer / NvencEncoder, so this header drags in
// neither <d3d11.h> nor shader-compilation APIs.
#pragma once
#include <string>

struct ID3D11Device; struct ID3D11DeviceContext;
struct ID3D11Texture2D; struct ID3D11ShaderResourceView;

namespace sylc {

// Subtitle burn-in parameters for one pack() call. Mirrors the display
// renderer's subtitle state (upload_subtitle texture + set_uniforms values):
// the overlay is composited into BOTH SBS eyes with the display's exact
// per-eye half-disparity convention, so the cast shows what the PC shows.
struct CastSubtitle {
    ID3D11ShaderResourceView* srv = nullptr;   // RGBA8 straight alpha; null = disabled
    int   enabled  = 0;                        // 0 = burn nothing (lossless copy)
    float disparity = 0.f;                     // normalized eye width; > 0 = pop-out
    float rect[4] = {0.f, 0.f, 0.f, 0.f};      // normalized (x, y, w, h) in eye space
};

class SbsNv12Packer {
public:
    SbsNv12Packer(); ~SbsNv12Packer();

    SbsNv12Packer(const SbsNv12Packer&) = delete;            // unique resource owner (raw Impl*):
    SbsNv12Packer& operator=(const SbsNv12Packer&) = delete; // a shallow copy would double-free impl_

    // Creates the output texture + luma/chroma RTVs + PS. tenBit=false -> NV12
    // (8-bit, R8/R8G8 plane views); true -> P010 (10-bit, R16/R16G16) for the
    // Main10 HDR cast.
    bool init(ID3D11Device* dev, std::string& err, bool tenBit = false);
    ID3D11Texture2D* outputTexture() const;           // the NV12/P010 3840x1080 to hand NVENC
    // 0..2=L Y/U/V, 3..5=R Y/U/V. `sub` burns the subtitle overlay into both
    // eyes; the two-argument form packs without one (identical to the old API).
    // planeScale multiplies every video sample before it is written: 1.0 for
    // 8-bit R8 planes and HW P010 R16 planes (already target-normalized);
    // 65535/1023 rescales SW-decoded 10-bit (value stored low in R16) up to
    // P010 MSB alignment. Mirrors the display renderer's plane_scale.
    void pack(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* const yuvSrv[6]);
    void pack(ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* const yuvSrv[6],
              const CastSubtitle& sub, float planeScale = 1.0f);
    void shutdown();

private:
    struct Impl; Impl* impl_;
};

} // namespace sylc
