// NVENC thin wrapper for SyLC Cast (PC sender) — one responsibility only:
//   NV12 D3D11 texture  ->  HEVC NAL byte packets.
//
// It knows nothing about SBS packing or the network transport. The renderer
// packs the decoded YUV into an NV12 side-by-side texture; this wrapper encodes
// that texture to HEVC; a transport layer streams the resulting packets.
//
// The NVENC encoder DLL (nvEncodeAPI64.dll) is DYNAMIC-LOADED from the user's
// NVIDIA driver at runtime — nothing NVIDIA-licensed is linked in. Only the MIT
// nv-codec-headers (nvEncodeAPI.h, vendored at ../third_party/nv-codec-headers/)
// are compiled. available() is therefore safe to call on a machine with no
// NVIDIA GPU (it returns false, no crash).
//
// This header keeps the NVENC / D3D11 types out of its interface (opaque Impl,
// exactly like NativeRenderer) so it can be included anywhere without dragging
// in <d3d11.h> or "nvEncodeAPI.h".
#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct ID3D11Device;
struct ID3D11Texture2D;

namespace sylc {

enum class CastMode { LosslessWired, CbrLowLatency };

struct NvencConfig {
    uint32_t width = 3840, height = 1080, fps = 24;
    CastMode mode = CastMode::CbrLowLatency;
    uint32_t cbrBitrateBps = 175'000'000;
    // HEVC Main10 over a P010 input surface, with BT.2020/ST2084 (PQ) VUI
    // signalling — the HDR cast path. false = the original 8-bit NV12 Main
    // session, byte-for-byte unchanged.
    bool main10 = false;
};

class NvencEncoder {
public:
    NvencEncoder(); ~NvencEncoder();

    NvencEncoder(const NvencEncoder&) = delete;            // unique resource owner (raw Impl*):
    NvencEncoder& operator=(const NvencEncoder&) = delete; // a shallow copy would double-free impl_

    static bool available();                    // nvEncodeAPI64.dll loads + CreateInstance ok
    bool open(ID3D11Device* dev, const NvencConfig& cfg, std::string& err);
    // Hot-reconfigure an OPEN session to a new bitrate/mode WITHOUT a teardown, via
    // NvEncReconfigureEncoder. The next encoded frame is forced to an IDR so the decoder
    // recovers instantly at the switch. Returns true when the driver accepts the seamless
    // reconfigure (the common same-mode CBR bitrate change). Returns false (err set) when
    // the driver rejects it — e.g. a lossless<->cbr mode switch, which alters the GOP
    // structure / tuning that NvEncReconfigureEncoder does not support; the caller then
    // falls back to an encoder-only close()+open()+registerInput() (packer/texture stay).
    bool reconfigure(const NvencConfig& cfg, std::string& err);
    bool supportsLossless() const;
    bool registerInput(ID3D11Texture2D* tex, void** outRegistered, std::string& err);
    bool encode(void* registeredInput, int64_t ptsMs, bool forceIdr,
                std::vector<std::vector<uint8_t>>& outPackets, std::string& err);
    void flush(std::vector<std::vector<uint8_t>>& outPackets);
    void close();

private:
    struct Impl; Impl* impl_;                   // opaque, like NativeRenderer
};

} // namespace sylc
