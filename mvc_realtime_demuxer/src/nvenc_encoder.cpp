// NVENC thin wrapper — implementation. See nvenc_encoder.h.
//
// Pipeline position: NV12 D3D11 texture -> HEVC NAL bytes. The NVENC entry point
// is dynamic-loaded (LoadLibraryW) from the NVIDIA driver; NOTHING is linked
// against an NVENC .lib. Every NV_ENC_* struct is zero-initialized and
// version-stamped (or the driver rejects it with NV_ENC_ERR_INVALID_VERSION),
// and every fl.nvEncXxx return is checked against NV_ENC_SUCCESS.
//
// Windows-only: the whole translation unit is gated on _WIN32 so it degrades to
// an empty object file anywhere else (it is only added to the build inside the
// Windows-only BUILD_NATIVE_RENDERER block, but the guard is belt-and-suspenders).
#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>          // LoadLibraryW / GetProcAddress / FreeLibrary, RECT, GUID
#include <d3d11.h>            // ID3D11Device, ID3D11Texture2D (complete types for the void* casts)
#include "nvEncodeAPI.h"      // vendored MIT nv-codec-headers (on the include path from Task 0)

#include "nvenc_encoder.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// File-local helpers (no NVENC types leak past this TU).
// ---------------------------------------------------------------------------
namespace {

// Signature of the single exported entry point we resolve by name.
typedef NVENCSTATUS (NVENCAPI *PFN_NvEncodeAPICreateInstance)(NV_ENCODE_API_FUNCTION_LIST*);

// Build a human-readable error string for a failed NVENC call. When an encoder
// session exists, the driver's own last-error text is appended.
std::string status_string(NV_ENCODE_API_FUNCTION_LIST& fl, void* encoder,
                          NVENCSTATUS s, const char* where) {
    std::string msg = where;
    msg += ": NVENC status ";
    msg += std::to_string(static_cast<int>(s));
    if (encoder && fl.nvEncGetLastErrorString) {
        const char* detail = fl.nvEncGetLastErrorString(encoder);
        if (detail && *detail) { msg += " ("; msg += detail; msg += ")"; }
    }
    return msg;
}

// Load nvEncodeAPI64.dll and populate a function list. Returns false (never
// crashes) when the DLL / entry point is absent (no NVIDIA driver) or when
// CreateInstance rejects the header version. On success `dll` owns a reference
// the caller must FreeLibrary.
bool load_nvenc_api(HMODULE& dll, NV_ENCODE_API_FUNCTION_LIST& fl) {
    dll = ::LoadLibraryW(L"nvEncodeAPI64.dll");
    if (!dll) return false;

    auto create = reinterpret_cast<PFN_NvEncodeAPICreateInstance>(
        ::GetProcAddress(dll, "NvEncodeAPICreateInstance"));
    if (!create) { ::FreeLibrary(dll); dll = nullptr; return false; }

    fl = NV_ENCODE_API_FUNCTION_LIST{};
    fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (create(&fl) != NV_ENC_SUCCESS) { ::FreeLibrary(dll); dll = nullptr; return false; }
    return true;
}

// Lock the output bitstream, copy its bytes into a fresh packet, unlock. Used by
// both encode() and flush(). Returns false with `err` set on any failure.
bool lock_and_append(NV_ENCODE_API_FUNCTION_LIST& fl, void* encoder, NV_ENC_OUTPUT_PTR buffer,
                     std::vector<std::vector<uint8_t>>& out, std::string& err) {
    NV_ENC_LOCK_BITSTREAM lb{};
    lb.version = NV_ENC_LOCK_BITSTREAM_VER;
    lb.outputBitstream = buffer;

    NVENCSTATUS s = fl.nvEncLockBitstream(encoder, &lb);
    if (s != NV_ENC_SUCCESS) { err = status_string(fl, encoder, s, "nvEncLockBitstream"); return false; }

    const uint8_t* p = static_cast<const uint8_t*>(lb.bitstreamBufferPtr);
    out.emplace_back(p, p + lb.bitstreamSizeInBytes);

    s = fl.nvEncUnlockBitstream(encoder, buffer);
    if (s != NV_ENC_SUCCESS) { err = status_string(fl, encoder, s, "nvEncUnlockBitstream"); return false; }
    return true;
}

// Fetch the P4 preset config for `tuning` and apply SyLC's fixed overrides into `out`.
// SHARED by open() and reconfigure() so the encode config can never drift between them —
// a drift would make a supposedly "seamless" reconfigure silently change more than the
// bitrate/mode the caller asked for. Requires an already-open encoder session
// (nvEncGetEncodePresetConfigEx needs the handle). Returns false with `err` set on failure.
bool build_encode_config(NV_ENCODE_API_FUNCTION_LIST& fl, void* encoder,
                         const sylc::NvencConfig& cfg, NV_ENC_TUNING_INFO tuning,
                         NV_ENC_CONFIG& out, std::string& err) {
    NV_ENC_PRESET_CONFIG pc{};
    pc.version           = NV_ENC_PRESET_CONFIG_VER;
    pc.presetCfg.version = NV_ENC_CONFIG_VER;
    NVENCSTATUS s = fl.nvEncGetEncodePresetConfigEx(encoder, NV_ENC_CODEC_HEVC_GUID,
                                                    NV_ENC_PRESET_P4_GUID, tuning, &pc);
    if (s != NV_ENC_SUCCESS) {
        err = status_string(fl, encoder, s, "nvEncGetEncodePresetConfigEx");
        return false;
    }
    out         = pc.presetCfg;
    out.version = NV_ENC_CONFIG_VER;

    // Force IPP (no B-frames) in BOTH modes. B-frame reorder makes the encoder buffer
    // frames and emit output asynchronously, which our single shared bitstream buffer
    // cannot hold. frameIntervalP=1 -> synchronous 1:1 output per submit, so the
    // single-buffer design in encode() stays correct for lossless too (QP stays
    // preset-managed; wired lossless has ample bandwidth for the lost B-frames).
    out.frameIntervalP = 1;

    if (cfg.main10) {
        // HDR cast: HEVC Main10 over a P010 surface, and the stream SAYS what
        // it carries — BT.2020 primaries + ST 2084 (PQ) transfer in the VUI,
        // limited range. Receivers (incl. Android MediaCodec / the Quest
        // compositor) read this in-band, so even a sender that predates the
        // HELLO_ACK "hdr" announcement decodes with the right color intent.
        out.profileGUID = NV_ENC_HEVC_PROFILE_MAIN10_GUID;
        out.encodeCodecConfig.hevcConfig.inputBitDepth  = NV_ENC_BIT_DEPTH_10;
        out.encodeCodecConfig.hevcConfig.outputBitDepth = NV_ENC_BIT_DEPTH_10;
        auto& vui = out.encodeCodecConfig.hevcConfig.hevcVUIParameters;
        vui.videoSignalTypePresentFlag  = 1;
        vui.videoFullRangeFlag          = 0;                            // limited
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries          = NV_ENC_VUI_COLOR_PRIMARIES_BT2020;
        vui.transferCharacteristics  = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SMPTE2084;
        vui.colourMatrix             = NV_ENC_VUI_MATRIX_COEFFS_BT2020_NCL;
    }

    if (cfg.mode == sylc::CastMode::CbrLowLatency) {
        const uint32_t fps = cfg.fps ? cfg.fps : 1u;
        out.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
        out.rcParams.averageBitRate  = cfg.cbrBitrateBps;
        out.rcParams.vbvBufferSize   = cfg.cbrBitrateBps / fps;   // one-frame VBV
        out.rcParams.vbvInitialDelay = out.rcParams.vbvBufferSize;
        out.rcParams.multiPass       = NV_ENC_MULTI_PASS_DISABLED;
        out.gopLength                = NVENC_INFINITE_GOPLENGTH;
        out.encodeCodecConfig.hevcConfig.idrPeriod    = NVENC_INFINITE_GOPLENGTH;
        out.encodeCodecConfig.hevcConfig.repeatSPSPPS = 1;   // SPS/PPS on every IDR
    }
    // LosslessWired: rate-control (CONSTQP / QP0) stays preset-managed.
    return true;
}

} // namespace

namespace sylc {

// ---------------------------------------------------------------------------
// Opaque state (kept out of the public header, exactly like NativeRenderer).
// ---------------------------------------------------------------------------
struct NvencEncoder::Impl {
    HMODULE                            dll = nullptr;   // nvEncodeAPI64.dll (FreeLibrary on close)
    NV_ENCODE_API_FUNCTION_LIST        fl{};            // cached driver function list
    void*                              encoder = nullptr;
    NV_ENC_CONFIG                      encodeConfig{};  // outlives nvEncInitializeEncoder
    NV_ENC_OUTPUT_PTR                  bitstreamBuffer = nullptr;
    std::vector<NV_ENC_REGISTERED_PTR> registered;      // handles to unregister at close
    uint32_t width = 0, height = 0, fps = 1;
    CastMode mode = CastMode::CbrLowLatency;            // current mode (open/reconfigure)
    // Input surface format: NV12 (8-bit Main) or YUV420_10BIT/P010 (Main10 HDR).
    NV_ENC_BUFFER_FORMAT bufFmt = NV_ENC_BUFFER_FORMAT_NV12;
};

NvencEncoder::NvencEncoder() : impl_(new Impl()) {}

NvencEncoder::~NvencEncoder() {
    close();
    delete impl_;
    impl_ = nullptr;
}

// static — pure probe: load, CreateInstance, then release. Safe with no GPU.
bool NvencEncoder::available() {
    HMODULE dll = nullptr;
    NV_ENCODE_API_FUNCTION_LIST fl{};
    if (!load_nvenc_api(dll, fl)) return false;
    ::FreeLibrary(dll);
    return true;
}

bool NvencEncoder::open(ID3D11Device* dev, const NvencConfig& cfg, std::string& err) {
    if (!impl_)          { err = "open: no impl";             return false; }
    if (!dev)            { err = "open: null device";         return false; }
    if (impl_->encoder)  { err = "open: already open";        return false; }

    if (!load_nvenc_api(impl_->dll, impl_->fl)) {
        err = "open: nvEncodeAPI64.dll load / CreateInstance failed (no NVENC driver?)";
        return false;
    }
    NV_ENCODE_API_FUNCTION_LIST& fl = impl_->fl;

    impl_->width  = cfg.width;
    impl_->height = cfg.height;
    impl_->fps    = cfg.fps ? cfg.fps : 1u;
    impl_->mode   = cfg.mode;
    impl_->bufFmt = cfg.main10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT
                               : NV_ENC_BUFFER_FORMAT_NV12;

    const NV_ENC_TUNING_INFO tuning = (cfg.mode == CastMode::LosslessWired)
        ? NV_ENC_TUNING_INFO_LOSSLESS
        : NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    // 1) Open a DirectX (D3D11) encode session on the caller's device.
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sp{};
    sp.version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
    sp.deviceType = NV_ENC_DEVICE_TYPE_DIRECTX;
    sp.device     = dev;
    sp.apiVersion = NVENCAPI_VERSION;
    NVENCSTATUS s = fl.nvEncOpenEncodeSessionEx(&sp, &impl_->encoder);
    if (s != NV_ENC_SUCCESS) {
        impl_->encoder = nullptr;  // driver leaves it undefined on failure
        err = status_string(fl, nullptr, s, "nvEncOpenEncodeSessionEx");
        close();
        return false;
    }

    // 2) Build the encode config (P4 preset for `tuning` + SyLC overrides). Shared with
    //    reconfigure() via build_encode_config so the two paths can never drift.
    if (!build_encode_config(fl, impl_->encoder, cfg, tuning, impl_->encodeConfig, err)) {
        close();
        return false;
    }

    // 3) Initialize the encoder from that config.
    NV_ENC_INITIALIZE_PARAMS ip{};
    ip.version      = NV_ENC_INITIALIZE_PARAMS_VER;
    ip.encodeGUID   = NV_ENC_CODEC_HEVC_GUID;
    ip.presetGUID   = NV_ENC_PRESET_P4_GUID;
    ip.tuningInfo   = tuning;
    ip.encodeWidth  = cfg.width;
    ip.encodeHeight = cfg.height;
    ip.darWidth     = cfg.width;
    ip.darHeight    = cfg.height;
    ip.frameRateNum = impl_->fps;
    ip.frameRateDen = 1;
    ip.enablePTD    = 1;
    ip.encodeConfig = &impl_->encodeConfig;
    s = fl.nvEncInitializeEncoder(impl_->encoder, &ip);
    if (s != NV_ENC_SUCCESS) {
        err = status_string(fl, impl_->encoder, s, "nvEncInitializeEncoder");
        close();
        return false;
    }

    // 4) Allocate one output bitstream buffer (single in-flight frame, v1).
    NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
    cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
    s = fl.nvEncCreateBitstreamBuffer(impl_->encoder, &cb);
    if (s != NV_ENC_SUCCESS) {
        err = status_string(fl, impl_->encoder, s, "nvEncCreateBitstreamBuffer");
        close();
        return false;
    }
    impl_->bitstreamBuffer = cb.bitstreamBuffer;
    return true;
}

bool NvencEncoder::reconfigure(const NvencConfig& cfg, std::string& err) {
    if (!impl_ || !impl_->encoder) { err = "reconfigure: not open"; return false; }
    NV_ENCODE_API_FUNCTION_LIST& fl = impl_->fl;
    if (!fl.nvEncReconfigureEncoder) {
        err = "reconfigure: driver has no nvEncReconfigureEncoder";
        return false;
    }

    const NV_ENC_TUNING_INFO tuning = (cfg.mode == CastMode::LosslessWired)
        ? NV_ENC_TUNING_INFO_LOSSLESS
        : NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;

    // Build the target config into a LOCAL. nvEncReconfigureEncoder validates atomically
    // (on rejection the session keeps its old config), so we must not overwrite
    // impl_->encodeConfig until the driver has ACCEPTED the change.
    NV_ENC_CONFIG newCfg{};
    if (!build_encode_config(fl, impl_->encoder, cfg, tuning, newCfg, err)) return false;

    const bool modeSwitch = (cfg.mode != impl_->mode);

    // NV_ENC_RECONFIGURE_PARAMS wraps a full re-init params block (same fields as open's
    // NV_ENC_INITIALIZE_PARAMS) + the new config. forceIDR makes the next frame an IDR so
    // the decoder recovers instantly at the switch; resetEncoder=0 keeps rate-control state
    // (seamless — correct for a same-mode bitrate change).
    NV_ENC_RECONFIGURE_PARAMS rp{};
    rp.version = NV_ENC_RECONFIGURE_PARAMS_VER;
    NV_ENC_INITIALIZE_PARAMS& ip = rp.reInitEncodeParams;
    ip.version      = NV_ENC_INITIALIZE_PARAMS_VER;
    ip.encodeGUID   = NV_ENC_CODEC_HEVC_GUID;
    ip.presetGUID   = NV_ENC_PRESET_P4_GUID;
    ip.tuningInfo   = tuning;
    ip.encodeWidth  = cfg.width;
    ip.encodeHeight = cfg.height;
    ip.darWidth     = cfg.width;
    ip.darHeight    = cfg.height;
    ip.frameRateNum = cfg.fps ? cfg.fps : 1u;
    ip.frameRateDen = 1;
    ip.enablePTD    = 1;
    ip.encodeConfig = &newCfg;
    rp.forceIDR     = 1;
    rp.resetEncoder = 0;

    NVENCSTATUS s = fl.nvEncReconfigureEncoder(impl_->encoder, &rp);
    if (s != NV_ENC_SUCCESS && modeSwitch) {
        // A lossless<->cbr mode switch changes tuning + rate-control mode (and the GOP
        // structure), which reconfigure may refuse without an internal reset. Retry once
        // resetting the encoder's state (still no session teardown). If it still fails the
        // caller falls back to a full encoder reopen.
        rp.resetEncoder = 1;
        s = fl.nvEncReconfigureEncoder(impl_->encoder, &rp);
    }
    if (s != NV_ENC_SUCCESS) {
        err = status_string(fl, impl_->encoder, s, "nvEncReconfigureEncoder");
        return false;
    }

    // Accepted — commit the new config + geometry so close()/a later reconfigure stay consistent.
    impl_->encodeConfig         = newCfg;
    impl_->encodeConfig.version = NV_ENC_CONFIG_VER;
    impl_->width  = cfg.width;
    impl_->height = cfg.height;
    impl_->fps    = cfg.fps ? cfg.fps : 1u;
    impl_->mode   = cfg.mode;
    return true;
}

bool NvencEncoder::supportsLossless() const {
    if (!impl_ || !impl_->encoder) return false;
    NV_ENC_CAPS_PARAM cp{};
    cp.version     = NV_ENC_CAPS_PARAM_VER;
    cp.capsToQuery = NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE;
    int v = 0;
    NVENCSTATUS s = impl_->fl.nvEncGetEncodeCaps(impl_->encoder, NV_ENC_CODEC_HEVC_GUID, &cp, &v);
    return (s == NV_ENC_SUCCESS) && (v != 0);
}

bool NvencEncoder::registerInput(ID3D11Texture2D* tex, void** outRegistered, std::string& err) {
    if (!impl_ || !impl_->encoder) { err = "registerInput: not open";      return false; }
    if (!tex)                      { err = "registerInput: null texture";  return false; }
    if (!outRegistered)            { err = "registerInput: null out";      return false; }

    NV_ENC_REGISTER_RESOURCE r{};
    r.version            = NV_ENC_REGISTER_RESOURCE_VER;
    r.resourceType       = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
    r.resourceToRegister = tex;
    r.width              = impl_->width;
    r.height             = impl_->height;
    r.pitch              = 0;                              // 0 for DirectX resources
    r.bufferFormat       = impl_->bufFmt;
    r.bufferUsage        = NV_ENC_INPUT_IMAGE;
    NVENCSTATUS s = impl_->fl.nvEncRegisterResource(impl_->encoder, &r);
    if (s != NV_ENC_SUCCESS) {
        err = status_string(impl_->fl, impl_->encoder, s, "nvEncRegisterResource");
        return false;
    }
    impl_->registered.push_back(r.registeredResource);
    *outRegistered = r.registeredResource;
    return true;
}

bool NvencEncoder::encode(void* registeredInput, int64_t ptsMs, bool forceIdr,
                          std::vector<std::vector<uint8_t>>& outPackets, std::string& err) {
    if (!impl_ || !impl_->encoder) { err = "encode: not open";    return false; }
    if (!registeredInput)          { err = "encode: null input";  return false; }
    NV_ENCODE_API_FUNCTION_LIST& fl = impl_->fl;

    // Map the registered resource for this frame.
    NV_ENC_MAP_INPUT_RESOURCE m{};
    m.version            = NV_ENC_MAP_INPUT_RESOURCE_VER;
    m.registeredResource = static_cast<NV_ENC_REGISTERED_PTR>(registeredInput);
    NVENCSTATUS s = fl.nvEncMapInputResource(impl_->encoder, &m);
    if (s != NV_ENC_SUCCESS) {
        err = status_string(fl, impl_->encoder, s, "nvEncMapInputResource");
        return false;
    }

    // Submit the picture.
    NV_ENC_PIC_PARAMS pp{};
    pp.version         = NV_ENC_PIC_PARAMS_VER;
    pp.inputBuffer     = m.mappedResource;
    pp.bufferFmt       = impl_->bufFmt;
    pp.inputWidth      = impl_->width;
    pp.inputHeight     = impl_->height;
    pp.outputBitstream = impl_->bitstreamBuffer;
    pp.inputTimeStamp  = static_cast<uint64_t>(ptsMs);
    pp.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;
    pp.encodePicFlags  = forceIdr
        ? (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS)
        : 0u;
    s = fl.nvEncEncodePicture(impl_->encoder, &pp);

    bool ok = true;
    if (s == NV_ENC_SUCCESS) {
        // Output is ready — lock it out (single in-flight frame + synchronous lock).
        ok = lock_and_append(fl, impl_->encoder, impl_->bitstreamBuffer, outPackets, err);
    } else if (s != NV_ENC_ERR_NEED_MORE_INPUT) {
        // NEED_MORE_INPUT = the encoder buffered this frame (e.g. B-frame reorder);
        // no output this call and not an error. Anything else is a real failure.
        err = status_string(fl, impl_->encoder, s, "nvEncEncodePicture");
        ok = false;
    }

    // Always unmap the resource we mapped above.
    NVENCSTATUS us = fl.nvEncUnmapInputResource(impl_->encoder, m.mappedResource);
    if (us != NV_ENC_SUCCESS && ok) {
        err = status_string(fl, impl_->encoder, us, "nvEncUnmapInputResource");
        ok = false;
    }
    return ok;
}

void NvencEncoder::flush(std::vector<std::vector<uint8_t>>& outPackets) {
    // No trailing output is ever produced in v1 (see below). The parameter stays
    // in the stable interface for a future buffering mode.
    (void)outPackets;
    if (!impl_ || !impl_->encoder) return;

    // Submit a bufferless end-of-stream marker (EOS flag, no input, no output
    // buffer) to signal end-of-stream, then RETURN — do NOT lock the bitstream.
    //
    // With frameIntervalP=1 (forced for BOTH modes) every encode() already locked
    // out its own output synchronously, so nothing is ever buffered at flush()
    // time: there is no pending frame to drain. The EOS submit returns
    // NV_ENC_SUCCESS but produces NO output, so a blocking nvEncLockBitstream here
    // would wait forever for a frame that never arrives (that was the previous
    // version's hang — a SUCCESS return does NOT imply a lockable frame). EOS
    // submit then return; the drain is a no-op.
    //
    // A future asynchronous / B-frame buffering mode must NOT reinstate an
    // unconditional (or merely SUCCESS-gated) blocking lock: it needs a real
    // output-buffer pool plus a submitted-minus-retrieved counter (or async
    // completion events / non-blocking locks) so it drains exactly the frames it
    // knows are pending. Out of scope for the single-buffer v1.
    NV_ENC_PIC_PARAMS eos{};
    eos.version        = NV_ENC_PIC_PARAMS_VER;
    eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    impl_->fl.nvEncEncodePicture(impl_->encoder, &eos);
}

void NvencEncoder::close() {
    if (!impl_) return;
    NV_ENCODE_API_FUNCTION_LIST& fl = impl_->fl;

    if (impl_->encoder) {
        for (NV_ENC_REGISTERED_PTR r : impl_->registered) {
            if (r && fl.nvEncUnregisterResource) fl.nvEncUnregisterResource(impl_->encoder, r);
        }
        if (impl_->bitstreamBuffer && fl.nvEncDestroyBitstreamBuffer) {
            fl.nvEncDestroyBitstreamBuffer(impl_->encoder, impl_->bitstreamBuffer);
        }
        if (fl.nvEncDestroyEncoder) fl.nvEncDestroyEncoder(impl_->encoder);
        impl_->encoder = nullptr;
    }
    impl_->registered.clear();
    impl_->bitstreamBuffer = nullptr;

    if (impl_->dll) { ::FreeLibrary(impl_->dll); impl_->dll = nullptr; }
}

} // namespace sylc

#endif // _WIN32
