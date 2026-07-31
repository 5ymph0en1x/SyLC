// DepthEngine: thin ONNX Runtime wrapper around DA3 depth models
// model — the inference core for SyLC's real-time 2D->3D "synth3d" mode
// (see .superpowers/sdd/2026-07-28-2d-to-3d-realtime, Task 2).
//
// onnxruntime.dll (and its execution providers: DirectML.dll, TensorRT on
// GPU builds that have it) is DYNAMIC-LOADED at runtime via LoadLibraryExW +
// GetProcAddress — nothing is linked against onnxruntime.lib. This mirrors
// the dynamic-loading idiom already used for NVENC (nvenc_encoder.cpp) and
// D3DCompile: only the vendored ONNX Runtime C API headers are compiled in.
//
// NOT thread-safe: single owner thread. Per spec §7 ("playback never dies
// for 3D"), init()/infer() run on the synth3d worker thread, never the GUI
// or decode thread, and any failure degrades to 2D rather than blocking.
#pragma once

#include <cstdint>
#include <string>

// Historical inference grid, and the default everywhere a dimension is optional:
// 756 = ViT patch 14 * 54 (the quality preset). The depth-preset feature also
// ships 518 = 14 * 37; any other multiple of the patch size would work as long
// as the exported graph matches it.
constexpr int kDefaultDepthSide = 756;

struct DepthConfig {
    std::wstring model_path;    // absolute path to the .onnx
    std::wstring ort_dir;       // dir containing onnxruntime.dll ("" = default search)
    // Convert positive camera-space depth Z to inverse depth 1/Z. This is the
    // physically correct quantity for binocular disparity and remains
    // scale/shift stabilized by DepthStabilizer.
    bool invert_output = false;
    // Backward-compatible square inference grid. `width`/`height`, when both
    // positive, take precedence; otherwise side is used for both dimensions.
    int side = kDefaultDepthSide;
    // Fixed rectangular inference grid, in pixels. Both dimensions must match
    // the model export and be positive. DA3 uses 14-pixel patches, so shipped
    // graphs use multiples of 14 even though the runtime merely validates > 0.
    int width = 0;
    int height = 0;
};

class DepthEngine {                     // NOT thread-safe; single owner thread
public:
    DepthEngine();
    ~DepthEngine();

    DepthEngine(const DepthEngine&) = delete;            // unique resource owner (raw Impl*):
    DepthEngine& operator=(const DepthEngine&) = delete;  // a shallow copy would double-free impl_

    // Loads onnxruntime.dll, builds the provider ladder (TensorRT -> DirectML
    // -> CPU) and creates the session. Takes seconds; call OFF the GUI thread.
    bool init(const DepthConfig& cfg, std::string& err);

    // in : view-major CHW float32, input_views() x 3 x H x W with W/H from
    //      DepthConfig; ImageNet-normalized RGB.
    //      Single-frame models report one view.
    // out: H x W float32 inverse depth (higher = closer), raw model scale.
    // confidence is optional; models without a confidence output yield 1.0.
    bool infer(const float* chw, float* out, std::string& err,
               float* confidence = nullptr);
    int input_views() const;

    const char* provider() const;       // "TensorRT" | "DirectML" | "CPU" | "none"

    void shutdown();                    // idempotent; also called by the destructor

private:
    struct Impl;
    Impl* impl_;                        // opaque, like NativeRenderer / NvencEncoder
};
