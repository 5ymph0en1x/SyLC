// DepthEngine implementation. See depth_engine.h.
//
// onnxruntime.dll and its execution providers are dynamic-loaded (no .lib is
// linked, see CMakeLists.txt): OrtGetApiBase is resolved via GetProcAddress,
// so this file builds and runs unmodified whether the deployed onnxruntime.dll
// carries DirectML, TensorRT, both, or neither (CPU-only fallback).
//
// The TensorRT rung is tried FIRST, through the OrtApi struct directly
// (CreateTensorRTProviderOptions / UpdateTensorRTProviderOptions /
// SessionOptionsAppendExecutionProvider_TensorRT_V2 / ReleaseTensorRTProviderOptions
// -- no GetProcAddress needed: these are struct members like CreateEnv, always
// present in this ABI version). This is the V2 API, which lets us set the
// engine-cache options (a from-scratch TensorRT engine compile is minutes; a
// cached one is seconds) -- the legacy `OrtSessionOptionsAppendExecutionProvider_
// Tensorrt` symbol (still probed via GetProcAddress, kept as a last-chance
// fallback) has no such options. DirectML/CPU are probed via GetProcAddress as
// before.
#define WIN32_LEAN_AND_MEAN
#include "depth_engine.h"
#include "onnxruntime_c_api.h"
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace {

// Copies the ORT error message and releases the status. Returns true when
// there was no error (status == nullptr), false (with err populated) otherwise.
bool check(const OrtApi* api, OrtStatus* status, std::string& err) {
    if (!status) return true;
    err = api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    return false;
}

// Same idiom as mvc_decoder.cpp's utf8() helper: OrtApi's provider-options
// strings (e.g. trt_engine_cache_path) are UTF-8, but ort_dir arrives as a
// wide path.
std::string utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int required = ::WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        nullptr, 0, nullptr, nullptr);
    if (required <= 0) return {};
    std::string result(static_cast<size_t>(required), '\0');
    ::WideCharToMultiByte(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), required, nullptr, nullptr);
    return result;
}

using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)(void);
using ProviderAppendFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);

}  // namespace

struct DepthEngine::Impl {
    HMODULE dll = nullptr;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSessionOptions* so = nullptr;
    OrtSession* session = nullptr;
    OrtMemoryInfo* mem = nullptr;
    std::string in_name, depth_name, confidence_name, provider = "none";
    // Non-fatal diagnostic scratch field (fix round 1, F3): records a failed
    // SetDllDirectoryW call below. Never causes init() to fail and is not
    // yet exposed by any accessor -- this file has no logging channel (see
    // the header comment) -- but the state is captured rather than silently
    // dropped, ready for whenever one exists.
    std::string err_;
    bool invert_output = false;
    int width = kDefaultDepthSide;
    int height = kDefaultDepthSide;
    // Sized for the 5D multi-view export [1, num_images, 3, H, W]; a plain 4D
    // export uses only the first 4 elements (in_rank tells Run how many).
    int64_t in_shape[5] = {1, 1, 3, kDefaultDepthSide, kDefaultDepthSide};
    size_t in_rank = 4;
    int input_views = 1;
};

DepthEngine::DepthEngine() : impl_(new Impl()) {}

DepthEngine::~DepthEngine() {
    shutdown();
    delete impl_;
}

bool DepthEngine::init(const DepthConfig& cfg, std::string& err) {
    Impl* d = impl_;
    d->invert_output = cfg.invert_output;
    const int requested_width =
        (cfg.width > 0 && cfg.height > 0) ? cfg.width : cfg.side;
    const int requested_height =
        (cfg.width > 0 && cfg.height > 0) ? cfg.height : cfg.side;
    if (requested_width <= 0 || requested_height <= 0) {
        err = "invalid inference grid " + std::to_string(requested_width) +
              "x" + std::to_string(requested_height);
        return false;
    }
    d->width = requested_width;
    d->height = requested_height;

    // 1) Load onnxruntime.dll — prefer ort_dir (also pulls DirectML.dll from
    //    the same directory), fall back to the normal DLL search path.
    //
    // SetDllDirectoryW(cfg.ort_dir) — mirrors tools_dev/setup_tensorrt.py's
    // own (already-working) _real_tensorrt_session_test(), which sets this
    // for the identical reason: LOAD_WITH_ALTERED_SEARCH_PATH below only
    // affects resolution of onnxruntime.dll's OWN direct dependencies for
    // THIS LoadLibraryExW call. It does NOT extend to NESTED, runtime-time
    // LoadLibrary calls made later by a transitively-loaded DLL -- e.g.
    // cudnn64_9.dll (loaded as a dependency of onnxruntime_providers_
    // tensorrt.dll) dynamically loads its own modular backend engine DLLs
    // (cudnn_ops64_9.dll etc.) on demand, at the first real cudnnCreate()
    // call during TensorRT engine BUILD -- and that nested load uses the
    // plain OS default search order, which does not include ort_dir unless
    // something process-wide has added it. Root-caused after a real engine
    // build reproducibly hard-crashed with "Invalid handle. Cannot load
    // symbol cudnnCreate" despite every required DLL being physically
    // present in ort_dir (see task-4-report.md, "The crash" and its
    // resolution).
    //
    // Fix round 1 (F3) — GATED, and never restored:
    //   - GATED: only called when cfg.ort_dir actually carries the opt-in
    //     TensorRT runtime (a cheap FindFirstFileW check for any
    //     nvinfer*.dll below), never for a plain DirectML-only deployment.
    //     SetDllDirectoryW mutates PROCESS-WIDE loader search-path state --
    //     a DML-only process (the common case) has no reason to have that
    //     state touched at all, and DepthEngine::init() runs on a
    //     background worker thread (see depth_engine.h), so an unconditional
    //     call would mutate global process state from a background thread
    //     for zero benefit whenever TensorRT isn't even in play.
    //   - NEVER RESTORED once set: cudnn64_9.dll's nested backend loads are
    //     not confined to this first CreateSession below -- they can also
    //     occur lazily during LATER infer() calls (a different input shape
    //     or a cache miss can trigger a fresh sub-engine build well after
    //     init() returns), so the search-path addition must outlive this
    //     function for the life of the process; there is no safe point at
    //     which to revert it.
    if (!cfg.ort_dir.empty()) {
        const std::wstring nvinfer_glob = cfg.ort_dir + L"\\nvinfer*.dll";
        WIN32_FIND_DATAW find_data{};
        HANDLE h_find = ::FindFirstFileW(nvinfer_glob.c_str(), &find_data);
        const bool has_trt_runtime = (h_find != INVALID_HANDLE_VALUE);
        if (h_find != INVALID_HANDLE_VALUE) ::FindClose(h_find);

        if (has_trt_runtime && !::SetDllDirectoryW(cfg.ort_dir.c_str())) {
            // Non-fatal: recorded, not returned. A failed SetDllDirectoryW
            // just means the ladder below may fall through to DML/CPU later
            // instead of reaching TensorRT -- not a reason to abort init().
            d->err_ = "SetDllDirectoryW failed (non-fatal): GetLastError=" +
                      std::to_string(::GetLastError());
        }
        const std::wstring path = cfg.ort_dir + L"\\onnxruntime.dll";
        d->dll = ::LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!d->dll) d->dll = ::LoadLibraryW(L"onnxruntime.dll");
    if (!d->dll) {
        err = "onnxruntime.dll not found";
        return false;
    }

    // 2) Resolve the C API entry point.
    auto get_api_base = reinterpret_cast<GetApiBaseFn>(::GetProcAddress(d->dll, "OrtGetApiBase"));
    if (!get_api_base) {
        err = "OrtGetApiBase not found in onnxruntime.dll";
        return false;
    }
    const OrtApiBase* base = get_api_base();
    if (!base) {
        err = "OrtGetApiBase() returned null";
        return false;
    }
    d->api = base->GetApi(ORT_API_VERSION);
    if (!d->api) {
        err = "GetApi(ORT_API_VERSION) failed — onnxruntime.dll is too old";
        return false;
    }
    const OrtApi* api = d->api;

    // 3) Env + session options.
    if (!check(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "sylc", &d->env), err)) return false;
    if (!check(api, api->CreateSessionOptions(&d->so), err)) return false;
    if (!check(api, api->SetIntraOpNumThreads(d->so, 2), err)) return false;
    if (!check(api, api->SetSessionGraphOptimizationLevel(d->so, ORT_ENABLE_ALL), err)) return false;

    // 4) Provider ladder: TensorRT (V2 API + engine cache) -> TensorRT (legacy
    //    symbol, no cache) -> DirectML -> CPU.
    //
    //    The V2 rung always runs (regardless of whether ort_dir carries the
    //    opt-in TensorRT runtime): CreateTensorRTProviderOptions and
    //    UpdateTensorRTProviderOptions only touch an in-memory options object,
    //    so they succeed on any onnxruntime.dll build. It is
    //    SessionOptionsAppendExecutionProvider_TensorRT_V2 that fails -- with an
    //    OrtStatus, not a crash -- when it tries to load the sibling
    //    onnxruntime_providers_tensorrt.dll and finds it absent, which is the
    //    normal case on a machine without ort_tensorrt/. That failure (and any
    //    earlier one) falls through to the legacy symbol probe, then DML, then
    //    CPU. `ladder_err` is intentionally never surfaced: every rung here is
    //    an expected, non-fatal ladder step -- provider() is the caller-visible
    //    signal of which one won.
    d->provider = "CPU";
    bool trt_appended = false;
    {
        OrtTensorRTProviderOptionsV2* trt_opts = nullptr;
        std::string ladder_err;
        if (check(api, api->CreateTensorRTProviderOptions(&trt_opts), ladder_err)) {
            std::string cache_path_utf8;
            if (!cfg.ort_dir.empty()) {
                const std::wstring cache_dir = cfg.ort_dir + L"\\trt_cache";
                ::CreateDirectoryW(cache_dir.c_str(), nullptr);  // idempotent
                cache_path_utf8 = utf8(cache_dir);
            }
            const char* keys[4] = {
                "trt_fp16_enable", "device_id",
                "trt_engine_cache_enable", "trt_engine_cache_path",
            };
            const char* values[4] = {"1", "0", "1", cache_path_utf8.c_str()};
            // Only ask for the cache when we have somewhere to put it.
            const size_t num_keys = cache_path_utf8.empty() ? 2u : 4u;
            if (check(api, api->UpdateTensorRTProviderOptions(
                               trt_opts, keys, values, num_keys),
                      ladder_err)) {
                trt_appended = check(
                    api,
                    api->SessionOptionsAppendExecutionProvider_TensorRT_V2(d->so, trt_opts),
                    ladder_err);
            }
        }
        // Partial objects are released regardless of how far the ladder got.
        if (trt_opts) api->ReleaseTensorRTProviderOptions(trt_opts);
    }

    if (trt_appended) {
        d->provider = "TensorRT";
    } else {
        // Last-chance fallback: the legacy (pre-V2, no cache options) symbol,
        // probed the same GetProcAddress way as DirectML below.
        auto trt_legacy = reinterpret_cast<ProviderAppendFn>(
            ::GetProcAddress(d->dll, "OrtSessionOptionsAppendExecutionProvider_Tensorrt"));
        if (trt_legacy) {
            OrtStatus* st = trt_legacy(d->so, 0);
            if (!st) d->provider = "TensorRT";
            else api->ReleaseStatus(st);
        }
    }

    auto dml = reinterpret_cast<ProviderAppendFn>(
        ::GetProcAddress(d->dll, "OrtSessionOptionsAppendExecutionProvider_DML"));
    if (d->provider == "CPU" && dml) {
        OrtStatus* st = dml(d->so, 0);
        if (!st) d->provider = "DirectML";
        else api->ReleaseStatus(st);
    }

    // 5) Create the session (wide path — ORTCHAR_T == wchar_t on Windows).
    if (!check(api, api->CreateSession(d->env, cfg.model_path.c_str(), d->so, &d->session), err))
        return false;

    // 6) Input/output names (default allocator) + input rank/dtype probe.
    OrtAllocator* alloc = nullptr;
    if (!check(api, api->GetAllocatorWithDefaultOptions(&alloc), err)) return false;

    size_t n_in = 0, n_out = 0;
    if (!check(api, api->SessionGetInputCount(d->session, &n_in), err)) return false;
    if (!check(api, api->SessionGetOutputCount(d->session, &n_out), err)) return false;
    if (n_in == 0 || n_out == 0) {
        err = "model has no inputs/outputs";
        return false;
    }

    char* name = nullptr;
    if (!check(api, api->SessionGetInputName(d->session, 0, alloc, &name), err)) return false;
    d->in_name = name;
    api->AllocatorFree(alloc, name);

    // Prefer the output literally named "predicted_depth" (the DA3 multi-view
    // export's depth head); fall back to output 0 for any other export shape.
    d->depth_name.clear();
    d->confidence_name.clear();
    for (size_t i = 0; i < n_out; ++i) {
        char* out_name = nullptr;
        if (!check(api, api->SessionGetOutputName(d->session, i, alloc, &out_name), err))
            return false;
        const std::string s = out_name;
        api->AllocatorFree(alloc, out_name);
        if (i == 0) d->depth_name = s;
        if (s == "predicted_depth" || s == "depth") d->depth_name = s;
        if (s == "depth_conf" || s == "confidence") d->confidence_name = s;
    }

    // Input rank: DA3-Small's plain export is 4D [1,3,H,W]; the multi-view
    // export (the one actually deployed) is 5D [1, num_images, 3, H, W].
    OrtTypeInfo* type_info = nullptr;
    if (!check(api, api->SessionGetInputTypeInfo(d->session, 0, &type_info), err)) return false;
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    if (!check(api, api->CastTypeInfoToTensorInfo(type_info, &tensor_info), err)) {
        api->ReleaseTypeInfo(type_info);
        return false;
    }
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    bool ok = check(api, api->GetTensorElementType(tensor_info, &elem_type), err);
    size_t rank = 0;
    if (ok) ok = check(api, api->GetDimensionsCount(tensor_info, &rank), err);
    int64_t probed_dims[5] = {1, 1, 3, d->height, d->width};
    if (ok && rank <= 5)
        ok = check(api, api->GetDimensions(tensor_info, probed_dims, rank), err);
    api->ReleaseTypeInfo(type_info);  // also releases tensor_info (owned by type_info)
    if (!ok) return false;

    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        err = "model must be exported with float32 I/O (re-export, see Task 1 step 5)";
        return false;
    }
    if (rank == 4) {
        d->in_rank = 4;
        d->input_views = 1;
        d->in_shape[0] = 1;
        d->in_shape[1] = 3;
        d->in_shape[2] = d->height;
        d->in_shape[3] = d->width;
    } else if (rank == 5) {
        d->in_rank = 5;
        d->input_views = probed_dims[1] > 0
            ? static_cast<int>(probed_dims[1]) : 1;
        d->in_shape[0] = 1;
        d->in_shape[1] = d->input_views;
        d->in_shape[2] = 3;
        d->in_shape[3] = d->height;
        d->in_shape[4] = d->width;
    } else {
        err = "unsupported model input rank " + std::to_string(rank) + " (expected 4 or 5)";
        return false;
    }

    // A fixed-shape export that disagrees with the requested grid would be fed
    // the wrong element count and answer with garbage (or an ORT shape error
    // deep in Run). Catch it here, where the message can name both sides.
    // A dynamic axis reports <= 0 and is left to the model's own resolution.
    const int64_t model_h = probed_dims[d->in_rank - 2];
    const int64_t model_w = probed_dims[d->in_rank - 1];
    if ((model_h > 0 && model_h != d->height) ||
        (model_w > 0 && model_w != d->width)) {
        err = "model input grid " + std::to_string(model_h) + "x" +
              std::to_string(model_w) + " does not match the requested grid " +
              std::to_string(d->height) + "x" + std::to_string(d->width);
        return false;
    }

    // 7) CPU memory descriptor for CreateTensorWithDataAsOrtValue.
    if (!check(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &d->mem), err))
        return false;

    return true;
}

bool DepthEngine::infer(const float* chw, float* out, std::string& err,
                        float* confidence) {
    Impl* d = impl_;
    if (!d->session) {
        err = "DepthEngine not initialized";
        return false;
    }
    const OrtApi* api = d->api;
    const size_t frame =
        static_cast<size_t>(d->width) * static_cast<size_t>(d->height);
    const size_t in_elems =
        static_cast<size_t>(d->input_views) * 3 * frame;

    OrtValue* in_tensor = nullptr;
    if (!check(api, api->CreateTensorWithDataAsOrtValue(
                        d->mem, const_cast<float*>(chw), in_elems * sizeof(float),
                        d->in_shape, d->in_rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor),
                err))
        return false;

    const char* in_names[1] = {d->in_name.c_str()};
    const bool request_confidence = confidence && !d->confidence_name.empty();
    const char* out_names[2] = {
        d->depth_name.c_str(),
        request_confidence ? d->confidence_name.c_str() : nullptr,
    };
    const OrtValue* inputs[1] = {in_tensor};
    OrtValue* out_vals[2] = {nullptr, nullptr};
    const size_t requested_outputs = request_confidence ? 2u : 1u;

    if (!check(api, api->Run(d->session, nullptr, in_names, inputs, 1,
                             out_names, requested_outputs, out_vals),
               err)) {
        api->ReleaseValue(in_tensor);
        return false;
    }

    auto tensor_data = [&](OrtValue* value, const char* label,
                           const float*& data, size_t& total) -> bool {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        if (!check(api, api->GetTensorTypeAndShape(value, &shape_info), err))
            return false;
        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        bool ok = check(api, api->GetTensorElementType(shape_info, &type), err);
        total = 0;
        if (ok) ok = check(api, api->GetTensorShapeElementCount(shape_info, &total), err);
        api->ReleaseTensorTypeAndShapeInfo(shape_info);
        if (!ok) return false;
        if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            err = std::string(label) + " output must be float32";
            return false;
        }
        if (total < frame || total % frame != 0) {
            err = std::string("unexpected ") + label + " element count " +
                  std::to_string(total) + " (not a multiple of " +
                  std::to_string(frame) + ")";
            return false;
        }
        float* mutable_data = nullptr;
        if (!check(api, api->GetTensorMutableData(
                           value, reinterpret_cast<void**>(&mutable_data)), err))
            return false;
        data = mutable_data;
        return true;
    };

    const float* depth_data = nullptr;
    size_t depth_total = 0;
    bool outputs_ok = tensor_data(
        out_vals[0], "depth", depth_data, depth_total);
    const float* confidence_data = nullptr;
    size_t confidence_total = 0;
    if (outputs_ok && request_confidence)
        outputs_ok = tensor_data(
            out_vals[1], "confidence", confidence_data, confidence_total);
    if (!outputs_ok) {
        for (size_t i = 0; i < requested_outputs; ++i)
            if (out_vals[i]) api->ReleaseValue(out_vals[i]);
        api->ReleaseValue(in_tensor);
        return false;
    }

    // Ranks 2/3/4/5 all land here as a flat buffer; take the LAST H*W
    // floats (a 5D model yields (1,1,N,N) — a single leading frame).
    const float* last = depth_data + (depth_total - frame);
    if (d->invert_output) {
        for (size_t i = 0; i < frame; ++i) {
            const float z = last[i];
            out[i] = std::isfinite(z) && z > 1.0e-6f ? 1.0f / z : 0.0f;
        }
    } else {
        std::memcpy(out, last, frame * sizeof(float));
    }
    if (confidence) {
        if (request_confidence) {
            const float* conf_last =
                confidence_data + (confidence_total - frame);
            std::memcpy(confidence, conf_last, frame * sizeof(float));
        } else {
            std::fill(confidence, confidence + frame, 1.0f);
        }
    }

    for (size_t i = 0; i < requested_outputs; ++i)
        if (out_vals[i]) api->ReleaseValue(out_vals[i]);
    api->ReleaseValue(in_tensor);
    return true;
}

const char* DepthEngine::provider() const { return impl_->provider.c_str(); }
int DepthEngine::input_views() const { return impl_->input_views; }

void DepthEngine::shutdown() {
    Impl* d = impl_;
    if (!d) return;
    if (d->api) {
        if (d->mem) {
            d->api->ReleaseMemoryInfo(d->mem);
            d->mem = nullptr;
        }
        if (d->session) {
            d->api->ReleaseSession(d->session);
            d->session = nullptr;
        }
        if (d->so) {
            d->api->ReleaseSessionOptions(d->so);
            d->so = nullptr;
        }
        if (d->env) {
            d->api->ReleaseEnv(d->env);
            d->env = nullptr;
        }
    }
    d->api = nullptr;
    if (d->dll) {
        ::FreeLibrary(d->dll);
        d->dll = nullptr;
    }
    d->provider = "none";
}
