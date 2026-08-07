#include "nvof_flow.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "nvOpticalFlowCuda.h"

namespace sylc {

namespace {

using CreateApiFn = NV_OF_STATUS (NVOFAPI *)(
    uint32_t, NV_OF_CUDA_API_FUNCTION_LIST*);

const char* of_status_name(NV_OF_STATUS status) {
    switch (status) {
        case NV_OF_SUCCESS: return "success";
        case NV_OF_ERR_OF_NOT_AVAILABLE: return "not available";
        case NV_OF_ERR_UNSUPPORTED_DEVICE: return "unsupported device";
        case NV_OF_ERR_DEVICE_DOES_NOT_EXIST: return "device missing";
        case NV_OF_ERR_INVALID_PTR: return "invalid pointer";
        case NV_OF_ERR_INVALID_PARAM: return "invalid parameter";
        case NV_OF_ERR_INVALID_CALL: return "invalid call";
        case NV_OF_ERR_INVALID_VERSION: return "invalid version";
        case NV_OF_ERR_OUT_OF_MEMORY: return "out of memory";
        case NV_OF_ERR_NOT_INITIALIZED: return "not initialized";
        case NV_OF_ERR_UNSUPPORTED_FEATURE: return "unsupported feature";
        default: return "generic error";
    }
}

std::string cuda_error(CUresult result, const char* operation) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream out;
    out << operation << ": " << (name ? name : "CUDA error");
    if (text) out << " (" << text << ")";
    return out.str();
}

}  // namespace

struct NvofFlow::Impl {
#ifdef _WIN32
    HMODULE module = nullptr;
#endif
    CUcontext context = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST api = {};
    NvOFHandle handle = nullptr;
    NvOFGPUBufferHandle input = nullptr;
    NvOFGPUBufferHandle reference = nullptr;
    NvOFGPUBufferHandle output = nullptr;
    NvOFGPUBufferHandle cost = nullptr;
    CUdeviceptr input_ptr = 0;
    CUdeviceptr reference_ptr = 0;
    CUdeviceptr output_ptr = 0;
    CUdeviceptr cost_ptr = 0;
    NV_OF_CUDA_BUFFER_STRIDE_INFO input_stride = {};
    NV_OF_CUDA_BUFFER_STRIDE_INFO reference_stride = {};
    NV_OF_CUDA_BUFFER_STRIDE_INFO output_stride = {};
    NV_OF_CUDA_BUFFER_STRIDE_INFO cost_stride = {};
    int width = 0;
    int height = 0;
    int output_width = 0;
    int output_height = 0;
    int grid = 1;
    std::string perf;

    bool of_ok(NV_OF_STATUS status, const char* operation,
               std::string& error) const {
        if (status == NV_OF_SUCCESS) return true;
        std::ostringstream out;
        out << operation << ": " << of_status_name(status)
            << " (" << static_cast<int>(status) << ")";
        if (handle && api.nvOFGetLastError) {
            char detail[512] = {};
            uint32_t size = sizeof(detail);
            if (api.nvOFGetLastError(handle, detail, &size) == NV_OF_SUCCESS &&
                detail[0]) {
                out << " — " << detail;
            }
        }
        error = out.str();
        return false;
    }

    bool create_buffer(uint32_t w, uint32_t h, NV_OF_BUFFER_USAGE usage,
                       NV_OF_BUFFER_FORMAT format,
                       NvOFGPUBufferHandle& buffer,
                       CUdeviceptr& pointer,
                       NV_OF_CUDA_BUFFER_STRIDE_INFO& stride,
                       std::string& error) {
        NV_OF_BUFFER_DESCRIPTOR descriptor = {};
        descriptor.width = w;
        descriptor.height = h;
        descriptor.bufferUsage = usage;
        descriptor.bufferFormat = format;
        if (!of_ok(api.nvOFCreateGPUBufferCuda(
                handle, &descriptor, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
                &buffer), "nvOFCreateGPUBufferCuda", error)) {
            return false;
        }
        pointer = api.nvOFGPUBufferGetCUdeviceptr(buffer);
        if (!pointer) {
            error = "nvOFGPUBufferGetCUdeviceptr returned null";
            return false;
        }
        if (!of_ok(api.nvOFGPUBufferGetStrideInfo(buffer, &stride),
                   "nvOFGPUBufferGetStrideInfo", error)) {
            return false;
        }
        return stride.numPlanes > 0 && stride.strideInfo[0].strideXInBytes > 0;
    }
};

NvofFlow::NvofFlow() : impl_(std::make_unique<Impl>()) {}
NvofFlow::~NvofFlow() { shutdown(); }

bool NvofFlow::initialize(int width, int height, const std::string& perf,
                          int grid,
                          std::string& error) {
    shutdown();
    impl_ = std::make_unique<Impl>();
    error.clear();
    if (width < 32 || height < 16) {
        error = "NVOFA dimensions are below 32x16";
        return false;
    }
    if (grid != 1 && grid != 2 && grid != 4) {
        error = "NVOFA grid must be 1, 2 or 4";
        return false;
    }
#ifndef _WIN32
    error = "NVOFA CUDA wrapper is Windows-only";
    return false;
#else
    Impl& state = *impl_;
    state.module = LoadLibraryW(L"nvofapi64.dll");
    if (!state.module) {
        error = "nvofapi64.dll is not installed";
        return false;
    }
    auto create_api = reinterpret_cast<CreateApiFn>(
        GetProcAddress(state.module, "NvOFAPICreateInstanceCuda"));
    if (!create_api) {
        error = "NvOFAPICreateInstanceCuda is missing from nvofapi64.dll";
        return false;
    }
    CUresult cu = cuInit(0);
    if (cu != CUDA_SUCCESS) {
        error = cuda_error(cu, "cuInit");
        return false;
    }
    CUdevice device = 0;
    cu = cuDeviceGet(&device, 0);
    if (cu != CUDA_SUCCESS) {
        error = cuda_error(cu, "cuDeviceGet");
        return false;
    }
#if CUDA_VERSION >= 13000
    cu = cuCtxCreate(&state.context, nullptr, 0, device);
#else
    cu = cuCtxCreate(&state.context, 0, device);
#endif
    if (cu != CUDA_SUCCESS) {
        error = cuda_error(cu, "cuCtxCreate");
        return false;
    }
    NV_OF_STATUS status = create_api(NV_OF_API_VERSION, &state.api);
    if (status != NV_OF_SUCCESS) {
        error = std::string("NvOFAPICreateInstanceCuda: ") +
                of_status_name(status);
        return false;
    }
    if (!state.api.nvCreateOpticalFlowCuda || !state.api.nvOFInit ||
        !state.api.nvOFExecute) {
        error = "NVOFA CUDA function table is incomplete";
        return false;
    }
    if (!state.of_ok(state.api.nvCreateOpticalFlowCuda(
            state.context, &state.handle),
            "nvCreateOpticalFlowCuda", error)) {
        return false;
    }
    NV_OF_INIT_PARAMS init = {};
    init.width = static_cast<uint32_t>(width);
    init.height = static_cast<uint32_t>(height);
    init.outGridSize = grid == 1 ? NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 :
                       grid == 2 ? NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 :
                                   NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = perf == "fast" ? NV_OF_PERF_LEVEL_FAST :
                     perf == "slow" ? NV_OF_PERF_LEVEL_SLOW :
                                      NV_OF_PERF_LEVEL_MEDIUM;
    init.enableOutputCost = NV_OF_TRUE;
    init.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    if (!state.of_ok(state.api.nvOFInit(state.handle, &init),
                     "nvOFInit", error)) {
        return false;
    }
    state.width = width;
    state.height = height;
    state.grid = grid;
    state.output_width = (width + grid - 1) / grid;
    state.output_height = (height + grid - 1) / grid;
    state.perf = perf;
    const uint32_t w = static_cast<uint32_t>(width);
    const uint32_t h = static_cast<uint32_t>(height);
    if (!state.create_buffer(w, h, NV_OF_BUFFER_USAGE_INPUT,
            NV_OF_BUFFER_FORMAT_GRAYSCALE8, state.input, state.input_ptr,
            state.input_stride, error) ||
        !state.create_buffer(w, h, NV_OF_BUFFER_USAGE_INPUT,
            NV_OF_BUFFER_FORMAT_GRAYSCALE8, state.reference,
            state.reference_ptr, state.reference_stride, error) ||
        !state.create_buffer(
            static_cast<uint32_t>(state.output_width),
            static_cast<uint32_t>(state.output_height),
            NV_OF_BUFFER_USAGE_OUTPUT,
            NV_OF_BUFFER_FORMAT_SHORT2, state.output, state.output_ptr,
            state.output_stride, error) ||
        !state.create_buffer(
            static_cast<uint32_t>(state.output_width),
            static_cast<uint32_t>(state.output_height),
            NV_OF_BUFFER_USAGE_COST,
            NV_OF_BUFFER_FORMAT_UINT8, state.cost, state.cost_ptr,
            state.cost_stride, error)) {
        return false;
    }
    return true;
#endif
}

bool NvofFlow::estimate(const uint8_t* input, int input_stride,
                        const uint8_t* reference, int reference_stride,
                        std::vector<float>& flow_x,
                        std::vector<float>& flow_y,
                        std::vector<float>& reliability,
                        std::string& error) {
    error.clear();
    Impl& state = *impl_;
    if (!state.handle || !input || !reference || input_stride < state.width ||
        reference_stride < state.width) {
        error = "NVOFA estimate called with an invalid session/frame";
        return false;
    }
    CUresult cu = cuCtxSetCurrent(state.context);
    if (cu != CUDA_SUCCESS) {
        error = cuda_error(cu, "cuCtxSetCurrent");
        return false;
    }
    auto upload = [&](CUdeviceptr destination, uint32_t pitch,
                      const uint8_t* source, int source_pitch,
                      const char* name) {
        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = source;
        copy.srcPitch = static_cast<size_t>(source_pitch);
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = destination;
        copy.dstPitch = pitch;
        copy.WidthInBytes = static_cast<size_t>(state.width);
        copy.Height = static_cast<size_t>(state.height);
        const CUresult result = cuMemcpy2D(&copy);
        if (result != CUDA_SUCCESS) error = cuda_error(result, name);
        return result == CUDA_SUCCESS;
    };
    if (!upload(state.input_ptr,
                state.input_stride.strideInfo[0].strideXInBytes,
                input, input_stride, "cuMemcpy2D(input)") ||
        !upload(state.reference_ptr,
                state.reference_stride.strideInfo[0].strideXInBytes,
                reference, reference_stride, "cuMemcpy2D(reference)")) {
        return false;
    }
    NV_OF_EXECUTE_INPUT_PARAMS execute_in = {};
    execute_in.inputFrame = state.input;
    execute_in.referenceFrame = state.reference;
    execute_in.disableTemporalHints = NV_OF_TRUE;
    NV_OF_EXECUTE_OUTPUT_PARAMS execute_out = {};
    execute_out.outputBuffer = state.output;
    execute_out.outputCostBuffer = state.cost;
    if (!state.of_ok(state.api.nvOFExecute(
            state.handle, &execute_in, &execute_out),
            "nvOFExecute", error)) {
        return false;
    }
    cu = cuCtxSynchronize();
    if (cu != CUDA_SUCCESS) {
        error = cuda_error(cu, "cuCtxSynchronize");
        return false;
    }
    const size_t count = static_cast<size_t>(state.output_width) *
                         state.output_height;
    std::vector<NV_OF_FLOW_VECTOR> packed(count);
    std::vector<uint8_t> cost(count);
    auto download = [&](void* destination, size_t destination_pitch,
                        CUdeviceptr source, uint32_t source_pitch,
                        size_t row_bytes, size_t rows, const char* name) {
        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = source;
        copy.srcPitch = source_pitch;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy.dstHost = destination;
        copy.dstPitch = destination_pitch;
        copy.WidthInBytes = row_bytes;
        copy.Height = rows;
        const CUresult result = cuMemcpy2D(&copy);
        if (result != CUDA_SUCCESS) error = cuda_error(result, name);
        return result == CUDA_SUCCESS;
    };
    if (!download(packed.data(),
                  static_cast<size_t>(state.output_width) *
                      sizeof(NV_OF_FLOW_VECTOR),
                  state.output_ptr,
                  state.output_stride.strideInfo[0].strideXInBytes,
                  static_cast<size_t>(state.output_width) *
                      sizeof(NV_OF_FLOW_VECTOR),
                  static_cast<size_t>(state.output_height),
                  "cuMemcpy2D(flow)") ||
        !download(cost.data(), static_cast<size_t>(state.output_width),
                  state.cost_ptr,
                  state.cost_stride.strideInfo[0].strideXInBytes,
                  static_cast<size_t>(state.output_width),
                  static_cast<size_t>(state.output_height),
                  "cuMemcpy2D(cost)")) {
        return false;
    }
    flow_x.resize(count);
    flow_y.resize(count);
    reliability.resize(count);
    for (size_t index = 0; index < count; ++index) {
        flow_x[index] = packed[index].flowx / 32.0f;
        flow_y[index] = packed[index].flowy / 32.0f;
        const float normalized_cost = cost[index] / 255.0f;
        // Cost is monotonic but not a calibrated probability.  The square
        // emphasizes high-quality matches and conservatively rejects the tail.
        reliability[index] = (1.0f - normalized_cost) *
                             (1.0f - normalized_cost);
    }
    return true;
}

void NvofFlow::shutdown() {
    if (!impl_) return;
    Impl& state = *impl_;
    if (state.context) cuCtxSetCurrent(state.context);
    auto destroy_buffer = [&](NvOFGPUBufferHandle& buffer) {
        if (buffer && state.api.nvOFDestroyGPUBufferCuda)
            state.api.nvOFDestroyGPUBufferCuda(buffer);
        buffer = nullptr;
    };
    destroy_buffer(state.cost);
    destroy_buffer(state.output);
    destroy_buffer(state.reference);
    destroy_buffer(state.input);
    if (state.handle && state.api.nvOFDestroy)
        state.api.nvOFDestroy(state.handle);
    state.handle = nullptr;
    if (state.context) cuCtxDestroy(state.context);
    state.context = nullptr;
#ifdef _WIN32
    if (state.module) FreeLibrary(state.module);
    state.module = nullptr;
#endif
}

int NvofFlow::width() const { return impl_ ? impl_->width : 0; }
int NvofFlow::height() const { return impl_ ? impl_->height : 0; }
int NvofFlow::output_width() const {
    return impl_ ? impl_->output_width : 0;
}
int NvofFlow::output_height() const {
    return impl_ ? impl_->output_height : 0;
}
int NvofFlow::grid() const { return impl_ ? impl_->grid : 0; }
std::string NvofFlow::backend() const {
    if (!impl_ || !impl_->handle) return "off";
    std::ostringstream out;
    out << "NVIDIA Optical Flow Accelerator/CUDA grid=" << impl_->grid
        << " perf=" << impl_->perf
        << " input_stride="
        << impl_->input_stride.strideInfo[0].strideXInBytes << "x"
        << impl_->input_stride.strideInfo[0].strideYInBytes
        << " output_stride="
        << impl_->output_stride.strideInfo[0].strideXInBytes << "x"
        << impl_->output_stride.strideInfo[0].strideYInBytes;
    return out.str();
}

}  // namespace sylc
