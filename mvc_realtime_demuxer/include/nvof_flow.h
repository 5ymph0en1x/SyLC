#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sylc {

// Small, optional CUDA-fronted NVIDIA Optical Flow Accelerator wrapper.
// It owns a private CUDA context and driver buffers; the shipping D3D11 path
// remains independent and falls back to synth3d_flow when this optional build
// feature or the hardware runtime is unavailable.
class NvofFlow {
public:
    NvofFlow();
    ~NvofFlow();
    NvofFlow(const NvofFlow&) = delete;
    NvofFlow& operator=(const NvofFlow&) = delete;

    bool initialize(int width, int height, const std::string& perf, int grid,
                    std::string& error);
    bool estimate(const uint8_t* input, int input_stride,
                  const uint8_t* reference, int reference_stride,
                  std::vector<float>& flow_x,
                  std::vector<float>& flow_y,
                  std::vector<float>& reliability,
                  std::string& error);
    void shutdown();

    int width() const;
    int height() const;
    int output_width() const;
    int output_height() const;
    int grid() const;
    std::string backend() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sylc
