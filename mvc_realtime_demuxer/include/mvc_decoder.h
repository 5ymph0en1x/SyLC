#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <string>

// Forward declare edge264 types to avoid header dependency
struct Edge264Decoder;
struct Edge264Frame;

namespace mvc_demux {

// Decoded MVC frame with both views
struct DecodedMVCFrame {
    // Base view (left eye)
    struct View {
        const uint8_t* y_plane;
        const uint8_t* cb_plane;
        const uint8_t* cr_plane;
        int width;
        int height;
        int stride_y;
        int stride_c;
    };

    View base_view;
    View dependent_view;  // MVC second view (right eye)

    bool has_mvc;  // True if dependent view is available
    int32_t frame_id;
    int32_t frame_id_mvc;

    // Frame dimensions (after cropping)
    int display_width;
    int display_height;
};

// MVC Decoder using edge264
// Decodes both base and dependent views with inter-view prediction support
class MVCDecoder {
public:
    MVCDecoder();
    ~MVCDecoder();

    // Initialize decoder
    // n_threads: -1 for auto, 0 for single-threaded, >0 for specific count
    bool init(int n_threads = -1);

    // Decode a NAL unit
    // Feed both base and dependent NAL units to the same decoder
    // Returns 0 on success, error code otherwise
    int decodeNAL(const uint8_t* nal_data, size_t nal_size);

    // Decode an Annex B access unit (multiple NALs, zero-copy).
    int decodeAnnexBStream(const uint8_t* data, size_t size);

    // Get next decoded frame if available
    // Returns true if frame was retrieved, false if no frame ready
    bool getFrame(DecodedMVCFrame& out_frame);

    // Check if decoder is initialized
    bool isInitialized() const { return decoder_ != nullptr; }

    // Flush decoder (for seeking)
    void flush();

    // Get last error message
    const char* getLastError() const { return last_error_.c_str(); }

private:
    Edge264Decoder* decoder_;
    std::string last_error_;

    // Convert edge264 frame to our frame structure
    void convertFrame(const Edge264Frame& src, DecodedMVCFrame& dst);
};

} // namespace mvc_demux
