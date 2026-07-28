#pragma once

#include <vector>
#include <deque>
#include <cstdint>
#include <memory>
#include <string>
#include <atomic>

// Forward declare edge264 types to avoid header dependency
struct Edge264Decoder;
struct Edge264Frame;

namespace mvc_demux {

// Decoded MVC frame with both views
struct DecodedMVCFrame {
    // Base view (left eye)
    struct View {
        // Tight, owned 4:2:0 planes. edge264's DPB storage is borrowed only
        // while it is copied here, then immediately returned to the decoder.
        std::vector<uint8_t> y_plane;
        std::vector<uint8_t> cb_plane;
        std::vector<uint8_t> cr_plane;
        int width = 0;
        int height = 0;
        int chroma_width = 0;
        int chroma_height = 0;
        int stride_y = 0;
        int stride_c = 0;
    };

    View base_view;
    View dependent_view;  // MVC second view (right eye)

    bool has_mvc = false;  // True if dependent view is available
    int32_t frame_id = 0;
    int32_t frame_id_mvc = 0;
    int32_t picture_order_cnt = 0;
    int32_t picture_order_cnt_mvc = 0;

    // Frame dimensions (after cropping)
    int display_width = 0;
    int display_height = 0;
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

    // Decode an Annex B access unit (multiple NALs). Compressed slices are
    // retained in owned storage while asynchronous edge264 workers use them.
    int decodeAnnexBStream(const uint8_t* data, size_t size);

    // Feed the two access units belonging to one MVC timestamp.
    int decodeAccessUnitPair(const uint8_t* base_data, size_t base_size,
                             const uint8_t* dependent_data, size_t dependent_size);

    // Get next decoded frame if available
    // Returns true if frame was retrieved, false if no frame ready
    bool getFrame(DecodedMVCFrame& out_frame);

    // Check if decoder is initialized
    bool isInitialized() const { return decoder_ != nullptr; }

    // Flush decoder (for seeking)
    void flush();

    // Bump delayed pictures at end of stream without resetting the decoder.
    void bumpFrames();

    // Deterministically release the native decoder.
    void close();

    // Cooperatively stop work at the next safe NAL/task boundary. This never
    // frees decoder memory from a foreign thread.
    void requestAbort();
    void clearAbort();

    // Probe/load edge264.dll and report a diagnostic suitable for Python/UI logs.
    static bool runtimeAvailable(std::string* diagnostic = nullptr);

    // Get last error message
    const char* getLastError() const { return last_error_.c_str(); }

private:
    Edge264Decoder* decoder_;
    int worker_threads_ = 0;
    std::atomic<bool> abort_requested_{false};
    std::string last_error_;
    std::deque<DecodedMVCFrame> ready_frames_;

    // Copy an edge264 borrowed frame into stable owned storage.
    bool convertFrame(const Edge264Frame& src, DecodedMVCFrame& dst);
    bool fetchNativeFrame(DecodedMVCFrame& out_frame);
    void cacheAvailableFrames();
};

} // namespace mvc_demux
