#pragma once

#include "h264_nal_parser.h"
#include "frame_ring_buffer.h"
#include <string>
#include <memory>
#include <queue>
#include <functional>

namespace mvc_demux {

// Frame data for a single view
struct FrameData {
    std::vector<uint8_t> data;
    uint64_t timestamp;  // In milliseconds
    bool isKeyframe;
};

// Callback for delivering separated streams
using FrameCallback = std::function<void(const FrameData& baseView, const FrameData& dependentView)>;

// MVC Real-time Demuxer
// Reads MKV file, extracts H.264 base and MVC dependent views, delivers them synchronously
class MVCDemuxer {
public:
    MVCDemuxer();
    ~MVCDemuxer();

    // Open an MKV file for demuxing
    bool open(const std::string& filePath);

    // Close the file
    void close();

    // Check if file is open
    bool isOpen() const { return isOpen_; }

    // Get video metadata
    struct VideoInfo {
        uint32_t width;
        uint32_t height;
        double fps;
        bool hasMVC;
        uint32_t trackCount;
    };

    VideoInfo getVideoInfo() const { return videoInfo_; }

    // Read next frame pair (base + dependent views)
    // Returns false when end of file reached
    bool readNextFramePair(FrameData& baseView, FrameData& dependentView);

    // Read and push directly into a pre-allocated ring buffer (zero-copy to Python).
    bool readNextFramePairIntoRing(FrameRingBuffer& buffer);

    // Alternative: Set callback for streaming mode
    void setFrameCallback(FrameCallback callback);

    // Process entire file with callback (streaming mode)
    bool processFile();

    // Seek to timestamp (milliseconds)
    bool seek(uint64_t timestampMs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool isOpen_;
    VideoInfo videoInfo_;
    H264NALParser nalParser_;
    FrameCallback frameCallback_;

    // Internal: Process video packet
    void processVideoPacket(const uint8_t* data, size_t size, uint64_t timestamp);

    // Internal: Separate NAL units into base and dependent streams
    void separateStreams(const std::vector<NALUnit>& nalUnits,
                        std::vector<uint8_t>& baseStream,
                        std::vector<uint8_t>& dependentStream);
};

} // namespace mvc_demux
