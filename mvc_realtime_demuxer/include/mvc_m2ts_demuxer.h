#pragma once

#include "m2ts_reader.h"
#include "h264_nal_parser.h"
#include <atomic>
#include <deque>
#include <memory>
#include <map>

namespace mvc_demux {

/**
 * MVC-aware M2TS Demuxer
 * Extracts H.264 base view and MVC dependent view from M2TS/TS files
 *
 * Handles PES packet reassembly and NAL unit separation for MVC content.
 */
class MVCM2TSDemuxer {
public:
    MVCM2TSDemuxer();
    ~MVCM2TSDemuxer();

    // Open M2TS or TS file
    bool open(const std::string& filePath);

    // Close file
    void close();

    // Check if open
    bool isOpen() const;

    // Video information
    struct VideoInfo {
        uint32_t width;
        uint32_t height;
        double fps;
        bool hasMVC;
        uint16_t baseVideoPid;
        uint16_t mvcVideoPid;
    };

    VideoInfo getVideoInfo() const;

    // Get codec private data (SPS/PPS extracted from stream)
    std::vector<uint8_t> getCodecPrivate() const;

    // Frame pair (base + dependent)
    struct FramePair {
        std::vector<uint8_t> baseData;
        std::vector<uint8_t> dependentData;
        int64_t timestamp;  // PTS in milliseconds
        bool isKeyframe;
    };

    // Read next frame pair
    // Returns false when EOF reached
    bool readNextFramePair(FramePair& framePair);

    // Seek to timestamp (milliseconds)
    bool seek(int64_t timestampMs);

    void setExternalDurationMs(int64_t durationMs) {
        if (durationMs > 0) externalDurationMs_ = durationMs;
    }
    int64_t getLastCueTimestamp() const { return lastSeekTimestampMs_; }

    // Cooperatively interrupt an in-flight optical-disc read/probe. This keeps
    // GUI fallback and shutdown bounded even when the source is slow or damaged.
    void requestAbort() { abortRequested_.store(true, std::memory_order_relaxed); }
    void clearAbort() { abortRequested_.store(false, std::memory_order_relaxed); }

private:
    std::unique_ptr<M2TSReader> reader_;
    H264NALParser nalParser_;

    VideoInfo videoInfo_;
    bool hasVideoInfo_;
    uint64_t dualPidStartPos_;  // Position where both H.264 and MVC PIDs are synchronized (for SSIF)
    std::atomic<bool> abortRequested_{false};
    int64_t externalDurationMs_ = 0;
    int64_t streamStartPts90k_ = -1;
    int64_t lastSeekTimestampMs_ = 0;

    // PES reassembly state
    struct PESState {
        std::vector<uint8_t> buffer;
        int64_t pts;  // Presentation Time Stamp (90kHz clock)
        int64_t dts;  // Decode Time Stamp
        bool hasStarted;
    };

    std::map<uint16_t, PESState> pesStates_;  // PID -> PES state

    // Codec private data (SPS/PPS)
    std::vector<uint8_t> codecPrivate_;
    bool hasCodecPrivate_;

    // Frame assembly state
    struct FrameState {
        std::vector<uint8_t> baseView;
        std::vector<uint8_t> dependentView;
        int64_t pts;
        bool isKeyframe;
        bool isComplete;
        // For dual-PID SSIF: track which PIDs have contributed data
        bool hasBasePidData;  // Have we received data from base PID (0x1011)?
        bool hasMvcPidData;   // Have we received data from MVC PID (0x1012)?
    };

    FrameState currentFrame_;

    // Frames consumed by the startup IDR probe must be replayed before normal
    // reading resumes. Keep this separate from pendingFrame_: finding the end
    // of an IDR necessarily pre-reads the first PES of the following access
    // unit, which already occupies pendingFrame_.
    std::deque<FramePair> replayFrames_;

    // Pending NAL data for next frame (when AUD detected mid-processing)
    struct PendingData {
        std::vector<uint8_t> baseView;
        std::vector<uint8_t> dependentView;
        int64_t pts;
        bool hasData;
        bool alreadyPrefixed;  // Track if codecPrivate was already prepended
        // For dual-PID SSIF: track which PIDs contributed to this pending frame
        bool hasBasePidData;
        bool hasMvcPidData;
    };
    PendingData pendingFrame_;

    // SSIF bulk-interleaved format support: Buffer for MVC-only frames
    // In SSIF files, MVC data arrives first (bulk), then H.264 data (bulk).
    // We buffer MVC frames by PTS, then match with H.264 frames when they arrive.
    std::map<int64_t, std::vector<uint8_t>> mvcBuffer_;  // PTS -> MVC data
    std::map<int64_t, std::vector<uint8_t>> h264Buffer_; // PTS -> H.264 data (new: bidirectional support)

    // Process TS packet for video PID
    void processVideoPacket(const M2TSReader::TSPacket& packet);

    // Parse PES packet header
    bool parsePESHeader(const std::vector<uint8_t>& pesData,
                       int64_t& pts, int64_t& dts,
                       size_t& headerLength);

    // Extract NAL units and separate by type
    void separateNALUnits(const std::vector<uint8_t>& nalData,
                         std::vector<uint8_t>& baseOut,
                         std::vector<uint8_t>& dependentOut);

    // Extract SPS/PPS for codec private
    void extractCodecPrivate(const std::vector<uint8_t>& nalData);

    // Parse SPS to extract dimensions
    void parseSPSForDimensions();

    // Check if frame is complete
    bool isFrameComplete() const;

    // Convert PTS (90kHz) to milliseconds
    int64_t ptsToMs(int64_t pts) const {
        return pts / 90;  // 90kHz to milliseconds
    }
};

} // namespace mvc_demux
