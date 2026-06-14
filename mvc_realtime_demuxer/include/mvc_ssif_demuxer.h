#pragma once

#include "mvc_demuxer.h"
#include "m2ts_reader.h"
#include "ssif_parser.h"
#include <memory>
#include <map>

namespace mvc_demux {

/**
 * MVC SSIF Demuxer
 *
 * Handles Blu-ray 3D files with separate M2TS streams for left and right eyes.
 * Uses SSIF (Stereo Interleaved File) to determine reading order.
 *
 * Format:
 * - BDMV/STREAM/00000.m2ts (left eye, base view)
 * - BDMV/STREAM/00001.m2ts (right eye, dependent view)
 * - BDMV/STREAM/SSIF/00000.ssif (interleaving instructions)
 */
class MVCSSIFDemuxer {
public:
    // Video information structure
    struct VideoInfo {
        uint32_t width;
        uint32_t height;
        double fps;
        bool hasMVC;
        uint16_t baseVideoPid;
        uint16_t mvcVideoPid;
    };

    // Frame pair structure
    struct FramePair {
        std::vector<uint8_t> baseData;
        std::vector<uint8_t> dependentData;
        uint64_t timestamp;
        bool isKeyframe;
    };

    MVCSSIFDemuxer();
    ~MVCSSIFDemuxer();

    /**
     * Open an SSIF-based 3D stream
     * @param ssifPath Path to the .ssif file or the base .m2ts file
     * @return true if successful
     */
    bool open(const std::string& ssifPath);

    /**
     * Close the demuxer
     */
    void close();

    /**
     * Read next stereo frame pair
     * @param framePair Output frame pair
     * @return true if successful
     */
    bool readNextFramePair(FramePair& framePair);

    /**
     * Get video information
     */
    VideoInfo getVideoInfo() const { return videoInfo_; }

    /**
     * Get codec private data (SPS/PPS)
     */
    const std::vector<uint8_t>& getCodecPrivate() const { return codecPrivate_; }

    /**
     * Check if codec private data is available
     */
    bool hasCodecPrivate() const { return hasCodecPrivate_; }

    /**
     * Seek to timestamp (not yet implemented)
     */
    bool seek(int64_t timestampMs);

private:
    struct PESState {
        std::vector<uint8_t> buffer;
        bool hasStarted;
        int64_t pts;
        int64_t dts;
    };

    struct PendingData {
        std::vector<uint8_t> baseView;
        std::vector<uint8_t> dependentView;
        int64_t pts;
        bool hasData;
        bool alreadyPrefixed;
    };

    // Frame data for buffering multiple frames
    struct BufferedFrame {
        std::vector<uint8_t> data;
        int64_t pts;
        bool isKeyframe;
    };

    // Process video packet from either stream
    void processVideoPacket(const M2TSReader::TSPacket& packet, bool isBase);

    // Parse PES header
    bool parsePESHeader(const std::vector<uint8_t>& pesData,
                       int64_t& pts, int64_t& dts,
                       size_t& headerLength);

    // Extract codec private (SPS/PPS)
    void extractCodecPrivate(const std::vector<uint8_t>& nalData);

    // Read and process extents according to SSIF instructions
    bool processExtents();

    // Read specific extent
    bool readExtent(const SSIFParser::Extent& extent);

    // Synchronize frames from both streams
    bool synchronizeFrames(FramePair& framePair);

    // STREAMING MODE methods
    bool readNextFramePairStreaming(FramePair& framePair);
    void processVideoPacketStreaming(const M2TSReader::TSPacket& packet);
    void separateNALUnits(const std::vector<uint8_t>& nalData, int64_t pts);

    std::unique_ptr<SSIFParser> ssifParser_;
    std::unique_ptr<M2TSReader> baseReader_;
    std::unique_ptr<M2TSReader> dependentReader_;
    std::unique_ptr<M2TSReader> ssifReader_;  // For streaming mode: read directly from SSIF

    VideoInfo videoInfo_;
    std::vector<uint8_t> codecPrivate_;
    bool hasCodecPrivate_;
    bool hasSPS_;  // Track if SPS has been found
    bool hasPPS_;  // Track if PPS has been found
    bool hasVideoInfo_;
    bool isStreamingMode_;  // True if reading directly from large SSIF file

    // PES state for both streams
    std::map<uint16_t, PESState> basePesStates_;
    std::map<uint16_t, PESState> dependentPesStates_;
    std::map<uint16_t, PESState> streamingPesState_;  // For streaming mode

    // Current extent being processed
    size_t currentExtentIndex_;

    // SOL 5A: Track stream positions for synchronized seeking
    int64_t base_stream_pos_;
    int64_t dependent_stream_pos_;

    // Pending frame data (for single-frame mode)
    PendingData pendingBase_;
    PendingData pendingDependent_;
    FramePair currentFrame_;

    // Frame queues (in case frames arrive out of sync)
    std::vector<FramePair> frameQueue_;

    // Buffered frames for PTS matching (SSIF streaming mode)
    // The SSIF format stores dependent frames AHEAD of base frames
    // So we buffer multiple frames and match by PTS
    std::vector<BufferedFrame> baseFrameBuffer_;
    std::vector<BufferedFrame> dependentFrameBuffer_;
    // SSIF can have significant offset (many MVC frames before base frames arrive)
    // Need large buffer to avoid dropping frames before matching
    static constexpr size_t MAX_FRAME_BUFFER_SIZE = 500;  // ~20 seconds at 24fps
    // MVC frames should have EXACT same PTS as base (same temporal instant)
    // Allow small tolerance for encoding differences (~1/10 frame = 375 ticks)
    static constexpr int64_t PTS_MATCH_TOLERANCE = 375;

    // PTS normalization: Blu-ray streams often start at non-zero PTS (e.g., ~11s)
    // We capture the first valid PTS and subtract it from all subsequent frames
    // to normalize timestamps to start from 0
    int64_t basePtsOffset_;
    bool basePtsInitialized_;

    bool isOpen_;
};

} // namespace mvc_demux
