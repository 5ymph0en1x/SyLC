#pragma once

#include "matroska_reader.h"
#include "h264_nal_parser.h"
#include <memory>
#include <queue>

namespace mvc_demux {

// MVC-aware Matroska Demuxer
// This class uses MatroskaReader to parse MKV files and extract MVC streams
class MVCMatroskaDemuxer {
public:
    MVCMatroskaDemuxer();
    ~MVCMatroskaDemuxer();

    // Open an MKV file
    bool open(const std::string& filePath);

    // Close the file
    void close();

    // Check if open
    bool isOpen() const;

    // Video information
    struct VideoInfo {
        uint32_t width;
        uint32_t height;
        double fps;
        bool hasMVC;
        uint32_t baseTrackNumber;
        uint32_t mvcTrackNumber;
    };

    VideoInfo getVideoInfo() const;

    // ========== SUBTITLE STREAMING SUPPORT ==========

    // Subtitle track information
    struct SubtitleTrackInfo {
        uint32_t trackNumber;
        std::string codecId;      // e.g., "S_HDMV/PGS", "S_TEXT/UTF8", "S_TEXT/ASS"
        std::string language;     // ISO 639-2 language code
        std::string name;         // Track name/title
        bool isPGS;               // True if this is a PGS bitmap subtitle
    };

    // Subtitle block data
    struct SubtitleBlock {
        uint32_t trackNumber;
        int64_t timestampMs;      // Timestamp in milliseconds
        std::vector<uint8_t> data;
    };

    // Get all subtitle tracks in the file
    std::vector<SubtitleTrackInfo> getSubtitleTracks() const;

    // Enable streaming for a specific subtitle track (0 = disable)
    void setActiveSubtitleTrack(uint32_t trackNumber);

    // Get currently active subtitle track (0 = none)
    uint32_t getActiveSubtitleTrack() const;

    // Check if subtitle data is available in the queue
    bool hasSubtitleData() const;

    // Read next subtitle block (non-blocking, returns false if queue empty)
    bool readNextSubtitleBlock(SubtitleBlock& block);

    // ================================================

    // Get codec private data (contains SPS/PPS in AVCC format)
    std::vector<uint8_t> getCodecPrivate() const;

    // Provide external duration hint (milliseconds) when Duration is missing in the container.
    void set_external_duration_ms(int64_t durationMs);

    // Rewind slightly after a failed seek (no IDR) and retry.
    bool rewind_after_failed_seek_ms(int64_t timestampMs, uint32_t backoffMs = 5000);

    // Frame pair (base + dependent)
    struct FramePair {
        std::vector<uint8_t> baseData;
        std::vector<uint8_t> dependentData;
        int64_t timestamp;
        bool isKeyframe;
    };

    // Read next frame pair
    bool readNextFramePair(FramePair& framePair);

    // Seek to timestamp (milliseconds)
    bool seek(int64_t timestampMs);

    // V8 INDEX-BASED SYNC: Get authoritative Cue timestamp from last seek
    // Returns -1 if no seek was performed or Cues unavailable
    int64_t getLastCueTimestamp() const;

    // V8 SEEK OPTIMIZATION: Get all Cue (keyframe) timestamps for efficient keyframe navigation
    // Returns a sorted vector of keyframe timestamps in milliseconds
    std::vector<int64_t> getCuesTimestamps() const;

    // V8 SEEK OPTIMIZATION: Seek directly to a specific Cue timestamp
    // This is more efficient than seek() because it skips the "nearest before" search
    bool seekToCue(int64_t cueTimestampMs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Analyze tracks to find MVC configuration
    void analyzeTracks();

    // Extract NAL units from block data and separate by type
    void separateNALUnits(const std::vector<uint8_t>& blockData,
                         std::vector<uint8_t>& baseOut,
                         std::vector<uint8_t>& dependentOut);

    void prepareCodecPrivateAnnexB();
};

} // namespace mvc_demux
