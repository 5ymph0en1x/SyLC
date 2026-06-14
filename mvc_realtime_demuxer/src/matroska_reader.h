#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace mvc_demux {

// Forward declarations for libmatroska types
// These will be defined in the implementation
class MatroskaFile;

// Track information
struct MatroskaTrack {
    uint32_t trackNumber;
    uint32_t trackUID;
    uint8_t trackType;  // 1=video, 2=audio, 17=subtitle
    std::string codecId;
    std::vector<uint8_t> codecPrivate;

    // Video-specific
    uint32_t pixelWidth;
    uint32_t pixelHeight;
    double frameRate;

    // MVC-specific
    bool isMVC;
    uint32_t mvcSubTrack;  // 0=none, 1=dependent, 2=base
};

// Block/Frame data
struct MatroskaBlock {
    uint32_t trackNumber;
    int64_t timestamp;  // In nanoseconds
    std::vector<uint8_t> data;
    bool isKeyframe;
    uint32_t frameCount;  // For laced frames
};

// Matroska file reader using libmatroska
class MatroskaReader {
public:
    MatroskaReader();
    ~MatroskaReader();

    // Open an MKV file
    bool open(const std::string& filePath);

    // Close the file
    void close();

    // Check if file is open
    bool isOpen() const;

    // Get all tracks
    std::vector<MatroskaTrack> getTracks() const;

    // Get a specific track by number
    MatroskaTrack getTrack(uint32_t trackNumber) const;

    // Read next block/frame
    // Returns false when EOF
    bool readNextBlock(MatroskaBlock& block);

    // Seek to timestamp (in milliseconds)
    // Optional trackNumber to filter blocks during seek
    bool seek(int64_t timestampMs, int32_t trackNumber = -1);

    // Get duration (in milliseconds)
    int64_t getDuration() const;

    // Get timecode scale (nanoseconds per unit)
    uint64_t getTimecodeScale() const;

    // Set external duration hint (e.g. from ffprobe)
    void setExternalDurationMs(int64_t durationMs);

    // Rewind helper for failed seeks (backoff strategy)
    bool rewind_after_failed_seek(int64_t timestampMs, uint32_t ms_backoff);

    // V8 INDEX-BASED SYNC: Get the authoritative Cue timestamp from last seek
    // This is the single source of truth for synchronization (T_cues = T_audio = T_video)
    // Returns -1 if no seek was performed or Cues unavailable
    int64_t getLastCueTimestamp() const;

    // V8 SEEK OPTIMIZATION: Get all Cue (keyframe) timestamps for efficient keyframe navigation
    // Returns a sorted vector of keyframe timestamps in milliseconds
    std::vector<int64_t> getCuesTimestamps() const;

    // V8 SEEK OPTIMIZATION: Seek directly to a specific Cue timestamp
    // This is more efficient than seek() because it skips the "nearest before" search
    // Returns false if the cueTimestampMs is not found in the index
    bool seekToCue(int64_t cueTimestampMs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mvc_demux
