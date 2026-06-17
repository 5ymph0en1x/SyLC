#pragma once

#include "mvc_demuxer.h"
#include "m2ts_reader.h"
#include "ssif_parser.h"
#include <memory>
#include <map>
#include <deque>
#include <utility>
#include <vector>
#include <atomic>

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
     * Seek to timestamp (milliseconds). In streaming mode this does a proportional
     * byte seek into the interleaved SSIF; the decoder then re-finds the next IDR.
     */
    bool seek(int64_t timestampMs);

    /**
     * Provide the media duration (ms) so seek() can map timestamp -> byte offset.
     */
    void setExternalDurationMs(int64_t ms) { externalDurationMs_ = ms; }

    /**
     * Provide a base-view EP_map seek table (parsed from the .clpi in Python):
     * parallel arrays of presentation timestamps (ms) and byte offsets into the
     * base .m2ts. Enables frame-accurate dual-file seeking. Sorted by ts ascending.
     */
    void setBaseSeekTable(const std::vector<int64_t>& ptsMs, const std::vector<uint64_t>& bytes);

    /**
     * Provide a frame-accurate SSIF seek table (parsed from the .clpi Extent Start Point
     * tables in Python): parallel arrays of presentation timestamps (ms) and EXACT byte
     * offsets of the interleaved-unit boundary in the .ssif that contains each base IDR.
     * When present, streaming-mode seek() jumps straight to the unit boundary (clean RAPI,
     * both views present) with no PTS binary-search or size-ratio estimate. Sorted by ts.
     */
    void setSsifSeekTable(const std::vector<int64_t>& ptsMs, const std::vector<uint64_t>& ssifBytes);

    /**
     * Cooperatively abort an in-flight read/scan. readNextFramePairStreaming() (and the
     * fallback PTS probe) poll this and return early, so a single cold/contended disc read
     * can never pin the decoder thread past the GUI watchdog into a force-terminate. Set from
     * Python before stopping the thread or when a newer seek supersedes the current scan;
     * cleared at the start of each seek/scan. Thread-safe (atomic).
     */
    void requestAbort() { abortRequested_.store(true, std::memory_order_relaxed); }
    void clearAbort()   { abortRequested_.store(false, std::memory_order_relaxed); }

    // ---- PGS subtitle streaming (dual-file/M2TS) ----
    // PIDs of PGS (Presentation Graphics) subtitle streams found in the base PMT.
    std::vector<uint16_t> getSubtitlePids() const;
    // Select a subtitle PID to stream (0 disables).
    void setSubtitlePid(int pid);
    // True if a reassembled subtitle block is queued.
    bool hasSubtitleData() const { return !subtitleQueue_.empty(); }
    // Pop one subtitle block (PTS in ms + raw PGS bytes). Returns false if none.
    bool readSubtitleBlock(int64_t& timestampMs, std::vector<uint8_t>& data);

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

    // DUAL-FILE MODE (base from 00001.m2ts, dependent from 00002.m2ts).
    // Robust seeking: each file is one contiguous view, so no SSIF interleave gaps.
    bool readNextFramePairDualFile(FramePair& framePair);
    // Shared base/dependent buffer matcher (decode-order base front, dep by PTS).
    bool tryMatchFramePair(FramePair& framePair, bool allowDropBase);
    // Push a freshly-assembled pending base/dependent frame into its buffer.
    void pushPendingBaseFrame();
    void pushPendingDependentFrame();
    // Collect a PGS subtitle packet (selected PID) into the subtitle PES reassembler.
    void collectSubtitlePacket(const M2TSReader::TSPacket& packet);
    // Binary-search a contiguous M2TS for the byte offset whose stream (pid) PTS is
    // at/just-before targetPts90k. Robust+accurate seek without needing an EP_map.
    uint64_t findByteForPts(M2TSReader* reader, uint16_t pid, int64_t targetPts90k, uint64_t hintByte = 0);
    // Proportional byte offset into the dependent file for a NORMALIZED time (ms). Used
    // instead of PTS binary-search for the dependent because its standalone .m2ts carries
    // unreliable (constant/garbage) PTS in places; byte∝time is monotonic by construction.
    uint64_t depProportionalByte(int64_t normMs);

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
    // MVC dependent frames pair with the base by nearest PTS. They are nominally the same
    // temporal instant, but in this interleave the dependent view's PTS can sit up to ~1 frame
    // off the base's (a ~40-50ms residual seen after a mid-stream seek + re-align). Use a ~1.5
    // frame window (24fps -> 6000 ticks ~= 67ms); the nearest-match search keeps it picking the
    // correct (closest) dependent frame, so this just admits the small offset, never mispairs.
    static constexpr int64_t PTS_MATCH_TOLERANCE = 6000;

    // PTS normalization: Blu-ray streams often start at non-zero PTS (e.g., ~11s)
    // We capture the first valid PTS and subtract it from all subsequent frames
    // to normalize timestamps to start from 0
    int64_t basePtsOffset_;
    bool basePtsInitialized_;

    int64_t externalDurationMs_ = 0;  // media duration for timestamp->byte seek mapping

    // DUAL-FILE MODE state
    bool dualFileMode_ = false;
    // Base-view EP_map seek table (pts_ms, byte offset into base .m2ts), sorted by ts.
    std::vector<std::pair<int64_t, uint64_t>> baseSeekTable_;
    // BD3D exact SSIF seek table (pts_ms, byte offset of the interleaved-unit boundary in the
    // .ssif). Built from the CLPI Extent Start Point tables; preferred over baseSeekTable_ in
    // streaming mode for byte-exact, both-views-aligned landings. Sorted by ts.
    std::vector<std::pair<int64_t, uint64_t>> ssifSeekTable_;
    int totalFramePairs_ = 0;  // emitted pair counter (was a static local)
    int depReseekBudget_ = 0;  // post-seek: allowed dependent re-seeks to align to base

    // PGS subtitle streaming state
    int selectedSubtitlePid_ = 0;                       // 0 = disabled
    std::map<uint16_t, PESState> subtitlePesStates_;    // PID -> reassembly
    struct SubtitleBlock { int64_t timestampMs; std::vector<uint8_t> data; };
    std::deque<SubtitleBlock> subtitleQueue_;           // ready PGS blocks

    bool isOpen_;

    // Cooperative-abort flag for readNextFramePairStreaming()/findByteForPts() (see requestAbort).
    std::atomic<bool> abortRequested_{false};
};

} // namespace mvc_demux
