#include "mvc_matroska_demuxer.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <deque>

namespace mvc_demux {

struct MVCMatroskaDemuxer::Impl {
    MatroskaReader reader;
    H264NALParser nalParser;

    VideoInfo videoInfo;
    std::vector<MatroskaTrack> allTracks;

    // Tracks configuration
    uint32_t baseTrackNum;
    uint32_t mvcTrackNum;
    bool hasSeparateTracks;
    bool codecPrivateInjected;
    std::vector<uint8_t> codecPrivateAnnexB;
    uint8_t nalLengthSize;  // NAL length prefix size (1, 2, or 4 bytes) from AVCC header

    // For reading blocks - Changed to deque for iteration support
    std::deque<MatroskaBlock> blockQueue;
    size_t maxBufferedBlocks;

    // ========== SUBTITLE STREAMING SUPPORT ==========
    std::vector<SubtitleTrackInfo> subtitleTracks;  // All detected subtitle tracks
    uint32_t activeSubtitleTrack;                    // Currently active track (0 = disabled)
    std::deque<SubtitleBlock> subtitleQueue;         // Queue of pending subtitle blocks
    size_t maxSubtitleQueueSize;                     // Max queue size before dropping old blocks
    // ================================================

    Impl()
        : baseTrackNum(0)
        , mvcTrackNum(0)
        , hasSeparateTracks(false)
        , codecPrivateInjected(false)
        , nalLengthSize(4)  // Default to 4-byte NAL length (most common)
        , activeSubtitleTrack(0)
        , maxSubtitleQueueSize(50)
    {
        videoInfo.width = 0;
        videoInfo.height = 0;
        videoInfo.fps = 0.0;
        videoInfo.hasMVC = false;
        videoInfo.baseTrackNumber = 0;
        videoInfo.mvcTrackNumber = 0;
        maxBufferedBlocks = 100; // Increased buffer size for safe interleaving
    }

    bool ensureBufferedBlocks() {
        // For separate tracks, we manage the buffer differently (in readNextFramePair)
        if (hasSeparateTracks) {
            return true;
        }
        if (blockQueue.size() >= 6) { // Maintain small buffer for single track
            return true;
        }
        while (blockQueue.size() < 6) {
            MatroskaBlock rawBlock;
            if (!reader.readNextBlock(rawBlock)) {
                return !blockQueue.empty();
            }

            // ========== SUBTITLE CAPTURE IN SINGLE-TRACK MODE ==========
            // Check for subtitle blocks BEFORE filtering for video
            if (activeSubtitleTrack > 0 && rawBlock.trackNumber == activeSubtitleTrack) {
                SubtitleBlock subBlock;
                subBlock.trackNumber = rawBlock.trackNumber;
                // MatroskaReader already returns timestamps in milliseconds!
                subBlock.timestampMs = rawBlock.timestamp;
                subBlock.data = std::move(rawBlock.data);
                subtitleQueue.push_back(std::move(subBlock));

                // Limit queue size
                while (subtitleQueue.size() > maxSubtitleQueueSize) {
                    subtitleQueue.pop_front();
                }
                continue;  // Continue looking for video blocks
            }
            // ============================================================

            if (rawBlock.trackNumber != baseTrackNum) {
                continue;
            }
            blockQueue.push_back(std::move(rawBlock));
            if (blockQueue.size() >= 6) {
                break;
            }
        }
        return !blockQueue.empty();
    }

    bool popNextVideoBlock(MatroskaBlock& outBlock) {
        if (hasSeparateTracks) {
            // Should not be called for separate tracks
            return reader.readNextBlock(outBlock);
        }
        if (blockQueue.empty()) {
            if (!ensureBufferedBlocks()) {
                return false;
            }
        }
        outBlock = std::move(blockQueue.front());
        blockQueue.pop_front();
        return true;
    }
};

MVCMatroskaDemuxer::MVCMatroskaDemuxer()
    : impl_(std::make_unique<Impl>())
{
}

MVCMatroskaDemuxer::~MVCMatroskaDemuxer() {
    close();
}

bool MVCMatroskaDemuxer::open(const std::string& filePath) {

    if (!impl_->reader.open(filePath)) {
        return false;
    }

    // Get all tracks
    impl_->allTracks = impl_->reader.getTracks();

    if (impl_->allTracks.empty()) {
        return false;
    }

    // Analyze tracks to find MVC configuration
    analyzeTracks();
    prepareCodecPrivateAnnexB();

    return true;
}

void MVCMatroskaDemuxer::analyzeTracks() {
    // Look for video tracks
    std::vector<MatroskaTrack> videoTracks;

    // Clear previous subtitle track info
    impl_->subtitleTracks.clear();

    for (const auto& track : impl_->allTracks) {
        if (track.trackType == 1) {  // Video
            videoTracks.push_back(track);

            // Optimized codec detection using substring position checks
            // Check for MVC (already set by parseTrackEntry, but also check codecId)
            if (track.isMVC ||
                (track.codecId.size() >= 15 && track.codecId.compare(0, 15, "V_MPEG4/ISO/MVC") == 0)) {
                impl_->videoInfo.hasMVC = true;
                impl_->mvcTrackNum = track.trackNumber;
            } else if (track.codecId.size() >= 15 &&
                      (track.codecId.compare(0, 15, "V_MPEG4/ISO/AVC") == 0 ||
                       track.codecId.find("H264") != std::string::npos)) {
                impl_->baseTrackNum = track.trackNumber;
            }
        }
        // ========== SUBTITLE TRACK DETECTION ==========
        else if (track.trackType == 17) {  // Subtitle
            SubtitleTrackInfo subInfo;
            subInfo.trackNumber = track.trackNumber;
            subInfo.codecId = track.codecId;

            // Check if this is a PGS (Blu-ray bitmap) subtitle
            // Codec IDs: "S_HDMV/PGS" for MKV, sometimes just contains "PGS"
            subInfo.isPGS = (track.codecId.find("HDMV/PGS") != std::string::npos) ||
                            (track.codecId.find("PGS") != std::string::npos);

            // Language and name would need to be extracted from track metadata
            // For now, use track number as identifier
            // TODO: Extract language from MatroskaTrack if available
            subInfo.language = "";  // Will be populated if MatroskaTrack has language field
            subInfo.name = "Track " + std::to_string(track.trackNumber);

            fprintf(stderr, "[SUB-DETECT] Found subtitle track: number=%u, codec=%s, isPGS=%d\n",
                    subInfo.trackNumber, subInfo.codecId.c_str(), subInfo.isPGS ? 1 : 0);
            fflush(stderr);

            impl_->subtitleTracks.push_back(subInfo);
        }
        // ================================================
    }

    // Determine configuration
    if (videoTracks.empty()) {
        return;
    }

    // Use first video track for resolution/fps
    const auto& firstVideo = videoTracks[0];
    impl_->videoInfo.width = firstVideo.pixelWidth;
    impl_->videoInfo.height = firstVideo.pixelHeight;
    impl_->videoInfo.fps = firstVideo.frameRate;

    // Check if we have separate MVC and base tracks
    if (impl_->baseTrackNum > 0 && impl_->mvcTrackNum > 0 &&
        impl_->baseTrackNum != impl_->mvcTrackNum) {
        impl_->hasSeparateTracks = true;
        impl_->videoInfo.hasMVC = true;
    } else if (videoTracks.size() == 1) {
        // Single track - check if it contains both streams interleaved
        impl_->baseTrackNum = videoTracks[0].trackNumber;
        impl_->mvcTrackNum = 0;
        impl_->hasSeparateTracks = false;

        // For single-track AVC, optimistically assume it could be interleaved MVC.
        // The NAL separation logic will handle the details.
        if (videoTracks[0].codecId.find("AVC") != std::string::npos) {
            impl_->videoInfo.hasMVC = true;
        }
    }

    impl_->videoInfo.baseTrackNumber = impl_->baseTrackNum;
    impl_->videoInfo.mvcTrackNumber = impl_->mvcTrackNum;

    // Summary log
    fprintf(stderr, "[DEMUX-INIT] Track analysis complete:\n");
    fprintf(stderr, "  - hasSeparateTracks: %s\n", impl_->hasSeparateTracks ? "YES" : "NO");
    fprintf(stderr, "  - baseTrackNum: %u, mvcTrackNum: %u\n", impl_->baseTrackNum, impl_->mvcTrackNum);
    fprintf(stderr, "  - subtitleTracks: %zu detected\n", impl_->subtitleTracks.size());
    for (const auto& st : impl_->subtitleTracks) {
        fprintf(stderr, "    * Track %u: %s (PGS=%d)\n", st.trackNumber, st.codecId.c_str(), st.isPGS ? 1 : 0);
    }
    fflush(stderr);
}

void MVCMatroskaDemuxer::close() {
    impl_->blockQueue.clear();
    impl_->subtitleQueue.clear();  // Clear subtitle queue
    impl_->activeSubtitleTrack = 0;
    impl_->reader.close();
}

bool MVCMatroskaDemuxer::isOpen() const {
    return impl_->reader.isOpen();
}

MVCMatroskaDemuxer::VideoInfo MVCMatroskaDemuxer::getVideoInfo() const {
    return impl_->videoInfo;
}

std::vector<uint8_t> MVCMatroskaDemuxer::getCodecPrivate() const {
    if (!impl_->reader.isOpen() || impl_->baseTrackNum == 0) {
        return std::vector<uint8_t>();
    }

    const auto& tracks = impl_->reader.getTracks();

    // Find the track with matching track number
    for (const auto& track : tracks) {
        if (track.trackNumber == impl_->baseTrackNum && track.trackType == 1) { // 1 = video
            return track.codecPrivate;
        }
    }

    return std::vector<uint8_t>();
}

void MVCMatroskaDemuxer::set_external_duration_ms(int64_t durationMs) {
    impl_->reader.setExternalDurationMs(durationMs);
}

bool MVCMatroskaDemuxer::rewind_after_failed_seek_ms(int64_t timestampMs, uint32_t backoffMs) {
    return impl_->reader.rewind_after_failed_seek(timestampMs, backoffMs);
}

void MVCMatroskaDemuxer::separateNALUnits(const std::vector<uint8_t>& blockData,
                                          std::vector<uint8_t>& baseOut,
                                          std::vector<uint8_t>& dependentOut) {
    baseOut.clear();
    dependentOut.clear();

    if (blockData.empty()) {
        return;
    }

    // OPTIMIZATION: Reserve space based on typical frame size to reduce reallocations
    // Typical 1080p MVC frame: base ~100KB, dependent ~50KB
    baseOut.reserve(blockData.size() * 2 / 3);  // Reserve ~67% for base
    dependentOut.reserve(blockData.size() / 3);  // Reserve ~33% for dependent

    // OPTIMIZATION: Start code as constant array (faster than push_back loop)
    static const uint8_t START_CODE[4] = {0x00, 0x00, 0x00, 0x01};

    // MKV stores H.264 in AVCC format: [N-byte length][NAL][N-byte length][NAL]...
    // where N is nalLengthSize (1, 2, or 4 bytes, from AVCC header)
    size_t pos = 0;
    const size_t blockSize = blockData.size();
    const uint8_t lengthSize = impl_->nalLengthSize;

    while (pos + lengthSize <= blockSize) {
        // Read NAL length using dynamic length size (big-endian)
        uint32_t nalSize = 0;
        for (uint8_t i = 0; i < lengthSize; ++i) {
            nalSize = (nalSize << 8) | blockData[pos + i];
        }
        pos += lengthSize;

        if (pos + nalSize > blockSize || nalSize == 0) {
            break;
        }

        const uint8_t* nalData = blockData.data() + pos;

        if (nalSize > 0) {
            // PRESERVE-ORDER FIX: Do NOT split base/dep. The previous code routed
            // NAL 14/15/20 (MVC-specific) to dependentOut and the player then
            // concatenated `base + dep`, which SCRAMBLED the NAL order from the
            // original AVCC block. edge264 needs to see NALs in their original
            // sequence — re-ordering breaks CABAC alignment for the slice
            // headers (Gravity 3D BD horizontal-band corruption). Emit ALL
            // NALs into baseOut in their original block order; dependentOut
            // stays empty (harmless — the player just concatenates).
            impl_->nalParser.parseUnit(nalData, nalSize);  // keep stats updated

            size_t oldSize = baseOut.size();
            baseOut.resize(oldSize + 4 + nalSize);
            std::memcpy(baseOut.data() + oldSize, START_CODE, 4);
            std::memcpy(baseOut.data() + oldSize + 4, nalData, nalSize);
        }

        pos += nalSize;
    }

    if (!impl_->codecPrivateAnnexB.empty() && !impl_->codecPrivateInjected) {
        std::vector<uint8_t> prefixed;
        prefixed.reserve(impl_->codecPrivateAnnexB.size() + baseOut.size());
        prefixed.insert(prefixed.end(), impl_->codecPrivateAnnexB.begin(), impl_->codecPrivateAnnexB.end());
        prefixed.insert(prefixed.end(), baseOut.begin(), baseOut.end());
        baseOut.swap(prefixed);
        impl_->codecPrivateInjected = true;
    }
}

bool MVCMatroskaDemuxer::readNextFramePair(FramePair& framePair) {
    if (!isOpen()) {
        return false;
    }

    framePair.baseData.clear();
    framePair.dependentData.clear();
    framePair.timestamp = 0;
    framePair.isKeyframe = false;

    if (impl_->hasSeparateTracks) {
        // Robust implementation for Separate Tracks
        // Buffers blocks until we find a matching pair (Base + MVC)
        
        while (true) {
            // 1. Scan buffer for the earliest Base block
            auto baseIt = impl_->blockQueue.end();
            for (auto it = impl_->blockQueue.begin(); it != impl_->blockQueue.end(); ++it) {
                if (it->trackNumber == impl_->baseTrackNum) {
                    baseIt = it;
                    break; // Found earliest base block
                }
            }

            if (baseIt != impl_->blockQueue.end()) {
                // 2. We have a Base block. Look for a matching MVC block.
                // Timestamps in MKV are usually precise, but allow small tolerance just in case
                int64_t targetTs = baseIt->timestamp;
                auto mvcIt = impl_->blockQueue.end();
                
                for (auto it = impl_->blockQueue.begin(); it != impl_->blockQueue.end(); ++it) {
                    if (it->trackNumber == impl_->mvcTrackNum) {
                        if (std::abs(static_cast<long long>(it->timestamp - targetTs)) < 50000) { // < 0.05ms tolerance (practically exact)
                            mvcIt = it;
                            break;
                        }
                    }
                }

                if (mvcIt != impl_->blockQueue.end()) {
                    // FOUND PAIR!
                    
                    // CRITICAL FIX: Convert AVCC (Length-Prefix) to Annex B (Start Codes)
                    // Separate Tracks contain raw AVCC blocks. We must parse and convert them.
                    // We use separateNALUnits which handles AVCC->AnnexB and NAL classification.
                    
                    std::vector<uint8_t> rawBase = std::move(baseIt->data);
                    std::vector<uint8_t> rawDep = std::move(mvcIt->data);
                    
                    // 1. Process Base Block
                    std::vector<uint8_t> baseFromBase;
                    std::vector<uint8_t> depFromBase;
                    separateNALUnits(rawBase, baseFromBase, depFromBase);
                    
                    // 2. Process Dependent Block
                    std::vector<uint8_t> baseFromDep;
                    std::vector<uint8_t> depFromDep;
                    separateNALUnits(rawDep, baseFromDep, depFromDep);
                    
                    // 3. Merge results
                    // Usually base block has base NALs, dep block has dep NALs.
                    // But separateNALUnits sorts them correctly by NAL type.
                    
                    framePair.baseData = std::move(baseFromBase);
                    if (!baseFromDep.empty()) {
                        framePair.baseData.insert(framePair.baseData.end(), baseFromDep.begin(), baseFromDep.end());
                    }
                    
                    framePair.dependentData = std::move(depFromDep);
                    if (!depFromBase.empty()) {
                        framePair.dependentData.insert(framePair.dependentData.begin(), depFromBase.begin(), depFromBase.end());
                    }

                    framePair.timestamp = baseIt->timestamp;  // Already in ms
                    framePair.isKeyframe = baseIt->isKeyframe || mvcIt->isKeyframe;
                    
                    // Cleanup: Remove these blocks
                    // ... (rest of the cleanup logic remains similar, just ensure iterators handled correctly)
                    
                    // Since we moved data out of iterators, we can just rebuild the queue excluding empty blocks
                    baseIt->data.clear(); 
                    mvcIt->data.clear();
                    
                    std::deque<MatroskaBlock> newQueue;
                    for (auto& b : impl_->blockQueue) {
                        if (b.data.empty()) continue; 
                        if (b.timestamp < targetTs - 100000000) continue; // Garbage collect old blocks
                        newQueue.push_back(std::move(b));
                    }
                    impl_->blockQueue = std::move(newQueue);
                    
                    return true;
                }
                
                // If we have Base but NO matching MVC yet:
                // We need to read more blocks.
                // BUT check if we have read "too far" past the Base block?
                // If buffer is full and we still don't have MVC, maybe MVC is missing for this frame.
                // Output Base only (2D fallback).
                if (impl_->blockQueue.size() > impl_->maxBufferedBlocks) {
                    // STEREO-SHEAR GUARD: emitting base-only here means the
                    // GPU will pair this base view with the NEXT frame's dep
                    // view (FrameId N vs FrameId N+1) — that's the classic
                    // stereoscopic shear. We track + log loudly so the
                    // condition is visible in diagnostics. The actual safety
                    // net is downstream: edge264 holds the frame until both
                    // ready0 and ready1 are set, and the Python POC-PAIR
                    // sentinel drops mismatched pairs.
                    static thread_local uint64_t lonely_base_count = 0;
                    ++lonely_base_count;
                    if (lonely_base_count <= 10 || (lonely_base_count % 100) == 0) {
                        fprintf(stderr,
                                "[MVC-DEMUX][LONELY-BASE #%llu] base ts=%lld ms has no matching dep "
                                "after %zu queued blocks — emitting 2D fallback\n",
                                (unsigned long long)lonely_base_count,
                                (long long)baseIt->timestamp,
                                impl_->blockQueue.size());
                    }

                    // Force output Base only (2D fallback)
                    std::vector<uint8_t> rawBase = std::move(baseIt->data);
                    std::vector<uint8_t> dummyDep;

                    // Convert AVCC to Annex B
                    separateNALUnits(rawBase, framePair.baseData, dummyDep);

                    // If any dep units were in base track (unlikely), attach them
                    framePair.dependentData = std::move(dummyDep);

                    framePair.timestamp = baseIt->timestamp;
                    framePair.isKeyframe = baseIt->isKeyframe;

                    // Remove base block
                    baseIt->data.clear(); // Mark as moved
                     std::deque<MatroskaBlock> newQueue;
                    for (auto& b : impl_->blockQueue) {
                        if (b.data.empty()) continue;
                        newQueue.push_back(std::move(b));
                    }
                    impl_->blockQueue = std::move(newQueue);

                    return true;
                }
            }
            
            // 3. Read next block from file
            MatroskaBlock newBlock;
            if (!impl_->reader.readNextBlock(newBlock)) {
                // EOF
                // If we have any remaining Base frames in buffer, flush them one by one
                if (baseIt != impl_->blockQueue.end()) {
                     framePair.baseData = std::move(baseIt->data);
                     framePair.timestamp = baseIt->timestamp;
                     framePair.isKeyframe = baseIt->isKeyframe;
                     baseIt->data.clear();
                     return true;
                }
                return false;
            }
            
            // Filter relevant tracks
            if (newBlock.trackNumber == impl_->baseTrackNum || newBlock.trackNumber == impl_->mvcTrackNum) {
                // Video track - add to video queue
                impl_->blockQueue.push_back(std::move(newBlock));
            }
            // ========== SUBTITLE BLOCK CAPTURE ==========
            else if (impl_->activeSubtitleTrack > 0 && newBlock.trackNumber == impl_->activeSubtitleTrack) {
                // Active subtitle track - add to subtitle queue
                SubtitleBlock subBlock;
                subBlock.trackNumber = newBlock.trackNumber;
                // MatroskaReader already returns timestamps in milliseconds!
                subBlock.timestampMs = newBlock.timestamp;
                subBlock.data = std::move(newBlock.data);

                impl_->subtitleQueue.push_back(std::move(subBlock));

                // Limit queue size to prevent memory bloat
                while (impl_->subtitleQueue.size() > impl_->maxSubtitleQueueSize) {
                    impl_->subtitleQueue.pop_front();
                }
            }
            // ============================================
        }

    } else {
        // Single track with interleaved NAL units (Existing logic works fine)
        while (true) {
            MatroskaBlock block;
            if (!impl_->popNextVideoBlock(block)) {
                return false;
            }

            // Sanity guard: drop suspiciously small blocks. A real H.264 MVC
            // Matroska block contains at minimum a slice header + payload —
            // anything under 32 bytes is almost certainly truncated and
            // would feed edge264 a malformed NAL stream, risking CABAC
            // desync that masquerades as stereo shear / banding downstream.
            if (block.data.size() < 32) {
                static thread_local uint64_t tiny_block_count = 0;
                ++tiny_block_count;
                if (tiny_block_count <= 5) {
                    fprintf(stderr,
                            "[MVC-DEMUX][TINY-BLOCK #%llu] dropping %zu-byte block at ts=%lld ms\n",
                            (unsigned long long)tiny_block_count,
                            block.data.size(),
                            (long long)block.timestamp);
                }
                continue;
            }

            framePair.timestamp = block.timestamp;  // Already in ms
            framePair.isKeyframe = block.isKeyframe;

            // Separate NAL units
            separateNALUnits(block.data, framePair.baseData, framePair.dependentData);

            // Return if we got valid data
            if (!framePair.baseData.empty() || !framePair.dependentData.empty()) {
                impl_->ensureBufferedBlocks();
                return true;
            }
        }
    }
}

void MVCMatroskaDemuxer::prepareCodecPrivateAnnexB() {
    impl_->codecPrivateAnnexB.clear();
    impl_->codecPrivateInjected = false;
    std::vector<uint8_t> avcc = getCodecPrivate();
    if (avcc.size() < 7 || avcc[0] != 1) {
        return;
    }

    // Read NAL length size from AVCC header byte 4 (lengthSizeMinusOne field)
    // The lower 2 bits indicate: 0 = 1 byte, 1 = 2 bytes, 3 = 4 bytes
    impl_->nalLengthSize = (avcc[4] & 0x03) + 1;
    fprintf(stderr, "[MVC-DEMUX] AVCC NAL length size: %u bytes\n", impl_->nalLengthSize);

    size_t offset = 5; // skip version, profile, compat, level, lengthSizeMinusOne
    auto appendAnnexB = [&](const uint8_t* data, size_t len) {
        static const uint8_t startCode[4] = {0x00, 0x00, 0x00, 0x01};
        size_t oldSize = impl_->codecPrivateAnnexB.size();
        impl_->codecPrivateAnnexB.resize(oldSize + 4 + len);
        std::memcpy(impl_->codecPrivateAnnexB.data() + oldSize, startCode, 4);
        std::memcpy(impl_->codecPrivateAnnexB.data() + oldSize + 4, data, len);
    };

    if (offset >= avcc.size()) {
        return;
    }
    uint8_t numSPS = avcc[offset++] & 0x1f;
    for (uint8_t i = 0; i < numSPS; ++i) {
        if (offset + 2 > avcc.size()) return;
        uint16_t length = (avcc[offset] << 8) | avcc[offset + 1];
        offset += 2;
        if (offset + length > avcc.size()) return;
        appendAnnexB(avcc.data() + offset, length);
        offset += length;
    }
    if (offset >= avcc.size()) {
        return;
    }
    uint8_t numPPS = avcc[offset++];
    for (uint8_t i = 0; i < numPPS; ++i) {
        if (offset + 2 > avcc.size()) return;
        uint16_t length = (avcc[offset] << 8) | avcc[offset + 1];
        offset += 2;
        if (offset + length > avcc.size()) return;
        appendAnnexB(avcc.data() + offset, length);
        offset += length;
    }

    // ========== MVC EXTENSION: Parse mvcC box ==========
    // MVC streams have an extension box after AVCC containing SubsetSPS and MVC PPS
    // The mvcC box is identified by signature "mvcC" (0x6d766343)
    // Structure: [box_size:4][mvcC:4][version:1][profile:1][compat:1][level:1][nalLen:1][numSSPS:1][SSPS entries][numPPS:1][PPS entries]

    // Search for mvcC signature in remaining data
    const uint8_t mvcC_sig[4] = {0x6d, 0x76, 0x63, 0x43};  // "mvcC"
    size_t mvcC_pos = 0;
    bool found_mvcC = false;

    for (size_t i = offset; i + 4 <= avcc.size(); ++i) {
        if (avcc[i] == mvcC_sig[0] && avcc[i+1] == mvcC_sig[1] &&
            avcc[i+2] == mvcC_sig[2] && avcc[i+3] == mvcC_sig[3]) {
            mvcC_pos = i + 4;  // Position after "mvcC" signature
            found_mvcC = true;
            fprintf(stderr, "[MVC-DEMUX] Found mvcC extension at offset %zu\n", i);
            break;
        }
    }

    if (found_mvcC && mvcC_pos + 6 <= avcc.size()) {
        // Parse mvcC header (similar structure to AVCC)
        // [version:1][profile:1][compat:1][level:1][nalLenMinusOne:1][numSSPS:1]
        size_t mvcOffset = mvcC_pos + 5;  // Skip version, profile, compat, level, nalLen

        if (mvcOffset < avcc.size()) {
            uint8_t numSubsetSPS = avcc[mvcOffset++] & 0x1F;
            fprintf(stderr, "[MVC-DEMUX] mvcC contains %u SubsetSPS\n", numSubsetSPS);

            for (uint8_t i = 0; i < numSubsetSPS; ++i) {
                if (mvcOffset + 2 > avcc.size()) break;
                uint16_t sspsLen = (avcc[mvcOffset] << 8) | avcc[mvcOffset + 1];
                mvcOffset += 2;
                if (mvcOffset + sspsLen > avcc.size()) break;
                fprintf(stderr, "[MVC-DEMUX] SubsetSPS #%u: %u bytes\n", i, sspsLen);
                appendAnnexB(avcc.data() + mvcOffset, sspsLen);
                mvcOffset += sspsLen;
            }

            // Parse MVC PPS entries
            if (mvcOffset < avcc.size()) {
                uint8_t numMvcPPS = avcc[mvcOffset++];
                fprintf(stderr, "[MVC-DEMUX] mvcC contains %u MVC PPS\n", numMvcPPS);

                for (uint8_t i = 0; i < numMvcPPS; ++i) {
                    if (mvcOffset + 2 > avcc.size()) break;
                    uint16_t ppsLen = (avcc[mvcOffset] << 8) | avcc[mvcOffset + 1];
                    mvcOffset += 2;
                    if (mvcOffset + ppsLen > avcc.size()) break;
                    fprintf(stderr, "[MVC-DEMUX] MVC PPS #%u: %u bytes\n", i, ppsLen);
                    appendAnnexB(avcc.data() + mvcOffset, ppsLen);
                    mvcOffset += ppsLen;
                }
            }
        }
    }
    // ===================================================
}

bool MVCMatroskaDemuxer::seek(int64_t timestampMs) {
    impl_->blockQueue.clear();
    impl_->subtitleQueue.clear();  // Clear subtitle queue on seek
    // Pass baseTrackNum to filter seek (avoids stopping on audio blocks with matching timestamps)
    return impl_->reader.seek(timestampMs, static_cast<int32_t>(impl_->baseTrackNum));
}

int64_t MVCMatroskaDemuxer::getLastCueTimestamp() const {
    // V8 INDEX-BASED SYNC: Forward to MatroskaReader
    return impl_->reader.getLastCueTimestamp();
}

std::vector<int64_t> MVCMatroskaDemuxer::getCuesTimestamps() const {
    // V8 SEEK OPTIMIZATION: Forward to MatroskaReader
    return impl_->reader.getCuesTimestamps();
}

bool MVCMatroskaDemuxer::seekToCue(int64_t cueTimestampMs) {
    // V8 SEEK OPTIMIZATION: Forward to MatroskaReader with block queue clear
    impl_->blockQueue.clear();
    impl_->subtitleQueue.clear();  // Clear subtitle queue on seek
    return impl_->reader.seekToCue(cueTimestampMs);
}

// ========== SUBTITLE STREAMING IMPLEMENTATION ==========

std::vector<MVCMatroskaDemuxer::SubtitleTrackInfo> MVCMatroskaDemuxer::getSubtitleTracks() const {
    return impl_->subtitleTracks;
}

void MVCMatroskaDemuxer::setActiveSubtitleTrack(uint32_t trackNumber) {
    fprintf(stderr, "[SUB-DEBUG] setActiveSubtitleTrack(%u) called\n", trackNumber);
    fflush(stderr);

    // Validate track number exists (0 = disable)
    if (trackNumber == 0) {
        impl_->activeSubtitleTrack = 0;
        impl_->subtitleQueue.clear();
        fprintf(stderr, "[SUB-DEBUG] Subtitles DISABLED\n");
        fflush(stderr);
        return;
    }

    // Check if track exists in our list
    bool found = false;
    for (const auto& track : impl_->subtitleTracks) {
        fprintf(stderr, "[SUB-DEBUG] Checking track %u against requested %u\n", track.trackNumber, trackNumber);
        if (track.trackNumber == trackNumber) {
            found = true;
            break;
        }
    }

    if (found) {
        impl_->activeSubtitleTrack = trackNumber;
        impl_->subtitleQueue.clear();  // Clear queue when switching tracks
        fprintf(stderr, "[SUB-DEBUG] Subtitles ENABLED for track %u\n", trackNumber);
    } else {
        fprintf(stderr, "[SUB-DEBUG] Track %u NOT FOUND in subtitleTracks list!\n", trackNumber);
    }
    fflush(stderr);
}

uint32_t MVCMatroskaDemuxer::getActiveSubtitleTrack() const {
    return impl_->activeSubtitleTrack;
}

bool MVCMatroskaDemuxer::hasSubtitleData() const {
    return !impl_->subtitleQueue.empty();
}

bool MVCMatroskaDemuxer::readNextSubtitleBlock(SubtitleBlock& block) {
    if (impl_->subtitleQueue.empty()) {
        return false;
    }

    block = std::move(impl_->subtitleQueue.front());
    impl_->subtitleQueue.pop_front();
    return true;
}

// ========================================================

} // namespace mvc_demux
