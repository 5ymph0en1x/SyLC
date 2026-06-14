#include "mvc_ssif_demuxer.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>  // for std::abs

namespace mvc_demux {

// NAL unit types
constexpr uint8_t NAL_TYPE_SPS = 7;
constexpr uint8_t NAL_TYPE_PPS = 8;

MVCSSIFDemuxer::MVCSSIFDemuxer()
    : ssifParser_(std::make_unique<SSIFParser>()),
      baseReader_(std::make_unique<M2TSReader>()),
      dependentReader_(std::make_unique<M2TSReader>()),
      ssifReader_(std::make_unique<M2TSReader>()),
      hasCodecPrivate_(false),
      hasSPS_(false),
      hasPPS_(false),
      hasVideoInfo_(false),
      isStreamingMode_(false),
      currentExtentIndex_(0),
      base_stream_pos_(0),
      dependent_stream_pos_(0),
      basePtsOffset_(0),
      basePtsInitialized_(false),
      isOpen_(false) {
    videoInfo_ = {};
    pendingBase_ = {};
    pendingDependent_ = {};
    currentFrame_ = {};
    pendingBase_.hasData = false;
    pendingDependent_.hasData = false;
}

MVCSSIFDemuxer::~MVCSSIFDemuxer() {
    close();
}

bool MVCSSIFDemuxer::open(const std::string& path) {
    std::cout << "[MVCSSIFDemuxer] Opening SSIF 3D stream: " << path << std::endl;

    // Determine if path is .ssif or .m2ts
    std::string ssifPath = path;
    if (path.find(".m2ts") != std::string::npos || path.find(".M2TS") != std::string::npos) {
        // Auto-detect SSIF file
        ssifPath = SSIFParser::detectSSIFPath(path);
        if (ssifPath.empty() || !SSIFParser::hasSSIF(path)) {
            std::cerr << "[MVCSSIFDemuxer] No SSIF file found for: " << path << std::endl;
            return false;
        }
        std::cout << "[MVCSSIFDemuxer] Auto-detected SSIF: " << ssifPath << std::endl;
    }

    // Parse SSIF file
    if (!ssifParser_->parse(ssifPath)) {
        std::cerr << "[MVCSSIFDemuxer] Failed to parse SSIF file" << std::endl;
        return false;
    }

    const auto& info = ssifParser_->getInfo();
    isStreamingMode_ = ssifParser_->isStreamingMode();

    if (isStreamingMode_) {
        // STREAMING MODE: Read directly from the large SSIF file
        std::cout << "[MVCSSIFDemuxer] Using STREAMING MODE for large SSIF" << std::endl;

        if (!ssifReader_->open(ssifPath)) {
            std::cerr << "[MVCSSIFDemuxer] Failed to open SSIF file for streaming: " << ssifPath << std::endl;
            return false;
        }
        std::cout << "[MVCSSIFDemuxer] Opened SSIF file for direct streaming" << std::endl;

        // Try to open base M2TS as fallback for audio
        if (!info.baseStreamPath.empty()) {
            if (baseReader_->open(info.baseStreamPath)) {
                std::cout << "[MVCSSIFDemuxer] Opened base stream (for audio/fallback)" << std::endl;
            }
        }
    } else {
        // DUAL-FILE MODE: Use separate M2TS files (original behavior)
        std::cout << "[SSIF] Using DUAL-FILE MODE" << std::endl;
        std::cout << "[SSIF] Validating stream files:" << std::endl;
        std::cout << "[SSIF]   Base stream: " << info.baseStreamPath << std::endl;
        std::cout << "[SSIF]   Dependent stream: " << info.dependentStreamPath << std::endl;

        // Open base stream (left eye)
        if (!baseReader_->open(info.baseStreamPath)) {
            std::cerr << "[MVCSSIFDemuxer] Failed to open base stream: " << info.baseStreamPath << std::endl;
            std::cerr << "[SSIF] ERROR: Companion .m2ts base file not found or unreadable" << std::endl;
            return false;
        }
        std::cout << "[MVCSSIFDemuxer] Opened base stream (left eye)" << std::endl;

        // Open dependent stream (right eye)
        if (!dependentReader_->open(info.dependentStreamPath)) {
            std::cerr << "[MVCSSIFDemuxer] Failed to open dependent stream: " << info.dependentStreamPath << std::endl;
            std::cerr << "[SSIF] ERROR: Companion .m2ts dependent file not found or unreadable" << std::endl;
            baseReader_->close();
            return false;
        }
        std::cout << "[MVCSSIFDemuxer] Opened dependent stream (right eye)" << std::endl;
    }

    // Read initial packets to find PAT/PMT and SPS/PPS
    M2TSReader::TSPacket packet;
    int maxProbePackets = 10000;

    // Use appropriate reader for probing
    M2TSReader* probeReader = isStreamingMode_ ? ssifReader_.get() : baseReader_.get();
    std::cout << "[MVCSSIFDemuxer] Probing " << (isStreamingMode_ ? "SSIF" : "base") << " stream for video info..." << std::endl;

    // Phase 1: Detect PIDs from PMT first (read enough packets to find PAT/PMT)
    uint16_t basePid = 0;
    uint16_t mvcPid = 0;
    for (int i = 0; i < 1000 && probeReader->readPacket(packet); i++) {
        auto videoPids = probeReader->getVideoPids();
        if (!videoPids.empty() && basePid == 0) {
            // Detect PIDs from PMT stream types
            const auto& programs = probeReader->getPrograms();
            for (const auto& prog : programs) {
                for (const auto& [pid, streamType] : prog.streamPids) {
                    if (streamType == 0x1B && basePid == 0) {  // H.264 base
                        basePid = pid;
                    } else if (streamType == 0x20 && mvcPid == 0) {  // MVC
                        mvcPid = pid;
                    }
                }
            }

            // Fallback heuristics
            if (basePid == 0 && mvcPid == 0 && videoPids.size() >= 2) {
                basePid = std::min(videoPids[0], videoPids[1]);
                mvcPid = std::max(videoPids[0], videoPids[1]);
            } else if (basePid != 0 && mvcPid == 0) {
                mvcPid = basePid + 1;
            } else if (basePid == 0 && mvcPid != 0) {
                basePid = mvcPid - 1;
            }

            if (basePid != 0) {
                std::cout << "[MVCSSIFDemuxer] Detected PIDs: base=0x" << std::hex << basePid
                          << ", mvc=0x" << mvcPid << std::dec << std::endl;
                break;
            }
        }
    }

    if (basePid == 0) {
        std::cerr << "[MVCSSIFDemuxer] Failed to detect video PIDs" << std::endl;
        close();
        return false;
    }

    // Set video info
    videoInfo_.width = 1920;
    videoInfo_.height = 1080;
    videoInfo_.fps = 23.976;
    videoInfo_.hasMVC = true;
    videoInfo_.baseVideoPid = basePid;
    videoInfo_.mvcVideoPid = mvcPid;
    hasVideoInfo_ = true;

    std::cout << "[MVCSSIFDemuxer] Using Blu-ray 3D SSIF dimensions: "
              << videoInfo_.width << "x" << videoInfo_.height
              << " @ " << videoInfo_.fps << " fps" << std::endl;

    // Phase 2: Extract codec_private from BASE .m2ts file (not SSIF)
    // SSIF file contains mostly dependent view data, base view SPS/PPS is in the base .m2ts
    std::cout << "[MVCSSIFDemuxer] Extracting codec private from base .m2ts file..." << std::endl;

    // Use baseReader_ which is opened on the base .m2ts file
    baseReader_->seek(0);

    for (int i = 0; i < maxProbePackets && baseReader_->readPacket(packet) && !hasCodecPrivate_; i++) {
        // Only process packets from base stream for codec_private
        if (packet.pid == basePid) {
            processVideoPacket(packet, true);
        }
    }

    // Flush pending PES buffer for base PID to extract remaining parameter sets
    if (!hasCodecPrivate_) {
        auto it = basePesStates_.find(basePid);
        if (it != basePesStates_.end()) {
            auto& state = it->second;
            if (state.hasStarted && !state.buffer.empty()) {
                int64_t pts, dts;
                size_t headerLength;
                if (parsePESHeader(state.buffer, pts, dts, headerLength)) {
                    std::vector<uint8_t> nalData(state.buffer.begin() + headerLength,
                                                state.buffer.end());
                    extractCodecPrivate(nalData);
                }
            }
        }
    }

    if (!hasCodecPrivate_) {
        std::cerr << "[MVCSSIFDemuxer] Failed to extract codec private (SPS/PPS)" << std::endl;
        close();
        return false;
    }

    // Reset readers to beginning
    if (isStreamingMode_) {
        // Streaming mode: reset SSIF reader only
        ssifReader_->seek(0);
        std::cout << "[MVCSSIFDemuxer] SSIF reader reset to start" << std::endl;
    } else {
        // Dual-file mode: reset both M2TS readers
        baseReader_->close();
        dependentReader_->close();

        if (!baseReader_->open(info.baseStreamPath)) {
            return false;
        }
        if (!dependentReader_->open(info.dependentStreamPath)) {
            baseReader_->close();
            return false;
        }
    }

    // CRITICAL FIX: Clear all PES states and buffers after codec_private extraction
    // The PES states contain partial data from probing the base .m2ts file
    // When streaming mode reads from the SSIF file (different content), these
    // stale states cause frame assembly corruption and wrong timestamps
    basePesStates_.clear();
    dependentPesStates_.clear();
    streamingPesState_.clear();

    // Clear frame buffers
    baseFrameBuffer_.clear();
    dependentFrameBuffer_.clear();
    frameQueue_.clear();

    // Reset pending data
    pendingBase_.baseView.clear();
    pendingBase_.dependentView.clear();
    pendingBase_.pts = 0;
    pendingBase_.hasData = false;
    pendingBase_.alreadyPrefixed = false;

    pendingDependent_.baseView.clear();
    pendingDependent_.dependentView.clear();
    pendingDependent_.pts = 0;
    pendingDependent_.hasData = false;
    pendingDependent_.alreadyPrefixed = false;

    // Reset PTS normalization for fresh timestamp calculation
    // The first valid PTS will be captured and used as offset
    basePtsOffset_ = 0;
    basePtsInitialized_ = false;

    std::cout << "[MVCSSIFDemuxer] All PES states and buffers cleared for fresh start" << std::endl;

    std::cout << "[MVCSSIFDemuxer] Base video PID: 0x" << std::hex
              << videoInfo_.baseVideoPid << std::dec << std::endl;

    // Reset extent index
    currentExtentIndex_ = 0;
    isOpen_ = true;

    std::cout << "[MVCSSIFDemuxer] SSIF demuxer opened successfully ("
              << (isStreamingMode_ ? "streaming" : "dual-file") << " mode)" << std::endl;
    return true;
}

void MVCSSIFDemuxer::close() {
    if (baseReader_) {
        baseReader_->close();
    }
    if (dependentReader_) {
        dependentReader_->close();
    }
    if (ssifReader_) {
        ssifReader_->close();
    }
    isOpen_ = false;
    isStreamingMode_ = false;
}

bool MVCSSIFDemuxer::readNextFramePair(FramePair& framePair) {
    if (!isOpen_) {
        return false;
    }

    if (isStreamingMode_) {
        // STREAMING MODE: Read directly from SSIF file
        // NAL units are separated by type: 1-5 = base, 14/20 = dependent
        return readNextFramePairStreaming(framePair);
    }

    // DUAL-FILE MODE: Process extents and accumulate frames from both streams
    const auto& extents = ssifParser_->getInfo().extents;

    while (currentExtentIndex_ < extents.size()) {
        const auto& extent = extents[currentExtentIndex_];

        // Read from the appropriate stream according to extent
        M2TSReader* reader = (extent.streamFileId == 0) ? baseReader_.get() : dependentReader_.get();
        bool isBase = (extent.streamFileId == 0);

        M2TSReader::TSPacket packet;
        if (reader->readPacket(packet)) {
            processVideoPacket(packet, isBase);

            // Check if we have synchronized frames
            if (synchronizeFrames(framePair)) {
                return true;
            }
        } else {
            // End of current extent, move to next
            currentExtentIndex_++;
        }
    }

    // Try to synchronize remaining buffered frames
    return synchronizeFrames(framePair);
}

bool MVCSSIFDemuxer::readNextFramePairStreaming(FramePair& framePair) {
    // STREAMING MODE: Read from interleaved SSIF file
    // SSIF contains packets from both base (PID 0x1011) and dependent (PID 0x1012) streams
    //
    // CRITICAL: SSIF format stores dependent (MVC) frames AHEAD of base frames in the file.
    // This means when reading linearly, we'll receive many MVC frames before their
    // corresponding base frames arrive. We MUST buffer frames and match by PTS.

    M2TSReader::TSPacket packet;
    int maxPackets = 500000;  // Increased limit to handle SSIF offset
    int packetsRead = 0;
    static int totalFramePairs = 0;

    // Get base and MVC PIDs
    uint16_t basePid = videoInfo_.baseVideoPid;
    uint16_t mvcPid = videoInfo_.mvcVideoPid;
    if (mvcPid == 0 && basePid != 0) {
        mvcPid = basePid + 1;
    }

    // First, check if we already have matching frames in our buffers
    auto tryMatchFrames = [&]() -> bool {
        if (baseFrameBuffer_.empty() || dependentFrameBuffer_.empty()) {
            return false;
        }

        // Wait for at least 3 base frames before emitting (to handle out-of-order arrival)
        // This ensures we can properly order frames by PTS
        // Exception: if we have many frames buffered, start emitting to avoid memory issues
        if (baseFrameBuffer_.size() < 3 && baseFrameBuffer_.size() < MAX_FRAME_BUFFER_SIZE / 2) {
            return false;  // Wait for more base frames
        }

        // Find the LOWEST PTS match (to emit frames in correct temporal order)
        // This is critical for decoder reference frame management
        int64_t bestPts = INT64_MAX;
        size_t bestBi = 0, bestDi = 0;
        bool foundMatch = false;

        for (size_t bi = 0; bi < baseFrameBuffer_.size(); bi++) {
            int64_t basePts = baseFrameBuffer_[bi].pts;

            for (size_t di = 0; di < dependentFrameBuffer_.size(); di++) {
                int64_t depPts = dependentFrameBuffer_[di].pts;
                int64_t ptsDiff = std::abs(basePts - depPts);

                if (ptsDiff <= PTS_MATCH_TOLERANCE && basePts < bestPts) {
                    bestPts = basePts;
                    bestBi = bi;
                    bestDi = di;
                    foundMatch = true;
                }
            }
        }

        if (foundMatch) {
            int64_t basePts = baseFrameBuffer_[bestBi].pts;

            // PTS NORMALIZATION: Capture first valid PTS as offset
            // Blu-ray streams often start at non-zero PTS (e.g., ~11s = 1048560 ticks)
            // We subtract this offset to get timestamps starting from 0
            if (!basePtsInitialized_) {
                basePtsOffset_ = basePts;
                basePtsInitialized_ = true;
                fprintf(stderr, "[SSIF] PTS normalization: first PTS=%lld (%.3fs), using as offset\n",
                        (long long)basePtsOffset_, basePtsOffset_ / 90000.0);
            }

            framePair.baseData = std::move(baseFrameBuffer_[bestBi].data);
            framePair.dependentData = std::move(dependentFrameBuffer_[bestDi].data);
            // Use normalized timestamp (PTS - offset) / 90
            int64_t normalizedPts = basePts - basePtsOffset_;
            framePair.timestamp = normalizedPts / 90;  // 90kHz to ms
            framePair.isKeyframe = baseFrameBuffer_[bestBi].isKeyframe;

            // Remove matched frames from buffers
            baseFrameBuffer_.erase(baseFrameBuffer_.begin() + bestBi);
            dependentFrameBuffer_.erase(dependentFrameBuffer_.begin() + bestDi);

            totalFramePairs++;
            if (totalFramePairs == 1 || totalFramePairs % 500 == 0) {
                fprintf(stderr, "[SSIF] Frame #%d: ts=%lld ms, base=%zu, dep=%zu%s\n",
                        totalFramePairs, (long long)framePair.timestamp,
                        framePair.baseData.size(), framePair.dependentData.size(),
                        framePair.isKeyframe ? " [IDR]" : "");
            }
            return true;
        }
        return false;
    };

    // Try matching existing buffered frames first
    if (tryMatchFrames()) {
        return true;
    }

    // Read more packets to fill buffers
    bool readSuccess = true;
    while (packetsRead < maxPackets && (readSuccess = ssifReader_->readPacket(packet))) {
        packetsRead++;

        // Process based on PID
        if (packet.pid == basePid) {
            processVideoPacket(packet, true);

            // If we got a valid base frame, add to buffer
            if (pendingBase_.hasData) {
                BufferedFrame bf;
                bf.data = std::move(pendingBase_.baseView);
                bf.pts = pendingBase_.pts;
                bf.isKeyframe = false;

                // Check for IDR
                for (size_t i = 0; i + 4 < bf.data.size(); i++) {
                    if (bf.data[i] == 0 && bf.data[i+1] == 0) {
                        size_t scLen = (bf.data[i+2] == 1) ? 3 :
                                       ((i + 3 < bf.data.size() && bf.data[i+2] == 0 && bf.data[i+3] == 1) ? 4 : 0);
                        if (scLen > 0 && i + scLen < bf.data.size()) {
                            uint8_t nalType = bf.data[i + scLen] & 0x1F;
                            if (nalType == 5) { bf.isKeyframe = true; break; }
                        }
                    }
                }

                baseFrameBuffer_.push_back(std::move(bf));
                pendingBase_.hasData = false;
                pendingBase_.baseView.clear();

                // Limit buffer size
                while (baseFrameBuffer_.size() > MAX_FRAME_BUFFER_SIZE) {
                    baseFrameBuffer_.erase(baseFrameBuffer_.begin());
                }
            }
        } else if (packet.pid == mvcPid) {
            processVideoPacket(packet, false);

            // If we got a valid dependent frame, add to buffer
            if (pendingDependent_.hasData) {
                BufferedFrame bf;
                bf.data = std::move(pendingDependent_.baseView);
                bf.pts = pendingDependent_.pts;
                bf.isKeyframe = false;

                dependentFrameBuffer_.push_back(std::move(bf));
                pendingDependent_.hasData = false;
                pendingDependent_.baseView.clear();

                // Limit buffer size
                while (dependentFrameBuffer_.size() > MAX_FRAME_BUFFER_SIZE) {
                    dependentFrameBuffer_.erase(dependentFrameBuffer_.begin());
                }
            }
        }

        // Try to match after adding new frames
        if (tryMatchFrames()) {
            return true;
        }

        // Early exit if we have enough buffered frames
        // This shouldn't normally happen if PTS matching is working correctly
        if (baseFrameBuffer_.size() >= MAX_FRAME_BUFFER_SIZE &&
            dependentFrameBuffer_.size() >= MAX_FRAME_BUFFER_SIZE) {
            // Buffer overflow - clear oldest frames to prevent memory issues
            baseFrameBuffer_.erase(baseFrameBuffer_.begin());
            dependentFrameBuffer_.erase(dependentFrameBuffer_.begin());
        }
    }

    // Only log on unexpected exit (EOF)
    if (!readSuccess) {
        fprintf(stderr, "[SSIF-BUFFER] EOF reached: baseBuffer=%zu, depBuffer=%zu\n",
                baseFrameBuffer_.size(), dependentFrameBuffer_.size());
    }

    return false;
}

void MVCSSIFDemuxer::processVideoPacketStreaming(const M2TSReader::TSPacket& packet) {
    // Process a video packet from streaming SSIF
    // Separate base and dependent NAL units based on NAL type

    auto& state = streamingPesState_[packet.pid];

    if (packet.payloadUnitStartIndicator) {
        // Flush previous PES
        if (state.hasStarted && !state.buffer.empty()) {
            int64_t pts, dts;
            size_t headerLength;

            if (parsePESHeader(state.buffer, pts, dts, headerLength)) {
                std::vector<uint8_t> nalData(state.buffer.begin() + headerLength,
                                            state.buffer.end());

                // Separate NAL units by type
                separateNALUnits(nalData, pts);
            }

            state.buffer.clear();
        }

        // Start new PES
        state.hasStarted = true;
    }

    // Accumulate payload
    if (state.hasStarted && !packet.payload.empty()) {
        state.buffer.insert(state.buffer.end(),
                          packet.payload.begin(),
                          packet.payload.end());
    }
}

void MVCSSIFDemuxer::separateNALUnits(const std::vector<uint8_t>& nalData, int64_t pts) {
    // Find and categorize NAL units
    // Base view: NAL types 1-8 (VCL slices, SPS, PPS)
    // Dependent view: NAL types 14, 20 (MVC prefix/slice)

    size_t i = 0;
    while (i < nalData.size()) {
        // Look for start code
        if (i + 2 < nalData.size() &&
            nalData[i] == 0x00 && nalData[i + 1] == 0x00) {

            size_t startCodeSize = 0;
            if (nalData[i + 2] == 0x01) {
                startCodeSize = 3;
            } else if (i + 3 < nalData.size() &&
                      nalData[i + 2] == 0x00 && nalData[i + 3] == 0x01) {
                startCodeSize = 4;
            }

            if (startCodeSize > 0) {
                size_t nalStart = i;
                i += startCodeSize;

                if (i < nalData.size()) {
                    uint8_t nalType = nalData[i] & 0x1F;

                    // Find next start code
                    size_t nextStart = i + 1;
                    while (nextStart + 2 < nalData.size()) {
                        if (nalData[nextStart] == 0x00 &&
                            nalData[nextStart + 1] == 0x00 &&
                            (nalData[nextStart + 2] == 0x01 ||
                             (nextStart + 3 < nalData.size() &&
                              nalData[nextStart + 2] == 0x00 &&
                              nalData[nextStart + 3] == 0x01))) {
                            break;
                        }
                        nextStart++;
                    }

                    // Copy NAL unit
                    std::vector<uint8_t> nalUnit(nalData.begin() + nalStart,
                                                 nalData.begin() + nextStart);

                    // Categorize by type
                    if (nalType == 14 || nalType == 20) {
                        // Dependent view (MVC)
                        if (!pendingDependent_.hasData) {
                            pendingDependent_.baseView.clear();
                            pendingDependent_.pts = pts;
                            pendingDependent_.hasData = false;
                        }
                        pendingDependent_.baseView.insert(pendingDependent_.baseView.end(),
                                                         nalUnit.begin(), nalUnit.end());
                        pendingDependent_.hasData = true;
                    } else if (nalType >= 1 && nalType <= 8) {
                        // Base view
                        if (!pendingBase_.hasData) {
                            pendingBase_.baseView.clear();
                            pendingBase_.pts = pts;
                            pendingBase_.hasData = false;
                        }
                        pendingBase_.baseView.insert(pendingBase_.baseView.end(),
                                                    nalUnit.begin(), nalUnit.end());
                        pendingBase_.hasData = true;
                    }

                    i = nextStart;
                    continue;
                }
            }
        }
        i++;
    }
}

void MVCSSIFDemuxer::processVideoPacket(const M2TSReader::TSPacket& packet, bool isBase) {
    auto& pesStates = isBase ? basePesStates_ : dependentPesStates_;
    auto& state = pesStates[packet.pid];
    static int basePesFlushCount = 0;
    static int mvcPesFlushCount = 0;

    if (packet.payloadUnitStartIndicator) {
        // Flush previous PES
        if (state.hasStarted && !state.buffer.empty()) {
            int64_t pts, dts;
            size_t headerLength;

            if (parsePESHeader(state.buffer, pts, dts, headerLength)) {
                std::vector<uint8_t> nalData(state.buffer.begin() + headerLength,
                                            state.buffer.end());

                if (!hasCodecPrivate_) {
                    extractCodecPrivate(nalData);
                }

                auto& pending = isBase ? pendingBase_ : pendingDependent_;

                // CRITICAL FIX: Only update pending if this PES has valid video content
                // For base: require NAL type 1 or 5 (slice) and size > 100 bytes
                // For MVC: require NAL type 20 (MVC slice) and size > 100 bytes
                // Small frames (skip/low-motion) can be 150-250 bytes
                bool hasValidContent = false;

                if (isBase) {
                    // Check for base slice (NAL type 1 or 5)
                    // Small frames (skip frames, low-motion B-frames) can be ~150-200 bytes
                    // Minimum 100 bytes to filter out parameter-set-only PES
                    for (size_t i = 0; i + 4 < nalData.size(); i++) {
                        if (nalData[i] == 0 && nalData[i+1] == 0) {
                            size_t scLen = 0;
                            if (nalData[i+2] == 1) scLen = 3;
                            else if (i + 3 < nalData.size() && nalData[i+2] == 0 && nalData[i+3] == 1) scLen = 4;
                            if (scLen > 0 && i + scLen < nalData.size()) {
                                uint8_t nalType = nalData[i + scLen] & 0x1F;
                                if ((nalType == 1 || nalType == 5) && nalData.size() > 100) {
                                    hasValidContent = true;
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    // Check for MVC slice (NAL type 20)
                    // MVC slices can be very small (226 bytes for skip frames)
                    // Only filter out parameter sets (NAL type 15 = Subset SPS)
                    for (size_t i = 0; i + 4 < nalData.size(); i++) {
                        if (nalData[i] == 0 && nalData[i+1] == 0) {
                            size_t scLen = 0;
                            if (nalData[i+2] == 1) scLen = 3;
                            else if (i + 3 < nalData.size() && nalData[i+2] == 0 && nalData[i+3] == 1) scLen = 4;
                            if (scLen > 0 && i + scLen < nalData.size()) {
                                uint8_t nalType = nalData[i + scLen] & 0x1F;
                                // NAL type 20 = MVC slice (valid frame data)
                                // Minimum 100 bytes to filter out tiny filler NALs
                                if (nalType == 20 && nalData.size() > 100) {
                                    hasValidContent = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                // Only update pending if valid content OR if pending is empty
                // This prevents small PES from overwriting valid frame data
                if (hasValidContent || !pending.hasData) {
                    pending.baseView = std::move(nalData);
                    pending.pts = pts;
                    pending.hasData = hasValidContent;

                    if (isBase) {
                        basePesFlushCount++;
                    } else {
                        mvcPesFlushCount++;
                    }
                }
            }

            state.buffer.clear();
        }

        // Start new PES
        state.hasStarted = true;
    }

    // Accumulate payload
    if (state.hasStarted && !packet.payload.empty()) {
        state.buffer.insert(state.buffer.end(),
                          packet.payload.begin(),
                          packet.payload.end());
    }
}

bool MVCSSIFDemuxer::parsePESHeader(const std::vector<uint8_t>& pesData,
                                    int64_t& pts, int64_t& dts,
                                    size_t& headerLength) {
    if (pesData.size() < 9) {
        return false;
    }

    // Check PES start code
    if (pesData[0] != 0x00 || pesData[1] != 0x00 || pesData[2] != 0x01) {
        return false;
    }

    uint8_t ptsDtsFlags = (pesData[7] >> 6) & 0x03;
    uint8_t pesHeaderLength = pesData[8];

    headerLength = 9 + pesHeaderLength;

    pts = 0;
    dts = 0;
    if (ptsDtsFlags >= 2 && pesData.size() >= 14) {
        pts = (static_cast<int64_t>(pesData[9] & 0x0E) << 29) |
              (static_cast<int64_t>(pesData[10]) << 22) |
              (static_cast<int64_t>(pesData[11] & 0xFE) << 14) |
              (static_cast<int64_t>(pesData[12]) << 7) |
              (static_cast<int64_t>(pesData[13]) >> 1);
    }

    if (ptsDtsFlags == 3 && pesData.size() >= 19) {
        dts = (static_cast<int64_t>(pesData[14] & 0x0E) << 29) |
              (static_cast<int64_t>(pesData[15]) << 22) |
              (static_cast<int64_t>(pesData[16] & 0xFE) << 14) |
              (static_cast<int64_t>(pesData[17]) << 7) |
              (static_cast<int64_t>(pesData[18]) >> 1);
    }

    return true;
}

void MVCSSIFDemuxer::extractCodecPrivate(const std::vector<uint8_t>& nalData) {
    // Extract SPS/PPS NAL units - need BOTH before marking complete
    size_t i = 0;
    while (i < nalData.size()) {
        if (i + 2 < nalData.size() &&
            nalData[i] == 0x00 && nalData[i + 1] == 0x00) {
            size_t startCodeSize = 0;
            if (nalData[i + 2] == 0x01) {
                startCodeSize = 3;
            } else if (i + 3 < nalData.size() &&
                      nalData[i + 2] == 0x00 && nalData[i + 3] == 0x01) {
                startCodeSize = 4;
            }

            if (startCodeSize > 0) {
                i += startCodeSize;
                if (i < nalData.size()) {
                    uint8_t nalType = nalData[i] & 0x1F;

                    // Find next start code
                    size_t nextStart = i + 1;
                    while (nextStart + 2 < nalData.size()) {
                        if (nalData[nextStart] == 0x00 &&
                            nalData[nextStart + 1] == 0x00 &&
                            (nalData[nextStart + 2] == 0x01 ||
                             (nextStart + 3 < nalData.size() &&
                              nalData[nextStart + 2] == 0x00 &&
                              nalData[nextStart + 3] == 0x01))) {
                            break;
                        }
                        nextStart++;
                    }

                    // Extract SPS (type 7) - only if not already found
                    if (nalType == NAL_TYPE_SPS && !hasSPS_) {
                        size_t nalStart = i - startCodeSize;
                        size_t nalLength = nextStart - nalStart;

                        // SPS should be at the beginning of codec private
                        std::vector<uint8_t> spsData(nalData.begin() + nalStart,
                                                     nalData.begin() + nalStart + nalLength);
                        codecPrivate_.insert(codecPrivate_.begin(), spsData.begin(), spsData.end());

                        std::cout << "[MVCSSIFDemuxer] Extracted SPS (NAL type 7) for codec private - "
                                  << nalLength << " bytes" << std::endl;
                        hasSPS_ = true;
                    }
                    // Extract PPS (type 8) - only if not already found
                    else if (nalType == NAL_TYPE_PPS && !hasPPS_) {
                        size_t nalStart = i - startCodeSize;
                        size_t nalLength = nextStart - nalStart;

                        codecPrivate_.insert(codecPrivate_.end(),
                                            nalData.begin() + nalStart,
                                            nalData.begin() + nalStart + nalLength);

                        std::cout << "[MVCSSIFDemuxer] Extracted PPS (NAL type 8) for codec private - "
                                  << nalLength << " bytes" << std::endl;
                        hasPPS_ = true;
                    }

                    // Only mark codec private complete when we have BOTH SPS and PPS
                    if (hasSPS_ && hasPPS_ && !hasCodecPrivate_) {
                        hasCodecPrivate_ = true;
                        std::cout << "[MVCSSIFDemuxer] Codec private complete (SPS + PPS): "
                                  << codecPrivate_.size() << " bytes" << std::endl;
                        return;  // Done, no need to parse more
                    }

                    i = nextStart;
                    continue;
                }
            }
        }
        i++;
    }
}

bool MVCSSIFDemuxer::synchronizeFrames(FramePair& framePair) {
    // Check if we have frames from both streams with matching PTS
    if (!pendingBase_.hasData || !pendingDependent_.hasData) {
        return false;
    }

    // For SSIF, frames should be pre-synchronized by the interleaving
    // but we can check PTS to be sure
    const int64_t PTS_TOLERANCE = 3003; // ~1 frame at 29.97fps (90kHz timebase)

    int64_t ptsDiff = std::abs(pendingBase_.pts - pendingDependent_.pts);

    // SOL 5C: Verify sync and warn if streams desync
    int64_t base_frame_ts = pendingBase_.pts / 90;  // Convert to ms
    int64_t dependent_frame_ts = pendingDependent_.pts / 90;
    int64_t delta_ms = std::abs(base_frame_ts - dependent_frame_ts);

    if (delta_ms > 100) {  // 100ms tolerance
        std::cerr << "[SSIF] Streams desync detected: " << delta_ms << "ms delta" << std::endl;
        // Note: Cannot resync here without M2TSReader::seek() being timestamp-aware
        // This warning helps diagnose SSIF playback issues
    }

    if (ptsDiff > PTS_TOLERANCE) {
        // Frames not synchronized - skip the older one
        if (pendingBase_.pts < pendingDependent_.pts) {
            std::cout << "[MVCSSIFDemuxer] PTS mismatch - skipping base frame (diff: "
                      << ptsDiff << ")" << std::endl;
            pendingBase_.hasData = false;
        } else {
            std::cout << "[MVCSSIFDemuxer] PTS mismatch - skipping dependent frame (diff: "
                      << ptsDiff << ")" << std::endl;
            pendingDependent_.hasData = false;
        }
        return false;
    }

    // Build frame pair
    framePair.baseData = std::move(pendingBase_.baseView);
    framePair.dependentData = std::move(pendingDependent_.baseView);

    // PTS NORMALIZATION: Capture first valid PTS as offset (dual-file mode)
    if (!basePtsInitialized_) {
        basePtsOffset_ = pendingBase_.pts;
        basePtsInitialized_ = true;
        fprintf(stderr, "[SSIF DUAL] PTS normalization: first PTS=%lld (%.3fs), using as offset\n",
                (long long)basePtsOffset_, basePtsOffset_ / 90000.0);
    }
    int64_t normalizedPts = pendingBase_.pts - basePtsOffset_;
    framePair.timestamp = normalizedPts / 90; // Convert 90kHz to milliseconds
    framePair.isKeyframe = false; // TODO: detect keyframes

    // Check for IDR in base view
    for (size_t i = 0; i + 4 < framePair.baseData.size(); i++) {
        if (framePair.baseData[i] == 0 && framePair.baseData[i+1] == 0 &&
            (framePair.baseData[i+2] == 1 ||
             (framePair.baseData[i+2] == 0 && framePair.baseData[i+3] == 1))) {
            size_t nalStart = (framePair.baseData[i+2] == 1) ? i + 3 : i + 4;
            if (nalStart < framePair.baseData.size()) {
                uint8_t nalType = framePair.baseData[nalStart] & 0x1F;
                if (nalType == 5) {  // IDR slice
                    framePair.isKeyframe = true;
                    break;
                }
            }
        }
    }

    // Clear pending frames
    pendingBase_.hasData = false;
    pendingDependent_.hasData = false;

    return true;
}

bool MVCSSIFDemuxer::seek(int64_t timestampMs) {
    // SOL 5B: Synchronize seek on both base AND dependent streams
    if (!isOpen_) {
        return false;
    }

    // Seek base stream
    if (!baseReader_->seek(timestampMs)) {
        std::cerr << "[SSIF] Base stream seek failed at " << timestampMs << "ms" << std::endl;
        return false;
    }
    base_stream_pos_ = timestampMs;
    std::cout << "[SSIF] Base stream seeked to " << timestampMs << "ms" << std::endl;

    // CRITICAL: Seek dependent stream to THE SAME position for sync
    if (dependentReader_ && !dependentReader_->seek(timestampMs)) {
        std::cerr << "[SSIF] Dependent stream seek failed, may desync" << std::endl;
        // Continue anyway, better than full failure
    } else {
        dependent_stream_pos_ = timestampMs;
        std::cout << "[SSIF] Dependent stream seeked to " << timestampMs << "ms" << std::endl;
    }

    // Clear pending frames to avoid stale data
    pendingBase_.hasData = false;
    pendingDependent_.hasData = false;

    return true;
}

} // namespace mvc_demux
