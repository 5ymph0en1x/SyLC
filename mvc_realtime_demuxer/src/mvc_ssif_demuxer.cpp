#include "mvc_ssif_demuxer.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>  // for std::abs
#include <chrono>   // DIAG: slow-read timing

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
    // Review fix DF-2 (finding 2): dualFileMode_ must not leak from a prior openDual() call on
    // a reused object. open() always (re)establishes SSIF/streaming mode, so force it false
    // here at the top before anything else runs; the branch below may still set it back to
    // true only via the historical (disabled) streaming-mode dual-file attempt path.
    dualFileMode_ = false;
    basePesFlushCount_ = 0;
    mvcPesFlushCount_ = 0;

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
        bool baseOpened = false;
        if (!info.baseStreamPath.empty()) {
            if (baseReader_->open(info.baseStreamPath)) {
                baseOpened = true;
                std::cout << "[MVCSSIFDemuxer] Opened base stream (for audio/fallback)" << std::endl;
            }
        }
        // A dual-file mode (separate 00001.m2ts + 00002.m2ts) was implemented but is DISABLED:
        // the standalone dependent .m2ts has corrupt/non-monotonic PTS in places. The SSIF
        // itself interleaves both views with CLEAN, consistent PTS everywhere (verified), so we
        // stream from it and seek by the base PID's clean PTS. (The base+dep PTS pair within the
        // interleave lead, which the read loop re-aligns after a seek.)
        (void)baseOpened;
        dualFileMode_ = false;
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
                if (parsePESHeader(state.buffer, pts, dts, headerLength) &&
                    headerLength <= state.buffer.size()) {   // defensive clamp: truncated/corrupt PES
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
    if (dualFileMode_) {
        // Dual-file: rewind base (used for codec_private probe) + dependent; free the SSIF handle.
        baseReader_->seek(0);
        dependentReader_->seek(0);
        if (ssifReader_) ssifReader_->close();
        std::cout << "[MVCSSIFDemuxer] Dual-file readers reset to start (SSIF handle released)" << std::endl;
    } else if (isStreamingMode_) {
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

bool MVCSSIFDemuxer::openDual(const std::string& basePath, const std::string& depPath) {
    // DUAL-SOURCE MODE (DF-2): base view and dependent view live in SEPARATE .m2ts files
    // (MakeMKV backup, no interleaved .ssif). This is a NEW entry point that reuses the
    // existing, battle-tested dual-file machinery (readNextFramePairDualFile / tryMatchFramePair
    // / findByteForPts / seek dualFileMode_ branch) which was implemented but previously
    // unreachable (dualFileMode_ was only ever left false). It bypasses the SSIF parser.
    std::cout << "[SSIF-DUAL] openDual: base=" << basePath << ", dep=" << depPath << std::endl;

    // Fresh state (openDual may be called on a reused object, but tests use a fresh one).
    close();
    hasCodecPrivate_ = false;
    hasSPS_ = false;
    hasPPS_ = false;
    codecPrivate_.clear();

    // Open both independent readers.
    if (!baseReader_->open(basePath)) {
        std::cerr << "[SSIF-DUAL] Failed to open base file: " << basePath << std::endl;
        return false;
    }
    if (!dependentReader_->open(depPath)) {
        std::cerr << "[SSIF-DUAL] Failed to open dependent file: " << depPath << std::endl;
        baseReader_->close();
        return false;
    }

    isStreamingMode_ = false;
    dualFileMode_ = true;

    M2TSReader::TSPacket packet;

    // Detect the base video PID from the base file's PMT (H.264 stream_type 0x1B).
    uint16_t basePid = 0;
    for (int i = 0; i < 20000 && baseReader_->readPacket(packet); i++) {
        for (const auto& prog : baseReader_->getPrograms()) {
            for (const auto& kv : prog.streamPids) {
                if (kv.second == 0x1B && basePid == 0) basePid = kv.first;  // H.264 (AVC)
            }
        }
        if (basePid != 0) break;
    }
    if (basePid == 0) {
        auto vids = baseReader_->getVideoPids();
        if (!vids.empty()) basePid = *std::min_element(vids.begin(), vids.end());
    }

    // Detect the dependent video PID from the dependent file's PMT. A MakeMKV-split
    // dependent .m2ts lists the view as 0x1B (AVC) or 0x20 (MVC).
    uint16_t mvcPid = 0;
    for (int i = 0; i < 20000 && dependentReader_->readPacket(packet); i++) {
        for (const auto& prog : dependentReader_->getPrograms()) {
            for (const auto& kv : prog.streamPids) {
                if ((kv.second == 0x1B || kv.second == 0x20) && mvcPid == 0) mvcPid = kv.first;
            }
        }
        if (mvcPid != 0) break;
    }
    if (mvcPid == 0) {
        auto vids = dependentReader_->getVideoPids();
        if (!vids.empty()) mvcPid = *std::min_element(vids.begin(), vids.end());
    }
    if (mvcPid == 0 && basePid != 0) mvcPid = basePid + 1;  // BD convention (dep = base+1)

    if (basePid == 0) {
        std::cerr << "[SSIF-DUAL] Failed to detect base video PID" << std::endl;
        close();
        return false;
    }

    videoInfo_.width = 1920;
    videoInfo_.height = 1080;
    videoInfo_.fps = 23.976;
    videoInfo_.hasMVC = true;
    videoInfo_.baseVideoPid = basePid;
    videoInfo_.mvcVideoPid = mvcPid;
    hasVideoInfo_ = true;
    std::cout << "[SSIF-DUAL] PIDs: base=0x" << std::hex << basePid
              << " dep=0x" << mvcPid << std::dec << std::endl;

    // Extract codec_private (SPS + PPS) from the BASE file. Same routine the SSIF path uses.
    baseReader_->seek(0);
    basePesStates_.clear();
    for (int i = 0; i < 200000 && baseReader_->readPacket(packet) && !hasCodecPrivate_; i++) {
        if (packet.pid == basePid) processVideoPacket(packet, true);
    }
    if (!hasCodecPrivate_) {
        // Non-fatal: the base .m2ts carries inline SPS/PPS in each IDR AU (passed through raw),
        // so playback works without a separate codec_private. Warn and continue (unlike the SSIF
        // open() which hard-fails) since get_codec_private() is optional for the dual-file path.
        std::cerr << "[SSIF-DUAL] WARNING: SPS/PPS not captured in probe window "
                  << "(continuing; base AUs carry inline parameter sets)" << std::endl;
    }

    // Reset BOTH readers to byte 0 and clear ALL reassembly/pairing/subtitle state so the
    // read loop resyncs cleanly from the start (same discipline as SSIF open()).
    baseReader_->seek(0);
    dependentReader_->seek(0);
    basePesStates_.clear();
    dependentPesStates_.clear();
    streamingPesState_.clear();
    baseFrameBuffer_.clear();
    dependentFrameBuffer_.clear();
    frameQueue_.clear();
    pendingBase_.baseView.clear();      pendingBase_.dependentView.clear();
    pendingBase_.hasData = false;       pendingBase_.pts = 0;  pendingBase_.alreadyPrefixed = false;
    pendingDependent_.baseView.clear(); pendingDependent_.dependentView.clear();
    pendingDependent_.hasData = false;  pendingDependent_.pts = 0; pendingDependent_.alreadyPrefixed = false;
    basePtsOffset_ = 0;
    basePtsInitialized_ = false;
    currentExtentIndex_ = 0;
    totalFramePairs_ = 0;
    depReseekBudget_ = 0;
    basePesFlushCount_ = 0;
    mvcPesFlushCount_ = 0;
    selectedSubtitlePid_ = 0;
    subtitlePesStates_.clear();
    subtitleQueue_.clear();
    abortRequested_.store(false, std::memory_order_relaxed);

    isOpen_ = true;
    std::cout << "[SSIF-DUAL] openDual complete (dual-file mode, "
              << (hasCodecPrivate_ ? "codec_private ready" : "inline params") << ")" << std::endl;
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
    // Review fix DF-2 (finding 2): dualFileMode_ leaked across object reuse — neither close()
    // nor open()'s non-streaming branch cleared it, so a demuxer instance that had been used
    // for openDual() and then reopened via open() (SSIF path) would still take the dual-file
    // read/seek branches against SSIF-mode readers. Clear it here unconditionally.
    dualFileMode_ = false;
    // Review fix DF-2 (finding 3): the PES-flush counters were function-static in
    // processVideoPacket() (shared by ALL instances/reuses); now instance members, reset
    // consistently with the other per-open counters (totalFramePairs_, depReseekBudget_).
    basePesFlushCount_ = 0;
    mvcPesFlushCount_ = 0;
}

bool MVCSSIFDemuxer::readNextFramePair(FramePair& framePair) {
    if (!isOpen_) {
        return false;
    }

    if (dualFileMode_) {
        // DUAL-FILE MODE: base from 00001.m2ts, dependent from 00002.m2ts (no interleave gaps)
        return readNextFramePairDualFile(framePair);
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

// Shared base/dependent buffer matcher. Emits the FIFO front of baseFrameBuffer_ (DECODE
// order — see corruption-fix note below) paired with its dependent view by PTS.
//   allowDropBase: streaming mode drops an unmatchable base front (dependent genuinely
//   absent in the interleave window); dual-file mode never drops (dependent always exists,
//   it just needs the readers to advance), so it passes false.
bool MVCSSIFDemuxer::tryMatchFramePair(FramePair& framePair, bool allowDropBase) {
    if (baseFrameBuffer_.empty() || dependentFrameBuffer_.empty()) {
        return false;
    }

    // CRITICAL (corruption fix): emit base frames in DECODE order — the FIFO order they
    // arrive in the stream (= DTS order) — NOT sorted by PTS. H.264 Blu-ray uses B-frames,
    // so presentation (PTS) order != decode order; feeding the decoder in PTS order makes it
    // decode B/P frames before their reference frames -> green frames + macroblock garbage.
    // The base PID arrives in decode order, so the FRONT of baseFrameBuffer_ is the next frame
    // to decode. Pair it with its dependent view by PTS (same temporal instant). edge264 then
    // performs presentation reordering itself (DPB), exactly as on the working MKV path.
    BufferedFrame& front = baseFrameBuffer_.front();
    int64_t basePts = front.pts;

    int bestDi = -1;
    int64_t bestDiff = PTS_MATCH_TOLERANCE + 1;
    for (size_t di = 0; di < dependentFrameBuffer_.size(); di++) {
        int64_t diff = std::abs(basePts - dependentFrameBuffer_[di].pts);
        if (diff <= PTS_MATCH_TOLERANCE && diff < bestDiff) {
            bestDiff = diff;
            bestDi = static_cast<int>(di);
        }
    }

    if (bestDi < 0) {
        if (allowDropBase && dependentFrameBuffer_.size() >= MAX_FRAME_BUFFER_SIZE) {
            baseFrameBuffer_.erase(baseFrameBuffer_.begin());
        }
        return false;
    }

    // PTS NORMALIZATION: Blu-ray streams often start at non-zero PTS; capture the first PTS
    // as offset so display timestamps start from 0. (PTS, not decode order, drives display.)
    if (!basePtsInitialized_) {
        basePtsOffset_ = basePts;
        basePtsInitialized_ = true;
        fprintf(stderr, "[SSIF] PTS normalization: first PTS=%lld (%.3fs), using as offset\n",
                (long long)basePtsOffset_, basePtsOffset_ / 90000.0);
    }

    // Review fix DF-2 (finding 1): capture the dependent frame's OWN PTS before it's moved out,
    // so framePair.depTimestamp reflects its real source timestamp instead of being left equal
    // to the base timestamp (which made any base<->dep delta assertion structurally vacuous —
    // it compared a value to itself). Same normalization offset as the base so both timestamps
    // stay on one shared zero-based timeline.
    int64_t depPts = dependentFrameBuffer_[bestDi].pts;

    framePair.baseData = std::move(front.data);
    framePair.dependentData = std::move(dependentFrameBuffer_[bestDi].data);
    framePair.timestamp = (basePts - basePtsOffset_) / 90;  // 90kHz -> ms (presentation ts)
    framePair.depTimestamp = (depPts - basePtsOffset_) / 90;  // dep's own PTS, same timeline
    framePair.isKeyframe = front.isKeyframe;

    baseFrameBuffer_.erase(baseFrameBuffer_.begin());
    dependentFrameBuffer_.erase(dependentFrameBuffer_.begin() + bestDi);

    totalFramePairs_++;
    if (totalFramePairs_ == 1 || totalFramePairs_ % 500 == 0) {
        fprintf(stderr, "[SSIF] Frame #%d: ts=%lld ms, base=%zu, dep=%zu%s\n",
                totalFramePairs_, (long long)framePair.timestamp,
                framePair.baseData.size(), framePair.dependentData.size(),
                framePair.isKeyframe ? " [IDR]" : "");
    }
    return true;
}

// Move a freshly-assembled pending base frame into baseFrameBuffer_ (with IDR detection).
void MVCSSIFDemuxer::pushPendingBaseFrame() {
    if (!pendingBase_.hasData) return;
    BufferedFrame bf;
    bf.data = std::move(pendingBase_.baseView);
    bf.pts = pendingBase_.pts;
    bf.isKeyframe = false;
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
    while (baseFrameBuffer_.size() > MAX_FRAME_BUFFER_SIZE) {
        baseFrameBuffer_.erase(baseFrameBuffer_.begin());
    }
}

// Move a freshly-assembled pending dependent frame into dependentFrameBuffer_.
void MVCSSIFDemuxer::pushPendingDependentFrame() {
    if (!pendingDependent_.hasData) return;
    BufferedFrame bf;
    bf.data = std::move(pendingDependent_.baseView);
    bf.pts = pendingDependent_.pts;
    bf.isKeyframe = false;
    dependentFrameBuffer_.push_back(std::move(bf));
    pendingDependent_.hasData = false;
    pendingDependent_.baseView.clear();
    while (dependentFrameBuffer_.size() > MAX_FRAME_BUFFER_SIZE) {
        dependentFrameBuffer_.erase(dependentFrameBuffer_.begin());
    }
}

bool MVCSSIFDemuxer::readNextFramePairStreaming(FramePair& framePair) {
    // STREAMING MODE: Read from interleaved SSIF file (fallback when the separate
    // dependent .m2ts is unavailable). Both PIDs arrive interleaved; buffer + match by PTS.
    M2TSReader::TSPacket packet;
    int maxPackets = 500000;
    int packetsRead = 0;

    // DIAG: detect slow reads (the GUI post-seek freeze). Logs packets/resyncs/elapsed.
    auto _t0 = std::chrono::steady_clock::now();
    long _r0 = ssifReader_ ? ssifReader_->getResyncCount() : 0;
    auto _logslow = [&](const char* where) {
        double ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - _t0).count();
        if (ms > 2000.0)
            fprintf(stderr, "[SSIF-SLOW] %s packets=%d resyncs=%ld elapsed=%.1fs baseBuf=%zu depBuf=%zu\n",
                    where, packetsRead,
                    (ssifReader_ ? ssifReader_->getResyncCount() : 0) - _r0,
                    ms / 1000.0, baseFrameBuffer_.size(), dependentFrameBuffer_.size());
    };

    uint16_t basePid = videoInfo_.baseVideoPid;
    uint16_t mvcPid = videoInfo_.mvcVideoPid;
    if (mvcPid == 0 && basePid != 0) {
        mvcPid = basePid + 1;
    }

    if (tryMatchFramePair(framePair, /*allowDropBase=*/false)) {
        return true;
    }

    bool readSuccess = true;
    while (packetsRead < maxPackets && (readSuccess = ssifReader_->readPacket(packet))) {
        packetsRead++;
        // Cooperative abort: a newer seek superseded this scan, or the thread is stopping.
        // Bail out so a long cold/contended dep-extent read can never pin the decoder thread
        // past the GUI watchdog (which would force-terminate). Checked cheaply (atomic, every
        // 256 packets). Returning false ends the scan; the caller handles it as no-IDR/stop.
        if ((packetsRead & 255) == 0 &&
            abortRequested_.load(std::memory_order_relaxed)) {
            fprintf(stderr, "[SSIF-ABORT] read aborted after %d packets (superseded/stopping)\n",
                    packetsRead);
            return false;
        }
        if (packet.pid == basePid) {
            processVideoPacket(packet, true);
            pushPendingBaseFrame();
        } else if (packet.pid == mvcPid) {
            processVideoPacket(packet, false);
            pushPendingDependentFrame();
        } else if (selectedSubtitlePid_ != 0 &&
                   packet.pid == static_cast<uint16_t>(selectedSubtitlePid_)) {
            collectSubtitlePacket(packet);
        }

        if (tryMatchFramePair(framePair, /*allowDropBase=*/false)) {
            _logslow("inloop-pair");
            return true;
        }
        // Catch-up to the interleave lead. The dependent leads the base by ~1-3s, so after a
        // mid-stream seek the base front's dependent view is BEHIND the landing (already passed)
        // and the base must advance to where the dependent buffer starts. If we just let both
        // buffers slide forward together the lead is preserved and pairing never re-establishes
        // (the post-seek failure). So: whenever the base front is behind the entire dependent
        // buffer, drop it — advancing the base toward the dependent's range until they overlap.
        if (dependentFrameBuffer_.size() >= 64 && !baseFrameBuffer_.empty()) {
            int64_t bf = baseFrameBuffer_.front().pts;
            int64_t dmin = dependentFrameBuffer_.front().pts;
            for (const auto& dd : dependentFrameBuffer_) dmin = std::min(dmin, dd.pts);
            if (bf < dmin - PTS_MATCH_TOLERANCE) {
                baseFrameBuffer_.erase(baseFrameBuffer_.begin());
            }
        }
    }

    if (!readSuccess) {
        fprintf(stderr, "[SSIF-BUFFER] EOF reached: baseBuffer=%zu, depBuffer=%zu\n",
                baseFrameBuffer_.size(), dependentFrameBuffer_.size());
    } else {
        int64_t bf = baseFrameBuffer_.empty() ? -90000 : baseFrameBuffer_.front().pts;
        int64_t dmn = -90000, dmx = -90000;
        if (!dependentFrameBuffer_.empty()) {
            dmn = dmx = dependentFrameBuffer_.front().pts;
            for (const auto& x : dependentFrameBuffer_) { dmn = std::min(dmn, x.pts); dmx = std::max(dmx, x.pts); }
        }
        fprintf(stderr, "[SSIF-STRM] maxPackets no-pair: baseFront=%.2fs depRange=%.2f..%.2fs baseBuf=%zu depBuf=%zu\n",
                bf / 90000.0, dmn / 90000.0, dmx / 90000.0,
                baseFrameBuffer_.size(), dependentFrameBuffer_.size());
    }
    _logslow("no-pair");
    return false;
}

bool MVCSSIFDemuxer::readNextFramePairDualFile(FramePair& framePair) {
    // DUAL-FILE MODE: base frames from baseReader_ (00001.m2ts), dependent frames from
    // dependentReader_ (00002.m2ts). Each file is one contiguous view, so we can advance
    // each independently to keep base+dependent aligned by PTS — no SSIF interleave gaps.

    // Cooperative abort (DF-2): a newer seek superseded this read, or the thread is stopping.
    // Bail immediately so a mid-read abort is a prompt clean stop (never a hang). read_next_*
    // releases the GIL, so requestAbort() from another thread is observed here.
    if (abortRequested_.load(std::memory_order_relaxed)) {
        return false;
    }

    uint16_t basePid = videoInfo_.baseVideoPid;
    uint16_t mvcPid = videoInfo_.mvcVideoPid;
    if (mvcPid == 0 && basePid != 0) {
        mvcPid = basePid + 1;
    }

    if (tryMatchFramePair(framePair, /*allowDropBase=*/false)) {
        return true;
    }

    M2TSReader::TSPacket packet;
    bool baseEof = false, depEof = false;
    const long MAX_ITERS = 1500000;  // ~288MB/file bound: a corrupt region fails fast, no hang
    long iters = 0;

    while (iters++ < MAX_ITERS) {
        // Cooperative abort inside the scan loop (checked cheaply, atomic, every 256 iters) so a
        // long unmatched scan on a cold/contended disc can never pin the decoder thread past the
        // GUI watchdog. Returning false ends the read; the caller treats it as no-pair/stop.
        if ((iters & 255) == 0 && abortRequested_.load(std::memory_order_relaxed)) {
            fprintf(stderr, "[SSIF-DUAL-ABORT] read aborted after %ld iters (superseded/stopping)\n", iters);
            return false;
        }
        bool progressed = false;

        // Keep base buffer populated; also harvest PGS subtitle packets from the base m2ts.
        if (!baseEof && baseFrameBuffer_.size() < MAX_FRAME_BUFFER_SIZE) {
            if (baseReader_->readPacket(packet)) {
                progressed = true;
                if (packet.pid == basePid) {
                    processVideoPacket(packet, true);
                    pushPendingBaseFrame();
                } else if (selectedSubtitlePid_ != 0 &&
                           packet.pid == static_cast<uint16_t>(selectedSubtitlePid_)) {
                    collectSubtitlePacket(packet);
                }
            } else {
                baseEof = true;
            }
        }

        // Keep dependent buffer populated.
        if (!depEof && dependentFrameBuffer_.size() < MAX_FRAME_BUFFER_SIZE) {
            if (dependentReader_->readPacket(packet)) {
                progressed = true;
                if (packet.pid == mvcPid) {
                    processVideoPacket(packet, false);
                    pushPendingDependentFrame();
                }
            } else {
                depEof = true;
            }
        }

        if (tryMatchFramePair(framePair, /*allowDropBase=*/false)) {
            return true;
        }

        // Post-seek alignment: if the dependent overshot the base (its nearest sane buffered PTS
        // is past the base front), re-seek the dependent to the base front's EXACT PTS via binary
        // search. Targeting the absolute base PTS (not a relative offset) is STABLE — it converges
        // in one step where the dependent PTS is clean, instead of oscillating.
        const int64_t SANE_WIN = 300LL * 90000;   // 300s window ignores corrupt/garbage dep PTS
        if (depReseekBudget_ > 0 && !baseFrameBuffer_.empty() && dependentFrameBuffer_.size() >= 16) {
            int64_t baseFront = baseFrameBuffer_.front().pts;
            int64_t nearest = SANE_WIN + 1;
            for (const auto& dprev : dependentFrameBuffer_) {
                int64_t off = dprev.pts - baseFront;
                if (std::llabs(off) < std::llabs(nearest)) nearest = off;
            }
            if (nearest > PTS_MATCH_TOLERANCE && nearest <= SANE_WIN) {
                uint64_t db = findByteForPts(dependentReader_.get(), mvcPid, baseFront);
                dependentReader_->seek(db);
                dependentFrameBuffer_.clear();
                dependentPesStates_.clear();
                pendingDependent_.hasData = false;
                pendingDependent_.baseView.clear();
                depReseekBudget_--;
                depEof = false;
                continue;
            }
        }

        // Progress guarantee when both buffers are full and unmatched: drop dependent frames that
        // cannot match the base front (older than it, or garbage-far), freeing slots so the
        // dependent advances. If none are droppable, skip the base front to keep moving.
        if (baseFrameBuffer_.size() >= MAX_FRAME_BUFFER_SIZE &&
            dependentFrameBuffer_.size() >= MAX_FRAME_BUFFER_SIZE) {
            int64_t baseFront = baseFrameBuffer_.front().pts;
            size_t before = dependentFrameBuffer_.size();
            for (size_t i = 0; i < dependentFrameBuffer_.size(); ) {
                int64_t off = dependentFrameBuffer_[i].pts - baseFront;
                if (off < -PTS_MATCH_TOLERANCE || off > SANE_WIN) {
                    dependentFrameBuffer_.erase(dependentFrameBuffer_.begin() + i);
                } else {
                    ++i;
                }
            }
            if (dependentFrameBuffer_.size() < before) {
                progressed = true;   // freed dependent slots -> can read more
            } else {
                baseFrameBuffer_.erase(baseFrameBuffer_.begin());  // base front unmatchable -> skip
                progressed = true;
            }
        }

        if (baseEof && depEof) break;
        if (!progressed) break;
    }

    if (!baseFrameBuffer_.empty() && !dependentFrameBuffer_.empty()) {
        fprintf(stderr, "[SSIF-DUAL] no pair: baseFront=%.3fs depRange=%.3f..%.3fs (baseBuf=%zu depBuf=%zu eof b=%d d=%d)\n",
                baseFrameBuffer_.front().pts / 90000.0,
                dependentFrameBuffer_.front().pts / 90000.0,
                dependentFrameBuffer_.back().pts / 90000.0,
                baseFrameBuffer_.size(), dependentFrameBuffer_.size(), (int)baseEof, (int)depEof);
    } else {
        fprintf(stderr, "[SSIF-DUAL] no pair (baseEof=%d depEof=%d baseBuf=%zu depBuf=%zu)\n",
                (int)baseEof, (int)depEof, baseFrameBuffer_.size(), dependentFrameBuffer_.size());
    }
    return false;
}

// Binary-search a contiguous M2TS for the byte whose (pid) PTS is at/just-before targetPts90k.
uint64_t MVCSSIFDemuxer::findByteForPts(M2TSReader* reader, uint16_t pid, int64_t targetPts90k, uint64_t hintByte) {
    if (!reader) return 0;
    int ps = reader->getPacketSize();
    if (ps <= 0) ps = 192;
    uint64_t lo = 0, hi = reader->getFileSize();
    if (hi < (uint64_t)ps * 2) return 0;
    uint64_t result = 0;
    // Coarse convergence (~2MB ≈ <1s of video): the decoder scans forward to the next IDR,
    // so we don't need byte precision.
    // 16 MB convergence (was 2 MB): on a COLD optical drive each probe is a ~2-4 s head-seek,
    // so refining to 2 MB cost ~5 probes ≈ 20 s and blew past the seek timeout → crash. The
    // forward IDR-scan reads the remaining ≤16 MB sequentially (cheap vs another cold head-seek).
    const uint64_t COARSE = 16 * 1024 * 1024;
    M2TSReader::TSPacket packet;
    int probes = 0;
    auto _t0 = std::chrono::steady_clock::now();

    // Bound the fallback (no-CLPI) search so a corrupt or missing-PID region can't turn a seek
    // into a multi-second "freeze" (Tokyo report #1/#5). Two guards:
    //  - per-probe packet cap: it must EXCEED the largest base-PID gap, which in the interleaved
    //    SSIF is an entire DEPENDENT extent (~tens of thousands of packets). The report's proposed
    //    1000-2000 would miss the base PTS that sits just past a dependent extent and break the
    //    search; 64000 (~12 MB) is the safe floor -- still ~3x cheaper worst case than the old
    //    200000 cap (~38 MB), and unused on the fast path (the exact CLPI map bypasses this).
    //  - wall-clock budget: abort runaway probing and return the best estimate found so far.
    constexpr int kMaxProbeScanPackets = 64000;
    constexpr double kSeekBudgetMs = 4000.0;
    auto elapsedMs = [&]() {
        return (double)std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - _t0).count();
    };
    bool budgetHit = false;

    // Read the first base-PID PTS at/after byte b (packet-aligned). -1 if none found ahead.
    auto ptsAt = [&](uint64_t b) -> int64_t {
        probes++;
        reader->seek((b / ps) * ps);
        for (int k = 0; k < kMaxProbeScanPackets && reader->readPacket(packet); k++) {
            if ((k & 511) == 0 && abortRequested_.load(std::memory_order_relaxed)) return -1;
            if (packet.pid == pid && packet.payloadUnitStartIndicator &&
                packet.payload.size() >= 14) {
                int64_t pts = 0, dts = 0; size_t hl = 0;
                if (parsePESHeader(packet.payload, pts, dts, hl) && pts > 0) return pts;
            }
        }
        return -1;
    };

    // REGRESSION FIX (cold-optical seek freeze): the old code BISECTED the whole 45 GB file —
    // ~13 iterations, each a head-seek to a far, COLD spot of the disc. Warm that's 0.01 s, but
    // on the real Blu-ray drive ~13 scattered cold seeks cost ~6 s per seek (no single read
    // >2 s, so it never tripped [SSIF-SLOW]). PTS<->byte is monotonic & ~linear, so an
    // INTERPOLATION search converges in ~3-4 probes. We seed the PTS endpoints from the known
    // first-PTS offset + duration (NO extra disc seeks), so the FIRST probe lands near target.
    int64_t loPts = basePtsInitialized_ ? basePtsOffset_ : ptsAt(lo);
    if (loPts < 0) return 0;
    if (targetPts90k <= loPts) return 0;             // target at/before start -> byte 0
    int64_t hiPts = (externalDurationMs_ > 0)
        ? loPts + externalDurationMs_ * 90           // estimate end PTS — avoids a cold end-seek
        : ptsAt(hi > 8u * 1024 * 1024 ? hi - 8u * 1024 * 1024 : 0);
    if (hiPts <= loPts) hiPts = loPts + 1;
    if (targetPts90k >= hiPts) return ((hi - COARSE) / ps) * ps;  // target near/after end

    // The bracket [lo,hi] always narrows (mid is kept strictly inside), so this is bounded by
    // the 40-iter cap even if the data were non-linear.
    for (int iter = 0; iter < 40 && lo + COARSE < hi; iter++) {
        if (abortRequested_.load(std::memory_order_relaxed)) return result;
        if (elapsedMs() > kSeekBudgetMs) { budgetHit = true; break; }
        uint64_t mid;
        if (iter == 0 && hintByte > lo && hintByte < hi) {
            mid = (hintByte / ps) * ps;   // EP_map-ratio seed: first probe lands near target
        } else {
            double frac = (hiPts > loPts) ? double(targetPts90k - loPts) / double(hiPts - loPts) : 0.5;
            if (frac < 0.0) frac = 0.0; else if (frac > 1.0) frac = 1.0;
            mid = lo + (uint64_t)(frac * double(hi - lo));
            mid = (mid / ps) * ps;
        }
        if (mid <= lo) mid = lo + (uint64_t)ps;
        if (mid + (uint64_t)ps >= hi) mid = hi - (uint64_t)ps;
        if (mid <= lo || mid >= hi) break;

        int64_t midPts = ptsAt(mid);
        if (midPts < 0) { hi = mid; continue; }       // no PTS ahead -> earlier
        if (midPts < targetPts90k) { lo = mid; loPts = midPts; result = mid; }  // before -> later
        else { hi = mid; hiPts = midPts; }            // at/after -> earlier
    }
    // FALLBACK (Tokyo report #3): if NO probe ever landed below the target, `result` is still 0.
    // Returning 0 would seek to the START of the film, and the Python IDR-scan would then read
    // GBs forward to a distant target -> freeze. Only a genuine start-target should map to 0
    // (handled by the early `targetPts90k <= loPts` return). So a 0 here for a non-start target
    // is an interpolation failure -> recover with a robust bounded BISECTION over the whole file.
    if (result == 0 && targetPts90k > loPts && !budgetHit) {
        uint64_t blo = 0, bhi = reader->getFileSize();
        for (int iter = 0; iter < 40 && blo + COARSE < bhi; iter++) {
            if (elapsedMs() > kSeekBudgetMs) { budgetHit = true; break; }
            uint64_t mid = (((blo + bhi) / 2) / ps) * ps;
            if (mid <= blo) break;
            int64_t midPts = ptsAt(mid);
            if (midPts < 0) bhi = mid;
            else if (midPts < targetPts90k) { blo = mid; result = mid; }
            else bhi = mid;
        }
    }
    double sec = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - _t0).count() / 1000.0;
    fprintf(stderr, "[SSIF-SEEK] findByteForPts: %d probes, %.2fs%s -> byte %llu\n",
            probes, sec, budgetHit ? " (budget hit, best estimate)" : "",
            (unsigned long long)result);
    return result;
}

uint64_t MVCSSIFDemuxer::depProportionalByte(int64_t normMs) {
    if (!dependentReader_) return 0;
    uint64_t fs = dependentReader_->getFileSize();
    if (externalDurationMs_ <= 0 || fs == 0) return 0;
    if (normMs < 0) normMs = 0;
    double frac = static_cast<double>(normMs) / static_cast<double>(externalDurationMs_);
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return static_cast<uint64_t>(frac * static_cast<double>(fs));
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

            if (parsePESHeader(state.buffer, pts, dts, headerLength) &&
                headerLength <= state.buffer.size()) {   // defensive clamp (Tokyo report #1)
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
    // Review fix DF-2 (finding 3): basePesFlushCount_/mvcPesFlushCount_ moved to instance
    // members (declared in the header) — they used to be `static` locals here, which are
    // shared by every MVCSSIFDemuxer instance/reuse in the process, not per-object counters.

    if (packet.payloadUnitStartIndicator) {
        // Flush previous PES
        if (state.hasStarted && !state.buffer.empty()) {
            int64_t pts, dts;
            size_t headerLength;

            if (parsePESHeader(state.buffer, pts, dts, headerLength) &&
                headerLength <= state.buffer.size()) {   // defensive clamp (Tokyo report #1)
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
                        basePesFlushCount_++;
                    } else {
                        mvcPesFlushCount_++;
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

    // CRASH FIX (Tokyo report #1): a TRUNCATED PES (common after a seek/resync on a real
    // optical disc — lost packet, false 0x47) gives headerLength > pesData.size(). A consumer
    // then builds std::vector(begin()+headerLength, end()) with begin()+headerLength PAST end()
    // → distance() is negative → size_t ~2^64 → access violation. Reject the truncated PES.
    if (headerLength > pesData.size()) {
        return false;
    }

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
    // Review fix DF-2 (finding 1): stamp the dependent's own PTS here too (legacy
    // extents-based SSIF path — reached only when the SSIF is small enough to skip streaming
    // mode). Same fix/rationale as tryMatchFramePair().
    framePair.depTimestamp = (pendingDependent_.pts - basePtsOffset_) / 90;
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
    if (!isOpen_) {
        return false;
    }
    if (timestampMs < 0) timestampMs = 0;

    // Clear all reassembly/pairing state so we resync cleanly from the new position.
    // Keep basePtsOffset_/basePtsInitialized_ so the normalized timeline stays consistent.
    auto clearState = [&]() {
        streamingPesState_.clear();
        basePesStates_.clear();
        dependentPesStates_.clear();
        baseFrameBuffer_.clear();
        dependentFrameBuffer_.clear();
        frameQueue_.clear();
        pendingBase_.hasData = false;       pendingBase_.baseView.clear();
        pendingDependent_.hasData = false;  pendingDependent_.baseView.clear();
    };

    // Map a timestamp to a packet-aligned byte offset within a reader (proportional).
    auto byteForTimestamp = [&](M2TSReader* r) -> uint64_t {
        if (!r) return 0;
        uint64_t fileSize = r->getFileSize();
        if (externalDurationMs_ <= 0 || fileSize == 0) return 0;
        double frac = static_cast<double>(timestampMs) / static_cast<double>(externalDurationMs_);
        if (frac < 0.0) frac = 0.0;
        if (frac > 1.0) frac = 1.0;
        return static_cast<uint64_t>(frac * static_cast<double>(fileSize));
    };

    if (dualFileMode_) {
        // DUAL-FILE MODE (robust + accurate): seek base and dependent independently to the
        // SAME target PTS. Each file is one contiguous view, so a PTS binary-search lands
        // within a GOP of the target with NO interleave-gap problem. If an EP_map table was
        // supplied (from the .clpi), use it for an exact base-IDR landing and align the
        // dependent to that IDR's PTS.
        uint16_t basePid = videoInfo_.baseVideoPid;
        uint16_t mvcPid = videoInfo_.mvcVideoPid;
        if (mvcPid == 0 && basePid != 0) mvcPid = basePid + 1;

        // Convert the normalized (from-zero) request to a raw stream PTS via the first-frame
        // offset captured during playback.
        int64_t targetPts90k = (int64_t)timestampMs * 90 + basePtsOffset_;
        uint64_t baseByte;
        if (!baseSeekTable_.empty()) {
            // EP_map table holds RAW (pts_ms, byte) straight from the .clpi -> exact IDR landing.
            int64_t reqRawMs = timestampMs + basePtsOffset_ / 90;
            int64_t bestRawMs = baseSeekTable_.front().first;
            uint64_t bestByte = baseSeekTable_.front().second;
            for (const auto& e : baseSeekTable_) {
                if (e.first <= reqRawMs) { bestRawMs = e.first; bestByte = e.second; }
                else break;
            }
            baseByte = bestByte;
            targetPts90k = bestRawMs * 90;  // align dependent to the chosen base IDR (raw)
        } else {
            baseByte = findByteForPts(baseReader_.get(), basePid, targetPts90k);
        }
        // Dependent: binary-search to the same target PTS. Accurate where the dependent PTS is
        // clean (the vast majority of the movie); in the small corrupt-PTS region the read loop
        // bounds the effort and the decoder falls back, rather than hanging or crashing.
        uint64_t depByte = findByteForPts(dependentReader_.get(), mvcPid, targetPts90k);

        bool ok = baseReader_ && baseReader_->seek(baseByte);
        ok = (dependentReader_ && dependentReader_->seek(depByte)) && ok;
        clearState();
        subtitlePesStates_.clear();
        subtitleQueue_.clear();
        depReseekBudget_ = 4;  // allow the dual-file read loop to re-align dep to base
        std::cout << "[SSIF-DUAL] seek " << timestampMs << "ms -> base byte " << baseByte
                  << " (" << baseByte / (1024.0*1024*1024) << "G), dep byte " << depByte
                  << " (" << depByte / (1024.0*1024*1024) << "G) [" << (ok ? "ok" : "FAIL") << "]"
                  << std::endl;
        return ok;
    }

    if (isStreamingMode_) {
        // STREAMING MODE (the real BD3D path): seek by the BASE PID's PTS, which is clean and
        // monotonic in the interleaved SSIF (the dependent view is paired by adjacency during
        // the read, so its own unreliable PTS doesn't matter for positioning). Binary-search
        // the SSIF for the base PTS -> accurate byte (no VBR/proportional error, no >4GB issue).
        // The EP_map table (if provided) snaps the target to a base IDR's PTS.
        uint16_t basePid = videoInfo_.baseVideoPid;
        int64_t targetPts90k = (int64_t)timestampMs * 90 + basePtsOffset_;

        // PREFERRED PATH: exact CLPI Extent-Start-Point map. Jump straight to the byte offset of
        // the interleaved-unit boundary that contains the target IDR (a clean RAPI with BOTH
        // views present from there), so the read loop pairs immediately -- no PTS binary-search,
        // no size-ratio estimate, no cold-disc probing. Validated byte-exact (~3 ms) on real discs.
        if (!ssifSeekTable_.empty()) {
            int64_t reqRawMs = timestampMs + basePtsOffset_ / 90;
            int64_t bestRawMs = ssifSeekTable_.front().first;
            uint64_t bestByte = ssifSeekTable_.front().second;
            for (const auto& e : ssifSeekTable_) {
                if (e.first <= reqRawMs) { bestRawMs = e.first; bestByte = e.second; } else break;
            }
            if (!ssifReader_ || !ssifReader_->seek(bestByte)) {
                std::cerr << "[SSIF] Exact-map seek to byte " << bestByte << " failed" << std::endl;
                return false;
            }
            clearState();
            subtitlePesStates_.clear();
            subtitleQueue_.clear();
            std::cout << "[SSIF] Exact seek " << timestampMs << "ms -> unit byte " << bestByte
                      << " (" << bestByte / (1024.0*1024*1024) << "G @ base IDR PTS "
                      << bestRawMs / 1000.0 << "s)" << std::endl;
            return true;
        }

        // FALLBACK PATH (no extent tables on disc): EP_map size-ratio hint + PTS binary-search.
        uint64_t hintByte = 0;
        if (!baseSeekTable_.empty()) {
            int64_t reqRawMs = timestampMs + basePtsOffset_ / 90;
            int64_t bestRawMs = baseSeekTable_.front().first;
            uint64_t bestByte = baseSeekTable_.front().second;   // EP_map M2TS byte for that IDR
            for (const auto& e : baseSeekTable_) {
                if (e.first <= reqRawMs) { bestRawMs = e.first; bestByte = e.second; } else break;
            }
            targetPts90k = bestRawMs * 90;
            // Seed the SSIF byte from the EP_map's ACCURATE M2TS byte, scaled by the interleave
            // size ratio (ssif = base + dependent). The first probe then lands near-target, so the
            // interpolation needs ~1-2 refinements instead of converging from a coarse guess.
            uint64_t m2tsSize = baseReader_ ? baseReader_->getFileSize() : 0;
            uint64_t ssifSize = ssifReader_ ? ssifReader_->getFileSize() : 0;
            if (m2tsSize > 0 && ssifSize > 0)
                hintByte = (uint64_t)((double)bestByte * (double)ssifSize / (double)m2tsSize);
        }
        uint64_t targetByte = findByteForPts(ssifReader_.get(), basePid, targetPts90k, hintByte);
        if (!ssifReader_ || !ssifReader_->seek(targetByte)) {
            std::cerr << "[SSIF] Streaming seek to byte " << targetByte << " failed" << std::endl;
            return false;
        }
        clearState();
        subtitlePesStates_.clear();
        subtitleQueue_.clear();
        std::cout << "[SSIF] Streaming seek " << timestampMs << "ms -> byte " << targetByte
                  << " (" << targetByte / (1024.0*1024*1024) << "G via base PTS "
                  << targetPts90k / 90000.0 << "s)" << std::endl;
        return true;
    }

    // DUAL-FILE MODE (legacy, rarely used): proportional byte seek on both M2TS readers.
    if (baseReader_) {
        baseReader_->seek(byteForTimestamp(baseReader_.get()));
        base_stream_pos_ = timestampMs;
    }
    if (dependentReader_) {
        dependentReader_->seek(byteForTimestamp(dependentReader_.get()));
        dependent_stream_pos_ = timestampMs;
    }
    clearState();
    return true;
}

void MVCSSIFDemuxer::setBaseSeekTable(const std::vector<int64_t>& ptsMs,
                                      const std::vector<uint64_t>& bytes) {
    baseSeekTable_.clear();
    size_t n = std::min(ptsMs.size(), bytes.size());
    baseSeekTable_.reserve(n);
    for (size_t i = 0; i < n; i++) baseSeekTable_.emplace_back(ptsMs[i], bytes[i]);
    std::sort(baseSeekTable_.begin(), baseSeekTable_.end());
    std::cout << "[SSIF-DUAL] base EP_map seek table set: " << baseSeekTable_.size()
              << " entries" << std::endl;
}

void MVCSSIFDemuxer::setSsifSeekTable(const std::vector<int64_t>& ptsMs,
                                      const std::vector<uint64_t>& ssifBytes) {
    ssifSeekTable_.clear();
    size_t n = std::min(ptsMs.size(), ssifBytes.size());
    ssifSeekTable_.reserve(n);
    for (size_t i = 0; i < n; i++) ssifSeekTable_.emplace_back(ptsMs[i], ssifBytes[i]);
    std::sort(ssifSeekTable_.begin(), ssifSeekTable_.end());
    std::cout << "[SSIF] exact Extent-Start-Point seek map set: " << ssifSeekTable_.size()
              << " entries (byte-exact streaming seek enabled)" << std::endl;
}

std::vector<uint16_t> MVCSSIFDemuxer::getSubtitlePids() const {
    std::vector<uint16_t> pids;
    const M2TSReader* r = baseReader_ ? baseReader_.get() : ssifReader_.get();
    if (!r) return pids;
    for (const auto& prog : r->getPrograms()) {
        for (const auto& kv : prog.streamPids) {
            if (kv.second == 0x90) {  // HDMV PGS subtitle
                if (std::find(pids.begin(), pids.end(), kv.first) == pids.end())
                    pids.push_back(kv.first);
            }
        }
    }
    std::sort(pids.begin(), pids.end());
    return pids;
}

void MVCSSIFDemuxer::setSubtitlePid(int pid) {
    selectedSubtitlePid_ = pid;
    subtitlePesStates_.clear();
    subtitleQueue_.clear();
    std::cout << "[SSIF-DUAL] subtitle PID set to 0x" << std::hex << pid << std::dec << std::endl;
}

bool MVCSSIFDemuxer::readSubtitleBlock(int64_t& timestampMs, std::vector<uint8_t>& data) {
    if (subtitleQueue_.empty()) return false;
    SubtitleBlock& b = subtitleQueue_.front();
    timestampMs = b.timestampMs;
    data = std::move(b.data);
    subtitleQueue_.pop_front();
    return true;
}

void MVCSSIFDemuxer::collectSubtitlePacket(const M2TSReader::TSPacket& packet) {
    // Reassemble one PGS PES and emit its RAW payload (the PGS segment bytes) with the PES PTS.
    // We do NOT parse/wrap segments here: a PGS Object (ODS) can exceed 64KB and SPAN multiple
    // PES, so per-PES segment parsing would cut a segment mid-way and corrupt it (the spurious
    // ~65532-byte blocks + flicker). The Python PGS parser's streaming path accumulates these
    // raw payloads (feed_pes_packet) and reassembles segments across feeds correctly.
    auto& state = subtitlePesStates_[packet.pid];

    if (packet.payloadUnitStartIndicator) {
        if (state.hasStarted && !state.buffer.empty()) {
            int64_t pts = 0, dts = 0; size_t hl = 0;
            if (parsePESHeader(state.buffer, pts, dts, hl) && state.buffer.size() > hl) {
                int64_t normPts = pts - basePtsOffset_;
                if (normPts < 0) normPts = 0;
                SubtitleBlock blk;
                blk.timestampMs = normPts / 90;
                blk.data.assign(state.buffer.begin() + hl, state.buffer.end());  // raw PGS segments
                if (!blk.data.empty()) {
                    subtitleQueue_.push_back(std::move(blk));
                    while (subtitleQueue_.size() > 512) subtitleQueue_.pop_front();
                }
            }
            state.buffer.clear();
        }
        state.hasStarted = true;
    }

    if (state.hasStarted && !packet.payload.empty()) {
        state.buffer.insert(state.buffer.end(), packet.payload.begin(), packet.payload.end());
    }
}

} // namespace mvc_demux
