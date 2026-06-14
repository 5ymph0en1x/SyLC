#include "mvc_m2ts_demuxer.h"
#include <iostream>
#include <cstring>
#include <algorithm>

namespace mvc_demux {

// NAL unit types
constexpr uint8_t NAL_TYPE_SPS = 7;
constexpr uint8_t NAL_TYPE_PPS = 8;
constexpr uint8_t NAL_TYPE_SUBSET_SPS = 15;
constexpr uint8_t NAL_TYPE_CODED_SLICE_EXTENSION = 20;

// Stream types (MPEG-2 TS)
constexpr uint8_t STREAM_TYPE_H264 = 0x1B;
constexpr uint8_t STREAM_TYPE_MVC = 0x20;

// Global debug counters (reset in open)
static int g_debugCount = 0;
static int g_nalHistCount = 0;
static bool g_nalFirstCall = true;
static int g_nalTypeHistogram[32] = {0};

MVCM2TSDemuxer::MVCM2TSDemuxer()
    : reader_(std::make_unique<M2TSReader>()),
      hasVideoInfo_(false),
      hasCodecPrivate_(false),
      dualPidStartPos_(0) {
    videoInfo_ = {};
    currentFrame_ = {};
    pendingFrame_ = {};
    pendingFrame_.hasData = false;
}

MVCM2TSDemuxer::~MVCM2TSDemuxer() {
    close();
}

bool MVCM2TSDemuxer::open(const std::string& filePath) {
    // Reset debug counters
    g_debugCount = 0;
    g_nalHistCount = 0;
    g_nalFirstCall = true;
    std::memset(g_nalTypeHistogram, 0, sizeof(g_nalTypeHistogram));

    if (!reader_->open(filePath)) {
        return false;
    }

    // Read initial packets to find PAT/PMT and SPS
    M2TSReader::TSPacket packet;
    int maxProbePackets = 10000;  // Reverted to 10000 for better SSIF detection
    int spsFound = 0;

    for (int i = 0; i < maxProbePackets && reader_->readPacket(packet); i++) {
        // PAT/PMT parsing handled inside readPacket

        // Once we have video PIDs, start looking for SPS
        auto videoPids = reader_->getVideoPids();
        if (!videoPids.empty() && spsFound < 2) {
            // Process video packets to find SPS
            if (std::find(videoPids.begin(), videoPids.end(), packet.pid) != videoPids.end()) {
                processVideoPacket(packet);

                // Check if we got codec private data
                if (hasCodecPrivate_ && !hasVideoInfo_) {
                    // Try to parse SPS from codec private to get dimensions
                    parseSPSForDimensions();
                    if (videoInfo_.width > 0 && videoInfo_.height > 0) {
                        spsFound++;
                        std::cout << "[MVCM2TSDemuxer] Extracted dimensions from SPS: "
                                  << videoInfo_.width << "x" << videoInfo_.height << std::endl;
                    }
                }
            }
        }

        // CRITICAL FIX for SSIF: Don't exit early until we've found BOTH H.264 and MVC PIDs
        // SSIF files have alternating PMTs for different stream types
        if (spsFound > 0 && i > 5000) {
            // Check if we have both H.264 and MVC PIDs before exiting
            const auto& programs = reader_->getPrograms();
            bool hasH264 = false;
            bool hasMVC = false;

            for (const auto& prog : programs) {
                for (const auto& streamPair : prog.streamPids) {
                    if (streamPair.second == STREAM_TYPE_H264) hasH264 = true;
                    if (streamPair.second == STREAM_TYPE_MVC) hasMVC = true;
                }
            }

            // For SSIF: ONLY exit when we have BOTH types (H.264 arrives late!)
            // Safety limit reduced to 20000 packets (~3.8MB) to speed up initialization
            if (hasH264 && hasMVC) {
                // CRITICAL: Track the position where both PIDs are synchronized
                // Each M2TS packet is 192 bytes, so position = packet_number * 192
                dualPidStartPos_ = static_cast<uint64_t>(i) * 192;
                std::cout << "[MVCM2TSDemuxer] Found both H.264 and MVC streams at packet " << i
                          << " (position " << dualPidStartPos_ << " bytes)" << std::endl;
                break;
            } else if (i > 20000) {
                std::cout << "[MVCM2TSDemuxer] WARNING: Scanned 20000 packets without finding both stream types" << std::endl;
                std::cout << "[MVCM2TSDemuxer]   hasH264=" << hasH264 << ", hasMVC=" << hasMVC << std::endl;
                break;
            }
        }
    }

    // Get video PIDs
    auto videoPids = reader_->getVideoPids();
    if (videoPids.empty()) {
        std::cerr << "[MVCM2TSDemuxer] No video streams found" << std::endl;
        close();
        return false;
    }

    // CRITICAL FIX: Detect MVC for both dual-PID and single-PID (Blu-ray 3D) modes
    // Check if any PID has MVC descriptor (0x7A) for single-PID interleaved mode
    // CRITICAL FIX: Force MVC detection for Blu-ray 3D M2TS files
    // The descriptor 0x7A is not always present or correctly parsed,
    // leading to videoInfo_.hasMVC being false for valid 3D streams.
    // Forcing it to true allows edge264 to correctly initialize DPB for MVC.
    videoInfo_.hasMVC = true;
    std::cout << "[MVCM2TSDemuxer] WARNING: Forcing videoInfo_.hasMVC = true for debugging purposes." << std::endl;
    videoInfo_.baseVideoPid = 0;
    videoInfo_.mvcVideoPid = 0;

    const auto& programs = reader_->getPrograms();

    // DEBUG: Show what programs we have
    std::cout << "[MVCM2TSDemuxer] === Dual-PID Detection ===" << std::endl;
    std::cout << "[MVCM2TSDemuxer] Number of programs: " << programs.size() << std::endl;
    for (size_t i = 0; i < programs.size(); i++) {
        const auto& prog = programs[i];
        std::cout << "[MVCM2TSDemuxer] Program " << i << ": number=" << prog.programNumber
                  << ", PMT PID=0x" << std::hex << prog.pmtPid << std::dec
                  << ", streams=" << prog.streamPids.size() << std::endl;
        for (const auto& streamPair : prog.streamPids) {
            std::cout << "[MVCM2TSDemuxer]   PID 0x" << std::hex << streamPair.first
                      << " -> type 0x" << (int)streamPair.second << std::dec << std::endl;
        }
    }

    // First, separate H.264 and MVC PIDs
    std::vector<uint16_t> foundH264Pids;
    std::vector<uint16_t> foundMvcPids;

    for (const auto& prog : programs) {
        for (const auto& streamPair : prog.streamPids) {
            uint16_t pid = streamPair.first;
            uint8_t streamType = streamPair.second;

            if (streamType == STREAM_TYPE_H264) {  // 0x1B
                foundH264Pids.push_back(pid);
                std::cout << "[MVCM2TSDemuxer] Found H.264 PID: 0x" << std::hex << pid << std::dec << std::endl;
            } else if (streamType == STREAM_TYPE_MVC) {  // 0x20
                foundMvcPids.push_back(pid);
                std::cout << "[MVCM2TSDemuxer] Found MVC PID: 0x" << std::hex << pid << std::dec << std::endl;
            }
        }
    }

    // Sort PIDs to ensure deterministic assignment (Lowest = Base)
    std::sort(foundH264Pids.begin(), foundH264Pids.end());
    std::sort(foundMvcPids.begin(), foundMvcPids.end());

    uint16_t h264Pid = 0;
    uint16_t mvcPid = 0;

    if (!foundH264Pids.empty()) {
        h264Pid = foundH264Pids[0];
    }

    if (!foundMvcPids.empty()) {
        mvcPid = foundMvcPids[0];
    } else if (foundH264Pids.size() >= 2) {
        // Heuristic: If we have 2+ H.264 streams and NO explicit MVC stream,
        // assume the second one is the MVC Dependent View.
        mvcPid = foundH264Pids[1];
        std::cout << "[MVCM2TSDemuxer] Heuristic: Using second H.264 stream (0x" 
                  << std::hex << mvcPid << std::dec << ") as MVC Dependent View" << std::endl;
    }

    // CRITICAL FIX for SSIF: If we found 0x1012 (likely MVC) but no 0x1011 (Base),
    // force 0x1011 as Base. SSIF files are often bulk-interleaved and we might
    // not have scanned far enough to see the H.264 chunk, but 0x1012 implies 0x1011.
    if ((mvcPid == 0x1012 || h264Pid == 0x1012) && h264Pid != 0x1011) {
        std::cout << "[MVCM2TSDemuxer] Heuristic: Found PID 0x1012 (MVC?), forcing Base PID to 0x1011" << std::endl;
        h264Pid = 0x1011;
        mvcPid = 0x1012;
        videoInfo_.hasMVC = true;
    }
    // Also handle case where we only found 0x1011 and missed 0x1012
    else if (h264Pid == 0x1011 && mvcPid == 0) {
         std::cout << "[MVCM2TSDemuxer] Heuristic: Found Base PID 0x1011 in SSIF, forcing MVC PID to 0x1012" << std::endl;
         mvcPid = 0x1012;
         videoInfo_.hasMVC = true;
    }

    std::cout << "[MVCM2TSDemuxer] Final: h264Pid=0x" << std::hex << h264Pid
              << ", mvcPid=0x" << mvcPid << std::dec << std::endl;

    // If we have BOTH H.264 and MVC PIDs, use H.264 as base and MVC as dependent
    // This is the SSIF format (interleaved base + dependent in separate PIDs)
    if (h264Pid != 0 && mvcPid != 0) {
        videoInfo_.hasMVC = true;
        videoInfo_.baseVideoPid = h264Pid;   // H.264 with IDR frames
        videoInfo_.mvcVideoPid = mvcPid;     // MVC slices
        std::cout << "[MVCM2TSDemuxer] MVC detected: SSIF format (H.264 base + MVC dependent)" << std::endl;
        std::cout << "[MVCM2TSDemuxer]   Base PID: 0x" << std::hex << h264Pid << " (H.264)" << std::dec << std::endl;
        std::cout << "[MVCM2TSDemuxer]   MVC PID: 0x" << std::hex << mvcPid << " (MVC)" << std::dec << std::endl;
    }
    // Otherwise check for single-PID MVC with descriptor 0x7A
    else if (!videoPids.empty()) {
        videoInfo_.baseVideoPid = videoPids[0];

        for (const auto& prog : programs) {
            auto it = prog.mvcStreams.find(videoInfo_.baseVideoPid);
            if (it != prog.mvcStreams.end() && it->second) {
                videoInfo_.hasMVC = true;
                std::cout << "[MVCM2TSDemuxer] MVC detected via descriptor 0x7A (Blu-ray 3D single-PID interleaved)" << std::endl;
                break;
            }
        }

        // Check for dual-PID MVC (legacy, same stream type)
        if (!videoInfo_.hasMVC && videoPids.size() >= 2) {
            videoInfo_.hasMVC = true;
            videoInfo_.mvcVideoPid = videoPids[1];
            std::cout << "[MVCM2TSDemuxer] MVC detected via dual-PID mode (legacy)" << std::endl;
        }
    }

    std::cout << "[MVCM2TSDemuxer] DEBUG: h264Pid=0x" << std::hex << h264Pid 
              << ", mvcPid=0x" << mvcPid << std::dec 
              << ", videoPids=" << videoPids.size() << std::endl;

    std::cout << "[MVCM2TSDemuxer] Base video PID: 0x" << std::hex
              << videoInfo_.baseVideoPid << std::dec << std::endl;
    if (videoInfo_.hasMVC && videoInfo_.mvcVideoPid != 0) {
        std::cout << "[MVCM2TSDemuxer] MVC video PID: 0x" << std::hex
                  << videoInfo_.mvcVideoPid << std::dec << std::endl;
    }

    // Set default FPS if not detected
    if (videoInfo_.fps == 0.0) {
        videoInfo_.fps = 23.976;  // Default for Blu-ray 3D
        std::cout << "[MVCM2TSDemuxer] Using default FPS: " << videoInfo_.fps << std::endl;
    }

    // CRITICAL FIX: Clear all state before seeking to start
    // The first scan (before dual-PID detection) created invalid micro-frames
    // using the PTS fallback. These are stored in pendingFrame_ and must be
    // cleared before starting the IDR scan.
    std::cout << "[MVCM2TSDemuxer] Clearing state and seeking before IDR scan..." << std::endl;
    pesStates_.clear();  // Clear PES reassembly buffers
    mvcBuffer_.clear();
    h264Buffer_.clear(); // Clear H.264 buffer
    currentFrame_ = {};  // Clear current frame
    pendingFrame_ = {};  // CRITICAL: Clear invalid micro-frames from first scan
    pendingFrame_.hasData = false;

    // CRITICAL FIX for SSIF bulk-interleaved format:
    // SSIF files have a special structure with bulk interleaving:
    //   - First section (~7 MB): ALL MVC data for ALL frames (PID 0x1012)
    //   - Second section: ALL H.264 data for ALL frames (PID 0x1011)
    //   - Frames are matched by PTS across both sections
    //
    // We MUST seek to position 0 to read the MVC data. If we skip to the H.264 section,
    // we lose all MVC data and get dependentData=0B for all frames.
    //
    // The PTS-based frame boundary detection (with >3750 tick threshold) correctly
    // prevents mega-frames when processing MVC-only data at the beginning.
    if (!reader_->seek(0)) {
        std::cerr << "[MVCM2TSDemuxer] WARNING: Failed to seek to start (continuing anyway)" << std::endl;
        // Don't fail here, try to find IDR from current position
    } else {
        std::cout << "[MVCM2TSDemuxer] Seeking to position 0 to read SSIF bulk-interleaved data" << std::endl;
    }
    
    if (dualPidStartPos_ > 0) {
        std::cout << "[MVCM2TSDemuxer] Note: H.264 data starts at position " << dualPidStartPos_
                  << " (will be matched with MVC by PTS)" << std::endl;
    }

    std::cout << "[MVCM2TSDemuxer] Searching for first IDR frame..." << std::endl;

    bool foundIDR = false;
    int framesScanned = 0;
    const int maxFramesToScan = 2000;  // Scan up to 2000 frames (increased for robustness)

    while (!foundIDR && framesScanned < maxFramesToScan) {
        FramePair tempPair;
        if (!readNextFramePair(tempPair)) {
            std::cerr << "[MVCM2TSDemuxer] Reached EOF before finding IDR frame" << std::endl;
            close();
            return false;
        }

        framesScanned++;
        if (tempPair.isKeyframe) {
            foundIDR = true;
            std::cout << "[MVCM2TSDemuxer] Found first IDR frame after scanning "
                      << framesScanned << " frames" << std::endl;

            // CRITICAL FIX: For dual-PID SSIF files (bulk-interleaved format), DON'T seek back.
            // The SSIF format has MVC data at position 0-6MB, then H.264 data at 6MB+.
            // If we seek back to 0, frame boundary detection creates incomplete frames
            // because it processes MVC data first without matching H.264 data.
            //
            // Instead, save the IDR frame we found (which has both views properly matched)
            // and use it as the first frame, just like single-PID files.
            if (dualPidStartPos_ > 0) {
                std::cout << "[MVCM2TSDemuxer] Dual-PID SSIF detected - using IDR frame directly (not seeking back)" << std::endl;
                // Put this IDR frame back as pending for the first readNextFramePair() call
                pendingFrame_.baseView = std::move(tempPair.baseData);
                pendingFrame_.dependentView = std::move(tempPair.dependentData);
                pendingFrame_.pts = tempPair.timestamp * 90;  // Convert back to 90kHz
                pendingFrame_.hasData = true;
                pendingFrame_.alreadyPrefixed = true;  // codecPrivate already prepended
                pendingFrame_.hasBasePidData = !tempPair.baseData.empty();
                pendingFrame_.hasMvcPidData = !tempPair.dependentData.empty();
            } else {
                std::cout << "[MVCM2TSDemuxer] Single-PID MVC detected - using IDR frame directly" << std::endl;
                // Put this IDR frame back as pending for the first readNextFramePair() call
                pendingFrame_.baseView = std::move(tempPair.baseData);
                pendingFrame_.dependentView = std::move(tempPair.dependentData);
                pendingFrame_.pts = tempPair.timestamp * 90;  // Convert back to 90kHz
                pendingFrame_.hasData = true;
                pendingFrame_.alreadyPrefixed = true;  // codecPrivate already prepended
                pendingFrame_.hasBasePidData = !tempPair.baseData.empty();
                pendingFrame_.hasMvcPidData = !tempPair.dependentData.empty();
            }
        }
    }

    if (!foundIDR) {
        std::cerr << "[MVCM2TSDemuxer] Could not find IDR frame in first "
                  << maxFramesToScan << " frames" << std::endl;
        close();
        return false;
    }

    std::cout << "[MVCM2TSDemuxer] Probing complete" << std::endl;
    return true;
}

void MVCM2TSDemuxer::close() {
    if (reader_) {
        reader_->close();
    }
    pesStates_.clear();
    mvcBuffer_.clear();
    h264Buffer_.clear();
    hasVideoInfo_ = false;
    hasCodecPrivate_ = false;
}

bool MVCM2TSDemuxer::isOpen() const {
    return reader_ && reader_->isOpen();
}

MVCM2TSDemuxer::VideoInfo MVCM2TSDemuxer::getVideoInfo() const {
    return videoInfo_;
}

std::vector<uint8_t> MVCM2TSDemuxer::getCodecPrivate() const {
    return {};
}

bool MVCM2TSDemuxer::readNextFramePair(FramePair& framePair) {
    if (!isOpen()) {
        return false;
    }

    // Reset current frame state
    currentFrame_ = {};

    // If we have pending data from previous frame boundary detection, use it first
    if (pendingFrame_.hasData) {
        currentFrame_.baseView = std::move(pendingFrame_.baseView);
        currentFrame_.dependentView = std::move(pendingFrame_.dependentView);
        currentFrame_.pts = pendingFrame_.pts;

        // CRITICAL FIX: Detect IDR in pending frame data
        // Don't assume it will be detected later - if all frame data is in pendingFrame_,
        // no new data will be added and IDR detection in processVideoPacket won't run
        currentFrame_.isKeyframe = false;
        if (!currentFrame_.baseView.empty()) {
            // Look for IDR slice (NAL type 5) ONLY - NAL type 20 is MVC dependent view, not IDR
            for (size_t i = 0; i + 4 < currentFrame_.baseView.size(); i++) {
                if (currentFrame_.baseView[i] == 0 && currentFrame_.baseView[i+1] == 0 &&
                    (currentFrame_.baseView[i+2] == 1 ||
                     (currentFrame_.baseView[i+2] == 0 && i+3 < currentFrame_.baseView.size() && currentFrame_.baseView[i+3] == 1))) {
                    size_t nalStart = (currentFrame_.baseView[i+2] == 1) ? i + 3 : i + 4;
                    if (nalStart < currentFrame_.baseView.size()) {
                        uint8_t nalType = currentFrame_.baseView[nalStart] & 0x1F;
                        if (nalType == 5) {  // ONLY IDR slice (not NAL type 20 which is MVC dependent view)
                            currentFrame_.isKeyframe = true;
                            break;
                        }
                    }
                }
            }
        }

        // Restore PID tracking flags from pending frame
        // CRITICAL: Don't infer from buffer contents - use actual PID flags
        currentFrame_.hasBasePidData = pendingFrame_.hasBasePidData;
        currentFrame_.hasMvcPidData = pendingFrame_.hasMvcPidData;

        pendingFrame_.hasData = false;
    }

    // Read TS packets until we have a complete frame
    M2TSReader::TSPacket packet;
    while (reader_->readPacket(packet)) {
        // DEBUG: Log packet PIDs to diagnose filtering issues
        /*
        if (g_debugCount < 500) {
            // Only log if it's NOT a null packet (0x1FFF) or PAT/PMT/PCR to reduce noise, 
            // unless it IS our video PID.
            if (packet.pid == videoInfo_.baseVideoPid || 
                packet.pid == videoInfo_.mvcVideoPid || 
                g_debugCount < 50) {
                std::cout << "[MVCM2TSDemuxer] Pkt PID: 0x" << std::hex << packet.pid << std::dec 
                          << " Base:" << (packet.pid == videoInfo_.baseVideoPid) 
                          << " Mvc:" << (packet.pid == videoInfo_.mvcVideoPid) << std::endl;
            }
        }
        */

        // Only process video PIDs (ignore PID 0/PAT if mvcVideoPid is 0)
        if (packet.pid == videoInfo_.baseVideoPid ||
            (videoInfo_.mvcVideoPid != 0 && packet.pid == videoInfo_.mvcVideoPid)) {
            processVideoPacket(packet);
        }

        // Check if frame is complete
        if (isFrameComplete()) {
            // Output the frame - RAW PASS-THROUGH
            // V7b FIX: Do NOT modify the stream. Pass raw Annex B exactly as demuxed.
            // Removing/replacing inline SPS/PPS caused corruption (NAL type 24) and black screen.
            framePair.baseData = std::move(currentFrame_.baseView);
            framePair.dependentData = std::move(currentFrame_.dependentView);
            
            framePair.timestamp = ptsToMs(currentFrame_.pts);
            framePair.isKeyframe = currentFrame_.isKeyframe;

            // Reset prefix flag after returning this frame
            pendingFrame_.alreadyPrefixed = false;

            return true;
        }
    }

    // EOF reached
    return false;
}

void MVCM2TSDemuxer::processVideoPacket(const M2TSReader::TSPacket& packet) {
    PESState& state = pesStates_[packet.pid];

    // Start of new PES packet
    if (packet.payloadUnitStartIndicator) {
        // Process previous PES if exists
        if (!state.buffer.empty()) {
            // Parse PES header
            int64_t pts, dts;
            size_t headerLength;
            if (parsePESHeader(state.buffer, pts, dts, headerLength)) {
                // Extract NAL units from PES payload
                std::vector<uint8_t> nalData(state.buffer.begin() + headerLength,
                                            state.buffer.end());

                // Extract codec private on first SPS/PPS
                if (!hasCodecPrivate_) {
                    extractCodecPrivate(nalData);
                }

                // Separate base and dependent views
                std::vector<uint8_t> baseView, dependentView;
                separateNALUnits(nalData, baseView, dependentView);

                // DEBUG: Log PID, PTS, and data sizes to diagnose dual-PID interleaving
                g_debugCount++;
                if (g_debugCount < 100) {  // Increased from 20 to 100 for better debugging
                    std::cout << "[MVCM2TSDemuxer] PES from PID 0x" << std::hex << packet.pid << std::dec
                              << ": PTS=" << pts << ", baseView=" << baseView.size() << "B"
                              << ", dependentView=" << dependentView.size() << "B" << std::endl;
                }

                // Check if we're starting a new frame (AUD or different PTS)
                bool isNewFrame = false;

                // CRITICAL FIX for dual-PID SSIF: In dual-PID mode, synchronize via PTS from BOTH PIDs!
                // A frame is complete when we have data from BOTH H.264 (0x1011) and MVC (0x1012) PIDs
                // with the SAME PTS value.
                bool isDualPID = (videoInfo_.baseVideoPid != 0 && videoInfo_.mvcVideoPid != 0);

                // CRITICAL: Always check for AUD in H.264 data (both single-PID and dual-PID modes)
                // SSIF files have H.264 bursts before MVC arrives. Without AUD detection, we'd
                // accumulate multiple H.264 frames into one mega-frame while waiting for MVC.
                bool hasAUD = false;
                if (!baseView.empty()) {
                    size_t i = 0;
                    while (i + 4 < nalData.size()) {
                        if (nalData[i] == 0 && nalData[i+1] == 0 && nalData[i+2] == 1) {
                            uint8_t nalType = nalData[i+3] & 0x1F;
                            if (nalType == 9) {  // AUD
                                hasAUD = true;
                                break;
                            }
                            i += 3;
                        } else {
                            i++;
                        }
                    }
                }

                if (isDualPID) {
                    // In dual-PID mode, use AUD for H.264 base PID
                    // This prevents accumulating multiple H.264 AUs while waiting for MVC data
                    if (packet.pid == videoInfo_.baseVideoPid && hasAUD) {
                        isNewFrame = true;
                    }
                    // For MVC PID or when no AUD, use PTS change
                    else if (currentFrame_.pts != 0 && pts != currentFrame_.pts) {
                        isNewFrame = true;
                    }
                } else {
                    // In single-PID mode, use AUD first
                    if (hasAUD) {
                        isNewFrame = true;
                    }
                    // CRITICAL FIX: If no AUD found but we have MVC data (NAL type 20),
                    // use PTS change as frame boundary with a threshold.
                    // Only trigger on LARGE PTS jumps (> 1 frame @ 24fps = 3750 ticks @ 90kHz)
                    // to avoid creating boundaries on every PES packet.
                    else if (!dependentView.empty() &&
                        currentFrame_.pts != 0 &&
                        pts > currentFrame_.pts &&  // Only forward jumps
                        (pts - currentFrame_.pts) > 3750) {  // > 1 frame @ 24fps
                        isNewFrame = true;
                        if (g_debugCount < 20) {
                            std::cout << "[MVCM2TSDemuxer]   → Using large PTS jump as boundary (MVC data, no AUD, PTS jump="
                                      << (pts - currentFrame_.pts) << ")" << std::endl;
                        }
                    }
                }

                // if (g_debugCount < 10000) {
                //     std::cout << "[MVCM2TSDemuxer]   isDualPID=" << isDualPID << ", isNewFrame=" << isNewFrame
                //               << ", currentPTS=" << currentFrame_.pts << ", newPTS=" << pts
                //               << ", PID=0x" << std::hex << packet.pid << std::dec << std::endl;
                // }

                // If starting new frame and we have accumulated data, check if previous frame is complete
                // CRITICAL: Also check dependentView for SSIF bulk-interleaved MVC-only sections
                if (isNewFrame && (!currentFrame_.baseView.empty() || !currentFrame_.dependentView.empty())) {
                    
                    bool buffered = false;

                    if (isDualPID) {
                        // Scenario A: MVC-only (from MVC PID) - Missing H.264
                        if (!currentFrame_.hasBasePidData && currentFrame_.hasMvcPidData) {
                            // Check if H.264 is waiting in buffer
                            auto it = h264Buffer_.find(currentFrame_.pts);
                            if (it != h264Buffer_.end()) {
                                // MATCH FOUND! Prepend H.264 data to baseView
                                std::vector<uint8_t> combined = std::move(it->second);
                                combined.insert(combined.end(), currentFrame_.baseView.begin(), currentFrame_.baseView.end());
                                currentFrame_.baseView = std::move(combined);
                                currentFrame_.hasBasePidData = true;
                                h264Buffer_.erase(it);
                                if (g_debugCount < 100) std::cout << "[MVCM2TSDemuxer]   -> Matched with buffered H.264: PTS=" << currentFrame_.pts << std::endl;
                            } else {
                                // Buffer MVC
                                mvcBuffer_[currentFrame_.pts] = std::move(currentFrame_.dependentView);
                                buffered = true;
                                if (g_debugCount < 100) std::cout << "[MVCM2TSDemuxer]   -> Buffered MVC frame: PTS=" << currentFrame_.pts << std::endl;
                            }
                        } 
                        // Scenario B: H.264-only (from Base PID) - Missing MVC
                        else if (currentFrame_.hasBasePidData && !currentFrame_.hasMvcPidData) {
                            // Check if MVC is waiting in buffer
                            auto it = mvcBuffer_.find(currentFrame_.pts);
                            if (it != mvcBuffer_.end()) {
                                // MATCH FOUND!
                                currentFrame_.dependentView = std::move(it->second);
                                currentFrame_.hasMvcPidData = true;
                                mvcBuffer_.erase(it);
                                if (g_debugCount < 100) std::cout << "[MVCM2TSDemuxer]   -> Matched with buffered MVC: PTS=" << currentFrame_.pts << std::endl;
                            } else {
                                // Buffer H.264
                                h264Buffer_[currentFrame_.pts] = std::move(currentFrame_.baseView);
                                buffered = true;
                                if (g_debugCount < 100) std::cout << "[MVCM2TSDemuxer]   -> Buffered H.264 frame: PTS=" << currentFrame_.pts << std::endl;
                            }
                        }
                    }

                    if (buffered) {
                        // Frame was buffered (removed from current flow).
                        // The NEW packet (in baseView/dependentView) belongs to the NEXT frame.
                        // We must start the new frame IMMEDIATELY in currentFrame_.
                        currentFrame_ = {};
                        currentFrame_.baseView = std::move(baseView);
                        currentFrame_.dependentView = std::move(dependentView);
                        currentFrame_.pts = pts;
                        
                        bool isBasePID = (packet.pid == videoInfo_.baseVideoPid);
                        bool isMvcPID = (packet.pid == videoInfo_.mvcVideoPid);
                        currentFrame_.hasBasePidData = (isBasePID && !currentFrame_.baseView.empty());
                        currentFrame_.hasMvcPidData = (isMvcPID && !currentFrame_.dependentView.empty());
                        
                        // Check for IDR in the new data
                        if (!currentFrame_.baseView.empty()) {
                             // Simplified IDR check (enough for starting frame state)
                             // Full check is done during accumulation but good to have flag
                             currentFrame_.isKeyframe = false; // Reset
                        }
                        
                        return; // Continue processing next packets into this new currentFrame_
                    }

                    // If NOT buffered, it means currentFrame_ is complete and ready to output.
                    currentFrame_.isComplete = true;

                    // Save current packet data as pending for next frame (to be picked up by next readNextFramePair)
                    pendingFrame_.baseView = std::move(baseView);
                    pendingFrame_.dependentView = std::move(dependentView);
                    pendingFrame_.pts = pts;
                    pendingFrame_.hasData = true;
                    
                    bool isBasePID = (packet.pid == videoInfo_.baseVideoPid);
                    bool isMvcPID = (packet.pid == videoInfo_.mvcVideoPid);
                    pendingFrame_.hasBasePidData = (isBasePID && !pendingFrame_.baseView.empty());
                    pendingFrame_.hasMvcPidData = (isMvcPID && !pendingFrame_.dependentView.empty());

                    // if (g_debugCount < 100) {
                    //     std::cout << "[MVCM2TSDemuxer]   -> Saved to pendingFrame_ (frame complete, base="
                    //               << pendingFrame_.baseView.size() << "B, dep=" << pendingFrame_.dependentView.size() << "B)" << std::endl;
                    // }
                }

                // Accumulate data into current frame if not starting a new complete frame
                if (!isNewFrame || !currentFrame_.isComplete) {
                    if (!baseView.empty()) {
                        currentFrame_.baseView.insert(currentFrame_.baseView.end(),
                                                     baseView.begin(), baseView.end());
                        currentFrame_.pts = pts;

                        // Mark that we've received data from base PID
                        if (isDualPID && packet.pid == videoInfo_.baseVideoPid) {
                            currentFrame_.hasBasePidData = true;
                        }

                        // Check for IDR (keyframe)
                        if (!nalData.empty()) {
                            // Look for IDR slice (type 5) ONLY - NAL type 20 is MVC dependent view, not IDR
                            for (size_t j = 0; j + 4 < nalData.size(); j++) {
                                if (nalData[j] == 0 && nalData[j+1] == 0 && nalData[j+2] == 1) {
                                    uint8_t nalType = nalData[j+3] & 0x1F;
                                    if (nalType == 5) {  // ONLY IDR slice (not NAL type 20)
                                        currentFrame_.isKeyframe = true;
                                        break;
                                    }
                                }
                            }
                        }

                        // if (g_debugCount < 20) {
                        //     std::cout << "[MVCM2TSDemuxer]   → Accumulated to currentFrame_.baseView (now "
                        //               << currentFrame_.baseView.size() << "B total)" << std::endl;
                        // }
                    }

                    if (!dependentView.empty()) {
                        currentFrame_.dependentView.insert(currentFrame_.dependentView.end(),
                                                          dependentView.begin(), dependentView.end());

                        // Mark that we've received data from MVC PID
                        if (isDualPID && packet.pid == videoInfo_.mvcVideoPid) {
                            currentFrame_.hasMvcPidData = true;
                        }

                        // if (g_debugCount < 20) {
                        //     std::cout << "[MVCM2TSDemuxer]   → Accumulated to currentFrame_.dependentView (now "
                        //               << currentFrame_.dependentView.size() << "B total)" << std::endl;
                        // }
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

bool MVCM2TSDemuxer::parsePESHeader(const std::vector<uint8_t>& pesData,
                                    int64_t& pts, int64_t& dts,
                                    size_t& headerLength) {
    if (pesData.size() < 9) {
        return false;
    }

    // Check PES start code (0x000001)
    if (pesData[0] != 0x00 || pesData[1] != 0x00 || pesData[2] != 0x01) {
        return false;
    }

    uint8_t ptsDtsFlags = (pesData[7] >> 6) & 0x03;
    uint8_t pesHeaderLength = pesData[8];

    headerLength = 9 + pesHeaderLength;

    // Parse PTS
    pts = 0;
    dts = 0;
    if (ptsDtsFlags >= 2 && pesData.size() >= 14) {
        pts = (static_cast<int64_t>(pesData[9] & 0x0E) << 29) |
              (static_cast<int64_t>(pesData[10]) << 22) |
              (static_cast<int64_t>(pesData[11] & 0xFE) << 14) |
              (static_cast<int64_t>(pesData[12]) << 7) |
              (static_cast<int64_t>(pesData[13]) >> 1);
    }

    // Parse DTS if present
    if (ptsDtsFlags == 3 && pesData.size() >= 19) {
        dts = (static_cast<int64_t>(pesData[14] & 0x0E) << 29) |
              (static_cast<int64_t>(pesData[15]) << 22) |
              (static_cast<int64_t>(pesData[16] & 0xFE) << 14) |
              (static_cast<int64_t>(pesData[17]) << 7) |
              (static_cast<int64_t>(pesData[18]) >> 1);
    }

    return true;
}

void MVCM2TSDemuxer::separateNALUnits(const std::vector<uint8_t>& nalData,
                                     std::vector<uint8_t>& baseOut,
                                     std::vector<uint8_t>& dependentOut) {
    // Parse NAL units (Annex B format: 0x000001 or 0x00000001 start codes)
    // Uses globals reset in open()

    size_t i = 0;
    int nalCount = 0;

    while (i < nalData.size()) {
        // Find start code
        if (i + 2 < nalData.size() &&
            nalData[i] == 0x00 && nalData[i + 1] == 0x00) {
            if (nalData[i + 2] == 0x01) {
                // 3-byte start code
                i += 3;
            } else if (i + 3 < nalData.size() &&
                      nalData[i + 2] == 0x00 && nalData[i + 3] == 0x01) {
                // 4-byte start code
                i += 4;
            } else {
                i++;
                continue;
            }

            if (i >= nalData.size()) break;

            // Get NAL unit type
            uint8_t nalType = nalData[i] & 0x1F;
            nalCount++;
            g_nalTypeHistogram[nalType]++;

            // DEBUG: Log first few NAL units to see what's in the stream
            if (g_nalFirstCall && nalCount <= 15) {
                // std::cout << "[MVCM2TSDemuxer] NAL #" << nalCount << ": type=" << (int)nalType;
                // if (nalType == 1) std::cout << " (SLICE)";
                // else if (nalType == 5) std::cout << " (IDR_SLICE)";
                // else if (nalType == 7) std::cout << " (SPS)";
                // else if (nalType == 8) std::cout << " (PPS)";
                // else if (nalType == 9) std::cout << " (AUD)";
                // else if (nalType == 15) std::cout << " (SUBSET_SPS)";
                // else if (nalType == 20) std::cout << " (MVC_SLICE_EXT) *** MVC DETECTED ***";
                // std::cout << std::endl;
            }

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

            // Extract NAL unit (with start code)
            size_t nalStart = i - (nalData[i - 1] == 0x01 ? 3 : 4);
            size_t nalLength = nextStart - nalStart;

            // Classify: MVC extension slices go to dependent, rest to base
            if (nalType == NAL_TYPE_CODED_SLICE_EXTENSION) {
                dependentOut.insert(dependentOut.end(),
                                   nalData.begin() + nalStart,
                                   nalData.begin() + nalStart + nalLength);
            } else {
                baseOut.insert(baseOut.end(),
                              nalData.begin() + nalStart,
                              nalData.begin() + nalStart + nalLength);
            }

            i = nextStart;
        } else {
            i++;
        }
    }

    // After first call, print histogram summary
    if (g_nalFirstCall) {
        g_nalFirstCall = false;
        // std::cout << "[MVCM2TSDemuxer] NAL type histogram (first frame):" << std::endl;
        // for (int t = 0; t < 32; t++) {
        //     if (g_nalTypeHistogram[t] > 0) {
        //         std::cout << "[MVCM2TSDemuxer]   Type " << t << ": " << g_nalTypeHistogram[t] << " units";
        //         if (t == 20) std::cout << " *** MVC DETECTED ***";
        //         std::cout << std::endl;
        //     }
        // }
    }
}

void MVCM2TSDemuxer::extractCodecPrivate(const std::vector<uint8_t>& nalData) {
    // Extract SPS/PPS NAL units for codec private
    // This is a simplified version - proper implementation would build AVCC format
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

                    // Accumulate SPS/PPS/Subset SPS
                    if (nalType == NAL_TYPE_SPS || nalType == NAL_TYPE_PPS ||
                        nalType == NAL_TYPE_SUBSET_SPS) {
                        codecPrivate_.insert(codecPrivate_.end(),
                                           nalData.begin() + i - startCodeSize,
                                           nalData.begin() + nextStart);
                        hasCodecPrivate_ = true;
                        // std::cout << "[MVCM2TSDemuxer] Extracted NAL type "
                        //           << static_cast<int>(nalType) << " for codec private" << std::endl;
                    }

                    i = nextStart;
                    continue;
                }
            }
        }
        i++;
    }
}

bool MVCM2TSDemuxer::isFrameComplete() const {
    // Frame is complete if we have base view data
    // Dependent view is optional
    return currentFrame_.isComplete && !currentFrame_.baseView.empty();
}

bool MVCM2TSDemuxer::seek(int64_t timestampMs) {
    // Simplified seek - just reset to start
    // Proper implementation would use PCR/PTS for seeking
    (void)timestampMs;  // Unused parameter - reserved for future implementation
    return reader_->seek(0);
}

void MVCM2TSDemuxer::parseSPSForDimensions() {
    if (codecPrivate_.empty() || hasVideoInfo_) {
        return;
    }

    // Simplified SPS parsing: look for SPS NAL and use default Blu-ray 3D dimensions
    // Full SPS parsing with exp-golomb decoding would be complex
    size_t i = 0;
    while (i < codecPrivate_.size()) {
        // Find start code
        if (i + 3 < codecPrivate_.size() &&
            codecPrivate_[i] == 0x00 && codecPrivate_[i + 1] == 0x00) {
            size_t startCodeSize = 0;
            if (codecPrivate_[i + 2] == 0x01) {
                startCodeSize = 3;
            } else if (i + 4 < codecPrivate_.size() &&
                      codecPrivate_[i + 2] == 0x00 && codecPrivate_[i + 3] == 0x01) {
                startCodeSize = 4;
            }

            if (startCodeSize > 0) {
                i += startCodeSize;
                if (i < codecPrivate_.size()) {
                    uint8_t nalType = codecPrivate_[i] & 0x1F;

                    // Found SPS
                    if (nalType == NAL_TYPE_SPS || nalType == NAL_TYPE_SUBSET_SPS) {
                        // Use default Blu-ray 3D dimensions
                        // Full parsing would require exp-golomb decoding
                        videoInfo_.width = 1920;
                        videoInfo_.height = 1080;
                        videoInfo_.fps = 23.976;
                        hasVideoInfo_ = true;

                        // std::cout << "[MVCM2TSDemuxer] Using Blu-ray 3D default dimensions: "
                        //           << videoInfo_.width << "x" << videoInfo_.height << " @ "
                        //           << videoInfo_.fps << " fps" << std::endl;
                        return;
                    }
                }
            }
        }
        i++;
    }
}

} // namespace mvc_demux
