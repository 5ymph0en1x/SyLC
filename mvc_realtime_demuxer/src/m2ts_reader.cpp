#include "m2ts_reader.h"
#include <algorithm>
#include <cstring>
#include <iostream>

namespace mvc_demux {

// TS packet constants
constexpr uint8_t TS_SYNC_BYTE = 0x47;
constexpr int TS_PACKET_SIZE = 188;
constexpr int M2TS_PACKET_SIZE = 192;
constexpr uint16_t PAT_PID = 0x0000;

// Stream types
constexpr uint8_t STREAM_TYPE_H264 = 0x1B;
constexpr uint8_t STREAM_TYPE_MVC = 0x20;

M2TSReader::M2TSReader()
    : fileSize_(0), packetSize_(0) {
}

M2TSReader::~M2TSReader() {
    close();
}

bool M2TSReader::open(const std::string& filePath) {
    // EXPLICIT 4 MB read buffer (see readBuffered). pubsetbuf was unreliable on MSVC: the
    // demuxer read 192-byte packets at ~2 MB/s while the SAME disc streams at 18 MB/s with
    // 1 MB reads (measured) — the optical drive re-seeks on tiny reads. Buffering ourselves
    // turns the post-seek dependent-extent read from ~20 s into <1 s.
    io_buffer_.resize(16u << 20);  // 16 MB — some disc regions only stream fast with >=16 MB reads
    bufPos_ = 0;
    bufLen_ = 0;
    bufFileStart_ = 0;
    file_.open(filePath, std::ios::binary);
    if (!file_) {
        std::cerr << "[M2TSReader] Failed to open file: " << filePath << std::endl;
        return false;
    }

    // Get file size
    file_.seekg(0, std::ios::end);
    fileSize_ = file_.tellg();
    file_.seekg(0, std::ios::beg);

    // Detect packet size (188 or 192 bytes)
    if (!detectPacketSize()) {
        std::cerr << "[M2TSReader] Failed to detect packet size" << std::endl;
        close();
        return false;
    }

    // OPTIMIZATION: Pre-allocate packet buffer to avoid malloc() on every read
    packetBuffer_.resize(packetSize_);

    std::cout << "[M2TSReader] Opened " << filePath << " (packet size: "
              << packetSize_ << " bytes)" << std::endl;

    return true;
}

void M2TSReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
    programs_.clear();
    pesBuffers_.clear();
}

bool M2TSReader::detectPacketSize() {
    // Read first 8KB to detect packet size. Fill the probe with a loop that tolerates
    // a short first read (optical Blu-ray drives can return a partial read right after
    // open / spin-up). Only a genuine EOF or repeated hard failure aborts.
    constexpr size_t PROBE_SIZE = 8192;
    std::vector<uint8_t> probe(PROBE_SIZE);

    file_.clear();
    file_.seekg(0, std::ios::beg);
    size_t bytesRead = 0;
    for (int tries = 0; tries < 8 && bytesRead < PROBE_SIZE; tries++) {
        file_.read(reinterpret_cast<char*>(probe.data()) + bytesRead,
                   static_cast<std::streamsize>(PROBE_SIZE - bytesRead));
        std::streamsize n = file_.gcount();
        if (n <= 0) {
            if (file_.eof()) break;     // genuine end of (tiny) file
            file_.clear();              // transient failure: clear and retry
            continue;
        }
        bytesRead += static_cast<size_t>(n);
    }
    file_.clear();
    file_.seekg(0, std::ios::beg);

    if (bytesRead < 1024) {
        return false;
    }

    // Look for sync bytes at regular intervals
    auto countSyncBytes = [&](int stride) -> int {
        int count = 0;
        for (size_t i = 0; i + stride * 5 < bytesRead; i++) {
            if (probe[i] == TS_SYNC_BYTE &&
                probe[i + stride] == TS_SYNC_BYTE &&
                probe[i + stride * 2] == TS_SYNC_BYTE &&
                probe[i + stride * 3] == TS_SYNC_BYTE) {
                count++;
            }
        }
        return count;
    };

    int count188 = countSyncBytes(TS_PACKET_SIZE);
    int count192 = countSyncBytes(M2TS_PACKET_SIZE);

    if (count192 > count188) {
        packetSize_ = M2TS_PACKET_SIZE;
        std::cout << "[M2TSReader] Detected M2TS format (192-byte packets)" << std::endl;
    } else if (count188 > 0) {
        packetSize_ = TS_PACKET_SIZE;
        std::cout << "[M2TSReader] Detected TS format (188-byte packets)" << std::endl;
    } else {
        std::cerr << "[M2TSReader] No valid sync pattern found" << std::endl;
        return false;
    }

    return true;
}

bool M2TSReader::readBuffered(uint8_t* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        if (bufPos_ >= bufLen_) {
            // Refill with ONE big sequential read. bufFileStart_ = file offset of io_buffer_[0]
            // so bufFileStart_+bufPos_ is always the logical read position (used by resync).
            if (!file_) return false;                       // fail/eof state -> no more data
            bufFileStart_ = static_cast<uint64_t>(file_.tellg());
            file_.read(io_buffer_.data(), static_cast<std::streamsize>(io_buffer_.size()));
            bufLen_ = static_cast<size_t>(file_.gcount());
            bufPos_ = 0;
            if (bufLen_ == 0) return false;                 // genuine EOF
        }
        size_t avail = bufLen_ - bufPos_;
        size_t take = (n - got < avail) ? (n - got) : avail;
        std::memcpy(dst + got, io_buffer_.data() + bufPos_, take);
        bufPos_ += take;
        got += take;
    }
    return true;
}

bool M2TSReader::readPacket(TSPacket& packet) {
    if (!file_.is_open()) {
        return false;
    }

    // Read one packet from the explicit big-chunk buffer (one big disk read serves thousands
    // of packets; pubsetbuf was unreliable on MSVC and left reads at ~2 MB/s).
    if (!readBuffered(packetBuffer_.data(), static_cast<size_t>(packetSize_))) {
        return false;
    }

    // Skip timecode if M2TS (first 4 bytes)
    int tcSkip = (packetSize_ == M2TS_PACKET_SIZE) ? 4 : 0;

    // Resync on sync-byte loss. Real Blu-ray 3D SSIF files contain non-TS regions at
    // the boundaries between interleaved base/dependent extents (measured: e.g. a
    // 1920-byte gap at 0.37 MB and a 4032-byte gap at 0.83 MB before the base view
    // first appears). A wrong-phase read must NOT end the stream: scan ahead for where
    // the packet cadence resumes and continue. Standard MPEG-TS demuxer behavior.
    if (packetBuffer_[tcSkip] != TS_SYNC_BYTE) {
        if (!resyncToNextPacket()) {
            return false;
        }
        // resyncToNextPacket() reloaded packetBuffer_ with a cadence-aligned packet.
    }

    // Parse TS header (4 bytes)
    const uint8_t* tsData = packetBuffer_.data() + tcSkip;
    packet.syncByte = tsData[0];
    if (packet.syncByte != TS_SYNC_BYTE) {
        return false;  // guard: should not happen after a successful resync
    }

    packet.transportErrorIndicator = (tsData[1] & 0x80) != 0;
    packet.payloadUnitStartIndicator = (tsData[1] & 0x40) != 0;
    packet.transportPriority = (tsData[1] & 0x20) != 0;
    packet.pid = ((tsData[1] & 0x1F) << 8) | tsData[2];

    packet.scramblingControl = (tsData[3] >> 6) & 0x03;
    packet.adaptationFieldExists = (tsData[3] & 0x20) != 0;
    packet.payloadExists = (tsData[3] & 0x10) != 0;
    packet.continuityCounter = tsData[3] & 0x0F;

    // Parse adaptation field if present
    int headerSize = 4;
    packet.pcr = 0;

    if (packet.adaptationFieldExists) {
        uint8_t adaptationLength = tsData[4];
        headerSize += 1 + adaptationLength;

        if (adaptationLength > 0 && (tsData[5] & 0x10)) {
            // PCR present
            packet.pcr = static_cast<uint64_t>(tsData[6]) << 25;
            packet.pcr |= static_cast<uint64_t>(tsData[7]) << 17;
            packet.pcr |= static_cast<uint64_t>(tsData[8]) << 9;
            packet.pcr |= static_cast<uint64_t>(tsData[9]) << 1;
            packet.pcr |= static_cast<uint64_t>(tsData[10]) >> 7;
        }
    }

    // Extract payload
    if (packet.payloadExists && headerSize < TS_PACKET_SIZE) {
        int payloadSize = TS_PACKET_SIZE - headerSize;
        packet.payload.assign(tsData + headerSize, tsData + headerSize + payloadSize);
    } else {
        packet.payload.clear();
    }

    // Parse PAT/PMT
    if (packet.pid == PAT_PID && packet.payloadUnitStartIndicator) {
        std::vector<uint8_t> section;
        if (parsePSISection(packet, section)) {
            parsePAT(section);
        }
    } else {
        // Check if this is a PMT PID
        for (const auto& prog : programs_) {
            if (packet.pid == prog.pmtPid && packet.payloadUnitStartIndicator) {
                std::vector<uint8_t> section;
                if (parsePSISection(packet, section)) {
                    parsePMT(section, packet.pid);
                }
                break;
            }
        }
    }

    return true;
}

bool M2TSReader::resyncToNextPacket() {
    resync_count_++;  // DIAG
    // Called right after readPacket() loaded a wrong-phase block into packetBuffer_.
    // Search forward (bounded) for the next position where the packet cadence resumes,
    // reposition file_ there, and reload packetBuffer_ with that aligned packet.
    const size_t ps = static_cast<size_t>(packetSize_);
    const int tcSkip = (packetSize_ == M2TS_PACKET_SIZE) ? 4 : 0;
    const size_t MAX_SCAN = 4 * 1024 * 1024;          // bound the resync effort (~4 MB)
    const size_t NEED_AHEAD = tcSkip + 3 * ps + 1;    // candidate + 3 lookahead packets

    // Seed the search buffer with the bad block we just read. With explicit buffering the
    // logical read position is bufFileStart_+bufPos_ (file_.tellg() is ahead by the unconsumed
    // buffer). Sync file_ there and drop the buffer so the forward CHUNK reads below continue
    // from right after the bad block; the next readPacket then refills cleanly past the resync.
    std::vector<uint8_t> buf(packetBuffer_.begin(), packetBuffer_.begin() + ps);
    uint64_t logicalPos = bufFileStart_ + bufPos_;
    uint64_t bufBase = logicalPos - ps;
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(logicalPos), std::ios::beg);
    bufPos_ = 0;
    bufLen_ = 0;

    size_t scanStart = 1;  // skip the known-bad boundary at offset 0
    while (buf.size() <= MAX_SCAN) {
        if (buf.size() >= NEED_AHEAD) {
            size_t limit = buf.size() - (tcSkip + 3 * ps);
            for (size_t b = scanStart; b < limit; b++) {
                // Require 4 consecutive sync bytes at the packet cadence: makes a
                // chance 0x47 inside payload an astronomically unlikely false resync.
                if (buf[b + tcSkip] == TS_SYNC_BYTE &&
                    buf[b + tcSkip + ps] == TS_SYNC_BYTE &&
                    buf[b + tcSkip + 2 * ps] == TS_SYNC_BYTE &&
                    buf[b + tcSkip + 3 * ps] == TS_SYNC_BYTE) {
                    uint64_t boundary = bufBase + b;
                    file_.clear();
                    file_.seekg(static_cast<std::streamoff>(boundary), std::ios::beg);
                    file_.read(reinterpret_cast<char*>(packetBuffer_.data()),
                               static_cast<std::streamsize>(ps));
                    return file_.gcount() == static_cast<std::streamsize>(ps) &&
                           packetBuffer_[tcSkip] == TS_SYNC_BYTE;
                }
            }
            scanStart = limit;  // don't rescan bytes already checked
        }
        // Append another chunk and keep scanning (lookahead may span the join).
        size_t old = buf.size();
        constexpr size_t CHUNK = 64 * 1024;
        buf.resize(old + CHUNK);
        file_.read(reinterpret_cast<char*>(buf.data()) + old,
                   static_cast<std::streamsize>(CHUNK));
        std::streamsize n = file_.gcount();
        if (n <= 0) break;  // EOF before a resync point was found
        buf.resize(old + static_cast<size_t>(n));
    }
    return false;
}

bool M2TSReader::parsePSISection(const TSPacket& packet, std::vector<uint8_t>& section) {
    if (packet.payload.empty()) {
        return false;
    }

    // Skip pointer field
    uint8_t pointerField = packet.payload[0];
    size_t offset = 1 + pointerField;

    if (offset + 3 > packet.payload.size()) {
        return false;
    }

    // Parse section length
    uint16_t sectionLength = ((packet.payload[offset + 1] & 0x0F) << 8) |
                             packet.payload[offset + 2];

    if (offset + 3 + sectionLength > packet.payload.size()) {
        return false;
    }

    section.assign(packet.payload.begin() + offset,
                   packet.payload.begin() + offset + 3 + sectionLength);
    return true;
}

void M2TSReader::parsePAT(const std::vector<uint8_t>& data) {
    if (data.size() < 8) {
        return;
    }

    uint8_t tableId = data[0];
    if (tableId != 0x00) { // PAT table_id
        return;
    }

    uint16_t sectionLength = ((data[1] & 0x0F) << 8) | data[2];

    // Parse program entries
    // CRITICAL FIX for SSIF: Don't clear programs_! PATs arrive frequently
    // and would wipe out all accumulated stream PIDs from previous PMTs.
    // Instead, check if program already exists and reuse it.
    std::vector<ProgramInfo> newPrograms;

    for (size_t i = 8; i + 4 <= 3 + sectionLength - 4; i += 4) {
        uint16_t programNumber = (data[i] << 8) | data[i + 1];
        uint16_t pmtPid = ((data[i + 2] & 0x1F) << 8) | data[i + 3];

        if (programNumber != 0) { // Skip network PID
            // Check if this program already exists
            ProgramInfo* existingProg = nullptr;
            for (auto& p : programs_) {
                if (p.programNumber == programNumber) {
                    existingProg = &p;
                    break;
                }
            }

            if (existingProg) {
                // Reuse existing program (keeps accumulated streamPids)
                newPrograms.push_back(*existingProg);
                // std::cout << "[M2TSReader] Reusing program " << programNumber
                //           << " (PMT PID: 0x" << std::hex << pmtPid << std::dec
                //           << ", accumulated PIDs: " << existingProg->streamPids.size() << ")" << std::endl;
            } else {
                // Create new program
                ProgramInfo prog;
                prog.programNumber = programNumber;
                prog.pmtPid = pmtPid;
                newPrograms.push_back(prog);
                // std::cout << "[M2TSReader] Found program " << programNumber
                //           << " (PMT PID: 0x" << std::hex << pmtPid << std::dec << ")" << std::endl;
            }
        }
    }

    programs_ = newPrograms;
}

void M2TSReader::parsePMT(const std::vector<uint8_t>& data, uint16_t pid) {
    if (data.size() < 12) {
        return;
    }

    uint8_t tableId = data[0];
    if (tableId != 0x02) { // PMT table_id
        return;
    }

    uint16_t sectionLength = ((data[1] & 0x0F) << 8) | data[2];
    uint16_t programInfoLength = ((data[10] & 0x0F) << 8) | data[11];

    // Find program
    ProgramInfo* prog = nullptr;
    for (auto& p : programs_) {
        if (p.pmtPid == pid) {
            prog = &p;
            break;
        }
    }

    if (!prog) {
        return;
    }

    // std::cout << "[M2TSReader] PMT Details - sectionLength: " << sectionLength
    //           << ", programInfoLength: " << programInfoLength << std::endl;

    // DEBUG: Parse program-level descriptors (often contains MVC info in Blu-ray 3D)
    if (programInfoLength > 0) {
        // std::cout << "[M2TSReader] === Program-level descriptors ===" << std::endl;
        size_t progDescPos = 12;
        size_t progDescEnd = 12 + programInfoLength;

        while (progDescPos + 2 <= progDescEnd && progDescPos < data.size()) {
            uint8_t descriptorTag = data[progDescPos];
            uint8_t descriptorLength = data[progDescPos + 1];

            // std::cout << "[M2TSReader]   Program descriptor tag: 0x" << std::hex << (int)descriptorTag
            //           << std::dec << " length: " << (int)descriptorLength << std::endl;

            if (descriptorTag == 0x7A) {
                // std::cout << "[M2TSReader] *** Found MVC descriptor (0x7A) at PROGRAM level! ***" << std::endl;
            }

            progDescPos += 2 + descriptorLength;
        }
    } else {
        // std::cout << "[M2TSReader] No program-level descriptors" << std::endl;
    }

    // Parse stream entries
    // std::cout << "[M2TSReader] === Elementary streams ===" << std::endl;
    size_t pos = 12 + programInfoLength;
    int streamCount = 0;

    while (pos + 5 <= 3 + sectionLength - 4) {
        uint8_t streamType = data[pos];
        uint16_t elementaryPid = ((data[pos + 1] & 0x1F) << 8) | data[pos + 2];
        uint16_t esInfoLength = ((data[pos + 3] & 0x0F) << 8) | data[pos + 4];

        streamCount++;
        // std::cout << "[M2TSReader] Stream #" << streamCount << ": type=0x" << std::hex << (int)streamType
        //           << std::dec << ", PID=0x" << std::hex << elementaryPid << std::dec
        //           << ", esInfoLength=" << esInfoLength << std::endl;

        prog->streamPids[elementaryPid] = streamType;

        // CRITICAL: Parse ES descriptors to detect MVC (Blu-ray 3D single-PID mode)
        // Blu-ray 3D uses descriptor tag 0x7A (MVC extension descriptor)
        bool hasMvcDescriptor = false;
        size_t descPos = pos + 5;
        size_t descEnd = descPos + esInfoLength;

        // DEBUG: Log ALL descriptor parsing, not just H.264
        if (esInfoLength > 0) {
            // std::cout << "[M2TSReader]   Parsing " << esInfoLength << " bytes of ES descriptors:" << std::endl;

            while (descPos + 2 <= descEnd && descPos < data.size()) {
                uint8_t descriptorTag = data[descPos];
                uint8_t descriptorLength = data[descPos + 1];

                // std::cout << "[M2TSReader]     Descriptor tag: 0x" << std::hex << (int)descriptorTag
                //           << std::dec << " length: " << (int)descriptorLength << std::endl;

                if (descriptorTag == 0x7A) {  // MVC extension descriptor
                    hasMvcDescriptor = true;
                    // std::cout << "[M2TSReader] *** Found MVC descriptor (0x7A) on PID 0x" << std::hex
                    //           << elementaryPid << std::dec << " - Blu-ray 3D detected! ***" << std::endl;
                    break;
                }

                descPos += 2 + descriptorLength;
            }
        } else {
            // std::cout << "[M2TSReader]   No ES descriptors for this stream" << std::endl;
        }

        if (hasMvcDescriptor) {
            prog->mvcStreams[elementaryPid] = true;
        }

        if (streamType == STREAM_TYPE_H264) {
            // std::cout << "[M2TSReader] Found H.264 stream (PID: 0x" << std::hex
            //           << elementaryPid << std::dec;
            if (hasMvcDescriptor) {
                // std::cout << ", MVC interleaved";
            }
            // std::cout << ")" << std::endl;
        } else if (streamType == STREAM_TYPE_MVC) {
            // std::cout << "[M2TSReader] Found MVC stream (PID: 0x" << std::hex
            //           << elementaryPid << std::dec << ")" << std::endl;
        }

        pos += 5 + esInfoLength;
    }
}

std::vector<uint16_t> M2TSReader::getVideoPids() const {
    std::vector<uint16_t> pids;
    for (const auto& prog : programs_) {
        for (const auto& [pid, streamType] : prog.streamPids) {
            if (streamType == STREAM_TYPE_H264 || streamType == STREAM_TYPE_MVC) {
                pids.push_back(pid);
            }
        }
    }
    return pids;
}

bool M2TSReader::seek(uint64_t bytePosition) {
    if (!file_.is_open()) {
        return false;
    }

    // Clear EOF/fail flags before seeking - CRITICAL for robust seeking
    file_.clear();

    // Align to packet boundary
    uint64_t alignedPos = (bytePosition / packetSize_) * packetSize_;
    file_.seekg(alignedPos, std::ios::beg);
    // Invalidate the read buffer so the next packet refills from the new position.
    bufPos_ = 0;
    bufLen_ = 0;
    bufFileStart_ = alignedPos;

    bool success = file_.good();
    if (!success) {
        std::cerr << "[M2TSReader] Seek to " << alignedPos << " failed" << std::endl;
    } else if (bytePosition == 0) {
        // Log only rewind to 0 to avoid spam
        std::cout << "[M2TSReader] Rewind to start (0)" << std::endl;
    }
    return success;
}

uint64_t M2TSReader::tell() {
    if (!file_.is_open()) {
        return 0;
    }
    // Logical read position (accounts for unconsumed bytes in io_buffer_).
    return bufFileStart_ + bufPos_;
}

} // namespace mvc_demux
