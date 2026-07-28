#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <deque>
#include <mutex>

#include "file_byte_source.h"

namespace mvc_demux {

/**
 * StreamTapBuffer — a bounded, thread-safe tee of raw bytes in file order.
 *
 * SyLC Cast audio: the cast's AudioTap used to open its OWN reader on the
 * media to decode audio — on an optical disc that is a THIRD concurrent
 * reader on one head (video demuxer + mpv audio + tap) and causes the
 * measured 45-120s seek thrash (8.4s single demux reads, frozen playback).
 * Instead, the M2TSReader tees every byte it already pulls off the disk into
 * this buffer, and the AudioTap demuxes the audio out of THAT: the disc
 * keeps exactly its two established readers.
 *
 * Shared via shared_ptr between the demuxer (Python-facing reads) and the
 * reader (appends): the buffer outlives either side, so a teardown on one
 * thread can never dangle the other.
 */
struct StreamTapBuffer {
    std::mutex m;
    size_t capacity = 0;                       // bytes; 0 = disabled
    size_t size = 0;                           // bytes currently queued
    uint64_t dropped = 0;                      // bytes discarded on overflow
    std::deque<std::vector<uint8_t>> chunks;

    void append(const uint8_t* data, size_t n) {
        if (!data || n == 0) return;
        std::lock_guard<std::mutex> lk(m);
        if (capacity == 0) return;
        chunks.emplace_back(data, data + n);
        size += n;
        while (size > capacity && !chunks.empty()) {   // drop-OLDEST: freshest audio wins
            size -= chunks.front().size();
            dropped += chunks.front().size();
            chunks.pop_front();
        }
    }

    std::vector<uint8_t> pop(size_t maxBytes) {
        std::vector<uint8_t> out;
        std::lock_guard<std::mutex> lk(m);
        while (!chunks.empty() && out.size() < maxBytes) {
            auto& front = chunks.front();
            if (out.size() + front.size() <= maxBytes) {
                out.insert(out.end(), front.begin(), front.end());
                size -= front.size();
                chunks.pop_front();
            } else {
                size_t take = maxBytes - out.size();
                out.insert(out.end(), front.begin(), front.begin() + take);
                front.erase(front.begin(), front.begin() + take);
                size -= take;
            }
        }
        return out;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(m);
        chunks.clear();
        size = 0;
    }
};

/**
 * M2TS/TS Reader
 * Parses MPEG-2 Transport Stream files (used in Blu-ray 3D)
 *
 * Format:
 * - M2TS: 192-byte packets (188 bytes + 4 bytes timecode)
 * - TS: 188-byte packets
 *
 * This reader extracts PES packets from TS streams and identifies
 * video PIDs for H.264/MVC content.
 */
class M2TSReader {
public:
    M2TSReader();
    ~M2TSReader();

    // Open M2TS or TS file
    bool open(const std::string& filePath);

    // Close file
    void close();

    // Check if file is open
    bool isOpen() const { return static_cast<bool>(source_); }

    // TS Packet (188 or 192 bytes)
    struct TSPacket {
        uint8_t syncByte;           // Always 0x47
        bool transportErrorIndicator;
        bool payloadUnitStartIndicator;
        bool transportPriority;
        uint16_t pid;               // Packet ID
        uint8_t scramblingControl;
        bool adaptationFieldExists;
        bool payloadExists;
        uint8_t continuityCounter;
        std::vector<uint8_t> payload;
        uint64_t pcr;               // Program Clock Reference (if present)
    };

    // Program info from PAT/PMT
    struct ProgramInfo {
        uint16_t programNumber;
        uint16_t pmtPid;
        std::map<uint16_t, uint8_t> streamPids; // PID -> stream_type
        std::map<uint16_t, bool> mvcStreams;     // PID -> has MVC descriptor (0x7A)
        bool hasMvcProgramDescriptor = false;    // program-level descriptor 0x7A
    };

    // Read next TS packet
    bool readPacket(TSPacket& packet);

    // Get detected packet size (188 or 192)
    int getPacketSize() const { return packetSize_; }

    // Get video PIDs (H.264 base and MVC extension)
    std::vector<uint16_t> getVideoPids() const;

    // Get program information
    const std::vector<ProgramInfo>& getPrograms() const { return programs_; }

    // Seek to byte position
    bool seek(uint64_t bytePosition);

    // Get current file position
    uint64_t tell();

    // Get file size
    uint64_t getFileSize() const { return fileSize_; }

    // DIAG: count of resync events (forward-scans on sync-byte loss)
    long getResyncCount() const { return resync_count_; }

    // Attach/detach the cast-audio stream tap (see StreamTapBuffer above).
    // Thread-safe against a concurrent readBuffered() refill on the demux
    // thread; pass nullptr to detach. seek() clears the attached buffer so
    // the tee never carries stale pre-seek bytes.
    void setStreamTap(std::shared_ptr<StreamTapBuffer> tap) {
        std::lock_guard<std::mutex> lk(tapMutex_);
        tap_ = std::move(tap);
    }

private:
    std::mutex tapMutex_;                       // guards tap_ (the pointer, not the buffer)
    std::shared_ptr<StreamTapBuffer> tap_;      // null when no cast audio tap is active

    std::shared_ptr<StreamTapBuffer> currentTap() {
        std::lock_guard<std::mutex> lk(tapMutex_);
        return tap_;
    }

    std::shared_ptr<FileByteSource> source_;
    // EXPLICIT big-chunk read buffer. pubsetbuf is unreliable on MSVC (the demuxer still
    // read ~2 MB/s while the SAME disc does 18 MB/s with 1 MB reads — measured), so we buffer
    // ourselves: one big source read fills io_buffer_, packets are served from it. Few large
    // sequential reads instead of thousands of tiny ones = optical drive streams at full speed.
    std::vector<char> io_buffer_;
    size_t bufPos_ = 0;            // consumed offset within io_buffer_
    size_t bufLen_ = 0;            // valid bytes currently in io_buffer_
    uint64_t bufFileStart_ = 0;    // file byte offset of io_buffer_[0]
    uint64_t fileSize_;
    int packetSize_;  // 188 or 192
    long resync_count_ = 0;  // DIAG: resync events

    // OPTIMIZATION: Pre-allocated buffer to avoid malloc() on every packet read
    std::vector<uint8_t> packetBuffer_;

    // Serve n bytes from io_buffer_, refilling with one big file_.read() when drained.
    // Returns false only at genuine EOF. Keeps bufFileStart_/bufPos_ = logical read position.
    bool readBuffered(uint8_t* dst, size_t n);

    // PAT/PMT parsing state
    std::vector<ProgramInfo> programs_;
    std::map<uint16_t, std::vector<uint8_t>> pesBuffers_; // PID -> accumulated PES data

    // Auto-detect packet size (188 or 192)
    bool detectPacketSize();

    // Forward-resync after a sync-byte loss (e.g. non-TS gaps at SSIF interleave
    // boundaries). Scans ahead for the next position where the packet cadence
    // resumes, repositions, and loads packetBuffer_ with that packet. Bounded;
    // returns false only at genuine EOF / unrecoverable stream.
    bool resyncToNextPacket();

    // Parse PAT (Program Association Table)
    void parsePAT(const std::vector<uint8_t>& data);

    // Parse PMT (Program Map Table)
    void parsePMT(const std::vector<uint8_t>& data, uint16_t pid);

    // Parse PSI (Program Specific Information) section
    bool parsePSISection(const TSPacket& packet, std::vector<uint8_t>& section);
};

} // namespace mvc_demux
