#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>

namespace mvc_demux {

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
    bool isOpen() const { return file_.is_open(); }

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

private:
    std::ifstream file_;
    uint64_t fileSize_;
    int packetSize_;  // 188 or 192

    // OPTIMIZATION: Pre-allocated buffer to avoid malloc() on every packet read
    std::vector<uint8_t> packetBuffer_;

    // PAT/PMT parsing state
    std::vector<ProgramInfo> programs_;
    std::map<uint16_t, std::vector<uint8_t>> pesBuffers_; // PID -> accumulated PES data

    // Auto-detect packet size (188 or 192)
    bool detectPacketSize();

    // Parse PAT (Program Association Table)
    void parsePAT(const std::vector<uint8_t>& data);

    // Parse PMT (Program Map Table)
    void parsePMT(const std::vector<uint8_t>& data, uint16_t pid);

    // Parse PSI (Program Specific Information) section
    bool parsePSISection(const TSPacket& packet, std::vector<uint8_t>& section);
};

} // namespace mvc_demux
