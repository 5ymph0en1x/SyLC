#include "ssif_parser.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace mvc_demux {

// SSIF packet size (M2TS packets are 192 bytes with 4-byte timestamp)
constexpr size_t SSIF_PACKET_SIZE = 192;

SSIFParser::SSIFParser() : parsed_(false) {
}

SSIFParser::~SSIFParser() {
}

std::string SSIFParser::detectSSIFPath(const std::string& m2tsPath) {
    // Convert BDMV/STREAM/00000.m2ts -> BDMV/STREAM/SSIF/00000.ssif

    // Find "STREAM" in path
    size_t streamPos = m2tsPath.rfind("STREAM");
    if (streamPos == std::string::npos) {
        // Try lowercase
        streamPos = m2tsPath.rfind("stream");
    }

    if (streamPos == std::string::npos) {
        return "";
    }

    // Find the filename (last component)
    size_t lastSlash = m2tsPath.find_last_of("/\\");
    if (lastSlash == std::string::npos) {
        return "";
    }

    std::string filename = m2tsPath.substr(lastSlash + 1);

    // Replace .m2ts with .ssif
    size_t extPos = filename.rfind(".m2ts");
    if (extPos == std::string::npos) {
        extPos = filename.rfind(".M2TS");
    }

    if (extPos != std::string::npos) {
        filename = filename.substr(0, extPos) + ".ssif";
    }

    // Build SSIF path: BDMV/STREAM/SSIF/filename.ssif
    std::string basePath = m2tsPath.substr(0, streamPos + 6); // Include "STREAM"
    char sep = (m2tsPath.find('\\') != std::string::npos) ? '\\' : '/';

    return basePath + sep + "SSIF" + sep + filename;
}

bool SSIFParser::hasSSIF(const std::string& m2tsPath) {
    std::string ssifPath = detectSSIFPath(m2tsPath);
    if (ssifPath.empty()) {
        return false;
    }

    std::ifstream file(ssifPath, std::ios::binary);
    return file.good();
}

bool SSIFParser::parse(const std::string& ssifPath) {
    std::cout << "[SSIFParser] Parsing SSIF file: " << ssifPath << std::endl;

    std::ifstream file(ssifPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[SSIFParser] Failed to open SSIF file" << std::endl;
        return false;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "[SSIFParser] SSIF file size: " << fileSize << " bytes" << std::endl;

    // STREAMING FIX: For large SSIF files (>100MB), use streaming mode
    // instead of reading entire file into memory
    constexpr size_t STREAMING_THRESHOLD = 100 * 1024 * 1024;  // 100 MB
    constexpr size_t HEADER_PROBE_SIZE = 64 * 1024;  // 64 KB for header

    std::vector<uint8_t> data;
    bool isLargeSSIF = (fileSize > STREAMING_THRESHOLD);

    if (isLargeSSIF) {
        // Large SSIF file (likely interleaved video data)
        // Only read header to probe format and find extent table
        std::cout << "[SSIFParser] Large SSIF detected (" << (fileSize / (1024*1024))
                  << " MB), using streaming mode" << std::endl;

        data.resize(HEADER_PROBE_SIZE);
        if (!file.read(reinterpret_cast<char*>(data.data()), HEADER_PROBE_SIZE)) {
            std::cerr << "[SSIFParser] Failed to read SSIF header" << std::endl;
            return false;
        }

        // Store actual file size for extent calculations
        info_.totalSize = fileSize;
        info_.ssifPath = ssifPath;

        // For large SSIF files, we use direct streaming mode
        // Generate synthetic extents that alternate between base/dependent
        // based on typical Blu-ray 3D interleaving pattern
        return parseStreamingSSIF(file, fileSize);
    }

    // Small SSIF: read entire file (original behavior)
    data.resize(fileSize);
    if (!file.read(reinterpret_cast<char*>(data.data()), fileSize)) {
        std::cerr << "[SSIFParser] Failed to read SSIF file" << std::endl;
        return false;
    }

    // Store path
    info_.ssifPath = ssifPath;

    // Check file signature
    if (fileSize < 16) {
        std::cerr << "[SSIFParser] SSIF file too small" << std::endl;
        return false;
    }

    // Debug: Print first 64 bytes to understand format
    std::cout << "[SSIFParser] File header (first 64 bytes):" << std::endl;
    std::cout << "[SSIFParser]   ";
    for (size_t i = 0; i < std::min(size_t(64), fileSize); i++) {
        printf("%02X ", data[i]);
        if ((i + 1) % 16 == 0 && i + 1 < std::min(size_t(64), fileSize)) {
            std::cout << std::endl << "[SSIFParser]   ";
        }
    }
    std::cout << std::endl;

    // Real Blu-ray 3D SSIF format:
    // Offset 0: "SSIF" magic (4 bytes)
    // Offset 4: Version (4 bytes)
    // Offset 8: Number of extents (4 bytes, big-endian)
    // Offset 12: Start of extent table
    // Each extent: 12 bytes (stream_id + start_packet + num_packets + flags)

    bool success = false;

    // Check for "SSIF" magic
    if (data[0] == 'S' && data[1] == 'S' && data[2] == 'I' && data[3] == 'F') {
        std::cout << "[SSIFParser] Found SSIF magic signature" << std::endl;

        uint32_t numExtents = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11];
        std::cout << "[SSIFParser] Extent count from header: " << numExtents << std::endl;

        if (numExtents > 0 && numExtents < 1000000) {
            success = parseSSIFv2(data);
        }
    }

    if (!success) {
        std::cout << "[SSIFParser] No SSIF magic found, trying heuristic parsing..." << std::endl;
        success = parseSSIFv2(data);
    }

    if (!success) {
        std::cout << "[SSIFParser] Trying v1 format..." << std::endl;
        success = parseSSIFv1(data);
    }

    if (success) {
        deriveStreamPaths();
        parsed_ = true;

        std::cout << "[SSIFParser] Successfully parsed SSIF:" << std::endl;
        std::cout << "[SSIFParser]   Base stream: " << info_.baseStreamPath << std::endl;
        std::cout << "[SSIFParser]   Dependent stream: " << info_.dependentStreamPath << std::endl;
        std::cout << "[SSIFParser]   Extents: " << info_.extents.size() << std::endl;
        std::cout << "[SSIFParser]   Virtual size: " << info_.totalSize << " bytes" << std::endl;

        // Sanity check
        if (info_.extents.size() < 10 || info_.totalSize < 1000000) {
            std::cerr << "[SSIFParser] WARNING: Parsed extent count or size seems too small!" << std::endl;
            std::cerr << "[SSIFParser] This may indicate incorrect parsing. Please check SSIF format." << std::endl;
        }
    }

    return success;
}

bool SSIFParser::parseSSIFv1(const std::vector<uint8_t>& data) {
    // Simple format: extent count followed by extent records
    // Each extent: 1 byte stream_id + 8 bytes offset + 8 bytes length

    if (data.size() < 4) return false;

    size_t pos = 0;

    // Skip potential header (look for reasonable extent count)
    while (pos + 20 < data.size()) {
        uint32_t extentCount = (data[pos] << 24) | (data[pos+1] << 16) |
                               (data[pos+2] << 8) | data[pos+3];

        if (extentCount > 0 && extentCount < 100000) {
            size_t expectedSize = pos + 4 + (extentCount * 17);
            if (expectedSize <= data.size()) {
                std::cout << "[SSIFParser] Found extent count: " << extentCount << " at offset " << pos << std::endl;
                pos += 4;

                uint64_t currentOffset = 0;
                for (uint32_t i = 0; i < extentCount; i++) {
                    if (pos + 17 > data.size()) break;

                    Extent ext;
                    ext.streamFileId = data[pos++];

                    ext.startByte = 0;
                    for (int j = 0; j < 8; j++) {
                        ext.startByte = (ext.startByte << 8) | data[pos++];
                    }

                    ext.length = 0;
                    for (int j = 0; j < 8; j++) {
                        ext.length = (ext.length << 8) | data[pos++];
                    }

                    ext.outputOffset = currentOffset;
                    currentOffset += ext.length;

                    info_.extents.push_back(ext);
                }

                info_.totalSize = currentOffset;
                return !info_.extents.empty();
            }
        }
        pos++;
    }

    return false;
}

bool SSIFParser::parseSSIFv2(const std::vector<uint8_t>& data) {
    // Blu-ray SSIF format (based on packet numbers, not byte offsets)
    // Each extent specifies: stream_id, start_packet, num_packets

    if (data.size() < 20) return false;

    // Look for extent table pattern
    // Typically: extent_count (4 bytes) followed by extent records
    // Each extent: stream_id (1 byte) + start_packet (4 bytes) + num_packets (4 bytes)

    size_t pos = 0;

    // Try different offsets for the extent count
    while (pos + 100 < data.size()) {
        uint32_t extentCount = (data[pos] << 24) | (data[pos+1] << 16) |
                               (data[pos+2] << 8) | data[pos+3];

        if (extentCount > 0 && extentCount < 100000) {
            size_t expectedSize = pos + 4 + (extentCount * 9);
            if (expectedSize <= data.size() + 1000) { // Allow some tolerance
                std::cout << "[SSIFParser] Trying packet-based format with " << extentCount
                          << " extents at offset " << pos << std::endl;

                pos += 4;
                std::vector<Extent> tempExtents;
                uint64_t currentOffset = 0;

                for (uint32_t i = 0; i < extentCount && pos + 9 <= data.size(); i++) {
                    Extent ext;
                    ext.streamFileId = data[pos++];

                    uint32_t startPacket = (data[pos] << 24) | (data[pos+1] << 16) |
                                          (data[pos+2] << 8) | data[pos+3];
                    pos += 4;

                    uint32_t numPackets = (data[pos] << 24) | (data[pos+1] << 16) |
                                         (data[pos+2] << 8) | data[pos+3];
                    pos += 4;

                    // Validate
                    if (ext.streamFileId > 1) break;
                    if (numPackets == 0 || numPackets > 1000000) break;

                    ext.startByte = static_cast<uint64_t>(startPacket) * SSIF_PACKET_SIZE;
                    ext.length = static_cast<uint64_t>(numPackets) * SSIF_PACKET_SIZE;
                    ext.outputOffset = currentOffset;

                    currentOffset += ext.length;
                    tempExtents.push_back(ext);
                }

                if (tempExtents.size() == extentCount && !tempExtents.empty()) {
                    info_.extents = std::move(tempExtents);
                    info_.totalSize = currentOffset;
                    std::cout << "[SSIFParser] Successfully parsed " << info_.extents.size()
                              << " extents (packet-based)" << std::endl;
                    return true;
                }
            }
        }
        pos++;
    }

    return false;
}

void SSIFParser::deriveStreamPaths() {
    // Derive base and dependent stream paths from SSIF path
    // BDMV/STREAM/SSIF/00000.ssif -> BDMV/STREAM/00000.m2ts and 00001.m2ts

    std::string ssifPath = info_.ssifPath;

    // Find SSIF directory
    size_t ssifDirPos = ssifPath.rfind("SSIF");
    if (ssifDirPos == std::string::npos) {
        ssifDirPos = ssifPath.rfind("ssif");
    }

    if (ssifDirPos == std::string::npos) {
        std::cerr << "[SSIFParser] Cannot derive stream paths - no SSIF directory found" << std::endl;
        return;
    }

    // Get the STREAM directory path (parent of SSIF)
    char sep = (ssifPath.find('\\') != std::string::npos) ? '\\' : '/';
    std::string streamDir = ssifPath.substr(0, ssifDirPos - 1);

    // Extract filename (e.g., "00000.ssif")
    size_t fileStart = ssifPath.find_last_of("/\\") + 1;
    std::string filename = ssifPath.substr(fileStart);

    // Replace .ssif with .m2ts
    size_t extPos = filename.rfind(".ssif");
    if (extPos == std::string::npos) {
        extPos = filename.rfind(".SSIF");
    }

    if (extPos != std::string::npos) {
        std::string baseName = filename.substr(0, extPos);

        // Base stream: same name (e.g., 00000.m2ts)
        info_.baseStreamPath = streamDir + sep + baseName + ".m2ts";

        // Dependent stream: increment number (e.g., 00001.m2ts)
        // Parse the numeric part
        int streamNum = 0;
        try {
            streamNum = std::stoi(baseName);
        } catch (...) {
            std::cerr << "[SSIFParser] Cannot parse stream number from: " << baseName << std::endl;
            return;
        }

        // Format with leading zeros (5 digits is typical)
        char depName[32];
        snprintf(depName, sizeof(depName), "%05d.m2ts", streamNum + 1);
        info_.dependentStreamPath = streamDir + sep + std::string(depName);
    }
}

int SSIFParser::findExtent(uint64_t offset) const {
    for (size_t i = 0; i < info_.extents.size(); i++) {
        const Extent& ext = info_.extents[i];
        if (offset >= ext.outputOffset && offset < ext.outputOffset + ext.length) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool SSIFParser::parseStreamingSSIF(std::ifstream& file, size_t fileSize) {
    // STREAMING MODE: For large SSIF files (40GB+), we don't parse extent tables
    // Instead, we set up the demuxer to read directly from the SSIF file
    // and separate base/dependent NAL units on-the-fly based on NAL type

    std::cout << "[SSIFParser] Streaming mode: direct SSIF read enabled" << std::endl;

    // Derive companion M2TS paths (for fallback or audio)
    deriveStreamPaths();

    // Mark this as a streaming SSIF (single interleaved file)
    // Create a single extent covering the entire file
    Extent fullExtent;
    fullExtent.streamFileId = 255;  // Special marker: interleaved SSIF
    fullExtent.startByte = 0;
    fullExtent.length = fileSize;
    fullExtent.outputOffset = 0;
    info_.extents.push_back(fullExtent);

    info_.totalSize = fileSize;
    parsed_ = true;
    isStreamingMode_ = true;

    std::cout << "[SSIFParser] Streaming SSIF ready:" << std::endl;
    std::cout << "[SSIFParser]   File size: " << (fileSize / (1024*1024)) << " MB" << std::endl;
    std::cout << "[SSIFParser]   Base stream (fallback): " << info_.baseStreamPath << std::endl;
    std::cout << "[SSIFParser]   Dependent stream: " << info_.dependentStreamPath << std::endl;

    return true;
}

} // namespace mvc_demux
