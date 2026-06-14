#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace mvc_demux {

/**
 * SSIF (Stereo Interleaved File) Parser
 *
 * Blu-ray 3D uses SSIF files to describe how to interleave two separate
 * M2TS streams (left eye and right eye) for 3D playback.
 *
 * Structure:
 * - BDMV/STREAM/00000.m2ts (base view - left eye)
 * - BDMV/STREAM/00001.m2ts (dependent view - right eye)
 * - BDMV/STREAM/SSIF/00000.ssif (interleaving instructions)
 */

class SSIFParser {
public:
    /**
     * Extent: defines a continuous block to read from a source stream
     */
    struct Extent {
        uint8_t streamFileId;    // 0 = base (left), 1 = dependent (right)
        uint64_t startByte;      // Starting byte offset in source file
        uint64_t length;         // Number of bytes to read
        uint64_t outputOffset;   // Offset in the virtual interleaved stream
    };

    struct SSIFInfo {
        std::string ssifPath;
        std::string baseStreamPath;      // Path to left eye M2TS
        std::string dependentStreamPath; // Path to right eye M2TS
        std::vector<Extent> extents;
        uint64_t totalSize;              // Total size of virtual interleaved stream
    };

    SSIFParser();
    ~SSIFParser();

    /**
     * Parse an SSIF file
     * @param ssifPath Path to the .ssif file
     * @return true if parsing succeeded
     */
    bool parse(const std::string& ssifPath);

    /**
     * Get the parsed SSIF information
     */
    const SSIFInfo& getInfo() const { return info_; }

    /**
     * Find which extent contains a given byte offset
     * @param offset Byte offset in the virtual interleaved stream
     * @return Index of the extent, or -1 if not found
     */
    int findExtent(uint64_t offset) const;

    /**
     * Auto-detect SSIF file from a M2TS path
     * Given BDMV/STREAM/00000.m2ts, returns BDMV/STREAM/SSIF/00000.ssif
     */
    static std::string detectSSIFPath(const std::string& m2tsPath);

    /**
     * Check if an SSIF file exists for this M2TS
     */
    static bool hasSSIF(const std::string& m2tsPath);

    /**
     * Check if parser is in streaming mode (large SSIF file)
     */
    bool isStreamingMode() const { return isStreamingMode_; }

private:
    bool parseSSIFv1(const std::vector<uint8_t>& data);
    bool parseSSIFv2(const std::vector<uint8_t>& data);
    bool parseStreamingSSIF(std::ifstream& file, size_t fileSize);
    void deriveStreamPaths();

    SSIFInfo info_;
    bool parsed_;
    bool isStreamingMode_ = false;
};

} // namespace mvc_demux
