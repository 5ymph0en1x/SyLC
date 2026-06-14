#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_set>

namespace mvc_demux {

// NAL Unit Types (from H.264/AVC and MVC specifications)
enum class NALUnitType : uint8_t {
    Unspecified = 0,
    CodedSliceNonIDR = 1,
    CodedSliceDataPartA = 2,
    CodedSliceDataPartB = 3,
    CodedSliceDataPartC = 4,
    CodedSliceIDR = 5,
    SEI = 6,
    SPS = 7,
    PPS = 8,
    AccessUnitDelimiter = 9,
    EndOfSequence = 10,
    EndOfStream = 11,
    FillerData = 12,
    SPSExtension = 13,
    PrefixNAL = 14,
    SubsetSPS = 15,           // MVC: Subset Sequence Parameter Set
    DepthParameterSet = 16,
    Reserved17 = 17,
    Reserved18 = 18,
    AuxiliarySlice = 19,
    SliceExtension = 20,      // MVC: Coded slice extension (dependent view)
    SliceExtensionDepth = 21  // MVC: Depth view extension
};

// Stream type identification
enum class StreamType {
    Unknown,
    BaseAVC,      // Base H.264/AVC view
    MVCDependent  // MVC dependent view
};

// NAL Unit structure
struct NALUnit {
    NALUnitType type;
    StreamType streamType;
    const uint8_t* data;
    size_t size;
    bool isMVC;
    uint8_t spsId;  // For PPS and slices, track which SPS they reference
};

// H.264/MVC NAL Unit Parser
class H264NALParser {
public:
    H264NALParser();
    ~H264NALParser();

    std::vector<NALUnit> parseBuffer(const uint8_t* buffer, size_t size);
    NALUnit parseUnit(const uint8_t* nalData, size_t nalSize);
    StreamType identifyStreamType(const NALUnit& nal);

    // Find start code prefix length (3 or 4 bytes)
    // Marked inline for performance in hot parsing path
    static inline int findStartCodePrefixLen(const uint8_t* data, size_t size) {
        if (size < 3) return 0;

        if (size >= 4) {
            uint32_t word;
            std::memcpy(&word, data, 4);
            if (word == 0x01000000u || word == 0x00000001u) {
                return 4;
            }
        }

        if (data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01) {
            return 3;
        }

        return 0;
    }

    // Extract SPS ID from PPS NAL unit
    static uint8_t extractSPSIDFromPPS(const uint8_t* data, size_t size);

    // Check if SEI is MVC-related
    static bool isMVCSEI(const uint8_t* data, size_t size);

    // Get MVC SPS IDs that have been detected
    const std::vector<uint8_t>& getMVCSPSIDs() const { return mvcSPSIDsVec_; }

private:
    std::unordered_set<uint8_t> mvcSPSIDs_;  // Fast lookup for MVC SPS IDs
    std::vector<uint8_t> mvcSPSIDsVec_;       // Vector version for API compatibility
    std::vector<uint8_t> buffer_;             // Internal buffer for incomplete NAL units
};

// Helper: Read Exponential-Golomb coded value (used in H.264)
uint32_t readExpGolomb(const uint8_t* data, size_t size, size_t& bitPos);

} // namespace mvc_demux
