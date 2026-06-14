#include "h264_nal_parser.h"
#include <cstring>
#include <algorithm>
#include <iostream>

namespace mvc_demux {

namespace {
constexpr size_t MAX_NAL_UNIT_SIZE = 4 * 1024 * 1024; // 4 MB guardrail for corrupt streams

// Scan forward to the next Annex B start code. Returns size on failure.
[[maybe_unused]] size_t resyncToStartCode(const uint8_t* data, size_t size, size_t offset) {
    if (!data || offset >= size) {
        return size;
    }
    for (size_t i = offset; i + 3 < size; ++i) {
        if (data[i] == 0x00 && data[i + 1] == 0x00 &&
            ((data[i + 2] == 0x01) || (data[i + 2] == 0x00 && data[i + 3] == 0x01))) {
            return i;
        }
    }
    // Also check the last possible 3-byte prefix.
    if (size >= 3 && data[size - 3] == 0x00 && data[size - 2] == 0x00 && data[size - 1] == 0x01) {
        return size - 3;
    }
    return size;
}
} // namespace

// Exponential-Golomb decoding helper (declaration needed for use in this file)
uint32_t readExpGolomb(const uint8_t* data, size_t size, size_t& bitPos);

H264NALParser::H264NALParser() {
}

H264NALParser::~H264NALParser() {
}

NALUnit H264NALParser::parseUnit(const uint8_t* nalData, size_t nalSize) {
    NALUnit nal;
    nal.data = nalData;
    nal.size = nalSize;
    nal.streamType = StreamType::Unknown;
    nal.isMVC = false;
    nal.spsId = 0;

    if (nalSize < 1) {
        return nal;
    }

    uint8_t nalHeader = nalData[0];
    nal.type = static_cast<NALUnitType>(nalHeader & 0x1F);

    // Identify stream type based on NAL type
    switch (nal.type) {
        case NALUnitType::SubsetSPS:
            // MVC Subset SPS - extract SPS ID and mark as MVC
            nal.streamType = StreamType::MVCDependent;
            nal.isMVC = true;
            if (nalSize > 1) {
                // SPS ID is in the next bytes (simplified extraction)
                size_t bitPos = 8; // Skip NAL header
                //uint8_t profileIdc = nalData[1]; // This is part of the following call
                readExpGolomb(nalData, nalSize, bitPos); // profile_idc
                readExpGolomb(nalData, nalSize, bitPos); // constraint_set_flags, level_idc etc.
                nal.spsId = static_cast<uint8_t>(readExpGolomb(nalData, nalSize, bitPos));

                // Track this as an MVC SPS using unordered_set for O(1) insertion check
                if (mvcSPSIDs_.insert(nal.spsId).second) {
                    // New SPS ID was inserted, update the vector for API compatibility
                    mvcSPSIDsVec_.push_back(nal.spsId);
                }
            }
            break;

        case NALUnitType::SliceExtension:
            // MVC coded slice extension (dependent view)
            nal.streamType = StreamType::MVCDependent;
            nal.isMVC = true;
            break;

        case NALUnitType::PrefixNAL:
            // MVC Prefix NAL unit - MUST stay with the following NAL type 20 slices!
            // Contains view_id, temporal_id, anchor_pic_flag etc.
            // If classified as BaseAVC, it gets separated from NAL 20 and causes decoder issues.
            nal.streamType = StreamType::MVCDependent;
            nal.isMVC = true;
            break;

        case NALUnitType::PPS:
            // Check if this PPS references an MVC SPS
            if (nalSize > 1) {
                nal.spsId = extractSPSIDFromPPS(nalData, nalSize);
                // O(1) lookup using unordered_set instead of O(n) vector search
                bool isMvcSps = (mvcSPSIDs_.find(nal.spsId) != mvcSPSIDs_.end());
                if (isMvcSps) {
                    nal.streamType = StreamType::MVCDependent;
                    nal.isMVC = true;
                } else {
                    nal.streamType = StreamType::BaseAVC;
                }
            }
            break;

        case NALUnitType::SEI:
            // Check if this is MVC SEI
            if (isMVCSEI(nalData, nalSize)) {
                nal.streamType = StreamType::MVCDependent;
                nal.isMVC = true;
            } else {
                nal.streamType = StreamType::BaseAVC;
            }
            break;

        case NALUnitType::SPS:
        case NALUnitType::CodedSliceIDR:
        case NALUnitType::CodedSliceNonIDR:
        case NALUnitType::AccessUnitDelimiter:
            // Standard AVC NAL units
            nal.streamType = StreamType::BaseAVC;
            break;

        case NALUnitType::EndOfSequence:
        case NALUnitType::EndOfStream:
            // These can be considered part of the base stream
            nal.streamType = StreamType::BaseAVC;
            break;

        default:
            // Unknown type, default to base stream for safety
            nal.streamType = StreamType::BaseAVC;
            break;
    }
    return nal;
}


// Optimized single-pass NAL parsing with efficient start code search
std::vector<NALUnit> H264NALParser::parseBuffer(const uint8_t* buffer, size_t size) {
    std::vector<NALUnit> nalUnits;
    if (!buffer || size == 0) return nalUnits;
    
    // Reserve space to reduce allocations (typical MVC frame has ~20-50 NAL units)
    nalUnits.reserve(32);

    size_t pos = 0;
    
    // Find first start code
    while (pos < size - 3) {
        int prefixLen = findStartCodePrefixLen(buffer + pos, size - pos);
        if (prefixLen > 0) {
            pos += prefixLen;
            break;
        }
        pos++;
    }
    
    if (pos >= size) return nalUnits;

    size_t nalStart = pos;
    
    // Single-pass search for subsequent start codes
    // Use efficient byte scanning instead of calling findStartCodePrefixLen repeatedly
    while (pos < size - 3) {
        // Fast scan for 0x00 bytes (start codes always begin with 0x00)
        const uint8_t* ptr = buffer + pos;
        const uint8_t* end = buffer + size - 3;
        
        // Look for potential start code pattern
        while (ptr < end && ptr[0] != 0x00) {
            ptr++;
        }
        
        pos = ptr - buffer;
        if (pos >= size - 3) break;
        
        // Check if this is actually a start code
        int prefixLen = findStartCodePrefixLen(buffer + pos, size - pos);
        if (prefixLen > 0) {
            // Found start of next NAL unit
            size_t nalSize = pos - nalStart;
            if (nalSize > 0) {
                if (nalSize > MAX_NAL_UNIT_SIZE) {
                    std::cerr << "[H264NALParser] Oversized NAL (" << nalSize
                              << " bytes). Dropping and attempting resync.\n";
                } else {
                    nalUnits.push_back(parseUnit(buffer + nalStart, nalSize));
                }
            }
            pos += prefixLen;
            nalStart = pos;
        } else {
            pos++;
        }
    }
    
    // Handle last NAL unit
    if (nalStart < size) {
        size_t nalSize = size - nalStart;
        if (nalSize > 0 && nalSize <= MAX_NAL_UNIT_SIZE) {
            nalUnits.push_back(parseUnit(buffer + nalStart, nalSize));
        } else if (nalSize > MAX_NAL_UNIT_SIZE) {
            std::cerr << "[H264NALParser] Dropping tail NAL (" << nalSize
                      << " bytes) that exceeds safety limit.\n";
        }
    }

    return nalUnits;
}

StreamType H264NALParser::identifyStreamType(const NALUnit& nal) {
    return nal.streamType;
}

uint8_t H264NALParser::extractSPSIDFromPPS(const uint8_t* data, size_t size) {
    if (size < 2) return 0;
    size_t bitPos = 8; // Skip NAL header
    readExpGolomb(data, size, bitPos); // Skip PPS ID
    return static_cast<uint8_t>(readExpGolomb(data, size, bitPos));
}

bool H264NALParser::isMVCSEI(const uint8_t* data, size_t size) {
    if (size < 2) return false;
    // SEI messages are a sequence of: payload_type, payload_size, payload_data
    size_t pos = 1; // Start after NAL header
    while(pos < size) {
        uint32_t payloadType = 0;
        uint8_t last_byte = 0xFF;
        while(pos < size && (last_byte = data[pos++]) == 0xFF) {
            payloadType += 255;
        }
        payloadType += last_byte;

        uint32_t payloadSize = 0;
        last_byte = 0xFF;
        while(pos < size && (last_byte = data[pos++]) == 0xFF) {
            payloadSize += 255;
        }
        payloadSize += last_byte;

        // MVC-related SEI payload types (from specification)
        if (payloadType == 15) { // MVC scalable nesting SEI
            return true;
        }
        pos += payloadSize;
    }
    return false;
}

// Optimized Exponential-Golomb decoding with bit manipulation
uint32_t readExpGolomb(const uint8_t* data, size_t size, size_t& bitPos) {
    // Fast path: Read leading zeros count
    int leadingZeros = 0;
    const size_t maxBits = size * 8;
    
    // Read bits more efficiently using byte operations where possible
    while (bitPos < maxBits && leadingZeros < 32) {
        size_t bytePos = bitPos >> 3;  // Division by 8 using bit shift
        size_t bitOffset = 7 - (bitPos & 7);  // Modulo 8 using bit mask
        bool bit = (data[bytePos] >> bitOffset) & 0x01;
        bitPos++;
        if (bit) {
            break;
        }
        leadingZeros++;
    }
    
    if (leadingZeros == 0) {
        return 0;  // Special case: single '1' bit means value 0
    }
    
    // Read the remaining bits to form the value
    uint32_t value = 0;
    for (int i = 0; i < leadingZeros && bitPos < maxBits; i++) {
        size_t bytePos = bitPos >> 3;
        size_t bitOffset = 7 - (bitPos & 7);
        bool bit = (data[bytePos] >> bitOffset) & 0x01;
        bitPos++;
        value = (value << 1) | (bit ? 1 : 0);
    }
    
    // Return ((1 << leadingZeros) - 1) + value
    // Using bit shift instead of pow/multiplication
    return ((1u << leadingZeros) - 1) + value;
}

} // namespace mvc_demux
