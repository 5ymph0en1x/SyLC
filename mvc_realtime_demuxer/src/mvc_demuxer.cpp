#include "mvc_demuxer.h"
#include <fstream>
#include <cstring>
#include <iostream>
#include "frame_ring_buffer.h"

namespace mvc_demux {

// Implementation details (PIMPL pattern)
struct MVCDemuxer::Impl {
    std::ifstream file;
    std::string filePath;
    size_t fileSize;
    size_t currentPos;

    // Buffers for accumulating frame data
    std::vector<uint8_t> baseBuffer;
    std::vector<uint8_t> dependentBuffer;
    uint64_t currentTimestamp;

    // For simplified implementation, we'll read raw H.264 data
    // In production, this should use proper MKV parsing (libmatroska)
    bool readRawData(uint8_t* buffer, size_t size) {
        if (!file.is_open() || file.eof()) return false;
        file.read(reinterpret_cast<char*>(buffer), size);
        return file.gcount() == static_cast<std::streamsize>(size);
    }
};

MVCDemuxer::MVCDemuxer()
    : impl_(std::make_unique<Impl>())
    , isOpen_(false)
{
    videoInfo_ = {0, 0, 0.0, false, 0};
}

MVCDemuxer::~MVCDemuxer() {
    close();
}

bool MVCDemuxer::open(const std::string& filePath) {
    close();

    impl_->filePath = filePath;
    impl_->file.open(filePath, std::ios::binary);

    if (!impl_->file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    // Get file size
    impl_->file.seekg(0, std::ios::end);
    impl_->fileSize = impl_->file.tellg();
    impl_->file.seekg(0, std::ios::beg);
    impl_->currentPos = 0;

    isOpen_ = true;

    // TODO: Parse MKV header to extract video metadata
    // For now, set defaults (caller should provide these externally)
    videoInfo_.width = 1920;
    videoInfo_.height = 1080;
    videoInfo_.fps = 23.976;
    videoInfo_.hasMVC = true;
    videoInfo_.trackCount = 1;

    return true;
}

void MVCDemuxer::close() {
    if (impl_->file.is_open()) {
        impl_->file.close();
    }
    isOpen_ = false;
}

void MVCDemuxer::separateStreams(const std::vector<NALUnit>& nalUnits,
                                 std::vector<uint8_t>& baseStream,
                                 std::vector<uint8_t>& dependentStream) {
    baseStream.clear();
    dependentStream.clear();
    
    // Pre-calculate total size needed to avoid multiple reallocations
    size_t baseSize = 0;
    size_t depSize = 0;
    
    for (const auto& nal : nalUnits) {
        if (nal.size == 0) continue;
        
        size_t nalTotalSize = nal.size + 4; // NAL + start code
        if (nal.streamType == StreamType::BaseAVC || nal.streamType == StreamType::Unknown) {
            baseSize += nalTotalSize;
        } else if (nal.streamType == StreamType::MVCDependent) {
            depSize += nalTotalSize;
        }
    }
    
    // Reserve capacity to avoid reallocations during insertion
    baseStream.reserve(baseSize);
    dependentStream.reserve(depSize);
    
    // Static start code to avoid repeated array creation
    static const uint8_t startCode[4] = {0x00, 0x00, 0x00, 0x01};

    for (const auto& nal : nalUnits) {
        std::vector<uint8_t>* targetStream = nullptr;

        // Route based on stream type
        switch (nal.streamType) {
            case StreamType::BaseAVC:
                targetStream = &baseStream;
                break;

            case StreamType::MVCDependent:
                targetStream = &dependentStream;
                break;

            case StreamType::Unknown:
            default:
                // Default to base stream for unknown types
                targetStream = &baseStream;
                break;
        }

        if (targetStream && nal.size > 0) {
            // Add start code before NAL unit
            targetStream->insert(targetStream->end(), startCode, startCode + 4);

            // Add NAL unit data
            targetStream->insert(targetStream->end(), nal.data, nal.data + nal.size);
        }
    }
}

bool MVCDemuxer::readNextFramePair(FrameData& baseView, FrameData& dependentView) {
    if (!isOpen_) return false;

    // Read a chunk of data
    const size_t chunkSize = 1024 * 1024; // 1 MB chunks
    std::vector<uint8_t> buffer(chunkSize);

    if (impl_->file.eof()) return false;

    impl_->file.read(reinterpret_cast<char*>(buffer.data()), chunkSize);
    size_t bytesRead = impl_->file.gcount();

    if (bytesRead == 0) return false;

    // Parse NAL units from the chunk
    auto nalUnits = nalParser_.parseBuffer(buffer.data(), bytesRead);

    if (nalUnits.empty()) {
        // No NAL units found, try reading more
        return readNextFramePair(baseView, dependentView);
    }

    // Separate into base and dependent streams
    std::vector<uint8_t> baseData, dependentData;
    separateStreams(nalUnits, baseData, dependentData);

    // Create frame data
    baseView.data = std::move(baseData);
    baseView.timestamp = impl_->currentTimestamp;
    baseView.isKeyframe = false;

    dependentView.data = std::move(dependentData);
    dependentView.timestamp = impl_->currentTimestamp;
    dependentView.isKeyframe = false;

    // Check for keyframes (IDR slices)
    for (const auto& nal : nalUnits) {
        if (nal.type == NALUnitType::CodedSliceIDR) {
            baseView.isKeyframe = true;
            dependentView.isKeyframe = true;
            break;
        }
    }

    // Increment timestamp (simplified - should use actual PTS from container)
    impl_->currentTimestamp += static_cast<uint64_t>(1000.0 / videoInfo_.fps);

    return !baseView.data.empty() || !dependentView.data.empty();
}

bool MVCDemuxer::readNextFramePairIntoRing(FrameRingBuffer& buffer) {
    FrameData baseView, dependentView;
    if (!readNextFramePair(baseView, dependentView)) {
        return false;
    }

    const bool keyframe = baseView.isKeyframe || dependentView.isKeyframe;
    buffer.push(baseView.data, dependentView.data, baseView.timestamp, keyframe);
    return true;
}

void MVCDemuxer::setFrameCallback(FrameCallback callback) {
    frameCallback_ = callback;
}

bool MVCDemuxer::processFile() {
    if (!isOpen_ || !frameCallback_) return false;

    impl_->currentTimestamp = 0;

    while (!impl_->file.eof()) {
        FrameData baseView, dependentView;

        if (!readNextFramePair(baseView, dependentView)) {
            break;
        }

        // Deliver via callback
        frameCallback_(baseView, dependentView);
    }

    return true;
}

bool MVCDemuxer::seek(uint64_t timestampMs) {
    // Simplified seek (not accurate without proper MKV index parsing)
    if (!isOpen_) return false;

    // Estimate file position based on timestamp
    double progress = static_cast<double>(timestampMs) / (1000.0 * videoInfo_.fps);
    size_t estimatedPos = static_cast<size_t>(impl_->fileSize * progress);

    impl_->file.seekg(estimatedPos);
    impl_->currentTimestamp = timestampMs;

    return true;
}

void MVCDemuxer::processVideoPacket(const uint8_t* data, size_t size, uint64_t timestamp) {
    // Parse NAL units
    auto nalUnits = nalParser_.parseBuffer(data, size);

    // Separate streams
    std::vector<uint8_t> baseData, dependentData;
    separateStreams(nalUnits, baseData, dependentData);

    // Accumulate in buffers
    impl_->baseBuffer.insert(impl_->baseBuffer.end(), baseData.begin(), baseData.end());
    impl_->dependentBuffer.insert(impl_->dependentBuffer.end(), dependentData.begin(), dependentData.end());
    impl_->currentTimestamp = timestamp;
}

} // namespace mvc_demux
