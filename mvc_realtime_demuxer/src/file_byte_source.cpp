#include "file_byte_source.h"

#include <algorithm>
#include <system_error>

namespace mvc_demux {

FileByteSource::FileByteSource(std::filesystem::path filePath)
    : filePath_(std::move(filePath)) {}

FileByteSource::~FileByteSource() {
    std::lock_guard<std::mutex> lock(ioMutex_);
    if (file_.is_open()) {
        file_.close();
    }
}

std::shared_ptr<FileByteSource> FileByteSource::open(
    const std::string& filePath, std::string* error) {
    std::error_code ec;
    auto canonical = std::filesystem::weakly_canonical(
        std::filesystem::path(filePath), ec);
    if (ec) {
        canonical = std::filesystem::path(filePath);
    }

    std::shared_ptr<FileByteSource> source(new FileByteSource(canonical));
    if (!source->initialize()) {
        if (error) *error = source->lastError();
        return nullptr;
    }
    if (error) error->clear();
    return source;
}

bool FileByteSource::initialize() {
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(filePath_, ec);
    if (ec) {
        setError("cannot stat " + filePath_.string() + ": " + ec.message());
        return false;
    }
    fileSize_ = static_cast<uint64_t>(bytes);

    file_.open(filePath_, std::ios::binary);
    if (!file_.is_open()) {
        setError("cannot open " + filePath_.string());
        return false;
    }
    return true;
}

std::size_t FileByteSource::readAt(
    uint64_t offset,
    uint8_t* destination,
    std::size_t length,
    const std::atomic<bool>* consumerCancelled) {
    if (!destination || length == 0 || offset >= fileSize_) return 0;

    const std::size_t requested = static_cast<std::size_t>(
        std::min<uint64_t>(length, fileSize_ - offset));

    std::lock_guard<std::mutex> lock(ioMutex_);
    if (!file_.is_open()) return 0;

    std::size_t total = 0;
    while (total < requested) {
        if (consumerCancelled &&
            consumerCancelled->load(std::memory_order_acquire)) {
            break;
        }
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(offset + total), std::ios::beg);
        file_.read(reinterpret_cast<char*>(destination + total),
                   static_cast<std::streamsize>(requested - total));
        // A short read is legal (optical media in particular), so loop instead
        // of treating it as end of stream; only a zero-byte read is terminal.
        const auto got = static_cast<std::size_t>(file_.gcount());
        if (got == 0) {
            if (total == 0) {
                setError("read failed at byte " + std::to_string(offset));
            }
            break;
        }
        total += got;
    }
    return total;
}

void FileByteSource::setError(std::string message) {
    std::lock_guard<std::mutex> lock(errorMutex_);
    lastError_ = std::move(message);
}

std::string FileByteSource::lastError() const noexcept {
    try {
        std::lock_guard<std::mutex> lock(errorMutex_);
        return lastError_;
    } catch (...) {
        return {};
    }
}

} // namespace mvc_demux
