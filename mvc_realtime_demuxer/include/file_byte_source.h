#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace mvc_demux {

/**
 * Random-access byte source for Blu-ray transport streams stored as files
 * (.m2ts, .ssif, or anything inside a mounted ISO).
 *
 * Positioned reads only: the caller supplies the absolute offset, so several
 * consumers can share one instance without fighting over a file cursor. Reads
 * are serialised because a single std::ifstream owns that cursor internally.
 *
 * This deliberately handles CLEAR streams only. SyLC does not decrypt copy
 * protection; a protected disc must be read through software that is licensed
 * to do so, and the resulting backup plays here like any other file.
 */
class FileByteSource {
public:
    static std::shared_ptr<FileByteSource> open(
        const std::string& filePath, std::string* error = nullptr);

    ~FileByteSource();

    FileByteSource(const FileByteSource&) = delete;
    FileByteSource& operator=(const FileByteSource&) = delete;

    /** Reads up to `length` bytes at `offset`; returns the byte count read. */
    std::size_t readAt(
        uint64_t offset,
        uint8_t* destination,
        std::size_t length,
        const std::atomic<bool>* consumerCancelled = nullptr);

    uint64_t size() const noexcept { return fileSize_; }
    std::string lastError() const noexcept;

private:
    explicit FileByteSource(std::filesystem::path filePath);
    bool initialize();
    void setError(std::string message);

    std::filesystem::path filePath_;
    std::ifstream file_;
    uint64_t fileSize_ = 0;

    mutable std::mutex ioMutex_;
    mutable std::mutex errorMutex_;
    std::string lastError_;
};

} // namespace mvc_demux
