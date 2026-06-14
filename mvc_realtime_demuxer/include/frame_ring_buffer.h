#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <utility>
#include <cstring>

namespace mvc_demux {

// Lightweight view returned to Python without copying buffers.
struct FrameBufferView {
    const uint8_t* basePtr{nullptr};
    size_t baseSize{0};
    const uint8_t* depPtr{nullptr};
    size_t depSize{0};
    uint64_t timestamp{0};
    bool isKeyframe{false};
    uint64_t sequence{0};
};

// Fixed-size ring buffer for access units (base + dependent).
// Pre-allocates buffers to avoid churn when processing very large files.
class FrameRingBuffer : public std::enable_shared_from_this<FrameRingBuffer> {
public:
    FrameRingBuffer(size_t capacity = 120, size_t maxFrameBytes = 4 * 1024 * 1024);

    // Push one pair of views. Returns false if the slot is still in use.
    bool push(const std::vector<uint8_t>& base,
              const std::vector<uint8_t>& dep,
              uint64_t timestamp,
              bool isKeyframe);

    // Pop oldest frame. Returns slot index so the caller can release it.
    bool pop(FrameBufferView& view, size_t& slotIndex);

    // Release a slot that was returned by pop().
    void release(size_t slotIndex);

    // Metrics
    size_t size() const;
    size_t capacity() const { return capacity_; }
    uint64_t dropped() const { return dropped_; }
    size_t maxFrameBytes() const { return maxFrameBytes_; }

    // Guard used by Python capsules to release slots automatically.
    struct SlotGuard {
        std::shared_ptr<FrameRingBuffer> owner;
        size_t index;
        explicit SlotGuard(std::shared_ptr<FrameRingBuffer> o, size_t i)
            : owner(std::move(o)), index(i) {}
        ~SlotGuard() {
            if (owner) {
                owner->release(index);
            }
        }
    };

private:
    struct Slot {
        std::vector<uint8_t> base;
        std::vector<uint8_t> dep;
        size_t baseSize{0};
        size_t depSize{0};
        uint64_t timestamp{0};
        bool isKeyframe{false};
        std::atomic<bool> inUse{false};

        Slot() = default;
        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;
        Slot(Slot&&) = default;
        Slot& operator=(Slot&&) = default;
    };

    void copyIntoStorage(std::vector<uint8_t>& target,
                         const std::vector<uint8_t>& src,
                         size_t& outSize);

    size_t capacity_;
    size_t maxFrameBytes_;
    std::vector<Slot> slots_;
    size_t head_{0};
    size_t tail_{0};
    size_t size_{0};
    uint64_t sequence_{0};
    uint64_t dropped_{0};
    mutable std::mutex mutex_;
};

} // namespace mvc_demux
