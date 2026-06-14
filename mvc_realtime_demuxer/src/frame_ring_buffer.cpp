#include "frame_ring_buffer.h"

#include <algorithm>
#include <atomic>
#include <cstring>

namespace mvc_demux {

FrameRingBuffer::FrameRingBuffer(size_t capacity, size_t maxFrameBytes)
    : capacity_(capacity ? capacity : 1), // Avoid zero-capacity buffers
      maxFrameBytes_(maxFrameBytes ? maxFrameBytes : 1024 * 1024),
      slots_(capacity_) {
    for (auto& slot : slots_) {
        slot.base.reserve(maxFrameBytes_);
        slot.dep.reserve(maxFrameBytes_);
    }
}

size_t FrameRingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_;
}

bool FrameRingBuffer::push(const std::vector<uint8_t>& base,
                           const std::vector<uint8_t>& dep,
                           uint64_t timestamp,
                           bool isKeyframe) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slots_[head_].inUse.load(std::memory_order_acquire)) {
        // Consumer still holds the slot we would overwrite.
        ++dropped_;
        return false;
    }

    // Drop oldest when full to keep memory bounded.
    if (size_ == capacity_) {
        tail_ = (tail_ + 1) % capacity_;
        --size_;
        ++dropped_;
    }

    Slot& slot = slots_[head_];
    copyIntoStorage(slot.base, base, slot.baseSize);
    copyIntoStorage(slot.dep, dep, slot.depSize);
    slot.timestamp = timestamp;
    slot.isKeyframe = isKeyframe;

    // Triple-buffer commit barrier: ensure all memcpy writes to slot.base /
    // slot.dep are globally visible BEFORE the slot is marked publishable.
    // Without this fence, the consumer thread may observe inUse=false while
    // the YUV payload writes are still in transit through the CPU's store
    // buffer — visible as horizontal tearing on the GPU upload.
    std::atomic_thread_fence(std::memory_order_release);

    slot.inUse.store(false, std::memory_order_release);

    head_ = (head_ + 1) % capacity_;
    ++size_;
    return true;
}

bool FrameRingBuffer::pop(FrameBufferView& view, size_t& slotIndex) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (size_ == 0) {
        return false;
    }

    Slot& slot = slots_[tail_];
    slot.inUse.store(true, std::memory_order_release);

    // Pair with the release fence in push(): guarantee that subsequent reads
    // of slot.base / slot.dep observe the producer's completed writes.
    std::atomic_thread_fence(std::memory_order_acquire);

    view.basePtr = slot.base.data();
    view.baseSize = slot.baseSize;
    view.depPtr = slot.dep.data();
    view.depSize = slot.depSize;
    view.timestamp = slot.timestamp;
    view.isKeyframe = slot.isKeyframe;
    view.sequence = sequence_++;

    slotIndex = tail_;
    tail_ = (tail_ + 1) % capacity_;
    --size_;
    return true;
}

void FrameRingBuffer::release(size_t slotIndex) {
    if (slotIndex >= slots_.size()) {
        return;
    }
    slots_[slotIndex].inUse.store(false, std::memory_order_release);
}

void FrameRingBuffer::copyIntoStorage(std::vector<uint8_t>& target,
                                      const std::vector<uint8_t>& src,
                                      size_t& outSize) {
    const size_t copySize = std::min(src.size(), maxFrameBytes_);
    if (target.capacity() < maxFrameBytes_) {
        target.reserve(maxFrameBytes_);
    }
    // Resize only up to maxFrameBytes_ to avoid runaway allocations.
    target.resize(copySize);
    if (!src.empty() && copySize > 0) {
        std::memcpy(target.data(), src.data(), copySize);
    }
    outSize = copySize;
}

} // namespace mvc_demux
