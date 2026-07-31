// Minimal fork-join helper shared by the synth3d CPU stages (optical flow,
// FB-consistency, DepthStabilizer reproject/step, luma pass). The index
// range is split into `parallel_chunk_count(total, max_threads)` contiguous
// [begin, end) chunks in index order; the body receives its chunk index so
// chunk-ordered reductions stay deterministic (the sequential-iteration
// winner of a min/max reduce is always reproduced). Threads are spawned per
// call: at ~10-20 service cycles/s the spawn cost (~0.1 ms) is noise next
// to the tens of ms each call parallelizes away.
#pragma once

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

inline int parallel_chunk_count(int total, int max_threads) {
    return std::max(1, std::min(max_threads, total));
}

inline void parallel_chunks(int total, int max_threads,
                            const std::function<void(int, int, int)>& body) {
    const int threads = parallel_chunk_count(total, max_threads);
    const int chunk = (total + threads - 1) / threads;
    if (threads == 1) {
        body(0, 0, total);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads) - 1);
    for (int t = 1; t < threads; ++t) {
        const int begin = t * chunk;
        const int end = std::min(total, begin + chunk);
        if (begin >= end) break;
        pool.emplace_back(body, t, begin, end);
    }
    body(0, 0, std::min(total, chunk));
    for (auto& th : pool) th.join();
}
