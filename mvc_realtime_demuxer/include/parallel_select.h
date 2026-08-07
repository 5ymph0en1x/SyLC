// Exact order statistics and order-preserving filtering of float arrays, in
// parallel, with the same bitwise-determinism guarantee parallel_chunks.h
// makes for per-pixel writes -- extended here to two operations that LOOK
// like reductions but are not:
//
//  - select_ranks(): the k-th smallest element of an array is a VALUE, not an
//    accumulation. Any correct selection algorithm returns the same value, so
//    replacing std::nth_element with a counting (radix) selection changes
//    nothing but the wall time. The parallel step sums integer histogram
//    counts, and integer addition is associative: the totals -- and therefore
//    the selected value -- are identical at any worker count and any chunk
//    grid. This is precisely the property the affine/Welford REDUCTIONS in
//    the stabilizer do not have (float association order changes rounding),
//    which is why they stay serial and this does not.
//
//  - filter_ge(): compaction that preserves source order. Chunks write their
//    matches at exclusive-prefix offsets and are concatenated in index order,
//    which reproduces the sequential result element-for-element at any worker
//    count.
//
// Ordering detail: keys order floats like operator< for every ordered value;
// -0.0f sorts below +0.0f where operator< calls them equal. The two patterns
// compare and subtract identically downstream, so a selection that lands on
// the tie is still value-identical. NaNs (which make nth_element's comparator
// contract void anyway) sort deterministically by payload here.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "parallel_chunks.h"

namespace sylc_select {

inline uint32_t float_key(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

inline float key_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// Reused across calls: sized for chunk-count x rank-count x bucket histograms
// so the per-map cost is bandwidth, not allocation.
struct Scratch {
    std::vector<uint32_t> hist;
    std::vector<int> chunk_counts;
};

// out[j] = the ranks[j]-th smallest element of data[0..n). ranks[j] < n.
// Three counting passes (11/11/10 key bits), all ranks resolved in the same
// passes; ~3n key transforms + 3n*nranks compares, spread over the pool.
inline void select_ranks(const float* data, int n, int max_threads,
                         const size_t* ranks, int nranks, float* out,
                         Scratch& scratch) {
    if (n <= 0 || nranks <= 0 || nranks > 8) return;  // contract: <= 8 ranks
    static const int kShift[3] = {21, 10, 0};
    static const int kBits[3] = {11, 11, 10};

    // prefix[j]: the key bits already decided (above the current field).
    uint32_t prefix[8] = {};
    int64_t residual[8] = {};
    for (int j = 0; j < nranks; ++j)
        residual[j] = static_cast<int64_t>(ranks[j]);

    for (int level = 0; level < 3; ++level) {
        const int shift = kShift[level];
        const int buckets = 1 << kBits[level];
        const uint32_t mask = static_cast<uint32_t>(buckets - 1);
        const int high_shift = shift + kBits[level];   // 32 on level 0
        const int nchunks = parallel_chunk_count(n, max_threads);
        // Level 0 has no prefix to match -- every rank reads the same
        // histogram, so build it once instead of nranks times.
        const int slots = (level == 0) ? 1 : nranks;

        scratch.hist.assign(
            static_cast<size_t>(nchunks) * slots * buckets, 0u);
        uint32_t* hist = scratch.hist.data();

        parallel_chunks(n, max_threads, [&](int chunk, int begin, int end) {
            uint32_t* h = hist +
                static_cast<size_t>(chunk) * slots * buckets;
            if (level == 0) {
                for (int i = begin; i < end; ++i)
                    ++h[(float_key(data[i]) >> shift) & mask];
            } else {
                for (int i = begin; i < end; ++i) {
                    const uint32_t k = float_key(data[i]);
                    const uint32_t high = k >> high_shift;
                    const uint32_t bucket = (k >> shift) & mask;
                    for (int j = 0; j < nranks; ++j)
                        if (high == prefix[j]) ++h[j * buckets + bucket];
                }
            }
        });

        // Serial walk: slots x buckets x nchunks integer adds, exact.
        for (int j = 0; j < nranks; ++j) {
            const int slot = (level == 0) ? 0 : j;
            int64_t r = residual[j];
            uint32_t found = 0;
            for (int b = 0; b < buckets; ++b) {
                int64_t count = 0;
                for (int c = 0; c < nchunks; ++c)
                    count += hist[(static_cast<size_t>(c) * slots + slot)
                                  * buckets + b];
                if (r < count) { found = static_cast<uint32_t>(b); break; }
                r -= count;
            }
            prefix[j] = (prefix[j] << kBits[level]) | found;
            residual[j] = r;
        }
    }

    for (int j = 0; j < nranks; ++j) out[j] = key_float(prefix[j]);
}

// dst = the elements of src whose paired gate value is >= threshold, in
// source order -- the exact sequence the sequential filter produces.
inline void filter_ge(const float* src, const float* gate, int n,
                      float threshold, int max_threads,
                      std::vector<float>& dst, Scratch& scratch) {
    if (n <= 0) { dst.clear(); return; }
    const int nchunks = parallel_chunk_count(n, max_threads);
    scratch.chunk_counts.assign(nchunks, 0);
    int* counts = scratch.chunk_counts.data();

    parallel_chunks(n, max_threads, [&](int chunk, int begin, int end) {
        int local = 0;
        for (int i = begin; i < end; ++i) local += (gate[i] >= threshold);
        counts[chunk] = local;
    });

    // Exclusive prefix in chunk order = each chunk's write offset.
    int total = 0;
    for (int c = 0; c < nchunks; ++c) {
        const int count = counts[c];
        counts[c] = total;
        total += count;
    }
    dst.resize(static_cast<size_t>(total));
    float* out = dst.data();

    parallel_chunks(n, max_threads, [&](int chunk, int begin, int end) {
        float* w = out + counts[chunk];
        for (int i = begin; i < end; ++i)
            if (gate[i] >= threshold) *w++ = src[i];
    });
}

}  // namespace sylc_select
