// Standalone timing harness for DepthStabilizer::step().
//
// The stabilizer is the largest and most volatile term in the depth cycle
// (measured 2026-08-06 on live logs: stab_ms median 23.8 ms, peaks 39.2 ms,
// against 10.5 ms of TensorRT inference), and it depends on nothing but
// <vector>/<cmath> and the chunk pool -- so it can be timed without the app,
// a GPU, or the models.
//
// The header guarantees step() is bitwise identical at any worker_threads
// value, so sweeping the thread count measures the parallel fraction directly:
// whatever refuses to speed up is the serial part.
//
// Build (MinGW, matches nothing shipped -- use for RATIOS):
//   g++ -std=c++17 -O3 -I mvc_realtime_demuxer/include \
//       tools_dev/bench_stabilizer.cpp mvc_realtime_demuxer/src/depth_stabilizer.cpp \
//       -o bench_stab.exe
// Build (MSVC, matches the shipped .pyd -- use for ABSOLUTE numbers):
//   cl /std:c++17 /O2 /Oi /Ot /arch:AVX2 /EHsc /I mvc_realtime_demuxer\include \
//      tools_dev\bench_stabilizer.cpp mvc_realtime_demuxer\src\depth_stabilizer.cpp \
//      /Fe:bench_stab.exe

#include "depth_stabilizer.h"
#include "parallel_select.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

// Deterministic, dependency-free noise so runs are comparable.
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    float unit() { return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f); }
};

// A plausible depth field: a few smooth lobes plus a foreground slab, drifting
// with `phase` so the temporal paths (motion, history, snap) do real work
// instead of settling into a constant.
void fill_depth(std::vector<float>& dst, size_t w, size_t h, float phase, Rng& rng) {
    for (size_t y = 0; y < h; ++y) {
        const float fy = static_cast<float>(y) / static_cast<float>(h);
        for (size_t x = 0; x < w; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(w);
            float v = 0.45f
                    + 0.25f * std::sin(6.0f * (fx + 0.15f * phase))
                    + 0.15f * std::cos(4.0f * (fy - 0.10f * phase));
            // A moving foreground subject.
            const float cx = 0.35f + 0.20f * std::sin(phase);
            const float dx = fx - cx, dy = fy - 0.55f;
            if (dx * dx * 4.0f + dy * dy < 0.02f) v += 0.30f;
            v += 0.01f * (rng.unit() - 0.5f);
            dst[y * w + x] = std::min(1.0f, std::max(0.0f, v));
        }
    }
}

void fill_unit(std::vector<float>& dst, float base, Rng& rng) {
    for (auto& v : dst) v = std::min(1.0f, std::max(0.0f, base + 0.10f * (rng.unit() - 0.5f)));
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// FNV-1a over the reference map, printed per grid: lets two BUILDS be compared
// for bit-exactness (the in-run verdict column only compares thread counts
// within one build).
uint32_t fnv1a(const std::vector<uint16_t>& v) {
    uint32_t h = 2166136261u;
    for (uint16_t x : v) {
        h = (h ^ (x & 0xffu)) * 16777619u;
        h = (h ^ (x >> 8)) * 16777619u;
    }
    return h;
}

// The stabilizer replaces std::nth_element with sylc_select::select_ranks on
// the strength of a mathematical claim: a k-th order statistic is the same
// VALUE whatever algorithm finds it, at any worker count. Claims get checked.
// Adversarial value patterns, boundary ranks, several thread counts -- each
// selection must equal what nth_element returns on a private copy, and the
// parallel filter must reproduce the sequential sequence exactly.
int selection_selftest() {
    int failures = 0;
    sylc_select::Scratch scratch;
    Rng rng(777u);
    const size_t sizes[] = {1, 2, 64, 999, 243432};
    for (size_t n : sizes) {
        for (int pattern = 0; pattern < 7; ++pattern) {
            std::vector<float> data(n);
            for (size_t i = 0; i < n; ++i) {
                switch (pattern) {
                    case 0: data[i] = rng.unit(); break;                      // uniform
                    case 1: data[i] = 0.5f; break;                            // all equal
                    case 2: data[i] = (i & 1) ? 0.25f : 0.75f; break;         // two values
                    case 3: data[i] = static_cast<float>(i) / n; break;       // sorted
                    case 4: data[i] = static_cast<float>(n - i) / n; break;   // reversed
                    case 5: data[i] = rng.unit() - 0.5f; break;               // negatives
                    case 6: data[i] = (i % 3 == 0) ? 0.0f
                                    : (i % 3 == 1) ? -0.0f : 1e-12f; break;   // +/-0, tiny
                }
            }
            const size_t ranks[] = {0, n - 1, n / 2,
                                    static_cast<size_t>(0.02 * (n - 1)),
                                    static_cast<size_t>(0.98 * (n - 1))};
            float expected[5];
            for (int r = 0; r < 5; ++r) {
                std::vector<float> copy(data);
                std::nth_element(copy.begin(), copy.begin() + ranks[r], copy.end());
                expected[r] = copy[ranks[r]];
            }
            for (int threads : {1, 3, 16}) {
                float got[5];
                sylc_select::select_ranks(data.data(), static_cast<int>(n),
                                          threads, ranks, 5, got, scratch);
                for (int r = 0; r < 5; ++r) {
                    if (got[r] != expected[r]) {
                        ++failures;
                        std::printf("  SELFTEST FAIL n=%zu pattern=%d threads=%d "
                                    "rank=%zu got=%.9g want=%.9g\n",
                                    n, pattern, threads, ranks[r],
                                    static_cast<double>(got[r]),
                                    static_cast<double>(expected[r]));
                    }
                }
            }
            // filter_ge vs the sequential filter, element-for-element.
            std::vector<float> gate(n);
            for (size_t i = 0; i < n; ++i) gate[i] = rng.unit();
            std::vector<float> want;
            want.reserve(n);
            for (size_t i = 0; i < n; ++i)
                if (gate[i] >= 0.30f) want.push_back(data[i]);
            for (int threads : {1, 3, 16}) {
                std::vector<float> got;
                sylc_select::filter_ge(data.data(), gate.data(),
                                       static_cast<int>(n), 0.30f, threads,
                                       got, scratch);
                if (got != want) {
                    ++failures;
                    std::printf("  SELFTEST FAIL filter n=%zu pattern=%d "
                                "threads=%d size=%zu want=%zu\n",
                                n, pattern, threads, got.size(), want.size());
                }
            }
        }
    }
    std::printf("selection selftest: %s\n",
                failures ? "FAILED" : "PASS (5 sizes x 7 patterns x 3 thread "
                                      "counts, ranks + filter vs std)");
    return failures;
}

struct Grid { const char* name; size_t w, h; };

void run_grid(const Grid& g, int iters) {
    const size_t n = g.w * g.h;
    std::printf("\n=== %s (%zux%zu = %zu pixels), %d iterations\n",
                g.name, g.w, g.h, n, iters);

    std::vector<float> raw(n), motion(n), confidence(n), boundary(n);
    std::vector<float> flow_x(n, 0.35f), flow_y(n, -0.20f), reliability(n, 0.85f);
    std::vector<uint16_t> out(n);

    std::printf("%8s %10s %10s %8s %12s\n",
                "threads", "step ms", "reproj ms", "speedup", "output");
    std::printf("%s\n", std::string(54, '-').c_str());

    double base_step = 0.0;
    // Every thread count must produce the SAME quantized map. The header
    // promises it for per-pixel writes; this checks it instead of trusting it.
    std::vector<uint16_t> reference;
    for (int threads : {1, 2, 4, 8, 16}) {
        DepthStabilizer stab(n);
        stab.worker_threads = threads;
        stab.set_source_dt_ms(41.7f);
        stab.set_update_dt_ms(41.7f);

        Rng rng(12345u);
        // Prime: the first step() takes the !primed_ branch.
        fill_depth(raw, g.w, g.h, 0.0f, rng);
        fill_unit(confidence, 0.80f, rng);
        fill_unit(motion, 0.20f, rng);
        fill_unit(boundary, 0.10f, rng);
        stab.step(raw.data(), out.data(), motion.data(), 0.0f,
                  confidence.data(), boundary.data());

        std::vector<double> step_ms, reproj_ms;
        for (int i = 0; i < iters; ++i) {
            const float phase = 0.05f * static_cast<float>(i + 1);
            fill_depth(raw, g.w, g.h, phase, rng);
            fill_unit(motion, 0.20f, rng);

            auto t0 = Clock::now();
            stab.reproject(flow_x.data(), flow_y.data(), reliability.data(), g.w, g.h);
            auto t1 = Clock::now();
            stab.step(raw.data(), out.data(), motion.data(), 0.02f,
                      confidence.data(), boundary.data());
            auto t2 = Clock::now();

            reproj_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            step_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }

        const double s = median(step_ms);
        const double r = median(reproj_ms);
        if (threads == 1) base_step = s;

        const char* verdict;
        if (reference.empty()) { reference = out; verdict = "reference"; }
        else if (out == reference) { verdict = "bit-exact"; }
        else {
            size_t diff = 0;
            for (size_t i = 0; i < out.size(); ++i) diff += (out[i] != reference[i]);
            static char msg[32];
            std::snprintf(msg, sizeof(msg), "DIFF %zu", diff);
            verdict = msg;
        }
        std::printf("%8d %10.2f %10.2f %8.2fx %12s\n",
                    threads, s, r, base_step / s, verdict);
    }
    std::printf("  reference map hash: %08x\n", fnv1a(reference));
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 40;
    std::printf("DepthStabilizer::step() timing harness\n");
    std::printf("(a 23.976 fps frame is 41.7 ms; the whole depth cycle must fit inside it)\n");

    if (selection_selftest() != 0) return 1;

    // The two grids the player actually selects: the Scope rectangle in use on
    // the measured session, and the square Quality grid.
    const Grid grids[] = {
        {"Scope   756x322", 756, 322},
        {"Quality 756x756", 756, 756},
    };
    for (const auto& g : grids) run_grid(g, iters);
    return 0;
}
