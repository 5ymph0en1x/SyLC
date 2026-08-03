// Minimal fork-join helper shared by the synth3d CPU stages (optical flow,
// FB-consistency, DepthStabilizer reproject/step, luma pass, boundary).
// The index range is split into `parallel_chunk_count(total, max_threads)`
// contiguous [begin, end) chunks in index order; the body receives its chunk
// index so chunk-ordered reductions stay deterministic (the sequential-
// iteration winner of a min/max reduce is always reproduced).
//
// Round 6 (2026-08-02): threads are NO LONGER spawned per call. The original
// spawn-per-call design assumed "~0.1 ms of spawn is noise next to the tens
// of ms each call parallelizes away" — true at 8 threads and 3 big regions,
// false once the flow stage runs concurrently with inference and the
// per-cycle region count grew: ~100 kernel thread creations per 40 ms cycle,
// contending with each other, measurably REGRESSED the very stages they
// parallelize. A single process-wide pool of parked workers now serves every
// call; the chunk-index -> [begin, end) mapping is unchanged, so outputs
// remain bitwise identical to the sequential path (pinned by
// test_flow_parallel.py). The CALLER participates in its own job (a
// 1-thread pool degenerates to the plain loop), and concurrent jobs (e.g.
// forward and backward flow) share the same workers.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

inline int parallel_chunk_count(int total, int max_threads) {
    return std::max(1, std::min(max_threads, total));
}

namespace sylc_chunks_detail {

struct Job {
    const std::function<void(int, int, int)>* body = nullptr;
    int total = 0;
    int chunk = 0;
    int nchunks = 0;
    int next = 0;                 // guarded by Pool::mtx_ (claims are locked)
    std::atomic<int> done{0};
};

class Pool {
public:
    static Pool& instance() {
        static Pool pool;
        return pool;
    }

    void run(Job& job) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            jobs_.push_back(&job);
        }
        cv_.notify_all();
        // The caller claims chunks from its OWN job. Claims happen UNDER the
        // mutex: a claimed-but-unfinished chunk keeps done < nchunks, so the
        // job (a stack object of this very frame) cannot be torn down while
        // any thread still works on it — the lifetime proof of this design.
        for (;;) {
            int index;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (job.next >= job.nchunks) break;
                index = job.next++;
            }
            execute_chunk(job, index);
        }
        std::unique_lock<std::mutex> lk(mtx_);
        done_cv_.wait(lk, [&] {
            return job.done.load(std::memory_order_acquire) >= job.nchunks;
        });
        jobs_.erase(std::remove(jobs_.begin(), jobs_.end(), &job),
                    jobs_.end());
    }

private:
    Pool() {
        const unsigned hc = std::max(
            2u, std::thread::hardware_concurrency());
        const unsigned count = std::min(16u, hc);
        workers_.reserve(count);
        for (unsigned i = 0; i < count; ++i)
            workers_.emplace_back([this] { worker(); });
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& worker_thread : workers_)
            if (worker_thread.joinable()) worker_thread.join();
    }

    bool claimable_locked() const {
        for (const Job* job : jobs_)
            if (job->next < job->nchunks) return true;
        return false;
    }

    void worker() {
        for (;;) {
            Job* job = nullptr;
            int index = 0;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || claimable_locked(); });
                if (stop_) return;
                for (Job* candidate : jobs_) {
                    if (candidate->next < candidate->nchunks) {
                        job = candidate;
                        index = candidate->next++;
                        break;
                    }
                }
                if (!job) continue;
            }
            execute_chunk(*job, index);
        }
    }

    void execute_chunk(Job& job, int index) {
        const int begin = index * job.chunk;
        const int end = std::min(job.total, begin + job.chunk);
        // Everything needed after the completing done-increment is read
        // BEFORE it: that increment may free the job (see run()).
        const int nchunks = job.nchunks;
        if (begin < end) (*job.body)(index, begin, end);
        if (job.done.fetch_add(1, std::memory_order_acq_rel) + 1 == nchunks) {
            std::lock_guard<std::mutex> lk(mtx_);
            done_cv_.notify_all();
        }
    }

    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::vector<Job*> jobs_;
    std::vector<std::thread> workers_;
    bool stop_ = false;
};

}  // namespace sylc_chunks_detail

inline void parallel_chunks(int total, int max_threads,
                            const std::function<void(int, int, int)>& body) {
    const int threads = parallel_chunk_count(total, max_threads);
    if (threads <= 1 || total <= 0) {
        if (total > 0) body(0, 0, total);
        return;
    }
    const int chunk = (total + threads - 1) / threads;
    sylc_chunks_detail::Job job;
    job.body = &body;
    job.total = total;
    job.chunk = chunk;
    job.nchunks = threads;
    sylc_chunks_detail::Pool::instance().run(job);
}
