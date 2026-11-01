#include "quspect_native/common/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace quspect::native::common {

namespace {

/// SetNumThreads 设置的线程数；0 表示未设置（使用 OpenMP 默认值）。
std::atomic<int> g_requested_threads{0};

}  // namespace

bool OpenMPEnabled() {
#ifdef QSPECT_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

int GetNumThreads() {
#ifdef QSPECT_HAS_OPENMP
    const int requested = g_requested_threads.load(std::memory_order_relaxed);
    if (requested > 0) {
        return requested;
    }
    const int max_threads = omp_get_max_threads();
    return max_threads > 0 ? max_threads : 1;
#else
    return 1;
#endif
}

void SetNumThreads(int num_threads) {
    if (num_threads < 1) {
        throw std::invalid_argument("num_threads must be >= 1");
    }
    g_requested_threads.store(num_threads, std::memory_order_relaxed);
}

int ThreadsForDim(uint64_t dim) {
    // 每个线程至少分到 kParallelMinDim 个振幅，避免小状态上 fork/join 开销超过计算量
    const uint64_t max_useful = dim / kParallelMinDim;
    const int requested = GetNumThreads();
    if (max_useful <= 1) {
        return 1;
    }
    return max_useful < static_cast<uint64_t>(requested) ? static_cast<int>(max_useful) : requested;
}

int CompileThreads() {
    if (const char* env = std::getenv("QUSPECT_COMPILE_THREADS")) {
        char* end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end != env && *end == '\0' && v >= 1) {
            return static_cast<int>(std::min<long>(v, 64));
        }
        return 1;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return static_cast<int>(std::min(8u, hw == 0 ? 1u : hw));
}

void RunTasks(const std::vector<std::function<void()>>& tasks, int threads) {
    const std::size_t count = tasks.size();
    std::vector<std::exception_ptr> errors(count);
    auto run = [&](std::size_t k) {
        try {
            tasks[k]();
        } catch (...) {
            errors[k] = std::current_exception();
        }
    };
    const std::size_t workers = std::min<std::size_t>(count, static_cast<std::size_t>(std::max(threads, 1)));
    if (workers <= 1) {
        for (std::size_t k = 0; k < count; ++k) {
            run(k);
        }
    } else {
        // 任务按下标轮流取走：线程 i 依次执行 i、i + workers、……
        auto lane = [&](std::size_t first) {
            for (std::size_t k = first; k < count; k += workers) {
                run(k);
            }
        };
        std::vector<std::thread> pool;
        std::size_t started = 1;
        try {
            for (; started < workers; ++started) {
                pool.emplace_back(lane, started);
            }
        } catch (const std::system_error&) {
            // 不能再创建线程：未分配出去的通道在本线程中执行
        }
        lane(0);
        for (std::size_t first = started; first < workers; ++first) {
            lane(first);
        }
        for (auto& t : pool) {
            t.join();
        }
    }
    for (const auto& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

struct WorkerPool::State {
    std::mutex mutex;
    std::condition_variable start;
    std::condition_variable done;
    std::vector<std::thread> threads;
    // 每次 Run 加一；工作线程据此判断是否有新批次
    std::uint64_t generation = 0;
    bool stop = false;
    const std::function<void(std::size_t, int)>* body = nullptr;
    std::size_t count = 0;
    std::atomic<std::size_t> next{0};
    // 本批次尚未结束的工作线程数
    int active = 0;

    void Drain(int lane) {
        for (;;) {
            const std::size_t k = next.fetch_add(1, std::memory_order_relaxed);
            if (k >= count) {
                return;
            }
            (*body)(k, lane);
        }
    }

    void Work(int lane) {
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                start.wait(lock, [&] { return stop || generation != seen; });
                if (stop) {
                    return;
                }
                seen = generation;
            }
            Drain(lane);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (--active == 0) {
                    done.notify_one();
                }
            }
        }
    }
};

WorkerPool::WorkerPool(int lanes) : state_(new State) {
    for (int lane = 1; lane < lanes; ++lane) {
        try {
            state_->threads.emplace_back([this, lane] { state_->Work(lane); });
        } catch (const std::system_error&) {
            // 不能再创建线程：用已有的通道
            break;
        }
    }
}

WorkerPool::~WorkerPool() {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->stop = true;
    }
    state_->start.notify_all();
    for (auto& t : state_->threads) {
        t.join();
    }
    delete state_;
}

int WorkerPool::lanes() const {
    return static_cast<int>(state_->threads.size()) + 1;
}

void WorkerPool::Run(std::size_t count, const std::function<void(std::size_t, int)>& body) {
    if (count == 0) {
        return;
    }
    if (state_->threads.empty() || count == 1) {
        for (std::size_t k = 0; k < count; ++k) {
            body(k, 0);
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->body = &body;
        state_->count = count;
        state_->next.store(0, std::memory_order_relaxed);
        state_->active = static_cast<int>(state_->threads.size());
        ++state_->generation;
    }
    state_->start.notify_all();
    state_->Drain(0);
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->done.wait(lock, [&] { return state_->active == 0; });
    state_->body = nullptr;
}

}  // namespace quspect::native::common
