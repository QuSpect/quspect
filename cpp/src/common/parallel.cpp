#include "quspect_native/common/parallel.hpp"

#include <atomic>
#include <stdexcept>

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

}  // namespace quspect::native::common
