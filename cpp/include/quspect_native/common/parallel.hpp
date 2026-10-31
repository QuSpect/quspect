// parallel.hpp — OpenMP 并行辅助工具
//
// 所有 OpenMP 调用（omp_* 函数与 #pragma omp）都集中在这里和 src/common/parallel.cpp 中，
// 内核代码只使用 ParallelForRange / BlockedReduce / ThreadsForDim。
//
// 未定义 QSPECT_HAS_OPENMP（构建时未链接 OpenMP）时，所有函数退化为单线程实现：
// ParallelForRange 直接调用一次 body(0, count)，GetNumThreads() 恒为 1。
//
// 兼容性约束：Windows 上 MSVC 的 /openmp 只支持 OpenMP 2.0，因此这里只使用
// `parallel` 指令与 omp_get_thread_num / omp_get_num_threads，不依赖任何 3.0+ 特性。

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef QSPECT_HAS_OPENMP
#    include <omp.h>
#endif

namespace quspect::native::common {

/// 每个线程至少处理的振幅数：维度低于 2 * kParallelMinDim 的状态保持单线程，
/// 更大的状态最多使用 dim / kParallelMinDim 个线程（fork/join 开销与计算量相当时不再增加线程）。
inline constexpr uint64_t kParallelMinDim = uint64_t{1} << 14;

/// BlockedReduce 每个归约块包含的迭代次数。块划分与线程数无关，因此归约结果与线程数无关。
inline constexpr int64_t kReductionBlockSize = int64_t{1} << 12;

/// 构建时是否启用了 OpenMP。
bool OpenMPEnabled();

/// 返回并行区域使用的线程数。
///
/// 未调用 SetNumThreads 时返回 omp_get_max_threads()（遵循 OMP_NUM_THREADS 等标准环境变量）；
/// 未启用 OpenMP 时恒为 1。
int GetNumThreads();

/// 设置后续并行区域使用的线程数（进程级全局设置，对所有调用线程生效）。
///
/// num_threads < 1 时抛出 std::invalid_argument。未启用 OpenMP 时只做参数校验。
void SetNumThreads(int num_threads);

/// 返回处理维度为 dim 的状态向量时应使用的线程数：min(GetNumThreads(), dim / kParallelMinDim)，至少为 1。
int ThreadsForDim(uint64_t dim);

/// 把 [0, count) 切成最多 num_threads 段互不重叠的连续区间，并行调用 body(begin, end)。
///
/// 区间划分与 OpenMP 的 schedule(static) 相同（每个线程一段连续区间）。num_threads <= 1、
/// count <= 1 或未启用 OpenMP 时直接调用一次 body(0, count)。body 不得抛出异常。
template <typename Body>
void ParallelForRange(int64_t count, int num_threads, const Body& body) {
    if (count <= 0) {
        return;
    }
#ifdef QSPECT_HAS_OPENMP
    if (num_threads > 1 && count > 1) {
#    pragma omp parallel num_threads(num_threads)
        {
            const int64_t thread_id = omp_get_thread_num();
            const int64_t team_size = omp_get_num_threads();
            // count <= 2^63 / team_size 时不会溢出；状态向量维度远小于该上限。
            const int64_t begin = count / team_size * thread_id
                                  + (thread_id < count % team_size ? thread_id : count % team_size);
            const int64_t length = count / team_size + (thread_id < count % team_size ? 1 : 0);
            if (length > 0) {
                body(begin, begin + length);
            }
        }
        return;
    }
#else
    (void) num_threads;
#endif
    body(int64_t{0}, count);
}

/// 确定性的并行求和：返回 N 个累加量在 [0, count) 上的总和。
///
/// body(begin, end, acc) 必须把区间 [begin, end) 的贡献累加到 acc[0..N) 中（acc 初始为 0）。
/// [0, count) 按固定的 kReductionBlockSize 分块，各块的部分和最后按块序号串行相加，
/// 因此结果与线程数完全无关（逐位一致），可保证带种子的测量采样在不同线程数下可复现。
/// 复数归约请拆成实部、虚部两个 double 累加量。
template <int N, typename Body>
std::array<double, N> BlockedReduce(int64_t count, int num_threads, const Body& body) {
    std::array<double, N> total{};
    if (count <= 0) {
        return total;
    }
    const int64_t num_blocks = (count + kReductionBlockSize - 1) / kReductionBlockSize;
    if (num_blocks == 1) {
        body(int64_t{0}, count, total.data());
        return total;
    }

    std::vector<double> partials(static_cast<std::size_t>(num_blocks) * N, 0.0);
    double* partial_data = partials.data();
    ParallelForRange(num_blocks, num_threads, [&](int64_t first_block, int64_t last_block) {
        for (int64_t block = first_block; block < last_block; ++block) {
            const int64_t begin = block * kReductionBlockSize;
            const int64_t end = (count - begin < kReductionBlockSize) ? count : begin + kReductionBlockSize;
            body(begin, end, partial_data + block * N);
        }
    });
    for (int64_t block = 0; block < num_blocks; ++block) {
        for (int n = 0; n < N; ++n) {
            total[n] += partial_data[block * N + n];
        }
    }
    return total;
}

}  // namespace quspect::native::common
