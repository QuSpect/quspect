#include <cmath>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <vector>

#include "quspect_native/common/parallel.hpp"
#include "quspect_native/kernels/sampling.hpp"

namespace quspect::native::kernels {

/// 累积概率前缀和（按下标顺序、双精度累加；顺序固定，采样结果与线程数无关）
template <typename FP>
std::vector<double> cumulative_probabilities(const std::complex<FP>* state, uint64_t dim) {
    std::vector<double> prefix_sum(dim);
    double accum = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        accum += static_cast<double>(std::norm(state[i]));
        prefix_sum[i] = accum;
    }
    return prefix_sum;
}

/// 在前缀和中二分查找随机数对应的 basis index
uint64_t sample_from_cumulative(const std::vector<double>& cumulative, double random01) {
    if (cumulative.empty()) {
        return 0;
    }
    const double r = random01 * cumulative.back();  // 总和应该接近 1
    const auto it = std::lower_bound(cumulative.begin(), cumulative.end(), r);
    const uint64_t idx = static_cast<uint64_t>(std::distance(cumulative.begin(), it));
    return std::min<uint64_t>(idx, cumulative.size() - 1);
}

// 显式实例化
template std::vector<double> cumulative_probabilities<float>(const std::complex<float>*, uint64_t);
template std::vector<double> cumulative_probabilities<double>(const std::complex<double>*, uint64_t);

}  // namespace quspect::native::kernels

namespace quspect::native::kernels::scalar {

/// 终端采样: 从状态向量的概率分布中采样一个 basis index
///
/// 算法：
/// 1. 构建累积概率前缀和
/// 2. 用随机数在前缀和中二分查找
///
/// 每次调用都重新构建前缀和（O(dim)）。对同一个状态多次采样时，应当只构建一次
/// cumulative_probabilities，再逐次调用 sample_from_cumulative（结果相同）。
template <typename FP>
uint64_t sample_from_statevector(const std::complex<FP>* state, uint64_t dim, double random01) {
    return sample_from_cumulative(cumulative_probabilities(state, dim), random01);
}

/// 单比特测量坍缩 kernel
///
/// 根据测量结果将状态向量坍缩到对应子空间：
///   - outcome == 0: 将所有 qubit==1 的振幅清零，除以 sqrt(p0)
///   - outcome == 1: 将所有 qubit==0 的振幅清零，除以 sqrt(1-p0)
///
/// Args:
///     state: 状态向量 (in/out)
///     dim: 状态向量维度 (2^n)
///     qubit: 被测量的 qubit 索引
///     outcome: 测量结果 (0 或 1)
template <typename FP>
void collapse_qubit(std::complex<FP>* state, uint64_t dim, uint32_t qubit, int outcome) {
    using Complex = std::complex<FP>;

    const uint64_t mask = 1ULL << qubit;
    const uint64_t keep_bit = outcome ? mask : 0;
    const int num_threads = common::ThreadsForDim(dim);

    // 第一遍：清零不匹配的振幅，同时累加保留振幅的平方模（每个迭代只写自己的振幅）
    const auto acc = common::BlockedReduce<1>(
        static_cast<int64_t>(dim), num_threads, [&](int64_t begin, int64_t end, double* partial) {
            double local = 0.0;
            for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
                if ((i & mask) == keep_bit) {
                    local += static_cast<double>(std::norm(state[i]));
                } else {
                    state[i] = Complex(0, 0);
                }
            }
            partial[0] += local;
        });
    const FP keep_prob = static_cast<FP>(acc[0]);

    // 第二遍：归一化（除以 sqrt(keep_prob)）
    if (keep_prob > 0) {
        const FP inv_norm = FP(1) / std::sqrt(keep_prob);
        common::ParallelForRange(static_cast<int64_t>(dim), num_threads, [&](int64_t begin, int64_t end) {
            for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
                state[i] *= inv_norm;
            }
        });
    }
}

// 显式实例化 — terminal sampling
template uint64_t sample_from_statevector<float>(const std::complex<float>*, uint64_t, double);
template uint64_t sample_from_statevector<double>(const std::complex<double>*, uint64_t, double);

// 显式实例化 — collapse
template void collapse_qubit<float>(std::complex<float>*, uint64_t, uint32_t, int);
template void collapse_qubit<double>(std::complex<double>*, uint64_t, uint32_t, int);

}  // namespace quspect::native::kernels::scalar
