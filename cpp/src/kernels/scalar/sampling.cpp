#include <cmath>

#include <complex>
#include <cstdint>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

/// 计算 qubit=0 边缘概率: P(q=0) = sum_{i: bit_q=0} |psi[i]|^2
template <typename FP>
FP compute_probability_zero(const std::complex<FP>* state, uint64_t dim, uint32_t qubit) {
    // 只遍历 qubit 位为 0 的半空间；BlockedReduce 保证结果与线程数无关（测量采样可复现）。
    const uint64_t low_mask = (1ULL << qubit) - 1;
    const auto acc = common::BlockedReduce<1>(
        static_cast<int64_t>(dim >> 1), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end, double* partial) {
            double local = 0.0;
            for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                const uint64_t i = (base & low_mask) | ((base & ~low_mask) << 1);
                local += static_cast<double>(std::norm(state[i]));
            }
            partial[0] += local;
        });
    return static_cast<FP>(acc[0]);
}

// 显式实例化
template float compute_probability_zero<float>(const std::complex<float>*, uint64_t, uint32_t);
template double compute_probability_zero<double>(const std::complex<double>*, uint64_t, uint32_t);

}  // namespace quspect::native::kernels::scalar
