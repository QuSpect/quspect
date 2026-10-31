#include <complex>
#include <cstdint>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

using common::ParallelForRange;
using common::ThreadsForDim;

// ============================================================
// CNOT: 受控 X 门
// control_mask = 1ULL << control_qubit
// control_value 决定控制位为 0 还是 1 时触发
// 标准 CNOT: control_value = control_mask (即控制位为 1 时翻转目标位)
// ============================================================

template <typename FP>
void apply_cnot(std::complex<FP>* state, uint64_t dim, uint32_t target, uint64_t control_mask, uint64_t control_value,
                const std::complex<FP>* /*unused*/) {
    const uint64_t tmask = 1ULL << target;

    // 每对 (i, j = i ^ tmask) 只由较小的 i 处理，迭代 i 只写 state[i] 与 state[j]，
    // 不同迭代写入的振幅互不相交，可以任意划分到多个线程。
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            // 如果控制位满足条件
            if ((i & control_mask) == control_value) {
                // 且目标位与另一状态的对应位不同
                uint64_t j = i ^ tmask;
                // 用 canonical ordering 避免重复交换
                if (i < j) {
                    auto tmp = state[i];
                    state[i] = state[j];
                    state[j] = tmp;
                }
            }
        }
    });
}

// ============================================================
// CZ: 受控 Z 门
// 目标 qubit 为 1 且控制位满足条件时乘以 -1
// ============================================================

template <typename FP>
void apply_cz(std::complex<FP>* state, uint64_t dim, uint32_t target, uint64_t control_mask, uint64_t control_value,
              const std::complex<FP>* /*unused*/) {
    const uint64_t tmask = 1ULL << target;

    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            if ((i & control_mask) == control_value && (i & tmask)) {
                state[i] *= std::complex<FP>(-1, 0);
            }
        }
    });
}

// ============================================================
// SWAP: 交换两个 qubit 的状态
// ============================================================

template <typename FP>
void apply_swap(std::complex<FP>* state, uint64_t dim, uint32_t qubit0, uint32_t qubit1) {
    const uint64_t mask0 = 1ULL << qubit0;
    const uint64_t mask1 = 1ULL << qubit1;

    // 只有 (b0, b1) == (0, 1) 的迭代写入，配对 (1, 0) 的迭代跳过，因此写入互不相交。
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            uint64_t b0 = (i >> qubit0) & 1;
            uint64_t b1 = (i >> qubit1) & 1;
            if (b0 == 0 && b1 == 1) {
                uint64_t j = i ^ mask0 ^ mask1;
                auto tmp = state[i];
                state[i] = state[j];
                state[j] = tmp;
            }
        }
    });
}

// 显式实例化
#define INSTANTIATE_TWO(FP)                                                                                            \
    template void apply_cnot<FP>(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);  \
    template void apply_cz<FP>(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);    \
    template void apply_swap<FP>(std::complex<FP>*, uint64_t, uint32_t, uint32_t);

INSTANTIATE_TWO(float)
INSTANTIATE_TWO(double)

#undef INSTANTIATE_TWO

}  // namespace quspect::native::kernels::scalar
