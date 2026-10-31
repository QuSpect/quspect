#include <cmath>

#include <complex>
#include <cstdint>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

using common::ParallelForRange;
using common::ThreadsForDim;

/// 插入零位：将 base 的第 q 位插入 0
/// 例如 insert_zero_bit(0b101, 1) = 0b1001
static uint64_t insert_zero_bit(uint64_t base, uint32_t q) {
    uint64_t low_mask = (1ULL << q) - 1;
    uint64_t high_mask = ~low_mask;
    return (base & low_mask) | ((base & high_mask) << 1);
}

/// 通用单比特门: 对 state[0..dim) 上的 qubit 施加 2x2 矩阵
template <typename FP>
void apply_single_qubit_gate(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    const uint64_t mask = 1ULL << qubit;
    const uint64_t half = dim >> 1;

    const auto& u00 = matrix[0];
    const auto& u01 = matrix[1];
    const auto& u10 = matrix[2];
    const auto& u11 = matrix[3];

    // 每个 base 对应一对互不相交的振幅 (i0, i1)，不同区间之间没有写冲突。
    ParallelForRange(static_cast<int64_t>(half), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
            uint64_t i0 = insert_zero_bit(base, qubit);
            uint64_t i1 = i0 | mask;

            auto a0 = state[i0];
            auto a1 = state[i1];

            state[i0] = u00 * a0 + u01 * a1;
            state[i1] = u10 * a0 + u11 * a1;
        }
    });
}

/// 对 qubit=1 的振幅乘以 phase（Z / S / Sdg / T / Tdg 的公共实现）
template <typename FP>
void apply_phase_on_one(std::complex<FP>* state, uint64_t dim, uint32_t qubit, std::complex<FP> phase) {
    const uint64_t mask = 1ULL << qubit;
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            if (i & mask) {
                state[i] *= phase;
            }
        }
    });
}

// ---- 单比特 named gate 矩阵常量 ----

template <typename FP>
void apply_x(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const std::complex<FP> X[4] = {
        std::complex<FP>(0, 0),
        std::complex<FP>(1, 0),
        std::complex<FP>(1, 0),
        std::complex<FP>(0, 0),
    };
    apply_single_qubit_gate(state, dim, qubit, X);
}

template <typename FP>
void apply_y(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const std::complex<FP> Y[4] = {
        std::complex<FP>(0, 0),
        std::complex<FP>(0, -1),
        std::complex<FP>(0, 1),
        std::complex<FP>(0, 0),
    };
    apply_single_qubit_gate(state, dim, qubit, Y);
}

template <typename FP>
void apply_z(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    apply_phase_on_one(state, dim, qubit, std::complex<FP>(-1, 0));
}

template <typename FP>
void apply_h(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP inv_sqrt2 = FP(1) / std::sqrt(FP(2));
    const std::complex<FP> H[4] = {
        std::complex<FP>(inv_sqrt2, 0),
        std::complex<FP>(inv_sqrt2, 0),
        std::complex<FP>(inv_sqrt2, 0),
        std::complex<FP>(-inv_sqrt2, 0),
    };
    apply_single_qubit_gate(state, dim, qubit, H);
}

template <typename FP>
void apply_s(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    apply_phase_on_one(state, dim, qubit, std::complex<FP>(0, 1));
}

template <typename FP>
void apply_sdg(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    apply_phase_on_one(state, dim, qubit, std::complex<FP>(0, -1));
}

template <typename FP>
void apply_t(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP phase = FP(1) / std::sqrt(FP(2));
    apply_phase_on_one(state, dim, qubit, std::complex<FP>(phase, phase));
}

template <typename FP>
void apply_tdg(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP phase = FP(1) / std::sqrt(FP(2));
    apply_phase_on_one(state, dim, qubit, std::complex<FP>(phase, -phase));
}

// ---- 参数化旋转门 ----
// RX(theta) = cos(theta/2) I - i sin(theta/2) X

template <typename FP>
void apply_rx(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    // matrix[0] = cos(theta/2), matrix[1] = -i * sin(theta/2)
    apply_single_qubit_gate(state, dim, qubit, matrix);
}

template <typename FP>
void apply_ry(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    apply_single_qubit_gate(state, dim, qubit, matrix);
}

template <typename FP>
void apply_rz(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    // RZ(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
    // matrix[0] = exp(-i*theta/2), matrix[3] = exp(i*theta/2)
    // 对角门可以直接逐元素操作
    const uint64_t mask = 1ULL << qubit;
    const std::complex<FP> phase0 = matrix[0];
    const std::complex<FP> phase1 = matrix[3];
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            state[i] *= (i & mask) ? phase1 : phase0;  // |1> / |0> component
        }
    });
}

// 显式实例化 — 单比特门
#define INSTANTIATE_SINGLE(FP)                                                                                         \
    template void apply_single_qubit_gate<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);         \
    template void apply_x<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_y<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_z<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_h<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_s<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_sdg<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                       \
    template void apply_t<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                         \
    template void apply_tdg<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                       \
    template void apply_rx<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                        \
    template void apply_ry<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);                        \
    template void apply_rz<FP>(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);

INSTANTIATE_SINGLE(float)
INSTANTIATE_SINGLE(double)

#undef INSTANTIATE_SINGLE

}  // namespace quspect::native::kernels::scalar
