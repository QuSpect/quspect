// AVX2 优化单比特门 — Phase 6
// 关键优化：使用 AVX2 原生指令加速最高频的门操作
// Z 门用 SIMD 批量符号翻转，H 门用 SIMD 乘加

#include <cmath>

#include <complex>
#include <cstdint>

#include "quspect_native/common/parallel.hpp"

#ifdef QSPECT_USE_AVX2
#    include <immintrin.h>
#endif

namespace quspect::native::kernels::avx2 {

#ifdef QSPECT_USE_AVX2

using common::ParallelForRange;
using common::ThreadsForDim;

namespace {

/// 插入零位
static inline uint64_t insert_zero_bit(uint64_t base, uint32_t q) {
    uint64_t low_mask = (1ULL << q) - 1;
    uint64_t high_mask = ~low_mask;
    return (base & low_mask) | ((base & high_mask) << 1);
}

}  // namespace

// 前置声明 — apply_x / apply_h 等函数在定义前调用了 apply_single_qubit_gate
template <typename FP>
void apply_single_qubit_gate(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix);

// ============================================================
// Z 门 — 用 AVX2 批量符号翻转 (最高频门的快速路径)
// ============================================================
template <typename FP>
void apply_z(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const uint64_t mask = 1ULL << qubit;
    const int num_threads = ThreadsForDim(dim);
    if constexpr (sizeof(FP) == sizeof(double)) {
        // 构造符号掩码: 置 1 的位翻转浮点符号位 (bit 63)
        const __m256d sign_flip = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ULL));

        // One AVX2 register contains two complex<double> amplitudes.
        // 并行划分以「振幅对」为单位（第 p 对 = state[2p], state[2p+1]），线程之间不会共享同一个寄存器块。
        const uint64_t num_pairs = dim / 2;
        ParallelForRange(static_cast<int64_t>(num_pairs), num_threads, [&](int64_t begin, int64_t end) {
            for (uint64_t p = static_cast<uint64_t>(begin); p < static_cast<uint64_t>(end); ++p) {
                const uint64_t i = 2 * p;
                const bool flip0 = (i & mask) != 0;
                const bool flip1 = ((i + 1) & mask) != 0;
                if (flip0 && flip1) {
                    __m256d v = _mm256_loadu_pd(reinterpret_cast<const double*>(&state[i]));
                    v = _mm256_xor_pd(v, sign_flip);
                    _mm256_storeu_pd(reinterpret_cast<double*>(&state[i]), v);
                } else if (flip0 || flip1) {
                    for (uint64_t j = i; j < i + 2; ++j) {
                        if (j & mask)
                            state[j] *= std::complex<FP>(-1, 0);
                    }
                }
            }
        });
        // 尾部
        for (uint64_t i = 2 * num_pairs; i < dim; ++i) {
            if (i & mask)
                state[i] *= std::complex<FP>(-1, 0);
        }
    } else {
        // One AVX2 register contains four complex<float> amplitudes.
        const __m256 sign_flip = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        // 并行划分以 4 个振幅为一组（第 g 组 = state[4g .. 4g+4)）。
        const uint64_t num_groups = dim / 4;
        ParallelForRange(static_cast<int64_t>(num_groups), num_threads, [&](int64_t begin, int64_t end) {
            for (uint64_t g = static_cast<uint64_t>(begin); g < static_cast<uint64_t>(end); ++g) {
                const uint64_t i = 4 * g;
                bool all_need = true;
                bool any_need = false;
                for (uint64_t j = i; j < i + 4; ++j) {
                    if (j & mask)
                        any_need = true;
                    else
                        all_need = false;
                }

                if (all_need) {
                    __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(&state[i]));
                    v = _mm256_xor_ps(v, sign_flip);
                    _mm256_storeu_ps(reinterpret_cast<float*>(&state[i]), v);
                } else if (any_need) {
                    for (uint64_t j = i; j < i + 4; ++j) {
                        if (j & mask)
                            state[j] *= std::complex<FP>(-1, 0);
                    }
                }
            }
        });
        for (uint64_t i = 4 * num_groups; i < dim; ++i) {
            if (i & mask)
                state[i] *= std::complex<FP>(-1, 0);
        }
    }
}

// ============================================================
// X 门 — swap 每对 (i0, i1)，编译器 -mavx2 会自动向量化循环体
// ============================================================
template <typename FP>
void apply_x(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP X_mat[8] = {0, 0, 1, 0, 1, 0, 0, 0};
    apply_single_qubit_gate<FP>(state, dim, qubit, reinterpret_cast<const std::complex<FP>*>(X_mat));
}

// ============================================================
// H 门
// ============================================================
template <typename FP>
void apply_h(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP inv_sqrt2 = FP(1) / std::sqrt(FP(2));
    const FP H_mat[8] = {inv_sqrt2, 0, inv_sqrt2, 0, inv_sqrt2, 0, -inv_sqrt2, 0};
    apply_single_qubit_gate<FP>(state, dim, qubit, reinterpret_cast<const std::complex<FP>*>(H_mat));
}

// ============================================================
// 通用单比特门 (auto-vectorized by -mavx2 -mfma)
// ============================================================
template <typename FP>
void apply_single_qubit_gate(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    const uint64_t mask = 1ULL << qubit;
    const uint64_t half = dim >> 1;

    const auto& u00 = matrix[0];
    const auto& u01 = matrix[1];
    const auto& u10 = matrix[2];
    const auto& u11 = matrix[3];

    // 编译器会用 -mfma 将此循环自动向量化:
    // - 连续的 (i0, i1) 对之间没有依赖
    // - 每次迭代 2 个独立 FMA 操作
    // 每个 base 对应一对互不相交的振幅 (i0, i1)，线程区间之间没有写冲突。
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

// ============================================================
// 标准门包装
// ============================================================

template <typename FP>
void apply_y(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP Y_mat[8] = {0, 0, 0, -1, 0, 1, 0, 0};
    apply_single_qubit_gate<FP>(state, dim, qubit, reinterpret_cast<const std::complex<FP>*>(Y_mat));
}

template <typename FP>
void apply_s(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP S_mat[8] = {1, 0, 0, 0, 0, 0, 0, 1};
    apply_single_qubit_gate<FP>(state, dim, qubit, reinterpret_cast<const std::complex<FP>*>(S_mat));
}

template <typename FP>
void apply_sdg(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP Sdg_mat[8] = {1, 0, 0, 0, 0, 0, 0, -1};
    apply_single_qubit_gate<FP>(state, dim, qubit, reinterpret_cast<const std::complex<FP>*>(Sdg_mat));
}

template <typename FP>
void apply_t(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP inv_sqrt2 = FP(1) / std::sqrt(FP(2));
    const std::complex<FP> T_mat[4] = {
        std::complex<FP>(1, 0),
        std::complex<FP>(0, 0),
        std::complex<FP>(0, 0),
        std::complex<FP>(inv_sqrt2, inv_sqrt2),
    };
    apply_single_qubit_gate<FP>(state, dim, qubit, T_mat);
}

template <typename FP>
void apply_tdg(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* /*unused*/) {
    const FP inv_sqrt2 = FP(1) / std::sqrt(FP(2));
    const std::complex<FP> Tdg_mat[4] = {
        std::complex<FP>(1, 0),
        std::complex<FP>(0, 0),
        std::complex<FP>(0, 0),
        std::complex<FP>(inv_sqrt2, -inv_sqrt2),
    };
    apply_single_qubit_gate<FP>(state, dim, qubit, Tdg_mat);
}

// 参数化门 — 与 scalar 相同的通用实现，由 -mfma auto-vectorize
template <typename FP>
void apply_rx(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    apply_single_qubit_gate<FP>(state, dim, qubit, matrix);
}

template <typename FP>
void apply_ry(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    apply_single_qubit_gate<FP>(state, dim, qubit, matrix);
}

template <typename FP>
void apply_rz(std::complex<FP>* state, uint64_t dim, uint32_t qubit, const std::complex<FP>* matrix) {
    apply_single_qubit_gate<FP>(state, dim, qubit, matrix);
}

// 显式实例化 — double
template void apply_x<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_y<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_z<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_h<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_s<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_sdg<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_t<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_tdg<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_rx<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_ry<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_rz<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);
template void apply_single_qubit_gate<double>(std::complex<double>*, uint64_t, uint32_t, const std::complex<double>*);

// 显式实例化 — float
template void apply_x<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_y<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_z<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_h<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_s<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_sdg<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_t<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_tdg<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_rx<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_ry<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_rz<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);
template void apply_single_qubit_gate<float>(std::complex<float>*, uint64_t, uint32_t, const std::complex<float>*);

#else
// AVX2 未启用时占位
#endif  // QSPECT_USE_AVX2

}  // namespace quspect::native::kernels::avx2
