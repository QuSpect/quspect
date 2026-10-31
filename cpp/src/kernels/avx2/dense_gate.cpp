// AVX2 优化稠密门 — Phase 6
// 利用 -mavx2 -mfma 编译器自动向量化

#include <complex>
#include <cstdint>
#include <vector>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels {

#ifdef QSPECT_USE_AVX2

namespace scalar {

// 定义在 src/kernels/scalar/dense_gate.cpp（同一静态库内）
template <typename FP>
void apply_dense_gate(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                      const std::complex<FP>* matrix);

}  // namespace scalar

namespace avx2 {

using common::ParallelForRange;
using common::ThreadsForDim;

/// 本实现支持的最大目标比特数；更大的门交给 scalar 实现（矩阵过大，向量化收益很小）。
constexpr uint32_t kMaxAvx2DenseTargets = 4;

// 通用 k 比特稠密门 — 与 scalar 相同实现，由编译器标志自动向量化
template <typename FP>
void apply_dense_gate(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                      const std::complex<FP>* matrix) {
    if (num_targets > kMaxAvx2DenseTargets) {
        scalar::apply_dense_gate<FP>(state, dim, target_qubits, num_targets, matrix);
        return;
    }

    // 构建目标 qubit 的掩码
    uint64_t mask = 0;
    for (uint32_t k = 0; k < num_targets; ++k) {
        mask |= (1ULL << target_qubits[k]);
    }

    const uint64_t matrix_dim = 1ULL << num_targets;

    // 迭代 base 只处理目标位全为 0 的基（其余直接跳过），并写入 base 对应的 2^k 个振幅；
    // 不同 base 的振幅组互不相交，因此可以按 base 任意划分线程区间，每个区间使用自己的 buffer。
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        std::complex<FP> buf[1ULL << kMaxAvx2DenseTargets];

        for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
            if (base & mask)
                continue;

            // 收集受影响的振幅
            for (uint64_t j = 0; j < matrix_dim; ++j) {
                uint64_t idx = base;
                for (uint32_t k = 0; k < num_targets; ++k) {
                    if (j & (1ULL << k)) {
                        idx |= (1ULL << target_qubits[k]);
                    }
                }
                buf[j] = state[idx];
            }

            // 施加矩阵变换
            for (uint64_t r = 0; r < matrix_dim; ++r) {
                uint64_t idx_r = base;
                for (uint32_t k = 0; k < num_targets; ++k) {
                    if (r & (1ULL << k)) {
                        idx_r |= (1ULL << target_qubits[k]);
                    }
                }

                std::complex<FP> acc(0, 0);
                for (uint64_t c = 0; c < matrix_dim; ++c) {
                    acc += matrix[r * matrix_dim + c] * buf[c];
                }
                state[idx_r] = acc;
            }
        }
    });
}

// 对角门 — 编译器用 -mfma 自动向量化 re*im + re*im 模式
template <typename FP>
void apply_diagonal_gate(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                         const std::complex<FP>* diag) {
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            uint64_t sub_idx = 0;
            for (uint32_t k = 0; k < num_targets; ++k) {
                if (i & (1ULL << target_qubits[k])) {
                    sub_idx |= (1ULL << k);
                }
            }
            state[i] *= diag[sub_idx];
        }
    });
}

// 显式实例化
template void apply_dense_gate<double>(std::complex<double>*, uint64_t, const uint32_t*, uint32_t,
                                       const std::complex<double>*);
template void apply_diagonal_gate<double>(std::complex<double>*, uint64_t, const uint32_t*, uint32_t,
                                          const std::complex<double>*);
template void apply_dense_gate<float>(std::complex<float>*, uint64_t, const uint32_t*, uint32_t,
                                      const std::complex<float>*);
template void apply_diagonal_gate<float>(std::complex<float>*, uint64_t, const uint32_t*, uint32_t,
                                         const std::complex<float>*);

}  // namespace avx2

#endif  // QSPECT_USE_AVX2

}  // namespace quspect::native::kernels
