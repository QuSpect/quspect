// AVX2 优化双比特门 — Phase 6
// 关键优化：利用 -mavx2 -mfma 编译器自动向量化
// CZ 门用 SIMD 批量符号翻转

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

// ============================================================
// CNOT — 受控 X 门
// 编译器用 -mavx2 自动向量化循环中的 swap 操作
// ============================================================
template <typename FP>
void apply_cnot(std::complex<FP>* state, uint64_t dim, uint32_t target, uint64_t control_mask, uint64_t control_value,
                const std::complex<FP>* /*unused*/) {
    const uint64_t tmask = 1ULL << target;

    // 每对 (i, i ^ tmask) 只由较小的 i 处理，迭代之间写入互不相交。
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            if ((i & control_mask) == control_value) {
                uint64_t j = i ^ tmask;
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
// CZ — 受控 Z 门 (SIMD 加速符号翻转)
// ============================================================
template <typename FP>
void apply_cz(std::complex<FP>* state, uint64_t dim, uint32_t target, uint64_t control_mask, uint64_t control_value,
              const std::complex<FP>* /*unused*/) {
    const uint64_t tmask = 1ULL << target;
    const int num_threads = ThreadsForDim(dim);
    if constexpr (sizeof(FP) == sizeof(double)) {
        const __m256d sign_flip = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000ULL));

        // One AVX2 register contains two complex<double> amplitudes.
        // 并行划分以「振幅对」为单位（第 p 对 = state[2p], state[2p+1]）。
        const uint64_t num_pairs = dim / 2;
        ParallelForRange(static_cast<int64_t>(num_pairs), num_threads, [&](int64_t begin, int64_t end) {
            for (uint64_t p = static_cast<uint64_t>(begin); p < static_cast<uint64_t>(end); ++p) {
                const uint64_t i = 2 * p;
                int count = 0;
                for (int k = 0; k < 2; ++k) {
                    uint64_t idx = i + k;
                    if ((idx & control_mask) == control_value && (idx & tmask))
                        count++;
                }

                if (count == 2) {
                    __m256d v = _mm256_loadu_pd(reinterpret_cast<const double*>(&state[i]));
                    v = _mm256_xor_pd(v, sign_flip);
                    _mm256_storeu_pd(reinterpret_cast<double*>(&state[i]), v);
                } else if (count > 0) {
                    for (int k = 0; k < 2; ++k) {
                        uint64_t idx = i + k;
                        if ((idx & control_mask) == control_value && (idx & tmask)) {
                            state[idx] *= std::complex<FP>(-1, 0);
                        }
                    }
                }
            }
        });
        for (uint64_t i = 2 * num_pairs; i < dim; ++i) {
            if ((i & control_mask) == control_value && (i & tmask)) {
                state[i] *= std::complex<FP>(-1, 0);
            }
        }
    } else {
        ParallelForRange(static_cast<int64_t>(dim), num_threads, [&](int64_t begin, int64_t end) {
            for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
                if ((i & control_mask) == control_value && (i & tmask)) {
                    state[i] *= std::complex<FP>(-1, 0);
                }
            }
        });
    }
}

// ============================================================
// SWAP — 交换两个 qubit
// ============================================================
template <typename FP>
void apply_swap(std::complex<FP>* state, uint64_t dim, uint32_t qubit0, uint32_t qubit1) {
    const uint64_t mask0 = 1ULL << qubit0;
    const uint64_t mask1 = 1ULL << qubit1;

    // 每对 (i, i ^ mask0 ^ mask1) 只由较小的 i 处理，迭代之间写入互不相交。
    ParallelForRange(static_cast<int64_t>(dim), ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            uint64_t b0 = (i >> qubit0) & 1;
            uint64_t b1 = (i >> qubit1) & 1;
            if (b0 == b1)
                continue;

            uint64_t j = i ^ mask0 ^ mask1;
            if (i < j) {
                auto tmp = state[i];
                state[i] = state[j];
                state[j] = tmp;
            }
        }
    });
}

// 显式实例化
template void apply_cnot<double>(std::complex<double>*, uint64_t, uint32_t, uint64_t, uint64_t,
                                 const std::complex<double>*);
template void apply_cz<double>(std::complex<double>*, uint64_t, uint32_t, uint64_t, uint64_t,
                               const std::complex<double>*);
template void apply_swap<double>(std::complex<double>*, uint64_t, uint32_t, uint32_t);
template void apply_cnot<float>(std::complex<float>*, uint64_t, uint32_t, uint64_t, uint64_t,
                                const std::complex<float>*);
template void apply_cz<float>(std::complex<float>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<float>*);
template void apply_swap<float>(std::complex<float>*, uint64_t, uint32_t, uint32_t);

#else
// AVX2 未启用时占位
#endif  // QSPECT_USE_AVX2

}  // namespace quspect::native::kernels::avx2
