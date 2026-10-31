#include <complex>
#include <cstdint>
#include <vector>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

namespace {

/// 从 target_qubits 列表中提取的 bit 组成局部 index
uint64_t extract_bits(uint64_t global_idx, const uint32_t* targets, uint32_t k) {
    uint64_t local = 0;
    for (uint32_t i = 0; i < k; ++i) {
        if ((global_idx >> targets[i]) & 1) {
            local |= (1ULL << i);
        }
    }
    return local;
}

/// 将 local bits 散射到全局 index 的目标位上
uint64_t scatter_bits(uint64_t local, const uint32_t* targets, uint32_t k) {
    uint64_t global = 0;
    for (uint32_t i = 0; i < k; ++i) {
        if ((local >> i) & 1) {
            global |= (1ULL << targets[i]);
        }
    }
    return global;
}

/// 构造 matrix_mask_list[x]: 局部 basis x 映射到全局 index 的异或 mask
void build_mask_list(uint64_t* mask_list, const uint32_t* targets, uint32_t k) {
    for (uint64_t x = 0; x < (1ULL << k); ++x) {
        mask_list[x] = scatter_bits(x, targets, k);
    }
}

}  // namespace

/// 通用 k 比特稠密门
///
/// 算法参考 qulacs update_ops_matrix_dense_multi.cpp。
/// 使用 2^k 大小的 thread-local buffer，避免分配 2^n 临时张量。
template <typename FP>
void apply_dense_gate(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                      const std::complex<FP>* matrix) {
    const uint64_t k = num_targets;
    const uint64_t matrix_dim = 1ULL << k;
    const uint64_t block_size = dim >> k;

    // 构造 mask 列表
    std::vector<uint64_t> mask_list(matrix_dim);
    build_mask_list(mask_list.data(), target_qubits, k);

    // 每个 base 对应一组互不相交的 2^k 个振幅；每个线程区间使用自己的临时 buffer。
    common::ParallelForRange(
        static_cast<int64_t>(block_size), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
            std::vector<std::complex<FP>> buffer(matrix_dim);

            for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                // 从 base 构造 basis0：将 base 的各位散射到非目标位
                uint64_t basis0 = 0;
                {
                    uint64_t b = base;
                    uint32_t pos = 0;
                    for (uint32_t bit = 0; bit < 64 && pos < (64 - k); ++bit) {
                        // 检查 bit 是否在 target_qubits 中
                        bool is_target = false;
                        for (uint32_t i = 0; i < k; ++i) {
                            if (target_qubits[i] == bit) {
                                is_target = true;
                                break;
                            }
                        }
                        if (!is_target) {
                            if (b & 1)
                                basis0 |= (1ULL << bit);
                            b >>= 1;
                            ++pos;
                        }
                    }
                }

                // 矩阵-向量乘法
                for (uint64_t y = 0; y < matrix_dim; ++y) {
                    buffer[y] = std::complex<FP>(0, 0);
                    for (uint64_t x = 0; x < matrix_dim; ++x) {
                        buffer[y] += matrix[y * matrix_dim + x] * state[basis0 ^ mask_list[x]];
                    }
                }

                // 写回
                for (uint64_t y = 0; y < matrix_dim; ++y) {
                    state[basis0 ^ mask_list[y]] = buffer[y];
                }
            }
        });
}

// 显式实例化
template void apply_dense_gate<float>(std::complex<float>*, uint64_t, const uint32_t*, uint32_t,
                                      const std::complex<float>*);
template void apply_dense_gate<double>(std::complex<double>*, uint64_t, const uint32_t*, uint32_t,
                                       const std::complex<double>*);

}  // namespace quspect::native::kernels::scalar
