#include "quspect_native/kernels/noise_kernels.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels {

namespace {

/// ||state||^2，按固定分块并行归约（结果与线程数无关，Kraus 选择可复现）
template <typename FP>
FP SquaredNorm(const std::complex<FP>* state, uint64_t dim) {
    const auto acc = common::BlockedReduce<1>(static_cast<int64_t>(dim), common::ThreadsForDim(dim),
                                              [&](int64_t begin, int64_t end, double* partial) {
                                                  double local = 0.0;
                                                  for (int64_t i = begin; i < end; ++i) {
                                                      local += static_cast<double>(std::norm(state[i]));
                                                  }
                                                  partial[0] += local;
                                              });
    return static_cast<FP>(acc[0]);
}

/// Extract bits from global_idx at positions specified by targets.
uint64_t extract_bits(uint64_t global_idx, const uint32_t* targets, uint32_t k) {
    uint64_t local = 0;
    for (uint32_t i = 0; i < k; ++i) {
        if ((global_idx >> targets[i]) & 1) {
            local |= (uint64_t{1} << i);
        }
    }
    return local;
}

/// Scatter local bits to global index positions specified by targets.
uint64_t scatter_bits(uint64_t local, const uint32_t* targets, uint32_t k) {
    uint64_t global = 0;
    for (uint32_t i = 0; i < k; ++i) {
        if ((local >> i) & 1) {
            global |= (uint64_t{1} << targets[i]);
        }
    }
    return global;
}

/// Build mask_list[x] = scatter_bits(x, targets, k) for x in [0, 2^k).
void build_mask_list(uint64_t* mask_list, const uint32_t* targets, uint32_t k) {
    for (uint64_t x = 0; x < (uint64_t{1} << k); ++x) {
        mask_list[x] = scatter_bits(x, targets, k);
    }
}

}  // namespace

// ============================================================
// apply_dense_unitary — standalone dense gate application
// ============================================================

template <typename FP>
void apply_dense_unitary(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                         const std::complex<FP>* matrix) {
    const uint64_t k = num_targets;
    const uint64_t matrix_dim = uint64_t{1} << k;
    const uint64_t block_size = dim >> k;

    std::vector<uint64_t> mask_list(matrix_dim);
    build_mask_list(mask_list.data(), target_qubits, k);

    // 每个 base 对应一组互不相交的 2^k 个振幅；每个线程区间使用自己的临时 buffer。
    common::ParallelForRange(
        static_cast<int64_t>(block_size), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
            std::vector<std::complex<FP>> buffer(matrix_dim);

            for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                // Construct basis0: scatter non-target bits.
                uint64_t basis0 = 0;
                {
                    uint64_t b = base;
                    uint32_t pos = 0;
                    for (uint32_t bit = 0; bit < 64 && pos < (64 - k); ++bit) {
                        bool is_target = false;
                        for (uint32_t i = 0; i < k; ++i) {
                            if (target_qubits[i] == bit) {
                                is_target = true;
                                break;
                            }
                        }
                        if (!is_target) {
                            if (b & 1)
                                basis0 |= (uint64_t{1} << bit);
                            b >>= 1;
                            ++pos;
                        }
                    }
                }

                // Matrix-vector multiply.
                for (uint64_t y = 0; y < matrix_dim; ++y) {
                    buffer[y] = std::complex<FP>(0, 0);
                    for (uint64_t x = 0; x < matrix_dim; ++x) {
                        buffer[y] += matrix[y * matrix_dim + x] * state[basis0 ^ mask_list[x]];
                    }
                }

                // Write back.
                for (uint64_t y = 0; y < matrix_dim; ++y) {
                    state[basis0 ^ mask_list[y]] = buffer[y];
                }
            }
        });
}

// ============================================================
// apply_kraus_monte_carlo — Monte Carlo Kraus selection
// ============================================================

template <typename FP>
uint32_t apply_kraus_monte_carlo(std::complex<FP>* state, uint64_t dim, const std::complex<FP>* kraus_ops,
                                 uint32_t num_ops, const uint32_t* target_qubits, uint32_t num_targets,
                                 std::mt19937_64& rng) {
    const uint64_t kdim = uint64_t{1} << num_targets;

    // Save original state.
    std::vector<std::complex<FP>> original(dim);
    std::copy(state, state + dim, original.begin());

    // Compute probabilities and transformed states for each Kraus op.
    std::vector<FP> probs(num_ops);
    std::vector<std::vector<std::complex<FP>>> results(num_ops);

    for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
        // Restore original state.
        std::copy(original.begin(), original.end(), state);

        const std::complex<FP>* kop = kraus_ops + op_idx * kdim * kdim;
        apply_dense_unitary<FP>(state, dim, target_qubits, num_targets, kop);

        // Store result.
        results[op_idx].assign(state, state + dim);

        // Probability = ||K_i|ψ⟩||².
        probs[op_idx] = SquaredNorm(state, dim);
    }

    // Normalize probabilities.
    FP total = 0;
    for (uint32_t i = 0; i < num_ops; ++i)
        total += probs[i];

    if (total < FP{1e-15}) {
        // All probabilities zero — keep original state.
        std::copy(original.begin(), original.end(), state);
        return 0;
    }

    for (auto& p : probs)
        p /= total;

    // Random selection.
    std::uniform_real_distribution<FP> dist(FP{0}, FP{1});
    FP r = dist(rng);
    FP cumulative = 0;
    uint32_t selected = 0;
    for (uint32_t i = 0; i < num_ops; ++i) {
        cumulative += probs[i];
        if (r <= cumulative || i + 1 == num_ops) {
            selected = i;
            break;
        }
    }

    // Apply selected Kraus operator and normalize.
    std::copy(results[selected].begin(), results[selected].end(), state);

    const FP norm = SquaredNorm(state, dim);
    if (norm > FP{1e-15}) {
        const FP inv_norm = FP{1} / std::sqrt(norm);
        common::ParallelForRange(static_cast<int64_t>(dim), common::ThreadsForDim(dim),
                                 [&](int64_t begin, int64_t end) {
                                     for (int64_t i = begin; i < end; ++i) {
                                         state[i] *= inv_norm;
                                     }
                                 });
    }

    return selected;
}

// Explicit instantiations.
template void apply_dense_unitary<float>(std::complex<float>*, uint64_t, const uint32_t*, uint32_t,
                                         const std::complex<float>*);
template void apply_dense_unitary<double>(std::complex<double>*, uint64_t, const uint32_t*, uint32_t,
                                          const std::complex<double>*);

template uint32_t apply_kraus_monte_carlo<float>(std::complex<float>*, uint64_t, const std::complex<float>*, uint32_t,
                                                 const uint32_t*, uint32_t, std::mt19937_64&);
template uint32_t apply_kraus_monte_carlo<double>(std::complex<double>*, uint64_t, const std::complex<double>*,
                                                  uint32_t, const uint32_t*, uint32_t, std::mt19937_64&);

}  // namespace quspect::native::kernels
