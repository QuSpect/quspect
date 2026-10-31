#pragma once

#include <cmath>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

namespace quspect::native::runtime {

/// Density matrix ρ = Σ p_i |ψ_i⟩⟨ψ_i| with shape dim×dim (dim = 2^n).
///
/// Stored row-major as a flat vector of size dim * dim.
/// Element (r, c) = data_[r * dim + c].
template <typename FP>
class DensityMatrix {
 public:
    using Complex = std::complex<FP>;
    using Rng = std::mt19937_64;

    explicit DensityMatrix(uint32_t num_qubits) : num_qubits_(num_qubits), dim_(uint64_t{1} << num_qubits) {
        amplitudes_.resize(dim_ * dim_, Complex{0, 0});
        amplitudes_[0] = Complex{1, 0};
    }

    uint32_t num_qubits() const {
        return num_qubits_;
    }
    uint64_t dim() const {
        return dim_;
    }
    uint64_t size() const {
        return dim_ * dim_;
    }

    Complex* data() {
        return amplitudes_.data();
    }
    const Complex* data() const {
        return amplitudes_.data();
    }

    void SetZero() {
        std::fill(amplitudes_.begin(), amplitudes_.end(), Complex{0, 0});
    }

    void SetPureZero() {
        std::fill(amplitudes_.begin(), amplitudes_.end(), Complex{0, 0});
        amplitudes_[0] = Complex{1, 0};
    }

    FP Trace() const {
        FP sum = 0;
        for (uint64_t i = 0; i < dim_; ++i) {
            sum += std::real(amplitudes_[i * dim_ + i]);
        }
        return sum;
    }

    FP SquaredNorm() const {
        FP sum = 0;
        for (const auto& v : amplitudes_) {
            sum += std::norm(v);
        }
        return sum;
    }

    void Normalize() {
        FP trace = Trace();
        if (trace < FP{1e-15})
            return;
        FP inv = FP{1} / trace;
        for (auto& v : amplitudes_) {
            v *= inv;
        }
    }

    // ---------------------------------------------------------------
    // ApplyUnitary: ρ → U ρ U†
    // ---------------------------------------------------------------
    void ApplyUnitary(const Complex* umat, const uint32_t* target_qubits, uint32_t num_targets) {
        const uint64_t k = num_targets;
        const uint64_t kdim = uint64_t{1} << k;
        const uint64_t block_size = dim_ >> k;

        // Build mask list.
        std::vector<uint64_t> mask_list(kdim);
        for (uint64_t x = 0; x < kdim; ++x) {
            uint64_t m = 0;
            for (uint32_t i = 0; i < k; ++i) {
                if ((x >> i) & 1)
                    m |= uint64_t{1} << target_qubits[i];
            }
            mask_list[x] = m;
        }

        // Auxiliary function: build basis0 from block index.
        auto build_basis0 = [&](uint64_t block_idx) -> uint64_t {
            uint64_t basis0 = 0;
            uint64_t b = block_idx;
            uint32_t pos = 0;
            uint32_t non_target_count = num_qubits_ - k;
            for (uint32_t bit = 0; bit < 64 && pos < non_target_count; ++bit) {
                bool is_target = false;
                for (uint32_t i = 0; i < k; ++i) {
                    if (target_qubits[i] == bit) {
                        is_target = true;
                        break;
                    }
                }
                if (!is_target) {
                    if (b & 1)
                        basis0 |= uint64_t{1} << bit;
                    b >>= 1;
                    ++pos;
                }
            }
            return basis0;
        };

        // Step 1: Apply U on the left: temp[row, col] = Σ_x U[y, x] * ρ[mask_x,
        // col]
        std::vector<Complex> temp(size());
        for (uint64_t block = 0; block < block_size; ++block) {
            uint64_t basis0 = build_basis0(block);
            for (uint64_t y = 0; y < kdim; ++y) {
                uint64_t row = basis0 | mask_list[y];
                for (uint64_t col = 0; col < dim_; ++col) {
                    Complex acc{0, 0};
                    for (uint64_t x = 0; x < kdim; ++x) {
                        Complex u_yx = umat[y * kdim + x];
                        if (std::norm(u_yx) < FP{1e-15})
                            continue;
                        uint64_t src_row = basis0 | mask_list[x];
                        acc += u_yx * amplitudes_[src_row * dim_ + col];
                    }
                    temp[row * dim_ + col] = acc;
                }
            }
        }

        // Step 2: Apply U† on the right: ρ'[row, col] = Σ_y temp[row, mask_y] *
        // conj(U[x, y])
        for (uint64_t block = 0; block < block_size; ++block) {
            uint64_t basis0 = build_basis0(block);
            for (uint64_t x = 0; x < kdim; ++x) {
                uint64_t col = basis0 | mask_list[x];
                for (uint64_t row = 0; row < dim_; ++row) {
                    Complex acc{0, 0};
                    for (uint64_t y = 0; y < kdim; ++y) {
                        uint64_t src_col = basis0 | mask_list[y];
                        Complex val = temp[row * dim_ + src_col];
                        if (std::norm(val) < FP{1e-15})
                            continue;
                        acc += val * std::conj(umat[x * kdim + y]);
                    }
                    amplitudes_[row * dim_ + col] = acc;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // ApplySingleKraus: ρ → K ρ K†
    // ---------------------------------------------------------------
    void ApplySingleKraus(const Complex* kop, const uint32_t* target_qubits, uint32_t num_targets) {
        ApplyUnitary(kop, target_qubits, num_targets);
    }

    // ---------------------------------------------------------------
    // ApplyKrausChannel: ρ → Σ_i K_i ρ K_i†
    // ---------------------------------------------------------------
    void ApplyKrausChannel(const Complex* kraus_ops, uint32_t num_ops, const uint32_t* target_qubits,
                           uint32_t num_targets) {
        uint64_t kdim = uint64_t{1} << num_targets;

        // Accumulate result in a temporary.
        std::vector<Complex> result(size(), Complex{0, 0});

        // Save original and work on a copy.
        std::vector<Complex> original = amplitudes_;

        for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
            // Restore original for each Kraus op.
            amplitudes_ = original;

            const Complex* kop = kraus_ops + op_idx * kdim * kdim;
            ApplyUnitary(kop, target_qubits, num_targets);

            // Add to result: result += K_i ρ K_i†
            for (uint64_t i = 0; i < size(); ++i) {
                result[i] += amplitudes_[i];
            }
        }

        amplitudes_ = std::move(result);
    }

    // ---------------------------------------------------------------
    // ApplyKrausMonteCarlo: randomly select one Kraus operator.
    //
    // Assumes the density matrix represents a pure state (single
    // non-zero entry at (basis, basis) where basis is the current
    // computational basis state).  After application, the state
    // may no longer be pure but we project back by normalizing
    // and returning the result as a pure state.
    // ---------------------------------------------------------------
    uint32_t ApplyKrausMonteCarlo(const Complex* kraus_ops, uint32_t num_ops, const uint32_t* target_qubits,
                                  uint32_t num_targets, Rng& rng) {
        uint64_t kdim = uint64_t{1} << num_targets;

        // Compute probabilities and transformed states for each Kraus op.
        std::vector<FP> probs(num_ops);
        std::vector<std::vector<Complex>> results(num_ops);
        std::vector<Complex> original = amplitudes_;

        for (uint32_t op_idx = 0; op_idx < num_ops; ++op_idx) {
            amplitudes_ = original;
            const Complex* kop = kraus_ops + op_idx * kdim * kdim;
            ApplyUnitary(kop, target_qubits, num_targets);

            results[op_idx] = amplitudes_;

            // Probability = Tr(K_i ρ K_i†) = Tr(ρ')
            FP tr = 0;
            for (uint64_t i = 0; i < dim_; ++i) {
                tr += std::real(amplitudes_[i * dim_ + i]);
            }
            probs[op_idx] = tr;
        }

        // Normalize probabilities.
        FP total = 0;
        for (FP p : probs)
            total += p;
        if (total < FP{1e-15}) {
            amplitudes_ = original;
            return 0;
        }

        for (auto& p : probs)
            p /= total;

        // Random selection.
        std::uniform_real_distribution<FP> dist(0, 1);
        FP r = dist(rng);
        FP cumulative = 0;
        uint32_t selected = 0;
        for (uint32_t i = 0; i < num_ops; ++i) {
            cumulative += probs[i];
            if (r <= cumulative) {
                selected = i;
                break;
            }
        }

        amplitudes_ = results[selected];
        Normalize();
        return selected;
    }

 private:
    uint32_t num_qubits_;
    uint64_t dim_;
    std::vector<Complex> amplitudes_;
};

}  // namespace quspect::native::runtime
