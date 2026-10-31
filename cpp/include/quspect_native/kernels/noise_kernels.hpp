#pragma once

#include <complex>
#include <cstdint>
#include <random>

namespace quspect::native::kernels {

/// Apply a k-qubit dense unitary to a state vector.
///
/// This is a standalone version usable outside the dispatch table,
/// for applying Kraus operators within Monte Carlo trajectories.
template <typename FP>
void apply_dense_unitary(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                         const std::complex<FP>* matrix);

/// Monte Carlo Kraus application on a state vector.
///
/// For each Kraus operator K_i:
///   1. Compute |ψ_i'⟩ = K_i |ψ⟩
///   2. Compute p_i = ⟨ψ_i'|ψ_i'⟩
/// Then randomly select operator i with probability proportional to p_i,
/// apply the selected operator, and normalize.
///
/// Returns the index of the selected Kraus operator.
template <typename FP>
uint32_t apply_kraus_monte_carlo(std::complex<FP>* state, uint64_t dim, const std::complex<FP>* kraus_ops,
                                 uint32_t num_ops, const uint32_t* target_qubits, uint32_t num_targets,
                                 std::mt19937_64& rng);

}  // namespace quspect::native::kernels
