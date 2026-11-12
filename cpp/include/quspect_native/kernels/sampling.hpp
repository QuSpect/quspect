#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace quspect::native::kernels {

/// Cumulative basis-state probabilities of a state vector.
///
/// out[i] = |state[0]|^2 + ... + |state[i]|^2, accumulated in double precision in index order.
/// Building it costs one pass over the state; sample_from_cumulative() then draws any number of
/// samples from the same state with one binary search each.
template <typename FP>
std::vector<double> cumulative_probabilities(const std::complex<FP>* state, uint64_t dim);

/// Basis index selected by one uniform number in [0, 1).
///
/// Returns the first index whose cumulative probability reaches random01 times the total (the
/// total is close to, but not exactly, 1), clamped to the last index.  For the same state and
/// random number this is the index the dispatch-table sampling kernel returns.
uint64_t sample_from_cumulative(const std::vector<double>& cumulative, double random01);

}  // namespace quspect::native::kernels
