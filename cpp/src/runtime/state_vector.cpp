#include "quspect_native/runtime/state_vector.hpp"

#include <cmath>

#include <algorithm>
#include <stdexcept>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::runtime {

namespace {

/// 并行地把整个振幅数组清零
template <typename FP>
void FillZero(std::vector<std::complex<FP>>* amplitudes) {
    std::complex<FP>* data = amplitudes->data();
    const auto dim = static_cast<uint64_t>(amplitudes->size());
    common::ParallelForRange(static_cast<int64_t>(dim), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        std::fill(data + begin, data + end, std::complex<FP>(0, 0));
    });
}

}  // namespace

// ============================================================
// StateVector<FP>
// ============================================================

template <typename FP>
StateVector<FP>::StateVector(uint32_t num_qubits) : num_qubits_(num_qubits), amplitudes_(1ULL << num_qubits) {
    SetZeroState();
}

template <typename FP>
StateVector<FP>::StateVector(uint32_t num_qubits, uint64_t basis)
    : num_qubits_(num_qubits), amplitudes_(1ULL << num_qubits) {
    SetComputationalBasis(basis);
}

template <typename FP>
void StateVector<FP>::SetZeroState() {
    FillZero(&amplitudes_);
    amplitudes_[0] = Complex(1, 0);
}

template <typename FP>
void StateVector<FP>::SetComputationalBasis(uint64_t basis) {
    if (basis >= dim()) {
        throw std::out_of_range("basis index exceeds state vector dimension");
    }
    FillZero(&amplitudes_);
    amplitudes_[basis] = Complex(1, 0);
}

template <typename FP>
void StateVector<FP>::Normalize() {
    FP norm = std::sqrt(SquaredNorm());
    if (norm > 0) {
        Complex* values = amplitudes_.data();
        common::ParallelForRange(static_cast<int64_t>(dim()), common::ThreadsForDim(dim()),
                                 [&](int64_t begin, int64_t end) {
                                     for (int64_t i = begin; i < end; ++i) {
                                         values[i] /= norm;
                                     }
                                 });
    }
}

template <typename FP>
FP StateVector<FP>::SquaredNorm() const {
    const Complex* values = amplitudes_.data();
    const auto acc = common::BlockedReduce<1>(static_cast<int64_t>(dim()), common::ThreadsForDim(dim()),
                                              [&](int64_t begin, int64_t end, double* partial) {
                                                  double local = 0.0;
                                                  for (int64_t i = begin; i < end; ++i) {
                                                      local += static_cast<double>(std::norm(values[i]));
                                                  }
                                                  partial[0] += local;
                                              });
    return static_cast<FP>(acc[0]);
}

template <typename FP>
auto StateVector<FP>::data() -> Complex* {
    return amplitudes_.data();
}

template <typename FP>
auto StateVector<FP>::data() const -> const Complex* {
    return amplitudes_.data();
}

template <typename FP>
uint32_t StateVector<FP>::num_qubits() const {
    return num_qubits_;
}

template <typename FP>
uint64_t StateVector<FP>::dim() const {
    return 1ULL << num_qubits_;
}

// 显式实例化
template class StateVector<float>;
template class StateVector<double>;

}  // namespace quspect::native::runtime
