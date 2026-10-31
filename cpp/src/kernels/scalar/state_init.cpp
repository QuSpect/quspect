#include <algorithm>
#include <complex>
#include <cstdint>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

/// 初始化状态为 |0...0>
template <typename FP>
void init_zero_state(std::complex<FP>* state, uint64_t dim) {
    common::ParallelForRange(static_cast<int64_t>(dim), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        std::fill(state + begin, state + end, std::complex<FP>(0, 0));
    });
    state[0] = std::complex<FP>(1, 0);
}

/// 初始化状态为指定计算基
template <typename FP>
void init_basis_state(std::complex<FP>* state, uint64_t dim, uint64_t basis) {
    common::ParallelForRange(static_cast<int64_t>(dim), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        std::fill(state + begin, state + end, std::complex<FP>(0, 0));
    });
    state[basis] = std::complex<FP>(1, 0);
}

// 显式实例化
template void init_zero_state<float>(std::complex<float>*, uint64_t);
template void init_zero_state<double>(std::complex<double>*, uint64_t);
template void init_basis_state<float>(std::complex<float>*, uint64_t, uint64_t);
template void init_basis_state<double>(std::complex<double>*, uint64_t, uint64_t);

}  // namespace quspect::native::kernels::scalar
