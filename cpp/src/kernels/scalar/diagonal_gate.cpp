#include <complex>
#include <cstdint>
#include <vector>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

/// 对角门：psi[i] *= diag[local_index]
/// 其中 local_index 是从 target_qubits 提取的局部 basis
template <typename FP>
void apply_diagonal_gate(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                         const std::complex<FP>* diag) {
    const uint64_t k = num_targets;

    common::ParallelForRange(static_cast<int64_t>(dim), common::ThreadsForDim(dim), [&](int64_t begin, int64_t end) {
        for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
            // 提取 target qubits 上的局部 index
            uint64_t local = 0;
            for (uint32_t q = 0; q < k; ++q) {
                if ((i >> target_qubits[q]) & 1) {
                    local |= (1ULL << q);
                }
            }
            state[i] *= diag[local];
        }
    });
}

// 显式实例化
template void apply_diagonal_gate<float>(std::complex<float>*, uint64_t, const uint32_t*, uint32_t,
                                         const std::complex<float>*);
template void apply_diagonal_gate<double>(std::complex<double>*, uint64_t, const uint32_t*, uint32_t,
                                          const std::complex<double>*);

}  // namespace quspect::native::kernels::scalar
