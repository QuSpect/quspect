// AVX2 优化期望值计算 — Phase 6
// 依赖 -mavx2 -mfma 编译器自动向量化；归约按固定分块并行（结果与线程数无关）。

#include <cmath>

#include <complex>
#include <cstdint>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::avx2 {

#ifdef QSPECT_USE_AVX2

namespace {

int popcount(uint64_t x) {
    return ::quspect::native::common::Popcount64(x);
}

}  // namespace

// 算法与 scalar::compute_pauli_expectation 相同，见 src/kernels/scalar/expectation.cpp。
template <typename FP>
std::complex<FP> compute_pauli_expectation(const std::complex<FP>* state, uint64_t dim, uint64_t bit_flip_mask,
                                           uint64_t phase_flip_mask, uint32_t pivot_qubit, int y_count) {
    using Complex = std::complex<FP>;

    // 相位因子 (-i)^{y_count}
    Complex i_pow_y(1, 0);
    int y_mod = y_count & 3;
    if (y_mod == 1)
        i_pow_y = Complex(0, -1);
    else if (y_mod == 2)
        i_pow_y = Complex(-1, 0);
    else if (y_mod == 3)
        i_pow_y = Complex(0, 1);

    const int num_threads = common::ThreadsForDim(dim);
    if (bit_flip_mask == 0) {
        // 纯对角项: sum += sign(i) * |psi[i]|^2
        const auto acc = common::BlockedReduce<1>(
            static_cast<int64_t>(dim), num_threads, [&](int64_t begin, int64_t end, double* partial) {
                double local = 0.0;
                for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
                    const double prob = static_cast<double>(std::norm(state[i]));
                    local += (popcount(i & phase_flip_mask) & 1) ? -prob : prob;
                }
                partial[0] += local;
            });
        return i_pow_y * Complex(static_cast<FP>(acc[0]), FP(0));
    }

    // 非对角项: 遍历 pivot 位为 0 的半空间，pivot 位为 0 已保证每个配对只计一次
    // （不能再要求 j > i，见 scalar 实现）。
    const uint64_t pivot_low_mask = (1ULL << pivot_qubit) - 1;
    const auto acc = common::BlockedReduce<2>(
        static_cast<int64_t>(dim >> 1), num_threads, [&](int64_t begin, int64_t end, double* partial) {
            double re = 0.0;
            double im = 0.0;
            for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                const uint64_t i = (base & pivot_low_mask) | ((base & ~pivot_low_mask) << 1);
                const uint64_t j = i ^ bit_flip_mask;
                const Complex term = std::conj(state[i]) * state[j];
                const double sign = (popcount(i & phase_flip_mask) & 1) ? -1.0 : 1.0;
                re += sign * static_cast<double>(term.real());
                im += sign * static_cast<double>(term.imag());
            }
            partial[0] += re;
            partial[1] += im;
        });
    const std::complex<double> phase_sum = std::complex<double>(static_cast<double>(i_pow_y.real()),
                                                                static_cast<double>(i_pow_y.imag()))
                                           * std::complex<double>(acc[0], acc[1]);
    return Complex(static_cast<FP>(2.0 * phase_sum.real()), FP(0));
}

// 显式实例化
template std::complex<float> compute_pauli_expectation<float>(const std::complex<float>*, uint64_t, uint64_t, uint64_t,
                                                              uint32_t, int);
template std::complex<double> compute_pauli_expectation<double>(const std::complex<double>*, uint64_t, uint64_t,
                                                                uint64_t, uint32_t, int);

#endif  // QSPECT_USE_AVX2

}  // namespace quspect::native::kernels::avx2
