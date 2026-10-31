#include <cmath>

#include <complex>
#include <cstdint>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/common/parallel.hpp"

namespace quspect::native::kernels::scalar {

namespace {

/// 计算 bit 的 popcount
int popcount(uint64_t x) {
    return ::quspect::native::common::Popcount64(x);
}

}  // namespace

/// Pauli 期望值 kernel
///
/// 对于 Pauli 字符串 P:
///   bit_flip_mask: 所有 X/Y 位置置 1
///   phase_flip_mask: 所有 Z/Y 位置置 1
///   pivot_qubit: bit_flip_mask 中任一置位（约定为最低置位）；只遍历该位为 0 的 i，
///                从而每个配对 (i, i ^ bit_flip_mask) 恰好计入一次
///   y_count: Y 的个数，用于计算 i^(y_count) 因子
///
/// 算法参考 qulacs stat_ops_expectation_value.cpp:
///   - 若 bit_flip_mask == 0 (纯对角项):
///       sum += (-1)^{popcount(i & phase_flip_mask)} * |psi[i]|^2
///   - 若 bit_flip_mask != 0:
///       遍历 pivot 位为 0 的 half space，对每对 (i, j=i^bit_flip_mask):
///       sum += phase_factor(i) * conj(psi[i]) * psi[j]
///       其中 phase_factor = (-i)^{y_count} * (-1)^{popcount(i & phase_flip_mask)} = P[i, j]
///       （该式对任意代表元 i 成立，与 i、j 的大小关系无关），最后结果取 2 * Re(sum)
template <typename FP>
std::complex<FP> compute_pauli_expectation(const std::complex<FP>* state, uint64_t dim, uint64_t bit_flip_mask,
                                           uint64_t phase_flip_mask, uint32_t pivot_qubit, int y_count) {
    using Complex = std::complex<FP>;

    Complex sum(0, 0);

    // 相位因子 (-i)^{y_count}
    // Y = i * X * Z, 矩阵元素 ⟨0|Y|1⟩ = -i, ⟨1|Y|0⟩ = i
    // 因此因子为 (-i)^{y_count} 而非 i^{y_count}
    // y_count % 4 -> 0:1, 1:-i, 2:-1, 3:i
    Complex i_pow_y(1, 0);
    int y_mod = y_count & 3;
    if (y_mod == 1) {
        i_pow_y = Complex(0, -1);  // (-i)^1 = -i
    } else if (y_mod == 2) {
        i_pow_y = Complex(-1, 0);  // (-i)^2 = -1
    } else if (y_mod == 3) {
        i_pow_y = Complex(0, 1);  // (-i)^3 = i
    }

    // 归约量拆成实部、虚部两个 double（OpenMP 2.0 不支持 std::complex 归约）；
    // BlockedReduce 的分块与线程数无关，结果与线程数逐位一致。
    const int num_threads = common::ThreadsForDim(dim);
    if (bit_flip_mask == 0) {
        // 纯对角项: sum += sign(i) * |psi[i]|^2，最后乘以 (-i)^{y_count}
        const auto acc = common::BlockedReduce<1>(
            static_cast<int64_t>(dim), num_threads, [&](int64_t begin, int64_t end, double* partial) {
                double local = 0.0;
                for (uint64_t i = static_cast<uint64_t>(begin); i < static_cast<uint64_t>(end); ++i) {
                    const double prob = static_cast<double>(std::norm(state[i]));
                    local += (popcount(i & phase_flip_mask) & 1) ? -prob : prob;
                }
                partial[0] += local;
            });
        sum = i_pow_y * Complex(static_cast<FP>(acc[0]), FP(0));
    } else {
        // 非对角项: 遍历 half_space
        // 对每对 (i, j=i^bfm), 期望值贡献为:
        //   P[i,j] * conj(psi[i]) * psi[j] + P[j,i] * conj(psi[j]) * psi[i]
        // = phase(i) * conj(psi[i]) * psi[j] + conj(phase(i) * conj(psi[i]) * psi[j])
        // = 2 * Re(phase(i) * conj(psi[i]) * psi[j])
        // 所以遍历 half space 后需要乘以 2
        //
        // pivot 位为 0 的 i 已唯一代表每个配对 (i, i ^ bit_flip_mask)。
        // 注意不能再额外要求 j > i：pivot 是 mask 的最低置位，而 i 与 j 的大小
        // 由 mask 的最高置位决定，叠加该条件会在 >= 2 个 X/Y 因子时丢掉一半配对。
        //
        // phase(i) = (-1)^{popcount(i & phase_flip_mask)} * (-i)^{y_count}，先累加
        // S = sum sign(i) * conj(psi[i]) * psi[j]，最后再乘以 (-i)^{y_count}。
        const uint64_t pivot_low_mask = (1ULL << pivot_qubit) - 1;
        const auto acc = common::BlockedReduce<2>(
            static_cast<int64_t>(dim >> 1), num_threads, [&](int64_t begin, int64_t end, double* partial) {
                double re = 0.0;
                double im = 0.0;
                for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                    // half_space: 在 pivot 位插入 0
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
        // 双倍贡献: 2 * Re(sum) (Pauli 期望值为实数)
        sum = Complex(static_cast<FP>(2.0 * phase_sum.real()), FP(0));
    }

    return sum;
}

// 显式实例化
template std::complex<float> compute_pauli_expectation<float>(const std::complex<float>*, uint64_t, uint64_t, uint64_t,
                                                              uint32_t, int);
template std::complex<double> compute_pauli_expectation<double>(const std::complex<double>*, uint64_t, uint64_t,
                                                                uint64_t, uint32_t, int);

}  // namespace quspect::native::kernels::scalar
