// 多 X/Y 因子 Pauli 期望值回归测试（B46 / B47）。
//
// 旧实现在 pivot 过滤之外又额外要求 j > i，导致 bit_flip_mask 含 >= 2 个置位时
// 丢掉一半的 (i, i ^ mask) 配对。这里用稠密矩阵作为参考实现进行交叉验证。

#include <cmath>

#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

namespace {

using Complex = std::complex<double>;
using DenseMatrix = std::vector<std::vector<Complex>>;

/// 返回单比特 Pauli 矩阵。
DenseMatrix PauliMatrix(char p) {
    const Complex kI(0, 1);
    switch (p) {
        case 'X':
            return {{0, 1}, {1, 0}};
        case 'Y':
            return {{0, -kI}, {kI, 0}};
        case 'Z':
            return {{1, 0}, {0, -1}};
        default:
            return {{1, 0}, {0, 1}};
    }
}

/// 计算 Kronecker 积 a ⊗ b。
DenseMatrix Kron(const DenseMatrix& a, const DenseMatrix& b) {
    const size_t ra = a.size();
    const size_t rb = b.size();
    DenseMatrix out(ra * rb, std::vector<Complex>(ra * rb));
    for (size_t i = 0; i < ra; ++i) {
        for (size_t j = 0; j < ra; ++j) {
            for (size_t k = 0; k < rb; ++k) {
                for (size_t l = 0; l < rb; ++l) {
                    out[i * rb + k][j * rb + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    return out;
}

/// 小端序稠密 Pauli 矩阵：label[q] 作用在 qubit q（基索引的第 q 位）。
DenseMatrix DensePauli(const std::string& label) {
    DenseMatrix m = {{1}};
    // 最高位 qubit 在 Kronecker 积最左侧。
    for (size_t q = label.size(); q-- > 0;) {
        m = Kron(m, PauliMatrix(label[q]));
    }
    return m;
}

/// 稠密参考：<psi|P|psi>。
Complex DenseExpectation(const std::string& label, const std::vector<Complex>& state) {
    const DenseMatrix m = DensePauli(label);
    Complex sum(0, 0);
    for (size_t i = 0; i < state.size(); ++i) {
        for (size_t j = 0; j < state.size(); ++j) {
            sum += std::conj(state[i]) * m[i][j] * state[j];
        }
    }
    return sum;
}

/// 通过 ComputePauliMasks + dispatch kernel 计算期望值。
Complex KernelExpectation(const std::string& label, const std::vector<Complex>& state) {
    std::vector<std::pair<uint32_t, char>> paulis;
    for (size_t q = 0; q < label.size(); ++q) {
        paulis.emplace_back(static_cast<uint32_t>(q), label[q]);
    }
    const auto masks = runtime::ComputePauliMasks(paulis);
    const auto& table = kernels::GetDispatchF64();
    return table.pauli_expect(state.data(), state.size(), masks.bit_flip_mask, masks.phase_flip_mask, masks.pivot_qubit,
                              masks.y_count);
}

/// 生成归一化随机态。
std::vector<Complex> RandomState(size_t num_qubits, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<Complex> state(size_t{1} << num_qubits);
    double norm = 0.0;
    for (auto& amp : state) {
        amp = Complex(dist(rng), dist(rng));
        norm += std::norm(amp);
    }
    const double inv = 1.0 / std::sqrt(norm);
    for (auto& amp : state) {
        amp *= inv;
    }
    return state;
}

}  // namespace

TEST(ExpectationMultiFlipTest, XXOnPlusPlusIsOne) {
    // |++> 的四个振幅均为 1/2。
    std::vector<Complex> state(4, Complex(0.5, 0));
    EXPECT_NEAR(std::real(KernelExpectation("XX", state)), 1.0, 1e-12);
}

TEST(ExpectationMultiFlipTest, YYOnPlusPlusIsZero) {
    std::vector<Complex> state(4, Complex(0.5, 0));
    EXPECT_NEAR(std::abs(KernelExpectation("YY", state)), 0.0, 1e-12);
}

TEST(ExpectationMultiFlipTest, XYOnTensorEigenstateIsOne) {
    // |+> (q0) ⊗ |y+> (q1)：<X0 Y1> = 1。
    const double h = 0.5;
    std::vector<Complex> state = {Complex(h, 0), Complex(h, 0), Complex(0, h), Complex(0, h)};
    EXPECT_NEAR(std::real(KernelExpectation("XY", state)), 1.0, 1e-12);
}

TEST(ExpectationMultiFlipTest, RandomStatesMatchDenseReference) {
    const std::vector<std::string> labels = {"XX",   "XY",   "YX",  "YY",  "XYZ",  "ZXY",  "YZX",  "XXX",
                                             "YYY",  "XIY",  "IXX", "YIY", "XYZI", "IYXY", "XXYY", "ZYZX",
                                             "XZZX", "YXXY", "X",   "Y",   "ZZ",   "IZY"};
    uint32_t seed = 1234;
    for (const auto& label : labels) {
        for (int trial = 0; trial < 3; ++trial) {
            const auto state = RandomState(label.size(), seed++);
            const Complex expected = DenseExpectation(label, state);
            const Complex actual = KernelExpectation(label, state);
            EXPECT_NEAR(std::real(actual), std::real(expected), 1e-10) << "label=" << label << " trial=" << trial;
            EXPECT_NEAR(std::imag(actual), 0.0, 1e-10) << "label=" << label;
            EXPECT_NEAR(std::imag(expected), 0.0, 1e-10) << "label=" << label;
        }
    }
}

TEST(ExpectationMultiFlipTest, Float32MatchesDenseReference) {
    const auto state = RandomState(3, 99);
    std::vector<std::complex<float>> state_f(state.size());
    for (size_t i = 0; i < state.size(); ++i) {
        state_f[i] = std::complex<float>(static_cast<float>(state[i].real()), static_cast<float>(state[i].imag()));
    }
    const auto masks = runtime::ComputePauliMasks({{0, 'Y'}, {1, 'X'}, {2, 'Y'}});
    const auto& table = kernels::GetDispatchF32();
    const auto actual = table.pauli_expect(state_f.data(), state_f.size(), masks.bit_flip_mask, masks.phase_flip_mask,
                                           masks.pivot_qubit, masks.y_count);
    EXPECT_NEAR(std::real(actual), std::real(DenseExpectation("YXY", state)), 1e-5);
}
