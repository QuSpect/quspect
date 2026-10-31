#include <cmath>

#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

namespace {

/// 测试任二 single-qubit gate 在 2-qubit 系统上的结果
template <typename FP>
void TestGateOnQubit0(SingleQubitKernel<FP> gate, const std::complex<FP>& expected00,
                      const std::complex<FP>& expected10) {
    constexpr uint32_t n = 2;
    std::vector<std::complex<FP>> state(1ULL << n);
    state[0] = std::complex<FP>(1, 0);  // |00>

    // 用一个虚拟矩阵调用 — 对于 named gate 快路径，matrix 参数被忽略
    std::complex<FP> dummy[4] = {};
    gate(state.data(), 1ULL << n, 0, dummy);

    // |00> = 经过 H|0> 门作用后
    EXPECT_NEAR(std::real(state[0]), std::real(expected00), 1e-6) << "real part mismatch at |00>";
    EXPECT_NEAR(std::imag(state[0]), std::imag(expected00), 1e-6) << "imag part mismatch at |00>";

    // |10> = |1>_0 ⊗ |0>_1
    EXPECT_NEAR(std::real(state[2]), std::real(expected10), 1e-6);
    EXPECT_NEAR(std::imag(state[2]), std::imag(expected10), 1e-6);
}

template <typename FP>
void TestZOnHighQubitAgainstScalarReference() {
    // q3 makes an entire AVX2 batch require a phase flip.  This specifically
    // catches confusing the register width in scalars with its width in
    // complex amplitudes (2 complex<double> / 4 complex<float>).
    constexpr uint64_t dim = 16;
    constexpr uint32_t qubit = 3;
    std::vector<std::complex<FP>> state(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        state[i] = {static_cast<FP>(0.125 * static_cast<double>(i + 1)),
                    static_cast<FP>(-0.0625 * static_cast<double>(i + 2))};
    }
    auto expected = state;
    for (uint64_t i = 0; i < dim; ++i) {
        if ((i >> qubit) & 1ULL) {
            expected[i] = -expected[i];
        }
    }

    const auto& table = []() -> const auto& {
        if constexpr (std::is_same_v<FP, double>) {
            return GetDispatchF64();
        } else {
            return GetDispatchF32();
        }
    }();
    table.z(state.data(), dim, qubit, nullptr);

    const FP tolerance = std::is_same_v<FP, double> ? FP(1e-12) : FP(1e-6);
    for (uint64_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(std::real(state[i]), std::real(expected[i]), tolerance) << "basis " << i;
        EXPECT_NEAR(std::imag(state[i]), std::imag(expected[i]), tolerance) << "basis " << i;
    }
}

}  // namespace

// qubit 0 = basis bit 0 (little-endian)
// |q1, q0>: index = q0 + q1*2
// |00> = idx0, |01> = idx1, |10> = idx2, |11> = idx3

TEST(SingleQubitGateTest, XOnQubit0) {
    const auto& table = GetDispatchF64();
    // X on q0: |00> -> |01> (index 0 -> index 1)
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.x(state.data(), 4, 0, dummy);

    EXPECT_NEAR(std::real(state[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(state[1]), 1.0, 1e-12);  // |01>
}

TEST(SingleQubitGateTest, HOnQubit0) {
    const auto& table = GetDispatchF64();
    // H on q0: |00> -> (|00> + |01>)/√2
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.h(state.data(), 4, 0, dummy);

    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(state[0]), inv_sqrt2, 1e-12);  // |00>
    EXPECT_NEAR(std::real(state[1]), inv_sqrt2, 1e-12);  // |01>
}

TEST(SingleQubitGateTest, ZOnQubit0) {
    const auto& table = GetDispatchF64();
    // Z|+> = |->
    // First make |+> on q0: H|0>
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.h(state.data(), 4, 0, dummy);
    // Now apply Z on q0
    table.z(state.data(), 4, 0, dummy);

    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    // |-> = (|0> - |1>)/√2 on q0: idx0=+inv_sqrt2, idx1=-inv_sqrt2
    EXPECT_NEAR(std::real(state[0]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::real(state[1]), -inv_sqrt2, 1e-12);
}

TEST(SingleQubitGateTest, ZOnHighQubitMatchesScalarReferenceFloat64) {
    TestZOnHighQubitAgainstScalarReference<double>();
}

TEST(SingleQubitGateTest, ZOnHighQubitMatchesScalarReferenceFloat32) {
    TestZOnHighQubitAgainstScalarReference<float>();
}

TEST(SingleQubitGateTest, YOnQubit0) {
    const auto& table = GetDispatchF64();
    // Y|0> = i|1>: |00> -> i|01> (idx 1)
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.y(state.data(), 4, 0, dummy);

    EXPECT_NEAR(std::real(state[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(state[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::imag(state[1]), 1.0, 1e-12);
}

TEST(SingleQubitGateTest, HOnQubit1) {
    const auto& table = GetDispatchF64();
    // |0>_1 上施加 H: |00> -> |0> ⊗ H|0> = |00> + |01> / √2
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.h(state.data(), 4, 1, dummy);

    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    // |01> has index 1 << 1 = 2? No — little endian:
    // qubit 1 = bit 1, bit 1=1 -> index 2
    EXPECT_NEAR(std::real(state[0]), inv_sqrt2, 1e-12);  // |00>
    EXPECT_NEAR(std::real(state[2]), inv_sqrt2, 1e-12);  // |01>_le = bit 1 = 1 -> index 2
}

TEST(SingleQubitGateTest, S_T_Dagger) {
    const auto& table = GetDispatchF64();
    std::complex<double> dummy[4] = {};

    // S|1> = i|1>
    {
        std::vector<std::complex<double>> state(2);
        state[1] = std::complex<double>(1, 0);
        table.s(state.data(), 2, 0, dummy);
        EXPECT_NEAR(std::imag(state[1]), 1.0, 1e-12);
    }
    // Sdg|1> = -i|1>
    {
        std::vector<std::complex<double>> state(2);
        state[1] = std::complex<double>(1, 0);
        table.sdg(state.data(), 2, 0, dummy);
        EXPECT_NEAR(std::imag(state[1]), -1.0, 1e-12);
    }
    // T|1> = e^{iπ/4}|1>
    {
        std::vector<std::complex<double>> state(2);
        state[1] = std::complex<double>(1, 0);
        table.t(state.data(), 2, 0, dummy);
        double phase = 1.0 / std::sqrt(2.0);
        EXPECT_NEAR(std::real(state[1]), phase, 1e-12);
        EXPECT_NEAR(std::imag(state[1]), phase, 1e-12);
    }
    // Tdg|1> = e^{-iπ/4}|1>
    {
        std::vector<std::complex<double>> state(2);
        state[1] = std::complex<double>(1, 0);
        table.tdg(state.data(), 2, 0, dummy);
        double phase = 1.0 / std::sqrt(2.0);
        EXPECT_NEAR(std::real(state[1]), phase, 1e-12);
        EXPECT_NEAR(std::imag(state[1]), -phase, 1e-12);
    }
}
