#include <cmath>

#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

namespace {

template <typename FP>
void TestCZOnHighQubitsAgainstScalarReference() {
    constexpr uint64_t dim = 16;
    constexpr uint32_t control = 2;
    constexpr uint32_t target = 3;
    const uint64_t control_mask = 1ULL << control;
    std::vector<std::complex<FP>> state(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        state[i] = {static_cast<FP>(0.0625 * static_cast<double>(i + 1)),
                    static_cast<FP>(-0.03125 * static_cast<double>(i + 3))};
    }
    auto expected = state;
    for (uint64_t i = 0; i < dim; ++i) {
        if ((i & control_mask) == control_mask && ((i >> target) & 1ULL)) {
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
    table.cz(state.data(), dim, target, control_mask, control_mask, nullptr);

    const FP tolerance = std::is_same_v<FP, double> ? FP(1e-12) : FP(1e-6);
    for (uint64_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(std::real(state[i]), std::real(expected[i]), tolerance) << "basis " << i;
        EXPECT_NEAR(std::imag(state[i]), std::imag(expected[i]), tolerance) << "basis " << i;
    }
}

}  // namespace

TEST(TwoQubitGateTest, CNOT00) {
    const auto& table = GetDispatchF64();
    // CNOT: control=0, target=1, |00> -> |00>
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.cnot(state.data(), 4, 1, 1ULL << 0, 1ULL << 0, dummy);

    EXPECT_NEAR(std::real(state[0]), 1.0, 1e-12);  // still |00>
    EXPECT_NEAR(std::real(state[3]), 0.0, 1e-12);
}

TEST(TwoQubitGateTest, CNOT11) {
    const auto& table = GetDispatchF64();
    // |11> -> |10>: state[3] -> state[1]
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    // 先准备 |11>: X on q0 and q1
    table.x(state.data(), 4, 0, dummy);
    table.x(state.data(), 4, 1, dummy);

    // 验证初始 |11>
    EXPECT_NEAR(std::real(state[3]), 1.0, 1e-12);

    // CNOT control=0, target=1
    table.cnot(state.data(), 4, 1, 1ULL << 0, 1ULL << 0, dummy);

    // |11> -> |10> (index 1 in le)
    EXPECT_NEAR(std::real(state[3]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(state[1]), 1.0, 1e-12);
}

TEST(TwoQubitGateTest, CZ) {
    const auto& table = GetDispatchF64();
    // CZ: control=0 (bit 0), target=1 (bit 1)
    // For |11> (index 3): both bits 1, 0 are set -> apply phase -1
    std::vector<std::complex<double>> state(4);
    std::complex<double> dummy[4] = {};
    // Prepare |+>|+>: H on both qubits
    state[0] = std::complex<double>(1, 0);
    table.h(state.data(), 4, 0, dummy);
    table.h(state.data(), 4, 1, dummy);
    // Now |++>: all amplitudes = 0.5

    table.cz(state.data(), 4, 1, 1ULL << 0, 1ULL << 0, dummy);

    // |11> should have phase flip
    EXPECT_NEAR(std::real(state[3]), -0.5, 1e-12);
    // Others unchanged
    EXPECT_NEAR(std::real(state[0]), 0.5, 1e-12);
    EXPECT_NEAR(std::real(state[1]), 0.5, 1e-12);
    EXPECT_NEAR(std::real(state[2]), 0.5, 1e-12);
}

TEST(TwoQubitGateTest, CZOnHighQubitsMatchesScalarReferenceFloat64) {
    TestCZOnHighQubitsAgainstScalarReference<double>();
}

TEST(TwoQubitGateTest, CZOnHighQubitsMatchesScalarReferenceFloat32) {
    TestCZOnHighQubitsAgainstScalarReference<float>();
}

TEST(TwoQubitGateTest, SWAP) {
    const auto& table = GetDispatchF64();
    // |01>_le = bit0=1, bit1=0 -> idx 1
    std::vector<std::complex<double>> state(4);
    state[1] = std::complex<double>(1, 0);  // |01>

    table.swap(state.data(), 4, 0, 1);

    // After swap: |10>_le = bit0=0, bit1=1 -> idx 2
    EXPECT_NEAR(std::real(state[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(state[2]), 1.0, 1e-12);
}
