#include <cmath>

#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

TEST(SamplingTest, DeterministicSample) {
    const auto& table = GetDispatchF64();
    // |0> 态: 一定采样到 index 0
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);

    uint64_t sample = table.sampling(state.data(), 2, 0.5);
    EXPECT_EQ(sample, 0);
}

TEST(SamplingTest, TwoStateEvenProbability) {
    const auto& table = GetDispatchF64();
    // |+> = (|0> + |1>)/√2: 各 50%
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(inv_sqrt2, 0);

    // random01 = 0.2 -> 应该采样到 0
    EXPECT_EQ(table.sampling(state.data(), 2, 0.2), 0);
    // random01 = 0.8 -> 应该采样到 1
    EXPECT_EQ(table.sampling(state.data(), 2, 0.8), 1);
}

TEST(SamplingTest, LargeState) {
    const auto& table = GetDispatchF64();
    // 3 qubit, |000> only
    std::vector<std::complex<double>> state(8);
    state[0] = std::complex<double>(1, 0);

    uint64_t sample = table.sampling(state.data(), 8, 0.99);
    EXPECT_EQ(sample, 0);
}

TEST(SamplingTest, ProbabilityZero) {
    const auto& table = GetDispatchF64();
    // |0> 态: P(q0=0) = 1.0
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);

    double p0 = table.prob0(state.data(), 2, 0);
    EXPECT_NEAR(p0, 1.0, 1e-10);
}

TEST(SamplingTest, ProbabilityZeroPlus) {
    const auto& table = GetDispatchF64();
    // |+> 态: P(q0=0) = 0.5
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(inv_sqrt2, 0);

    double p0 = table.prob0(state.data(), 2, 0);
    EXPECT_NEAR(p0, 0.5, 1e-10);
}
