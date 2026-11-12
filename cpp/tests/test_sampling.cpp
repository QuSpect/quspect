#include <cmath>

#include <complex>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/kernels/sampling.hpp"

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

TEST(SamplingTest, CumulativeSamplingMatchesTheSingleShotKernel) {
    // The executor builds the cumulative distribution once and searches it for
    // every shot; each sample must be the one table.sampling() returns for the
    // same random number, zero-probability states and the interval ends included.
    const auto& table = GetDispatchF64();
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    std::vector<std::complex<double>> state(64);
    double norm = 0.0;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i % 5 == 0) {
            continue;  // zero amplitude
        }
        state[i] = std::complex<double>(dist01(rng) - 0.5, dist01(rng) - 0.5);
        norm += std::norm(state[i]);
    }
    for (auto& amplitude : state) {
        amplitude /= std::sqrt(norm);
    }

    const std::vector<double> cumulative = cumulative_probabilities(state.data(), state.size());
    ASSERT_EQ(cumulative.size(), state.size());
    EXPECT_NEAR(cumulative.back(), 1.0, 1e-12);

    std::vector<double> draws = {0.0, std::nextafter(1.0, 0.0), 0.5};
    for (int i = 0; i < 2000; ++i) {
        draws.push_back(dist01(rng));
    }
    for (double r : draws) {
        EXPECT_EQ(sample_from_cumulative(cumulative, r), table.sampling(state.data(), state.size(), r)) << r;
    }
}

TEST(SamplingTest, CumulativeSamplingSinglePrecision) {
    const auto& table = GetDispatchF32();
    std::vector<std::complex<float>> state(8);
    for (size_t i = 0; i < state.size(); ++i) {
        state[i] = std::complex<float>(static_cast<float>(i + 1) / 20.0f, 0.1f);
    }
    const std::vector<double> cumulative = cumulative_probabilities(state.data(), state.size());
    for (int i = 0; i < 100; ++i) {
        const double r = (static_cast<double>(i) + 0.5) / 100.0;
        EXPECT_EQ(sample_from_cumulative(cumulative, r), table.sampling(state.data(), state.size(), r)) << r;
    }
}
