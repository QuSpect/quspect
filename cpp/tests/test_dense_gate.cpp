#include <cmath>

#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

TEST(DenseGateTest, SingleQubitDenseGate) {
    const auto& table = GetDispatchF64();
    // X on qubit 0 via dense gate: |00> -> |01> (index 0 -> index 1)
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);  // |00>

    uint32_t target = 0;
    // X matrix row-major: [[0,1],[1,0]]
    std::vector<std::complex<double>> matrix = {
        {0, 0},
        {1, 0},
        {1, 0},
        {0, 0},
    };

    table.dense_k(state.data(), 4, &target, 1, matrix.data());

    // |00> -> X on q0 -> |01> (index 1)
    EXPECT_NEAR(std::real(state[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(state[1]), 1.0, 1e-12);
}

TEST(DenseGateTest, SingleQubitIdentity) {
    const auto& table = GetDispatchF64();
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);

    uint32_t target = 1;
    std::vector<std::complex<double>> matrix = {
        {1, 0},
        {0, 0},
        {0, 0},
        {1, 0},
    };

    table.dense_k(state.data(), 4, &target, 1, matrix.data());

    EXPECT_NEAR(std::real(state[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::real(state[2]), 0.0, 1e-12);
}

TEST(DenseGateTest, DiagonalGate) {
    const auto& table = GetDispatchF64();
    // Z = diag(1, -1) on qubit 0, acting on |++>
    std::vector<std::complex<double>> state(4);
    state[0] = std::complex<double>(1, 0);
    std::complex<double> dummy[4] = {};
    table.h(state.data(), 4, 0, dummy);
    table.h(state.data(), 4, 1, dummy);
    // |++>: all = 0.5

    uint32_t target = 0;
    std::vector<std::complex<double>> diag = {
        {1, 0},   // local 0: no phase
        {-1, 0},  // local 1: -1 phase
    };

    table.diagonal_k(state.data(), 4, &target, 1, diag.data());

    // Z on q0: flips sign when q0=1
    // idx 0: q0=0 → +0.5
    // idx 1: q0=1 → -0.5
    // idx 2: q0=0 → +0.5
    // idx 3: q0=1 → -0.5
    EXPECT_NEAR(std::real(state[0]), 0.5, 1e-12);
    EXPECT_NEAR(std::real(state[1]), -0.5, 1e-12);
    EXPECT_NEAR(std::real(state[2]), 0.5, 1e-12);
    EXPECT_NEAR(std::real(state[3]), -0.5, 1e-12);
}
