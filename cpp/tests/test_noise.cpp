#include <cmath>

#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "quspect_native/kernels/noise_kernels.hpp"
#include "quspect_native/runtime/density_matrix.hpp"

namespace quspect::native {
namespace {

using Complex = std::complex<double>;

// ============================================================
// DensityMatrix tests
// ============================================================

TEST(DensityMatrixTest, InitPure) {
    runtime::DensityMatrix<double> dm(2);
    EXPECT_EQ(dm.num_qubits(), 2);
    EXPECT_EQ(dm.dim(), 4);
    // |00><00| — first element should be 1
    EXPECT_NEAR(std::real(dm.data()[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::imag(dm.data()[0]), 0.0, 1e-12);
    // All other elements should be 0
    for (uint64_t i = 1; i < dm.size(); ++i) {
        EXPECT_NEAR(std::abs(dm.data()[i]), 0.0, 1e-12);
    }
}

TEST(DensityMatrixTest, Trace) {
    runtime::DensityMatrix<double> dm(2);
    double tr = dm.Trace();
    EXPECT_NEAR(tr, 1.0, 1e-12);
}

TEST(DensityMatrixTest, ApplyUnitaryX) {
    // Apply X on qubit 0 of |00><00| → |01><01|
    // Convention: qubit 0 = bit 0 (LSB), |q1,q0>: idx = q0 + 2*q1
    // X on q0: |00> → |01> (idx 1)
    runtime::DensityMatrix<double> dm(2);
    uint32_t qubits[] = {0};
    // X matrix (row-major): [[0, 1], [1, 0]]
    Complex xmat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    dm.ApplyUnitary(xmat, qubits, 1);

    // Result should be |01><01|: element (1, 1) = 1
    uint64_t idx = 1 * dm.dim() + 1;  // row 1, col 1
    EXPECT_NEAR(std::real(dm.data()[idx]), 1.0, 1e-12);
    EXPECT_NEAR(dm.Trace(), 1.0, 1e-12);
}

TEST(DensityMatrixTest, ApplyKrausChannelDepolarizing) {
    // Depolarizing channel on 1 qubit with p=0.75
    // Kraus: K0 = sqrt(1-3p/4)*I, K1 = sqrt(p)/2 * X, K2 = sqrt(p)/2 * Y,
    // K3 = sqrt(p)/2 * Z
    // For p=0.75: sqrt(1-3*0.75/4) = sqrt(0.4375) = 0.6614...
    // sqrt(p)/2 = sqrt(0.75)/2 = 0.4330...
    double p = 0.75;
    double s1 = std::sqrt(1 - 3 * p / 4);
    double s2 = std::sqrt(p) / 2;

    Complex I_mat[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};   // [[1,0],[0,1]]
    Complex X_mat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};   // [[0,1],[1,0]]
    Complex Y_mat[4] = {{0, 0}, {0, -1}, {0, 1}, {0, 0}};  // [[0,-i],[i,0]]
    Complex Z_mat[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};  // [[1,0],[0,-1]]

    // concatenated kraus_ops: K0, K1, K2, K3
    Complex kraus_ops[16];
    for (int i = 0; i < 4; ++i)
        kraus_ops[i] = I_mat[i] * s1;
    for (int i = 0; i < 4; ++i)
        kraus_ops[4 + i] = X_mat[i] * s2;
    for (int i = 0; i < 4; ++i)
        kraus_ops[8 + i] = Y_mat[i] * s2;
    for (int i = 0; i < 4; ++i)
        kraus_ops[12 + i] = Z_mat[i] * s2;

    runtime::DensityMatrix<double> dm(1);
    uint32_t qubits[] = {0};
    dm.ApplyKrausChannel(kraus_ops, 4, qubits, 1);

    // For p=0.75 with this parameterization:
    //   ρ'[0,0] = (1-3p/4 + p/4) = 1 - p/2 = 0.625
    //   ρ'[1,1] = (p/4 + p/4) = p/2 = 0.375
    EXPECT_NEAR(dm.Trace(), 1.0, 1e-12);
    EXPECT_NEAR(std::real(dm.data()[0]), 0.625, 1e-2);             // first diagonal
    EXPECT_NEAR(std::real(dm.data()[dm.dim() + 1]), 0.375, 1e-2);  // second
}

// A unitary must preserve the matrix: entries of any size survive, and only exact zeros of the
// operator may be skipped.  Entries below sqrt(1e-15) ~ 3.16e-8 used to be zeroed at every gate.

TEST(DensityMatrixTest, IdentityKeepsSmallEntries) {
    runtime::DensityMatrix<double> dm(1);
    dm.data()[0] = Complex{1.0 - 1e-9, 0};
    dm.data()[1] = Complex{2e-10, -3e-10};
    dm.data()[2] = Complex{2e-10, 3e-10};
    dm.data()[3] = Complex{1e-9, 0};
    const std::vector<Complex> before(dm.data(), dm.data() + dm.size());

    uint32_t qubits[] = {0};
    Complex identity[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    dm.ApplyUnitary(identity, qubits, 1);

    for (uint64_t i = 0; i < dm.size(); ++i) {
        EXPECT_EQ(dm.data()[i], before[i]) << "entry " << i;
    }
}

TEST(DensityMatrixTest, UnitaryKeepsSmallPopulationAndTinyRotation) {
    // X swaps the populations exactly, whatever their size.
    runtime::DensityMatrix<double> swapped(1);
    swapped.data()[0] = Complex{1.0 - 1e-12, 0};
    swapped.data()[3] = Complex{1e-12, 0};
    uint32_t qubits[] = {0};
    Complex xmat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    swapped.ApplyUnitary(xmat, qubits, 1);
    EXPECT_EQ(swapped.data()[0], (Complex{1e-12, 0}));
    EXPECT_EQ(swapped.data()[3], (Complex{1.0 - 1e-12, 0}));

    // RX(2e-9): off-diagonal entries of the gate are -i * 1e-9, below the old cut.
    const double half = 1e-9;
    Complex rx[4] = {{std::cos(half), 0}, {0, -std::sin(half)}, {0, -std::sin(half)}, {std::cos(half), 0}};
    runtime::DensityMatrix<double> rotated(1);
    rotated.ApplyUnitary(rx, qubits, 1);
    // |0><0| -> [[cos^2, i sin cos], [-i sin cos, sin^2]]
    EXPECT_NEAR(std::imag(rotated.data()[1]), std::sin(half) * std::cos(half), 1e-24);
    EXPECT_NEAR(std::imag(rotated.data()[2]), -std::sin(half) * std::cos(half), 1e-24);
    EXPECT_NEAR(std::real(rotated.data()[3]), std::sin(half) * std::sin(half), 1e-33);
    EXPECT_NEAR(rotated.Trace(), 1.0, 1e-15);
}

TEST(DensityMatrixTest, LongSmallAngleCircuitPreservesTrace) {
    // 2000 Trotter-like steps of small rotations on three qubits: populations of the excited
    // states stay far below 3.16e-8 for most of the circuit.
    const uint32_t num_qubits = 3;
    const double half = 1e-5;
    Complex rx[4] = {{std::cos(half), 0}, {0, -std::sin(half)}, {0, -std::sin(half)}, {std::cos(half), 0}};
    Complex cx[16] = {};
    cx[0 * 4 + 0] = cx[1 * 4 + 3] = cx[2 * 4 + 2] = cx[3 * 4 + 1] = Complex{1, 0};

    runtime::DensityMatrix<double> dm(num_qubits);
    std::vector<Complex> state(uint64_t{1} << num_qubits, Complex{0, 0});
    state[0] = Complex{1, 0};
    for (int step = 0; step < 2000; ++step) {
        for (uint32_t q = 0; q + 1 < num_qubits; ++q) {
            uint32_t pair[] = {q, q + 1};
            dm.ApplyUnitary(cx, pair, 2);
            kernels::apply_dense_unitary<double>(state.data(), state.size(), pair, 2, cx);
        }
        for (uint32_t q = 0; q < num_qubits; ++q) {
            uint32_t one[] = {q};
            dm.ApplyUnitary(rx, one, 1);
            kernels::apply_dense_unitary<double>(state.data(), state.size(), one, 1, rx);
        }
    }

    EXPECT_NEAR(dm.Trace(), 1.0, 1e-12);
    for (uint64_t row = 0; row < dm.dim(); ++row) {
        for (uint64_t col = 0; col < dm.dim(); ++col) {
            const Complex expected = state[row] * std::conj(state[col]);
            EXPECT_NEAR(std::abs(dm.data()[row * dm.dim() + col] - expected), 0.0, 1e-13)
                << "entry (" << row << ", " << col << ")";
        }
    }
}

TEST(DensityMatrixTest, KrausChannelKeepsSmallPopulation) {
    // Bit flip with probability p on diag(1 - e, e): diag((1-p)(1-e) + p e, (1-p) e + p (1-e)).
    const double p = 1e-10;
    const double e = 1e-12;
    const double s0 = std::sqrt(1 - p);
    const double s1 = std::sqrt(p);
    Complex kraus_ops[8] = {{s0, 0}, {0, 0}, {0, 0}, {s0, 0}, {0, 0}, {s1, 0}, {s1, 0}, {0, 0}};

    runtime::DensityMatrix<double> dm(1);
    dm.data()[0] = Complex{1 - e, 0};
    dm.data()[3] = Complex{e, 0};
    uint32_t qubits[] = {0};
    dm.ApplyKrausChannel(kraus_ops, 2, qubits, 1);

    const double excited = (1 - p) * e + p * (1 - e);
    EXPECT_NEAR(std::real(dm.data()[3]), excited, 1e-12 * excited);
    EXPECT_NEAR(dm.Trace(), 1.0, 1e-15);
}

TEST(DensityMatrixTest, RejectsQubitCountsWhoseMatrixCannotExist) {
    // dim * dim overflowed for 32..63 qubits (an empty buffer that was then written) and the
    // shift itself for 64 and more.
    const uint32_t max_qubits = runtime::DensityMatrix<double>::MaxQubits();
    EXPECT_EQ(max_qubits, 29U);
    EXPECT_EQ(runtime::DensityMatrix<float>::MaxQubits(), 29U);
    for (const uint32_t num_qubits : {max_qubits + 1, uint32_t{31}, uint32_t{32}, uint32_t{40}, uint32_t{63},
                                      uint32_t{64}, uint32_t{65}, std::numeric_limits<uint32_t>::max()}) {
        EXPECT_THROW(runtime::DensityMatrix<double>{num_qubits}, std::invalid_argument) << num_qubits << " qubits";
        EXPECT_THROW(runtime::DensityMatrix<float>{num_qubits}, std::invalid_argument) << num_qubits << " qubits";
    }
    const runtime::DensityMatrix<double> empty(0);
    EXPECT_EQ(empty.dim(), 1U);
    EXPECT_EQ(empty.size(), 1U);
}

// ============================================================
// Noise kernel tests
// ============================================================

TEST(NoiseKernelTest, ApplyDenseUnitary) {
    // Apply X gate to |0> state → |1>
    std::vector<Complex> state = {{1, 0}, {0, 0}};
    uint32_t qubits[] = {0};
    // X matrix
    Complex xmat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};

    kernels::apply_dense_unitary<double>(state.data(), 2, qubits, 1, xmat);

    EXPECT_NEAR(std::abs(state[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state[1]), 1.0, 1e-12);
}

TEST(NoiseKernelTest, ApplyKrausMonteCarlo) {
    // Apply X gate via Monte Carlo (deterministic: only one Kraus op = X)
    std::vector<Complex> state = {{1, 0}, {0, 0}};
    uint32_t qubits[] = {0};
    Complex xmat[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    std::mt19937_64 rng(42);

    uint32_t selected = kernels::apply_kraus_monte_carlo<double>(state.data(), 2, xmat, 1, qubits, 1, rng);

    EXPECT_EQ(selected, 0);
    EXPECT_NEAR(std::abs(state[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(state[1]), 1.0, 1e-12);
}

TEST(NoiseKernelTest, KrausMonteCarloProbabilities) {
    // Identity channel: two Kraus ops, both are I/√2
    // Should sample each with ~50% probability
    std::vector<Complex> state = {{1, 0}, {0, 0}};
    uint32_t qubits[] = {0};
    Complex inv_sqrt2 = {1.0 / std::sqrt(2.0), 0};
    // K0 = I/√2, K1 = I/√2
    Complex kraus_ops[8];
    for (int i = 0; i < 4; ++i)
        kraus_ops[i] = (i == 0 || i == 3) ? inv_sqrt2 : Complex{0, 0};
    for (int i = 0; i < 4; ++i)
        kraus_ops[4 + i] = (i == 0 || i == 3) ? inv_sqrt2 : Complex{0, 0};

    int count0 = 0;
    int count1 = 0;
    std::mt19937_64 rng(12345);
    for (int shot = 0; shot < 1000; ++shot) {
        std::vector<Complex> sv = {{1, 0}, {0, 0}};
        uint32_t sel = kernels::apply_kraus_monte_carlo<double>(sv.data(), 2, kraus_ops, 2, qubits, 1, rng);
        if (sel == 0)
            count0++;
        else
            count1++;
    }
    // Should be roughly 50/50
    EXPECT_GT(count0, 400);
    EXPECT_GT(count1, 400);
    // State after either Kraus op should still be |0>
    // (I/√2 |0> = |0>/√2, then normalized to |0>)
}

}  // namespace
}  // namespace quspect::native
