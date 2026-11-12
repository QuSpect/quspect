#include <cmath>

#include <complex>
#include <new>
#include <string>

#include <gtest/gtest.h>

#include "quspect_native/runtime/state_vector.hpp"

using namespace quspect::native::runtime;

TEST(StateVectorTest, InitZeroStateDouble) {
    StateVector<double> sv(3);
    EXPECT_EQ(sv.num_qubits(), 3);
    EXPECT_EQ(sv.dim(), 8);

    // |000> = 1, rest = 0
    EXPECT_NEAR(std::real(sv.data()[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::imag(sv.data()[0]), 0.0, 1e-12);
    for (uint64_t i = 1; i < sv.dim(); ++i) {
        EXPECT_NEAR(std::real(sv.data()[i]), 0.0, 1e-12);
        EXPECT_NEAR(std::imag(sv.data()[i]), 0.0, 1e-12);
    }
}

TEST(StateVectorTest, InitZeroStateFloat) {
    StateVector<float> sv(2);
    EXPECT_EQ(sv.num_qubits(), 2);
    EXPECT_EQ(sv.dim(), 4);
    EXPECT_NEAR(std::real(sv.data()[0]), 1.0f, 1e-6f);
}

TEST(StateVectorTest, InitBasisState) {
    // |101> = basis 5 (little-endian: q0=1, q1=0, q2=1)
    StateVector<double> sv(3, 5);
    for (uint64_t i = 0; i < sv.dim(); ++i) {
        if (i == 5) {
            EXPECT_NEAR(std::real(sv.data()[i]), 1.0, 1e-12);
        } else {
            EXPECT_NEAR(std::real(sv.data()[i]), 0.0, 1e-12);
        }
    }
}

TEST(StateVectorTest, SetZeroStateResets) {
    StateVector<double> sv(2, 3);  // |11>
    sv.SetZeroState();             // back to |00>
    EXPECT_NEAR(std::real(sv.data()[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::real(sv.data()[3]), 0.0, 1e-12);
}

TEST(StateVectorTest, Normalize) {
    StateVector<double> sv(1);
    sv.data()[0] = std::complex<double>(3, 0);
    sv.data()[1] = std::complex<double>(4, 0);
    sv.Normalize();

    EXPECT_NEAR(std::real(sv.data()[0]), 0.6, 1e-12);
    EXPECT_NEAR(std::real(sv.data()[1]), 0.8, 1e-12);
}

TEST(StateVectorTest, SquaredNorm) {
    StateVector<double> sv(1);
    sv.data()[0] = std::complex<double>(1, 0);
    sv.data()[1] = std::complex<double>(2, 0);
    EXPECT_NEAR(sv.SquaredNorm(), 5.0, 1e-12);
}

TEST(StateVectorTest, LargeQubitCount) {
    // 确保 10 qubit (1024 维) 不会出错
    StateVector<double> sv(10);
    EXPECT_EQ(sv.dim(), 1024);
    EXPECT_NEAR(std::real(sv.data()[0]), 1.0, 1e-12);
}

TEST(StateVectorTest, UnallocatableWidthNamesTheState) {
    // 2^56 amplitudes and more exceed every address space, so nothing is allocated.  The failure
    // must be an allocation error (MemoryError in Python) that names the width, not the
    // std::length_error text "vector".
    for (uint32_t num_qubits : {56U, 59U, 60U, 63U, 64U, 200U}) {
        try {
            StateVector<double> sv(num_qubits);
            FAIL() << "allocated " << num_qubits << " qubits";
        } catch (const std::bad_alloc& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("state vector of " + std::to_string(num_qubits) + " qubits"), std::string::npos)
                << message;
            EXPECT_NE(message.find("2^" + std::to_string(num_qubits + 4) + " bytes"), std::string::npos) << message;
        }
    }
    try {
        StateVector<float> sv(60, 1);
        FAIL() << "allocated 60 qubits";
    } catch (const std::bad_alloc& error) {
        EXPECT_NE(std::string(error.what()).find("2^63 bytes"), std::string::npos) << error.what();
    }
}
