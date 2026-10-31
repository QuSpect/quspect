#include <stdexcept>

#include <gtest/gtest.h>

#include "quspect_native/runtime/classical_register.hpp"

using namespace quspect::native::runtime;

TEST(ClassicalRegisterTest, InitZero) {
    ClassicalRegisterFile reg(4);
    EXPECT_EQ(reg.num_clbits(), 4);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(reg.Get(i), 0);
    }
}

TEST(ClassicalRegisterTest, SetAndGet) {
    ClassicalRegisterFile reg(3);
    reg.Set(0, 1);
    reg.Set(2, 1);

    EXPECT_EQ(reg.Get(0), 1);
    EXPECT_EQ(reg.Get(1), 0);
    EXPECT_EQ(reg.Get(2), 1);
}

TEST(ClassicalRegisterTest, ToBitstring) {
    ClassicalRegisterFile reg(3);
    reg.Set(0, 1);
    reg.Set(1, 0);
    reg.Set(2, 1);
    // Index 0 is LSB, so bitstring MSB..LSB = "101"
    EXPECT_EQ(reg.ToBitstring(), "101");
}

TEST(ClassicalRegisterTest, OutOfRange) {
    ClassicalRegisterFile reg(2);
    EXPECT_THROW(reg.Set(2, 0), std::out_of_range);
    EXPECT_THROW(reg.Get(3), std::out_of_range);
}

TEST(ClassicalRegisterTest, InvalidValue) {
    ClassicalRegisterFile reg(1);
    EXPECT_THROW(reg.Set(0, 2), std::invalid_argument);
}
