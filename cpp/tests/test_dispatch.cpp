#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

TEST(DispatchTest, CpuDetectionDoesNotCrash) {
    // CPU 检测至少不应崩溃
    bool avx2 = CpuSupportsAVX2();
    (void) avx2;
}

TEST(DispatchTest, GetDispatchF32ReturnsTable) {
    const auto& table = GetDispatchF32();
    EXPECT_NE(table.backend_name, nullptr);
    const char* expected = CpuSupportsAVX2() ? "avx2" : "scalar";
    EXPECT_STREQ(table.backend_name, expected);

    // 单比特门不应为 null
    EXPECT_NE(table.x, nullptr);
    EXPECT_NE(table.y, nullptr);
    EXPECT_NE(table.z, nullptr);
    EXPECT_NE(table.h, nullptr);
    EXPECT_NE(table.s, nullptr);
    EXPECT_NE(table.sdg, nullptr);
    EXPECT_NE(table.t, nullptr);
    EXPECT_NE(table.tdg, nullptr);
    EXPECT_NE(table.rx, nullptr);
    EXPECT_NE(table.ry, nullptr);
    EXPECT_NE(table.rz, nullptr);

    // 双比特门不应为 null
    EXPECT_NE(table.cnot, nullptr);
    EXPECT_NE(table.cz, nullptr);
    EXPECT_NE(table.swap, nullptr);

    // 通用门不应为 null
    EXPECT_NE(table.dense_k, nullptr);
    EXPECT_NE(table.diagonal_k, nullptr);

    // 统计函数不应为 null
    EXPECT_NE(table.prob0, nullptr);
    EXPECT_NE(table.sampling, nullptr);
    EXPECT_NE(table.pauli_expect, nullptr);
}

TEST(DispatchTest, GetDispatchF64ReturnsTable) {
    const auto& table = GetDispatchF64();
    EXPECT_NE(table.backend_name, nullptr);
    const char* expected = CpuSupportsAVX2() ? "avx2" : "scalar";
    EXPECT_STREQ(table.backend_name, expected);

    EXPECT_NE(table.x, nullptr);
    EXPECT_NE(table.h, nullptr);
    EXPECT_NE(table.cnot, nullptr);
    EXPECT_NE(table.dense_k, nullptr);
}
