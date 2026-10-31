// 链接 scalar dispatch 变体（qspect_native_scalar_static）的测试。
// 该变体不含任何 AVX2 目标文件，无论 CPU 是否支持 AVX2，都必须调度到 scalar 内核。

#include <gtest/gtest.h>

#include "quspect_native/kernels/dispatch_table.hpp"

using namespace quspect::native::kernels;

TEST(DispatchScalarTest, CpuDetectionDoesNotCrash) {
    bool avx2 = CpuSupportsAVX2();
    (void) avx2;
}

TEST(DispatchScalarTest, F32BackendIsAlwaysScalar) {
    const auto& table = GetDispatchF32();
    ASSERT_NE(table.backend_name, nullptr);
    EXPECT_STREQ(table.backend_name, "scalar");
    EXPECT_NE(table.x, nullptr);
    EXPECT_NE(table.cnot, nullptr);
    EXPECT_NE(table.dense_k, nullptr);
    EXPECT_NE(table.pauli_expect, nullptr);
}

TEST(DispatchScalarTest, F64BackendIsAlwaysScalar) {
    const auto& table = GetDispatchF64();
    ASSERT_NE(table.backend_name, nullptr);
    EXPECT_STREQ(table.backend_name, "scalar");
    EXPECT_NE(table.x, nullptr);
    EXPECT_NE(table.cnot, nullptr);
    EXPECT_NE(table.dense_k, nullptr);
    EXPECT_NE(table.pauli_expect, nullptr);
}
