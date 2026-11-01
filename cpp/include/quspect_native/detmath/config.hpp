#pragma once

// 确定性数值层（detmath）的编译规则。
//
// detmath 的目标是：同一输入在所有平台（macOS arm64 / clang、Linux x86_64 与 aarch64 / gcc、
// Windows AMD64 / MSVC）上得到逐位相同的结果。为此：
//
//   * 只使用 IEEE 754 正确舍入的运算：+ − × ÷、sqrt、显式的 std::fma，以及 CORE-MATH 提供的
//     正确舍入超越函数（scalar.hpp）；不调用平台 libm 的超越函数、BLAS / LAPACK；
//   * 运算顺序在源码中固定：不允许编译器把 a·b + c 收缩成 FMA（clang 在 arm64 上默认收缩，
//     gcc 在 aarch64 上默认 -ffp-contract=fast），不允许 -ffast-math 式的重结合；
//   * 不使用 SIMD 水平归约、不依赖线程数。
//
// 头文件中的内联函数会在包含它的编译单元中实例化，所以每个包含 detmath 运算头文件的源文件都
// 必须按上述规则编译（CMake 中的 QSPECT_EXACT_FP_SRCS：GCC / Clang 为 -ffp-contract=off
// -fno-fast-math，MSVC 为 /fp:precise，并定义 QSPECT_EXACT_FP=1）。否则同一个内联函数在不同
// 编译单元中可能被编译成不同的指令序列，链接器任取其一。
#if !defined(QSPECT_EXACT_FP)
#error "detmath headers require the exact floating-point compile rules (add the source to QSPECT_EXACT_FP_SRCS)"
#endif

#if defined(__FAST_MATH__)
#error "detmath must not be compiled with -ffast-math"
#endif

#if defined(_M_IX86) && !defined(_M_IX86_FP)
#error "detmath requires SSE2 floating point (x87 extended precision is not deterministic)"
#endif
#if defined(__i386__) && !defined(__SSE2_MATH__)
#error "detmath requires SSE2 floating point (x87 extended precision is not deterministic)"
#endif
