// bit_ops.hpp — 可移植的 64 位位运算工具
//
// GCC / Clang 使用编译器内建函数；MSVC 没有 __builtin_*，因此：
//   - Ctz64 使用 _BitScanForward64（x64 / ARM64）或两次 _BitScanForward（x86）
//   - Popcount64 / Parity64 使用纯软件实现
//
// 注意：MSVC 下刻意不使用 __popcnt64。它会直接生成 POPCNT 指令，而 scalar
// 扩展模块需要能在不支持 POPCNT 的老 CPU 上运行。

#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#endif

namespace quspect::native::common {

/// 软件 popcount（SWAR 算法），所有平台均可用，主要用于 MSVC 和测试对照。
inline int PortablePopcount64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<int>((x * 0x0101010101010101ULL) >> 56);
}

/// 返回 x 中置 1 的位数。
inline int Popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    return PortablePopcount64(x);
#endif
}

/// 返回 x 最低置位的下标（trailing zero count）。x == 0 时返回 64。
inline int Ctz64(uint64_t x) {
    if (x == 0) {
        return 64;
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long index = 0;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
#elif defined(_MSC_VER)
    unsigned long index = 0;
    const auto low = static_cast<unsigned long>(x & 0xFFFFFFFFULL);
    if (low != 0) {
        _BitScanForward(&index, low);
        return static_cast<int>(index);
    }
    _BitScanForward(&index, static_cast<unsigned long>(x >> 32));
    return static_cast<int>(index) + 32;
#else
    int count = 0;
    while ((x & 1ULL) == 0) {
        x >>= 1;
        ++count;
    }
    return count;
#endif
}

/// 返回 x 中置 1 位数的奇偶性：奇数个返回 1，偶数个返回 0。
inline int Parity64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_parityll(x);
#else
    return PortablePopcount64(x) & 1;
#endif
}

}  // namespace quspect::native::common
