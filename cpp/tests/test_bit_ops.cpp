#include <cstdint>
#include <random>

#include <gtest/gtest.h>

#include "quspect_native/common/bit_ops.hpp"

using quspect::native::common::Ctz64;
using quspect::native::common::Parity64;
using quspect::native::common::Popcount64;
using quspect::native::common::PortablePopcount64;

namespace {

// 逐位计数的参考实现
int ReferencePopcount(uint64_t x) {
    int count = 0;
    for (int bit = 0; bit < 64; ++bit) {
        count += static_cast<int>((x >> bit) & 1ULL);
    }
    return count;
}

int ReferenceCtz(uint64_t x) {
    for (int bit = 0; bit < 64; ++bit) {
        if ((x >> bit) & 1ULL) {
            return bit;
        }
    }
    return 64;
}

}  // namespace

TEST(BitOpsTest, PopcountEdgeCases) {
    EXPECT_EQ(Popcount64(0), 0);
    EXPECT_EQ(Popcount64(1), 1);
    EXPECT_EQ(Popcount64(~0ULL), 64);
    EXPECT_EQ(Popcount64(0x8000000000000000ULL), 1);
    EXPECT_EQ(Popcount64(0xAAAAAAAAAAAAAAAAULL), 32);
    EXPECT_EQ(PortablePopcount64(0), 0);
    EXPECT_EQ(PortablePopcount64(~0ULL), 64);
}

TEST(BitOpsTest, CtzEdgeCases) {
    EXPECT_EQ(Ctz64(0), 64);
    EXPECT_EQ(Ctz64(1), 0);
    EXPECT_EQ(Ctz64(~0ULL), 0);
    EXPECT_EQ(Ctz64(0x8000000000000000ULL), 63);
    EXPECT_EQ(Ctz64(0x100000000ULL), 32);
    EXPECT_EQ(Ctz64(0x80000000ULL), 31);
    EXPECT_EQ(Ctz64(0b101000ULL), 3);
}

TEST(BitOpsTest, ParityEdgeCases) {
    EXPECT_EQ(Parity64(0), 0);
    EXPECT_EQ(Parity64(1), 1);
    EXPECT_EQ(Parity64(0b11ULL), 0);
    EXPECT_EQ(Parity64(0b111ULL), 1);
    EXPECT_EQ(Parity64(~0ULL), 0);
    EXPECT_EQ(Parity64(0x8000000000000001ULL), 0);
    EXPECT_EQ(Parity64(0x8000000000000000ULL), 1);
}

TEST(BitOpsTest, SingleBitsMatchReference) {
    for (int bit = 0; bit < 64; ++bit) {
        const uint64_t x = 1ULL << bit;
        EXPECT_EQ(Popcount64(x), 1) << "bit=" << bit;
        EXPECT_EQ(PortablePopcount64(x), 1) << "bit=" << bit;
        EXPECT_EQ(Ctz64(x), bit) << "bit=" << bit;
        EXPECT_EQ(Parity64(x), 1) << "bit=" << bit;
    }
}

TEST(BitOpsTest, RandomValuesMatchReference) {
    std::mt19937_64 rng(20261027);
    for (int i = 0; i < 10000; ++i) {
        // 混合稠密与稀疏的随机数，覆盖高位 / 低位各种分布
        uint64_t x = rng();
        if (i % 3 == 1) {
            x &= rng();
        } else if (i % 3 == 2) {
            x &= rng() & rng() & rng();
        }
        const int expected_popcount = ReferencePopcount(x);
        EXPECT_EQ(Popcount64(x), expected_popcount) << "x=" << x;
        EXPECT_EQ(PortablePopcount64(x), expected_popcount) << "x=" << x;
        EXPECT_EQ(Ctz64(x), ReferenceCtz(x)) << "x=" << x;
        EXPECT_EQ(Parity64(x), expected_popcount & 1) << "x=" << x;
    }
}
