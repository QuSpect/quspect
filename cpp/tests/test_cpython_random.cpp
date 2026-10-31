#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/cpython_random.hpp"

namespace {

using quspect::native::transpiler::CPythonRandom;

// 期望值由 CPython 3.13 生成：
//   r = random.Random(seed)
//   bits32 = [r.getrandbits(32) for _ in range(4)]
//   bits64 = [r.getrandbits(64) for _ in range(2)]                    （新的 Random(seed)）
//   mixed  = [r.getrandbits(k) for k in (1, 5, 17, 31, 33, 40, 63)]   （新的 Random(seed)）
//   below  = [r._randbelow(n) for n in (2, 3, 5, 7, 100, 1000, 2**31 + 1, 2**40 + 3)]
//   checksum: x = (x * 31 + r.getrandbits(32)) % 2**64，共 2000 次（跨越多次 624 字重算）
//   state0/state1/state623 = random.Random(seed).getstate()[1][0, 1, 623]
struct Expected {
    std::uint64_t seed;
    std::uint32_t bits32[4];
    std::uint64_t bits64[2];
    std::uint64_t mixed[7];
    std::uint64_t below[8];
    std::uint64_t checksum;
    std::uint32_t state0;
    std::uint32_t state1;
    std::uint32_t state623;
};

constexpr int kMixedBits[7] = {1, 5, 17, 31, 33, 40, 63};
constexpr std::uint64_t kBelow[8] = {2, 3, 5, 7, 100, 1000, (1ULL << 31) + 1, (1ULL << 40) + 3};

const Expected kExpected[] = {
    {0ULL,
     {3626764237U, 1654615998U, 3255389356U, 3823568514U},
     {7106521602475165645ULL, 16422101724900707500ULL},
     {1ULL, 12ULL, 99346ULL, 1911784257ULL, 1806341205ULL, 1061968961082ULL, 4481891911369680482ULL},
     {1ULL, 1ULL, 0ULL, 2ULL, 65ULL, 497ULL, 1739178872ULL, 669284121952ULL},
     7953043930204808742ULL,
     2147483648U,
     766982754U,
     2902720905U},
    {1ULL,
     {577090037U, 2444712010U, 3639700191U, 3445702192U},
     {10499958131665514997ULL, 14799178230035213023ULL},
     {0ULL, 18ULL, 111074ULL, 1722851096ULL, 3280387012ULL, 129944532028ULL, 7018639715332314491ULL},
     {0ULL, 2ULL, 0ULL, 2ULL, 15ULL, 507ULL, 1930549411ULL, 207060179245ULL},
     12340158826204599674ULL,
     2147483648U,
     163610392U,
     1781494222U},
    {42ULL,
     {2746317213U, 478163327U, 107420369U, 3184935163U},
     {2053695854357871005ULL, 13679192365072849617ULL},
     {1ULL, 3ULL, 3278ULL, 1592467581ULL, 1181241943ULL, 151282538206ULL, 945351111358374057ULL},
     {0ULL, 0ULL, 2ULL, 1ULL, 28ULL, 142ULL, 440213415ULL, 70531617177ULL},
     14092032207236291458ULL,
     2147483648U,
     3564348608U,
     3831079317U},
    {4294967301ULL,
     {675479763U, 2085189291U, 1213270837U, 3822202474U},
     {8955819811489906899ULL, 16416234625733561141ULL},
     {0ULL, 15ULL, 37026ULL, 1911101237ULL, 6891077597ULL, 386609434369ULL, 1580140662990693877ULL},
     {0ULL, 1ULL, 2ULL, 4ULL, 1ULL, 363ULL, 735810335ULL, 536073183864ULL},
     7703917779143746533ULL,
     2147483648U,
     3029627165U,
     656747425U},
    {18446744073709551615ULL,
     {93740670U, 1068495656U, 1452108352U, 2657516307U},
     {4589153898531806846ULL, 11413945628603804224ULL},
     {0ULL, 7ULL, 44314ULL, 1328758153ULL, 910393425ULL, 105721161155ULL, 8308523266359230708ULL},
     {0ULL, 0ULL, 2ULL, 4ULL, 27ULL, 467ULL, 408097150ULL, 119153246635ULL},
     267399427469839883ULL,
     2147483648U,
     2932382577U,
     861141058U},
    {123456789ULL,
     {2754794679U, 1899526012U, 2328685183U, 3049235403U},
     {8158402102196098231ULL, 13096366336019065471ULL},
     {1ULL, 14ULL, 71065ULL, 1524617701ULL, 8560621697ULL, 1008644087602ULL, 5552776204723800935ULL},
     {1ULL, 2ULL, 4ULL, 3ULL, 38ULL, 705ULL, 2004409099ULL, 968083923990ULL},
     639303860592450981ULL,
     2147483648U,
     2909095743U,
     2402112229U},
};

TEST(CPythonRandomTest, SeedStateMatchesCPython) {
    for (const auto& e : kExpected) {
        const auto state = CPythonRandom(e.seed).State();
        ASSERT_EQ(state.size(), 625U);
        EXPECT_EQ(state[0], e.state0) << e.seed;
        EXPECT_EQ(state[1], e.state1) << e.seed;
        EXPECT_EQ(state[623], e.state623) << e.seed;
        EXPECT_EQ(state[624], 624U) << e.seed;
    }
}

TEST(CPythonRandomTest, GetRandBitsMatchesCPython) {
    for (const auto& e : kExpected) {
        CPythonRandom r32(e.seed);
        for (std::uint32_t v : e.bits32) {
            EXPECT_EQ(r32.GetRandBits(32), v) << e.seed;
        }
        CPythonRandom r64(e.seed);
        for (std::uint64_t v : e.bits64) {
            EXPECT_EQ(r64.GetRandBits(64), v) << e.seed;
        }
        CPythonRandom rmix(e.seed);
        for (int i = 0; i < 7; ++i) {
            EXPECT_EQ(rmix.GetRandBits(kMixedBits[i]), e.mixed[i]) << e.seed << " k=" << kMixedBits[i];
        }
    }
}

TEST(CPythonRandomTest, RandBelowMatchesCPython) {
    for (const auto& e : kExpected) {
        CPythonRandom rng(e.seed);
        for (int i = 0; i < 8; ++i) {
            EXPECT_EQ(rng.RandBelow(kBelow[i]), e.below[i]) << e.seed << " n=" << kBelow[i];
        }
    }
}

TEST(CPythonRandomTest, LongSequenceChecksumMatchesCPython) {
    for (const auto& e : kExpected) {
        CPythonRandom rng(e.seed);
        std::uint64_t x = 0;
        for (int i = 0; i < 2000; ++i) {
            x = x * 31 + rng.GetRandBits(32);
        }
        EXPECT_EQ(x, e.checksum) << e.seed;
    }
}

TEST(CPythonRandomTest, ChoiceSequenceMatchesCPython) {
    // [random.Random(7).choice(range(6)) for _ in range(12)]
    const std::uint64_t expected[12] = {2, 1, 3, 5, 0, 0, 4, 0, 2, 4, 0, 4};
    CPythonRandom rng(7);
    for (std::uint64_t v : expected) {
        EXPECT_EQ(rng.RandBelow(6), v);
    }
}

TEST(CPythonRandomTest, StateRoundTripContinuesSequence) {
    CPythonRandom a(42);
    for (int i = 0; i < 700; ++i) {
        a.GenRandUint32();
    }
    CPythonRandom b = CPythonRandom::FromState(a.State());
    EXPECT_FALSE(b.used());
    for (int i = 0; i < 1000; ++i) {
        ASSERT_EQ(a.GenRandUint32(), b.GenRandUint32());
    }
    EXPECT_TRUE(b.used());
    EXPECT_EQ(a.State(), b.State());
}

TEST(CPythonRandomTest, RejectsInvalidArguments) {
    CPythonRandom rng(1);
    EXPECT_THROW(rng.RandBelow(0), std::invalid_argument);
    EXPECT_THROW(rng.GetRandBits(65), std::invalid_argument);
    EXPECT_THROW(rng.GetRandBits(-1), std::invalid_argument);
    EXPECT_EQ(rng.GetRandBits(0), 0U);
    EXPECT_FALSE(rng.used());
    EXPECT_THROW(CPythonRandom::FromState(std::vector<std::uint32_t>(10, 0)), std::invalid_argument);
    std::vector<std::uint32_t> bad(625, 0);
    bad[624] = 625;
    EXPECT_THROW(CPythonRandom::FromState(bad), std::invalid_argument);
}

}  // namespace
