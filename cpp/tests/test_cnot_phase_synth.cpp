// 受耦合约束的 CNOT-相位最优综合（cnot_phase_synth）的 gtest。
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/cnot_phase_synth.hpp"

namespace {

using quspect::native::transpiler::CnotPhaseResult;
using quspect::native::transpiler::SynthesizeCnotPhase;

using Edges = std::vector<std::pair<int, int>>;

Edges AllPairs(int n) {
    Edges e;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            if (a != b)
                e.emplace_back(a, b);
    return e;
}

Edges Line(int n) {
    Edges e;
    for (int a = 0; a + 1 < n; ++a) {
        e.emplace_back(a, a + 1);
        e.emplace_back(a + 1, a);
    }
    return e;
}

// 按解重放：检查终态行与必需项都满足
void Check(int n, const CnotPhaseResult& r, const std::vector<std::uint32_t>& rows,
           const std::vector<std::uint32_t>& required, const Edges& edges) {
    ASSERT_TRUE(r.found);
    std::vector<std::uint32_t> cur(n);
    std::vector<std::uint32_t> seen;
    for (int i = 0; i < n; ++i) {
        cur[i] = 1U << i;
        seen.push_back(cur[i]);
    }
    for (const auto& [c, t] : r.moves) {
        bool ok = false;
        for (const auto& e : edges)
            ok = ok || (e.first == c && e.second == t);
        ASSERT_TRUE(ok);
        cur[t] ^= cur[c];
        seen.push_back(cur[t]);
    }
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(cur[i], rows[i]);
    for (std::uint32_t m : required) {
        bool hit = false;
        for (std::uint32_t s : seen)
            hit = hit || s == m;
        EXPECT_TRUE(hit) << m;
    }
}

TEST(CnotPhaseSynth, CczIsSixOnAllToAllAndEightOnALine) {
    const std::vector<std::uint32_t> id{1, 2, 4};
    const std::vector<std::uint32_t> ccz{1, 2, 4, 3, 5, 6, 7};
    const auto full = SynthesizeCnotPhase(3, AllPairs(3), id, ccz, 12, 100000);
    Check(3, full, id, ccz, AllPairs(3));
    EXPECT_EQ(full.moves.size(), 6U);
    const auto line = SynthesizeCnotPhase(3, Line(3), id, ccz, 12, 100000);
    Check(3, line, id, ccz, Line(3));
    EXPECT_EQ(line.moves.size(), 8U);
    EXPECT_FALSE(SynthesizeCnotPhase(3, AllPairs(3), id, ccz, 5, 100000).found);
}

TEST(CnotPhaseSynth, SwapIsThreeAndPermutationOnALine) {
    // SWAP：行 (2, 1)
    const auto swap = SynthesizeCnotPhase(2, Line(2), {2, 1}, {}, 5, 1000);
    Check(2, swap, {2, 1}, {}, Line(2));
    EXPECT_EQ(swap.moves.size(), 3U);
    // 只需一个奇偶 x0^x1、终态恒等：2 个 CX
    const auto zz = SynthesizeCnotPhase(2, Line(2), {1, 2}, {3}, 5, 1000);
    Check(2, zz, {1, 2}, {3}, Line(2));
    EXPECT_EQ(zz.moves.size(), 2U);
    // 线 0-1-2 上的 CX(0, 2)（行 (1, 2, 5)）需要 4 个 CX
    const auto far = SynthesizeCnotPhase(3, Line(3), {1, 2, 5}, {}, 6, 100000);
    Check(3, far, {1, 2, 5}, {}, Line(3));
    EXPECT_EQ(far.moves.size(), 4U);
}

TEST(CnotPhaseSynth, BudgetAndArguments) {
    const std::vector<std::uint32_t> id{1, 2, 4, 8};
    const std::vector<std::uint32_t> all{3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15};
    const auto r = SynthesizeCnotPhase(4, Line(4), id, all, 30, 10);
    EXPECT_FALSE(r.found);
    EXPECT_GT(r.expanded, 10);
    EXPECT_THROW(SynthesizeCnotPhase(7, {}, std::vector<std::uint32_t>(7, 0), {}, 1, 10), std::invalid_argument);
    EXPECT_THROW(SynthesizeCnotPhase(2, {{0, 0}}, {1, 2}, {}, 1, 10), std::invalid_argument);
    // 已在目标：0 个 CX
    const auto zero = SynthesizeCnotPhase(2, Line(2), {1, 2}, {1, 2}, 3, 10);
    EXPECT_TRUE(zero.found);
    EXPECT_TRUE(zero.moves.empty());
}

}  // namespace
