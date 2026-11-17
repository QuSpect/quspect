// T 门数优化搜索内核（tcount）的 gtest：结果必须与输入差一个 Clifford 相位多项式，
// 并且与线程数无关。
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/tcount.hpp"

namespace {

using quspect::native::transpiler::TCountColumns;
using quspect::native::transpiler::TCountFlipProblem;
using quspect::native::transpiler::TCountFlipWalks;
using quspect::native::transpiler::TCountRmProblem;
using quspect::native::transpiler::TCountRmSearch;
using quspect::native::transpiler::TCountTerm;
using quspect::native::transpiler::TCountToddBatch;
using quspect::native::transpiler::TCountToddProblem;

// f_a - f_b 的多线性形式：三次系数 mod 8、二次系数 mod 4、一次系数 mod 2 全为 0 时两者差一个 Clifford。
bool CliffordEquivalent(const TCountColumns& a, const TCountColumns& b, int k) {
    std::map<int, int> lin;
    std::map<std::pair<int, int>, int> quad;
    std::map<std::array<int, 3>, int> cub;
    auto add = [&](std::uint64_t col, int sign) {
        std::vector<int> bits;
        for (int i = 0; i < k; ++i)
            if ((col >> i) & 1u)
                bits.push_back(i);
        for (std::size_t x = 0; x < bits.size(); ++x) {
            lin[bits[x]] += sign;
            for (std::size_t y = x + 1; y < bits.size(); ++y) {
                quad[{bits[x], bits[y]}] -= 2 * sign;
                for (std::size_t z = y + 1; z < bits.size(); ++z)
                    cub[{bits[x], bits[y], bits[z]}] += 4 * sign;
            }
        }
    };
    for (auto c : a)
        add(c, 1);
    for (auto c : b)
        add(c, -1);
    for (const auto& kv : cub)
        if (((kv.second % 8) + 8) % 8 != 0)
            return false;
    for (const auto& kv : quad)
        if (((kv.second % 4) + 4) % 4 != 0)
            return false;
    for (const auto& kv : lin)
        if (((kv.second % 2) + 2) % 2 != 0)
            return false;
    return true;
}

// CCZ(x0, x1, x2) 的七个奇偶。
TCountColumns Ccz(int a, int b, int c) {
    const std::uint64_t x = std::uint64_t{1} << a, y = std::uint64_t{1} << b, z = std::uint64_t{1} << c;
    return {x, y, z, x ^ y, x ^ z, y ^ z, x ^ y ^ z};
}

TCountColumns Xor(const TCountColumns& a, const TCountColumns& b) {
    std::set<std::uint64_t> s(a.begin(), a.end());
    for (auto c : b) {
        if (s.count(c))
            s.erase(c);
        else
            s.insert(c);
    }
    return TCountColumns(s.begin(), s.end());
}

TEST(TCount, ToddReducesSixToffolis) {
    // 7 个变量上六个 CCZ 的 24 个奇偶：贪心 TODD 降到 17 个（与 Python 参考实现相同）。
    TCountColumns cols;
    for (const auto& t : {Ccz(0, 1, 2), Ccz(2, 3, 4), Ccz(1, 4, 5), Ccz(0, 3, 5), Ccz(1, 2, 6), Ccz(4, 5, 6)})
        cols = Xor(cols, t);
    ASSERT_EQ(cols.size(), 24u);
    const auto out = TCountToddBatch({TCountToddProblem{cols, 7, 0, 0}}, 1);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].size(), 17u);
    EXPECT_TRUE(CliffordEquivalent(cols, out[0], 7));
}

TEST(TCount, ToddIsIndependentOfThreadsAndBatching) {
    TCountColumns cols;
    for (const auto& t : {Ccz(0, 1, 2), Ccz(2, 3, 4), Ccz(1, 4, 5), Ccz(0, 3, 5), Ccz(1, 2, 6), Ccz(4, 5, 6)})
        cols = Xor(cols, t);
    const TCountToddProblem prob{cols, 7, 20, 5};
    const auto one = TCountToddBatch({prob}, 1);
    const auto many = TCountToddBatch({prob}, 4);
    const auto batch = TCountToddBatch({prob, prob, prob}, 3);
    EXPECT_EQ(one[0], many[0]);
    EXPECT_EQ(one[0], batch[0]);
    EXPECT_EQ(one[0], batch[2]);
    EXPECT_LE(one[0].size(), cols.size());
    EXPECT_TRUE(CliffordEquivalent(cols, one[0], 7));
}

TEST(TCount, RmSearchKeepsTheUnitaryUpToClifford) {
    TCountColumns cols;
    for (const auto& t : {Ccz(0, 1, 2), Ccz(2, 3, 4), Ccz(1, 4, 5), Ccz(0, 3, 5), Ccz(1, 2, 6), Ccz(4, 5, 6)})
        cols = Xor(cols, t);
    const auto todd = TCountToddBatch({TCountToddProblem{cols, 7, 0, 0}}, 1);
    const auto out = TCountRmSearch({TCountRmProblem{cols, 7, 5, 1}, TCountRmProblem{cols, 7, 5, 2}}, 200, 2);
    ASSERT_EQ(out.size(), 2u);
    for (const auto& res : out) {
        EXPECT_LE(res.size(), todd[0].size());
        EXPECT_TRUE(CliffordEquivalent(cols, res, 7));
    }
    const auto again = TCountRmSearch({TCountRmProblem{cols, 7, 5, 1}}, 200, 1);
    EXPECT_EQ(out[0], again[0]);
}

TEST(TCount, FlipWalkFindsKaratsubaRankForTwoByTwoProduct) {
    // 两个一次多项式的乘积：(a0 + a1 x)(b0 + b1 x)，4 项的朴素分解可以降到秩 3。
    std::vector<TCountTerm> start = {{1, 1, 1}, {1, 2, 2}, {2, 1, 2}, {2, 2, 4}};
    const auto out = TCountFlipWalks({TCountFlipProblem{start, 20000, 0}}, 50, 1);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].size(), 3u);
    std::set<std::array<int, 3>> tensor;
    for (const auto& t : out[0])
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 3; ++k)
                    if (((t[0] >> i) & 1u) && ((t[1] >> j) & 1u) && ((t[2] >> k) & 1u)) {
                        const std::array<int, 3> e{i, j, k};
                        if (tensor.count(e))
                            tensor.erase(e);
                        else
                            tensor.insert(e);
                    }
    const std::set<std::array<int, 3>> want = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 2}};
    EXPECT_EQ(tensor, want);
}

TEST(TCount, RejectsMoreThan64Variables) {
    EXPECT_THROW(TCountToddBatch({TCountToddProblem{{1, 2, 3}, 65, 0, 0}}, 1), std::invalid_argument);
}

}  // namespace
