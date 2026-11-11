// 三比特数值实例化搜索（synthesis/instantiate）的 gtest：找到的线路等于目标（相差全局相位）、
// CZ 数不超过上限、只用允许的比特对、同一输入两次运行逐位相同。
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/synthesis/instantiate.hpp"
#include "quspect_native/synthesis/three_qubit.hpp"

namespace {

using namespace quspect::native::synthesis;

M8 Identity() {
    M8 u{};
    for (int i = 0; i < 8; ++i) {
        u[static_cast<std::size_t>(i * 9)] = Cx{1.0, 0.0};
    }
    return u;
}

// 由经典可逆映射 f（基态 x ↦ f(x)，比特 0 为最高位）得到的置换矩阵。
M8 Permutation(int (*f)(int)) {
    M8 u{};
    for (int x = 0; x < 8; ++x) {
        u[static_cast<std::size_t>(f(x) * 8 + x)] = Cx{1.0, 0.0};
    }
    return u;
}

int Toffoli(int x) {
    return ((x & 6) == 6) ? (x ^ 1) : x;  // 控制 0、1，目标 2
}

int CzChainValue(int x) {
    return x;
}

double Distance(const M8& u, const M8& v) {
    double re = 0.0, im = 0.0;
    for (int e = 0; e < 64; ++e) {
        re += u[e].re * v[e].re + u[e].im * v[e].im;
        im += u[e].re * v[e].im - u[e].im * v[e].re;
    }
    const double mag = std::hypot(re, im);
    const double wr = re / mag, wi = im / mag;  // v ≈ w·u
    double m = 0.0;
    for (int e = 0; e < 64; ++e) {
        const double dr = v[e].re - (wr * u[e].re - wi * u[e].im);
        const double di = v[e].im - (wr * u[e].im + wi * u[e].re);
        m = std::max(m, std::max(std::fabs(dr), std::fabs(di)));
    }
    return m;
}

TEST(Instantiate, IdentityNeedsNoCz) {
    InstantiateOptions opt;
    opt.max_cz = 3;
    const auto r = InstantiateSearch({Identity()}, {{0, 1}, {1, 2}}, opt);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.czs.size(), 0u);
    EXPECT_LE(r.residual, 1e-12);
}

TEST(Instantiate, TwoCzChain) {
    // CZ(0,1)·CZ(1,2) 是对角阵：diag((−1)^{x0 x1 + x1 x2})
    M8 u{};
    for (int x = 0; x < 8; ++x) {
        const int b0 = (x >> 2) & 1, b1 = (x >> 1) & 1, b2 = x & 1;
        u[static_cast<std::size_t>(x * 9)] = Cx{((b0 & b1) ^ (b1 & b2)) ? -1.0 : 1.0, 0.0};
    }
    (void)CzChainValue;
    InstantiateOptions opt;
    opt.max_cz = 5;
    const auto r = InstantiateSearch({u}, {{0, 1}, {1, 2}}, opt);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.czs.size(), 2u);
    EXPECT_LE(Distance(u, InstantiateUnitary(r.czs, r.params)), 1e-12);
}

TEST(Instantiate, ToffoliOnTriangleAndDeterminism) {
    const M8 u = Permutation(Toffoli);
    InstantiateOptions opt;
    opt.max_cz = 8;
    opt.heuristic_factor = 20.0;
    opt.budget = 40000;
    const std::vector<std::array<int, 2>> edges{{0, 1}, {0, 2}, {1, 2}};
    const auto r = InstantiateSearch({u}, edges, opt);
    ASSERT_TRUE(r.found);
    EXPECT_LE(r.czs.size(), 8u);
    EXPECT_GE(r.czs.size(), 6u);  // Toffoli 至少要 6 个两比特门
    EXPECT_LE(Distance(u, InstantiateUnitary(r.czs, r.params)), 1e-12);
    const auto again = InstantiateSearch({u}, edges, opt);
    ASSERT_EQ(again.czs, r.czs);
    ASSERT_EQ(again.params.size(), r.params.size());
    EXPECT_EQ(std::memcmp(again.params.data(), r.params.data(), r.params.size() * sizeof(double)), 0);
    EXPECT_EQ(again.iterations, r.iterations);
}

TEST(Instantiate, RespectsEdgesAndBound) {
    const M8 u = Permutation(Toffoli);
    InstantiateOptions opt;
    opt.max_cz = 3;  // Toffoli 不可能
    opt.budget = 3000;
    const auto r = InstantiateSearch({u}, {{0, 1}, {1, 2}}, opt);
    EXPECT_FALSE(r.found);
    EXPECT_LE(r.iterations, opt.budget + opt.max_iters + opt.polish_iters);
    opt.max_cz = 12;
    opt.budget = 40000;
    opt.heuristic_factor = 20.0;
    const auto s = InstantiateSearch({u}, {{0, 1}, {1, 2}}, opt);
    ASSERT_TRUE(s.found);
    for (const auto& e : s.czs) {
        EXPECT_TRUE((e[0] == 0 && e[1] == 1) || (e[0] == 1 && e[1] == 2));
    }
}

TEST(Instantiate, RejectsBadEdges) {
    InstantiateOptions opt;
    EXPECT_THROW(InstantiateSearch({Identity()}, {{0, 3}}, opt), std::invalid_argument);
    EXPECT_THROW(InstantiateUnitary({{0, 1}}, std::vector<double>(9, 0.0)), std::invalid_argument);
}

}  // namespace
