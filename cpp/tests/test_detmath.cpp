// 确定性数值层（detmath）的 gtest：正确舍入的标量函数、复数运算与小矩阵分解。
//
// 分解的精度对随机、病态（条件数 1e12）、简并（重特征值 / 重奇异值）与秩亏输入检查：重构误差与
// 正交性误差都在 n·ε 的若干倍以内（与 LAPACK 的 dsyevd / zgesdd 同一量级或更好）。跨平台逐位一致性
// 由 tools/det_fingerprint.py 在各平台上比对。
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/detmath/scalar.hpp"

// vendored CORE-MATH（C 链接）
extern "C" double cr_atan2(double, double);

namespace {

namespace dm = quspect::native::detmath;
using dm::Cx;
using C = std::complex<double>;

constexpr double kPi = 3.141592653589793;
constexpr double kEps = 2.220446049250313e-16;

std::uint64_t Bits(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

// 两个有限 double 之间相差的 ulp 数。
std::int64_t UlpDistance(double a, double b) {
    auto key = [](double x) {
        const auto u = static_cast<std::int64_t>(Bits(x));
        return u < 0 ? std::numeric_limits<std::int64_t>::min() - u : u;
    };
    const std::int64_t d = key(a) - key(b);
    return d < 0 ? -d : d;
}

std::vector<Cx> RandomComplex(std::mt19937_64& rng, int n) {
    std::normal_distribution<double> normal;
    std::vector<Cx> a(static_cast<std::size_t>(n * n));
    for (Cx& z : a) {
        z = Cx{normal(rng), normal(rng)};
    }
    return a;
}

std::vector<Cx> RandomUnitary(std::mt19937_64& rng, int n) {
    const std::vector<Cx> g = RandomComplex(rng, n);
    std::vector<Cx> q(g.size());
    std::vector<Cx> r(g.size());
    dm::Qr(n, g.data(), q.data(), r.data());
    return q;
}

std::vector<double> RandomOrthogonal(std::mt19937_64& rng, int n) {
    std::normal_distribution<double> normal;
    std::vector<Cx> g(static_cast<std::size_t>(n * n));
    for (Cx& z : g) {
        z = Cx{normal(rng), 0.0};
    }
    std::vector<Cx> q(g.size());
    std::vector<Cx> r(g.size());
    dm::Qr(n, g.data(), q.data(), r.data());
    std::vector<double> out(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        out[i] = q[i].re;
    }
    return out;
}

// max |a^† a − I|
double UnitarityError(int n, const Cx* a) {
    std::vector<Cx> g(static_cast<std::size_t>(n * n));
    dm::MatMulAdjA(n, a, a, g.data());
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            err = std::max(err, std::abs(C(g[i * n + j].re, g[i * n + j].im) - C(i == j ? 1.0 : 0.0, 0.0)));
        }
    }
    return err;
}

double OrthogonalityError(int n, const double* v) {
    double err = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                s += v[k * n + i] * v[k * n + j];
            }
            err = std::max(err, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    }
    return err;
}

// ---- 标量函数 --------------------------------------------------------------------

TEST(DetmathScalar, SpecialValues) {
    EXPECT_EQ(Bits(dm::Sin(0.0)), Bits(0.0));
    EXPECT_EQ(Bits(dm::Sin(-0.0)), Bits(-0.0));
    EXPECT_EQ(dm::Cos(0.0), 1.0);
    EXPECT_EQ(dm::Exp(0.0), 1.0);
    EXPECT_EQ(dm::Log(1.0), 0.0);
    EXPECT_EQ(dm::Hypot(3.0, 4.0), 5.0);
    EXPECT_EQ(dm::Atan2(0.0, -1.0), kPi);
    EXPECT_EQ(dm::Atan2(-0.0, -1.0), -kPi);
    EXPECT_EQ(dm::Atan2(1.0, 0.0), kPi / 2);
    EXPECT_EQ(dm::Acos(-1.0), kPi);
    EXPECT_EQ(dm::Asin(1.0), kPi / 2);
    EXPECT_EQ(dm::Atan(1.0), kPi / 4);
    EXPECT_TRUE(std::isnan(dm::Sin(std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(dm::Log(-1.0)));
    EXPECT_EQ(dm::Exp(-1000.0), 0.0);
    // 已知的正确舍入值（与 MPFR 一致）
    EXPECT_EQ(dm::Sin(1.0), 0x1.aed548f090ceep-1);
    EXPECT_EQ(dm::Cos(1.0), 0x1.14a280fb5068cp-1);
    EXPECT_EQ(dm::Exp(1.0), 0x1.5bf0a8b145769p+1);
    EXPECT_EQ(dm::Tan(1.0), 0x1.8eb245cbee3a6p+0);
}

TEST(DetmathScalar, AgreesWithLibmToOneUlp) {
    // 正确舍入的结果与任何误差 < 1 ulp 的 libm 至多差 1 ulp
    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> wide(-50.0, 50.0);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    for (int i = 0; i < 20000; ++i) {
        const double x = wide(rng);
        const double y = wide(rng);
        const double u = unit(rng);
        EXPECT_LE(UlpDistance(dm::Sin(x), std::sin(x)), 1) << x;
        EXPECT_LE(UlpDistance(dm::Cos(x), std::cos(x)), 1) << x;
        // 部分 libm（如 macOS）的 tan 误差超过 1 ulp
        EXPECT_LE(UlpDistance(dm::Tan(x), std::tan(x)), 4) << x;
        EXPECT_LE(UlpDistance(dm::Exp(x), std::exp(x)), 1) << x;
        EXPECT_LE(UlpDistance(dm::Log(std::fabs(x)), std::log(std::fabs(x))), 1) << x;
        EXPECT_LE(UlpDistance(dm::Atan2(y, x), std::atan2(y, x)), 1) << y << " " << x;
        EXPECT_LE(UlpDistance(dm::Hypot(y, x), std::hypot(y, x)), 1) << y << " " << x;
        EXPECT_LE(UlpDistance(dm::Atan(x), std::atan(x)), 1) << x;
        EXPECT_LE(UlpDistance(dm::Asin(u), std::asin(u)), 1) << u;
        EXPECT_LE(UlpDistance(dm::Acos(u), std::acos(u)), 1) << u;
    }
}

// std::fma 必须正确舍入（CORE-MATH 与 detmath 都依赖它；没有 FMA 指令的平台上由 C 运行库用软件
// 实现，例如 Windows UCRT）。用例的正确结果由有理数精确计算后舍入得到，其中大多数与先舍入乘积再
// 相加的结果不同（包括抵消、接近舍入中点与次正规结果）。输入经 volatile 读取，防止编译期求值。
TEST(DetmathScalar, FmaIsCorrectlyRounded) {
    struct Case {
        double a, b, c, expected;
    };
    static const Case kCases[] = {
    {0x1.0000000000001p+0, 0x1.0000000000001p+0, -0x1.0000000000002p+0, 0x1.0000000000000p-104},
    {0x1.0000000400000p+0, 0x1.fffffff800000p-1, -0x1.0000000000000p+0, -0x1.0000000000000p-60},
    {0x1.6f31ba6c42e88p+17, 0x1.09ad79962b56ap+14, -0x1.7d134cfaf2d5fp+31, -0x1.59d064b9d2760p-22},
    {0x1.4b8909b1646a8p+23, 0x1.32cc91c53efa9p+30, -0x1.8d52bb91d3e4fp+53, -0x1.5e489dc831800p-7},
    {0x1.d9f21f378763ap-23, 0x1.0ec4c62683ebap+25, -0x1.f549a925099f7p+2, -0x1.21e764901bb80p-55},
    {0x1.4e50fbb7192dcp-16, 0x1.0a125ac631922p-18, -0x1.5b781dc93ee6bp-34, 0x1.56c89780fd380p-90},
    {0x1.dfe06a59a3566p+19, 0x1.158b3ed7a9377p+18, -0x1.04216bdcac6b2p+38, -0x1.02494c05cdcb0p-18},
    {0x1.6d1a0d755a241p-22, 0x1.658f9c202b9e6p+24, -0x1.fdf224fbecec1p+2, 0x1.53bdf7ecafd98p-52},
    {0x1.a57d525c2c155p-21, 0x1.cf012c47fdc2ep-12, -0x1.7d27cb2d0be70p-32, 0x1.b145d244ec946p-85},
    {0x1.4f0fd2ea0dbfep+13, 0x1.7047c7f9d454ap-19, -0x1.e204b2503bed9p-6, 0x1.74e4b0a4435b0p-60},
    {0x1.f298d5eae64c3p-6, 0x1.e57849a2121c0p-8, -0x1.d8c2ea29c4b37p-13, 0x1.973bf91d76a80p-67},
    {0x1.6fe6293953712p-27, 0x1.fedd3f3cee337p+21, -0x1.6f153d597f90ap-5, -0x1.73e634a501a91p-57},
    {0x1.64db1d608a74cp+15, 0x1.01c09cf32b48ap-11, -0x1.674c779c71a0cp+4, 0x1.bd5fd2e473df0p-49},
    {0x1.389b2cc16d9b4p-10, 0x1.8de7e5ae326c4p-1, -0x1.e5e3d0e4df97ap-11, -0x1.725b01e5268c0p-65},
    {0x1.c4bf65fd9c0bep+29, 0x1.118f8d7b200acp+16, -0x1.e3cdfce497e06p+45, 0x1.417dffc697fa8p-7},
    {0x1.0de038d4b5d82p+19, 0x1.20d4649193accp-6, -0x1.307c27a36a9afp+13, -0x1.fa32c422010d0p-40},
    {0x1.2de23d223e2ecp+15, 0x1.8af237fbe05f0p-19, -0x1.d1bbd3f91e9d0p-4, 0x1.04ca4d0ed6500p-58},
    {0x1.e4fcd62755ee9p+30, 0x1.317ca2c087c38p+2, -0x1.215eacf8ab63fp+33, 0x1.781b9b376cf7cp-19},
    {0x1.c0afc24a75384p-2, 0x1.52fc0ef89e064p+16, -0x1.2910eae375f5ap+15, 0x1.04f0a540a0fc8p-37},
    {0x1.1c220fc990d5cp+29, 0x1.f7aee1c4606b1p-17, -0x1.1784835fd5c61p+13, 0x1.308c8f9a5324ep-39},
    {0x1.0b3f892c11749p-21, 0x1.028b619d3b82ep+29, -0x1.0de789a51b962p+8, -0x1.53988866658e2p-44},
    {0x1.eff63e01da4bcp-11, 0x1.eb23ae7109b18p+20, -0x1.dbc134c632472p+10, 0x1.86c4dca8a8b40p-44},
    {0x1.94ea75811493cp-11, 0x1.febe3b9738dbap-20, -0x1.93ebfd2dcb0d3p-30, -0x1.e7410f09af9a0p-85},
    {0x1.fae4781cd4c70p+4, 0x1.e2aa3e937b8d5p-28, -0x1.ddd9a0e2b357cp-23, 0x1.17e3f5688c980p-79},
    {0x1.04082647a984fp-14, 0x1.2ef9b3bf3f136p-4, -0x1.33bf3fa9ba991p-18, -0x1.333398900ac00p-79},
    {0x1.dfc547fea3badp-28, 0x1.b3d6d1f7b0eabp-9, -0x1.986768e010e11p-36, -0x1.30111013d6b88p-92},
    {0x1.98f58a1b13046p+30, 0x1.65f9049c84b16p+26, -0x1.1dee1be47ace9p+57, 0x1.10e39ec3aa010p+2},
    {0x1.26de8371e7651p-21, 0x1.2d22c611182e2p-5, -0x1.5adbae3e2c7d6p-26, -0x1.13b52a56a7e00p-86},
    {0x1.e20c9ddff765dp+9, 0x1.c743d8b8203c0p+3, -0x1.aca216fddea65p+13, 0x1.4be281835dcc0p-40},
    {0x1.8883a23304286p-10, 0x1.37f3c9fd27594p-12, -0x1.de4db4b271404p-22, -0x1.87fd9b8c99100p-79},
    {0x1.708452aace488p-7, 0x1.bbab4b8f714edp+4, -0x1.3f55c81b0626bp-2, -0x1.7ffdf79bb9c30p-56},
    {0x1.3fbd64e9a7836p-9, 0x1.8a5517688ea41p+0, -0x1.ec83c46eb88ebp-9, 0x1.4a1d1e7c9e1b6p-61},
    {0x1.52787979a272fp+2, 0x1.0abfd063077b1p+20, -0x1.60aec99fd5b96p+22, 0x1.f08fbd937e0fep-31},
    {0x1.7efbab51a5b4cp+18, 0x1.eac6e2a2e3b2ep+4, -0x1.6f1b9fcf52bbcp+23, -0x1.dbc5782911458p-30},
    {0x1.c0bab1b750199p-14, 0x1.a795416f15febp+14, -0x1.733d0d80b9d62p+1, -0x1.8187f71ed831ap-53},
    {0x1.eb171b5052d26p+11, 0x1.8933e657be520p+23, -0x1.79250433ea513p+35, -0x1.1556c3ad57500p-20},
    {0x1.a91b7a4c0881cp-10, 0x1.81e92f9d230aep-1, -0x1.406ac5ffd1becp-10, 0x1.366007bace184p-62},
    {0x1.789da43055aa6p+21, 0x1.08ec51d0b2ce0p-13, -0x1.85be3b021cfc1p+8, -0x1.a2d0af2689b00p-46},
    {0x1.e8017e17c0c84p-4, 0x1.066ad6ac45a17p+23, -0x1.f43d30e42a478p+19, -0x1.859c72323c120p-36},
    {0x1.885246b596f25p-7, 0x1.ef4d65500c5c5p+3, -0x1.7b86d95dc3050p-3, 0x1.69dedd6801e40p-62},
    {0x1.7f9eddb31219dp+22, 0x1.0b04e40cfe8bdp+24, -0x1.9022057231043p+46, 0x1.f18db1ebc63a4p-8},
    {0x1.3c8bee851703cp+11, 0x1.9874a9c3ff974p-20, -0x1.f90f4576ec137p-9, -0x1.477c0562ac8d0p-61},
    {0x1.00000000d7721p+0, 0x1.000000009af71p+0, -0x1.7269200000000p-32, 0x1.0000000000000p+0},
    {0x1.000000008a9a0p+0, 0x1.000000000e248p+0, -0x1.317d000000000p-33, 0x1.0000000000000p+0},
    {0x1.0000000064259p+0, 0x1.0000000051ee9p+0, -0x1.6c28400000000p-33, 0x1.0000000000000p+0},
    {0x1.00000000e1d3cp+0, 0x1.000000005c8a6p+0, -0x1.3e5e200000000p-32, 0x1.0000000000000p+0},
    {0x1.000000007025dp+0, 0x1.000000005c6eep+0, -0x1.9929600000000p-33, 0x1.0000000000000p+0},
    {0x1.00000000156efp+0, 0x1.00000000f173bp+0, -0x1.06e2a00000000p-32, 0x1.0000000000000p+0},
    {0x1.0000000073d62p+0, 0x1.0000000054dcap+0, -0x1.9165800000000p-33, 0x1.0000000000000p+0},
    {0x1.000000001bb8bp+0, 0x1.00000000445dbp+0, -0x1.8059800000000p-34, 0x1.0000000000000p+0},
    {0x1.0000000038be9p+0, 0x1.00000000a27bdp+0, -0x1.b674c00000000p-33, 0x1.0000000000000p+0},
    {0x1.000000005cf3dp+0, 0x1.00000000f71f8p+0, -0x1.5413500000000p-32, 0x1.0000000000000p+0},
    {0x1.0000000000000p-537, 0x1.0000000000000p-537, -0x0.0000000000003p-1022, -0x0.0000000000002p-1022},
    {0x1.8000000000000p-999, 0x1.0000000000000p-60, 0x0.0000000000001p-1022, 0x0.000000000c001p-1022},
    };
    for (const Case& k : kCases) {
        volatile double a = k.a;
        volatile double b = k.b;
        volatile double c = k.c;
        const double got = std::fma(a, b, c);
        EXPECT_EQ(Bits(got), Bits(k.expected)) << std::hexfloat << k.a << " * " << k.b << " + " << k.c;
    }
}

// CORE-MATH 的 atan2 在这个输入上走到"意外的最坏情况"分支（上游打印后 exit(1)，本地改为继续舍入）。
TEST(DetmathScalar, Atan2HardCaseDoesNotExit) {
    volatile double y = 0x1.7fffffffffffdp-53;
    volatile double x = 0x1.ffffffffffffap-3;
    EXPECT_EQ(Bits(dm::Atan2(y, x)), Bits(0x1.8000000000002p-51));
}

// CORE-MATH 的 atan2 在该提交对下面的输入走到"意外最坏情况"分支（上游会 exit(1)；本地修改为
// 继续舍入 192 位近似值，见 third_party/core-math/README.md）。精确值比舍入中点只高约
// 2^-152.6 ulp；期望值是 mpmath 2000 位的正确舍入。直接调 cr_atan2（dm::Atan2 对 |y/x| ≤ 2^-30
// 走级数，不进 CORE-MATH），同时覆盖 ±1 ulp 的邻域与按 2 的幂缩放的同一比值。
TEST(DetmathScalar, CoreMathAtan2WorstCaseIsCorrectlyRounded) {
    struct Case {
        double y, x, expected;
    };
    const Case cases[] = {
        {0x1.7fffffffffffcp-53, 0x1.ffffffffffff9p-3, 0x1.8000000000001p-51},
        {0x1.7fffffffffffcp-53, 0x1.ffffffffffffap-3, 0x1.8000000000000p-51},
        {0x1.7fffffffffffcp-53, 0x1.ffffffffffffbp-3, 0x1.8000000000000p-51},
        {0x1.7fffffffffffdp-53, 0x1.ffffffffffff9p-3, 0x1.8000000000002p-51},
        {0x1.7fffffffffffdp-53, 0x1.ffffffffffffap-3, 0x1.8000000000002p-51},
        {0x1.7fffffffffffdp-53, 0x1.ffffffffffffbp-3, 0x1.8000000000001p-51},
        {0x1.7fffffffffffep-53, 0x1.ffffffffffff9p-3, 0x1.8000000000003p-51},
        {0x1.7fffffffffffep-53, 0x1.ffffffffffffap-3, 0x1.8000000000003p-51},
        {0x1.7fffffffffffep-53, 0x1.ffffffffffffbp-3, 0x1.8000000000002p-51},
        {0x1.7fffffffffffdp-953, 0x1.ffffffffffffap-903, 0x1.8000000000002p-51},
        {0x1.7fffffffffffdp+247, 0x1.ffffffffffffap+297, 0x1.8000000000002p-51},
        {-0x1.7fffffffffffdp-53, 0x1.ffffffffffffap-3, -0x1.8000000000002p-51},
    };
    for (const Case& c : cases) {
        volatile double y = c.y;
        volatile double x = c.x;
        EXPECT_EQ(Bits(cr_atan2(y, x)), Bits(c.expected)) << std::hexfloat << c.y << ", " << c.x;
    }
}

TEST(DetmathScalar, PowExactCases) {
    volatile double two = 2.0;
    EXPECT_EQ(dm::Pow(two, 10.0), 1024.0);
    EXPECT_EQ(dm::Pow(two, -3.0), 0.125);
    EXPECT_EQ(dm::Pow(9.0, 0.5), 3.0);
    EXPECT_EQ(dm::Pow(two, 0.0), 1.0);
    EXPECT_EQ(dm::Pow(-two, 3.0), -8.0);
    EXPECT_TRUE(std::isnan(dm::Pow(-two, 0.5)));
    // 3^0.5 的正确舍入值（mpmath 200 位）
    EXPECT_EQ(Bits(dm::Pow(3.0, 0.5)), Bits(0x1.bb67ae8584caap+0));
}

TEST(DetmathComplex, DivisionSqrtAndExp) {
    const Cx q = dm::Div(Cx{1.0, 2.0}, Cx{3.0, -4.0});
    const C ref = C(1, 2) / C(3, -4);
    EXPECT_NEAR(q.re, ref.real(), 1e-16);
    EXPECT_NEAR(q.im, ref.imag(), 1e-16);
    const Cx r = dm::Sqrt(Cx{-4.0, 0.0});
    EXPECT_EQ(r.re, 0.0);
    EXPECT_EQ(r.im, 2.0);
    const Cx rn = dm::Sqrt(Cx{-4.0, -0.0});
    EXPECT_EQ(rn.im, -2.0);
    const Cx i = dm::Sqrt(Cx{0.0, 2.0});
    EXPECT_EQ(i.re, 1.0);
    EXPECT_EQ(i.im, 1.0);
    std::mt19937_64 rng(5);
    std::normal_distribution<double> normal;
    for (int k = 0; k < 1000; ++k) {
        const Cx z{normal(rng), normal(rng)};
        const Cx s = dm::Sqrt(z);
        const C sr = std::sqrt(C(z.re, z.im));
        EXPECT_LT(std::abs(C(s.re, s.im) - sr), 4 * kEps * std::abs(sr));
        const Cx e = dm::Exp(z);
        const C er = std::exp(C(z.re, z.im));
        EXPECT_LT(std::abs(C(e.re, e.im) - er), 4 * kEps * std::abs(er));
    }
    EXPECT_EQ(dm::ExpI(0.0).re, 1.0);
}

// ---- 行列式与乘法 ------------------------------------------------------------------

TEST(DetmathLinalg, DeterminantAndProducts) {
    std::mt19937_64 rng(1);
    for (int n : {1, 2, 3, 4, 8}) {
        for (int t = 0; t < 20; ++t) {
            const std::vector<Cx> u = RandomUnitary(rng, n);
            EXPECT_LT(UnitarityError(n, u.data()), 1e-14);
            EXPECT_NEAR(dm::Abs(dm::Det(n, u.data())), 1.0, 1e-13);
        }
    }
    // 置换矩阵与奇异矩阵
    const double perm[9] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
    EXPECT_EQ(dm::DetReal(3, perm), -1.0);
    const double sing[4] = {1, 2, 2, 4};
    EXPECT_EQ(dm::DetReal(2, sing), 0.0);
    // det(AB) = det A det B
    const std::vector<Cx> a = RandomComplex(rng, 4);
    const std::vector<Cx> b = RandomComplex(rng, 4);
    std::vector<Cx> ab(16);
    dm::MatMul(4, a.data(), b.data(), ab.data());
    const Cx dab = dm::Det(4, ab.data());
    const Cx prod = dm::Mul(dm::Det(4, a.data()), dm::Det(4, b.data()));
    EXPECT_LT(dm::Abs(dm::Sub(dab, prod)), 1e-12 * dm::Abs(prod));
}

// ---- 实对称特征分解 ---------------------------------------------------------------

void CheckEigh(int n, const std::vector<double>& a, double tol) {
    std::vector<double> w(static_cast<std::size_t>(n));
    std::vector<double> v(static_cast<std::size_t>(n * n));
    ASSERT_TRUE(dm::EighSym(n, a.data(), w.data(), v.data()));
    EXPECT_LT(OrthogonalityError(n, v.data()), 8 * n * kEps);
    double norm = 0.0;
    for (double x : a) {
        norm = std::max(norm, std::fabs(x));
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                s += v[i * n + k] * w[k] * v[j * n + k];
            }
            EXPECT_LT(std::fabs(s - a[i * n + j]), tol * std::max(norm, 1e-300));
        }
    }
    for (int i = 1; i < n; ++i) {
        EXPECT_LE(w[i - 1], w[i]);
    }
}

TEST(DetmathLinalg, EighRandomDegenerateAndIllConditioned) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> normal;
    for (int n : {2, 3, 4, 8}) {
        for (int t = 0; t < 50; ++t) {
            // 随机
            std::vector<double> a(static_cast<std::size_t>(n * n));
            for (int i = 0; i < n; ++i) {
                for (int j = i; j < n; ++j) {
                    a[i * n + j] = a[j * n + i] = normal(rng);
                }
            }
            CheckEigh(n, a, 16 * n * kEps);
            // Q diag(λ) Q^T：λ 有重根（简并）或跨 12 个数量级（病态）
            const std::vector<double> q = RandomOrthogonal(rng, n);
            std::vector<double> lam(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                lam[i] = t % 2 == 0 ? static_cast<double>(i / 2) : std::pow(10.0, -12.0 * i / (n - 1));
            }
            std::vector<double> m(static_cast<std::size_t>(n * n), 0.0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    for (int k = 0; k < n; ++k) {
                        m[i * n + j] += q[i * n + k] * lam[k] * q[j * n + k];
                    }
                }
            }
            CheckEigh(n, m, 16 * n * kEps);
        }
    }
    // 单位阵、零矩阵、对角阵
    CheckEigh(4, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, kEps);
    CheckEigh(4, std::vector<double>(16, 0.0), kEps);
    CheckEigh(3, {3, 0, 0, 0, -1, 0, 0, 0, 2}, kEps);
}

TEST(DetmathLinalg, JointDiagonalizationOfCommutingPairs) {
    std::mt19937_64 rng(4);
    std::normal_distribution<double> normal;
    for (int t = 0; t < 400; ++t) {
        const int n = t % 3 == 0 ? 8 : 4;
        const std::vector<double> q = RandomOrthogonal(rng, n);
        // 两组特征值：各自可以有重根，但组合起来可分；t % 4 == 1 时整组完全简并
        std::vector<double> la(static_cast<std::size_t>(n));
        std::vector<double> lb(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const double phase = t % 4 == 1 ? 0.3 : (t % 4 == 2 ? 0.5 * (i % 2) : normal(rng));
            la[i] = std::cos(phase + (t % 4 == 3 ? 0.25 * (i / 2) : 0.0));
            lb[i] = std::sin(phase + (t % 4 == 3 ? 0.25 * (i / 2) : 0.0));
        }
        std::vector<double> a(static_cast<std::size_t>(n * n), 0.0);
        std::vector<double> b(static_cast<std::size_t>(n * n), 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    a[i * n + j] += q[i * n + k] * la[k] * q[j * n + k];
                    b[i * n + j] += q[i * n + k] * lb[k] * q[j * n + k];
                }
            }
        }
        std::vector<double> v(static_cast<std::size_t>(n * n));
        ASSERT_TRUE(dm::JointDiagSym(n, a.data(), b.data(), v.data()));
        EXPECT_LT(OrthogonalityError(n, v.data()), 8 * n * kEps);
        for (const std::vector<double>* m : {&a, &b}) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) {
                        continue;
                    }
                    double s = 0.0;
                    for (int k = 0; k < n; ++k) {
                        for (int l = 0; l < n; ++l) {
                            s += v[k * n + i] * (*m)[k * n + l] * v[l * n + j];
                        }
                    }
                    EXPECT_LT(std::fabs(s), 1e-14) << "trial " << t;
                }
            }
        }
    }
}

// ---- 奇异值分解与 QR -----------------------------------------------------------------

void CheckSvd(int n, const std::vector<Cx>& a, double tol) {
    std::vector<Cx> u(static_cast<std::size_t>(n * n));
    std::vector<Cx> vh(static_cast<std::size_t>(n * n));
    std::vector<double> s(static_cast<std::size_t>(n));
    ASSERT_TRUE(dm::Svd(n, a.data(), u.data(), s.data(), vh.data()));
    EXPECT_LT(UnitarityError(n, u.data()), 16 * n * kEps);
    EXPECT_LT(UnitarityError(n, vh.data()), 16 * n * kEps);
    double norm = 0.0;
    for (const Cx& z : a) {
        norm = std::max(norm, dm::Abs(z));
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += C(u[i * n + k].re, u[i * n + k].im) * s[k] * C(vh[k * n + j].re, vh[k * n + j].im);
            }
            EXPECT_LT(std::abs(acc - C(a[i * n + j].re, a[i * n + j].im)), tol * std::max(norm, 1e-300));
        }
    }
    for (int i = 1; i < n; ++i) {
        EXPECT_GE(s[i - 1], s[i]);
    }
}

TEST(DetmathLinalg, SvdRandomRankDeficientAndDegenerate) {
    std::mt19937_64 rng(6);
    for (int n : {2, 4, 8}) {
        for (int t = 0; t < 50; ++t) {
            CheckSvd(n, RandomComplex(rng, n), 16 * n * kEps);
            // U diag(σ) V^†：秩 1、重奇异值、病态
            const std::vector<Cx> u = RandomUnitary(rng, n);
            const std::vector<Cx> v = RandomUnitary(rng, n);
            for (int kind = 0; kind < 3; ++kind) {
                std::vector<Cx> a(static_cast<std::size_t>(n * n), Cx{0.0, 0.0});
                for (int k = 0; k < n; ++k) {
                    const double sigma =
                        kind == 0 ? (k == 0 ? 2.0 : 0.0) : (kind == 1 ? 1.0 + (k / 2) : std::pow(10.0, -12.0 * k / (n - 1)));
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            a[i * n + j] = dm::Add(a[i * n + j], dm::Scale(dm::Mul(u[i * n + k], dm::Conj(v[j * n + k])), sigma));
                        }
                    }
                }
                CheckSvd(n, a, 16 * n * kEps);
            }
        }
    }
    // 秩 1 的张量积（_factor_local 的输入形状）：最大奇异值 2，其余为 0
    std::vector<Cx> a(16);
    const std::vector<Cx> x = RandomUnitary(rng, 2);
    const std::vector<Cx> y = RandomUnitary(rng, 2);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i * 4 + j] = dm::Mul(x[i], y[j]);
        }
    }
    std::vector<Cx> u(16);
    std::vector<Cx> vh(16);
    double s[4];
    ASSERT_TRUE(dm::Svd(4, a.data(), u.data(), s, vh.data()));
    EXPECT_NEAR(s[0], 2.0, 4 * kEps);
    EXPECT_LT(s[1], 1e-15);
    // 零矩阵
    CheckSvd(3, std::vector<Cx>(9, Cx{0.0, 0.0}), kEps);
}

TEST(DetmathLinalg, QrIsUniqueWithNonNegativeDiagonal) {
    std::mt19937_64 rng(8);
    for (int n : {1, 2, 4, 8}) {
        for (int t = 0; t < 30; ++t) {
            const std::vector<Cx> a = RandomComplex(rng, n);
            std::vector<Cx> q(a.size());
            std::vector<Cx> r(a.size());
            dm::Qr(n, a.data(), q.data(), r.data());
            EXPECT_LT(UnitarityError(n, q.data()), 16 * n * kEps);
            std::vector<Cx> qr(a.size());
            dm::MatMul(n, q.data(), r.data(), qr.data());
            for (std::size_t i = 0; i < a.size(); ++i) {
                EXPECT_LT(dm::Abs(dm::Sub(qr[i], a[i])), 32 * n * kEps);
            }
            for (int i = 0; i < n; ++i) {
                EXPECT_GE(r[i * n + i].re, 0.0);
                EXPECT_EQ(r[i * n + i].im, 0.0);
                for (int j = 0; j < i; ++j) {
                    EXPECT_EQ(r[i * n + j].re, 0.0);
                    EXPECT_EQ(r[i * n + j].im, 0.0);
                }
            }
        }
    }
}

// 同一输入重复计算的结果逐位相同（没有未初始化内存或依赖调用历史的状态）。
TEST(DetmathLinalg, RepeatedCallsAreBitIdentical) {
    std::mt19937_64 rng(10);
    const std::vector<Cx> a = RandomComplex(rng, 4);
    std::vector<Cx> u1(16);
    std::vector<Cx> u2(16);
    std::vector<Cx> v1(16);
    std::vector<Cx> v2(16);
    double s1[4];
    double s2[4];
    ASSERT_TRUE(dm::Svd(4, a.data(), u1.data(), s1, v1.data()));
    ASSERT_TRUE(dm::Svd(4, a.data(), u2.data(), s2, v2.data()));
    EXPECT_EQ(std::memcmp(u1.data(), u2.data(), sizeof(Cx) * 16), 0);
    EXPECT_EQ(std::memcmp(v1.data(), v2.data(), sizeof(Cx) * 16), 0);
    EXPECT_EQ(std::memcmp(s1, s2, sizeof(s1)), 0);
}

}  // namespace
