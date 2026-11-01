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
