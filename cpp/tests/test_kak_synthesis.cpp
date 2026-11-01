// 两比特 KAK 合成（synthesis/kak_synthesis，确定性数值层）与 numpy 语义辅助代码（numpy_compat、
// 单比特合成）的 gtest。
//
// KAK 合成不依赖外部库，这里直接检查算法：KAK 分解能重建输入、坐标落在 Weyl 室内、合成出的门序列
// 等于输入（相差全局相位）且纠缠门数最少、欧拉合成正确。numpy 语义部分在生产环境中调用 numpy 链接的
// BLAS / LAPACK，这里用测试内的朴素实现（三重循环乘法、Jacobi 特征分解与奇异值分解、部分主元 LU，
// 接口与 Fortran / CBLAS 相同）代替，检查各种逐元素运算顺序的定义与调用约定。
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/synthesis/kak_synthesis.hpp"
#include "quspect_native/synthesis/numpy_compat.hpp"
#include "quspect_native/synthesis/one_qubit_synthesis.hpp"

namespace {

using namespace quspect::native::synthesis;
using I = std::int64_t;
using C = std::complex<double>;

C ToC(const Cx& z) {
    return {z.re, z.im};
}
Cx FromC(const C& z) {
    return {z.real(), z.imag()};
}

// ---- 测试用 BLAS / LAPACK（ILP64，行列约定与 CBLAS / Fortran 相同）----------------

// 行主序下按转置标志取 op(A)(i, k)。
C OpAt(const Cx* a, I lda, int trans, I i, I k) {
    return trans == 111 ? ToC(a[i * lda + k]) : ToC(a[k * lda + i]);
}

void TestZgemm(int /*order*/, int ta, int tb, I m, I n, I k, const void* alpha, const void* a, I lda, const void* b,
               I ldb, const void* beta, void* c, I ldc) {
    const C al = ToC(*static_cast<const Cx*>(alpha));
    const C be = ToC(*static_cast<const Cx*>(beta));
    const auto* pa = static_cast<const Cx*>(a);
    const auto* pb = static_cast<const Cx*>(b);
    auto* pc = static_cast<Cx*>(c);
    for (I i = 0; i < m; ++i) {
        for (I j = 0; j < n; ++j) {
            C s = 0;
            for (I l = 0; l < k; ++l) {
                s += OpAt(pa, lda, ta, i, l) * OpAt(pb, ldb, tb, l, j);
            }
            const C prev = be == C(0) ? C(0) : be * ToC(pc[i * ldc + j]);
            pc[i * ldc + j] = FromC(al * s + prev);
        }
    }
}

void TestZsyrk(int /*order*/, int /*uplo*/, int trans, I n, I k, const void* alpha, const void* a, I lda,
               const void* /*beta*/, void* c, I ldc) {
    // C = alpha · op(A) · op(A)^T（只写上三角）
    const C al = ToC(*static_cast<const Cx*>(alpha));
    const auto* pa = static_cast<const Cx*>(a);
    auto* pc = static_cast<Cx*>(c);
    for (I i = 0; i < n; ++i) {
        for (I j = i; j < n; ++j) {
            C s = 0;
            for (I l = 0; l < k; ++l) {
                s += OpAt(pa, lda, trans, i, l) * OpAt(pa, lda, trans, j, l);
            }
            pc[i * ldc + j] = FromC(al * s);
        }
    }
}

void TestDgemv(int /*order*/, int /*trans*/, I n, I m, double alpha, const double* a, I lda, const double* x, I incx,
               double beta, double* y, I incy) {
    // 行主序 Trans：y = alpha · A^T x，A 为 n×m
    for (I j = 0; j < m; ++j) {
        double s = 0;
        for (I i = 0; i < n; ++i) {
            s += a[i * lda + j] * x[i * incx];
        }
        y[j * incy] = alpha * s + (beta == 0 ? 0.0 : beta * y[j * incy]);
    }
}

void TestZdotc(I n, const void* x, I incx, const void* y, I incy, void* out) {
    const auto* px = static_cast<const Cx*>(x);
    const auto* py = static_cast<const Cx*>(y);
    C s = 0;
    for (I i = 0; i < n; ++i) {
        s += std::conj(ToC(px[i * incx])) * ToC(py[i * incy]);
    }
    *static_cast<Cx*>(out) = FromC(s);
}

template <typename T>
I TestGetrf(I* m, I* /*n*/, T* a, I* lda, I* ipiv, I* info) {
    // 部分主元 LU（列主序）
    *info = 0;
    for (I k = 0; k < *m; ++k) {
        I p = k;
        for (I i = k + 1; i < *m; ++i) {
            if (std::abs(a[i + k * *lda]) > std::abs(a[p + k * *lda])) {
                p = i;
            }
        }
        ipiv[k] = p + 1;
        if (a[p + k * *lda] == T(0)) {
            if (*info == 0) {
                *info = k + 1;
            }
            continue;
        }
        if (p != k) {
            for (I j = 0; j < *m; ++j) {
                std::swap(a[k + j * *lda], a[p + j * *lda]);
            }
        }
        for (I i = k + 1; i < *m; ++i) {
            a[i + k * *lda] /= a[k + k * *lda];
            for (I j = k + 1; j < *m; ++j) {
                a[i + j * *lda] -= a[i + k * *lda] * a[k + j * *lda];
            }
        }
    }
    return 0;
}

I TestZgetrf(I* m, I* n, Cx* a, I* lda, I* ipiv, I* info) {
    return TestGetrf(m, n, reinterpret_cast<C*>(a), lda, ipiv, info);
}

I TestDgetrf(I* m, I* n, double* a, I* lda, I* ipiv, I* info) {
    return TestGetrf(m, n, a, lda, ipiv, info);
}

I TestDsyevd(char* /*jobz*/, char* /*uplo*/, I* n, double* a, I* lda, double* w, double* work, I* lwork, I* iwork,
             I* liwork, I* info) {
    *info = 0;
    if (*lwork == -1 || *liwork == -1) {
        work[0] = 64;
        iwork[0] = 32;
        return 0;
    }
    // 循环 Jacobi（列主序，下三角有效）
    const I nn = *n;
    std::vector<double> m(static_cast<std::size_t>(nn * nn));
    std::vector<double> v(static_cast<std::size_t>(nn * nn), 0.0);
    for (I i = 0; i < nn; ++i) {
        for (I j = 0; j < nn; ++j) {
            m[i * nn + j] = i >= j ? a[i + j * *lda] : a[j + i * *lda];
        }
        v[i * nn + i] = 1.0;
    }
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0;
        for (I p = 0; p < nn; ++p) {
            for (I q = p + 1; q < nn; ++q) {
                off += m[p * nn + q] * m[p * nn + q];
            }
        }
        if (off < 1e-30) {
            break;
        }
        for (I p = 0; p < nn; ++p) {
            for (I q = p + 1; q < nn; ++q) {
                if (std::fabs(m[p * nn + q]) < 1e-300) {
                    continue;
                }
                const double theta = (m[q * nn + q] - m[p * nn + p]) / (2 * m[p * nn + q]);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1);
                const double s = t * c;
                for (I k = 0; k < nn; ++k) {
                    const double mkp = m[k * nn + p];
                    const double mkq = m[k * nn + q];
                    m[k * nn + p] = c * mkp - s * mkq;
                    m[k * nn + q] = s * mkp + c * mkq;
                }
                for (I k = 0; k < nn; ++k) {
                    const double mpk = m[p * nn + k];
                    const double mqk = m[q * nn + k];
                    m[p * nn + k] = c * mpk - s * mqk;
                    m[q * nn + k] = s * mpk + c * mqk;
                }
                for (I k = 0; k < nn; ++k) {
                    const double vkp = v[k * nn + p];
                    const double vkq = v[k * nn + q];
                    v[k * nn + p] = c * vkp - s * vkq;
                    v[k * nn + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<I> order(static_cast<std::size_t>(nn));
    for (I i = 0; i < nn; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](I x, I y) { return m[x * nn + x] < m[y * nn + y]; });
    for (I j = 0; j < nn; ++j) {
        w[j] = m[order[j] * nn + order[j]];
        for (I i = 0; i < nn; ++i) {
            a[i + j * *lda] = v[i * nn + order[j]];
        }
    }
    return 0;
}

I TestZgesdd(char* /*jobz*/, I* m, I* /*n*/, Cx* a, I* lda, double* s, Cx* u, I* ldu, Cx* vt, I* ldvt, Cx* work,
             I* lwork, double* /*rwork*/, I* /*iwork*/, I* info) {
    *info = 0;
    if (*lwork == -1) {
        work[0] = Cx{64.0, 0.0};
        return 0;
    }
    // A = U Σ V^H：对 A^H A 做 Hermite Jacobi 特征分解（实数化为 2n×2n 对称矩阵），按奇异值降序
    const I nn = *m;
    std::vector<C> am(static_cast<std::size_t>(nn * nn));
    for (I i = 0; i < nn; ++i) {
        for (I j = 0; j < nn; ++j) {
            am[i * nn + j] = ToC(a[i + j * *lda]);
        }
    }
    // H = A^H A
    std::vector<C> h(static_cast<std::size_t>(nn * nn), 0.0);
    for (I i = 0; i < nn; ++i) {
        for (I j = 0; j < nn; ++j) {
            for (I k = 0; k < nn; ++k) {
                h[i * nn + j] += std::conj(am[k * nn + i]) * am[k * nn + j];
            }
        }
    }
    // [[Re, -Im], [Im, Re]] 的特征向量 (x, y) 给出复特征向量 x + iy（每个特征值出现两次）
    const I r = 2 * nn;
    std::vector<double> big(static_cast<std::size_t>(r * r));
    for (I i = 0; i < nn; ++i) {
        for (I j = 0; j < nn; ++j) {
            big[i + j * r] = h[i * nn + j].real();
            big[(i + nn) + (j + nn) * r] = h[i * nn + j].real();
            big[(i + nn) + j * r] = h[i * nn + j].imag();
            big[i + (j + nn) * r] = -h[i * nn + j].imag();
        }
    }
    std::vector<double> w(static_cast<std::size_t>(r));
    std::vector<double> wk(64);
    std::vector<I> iw(32);
    I lw = 64;
    I liw = 32;
    I inf = 0;
    char jz = 'V';
    char ul = 'L';
    I rr = r;
    TestDsyevd(&jz, &ul, &rr, big.data(), &rr, w.data(), wk.data(), &lw, iw.data(), &liw, &inf);
    // 从大到小取 nn 个互相正交的复向量
    std::vector<std::vector<C>> vecs;
    std::vector<double> vals;
    for (I j = r - 1; j >= 0 && static_cast<I>(vecs.size()) < nn; --j) {
        std::vector<C> x(static_cast<std::size_t>(nn));
        for (I i = 0; i < nn; ++i) {
            x[i] = C(big[i + j * r], big[(i + nn) + j * r]);
        }
        for (const auto& y : vecs) {
            C d = 0;
            for (I i = 0; i < nn; ++i) {
                d += std::conj(y[i]) * x[i];
            }
            for (I i = 0; i < nn; ++i) {
                x[i] -= d * y[i];
            }
        }
        double norm = 0;
        for (const C& z : x) {
            norm += std::norm(z);
        }
        if (norm < 1e-12) {
            continue;
        }
        norm = std::sqrt(norm);
        for (C& z : x) {
            z /= norm;
        }
        vecs.push_back(x);
        vals.push_back(std::sqrt(std::max(0.0, w[j])));
    }
    if (static_cast<I>(vecs.size()) < nn) {
        *info = 1;
        return 0;
    }
    // U 的列：A v / σ（σ = 0 时用 Gram–Schmidt 补全）
    std::vector<std::vector<C>> ucols;
    for (I j = 0; j < nn; ++j) {
        std::vector<C> col(static_cast<std::size_t>(nn), 0.0);
        if (vals[j] > 1e-12) {
            for (I i = 0; i < nn; ++i) {
                for (I k = 0; k < nn; ++k) {
                    col[i] += am[i * nn + k] * vecs[j][k];
                }
                col[i] /= vals[j];
            }
        } else {
            for (I e = 0; e < nn; ++e) {
                std::vector<C> x(static_cast<std::size_t>(nn), 0.0);
                x[e] = 1.0;
                for (const auto& y : ucols) {
                    C d = 0;
                    for (I i = 0; i < nn; ++i) {
                        d += std::conj(y[i]) * x[i];
                    }
                    for (I i = 0; i < nn; ++i) {
                        x[i] -= d * y[i];
                    }
                }
                double norm = 0;
                for (const C& z : x) {
                    norm += std::norm(z);
                }
                if (norm > 1e-6) {
                    for (C& z : x) {
                        z /= std::sqrt(norm);
                    }
                    col = x;
                    break;
                }
            }
        }
        ucols.push_back(col);
    }
    for (I j = 0; j < nn; ++j) {
        s[j] = vals[j];
        for (I i = 0; i < nn; ++i) {
            u[i + j * *ldu] = FromC(ucols[j][i]);
            vt[j + i * *ldvt] = FromC(std::conj(vecs[j][i]));  // VT 的第 j 行 = v_j^H
        }
    }
    return 0;
}

LinalgFunctions TestFunctions() {
    LinalgFunctions f;
    f.ilp64 = true;
    f.zgemm = reinterpret_cast<std::uintptr_t>(&TestZgemm);
    f.zsyrk = reinterpret_cast<std::uintptr_t>(&TestZsyrk);
    f.dgemv = reinterpret_cast<std::uintptr_t>(&TestDgemv);
    f.zdotc_sub = reinterpret_cast<std::uintptr_t>(&TestZdotc);
    f.zgesdd = reinterpret_cast<std::uintptr_t>(&TestZgesdd);
    f.dsyevd = reinterpret_cast<std::uintptr_t>(&TestDsyevd);
    f.zgetrf = reinterpret_cast<std::uintptr_t>(&TestZgetrf);
    f.dgetrf = reinterpret_cast<std::uintptr_t>(&TestDgetrf);
    return f;
}

// ---- 常量（与 two_qubit.py 相同的定义）--------------------------------------------

using CM = std::vector<C>;  // 行主序方阵

CM MatMul(const CM& a, const CM& b) {
    const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.size()))));
    CM out(a.size(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                out[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    return out;
}

CM Kron(const CM& a, const CM& b) {
    CM out(16);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    out[(i * 2 + k) * 4 + j * 2 + l] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    return out;
}

CM Dagger(const CM& a) {
    const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.size()))));
    CM out(a.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out[j * n + i] = std::conj(a[i * n + j]);
        }
    }
    return out;
}

template <std::size_t N>
std::array<Cx, N> ToArr(const CM& m) {
    std::array<Cx, N> out;
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = FromC(m[i]);
    }
    return out;
}

template <std::size_t N>
CM FromArr(const std::array<Cx, N>& m) {
    CM out(N);
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = ToC(m[i]);
    }
    return out;
}

const double kS = 1.0 / std::sqrt(2.0);
const CM kI2{1, 0, 0, 1};
const CM kX{0, 1, 1, 0};
const CM kY{0, C(0, -1), C(0, 1), 0};
const CM kZ{1, 0, 0, -1};
const CM kH{kS, kS, kS, -kS};
const CM kSg{1, 0, 0, C(0, 1)};
const CM kRx90{kS, C(0, -kS), C(0, -kS), kS};
const CM kSX{C(0.5, 0.5), C(0.5, -0.5), C(0.5, -0.5), C(0.5, 0.5)};
const CM kMagic{kS, 0, 0, C(0, kS), 0, C(0, kS), kS, 0, 0, C(0, kS), -kS, 0, kS, 0, 0, C(0, -kS)};
const CM kCZ{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1};
const CM kCX{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
const CM kSwap{1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

// KakConstants::Make() 与 two_qubit.py 的常量矩阵数值相同。
TEST(KakConstants, MatchTheModuleConstants) {
    const KakConstants c = KakConstants::Make();
    auto same = [](const auto& got, const CM& want) {
        for (std::size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(ToC(got[i]), want[i]) << i;
        }
    };
    same(c.magic, kMagic);
    same(c.magic_dag, Dagger(kMagic));
    same(c.h, kH);
    same(c.sx, kSX);
    same(c.cz4, kCZ);
    same(c.swap4, kSwap);
    same(c.pauli_pairs[1], Kron(kY, kY));
    same(c.swap_ab, Kron(kSg, kSg));
    same(c.swap_bc, Kron(kRx90, kRx90));
    same(c.flip_ac, Kron(kY, kI2));
    const double phases[4][3] = {{1, -1, 1}, {1, 1, -1}, {-1, -1, -1}, {-1, 1, 1}};
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(c.phase_map[static_cast<std::size_t>(r * 4)], 1.0);
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(c.phase_map[static_cast<std::size_t>(r * 4 + 1 + k)], phases[r][k]);
            EXPECT_EQ(c.can_phases[static_cast<std::size_t>(r * 3 + k)], phases[r][k]);
        }
    }
}

KakSynthesizer MakeSynth() {
    KakSynthesizer s;
    EntanglerSpec cz;
    cz.name = "cz";
    cz.matrix = ToArr<16>(kCZ);
    cz.before = {ToArr<4>(kI2), ToArr<4>(kI2)};
    cz.after = {ToArr<4>(kI2), ToArr<4>(kI2)};
    cz.symmetric = true;
    EXPECT_EQ(s.AddEntangler(cz), 0);
    EntanglerSpec cx;
    cx.name = "cx";
    cx.matrix = ToArr<16>(kCX);
    cx.before = {ToArr<4>(kI2), ToArr<4>(kH)};
    cx.after = {ToArr<4>(kI2), ToArr<4>(kH)};
    cx.symmetric = false;
    EXPECT_EQ(s.AddEntangler(cx), 1);
    return s;
}

CM RandomUnitary(std::mt19937_64& rng, int n) {
    std::normal_distribution<double> normal;
    CM a(static_cast<std::size_t>(n * n));
    for (C& z : a) {
        z = C(normal(rng), normal(rng));
    }
    // Gram–Schmidt（按列）
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            C d = 0;
            for (int i = 0; i < n; ++i) {
                d += std::conj(a[i * n + k]) * a[i * n + j];
            }
            for (int i = 0; i < n; ++i) {
                a[i * n + j] -= d * a[i * n + k];
            }
        }
        double norm = 0;
        for (int i = 0; i < n; ++i) {
            norm += std::norm(a[i * n + j]);
        }
        for (int i = 0; i < n; ++i) {
            a[i * n + j] /= std::sqrt(norm);
        }
    }
    return a;
}

CM Canonical(double a, double b, double c) {
    // exp(i(a XX + b YY + c ZZ))：在 magic 基下为对角阵
    const double phases[4][3] = {{1, -1, 1}, {1, 1, -1}, {-1, -1, -1}, {-1, 1, 1}};
    CM d(16, 0.0);
    for (int r = 0; r < 4; ++r) {
        d[r * 5] = std::exp(C(0, phases[r][0] * a + phases[r][1] * b + phases[r][2] * c));
    }
    return MatMul(MatMul(kMagic, d), Dagger(kMagic));
}

double DistanceUpToPhase(const CM& u, const CM& v) {
    C overlap = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        overlap += std::conj(v[i]) * u[i];
    }
    const C phase = overlap / std::abs(overlap);
    double d = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        d = std::max(d, std::abs(u[i] - phase * v[i]));
    }
    return d;
}

CM OpsUnitary(const std::vector<Op>& ops, const CM& entangler) {
    CM u = Kron(kI2, kI2);
    for (const Op& op : ops) {
        CM g;
        if (op.kind == OpKind::kEntangler) {
            g = op.q0 == 0 ? entangler : MatMul(MatMul(kSwap, entangler), kSwap);
        } else {
            CM m;
            switch (op.kind) {
                case OpKind::kRz:
                    m = {std::exp(C(0, -op.param / 2)), 0, 0, std::exp(C(0, op.param / 2))};
                    break;
                case OpKind::kH:
                    m = kH;
                    break;
                case OpKind::kSX:
                    m = kSX;
                    break;
                default:
                    m = kX;
                    break;
            }
            g = op.q0 == 0 ? Kron(m, kI2) : Kron(kI2, m);
        }
        u = MatMul(g, u);
    }
    return u;
}

int CountEntanglers(const std::vector<Op>& ops) {
    return static_cast<int>(std::count_if(ops.begin(), ops.end(), [](const Op& o) { return o.kind == OpKind::kEntangler; }));
}

// ---- 测试 ----------------------------------------------------------------------

TEST(NumpyCompat, MultiplicationOrders) {
    const Cx a{1.0 / 3.0, -2.0 / 7.0};
    const Cx b{5.0 / 11.0, 3.0 / 13.0};
    const Cx p = Mul(MulMode::kPlain, a, b);
    const Cx f1 = Mul(MulMode::kFma1, a, b);
    const Cx f2 = Mul(MulMode::kFma2, a, b);
    const C ref = ToC(a) * ToC(b);
    for (const Cx& z : {p, f1, f2}) {
        EXPECT_NEAR(z.re, ref.real(), 1e-15);
        EXPECT_NEAR(z.im, ref.imag(), 1e-15);
    }
    EXPECT_EQ(f1.re, std::fma(a.re, b.re, -(a.im * b.im)));
    EXPECT_EQ(f2.im, std::fma(a.im, b.re, a.re * b.im));
    // 实数乘法与混合运算
    ArithModes m;
    m.py_mixed = true;
    const Cx mixed = PyMulReal(m, Cx{-0.0, -1.0}, 0.5);
    EXPECT_TRUE(std::signbit(mixed.re));
    EXPECT_EQ(mixed.im, -0.5);
}

TEST(NumpyCompat, DivisionAndScalarFunctions) {
    for (DivMode mode : {DivMode::kPlain, DivMode::kFma}) {
        const Cx q = NpDiv(mode, Cx{1.0, 2.0}, Cx{3.0, -4.0});
        const C ref = C(1, 2) / C(3, -4);
        EXPECT_NEAR(q.re, ref.real(), 1e-15);
        EXPECT_NEAR(q.im, ref.imag(), 1e-15);
    }
    const Cx r = PySqrt(Cx{-4.0, 0.0});
    EXPECT_EQ(r.re, 0.0);
    EXPECT_EQ(r.im, 2.0);
    const Cx z = PySqrt(Cx{0.0, -0.0});
    EXPECT_TRUE(std::signbit(z.im));
    ArithModes m;
    const Cx e = PyExp(m, Cx{0.0, 3.141592653589793 / 2});
    EXPECT_NEAR(e.re, 0.0, 1e-15);
    EXPECT_EQ(e.im, 1.0);
    EXPECT_EQ(PyAtan2(-0.0, -1.0), -3.141592653589793);
    EXPECT_EQ(PyAtan2(0.0, 1.0), 0.0);
    EXPECT_EQ(PyPow(3.0, 2.0), 9.0);
    EXPECT_EQ(PyPow(0.0, 2.0), 0.0);
}

TEST(NumpyLinalg, MatMulSyrkAndDecompositions) {
    const NumpyLinalg np(TestFunctions(), ArithModes{});
    std::mt19937_64 rng(7);
    const M4 a = ToArr<16>(RandomUnitary(rng, 4));
    const M4 b = ToArr<16>(RandomUnitary(rng, 4));
    M4 c;
    np.MatMul(4, View(a), ViewT(b), c.data());
    CM bt(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            bt[i * 4 + j] = ToC(b[j * 4 + i]);
        }
    }
    const CM ref = MatMul(FromArr(a), bt);
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(std::abs(ToC(c[i]) - ref[i]), 0.0, 1e-12);
    }
    // a.T @ a（syrk 分支，结果对称）
    np.MatMul(4, ViewT(a), View(a), c.data());
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(c[i * 4 + j].re, c[j * 4 + i].re);
            EXPECT_EQ(c[i * 4 + j].im, c[j * 4 + i].im);
        }
    }
    // det：酉矩阵的行列式模为 1
    const Cx d = np.Det(4, View(a));
    EXPECT_NEAR(std::abs(ToC(d)), 1.0, 1e-12);
    // svd：秩 1 的矩阵
    M4 u;
    M4 vh;
    std::array<double, 4> sv;
    M4 rank1;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            rank1[i * 4 + j] = FromC(ToC(a[i * 4]) * std::conj(ToC(b[j * 4])) * 2.0);
        }
    }
    ASSERT_TRUE(np.Svd4(rank1, &u, &sv, &vh));
    EXPECT_NEAR(sv[0], 2.0, 1e-10);
    EXPECT_NEAR(sv[1], 0.0, 1e-6);  // 测试实现经由 A^H A，零奇异值只精确到 sqrt(eps)
}

TEST(KakSynthesizer, KakReconstructsAndLiesInTheChamber) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 20; ++trial) {
        const CM u = trial == 0 ? kCZ : (trial == 1 ? kSwap : RandomUnitary(rng, 4));
        const Kak kak = s.Kak4(ToArr<16>(u));
        const double a = kak.v[0];
        const double b = kak.v[1];
        const double c = kak.v[2];
        EXPECT_LE(a, 3.141592653589793 / 4 + 1e-12);
        EXPECT_GE(a + 1e-12, b);
        EXPECT_GE(b + 1e-12, std::fabs(c));
        const CM rebuilt = MatMul(MatMul(FromArr(kak.k1), Canonical(a, b, c)), FromArr(kak.k2));
        CM scaled = rebuilt;
        for (C& z : scaled) {
            z *= std::exp(C(0, kak.phase));
        }
        double err = 0;
        for (int i = 0; i < 16; ++i) {
            err = std::max(err, std::abs(scaled[i] - u[i]));
        }
        EXPECT_LT(err, 1e-9) << "trial " << trial;
        // 局部因子可分解为张量积
        const auto f = s.FactorLocal(kak.k1);
        EXPECT_LT(DistanceUpToPhase(Kron(FromArr(f[0]), FromArr(f[1])), FromArr(kak.k1)), 1e-9);
    }
}

TEST(KakSynthesizer, ExactSynthesisUsesTheMinimalEntanglerCount) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(5);
    struct Case {
        CM u;
        int expected;
    };
    const CM local = Kron(RandomUnitary(rng, 2), RandomUnitary(rng, 2));
    const CM local2 = Kron(RandomUnitary(rng, 2), RandomUnitary(rng, 2));
    std::vector<Case> cases = {
        {local, 0},
        {MatMul(MatMul(local, kCZ), local2), 1},
        {MatMul(MatMul(local, Canonical(0.4, 0.2, 0.0)), local2), 2},
        {MatMul(MatMul(local, Canonical(0.5, 0.3, 0.1)), local2), 3},
        {kSwap, 3},
        {RandomUnitary(rng, 4), 3},
    };
    for (int ent = 0; ent < 2; ++ent) {
        for (int euler = 0; euler < 2; ++euler) {
            for (const Case& c : cases) {
                SynthOptions opt;
                opt.entangler = ent;
                opt.euler = euler;
                bool uses_flag = false;
                const std::vector<Op> ops = s.SynthesizeExact(ToArr<16>(c.u), opt, &uses_flag);
                EXPECT_EQ(CountEntanglers(ops), c.expected);
                EXPECT_LT(DistanceUpToPhase(OpsUnitary(ops, ent == 0 ? kCZ : kCX), c.u), 1e-8);
                for (const Op& op : ops) {
                    if (op.kind == OpKind::kH) {
                        EXPECT_EQ(euler, 0);
                    }
                    if (op.kind == OpKind::kSX) {
                        EXPECT_EQ(euler, 1);
                    }
                }
            }
        }
    }
}

TEST(KakSynthesizer, SwapFaceTemplateHasOneInnerRotation) {
    const KakSynthesizer s = MakeSynth();
    CM zz(16, 0.0);
    const double d[4] = {1, -1, -1, 1};
    for (int i = 0; i < 4; ++i) {
        zz[i * 5] = std::exp(C(0, -0.15 * d[i]));
    }
    const CM u = MatMul(kSwap, zz);
    SynthOptions opt;
    opt.entangler = 1;  // cx
    const auto ops = s.SynthesizeSwapFace(ToArr<16>(u), opt);
    ASSERT_TRUE(ops.has_value());
    EXPECT_EQ(CountEntanglers(*ops), 3);
    EXPECT_LT(DistanceUpToPhase(OpsUnitary(*ops, kCX), u), 1e-8);
    // CZ 不在 SWAP 面上
    EXPECT_FALSE(s.SynthesizeSwapFace(ToArr<16>(kCZ), opt).has_value());
    // 模块开关打开时普通合成也用 SWAP 形模板
    opt.swap_face_template = true;
    bool uses_flag = false;
    const auto shaped = s.SynthesizeExact(ToArr<16>(u), opt, &uses_flag);
    EXPECT_TRUE(uses_flag);
    EXPECT_EQ(CountEntanglers(shaped), 3);
}

TEST(KakSynthesizer, EulerOpsReproduceTheMatrix) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(3);
    std::vector<CM> mats = {kI2, kH, kX, kZ, kSX};
    for (int i = 0; i < 10; ++i) {
        mats.push_back(RandomUnitary(rng, 2));
    }
    for (int basis = 0; basis < 2; ++basis) {
        for (const CM& m : mats) {
            std::vector<Op> ops;
            s.EulerOps(ToArr<4>(m), 0, basis, 1e-9, &ops);
            CM rebuilt = kI2;
            for (const Op& op : ops) {
                CM g = op.kind == OpKind::kRz ? CM{std::exp(C(0, -op.param / 2)), 0, 0, std::exp(C(0, op.param / 2))}
                       : op.kind == OpKind::kH ? kH
                       : op.kind == OpKind::kSX ? kSX
                                                : kX;
                rebuilt = MatMul(g, rebuilt);
            }
            EXPECT_LT(DistanceUpToPhase(rebuilt, m), 1e-9);
            EXPECT_LE(ops.size(), 5u);
        }
    }
    std::vector<Op> ops;
    s.EulerOps(ToArr<4>(kI2), 0, 0, 1e-9, &ops);
    EXPECT_TRUE(ops.empty());
}

TEST(KakSynthesizer, ApproximationFilterAndUnitarityCheck) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(9);
    const Kak kak = s.Kak4(ToArr<16>(RandomUnitary(rng, 4)));
    EXPECT_TRUE(s.ApproximationImpossible(kak, 1e-12, 1e-9, 2));
    // CP(1e-6)：去掉相互作用的不保真度约 5e-14，预算 1e-12 时可以近似
    CM cp(16, 0.0);
    cp[0] = cp[5] = cp[10] = 1.0;
    cp[15] = std::exp(C(0, 2e-6));
    EXPECT_FALSE(s.ApproximationImpossible(s.Kak4(ToArr<16>(cp)), 1e-12, 1e-9, 2));
    CM bad = kCZ;
    bad[0] = 2.0;
    EXPECT_THROW(s.CheckUnitary(ToArr<16>(bad)), NotUnitary);
    EXPECT_NO_THROW(s.CheckUnitary(ToArr<16>(kCZ)));
}

// ---- 单比特合成 ------------------------------------------------------------------

std::array<M2, kNumFixedGates> FixedMatrices() {
    const C t = std::exp(C(0, 3.141592653589793 / 4));
    const std::vector<CM> mats = {kH, kX, kY, kZ, kSg, {1, 0, 0, C(0, -1)}, {1, 0, 0, t},
                                  {1, 0, 0, std::conj(t)}, kSX, Dagger(kSX)};
    std::array<M2, kNumFixedGates> out;
    for (int i = 0; i < kNumFixedGates; ++i) {
        out[static_cast<std::size_t>(i)] = ToArr<4>(mats[static_cast<std::size_t>(i)]);
    }
    return out;
}

CM Gate1Matrix(const Gate1& g, const std::array<M2, kNumFixedGates>& fixed) {
    switch (g.kind) {
        case 0:
            return FromArr(fixed[static_cast<std::size_t>(g.index)]);
        case 1:
            return {std::exp(C(0, -g.p0 / 2)), 0, 0, std::exp(C(0, g.p0 / 2))};
        case 2:
            return {std::cos(g.p0 / 2), C(0, -std::sin(g.p0 / 2)), C(0, -std::sin(g.p0 / 2)), std::cos(g.p0 / 2)};
        case 3:
            return {std::cos(g.p0 / 2), -std::sin(g.p0 / 2), std::sin(g.p0 / 2), std::cos(g.p0 / 2)};
        default: {
            const double c = std::cos(g.p0 / 2);
            const double s = std::sin(g.p0 / 2);
            return {c, -s * std::exp(C(0, g.p2)), s * std::exp(C(0, g.p1)), c * std::exp(C(0, g.p1 + g.p2))};
        }
    }
}

TEST(OneQubitSynthesizer, SynthesizesInSeveralBases) {
    const auto fixed = FixedMatrices();
    const OneQubitSynthesizer synth(ArithModes{}, fixed);
    OneQubitBasis zxz;  // {rz, h, x}
    zxz.rz = zxz.h = zxz.x = true;
    zxz.named = {kGateH, kGateX};
    OneQubitBasis zsx;  // {rz, sx, x}
    zsx.rz = zsx.sx = zsx.x = true;
    zsx.named = {kGateX, kGateSX};
    OneQubitBasis u3;
    u3.u3 = true;
    OneQubitBasis clifford_t;  // {h, s, t}：只能表示 Clifford+T
    clifford_t.h = true;
    clifford_t.named = {kGateH, kGateS, kGateT};
    clifford_t.has_word[1 + 4] = true;
    clifford_t.words[1 + 4] = {kGateT};
    clifford_t.has_word[2 + 4] = true;
    clifford_t.words[2 + 4] = {kGateS};
    clifford_t.has_word[3 + 4] = true;
    clifford_t.words[3 + 4] = {kGateS, kGateT};
    std::mt19937_64 rng(17);
    std::vector<CM> mats = {kI2, kH, kX, kSX, kSg};
    for (int i = 0; i < 8; ++i) {
        mats.push_back(RandomUnitary(rng, 2));
    }
    for (const OneQubitBasis* basis : {&zxz, &zsx, &u3}) {
        for (const CM& m : mats) {
            if (basis == &u3 && std::abs(m[1]) < 1e-12 && std::abs(m[0] - m[3]) > 1e-9) {
                // 没有 rz 类门时对角矩阵只能走 rz（与 Python 相同：无法表示）
                EXPECT_FALSE(synth.Synthesize(ToArr<4>(m), *basis).has_value());
                continue;
            }
            const auto gates = synth.Synthesize(ToArr<4>(m), *basis);
            ASSERT_TRUE(gates.has_value());
            CM rebuilt = kI2;
            for (const Gate1& g : *gates) {
                rebuilt = MatMul(Gate1Matrix(g, fixed), rebuilt);
            }
            EXPECT_LT(DistanceUpToPhase(rebuilt, m), 1e-9);
            EXPECT_LE(gates->size(), 5u);
        }
    }
    // 恒等、具名门
    EXPECT_TRUE(synth.Synthesize(ToArr<4>(kI2), zxz)->empty());
    const auto h = synth.Synthesize(ToArr<4>(kH), zxz);
    ASSERT_EQ(h->size(), 1u);
    EXPECT_EQ((*h)[0].index, kGateH);
    // Clifford+T：S·T 可以表示，随机酉矩阵不行
    const CM st = MatMul(FromArr(fixed[kGateT]), FromArr(fixed[kGateS]));
    EXPECT_TRUE(synth.Synthesize(ToArr<4>(st), clifford_t).has_value());
    EXPECT_FALSE(synth.Synthesize(ToArr<4>(mats.back()), clifford_t).has_value());
}

TEST(OneQubitSynthesizer, ZyzAnglesRebuildTheMatrix) {
    const OneQubitSynthesizer synth(ArithModes{}, FixedMatrices());
    std::mt19937_64 rng(23);
    for (int i = 0; i < 20; ++i) {
        const CM m = RandomUnitary(rng, 2);
        const auto a = synth.ZyzAngles(ToArr<4>(m));
        const Gate1 g{4, 0, a[0], a[1], a[2]};
        EXPECT_LT(DistanceUpToPhase(Gate1Matrix(g, FixedMatrices()), m), 1e-12);
        EXPECT_GE(a[1], 0.0);
        EXPECT_LT(a[1], 2 * 3.141592653589793);
    }
}

}  // namespace
