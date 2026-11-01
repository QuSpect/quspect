#include "quspect_native/synthesis/numpy_compat.hpp"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace quspect::native::synthesis {

namespace {

// ---- CBLAS / LAPACK 调用约定 -------------------------------------------------
//
// 与 numpy 的声明一致（numpy/_core/src/common/npy_cblas_base.h、numpy/linalg/umath_linalg.cpp）：
// CBLAS 的枚举为 int；LAPACK 为 Fortran 接口，全部按指针传参，不带字符串长度隐藏参数。
constexpr int kRowMajor = 101;
constexpr int kNoTrans = 111;
constexpr int kTrans = 112;
constexpr int kUpper = 121;

template <typename I>
using ZgemmFn = void (*)(int, int, int, I, I, I, const void*, const void*, I, const void*, I, const void*, void*, I);
template <typename I>
using ZsyrkFn = void (*)(int, int, int, I, I, const void*, const void*, I, const void*, void*, I);
template <typename I>
using DgemvFn = void (*)(int, int, I, I, double, const double*, I, const double*, I, double, double*, I);
template <typename I>
using ZdotcFn = void (*)(I, const void*, I, const void*, I, void*);
template <typename I>
using ZgesddFn = I (*)(char*, I*, I*, Cx*, I*, double*, Cx*, I*, Cx*, I*, Cx*, I*, double*, I*, I*);
template <typename I>
using DsyevdFn = I (*)(char*, char*, I*, double*, I*, double*, double*, I*, I*, I*, I*);
template <typename I>
using ZgetrfFn = I (*)(I*, I*, Cx*, I*, I*, I*);
template <typename I>
using DgetrfFn = I (*)(I*, I*, double*, I*, I*, I*);

template <typename F>
F Fn(std::uintptr_t address) {
    return reinterpret_cast<F>(address);
}

// 4×4 分解的工作区上限（实际约为 zgesdd 264、dsyevd 57 / 23）。
constexpr std::int64_t kMaxWork = 1024;

const Cx kOne{1.0, 0.0};
const Cx kZero{0.0, 0.0};

// numpy matmul.c.src: is_blasable2d（步长单位为元素，itemsize 已约去）。
bool IsBlasable2d(std::ptrdiff_t stride1, std::ptrdiff_t stride2, std::ptrdiff_t d2) {
    return stride2 == 1 && stride1 >= d2;
}

}  // namespace

// ---- 逐元素运算 ----------------------------------------------------------------

#if defined(_MSC_VER)
#define QSPECT_NOINLINE __declspec(noinline)
#else
#define QSPECT_NOINLINE __attribute__((noinline))
#endif

QSPECT_NOINLINE double SeparateSin(double x) {
    return std::sin(x);
}

QSPECT_NOINLINE double SeparateCos(double x) {
    return std::cos(x);
}

void JointSinCos(double x, double* s, double* c) {
#if defined(__APPLE__)
    __sincos(x, s, c);
#elif defined(__GLIBC__)
    ::sincos(x, s, c);
#else
    *s = std::sin(x);
    *c = std::cos(x);
#endif
}

double PyPow(double x, double y) {
    // float_pow 的特殊情形（只处理 x ≥ 0）：y == 0 → 1；x == 1 → 1；x == 0 → 0（y > 0）
    if (y == 0.0 || x == 1.0) {
        return 1.0;
    }
    if (x == 0.0) {
        return 0.0;
    }
    volatile double exponent = y;
    return std::pow(x, exponent);
}

Cx Mul(MulMode mode, const Cx& a, const Cx& b) {
    switch (mode) {
        case MulMode::kFma1:
            return {std::fma(a.re, b.re, -(a.im * b.im)), std::fma(a.re, b.im, a.im * b.re)};
        case MulMode::kFma2:
            return {std::fma(-a.im, b.im, a.re * b.re), std::fma(a.im, b.re, a.re * b.im)};
        case MulMode::kPlain:
        default:
            return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
    }
}

Cx NpDiv(DivMode mode, const Cx& a, const Cx& b) {
    const double in2r_abs = std::fabs(b.re);
    const double in2i_abs = std::fabs(b.im);
    const bool fma = mode == DivMode::kFma;
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) {
            return {a.re / in2r_abs, a.im / in2i_abs};
        }
        const double rat = b.im / b.re;
        const double scl = 1.0 / (fma ? std::fma(b.im, rat, b.re) : b.re + b.im * rat);
        if (fma) {
            return {std::fma(a.im, rat, a.re) * scl, std::fma(-a.re, rat, a.im) * scl};
        }
        return {(a.re + a.im * rat) * scl, (a.im - a.re * rat) * scl};
    }
    const double rat = b.re / b.im;
    const double scl = 1.0 / (fma ? std::fma(b.re, rat, b.im) : b.im + b.re * rat);
    if (fma) {
        return {std::fma(a.re, rat, a.im) * scl, std::fma(a.im, rat, -a.re) * scl};
    }
    return {(a.re * rat + a.im) * scl, (a.im * rat - a.re) * scl};
}

Cx PyMulReal(const ArithModes& m, const Cx& a, double f) {
    if (m.py_mixed) {
        return {a.re * f, a.im * f};
    }
    return Mul(m.py, a, Cx{f, 0.0});
}

double PyAbs(const Cx& a) {
    return std::hypot(a.re, a.im);
}

Cx PySqrt(const Cx& z) {
    // CPython cmathmodule.c: cmath_sqrt_impl（有限输入）
    constexpr int kScaleUp = 2 * (DBL_MANT_DIG / 2) + 1;
    constexpr int kScaleDown = -(kScaleUp + 1) / 2;
    if (z.re == 0.0 && z.im == 0.0) {
        return {0.0, z.im};
    }
    double ax = std::fabs(z.re);
    const double ay = std::fabs(z.im);
    double s;
    if (ax < DBL_MIN && ay < DBL_MIN) {
        ax = std::ldexp(ax, kScaleUp);
        s = std::ldexp(std::sqrt(ax + std::hypot(ax, std::ldexp(ay, kScaleUp))), kScaleDown);
    } else {
        ax /= 8.0;
        s = 2.0 * std::sqrt(ax + std::hypot(ax, ay / 8.0));
    }
    const double d = ay / (2.0 * s);
    if (z.re >= 0.0) {
        return {s, std::copysign(d, z.im)};
    }
    return {d, std::copysign(s, z.im)};
}

Cx PyExp(const ArithModes& m, const Cx& z) {
    // CPython cmathmodule.c: cmath_exp_impl（有限输入）
    static const double kLogLargeDouble = std::log(DBL_MAX / 4.0);  // CM_LOG_LARGE_DOUBLE
    constexpr double kE = 2.7182818284590452354;                     // Py_MATH_E
    double s;
    double c;
    if (m.py_exp_joint) {
        JointSinCos(z.im, &s, &c);
    } else {
        s = SeparateSin(z.im);
        c = SeparateCos(z.im);
    }
    if (z.re > kLogLargeDouble) {
        const double l = std::exp(z.re - 1.0);
        return {l * c * kE, l * s * kE};
    }
    const double l = std::exp(z.re);
    return {l * c, l * s};
}

Cx NpExpImag(const ArithModes& m, const Cx& z) {
    double s;
    double c;
    if (m.np_exp_joint) {
        JointSinCos(z.im, &s, &c);
    } else {
        s = SeparateSin(z.im);
        c = SeparateCos(z.im);
    }
    const double l = std::exp(z.re);
    return {l * c, l * s};
}

double PyPhase(const Cx& z) {
    // cmath.phase → c_atan2：有限输入时与 m_atan2 相同
    return PyAtan2(z.im, z.re);
}

double PyAtan2(double y, double x) {
    // CPython mathmodule.c: m_atan2（有限输入）
    if (y == 0.0) {
        if (std::copysign(1.0, x) == 1.0) {
            return std::copysign(0.0, y);
        }
        return std::copysign(3.141592653589793238462643383279502884197, y);
    }
    return std::atan2(y, x);
}

// ---- BLAS / LAPACK -----------------------------------------------------------

NumpyLinalg::NumpyLinalg(const LinalgFunctions& fns, const ArithModes& modes) : fns_(fns), modes_(modes) {
    if (!fns_.Complete()) {
        throw std::invalid_argument("incomplete BLAS/LAPACK function table");
    }
    // 工作区大小查询（numpy 每次调用前都查询一次；结果只取决于问题规模，这里查一次即可）。
    auto query = [this](auto tag) {
        using I = decltype(tag);
        I m = 4;
        I lda = 4;
        I info = 0;
        // zgesdd（jobz = 'A'）
        {
            char jobz = 'A';
            Cx a[16]{};
            double s[4];
            Cx u[16];
            Cx vt[16];
            Cx work_query{};
            I lwork = -1;
            double rwork[200];
            I iwork[32];
            Fn<ZgesddFn<I>>(fns_.zgesdd)(&jobz, &m, &m, a, &lda, s, u, &lda, vt, &lda, &work_query, &lwork, rwork,
                                         iwork, &info);
            if (info != 0) {
                throw std::runtime_error("zgesdd workspace query failed");
            }
            I count = static_cast<I>(work_query.re);
            if (count == 0) {
                count = 1;
            }
            svd_lwork_ = count;
        }
        // dsyevd（jobz = 'V', uplo = 'L'）
        {
            char jobz = 'V';
            char uplo = 'L';
            double a[16]{};
            double w[4];
            double work_query = 0.0;
            I iwork_query = 0;
            I lwork = -1;
            I liwork = -1;
            Fn<DsyevdFn<I>>(fns_.dsyevd)(&jobz, &uplo, &m, a, &lda, w, &work_query, &lwork, &iwork_query, &liwork,
                                         &info);
            if (info != 0) {
                throw std::runtime_error("dsyevd workspace query failed");
            }
            eigh_lwork_ = static_cast<I>(work_query);
            eigh_liwork_ = iwork_query;
        }
    };
    if (fns_.ilp64) {
        query(std::int64_t{});
    } else {
        query(std::int32_t{});
    }
    if (svd_lwork_ <= 0 || svd_lwork_ > kMaxWork || eigh_lwork_ <= 0 || eigh_lwork_ > kMaxWork || eigh_liwork_ <= 0
        || eigh_liwork_ > kMaxWork) {
        throw std::runtime_error("unexpected LAPACK workspace size");
    }
}

void NumpyLinalg::MatMul(int n, const CView& a, const CView& b, Cx* out) const {
    // numpy matmul.c.src：@TYPE@_matmul 的 matrix @ matrix 分支（n ≥ 2，输出 C 连续）。
    const std::ptrdiff_t d = n;
    Cx tmp1[16];
    Cx tmp2[16];
    const Cx* ip1 = a.data;
    const Cx* ip2 = b.data;
    std::ptrdiff_t is1_m = a.rs;
    std::ptrdiff_t is1_n = a.cs;
    std::ptrdiff_t is2_n = b.rs;
    std::ptrdiff_t is2_p = b.cs;
    const bool i1blasable = IsBlasable2d(is1_m, is1_n, d) || IsBlasable2d(is1_n, is1_m, d);
    const bool i2blasable = IsBlasable2d(is2_n, is2_p, d) || IsBlasable2d(is2_p, is2_n, d);
    if (!i1blasable) {
        // matrix_copy：按步长较小的轴决定复制方向
        const bool transpose = std::abs(is1_m) < std::abs(is1_n);
        const std::ptrdiff_t tm = transpose ? 1 : d;
        const std::ptrdiff_t tn = transpose ? d : 1;
        for (std::ptrdiff_t i = 0; i < d; ++i) {
            for (std::ptrdiff_t j = 0; j < d; ++j) {
                tmp1[i * tm + j * tn] = a.data[i * is1_m + j * is1_n];
            }
        }
        ip1 = tmp1;
        is1_m = tm;
        is1_n = tn;
    }
    if (!i2blasable) {
        const bool transpose = std::abs(is2_n) < std::abs(is2_p);
        const std::ptrdiff_t tn = transpose ? 1 : d;
        const std::ptrdiff_t tp = transpose ? d : 1;
        for (std::ptrdiff_t i = 0; i < d; ++i) {
            for (std::ptrdiff_t j = 0; j < d; ++j) {
                tmp2[i * tn + j * tp] = b.data[i * is2_n + j * is2_p];
            }
        }
        ip2 = tmp2;
        is2_n = tn;
        is2_p = tp;
    }
    // matmul_matrixmatrix（输出 C 连续：ldc = n，不走 o_transpose）
    int trans1;
    int trans2;
    std::ptrdiff_t lda;
    std::ptrdiff_t ldb;
    if (IsBlasable2d(is1_m, is1_n, d)) {
        trans1 = kNoTrans;
        lda = is1_m;
    } else {
        trans1 = kTrans;
        lda = is1_n;
    }
    if (IsBlasable2d(is2_n, is2_p, d)) {
        trans2 = kNoTrans;
        ldb = is2_n;
    } else {
        trans2 = kTrans;
        ldb = is2_p;
    }
    const bool syrk = ip1 == ip2 && is1_m == is2_p && is1_n == is2_n && trans1 != trans2;
    auto call = [&](auto tag) {
        using I = decltype(tag);
        const I nn = static_cast<I>(n);
        if (syrk) {
            Fn<ZsyrkFn<I>>(fns_.zsyrk)(kRowMajor, kUpper, trans1, nn, nn, &kOne, ip1,
                                       static_cast<I>(trans1 == kNoTrans ? lda : ldb), &kZero, out, nn);
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    out[j * n + i] = out[i * n + j];
                }
            }
        } else {
            Fn<ZgemmFn<I>>(fns_.zgemm)(kRowMajor, trans1, trans2, nn, nn, nn, &kOne, ip1, static_cast<I>(lda), ip2,
                                       static_cast<I>(ldb), &kZero, out, nn);
        }
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
}

void NumpyLinalg::MatMul(int n, const RView& a, const CView& b, Cx* out) const {
    // gufunc 的类型转换副本按 C 顺序分配（nditer: npyiter_new_temp_array 中核心维度按 C 顺序补齐）
    Cx tmp[16];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp[i * n + j] = Cx{a.data[i * a.rs + j * a.cs], 0.0};
        }
    }
    MatMul(n, CView{tmp, n, 1}, b, out);
}

void NumpyLinalg::MatMul(int n, const CView& a, const RView& b, Cx* out) const {
    Cx tmp[16];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp[i * n + j] = Cx{b.data[i * b.rs + j * b.cs], 0.0};
        }
    }
    MatMul(n, a, CView{tmp, n, 1}, out);
}

namespace {

// dgemv 的两种整数宽度各自放在独立的不内联函数里。若写成同一函数中的两个分支，两次调用只差在
// 整数常量实参的类型上，GCC 的尾合并（-ftree-tail-merge）会把它们合并成一次调用（aarch64 GCC 13
// 实测保留了 32 位版本）：在栈上传递的 64 位实参只写了低 32 位，ILP64 库读到的是高位垃圾。
#if defined(__GNUC__) && !defined(__clang__)
#define QSPECT_DISTINCT_CALL __attribute__((noinline, no_icf))
#else
#define QSPECT_DISTINCT_CALL QSPECT_NOINLINE
#endif

template <typename I>
QSPECT_DISTINCT_CALL void CallDgemvT4(std::uintptr_t fn, const double* m, const double* v, double* out) {
    Fn<DgemvFn<I>>(fn)(kRowMajor, kTrans, I{4}, I{4}, 1.0, m, I{4}, v, I{1}, 0.0, out, I{1});
}

}  // namespace

void NumpyLinalg::MatVecT4(const double* m, const double* v, double* out) const {
    // DOUBLE_matmul 的 matrix_vector 分支：m.T 不是 C 可用布局，按 RowMajor + Trans 调 dgemv。
    if (fns_.ilp64) {
        CallDgemvT4<std::int64_t>(fns_.dgemv, m, v, out);
    } else {
        CallDgemvT4<std::int32_t>(fns_.dgemv, m, v, out);
    }
}

Cx NumpyLinalg::Vdot16(const Cx* a, const Cx* b) const {
    Cx tmp{};
    auto call = [&](auto tag) {
        using I = decltype(tag);
        Fn<ZdotcFn<I>>(fns_.zdotc_sub)(I{16}, a, I{1}, b, I{1}, &tmp);
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
    // CDOUBLE_dot: sum = 0; sum += tmp
    return {0.0 + tmp.re, 0.0 + tmp.im};
}

Cx NumpyLinalg::Det(int n, const CView& a) const {
    Cx f[16];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f[i + j * n] = a.data[i * a.rs + j * a.cs];  // Fortran 顺序
        }
    }
    std::int64_t pivots[4];
    std::int64_t info = 0;
    auto call = [&](auto tag) {
        using I = decltype(tag);
        I m = static_cast<I>(n);
        I lda = m;
        I ipiv[4];
        I inf = 0;
        Fn<ZgetrfFn<I>>(fns_.zgetrf)(&m, &m, f, &lda, ipiv, &inf);
        info = inf;
        for (int i = 0; i < n; ++i) {
            pivots[i] = ipiv[i];
        }
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
    Cx sign;
    double logdet;
    if (info == 0) {
        int change_sign = 0;
        for (int i = 0; i < n; ++i) {
            change_sign += (pivots[i] != i + 1) ? 1 : 0;
        }
        sign = (change_sign % 2) ? Cx{-1.0, 0.0} : Cx{1.0, 0.0};
        logdet = 0.0;
        for (int i = 0; i < n; ++i) {
            const Cx& e = f[i * (n + 1)];
            const double abs_element = std::hypot(e.re, e.im);
            const Cx sign_element{e.re / abs_element, e.im / abs_element};
            sign = Mul(modes_.linalg, sign, sign_element);
            logdet += std::log(abs_element);
        }
    } else {
        sign = Cx{0.0, 0.0};
        logdet = -HUGE_VAL;
    }
    return Mul(modes_.linalg, sign, Cx{std::exp(logdet), 0.0});
}

bool NumpyLinalg::DetNegative4(const RView& a) const {
    double f[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            f[i + j * 4] = a.data[i * a.rs + j * a.cs];
        }
    }
    std::int64_t pivots[4];
    std::int64_t info = 0;
    auto call = [&](auto tag) {
        using I = decltype(tag);
        I m = 4;
        I lda = 4;
        I ipiv[4];
        I inf = 0;
        Fn<DgetrfFn<I>>(fns_.dgetrf)(&m, &m, f, &lda, ipiv, &inf);
        info = inf;
        for (int i = 0; i < 4; ++i) {
            pivots[i] = ipiv[i];
        }
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
    double sign;
    double logdet;
    if (info == 0) {
        int change_sign = 0;
        for (int i = 0; i < 4; ++i) {
            change_sign += (pivots[i] != i + 1) ? 1 : 0;
        }
        sign = (change_sign % 2) ? -1.0 : 1.0;
        logdet = 0.0;
        for (int i = 0; i < 4; ++i) {
            double e = f[i * 5];
            if (e < 0.0) {
                sign = -sign;
                e = -e;
            }
            logdet += std::log(e);
        }
    } else {
        sign = 0.0;
        logdet = -HUGE_VAL;
    }
    return sign * std::exp(logdet) < 0.0;
}

bool NumpyLinalg::Svd4(const M4& a, M4* u, std::array<double, 4>* s, M4* vh) const {
    Cx f[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            f[i + j * 4] = a[i * 4 + j];
        }
    }
    for (const Cx& z : f) {
        if (!std::isfinite(z.re) || !std::isfinite(z.im)) {
            return false;
        }
    }
    Cx uf[16];
    Cx vtf[16];
    double sv[4];
    std::int64_t info = 0;
    auto call = [&](auto tag) {
        using I = decltype(tag);
        char jobz = 'A';
        I m = 4;
        I ld = 4;
        I lwork = static_cast<I>(svd_lwork_);
        Cx work[kMaxWork];
        double rwork[200];
        I iwork[32];
        I inf = 0;
        Fn<ZgesddFn<I>>(fns_.zgesdd)(&jobz, &m, &m, f, &ld, sv, uf, &ld, vtf, &ld, work, &lwork, rwork, iwork, &inf);
        info = inf;
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
    if (info != 0) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        (*s)[i] = sv[i];
        for (int j = 0; j < 4; ++j) {
            (*u)[i * 4 + j] = uf[i + j * 4];
            (*vh)[i * 4 + j] = vtf[i + j * 4];
        }
    }
    return true;
}

bool NumpyLinalg::Eigh4(const std::array<double, 16>& a, std::array<double, 4>* w, std::array<double, 16>* v) const {
    double f[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            f[i + j * 4] = a[i * 4 + j];
        }
    }
    double wv[4];
    std::int64_t info = 0;
    auto call = [&](auto tag) {
        using I = decltype(tag);
        char jobz = 'V';
        char uplo = 'L';
        I m = 4;
        I lda = 4;
        I lwork = static_cast<I>(eigh_lwork_);
        I liwork = static_cast<I>(eigh_liwork_);
        double work[kMaxWork];
        I iwork[kMaxWork];
        I inf = 0;
        Fn<DsyevdFn<I>>(fns_.dsyevd)(&jobz, &uplo, &m, f, &lda, wv, work, &lwork, iwork, &liwork, &inf);
        info = inf;
    };
    if (fns_.ilp64) {
        call(std::int64_t{});
    } else {
        call(std::int32_t{});
    }
    if (info != 0) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        (*w)[i] = wv[i];
        for (int j = 0; j < 4; ++j) {
            (*v)[i * 4 + j] = f[i + j * 4];
        }
    }
    return true;
}

}  // namespace quspect::native::synthesis
