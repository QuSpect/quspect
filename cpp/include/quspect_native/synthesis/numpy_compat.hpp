#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "quspect_native/detmath/cx.hpp"

namespace quspect::native::synthesis {

// 与 numpy / CPython 逐位一致的小矩阵运算。
//
// 两比特 KAK 合成（quspect.synthesis.two_qubit）的 Python 参考实现由一串 numpy 调用组成：矩阵乘法走
// BLAS（zgemm / zsyrk / dgemv），分解走 LAPACK（zgesdd / dsyevd / zgetrf），逐元素复数运算走 numpy 的
// ufunc 内循环，标量运算走 CPython 的 complex / cmath。分解出的奇异向量 / 特征向量（以及退化子空间里
// 的基）取决于具体的 LAPACK 实现，所以这里不自己实现分解，而是：
//
//   * BLAS / LAPACK：调用 numpy 自己链接的那一份库（地址由 Python 端从 numpy 的构建配置解析出来，
//     macOS 上是 Accelerate，wheel 里是 scipy-openblas），并按 numpy 的方式调用——同样的驱动函数、
//     同样的参数（转置标志、leading dimension、工作区大小查询）、同样的数据布局；
//   * 逐元素运算：不同平台上 numpy / CPython 的复数乘除是否被编译器收缩成 FMA、收缩成哪一种并不
//     相同，这里把几种可能的运算顺序都写出来（``MulMode`` / ``DivMode``），由 Python 端在加载时用
//     随机输入逐一比对 numpy / CPython 的实际结果，选出唯一一致的那种；都不一致时不启用原生路径。
//
// 本文件以 -ffp-contract=off 编译：所有 FMA 都显式写成 std::fma。

// 复数（与 numpy complex128 / std::complex<double> 布局相同）。
using Cx = detmath::Cx;

using M2 = std::array<Cx, 4>;   // 2×2，行主序
using M4 = std::array<Cx, 16>;  // 4×4，行主序

// numpy 语义的原生实现不能复刻 Python 参考结果（比较落在阈值附近、输入不是有限值等），需要退回
// Python 参考实现。
class Fallback : public std::runtime_error {
 public:
    explicit Fallback(const char* what) : std::runtime_error(what) {}
};

// 复数乘法 a·b 的运算顺序。
enum class MulMode : int {
    kPlain = 0,  // (ar·br − ai·bi, ar·bi + ai·br)，每个乘积单独舍入
    kFma1 = 1,   // (fma(ar, br, −(ai·bi)), fma(ar, bi, ai·br))
    kFma2 = 2,   // (fma(−ai, bi, ar·br), fma(ai, br, ar·bi))
};

// numpy 复数除法内循环（Smith 算法，乘以倒数）是否收缩成 FMA。
enum class DivMode : int {
    kPlain = 0,
    kFma = 1,
};

// 各处运算语义（由 Python 端探测）。
struct ArithModes {
    MulMode arr_scalar = MulMode::kFma1;   // numpy：数组 * 复数标量（数组在左）
    MulMode scalar_arr = MulMode::kFma1;   // numpy：复数标量 * 数组（标量在左）
    MulMode kron = MulMode::kFma1;         // numpy：_kron2 的广播乘法
    MulMode strided_real = MulMode::kFma1; // numpy：跨步复数视图 * 实数标量（u[:, 0] * scale）
    MulMode np_scalar = MulMode::kFma1;    // numpy 标量（np.complex128）乘法
    MulMode linalg = MulMode::kFma1;       // numpy.linalg 内部（det 的符号累乘）
    MulMode py = MulMode::kFma1;           // CPython complex * complex
    bool py_mixed = false;                 // CPython complex * float 是否为混合运算（3.14 起）
    DivMode np_div = DivMode::kFma;        // numpy：数组 / 复数标量
    // cmath.exp / numpy 的复数 exp 内部的 sin、cos 是否由编译器合并成一次 sincos 调用（macOS 的
    // __sincos_stret 与单独的 sin 结果可能差 1 ulp）。
    bool py_exp_joint = false;
    bool np_exp_joint = false;
};

Cx Mul(MulMode mode, const Cx& a, const Cx& b);
// numpy 复数除法内循环（CDOUBLE_divide）。
Cx NpDiv(DivMode mode, const Cx& a, const Cx& b);
// CPython complex * float（float 在右；3.13 及以前转换成 (f, 0.0) 后做完整复数乘法）。
Cx PyMulReal(const ArithModes& m, const Cx& a, double f);
// CPython ``abs(complex)``（_Py_c_abs，有限输入时为 hypot）。
double PyAbs(const Cx& a);
// CPython ``cmath.sqrt`` / ``cmath.exp`` / ``cmath.phase``（有限输入）。
Cx PySqrt(const Cx& z);
Cx PyExp(const ArithModes& m, const Cx& z);
// numpy ``np.exp`` 作用在实部为 ±0 的复数上（libm cexp：(cos y, sin y)）。
Cx NpExpImag(const ArithModes& m, const Cx& z);
double PyPhase(const Cx& z);
// CPython ``math.atan2``（m_atan2）。
double PyAtan2(double y, double x);
// libm sin / cos。Python 端是两次独立调用；编译器可能把同一参数的 sin、cos 合并成 sincos
// （结果不保证与分开调用逐位相同），所以经由不内联的函数调用。
double SeparateSin(double x);
double SeparateCos(double x);
// 一次调用同时求 sin、cos（macOS：__sincos_stret；glibc：sincos；其他平台：分别调用）。
void JointSinCos(double x, double* s, double* c);
// CPython ``float ** float``（x ≥ 0 的有限值：libm pow；编译器会把常数指数 2 的 pow 换成乘法，
// 这里保证调用的是 libm）。
double PyPow(double x, double y);

// numpy 链接的 BLAS / LAPACK 函数地址（0 表示缺失）与整数宽度。
struct LinalgFunctions {
    bool ilp64 = true;
    std::uintptr_t zgemm = 0;
    std::uintptr_t zsyrk = 0;
    std::uintptr_t dgemv = 0;
    std::uintptr_t zdotc_sub = 0;
    std::uintptr_t zgesdd = 0;
    std::uintptr_t dsyevd = 0;
    std::uintptr_t zgetrf = 0;
    std::uintptr_t dgetrf = 0;

    bool Complete() const {
        return zgemm && zsyrk && dgemv && zdotc_sub && zgesdd && dsyevd && zgetrf && dgetrf;
    }
};

// 二维矩阵视图：元素步长（单位为元素，不是字节）。numpy 的 ``a.T`` 是 (1, n) 步长的视图。
struct CView {
    const Cx* data;
    std::ptrdiff_t rs;  // 行步长
    std::ptrdiff_t cs;  // 列步长
};
struct RView {
    const double* data;
    std::ptrdiff_t rs;
    std::ptrdiff_t cs;
};

inline CView View(const M2& m) {
    return {m.data(), 2, 1};
}
inline CView View(const M4& m) {
    return {m.data(), 4, 1};
}
// numpy 的 ``m.T``（C 连续数组的转置视图）。
inline CView ViewT(const M2& m) {
    return {m.data(), 1, 2};
}
inline CView ViewT(const M4& m) {
    return {m.data(), 1, 4};
}

class NumpyLinalg {
 public:
    NumpyLinalg(const LinalgFunctions& fns, const ArithModes& modes);

    const ArithModes& modes() const {
        return modes_;
    }

    // ``a @ b``（n×n 复数，n = 2 或 4），输出 C 连续。复刻 numpy matmul 内循环的 BLAS 选择：
    // 两个操作数可直接交给 BLAS 时按步长决定转置标志；``x.T @ x`` 走 syrk。
    void MatMul(int n, const CView& a, const CView& b, Cx* out) const;
    // 实数操作数先按 numpy gufunc 的类型转换规则复制成 C 连续的复数数组（(x, 0.0)）。
    void MatMul(int n, const RView& a, const CView& b, Cx* out) const;
    void MatMul(int n, const CView& a, const RView& b, Cx* out) const;

    // ``m.T @ v``（4×4 实数矩阵的转置视图乘 4 维实数向量；numpy 走 dgemv）。
    void MatVecT4(const double* m, const double* v, double* out) const;

    // ``np.vdot(a, b)``（16 个元素，cblas_zdotc_sub）。
    Cx Vdot16(const Cx* a, const Cx* b) const;

    // ``np.linalg.det``（复数 n×n，n = 2 或 4）：zgetrf + numpy 的 slogdet 累乘。
    Cx Det(int n, const CView& a) const;
    // ``np.linalg.det(a) < 0``（4×4 实数）。
    bool DetNegative4(const RView& a) const;

    // ``np.linalg.svd(a)``（4×4 复数，full_matrices=True）：u、s、vh 均按 numpy 的逻辑下标排列（行主序）。
    // LAPACK 失败或输入不是有限值时返回 false。
    bool Svd4(const M4& a, M4* u, std::array<double, 4>* s, M4* vh) const;

    // ``np.linalg.eigh(a)``（4×4 实对称，UPLO='L'）：特征向量按 numpy 的逻辑下标（行主序，列为向量）。
    bool Eigh4(const std::array<double, 16>& a, std::array<double, 4>* w, std::array<double, 16>* v) const;

 private:
    LinalgFunctions fns_;
    ArithModes modes_;
    std::int64_t svd_lwork_ = 0;
    std::int64_t eigh_lwork_ = 0;
    std::int64_t eigh_liwork_ = 0;
};

}  // namespace quspect::native::synthesis
