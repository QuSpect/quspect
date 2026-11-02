#pragma once

#include <cmath>

#include "quspect_native/detmath/config.hpp"
#include "quspect_native/detmath/cx.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::detmath {

// 复数运算：每个函数的运算顺序固定（见各函数的注释），不含 FMA。

inline Cx Add(const Cx& a, const Cx& b) {
    return {a.re + b.re, a.im + b.im};
}

inline Cx Sub(const Cx& a, const Cx& b) {
    return {a.re - b.re, a.im - b.im};
}

inline Cx Neg(const Cx& a) {
    return {-a.re, -a.im};
}

inline Cx Conj(const Cx& a) {
    return {a.re, -a.im};
}

// (ar·br − ai·bi, ar·bi + ai·br)，每个乘积单独舍入。
inline Cx Mul(const Cx& a, const Cx& b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// conj(a)·b = (ar·br + ai·bi, ar·bi − ai·br)。
inline Cx MulConj(const Cx& a, const Cx& b) {
    return {a.re * b.re + a.im * b.im, a.re * b.im - a.im * b.re};
}

inline Cx Scale(const Cx& a, double s) {
    return {a.re * s, a.im * s};
}

inline Cx DivReal(const Cx& a, double s) {
    return {a.re / s, a.im / s};
}

// |a|²（ar² + ai²）。
inline double Norm2(const Cx& a) {
    return a.re * a.re + a.im * a.im;
}

// |a|（正确舍入的 hypot）。
inline double Abs(const Cx& a) {
    return Hypot(a.re, a.im);
}

// arg a ∈ [−π, π]（atan2(im, re)，带符号零的约定与 C 标准相同）。
inline double Arg(const Cx& a) {
    return Atan2(a.im, a.re);
}

// e^{iθ} = (cos θ, sin θ)。
inline Cx ExpI(double theta) {
    return {Cos(theta), Sin(theta)};
}

// e^{z} = e^{re}·(cos im, sin im)。re = ±0 时 e^{re} = 1（精确）。
inline Cx Exp(const Cx& z) {
    const double e = Exp(z.re);
    return {e * Cos(z.im), e * Sin(z.im)};
}

// a / b（Smith 算法，按 |br|、|bi| 的大小选择分支）。
Cx Div(const Cx& a, const Cx& b);

// 主平方根（实部 ≥ 0，虚部符号与 z.im 相同；z.im = ±0 且 z.re < 0 时虚部为 ±sqrt(−re)）。
Cx Sqrt(const Cx& z);

}  // namespace quspect::native::detmath
