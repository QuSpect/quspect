#include "quspect_native/detmath/complex.hpp"

#include <cmath>

namespace quspect::native::detmath {

Cx Div(const Cx& a, const Cx& b) {
    // Smith (1962)：先除以模较大的分量，避免 br² + bi² 上溢 / 下溢
    if (std::fabs(b.re) >= std::fabs(b.im)) {
        const double r = b.im / b.re;
        const double d = b.re + b.im * r;
        return {(a.re + a.im * r) / d, (a.im - a.re * r) / d};
    }
    const double r = b.re / b.im;
    const double d = b.im + b.re * r;
    return {(a.re * r + a.im) / d, (a.im * r - a.re) / d};
}

Cx Sqrt(const Cx& z) {
    if (z.re == 0.0 && z.im == 0.0) {
        return {0.0, z.im};
    }
    // s = sqrt((|z| + |re|) / 2)：两个非负数相加，没有相消；0.5 倍先乘，避免上溢
    const double t = Hypot(z.re, z.im);
    const double s = Sqrt(0.5 * t + 0.5 * std::fabs(z.re));
    if (z.re >= 0.0) {
        return {s, z.im / (2.0 * s)};
    }
    return {std::fabs(z.im) / (2.0 * s), std::copysign(s, z.im)};
}

}  // namespace quspect::native::detmath
