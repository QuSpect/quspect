#include "quspect_native/detmath/scalar.hpp"

#include <cfloat>
#include <cmath>
#include <utility>

// CORE-MATH（cpp/third_party/core-math）的 C 接口：binary64、舍入到最近时正确舍入。
extern "C" {
double cr_sin(double);
double cr_cos(double);
double cr_tan(double);
double cr_asin(double);
double cr_acos(double);
double cr_atan(double);
double cr_atan2(double, double);
double cr_exp(double);
double cr_log(double);
double cr_pow(double, double);
}

namespace quspect::native::detmath {

double Sin(double x) {
    return cr_sin(x);
}

double Cos(double x) {
    return cr_cos(x);
}

double Tan(double x) {
    return cr_tan(x);
}

double Asin(double x) {
    return cr_asin(x);
}

double Acos(double x) {
    return cr_acos(x);
}

double Atan(double x) {
    return cr_atan(x);
}

namespace {

// π 与 π/2 的双精度和（hi + lo，lo 为余项的舍入值）。
constexpr double kPiHi = 0x1.921fb54442d18p+1;
constexpr double kPiLo = 0x1.1a62633145c07p-53;
constexpr double kHalfPiHi = 0x1.921fb54442d18p+0;
constexpr double kHalfPiLo = 0x1.1a62633145c07p-54;

// atan(n / d)，0 < |n| ≤ 2^-30·|d|（d > 0）：q = RN(n/d)，余数 r = n − q·d 由 fma 精确求得，
// atan(n/d) = q + r/d − q³/3 + O(q⁵)；修正项远小于 ulp(q)，一次舍入。
double AtanOfSmallQuotient(double n, double d) {
    const double q = n / d;
    if (std::fabs(q) < 0x1p-1000) {
        return q;
    }
    const double r = std::fma(-q, d, n);
    const double corr = r / d - (q * q * q) / 3.0;
    return q + corr;
}

}  // namespace

double Atan2(double y, double x) {
    // |y/x| ≤ 2^-30 或 ≥ 2^30（有限非零输入）：CORE-MATH 在指数差 ≥ 53 时走慢速的精确路径
    // （保存 / 恢复浮点异常标志），且在这一区域曾出现过意外的最坏情况（见 third_party/core-math/
    // README.md）。这里直接用 atan(t) = t − t³/3 求值：只用 IEEE 运算与 fma，结果与平台无关，
    // 除非精确值离舍入中点不到约 2^-50 ulp，否则就是正确舍入的结果。
    const double ax = std::fabs(x);
    const double ay = std::fabs(y);
    if (std::isfinite(ax) && std::isfinite(ay) && ax != 0.0 && ay != 0.0) {
        if (ay <= 0x1p-30 * ax) {
            if (x > 0.0) {
                return AtanOfSmallQuotient(y, x);
            }
            // atan2(y, x) = sign(y)·π − atan(|y| / |x|)
            const double a = AtanOfSmallQuotient(ay, ax);
            const double r = kPiHi + (kPiLo - a);
            return y > 0.0 ? r : -r;
        }
        if (ax <= 0x1p-30 * ay) {
            // atan2(y, x) = sign(y)·π/2 − atan(x / |y|)
            const double a = AtanOfSmallQuotient(x, ay);
            const double r = kHalfPiHi + (kHalfPiLo - a);
            return y > 0.0 ? r : -r;
        }
    }
    return cr_atan2(y, x);
}

double Exp(double x) {
    return cr_exp(x);
}

double Log(double x) {
    return cr_log(x);
}

double Hypot(double x, double y) {
    // √(a² + b²)（a ≥ b ≥ 0）：a²、b² 由 fma 精确拆成双精度和，r = √(hi)，再用精确的余量
    // S − r²（fma）做一次牛顿修正 r + (S − r²)/(2r)，最后一次舍入。只用 IEEE 运算、fma 与 sqrt，
    // 与平台无关；误差远小于 1 ulp（精确值离舍入中点约 2^-45 ulp 以内时才可能与正确舍入不同）。
    // 不调用 CORE-MATH 的 cr_hypot：它在每次调用时保存 / 恢复浮点异常标志（非 x86 平台上是
    // fegetexceptflag / fesetexceptflag 的库调用），在编译路径上的开销很大。
    double a = std::fabs(x);
    double b = std::fabs(y);
    if (!(a <= DBL_MAX) || !(b <= DBL_MAX)) {
        if (std::isinf(a) || std::isinf(b)) {
            return HUGE_VAL;
        }
        return x + y;  // NaN
    }
    if (a < b) {
        std::swap(a, b);
    }
    if (b == 0.0) {
        return a;
    }
    // 缩放到不会溢出 / 下溢的范围（乘 2 的幂是精确的）
    double scale = 1.0;
    if (a > 0x1p500) {
        a *= 0x1p-600;
        b *= 0x1p-600;
        scale = 0x1p600;
    } else if (a < 0x1p-500) {
        a *= 0x1p600;
        b *= 0x1p600;
        scale = 0x1p-600;
    }
    const double h = a * a;
    const double hl = std::fma(a, a, -h);
    const double l = b * b;
    const double ll = std::fma(b, b, -l);
    const double sh = h + l;
    const double sl = ((h - sh) + l) + (hl + ll);  // h ≥ l：快速两数和的误差项
    const double r = std::sqrt(sh);
    const double d = std::fma(-r, r, sh) + sl;     // S − r²
    return (r + d / (2.0 * r)) * scale;
}

double Pow(double x, double y) {
    return cr_pow(x, y);
}

}  // namespace quspect::native::detmath
