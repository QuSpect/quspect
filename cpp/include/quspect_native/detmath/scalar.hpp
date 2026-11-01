#pragma once

#include <cmath>

#include "quspect_native/detmath/config.hpp"

namespace quspect::native::detmath {

// 正确舍入的标量函数（binary64，舍入到最近）。
//
// 超越函数由 CORE-MATH（cpp/third_party/core-math）实现：结果是精确值舍入到最近的 double，
// 与平台、编译器、CPU 无关。sqrt、remainder、floor、fabs 由 IEEE 754 规定为精确（或正确舍入）
// 运算，各平台的实现结果相同，直接使用标准库。
//
// 定义域与 C 标准库相同（NaN / ±Inf / 越界输入返回与 libm 相同的特殊值）。

double Sin(double x);
double Cos(double x);
double Tan(double x);
double Asin(double x);
double Acos(double x);
double Atan(double x);
double Atan2(double y, double x);
double Exp(double x);
double Log(double x);
double Hypot(double x, double y);

inline double Sqrt(double x) {
    return std::sqrt(x);
}

// IEEE 754 remainder：x − n·y（n 为最接近 x / y 的整数，平局取偶数），结果总是精确的。
inline double Remainder(double x, double y) {
    return std::remainder(x, y);
}

}  // namespace quspect::native::detmath
