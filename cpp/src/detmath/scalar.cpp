#include "quspect_native/detmath/scalar.hpp"

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
double cr_hypot(double, double);
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

double Atan2(double y, double x) {
    return cr_atan2(y, x);
}

double Exp(double x) {
    return cr_exp(x);
}

double Log(double x) {
    return cr_log(x);
}

double Hypot(double x, double y) {
    return cr_hypot(x, y);
}

}  // namespace quspect::native::detmath
