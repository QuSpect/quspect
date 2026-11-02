#pragma once

namespace quspect::native::detmath {

// 复数（与 numpy complex128 / std::complex<double> 布局相同）。
//
// 只有数据定义，不含运算，可以在任何编译单元中使用；运算见 complex.hpp。
struct Cx {
    double re;
    double im;
};

}  // namespace quspect::native::detmath
