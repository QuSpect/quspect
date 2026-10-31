#pragma once

#include <pybind11/pybind11.h>

#include "quspect_native/synthesis/numpy_compat.hpp"

namespace quspect::native::pybind {

// _KakSynthesizer 对象所用的 numpy BLAS / LAPACK 封装（numpy 语义的 2×2 / 4×4 乘法等）；
// obj 不是 _KakSynthesizer 时返回 nullptr。返回的指针在 obj 存活期间有效。
const synthesis::NumpyLinalg* KakLinalg(pybind11::handle obj);

// 绑定两比特 KAK 合成（_KakSynthesizer）及其 numpy 语义探测函数。
void BindSynthesis(pybind11::module_& m);

}  // namespace quspect::native::pybind
