#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定两比特 KAK 合成（_KakSynthesizer）及其 numpy 语义探测函数。
void BindSynthesis(pybind11::module_& m);

}  // namespace quspect::native::pybind
