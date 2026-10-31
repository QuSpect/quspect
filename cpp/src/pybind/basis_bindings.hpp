#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定目标门集翻译（quspect.transpiler._basis.translate_to_target）的原生实现 _translate_to_target。
void BindBasis(pybind11::module_& m);

}  // namespace quspect::native::pybind
