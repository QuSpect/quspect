#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定单比特门最小化（quspect.transpiler._gauge._minimize）的整体原生实现 _gauge_minimize。
void BindGauge(pybind11::module_& m);

}  // namespace quspect::native::pybind
