#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定线路 IR（_Circuit）、类型登记函数与 OpenQASM 2.0 解析快速路径（_qasm2_parse）。
void BindIr(pybind11::module_& m);

}  // namespace quspect::native::pybind
