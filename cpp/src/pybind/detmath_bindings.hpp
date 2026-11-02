#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 确定性数值层（detmath）的测试入口：标量函数与小矩阵分解（跨平台逐位一致性检查用）。
void BindDetmath(pybind11::module_& m);

}  // namespace quspect::native::pybind
