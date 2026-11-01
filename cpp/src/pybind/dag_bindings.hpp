#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定 DAGCircuit 的 C++ 图存储内核（_DagCore）与类型登记函数。
void BindDagCore(pybind11::module_& m);

}  // namespace quspect::native::pybind
