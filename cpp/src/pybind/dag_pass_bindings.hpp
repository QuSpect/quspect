#pragma once

#include <pybind11/pybind11.h>

#include "dag_core.hpp"

namespace quspect::native::pybind {

// 在 _DagCore 上登记原生优化 pass 与收敛签名（由 BindDagCore 调用）。
void BindDagPasses(pybind11::module_& m, pybind11::class_<dag_core::PyDagCore>& core);

}  // namespace quspect::native::pybind
