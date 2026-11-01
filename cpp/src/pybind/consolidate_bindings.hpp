#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 块合并的扫描（_consolidate_blocks、_single_qubit_runs）。
void BindConsolidate(pybind11::module_& m);

}  // namespace quspect::native::pybind
