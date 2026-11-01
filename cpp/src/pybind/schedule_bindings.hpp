#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 深度调度的输入准备（_schedule_info、_block_schedule_inputs）。
void BindSchedule(pybind11::module_& m);

}  // namespace quspect::native::pybind
