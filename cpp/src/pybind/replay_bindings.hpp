#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定路由事件回放（quspect.transpiler._mapping.SABRE._finish 与 estimate_two_qubit_cost）的原生实现：
// _ReplayTable、_replay_finish。
void BindReplay(pybind11::module_& m);

}  // namespace quspect::native::pybind
