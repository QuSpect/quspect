#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定线路级的批量辅助：QuantumCircuit 的比特计数、transpile 的门数 / 深度统计、compose 的
// 比特重映射与两比特合成结果到指令的转换（都是 Python 端逐条循环的等价实现）。
void BindCircuitHelpers(pybind11::module_& m);

}  // namespace quspect::native::pybind
