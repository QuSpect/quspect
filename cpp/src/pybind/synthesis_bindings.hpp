#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定两比特 / 单比特合成（_KakSynthesizer）与确定性门矩阵（_gate_matrix）。
void BindSynthesis(pybind11::module_& m);

}  // namespace quspect::native::pybind
