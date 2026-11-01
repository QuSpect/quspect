#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定相位折叠（quspect.transpiler._phase_fold.PhaseFoldingPass.run）的原生实现 _phase_fold。
void BindPhaseFold(pybind11::module_& m);

}  // namespace quspect::native::pybind
