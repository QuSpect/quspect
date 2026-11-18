#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 绑定 T 门数优化（quspect.transpiler._tcount_poly）的原生搜索内核：
// _tcount_todd_batch、_tcount_rm_search、_tcount_flip_walks。
void BindTCount(pybind11::module_& m);

}  // namespace quspect::native::pybind
