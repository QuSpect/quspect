#pragma once

#include <pybind11/pybind11.h>

namespace quspect::native::pybind {

// 驻留的属性名：实例 / 类型字典中的键是驻留字符串，查找时按身份命中，不必逐字节比较。
inline pybind11::str InternedStr(const char* s) {
    PyObject* o = PyUnicode_InternFromString(s);
    if (o == nullptr) {
        throw pybind11::error_already_set();
    }
    return pybind11::reinterpret_steal<pybind11::str>(o);
}

}  // namespace quspect::native::pybind
