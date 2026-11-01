// 深度调度（quspect.transpiler._schedule）的输入准备：把指令序列直接整理成 _Info 的 CSR 数组与
// 块调度的输入，不经逐条 Python 属性访问。门的对易轴先查名字表（与 _axes 相同的 dict），表中没有
// 的门回调 Python 的 _axes（同一个缓存 dict）；块调度的门矩阵同样按 Gate 为键查 Python dict，
// 未命中时回调 Python 求矩阵。指令不是普通 CircuitInstruction（子类等）时返回 None，由 Python 端
// 按原实现处理。
#include "schedule_bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "py_interned.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

constexpr long kNone = -1;  // _schedule._NONE

struct ScheduleNames {
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str clbits = InternedStr("clbits");
    py::str gate = InternedStr("gate");
    py::str name = InternedStr("name");
};

ScheduleNames& Names() {
    static ScheduleNames* names = new ScheduleNames();
    return *names;
}

py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

// 比特元组：由 int 组成的 tuple，否则返回 false（交给 Python）。
bool ReadBits(PyObject* t, std::vector<long long>& out) {
    out.clear();
    if (!PyTuple_CheckExact(t)) {
        return false;
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(t); ++i) {
        PyObject* item = PyTuple_GET_ITEM(t, i);
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}

struct Instr {
    PyObject* obj;
    bool is_gate;
    bool is_barrier;
    py::object gate;
    std::vector<long long> qubits;
    std::vector<long long> clbits;
};

// 解码全部指令；遇到不能直接处理的指令返回 false。
bool Decode(const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
            const py::object& type_barrier, std::vector<Instr>& out) {
    ScheduleNames& n = Names();
    py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(instrs.ptr(), "instructions"));
    if (!fast) {
        throw py::error_already_set();
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast.ptr());
    PyObject** items = PySequence_Fast_ITEMS(fast.ptr());
    out.resize(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* ins = items[i];
        if (Py_TYPE(ins) != reinterpret_cast<PyTypeObject*>(instr_cls.ptr())) {
            return false;
        }
        Instr& r = out[static_cast<std::size_t>(i)];
        r.obj = ins;
        py::object type = Attr(ins, n.type);
        r.is_gate = type.is(type_gate);
        r.is_barrier = type.is(type_barrier);
        r.gate = Attr(ins, n.gate);
        if (!ReadBits(Attr(ins, n.qubits).ptr(), r.qubits) || !ReadBits(Attr(ins, n.clbits).ptr(), r.clbits)) {
            return false;
        }
    }
    return true;
}

py::object IntList(const std::vector<long>& v) {
    PyObject* out = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (out == nullptr) {
        throw py::error_already_set();
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), PyLong_FromLong(v[i]));
    }
    return py::reinterpret_steal<py::object>(out);
}

py::object BoolList(const std::vector<bool>& v) {
    PyObject* out = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (out == nullptr) {
        throw py::error_already_set();
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        PyObject* b = v[i] ? Py_True : Py_False;
        Py_INCREF(b);
        PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), b);
    }
    return py::reinterpret_steal<py::object>(out);
}

// 一条指令每个比特上的对易轴（_schedule._axes）。
void Axes(const Instr& ins, const py::dict& axes_1q, const py::dict& axes_mq, const py::object& axes_fn,
          const py::object& cache, std::vector<long>& out) {
    const std::size_t n = ins.qubits.size();
    out.assign(n, kNone);
    if (!ins.is_gate || ins.gate.is_none()) {
        return;
    }
    py::object name = Attr(ins.gate.ptr(), Names().name);
    if (PyUnicode_CheckExact(name.ptr())) {
        if (n == 1) {
            PyObject* hit = PyDict_GetItemWithError(axes_1q.ptr(), name.ptr());
            if (hit == nullptr && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            if (hit != nullptr && PyLong_CheckExact(hit)) {
                out[0] = PyLong_AsLong(hit);
                return;
            }
        } else if (n > 1) {
            PyObject* hit = PyDict_GetItemWithError(axes_mq.ptr(), name.ptr());
            if (hit == nullptr && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            if (hit != nullptr && PyTuple_CheckExact(hit) && static_cast<std::size_t>(PyTuple_GET_SIZE(hit)) == n) {
                for (std::size_t i = 0; i < n; ++i) {
                    out[i] = PyLong_AsLong(PyTuple_GET_ITEM(hit, static_cast<Py_ssize_t>(i)));
                }
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                return;
            }
        }
    }
    // 名字表之外的门：Python 的 _axes（矩阵判断，按 Gate 缓存）。
    py::object ax = axes_fn(py::reinterpret_borrow<py::object>(ins.obj), cache);
    py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(ax.ptr(), "axes"));
    if (!fast) {
        throw py::error_already_set();
    }
    const Py_ssize_t m = PySequence_Fast_GET_SIZE(fast.ptr());
    out.resize(static_cast<std::size_t>(m));
    for (Py_ssize_t i = 0; i < m; ++i) {
        out[static_cast<std::size_t>(i)] = PyLong_AsLong(PySequence_Fast_GET_ITEM(fast.ptr(), i));
    }
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
}

}  // namespace

void BindSchedule(py::module_& m) {
    m.def(
        "_schedule_info",
        [](const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
           const py::object& type_barrier, const py::dict& axes_1q, const py::dict& axes_mq,
           const py::object& axes_fn, const py::object& cache) -> py::object {
            std::vector<Instr> decoded;
            if (!Decode(instrs, instr_cls, type_gate, type_barrier, decoded)) {
                return py::none();
            }
            long long top = -1;  // max(q for ins for q in ins.qubits, default=-1)
            for (const Instr& ins : decoded) {
                for (long long q : ins.qubits) {
                    top = std::max(top, q);
                }
            }
            const long long width = top + 1;
            std::vector<long> offsets{0};
            std::vector<long> res;
            std::vector<long> axes;
            std::vector<bool> two;
            std::vector<bool> counted;
            std::vector<long> ax;
            offsets.reserve(decoded.size() + 1);
            for (const Instr& ins : decoded) {
                // _instruction_resources
                const std::size_t begin = res.size();
                auto contains = [&](long v) {
                    for (std::size_t k = begin; k < res.size(); ++k) {
                        if (res[k] == v) {
                            return true;
                        }
                    }
                    return false;
                };
                if (ins.is_barrier && ins.qubits.empty()) {
                    for (long long q = 0; q < width; ++q) {
                        if (!contains(static_cast<long>(q))) {
                            res.push_back(static_cast<long>(q));
                            axes.push_back(kNone);
                        }
                    }
                } else {
                    Axes(ins, axes_1q, axes_mq, axes_fn, cache, ax);
                    if (ax.size() != ins.qubits.size()) {
                        throw py::value_error("zip() argument 2 is shorter or longer than argument 1");
                    }
                    for (std::size_t k = 0; k < ins.qubits.size(); ++k) {
                        const auto q = static_cast<long>(ins.qubits[k]);
                        if (!contains(q)) {
                            res.push_back(q);
                            axes.push_back(ax[k]);
                        }
                    }
                }
                for (long long c : ins.clbits) {
                    const auto r = static_cast<long>(width + c);
                    if (!contains(r)) {
                        res.push_back(r);
                        axes.push_back(kNone);
                    }
                }
                offsets.push_back(static_cast<long>(res.size()));
                two.push_back(ins.is_gate && ins.qubits.size() == 2);
                counted.push_back(!ins.is_barrier && !ins.qubits.empty());
            }
            return py::make_tuple(width, IntList(offsets), IntList(res), IntList(axes), BoolList(two),
                                  BoolList(counted));
        },
        py::arg("instructions"), py::arg("instr_cls"), py::arg("type_gate"), py::arg("type_barrier"),
        py::arg("axes_1q"), py::arg("axes_mq"), py::arg("axes_fn"), py::arg("cache"),
        "Scheduling nodes of an instruction sequence (quspect.transpiler._schedule._Info), or None.");

    m.def(
        "_block_schedule_inputs",
        [](const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
           const py::object& type_barrier, const py::dict& index, const py::object& matrix_fn) -> py::object {
            std::vector<Instr> decoded;
            if (!Decode(instrs, instr_cls, type_gate, type_barrier, decoded)) {
                return py::none();
            }
            std::vector<long> qubit_offsets{0};
            std::vector<long> qubits;
            std::vector<long> kind;
            std::vector<long> matrix_offset;
            py::list mats;
            long size = 0;
            for (const Instr& ins : decoded) {
                for (long long q : ins.qubits) {
                    qubits.push_back(static_cast<long>(q));
                }
                qubit_offsets.push_back(static_cast<long>(qubits.size()));
                const std::size_t nq = ins.qubits.size();
                long k = ins.is_gate && (nq == 1 || nq == 2) && ins.clbits.empty() ? static_cast<long>(nq) : 0;
                if (k == 0 && ins.is_barrier && nq == 0) {
                    k = 3;
                }
                kind.push_back(k);
                long off = -1;
                if ((k == 1 || k == 2) && !ins.gate.is_none()) {
                    PyObject* hit = PyDict_GetItemWithError(index.ptr(), ins.gate.ptr());
                    if (hit != nullptr) {
                        off = PyLong_AsLong(hit);
                    } else if (PyErr_Occurred()) {
                        if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
                            throw py::error_already_set();
                        }
                        PyErr_Clear();  // 不可哈希的门
                    } else {
                        // matrix_fn(gate, k)：形状正确的展平矩阵，否则 None
                        py::object flat = matrix_fn(ins.gate, k);
                        if (!flat.is_none()) {
                            off = size;
                            size += static_cast<long>(py::len(flat));
                            mats.append(flat);
                        }
                        index[ins.gate] = off;
                    }
                }
                matrix_offset.push_back(off);
            }
            return py::make_tuple(IntList(qubit_offsets), IntList(qubits), IntList(kind), IntList(matrix_offset),
                                  mats);
        },
        py::arg("instructions"), py::arg("instr_cls"), py::arg("type_gate"), py::arg("type_barrier"), py::arg("index"),
        py::arg("matrix_fn"), "Inputs of _block_schedule for an instruction sequence, or None.");
}

}  // namespace quspect::native::pybind
