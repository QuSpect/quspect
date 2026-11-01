#pragma once

// 单比特门运行类原生实现（单比特门最小化 gauge_bindings.cpp、目标门集翻译 basis_bindings.cpp）
// 共用的辅助：指令属性读取、确定性的 2×2 乘法、按矩阵字节缓存的合成回调、指令的直接构造。
//
// 本头文件包含 detmath 运算，只能在 QSPECT_EXACT_FP_SRCS 中的源文件里使用。

#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "py_interned.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/transpiler/one_qubit_gauge.hpp"

namespace quspect::native::pybind::runs {

namespace py = pybind11;

using transpiler::Complex;
using transpiler::Matrix2;

// 输入不能逐项复刻：交回 Python。
struct Decline {};

struct RunNames {
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str clbits = InternedStr("clbits");
    py::str gate = InternedStr("gate");
    py::str label = InternedStr("label");
    py::str channel = InternedStr("channel");
    py::str name = InternedStr("name");
    py::str params = InternedStr("params");
    py::str num_qubits = InternedStr("num_qubits");
    py::str empty = InternedStr("");
};

inline RunNames& Names() {
    static RunNames* names = new RunNames();
    return *names;
}

inline py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

inline bool Truthy(PyObject* obj) {
    const int r = PyObject_IsTrue(obj);
    if (r < 0) {
        throw py::error_already_set();
    }
    return r != 0;
}

// 由 int 组成的 tuple（元素为普通 int，不溢出 long long），否则 Decline。
inline void ReadBits(PyObject* t, std::vector<long long>* out) {
    out->clear();
    if (!PyTuple_CheckExact(t)) {
        throw Decline{};
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(t); ++i) {
        PyObject* item = PyTuple_GET_ITEM(t, i);
        if (!PyLong_CheckExact(item)) {
            throw Decline{};
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            throw Decline{};
        }
        out->push_back(v);
    }
}

// C 连续、本机字节序 complex128 的 2×2 数组 → Matrix2；其他布局 Decline（交给 Python）。
inline Matrix2 ReadMatrix2(PyObject* obj) {
    if (!py::isinstance<py::array>(obj)) {
        throw Decline{};
    }
    auto arr = py::reinterpret_borrow<py::array>(obj);
    const py::dtype dt = arr.dtype();
    if (dt.kind() != 'c' || dt.itemsize() != 16 || dt.byteorder() != '=' || arr.ndim() != 2 || arr.shape(0) != 2
        || arr.shape(1) != 2 || (arr.flags() & py::array::c_style) == 0) {
        throw Decline{};
    }
    Matrix2 m;
    std::memcpy(m.data(), arr.data(), sizeof(m));
    return m;
}

inline py::array_t<std::complex<double>> MatrixArray(const Matrix2& m) {
    py::array_t<std::complex<double>> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
    std::memcpy(out.mutable_data(), m.data(), sizeof(m));
    return out;
}

inline Matrix2 Identity2() {
    return {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}};
}

// 2×2 复数乘法（确定性数值层：固定求和顺序，不含 FMA）。
class Mat2 {
 public:
    // ``a @ b``（与 detmath::MatMul(2, ...) 的运算逐项相同，内联展开）。
    Matrix2 Mul(const Matrix2& a, const Matrix2& b) const {
        auto entry = [&a, &b](int i, int j) {
            const Complex s0 = detmath::Add(Complex{0.0, 0.0}, detmath::Mul(a[i * 2], b[j]));
            return detmath::Add(s0, detmath::Mul(a[i * 2 + 1], b[2 + j]));
        };
        return {entry(0, 0), entry(0, 1), entry(1, 0), entry(1, 1)};
    }
    // ``a.T @ b``。
    Matrix2 MulT(const Matrix2& a, const Matrix2& b) const {
        Matrix2 out;
        detmath::MatMulTransA(2, In(a), In(b), Out(&out));
        return out;
    }
    // ``a @ np.eye(2)``。
    Matrix2 MulRealIdentity(const Matrix2& a) const {
        return Mul(a, Identity2());
    }

 private:
    static const detmath::Cx* In(const Matrix2& m) {
        return m.data();
    }
    static detmath::Cx* Out(Matrix2* m) {
        return m->data();
    }
};

// ``synthesize_one_qubit(m, basis)`` 的回调：逐位相同的矩阵只调用一次（函数是纯的，结果相同）。
// 结果为 Python list 或 None；其他类型 Decline。
class SynthCache {
 public:
    explicit SynthCache(py::object synthesize) : synthesize_(std::move(synthesize)) {}

    py::object Get(const Matrix2& m) {
        std::array<std::uint64_t, 8> key;
        static_assert(sizeof(key) == sizeof(Matrix2), "Matrix2 is four (re, im) pairs");
        std::memcpy(key.data(), m.data(), sizeof(key));
        const auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        py::object seq = synthesize_(MatrixArray(m));
        if (!seq.is_none() && !PyList_CheckExact(seq.ptr())) {
            throw Decline{};
        }
        cache_.emplace(key, seq);
        return seq;
    }

    std::size_t size() const {
        return cache_.size();
    }

 private:
    struct KeyHash {
        std::size_t operator()(const std::array<std::uint64_t, 8>& k) const {
            std::uint64_t h = 0x9E3779B97F4A7C15ULL;
            for (std::uint64_t v : k) {
                h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
            }
            return static_cast<std::size_t>(h);
        }
    };
    py::object synthesize_;
    std::unordered_map<std::array<std::uint64_t, 8>, py::object, KeyHash> cache_;
};

// 指令构造：CircuitInstruction(type=GATE, gate=gate, qubits=(q,))。普通单比特 Gate、非负 int 比特
// 时按数据类 __init__ 的字段顺序直接写入（__post_init__ 的校验在这里必然通过），否则调用构造函数
// （同样的校验与异常）。
class InstructionFactory {
 public:
    InstructionFactory(py::object instr_cls, py::object type_gate, py::object gate_cls)
        : instr_cls_(std::move(instr_cls)), type_gate_(std::move(type_gate)), gate_cls_(std::move(gate_cls)) {}

    const py::object& instr_cls() const {
        return instr_cls_;
    }

    py::object OneQubit(PyObject* gate, long long q) const {
        RunNames& n = Names();
        PyObject* qobj = PyLong_FromLongLong(q);
        if (qobj == nullptr) {
            throw py::error_already_set();
        }
        py::object qubits = py::reinterpret_steal<py::object>(PyTuple_Pack(1, qobj));
        Py_DECREF(qobj);
        if (!qubits) {
            throw py::error_already_set();
        }
        bool simple = q >= 0 && Py_TYPE(gate) == reinterpret_cast<PyTypeObject*>(gate_cls_.ptr());
        if (simple) {
            py::object nq = Attr(gate, n.num_qubits);
            simple = PyLong_CheckExact(nq.ptr()) && PyLong_AsLong(nq.ptr()) == 1;
            if (PyErr_Occurred()) {
                PyErr_Clear();
                simple = false;
            }
        }
        if (!simple) {
            py::dict kw;
            kw[n.type] = type_gate_;
            kw[n.gate] = py::handle(gate);
            kw[n.qubits] = qubits;
            return instr_cls_(**kw);
        }
        return Fields(type_gate_.ptr(), qubits.ptr(), gate, Names().empty.ptr());
    }

    // CircuitInstruction(type=GATE, gate=gate, qubits=qubits, clbits=(), label=label)：普通 Gate、
    // gate.num_qubits == len(qubits) ≤ 2、比特为互不相同的非负 int 时直接写字段，否则调用构造函数。
    py::object Gate(PyObject* gate, PyObject* qubits, PyObject* label) const {
        if (Simple(gate, qubits) && PyUnicode_CheckExact(label)) {
            return Fields(type_gate_.ptr(), qubits, gate, label);
        }
        RunNames& n = Names();
        py::dict kw;
        kw[n.type] = type_gate_;
        kw[n.gate] = py::handle(gate);
        kw[n.qubits] = py::handle(qubits);
        kw[n.clbits] = py::tuple();
        kw[n.label] = py::handle(label);
        return instr_cls_(**kw);
    }

    // 两比特门：CircuitInstruction(type=GATE, gate=gate, qubits=(a, b))（构造函数）。
    py::object TwoQubit(PyObject* gate, long long a, long long b) const {
        RunNames& n = Names();
        py::dict kw;
        kw[n.type] = type_gate_;
        kw[n.gate] = py::handle(gate);
        kw[n.qubits] = py::make_tuple(a, b);
        return instr_cls_(**kw);
    }

 private:
    bool Simple(PyObject* gate, PyObject* qubits) const {
        if (Py_TYPE(gate) != reinterpret_cast<PyTypeObject*>(gate_cls_.ptr()) || !PyTuple_CheckExact(qubits)) {
            return false;
        }
        const Py_ssize_t n = PyTuple_GET_SIZE(qubits);
        if (n < 1 || n > 2) {
            return false;
        }
        long long q[2] = {0, 0};
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* item = PyTuple_GET_ITEM(qubits, i);
            if (!PyLong_CheckExact(item)) {
                return false;
            }
            int overflow = 0;
            q[i] = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (overflow != 0 || q[i] < 0) {
                return false;
            }
        }
        if (n == 2 && q[0] == q[1]) {
            return false;
        }
        py::object nq = Attr(gate, Names().num_qubits);
        if (!PyLong_CheckExact(nq.ptr())) {
            return false;
        }
        const long v = PyLong_AsLong(nq.ptr());
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return v == n;
    }

    py::object Fields(PyObject* type, PyObject* qubits, PyObject* gate, PyObject* label) const {
        RunNames& n = Names();
        auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls_.ptr());
        py::object empty = py::reinterpret_steal<py::object>(PyTuple_New(0));
        if (!empty) {
            throw py::error_already_set();
        }
        PyObject* raw = PyBaseObject_Type.tp_new(cls, empty.ptr(), nullptr);
        if (raw == nullptr) {
            throw py::error_already_set();
        }
        py::object obj = py::reinterpret_steal<py::object>(raw);
        auto set = [&](const py::str& name, PyObject* value) {
            if (PyObject_GenericSetAttr(obj.ptr(), name.ptr(), value) != 0) {
                throw py::error_already_set();
            }
        };
        set(n.type, type);
        set(n.qubits, qubits);
        set(n.clbits, empty.ptr());
        set(n.gate, gate);
        set(n.channel, Py_None);
        set(n.label, label);
        return obj;
    }

    py::object instr_cls_;
    py::object type_gate_;
    py::object gate_cls_;
};

}  // namespace quspect::native::pybind::runs
