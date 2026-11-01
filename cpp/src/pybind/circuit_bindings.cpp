// 线路级的批量辅助（Python 端逐条循环的等价实现）：
// * _instruction_bit_counts：QuantumCircuit.num_qubits / num_clbits（各比特元组的最大值 + 1）；
// * _instruction_stats：transpile 的 (两比特门数, 深度, 两比特门深度, 门数)；
// * _remap_instruction_qubits：QuantumCircuit.compose 按比特映射逐条重建指令；
// * _synth_op_instructions：two_qubit.ops_to_instructions（合成结果 -> 指令）；
// * _unsupported_gate_indices：分解时需要逐条处理的指令（门名不在目标门集中的门）的下标；
// * _flatten_slots：块重综合的槽位表展开（None 跳过、list 展开、其他原样）。
// 只处理 CircuitInstruction 本身、比特为 int 组成的 tuple 的输入；其他情形返回 None，由 Python 端
// 按原实现处理（因此异常类型与信息都来自原实现）。直接构造的指令按数据类 __init__ 的字段顺序
// 逐个写入，只在 __post_init__ 的校验必然通过时这样做。
#include "circuit_bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>

#include "py_interned.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

struct CircuitNames {
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str clbits = InternedStr("clbits");
    py::str gate = InternedStr("gate");
    py::str channel = InternedStr("channel");
    py::str label = InternedStr("label");
    py::str name = InternedStr("name");
    py::str num_qubits = InternedStr("num_qubits");
    py::str params = InternedStr("params");
};

CircuitNames& Names() {
    static CircuitNames* names = new CircuitNames();
    return *names;
}

// 比特编号的上限（超过时交回 Python；深度统计按编号下标数组）。
constexpr long long kMaxBit = 1LL << 24;

py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

// 由 [0, kMaxBit) 内的 int 组成的 tuple -> out；否则返回 false。
bool ReadBits(PyObject* tuple, std::vector<long long>& out) {
    out.clear();
    if (!PyTuple_CheckExact(tuple)) {
        return false;
    }
    const Py_ssize_t n = PyTuple_GET_SIZE(tuple);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0 || v < 0 || v >= kMaxBit) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}

// 按数据类 __init__ 的字段顺序（type, qubits, clbits, gate, channel, label）直接写入。
py::object NewInstruction(PyTypeObject* cls, PyObject* type, PyObject* qubits, PyObject* clbits, PyObject* gate,
                          PyObject* channel, PyObject* label) {
    static PyObject* empty_args = PyTuple_New(0);  // 进程内常驻
    PyObject* raw = PyBaseObject_Type.tp_new(cls, empty_args, nullptr);
    if (raw == nullptr) {
        throw py::error_already_set();
    }
    py::object obj = py::reinterpret_steal<py::object>(raw);
    CircuitNames& n = Names();
    if (PyObject_GenericSetAttr(raw, n.type.ptr(), type) != 0 || PyObject_GenericSetAttr(raw, n.qubits.ptr(), qubits) != 0
        || PyObject_GenericSetAttr(raw, n.clbits.ptr(), clbits) != 0
        || PyObject_GenericSetAttr(raw, n.gate.ptr(), gate) != 0
        || PyObject_GenericSetAttr(raw, n.channel.ptr(), channel) != 0
        || PyObject_GenericSetAttr(raw, n.label.ptr(), label) != 0) {
        throw py::error_already_set();
    }
    return obj;
}

// num_qubits / num_clbits：max_q = -1；对每条指令 if qubits: max_q = max(max_q, max(qubits))。
py::object BitCounts(const py::list& instrs, const py::object& instr_cls) {
    auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls.ptr());
    CircuitNames& n = Names();
    long long max_q = -1;
    long long max_c = -1;
    std::vector<long long> bits;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(instrs.ptr()); ++i) {
        PyObject* ins = PyList_GET_ITEM(instrs.ptr(), i);
        if (Py_TYPE(ins) != cls) {
            return py::none();
        }
        if (!ReadBits(Attr(ins, n.qubits).ptr(), bits)) {
            return py::none();
        }
        for (long long q : bits) {
            max_q = std::max(max_q, q);
        }
        if (!ReadBits(Attr(ins, n.clbits).ptr(), bits)) {
            return py::none();
        }
        for (long long c : bits) {
            max_c = std::max(max_c, c);
        }
    }
    return py::make_tuple(max_q + 1, max_c + 1);
}

// _transpile 的统计：
//   two  = sum(1 for i in c if i.is_gate and len(i.qubits) == 2)
//   d    = _depth(c, two_qubit_only=False)，d2 = _depth(c, two_qubit_only=True)
//   gates = sum(1 for i in c if i.is_gate)
py::object Stats(const py::list& instrs, const py::object& instr_cls, const py::object& type_gate,
                 const py::object& type_barrier) {
    auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls.ptr());
    CircuitNames& n = Names();
    std::vector<long long> level;
    std::vector<long long> level2;
    std::vector<long long> qs;
    long long two = 0;
    long long gates = 0;
    long long deepest = 0;
    long long deepest2 = 0;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(instrs.ptr()); ++i) {
        PyObject* ins = PyList_GET_ITEM(instrs.ptr(), i);
        if (Py_TYPE(ins) != cls) {
            return py::none();
        }
        py::object type = Attr(ins, n.type);
        if (!ReadBits(Attr(ins, n.qubits).ptr(), qs)) {
            return py::none();
        }
        const bool is_gate = type.ptr() == type_gate.ptr();
        const bool is_two = is_gate && qs.size() == 2;
        gates += is_gate ? 1 : 0;
        two += is_two ? 1 : 0;
        if (type.ptr() == type_barrier.ptr() || qs.empty()) {
            continue;
        }
        long long top = 0;
        for (long long q : qs) {
            if (static_cast<std::size_t>(q) >= level.size()) {
                level.resize(static_cast<std::size_t>(q) + 1, 0);
                level2.resize(static_cast<std::size_t>(q) + 1, 0);
            }
            top = std::max(top, level[static_cast<std::size_t>(q)]);
        }
        const long long lv = top + 1;
        for (long long q : qs) {
            level[static_cast<std::size_t>(q)] = lv;
        }
        deepest = std::max(deepest, lv);
        if (is_two) {
            const long long lv2 =
                1 + std::max(level2[static_cast<std::size_t>(qs[0])], level2[static_cast<std::size_t>(qs[1])]);
            level2[static_cast<std::size_t>(qs[0])] = lv2;
            level2[static_cast<std::size_t>(qs[1])] = lv2;
            deepest2 = std::max(deepest2, lv2);
        }
    }
    return py::make_tuple(two, deepest, deepest2, gates);
}

// compose 的逐条重建：CircuitInstruction(type, gate, channel, qubits=tuple(m[q] for q in qubits),
// clbits, label)，m = dict(enumerate(mapping))。映射由互不相同的非负 int 组成时，原指令通过的
// 校验（只涉及比特非负、互不相同）在映射后同样通过，于是直接构造。
py::object Remap(const py::list& instrs, const py::object& instr_cls, const py::sequence& mapping) {
    auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls.ptr());
    CircuitNames& n = Names();
    std::vector<PyObject*> map;  // 借用：mapping 的元素
    py::object fast_map = py::reinterpret_steal<py::object>(PySequence_Fast(mapping.ptr(), "mapping"));
    if (!fast_map) {
        throw py::error_already_set();
    }
    std::vector<long long> values;
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast_map.ptr()); ++i) {
        PyObject* v = PySequence_Fast_GET_ITEM(fast_map.ptr(), i);
        if (!PyLong_CheckExact(v)) {
            return py::none();
        }
        int overflow = 0;
        const long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
        if (overflow != 0 || x < 0) {
            return py::none();
        }
        map.push_back(v);
        values.push_back(x);
    }
    std::sort(values.begin(), values.end());
    if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
        return py::none();
    }
    const Py_ssize_t count = PyList_GET_SIZE(instrs.ptr());
    py::list out(count);
    std::vector<long long> qs;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* ins = PyList_GET_ITEM(instrs.ptr(), i);
        if (Py_TYPE(ins) != cls) {
            return py::none();
        }
        py::object qubits = Attr(ins, n.qubits);
        if (!ReadBits(qubits.ptr(), qs)) {
            return py::none();
        }
        py::object mapped = py::reinterpret_steal<py::object>(PyTuple_New(static_cast<Py_ssize_t>(qs.size())));
        if (!mapped) {
            throw py::error_already_set();
        }
        for (std::size_t k = 0; k < qs.size(); ++k) {
            if (static_cast<std::size_t>(qs[k]) >= map.size()) {
                return py::none();  // KeyError：交给 Python
            }
            PyObject* v = map[static_cast<std::size_t>(qs[k])];
            Py_INCREF(v);
            PyTuple_SET_ITEM(mapped.ptr(), static_cast<Py_ssize_t>(k), v);
        }
        py::object ni = NewInstruction(cls, Attr(ins, n.type).ptr(), mapped.ptr(), Attr(ins, n.clbits).ptr(),
                                       Attr(ins, n.gate).ptr(), Attr(ins, n.channel).ptr(), Attr(ins, n.label).ptr());
        PyList_SET_ITEM(out.ptr(), i, ni.release().ptr());
    }
    return out;
}

// ops_to_instructions(ops, qubits)：
//   [CircuitInstruction(type=GATE, gate=(rz.with_params(*params) if name == "rz" else fixed[name]),
//                       qubits=tuple(qubits[q] for q in local), clbits=(), label="") for name, local, params in ops]
// rz 模板是普通 Gate、参数为 float 组成的非空 tuple 时直接构造 Gate(name, num_qubits, params, label)
// （__post_init__ 对这种参数不做任何改动）；比特映射后为 1 / 2 个互不相同的非负 int 且门的比特数
// 相符时直接构造指令。其他情形返回 None。
py::object SynthOps(const py::sequence& ops, const py::sequence& qubits, const py::object& rz,
                    const py::dict& fixed, const py::object& instr_cls, const py::object& type_gate,
                    const py::object& gate_cls) {
    auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls.ptr());
    auto* gcls = reinterpret_cast<PyTypeObject*>(gate_cls.ptr());
    CircuitNames& n = Names();
    if (Py_TYPE(rz.ptr()) != gcls) {
        return py::none();
    }
    py::object rz_name = Attr(rz.ptr(), n.name);
    py::object rz_nq = Attr(rz.ptr(), n.num_qubits);
    py::object rz_label = Attr(rz.ptr(), n.label);
    py::object fast_q = py::reinterpret_steal<py::object>(PySequence_Fast(qubits.ptr(), "qubits"));
    if (!fast_q) {
        throw py::error_already_set();
    }
    const Py_ssize_t nq_map = PySequence_Fast_GET_SIZE(fast_q.ptr());
    py::object fast_ops = py::reinterpret_steal<py::object>(PySequence_Fast(ops.ptr(), "ops"));
    if (!fast_ops) {
        throw py::error_already_set();
    }
    static PyObject* empty_tuple = PyTuple_New(0);  // 进程内常驻
    static PyObject* empty_str = PyUnicode_FromString("");
    static PyObject* rz_str = PyUnicode_InternFromString("rz");
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast_ops.ptr());
    py::list out(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* op = PySequence_Fast_GET_ITEM(fast_ops.ptr(), i);
        if (!PyTuple_CheckExact(op) || PyTuple_GET_SIZE(op) != 3) {
            return py::none();
        }
        PyObject* name = PyTuple_GET_ITEM(op, 0);
        PyObject* local = PyTuple_GET_ITEM(op, 1);
        PyObject* params = PyTuple_GET_ITEM(op, 2);
        if (!PyUnicode_CheckExact(name) || !PyTuple_CheckExact(local)) {
            return py::none();
        }
        py::object gate;
        const int is_rz = PyUnicode_Compare(name, rz_str);
        if (is_rz == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (is_rz == 0) {
            if (!PyTuple_CheckExact(params) || PyTuple_GET_SIZE(params) == 0) {
                return py::none();
            }
            for (Py_ssize_t k = 0; k < PyTuple_GET_SIZE(params); ++k) {
                if (!PyFloat_CheckExact(PyTuple_GET_ITEM(params, k))) {
                    return py::none();
                }
            }
            PyObject* raw = PyBaseObject_Type.tp_new(gcls, empty_tuple, nullptr);
            if (raw == nullptr) {
                throw py::error_already_set();
            }
            gate = py::reinterpret_steal<py::object>(raw);
            if (PyObject_GenericSetAttr(raw, n.name.ptr(), rz_name.ptr()) != 0
                || PyObject_GenericSetAttr(raw, n.num_qubits.ptr(), rz_nq.ptr()) != 0
                || PyObject_GenericSetAttr(raw, n.params.ptr(), params) != 0
                || PyObject_GenericSetAttr(raw, n.label.ptr(), rz_label.ptr()) != 0) {
                throw py::error_already_set();
            }
        } else {
            PyObject* g = PyDict_GetItemWithError(fixed.ptr(), name);
            if (g == nullptr) {
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                return py::none();  // KeyError：交给 Python
            }
            gate = py::reinterpret_borrow<py::object>(g);
        }
        const Py_ssize_t width = PyTuple_GET_SIZE(local);
        if (width < 1 || width > 2 || Py_TYPE(gate.ptr()) != gcls) {
            return py::none();
        }
        py::object gnq = Attr(gate.ptr(), n.num_qubits);
        if (!PyLong_CheckExact(gnq.ptr()) || PyLong_AsLongLong(gnq.ptr()) != width) {
            PyErr_Clear();
            return py::none();
        }
        py::object mapped = py::reinterpret_steal<py::object>(PyTuple_New(width));
        if (!mapped) {
            throw py::error_already_set();
        }
        long long seen[2] = {-1, -1};
        for (Py_ssize_t k = 0; k < width; ++k) {
            PyObject* lq = PyTuple_GET_ITEM(local, k);
            if (!PyLong_CheckExact(lq)) {
                return py::none();
            }
            const long long idx = PyLong_AsLongLong(lq);
            if (idx < 0 || idx >= nq_map) {
                PyErr_Clear();
                return py::none();
            }
            PyObject* q = PySequence_Fast_GET_ITEM(fast_q.ptr(), idx);
            if (!PyLong_CheckExact(q)) {
                return py::none();
            }
            int overflow = 0;
            const long long qv = PyLong_AsLongLongAndOverflow(q, &overflow);
            if (overflow != 0 || qv < 0) {
                return py::none();
            }
            seen[k] = qv;
            Py_INCREF(q);
            PyTuple_SET_ITEM(mapped.ptr(), k, q);
        }
        if (width == 2 && seen[0] == seen[1]) {
            return py::none();
        }
        py::object ins = NewInstruction(cls, type_gate.ptr(), mapped.ptr(), empty_tuple, gate.ptr(), Py_None, empty_str);
        PyList_SET_ITEM(out.ptr(), i, ins.release().ptr());
    }
    return out;
}

// 需要逐条处理的指令下标：is_gate and gate is not None and gate.name not in supported 的门，以及
// 无法在这里判断的指令（不是 CircuitInstruction 本身、type 不是 InstructionType 的成员、门名
// 不是 str）。其余指令（非门、无门、门名在 supported 中）原样保留，调用方按区间整段复制。
// supported 不是 set / frozenset 时返回 None。
py::object UnsupportedIndices(const py::list& instrs, const py::object& instr_cls, const py::object& type_gate,
                              const py::tuple& other_types, const py::object& supported) {
    if (!PyAnySet_Check(supported.ptr())) {
        return py::none();
    }
    auto* cls = reinterpret_cast<PyTypeObject*>(instr_cls.ptr());
    CircuitNames& n = Names();
    py::list out;
    auto take = [&](Py_ssize_t i) {
        py::object idx = py::reinterpret_steal<py::object>(PyLong_FromSsize_t(i));
        if (!idx || PyList_Append(out.ptr(), idx.ptr()) != 0) {
            throw py::error_already_set();
        }
    };
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(instrs.ptr()); ++i) {
        PyObject* ins = PyList_GET_ITEM(instrs.ptr(), i);
        if (Py_TYPE(ins) != cls) {
            take(i);
            continue;
        }
        py::object type = Attr(ins, n.type);
        if (type.ptr() != type_gate.ptr()) {
            bool known = false;
            for (py::handle t : other_types) {
                known = known || t.ptr() == type.ptr();
            }
            if (!known) {
                take(i);
            }
            continue;
        }
        py::object gate = Attr(ins, n.gate);
        if (gate.is_none()) {
            continue;
        }
        py::object name = Attr(gate.ptr(), n.name);
        if (!PyUnicode_CheckExact(name.ptr())) {
            take(i);
            continue;
        }
        const int has = PySet_Contains(supported.ptr(), name.ptr());
        if (has < 0) {
            throw py::error_already_set();
        }
        if (has == 0) {
            take(i);
        }
    }
    return out;
}

// for slot in slots: None 跳过；isinstance(slot, list) 时 extend；否则 append。
py::list FlattenSlots(const py::list& slots) {
    py::list out;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(slots.ptr()); ++i) {
        PyObject* slot = PyList_GET_ITEM(slots.ptr(), i);
        if (slot == Py_None) {
            continue;
        }
        if (PyList_Check(slot)) {
            // list.extend(list 子类) 同样按迭代协议取元素
            py::object items = py::reinterpret_steal<py::object>(PySequence_List(slot));
            if (!items) {
                throw py::error_already_set();
            }
            for (Py_ssize_t k = 0; k < PyList_GET_SIZE(items.ptr()); ++k) {
                if (PyList_Append(out.ptr(), PyList_GET_ITEM(items.ptr(), k)) != 0) {
                    throw py::error_already_set();
                }
            }
            continue;
        }
        if (PyList_Append(out.ptr(), slot) != 0) {
            throw py::error_already_set();
        }
    }
    return out;
}

}  // namespace

void BindCircuitHelpers(py::module_& m) {
    m.def("_flatten_slots", &FlattenSlots, py::arg("slots"),
          "Flatten a resynthesis slot list: skip None, extend lists, append anything else.");
    m.def("_unsupported_gate_indices", &UnsupportedIndices, py::arg("instrs"), py::arg("instr_cls"),
          py::arg("type_gate"), py::arg("other_types"), py::arg("supported"),
          "Indices of the instructions a decomposition pass must look at one by one, or None.");
    m.def("_instruction_bit_counts", &BitCounts, py::arg("instrs"), py::arg("instr_cls"),
          "(num_qubits, num_clbits) of a QuantumCircuit instruction list, or None.");
    m.def("_instruction_stats", &Stats, py::arg("instrs"), py::arg("instr_cls"), py::arg("type_gate"),
          py::arg("type_barrier"),
          "(two-qubit gates, depth, two-qubit depth, gates) of an instruction list, or None.");
    m.def("_remap_instruction_qubits", &Remap, py::arg("instrs"), py::arg("instr_cls"), py::arg("mapping"),
          "Instructions with qubit q renamed to mapping[q] (QuantumCircuit.compose), or None.");
    m.def("_synth_op_instructions", &SynthOps, py::arg("ops"), py::arg("qubits"), py::arg("rz"), py::arg("fixed"),
          py::arg("instr_cls"), py::arg("type_gate"), py::arg("gate_cls"),
          "two_qubit.ops_to_instructions in one call, or None.");
}

}  // namespace quspect::native::pybind
