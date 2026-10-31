// 线路 IR（ir::Circuit）与 OpenQASM 2.0 快速路径的 Python 绑定。
//
// Python 的 Gate / CircuitInstruction 仍是冻结数据类（用户代码依赖 dataclasses.replace 等），
// 这里负责在它们与 C++ 的紧凑表示之间批量转换：
//   * _Circuit.from_instructions(seq)  Python 指令序列 -> C++ 线路（保留原对象，回转时原样返回）；
//   * _Circuit.to_instructions()       C++ 线路 -> Python 指令列表（直接按字段顺序构造对象，
//                                      不经过 __init__，结果与正常构造完全相同）；
//   * _qasm2_parse(source, library)    QASM 文本 -> C++ 线路；参考实现会报错或本解析器不处理
//                                      的输入返回 None，由 Python 端回退到参考实现。
#include "ir_bindings.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/pybind11.h>

#include "quspect_native/ir/circuit.hpp"
#include "quspect_native/ir/operation.hpp"
#include "quspect_native/ir/qasm2_parser.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using ir::Circuit;
using ir::InstRef;
using ir::Operation;
using ir::OpKind;

// 由 quspect.circuit._native 在首次使用时登记的 Python 类型与门对象。
struct IrRegistry {
    py::object gate_cls;
    py::object instr_cls;
    py::object type_gate;
    py::object type_measure;
    py::object type_barrier;
    py::object type_channel;
    // 每种标准门的单例（固定门）或模板（参数门，提供名字与标签）；没有时为空。
    std::vector<py::object> gates = std::vector<py::object>(ir::kNumOpKinds);
    py::object empty_str;
    py::object empty_tuple;
    bool ready = false;
};

IrRegistry& Reg() {
    static IrRegistry* reg = new IrRegistry();  // 进程结束时不析构 Python 对象
    return *reg;
}

struct IrNames {
    py::str type{"type"};
    py::str qubits{"qubits"};
    py::str clbits{"clbits"};
    py::str gate{"gate"};
    py::str channel{"channel"};
    py::str label{"label"};
    py::str name{"name"};
    py::str num_qubits{"num_qubits"};
    py::str params{"params"};
};

IrNames& Names() {
    static IrNames* names = new IrNames();
    return *names;
}

void RequireRegistry() {
    if (!Reg().ready) {
        throw std::runtime_error("quspect IR types are not registered");
    }
}

py::object GetAttr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

// 不经过 __init__ / 冻结的 __setattr__，按字段顺序写入（与数据类 __init__ 的效果相同）。
void SetField(PyObject* obj, const py::str& name, PyObject* value) {
    if (PyObject_GenericSetAttr(obj, name.ptr(), value) != 0) {
        throw py::error_already_set();
    }
}

py::object NewInstance(const py::object& cls) {
    auto* type = reinterpret_cast<PyTypeObject*>(cls.ptr());
    PyObject* obj = PyBaseObject_Type.tp_new(type, Reg().empty_tuple.ptr(), nullptr);
    if (obj == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(obj);
}

py::object BitTuple(const std::uint32_t* bits, std::size_t n) {
    if (n == 0) {
        return Reg().empty_tuple;
    }
    PyObject* t = PyTuple_New(static_cast<Py_ssize_t>(n));
    if (t == nullptr) {
        throw py::error_already_set();
    }
    for (std::size_t i = 0; i < n; ++i) {
        PyTuple_SET_ITEM(t, static_cast<Py_ssize_t>(i), PyLong_FromUnsignedLong(bits[i]));
    }
    return py::reinterpret_steal<py::object>(t);
}

void ReadBits(PyObject* seq, std::vector<std::uint32_t>& out) {
    out.clear();
    py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(seq, "bits must be a sequence"));
    if (!fast) {
        throw py::error_already_set();
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast.ptr());
    PyObject** items = PySequence_Fast_ITEMS(fast.ptr());
    for (Py_ssize_t i = 0; i < n; ++i) {
        const long long v = PyLong_AsLongLong(items[i]);
        if (v == -1 && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (v < 0 || v > 0x7fffffffLL) {
            throw py::value_error("bit index out of range for the native circuit");
        }
        out.push_back(static_cast<std::uint32_t>(v));
    }
}

// 线路 + 每条指令对应的原 Python 对象（来自 Python 的指令；QASM 解析出的指令没有）。
class PyCircuit {
public:
    Circuit circuit;
    std::vector<py::object> hosts;

    static PyCircuit from_instructions(const py::iterable& instructions) {
        RequireRegistry();
        IrRegistry& reg = Reg();
        IrNames& n = Names();
        PyCircuit out;
        std::vector<std::uint32_t> qs;
        std::vector<std::uint32_t> cs;
        for (py::handle h : instructions) {
            PyObject* instr = h.ptr();
            py::object type = GetAttr(instr, n.type);
            py::object qubits = GetAttr(instr, n.qubits);
            py::object clbits = GetAttr(instr, n.clbits);
            ReadBits(qubits.ptr(), qs);
            ReadBits(clbits.ptr(), cs);
            Operation op;
            if (type.is(reg.type_gate)) {
                py::object gate = GetAttr(instr, n.gate);
                py::object name = GetAttr(gate.ptr(), n.name);
                py::object params = GetAttr(gate.ptr(), n.params);
                Py_ssize_t len = 0;
                const char* s = PyUnicode_AsUTF8AndSize(name.ptr(), &len);
                if (s == nullptr) {
                    throw py::error_already_set();
                }
                const std::string_view gname(s, static_cast<std::size_t>(len));
                if (auto kind = ir::GateKindFromName(gname)) {
                    op.kind = *kind;
                } else {
                    op.kind = OpKind::kOpaque;
                    op.ext = out.circuit.intern_name(gname);
                }
                py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(params.ptr(), "params"));
                if (!fast) {
                    throw py::error_already_set();
                }
                const Py_ssize_t np = PySequence_Fast_GET_SIZE(fast.ptr());
                PyObject** items = PySequence_Fast_ITEMS(fast.ptr());
                bool numeric = np <= static_cast<Py_ssize_t>(Operation::kMaxInlineParams);
                for (Py_ssize_t i = 0; numeric && i < np; ++i) {
                    numeric = PyFloat_CheckExact(items[i]) != 0;
                }
                if (numeric) {
                    op.num_params = static_cast<std::uint8_t>(np);
                    for (Py_ssize_t i = 0; i < np; ++i) {
                        op.params[static_cast<std::size_t>(i)] = PyFloat_AS_DOUBLE(items[i]);
                    }
                } else {
                    op.flags |= Operation::kSymbolic;  // 参数只保存在原 Python 对象中
                }
            } else if (type.is(reg.type_measure)) {
                op.kind = OpKind::kMeasure;
            } else if (type.is(reg.type_barrier)) {
                op.kind = OpKind::kBarrier;
            } else {
                op.kind = OpKind::kChannel;
            }
            const InstRef ref = out.circuit.append(op, qs.data(), qs.size(), cs.data(), cs.size());
            out.circuit.at_mutable(ref).host = static_cast<std::uint32_t>(out.hosts.size());
            out.hosts.push_back(py::reinterpret_borrow<py::object>(h));
        }
        return out;
    }

    py::list to_instructions() const {
        RequireRegistry();
        py::list out(circuit.size());
        for (InstRef i = 0; i < circuit.size(); ++i) {
            out[i] = instruction_object(i);
        }
        return out;
    }

    py::object instruction_object(InstRef i) const {
        const ir::Instruction& inst = circuit[i];
        if (inst.host != Operation::kNoExt) {
            return hosts[inst.host];
        }
        IrRegistry& reg = Reg();
        IrNames& n = Names();
        py::object obj = NewInstance(reg.instr_cls);
        py::object qubits = BitTuple(inst.qubits.data(), inst.qubits.size());
        py::object clbits = BitTuple(inst.clbits.data(), inst.clbits.size());
        py::object gate = py::none();
        py::object type;
        if (inst.op.is_gate()) {
            type = reg.type_gate;
            gate = gate_object(inst.op);
        } else if (inst.op.kind == OpKind::kMeasure) {
            type = reg.type_measure;
        } else if (inst.op.kind == OpKind::kBarrier) {
            type = reg.type_barrier;
        } else {
            throw std::runtime_error("channel instructions need their Python object");
        }
        SetField(obj.ptr(), n.type, type.ptr());
        SetField(obj.ptr(), n.qubits, qubits.ptr());
        SetField(obj.ptr(), n.clbits, clbits.ptr());
        SetField(obj.ptr(), n.gate, gate.ptr());
        SetField(obj.ptr(), n.channel, Py_None);
        SetField(obj.ptr(), n.label, reg.empty_str.ptr());
        return obj;
    }

    py::object gate_object(const Operation& op) const {
        IrRegistry& reg = Reg();
        if (op.kind == OpKind::kOpaque || (op.flags & Operation::kSymbolic) != 0) {
            throw std::runtime_error("gate needs its Python object");
        }
        const py::object& known = reg.gates[static_cast<std::size_t>(op.kind)];
        if (!known) {
            throw std::runtime_error("gate kind is not registered");
        }
        if (ir::GetOpInfo(op.kind).num_params == 0) {
            return known;  // 固定门：共享的模块级单例
        }
        // 参数门：等价于 template.with_params(*params)。
        IrNames& n = Names();
        py::object gate = NewInstance(reg.gate_cls);
        py::object name = GetAttr(known.ptr(), n.name);
        py::object num_qubits = GetAttr(known.ptr(), n.num_qubits);
        py::object label = GetAttr(known.ptr(), n.label);
        PyObject* params = PyTuple_New(op.num_params);
        if (params == nullptr) {
            throw py::error_already_set();
        }
        py::object params_obj = py::reinterpret_steal<py::object>(params);
        for (std::size_t k = 0; k < op.num_params; ++k) {
            PyObject* f = PyFloat_FromDouble(op.params[k]);
            if (f == nullptr) {
                throw py::error_already_set();
            }
            PyTuple_SET_ITEM(params, static_cast<Py_ssize_t>(k), f);
        }
        SetField(gate.ptr(), n.name, name.ptr());
        SetField(gate.ptr(), n.num_qubits, num_qubits.ptr());
        SetField(gate.ptr(), n.params, params);
        SetField(gate.ptr(), n.label, label.ptr());
        return gate;
    }
};

double PythonFloat(const char* text, std::size_t length) {
    const std::string s(text, length);
    char* end = nullptr;
    const double v = PyOS_string_to_double(s.c_str(), &end, nullptr);
    if ((v == -1.0 && PyErr_Occurred()) || end != s.c_str() + s.size()) {
        PyErr_Clear();
        throw std::invalid_argument("bad float literal");
    }
    return v;
}

// 编译好的门库（按库文本缓存）。
struct LibraryCache {
    std::mutex mutex;
    std::string text;
    std::shared_ptr<const ir::Qasm2Library> lib;
};

std::shared_ptr<const ir::Qasm2Library> Library(const std::string& text, const ir::Qasm2Options& options) {
    static LibraryCache* cache = new LibraryCache();
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (!cache->lib || cache->text != text) {
        cache->lib = ir::CompileQasm2Library(text, options);
        cache->text = text;
    }
    return cache->lib;
}

py::list IntList(const std::vector<std::uint32_t>& v) {
    py::list out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i] = v[i];
    }
    return out;
}

}  // namespace

void BindIr(py::module_& m) {
    m.def(
        "_register_ir_types",
        [](py::object gate_cls, py::object instr_cls, py::object type_gate, py::object type_measure,
           py::object type_barrier, py::object type_channel, const py::dict& gates) {
            IrRegistry& reg = Reg();
            reg.gate_cls = std::move(gate_cls);
            reg.instr_cls = std::move(instr_cls);
            reg.type_gate = std::move(type_gate);
            reg.type_measure = std::move(type_measure);
            reg.type_barrier = std::move(type_barrier);
            reg.type_channel = std::move(type_channel);
            for (auto item : gates) {
                const std::string name = py::str(item.first);
                if (auto kind = ir::GateKindFromName(name)) {
                    reg.gates[static_cast<std::size_t>(*kind)] = py::reinterpret_borrow<py::object>(item.second);
                }
            }
            reg.empty_str = py::str("");
            reg.empty_tuple = py::tuple();
            reg.ready = true;
        },
        "Register the Python gate / instruction types used by the native circuit IR.");

    py::class_<PyCircuit>(m, "_Circuit", py::module_local(),
                          "Contiguous C++ circuit IR (instructions with an implicit DAG).")
        .def_static("from_instructions", &PyCircuit::from_instructions, py::arg("instructions"),
                    "Build a native circuit from CircuitInstruction objects (kept for the round trip).")
        .def("to_instructions", &PyCircuit::to_instructions, "Return the CircuitInstruction objects as a list.")
        .def("__len__", [](const PyCircuit& c) { return c.circuit.size(); })
        .def_property_readonly("num_qubits", [](const PyCircuit& c) { return c.circuit.num_qubits(); })
        .def_property_readonly("num_clbits", [](const PyCircuit& c) { return c.circuit.num_clbits(); })
        .def(
            "gate_name",
            [](const PyCircuit& c, InstRef i) {
                if (i >= c.circuit.size()) {
                    throw py::index_error("instruction index out of range");
                }
                return std::string(c.circuit.gate_name(c.circuit[i]));
            },
            py::arg("index"))
        .def(
            "predecessors",
            [](const PyCircuit& c, InstRef i) {
                if (i >= c.circuit.size()) {
                    throw py::index_error("instruction index out of range");
                }
                std::vector<std::uint32_t> out;
                c.circuit.foreach_predecessor(i, [&](InstRef p) { out.push_back(p); });
                return IntList(out);
            },
            py::arg("index"), "Direct predecessors of an instruction (implicit DAG).")
        .def(
            "successors",
            [](const PyCircuit& c, InstRef i) {
                if (i >= c.circuit.size()) {
                    throw py::index_error("instruction index out of range");
                }
                std::vector<std::uint32_t> out;
                c.circuit.foreach_successor(i, [&](InstRef s) { out.push_back(s); });
                return IntList(out);
            },
            py::arg("index"), "Direct successors of an instruction (implicit DAG).")
        .def(
            "qubit_chain", [](const PyCircuit& c, std::uint32_t q) { return IntList(c.circuit.qubit_chain(q)); },
            py::arg("qubit"), "Indices of the instructions acting on a qubit, in program order.")
        .def(
            "depth",
            [](const PyCircuit& c) {
                std::uint32_t depth = 0;
                c.circuit.asap_layers(&depth);
                return depth;
            },
            "Number of ASAP layers of the dependency DAG.");

    m.def(
        "_qasm2_parse",
        [](const py::str& source, const std::string& library) -> py::object {
            RequireRegistry();
            ir::Qasm2Options options;
            options.parse_real = &PythonFloat;
            Py_ssize_t len = 0;
            const char* s = PyUnicode_AsUTF8AndSize(source.ptr(), &len);
            if (s == nullptr) {
                throw py::error_already_set();
            }
            std::shared_ptr<const ir::Qasm2Library> lib;
            try {
                lib = Library(library, options);
            } catch (const std::invalid_argument&) {
                return py::none();
            }
            PyCircuit out;
            bool ok = false;
            try {
                ok = ir::ParseQasm2(std::string_view(s, static_cast<std::size_t>(len)), *lib, out.circuit, nullptr,
                                    options);
            } catch (const std::invalid_argument&) {
                ok = false;
            }
            if (!ok) {
                return py::none();
            }
            return py::cast(std::move(out));
        },
        py::arg("source"), py::arg("library"),
        "Parse OpenQASM 2.0 into a native circuit; None when the reference importer must handle the input.");
}

}  // namespace quspect::native::pybind
