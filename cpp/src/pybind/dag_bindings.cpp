// DAGCircuit 的 C++ 图存储内核（quspect.dag.dag_circuit.DAGCircuit 的基类 _DagCore）。
//
// Python 端的 DAGCircuit 继承这里绑定的 _DagCore：图结构（邻接表、拓扑序、增量拓扑编号、
// 事务日志、沿线遍历）全部在 ir::DagStore 中完成。节点对象（DAGNode 数据类）按需创建：内核
// 只记下节点的种类与指令，Python 端第一次取用节点时才创建对象并缓存（见 dag_core.hpp）；
// DAGCircuit._nodes 是按 dict 语义工作的映射视图（get_node 直接绑定它的 __getitem__）。线标签在边界上转换：
// 规范标签 "q<i>" / "c<i>" 直接解析成编号，其他标签登记到表中。
#include "dag_bindings.hpp"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "quspect_native/ir/dag_store.hpp"
#include "quspect_native/ir/id_map.hpp"

#include "dag_core.hpp"
#include "dag_pass_bindings.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

using dag_core::DagRegistry;
using dag_core::PyDagCore;
using dag_core::Registry;
using ir::NodeId;

namespace dag_core {

namespace {

// DAGCircuit._nodes：node_id -> DAGNode 的只读映射视图。
struct NodeTableObject {
    PyObject_HEAD
    PyDagCore* core;  // 内核析构时置空
};

PyTypeObject*& NodeTableType() {
    static PyTypeObject* type = nullptr;
    return type;
}

// C++ 异常 -> Python 异常（在 C API 入口处调用）。
void SetPythonError() {
    try {
        throw;
    } catch (py::error_already_set& e) {
        e.restore();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown error in the DAG node table");
    }
}

void SetKeyError(PyObject* key) {
    PyObject* args = PyTuple_Pack(1, key);
    if (args != nullptr) {
        PyErr_SetObject(PyExc_KeyError, args);
        Py_DECREF(args);
    }
}

// 按 dict 的判等规则把键解析成节点编号：int（含 bool 与子类）、值为整数的 float、可 __index__
// 且与该整数相等的对象。返回 0：不可能是节点编号；1：*out 有效；-1：异常（如键不可哈希）。
int KeyToId(PyObject* key, NodeId* out) {
    if (PyLong_Check(key)) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(key, &overflow);
        if (v == -1 && PyErr_Occurred()) {
            return -1;
        }
        if (overflow != 0) {
            return 0;
        }
        *out = static_cast<NodeId>(v);
        return 1;
    }
    if (PyFloat_Check(key)) {
        const double d = PyFloat_AsDouble(key);
        if (d == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (!(d >= -9.2e18 && d <= 9.2e18) || d != static_cast<double>(static_cast<long long>(d))) {
            return 0;
        }
        *out = static_cast<NodeId>(static_cast<long long>(d));
        return 1;
    }
    if (PyObject_Hash(key) == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (!PyIndex_Check(key)) {
        return 0;
    }
    PyObject* idx = PyNumber_Index(key);
    if (idx == nullptr) {
        PyErr_Clear();
        return 0;
    }
    int result = 0;
    const Py_hash_t hk = PyObject_Hash(key);
    const Py_hash_t hi = PyObject_Hash(idx);
    if (hk != -1 && hk == hi && PyObject_RichCompareBool(idx, key, Py_EQ) == 1) {
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(idx, &overflow);
        if (overflow == 0 && !(v == -1 && PyErr_Occurred())) {
            *out = static_cast<NodeId>(v);
            result = 1;
        }
    }
    PyErr_Clear();
    Py_DECREF(idx);
    return result;
}

PyObject* TableGetItem(PyObject* self, PyObject* key) {
    PyDagCore* core = reinterpret_cast<NodeTableObject*>(self)->core;
    NodeId id = 0;
    const int r = KeyToId(key, &id);
    if (r < 0) {
        return nullptr;
    }
    if (r == 0 || core == nullptr) {
        SetKeyError(key);
        return nullptr;
    }
    try {
        PyObject* node = core->node_ptr(id);
        if (node == nullptr) {
            SetKeyError(key);
            return nullptr;
        }
        Py_INCREF(node);
        return node;
    } catch (...) {
        SetPythonError();
        return nullptr;
    }
}

int TableContains(PyObject* self, PyObject* key) {
    PyDagCore* core = reinterpret_cast<NodeTableObject*>(self)->core;
    NodeId id = 0;
    const int r = KeyToId(key, &id);
    if (r <= 0) {
        return r;
    }
    return core != nullptr && core->store.has_node(id) ? 1 : 0;
}

Py_ssize_t TableLength(PyObject* self) {
    PyDagCore* core = reinterpret_cast<NodeTableObject*>(self)->core;
    return core == nullptr ? 0 : static_cast<Py_ssize_t>(core->store.num_nodes());
}

PyObject* TableKeys(PyObject* self, PyObject* /*unused*/) {
    PyDagCore* core = reinterpret_cast<NodeTableObject*>(self)->core;
    if (core == nullptr) {
        return PyList_New(0);
    }
    try {
        return core->node_id_list().release().ptr();
    } catch (...) {
        SetPythonError();
        return nullptr;
    }
}

PyObject* TableIter(PyObject* self) {
    PyObject* keys = TableKeys(self, nullptr);
    if (keys == nullptr) {
        return nullptr;
    }
    PyObject* it = PyObject_GetIter(keys);
    Py_DECREF(keys);
    return it;
}

PyObject* TableValues(PyObject* self, PyObject* /*unused*/) {
    PyDagCore* core = reinterpret_cast<NodeTableObject*>(self)->core;
    if (core == nullptr) {
        return PyList_New(0);
    }
    try {
        return core->node_object_list().release().ptr();
    } catch (...) {
        SetPythonError();
        return nullptr;
    }
}

PyObject* TableItems(PyObject* self, PyObject* /*unused*/) {
    PyObject* keys = TableKeys(self, nullptr);
    if (keys == nullptr) {
        return nullptr;
    }
    PyObject* values = TableValues(self, nullptr);
    if (values == nullptr) {
        Py_DECREF(keys);
        return nullptr;
    }
    const Py_ssize_t n = PyList_GET_SIZE(keys);
    PyObject* out = PyList_New(n);
    for (Py_ssize_t i = 0; out != nullptr && i < n; ++i) {
        PyObject* pair = PyTuple_Pack(2, PyList_GET_ITEM(keys, i), PyList_GET_ITEM(values, i));
        if (pair == nullptr) {
            Py_CLEAR(out);
            break;
        }
        PyList_SET_ITEM(out, i, pair);
    }
    Py_DECREF(keys);
    Py_DECREF(values);
    return out;
}

PyObject* TableGet(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
    if (nargs < 1 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "get expected 1 or 2 arguments");
        return nullptr;
    }
    PyObject* dflt = nargs == 2 ? args[1] : Py_None;
    const int has = TableContains(self, args[0]);
    if (has < 0) {
        return nullptr;
    }
    if (has == 0) {
        Py_INCREF(dflt);
        return dflt;
    }
    return TableGetItem(self, args[0]);
}

void TableDealloc(PyObject* self) {
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

PyMethodDef kTableMethods[] = {
    {"keys", TableKeys, METH_NOARGS, "Node ids in insertion order (list)."},
    {"values", TableValues, METH_NOARGS, "Nodes in insertion order (list)."},
    {"items", TableItems, METH_NOARGS, "(node_id, node) pairs in insertion order (list)."},
    {"get", reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(TableGet)), METH_FASTCALL,
     "Return the node for node_id, or default."},
    {nullptr, nullptr, 0, nullptr},
};

PyType_Slot kTableSlots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(TableDealloc)},
    {Py_tp_iter, reinterpret_cast<void*>(TableIter)},
    {Py_tp_methods, kTableMethods},
    {Py_mp_subscript, reinterpret_cast<void*>(TableGetItem)},
    {Py_mp_length, reinterpret_cast<void*>(TableLength)},
    {Py_sq_contains, reinterpret_cast<void*>(TableContains)},
    {Py_tp_doc, const_cast<char*>("node_id -> DAGNode view of a DAGCircuit (nodes are created on first access).")},
    {0, nullptr},
};

PyType_Spec kTableSpec = {
    "quspect._DagNodeTable",
    sizeof(NodeTableObject),
    0,
    Py_TPFLAGS_DEFAULT,
    kTableSlots,
};

}  // namespace

PyObject* NewNodeTable(PyDagCore* core) {
    PyTypeObject* type = NodeTableType();
    if (type == nullptr) {
        type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&kTableSpec));
        if (type == nullptr) {
            return nullptr;
        }
        NodeTableType() = type;  // 进程内常驻
    }
    auto* obj = PyObject_New(NodeTableObject, type);
    if (obj == nullptr) {
        return nullptr;
    }
    obj->core = core;
    return reinterpret_cast<PyObject*>(obj);
}

void DetachNodeTable(PyObject* table) noexcept {
    if (table != nullptr) {
        reinterpret_cast<NodeTableObject*>(table)->core = nullptr;
    }
}

}  // namespace dag_core

namespace {

// 核对 set((a, b)) 的迭代顺序：a、b 互不相同且 (a & 7) != (b & 7) 时按 (x & 7) 排列。
bool PairOrderBySlot() {
    for (long long a = 0; a < 80; ++a) {
        for (long long b = 0; b < 80; ++b) {
            if ((a & 7) == (b & 7)) {
                continue;
            }
            py::tuple t = py::make_tuple(a, b);
            py::object set = py::reinterpret_steal<py::object>(PySet_New(t.ptr()));
            if (!set) {
                throw py::error_already_set();
            }
            py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(set.ptr()));
            py::object first = py::reinterpret_steal<py::object>(PyIter_Next(it.ptr()));
            if (!first) {
                throw py::error_already_set();
            }
            const long long expect = (a & 7) < (b & 7) ? a : b;
            if (first.cast<long long>() != expect) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

void BindDagCore(py::module_& m) {
    m.def(
        "_register_dag_types",
        [](py::object input, py::object output, py::object operation, py::object dag_node_cls,
           py::object scan_fallback, py::object instr_cls, py::object type_gate, py::object type_measure,
           py::object type_barrier, py::object type_channel) {
            DagRegistry& reg = Registry();
            reg.node_type_input = std::move(input);
            reg.node_type_output = std::move(output);
            reg.node_type_operation = std::move(operation);
            reg.dag_node_cls = std::move(dag_node_cls);
            reg.scan_fallback = std::move(scan_fallback);
            reg.instr_cls = std::move(instr_cls);
            reg.type_gate = std::move(type_gate);
            reg.type_measure = std::move(type_measure);
            reg.type_barrier = std::move(type_barrier);
            reg.type_channel = std::move(type_channel);
            reg.pair_order_by_slot = PairOrderBySlot();
            reg.ready = true;
        },
        "Register the Python DAG node types used by _DagCore (called by quspect.dag.dag_circuit).");

    py::class_<PyDagCore> core(m, "_DagCore", py::module_local(), "C++ graph storage behind quspect.dag.DAGCircuit.");
    core.def(py::init<>())
        .def("_nodes_dict", &PyDagCore::nodes_dict, "Return the node_id -> DAGNode dict (shared, do not mutate).")
        .def_property_readonly(
            "num_qubits", [](const PyDagCore& d) { return d.store.num_qubits(); },
            "Return the number of qubit lines in this DAG.")
        .def_property_readonly(
            "num_clbits", [](const PyDagCore& d) { return d.store.num_clbits(); },
            "Return the number of classical bit lines in this DAG.")
        .def_property_readonly(
            "modification_count", [](const PyDagCore& d) { return d.store.version(); },
            R"doc(Return a counter that increases on every structural modification.

Adding or removing nodes or edges, and replacing a node's instruction through
:meth:`substitute_node_instruction`, increase the counter. Comparing two
readings is a cheap way to detect whether a pass touched the DAG.

Returns:
    A monotonically increasing integer.)doc")
        .def(
            "num_nodes", [](const PyDagCore& d) { return d.store.num_nodes(); }, "Return the total number of nodes.")
        .def(
            "num_edges", [](const PyDagCore& d) { return d.store.num_edges(); }, "Return the total number of edges.")
        .def(
            "set_qubit_count", [](PyDagCore& d, std::int64_t n) { d.store.set_num_qubits(n); }, py::arg("n"),
            "Set the number of qubit lines.")
        .def(
            "set_clbit_count", [](PyDagCore& d, std::int64_t n) { d.store.set_num_clbits(n); }, py::arg("n"),
            "Set the number of classical bit lines.")
        .def("add_node", &PyDagCore::add_node, py::arg("node"),
             R"doc(Add a node to the DAG.

Args:
    node: The DAGNode to add.

Raises:
    ValueError: If a node with the same ID already exists.)doc")
        .def("remove_node", &PyDagCore::remove_node, py::arg("node_id"),
             R"doc(Remove a node and all its incident edges from the DAG.

The neighbours are **not** reconnected; callers that want to keep the wires
connected should use :meth:`remove_node_and_reconnect`. Runs in O(degree) time.

Args:
    node_id: The ID of the node to remove.

Raises:
    KeyError: If the node_id is not found.)doc")
        .def("_remove_node_and_reconnect", &PyDagCore::remove_node_and_reconnect, py::arg("node_id"),
             py::arg("wires"))
        .def("_bump_version", [](PyDagCore& d) { d.store.bump_version(); })
        .def("add_edge", &PyDagCore::add_edge, py::arg("edge"), py::kw_only(), py::arg("defer_order_check") = false,
             R"doc(Add a directed edge to the DAG.

Args:
    edge: The DAGEdge to add.
    defer_order_check: Internal flag for reordering passes. When True the
        incremental topological numbering is not checked; the caller must
        call :meth:`reverse_adjacent_edge_order` afterwards.)doc")
        .def("remove_edge", &PyDagCore::remove_edge, py::arg("source_id"), py::arg("target_id"), py::arg("wire"),
             R"doc(Remove one directed edge from the DAG.

Unlike :meth:`remove_node`, the endpoints are kept.

Args:
    source_id: Source node ID.
    target_id: Target node ID.
    wire: Wire label of the edge (e.g. ``"q0"``).

Raises:
    KeyError: If no such edge exists.)doc")
        .def("_edge_triples", &PyDagCore::edge_triples)
        .def(
            "begin_transaction", [](PyDagCore& d) { return d.store.begin_transaction(); },
            R"doc(Start (or nest into) an undo journal for edge modifications.

While a transaction is open, edge insertions and removals (also those
performed by :meth:`remove_node`, whose node removal itself is *not* undone)
are journaled so that :meth:`rollback` can restore the exact previous edge
order.

Returns:
    A savepoint to pass to :meth:`rollback`.)doc")
        .def(
            "rollback",
            [](PyDagCore& d, std::int64_t savepoint) {
                try {
                    d.store.rollback(savepoint);
                } catch (const std::logic_error& e) {
                    PyErr_SetString(PyExc_RuntimeError, e.what());
                    throw py::error_already_set();
                }
            },
            py::arg("savepoint"),
            R"doc(Undo all journaled edge modifications made after ``savepoint``.

Args:
    savepoint: A value returned by :meth:`begin_transaction`.

Raises:
    RuntimeError: If no transaction is open.)doc")
        .def(
            "end_transaction", [](PyDagCore& d) { d.store.end_transaction(); },
            R"doc(Close the innermost transaction, keeping all modifications.

The journal is discarded when the outermost transaction ends.)doc")
        .def(
            "predecessors", [](const PyDagCore& d, NodeId id) { return d.adjacent(d.store.in_adj(id)); },
            py::arg("node_id"),
            R"doc(Return node_ids of all direct predecessors (one entry per edge).)doc")
        .def(
            "successors", [](const PyDagCore& d, NodeId id) { return d.adjacent(d.store.out_adj(id)); },
            py::arg("node_id"),
            R"doc(Return node_ids of all direct successors (one entry per edge).)doc")
        .def(
            "out_edges_with_wires",
            [](PyDagCore& d, NodeId id) { return d.adjacent_with_wires(d.store.out_adj(id)); }, py::arg("node_id"),
            R"doc(Return ``(target_id, wire)`` for every outgoing edge, in edge order.)doc")
        .def(
            "in_edges_with_wires", [](PyDagCore& d, NodeId id) { return d.adjacent_with_wires(d.store.in_adj(id)); },
            py::arg("node_id"), R"doc(Return ``(source_id, wire)`` for every incoming edge, in edge order.)doc")
        .def("has_edge", &PyDagCore::has_edge, py::arg("source_id"), py::arg("target_id"),
             py::arg("wire") = py::none(),
             R"doc(Return whether a direct edge ``source -> target`` exists.

Args:
    source_id: Source node ID.
    target_id: Target node ID.
    wire: If given, the edge must carry this wire label.

Returns:
    True if a matching edge exists. Runs in O(out-degree).)doc")
        .def(
            "wire_predecessor",
            [](const PyDagCore& d, NodeId id, const py::str& wire) { return d.wire_neighbour(id, wire, false); },
            py::arg("node_id"), py::arg("wire"),
            R"doc(Return the node feeding ``node_id`` on ``wire`` (``None`` if there is none).

If several edges match, the most recently added wins.)doc")
        .def(
            "wire_successor",
            [](const PyDagCore& d, NodeId id, const py::str& wire) { return d.wire_neighbour(id, wire, true); },
            py::arg("node_id"), py::arg("wire"),
            R"doc(Return the node following ``node_id`` on ``wire`` (``None`` if there is none).

If several edges match, the most recently added wins.)doc")
        .def("operation_nodes", &PyDagCore::operation_nodes, "Return all OPERATION nodes in topological order.")
        .def("topological_order", &PyDagCore::topological_order,
             R"doc(Return node_ids in topological order.

Uses Kahn's algorithm. The DAG must be acyclic (guaranteed by construction from
a sequential circuit). The result is cached until the next structural
modification.

Returns:
    List of node IDs in topological order (INPUTs first, OUTPUTs last).

Raises:
    RuntimeError: If a cycle is detected.)doc")
        .def("ensure_topological_positions", &PyDagCore::ensure_topological_positions,
             R"doc(Compute the incremental topological numbering if it is missing.

Raises:
    RuntimeError: If the DAG contains a cycle.)doc")
        .def("reverse_adjacent_edge_order", &PyDagCore::reverse_adjacent_edge_order, py::arg("first_id"),
             py::arg("second_id"),
             R"doc(Repair the topological numbering after ``second`` was moved before ``first``.

This is the incremental (Pearce-Kelly) step used by commutation reordering.

Args:
    first_id: The node that used to come first.
    second_id: The node that now comes first.

Returns:
    True if the DAG is still acyclic, False if a cycle was created (the
    numbering is left unchanged in that case).)doc")
        .def("layers", &PyDagCore::layers,
             R"doc(Partition nodes into layers (depth levels) for scheduling.

Returns:
    A list of layers, each layer is a list of node_ids.)doc")
        .def("critical_path_length", &PyDagCore::critical_path_length,
             "Return the number of operation nodes on the longest dependency chain.")
        .def(
            "nodes_on_wire",
            [](py::object self, const py::str& wire) { return self.cast<PyDagCore&>().nodes_on_wire(self, wire); },
            py::arg("wire"),
            R"doc(Return node_ids on a given qubit/bit wire in topological order.

Args:
    wire: Wire identifier, e.g. ``"q0"`` or ``"c1"``.

Returns:
    List of node IDs that involve the given wire.)doc")
        .def(
            "qubit_wires",
            [](const PyDagCore& d) {
                py::list out;
                for (std::int64_t i = 0; i < d.store.num_qubits(); ++i) {
                    out.append(i);
                }
                return out;
            },
            "Return the list of qubit indices present in this DAG.")
        .def(
            "clbit_wires",
            [](const PyDagCore& d) {
                py::list out;
                for (std::int64_t i = 0; i < d.store.num_clbits(); ++i) {
                    out.append(i);
                }
                return out;
            },
            "Return the list of classical bit indices present in this DAG.")
        .def("_wire_sequences", &PyDagCore::wire_sequences, py::arg("cache"))
        .def("_operation_instructions", &PyDagCore::operation_instructions,
             "Instructions of the OPERATION nodes in topological order, or None (see DAGToCircuitCompiler).")
        .def("_nodes_of_type", &PyDagCore::nodes_of_type, py::arg("node_type"),
             "Nodes whose node_type equals node_type, in insertion order.")
        .def(
            "_set_node_instruction",
            [](PyDagCore& d, NodeId id, const py::object& instr) {
                d.set_instruction(id, instr);
                return true;
            },
            py::arg("node_id"), py::arg("instruction"),
            "node.instruction = instruction without creating the node object when possible.")
        .def(
            "_remove_lazy_node_and_reconnect",
            [](PyDagCore& d, NodeId id) {
                if (!d.is_lazy(id)) {
                    return false;
                }
                std::vector<ir::WireId> wires;
                d.lazy_reconnect_wires(id, wires);
                try {
                    d.store.remove_node_and_reconnect(id, wires.data(), wires.size());
                } catch (const ir::DagKeyError& e) {
                    throw py::key_error(e.what());
                }
                d.forget_node(id);
                return true;
            },
            py::arg("node_id"),
            "remove_node_and_reconnect for a node whose object has not been created; False otherwise.")
        .def("_core_getstate", &PyDagCore::get_state)
        .def("_core_setstate", &PyDagCore::set_state, py::arg("state"))
        .def("_build_from_instructions", &PyDagCore::build_from_instructions, py::arg("instructions"),
             py::arg("num_qubits"), py::arg("num_clbits"));
    BindDagPasses(m, core);
}

}  // namespace quspect::native::pybind
