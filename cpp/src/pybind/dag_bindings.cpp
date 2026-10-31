// DAGCircuit 的 C++ 图存储内核（quspect.dag.dag_circuit.DAGCircuit 的基类 _DagCore）。
//
// Python 端的 DAGCircuit 继承这里绑定的 _DagCore：图结构（邻接表、拓扑序、增量拓扑编号、
// 事务日志、沿线遍历）全部在 ir::DagStore 中完成，节点对象（DAGNode 数据类）仍是 Python
// 对象，保存在共享的 dict 中（Python 端的 get_node 直接查这个 dict）。线标签在边界上转换：
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

void BindDagCore(py::module_& m) {
    m.def(
        "_register_dag_types",
        [](py::object input, py::object output, py::object operation, py::object dag_node_cls,
           py::object scan_fallback) {
            DagRegistry& reg = Registry();
            reg.node_type_input = std::move(input);
            reg.node_type_output = std::move(output);
            reg.node_type_operation = std::move(operation);
            reg.dag_node_cls = std::move(dag_node_cls);
            reg.scan_fallback = std::move(scan_fallback);
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
        .def("_core_getstate", &PyDagCore::get_state)
        .def("_core_setstate", &PyDagCore::set_state, py::arg("state"))
        .def("_build_from_instructions", &PyDagCore::build_from_instructions, py::arg("instructions"),
             py::arg("num_qubits"), py::arg("num_clbits"));
    BindDagPasses(m, core);
}

}  // namespace quspect::native::pybind
