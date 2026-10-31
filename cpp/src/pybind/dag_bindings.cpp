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

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using ir::DagNodeKind;
using ir::DagStore;
using ir::NodeId;
using ir::WireId;

// 由 quspect.dag.dag_circuit 在导入时登记的 Python 类型与辅助函数。
struct DagRegistry {
    py::object node_type_input;
    py::object node_type_output;
    py::object node_type_operation;
    py::object dag_node_cls;
    py::object scan_fallback;  // 非规范线标签的 nodes_on_wire 回退：_scan_wire_fallback(dag, label)
    bool ready = false;
};

DagRegistry& Registry() {
    static DagRegistry* reg = new DagRegistry();  // 进程结束时不析构 Python 对象
    return *reg;
}

void RequireRegistry() {
    if (!Registry().ready) {
        throw std::runtime_error("quspect DAG core types are not registered");
    }
}

// 预先驻留的属性名。
struct Names {
    py::str node_id{"node_id"};
    py::str node_type{"node_type"};
    py::str instruction{"instruction"};
    py::str qubits{"qubits"};
    py::str clbits{"clbits"};
    py::str label{"label"};
    py::str metadata{"metadata"};
    py::str source_id{"source_id"};
    py::str target_id{"target_id"};
    py::str wire{"wire"};
};

Names& N() {
    static Names* names = new Names();
    return *names;
}

NodeId AsNodeId(PyObject* obj) {
    const long long v = PyLong_AsLongLong(obj);
    if (v == -1 && PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return static_cast<NodeId>(v);
}

// 解析规范线标签：'q' 或 'c' 后接无前导零的十进制数。
std::optional<WireId> ParseCanonical(const char* s, Py_ssize_t n) {
    if (n < 2 || n > 12 || (s[0] != 'q' && s[0] != 'c')) {
        return std::nullopt;
    }
    if (s[1] == '0' && n > 2) {
        return std::nullopt;
    }
    std::int64_t v = 0;
    for (Py_ssize_t i = 1; i < n; ++i) {
        const char ch = s[i];
        if (ch < '0' || ch > '9') {
            return std::nullopt;
        }
        v = v * 10 + (ch - '0');
    }
    return s[0] == 'q' ? ir::QubitWire(v) : ir::ClbitWire(v);
}

class PyDagCore {
public:
    PyDagCore() : nodes_(py::dict()) {
    }

    DagStore store;

    py::dict nodes_dict() const {
        return nodes_;
    }

    // ---- 线标签 ----

    // 标签 -> 编号（必要时登记新标签）。label 必须是 str。
    WireId wire_id(PyObject* label) {
        if (!PyUnicode_Check(label)) {
            throw py::type_error("wire label must be a str");
        }
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(label, &n);
        if (s == nullptr) {
            throw py::error_already_set();
        }
        if (auto w = ParseCanonical(s, n)) {
            return *w;
        }
        std::string key(s, static_cast<std::size_t>(n));
        auto it = other_ids_.find(key);
        if (it != other_ids_.end()) {
            return it->second;
        }
        const WireId id = next_other_--;
        other_ids_.emplace(std::move(key), id);
        labels_[id] = py::reinterpret_borrow<py::object>(label);
        return id;
    }

    // 查找已登记的标签（不登记）；未登记的标签不可能出现在任何边上。
    std::optional<WireId> find_wire(PyObject* label) const {
        if (!PyUnicode_Check(label)) {
            return std::nullopt;
        }
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(label, &n);
        if (s == nullptr) {
            throw py::error_already_set();
        }
        if (auto w = ParseCanonical(s, n)) {
            return *w;
        }
        auto it = other_ids_.find(std::string(s, static_cast<std::size_t>(n)));
        if (it == other_ids_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    py::object label(WireId id) {
        if (py::object* cached = labels_.find(id); cached != nullptr && *cached) {
            return *cached;
        }
        py::object s = py::reinterpret_steal<py::object>(
            PyUnicode_FromFormat("%c%lld", id % 2 == 0 ? 'q' : 'c', static_cast<long long>(id / 2)));
        if (!s) {
            throw py::error_already_set();
        }
        labels_[id] = s;
        return s;
    }

    // 节点所在的线（qubits 在前、clbits 在后，按集合迭代顺序），与 Python 的 _node_wires 一致。
    void node_wires(PyObject* node, std::vector<WireId>& out) {
        out.clear();
        append_bits(node, N().qubits.ptr(), 'q', out);
        append_bits(node, N().clbits.ptr(), 'c', out);
    }

    // ---- 节点 ----

    void add_node(const py::object& node) {
        RequireRegistry();
        PyObject* obj = node.ptr();
        py::object id_obj = py::reinterpret_steal<py::object>(PyObject_GetAttr(obj, N().node_id.ptr()));
        if (!id_obj) {
            throw py::error_already_set();
        }
        const NodeId id = AsNodeId(id_obj.ptr());
        py::object type_obj = py::reinterpret_steal<py::object>(PyObject_GetAttr(obj, N().node_type.ptr()));
        if (!type_obj) {
            throw py::error_already_set();
        }
        const DagNodeKind kind = kind_of(type_obj);
        if (store.has_node(id)) {
            throw py::value_error("Node with id=" + std::to_string(id) + " already exists in DAG.");
        }
        node_wires(obj, wires_buf_);
        insert_node(id, kind, node);
    }

    void remove_node(const py::object& node_id) {
        const NodeId id = AsNodeId(node_id.ptr());
        try {
            store.remove_node(id);
        } catch (const ir::DagKeyError& e) {
            throw py::key_error(e.what());
        }
        drop_node(id);
    }

    void remove_node_and_reconnect(NodeId id, const py::list& wires) {
        std::vector<WireId> ids;
        ids.reserve(wires.size());
        for (py::handle w : wires) {
            ids.push_back(wire_id(w.ptr()));
        }
        try {
            store.remove_node_and_reconnect(id, ids.data(), ids.size());
        } catch (const ir::DagKeyError& e) {
            throw py::key_error(e.what());
        }
        drop_node(id);
    }

    // ---- 边 ----

    void add_edge(const py::object& edge, bool defer_order_check) {
        PyObject* obj = edge.ptr();
        py::object s = py::reinterpret_steal<py::object>(PyObject_GetAttr(obj, N().source_id.ptr()));
        if (!s) {
            throw py::error_already_set();
        }
        py::object t = py::reinterpret_steal<py::object>(PyObject_GetAttr(obj, N().target_id.ptr()));
        if (!t) {
            throw py::error_already_set();
        }
        py::object w = py::reinterpret_steal<py::object>(PyObject_GetAttr(obj, N().wire.ptr()));
        if (!w) {
            throw py::error_already_set();
        }
        store.add_edge(AsNodeId(s.ptr()), AsNodeId(t.ptr()), wire_id(w.ptr()), defer_order_check);
    }

    void remove_edge(NodeId source, NodeId target, const py::str& wire) {
        const auto w = find_wire(wire.ptr());
        try {
            if (!w) {
                throw ir::DagEdgeNotFound();
            }
            store.remove_edge(source, target, *w);
        } catch (const ir::DagEdgeNotFound&) {
            throw py::key_error("Edge " + std::to_string(source) + "->" + std::to_string(target) + " on "
                                + py::repr(wire).cast<std::string>() + " not found.");
        }
    }

    py::list edge_triples() {
        std::vector<ir::DagEdge> edges;
        store.edges(edges);
        py::list out(edges.size());
        for (std::size_t i = 0; i < edges.size(); ++i) {
            out[i] = py::make_tuple(edges[i].source, edges[i].target, label(edges[i].wire));
        }
        return out;
    }

    // ---- 查询 ----

    py::list adjacent(const DagStore::AdjList& adj) const {
        PyObject* out = PyList_New(static_cast<Py_ssize_t>(adj.size()));
        if (out == nullptr) {
            throw py::error_already_set();
        }
        for (std::size_t i = 0; i < adj.size(); ++i) {
            PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), PyLong_FromLongLong(adj[i].nbr));
        }
        return py::reinterpret_steal<py::list>(out);
    }

    py::list adjacent_with_wires(const DagStore::AdjList& adj) {
        py::list out(adj.size());
        for (std::size_t i = 0; i < adj.size(); ++i) {
            out[i] = py::make_tuple(adj[i].nbr, label(adj[i].wire));
        }
        return out;
    }

    py::object wire_neighbour(NodeId id, const py::str& wire, bool successor) const {
        const auto w = find_wire(wire.ptr());
        if (!w) {
            return py::none();
        }
        const auto r = successor ? store.wire_successor(id, *w) : store.wire_predecessor(id, *w);
        if (!r) {
            return py::none();
        }
        return py::reinterpret_steal<py::object>(PyLong_FromLongLong(*r));
    }

    bool has_edge(NodeId source, NodeId target, const py::object& wire) const {
        if (wire.is_none()) {
            return store.has_edge(source, target, std::nullopt);
        }
        const auto w = find_wire(wire.ptr());
        if (!w) {
            return false;
        }
        return store.has_edge(source, target, *w);
    }

    // 调用 DagStore 并把它的异常翻译成 Python 的 KeyError / RuntimeError。
    template <typename F>
    static decltype(auto) guarded(F&& f) {
        try {
            return f();
        } catch (const ir::DagKeyError& e) {
            throw_key_error(e);
        } catch (const ir::DagCycleError& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::error_already_set();
        }
    }

    const std::vector<NodeId>& topo() {
        return guarded([this]() -> const std::vector<NodeId>& { return store.topological_order(); });
    }

    py::list topological_order() {
        const std::vector<NodeId>& order = topo();
        if (topo_list_version_ != store.version() || !topo_list_) {
            topo_list_ = id_list(order);
            topo_list_version_ = store.version();
        }
        return py::reinterpret_steal<py::list>(PyList_GetSlice(topo_list_.ptr(), 0, PY_SSIZE_T_MAX));
    }

    py::list operation_nodes() {
        const std::vector<NodeId>& order = topo();
        py::list out;
        for (NodeId id : order) {
            const NodeEntry* e = objects_.find(id);
            if (e->kind == DagNodeKind::kOperation) {
                out.append(e->obj);
            }
        }
        return out;
    }

    bool reverse_adjacent_edge_order(NodeId first, NodeId second) {
        return guarded([&] { return store.reverse_adjacent_edge_order(first, second); });
    }

    void ensure_topological_positions() {
        guarded([this] { store.ensure_topological_positions(); });
    }

    py::list layers() {
        const std::vector<std::vector<NodeId>> l = guarded([this] { return store.layers(); });
        py::list out(l.size());
        for (std::size_t i = 0; i < l.size(); ++i) {
            out[i] = id_list(l[i]);
        }
        return out;
    }

    std::int64_t critical_path_length() {
        return guarded([this] { return store.critical_path_length(); });
    }

    py::list nodes_on_wire(const py::object& self, const py::str& wire) {
        if (wire_cache_version_ != store.version()) {
            wire_cache_.clear();
            other_wire_cache_.clear();
            wire_cache_version_ = store.version();
        }
        Py_ssize_t n = 0;
        const char* s = PyUnicode_AsUTF8AndSize(wire.ptr(), &n);
        if (s == nullptr) {
            throw py::error_already_set();
        }
        const auto canonical = ParseCanonical(s, n);
        py::object cached;
        if (canonical) {
            if (py::object* hit = wire_cache_.find(*canonical); hit != nullptr && *hit) {
                cached = *hit;
            } else {
                if (!store.walk_wire(*canonical, *canonical, ids_buf_)) {
                    guarded([this, &canonical] { store.scan_wire(*canonical, ids_buf_); });
                }
                cached = id_list(ids_buf_);
                wire_cache_[*canonical] = cached;
            }
        } else {
            std::string key(s, static_cast<std::size_t>(n));
            auto it = other_wire_cache_.find(key);
            if (it != other_wire_cache_.end()) {
                cached = it->second;
            } else {
                // 非规范标签的回退语义（int() 解析等）交给 Python 参考实现。
                RequireRegistry();
                cached = Registry().scan_fallback(self, wire);
                other_wire_cache_.emplace(std::move(key), cached);
            }
        }
        return py::reinterpret_steal<py::list>(PyList_GetSlice(cached.ptr(), 0, PY_SSIZE_T_MAX));
    }

    // 每条线上依次出现的操作指令的 repr（quspect.dag.passes.base._wire_sequences 的实现）：
    // qubit q 的键为 q，clbit c 的键为 -1 - c。cache 为 id(instruction) -> (instruction, repr)。
    py::dict wire_sequences(const py::object& cache) {
        const std::vector<NodeId>& order = topo();
        const bool use_cache = !cache.is_none();
        PyObject* cache_dict = cache.ptr();
        py::dict wires;
        for (NodeId id : order) {
            const NodeEntry* e = objects_.find(id);
            if (e->kind != DagNodeKind::kOperation) {
                continue;
            }
            PyObject* node = e->obj.ptr();
            py::object instr = py::reinterpret_steal<py::object>(PyObject_GetAttr(node, N().instruction.ptr()));
            if (!instr) {
                throw py::error_already_set();
            }
            py::object key;
            if (!use_cache) {
                key = py::reinterpret_steal<py::object>(PyObject_Repr(instr.ptr()));
                if (!key) {
                    throw py::error_already_set();
                }
            } else {
                py::object ident = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(instr.ptr()));
                PyObject* hit = PyDict_GetItemWithError(cache_dict, ident.ptr());
                if (hit == nullptr && PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                if (hit != nullptr && hit != Py_None) {
                    py::object first = py::reinterpret_steal<py::object>(PySequence_GetItem(hit, 0));
                    if (!first) {
                        throw py::error_already_set();
                    }
                    if (first.ptr() == instr.ptr()) {
                        key = py::reinterpret_steal<py::object>(PySequence_GetItem(hit, 1));
                        if (!key) {
                            throw py::error_already_set();
                        }
                    }
                }
                if (!key) {
                    key = py::reinterpret_steal<py::object>(PyObject_Repr(instr.ptr()));
                    if (!key) {
                        throw py::error_already_set();
                    }
                    py::tuple entry = py::make_tuple(instr, key);
                    if (PyDict_SetItem(cache_dict, ident.ptr(), entry.ptr()) != 0) {
                        throw py::error_already_set();
                    }
                }
            }
            append_wire_keys(wires, node, N().qubits.ptr(), false, key);
            append_wire_keys(wires, node, N().clbits.ptr(), true, key);
        }
        return wires;
    }

    // ---- 状态（拷贝 / pickle） ----

    py::tuple get_state() const {
        const ir::DagState st = store.export_state();
        py::list nodes;
        for (const auto& n : st.nodes) {
            py::list wires;
            for (WireId w : n.wires) {
                wires.append(w);
            }
            nodes.append(py::make_tuple(n.id, static_cast<int>(n.kind), wires, objects_.find(n.id)->obj));
        }
        py::list edges;
        for (const auto& e : st.edges) {
            edges.append(py::make_tuple(e.seq, e.edge.source, e.edge.target, e.edge.wire));
        }
        auto pairs = [](const auto& v) {
            py::list out;
            for (const auto& [a, b] : v) {
                out.append(py::make_tuple(a, b));
            }
            return out;
        };
        py::list ambiguous;
        for (WireId w : st.ambiguous_wires) {
            ambiguous.append(w);
        }
        py::list journal;
        for (const auto& je : st.journal) {
            journal.append(py::make_tuple(static_cast<int>(je.kind), je.key, je.edge.source, je.edge.target,
                                          je.edge.wire, je.old_pos,
                                          je.has_positions ? py::object(pairs(je.positions)) : py::none()));
        }
        py::list labels;
        for (const auto& entry : other_ids_) {
            labels.append(py::make_tuple(entry.second, *labels_.find(entry.second)));  // 登记时已保存标签对象
        }
        return py::make_tuple(1, nodes, edges, st.next_seq, pairs(st.wire_input), ambiguous, pairs(st.wire_count),
                              st.num_qubits, st.num_clbits, st.version,
                              st.has_positions ? py::object(pairs(st.positions)) : py::none(), st.next_pos,
                              st.journal_open, journal, st.txn_depth, labels, next_other_);
    }

    void set_state(const py::tuple& t) {
        if (t.size() != 17 || t[0].cast<int>() != 1) {
            throw py::value_error("unsupported DAG state");
        }
        auto pair_vec = [](const py::handle& h) {
            std::vector<std::pair<std::int64_t, std::int64_t>> out;
            for (py::handle item : h) {
                auto tup = py::reinterpret_borrow<py::tuple>(item);
                out.emplace_back(tup[0].cast<std::int64_t>(), tup[1].cast<std::int64_t>());
            }
            return out;
        };
        ir::DagState st;
        std::vector<std::pair<NodeId, py::object>> node_objs;
        for (py::handle item : t[1]) {
            auto tup = py::reinterpret_borrow<py::tuple>(item);
            ir::DagState::Node n;
            n.id = tup[0].cast<NodeId>();
            const int kind = tup[1].cast<int>();
            if (kind < 0 || kind > 2) {
                throw py::value_error("invalid node kind in DAG state");
            }
            n.kind = static_cast<DagNodeKind>(kind);
            for (py::handle w : tup[2]) {
                n.wires.push_back(w.cast<WireId>());
            }
            node_objs.emplace_back(n.id, py::reinterpret_borrow<py::object>(tup[3]));
            st.nodes.push_back(std::move(n));
        }
        for (py::handle item : t[2]) {
            auto tup = py::reinterpret_borrow<py::tuple>(item);
            st.edges.push_back(ir::DagState::Edge{
                tup[0].cast<std::int64_t>(),
                ir::DagEdge{tup[1].cast<NodeId>(), tup[2].cast<NodeId>(), tup[3].cast<WireId>()}});
        }
        st.next_seq = t[3].cast<std::int64_t>();
        st.wire_input = pair_vec(t[4]);
        for (py::handle w : t[5]) {
            st.ambiguous_wires.push_back(w.cast<WireId>());
        }
        st.wire_count = pair_vec(t[6]);
        st.num_qubits = t[7].cast<std::int64_t>();
        st.num_clbits = t[8].cast<std::int64_t>();
        st.version = t[9].cast<std::int64_t>();
        st.has_positions = !t[10].is_none();
        if (st.has_positions) {
            st.positions = pair_vec(t[10]);
        }
        st.next_pos = t[11].cast<std::int64_t>();
        st.journal_open = t[12].cast<bool>();
        for (py::handle item : t[13]) {
            auto tup = py::reinterpret_borrow<py::tuple>(item);
            ir::DagState::JournalEntry je;
            je.kind = static_cast<std::uint8_t>(tup[0].cast<int>());
            je.key = tup[1].cast<std::int64_t>();
            je.edge = ir::DagEdge{tup[2].cast<NodeId>(), tup[3].cast<NodeId>(), tup[4].cast<WireId>()};
            je.old_pos = tup[5].cast<std::int64_t>();
            je.has_positions = !tup[6].is_none();
            if (je.has_positions) {
                je.positions = pair_vec(tup[6]);
            }
            st.journal.push_back(std::move(je));
        }
        st.txn_depth = t[14].cast<std::int64_t>();
        try {
            store.import_state(st);
        } catch (const std::invalid_argument& e) {
            throw py::value_error(e.what());
        }
        // 节点对象与线标签表。
        PyDict_Clear(nodes_.ptr());
        objects_.clear();
        for (std::size_t i = 0; i < node_objs.size(); ++i) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(node_objs[i].first));
            if (PyDict_SetItem(nodes_.ptr(), key.ptr(), node_objs[i].second.ptr()) != 0) {
                throw py::error_already_set();
            }
            NodeEntry& e = objects_[node_objs[i].first];
            e.obj = node_objs[i].second;
            e.kind = st.nodes[i].kind;
        }
        other_ids_.clear();
        labels_.clear();
        for (py::handle item : t[15]) {
            auto tup = py::reinterpret_borrow<py::tuple>(item);
            const WireId id = tup[0].cast<WireId>();
            other_ids_[tup[1].cast<std::string>()] = id;
            labels_[id] = py::reinterpret_borrow<py::object>(tup[1]);
        }
        next_other_ = t[16].cast<WireId>();
        topo_list_ = py::object();
        topo_list_version_ = -1;
        wire_cache_.clear();
        other_wire_cache_.clear();
        wire_cache_version_ = -1;
    }

    // ---- 线路 -> DAG（CircuitCompiler.compile 的批量实现） ----

    void build_from_instructions(const py::sequence& instructions, std::int64_t num_qubits, std::int64_t num_clbits) {
        RequireRegistry();
        DagRegistry& reg = Registry();
        store.set_num_qubits(num_qubits);
        store.set_num_clbits(num_clbits);
        NodeId next_id = 0;
        std::vector<NodeId> qubit_last(static_cast<std::size_t>(num_qubits));
        std::vector<NodeId> clbit_last(static_cast<std::size_t>(num_clbits));
        py::object empty_label = py::str("");

        auto boundary = [&](const py::object& type, char kind, std::int64_t bit, const char* prefix) {
            py::object bits = py::reinterpret_steal<py::object>(PySet_New(nullptr));
            py::object bit_obj = py::reinterpret_steal<py::object>(PyLong_FromLongLong(bit));
            if (PySet_Add(bits.ptr(), bit_obj.ptr()) != 0) {
                throw py::error_already_set();
            }
            py::object none_bits = py::reinterpret_steal<py::object>(PySet_New(nullptr));
            py::object lbl = py::reinterpret_steal<py::object>(
                PyUnicode_FromFormat("%s:%c%lld", prefix, kind, static_cast<long long>(bit)));
            py::object node = kind == 'q' ? make_node(next_id, type, py::none(), bits, none_bits, lbl)
                                          : make_node(next_id, type, py::none(), none_bits, bits, lbl);
            wires_buf_.assign(1, kind == 'q' ? ir::QubitWire(bit) : ir::ClbitWire(bit));
            insert_node(next_id, type.is(reg.node_type_input) ? DagNodeKind::kInput : DagNodeKind::kOutput, node);
            return next_id++;
        };

        for (std::int64_t q = 0; q < num_qubits; ++q) {
            qubit_last[static_cast<std::size_t>(q)] = boundary(reg.node_type_input, 'q', q, "IN");
        }
        for (std::int64_t c = 0; c < num_clbits; ++c) {
            clbit_last[static_cast<std::size_t>(c)] = boundary(reg.node_type_input, 'c', c, "IN");
        }

        std::vector<std::int64_t> qs;
        std::vector<std::int64_t> cs;
        for (py::handle instr : instructions) {
            py::object qubits = py::reinterpret_steal<py::object>(PyObject_GetAttr(instr.ptr(), N().qubits.ptr()));
            if (!qubits) {
                throw py::error_already_set();
            }
            py::object clbits = py::reinterpret_steal<py::object>(PyObject_GetAttr(instr.ptr(), N().clbits.ptr()));
            if (!clbits) {
                throw py::error_already_set();
            }
            py::object qset = py::reinterpret_steal<py::object>(PySet_New(qubits.ptr()));
            if (!qset) {
                throw py::error_already_set();
            }
            py::object cset = py::reinterpret_steal<py::object>(PySet_New(clbits.ptr()));
            if (!cset) {
                throw py::error_already_set();
            }
            set_items(qset.ptr(), qs);
            set_items(cset.ptr(), cs);
            py::object node = make_node(next_id, reg.node_type_operation, py::reinterpret_borrow<py::object>(instr),
                                        qset, cset, empty_label);
            wires_buf_.clear();
            for (std::int64_t q : qs) {
                wires_buf_.push_back(bit_wire(q, 'q'));
            }
            for (std::int64_t c : cs) {
                wires_buf_.push_back(bit_wire(c, 'c'));
            }
            const NodeId op_id = next_id++;
            insert_node(op_id, DagNodeKind::kOperation, node);
            for (std::int64_t q : qs) {
                NodeId& last = last_ref(qubit_last, q);
                store.add_edge(last, op_id, ir::QubitWire(q));
                last = op_id;
            }
            for (std::int64_t c : cs) {
                NodeId& last = last_ref(clbit_last, c);
                store.add_edge(last, op_id, ir::ClbitWire(c));
                last = op_id;
            }
        }

        for (std::int64_t q = 0; q < num_qubits; ++q) {
            const NodeId out = boundary(reg.node_type_output, 'q', q, "OUT");
            store.add_edge(qubit_last[static_cast<std::size_t>(q)], out, ir::QubitWire(q));
        }
        for (std::int64_t c = 0; c < num_clbits; ++c) {
            const NodeId out = boundary(reg.node_type_output, 'c', c, "OUT");
            store.add_edge(clbit_last[static_cast<std::size_t>(c)], out, ir::ClbitWire(c));
        }
    }

private:
    struct NodeEntry {
        py::object obj;
        DagNodeKind kind = DagNodeKind::kOperation;
    };

    [[noreturn]] static void throw_key_error(const ir::DagKeyError& e) {
        if (e.has_int_key()) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(e.key()));
            PyErr_SetObject(PyExc_KeyError, key.ptr());
            throw py::error_already_set();
        }
        throw py::key_error(e.what());
    }

    static NodeId& last_ref(std::vector<NodeId>& last, std::int64_t bit) {
        if (bit < 0 || static_cast<std::size_t>(bit) >= last.size()) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(bit));
            PyErr_SetObject(PyExc_KeyError, key.ptr());
            throw py::error_already_set();
        }
        return last[static_cast<std::size_t>(bit)];
    }

    static py::list id_list(const std::vector<NodeId>& ids) {
        PyObject* out = PyList_New(static_cast<Py_ssize_t>(ids.size()));
        if (out == nullptr) {
            throw py::error_already_set();
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), PyLong_FromLongLong(ids[i]));
        }
        return py::reinterpret_steal<py::list>(out);
    }

    static void set_items(PyObject* set, std::vector<std::int64_t>& out) {
        out.clear();
        py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(set));
        if (!it) {
            throw py::error_already_set();
        }
        while (PyObject* item = PyIter_Next(it.ptr())) {
            const long long v = PyLong_AsLongLong(item);
            Py_DECREF(item);
            if (v == -1 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            out.push_back(v);
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
    }

    // 比特 -> 线编号；负数比特对应非规范标签（"q-1"）。
    WireId bit_wire(std::int64_t bit, char kind) {
        if (bit >= 0) {
            return kind == 'q' ? ir::QubitWire(bit) : ir::ClbitWire(bit);
        }
        py::object lbl =
            py::reinterpret_steal<py::object>(PyUnicode_FromFormat("%c%lld", kind, static_cast<long long>(bit)));
        return wire_id(lbl.ptr());
    }

    void append_bits(PyObject* node, PyObject* attr, char kind, std::vector<WireId>& out) {
        py::object bits = py::reinterpret_steal<py::object>(PyObject_GetAttr(node, attr));
        if (!bits) {
            throw py::error_already_set();
        }
        py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(bits.ptr()));
        if (!it) {
            throw py::error_already_set();
        }
        while (PyObject* item = PyIter_Next(it.ptr())) {
            py::object owned = py::reinterpret_steal<py::object>(item);
            const long long v = PyLong_AsLongLong(item);
            if (v == -1 && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            out.push_back(bit_wire(v, kind));
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
    }

    // wires[w].append(key)，w 为 node.<attr> 中每个比特（clbit 映射为 -1 - c）。
    static void append_wire_keys(py::dict& wires, PyObject* node, PyObject* attr, bool clbit, const py::object& key) {
        py::object bits = py::reinterpret_steal<py::object>(PyObject_GetAttr(node, attr));
        if (!bits) {
            throw py::error_already_set();
        }
        py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(bits.ptr()));
        if (!it) {
            throw py::error_already_set();
        }
        while (PyObject* item = PyIter_Next(it.ptr())) {
            py::object bit = py::reinterpret_steal<py::object>(item);
            py::object w = bit;
            if (clbit) {
                // -1 - c
                py::object minus_one = py::reinterpret_steal<py::object>(PyLong_FromLong(-1));
                w = py::reinterpret_steal<py::object>(PyNumber_Subtract(minus_one.ptr(), bit.ptr()));
                if (!w) {
                    throw py::error_already_set();
                }
            }
            PyObject* seq = PyDict_GetItemWithError(wires.ptr(), w.ptr());
            if (seq == nullptr) {
                if (PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                py::list fresh;
                if (PyDict_SetItem(wires.ptr(), w.ptr(), fresh.ptr()) != 0) {
                    throw py::error_already_set();
                }
                seq = fresh.ptr();
            }
            if (PyList_Append(seq, key.ptr()) != 0) {
                throw py::error_already_set();
            }
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
    }

    DagNodeKind kind_of(const py::object& type_obj) const {
        DagRegistry& reg = Registry();
        if (type_obj.is(reg.node_type_input)) {
            return DagNodeKind::kInput;
        }
        if (type_obj.is(reg.node_type_output)) {
            return DagNodeKind::kOutput;
        }
        return DagNodeKind::kOperation;
    }

    // 以 DAGNode 数据类的 __init__ 相同的顺序逐个设置字段（不经过 __init__）。
    static py::object make_node(NodeId id, const py::object& type, const py::object& instruction,
                                const py::object& qubits, const py::object& clbits, const py::object& lbl) {
        DagRegistry& reg = Registry();
        auto* cls = reinterpret_cast<PyTypeObject*>(reg.dag_node_cls.ptr());
        py::object empty = py::reinterpret_steal<py::object>(PyTuple_New(0));
        py::object node = py::reinterpret_steal<py::object>(PyBaseObject_Type.tp_new(cls, empty.ptr(), nullptr));
        if (!node) {
            throw py::error_already_set();
        }
        py::object id_obj = py::reinterpret_steal<py::object>(PyLong_FromLongLong(id));
        py::object metadata = py::reinterpret_steal<py::object>(PyDict_New());
        Names& n = N();
        if (PyObject_SetAttr(node.ptr(), n.node_id.ptr(), id_obj.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.node_type.ptr(), type.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.instruction.ptr(), instruction.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.qubits.ptr(), qubits.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.clbits.ptr(), clbits.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.label.ptr(), lbl.ptr()) != 0
            || PyObject_SetAttr(node.ptr(), n.metadata.ptr(), metadata.ptr()) != 0) {
            throw py::error_already_set();
        }
        return node;
    }

    // wires_buf_ 中是节点的线。
    void insert_node(NodeId id, DagNodeKind kind, const py::object& node) {
        store.add_node(id, kind, wires_buf_.data(), wires_buf_.size());
        py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(id));
        if (PyDict_SetItem(nodes_.ptr(), key.ptr(), node.ptr()) != 0) {
            throw py::error_already_set();
        }
        NodeEntry& e = objects_[id];
        e.obj = node;
        e.kind = kind;
    }

    void drop_node(NodeId id) {
        py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(id));
        if (PyDict_DelItem(nodes_.ptr(), key.ptr()) != 0) {
            PyErr_Clear();
        }
        objects_.erase(id);
    }

    py::dict nodes_;
    ir::IdMap<NodeEntry> objects_;
    std::unordered_map<std::string, WireId> other_ids_;
    ir::IdMap<py::object> labels_;
    WireId next_other_ = -1;

    py::object topo_list_;
    std::int64_t topo_list_version_ = -1;
    ir::IdMap<py::object> wire_cache_;
    std::unordered_map<std::string, py::object> other_wire_cache_;
    std::int64_t wire_cache_version_ = -1;

    std::vector<WireId> wires_buf_;
    std::vector<NodeId> ids_buf_;
};

}  // namespace

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

    py::class_<PyDagCore>(m, "_DagCore", py::module_local(),
                          "C++ graph storage behind quspect.dag.DAGCircuit.")
        .def(py::init<>())
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
}

}  // namespace quspect::native::pybind
