#include "quspect_native/ir/dag_passes.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

#include "quspect_native/ir/id_map.hpp"

namespace quspect::native::ir {

// ---------------------------------------------------------------------------
// 门名驻留
// ---------------------------------------------------------------------------

namespace {

struct NameTable {
    std::unordered_map<std::string, std::int32_t> ids;
    NameTable() {
        static const char* const kFixed[kNumFixedGateNames] = {
            "id", "h",  "x",  "y",  "z",  "s",  "sdg", "t",    "tdg", "rx",  "ry", "rz",
            "p",  "u1", "u2", "u3", "cx", "cz", "swap", "crx", "cry", "crz", "cp"};
        for (std::int32_t i = 0; i < kNumFixedGateNames; ++i) {
            ids.emplace(kFixed[i], i);
        }
    }
};

NameTable& Names() {
    static NameTable* table = new NameTable();  // 进程内常驻
    return *table;
}

}  // namespace

std::int32_t InternGateName(std::string_view name) {
    NameTable& t = Names();
    auto it = t.ids.find(std::string(name));
    if (it != t.ids.end()) {
        return it->second;
    }
    const auto id = static_cast<std::int32_t>(t.ids.size());
    t.ids.emplace(std::string(name), id);
    return id;
}

namespace {

// ---------------------------------------------------------------------------
// 门族（与各 Python 模块中的常量相同）
// ---------------------------------------------------------------------------

constexpr std::uint32_t Bit(std::int32_t name) noexcept {
    return std::uint32_t{1} << static_cast<std::uint32_t>(name);
}

bool InSet(std::uint32_t set, std::int32_t name) noexcept {
    return name >= 0 && name < kNumFixedGateNames && (set & Bit(name)) != 0;
}

// gate_cancellation.SELF_INVERSE_GATES
constexpr std::uint32_t kSelfInverse =
    Bit(kNameH) | Bit(kNameX) | Bit(kNameY) | Bit(kNameZ) | Bit(kNameCx) | Bit(kNameCz) | Bit(kNameSwap) | Bit(kNameId);
// gate_cancellation.ROTATION_FAMILIES
constexpr std::uint32_t kCancelRotations = Bit(kNameRx) | Bit(kNameRy) | Bit(kNameRz) | Bit(kNameP) | Bit(kNameCrx)
                                           | Bit(kNameCry) | Bit(kNameCrz) | Bit(kNameCp);
// commutation.Z_FAMILY / X_FAMILY / CNOT_LIKE
constexpr std::uint32_t kZFamily = Bit(kNameZ) | Bit(kNameS) | Bit(kNameSdg) | Bit(kNameT) | Bit(kNameTdg)
                                   | Bit(kNameRz) | Bit(kNameP) | Bit(kNameCz);
constexpr std::uint32_t kXFamily = Bit(kNameX) | Bit(kNameRx);
constexpr std::uint32_t kCnotLike = Bit(kNameCx) | Bit(kNameCz);
// single_qubit_fusion.SINGLE_QUBIT_GATES
constexpr std::uint32_t kSingleQubit = Bit(kNameId) | Bit(kNameH) | Bit(kNameX) | Bit(kNameY) | Bit(kNameZ)
                                       | Bit(kNameS) | Bit(kNameSdg) | Bit(kNameT) | Bit(kNameTdg) | Bit(kNameRx)
                                       | Bit(kNameRy) | Bit(kNameRz) | Bit(kNameP) | Bit(kNameU1) | Bit(kNameU2)
                                       | Bit(kNameU3);
// diagonal_before_measure.DIAGONAL_ELIMINABLE
constexpr std::uint32_t kDiagonal = Bit(kNameId) | Bit(kNameZ) | Bit(kNameS) | Bit(kNameSdg) | Bit(kNameT)
                                    | Bit(kNameTdg) | Bit(kNameRz) | Bit(kNameP) | Bit(kNameU1);
// identity_removal._ROTATIONS / _MATRIX_CHECKED
constexpr std::uint32_t kIdentityRotations = Bit(kNameRx) | Bit(kNameRy) | Bit(kNameRz) | Bit(kNameP) | Bit(kNameU1);
constexpr std::uint32_t kMatrixChecked = Bit(kNameU2) | Bit(kNameU3);

constexpr double kAngleTolerance = 1e-10;  // gate_cancellation.ANGLE_TOLERANCE
constexpr double kIdentityAtol = 1e-9;     // identity_removal._ATOL
constexpr double kPi = 3.141592653589793;  // math.pi

// ---------------------------------------------------------------------------
// 公用的 DAG 操作（与 Python 端 DAGCircuit 方法的调用序列相同）
// ---------------------------------------------------------------------------

template <typename A, typename B>
bool SameBits(const A& a, const B& b) noexcept {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// dag.nodes_on_wire("q<i>")：沿线遍历，不是简单链时退回扫描拓扑序。
void NodesOnWire(DagStore& dag, WireId wire, std::vector<NodeId>& out) {
    if (!dag.walk_wire(wire, wire, out)) {
        dag.scan_wire(wire, out);
    }
}

std::string WireLabel(WireId w) {
    return (w % 2 == 0 ? "q" : "c") + std::to_string(w / 2);
}

// 指令的线：qubits（有序）在前、clbits 在后，不去重。
void InstructionWires(const PassNode& n, std::vector<WireId>& out) {
    out.clear();
    for (std::int64_t q : n.qubits) {
        out.push_back(QubitWire(q));
    }
    for (std::int64_t c : n.clbits) {
        out.push_back(ClbitWire(c));
    }
}

// dag.remove_node(nid)
void RemoveNode(DagStore& dag, DagPassHost& host, NodeId id) {
    dag.remove_node(id);
    host.node_removed(id);
}

// 各 pass 的 _remove_node(dag, nid, wire)：只在被扫描的线上把前驱与后继重新连起来。
void RemoveNodeOnWire(DagStore& dag, DagPassHost& host, NodeId id, WireId wire) {
    const auto pred = dag.wire_predecessor(id, wire);
    const auto succ = dag.wire_successor(id, wire);
    RemoveNode(dag, host, id);
    if (pred && succ) {
        dag.add_edge(*pred, *succ, wire);
    }
}

// dag.remove_node_and_reconnect(nid)：指令的 qubits + clbits（保持顺序去重）。
void RemoveNodeAndReconnect(DagStore& dag, DagPassHost& host, NodeId id, std::vector<WireId>& buf) {
    const PassNode& n = host.node(id);
    if (n.type != PassInstrType::kNone) {
        InstructionWires(n, buf);
        std::vector<WireId> unique;
        unique.reserve(buf.size());
        for (WireId w : buf) {
            if (std::find(unique.begin(), unique.end(), w) == unique.end()) {
                unique.push_back(w);
            }
        }
        buf.swap(unique);
    } else {
        // 没有指令的节点：线来自节点的 qubits / clbits 集合（加入时记录）。
        buf.clear();
        if (const DagStore::WireList* wires = dag.node_wires(id)) {
            buf.assign(wires->begin(), wires->end());
        }
    }
    dag.remove_node_and_reconnect(id, buf.data(), buf.size());
    host.node_removed(id);
}

// 节点上门（指令非 None 且 gate 非 None）的过滤：nodes_on_wire 中 is_gate 的节点。
void GateNodesOnWire(DagStore& dag, DagPassHost& host, WireId wire, std::vector<NodeId>& ops,
                     std::vector<NodeId>& out) {
    NodesOnWire(dag, wire, ops);
    out.clear();
    for (NodeId nid : ops) {
        const PassNode& n = host.node(nid);
        if (n.is_gate() && n.has_gate) {
            out.push_back(nid);
        }
    }
}

// 结束事务（对应 Python 的 try / finally: dag.end_transaction()）。
class TransactionScope {
public:
    explicit TransactionScope(DagStore& dag) : dag_(dag), savepoint_(dag.begin_transaction()) {
    }
    ~TransactionScope() {
        dag_.end_transaction();
    }
    TransactionScope(const TransactionScope&) = delete;
    TransactionScope& operator=(const TransactionScope&) = delete;
    std::int64_t savepoint() const noexcept {
        return savepoint_;
    }

private:
    DagStore& dag_;
    std::int64_t savepoint_;
};

// ---------------------------------------------------------------------------
// GateCancellation
// ---------------------------------------------------------------------------

bool GatesCancel(const PassNode& a, const PassNode& b) {
    if (a.name == b.name) {
        if (InSet(kSelfInverse, a.name)) {
            return true;
        }
        if (InSet(kCancelRotations, a.name) && !a.params.empty() && !b.params.empty()) {
            if (a.params.size() != b.params.size()) {
                return false;
            }
            if (!a.all_concrete() || !b.all_concrete()) {
                return false;
            }
            for (std::size_t i = 0; i < a.params.size(); ++i) {
                if (!(std::fabs(a.params[i] + b.params[i]) < kAngleTolerance)) {
                    return false;
                }
            }
            return true;
        }
    }
    return (a.name == kNameS && b.name == kNameSdg) || (a.name == kNameSdg && b.name == kNameS)
           || (a.name == kNameT && b.name == kNameTdg) || (a.name == kNameTdg && b.name == kNameT);
}

bool ConsecutiveOnAllWires(DagStore& dag, DagPassHost& host, NodeId a, NodeId b, std::vector<WireId>& buf) {
    const PassNode& na = host.node(a);
    const PassNode& nb = host.node(b);
    if (na.type == PassInstrType::kNone || nb.type == PassInstrType::kNone) {
        return false;
    }
    if (!SameBits(na.qubits, nb.qubits) || !SameBits(na.clbits, nb.clbits)) {
        return false;
    }
    InstructionWires(na, buf);
    if (buf.empty()) {
        return false;
    }
    for (WireId w : buf) {
        if (!dag.has_edge(a, b, w)) {
            return false;
        }
    }
    return true;
}

void RemoveNodePair(DagStore& dag, DagPassHost& host, NodeId a, NodeId b, std::vector<WireId>& buf) {
    const PassNode& na = host.node(a);
    if (na.type == PassInstrType::kNone) {
        return;
    }
    InstructionWires(na, buf);
    struct Reconnect {
        NodeId pred;
        NodeId succ;
        WireId wire;
    };
    std::vector<Reconnect> reconnect;
    reconnect.reserve(buf.size());
    for (WireId w : buf) {
        const auto pred = dag.wire_predecessor(a, w);
        const auto succ = dag.wire_successor(b, w);
        if (!pred || !succ) {
            throw DagPassError("Wire " + WireLabel(w) + " of node " + std::to_string(a) + "/" + std::to_string(b)
                               + " is broken.");
        }
        reconnect.push_back(Reconnect{*pred, *succ, w});
    }
    RemoveNode(dag, host, a);
    RemoveNode(dag, host, b);
    for (const Reconnect& r : reconnect) {
        dag.add_edge(r.pred, r.succ, r.wire);
    }
}

bool CancelOnWire(DagStore& dag, DagPassHost& host, WireId wire, IdMap<std::uint8_t>& removed,
                  std::vector<NodeId>& ops, std::vector<NodeId>& gates, std::vector<WireId>& buf) {
    NodesOnWire(dag, wire, ops);
    gates.clear();
    for (NodeId nid : ops) {
        if (!removed.contains(nid) && host.node(nid).is_gate()) {
            gates.push_back(nid);
        }
    }
    if (gates.size() < 2) {
        return false;
    }
    bool canceled = false;
    std::size_t i = 0;
    while (i + 1 < gates.size()) {
        const NodeId a = gates[i];
        const NodeId b = gates[i + 1];
        if (removed.contains(a)) {
            i += 1;
            continue;
        }
        if (removed.contains(b)) {
            i += 2;
            continue;
        }
        const PassNode& na = host.node(a);
        const PassNode& nb = host.node(b);
        if (na.type == PassInstrType::kNone || nb.type == PassInstrType::kNone || !na.has_gate || !nb.has_gate
            || !SameBits(na.qubits, nb.qubits)) {
            i += 1;
            continue;
        }
        if (GatesCancel(na, nb) && ConsecutiveOnAllWires(dag, host, a, b, buf)) {
            RemoveNodePair(dag, host, a, b, buf);
            removed[a] = 1;
            removed[b] = 1;
            gates.erase(gates.begin() + static_cast<std::ptrdiff_t>(i),
                        gates.begin() + static_cast<std::ptrdiff_t>(i + 2));
            canceled = true;
        } else {
            i += 1;
        }
    }
    return canceled;
}

// ---------------------------------------------------------------------------
// CommutationAnalysis
// ---------------------------------------------------------------------------

bool QubitsIntersect(const DagStore::WireList* a, const DagStore::WireList* b) noexcept {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    for (WireId x : *a) {
        if (x % 2 != 0) {
            continue;
        }
        for (WireId y : *b) {
            if (x == y) {
                return true;
            }
        }
    }
    return false;
}

bool OpsCommute(DagStore& dag, DagPassHost& host, NodeId ida, NodeId idb) {
    const PassNode& a = host.node(ida);
    const PassNode& b = host.node(idb);
    if (a.type == PassInstrType::kNone || b.type == PassInstrType::kNone) {
        return false;
    }
    if (!a.has_gate || !b.has_gate) {
        return false;
    }
    const std::int32_t na = a.name;
    const std::int32_t nb = b.name;
    const auto& qa = a.qubits;
    const auto& qb = b.qubits;

    // 1. 不相交的比特
    if (!QubitsIntersect(dag.node_wires(ida), dag.node_wires(idb))) {
        return true;
    }
    // 2. 都是对角门
    if (InSet(kZFamily, na) && InSet(kZFamily, nb)) {
        return true;
    }
    // 3. 相同的 CNOT 类操作；共享控制位或目标位的 CNOT
    if (na == nb && InSet(kCnotLike, na) && SameBits(qa, qb)) {
        return true;
    }
    if (na == kNameCx && nb == kNameCx && qa.size() == 2 && qb.size() == 2 && (qa[0] == qb[0] || qa[1] == qb[1])) {
        return true;
    }
    // 4. X 族作用在 CNOT 的目标位
    if (na == kNameCx && InSet(kXFamily, nb) && qa.size() == 2 && qb.size() == 1 && qb[0] == qa[1]) {
        return true;
    }
    if (nb == kNameCx && InSet(kXFamily, na) && qb.size() == 2 && qa.size() == 1 && qa[0] == qb[1]) {
        return true;
    }
    // 5. Z 族作用在 CNOT 的控制位
    if (na == kNameCx && InSet(kZFamily, nb) && qa.size() == 2 && qb.size() == 1 && qb[0] == qa[0]) {
        return true;
    }
    return nb == kNameCx && InSet(kZFamily, na) && qb.size() == 2 && qa.size() == 1 && qa[0] == qb[0];
}

// 两个节点共享的线：sorted(qubits 交集) 在前、sorted(clbits 交集) 在后。
void SharedWires(DagStore& dag, NodeId a, NodeId b, SmallVector<WireId, 4>& out) {
    out.clear();
    const DagStore::WireList* wa = dag.node_wires(a);
    const DagStore::WireList* wb = dag.node_wires(b);
    if (wa == nullptr || wb == nullptr) {
        return;
    }
    for (WireId x : *wa) {
        bool shared = false;
        for (WireId y : *wb) {
            if (x == y) {
                shared = true;
                break;
            }
        }
        if (shared && std::find(out.begin(), out.end(), x) == out.end()) {
            out.push_back(x);
        }
    }
    std::sort(out.begin(), out.end(), [](WireId x, WireId y) {
        const bool cx = x % 2 != 0;
        const bool cy = y % 2 != 0;
        if (cx != cy) {
            return !cx;
        }
        return x < y;
    });
}

void RemoveEdgeChecked(DagStore& dag, NodeId s, NodeId t, WireId w) {
    try {
        dag.remove_edge(s, t, w);
    } catch (const DagEdgeNotFound&) {
        throw DagKeyError("Edge " + std::to_string(s) + "->" + std::to_string(t) + " on '" + WireLabel(w)
                          + "' not found.");
    }
}

bool SwapTwoNodes(DagStore& dag, WireId wire, NodeId a, NodeId b) {
    SmallVector<WireId, 4> shared;
    SharedWires(dag, a, b, shared);
    if (std::find(shared.begin(), shared.end(), wire) == shared.end()) {
        return false;
    }
    for (WireId w : shared) {
        if (!dag.has_edge(a, b, w)) {
            return false;
        }
    }
    struct Rewire {
        NodeId pred;
        NodeId succ;
        WireId wire;
    };
    SmallVector<Rewire, 4> rewires;
    for (WireId w : shared) {
        const auto pred = dag.wire_predecessor(a, w);
        const auto succ = dag.wire_successor(b, w);
        if (!pred || !succ) {
            return false;
        }
        rewires.push_back(Rewire{*pred, *succ, w});
    }
    dag.ensure_topological_positions();
    TransactionScope txn(dag);
    for (const Rewire& r : rewires) {
        RemoveEdgeChecked(dag, r.pred, a, r.wire);
        RemoveEdgeChecked(dag, a, b, r.wire);
        RemoveEdgeChecked(dag, b, r.succ, r.wire);
    }
    for (const Rewire& r : rewires) {
        dag.add_edge(r.pred, b, r.wire, true);
        dag.add_edge(b, a, r.wire, true);
        dag.add_edge(a, r.succ, r.wire, true);
    }
    if (!dag.reverse_adjacent_edge_order(a, b)) {
        dag.rollback(txn.savepoint());
        return false;
    }
    return true;
}

bool SwapChain(DagStore& dag, WireId wire, std::vector<NodeId>& gates, std::size_t target, std::size_t source) {
    const std::vector<NodeId> original = gates;
    TransactionScope txn(dag);
    for (std::size_t k = source; k > target + 1; --k) {
        if (!SwapTwoNodes(dag, wire, gates[k - 1], gates[k])) {
            dag.rollback(txn.savepoint());
            gates = original;
            return false;
        }
        std::swap(gates[k - 1], gates[k]);
    }
    return true;
}

bool CanSkipAll(DagStore& dag, DagPassHost& host, const std::vector<NodeId>& gates, std::size_t start,
                std::size_t end) {
    const NodeId target = gates[start];
    for (std::size_t k = start + 1; k < end; ++k) {
        if (!OpsCommute(dag, host, target, gates[k])) {
            return false;
        }
    }
    return true;
}

bool ReorderOnWire(DagStore& dag, DagPassHost& host, WireId wire, std::int64_t max_lookahead,
                   std::vector<NodeId>& ops, std::vector<NodeId>& gates) {
    GateNodesOnWire(dag, host, wire, ops, gates);
    if (gates.size() < 2) {
        return false;
    }
    bool reordered = false;
    std::size_t i = 0;
    while (i < gates.size()) {
        const PassNode& ni = host.node(gates[i]);
        const std::int32_t name_i = ni.name;
        std::ptrdiff_t best_j = -1;
        // range(i + 1, min(i + MAX_LOOKAHEAD + 1, len(gate_nodes)))
        const std::int64_t stop =
            std::min<std::int64_t>(static_cast<std::int64_t>(i) + max_lookahead + 1,
                                   static_cast<std::int64_t>(gates.size()));
        for (std::int64_t j = static_cast<std::int64_t>(i) + 1; j < stop; ++j) {
            const PassNode& nj = host.node(gates[static_cast<std::size_t>(j)]);
            if (nj.name == name_i && SameBits(nj.qubits, ni.qubits)
                && CanSkipAll(dag, host, gates, i, static_cast<std::size_t>(j))) {
                best_j = j;
                break;
            }
        }
        if (best_j > static_cast<std::ptrdiff_t>(i) + 1) {
            if (SwapChain(dag, wire, gates, i, static_cast<std::size_t>(best_j))) {
                reordered = true;
                continue;
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    return reordered;
}

// ---------------------------------------------------------------------------
// RotationMerging
// ---------------------------------------------------------------------------

// rotation_merging.ROTATION_FAMILIES：rx -> rx, ry -> ry, rz / p / u1 -> rz。
std::int32_t RotationFamily(std::int32_t name) noexcept {
    switch (name) {
        case kNameRx:
            return kNameRx;
        case kNameRy:
            return kNameRy;
        case kNameRz:
        case kNameP:
        case kNameU1:
            return kNameRz;
        default:
            return kNameNone;
    }
}

bool MergeOnWire(DagStore& dag, DagPassHost& host, WireId wire, std::vector<NodeId>& ops,
                 std::vector<NodeId>& gates) {
    GateNodesOnWire(dag, host, wire, ops, gates);
    if (gates.size() < 2) {
        return false;
    }
    bool merged = false;
    std::size_t i = 0;
    while (i + 1 < gates.size()) {
        const NodeId a = gates[i];
        const NodeId b = gates[i + 1];
        const PassNode& na = host.node(a);
        const PassNode& nb = host.node(b);
        if (na.type == PassInstrType::kNone || nb.type == PassInstrType::kNone || !na.has_gate || !nb.has_gate) {
            i += 1;
            continue;
        }
        const std::int32_t family_a = RotationFamily(na.name);
        const std::int32_t family_b = RotationFamily(nb.name);
        if (family_a == kNameNone || family_b == kNameNone || family_a != family_b) {
            i += 1;
            continue;
        }
        // _get_rotation_angle：第一个参数为具体数值时取 float(params[0])。
        if (na.params.empty() || na.concrete[0] == 0 || nb.params.empty() || nb.concrete[0] == 0) {
            i += 1;
            continue;
        }
        if (!dag.has_edge(a, b, wire)) {
            i += 1;
            continue;
        }
        const double merged_angle = na.params[0] + nb.params[0];
        std::int32_t gate_name = kNameP;  // p, u1
        if (na.name == kNameRx || na.name == kNameRy || na.name == kNameRz) {
            gate_name = na.name;
        }
        host.substitute_rotation(a, gate_name, merged_angle);
        RemoveNodeOnWire(dag, host, b, wire);
        gates.erase(gates.begin() + static_cast<std::ptrdiff_t>(i + 1));
        merged = true;
    }
    return merged;
}

// ---------------------------------------------------------------------------
// SingleQubitGateFusion（基矢模式）
// ---------------------------------------------------------------------------

bool IsFusible(DagPassHost& host, NodeId nid) {
    const PassNode& n = host.node(nid);
    return n.type != PassInstrType::kNone && n.has_gate && InSet(kSingleQubit, n.name) && n.num_qubits == 1
           && n.all_concrete();
}

bool ResynthOnWire(DagStore& dag, DagPassHost& host, WireId wire, std::vector<NodeId>& nodes,
                   std::vector<NodeId>& block, std::vector<WireId>& buf) {
    NodesOnWire(dag, wire, nodes);
    bool changed = false;
    std::size_t i = 0;
    const std::size_t n = nodes.size();
    while (i < n) {
        if (!IsFusible(host, nodes[i])) {
            i += 1;
            continue;
        }
        std::size_t j = i;
        while (j + 1 < n && IsFusible(host, nodes[j + 1]) && dag.has_edge(nodes[j], nodes[j + 1], wire)) {
            j += 1;
        }
        block.assign(nodes.begin() + static_cast<std::ptrdiff_t>(i), nodes.begin() + static_cast<std::ptrdiff_t>(j + 1));
        i = j + 1;

        const std::optional<std::size_t> replacement = host.resynthesize(block);
        if (!replacement) {
            continue;
        }
        const std::size_t m = *replacement;
        for (std::size_t k = 0; k < std::min(m, block.size()); ++k) {
            host.apply_replacement(block[k], k);
        }
        for (std::size_t k = m; k < block.size(); ++k) {
            RemoveNodeAndReconnect(dag, host, block[k], buf);
        }
        changed = true;
    }
    return changed;
}

// ---------------------------------------------------------------------------
// EliminateDiagonalBeforeMeasurement
// ---------------------------------------------------------------------------

bool EliminateOnWire(DagStore& dag, DagPassHost& host, WireId wire, std::vector<NodeId>& ops) {
    NodesOnWire(dag, wire, ops);
    // 每个节点在这条线上的后继（在开始时计算，之后不更新，与 Python 相同）
    std::vector<SmallVector<NodeId, 2>> succ_on_wire(ops.size());
    for (std::size_t k = 0; k < ops.size(); ++k) {
        for (const DagAdj& e : dag.out_adj(ops[k])) {
            if (e.wire == wire) {
                succ_on_wire[k].push_back(e.nbr);
            }
        }
    }
    bool eliminated = false;
    std::vector<NodeId> all_succs;
    for (std::size_t k = 0; k < ops.size(); ++k) {
        const NodeId nid = ops[k];
        const PassNode& node = host.node(nid);
        if (!node.is_gate() || !node.has_gate) {
            continue;
        }
        if (!InSet(kDiagonal, node.name)) {
            continue;
        }
        const auto& successors = succ_on_wire[k];
        if (successors.size() != 1) {
            continue;
        }
        if (!host.node(successors[0]).is_measure()) {
            continue;
        }
        all_succs.clear();
        for (const DagAdj& e : dag.out_adj(nid)) {
            all_succs.push_back(e.nbr);
        }
        bool blocked = false;
        for (NodeId s : all_succs) {
            if (host.node(s).is_gate() && s != successors[0]) {
                blocked = true;
                break;
            }
        }
        if (!blocked) {
            RemoveNodeOnWire(dag, host, nid, wire);
            eliminated = true;
        }
    }
    return eliminated;
}

}  // namespace

// ---------------------------------------------------------------------------
// 入口
// ---------------------------------------------------------------------------

void RunGateCancellation(DagStore& dag, DagPassHost& host) {
    IdMap<std::uint8_t> removed;
    std::vector<NodeId> ops;
    std::vector<NodeId> gates;
    std::vector<WireId> buf;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
            if (CancelOnWire(dag, host, QubitWire(q), removed, ops, gates, buf)) {
                changed = true;
            }
        }
    }
}

void RunCommutationAnalysis(DagStore& dag, DagPassHost& host, std::int64_t max_lookahead) {
    std::vector<NodeId> ops;
    std::vector<NodeId> gates;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
            if (ReorderOnWire(dag, host, QubitWire(q), max_lookahead, ops, gates)) {
                changed = true;
            }
        }
    }
}

void RunRotationMerging(DagStore& dag, DagPassHost& host) {
    std::vector<NodeId> ops;
    std::vector<NodeId> gates;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
            if (MergeOnWire(dag, host, QubitWire(q), ops, gates)) {
                changed = true;
            }
        }
    }
}

void RunSingleQubitResynthesis(DagStore& dag, DagPassHost& host) {
    std::vector<NodeId> nodes;
    std::vector<NodeId> block;
    std::vector<WireId> buf;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
            if (ResynthOnWire(dag, host, QubitWire(q), nodes, block, buf)) {
                changed = true;
            }
        }
    }
}

void RunEliminateDiagonalBeforeMeasurement(DagStore& dag, DagPassHost& host) {
    std::vector<NodeId> ops;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
            if (EliminateOnWire(dag, host, QubitWire(q), ops)) {
                changed = true;
            }
        }
    }
}

bool RunRemoveIdentityGates(DagStore& dag, DagPassHost& host) {
    std::vector<NodeId> ops;
    std::vector<NodeId> doomed;
    for (std::int64_t q = 0; q < dag.num_qubits(); ++q) {
        NodesOnWire(dag, QubitWire(q), ops);
        for (NodeId nid : ops) {
            const PassNode& n = host.node(nid);
            if (!n.is_gate() || !n.has_gate) {
                continue;
            }
            // is_identity_gate(gate)
            if (n.num_qubits != 1 || n.params.empty() || !n.all_concrete()) {
                continue;
            }
            bool identity = false;
            if (InSet(kIdentityRotations, n.name)) {
                const double x = n.params[0];
                if (std::isinf(x)) {
                    return false;  // math.fmod(inf, 2π) 抛 ValueError
                }
                const double t = std::fmod(x, 2.0 * kPi);
                identity = std::fabs(t) < kIdentityAtol || std::fabs(std::fabs(t) - 2.0 * kPi) < kIdentityAtol;
            } else if (InSet(kMatrixChecked, n.name)) {
                identity = host.matrix_identity(nid);
            }
            if (identity) {
                doomed.push_back(nid);
            }
        }
    }
    std::vector<WireId> buf;
    for (NodeId nid : doomed) {
        if (dag.has_node(nid)) {
            RemoveNodeAndReconnect(dag, host, nid, buf);
        }
    }
    return true;
}

}  // namespace quspect::native::ir
