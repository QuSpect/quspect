#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quspect_native/ir/id_map.hpp"
#include "quspect_native/ir/small_vector.hpp"

namespace quspect::native::ir {

// DAGCircuit 的图存储引擎。
//
// 与 Python 参考实现（quspect.dag.dag_circuit 中的纯 Python 内核）逐项语义一致：
// 每个节点按边序号（全局插入顺序）保存出边 / 入边小表，局部查询 O(度)；拓扑序用 Kahn 算法
// 并缓存到下一次结构修改；交换相邻节点的重排 pass 用增量拓扑编号（Pearce–Kelly）做局部
// 环检测；事务日志支持把边的增删精确回滚到原来的边序。
//
// 节点编号（NodeId）是任意 64 位整数；线（WireId）由调用方分配：规范线标签 "q<i>" 为 2i，
// "c<i>" 为 2i+1，其他标签为负数。节点的 "成员线"（qubits/clbits 对应的规范线）在加入时
// 记录，用于沿线遍历时校验节点确实位于该线上。

using NodeId = std::int64_t;
using WireId = std::int64_t;
using EdgeSeq = std::int64_t;

enum class DagNodeKind : std::uint8_t { kInput = 0, kOutput = 1, kOperation = 2 };

// 规范线标签的编号。
constexpr WireId QubitWire(std::int64_t q) noexcept {
    return 2 * q;
}
constexpr WireId ClbitWire(std::int64_t c) noexcept {
    return 2 * c + 1;
}

// 邻接表项：(边序号, 邻居节点, 线)。
struct DagAdj {
    EdgeSeq seq;
    NodeId nbr;
    WireId wire;
};

struct DagEdge {
    NodeId source;
    NodeId target;
    WireId wire;
};

// 访问不存在的节点 / 边。has_int_key 为真时对应 Python 的 KeyError(key)（整数键），
// 否则对应 KeyError(message)。
class DagKeyError : public std::exception {
public:
    explicit DagKeyError(NodeId key) : key_(key), has_int_key_(true), message_(std::to_string(key)) {
    }
    explicit DagKeyError(std::string message) : message_(std::move(message)) {
    }
    const char* what() const noexcept override {
        return message_.c_str();
    }
    NodeId key() const noexcept {
        return key_;
    }
    bool has_int_key() const noexcept {
        return has_int_key_;
    }

private:
    NodeId key_ = 0;
    bool has_int_key_ = false;
    std::string message_;
};

// 找不到要删除的边（调用方据此拼出带线标签的错误信息）。
class DagEdgeNotFound : public std::exception {
public:
    const char* what() const noexcept override {
        return "edge not found";
    }
};

// 拓扑排序发现环。
class DagCycleError : public std::runtime_error {
public:
    DagCycleError()
        : std::runtime_error(
              "Cycle detected in DAG \xe2\x80\x94 this should not happen for circuits built from sequential "
              "instructions.") {
    }
};

// DagStore 的完整状态（用于拷贝 / 序列化）：节点按插入顺序、边按序号，外加线的簿记、
// 版本号、增量拓扑编号与事务日志。拓扑序缓存不在其中（按需重算）。
struct DagState {
    struct Node {
        NodeId id;
        DagNodeKind kind;
        std::vector<WireId> wires;
    };
    struct Edge {
        EdgeSeq seq;
        DagEdge edge;
    };
    struct JournalEntry {
        std::uint8_t kind;  // 0 add, 1 remove, 2 pos, 3 位置编号失效
        std::int64_t key;
        DagEdge edge;
        std::int64_t old_pos;
        bool has_positions;
        std::vector<std::pair<NodeId, std::int64_t>> positions;
    };
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    EdgeSeq next_seq = 0;
    std::vector<std::pair<WireId, NodeId>> wire_input;
    std::vector<WireId> ambiguous_wires;
    std::vector<std::pair<WireId, std::int64_t>> wire_count;
    std::int64_t num_qubits = 0;
    std::int64_t num_clbits = 0;
    std::int64_t version = 0;
    bool has_positions = false;
    std::vector<std::pair<NodeId, std::int64_t>> positions;
    std::int64_t next_pos = 0;
    bool journal_open = false;
    std::vector<JournalEntry> journal;
    std::int64_t txn_depth = 0;
};

class DagStore {
public:
    using AdjList = SmallVector<DagAdj, 3>;
    using WireList = SmallVector<WireId, 4>;

    DagStore() = default;

    // ---- 计数 / 版本 ----
    std::int64_t num_qubits() const noexcept {
        return num_qubits_;
    }
    std::int64_t num_clbits() const noexcept {
        return num_clbits_;
    }
    void set_num_qubits(std::int64_t n) noexcept {
        num_qubits_ = n;
    }
    void set_num_clbits(std::int64_t n) noexcept {
        num_clbits_ = n;
    }
    std::size_t num_nodes() const noexcept {
        return num_alive_;
    }
    std::size_t num_edges() const noexcept {
        return num_edges_;
    }
    // 每次结构修改（增删节点 / 边）以及 bump_version() 都会增加。
    std::int64_t version() const noexcept {
        return version_;
    }
    void bump_version() noexcept {
        ++version_;
    }

    // ---- 节点 ----
    bool has_node(NodeId id) const noexcept {
        return slot_of_.contains(id);
    }
    // 加入节点；wires 为节点所在的线（qubits 在前、clbits 在后，规范编号）。
    // 抛出 std::invalid_argument：编号已存在。
    void add_node(NodeId id, DagNodeKind kind, const WireId* wires, std::size_t num_wires);
    // 删除节点及其全部关联边（不重新连接邻居）。抛出 DagKeyError：节点不存在。
    void remove_node(NodeId id);
    // 删除节点并在 wires 的每条线上把前驱与后继重新连起来（wires 已去重、按顺序）。
    void remove_node_and_reconnect(NodeId id, const WireId* wires, std::size_t num_wires);
    // 节点种类；节点不存在时抛出 DagKeyError。
    DagNodeKind kind(NodeId id) const;
    // 按插入顺序（Python dict 顺序）列出全部节点。
    void node_ids(std::vector<NodeId>& out) const;

    // ---- 边 ----
    void add_edge(NodeId source, NodeId target, WireId wire, bool defer_order_check = false);
    // 删除第一条匹配的边。抛出 DagEdgeNotFound。
    void remove_edge(NodeId source, NodeId target, WireId wire);
    // 按边序号列出全部边。
    void edges(std::vector<DagEdge>& out) const;

    const AdjList& out_adj(NodeId id) const noexcept;
    const AdjList& in_adj(NodeId id) const noexcept;
    bool has_edge(NodeId source, NodeId target, std::optional<WireId> wire) const noexcept;
    // 线 wire 上的前驱 / 后继（多条匹配时取最后加入的一条）。
    std::optional<NodeId> wire_predecessor(NodeId id, WireId wire) const noexcept;
    std::optional<NodeId> wire_successor(NodeId id, WireId wire) const noexcept;

    // ---- 事务 ----
    std::int64_t begin_transaction();
    // 抛出 std::logic_error：没有打开的事务。
    void rollback(std::int64_t savepoint);
    void end_transaction() noexcept;

    // ---- 拓扑 ----
    // 缓存的拓扑序（Kahn 算法，队列按节点插入顺序初始化）。抛出 DagCycleError / DagKeyError。
    const std::vector<NodeId>& topological_order();
    void ensure_topological_positions();
    // second 被移到 first 之前后修复拓扑编号；产生环时返回 false。
    bool reverse_adjacent_edge_order(NodeId first, NodeId second);
    std::vector<std::vector<NodeId>> layers();
    std::int64_t critical_path_length();

    // ---- 沿线查询 ----
    // 从线的 INPUT 节点沿 wire 标记的边走到底；member 为校验节点成员资格所用的规范线
    // （标签无法解析时传 std::nullopt）。线不是简单链时返回 false。
    bool walk_wire(WireId wire, std::optional<WireId> member, std::vector<NodeId>& out) const;
    // walk_wire 失败时退回扫描拓扑序：列出成员线含 member 的全部节点。
    void scan_wire(WireId member, std::vector<NodeId>& out);

    // 节点 id 是否位于成员线 member 上（加入时记录）。
    bool on_member_wire(NodeId id, WireId member) const noexcept;
    // 节点加入时记录的成员线（qubits 在前、clbits 在后）；节点不存在时返回 nullptr。
    const WireList* node_wires(NodeId id) const noexcept;

    // ---- 状态导出 / 导入（拷贝、序列化） ----
    DagState export_state() const;
    // 用 state 替换当前全部内容。抛出 std::invalid_argument：state 不自洽。
    void import_state(const DagState& state);

private:
    struct NodeSlot {
        NodeId id = 0;
        DagNodeKind kind = DagNodeKind::kOperation;
        bool alive = false;
        WireList wires;
    };
    struct EdgeRec {
        DagEdge edge{};
        bool alive = false;
    };
    struct Adjacency {
        AdjList out;
        AdjList in;
    };
    using PosMap = IdMap<std::int64_t>;
    enum class JournalKind : std::uint8_t { kAdd, kRemove, kPos, kPosInvalidate };
    struct JournalEntry {
        JournalKind kind;
        std::int64_t key;
        DagEdge edge;
        std::int64_t old_pos;
        std::shared_ptr<PosMap> positions;
    };

    void touch() noexcept {
        ++version_;
        topo_valid_ = false;
    }
    void link(EdgeSeq seq, const DagEdge& edge);
    DagEdge unlink(EdgeSeq seq);
    std::optional<EdgeSeq> find_edge_seq(NodeId source, NodeId target, WireId wire) const noexcept;
    void invalidate_positions();
    PosMap& ensure_positions();
    void compact_order();
    const NodeSlot* slot(NodeId id) const noexcept;

    std::vector<NodeSlot> slots_;  // 插入顺序；删除的节点留空位，定期压缩
    IdMap<std::int32_t> slot_of_;  // 节点编号 -> slots_ 下标
    std::size_t num_alive_ = 0;

    std::vector<EdgeRec> edges_;  // 边序号 -> 边
    std::size_t num_edges_ = 0;
    EdgeSeq next_seq_ = 0;
    IdMap<Adjacency> adj_;  // 节点编号 -> 出边 / 入边（也允许指向不存在的节点）

    IdMap<NodeId> wire_input_;
    IdMap<std::uint8_t> ambiguous_wires_;
    IdMap<std::int64_t> wire_count_;

    std::int64_t num_qubits_ = 0;
    std::int64_t num_clbits_ = 0;

    std::vector<NodeId> topo_cache_;
    bool topo_valid_ = false;
    std::int64_t version_ = 0;

    std::shared_ptr<PosMap> pos_;
    std::int64_t next_pos_ = 0;

    bool journal_open_ = false;
    std::vector<JournalEntry> journal_;
    std::int64_t txn_depth_ = 0;
};

}  // namespace quspect::native::ir
