// DagStore（DAGCircuit 的 C++ 图存储）的 gtest：按线路构图、拓扑序与插入顺序、沿线遍历、
// 删除并重连、事务回滚恢复原边序、增量拓扑编号的环检测、分层与关键路径。
// 语义参考 quspect.dag.dag_circuit 中的纯 Python 内核（tests/dag/test_dag_core_parity.py
// 另有与 Python 内核的随机对拍）。
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/ir/dag_store.hpp"

namespace {

using quspect::native::ir::ClbitWire;
using quspect::native::ir::DagCycleError;
using quspect::native::ir::DagEdgeNotFound;
using quspect::native::ir::DagKeyError;
using quspect::native::ir::DagNodeKind;
using quspect::native::ir::DagStore;
using quspect::native::ir::NodeId;
using quspect::native::ir::QubitWire;
using quspect::native::ir::WireId;

void AddNode(DagStore& d, NodeId id, DagNodeKind kind, std::vector<WireId> wires) {
    d.add_node(id, kind, wires.data(), wires.size());
}

// 与 CircuitCompiler 相同的构图：INPUT q0..q(n-1)，各门，OUTPUT。返回门节点编号。
std::vector<NodeId> BuildCircuit(DagStore& d, int nq, const std::vector<std::vector<int>>& gates) {
    NodeId next = 0;
    std::vector<NodeId> last(static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        AddNode(d, next, DagNodeKind::kInput, {QubitWire(q)});
        last[static_cast<std::size_t>(q)] = next++;
    }
    std::vector<NodeId> ops;
    for (const auto& g : gates) {
        std::vector<WireId> wires;
        for (int q : g) {
            wires.push_back(QubitWire(q));
        }
        AddNode(d, next, DagNodeKind::kOperation, wires);
        for (int q : g) {
            d.add_edge(last[static_cast<std::size_t>(q)], next, QubitWire(q));
            last[static_cast<std::size_t>(q)] = next;
        }
        ops.push_back(next++);
    }
    for (int q = 0; q < nq; ++q) {
        AddNode(d, next, DagNodeKind::kOutput, {QubitWire(q)});
        d.add_edge(last[static_cast<std::size_t>(q)], next, QubitWire(q));
        ++next;
    }
    return ops;
}

TEST(DagStore, BuildAndTopologicalOrder) {
    DagStore d;
    auto ops = BuildCircuit(d, 2, {{0}, {0, 1}, {1}});
    EXPECT_EQ(d.num_nodes(), 7u);
    EXPECT_EQ(d.num_edges(), 6u);
    // 每次增删节点 / 边都计一次修改
    EXPECT_EQ(d.version(), 13);
    const auto& topo = d.topological_order();
    EXPECT_EQ(topo, (std::vector<NodeId>{0, 1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(d.wire_successor(ops[0], QubitWire(0)), ops[1]);
    EXPECT_EQ(d.wire_predecessor(ops[2], QubitWire(1)), ops[1]);
    EXPECT_FALSE(d.wire_successor(ops[0], QubitWire(1)).has_value());
    EXPECT_TRUE(d.has_edge(ops[0], ops[1], std::nullopt));
    EXPECT_TRUE(d.has_edge(ops[0], ops[1], QubitWire(0)));
    EXPECT_FALSE(d.has_edge(ops[0], ops[1], QubitWire(1)));
}

TEST(DagStore, WalkWireAndScanFallback) {
    DagStore d;
    auto ops = BuildCircuit(d, 2, {{0}, {0, 1}, {0}});
    std::vector<NodeId> out;
    ASSERT_TRUE(d.walk_wire(QubitWire(0), QubitWire(0), out));
    EXPECT_EQ(out, (std::vector<NodeId>{0, ops[0], ops[1], ops[2], 5}));
    ASSERT_TRUE(d.walk_wire(QubitWire(1), QubitWire(1), out));
    EXPECT_EQ(out, (std::vector<NodeId>{1, ops[1], 6}));
    // 断开一条边后线不再是完整的链：走线失败，扫描拓扑序仍能给出线上的节点
    d.remove_edge(ops[1], ops[2], QubitWire(0));
    EXPECT_FALSE(d.walk_wire(QubitWire(0), QubitWire(0), out));
    d.scan_wire(QubitWire(0), out);
    EXPECT_EQ(out.size(), 5u);
    EXPECT_THROW(d.remove_edge(ops[1], ops[2], QubitWire(0)), DagEdgeNotFound);
}

TEST(DagStore, RemoveNodeAndReconnect) {
    DagStore d;
    auto ops = BuildCircuit(d, 1, {{0}, {0}, {0}});
    const WireId w = QubitWire(0);
    d.remove_node_and_reconnect(ops[1], &w, 1);
    EXPECT_FALSE(d.has_node(ops[1]));
    EXPECT_EQ(d.wire_successor(ops[0], w), ops[2]);
    std::vector<NodeId> out;
    ASSERT_TRUE(d.walk_wire(w, w, out));
    EXPECT_EQ(out, (std::vector<NodeId>{0, ops[0], ops[2], 4}));
    EXPECT_THROW(d.remove_node(ops[1]), DagKeyError);
}

TEST(DagStore, ReAddedNodeGoesToEndOfInsertionOrder) {
    DagStore d;
    AddNode(d, 0, DagNodeKind::kOperation, {});
    AddNode(d, 1, DagNodeKind::kOperation, {});
    AddNode(d, 2, DagNodeKind::kOperation, {});
    d.remove_node(0);
    AddNode(d, 0, DagNodeKind::kOperation, {});
    std::vector<NodeId> ids;
    d.node_ids(ids);
    EXPECT_EQ(ids, (std::vector<NodeId>{1, 2, 0}));
    EXPECT_EQ(d.topological_order(), (std::vector<NodeId>{1, 2, 0}));
    EXPECT_THROW(AddNode(d, 1, DagNodeKind::kOperation, {}), std::invalid_argument);
}

TEST(DagStore, RollbackRestoresEdgeOrder) {
    DagStore d;
    auto ops = BuildCircuit(d, 1, {{0}, {0}});
    const WireId w = QubitWire(0);
    std::vector<quspect::native::ir::DagEdge> before;
    d.edges(before);
    const auto sp = d.begin_transaction();
    // 交换两个门：pred->a->b->succ 变为 pred->b->a->succ
    d.remove_edge(0, ops[0], w);
    d.remove_edge(ops[0], ops[1], w);
    d.remove_edge(ops[1], 3, w);
    d.add_edge(0, ops[1], w, true);
    d.add_edge(ops[1], ops[0], w, true);
    d.add_edge(ops[0], 3, w, true);
    EXPECT_EQ(d.wire_successor(0, w), ops[1]);
    d.rollback(sp);
    d.end_transaction();
    std::vector<quspect::native::ir::DagEdge> after;
    d.edges(after);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(before[i].source, after[i].source);
        EXPECT_EQ(before[i].target, after[i].target);
    }
    EXPECT_EQ(d.out_adj(0)[0].nbr, ops[0]);
    EXPECT_THROW(d.rollback(0), std::logic_error);
}

TEST(DagStore, ReverseAdjacentEdgeOrderDetectsCycles) {
    DagStore d;
    auto ops = BuildCircuit(d, 2, {{0}, {0, 1}, {1}});
    d.ensure_topological_positions();
    const WireId w0 = QubitWire(0);
    // 合法交换：a 与 cx 在 q0 上交换
    d.begin_transaction();
    d.remove_edge(0, ops[0], w0);
    d.remove_edge(ops[0], ops[1], w0);
    d.remove_edge(ops[1], 5, w0);  // 5 = OUTPUT q0
    d.add_edge(0, ops[1], w0, true);
    d.add_edge(ops[1], ops[0], w0, true);
    d.add_edge(ops[0], 5, w0, true);
    EXPECT_TRUE(d.reverse_adjacent_edge_order(ops[0], ops[1]));
    d.end_transaction();
    EXPECT_EQ(d.topological_order().size(), 7u);
    // 制造环：ops[2] -> ops[1]（ops[1] -> ops[2] 已存在）
    d.add_edge(ops[2], ops[1], QubitWire(5), true);
    EXPECT_FALSE(d.reverse_adjacent_edge_order(ops[1], ops[2]));
    EXPECT_THROW(d.topological_order(), DagCycleError);
}

TEST(DagStore, EdgeToMissingNodeRaisesKeyErrorInTopologicalOrder) {
    DagStore d;
    AddNode(d, 0, DagNodeKind::kOperation, {});
    d.add_edge(0, 42, QubitWire(0));
    try {
        d.topological_order();
        FAIL() << "expected DagKeyError";
    } catch (const DagKeyError& e) {
        EXPECT_TRUE(e.has_int_key());
        EXPECT_EQ(e.key(), 42);
    }
}

TEST(DagStore, LayersAndCriticalPath) {
    DagStore d;
    BuildCircuit(d, 3, {{0}, {1}, {0, 1}, {2}, {1, 2}});
    const auto layers = d.layers();
    // INPUT / {q0, q1, q2 门} / {cx01} / {cx12} / OUTPUT ...
    EXPECT_EQ(layers[0].size(), 3u);
    EXPECT_EQ(d.critical_path_length(), static_cast<std::int64_t>(layers.size()) - 2);
    DagStore empty;
    EXPECT_EQ(empty.layers().size(), 1u);
    EXPECT_EQ(empty.critical_path_length(), 0);
}

TEST(DagStore, AmbiguousWireDisablesWalk) {
    DagStore d;
    BuildCircuit(d, 1, {{0}});
    AddNode(d, 100, DagNodeKind::kInput, {QubitWire(0)});
    std::vector<NodeId> out;
    EXPECT_FALSE(d.walk_wire(QubitWire(0), QubitWire(0), out));
    EXPECT_TRUE(d.on_member_wire(100, QubitWire(0)));
    EXPECT_FALSE(d.on_member_wire(100, ClbitWire(0)));
}

}  // namespace
