// DAG 优化 pass（ir/dag_passes）的 gtest：用纯 C++ 宿主在 DagStore 上构图，检查各 pass 的结果
// 与 quspect.dag.passes 中 Python 实现的语义一致（门的消去、对易重排、旋转合并、重综合的替换 /
// 删除、测量前对角门消除、恒等门移除）。与 Python 实现的逐项对拍见 tests/dag/passes/test_native_passes.py。
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/ir/dag_passes.hpp"
#include "quspect_native/ir/dag_store.hpp"

namespace {

using quspect::native::ir::DagKeyError;
using quspect::native::ir::DagNodeKind;
using quspect::native::ir::DagPassHost;
using quspect::native::ir::DagStore;
using quspect::native::ir::InternGateName;
using quspect::native::ir::NodeId;
using quspect::native::ir::PassInstrType;
using quspect::native::ir::PassNode;
using quspect::native::ir::QubitWire;
using quspect::native::ir::WireId;

struct Gate {
    std::string name;
    std::vector<std::int64_t> qubits;
    std::vector<double> params;
    bool measure = false;
};

PassNode MakeNode(const Gate& g) {
    PassNode n;
    n.operation = true;
    n.type = g.measure ? PassInstrType::kMeasure : PassInstrType::kGate;
    n.has_gate = !g.measure;
    n.name = g.measure ? -1 : InternGateName(g.name);
    n.num_qubits = static_cast<std::int64_t>(g.qubits.size());
    for (std::int64_t q : g.qubits) {
        n.qubits.push_back(q);
    }
    if (g.measure) {
        for (std::int64_t q : g.qubits) {
            n.clbits.push_back(q);
        }
    }
    for (double p : g.params) {
        n.params.push_back(p);
        n.concrete.push_back(1);
    }
    return n;
}

// 测试宿主：节点信息放在 map 中；重综合把任意单比特串换成一个 "u"。
class TestHost : public DagPassHost {
public:
    std::map<NodeId, PassNode> nodes;
    int substitutions = 0;
    std::optional<std::size_t> replacement_size;

    const PassNode& node(NodeId id) override {
        auto it = nodes.find(id);
        if (it == nodes.end()) {
            throw DagKeyError(id);
        }
        return it->second;
    }
    void node_removed(NodeId id) override {
        nodes.erase(id);
    }
    void substitute_rotation(NodeId id, std::int32_t gate_name, double angle) override {
        PassNode& n = nodes.at(id);
        n.name = gate_name;
        n.params.clear();
        n.params.push_back(angle);
        n.concrete.clear();
        n.concrete.push_back(1);
        ++substitutions;
    }
    bool matrix_identity(NodeId id) override {
        const PassNode& n = nodes.at(id);
        return n.params.size() == 3 && n.params[0] == 0.0 && n.params[1] + n.params[2] == 0.0;
    }
    std::optional<std::size_t> resynthesize(const std::vector<NodeId>& block) override {
        // 已经是 u3 的串不再替换（真实宿主只在严格更短时替换，pass 因此收敛）
        bool all_u3 = true;
        for (NodeId id : block) {
            all_u3 = all_u3 && nodes.at(id).name == InternGateName("u3");
        }
        if (block.size() < 2 || all_u3) {
            return std::nullopt;
        }
        return replacement_size;
    }
    void apply_replacement(NodeId id, std::size_t) override {
        PassNode& n = nodes.at(id);
        n.name = InternGateName("u3");
        n.params.clear();
        n.concrete.clear();
        ++substitutions;
    }
};

// 与 CircuitCompiler 相同的构图：INPUT q0..q(n-1)，各门，OUTPUT。返回门节点编号。
std::vector<NodeId> Build(DagStore& d, TestHost& host, int nq, const std::vector<Gate>& gates) {
    d.set_num_qubits(nq);
    NodeId next = 0;
    std::vector<NodeId> last(static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        const WireId w = QubitWire(q);
        d.add_node(next, DagNodeKind::kInput, &w, 1);
        host.nodes[next] = PassNode{};
        last[static_cast<std::size_t>(q)] = next++;
    }
    std::vector<NodeId> ops;
    for (const Gate& g : gates) {
        std::vector<WireId> wires;
        for (std::int64_t q : g.qubits) {
            wires.push_back(QubitWire(q));
        }
        const NodeId id = next++;
        d.add_node(id, DagNodeKind::kOperation, wires.data(), wires.size());
        host.nodes[id] = MakeNode(g);
        for (std::int64_t q : g.qubits) {
            d.add_edge(last[static_cast<std::size_t>(q)], id, QubitWire(q));
            last[static_cast<std::size_t>(q)] = id;
        }
        ops.push_back(id);
    }
    for (int q = 0; q < nq; ++q) {
        const WireId w = QubitWire(q);
        d.add_node(next, DagNodeKind::kOutput, &w, 1);
        host.nodes[next] = PassNode{};
        d.add_edge(last[static_cast<std::size_t>(q)], next, w);
        ++next;
    }
    return ops;
}

// 每条线上的门名序列。
std::vector<std::string> WireNames(DagStore& d, TestHost& host, int q) {
    std::vector<NodeId> ids;
    EXPECT_TRUE(d.walk_wire(QubitWire(q), QubitWire(q), ids));
    std::vector<std::string> out;
    static const std::map<std::int32_t, std::string> kNames = [] {
        std::map<std::int32_t, std::string> m;
        for (const char* s : {"h", "x", "z", "s", "sdg", "t", "tdg", "rx", "rz", "p", "cx", "cz", "u3", "u1"}) {
            m[InternGateName(s)] = s;
        }
        return m;
    }();
    for (NodeId id : ids) {
        const PassNode& n = host.node(id);
        if (n.type == PassInstrType::kMeasure) {
            out.push_back("measure");
        } else if (n.operation) {
            out.push_back(kNames.at(n.name));
        }
    }
    return out;
}

TEST(DagPasses, CancelsSelfInverseAndInversePairs) {
    DagStore d;
    TestHost host;
    Build(d, host, 2, {{"h", {0}}, {"h", {0}}, {"s", {1}}, {"sdg", {1}}, {"cx", {0, 1}}, {"cx", {0, 1}}, {"t", {0}}});
    RunGateCancellation(d, host);
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"t"}));
    EXPECT_TRUE(WireNames(d, host, 1).empty());
}

TEST(DagPasses, CancellationNeedsAdjacencyOnEveryWire) {
    // CX(q0,q1) - RZ(q1) - CX(q0,q1)：两个 CX 在 q0 上相邻、在 q1 上不相邻
    DagStore d;
    TestHost host;
    Build(d, host, 2, {{"cx", {0, 1}}, {"rz", {1}, {0.3}}, {"cx", {0, 1}}});
    RunGateCancellation(d, host);
    EXPECT_EQ(WireNames(d, host, 1), (std::vector<std::string>{"cx", "rz", "cx"}));
}

TEST(DagPasses, CancelsOppositeRotations) {
    DagStore d;
    TestHost host;
    Build(d, host, 1, {{"rz", {0}, {0.25}}, {"rz", {0}, {-0.25}}, {"rx", {0}, {0.5}}, {"rx", {0}, {0.4}}});
    RunGateCancellation(d, host);
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"rx", "rx"}));
}

TEST(DagPasses, CommutationBringsEqualGatesTogether) {
    // h(0) cx(0,1) z(0)?  用 rz 与 cx 控制位对易：rz(0) cx(0,1) rz(0) -> rz rz cx
    DagStore d;
    TestHost host;
    Build(d, host, 2, {{"t", {0}}, {"cx", {0, 1}}, {"x", {1}}, {"cx", {0, 1}}});
    RunCommutationAnalysis(d, host, 5);
    // cx 与 x 在目标位上对易：x 被移到第二个 cx 之后
    EXPECT_EQ(WireNames(d, host, 1), (std::vector<std::string>{"cx", "cx", "x"}));
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"t", "cx", "cx"}));
    RunGateCancellation(d, host);
    EXPECT_EQ(WireNames(d, host, 1), (std::vector<std::string>{"x"}));
}

TEST(DagPasses, CommutationRespectsNonCommutingGates) {
    DagStore d;
    TestHost host;
    Build(d, host, 2, {{"cx", {0, 1}}, {"h", {1}}, {"cx", {0, 1}}});
    RunCommutationAnalysis(d, host, 5);
    EXPECT_EQ(WireNames(d, host, 1), (std::vector<std::string>{"cx", "h", "cx"}));
}

TEST(DagPasses, MergesSameAxisRotations) {
    DagStore d;
    TestHost host;
    const auto ops = Build(d, host, 1, {{"rz", {0}, {0.25}}, {"p", {0}, {0.5}}, {"u1", {0}, {1.0}}, {"rx", {0}, {0.5}}});
    RunRotationMerging(d, host);
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"rz", "rx"}));
    EXPECT_DOUBLE_EQ(host.node(ops[0]).params[0], 1.75);
    EXPECT_EQ(host.substitutions, 2);
}

TEST(DagPasses, ResynthesisReplacesAndSplicesRuns) {
    DagStore d;
    TestHost host;
    host.replacement_size = 1;
    Build(d, host, 2, {{"h", {0}}, {"t", {0}}, {"h", {0}}, {"cx", {0, 1}}, {"h", {0}}});
    RunSingleQubitResynthesis(d, host);
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"u3", "cx", "h"}));
    // 替换门多于原串时只替换原串中的节点（zip 截断）
    DagStore d2;
    TestHost host2;
    host2.replacement_size = 3;
    Build(d2, host2, 1, {{"h", {0}}, {"t", {0}}});
    RunSingleQubitResynthesis(d2, host2);
    EXPECT_EQ(WireNames(d2, host2, 0), (std::vector<std::string>{"u3", "u3"}));
}

TEST(DagPasses, EliminatesDiagonalBeforeMeasurement) {
    DagStore d;
    TestHost host;
    Build(d, host, 1, {{"h", {0}}, {"t", {0}}, {"z", {0}}, {"", {0}, {}, true}});
    RunEliminateDiagonalBeforeMeasurement(d, host);
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"h", "measure"}));
}

TEST(DagPasses, RemovesIdentityRotations) {
    DagStore d;
    TestHost host;
    const double two_pi = 2.0 * 3.141592653589793;
    Build(d, host, 1,
          {{"rz", {0}, {0.0}}, {"h", {0}}, {"rx", {0}, {two_pi}}, {"rz", {0}, {0.1}}, {"u3", {0}, {0.0, 0.2, -0.2}}});
    EXPECT_TRUE(RunRemoveIdentityGates(d, host));
    EXPECT_EQ(WireNames(d, host, 0), (std::vector<std::string>{"h", "rz"}));
}

TEST(DagPasses, IdentityRemovalDefersInfiniteAngles) {
    DagStore d;
    TestHost host;
    Build(d, host, 1, {{"rz", {0}, {0.0}}, {"rz", {0}, {INFINITY}}});
    const std::size_t before = d.num_nodes();
    EXPECT_FALSE(RunRemoveIdentityGates(d, host));
    EXPECT_EQ(d.num_nodes(), before);  // 没有修改
}

TEST(DagPasses, InternedNamesAreStable) {
    EXPECT_EQ(InternGateName("cx"), quspect::native::ir::kNameCx);
    const std::int32_t a = InternGateName("my_custom_gate");
    EXPECT_EQ(InternGateName("my_custom_gate"), a);
    EXPECT_GE(a, quspect::native::ir::kNumFixedGateNames);
}

}  // namespace
