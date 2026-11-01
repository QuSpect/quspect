// ir::Circuit（连续存储 + 隐式 DAG 的线路 IR）的 gtest：前驱 / 后继、按比特的指令链、
// ASAP 分层、比特数推断、名字表与操作的静态信息。
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/ir/circuit.hpp"
#include "quspect_native/ir/operation.hpp"

namespace {

using quspect::native::ir::Circuit;
using quspect::native::ir::GateKindFromName;
using quspect::native::ir::GetOpInfo;
using quspect::native::ir::InstRef;
using quspect::native::ir::kNoInst;
using quspect::native::ir::Operation;
using quspect::native::ir::OpKind;

std::vector<InstRef> Preds(const Circuit& c, InstRef i) {
    std::vector<InstRef> out;
    c.foreach_predecessor(i, [&](InstRef p) { out.push_back(p); });
    return out;
}

std::vector<InstRef> Succs(const Circuit& c, InstRef i) {
    std::vector<InstRef> out;
    c.foreach_successor(i, [&](InstRef s) { out.push_back(s); });
    return out;
}

TEST(IrCircuit, ImplicitDagFromAppendOrder) {
    Circuit c;
    const InstRef h = c.append(Operation::Gate(OpKind::kH), {0});
    const InstRef cx = c.append(Operation::Gate(OpKind::kCX), {0, 1});
    const InstRef rz = c.append(Operation::Gate(OpKind::kRZ, {0.5}), {1});
    const InstRef x = c.append(Operation::Gate(OpKind::kX), {2});
    Operation measure;
    measure.kind = OpKind::kMeasure;
    const InstRef m = c.append(measure, {1}, {0});

    EXPECT_EQ(c.size(), 5u);
    EXPECT_EQ(c.num_qubits(), 3u);
    EXPECT_EQ(c.num_clbits(), 1u);
    EXPECT_EQ(c[cx].preds[0], h);
    EXPECT_EQ(c[cx].preds[1], kNoInst);
    EXPECT_EQ(Preds(c, rz), (std::vector<InstRef>{cx}));
    EXPECT_TRUE(Preds(c, x).empty());
    EXPECT_EQ(Succs(c, h), (std::vector<InstRef>{cx}));
    EXPECT_EQ(Succs(c, cx), (std::vector<InstRef>{rz}));
    EXPECT_EQ(Succs(c, rz), (std::vector<InstRef>{m}));
    EXPECT_EQ(c.last_on_qubit(1), m);
    EXPECT_EQ(c.last_on_clbit(0), m);
    EXPECT_EQ(c.last_on_qubit(7), kNoInst);
    EXPECT_DOUBLE_EQ(c[rz].op.params[0], 0.5);
    EXPECT_EQ(c[rz].op.num_params, 1);

    // 追加后后继索引失效并重建
    const InstRef h2 = c.append(Operation::Gate(OpKind::kH), {0});
    EXPECT_EQ(Succs(c, cx), (std::vector<InstRef>{rz, h2}));
}

TEST(IrCircuit, DuplicatePredecessorsAreReportedOnce) {
    Circuit c;
    const InstRef a = c.append(Operation::Gate(OpKind::kCX), {0, 1});
    const InstRef b = c.append(Operation::Gate(OpKind::kCZ), {1, 0});
    EXPECT_EQ(Preds(c, b), (std::vector<InstRef>{a}));
    EXPECT_EQ(Succs(c, a), (std::vector<InstRef>{b}));
}

TEST(IrCircuit, QubitChainAndLayers) {
    Circuit c;
    c.append(Operation::Gate(OpKind::kH), {0});        // 0
    c.append(Operation::Gate(OpKind::kH), {1});        // 1
    c.append(Operation::Gate(OpKind::kCX), {0, 1});    // 2
    c.append(Operation::Gate(OpKind::kT), {1});        // 3
    c.append(Operation::Gate(OpKind::kCX), {1, 2});    // 4
    EXPECT_EQ(c.qubit_chain(1), (std::vector<InstRef>{1, 2, 3, 4}));
    EXPECT_EQ(c.qubit_chain(2), (std::vector<InstRef>{4}));
    std::uint32_t depth = 0;
    const auto layers = c.asap_layers(&depth);
    EXPECT_EQ(layers, (std::vector<std::uint32_t>{0, 0, 1, 2, 3}));
    EXPECT_EQ(depth, 4u);
    Circuit empty;
    empty.asap_layers(&depth);
    EXPECT_EQ(depth, 0u);
}

TEST(IrCircuit, OpaqueNamesAndOpTable) {
    Circuit c;
    Operation op;
    op.kind = OpKind::kOpaque;
    op.ext = c.intern_name("my_gate");
    EXPECT_EQ(c.intern_name("my_gate"), op.ext);
    const InstRef i = c.append(op, {0, 3});
    EXPECT_EQ(c.gate_name(c[i]), "my_gate");
    EXPECT_EQ(c.num_qubits(), 4u);
    EXPECT_EQ(GetOpInfo(OpKind::kCU3).name, "cu3");
    EXPECT_EQ(GetOpInfo(OpKind::kCU3).num_params, 3);
    EXPECT_EQ(GetOpInfo(OpKind::kCCX).num_qubits, 3);
    EXPECT_EQ(GateKindFromName("sxdg"), OpKind::kSXdg);
    EXPECT_EQ(GateKindFromName("u1"), OpKind::kU1);
    EXPECT_FALSE(GateKindFromName("measure").has_value());
    EXPECT_FALSE(GateKindFromName("rzz").has_value());
}

}  // namespace
