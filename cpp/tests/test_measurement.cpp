#include <cmath>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/executor/executor.hpp"
#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

// ============================================================
// 辅助函数
// ============================================================

static runtime::GateOperation MakeHGate(uint32_t qubit) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "h";
    g.qubits = {qubit};
    return g;
}

static runtime::GateOperation MakeXGate(uint32_t qubit) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "x";
    g.qubits = {qubit};
    return g;
}

static runtime::GateOperation MakeCXGate(uint32_t control, uint32_t target) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "cx";
    g.qubits = {control, target};
    return g;
}

static runtime::MeasureOperation MakeMeasure(const std::vector<uint32_t>& qubits, const std::vector<uint32_t>& clbits) {
    runtime::MeasureOperation m;
    m.qubits = qubits;
    m.clbits = clbits;
    return m;
}

// ============================================================
// collapse kernel 单元测试
// ============================================================

TEST(CollapseKernelTest, ZeroState_Outcome0_StaysZero) {
    const auto& table = kernels::GetDispatchF64();
    // |0⟩, measure qubit 0 → outcome 0, state should remain |0⟩
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);
    state[1] = std::complex<double>(0, 0);

    table.collapse(state.data(), 2, 0, 0);

    EXPECT_NEAR(std::norm(state[0]), 1.0, 1e-10);
    EXPECT_NEAR(std::norm(state[1]), 0.0, 1e-10);
}

TEST(CollapseKernelTest, ZeroState_Outcome1_ShouldNotHappen) {
    const auto& table = kernels::GetDispatchF64();
    // |0⟩, measure qubit 0 → outcome 1 (概率为 0，但逻辑上可处理)
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);
    state[1] = std::complex<double>(0, 0);

    table.collapse(state.data(), 2, 0, 1);

    // 所有振幅应该为 0（概率为 0 的 outcome 不合法）
    EXPECT_NEAR(std::norm(state[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::norm(state[1]), 0.0, 1e-10);
}

TEST(CollapseKernelTest, PlusState_Outcome0_Normalized) {
    const auto& table = kernels::GetDispatchF64();
    // |+⟩ = (|0⟩ + |1⟩)/√2, measure qubit 0 → outcome 0
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(inv_sqrt2, 0);

    table.collapse(state.data(), 2, 0, 0);

    // 坍缩后应该为 |0⟩，即只有 state[0] 非零，归一化后 |state[0]|^2 = 1
    EXPECT_NEAR(std::norm(state[0]), 1.0, 1e-10);
    EXPECT_NEAR(std::norm(state[1]), 0.0, 1e-10);
}

TEST(CollapseKernelTest, PlusState_Outcome1_Normalized) {
    const auto& table = kernels::GetDispatchF64();
    // |+⟩, measure qubit 0 → outcome 1
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(inv_sqrt2, 0);

    table.collapse(state.data(), 2, 0, 1);

    // 坍缩后应该为 |1⟩
    EXPECT_NEAR(std::norm(state[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::norm(state[1]), 1.0, 1e-10);
}

TEST(CollapseKernelTest, TwoQubitBellState_MeasureQ0_Outcome0) {
    const auto& table = kernels::GetDispatchF64();
    // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, measure qubit 0 → outcome 0 → |00⟩
    std::vector<std::complex<double>> state(4);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[3] = std::complex<double>(inv_sqrt2, 0);

    table.collapse(state.data(), 4, 0, 0);

    // |11⟩ 的 qubit 0 = 1，被清零。剩下 |00⟩，归一化后 |state[0]|^2 = 1
    EXPECT_NEAR(std::norm(state[0]), 1.0, 1e-10);
    EXPECT_NEAR(std::norm(state[1]), 0.0, 1e-10);
    EXPECT_NEAR(std::norm(state[2]), 0.0, 1e-10);
    EXPECT_NEAR(std::norm(state[3]), 0.0, 1e-10);
}

TEST(CollapseKernelTest, Float32Precision) {
    const auto& table = kernels::GetDispatchF32();
    std::vector<std::complex<float>> state(2);
    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    state[0] = std::complex<float>(inv_sqrt2, 0);
    state[1] = std::complex<float>(inv_sqrt2, 0);

    table.collapse(state.data(), 2, 0, 0);

    EXPECT_NEAR(std::norm(state[0]), 1.0f, 1e-6f);
    EXPECT_NEAR(std::norm(state[1]), 0.0f, 1e-6f);
}

// ============================================================
// 终端采样测试 (SimulationExecutor)
// ============================================================

TEST(TerminalSamplingTest, ZeroState_ZMeasure_AlwaysZero) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 100;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    // |0⟩ 态测量 Z 应始终得到 0
    ASSERT_EQ(result.packed_samples.size(), 100);
    for (auto s : result.packed_samples) {
        EXPECT_EQ(s, 0);
    }
    // counts 应只有 "0"
    ASSERT_TRUE(result.counts.count("0"));
    EXPECT_EQ(result.counts.at("0"), 100);
    EXPECT_EQ(result.metadata.at("measurement_mode"), "terminal");
}

TEST(TerminalSamplingTest, PlusState_ZMeasure_Statistical) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 2000;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 2000);

    // 统计 0 和 1 的频次
    size_t zeros = std::count(result.packed_samples.begin(), result.packed_samples.end(), 0);
    size_t ones = std::count(result.packed_samples.begin(), result.packed_samples.end(), 1);

    // |+⟩ 态下 0 和 1 各约 50%，允许 ±5% 的统计误差
    double ratio_zero = static_cast<double>(zeros) / 2000.0;
    EXPECT_NEAR(ratio_zero, 0.5, 0.05);

    // counts 应包含 "0" 和 "1"
    EXPECT_TRUE(result.counts.count("0"));
    EXPECT_TRUE(result.counts.count("1"));
    EXPECT_EQ(result.counts.at("0") + result.counts.at("1"), 2000);
}

TEST(TerminalSamplingTest, NoMeasureOps_NoShots_EmptySamples) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    // 无测量操作
    prog.ops.push_back(MakeHGate(0));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 100;  // 有 shots 但无测量操作
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    // 没有测量操作时不应采样
    EXPECT_TRUE(result.packed_samples.empty());
    EXPECT_TRUE(result.counts.empty());
    EXPECT_EQ(result.metadata.at("shots"), "0");
}

TEST(TerminalSamplingTest, StatevectorAndSamplesTogether) {
    // 同时请求 statevector 和采样
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 10;
    req.return_statevector = true;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    // statevector 应返回测量前的末态
    ASSERT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), 1.0, 1e-10);

    // 采样应正常工作
    ASSERT_EQ(result.packed_samples.size(), 10);
}

TEST(TerminalSamplingTest, MultiQubitMeasurement) {
    // 2-qubit |00⟩, 同时测量两个 qubit
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 2;
    prog.ops.push_back(MakeMeasure({0, 1}, {0, 1}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 100;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 100);
    for (auto s : result.packed_samples) {
        EXPECT_EQ(s, 0);  // |00⟩ → bitstring "00" = 0
    }
}

// ============================================================
// 中途测量测试 (SimulationExecutor)
// ============================================================

TEST(MidCircuitMeasurementTest, H_Measure_X_OnZero) {
    // 线路: H|0⟩ → Measure q0→c0 → if c0==0: X|0⟩
    // 由于测量后状态坍缩，如果结果为 0 则 X 门作用
    // 效果: 50% 概率测量到 0 并施加 X (最终 |1⟩, c0=0),
    //       50% 概率测量到 1 (最终 |1⟩, c0=1)
    // 但这不对... 让我仔细想想。
    //
    // 更简单的测试: H|0⟩ = |+⟩ → Measure q0→c0
    // 测量到 0 → 坍缩为 |0⟩, 结果 bit=0
    // 测量到 1 → 坍缩为 |1⟩, 结果 bit=1
    // 后续加 X 门: 如果测量到 0 则翻转为 1，最终 c0=0 但 state=|1⟩
    // 如果测量到 1 则翻转为 0，最终 c0=1 但 state=|0⟩
    //
    // 经典寄存器记录的是测量结果，不是最终状态。
    // 所以 c0=0 概率 50%, c0=1 概率 50%。
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));           // |+⟩
    prog.ops.push_back(MakeMeasure({0}, {0}));  // 测量 q0→c0
    prog.ops.push_back(MakeXGate(0));           // X 门（测量后执行）

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 1000;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 1000);

    size_t zeros = std::count(result.packed_samples.begin(), result.packed_samples.end(), 0);
    size_t ones = std::count(result.packed_samples.begin(), result.packed_samples.end(), 1);

    // 经典寄存器记录测量结果，应大约 50/50
    double ratio_zero = static_cast<double>(zeros) / 1000.0;
    EXPECT_NEAR(ratio_zero, 0.5, 0.06);

    // 验证测量模式
    EXPECT_EQ(result.metadata.at("measurement_mode"), "mid_circuit");
}

TEST(MidCircuitMeasurementTest, TwoMeasurements) {
    // 线路: H|0⟩ → Measure q0→c0 → H|0⟩ → Measure q0→c1
    // 两次独立测量，中间通过 H 重置叠加态
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 2;
    prog.ops.push_back(MakeHGate(0));           // |+⟩
    prog.ops.push_back(MakeMeasure({0}, {0}));  // 第一次测量 → c0
    prog.ops.push_back(MakeHGate(0));           // 坍缩态 → H → |+⟩ 或 |-⟩
    prog.ops.push_back(MakeMeasure({0}, {1}));  // 第二次测量 → c1

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 1000;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 1000);

    // 统计每个 bitstring
    size_t c00 = 0, c01 = 0, c10 = 0, c11 = 0;
    for (auto s : result.packed_samples) {
        if (s == 0)
            c00++;
        else if (s == 1)
            c01++;
        else if (s == 2)
            c10++;
        else if (s == 3)
            c11++;
    }

    // 第一次测量: 50/50 得到 0 或 1
    // 第二次测量: 无论第一次结果如何，H 重置叠加态后 50/50
    // 所以四种组合应各约 25%
    EXPECT_NEAR(static_cast<double>(c00) / 1000.0, 0.25, 0.06);
    EXPECT_NEAR(static_cast<double>(c01) / 1000.0, 0.25, 0.06);
    EXPECT_NEAR(static_cast<double>(c10) / 1000.0, 0.25, 0.06);
    EXPECT_NEAR(static_cast<double>(c11) / 1000.0, 0.25, 0.06);
}

TEST(MidCircuitMeasurementTest, BellState_PartialMeasurement) {
    // |Φ⁺⟩ → Measure q0→c0 → Measure q1→c1
    // 测量结果总是 00 或 11 (perfect correlation)
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 2;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeCXGate(0, 1));
    prog.ops.push_back(MakeMeasure({0}, {0}));  // 先测 q0
    prog.ops.push_back(MakeMeasure({1}, {1}));  // 再测 q1

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 500;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 500);

    // Bell 态只产生 00 或 11
    for (auto s : result.packed_samples) {
        EXPECT_TRUE(s == 0 || s == 3) << "Unexpected bitstring: " << s;
    }

    size_t c00 = std::count(result.packed_samples.begin(), result.packed_samples.end(), 0);
    size_t c11 = std::count(result.packed_samples.begin(), result.packed_samples.end(), 3);

    EXPECT_NEAR(static_cast<double>(c00) / 500.0, 0.5, 0.08);
    EXPECT_NEAR(static_cast<double>(c11) / 500.0, 0.5, 0.08);
}

TEST(MidCircuitMeasurementTest, StatevectorFromLastShot) {
    // 中途测量后，最后一个 shot 的末态应该被返回
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 100;
    req.return_statevector = true;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    // statevector 应存在（最后一个 shot 的末态）
    ASSERT_EQ(result.statevector_f64.size(), 2);
    // 坍缩后状态应归一化
    double norm = std::norm(result.statevector_f64[0]) + std::norm(result.statevector_f64[1]);
    EXPECT_NEAR(norm, 1.0, 1e-10);
}

// ============================================================
// 种子可复现性测试
// ============================================================

TEST(SeedReproducibilityTest, SameSeedSameResults) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 100;
    req.seed = 42;
    req.return_samples = true;

    auto result1 = exec.Run(prog, req);
    auto result2 = exec.Run(prog, req);

    // 相同种子应产生相同结果
    ASSERT_EQ(result1.packed_samples.size(), result2.packed_samples.size());
    for (size_t i = 0; i < result1.packed_samples.size(); ++i) {
        EXPECT_EQ(result1.packed_samples[i], result2.packed_samples[i]);
    }
    EXPECT_EQ(result1.counts, result2.counts);
}

TEST(SeedReproducibilityTest, DifferentSeedDifferentResults) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req1;
    req1.shots = 100;
    req1.seed = 42;
    req1.return_samples = true;

    executor::ExecutionRequest req2;
    req2.shots = 100;
    req2.seed = 12345;
    req2.return_samples = true;

    auto result1 = exec.Run(prog, req1);
    auto result2 = exec.Run(prog, req2);

    // 不同种子几乎肯定产生不同结果（统计上）
    bool all_same = true;
    for (size_t i = 0; i < result1.packed_samples.size(); ++i) {
        if (result1.packed_samples[i] != result2.packed_samples[i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Different seeds unexpectedly produced identical results";
}

// ============================================================
// Float32 精度中途测量测试
// ============================================================

TEST(MidCircuitMeasurementTest, Float32Precision) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 1;
    prog.ops.push_back(MakeHGate(0));
    prog.ops.push_back(MakeMeasure({0}, {0}));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat32;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 500;
    req.return_samples = true;

    auto result = exec.Run(prog, req);

    ASSERT_EQ(result.packed_samples.size(), 500);

    size_t zeros = std::count(result.packed_samples.begin(), result.packed_samples.end(), 0);
    double ratio_zero = static_cast<double>(zeros) / 500.0;
    EXPECT_NEAR(ratio_zero, 0.5, 0.08);
}

// ============================================================
// 边界情况测试
// ============================================================

TEST(EdgeCaseTest, ZeroQubitCircuit) {
    runtime::QSALProgram prog;
    prog.num_qubits = 0;

    executor::SimulationExecutor::Options opts;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    auto result = exec.Run(prog, req);

    EXPECT_TRUE(result.metadata.count("error"));
}

TEST(EdgeCaseTest, NoOpsCircuit) {
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;

    executor::SimulationExecutor::Options opts;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = exec.Run(prog, req);

    // 应返回 |00⟩
    ASSERT_EQ(result.statevector_f64.size(), 4);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), 1.0, 1e-10);
    EXPECT_EQ(result.metadata.at("measurement_mode"), "terminal");
}

TEST(EdgeCaseTest, MeasurementWithoutClbits) {
    // 测量操作没有指定 clbits 时，使用 qubit 自身作为默认 clbit
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 2;
    runtime::MeasureOperation m;
    m.qubits = {0, 1};
    // clbits 为空，引擎中应跳过 creg 写入
    m.clbits = {};
    prog.ops.push_back(m);

    executor::SimulationExecutor::Options opts;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.shots = 10;
    req.return_samples = true;

    // 不应崩溃
    auto result = exec.Run(prog, req);
    EXPECT_EQ(result.packed_samples.size(), 10);
}
