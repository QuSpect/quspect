#include <cmath>

#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/executor/executor.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

// ============================================================
// 辅助函数：创建单比特门操作
// ============================================================
static runtime::GateOperation MakeNamedGate(const std::string& name, uint32_t qubit) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = name;
    g.qubits = {qubit};
    return g;
}

static runtime::GateOperation MakeRxGate(uint32_t qubit, double theta) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "rx";
    g.qubits = {qubit};
    g.params_f64 = {theta};
    return g;
}

static runtime::GateOperation MakeRyGate(uint32_t qubit, double theta) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "ry";
    g.qubits = {qubit};
    g.params_f64 = {theta};
    return g;
}

static runtime::GateOperation MakeRzGate(uint32_t qubit, double theta) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "rz";
    g.qubits = {qubit};
    g.params_f64 = {theta};
    return g;
}

static runtime::GateOperation MakeCXGate(uint32_t control, uint32_t target) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "cx";
    g.qubits = {control, target};
    return g;
}

static runtime::GateOperation MakeSwapGate(uint32_t q0, uint32_t q1) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "swap";
    g.qubits = {q0, q1};
    return g;
}

// ============================================================
// 基础线路测试
// ============================================================

TEST(CircuitExecutionTest, EmptyCircuit) {
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 4);
    // 初始态 |00> = 1
    EXPECT_NEAR(std::real(result.statevector_f64[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[2]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[3]), 0.0, 1e-12);
}

TEST(CircuitExecutionTest, XGateCreatesOne) {
    // X on q0: |0> -> |1>
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("x", 0));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), 1.0, 1e-12);
}

TEST(CircuitExecutionTest, BellState) {
    // H on q0, CNOT(q0, q1) -> Bell state (|00> + |11>)/√2
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));
    prog.ops.push_back(MakeCXGate(0, 1));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 4);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    // |00> = index 0
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::imag(result.statevector_f64[0]), 0.0, 1e-12);
    // |01> = index 1
    EXPECT_NEAR(std::real(result.statevector_f64[1]), 0.0, 1e-12);
    // |10> = index 2
    EXPECT_NEAR(std::real(result.statevector_f64[2]), 0.0, 1e-12);
    // |11> = index 3
    EXPECT_NEAR(std::real(result.statevector_f64[3]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::imag(result.statevector_f64[3]), 0.0, 1e-12);
}

TEST(CircuitExecutionTest, GHZState) {
    // H on q0, CNOT(0,1), CNOT(0,2) -> GHZ (|000> + |111>)/√2
    runtime::QSALProgram prog;
    prog.num_qubits = 3;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));
    prog.ops.push_back(MakeCXGate(0, 1));
    prog.ops.push_back(MakeCXGate(0, 2));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 8);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // |000> = index 0
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-12);
    // |111> = index 7
    EXPECT_NEAR(std::real(result.statevector_f64[7]), inv_sqrt2, 1e-12);
    // others should be 0
    for (uint64_t i = 1; i < 7; ++i) {
        EXPECT_NEAR(std::abs(result.statevector_f64[i]), 0.0, 1e-12) << "at i=" << i;
    }
}

// ============================================================
// 参数化门测试
// ============================================================

TEST(CircuitExecutionTest, RXPiPulse) {
    // RX(pi) on |0> = -i|1>
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeRxGate(0, M_PI));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::abs(result.statevector_f64[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.statevector_f64[1]), 1.0, 1e-10);
}

TEST(CircuitExecutionTest, RYPiPulse) {
    // RY(pi) on |0> = |1>
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeRyGate(0, M_PI));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::abs(result.statevector_f64[0]), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(result.statevector_f64[1]), 1.0, 1e-10);
}

TEST(CircuitExecutionTest, RZGate) {
    // RZ(theta) on |+> should give:
    // |+_theta> = (|0> + e^{i*theta}|1>)/√2
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));    // |+>
    prog.ops.push_back(MakeRzGate(0, M_PI / 2));  // RZ(pi/2)

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);

    // RZ(pi/2): exp(i*pi/4)*|0> + exp(-i*pi/4)*|1> all over √2
    // Wait, actually RZ(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
    // So RZ(pi/2)|+> = (exp(-i*pi/4)|0> + exp(i*pi/4)|1>)/√2
    double cos_pi4 = std::cos(M_PI / 4);
    double sin_pi4 = std::sin(M_PI / 4);

    // |0>: exp(-i*pi/4)/√2 = (cos(pi/4) - i*sin(pi/4))/√2
    EXPECT_NEAR(std::real(result.statevector_f64[0]), cos_pi4 * inv_sqrt2, 1e-10);
    EXPECT_NEAR(std::imag(result.statevector_f64[0]), -sin_pi4 * inv_sqrt2, 1e-10);

    // |1>: exp(i*pi/4)/√2 = (cos(pi/4) + i*sin(pi/4))/√2
    EXPECT_NEAR(std::real(result.statevector_f64[1]), cos_pi4 * inv_sqrt2, 1e-10);
    EXPECT_NEAR(std::imag(result.statevector_f64[1]), sin_pi4 * inv_sqrt2, 1e-10);
}

// ============================================================
// 双比特门组合测试
// ============================================================

TEST(CircuitExecutionTest, SWAPGate) {
    // X on q0, then SWAP(q0,q1): |01> -> |10>
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("x", 0));  // |01> (idx 1)
    prog.ops.push_back(MakeSwapGate(0, 1));     // swap -> |10> (idx 2)

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 4);
    EXPECT_NEAR(std::abs(result.statevector_f64[2]), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(result.statevector_f64[1]), 0.0, 1e-12);
}

TEST(CircuitExecutionTest, CZGate) {
    // |++> then CZ → Bell-like state with phase
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));  // |+> on q0
    prog.ops.push_back(MakeNamedGate("h", 1));  // |+> on q1
    // CZ: qubits[0]=control, qubits[1]=target
    runtime::GateOperation cz;
    cz.opcode = runtime::OpCode::kNamedGate;
    cz.name = "cz";
    cz.qubits = {0, 1};
    prog.ops.push_back(cz);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 4);
    // (|00> + |01> + |10> - |11>)/2
    double half = 0.5;
    EXPECT_NEAR(std::real(result.statevector_f64[0]), half, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), half, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[2]), half, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[3]), -half, 1e-12);
}

// ============================================================
// Barrier 测试
// ============================================================

TEST(CircuitExecutionTest, BarrierIsNoop) {
    // Barrier 应该被跳过，不影响状态
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("x", 0));

    runtime::GateOperation barrier;
    barrier.opcode = runtime::OpCode::kBarrier;
    barrier.qubits = {0};
    prog.ops.push_back(barrier);

    prog.ops.push_back(MakeNamedGate("x", 0));  // X*X = I, back to |0>

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_NEAR(std::real(result.statevector_f64[0]), 1.0, 1e-12);
}

// ============================================================
// 精度切换测试
// ============================================================

TEST(CircuitExecutionTest, SinglePrecision) {
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat32;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f32.size(), 2);
    EXPECT_EQ(result.statevector_f64.size(), 0);  // double vector should be empty

    float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    EXPECT_NEAR(std::real(result.statevector_f32[0]), inv_sqrt2, 1e-5f);
    EXPECT_NEAR(std::real(result.statevector_f32[1]), inv_sqrt2, 1e-5f);
}

// ============================================================
// 多门序列测试
// ============================================================

TEST(CircuitExecutionTest, XHSequence) {
    // X then H: X|0>=|1>, H|1>=(|0>-|1>)/√2
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("x", 0));
    prog.ops.push_back(MakeNamedGate("h", 0));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), -inv_sqrt2, 1e-12);
}

TEST(CircuitExecutionTest, SGate) {
    // H then S: S|+> = (|0> + i|1>)/√2
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 0));
    prog.ops.push_back(MakeNamedGate("s", 0));

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::imag(result.statevector_f64[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::imag(result.statevector_f64[1]), inv_sqrt2, 1e-12);
}

TEST(CircuitExecutionTest, GappedQubitIndices) {
    // 使用非连续的 qubit index，测试 qubit index 映射正确
    // 2 qubits: q1 is bit 1, q3 is bit 2?  No — num_qubits=4 means dim=16
    // q0 unused, q2 unused
    // H on q1, X on q3
    runtime::QSALProgram prog;
    prog.num_qubits = 4;
    prog.num_clbits = 0;
    prog.ops.push_back(MakeNamedGate("h", 1));  // q1=bit1
    prog.ops.push_back(MakeNamedGate("x", 3));  // q3=bit3

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(prog, req);

    EXPECT_EQ(result.statevector_f64.size(), 16);

    // Initial: |0000> -> H on q1: (|0000> + |0010>)/√2 -> X on q3:
    // (|1000> + |1010>)/√2
    // indices: q3,q2,q1,q0 in LE: q3=bit3, q2=bit2, q1=bit1, q0=bit0
    // |1000> = bit3=1 = idx 8
    // |1010> = bit3=1, bit1=1 = idx 10
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(result.statevector_f64[8]), inv_sqrt2, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[10]), inv_sqrt2, 1e-12);

    // All others should be 0
    double total = 0;
    for (uint64_t i = 0; i < 16; ++i) {
        total += std::norm(result.statevector_f64[i]);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}
