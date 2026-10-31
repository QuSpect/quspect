#include <cmath>

#include <complex>
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

static runtime::GateOperation MakeCXGate(uint32_t control, uint32_t target) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "cx";
    g.qubits = {control, target};
    return g;
}

static runtime::GateOperation MakeXGate(uint32_t qubit) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = "x";
    g.qubits = {qubit};
    return g;
}

// ============================================================
// ComputePauliMasks 单元测试
// ============================================================

TEST(ComputePauliMasksTest, SingleX) {
    auto info = runtime::ComputePauliMasks({{0, 'X'}});
    EXPECT_EQ(info.bit_flip_mask, 1ULL << 0);
    EXPECT_EQ(info.phase_flip_mask, 0);
    EXPECT_EQ(info.pivot_qubit, 0);
    EXPECT_EQ(info.y_count, 0);
}

TEST(ComputePauliMasksTest, SingleZ) {
    auto info = runtime::ComputePauliMasks({{2, 'Z'}});
    EXPECT_EQ(info.bit_flip_mask, 0);
    EXPECT_EQ(info.phase_flip_mask, 1ULL << 2);
    EXPECT_EQ(info.pivot_qubit, 0);  // 纯对角项，pivot 为 0
    EXPECT_EQ(info.y_count, 0);
}

TEST(ComputePauliMasksTest, SingleY) {
    auto info = runtime::ComputePauliMasks({{1, 'Y'}});
    EXPECT_EQ(info.bit_flip_mask, 1ULL << 1);
    EXPECT_EQ(info.phase_flip_mask, 1ULL << 1);
    EXPECT_EQ(info.pivot_qubit, 1);
    EXPECT_EQ(info.y_count, 1);
}

TEST(ComputePauliMasksTest, MultiQubitMixed) {
    // X0 Z1 Y2
    auto info = runtime::ComputePauliMasks({{0, 'X'}, {1, 'Z'}, {2, 'Y'}});
    EXPECT_EQ(info.bit_flip_mask, (1ULL << 0) | (1ULL << 2));    // X on q0, Y on q2
    EXPECT_EQ(info.phase_flip_mask, (1ULL << 1) | (1ULL << 2));  // Z on q1, Y on q2
    EXPECT_EQ(info.pivot_qubit, 0);                              // bit_flip_mask LSB = bit 0
    EXPECT_EQ(info.y_count, 1);
}

TEST(ComputePauliMasksTest, AllIdentity) {
    // I0 I1 — 空的 paulis 列表等价于恒等算符
    auto info = runtime::ComputePauliMasks({});
    EXPECT_EQ(info.bit_flip_mask, 0);
    EXPECT_EQ(info.phase_flip_mask, 0);
    EXPECT_EQ(info.y_count, 0);
}

TEST(ComputePauliMasksTest, ExplicitIdentity) {
    auto info = runtime::ComputePauliMasks({{0, 'I'}, {1, 'I'}});
    EXPECT_EQ(info.bit_flip_mask, 0);
    EXPECT_EQ(info.phase_flip_mask, 0);
    EXPECT_EQ(info.y_count, 0);
}

TEST(ComputePauliMasksTest, LowercasePauliChars) {
    auto info_x = runtime::ComputePauliMasks({{0, 'x'}});
    EXPECT_EQ(info_x.bit_flip_mask, 1ULL << 0);

    auto info_y = runtime::ComputePauliMasks({{0, 'y'}});
    EXPECT_EQ(info_y.bit_flip_mask, 1ULL << 0);
    EXPECT_EQ(info_y.phase_flip_mask, 1ULL << 0);
    EXPECT_EQ(info_y.y_count, 1);

    auto info_z = runtime::ComputePauliMasks({{0, 'z'}});
    EXPECT_EQ(info_z.phase_flip_mask, 1ULL << 0);
}

TEST(ComputePauliMasksTest, PivotSelectsLowestBitFlipBit) {
    // Y 在 qubit 3 和 qubit 0 — pivot 应为 0
    auto info = runtime::ComputePauliMasks({{3, 'X'}, {0, 'Y'}});
    EXPECT_EQ(info.pivot_qubit, 0);
}

TEST(ComputePauliMasksTest, InvalidQubitIndexThrows) {
    EXPECT_THROW(runtime::ComputePauliMasks({{64, 'X'}}), std::invalid_argument);
    EXPECT_THROW(runtime::ComputePauliMasks({{100, 'Z'}}), std::invalid_argument);
}

TEST(ComputePauliMasksTest, InvalidPauliCharThrows) {
    EXPECT_THROW(runtime::ComputePauliMasks({{0, 'W'}}), std::invalid_argument);
    EXPECT_THROW(runtime::ComputePauliMasks({{0, 'A'}}), std::invalid_argument);
}

// ============================================================
// Pauli 期望值 kernel 测试（直接调用 dispatch table）
// ============================================================

TEST(ExpectationKernelTest, PauliZ_ZeroState) {
    const auto& table = kernels::GetDispatchF64();
    // <0|Z|0> = 1
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);  // |0>

    auto result = table.pauli_expect(state.data(), 2,
                                     0,          // bit_flip_mask (no X/Y)
                                     1ULL << 0,  // phase_flip_mask (Z on q0)
                                     0,          // pivot (unused for diagonal)
                                     0           // y_count
    );

    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

TEST(ExpectationKernelTest, PauliX_PlusState) {
    const auto& table = kernels::GetDispatchF64();
    // <+|X|+> = 1
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(inv_sqrt2, 0);  // |+>

    auto result = table.pauli_expect(state.data(), 2,
                                     1ULL << 0,  // bit_flip_mask (X on q0)
                                     0,          // phase_flip_mask (no Z/Y phase)
                                     0,          // pivot qubit
                                     0           // y_count
    );

    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

TEST(ExpectationKernelTest, PauliY_ZeroState) {
    const auto& table = kernels::GetDispatchF64();
    // <0|Y|0> = 0
    std::vector<std::complex<double>> state(2);
    state[0] = std::complex<double>(1, 0);  // |0>

    auto result = table.pauli_expect(state.data(), 2,
                                     1ULL << 0,  // bit_flip_mask (Y has X component)
                                     1ULL << 0,  // phase_flip_mask (Y has Z component)
                                     0,          // pivot qubit
                                     1           // y_count = 1
    );

    EXPECT_NEAR(std::abs(result), 0.0, 1e-10);
}

TEST(ExpectationKernelTest, PauliX_MinusState) {
    const auto& table = kernels::GetDispatchF64();
    // <-|X|-> = -1
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(-inv_sqrt2, 0);  // |->

    auto result = table.pauli_expect(state.data(), 2,
                                     1ULL << 0,  // bit_flip_mask
                                     0, 0, 0);

    EXPECT_NEAR(std::real(result), -1.0, 1e-10);
}

// ============================================================
// 多 qubit Pauli 期望值测试 (Bell 态)
// ============================================================

TEST(ExpectationKernelTest, BellState_ZZ) {
    const auto& table = kernels::GetDispatchF64();
    // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, ⟨Z₀Z₁⟩ = 1
    std::vector<std::complex<double>> state(4);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);  // |00⟩
    state[3] = std::complex<double>(inv_sqrt2, 0);  // |11⟩

    // Z₀Z₁: bit_flip_mask=0, phase_flip_mask = qubit 0 and 1
    uint64_t phase_flip_mask = (1ULL << 0) | (1ULL << 1);
    auto result = table.pauli_expect(state.data(), 4, 0, phase_flip_mask, 0, 0);

    // ⟨Φ⁺|Z₀Z₁|Φ⁺⟩ = (+1*0.5) + (+1*0.5) = 1
    // |00⟩: popcount(00 & 11)=0, sign=+1, prob=0.5 → +0.5
    // |11⟩: popcount(11 & 11)=2, sign=+1, prob=0.5 → +0.5
    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

TEST(ExpectationKernelTest, BellState_XX) {
    const auto& table = kernels::GetDispatchF64();
    // |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, ⟨X₀X₁⟩ = 1
    std::vector<std::complex<double>> state(4);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);  // |00⟩
    state[3] = std::complex<double>(inv_sqrt2, 0);  // |11⟩

    // X₀X₁: bit_flip_mask = qubit 0 and 1, phase_flip_mask = 0, pivot = 0
    uint64_t bit_flip_mask = (1ULL << 0) | (1ULL << 1);
    auto result = table.pauli_expect(state.data(), 4, bit_flip_mask, 0, 0, 0);

    // ⟨Φ⁺|X₀X₁|Φ⁺⟩ = 1
    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

TEST(ExpectationKernelTest, BellState_ZZ_NegativeCorrelation) {
    const auto& table = kernels::GetDispatchF64();
    // |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2, ⟨Z₀Z₁⟩ = -1
    std::vector<std::complex<double>> state(4);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[1] = std::complex<double>(inv_sqrt2, 0);  // |01⟩
    state[2] = std::complex<double>(inv_sqrt2, 0);  // |10⟩

    uint64_t phase_flip_mask = (1ULL << 0) | (1ULL << 1);
    auto result = table.pauli_expect(state.data(), 4, 0, phase_flip_mask, 0, 0);

    // |01⟩: popcount(01 & 11)=1, sign=-1, prob=0.5 → -0.5
    // |10⟩: popcount(10 & 11)=1, sign=-1, prob=0.5 → -0.5
    EXPECT_NEAR(std::real(result), -1.0, 1e-10);
}

// ============================================================
// 3-qubit GHZ 态 Pauli 期望值测试
// ============================================================

TEST(ExpectationKernelTest, GHZ_ZZZ) {
    const auto& table = kernels::GetDispatchF64();
    // |GHZ⟩ = (|000⟩ + |111⟩)/√2, ⟨Z₀Z₁Z₂⟩ = 0
    // Z₀Z₁Z₂|000⟩ = +|000⟩, Z₀Z₁Z₂|111⟩ = -|111⟩
    // expectation = (1 * 0.5) + (-1 * 0.5) = 0
    std::vector<std::complex<double>> state(8);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);  // |000⟩
    state[7] = std::complex<double>(inv_sqrt2, 0);  // |111⟩

    uint64_t phase_flip_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 2);
    auto result = table.pauli_expect(state.data(), 8, 0, phase_flip_mask, 0, 0);

    EXPECT_NEAR(std::abs(result), 0.0, 1e-10);
}

TEST(ExpectationKernelTest, GHZ_XXX) {
    const auto& table = kernels::GetDispatchF64();
    // |GHZ⟩ = (|000⟩ + |111⟩)/√2, ⟨X₀X₁X₂⟩ = 1
    std::vector<std::complex<double>> state(8);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);  // |000⟩
    state[7] = std::complex<double>(inv_sqrt2, 0);  // |111⟩

    uint64_t bit_flip_mask = (1ULL << 0) | (1ULL << 1) | (1ULL << 2);
    auto result = table.pauli_expect(state.data(), 8, bit_flip_mask, 0, 0, 0);

    // XXX flips all bits: 000 ↔ 111
    // contribution from (000, 111) pair: 2 * Re(conj(inv_sqrt2) * inv_sqrt2) = 2 * 0.5 = 1
    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

TEST(ExpectationKernelTest, GHZ_SingleZ_IsZero) {
    const auto& table = kernels::GetDispatchF64();
    // 对 GHZ 态，单 qubit 约化态为最大混合态，⟨Z₀⟩ = 0
    std::vector<std::complex<double>> state(8);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[7] = std::complex<double>(inv_sqrt2, 0);

    auto result = table.pauli_expect(state.data(), 8,
                                     0,          // bit_flip_mask
                                     1ULL << 0,  // phase_flip_mask (Z on q0)
                                     0, 0);

    EXPECT_NEAR(std::abs(result), 0.0, 1e-10);
}

// ============================================================
// Y 相关期望值测试
// ============================================================

TEST(ExpectationKernelTest, PauliY_EigenstatePlus) {
    const auto& table = kernels::GetDispatchF64();
    // |y+⟩ = (|0⟩ + i|1⟩)/√2, <y+|Y|y+> = 1
    std::vector<std::complex<double>> state(2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    state[0] = std::complex<double>(inv_sqrt2, 0);
    state[1] = std::complex<double>(0, inv_sqrt2);  // i/√2

    auto result = table.pauli_expect(state.data(), 2,
                                     1ULL << 0,  // bit_flip_mask (Y has X component)
                                     1ULL << 0,  // phase_flip_mask (Y has Z component)
                                     0,          // pivot qubit
                                     1           // y_count = 1
    );

    EXPECT_NEAR(std::real(result), 1.0, 1e-10);
}

// ============================================================
// SimulationExecutor 端到端 observable 聚合测试
// ============================================================

TEST(ExecutorExpectationTest, SingleZOnZeroState) {
    // 对 |0⟩ 测量 ⟨Z⟩，结果应为 1
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;

    runtime::ObservableTerm obs;
    obs.label = "Z0";
    obs.coefficient = std::complex<double>(2.0, 0);  // 2*Z
    obs.paulis = {{0, 'Z'}};
    prog.observables.push_back(obs);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("Z0"));
    // ⟨2Z⟩ = 2 * 1 = 2
    EXPECT_NEAR(result.expectations_real.at("Z0"), 2.0, 1e-10);
}

TEST(ExecutorExpectationTest, BellStateWithObservables) {
    // 制备 Bell 态 |Φ⁺⟩，验证 ⟨Z₀Z₁⟩=1, ⟨X₀X₁⟩=1
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;

    // H on q0, then CX q0→q1 → Bell state
    runtime::ProgramOperation h_op = MakeHGate(0);
    runtime::ProgramOperation cx_op = MakeCXGate(0, 1);
    prog.ops = {h_op, cx_op};

    // 添加两个可观测量
    runtime::ObservableTerm obs_zz;
    obs_zz.label = "ZZ";
    obs_zz.coefficient = std::complex<double>(1.0, 0);
    obs_zz.paulis = {{0, 'Z'}, {1, 'Z'}};
    prog.observables.push_back(obs_zz);

    runtime::ObservableTerm obs_xx;
    obs_xx.label = "XX";
    obs_xx.coefficient = std::complex<double>(1.0, 0);
    obs_xx.paulis = {{0, 'X'}, {1, 'X'}};
    prog.observables.push_back(obs_xx);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("ZZ"));
    ASSERT_TRUE(result.expectations_real.count("XX"));
    EXPECT_NEAR(result.expectations_real.at("ZZ"), 1.0, 1e-10);
    EXPECT_NEAR(result.expectations_real.at("XX"), 1.0, 1e-10);
}

TEST(ExecutorExpectationTest, ObservableWithCoefficient) {
    // H = 0.5*Z₀ + 0.5*X₀ on |0⟩
    // ⟨0|H|0⟩ = 0.5*1 + 0.5*0 = 0.5
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;

    runtime::ObservableTerm obs_z;
    obs_z.label = "H_Z_part";
    obs_z.coefficient = std::complex<double>(0.5, 0);
    obs_z.paulis = {{0, 'Z'}};
    prog.observables.push_back(obs_z);

    runtime::ObservableTerm obs_x;
    obs_x.label = "H_X_part";
    obs_x.coefficient = std::complex<double>(0.5, 0);
    obs_x.paulis = {{0, 'X'}};
    prog.observables.push_back(obs_x);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("H_Z_part"));
    ASSERT_TRUE(result.expectations_real.count("H_X_part"));
    EXPECT_NEAR(result.expectations_real.at("H_Z_part"), 0.5, 1e-10);  // 0.5 * ⟨Z⟩ = 0.5
    EXPECT_NEAR(result.expectations_real.at("H_X_part"), 0.0, 1e-10);  // 0.5 * ⟨X⟩ = 0.0
}

TEST(ExecutorExpectationTest, EmptyObservablesWhenNotRequested) {
    // 不设置 return_expectations 时，expectations_real 应为空
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;

    runtime::ObservableTerm obs;
    obs.label = "Z0";
    obs.coefficient = std::complex<double>(1.0, 0);
    obs.paulis = {{0, 'Z'}};
    prog.observables.push_back(obs);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = false;  // 不请求可观测量

    auto result = exec.Run(prog, req);

    // 未请求期望值时不应计算
    EXPECT_TRUE(result.expectations_real.empty());
}

TEST(ExecutorExpectationTest, Float32Precision) {
    // 验证 float32 精度下可观测量计算也能工作
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;

    runtime::ObservableTerm obs;
    obs.label = "Z0";
    obs.coefficient = std::complex<double>(1.0, 0);
    obs.paulis = {{0, 'Z'}};
    prog.observables.push_back(obs);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat32;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("Z0"));
    EXPECT_NEAR(result.expectations_real.at("Z0"), 1.0, 1e-6);  // float32 精度较低
}

TEST(ExecutorExpectationTest, XGateThenExpectation) {
    // X|0⟩ = |1⟩, ⟨Z⟩ = -1
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops = {MakeXGate(0)};

    runtime::ObservableTerm obs;
    obs.label = "Z0";
    obs.coefficient = std::complex<double>(1.0, 0);
    obs.paulis = {{0, 'Z'}};
    prog.observables.push_back(obs);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("Z0"));
    EXPECT_NEAR(result.expectations_real.at("Z0"), -1.0, 1e-10);
}

TEST(ExecutorExpectationTest, MultipleObservablesAggregation) {
    // |Φ⁺⟩ Bell state, H = Z₀Z₁ - X₀X₁
    // ⟨H⟩ = 1 - 1 = 0
    runtime::QSALProgram prog;
    prog.num_qubits = 2;
    prog.num_clbits = 0;

    runtime::ProgramOperation h_op = MakeHGate(0);
    runtime::ProgramOperation cx_op = MakeCXGate(0, 1);
    prog.ops = {h_op, cx_op};

    runtime::ObservableTerm obs_zz;
    obs_zz.label = "ZZ";
    obs_zz.coefficient = std::complex<double>(1.0, 0);
    obs_zz.paulis = {{0, 'Z'}, {1, 'Z'}};
    prog.observables.push_back(obs_zz);

    runtime::ObservableTerm obs_xx_neg;
    obs_xx_neg.label = "neg_XX";
    obs_xx_neg.coefficient = std::complex<double>(-1.0, 0);
    obs_xx_neg.paulis = {{0, 'X'}, {1, 'X'}};
    prog.observables.push_back(obs_xx_neg);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    ASSERT_TRUE(result.expectations_real.count("ZZ"));
    ASSERT_TRUE(result.expectations_real.count("neg_XX"));
    EXPECT_NEAR(result.expectations_real.at("ZZ"), 1.0, 1e-10);
    EXPECT_NEAR(result.expectations_real.at("neg_XX"), -1.0, 1e-10);
}

TEST(ExecutorExpectationTest, TermsWithSameLabelAreSummed) {
    // |0>: <Z>=1 and <X>=0.  Both Pauli terms belong to one Hamiltonian.
    runtime::QSALProgram prog;
    prog.num_qubits = 1;

    runtime::ObservableTerm z_term;
    z_term.label = "energy";
    z_term.coefficient = std::complex<double>(0.75, 0.0);
    z_term.paulis = {{0, 'Z'}};
    prog.observables.push_back(z_term);

    runtime::ObservableTerm x_term;
    x_term.label = "energy";
    x_term.coefficient = std::complex<double>(-0.25, 0.0);
    x_term.paulis = {{0, 'X'}};
    prog.observables.push_back(x_term);

    executor::SimulationExecutor exec({});
    executor::ExecutionRequest req;
    req.return_expectations = true;
    const auto result = exec.Run(prog, req);

    ASSERT_EQ(result.expectations_real.size(), 1);
    EXPECT_NEAR(result.expectations_real.at("energy"), 0.75, 1e-10);
    ASSERT_EQ(result.expectations_complex.size(), 1);
    EXPECT_NEAR(std::real(result.expectations_complex.at("energy")), 0.75, 1e-10);
}

TEST(ExecutorExpectationTest, StatevectorAndExpectationsTogether) {
    // 同时请求 statevector 和期望值
    runtime::QSALProgram prog;
    prog.num_qubits = 1;
    prog.num_clbits = 0;
    prog.ops = {MakeHGate(0)};  // |+⟩ state

    runtime::ObservableTerm obs;
    obs.label = "X0";
    obs.coefficient = std::complex<double>(1.0, 0);
    obs.paulis = {{0, 'X'}};
    prog.observables.push_back(obs);

    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    executor::SimulationExecutor exec(opts);

    executor::ExecutionRequest req;
    req.return_statevector = true;
    req.return_expectations = true;

    auto result = exec.Run(prog, req);

    // statevector 应包含 |+⟩
    ASSERT_EQ(result.statevector_f64.size(), 2);
    double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), inv_sqrt2, 1e-10);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), inv_sqrt2, 1e-10);

    // ⟨X⟩ = 1 for |+⟩
    ASSERT_TRUE(result.expectations_real.count("X0"));
    EXPECT_NEAR(result.expectations_real.at("X0"), 1.0, 1e-10);
}
