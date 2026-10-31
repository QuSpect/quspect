#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/executor/executor.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

namespace {

runtime::ChannelOperation MakeBitFlipChannel(uint32_t qubit, double probability) {
    runtime::ChannelOperation channel;
    channel.channel_kind = "bit_flip";
    channel.qubits = {qubit};
    channel.params_f64 = {probability};
    channel.params_f32 = {static_cast<float>(probability)};
    channel.num_kraus_ops = 2;

    const double identity_scale = std::sqrt(1.0 - probability);
    const double flip_scale = std::sqrt(probability);
    channel.kraus_operators_f64 = {
        {identity_scale, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {identity_scale, 0.0},
        {0.0, 0.0}, {flip_scale, 0.0}, {flip_scale, 0.0}, {0.0, 0.0},
    };
    channel.kraus_operators_f32.assign(channel.kraus_operators_f64.begin(), channel.kraus_operators_f64.end());
    return channel;
}

runtime::QSALProgram MakeTwoIndependentBitFlipProgram() {
    runtime::QSALProgram program;
    program.num_qubits = 2;
    program.num_clbits = 2;
    program.ops.emplace_back(MakeBitFlipChannel(0, 0.5));
    program.ops.emplace_back(MakeBitFlipChannel(1, 0.5));

    runtime::MeasureOperation measure;
    measure.qubits = {0, 1};
    measure.clbits = {0, 1};
    program.ops.emplace_back(std::move(measure));
    return program;
}

}  // namespace

TEST(ExecutorTest, SimulationExecutorConstruction) {
    executor::SimulationExecutor::Options opts;
    opts.precision = executor::PrecisionKind::kFloat64;
    opts.seed = 42;

    executor::SimulationExecutor sim(opts);
    // 构造成功
}

TEST(ExecutorTest, SimulationExecutorRunReturnsStatevector) {
    // Phase 2: 执行 X 门线路，验证返回 statevector
    executor::SimulationExecutor::Options opts;
    executor::SimulationExecutor sim(opts);

    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.num_clbits = 0;

    runtime::GateOperation xgate;
    xgate.opcode = runtime::OpCode::kNamedGate;
    xgate.name = "x";
    xgate.qubits = {0};
    program.ops.push_back(xgate);

    executor::ExecutionRequest req;
    req.return_statevector = true;

    auto result = sim.Run(program, req);

    // |0> -> X -> |1>
    EXPECT_EQ(result.statevector_f64.size(), 2);
    EXPECT_NEAR(std::real(result.statevector_f64[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::real(result.statevector_f64[1]), 1.0, 1e-12);
    EXPECT_EQ(result.metadata.at("num_qubits"), "1");
    EXPECT_EQ(result.metadata.at("precision"), "double");
}

TEST(ExecutorTest, HardwareExecutorThrows) {
    executor::HardwareExecutor hw;
    runtime::QSALProgram program;
    executor::ExecutionRequest req;

    EXPECT_THROW(hw.Run(program, req), std::runtime_error);
}

TEST(ExecutorTest, ConsecutiveChannelsUseIndependentRandomDrawsWithinEveryShot) {
    const auto program = MakeTwoIndependentBitFlipProgram();
    executor::SimulationExecutor::Options opts;
    opts.seed = 20260810;
    executor::SimulationExecutor sim(opts);

    executor::ExecutionRequest req;
    req.shots = 4096;
    req.return_samples = true;
    const auto result = sim.Run(program, req);

    ASSERT_EQ(result.packed_samples.size(), req.shots);
    ASSERT_EQ(result.counts.size(), 4);
    // If every channel recreated an RNG from the same per-shot seed, the two
    // p=0.5 flips would be perfectly correlated and 01/10 would never occur.
    for (const char* outcome : {"00", "01", "10", "11"}) {
        ASSERT_TRUE(result.counts.count(outcome)) << outcome;
        EXPECT_GT(result.counts.at(outcome), 700) << outcome;
        EXPECT_LT(result.counts.at(outcome), 1400) << outcome;
    }
    EXPECT_EQ(result.metadata.at("measurement_mode"), "noisy_trajectory");
    EXPECT_EQ(result.metadata.at("has_noise"), "true");
}

TEST(ExecutorTest, NoisyTrajectorySeedIsExactlyReproducible) {
    const auto program = MakeTwoIndependentBitFlipProgram();
    executor::SimulationExecutor first({executor::PrecisionKind::kFloat64, 1});
    executor::SimulationExecutor second({executor::PrecisionKind::kFloat64, 2});

    executor::ExecutionRequest req;
    req.shots = 512;
    req.return_samples = true;
    req.seed = 8675309;

    const auto first_result = first.Run(program, req);
    const auto second_result = second.Run(program, req);
    EXPECT_EQ(first_result.packed_samples, second_result.packed_samples);
    EXPECT_EQ(first_result.counts, second_result.counts);
}

TEST(ExecutorTest, NoisyTerminalMeasurementHonorsClassicalBitMapping) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.num_clbits = 3;
    program.ops.emplace_back(MakeBitFlipChannel(0, 1.0));

    runtime::MeasureOperation measure;
    measure.qubits = {0};
    measure.clbits = {2};
    program.ops.emplace_back(std::move(measure));

    executor::SimulationExecutor sim({executor::PrecisionKind::kFloat64, 42});
    executor::ExecutionRequest req;
    req.shots = 32;
    req.return_samples = true;
    const auto result = sim.Run(program, req);

    ASSERT_EQ(result.packed_samples.size(), req.shots);
    for (const uint64_t sample : result.packed_samples) {
        EXPECT_EQ(sample, 1ULL << 2);
    }
    ASSERT_EQ(result.counts.size(), 1);
    EXPECT_EQ(result.counts.at("100"), req.shots);
}
