#include <cmath>

#include <complex>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/executor/gradient_executor.hpp"
#include "quspect_native/runtime/program.hpp"

using namespace quspect::native;

namespace {

runtime::GateOperation FixedGate(const std::string& name, std::vector<uint32_t> qubits) {
    runtime::GateOperation gate;
    gate.opcode = runtime::OpCode::kNamedGate;
    gate.name = name;
    gate.qubits = std::move(qubits);
    return gate;
}

runtime::ParameterExpression Expression(double constant, std::initializer_list<std::pair<std::string, double>> terms) {
    runtime::ParameterExpression expression;
    expression.constant = constant;
    expression.terms.assign(terms.begin(), terms.end());
    return expression;
}

runtime::GateOperation Rotation(const std::string& name, uint32_t qubit,
                                const runtime::ParameterExpression& expression) {
    auto gate = FixedGate(name, {qubit});
    gate.parameter_expressions.push_back(expression);
    return gate;
}

void AddObservable(runtime::QSALProgram& program, const std::string& label, double coefficient,
                   std::vector<std::pair<uint32_t, char>> paulis) {
    runtime::ObservableTerm term;
    term.label = label;
    term.coefficient = {coefficient, 0.0};
    term.paulis = std::move(paulis);
    program.observables.push_back(std::move(term));
}

double Expectation(const runtime::QSALProgram& program, const std::unordered_map<std::string, double>& values,
                   std::size_t observable = 0) {
    return executor::AdjointGradientExecutor()
        .Run(program, values, executor::PrecisionKind::kFloat64)
        .expectations.at(observable);
}

}  // namespace

TEST(AdjointGradientTest, RyZExpectationMatchesAnalyticFormula) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z", 1.0, {{0, 'Z'}});

    const double theta = 0.731;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}},
                                                                executor::PrecisionKind::kFloat64);

    ASSERT_EQ(result.parameter_names, std::vector<std::string>({"theta"}));
    ASSERT_EQ(result.observable_labels, std::vector<std::string>({"Z"}));
    ASSERT_EQ(result.expectations.size(), 1);
    ASSERT_EQ(result.gradients.size(), 1);
    EXPECT_NEAR(result.expectations[0], std::cos(theta), 1e-12);
    EXPECT_NEAR(result.gradients[0][0], -std::sin(theta), 1e-12);
}

TEST(AdjointGradientTest, RxWorksInFloatPrecisionAndReturnsDouble) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("rx", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z", 1.0, {{0, 'Z'}});

    const double theta = -0.43;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}},
                                                                executor::PrecisionKind::kFloat32);

    EXPECT_NEAR(result.expectations[0], std::cos(theta), 2e-6);
    EXPECT_NEAR(result.gradients[0][0], -std::sin(theta), 2e-6);
}

TEST(AdjointGradientTest, PauliYActionAndRzGradientUseCorrectPhase) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(FixedGate("h", {0}));
    program.ops.push_back(Rotation("rz", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Y", 1.0, {{0, 'Y'}});

    const double theta = 0.37;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}});

    EXPECT_NEAR(result.expectations[0], std::sin(theta), 1e-12);
    EXPECT_NEAR(result.gradients[0][0], std::cos(theta), 1e-12);
}

TEST(AdjointGradientTest, AffineExpressionsAndSharedParametersAccumulateInProgramOrder) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    // Deliberately not lexical order: result columns must follow this vector.
    program.parameter_names = {"b", "a"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.1, {{"a", 2.0}, {"b", -0.25}})));
    program.ops.push_back(Rotation("ry", 0, Expression(-0.2, {{"a", -0.5}, {"b", 0.75}})));
    AddObservable(program, "energy", 1.0, {{0, 'Z'}});

    const double a = 0.41;
    const double b = -0.28;
    const double total_angle = -0.1 + 1.5 * a + 0.5 * b;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"a", a}, {"b", b}});

    ASSERT_EQ(result.parameter_names, std::vector<std::string>({"b", "a"}));
    EXPECT_NEAR(result.expectations[0], std::cos(total_angle), 1e-12);
    EXPECT_NEAR(result.gradients[0][0], -0.5 * std::sin(total_angle), 1e-12);
    EXPECT_NEAR(result.gradients[0][1], -1.5 * std::sin(total_angle), 1e-12);
}

TEST(AdjointGradientTest, PreparedObjectOwnsItsTapeAndRunsRepeatedParameterPoints) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z", 1.0, {{0, 'Z'}});
    AddObservable(program, "X", 1.0, {{0, 'X'}});

    executor::PreparedAdjointGradient prepared(program);
    ASSERT_EQ(prepared.parameter_names(), std::vector<std::string>({"theta"}));
    ASSERT_EQ(prepared.observable_labels(), std::vector<std::string>({"Z", "X"}));

    // The prepared plan owns its program/tape and is independent of the source.
    program.ops.clear();
    program.observables.clear();
    program.parameter_names.clear();

    const auto first = prepared.Run({{"theta", 0.1}});
    const auto second = prepared.Run({{"theta", 0.8}});
    const executor::PreparedAdjointGradient copied = prepared;
    const auto third = copied.Run({{"theta", -0.35}});

    EXPECT_NEAR(first.expectations[0], std::cos(0.1), 1e-12);
    EXPECT_NEAR(first.gradients[0][0], -std::sin(0.1), 1e-12);
    EXPECT_NEAR(second.expectations[1], std::sin(0.8), 1e-12);
    EXPECT_NEAR(second.gradients[1][0], std::cos(0.8), 1e-12);
    EXPECT_NEAR(third.expectations[0], std::cos(-0.35), 1e-12);
    EXPECT_NEAR(third.gradients[0][0], -std::sin(-0.35), 1e-12);
}

TEST(AdjointGradientTest, PreparedObjectSupportsConcurrentReadOnlyRuns) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z", 1.0, {{0, 'Z'}});
    const executor::PreparedAdjointGradient prepared(program);

    auto first = std::async(std::launch::async, [&prepared] { return prepared.Run({{"theta", 0.2}}); });
    auto second = std::async(std::launch::async, [&prepared] { return prepared.Run({{"theta", -0.7}}); });
    const auto first_result = first.get();
    const auto second_result = second.get();

    EXPECT_NEAR(first_result.expectations[0], std::cos(0.2), 1e-12);
    EXPECT_NEAR(first_result.gradients[0][0], -std::sin(0.2), 1e-12);
    EXPECT_NEAR(second_result.expectations[0], std::cos(-0.7), 1e-12);
    EXPECT_NEAR(second_result.gradients[0][0], -std::sin(-0.7), 1e-12);
}

TEST(AdjointGradientTest, MultipleTermsWithSameLabelFormOneHamiltonian) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "energy", 0.5, {{0, 'Z'}});
    AddObservable(program, "energy", 0.25, {{0, 'Z'}});
    AddObservable(program, "energy", 0.2, {});  // identity
    AddObservable(program, "x", 1.0, {{0, 'X'}});

    const double theta = 0.62;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}});

    ASSERT_EQ(result.observable_labels, std::vector<std::string>({"energy", "x"}));
    ASSERT_EQ(result.expectations.size(), 2);
    ASSERT_EQ(result.gradients.size(), 2);
    EXPECT_NEAR(result.expectations[0], 0.75 * std::cos(theta) + 0.2, 1e-12);
    EXPECT_NEAR(result.gradients[0][0], -0.75 * std::sin(theta), 1e-12);
    EXPECT_NEAR(result.expectations[1], std::sin(theta), 1e-12);
    EXPECT_NEAR(result.gradients[1][0], std::cos(theta), 1e-12);
}

TEST(AdjointGradientTest, MultiQubitCircuitMatchesCentralFiniteDifference) {
    runtime::QSALProgram program;
    program.num_qubits = 2;
    program.parameter_names = {"a", "b"};
    program.ops.push_back(FixedGate("h", {0}));
    program.ops.push_back(Rotation("ry", 1, Expression(0.1, {{"a", 1.0}})));
    program.ops.push_back(FixedGate("cx", {0, 1}));
    program.ops.push_back(Rotation("rz", 0, Expression(-0.2, {{"b", 1.0}})));
    program.ops.push_back(Rotation("rx", 1, Expression(0.0, {{"a", 0.3}, {"b", -0.4}})));
    AddObservable(program, "H", 0.7, {{0, 'Z'}});
    AddObservable(program, "H", -0.2, {{0, 'X'}, {1, 'X'}});
    AddObservable(program, "H", 0.35, {{0, 'Y'}, {1, 'Z'}});

    const std::unordered_map<std::string, double> values{{"a", 0.31}, {"b", -0.47}};
    const auto result = executor::AdjointGradientExecutor().Run(program, values);
    const double epsilon = 1e-6;

    for (std::size_t p = 0; p < program.parameter_names.size(); ++p) {
        const auto& name = program.parameter_names[p];
        auto plus = values;
        auto minus = values;
        plus[name] += epsilon;
        minus[name] -= epsilon;
        const double finite_difference = (Expectation(program, plus) - Expectation(program, minus)) / (2 * epsilon);
        EXPECT_NEAR(result.gradients[0][p], finite_difference, 2e-9) << "parameter " << name;
    }
}

TEST(AdjointGradientTest, HighQubitZGateMatchesAnalyticGradient) {
    runtime::QSALProgram program;
    program.num_qubits = 4;
    program.parameter_names = {"theta"};
    for (uint32_t qubit = 0; qubit < 4; ++qubit) {
        program.ops.push_back(FixedGate("h", {qubit}));
    }
    program.ops.push_back(FixedGate("z", {3}));
    program.ops.push_back(Rotation("ry", 3, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z3", 1.0, {{3, 'Z'}});

    const double theta = 0.37;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}});
    const auto float_result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}},
                                                                      executor::PrecisionKind::kFloat32);

    EXPECT_NEAR(result.expectations[0], std::sin(theta), 1e-12);
    EXPECT_NEAR(result.gradients[0][0], std::cos(theta), 1e-12);
    EXPECT_NEAR(float_result.expectations[0], std::sin(theta), 2e-6);
    EXPECT_NEAR(float_result.gradients[0][0], std::cos(theta), 2e-6);
}

TEST(AdjointGradientTest, HighQubitCZGateMatchesAnalyticGradient) {
    runtime::QSALProgram program;
    program.num_qubits = 4;
    program.parameter_names = {"theta"};
    for (uint32_t qubit = 0; qubit < 4; ++qubit) {
        program.ops.push_back(FixedGate("h", {qubit}));
    }
    program.ops.push_back(FixedGate("cz", {2, 3}));
    program.ops.push_back(Rotation("ry", 2, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "X2Z3", 1.0, {{2, 'X'}, {3, 'Z'}});

    const double theta = -0.43;
    const auto result = executor::AdjointGradientExecutor().Run(program, {{"theta", theta}});

    EXPECT_NEAR(result.expectations[0], std::cos(theta), 1e-12);
    EXPECT_NEAR(result.gradients[0][0], -std::sin(theta), 1e-12);
}

TEST(AdjointGradientTest, RejectsParameterBindingErrors) {
    runtime::QSALProgram program;
    program.num_qubits = 1;
    program.parameter_names = {"theta"};
    program.ops.push_back(Rotation("ry", 0, Expression(0.0, {{"theta", 1.0}})));
    AddObservable(program, "Z", 1.0, {{0, 'Z'}});

    executor::AdjointGradientExecutor gradient;
    EXPECT_THROW(gradient.Run(program, {}), std::invalid_argument);
    EXPECT_THROW(gradient.Run(program, {{"theta", 0.2}, {"extra", 1.0}}), std::invalid_argument);
}

TEST(AdjointGradientTest, RejectsNonHermitianAndUnsupportedPrograms) {
    runtime::QSALProgram non_hermitian;
    non_hermitian.num_qubits = 1;
    runtime::ObservableTerm bad_term;
    bad_term.label = "bad";
    bad_term.coefficient = {1.0, 0.1};
    bad_term.paulis = {{0, 'Z'}};
    non_hermitian.observables.push_back(bad_term);

    executor::AdjointGradientExecutor gradient;
    EXPECT_THROW(gradient.Run(non_hermitian, {}), std::invalid_argument);

    runtime::QSALProgram dense;
    dense.num_qubits = 1;
    runtime::GateOperation dense_gate;
    dense_gate.opcode = runtime::OpCode::kDenseGate;
    dense_gate.name = "custom";
    dense_gate.qubits = {0};
    dense.ops.push_back(dense_gate);
    AddObservable(dense, "Z", 1.0, {{0, 'Z'}});
    EXPECT_THROW(gradient.Run(dense, {}), std::invalid_argument);

    runtime::QSALProgram empty_observable;
    empty_observable.num_qubits = 1;
    EXPECT_THROW(gradient.Run(empty_observable, {}), std::invalid_argument);
}
