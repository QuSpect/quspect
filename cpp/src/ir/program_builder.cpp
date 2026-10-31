// program_builder.cpp — 编程式 QSALProgram 构建辅助
// Phase 5: 完整实现

#include "quspect_native/ir/program_builder.hpp"

#include <cstdint>
#include <memory>

#include "quspect_native/runtime/program.hpp"

namespace quspect::native::ir {

ProgramBuilder::ProgramBuilder() : prog_(std::make_unique<runtime::QSALProgram>()) {
}

ProgramBuilder::ProgramBuilder(uint32_t num_qubits, uint32_t num_clbits)
    : prog_(std::make_unique<runtime::QSALProgram>()) {
    prog_->num_qubits = num_qubits;
    prog_->num_clbits = num_clbits;
}

ProgramBuilder& ProgramBuilder::SetNumQubits(uint32_t n) {
    prog_->num_qubits = n;
    return *this;
}

ProgramBuilder& ProgramBuilder::SetNumClbits(uint32_t n) {
    prog_->num_clbits = n;
    return *this;
}

ProgramBuilder& ProgramBuilder::AddGate(const std::string& name, const std::vector<uint32_t>& qubits,
                                        const std::vector<double>& params) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = name;
    g.qubits = qubits;
    g.params_f64 = params;
    g.params_f32.assign(params.begin(), params.end());
    g.parameter_expressions.reserve(params.size());
    for (double parameter : params) {
        runtime::ParameterExpression expression;
        expression.constant = parameter;
        g.parameter_expressions.push_back(std::move(expression));
    }
    prog_->ops.push_back(std::move(g));
    return *this;
}

ProgramBuilder& ProgramBuilder::AddBarrier(const std::vector<uint32_t>& qubits) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kBarrier;
    g.name = "barrier";
    g.qubits = qubits;
    prog_->ops.push_back(std::move(g));
    return *this;
}

ProgramBuilder& ProgramBuilder::AddMeasure(const std::vector<uint32_t>& qubits, const std::vector<uint32_t>& clbits) {
    runtime::MeasureOperation m;
    m.qubits = qubits;
    m.clbits = clbits;
    prog_->ops.push_back(std::move(m));
    return *this;
}

ProgramBuilder& ProgramBuilder::AddObservable(const std::string& label,
                                              const std::vector<std::pair<uint32_t, char>>& paulis,
                                              std::complex<double> coefficient) {
    runtime::ObservableTerm term;
    term.label = label;
    term.coefficient = coefficient;
    term.paulis = paulis;
    prog_->observables.push_back(std::move(term));
    return *this;
}

std::unique_ptr<runtime::QSALProgram> ProgramBuilder::Build() {
    return std::move(prog_);
}

}  // namespace quspect::native::ir
