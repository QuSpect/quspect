#include "quspect_native/executor/gradient_executor.hpp"

#include <cmath>

#include <algorithm>
#include <array>
#include <cctype>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/common/parallel.hpp"
#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/runtime/program.hpp"
#include "quspect_native/runtime/state_vector.hpp"

namespace quspect::native::executor {
namespace {

constexpr double kHermitianTolerance = 1e-12;

using ParameterIndex = std::unordered_map<std::string, std::size_t>;
using PauliWord = std::vector<std::pair<uint32_t, char>>;

enum class GateKind {
    kId,
    kX,
    kY,
    kZ,
    kH,
    kS,
    kSdg,
    kT,
    kTdg,
    kRx,
    kRy,
    kRz,
    kCnot,
    kCz,
    kSwap,
    kCcnot,
};

struct CompiledGate {
    GateKind kind = GateKind::kId;
    std::array<uint32_t, 3> qubits{};
    uint32_t num_qubits = 0;
    // For rotations: intrinsic angle = angle_constant + sum(coeff * value[index]).
    double angle_constant = 0.0;
    std::vector<std::pair<std::size_t, double>> jacobian;
};

struct CompiledPauliTerm {
    double coefficient = 0.0;
    uint64_t bit_flip_mask = 0;
    uint64_t phase_flip_mask = 0;
    int y_count = 0;
};

struct CompiledHamiltonian {
    std::string label;
    std::vector<CompiledPauliTerm> terms;
};

struct PreparedData {
    uint32_t num_qubits = 0;
    std::vector<std::string> parameter_names;
    ParameterIndex parameter_indices;
    std::vector<CompiledGate> tape;
    std::vector<CompiledHamiltonian> hamiltonians;
    std::vector<std::string> observable_labels;
};

bool IsRotation(GateKind kind) {
    return kind == GateKind::kRx || kind == GateKind::kRy || kind == GateKind::kRz;
}

GateKind ParseGateKind(const std::string& name) {
    if (name == "id") {
        return GateKind::kId;
    }
    if (name == "x") {
        return GateKind::kX;
    }
    if (name == "y") {
        return GateKind::kY;
    }
    if (name == "z") {
        return GateKind::kZ;
    }
    if (name == "h") {
        return GateKind::kH;
    }
    if (name == "s") {
        return GateKind::kS;
    }
    if (name == "sdg") {
        return GateKind::kSdg;
    }
    if (name == "t") {
        return GateKind::kT;
    }
    if (name == "tdg") {
        return GateKind::kTdg;
    }
    if (name == "rx") {
        return GateKind::kRx;
    }
    if (name == "ry") {
        return GateKind::kRy;
    }
    if (name == "rz") {
        return GateKind::kRz;
    }
    if (name == "cx" || name == "cnot") {
        return GateKind::kCnot;
    }
    if (name == "cz") {
        return GateKind::kCz;
    }
    if (name == "swap") {
        return GateKind::kSwap;
    }
    if (name == "ccx" || name == "ccnot") {
        return GateKind::kCcnot;
    }
    throw std::invalid_argument("AdjointGradientExecutor: unsupported named gate '" + name + "'");
}

uint32_t ExpectedQubitCount(GateKind kind) {
    switch (kind) {
        case GateKind::kCnot:
        case GateKind::kCz:
        case GateKind::kSwap:
            return 2;
        case GateKind::kCcnot:
            return 3;
        default:
            return 1;
    }
}

ParameterIndex CompileParameterIndex(const std::vector<std::string>& parameter_names) {
    ParameterIndex indices;
    indices.reserve(parameter_names.size());
    for (std::size_t i = 0; i < parameter_names.size(); ++i) {
        const auto& name = parameter_names[i];
        if (name.empty()) {
            throw std::invalid_argument("AdjointGradientExecutor: parameter name cannot be empty");
        }
        if (!indices.emplace(name, i).second) {
            throw std::invalid_argument("AdjointGradientExecutor: duplicate parameter name '" + name + "'");
        }
    }
    return indices;
}

std::vector<double> BindParameterValues(const PreparedData& prepared,
                                        const std::unordered_map<std::string, double>& parameter_values) {
    std::vector<double> ordered_values(prepared.parameter_names.size(), 0.0);
    for (std::size_t i = 0; i < prepared.parameter_names.size(); ++i) {
        const auto& name = prepared.parameter_names[i];
        const auto value_it = parameter_values.find(name);
        if (value_it == parameter_values.end()) {
            throw std::invalid_argument("AdjointGradientExecutor: missing value for parameter '" + name + "'");
        }
        if (!std::isfinite(value_it->second)) {
            throw std::invalid_argument("AdjointGradientExecutor: non-finite value for parameter '" + name + "'");
        }
        ordered_values[i] = value_it->second;
    }
    for (const auto& [name, value] : parameter_values) {
        (void) value;
        if (prepared.parameter_indices.find(name) == prepared.parameter_indices.end()) {
            throw std::invalid_argument("AdjointGradientExecutor: unexpected value for parameter '" + name + "'");
        }
    }
    return ordered_values;
}

void ValidateGateQubits(const runtime::GateOperation& gate, GateKind kind, uint32_t num_qubits) {
    const uint32_t expected = ExpectedQubitCount(kind);
    if (gate.qubits.size() != expected) {
        throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name + "' expects "
                                    + std::to_string(expected) + " qubits, got " + std::to_string(gate.qubits.size()));
    }
    std::unordered_set<uint32_t> seen;
    for (uint32_t qubit : gate.qubits) {
        if (qubit >= num_qubits) {
            throw std::out_of_range("AdjointGradientExecutor: gate '" + gate.name + "' references qubit "
                                    + std::to_string(qubit) + " outside a " + std::to_string(num_qubits)
                                    + "-qubit program");
        }
        if (!seen.insert(qubit).second) {
            throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name + "' contains a duplicate qubit");
        }
    }
}

CompiledGate CompileNamedGate(const runtime::GateOperation& gate, uint32_t num_qubits,
                              const ParameterIndex& parameter_indices) {
    CompiledGate compiled;
    compiled.kind = ParseGateKind(gate.name);
    ValidateGateQubits(gate, compiled.kind, num_qubits);
    compiled.num_qubits = static_cast<uint32_t>(gate.qubits.size());
    std::copy(gate.qubits.begin(), gate.qubits.end(), compiled.qubits.begin());

    if (!IsRotation(compiled.kind)) {
        if (!gate.parameter_expressions.empty() || !gate.params_f64.empty() || !gate.params_f32.empty()) {
            throw std::invalid_argument("AdjointGradientExecutor: fixed gate '" + gate.name
                                        + "' cannot have parameters");
        }
        return compiled;
    }

    if (gate.parameter_expressions.size() > 1) {
        throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name
                                    + "' expects one parameter expression");
    }
    if (gate.params_f64.size() > 1 || gate.params_f32.size() > 1) {
        throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name
                                    + "' expects exactly one intrinsic parameter");
    }
    if (!gate.parameter_expressions.empty()) {
        const auto& expression = gate.parameter_expressions.front();
        if (!std::isfinite(expression.constant)) {
            throw std::invalid_argument("AdjointGradientExecutor: non-finite parameter-expression constant");
        }
        compiled.angle_constant = expression.constant;
        std::map<std::size_t, double> combined_terms;
        for (const auto& [name, coefficient] : expression.terms) {
            const auto parameter_it = parameter_indices.find(name);
            if (parameter_it == parameter_indices.end()) {
                throw std::invalid_argument("AdjointGradientExecutor: expression references unknown parameter '" + name
                                            + "'");
            }
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument("AdjointGradientExecutor: non-finite coefficient for parameter '" + name
                                            + "'");
            }
            combined_terms[parameter_it->second] += coefficient;
            if (!std::isfinite(combined_terms[parameter_it->second])) {
                throw std::invalid_argument("AdjointGradientExecutor: combined coefficient for parameter '" + name
                                            + "' is non-finite");
            }
        }
        for (const auto& [parameter_index, coefficient] : combined_terms) {
            if (coefficient != 0.0) {
                compiled.jacobian.emplace_back(parameter_index, coefficient);
            }
        }
    } else if (gate.params_f64.size() == 1) {
        compiled.angle_constant = gate.params_f64.front();
    } else if (gate.params_f32.size() == 1) {
        compiled.angle_constant = static_cast<double>(gate.params_f32.front());
    } else {
        throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name
                                    + "' requires exactly one numeric or symbolic parameter");
    }
    if (!std::isfinite(compiled.angle_constant)) {
        throw std::invalid_argument("AdjointGradientExecutor: gate '" + gate.name + "' has a non-finite angle");
    }
    return compiled;
}

std::vector<CompiledGate> CompileTape(const runtime::QSALProgram& program, const ParameterIndex& parameter_indices) {
    std::vector<CompiledGate> tape;
    tape.reserve(program.ops.size());
    for (const auto& operation : program.ops) {
        if (std::holds_alternative<runtime::MeasureOperation>(operation)) {
            throw std::invalid_argument("AdjointGradientExecutor: measurement operations are not supported");
        }
        if (std::holds_alternative<runtime::ChannelOperation>(operation)) {
            throw std::invalid_argument("AdjointGradientExecutor: noise channels are not supported");
        }
        const auto& gate = std::get<runtime::GateOperation>(operation);
        switch (gate.opcode) {
            case runtime::OpCode::kBarrier:
                continue;
            case runtime::OpCode::kMeasure:
                throw std::invalid_argument("AdjointGradientExecutor: measurement operations are not supported");
            case runtime::OpCode::kDenseGate:
                throw std::invalid_argument("AdjointGradientExecutor: dense gates are not supported");
            case runtime::OpCode::kDiagonalGate:
                throw std::invalid_argument("AdjointGradientExecutor: diagonal gates are not supported");
            case runtime::OpCode::kNamedGate:
                tape.push_back(CompileNamedGate(gate, program.num_qubits, parameter_indices));
                break;
        }
    }
    return tape;
}

char NormalizePauli(char pauli) {
    const char normalized = static_cast<char>(std::toupper(static_cast<unsigned char>(pauli)));
    if (normalized != 'I' && normalized != 'X' && normalized != 'Y' && normalized != 'Z') {
        throw std::invalid_argument("AdjointGradientExecutor: invalid Pauli character '" + std::string(1, pauli) + "'");
    }
    return normalized;
}

PauliWord CanonicalizePaulis(const runtime::ObservableTerm& term, uint32_t num_qubits) {
    PauliWord word;
    word.reserve(term.paulis.size());
    std::unordered_set<uint32_t> seen;
    for (const auto& [qubit, raw_pauli] : term.paulis) {
        if (qubit >= num_qubits) {
            throw std::out_of_range("AdjointGradientExecutor: observable '" + term.label + "' references qubit "
                                    + std::to_string(qubit) + " outside a " + std::to_string(num_qubits)
                                    + "-qubit program");
        }
        if (!seen.insert(qubit).second) {
            throw std::invalid_argument("AdjointGradientExecutor: observable '" + term.label
                                        + "' contains duplicate Pauli entries for qubit " + std::to_string(qubit));
        }
        const char pauli = NormalizePauli(raw_pauli);
        if (pauli != 'I') {
            word.emplace_back(qubit, pauli);
        }
    }
    std::sort(word.begin(), word.end());
    return word;
}

void CompileHamiltonians(const runtime::QSALProgram& program, PreparedData* prepared) {
    if (program.observables.empty()) {
        throw std::invalid_argument("AdjointGradientExecutor: at least one observable is required");
    }

    std::unordered_map<std::string, std::size_t> label_indices;
    std::vector<std::map<PauliWord, double>> grouped_terms;
    for (const auto& term : program.observables) {
        if (!std::isfinite(term.coefficient.real()) || !std::isfinite(term.coefficient.imag())) {
            throw std::invalid_argument("AdjointGradientExecutor: observable '" + term.label
                                        + "' has a non-finite coefficient");
        }
        if (std::abs(term.coefficient.imag()) > kHermitianTolerance) {
            throw std::invalid_argument("AdjointGradientExecutor: observable '" + term.label
                                        + "' is non-Hermitian; Pauli coefficients must be real");
        }
        auto label_it = label_indices.find(term.label);
        std::size_t label_index = 0;
        if (label_it == label_indices.end()) {
            label_index = prepared->observable_labels.size();
            prepared->observable_labels.push_back(term.label);
            label_indices.emplace(term.label, label_index);
            grouped_terms.emplace_back();
        } else {
            label_index = label_it->second;
        }
        auto& combined_coefficient = grouped_terms[label_index][CanonicalizePaulis(term, program.num_qubits)];
        combined_coefficient += term.coefficient.real();
        if (!std::isfinite(combined_coefficient)) {
            throw std::invalid_argument("AdjointGradientExecutor: combined coefficient for observable '" + term.label
                                        + "' is non-finite");
        }
    }

    prepared->hamiltonians.reserve(prepared->observable_labels.size());
    for (std::size_t i = 0; i < prepared->observable_labels.size(); ++i) {
        CompiledHamiltonian hamiltonian;
        hamiltonian.label = prepared->observable_labels[i];
        for (const auto& [paulis, coefficient] : grouped_terms[i]) {
            if (coefficient == 0.0) {
                continue;
            }
            const auto masks = runtime::ComputePauliMasks(paulis);
            hamiltonian.terms.push_back({coefficient, masks.bit_flip_mask, masks.phase_flip_mask, masks.y_count});
        }
        prepared->hamiltonians.push_back(std::move(hamiltonian));
    }
}

PreparedData PrepareProgram(const runtime::QSALProgram& program) {
    if (program.num_qubits == 0) {
        throw std::invalid_argument("AdjointGradientExecutor: num_qubits must be greater than zero");
    }
    if (program.num_qubits >= 63) {
        throw std::invalid_argument("AdjointGradientExecutor: num_qubits must be less than 63");
    }

    PreparedData prepared;
    prepared.num_qubits = program.num_qubits;
    prepared.parameter_names = program.parameter_names;
    prepared.parameter_indices = CompileParameterIndex(prepared.parameter_names);
    prepared.tape = CompileTape(program, prepared.parameter_indices);
    CompileHamiltonians(program, &prepared);
    return prepared;
}

template <typename FP>
const kernels::KernelDispatchTable<FP>& GetDispatch();

template <>
const kernels::KernelDispatchTable<float>& GetDispatch<float>() {
    return kernels::GetDispatchF32();
}

template <>
const kernels::KernelDispatchTable<double>& GetDispatch<double>() {
    return kernels::GetDispatchF64();
}

template <typename FP>
void ApplyRotation(runtime::StateVector<FP>& state, const kernels::KernelDispatchTable<FP>& table, GateKind kind,
                   uint32_t qubit, double angle) {
    using Complex = std::complex<FP>;
    const FP half_angle = static_cast<FP>(angle * 0.5);
    const FP cosine = std::cos(half_angle);
    const FP sine = std::sin(half_angle);
    if (kind == GateKind::kRx) {
        const Complex matrix[4] = {{cosine, 0}, {0, -sine}, {0, -sine}, {cosine, 0}};
        table.rx(state.data(), state.dim(), qubit, matrix);
    } else if (kind == GateKind::kRy) {
        const Complex matrix[4] = {{cosine, 0}, {-sine, 0}, {sine, 0}, {cosine, 0}};
        table.ry(state.data(), state.dim(), qubit, matrix);
    } else {
        const Complex matrix[4] = {std::exp(Complex(0, -half_angle)), Complex(0, 0), Complex(0, 0),
                                   std::exp(Complex(0, half_angle))};
        table.rz(state.data(), state.dim(), qubit, matrix);
    }
}

template <typename FP>
void ApplyGate(runtime::StateVector<FP>& state, const kernels::KernelDispatchTable<FP>& table, const CompiledGate& gate,
               double angle, bool dagger) {
    switch (gate.kind) {
        case GateKind::kId:
            break;
        case GateKind::kX:
            table.x(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kY:
            table.y(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kZ:
            table.z(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kH:
            table.h(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kS:
            (dagger ? table.sdg : table.s)(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kSdg:
            (dagger ? table.s : table.sdg)(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kT:
            (dagger ? table.tdg : table.t)(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kTdg:
            (dagger ? table.t : table.tdg)(state.data(), state.dim(), gate.qubits[0], nullptr);
            break;
        case GateKind::kRx:
        case GateKind::kRy:
        case GateKind::kRz:
            ApplyRotation(state, table, gate.kind, gate.qubits[0], dagger ? -angle : angle);
            break;
        case GateKind::kCnot: {
            const uint64_t control_mask = 1ULL << gate.qubits[0];
            table.cnot(state.data(), state.dim(), gate.qubits[1], control_mask, control_mask, nullptr);
            break;
        }
        case GateKind::kCz: {
            const uint64_t control_mask = 1ULL << gate.qubits[0];
            table.cz(state.data(), state.dim(), gate.qubits[1], control_mask, control_mask, nullptr);
            break;
        }
        case GateKind::kSwap:
            table.swap(state.data(), state.dim(), gate.qubits[0], gate.qubits[1]);
            break;
        case GateKind::kCcnot: {
            const uint64_t control_mask = (1ULL << gate.qubits[0]) | (1ULL << gate.qubits[1]);
            table.cnot(state.data(), state.dim(), gate.qubits[2], control_mask, control_mask, nullptr);
            break;
        }
    }
}

template <typename FP>
runtime::StateVector<FP> ApplyHamiltonian(const CompiledHamiltonian& hamiltonian,
                                          const runtime::StateVector<FP>& state) {
    using Complex = std::complex<FP>;
    runtime::StateVector<FP> output = state;
    std::fill(output.data(), output.data() + output.dim(), Complex(0, 0));

    for (const auto& term : hamiltonian.terms) {
        Complex y_phase(1, 0);
        switch (term.y_count & 3) {
            case 1:
                y_phase = Complex(0, 1);
                break;
            case 2:
                y_phase = Complex(-1, 0);
                break;
            case 3:
                y_phase = Complex(0, -1);
                break;
            default:
                break;
        }
        const FP coefficient = static_cast<FP>(term.coefficient);
        const Complex* input = state.data();
        Complex* out = output.data();
        // basis -> basis ^ bit_flip_mask 是双射，同一 term 内各迭代写入的 target 互不相同。
        common::ParallelForRange(
            static_cast<int64_t>(state.dim()), common::ThreadsForDim(state.dim()), [&](int64_t begin, int64_t end) {
                for (uint64_t basis = static_cast<uint64_t>(begin); basis < static_cast<uint64_t>(end); ++basis) {
                    const uint64_t target = basis ^ term.bit_flip_mask;
                    const FP sign = common::Parity64(basis & term.phase_flip_mask) ? FP(-1) : FP(1);
                    out[target] += coefficient * sign * y_phase * input[basis];
                }
            });
    }
    return output;
}

template <typename FP>
std::complex<double> InnerProduct(const runtime::StateVector<FP>& bra, const runtime::StateVector<FP>& ket) {
    const auto* bra_data = bra.data();
    const auto* ket_data = ket.data();
    const auto acc = common::BlockedReduce<2>(static_cast<int64_t>(bra.dim()), common::ThreadsForDim(bra.dim()),
                                              [&](int64_t begin, int64_t end, double* partial) {
                                                  double re = 0.0;
                                                  double im = 0.0;
                                                  for (int64_t i = begin; i < end; ++i) {
                                                      const double br = static_cast<double>(bra_data[i].real());
                                                      const double bi = static_cast<double>(bra_data[i].imag());
                                                      const double kr = static_cast<double>(ket_data[i].real());
                                                      const double ki = static_cast<double>(ket_data[i].imag());
                                                      // conj(b) * k = (br*kr + bi*ki) + i (br*ki - bi*kr)
                                                      re += br * kr + bi * ki;
                                                      im += br * ki - bi * kr;
                                                  }
                                                  partial[0] += re;
                                                  partial[1] += im;
                                              });
    return {acc[0], acc[1]};
}

template <typename FP>
std::complex<double> ExpectRotationDaggerDerivative(const runtime::StateVector<FP>& primal,
                                                    const runtime::StateVector<FP>& costate, const CompiledGate& gate,
                                                    double angle) {
    using Complex = std::complex<FP>;
    const FP half_angle = static_cast<FP>(angle * 0.5);
    const FP sine_half = std::sin(half_angle) * FP(0.5);
    const FP cosine_half = std::cos(half_angle) * FP(0.5);
    Complex derivative[4];
    if (gate.kind == GateKind::kRx) {
        derivative[0] = Complex(-sine_half, 0);
        derivative[1] = Complex(0, cosine_half);
        derivative[2] = Complex(0, cosine_half);
        derivative[3] = Complex(-sine_half, 0);
    } else if (gate.kind == GateKind::kRy) {
        derivative[0] = Complex(-sine_half, 0);
        derivative[1] = Complex(cosine_half, 0);
        derivative[2] = Complex(-cosine_half, 0);
        derivative[3] = Complex(-sine_half, 0);
    } else {
        derivative[0] = Complex(0, FP(0.5)) * std::exp(Complex(0, half_angle));
        derivative[1] = Complex(0, 0);
        derivative[2] = Complex(0, 0);
        derivative[3] = Complex(0, FP(-0.5)) * std::exp(Complex(0, -half_angle));
    }

    const uint32_t qubit = gate.qubits[0];
    const uint64_t low_mask = (1ULL << qubit) - 1;
    const auto* primal_data = primal.data();
    const auto* costate_data = costate.data();
    const auto acc = common::BlockedReduce<2>(
        static_cast<int64_t>(primal.dim() / 2), common::ThreadsForDim(primal.dim()),
        [&](int64_t begin, int64_t end, double* partial) {
            double re = 0.0;
            double im = 0.0;
            for (uint64_t base = static_cast<uint64_t>(begin); base < static_cast<uint64_t>(end); ++base) {
                const uint64_t i0 = (base & low_mask) | ((base & ~low_mask) << 1);
                const uint64_t i1 = i0 | (1ULL << qubit);
                const Complex d0 = derivative[0] * costate_data[i0] + derivative[1] * costate_data[i1];
                const Complex d1 = derivative[2] * costate_data[i0] + derivative[3] * costate_data[i1];
                const std::complex<double> p0(static_cast<double>(primal_data[i0].real()),
                                              static_cast<double>(primal_data[i0].imag()));
                const std::complex<double> p1(static_cast<double>(primal_data[i1].real()),
                                              static_cast<double>(primal_data[i1].imag()));
                const std::complex<double> v0(static_cast<double>(d0.real()), static_cast<double>(d0.imag()));
                const std::complex<double> v1(static_cast<double>(d1.real()), static_cast<double>(d1.imag()));
                const std::complex<double> contribution = std::conj(p0) * v0 + std::conj(p1) * v1;
                re += contribution.real();
                im += contribution.imag();
            }
            partial[0] += re;
            partial[1] += im;
        });
    return {acc[0], acc[1]};
}

std::vector<double> BindAngles(const PreparedData& prepared, const std::vector<double>& parameter_values) {
    std::vector<double> angles(prepared.tape.size(), 0.0);
    for (std::size_t gate_index = 0; gate_index < prepared.tape.size(); ++gate_index) {
        const auto& gate = prepared.tape[gate_index];
        if (!IsRotation(gate.kind)) {
            continue;
        }
        double angle = gate.angle_constant;
        for (const auto& [parameter_index, coefficient] : gate.jacobian) {
            angle += coefficient * parameter_values[parameter_index];
        }
        if (!std::isfinite(angle)) {
            throw std::invalid_argument("AdjointGradientExecutor: bound gate parameter is non-finite");
        }
        angles[gate_index] = angle;
    }
    return angles;
}

template <typename FP>
GradientResult RunPreparedTyped(const PreparedData& prepared,
                                const std::unordered_map<std::string, double>& parameter_values) {
    const auto ordered_values = BindParameterValues(prepared, parameter_values);
    const auto angles = BindAngles(prepared, ordered_values);
    const auto& table = GetDispatch<FP>();

    runtime::StateVector<FP> primal(prepared.num_qubits);
    for (std::size_t gate_index = 0; gate_index < prepared.tape.size(); ++gate_index) {
        ApplyGate(primal, table, prepared.tape[gate_index], angles[gate_index], false);
    }

    std::vector<runtime::StateVector<FP>> costates;
    costates.reserve(prepared.hamiltonians.size());
    GradientResult result;
    result.parameter_names = prepared.parameter_names;
    result.observable_labels = prepared.observable_labels;
    result.expectations.resize(prepared.hamiltonians.size(), 0.0);
    result.gradients.assign(prepared.hamiltonians.size(), std::vector<double>(prepared.parameter_names.size(), 0.0));
    for (std::size_t i = 0; i < prepared.hamiltonians.size(); ++i) {
        costates.push_back(ApplyHamiltonian(prepared.hamiltonians[i], primal));
        result.expectations[i] = InnerProduct(primal, costates.back()).real();
    }

    for (std::size_t reverse_index = prepared.tape.size(); reverse_index-- > 0;) {
        const auto& gate = prepared.tape[reverse_index];
        const double angle = angles[reverse_index];
        ApplyGate(primal, table, gate, angle, true);

        if (IsRotation(gate.kind) && !gate.jacobian.empty()) {
            for (std::size_t hamiltonian_index = 0; hamiltonian_index < costates.size(); ++hamiltonian_index) {
                const double intrinsic_gradient
                    = 2.0 * ExpectRotationDaggerDerivative(primal, costates[hamiltonian_index], gate, angle).real();
                for (const auto& [parameter_index, coefficient] : gate.jacobian) {
                    result.gradients[hamiltonian_index][parameter_index] += coefficient * intrinsic_gradient;
                }
            }
        }

        for (auto& costate : costates) {
            ApplyGate(costate, table, gate, angle, true);
        }
    }
    return result;
}

}  // namespace

struct PreparedAdjointGradient::Impl {
    runtime::QSALProgram program;
    PreparedData prepared;

    explicit Impl(const runtime::QSALProgram& source) : program(source), prepared(PrepareProgram(program)) {
    }
};

PreparedAdjointGradient::PreparedAdjointGradient(const runtime::QSALProgram& program)
    : impl_(std::make_shared<Impl>(program)) {
}

PreparedAdjointGradient::~PreparedAdjointGradient() = default;

GradientResult PreparedAdjointGradient::Run(const std::unordered_map<std::string, double>& parameter_values,
                                            PrecisionKind precision) const {
    switch (precision) {
        case PrecisionKind::kFloat32:
            return RunPreparedTyped<float>(impl_->prepared, parameter_values);
        case PrecisionKind::kFloat64:
            return RunPreparedTyped<double>(impl_->prepared, parameter_values);
    }
    throw std::invalid_argument("AdjointGradientExecutor: unsupported precision");
}

const std::vector<std::string>& PreparedAdjointGradient::parameter_names() const noexcept {
    return impl_->prepared.parameter_names;
}

const std::vector<std::string>& PreparedAdjointGradient::observable_labels() const noexcept {
    return impl_->prepared.observable_labels;
}

GradientResult AdjointGradientExecutor::Run(const runtime::QSALProgram& program,
                                            const std::unordered_map<std::string, double>& parameter_values,
                                            PrecisionKind precision) const {
    return PreparedAdjointGradient(program).Run(parameter_values, precision);
}

}  // namespace quspect::native::executor
