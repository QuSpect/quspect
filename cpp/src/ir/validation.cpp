// validation.cpp — QSALProgram 输入校验
// Phase 5: 完整校验实现

#include "quspect_native/ir/validation.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "quspect_native/runtime/program.hpp"

namespace quspect::native::ir {

namespace {

/// 已知的 named gate 名称（与 decoder 保持一致）
bool IsKnownGate(const std::string& name) {
    static const std::vector<std::string> known = {
        "x",  "y",  "z",    "h",  "s",    "sdg", "t",   "tdg",   "rx",      "ry",
        "rz", "cx", "cnot", "cz", "swap", "id",  "ccx", "ccnot", "barrier",
    };
    return std::find(known.begin(), known.end(), name) != known.end();
}

/// 参数化门所需的参数个数
int ExpectedParamCount(const std::string& name) {
    if (name == "rx" || name == "ry" || name == "rz")
        return 1;
    return 0;
}

/// 门所需的 qubit 数
int ExpectedQubitCount(const std::string& name) {
    if (name == "ccx" || name == "ccnot")
        return 3;
    if (name == "cx" || name == "cnot" || name == "cz" || name == "swap")
        return 2;
    return 1;
}

}  // namespace

std::vector<std::string> Validator::Validate(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;

    auto bounds = CheckIndexBounds(prog);
    errors.insert(errors.end(), bounds.begin(), bounds.end());

    auto names = CheckGateNames(prog);
    errors.insert(errors.end(), names.begin(), names.end());

    auto params = CheckParameterCount(prog);
    errors.insert(errors.end(), params.begin(), params.end());

    auto meas = CheckMeasureConsistency(prog);
    errors.insert(errors.end(), meas.begin(), meas.end());

    auto channels = CheckChannels(prog);
    errors.insert(errors.end(), channels.begin(), channels.end());

    return errors;
}

std::vector<std::string> Validator::CheckIndexBounds(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;

    for (size_t i = 0; i < prog.ops.size(); ++i) {
        const auto& op_var = prog.ops[i];
        std::string ctx = "op[" + std::to_string(i) + "]";

        if (std::holds_alternative<runtime::GateOperation>(op_var)) {
            const auto& gate = std::get<runtime::GateOperation>(op_var);
            for (auto q : gate.qubits) {
                if (q >= prog.num_qubits) {
                    std::ostringstream oss;
                    oss << ctx << " gate '" << gate.name << "': qubit " << q << " >= num_qubits " << prog.num_qubits;
                    errors.push_back(oss.str());
                }
            }
        } else if (std::holds_alternative<runtime::MeasureOperation>(op_var)) {
            const auto& meas = std::get<runtime::MeasureOperation>(op_var);
            for (auto q : meas.qubits) {
                if (q >= prog.num_qubits) {
                    std::ostringstream oss;
                    oss << ctx << " measure: qubit " << q << " >= num_qubits " << prog.num_qubits;
                    errors.push_back(oss.str());
                }
            }
            for (auto c : meas.clbits) {
                if (c >= prog.num_clbits) {
                    std::ostringstream oss;
                    oss << ctx << " measure: clbit " << c << " >= num_clbits " << prog.num_clbits;
                    errors.push_back(oss.str());
                }
            }
        } else if (std::holds_alternative<runtime::ChannelOperation>(op_var)) {
            const auto& channel = std::get<runtime::ChannelOperation>(op_var);
            for (auto q : channel.qubits) {
                if (q >= prog.num_qubits) {
                    std::ostringstream oss;
                    oss << ctx << " channel '" << channel.channel_kind << "': qubit " << q
                        << " >= num_qubits " << prog.num_qubits;
                    errors.push_back(oss.str());
                }
            }
        }
    }

    return errors;
}

std::vector<std::string> Validator::CheckGateNames(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;

    for (size_t i = 0; i < prog.ops.size(); ++i) {
        const auto& op_var = prog.ops[i];
        if (std::holds_alternative<runtime::GateOperation>(op_var)) {
            const auto& gate = std::get<runtime::GateOperation>(op_var);
            if (gate.opcode == runtime::OpCode::kBarrier)
                continue;
            if (gate.opcode == runtime::OpCode::kDenseGate)
                continue;
            if (gate.opcode == runtime::OpCode::kDiagonalGate)
                continue;

            if (!IsKnownGate(gate.name)) {
                errors.push_back("op[" + std::to_string(i) + "]: unknown gate '" + gate.name + "'");
            }
        }
    }

    return errors;
}

std::vector<std::string> Validator::CheckParameterCount(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;

    for (size_t i = 0; i < prog.ops.size(); ++i) {
        const auto& op_var = prog.ops[i];
        if (std::holds_alternative<runtime::GateOperation>(op_var)) {
            const auto& gate = std::get<runtime::GateOperation>(op_var);
            if (gate.opcode != runtime::OpCode::kNamedGate)
                continue;

            int expected = ExpectedParamCount(gate.name);
            // Symbolic programs intentionally have no numeric parameter cache.
            // The affine expression vector is the canonical count whenever it
            // is present; legacy/numeric-only programs still use params_f64.
            int actual = static_cast<int>(gate.parameter_expressions.empty() ? gate.params_f64.size()
                                                                             : gate.parameter_expressions.size());
            if (actual != expected) {
                std::ostringstream oss;
                oss << "op[" << i << "] gate '" << gate.name << "': expected " << expected << " params, got " << actual;
                errors.push_back(oss.str());
            }

            int expected_q = ExpectedQubitCount(gate.name);
            int actual_q = static_cast<int>(gate.qubits.size());
            if (actual_q != expected_q) {
                std::ostringstream oss;
                oss << "op[" << i << "] gate '" << gate.name << "': expected " << expected_q << " qubits, got "
                    << actual_q;
                errors.push_back(oss.str());
            }
        }
    }

    return errors;
}

std::vector<std::string> Validator::CheckMeasureConsistency(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;

    for (size_t i = 0; i < prog.ops.size(); ++i) {
        const auto& op_var = prog.ops[i];
        if (std::holds_alternative<runtime::MeasureOperation>(op_var)) {
            const auto& meas = std::get<runtime::MeasureOperation>(op_var);
            if (meas.qubits.size() != meas.clbits.size() && !meas.clbits.empty()) {
                std::ostringstream oss;
                oss << "op[" << i << "] measure: qubits count " << meas.qubits.size() << " != clbits count "
                    << meas.clbits.size();
                errors.push_back(oss.str());
            }
        }
    }

    return errors;
}

std::vector<std::string> Validator::CheckChannels(const runtime::QSALProgram& prog) {
    std::vector<std::string> errors;
    constexpr double tolerance = 1e-8;

    for (size_t i = 0; i < prog.ops.size(); ++i) {
        const auto& op_var = prog.ops[i];
        if (!std::holds_alternative<runtime::ChannelOperation>(op_var)) {
            continue;
        }
        const auto& channel = std::get<runtime::ChannelOperation>(op_var);
        const std::string ctx = "op[" + std::to_string(i) + "] channel '" + channel.channel_kind + "'";

        if (channel.qubits.empty() || channel.qubits.size() > 10) {
            errors.push_back(ctx + ": serialized Kraus target count must be between 1 and 10");
            continue;
        }
        std::set<uint32_t> unique_qubits(channel.qubits.begin(), channel.qubits.end());
        if (unique_qubits.size() != channel.qubits.size()) {
            errors.push_back(ctx + ": target qubits must be unique");
        }
        if (channel.num_kraus_ops == 0) {
            errors.push_back(ctx + ": must contain at least one Kraus operator");
            continue;
        }

        const uint64_t matrix_dim = uint64_t{1} << channel.qubits.size();
        const uint64_t expected_size = static_cast<uint64_t>(channel.num_kraus_ops) * matrix_dim * matrix_dim;
        if (channel.kraus_operators_f64.size() != expected_size
            || channel.kraus_operators_f32.size() != expected_size) {
            errors.push_back(ctx + ": flattened Kraus data has the wrong size");
            continue;
        }

        // Verify sum_i K_i^dagger K_i = I.  Without this check the trajectory
        // kernel would normalize invalid maps and silently make them nonlinear.
        bool trace_preserving = true;
        for (uint64_t col_a = 0; col_a < matrix_dim && trace_preserving; ++col_a) {
            for (uint64_t col_b = 0; col_b < matrix_dim; ++col_b) {
                std::complex<double> value(0.0, 0.0);
                for (uint32_t op = 0; op < channel.num_kraus_ops; ++op) {
                    const uint64_t offset = static_cast<uint64_t>(op) * matrix_dim * matrix_dim;
                    for (uint64_t row = 0; row < matrix_dim; ++row) {
                        value += std::conj(channel.kraus_operators_f64[offset + row * matrix_dim + col_a])
                                 * channel.kraus_operators_f64[offset + row * matrix_dim + col_b];
                    }
                }
                const std::complex<double> expected =
                    (col_a == col_b) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
                if (std::abs(value - expected) > tolerance) {
                    trace_preserving = false;
                    break;
                }
            }
        }
        if (!trace_preserving) {
            errors.push_back(ctx + ": Kraus operators are not trace-preserving");
        }
    }
    return errors;
}

}  // namespace quspect::native::ir
