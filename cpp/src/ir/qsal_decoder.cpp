// qsal_decoder.cpp — QSAL JSON → QSALProgram 解码
// Phase 5: 完整解码实现，使用 nlohmann/json

#include "quspect_native/ir/qsal_decoder.hpp"

#include <cmath>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "quspect_native/runtime/program.hpp"

// ordered_json keeps the order of parameter names inside an affine ``terms``
// object.  That lets QSALProgram::parameter_names faithfully reflect the first
// occurrence order in the serialized circuit.
using json = nlohmann::ordered_json;

namespace quspect::native::ir {

namespace {

/// 支持的 QSAL 版本列表
const std::vector<std::string> kSupportedVersions = {"0.1.0"};

/// 字符串转小写
std::string ToLower(const std::string& s) {
    std::string r = s;
    for (auto& c : r)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

/// 是否为参数化门，这类门的 gate_type 可能为 "parameterized"
bool IsParameterizedGate(const std::string& name) {
    std::string lower = ToLower(name);
    return lower == "rx" || lower == "ry" || lower == "rz" || lower == "p" || lower == "u1" || lower == "u2"
           || lower == "u3" || lower == "crx" || lower == "cry" || lower == "crz" || lower == "cp" || lower == "cu3";
}

/// 已知的 named gate 名称集合
bool IsKnownNamedGate(const std::string& name) {
    static const std::vector<std::string> known = {
        "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cnot", "cz", "swap", "id", "ccx", "ccnot",
    };
    std::string lower = ToLower(name);
    return std::find(known.begin(), known.end(), lower) != known.end();
}

/// 创建 Named Gate 操作
runtime::GateOperation MakeNamedGate(const std::string& name, const std::vector<uint32_t>& qubits,
                                     const std::vector<double>& params,
                                     const std::vector<runtime::ParameterExpression>& parameter_expressions = {}) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kNamedGate;
    g.name = ToLower(name);
    g.qubits = qubits;
    g.params_f64 = params;
    g.params_f32.assign(params.begin(), params.end());
    g.parameter_expressions = parameter_expressions;
    return g;
}

/// 无参数、以固定 2x2 矩阵执行的标准门（√X 与其逆）
bool IsFixedDenseGate(const std::string& name) {
    std::string lower = ToLower(name);
    return lower == "sx" || lower == "sxdg";
}

/// 为 sx / sxdg 生成 kDenseGate 操作：SX = ½[[1+i, 1-i], [1-i, 1+i]]，SXdg = SX^†
runtime::GateOperation MakeFixedDenseGate(const std::string& name, const std::vector<uint32_t>& qubits,
                                          std::string& error) {
    std::string lower = ToLower(name);
    if (qubits.size() != 1) {
        error = lower + " requires 1 qubit";
        return {};
    }
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kDenseGate;
    g.name = lower;
    g.qubits = qubits;
    const double sign = lower == "sx" ? 1.0 : -1.0;
    const std::complex<double> diag(0.5, 0.5 * sign);
    const std::complex<double> off(0.5, -0.5 * sign);
    g.dense_matrix_f64 = {diag, off, off, diag};
    g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    return g;
}

/// 是否需要通过完整矩阵计算来处理
bool IsDenseComputeGate(const std::string& name) {
    std::string lower = ToLower(name);
    return lower == "p" || lower == "u1" || lower == "u2" || lower == "u3" || lower == "crx" || lower == "cry"
           || lower == "crz" || lower == "cp" || lower == "cu3";
}

/// 为参数化门计算完整矩阵并生成 kDenseGate 操作
runtime::GateOperation MakeDenseComputeGate(const std::string& name, const std::vector<uint32_t>& qubits,
                                            const std::vector<double>& params,
                                            const std::vector<runtime::ParameterExpression>& parameter_expressions,
                                            bool all_numeric, std::string& error) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kDenseGate;
    g.name = ToLower(name);
    g.qubits = qubits;
    g.params_f64 = params;
    g.params_f32.assign(params.begin(), params.end());
    g.parameter_expressions = parameter_expressions;

    std::string lower = ToLower(name);

    const size_t expected_parameters = (lower == "u3" || lower == "cu3") ? 3 : (lower == "u2" ? 2 : 1);
    if (parameter_expressions.size() != expected_parameters) {
        error = lower + " requires " + std::to_string(expected_parameters) + " parameter"
                + (expected_parameters == 1 ? "" : "s");
        return {};
    }
    const size_t expected_qubits
        = (lower == "crx" || lower == "cry" || lower == "crz" || lower == "cp" || lower == "cu3") ? 2 : 1;
    if (qubits.size() != expected_qubits) {
        error = lower + " requires " + std::to_string(expected_qubits) + " qubit" + (expected_qubits == 1 ? "" : "s");
        return {};
    }

    // Preserve symbolic dense-gate parameters in the IR.  Their matrix cannot
    // be materialized until parameters are bound; the ordinary simulation
    // executor rejects such a program explicitly, while variational executors
    // can inspect the expressions and report gate support independently.
    if (!all_numeric) {
        return g;
    }

    if (lower == "p" || lower == "u1") {
        if (params.empty()) {
            error = lower + " requires 1 parameter";
            return {};
        }
        double theta = params[0];
        g.dense_matrix_f64 = {
            std::complex<double>(1, 0),
            std::complex<double>(0, 0),
            std::complex<double>(0, 0),
            std::complex<double>(std::cos(theta), std::sin(theta)),
        };
        g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    } else if (lower == "u2" || lower == "u3") {
        if (lower == "u2" && params.size() < 2) {
            error = "u2 requires 2 parameters (phi, lambda)";
            return {};
        }
        if (params.size() < 3) {
            if (lower == "u3") {
                error = "u3 requires 3 parameters (theta, phi, lambda)";
                return {};
            }
        }
        double theta = lower == "u2" ? std::acos(-1.0) / 2.0 : params[0];
        double phi = lower == "u2" ? params[0] : params[1];
        double lambda = lower == "u2" ? params[1] : params[2];
        double cos_half = std::cos(theta / 2.0);
        double sin_half = std::sin(theta / 2.0);
        g.dense_matrix_f64 = {
            std::complex<double>(cos_half, 0),
            std::complex<double>(-sin_half * std::cos(lambda), -sin_half * std::sin(lambda)),
            std::complex<double>(sin_half * std::cos(phi), sin_half * std::sin(phi)),
            std::complex<double>(cos_half * std::cos(phi + lambda), cos_half * std::sin(phi + lambda)),
        };
        g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    } else if (lower == "crx" || lower == "cry") {
        if (params.empty()) {
            error = lower + " requires 1 parameter (theta)";
            return {};
        }
        double theta = params[0];
        double cos_half = std::cos(theta / 2.0);
        double sin_half = std::sin(theta / 2.0);
        // targets = [ctrl, tgt]; local basis: bit 0 = ctrl, bit 1 = tgt
        // row-major 4x4: M[row*4+col]
        g.dense_matrix_f64.resize(16, std::complex<double>(0, 0));
        g.dense_matrix_f64[0] = std::complex<double>(1, 0);         // |0,0⟩→|0,0⟩
        g.dense_matrix_f64[5] = std::complex<double>(cos_half, 0);  // |1,0⟩→|1,0⟩
        g.dense_matrix_f64[7] = lower == "crx" ? std::complex<double>(0, -sin_half)
                                               : std::complex<double>(-sin_half, 0);  // |1,1⟩→|1,0⟩
        g.dense_matrix_f64[10] = std::complex<double>(1, 0);                          // |0,1⟩→|0,1⟩
        g.dense_matrix_f64[13] = lower == "crx" ? std::complex<double>(0, -sin_half)
                                                : std::complex<double>(sin_half, 0);  // |1,0⟩→|1,1⟩
        g.dense_matrix_f64[15] = std::complex<double>(cos_half, 0);                   // |1,1⟩→|1,1⟩
        g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    } else if (lower == "crz" || lower == "cp") {
        if (params.empty()) {
            error = lower + " requires 1 parameter (theta)";
            return {};
        }
        double theta = params[0];
        double half = lower == "crz" ? theta / 2.0 : 0.0;
        // targets = [ctrl, tgt]; local basis: bit 0 = ctrl, bit 1 = tgt
        g.dense_matrix_f64.resize(16, std::complex<double>(0, 0));
        g.dense_matrix_f64[0] = std::complex<double>(1, 0);                             // |0,0⟩→|0,0⟩
        g.dense_matrix_f64[5] = std::complex<double>(std::cos(half), -std::sin(half));  // |1,0⟩→|1,0⟩
        g.dense_matrix_f64[10] = std::complex<double>(1, 0);                            // |0,1⟩→|0,1⟩
        g.dense_matrix_f64[15] = lower == "crz"
                                     ? std::complex<double>(std::cos(half), std::sin(half))
                                     : std::complex<double>(std::cos(theta), std::sin(theta));  // |1,1⟩→|1,1⟩
        g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    } else if (lower == "cu3") {
        double theta = params[0], phi = params[1], lambda = params[2];
        double cos_half = std::cos(theta / 2.0);
        double sin_half = std::sin(theta / 2.0);
        g.dense_matrix_f64.resize(16, std::complex<double>(0, 0));
        g.dense_matrix_f64[0] = std::complex<double>(1, 0);
        g.dense_matrix_f64[10] = std::complex<double>(1, 0);
        g.dense_matrix_f64[5] = std::complex<double>(cos_half, 0);
        g.dense_matrix_f64[7] = std::complex<double>(-sin_half * std::cos(lambda), -sin_half * std::sin(lambda));
        g.dense_matrix_f64[13] = std::complex<double>(sin_half * std::cos(phi), sin_half * std::sin(phi));
        g.dense_matrix_f64[15] = std::complex<double>(cos_half * std::cos(phi + lambda),
                                                      cos_half * std::sin(phi + lambda));
        g.dense_matrix_f32.assign(g.dense_matrix_f64.begin(), g.dense_matrix_f64.end());
    } else {
        error = "unknown dense-compute gate: '" + name + "'";
        return {};
    }

    return g;
}

/// 创建 Barrier 操作
runtime::GateOperation MakeBarrier(const std::vector<uint32_t>& qubits) {
    runtime::GateOperation g;
    g.opcode = runtime::OpCode::kBarrier;
    g.name = "barrier";
    g.qubits = qubits;
    return g;
}

struct ParsedParameters {
    std::vector<runtime::ParameterExpression> expressions;
    // This cache is populated only if every expression is a constant.  A
    // symbolic gate must never acquire an implicit zero-valued parameter.
    std::vector<double> numeric_values;
    bool all_numeric = true;
};

bool ParseComplexValue(const json& value, std::complex<double>& output, std::string& error) {
    if (value.is_number()) {
        output = std::complex<double>(value.get<double>(), 0.0);
        if (!std::isfinite(output.real())) {
            error = "Kraus matrix entries must be finite";
            return false;
        }
        return true;
    }
    if (!value.is_array() || value.size() != 2 || !value[0].is_number() || !value[1].is_number()) {
        error = "Kraus matrix entries must be numbers or [real, imag] pairs";
        return false;
    }
    output = std::complex<double>(value[0].get<double>(), value[1].get<double>());
    if (!std::isfinite(output.real()) || !std::isfinite(output.imag())) {
        error = "Kraus matrix entries must be finite";
        return false;
    }
    return true;
}

std::complex<double> PauliElement(uint32_t pauli, uint32_t row, uint32_t col) {
    if (pauli == 0) {
        return (row == col) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
    }
    if (pauli == 1) {
        return (row != col) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
    }
    if (pauli == 2) {
        if (row == 0 && col == 1) {
            return std::complex<double>(0.0, -1.0);
        }
        if (row == 1 && col == 0) {
            return std::complex<double>(0.0, 1.0);
        }
        return std::complex<double>(0.0, 0.0);
    }
    if (row == col) {
        return (row == 0) ? std::complex<double>(1.0, 0.0) : std::complex<double>(-1.0, 0.0);
    }
    return std::complex<double>(0.0, 0.0);
}

/// Verify that a parameterized Pauli fast path represents exactly the same
/// quantum map as the serialized Kraus data.  Raw/hand-written QSAL therefore
/// cannot change semantics by supplying conflicting metadata.
bool MatchesPauliFastPath(const runtime::ChannelOperation& channel) {
    const bool is_depolarizing = channel.channel_kind == "depolarizing";
    const bool is_single_qubit_pauli = channel.channel_kind == "bit_flip" || channel.channel_kind == "phase_flip"
                                       || channel.channel_kind == "bit_phase_flip" || channel.channel_kind == "pauli";
    if ((!is_depolarizing && !is_single_qubit_pauli) || channel.qubits.empty()) {
        return false;
    }
    // Python currently exposes depolarizing Kraus maps for one and two qubits.
    // Keeping this check small also makes the one-time superoperator comparison
    // cheap during decode.
    if ((is_single_qubit_pauli && channel.qubits.size() != 1) || (is_depolarizing && channel.qubits.size() > 2)) {
        return false;
    }

    std::vector<double> pauli_weights(4, 0.0);
    if (!is_depolarizing) {
        if (channel.channel_kind == "pauli") {
            if (channel.params_f64.size() != 3) {
                return false;
            }
            pauli_weights[1] = channel.params_f64[0];
            pauli_weights[2] = channel.params_f64[1];
            pauli_weights[3] = channel.params_f64[2];
        } else {
            if (channel.params_f64.size() != 1) {
                return false;
            }
            const uint32_t pauli = channel.channel_kind == "bit_flip"     ? 1
                                   : channel.channel_kind == "phase_flip" ? 3
                                                                          : 2;
            pauli_weights[pauli] = channel.params_f64[0];
        }
        pauli_weights[0] = 1.0 - pauli_weights[1] - pauli_weights[2] - pauli_weights[3];
    } else if (channel.params_f64.size() != 1) {
        return false;
    }

    const uint64_t dim = uint64_t{1} << channel.qubits.size();
    const uint64_t matrix_size = dim * dim;
    constexpr double tolerance = 1e-9;
    for (uint64_t out_row = 0; out_row < dim; ++out_row) {
        for (uint64_t out_col = 0; out_col < dim; ++out_col) {
            for (uint64_t in_row = 0; in_row < dim; ++in_row) {
                for (uint64_t in_col = 0; in_col < dim; ++in_col) {
                    std::complex<double> actual(0.0, 0.0);
                    for (uint32_t op = 0; op < channel.num_kraus_ops; ++op) {
                        const uint64_t offset = static_cast<uint64_t>(op) * matrix_size;
                        actual += channel.kraus_operators_f64[offset + out_row * dim + in_row]
                                  * std::conj(channel.kraus_operators_f64[offset + out_col * dim + in_col]);
                    }

                    std::complex<double> expected(0.0, 0.0);
                    if (is_depolarizing) {
                        const double p = channel.params_f64[0];
                        if (out_row == in_row && out_col == in_col) {
                            expected += 1.0 - p;
                        }
                        if (in_row == in_col && out_row == out_col) {
                            expected += p / static_cast<double>(dim);
                        }
                    } else {
                        for (uint32_t pauli = 0; pauli < 4; ++pauli) {
                            expected += pauli_weights[pauli]
                                        * PauliElement(pauli, static_cast<uint32_t>(out_row),
                                                       static_cast<uint32_t>(in_row))
                                        * std::conj(PauliElement(pauli, static_cast<uint32_t>(out_col),
                                                                 static_cast<uint32_t>(in_col)));
                        }
                    }
                    if (std::abs(actual - expected) > tolerance) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

runtime::ChannelOperation ParseChannel(const json& channel_json, const std::vector<uint32_t>& qubits,
                                       std::string& error) {
    runtime::ChannelOperation channel;
    channel.qubits = qubits;
    channel.channel_kind = ToLower(
        channel_json.value("channel_kind", channel_json.value("channel_type", std::string("kraus"))));
    if (channel.channel_kind == "gen_amp_damp") {
        channel.channel_kind = "generalized_amplitude_damping";
    }

    const uint32_t declared_qubits = channel_json.value("num_qubits", static_cast<uint32_t>(qubits.size()));
    if (declared_qubits != qubits.size()) {
        error = "channel num_qubits does not match instruction qubits";
        return channel;
    }
    constexpr size_t kMaxSerializedKrausQubits = 10;
    if (qubits.empty() || qubits.size() > kMaxSerializedKrausQubits) {
        error = "serialized Kraus channels must act on between 1 and 10 qubits";
        return channel;
    }

    auto append_param = [&](const char* name) -> bool {
        if (!channel_json.contains(name) || !channel_json.at(name).is_number()) {
            error = std::string("channel '") + channel.channel_kind + "' missing numeric '" + name + "'";
            return false;
        }
        const double value = channel_json.at(name).get<double>();
        if (!std::isfinite(value)) {
            error = std::string("channel parameter '") + name + "' must be finite";
            return false;
        }
        channel.params_f64.push_back(value);
        channel.params_f32.push_back(static_cast<float>(value));
        return true;
    };

    if (channel.channel_kind == "bit_flip" || channel.channel_kind == "phase_flip"
        || channel.channel_kind == "bit_phase_flip" || channel.channel_kind == "depolarizing") {
        if (!append_param("p")) {
            return channel;
        }
        if (channel.params_f64[0] < 0.0 || channel.params_f64[0] > 1.0) {
            error = "channel probability p must be in [0, 1]";
            return channel;
        }
    } else if (channel.channel_kind == "pauli") {
        if (!append_param("px") || !append_param("py") || !append_param("pz")) {
            return channel;
        }
        if (std::any_of(channel.params_f64.begin(), channel.params_f64.end(), [](double p) { return p < 0.0; })
            || channel.params_f64[0] + channel.params_f64[1] + channel.params_f64[2] > 1.0) {
            error = "Pauli channel probabilities must be non-negative and sum to at most 1";
            return channel;
        }
    } else if (channel.channel_kind == "amplitude_damping" || channel.channel_kind == "phase_damping") {
        if (!append_param("gamma")) {
            return channel;
        }
        if (channel.params_f64[0] < 0.0 || channel.params_f64[0] > 1.0) {
            error = "channel gamma must be in [0, 1]";
            return channel;
        }
    } else if (channel.channel_kind == "generalized_amplitude_damping") {
        if (!append_param("gamma") || !append_param("p_thermal")) {
            return channel;
        }
        if (channel.params_f64[0] < 0.0 || channel.params_f64[0] > 1.0 || channel.params_f64[1] < 0.0
            || channel.params_f64[1] > 0.5) {
            error = "generalized amplitude damping requires gamma in [0, 1] and p_thermal in [0, 0.5]";
            return channel;
        }
    }

    if (!channel_json.contains("kraus_ops") || !channel_json.at("kraus_ops").is_array()
        || channel_json.at("kraus_ops").empty()) {
        error = "channel '" + channel.channel_kind + "' is missing a non-empty kraus_ops array";
        return channel;
    }

    const uint64_t matrix_dim = uint64_t{1} << qubits.size();
    const auto& kraus_ops = channel_json.at("kraus_ops");
    if (kraus_ops.size() > std::numeric_limits<uint32_t>::max()) {
        error = "too many Kraus operators";
        return channel;
    }
    channel.num_kraus_ops = static_cast<uint32_t>(kraus_ops.size());

    // Validate the full nested shape before allocating the flattened buffer.
    // This prevents a small malformed document from triggering a huge reserve
    // based only on a declared target width.
    for (const auto& matrix : kraus_ops) {
        if (!matrix.is_array() || matrix.size() != matrix_dim) {
            error = "Kraus operator has the wrong row count";
            return channel;
        }
        for (const auto& row : matrix) {
            if (!row.is_array() || row.size() != matrix_dim) {
                error = "Kraus operator has the wrong column count";
                return channel;
            }
        }
    }
    const uint64_t entries_per_matrix = matrix_dim * matrix_dim;
    if (channel.num_kraus_ops > std::numeric_limits<size_t>::max() / entries_per_matrix) {
        error = "Kraus operator data is too large";
        return channel;
    }
    channel.kraus_operators_f64.reserve(static_cast<size_t>(channel.num_kraus_ops * entries_per_matrix));

    for (const auto& matrix : kraus_ops) {
        for (const auto& row : matrix) {
            for (const auto& value : row) {
                std::complex<double> parsed;
                if (!ParseComplexValue(value, parsed, error)) {
                    return channel;
                }
                channel.kraus_operators_f64.push_back(parsed);
            }
        }
    }
    channel.kraus_operators_f32.assign(channel.kraus_operators_f64.begin(), channel.kraus_operators_f64.end());
    channel.use_pauli_fast_path = MatchesPauliFastPath(channel);
    return channel;
}

void RegisterParameterName(const std::string& name, std::vector<std::string>& parameter_names) {
    if (std::find(parameter_names.begin(), parameter_names.end(), name) == parameter_names.end()) {
        parameter_names.push_back(name);
    }
}

bool ParseParameterExpression(const json& value, runtime::ParameterExpression& expression,
                              std::vector<std::string>& parameter_names, std::string& error) {
    if (value.is_number()) {
        expression.constant = value.get<double>();
        return true;
    }

    if (!value.is_object()) {
        error = "unsupported parameter format: expected a number, _param, or _expr object";
        return false;
    }

    if (value.contains("_param")) {
        if (!value.at("_param").is_string() || value.at("_param").get<std::string>().empty()) {
            error = "parameter _param must be a non-empty string";
            return false;
        }
        const auto name = value.at("_param").get<std::string>();
        expression.terms.emplace_back(name, 1.0);
        RegisterParameterName(name, parameter_names);
        return true;
    }

    if (!value.contains("_expr") || !value.at("_expr").is_object()) {
        error = "unsupported parameter object: expected _param or _expr";
        return false;
    }

    const auto& affine = value.at("_expr");
    if (affine.contains("constant")) {
        if (!affine.at("constant").is_number()) {
            error = "parameter _expr.constant must be numeric";
            return false;
        }
        expression.constant = affine.at("constant").get<double>();
    }

    if (!affine.contains("terms")) {
        return true;
    }
    if (!affine.at("terms").is_object()) {
        error = "parameter _expr.terms must be an object mapping names to coefficients";
        return false;
    }

    for (const auto& [name, coefficient_json] : affine.at("terms").items()) {
        if (name.empty() || !coefficient_json.is_number()) {
            error = "parameter _expr.terms must contain non-empty names and numeric coefficients";
            return false;
        }
        const double coefficient = coefficient_json.get<double>();
        // A zero coefficient is not an actual dependency and should neither
        // make the expression symbolic nor enter the global parameter list.
        if (coefficient == 0.0) {
            continue;
        }
        expression.terms.emplace_back(name, coefficient);
        RegisterParameterName(name, parameter_names);
    }
    return true;
}

ParsedParameters ExtractParameters(const json& params_array, std::vector<std::string>& parameter_names,
                                   std::string& error) {
    ParsedParameters result;
    if (!params_array.is_array()) {
        error = "gate params must be an array";
        return result;
    }

    for (const auto& value : params_array) {
        runtime::ParameterExpression expression;
        if (!ParseParameterExpression(value, expression, parameter_names, error)) {
            return {};
        }
        result.all_numeric = result.all_numeric && !expression.IsSymbolic();
        result.expressions.push_back(std::move(expression));
    }

    if (result.all_numeric) {
        result.numeric_values.reserve(result.expressions.size());
        for (const auto& expression : result.expressions) {
            result.numeric_values.push_back(expression.constant);
        }
    }
    return result;
}

/// 解析单条 QSAL 指令，返回 ProgramOperation
runtime::ProgramOperation ParseInstruction(const json& instr, std::vector<std::string>& parameter_names,
                                           std::string& error) {
    std::string itype = instr.value("type", "");

    std::vector<uint32_t> qubits;
    for (const auto& q : instr.at("qubits")) {
        qubits.push_back(q.get<uint32_t>());
    }

    std::vector<uint32_t> clbits;
    if (instr.contains("clbits") && !instr.at("clbits").is_null()) {
        for (const auto& c : instr.at("clbits")) {
            clbits.push_back(c.get<uint32_t>());
        }
    }

    if (itype == "MEASURE" || itype == "measure") {
        runtime::MeasureOperation m;
        m.qubits = qubits;
        m.clbits = clbits;
        return m;
    }

    if (itype == "BARRIER" || itype == "barrier") {
        return MakeBarrier(qubits);
    }

    if (itype == "CHANNEL" || itype == "channel") {
        if (!instr.contains("channel") || !instr.at("channel").is_object()) {
            error = "CHANNEL instruction missing 'channel' field";
            return {};
        }
        return ParseChannel(instr.at("channel"), qubits, error);
    }

    if (itype == "GATE" || itype == "gate") {
        if (!instr.contains("gate") || instr.at("gate").is_null()) {
            error = "GATE instruction missing 'gate' field";
            return {};
        }

        const auto& gd = instr.at("gate");
        std::string gate_name = gd.value("name", "");
        std::string gate_type = gd.value("gate_type", "standard");

        if (gate_name.empty()) {
            error = "GATE instruction has empty gate name";
            return {};
        }

        if (gate_type == "parameterized" || IsParameterizedGate(gate_name)) {
            auto params = ExtractParameters(gd.value("params", json::array()), parameter_names, error);
            if (!error.empty())
                return {};

            // 对非原生内核门使用完整矩阵计算（kDenseGate）
            if (IsDenseComputeGate(gate_name)) {
                return MakeDenseComputeGate(gate_name, qubits, params.numeric_values, params.expressions,
                                            params.all_numeric, error);
            }

            if (!IsKnownNamedGate(gate_name)) {
                error = "unknown parameterized gate: '" + gate_name + "'";
                return {};
            }

            return MakeNamedGate(gate_name, qubits, params.numeric_values, params.expressions);
        }

        // 标准门
        if (IsFixedDenseGate(gate_name)) {
            return MakeFixedDenseGate(gate_name, qubits, error);
        }
        if (!IsKnownNamedGate(gate_name)) {
            error = "unknown standard gate: '" + gate_name + "'";
            return {};
        }

        return MakeNamedGate(gate_name, qubits, {});
    }

    error = "unknown instruction type: '" + itype + "'";
    return runtime::GateOperation{};  // 占位返回值
}

runtime::ObservableTerm ParseObservableTerm(const std::string& label, const json& term_json, std::string& error) {
    runtime::ObservableTerm term;
    term.label = label;

    // coefficient 可以是实数，或 [real, imag] 形式的复数对（QSAL 对非实系数的编码）。
    term.coefficient = std::complex<double>(1.0, 0.0);
    if (term_json.contains("coefficient")) {
        const auto& coefficient = term_json.at("coefficient");
        if (coefficient.is_number()) {
            term.coefficient = std::complex<double>(coefficient.get<double>(), 0.0);
        } else if (coefficient.is_array() && coefficient.size() == 2 && coefficient[0].is_number()
                   && coefficient[1].is_number()) {
            term.coefficient = std::complex<double>(coefficient[0].get<double>(), coefficient[1].get<double>());
        } else {
            error = "observable '" + label + "' has a non-numeric coefficient";
            return term;
        }
    }

    if (term_json.contains("paulis")) {
        if (!term_json.at("paulis").is_array()) {
            error = "observable '" + label + "' has a non-array paulis field";
            return term;
        }
        for (const auto& p : term_json.at("paulis")) {
            if (!p.is_array() || p.size() != 2 || !p[0].is_number_unsigned() || !p[1].is_string()
                || p[1].get<std::string>().size() != 1) {
                error = "observable '" + label + "' has an invalid Pauli entry";
                return term;
            }
            term.paulis.emplace_back(p[0].get<uint32_t>(), static_cast<char>(p[1].get<std::string>()[0]));
        }
    }
    return term;
}

/// Expand a serialized operator into one ObservableTerm per Pauli term.  Terms
/// from the same operator deliberately share a label so executors can aggregate
/// them into a single Hamiltonian expectation value.
std::vector<runtime::ObservableTerm> ParseObservable(const std::string& label, const json& op_json,
                                                     std::string& error) {
    std::vector<runtime::ObservableTerm> result;

    // 支持两种格式：
    //   1. 单个 term 直接：{"paulis": [[0, "Z"], [1, "Z"]], "coefficient": 1.0}
    //   2. multi-term operator：{"terms": [{...}, ...]}
    if (op_json.contains("terms")) {
        const auto& terms = op_json.at("terms");
        if (!terms.is_array() || terms.empty()) {
            error = "observable '" + label + "' has empty or invalid terms list";
            return result;
        }
        result.reserve(terms.size());
        for (const auto& term_json : terms) {
            auto term = ParseObservableTerm(label, term_json, error);
            if (!error.empty()) {
                return {};
            }
            result.push_back(std::move(term));
        }
    } else if (op_json.contains("paulis")) {
        result.push_back(ParseObservableTerm(label, op_json, error));
    } else {
        error = "observable '" + label + "' missing 'terms' or 'paulis' field";
        return result;
    }
    return result;
}

}  // namespace

// ============================================================
// QSALDecoder::Impl
// ============================================================

struct QSALDecoder::Impl {
    std::string last_error;
};

// ============================================================
// 构造与析构
// ============================================================

QSALDecoder::QSALDecoder() : impl_(std::make_unique<Impl>()) {
}

QSALDecoder::~QSALDecoder() = default;

// ============================================================
// Decode
// ============================================================

std::unique_ptr<runtime::QSALProgram> QSALDecoder::Decode(const std::string& qsal_json) {
    impl_->last_error.clear();

    // Step 1: 解析 JSON
    json doc;
    try {
        doc = json::parse(qsal_json);
    } catch (const json::parse_error& e) {
        impl_->last_error = std::string("JSON parse error: ") + e.what();
        return nullptr;
    }

    // Step 2: 版本检查
    std::string version = doc.value("qsal_version", "unknown");
    if (std::find(kSupportedVersions.begin(), kSupportedVersions.end(), version) == kSupportedVersions.end()) {
        impl_->last_error = "unsupported QSAL version: '" + version + "'";
        return nullptr;
    }

    // Step 3: 构建程序对象
    auto prog = std::make_unique<runtime::QSALProgram>();

    // 解析 circuit
    if (!doc.contains("circuit") || doc.at("circuit").is_null()) {
        impl_->last_error = "QSAL document missing 'circuit' field";
        return nullptr;
    }

    const auto& circuit = doc.at("circuit");
    prog->num_qubits = circuit.value("num_qubits", 0);
    prog->num_clbits = circuit.value("num_clbits", 0);

    // 解析 instructions
    if (circuit.contains("instructions")) {
        for (const auto& instr : circuit.at("instructions")) {
            std::string err;
            auto op = ParseInstruction(instr, prog->parameter_names, err);
            if (!err.empty()) {
                impl_->last_error = "instruction parse error: " + err;
                return nullptr;
            }

            if (std::holds_alternative<runtime::GateOperation>(op)
                || std::holds_alternative<runtime::MeasureOperation>(op)
                || std::holds_alternative<runtime::ChannelOperation>(op)) {
                prog->ops.push_back(std::move(op));
            }
        }
    }

    // 解析 observables
    if (doc.contains("observables") && !doc.at("observables").is_null()) {
        for (const auto& obs_data : doc.at("observables")) {
            std::string label = obs_data.value("name", "");
            std::string err;
            auto terms = ParseObservable(label, obs_data.at("operator"), err);
            if (!err.empty()) {
                impl_->last_error = "observable parse error: " + err;
                return nullptr;
            }
            prog->observables.insert(prog->observables.end(), std::make_move_iterator(terms.begin()),
                                     std::make_move_iterator(terms.end()));
        }
    }

    return prog;
}

const std::string& QSALDecoder::LastError() const {
    return impl_->last_error;
}

}  // namespace quspect::native::ir
