#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace quspect::native::runtime {

// ============================================================
// 操作码
// ============================================================

enum class OpCode {
    kNamedGate,
    kDenseGate,
    kDiagonalGate,
    kMeasure,
    kBarrier,
};

// ============================================================
// 门操作
// ============================================================

/// A gate's intrinsic parameter expressed as an affine function of global
/// circuit parameters: constant + sum(coefficient * parameter).
///
/// Keeping this representation in the compiled program allows variational
/// algorithms to bind new values without serializing and decoding the circuit
/// for every optimizer step.  The adjoint-gradient engine also uses ``terms``
/// directly as the local-to-global Jacobian.
struct ParameterExpression {
    double constant = 0.0;
    std::vector<std::pair<std::string, double>> terms;

    bool IsSymbolic() const noexcept {
        return !terms.empty();
    }
};

struct GateOperation {
    OpCode opcode;
    std::string name;                                    // 门名称，如 "h", "cx", "rx"
    std::vector<uint32_t> qubits;                        // 目标 qubit 列表
    std::vector<double> params_f64;                      // 数值化参数 (double)
    std::vector<float> params_f32;                       // 数值化参数 (float)
    std::vector<ParameterExpression> parameter_expressions;  // 编译后的仿射参数表达式
    std::vector<std::complex<double>> dense_matrix_f64;  // 通用密集矩阵 (row-major)
    std::vector<std::complex<float>> dense_matrix_f32;
    uint64_t control_mask = 0;   // 受控门的控制位 mask
    uint64_t control_value = 0;  // 受控门的控制位期望值
};

// ============================================================
// 测量操作
// ============================================================

struct MeasureOperation {
    std::vector<uint32_t> qubits;  // 被测量的 qubit
    std::vector<uint32_t> clbits;  // 结果写入的 classical register 位
};

// ============================================================
// 噪声信道操作
// ============================================================

/// A CPTP channel compiled to a Kraus representation.
///
/// ``channel_kind`` keeps the canonical channel type so the runtime can use
/// fast paths for Pauli mixtures.  The flattened Kraus matrices remain the
/// source of truth and provide a generic fallback for damping and user-defined
/// channels.  Matrices are stored consecutively in row-major order.
struct ChannelOperation {
    std::string channel_kind;
    std::vector<uint32_t> qubits;
    std::vector<double> params_f64;
    std::vector<float> params_f32;
    // Set by the decoder only after the parameterized Pauli channel is
    // verified to represent the same map as the serialized Kraus matrices.
    bool use_pauli_fast_path = false;
    uint32_t num_kraus_ops = 0;
    std::vector<std::complex<double>> kraus_operators_f64;
    std::vector<std::complex<float>> kraus_operators_f32;
};

// ============================================================
// 可观测量项
// ============================================================

struct ObservableTerm {
    std::string label;
    std::complex<double> coefficient;
    // (qubit_index, pauli_char) — e.g. (0, 'X'), (2, 'Z')
    std::vector<std::pair<uint32_t, char>> paulis;
};

// ============================================================
// 线路程序
// ============================================================

using ProgramOperation = std::variant<GateOperation, MeasureOperation, ChannelOperation>;

struct QSALProgram {
    uint32_t num_qubits = 0;
    uint32_t num_clbits = 0;
    std::vector<ProgramOperation> ops;
    std::vector<ObservableTerm> observables;
    std::vector<std::string> parameter_names;  // 按在线路中首次出现的顺序排列
};

// ============================================================
// Pauli 字符串 mask 分解
// ============================================================

/// Pauli 字符串的 mask 分解结果，作为 pauli_expect kernel 的输入参数
struct PauliMaskInfo {
    uint64_t bit_flip_mask = 0;    ///< X/Y 位置置 1
    uint64_t phase_flip_mask = 0;  ///< Z/Y 位置置 1
    uint32_t pivot_qubit = 0;      ///< bit_flip_mask 的最低置位，用于防止双计数
    int y_count = 0;               ///< Y 的数量，用于计算 i^{y_count} 相位因子
};

/// 将 pauli 列表转换为 mask 参数
///
/// 遍历 (qubit, pauli_char) 列表:
///   - ``'X'`` → 设置 bit_flip_mask
///   - ``'Z'`` → 设置 phase_flip_mask
///   - ``'Y'`` → 设置 bit_flip_mask 和 phase_flip_mask，y_count++
///   - ``'I'`` → 无操作
///
/// pivot_qubit 选为 bit_flip_mask 的最低置位（若 bit_flip_mask == 0 则设为 0）。
///
/// Args:
///     paulis: (qubit_index, pauli_char) 列表，每个 pauli_char 为 'I'/'X'/'Y'/'Z'
///
/// Returns:
///     包含 bit_flip_mask, phase_flip_mask, pivot_qubit, y_count 的 PauliMaskInfo
///
/// Raises:
///     std::invalid_argument: 若 qubit 索引 >= 64 或 pauli_char 无效
PauliMaskInfo ComputePauliMasks(const std::vector<std::pair<uint32_t, char>>& paulis);

}  // namespace quspect::native::runtime
