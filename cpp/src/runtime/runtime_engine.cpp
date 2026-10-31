#include "quspect_native/runtime/runtime_engine.hpp"

#include <cmath>

#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/kernels/noise_kernels.hpp"
#include "quspect_native/runtime/classical_register.hpp"
#include "quspect_native/runtime/program.hpp"
#include "quspect_native/runtime/state_vector.hpp"

namespace quspect::native::runtime {

// 内核命名空间别名
namespace nkernels = ::quspect::native::kernels;

// ============================================================
// RuntimeEngine::Impl
// ============================================================

template <typename FP>
struct RuntimeEngine<FP>::Impl {
    const QSALProgram& program;
    const nkernels::KernelDispatchTable<FP>& table;
    StateVector<FP> state;
    ClassicalRegisterFile creg;
    std::mt19937_64 rng;

    Impl(const QSALProgram& prog, const nkernels::KernelDispatchTable<FP>& tbl, uint64_t seed)
        : program(prog), table(tbl), state(prog.num_qubits), creg(prog.num_clbits), rng(seed) {
    }
};

// ============================================================
// 构造与析构
// ============================================================

template <typename FP>
RuntimeEngine<FP>::RuntimeEngine(const QSALProgram& program, const nkernels::KernelDispatchTable<FP>& table,
                                 uint64_t seed)
    : impl_(std::make_unique<Impl>(program, table, seed)) {
}

template <typename FP>
RuntimeEngine<FP>::~RuntimeEngine() = default;

// ============================================================
// Reset
// ============================================================

template <typename FP>
void RuntimeEngine<FP>::Reset() {
    impl_->state.SetZeroState();
    impl_->creg = ClassicalRegisterFile(impl_->program.num_clbits);
}

// ============================================================
// Run — 主执行循环
// ============================================================

template <typename FP>
void RuntimeEngine<FP>::Run(bool execute_measurements) {
    auto& state = impl_->state;
    auto& creg = impl_->creg;
    const auto& table = impl_->table;
    const auto& program = impl_->program;
    auto& rng = impl_->rng;

    const uint64_t dim = state.dim();

    // 用于测量的随机数分布
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    for (const auto& op_var : program.ops) {
        if (std::holds_alternative<GateOperation>(op_var)) {
            const auto& gate = std::get<GateOperation>(op_var);

            switch (gate.opcode) {
                case OpCode::kNamedGate: {
                    // ---- 单比特门 ----
                    if (gate.name == "x") {
                        table.x(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "y") {
                        table.y(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "z") {
                        table.z(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "h") {
                        table.h(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "s") {
                        table.s(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "sdg") {
                        table.sdg(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "t") {
                        table.t(state.data(), dim, gate.qubits[0], nullptr);
                    } else if (gate.name == "tdg") {
                        table.tdg(state.data(), dim, gate.qubits[0], nullptr);
                    }
                    // ---- 参数化单比特门 ----
                    else if (gate.name == "rx") {
                        // RX(theta) = cos(theta/2) I - i sin(theta/2) X
                        FP theta = static_cast<FP>(gate.params_f64[0]);
                        Complex half(static_cast<FP>(0.5), static_cast<FP>(0));
                        Complex u00 = std::cos(half * Complex(theta, 0));
                        Complex u01 = Complex(0, -1) * std::sin(half * Complex(theta, 0));
                        Complex u10 = u01;
                        Complex u11 = u00;
                        Complex m[4] = {u00, u01, u10, u11};
                        table.rx(state.data(), dim, gate.qubits[0], m);
                    } else if (gate.name == "ry") {
                        // RY(theta) = cos(theta/2) I - i sin(theta/2) Y
                        FP theta = static_cast<FP>(gate.params_f64[0]);
                        Complex half(static_cast<FP>(0.5), static_cast<FP>(0));
                        Complex u00 = std::cos(half * Complex(theta, 0));
                        Complex u01 = -std::sin(half * Complex(theta, 0));
                        Complex u10 = std::sin(half * Complex(theta, 0));
                        Complex u11 = u00;
                        Complex m[4] = {u00, u01, u10, u11};
                        table.ry(state.data(), dim, gate.qubits[0], m);
                    } else if (gate.name == "rz") {
                        // RZ(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
                        FP theta = static_cast<FP>(gate.params_f64[0]);
                        Complex half(static_cast<FP>(0.5), static_cast<FP>(0));
                        Complex u00 = std::exp(Complex(0, -1) * half * Complex(theta, 0));
                        Complex u11 = std::exp(Complex(0, 1) * half * Complex(theta, 0));
                        Complex m[4] = {u00, Complex(0, 0), Complex(0, 0), u11};
                        table.rz(state.data(), dim, gate.qubits[0], m);
                    }
                    // ---- 双比特门 ----
                    else if (gate.name == "cx" || gate.name == "cnot") {
                        // control = qubits[0], target = qubits[1]
                        uint64_t cmask = 1ULL << gate.qubits[0];
                        table.cnot(state.data(), dim, gate.qubits[1], cmask, cmask, nullptr);
                    } else if (gate.name == "cz") {
                        uint64_t cmask = 1ULL << gate.qubits[0];
                        table.cz(state.data(), dim, gate.qubits[1], cmask, cmask, nullptr);
                    } else if (gate.name == "swap") {
                        table.swap(state.data(), dim, gate.qubits[0], gate.qubits[1]);
                    } else if (gate.name == "id") {
                        // id gate: no-op — does not affect state
                    }
                    // ---- 三比特门 ----
                    else if (gate.name == "ccx" || gate.name == "ccnot") {
                        // Toffoli:  controlled-controlled-X
                        // qubits[0] = control 0, qubits[1] = control 1, qubits[2] = target
                        uint64_t cmask = (1ULL << gate.qubits[0]) | (1ULL << gate.qubits[1]);
                        table.cnot(state.data(), dim, gate.qubits[2], cmask, cmask, nullptr);
                    } else {
                        throw std::runtime_error("Unknown named gate: " + gate.name);
                    }
                    break;
                }

                case OpCode::kDenseGate: {
                    if constexpr (sizeof(FP) == sizeof(double)) {
                        table.dense_k(state.data(), dim, gate.qubits.data(), static_cast<uint32_t>(gate.qubits.size()),
                                      gate.dense_matrix_f64.data());
                    } else {
                        table.dense_k(state.data(), dim, gate.qubits.data(), static_cast<uint32_t>(gate.qubits.size()),
                                      gate.dense_matrix_f32.data());
                    }
                    break;
                }

                case OpCode::kDiagonalGate: {
                    if constexpr (sizeof(FP) == sizeof(double)) {
                        table.diagonal_k(state.data(), dim, gate.qubits.data(),
                                         static_cast<uint32_t>(gate.qubits.size()), gate.dense_matrix_f64.data());
                    } else {
                        table.diagonal_k(state.data(), dim, gate.qubits.data(),
                                         static_cast<uint32_t>(gate.qubits.size()), gate.dense_matrix_f32.data());
                    }
                    break;
                }

                case OpCode::kBarrier:
                    // Barrier 不作用到状态向量
                    break;

                case OpCode::kMeasure:
                    // 测量门（GateOperation 形式）— Phase 4
                    if (execute_measurements) {
                        // 对每个被测量的 qubit 逐个处理
                        for (size_t idx = 0; idx < gate.qubits.size(); ++idx) {
                            uint32_t qubit = gate.qubits[idx];

                            // 计算 P(qubit=0)
                            FP p0 = table.prob0(state.data(), dim, qubit);

                            // 采样测量结果
                            double r = dist01(rng);
                            int outcome = (static_cast<double>(p0) > r) ? 0 : 1;

                            // 坍缩状态向量
                            table.collapse(state.data(), dim, qubit, outcome);

                            // 写入经典寄存器（如果指定了 clbits）
                            if (idx < gate.qubits.size()) {
                                // kMeasure 形式的 GateOperation 不直接带 clbits
                                // 见 MeasureOperation 分支
                            }
                        }
                    }
                    break;
            }
        } else if (std::holds_alternative<ChannelOperation>(op_var)) {
            const auto& channel = std::get<ChannelOperation>(op_var);

            // Pauli mixtures do not need the generic Kraus kernel.  Sampling
            // the Pauli label directly avoids allocating one full transformed
            // state per Kraus operator, which is especially important for
            // repeated depolarizing noise on larger state vectors.
            auto apply_pauli = [&](uint32_t qubit, uint32_t pauli) {
                if (pauli == 1) {
                    table.x(state.data(), dim, qubit, nullptr);
                } else if (pauli == 2) {
                    table.y(state.data(), dim, qubit, nullptr);
                } else if (pauli == 3) {
                    table.z(state.data(), dim, qubit, nullptr);
                }
            };

            bool used_fast_path = false;
            const auto& params = channel.params_f64;
            if (channel.use_pauli_fast_path && channel.channel_kind == "bit_flip" && params.size() == 1
                && channel.qubits.size() == 1) {
                if (dist01(rng) < params[0]) {
                    apply_pauli(channel.qubits[0], 1);
                }
                used_fast_path = true;
            } else if (channel.use_pauli_fast_path && channel.channel_kind == "phase_flip" && params.size() == 1
                       && channel.qubits.size() == 1) {
                if (dist01(rng) < params[0]) {
                    apply_pauli(channel.qubits[0], 3);
                }
                used_fast_path = true;
            } else if (channel.use_pauli_fast_path && channel.channel_kind == "bit_phase_flip" && params.size() == 1
                       && channel.qubits.size() == 1) {
                if (dist01(rng) < params[0]) {
                    apply_pauli(channel.qubits[0], 2);
                }
                used_fast_path = true;
            } else if (channel.use_pauli_fast_path && channel.channel_kind == "pauli" && params.size() == 3
                       && channel.qubits.size() == 1) {
                const double r = dist01(rng);
                if (r < params[0]) {
                    apply_pauli(channel.qubits[0], 1);
                } else if (r < params[0] + params[1]) {
                    apply_pauli(channel.qubits[0], 2);
                } else if (r < params[0] + params[1] + params[2]) {
                    apply_pauli(channel.qubits[0], 3);
                }
                used_fast_path = true;
            } else if (channel.use_pauli_fast_path && channel.channel_kind == "depolarizing" && params.size() == 1
                       && !channel.qubits.empty() && channel.qubits.size() < 32) {
                // QuSpect follows the depolarizing-strength convention
                // E(rho) = (1-p)rho + p I/d.  Equivalently, each of the 4^k
                // Pauli strings has weight p/4^k in addition to the identity
                // contribution 1-p.
                const uint32_t bits = static_cast<uint32_t>(2 * channel.qubits.size());
                const uint64_t pauli_space = uint64_t{1} << bits;
                const double non_identity_probability =
                    params[0] * static_cast<double>(pauli_space - 1) / static_cast<double>(pauli_space);
                if (dist01(rng) < non_identity_probability) {
                    std::uniform_int_distribution<uint64_t> select_pauli(1, pauli_space - 1);
                    uint64_t pauli_word = select_pauli(rng);
                    for (uint32_t qubit : channel.qubits) {
                        apply_pauli(qubit, static_cast<uint32_t>(pauli_word & 0x3ULL));
                        pauli_word >>= 2;
                    }
                }
                used_fast_path = true;
            }

            if (!used_fast_path) {
                if (channel.num_kraus_ops == 0 || channel.qubits.empty()) {
                    throw std::runtime_error("Invalid Kraus channel '" + channel.channel_kind + "'");
                }
                // Public Kraus matrices use the conventional big-endian
                // tensor-product order (the first numpy kron factor is the
                // most-significant local bit).  The native dense kernel maps
                // targets[i] to local bit i, so reverse only at this boundary.
                std::vector<uint32_t> kernel_qubits(channel.qubits.rbegin(), channel.qubits.rend());
                if constexpr (sizeof(FP) == sizeof(double)) {
                    kernels::apply_kraus_monte_carlo<double>(
                        state.data(), dim, channel.kraus_operators_f64.data(), channel.num_kraus_ops,
                        kernel_qubits.data(), static_cast<uint32_t>(kernel_qubits.size()), rng);
                } else {
                    kernels::apply_kraus_monte_carlo<float>(
                        state.data(), dim, channel.kraus_operators_f32.data(), channel.num_kraus_ops,
                        kernel_qubits.data(), static_cast<uint32_t>(kernel_qubits.size()), rng);
                }
            }
        } else if (std::holds_alternative<MeasureOperation>(op_var)) {
            // 测量操作 — Phase 4
            if (execute_measurements) {
                const auto& meas = std::get<MeasureOperation>(op_var);

                // 对每个 qubit-clbit 对依次测量
                for (size_t idx = 0; idx < meas.qubits.size(); ++idx) {
                    uint32_t qubit = meas.qubits[idx];

                    // 计算 P(qubit=0)
                    FP p0 = table.prob0(state.data(), dim, qubit);

                    // 采样测量结果
                    double r = dist01(rng);
                    int outcome = (static_cast<double>(p0) > r) ? 0 : 1;

                    // 坍缩状态向量
                    table.collapse(state.data(), dim, qubit, outcome);

                    // 写入经典寄存器
                    if (idx < meas.clbits.size()) {
                        creg.Set(meas.clbits[idx], static_cast<uint8_t>(outcome));
                    }
                }
            }
        }
    }
}

// ============================================================
// 访问器
// ============================================================

template <typename FP>
StateVector<FP>& RuntimeEngine<FP>::GetStateVector() {
    return impl_->state;
}

template <typename FP>
const StateVector<FP>& RuntimeEngine<FP>::GetStateVector() const {
    return impl_->state;
}

template <typename FP>
ClassicalRegisterFile& RuntimeEngine<FP>::GetClassicalRegister() {
    return impl_->creg;
}

template <typename FP>
const ClassicalRegisterFile& RuntimeEngine<FP>::GetClassicalRegister() const {
    return impl_->creg;
}

template <typename FP>
uint64_t RuntimeEngine<FP>::SampleBasis() {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    const auto& table = impl_->table;
    const auto& state = impl_->state;
    return table.sampling(state.data(), state.dim(), dist01(impl_->rng));
}

// 显式实例化
template class RuntimeEngine<float>;
template class RuntimeEngine<double>;

}  // namespace quspect::native::runtime
