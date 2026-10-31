#include <algorithm>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "quspect_native/executor/executor.hpp"
#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/runtime/classical_register.hpp"
#include "quspect_native/runtime/program.hpp"
#include "quspect_native/runtime/runtime_engine.hpp"
#include "quspect_native/runtime/state_vector.hpp"

namespace quspect::native::executor {

// ============================================================
// SimulationExecutor
// ============================================================

struct SimulationExecutor::Impl {
    Options options;
};

SimulationExecutor::SimulationExecutor(Options options) : impl_(std::make_unique<Impl>(Impl{options})) {
}

SimulationExecutor::~SimulationExecutor() = default;

namespace {

/// 检测线路中是否存在中途测量（非尾部的测量操作）
///
/// 扫描 ops 列表，若在任一 MeasureOperation 之后还存在
/// 非 Barrier/非 Measure 的 GateOperation，则为中途测量。
bool HasMidCircuitMeasurements(const std::vector<runtime::ProgramOperation>& ops) {
    bool seen_measure = false;
    for (const auto& op_var : ops) {
        if (std::holds_alternative<runtime::MeasureOperation>(op_var)) {
            seen_measure = true;
        } else if (std::holds_alternative<runtime::GateOperation>(op_var)) {
            const auto& gate = std::get<runtime::GateOperation>(op_var);
            if (gate.opcode == runtime::OpCode::kMeasure) {
                seen_measure = true;
            } else if (gate.opcode != runtime::OpCode::kBarrier) {
                // 非 Barrier 的门操作
                if (seen_measure) {
                    return true;  // 测量之后还有门 → 中途测量
                }
            }
        } else if (std::holds_alternative<runtime::ChannelOperation>(op_var) && seen_measure) {
            return true;  // a channel after a measurement is also quantum evolution
        }
    }
    return false;
}

bool HasNoiseChannels(const std::vector<runtime::ProgramOperation>& ops) {
    return std::any_of(ops.begin(), ops.end(), [](const auto& op) {
        return std::holds_alternative<runtime::ChannelOperation>(op);
    });
}

/// 收集所有终端测量操作的 (qubit, clbit) 映射
std::vector<std::pair<uint32_t, uint32_t>> CollectTerminalMeasures(const std::vector<runtime::ProgramOperation>& ops) {
    std::vector<std::pair<uint32_t, uint32_t>> mapping;
    for (const auto& op_var : ops) {
        if (std::holds_alternative<runtime::MeasureOperation>(op_var)) {
            const auto& meas = std::get<runtime::MeasureOperation>(op_var);
            for (size_t i = 0; i < meas.qubits.size(); ++i) {
                uint32_t q = meas.qubits[i];
                uint32_t c = (i < meas.clbits.size()) ? meas.clbits[i] : q;
                mapping.emplace_back(q, c);
            }
        }
    }
    return mapping;
}

/// 从 sampled basis index 中按 (qubit, clbit) 映射构建 packed bitstring
uint64_t BasisToPackedSample(uint64_t basis, const std::vector<std::pair<uint32_t, uint32_t>>& mapping) {
    uint64_t packed = 0;
    for (const auto& [qubit, clbit] : mapping) {
        uint64_t bit = (basis >> qubit) & 1ULL;
        const uint64_t mask = 1ULL << clbit;
        packed = (packed & ~mask) | (bit << clbit);
    }
    return packed;
}

/// 将 packed uint64 转为 num_bits 位的 bitstring (bit 0 = LSB)
std::string PackedToBitstring(uint64_t packed, uint32_t num_bits) {
    if (num_bits == 0)
        return "";
    std::string s(num_bits, '0');
    for (uint32_t i = 0; i < num_bits; ++i) {
        s[num_bits - 1 - i] = ((packed >> i) & 1ULL) ? '1' : '0';
    }
    return s;
}

void RejectSymbolicParameters(const runtime::QSALProgram& program) {
    for (size_t op_index = 0; op_index < program.ops.size(); ++op_index) {
        if (!std::holds_alternative<runtime::GateOperation>(program.ops[op_index])) {
            continue;
        }
        const auto& gate = std::get<runtime::GateOperation>(program.ops[op_index]);
        for (const auto& expression : gate.parameter_expressions) {
            if (expression.IsSymbolic()) {
                const auto& parameter_name = expression.terms.front().first;
                throw std::invalid_argument("SimulationExecutor cannot execute symbolic parameter '" + parameter_name
                                            + "' in op[" + std::to_string(op_index)
                                            + "]; bind parameter values or use a variational executor");
            }
        }
    }
}

}  // namespace

ExecutionResult SimulationExecutor::Run(const runtime::QSALProgram& program, const ExecutionRequest& request) {
    ExecutionResult result;

    RejectSymbolicParameters(program);

    // 输入校验
    if (program.num_qubits == 0) {
        result.metadata["error"] = "num_qubits must be > 0";
        return result;
    }
    if (program.num_clbits > 64) {
        throw std::invalid_argument("SimulationExecutor supports at most 64 classical bits");
    }

    bool is_mid_circuit = HasMidCircuitMeasurements(program.ops);
    bool has_noise = HasNoiseChannels(program.ops);
    uint64_t shots = request.shots;
    bool has_terminal_measures = !is_mid_circuit && !CollectTerminalMeasures(program.ops).empty();

    // 若请求了 sampling 但未设置 shots，默认至少 1 shot
    if ((request.return_samples || request.return_expectations) && shots == 0) {
        // 仅当没有显式设置 shots 且有测量操作时才需要至少 1 shot
        if (is_mid_circuit || has_terminal_measures) {
            shots = 1;
        }
    }
    // 若未请求采样且无 shots 设定，不做采样
    if (shots == 0 && !request.return_samples) {
        shots = 0;
    }
    // 若 shots > 0 但无测量，执行 shots 次无测量线路没有意义
    bool has_any_measure = is_mid_circuit || has_terminal_measures;
    if (shots > 0 && !has_any_measure) {
        shots = 0;
    }

    uint64_t effective_shots = (shots > 0) ? shots : 0;
    uint64_t master_seed = request.seed.value_or(impl_->options.seed);

    // 辅助 lambda：计算可观测量期望值
    auto compute_expectations = [&](const auto& engine, const auto& table) {
        if (request.return_expectations && !program.observables.empty()) {
            const auto& sv = engine.GetStateVector();
            for (const auto& obs : program.observables) {
                auto masks = runtime::ComputePauliMasks(obs.paulis);
                auto expect = table.pauli_expect(sv.data(), sv.dim(), masks.bit_flip_mask, masks.phase_flip_mask,
                                                 masks.pivot_qubit, masks.y_count);
                std::complex<double> expect_double(std::real(expect), std::imag(expect));
                auto weighted = expect_double * obs.coefficient;
                result.expectations_real[obs.label] += std::real(weighted);
                result.expectations_complex[obs.label] += weighted;
            }
        }
    };

    // ============================================================
    // 路径 A: 终端测量或无测量（快速路径）
    // ============================================================
    const bool requires_per_shot_trajectory = is_mid_circuit || (has_noise && effective_shots > 0);
    if (!requires_per_shot_trajectory) {
        // 单次执行线路（跳过终端测量门）
        if (impl_->options.precision == PrecisionKind::kFloat64) {
            const auto& table = kernels::GetDispatchF64();
            runtime::RuntimeEngine<double> engine(program, table, master_seed);
            engine.Run(/*execute_measurements=*/false);

            // 输出 statevector（测量前末态）
            if (request.return_statevector) {
                const auto& sv = engine.GetStateVector();
                result.statevector_f64.assign(sv.data(), sv.data() + sv.dim());
            }

            // 计算期望值
            compute_expectations(engine, table);

            // 终端采样
            if (effective_shots > 0 && has_terminal_measures) {
                const auto& sv = engine.GetStateVector();
                auto mapping = CollectTerminalMeasures(program.ops);
                uint32_t num_clbits = program.num_clbits;

                std::mt19937_64 sampling_rng(master_seed);
                std::uniform_real_distribution<double> dist01(0.0, 1.0);

                for (uint64_t shot = 0; shot < effective_shots; ++shot) {
                    double r = dist01(sampling_rng);
                    uint64_t basis = table.sampling(sv.data(), sv.dim(), r);
                    uint64_t packed = BasisToPackedSample(basis, mapping);

                    if (request.return_samples) {
                        result.packed_samples.push_back(packed);
                    }

                    if (num_clbits > 0) {
                        std::string bs = PackedToBitstring(packed, num_clbits);
                        result.counts[bs]++;
                    }
                }
            }

            result.metadata["precision"] = "double";
        } else {
            const auto& table = kernels::GetDispatchF32();
            runtime::RuntimeEngine<float> engine(program, table, master_seed);
            engine.Run(/*execute_measurements=*/false);

            // 输出 statevector
            if (request.return_statevector) {
                const auto& sv = engine.GetStateVector();
                result.statevector_f32.assign(sv.data(), sv.data() + sv.dim());
            }

            // 计算期望值
            compute_expectations(engine, table);

            // 终端采样
            if (effective_shots > 0 && has_terminal_measures) {
                const auto& sv = engine.GetStateVector();
                auto mapping = CollectTerminalMeasures(program.ops);
                uint32_t num_clbits = program.num_clbits;

                std::mt19937_64 sampling_rng(master_seed);
                std::uniform_real_distribution<double> dist01(0.0, 1.0);

                for (uint64_t shot = 0; shot < effective_shots; ++shot) {
                    double r = dist01(sampling_rng);
                    uint64_t basis = table.sampling(sv.data(), sv.dim(), r);
                    uint64_t packed = BasisToPackedSample(basis, mapping);

                    if (request.return_samples) {
                        result.packed_samples.push_back(packed);
                    }

                    if (num_clbits > 0) {
                        std::string bs = PackedToBitstring(packed, num_clbits);
                        result.counts[bs]++;
                    }
                }
            }

            result.metadata["precision"] = "single";
        }

        result.metadata["measurement_mode"] = has_noise ? "noisy_single_trajectory" : "terminal";
    }
    // ============================================================
    // 路径 B: 中途测量（逐 shot 执行）
    // ============================================================
    else {
        // 为每个 shot 使用独立种子，确保可复现性
        std::mt19937_64 master_rng(master_seed);
        const auto terminal_mapping = CollectTerminalMeasures(program.ops);

        for (uint64_t shot = 0; shot < effective_shots; ++shot) {
            uint64_t shot_seed = master_rng();

            if (impl_->options.precision == PrecisionKind::kFloat64) {
                const auto& table = kernels::GetDispatchF64();
                runtime::RuntimeEngine<double> engine(program, table, shot_seed);
                engine.Run(/*execute_measurements=*/is_mid_circuit);

                // 最后一个 shot 输出 statevector
                if (request.return_statevector && shot == effective_shots - 1) {
                    const auto& sv = engine.GetStateVector();
                    result.statevector_f64.assign(sv.data(), sv.data() + sv.dim());
                }

                // 最后一个 shot 计算期望值
                if (request.return_expectations && shot == effective_shots - 1) {
                    compute_expectations(engine, table);
                }

                uint64_t packed = 0;
                if (is_mid_circuit) {
                    const auto& creg = engine.GetClassicalRegister();
                    for (uint32_t b = 0; b < program.num_clbits; ++b) {
                        if (creg.Get(b)) {
                            packed |= (1ULL << b);
                        }
                    }
                } else {
                    // Terminal measurements can be sampled once from the
                    // final basis distribution.  This preserves the state
                    // vector and uses the trajectory's advancing RNG after
                    // all preceding noise channels.
                    packed = BasisToPackedSample(engine.SampleBasis(), terminal_mapping);
                }

                if (request.return_samples) {
                    result.packed_samples.push_back(packed);
                }

                if (program.num_clbits > 0) {
                    std::string bs = PackedToBitstring(packed, program.num_clbits);
                    result.counts[bs]++;
                }
            } else {
                const auto& table = kernels::GetDispatchF32();
                runtime::RuntimeEngine<float> engine(program, table, shot_seed);
                engine.Run(/*execute_measurements=*/is_mid_circuit);

                if (request.return_statevector && shot == effective_shots - 1) {
                    const auto& sv = engine.GetStateVector();
                    result.statevector_f32.assign(sv.data(), sv.data() + sv.dim());
                }

                if (request.return_expectations && shot == effective_shots - 1) {
                    compute_expectations(engine, table);
                }

                uint64_t packed = 0;
                if (is_mid_circuit) {
                    const auto& creg = engine.GetClassicalRegister();
                    for (uint32_t b = 0; b < program.num_clbits; ++b) {
                        if (creg.Get(b)) {
                            packed |= (1ULL << b);
                        }
                    }
                } else {
                    packed = BasisToPackedSample(engine.SampleBasis(), terminal_mapping);
                }

                if (request.return_samples) {
                    result.packed_samples.push_back(packed);
                }

                if (program.num_clbits > 0) {
                    std::string bs = PackedToBitstring(packed, program.num_clbits);
                    result.counts[bs]++;
                }
            }
        }

        result.metadata["precision"] = (impl_->options.precision == PrecisionKind::kFloat64) ? "double" : "single";
        result.metadata["measurement_mode"] = is_mid_circuit ? "mid_circuit" : "noisy_trajectory";
    }

    uint32_t num_qubits = program.num_qubits;
    result.metadata["num_qubits"] = std::to_string(num_qubits);
    result.metadata["num_ops"] = std::to_string(static_cast<uint32_t>(program.ops.size()));
    result.metadata["shots"] = std::to_string(effective_shots);
    result.metadata["has_noise"] = has_noise ? "true" : "false";

    return result;
}

// ============================================================
// HardwareExecutor — 占位
// ============================================================

HardwareExecutor::HardwareExecutor() = default;
HardwareExecutor::~HardwareExecutor() = default;

ExecutionResult HardwareExecutor::Run(const runtime::QSALProgram& /*program*/, const ExecutionRequest& /*request*/) {
    throw std::runtime_error("NotImplemented: hardware backend is not available yet");
}

}  // namespace quspect::native::executor
