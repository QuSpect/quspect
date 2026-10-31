// observable.cpp — ObservableTerm 工具函数
// Phase 3: Pauli 期望值计算与 observable 聚合

#include <cstdint>
#include <stdexcept>
#include <string>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/runtime/program.hpp"

namespace quspect::native::runtime {

// ============================================================
// ComputePauliMasks
// ============================================================

PauliMaskInfo ComputePauliMasks(const std::vector<std::pair<uint32_t, char>>& paulis) {
    PauliMaskInfo info;

    for (const auto& [qubit, pauli] : paulis) {
        if (qubit >= 64) {
            throw std::invalid_argument("ComputePauliMasks: qubit index " + std::to_string(qubit)
                                        + " exceeds maximum supported (63)");
        }

        uint64_t mask = 1ULL << qubit;

        switch (pauli) {
            case 'X':
            case 'x':
                info.bit_flip_mask |= mask;
                break;
            case 'Z':
            case 'z':
                info.phase_flip_mask |= mask;
                break;
            case 'Y':
            case 'y':
                info.bit_flip_mask |= mask;
                info.phase_flip_mask |= mask;
                info.y_count++;
                break;
            case 'I':
            case 'i':
                // 恒等算符，无操作
                break;
            default:
                throw std::invalid_argument("ComputePauliMasks: invalid Pauli character '" + std::string(1, pauli)
                                            + "', expected one of 'I', 'X', 'Y', 'Z'");
        }
    }

    // 选取 bit_flip_mask 的最低置位作为 pivot_qubit
    // 若 bit_flip_mask == 0（纯对角项），pivot_qubit 设为 0（kernel 中会忽略）
    if (info.bit_flip_mask != 0) {
        info.pivot_qubit = static_cast<uint32_t>(common::Ctz64(info.bit_flip_mask));
    } else {
        info.pivot_qubit = 0;
    }

    return info;
}

}  // namespace quspect::native::runtime
