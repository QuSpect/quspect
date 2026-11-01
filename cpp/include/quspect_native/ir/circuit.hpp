#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "quspect_native/ir/operation.hpp"
#include "quspect_native/ir/small_vector.hpp"

namespace quspect::native::ir {

// 指令在线路中的下标（tweedledum 的 InstRef）。
using InstRef = std::uint32_t;
constexpr InstRef kNoInst = 0xffffffffu;

using BitList = SmallVector<std::uint32_t, 3>;

// 一条指令：操作 + 比特 + 各操作数线上的前驱（隐式 DAG）。
//
// preds[i] 是第 i 个操作数所在线（先 qubits 后 clbits）上的上一条指令，线的起点为 kNoInst。
// 后继不显式保存：需要时由 Circuit 一次性建出 CSR 索引。
struct Instruction {
    Operation op;
    BitList qubits;
    SmallVector<std::uint32_t, 2> clbits;
    SmallVector<InstRef, 3> preds;
    std::uint32_t host = Operation::kNoExt;  // 宿主语言附带的对象（例如 Python 指令）编号

    std::size_t num_wires() const noexcept {
        return qubits.size() + clbits.size();
    }
};

// 连续存储的量子线路（tweedledum 风格）：指令按程序顺序存放在一个 std::vector 中，以下标互相
// 引用，没有逐节点的堆分配；每条线记录最后一条指令，追加指令时即得到各操作数线上的前驱，
// 因此线路本身就是 DAG（程序顺序即一个拓扑序），C++ pass 可以直接在上面遍历依赖。
//
// 比特数按用到的最大编号 + 1 推断（与 Python QuantumCircuit 一致）。
class Circuit {
public:
    Circuit() = default;

    // ---- 构造 ----
    InstRef append(const Operation& op, const std::uint32_t* qubits, std::size_t num_qubits,
                   const std::uint32_t* clbits = nullptr, std::size_t num_clbits = 0);
    InstRef append(const Operation& op, std::initializer_list<std::uint32_t> qubits,
                   std::initializer_list<std::uint32_t> clbits = {}) {
        return append(op, qubits.begin(), qubits.size(), clbits.begin(), clbits.size());
    }
    void reserve(std::size_t n) {
        insts_.reserve(n);
    }
    void clear();

    // ---- 访问 ----
    std::size_t size() const noexcept {
        return insts_.size();
    }
    bool empty() const noexcept {
        return insts_.empty();
    }
    const Instruction& operator[](InstRef i) const noexcept {
        return insts_[i];
    }
    Instruction& at_mutable(InstRef i) noexcept {
        return insts_[i];
    }
    std::vector<Instruction>::const_iterator begin() const noexcept {
        return insts_.begin();
    }
    std::vector<Instruction>::const_iterator end() const noexcept {
        return insts_.end();
    }
    std::uint32_t num_qubits() const noexcept {
        return static_cast<std::uint32_t>(last_qubit_.size());
    }
    std::uint32_t num_clbits() const noexcept {
        return static_cast<std::uint32_t>(last_clbit_.size());
    }
    // 线上最后一条指令（没有时为 kNoInst）。
    InstRef last_on_qubit(std::uint32_t q) const noexcept {
        return q < last_qubit_.size() ? last_qubit_[q] : kNoInst;
    }
    InstRef last_on_clbit(std::uint32_t c) const noexcept {
        return c < last_clbit_.size() ? last_clbit_[c] : kNoInst;
    }

    // ---- 名字表（kOpaque 门） ----
    std::uint32_t intern_name(std::string_view name);
    std::string_view name(std::uint32_t id) const {
        return names_.at(id);
    }
    // 指令的门名（标准门查表，kOpaque 查名字表；非门指令返回 "measure" 等）。
    std::string_view gate_name(const Instruction& inst) const;

    // ---- 隐式 DAG ----
    // 对每个（去重后的）直接前驱调用 fn(InstRef)。
    template <typename F>
    void foreach_predecessor(InstRef i, F&& fn) const {
        const auto& preds = insts_[i].preds;
        for (std::size_t k = 0; k < preds.size(); ++k) {
            const InstRef p = preds[k];
            if (p == kNoInst) {
                continue;
            }
            bool seen = false;
            for (std::size_t j = 0; j < k; ++j) {
                seen = seen || preds[j] == p;
            }
            if (!seen) {
                fn(p);
            }
        }
    }
    // 对每个（去重后的）直接后继调用 fn(InstRef)，按后继下标递增。
    template <typename F>
    void foreach_successor(InstRef i, F&& fn) const {
        ensure_successors();
        for (std::uint32_t k = succ_offsets_[i]; k < succ_offsets_[i + 1]; ++k) {
            fn(succ_[k]);
        }
    }
    // 某个比特上的全部指令（程序顺序）。
    std::vector<InstRef> qubit_chain(std::uint32_t q) const;
    // 每条指令的 ASAP 层号（无前驱的指令在第 0 层）以及层数。
    std::vector<std::uint32_t> asap_layers(std::uint32_t* depth = nullptr) const;

private:
    void ensure_successors() const;

    std::vector<Instruction> insts_;
    std::vector<InstRef> last_qubit_;
    std::vector<InstRef> last_clbit_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, std::uint32_t> name_ids_;

    // 后继 CSR 索引（惰性构建，追加指令后失效）。
    mutable bool succ_valid_ = false;
    mutable std::vector<std::uint32_t> succ_offsets_;
    mutable std::vector<InstRef> succ_;
};

}  // namespace quspect::native::ir
