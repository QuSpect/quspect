#include "quspect_native/ir/circuit.hpp"

#include <algorithm>

namespace quspect::native::ir {

std::optional<OpKind> GateKindFromName(std::string_view name) noexcept {
    for (std::size_t k = 0; k < static_cast<std::size_t>(OpKind::kOpaque); ++k) {
        if (kOpTable[k].name == name) {
            return static_cast<OpKind>(k);
        }
    }
    return std::nullopt;
}

namespace {

void Grow(std::vector<InstRef>& last, std::uint32_t bit) {
    if (bit >= last.size()) {
        last.resize(static_cast<std::size_t>(bit) + 1, kNoInst);
    }
}

}  // namespace

InstRef Circuit::append(const Operation& op, const std::uint32_t* qubits, std::size_t num_qubits,
                        const std::uint32_t* clbits, std::size_t num_clbits) {
    if (insts_.size() >= kNoInst) {
        throw std::length_error("circuit has too many instructions");
    }
    const auto ref = static_cast<InstRef>(insts_.size());
    Instruction inst;
    inst.op = op;
    inst.qubits.assign(qubits, qubits + num_qubits);
    inst.clbits.assign(clbits, clbits + num_clbits);
    inst.preds.reserve(num_qubits + num_clbits);
    for (std::size_t i = 0; i < num_qubits; ++i) {
        Grow(last_qubit_, qubits[i]);
        inst.preds.push_back(last_qubit_[qubits[i]]);
    }
    for (std::size_t i = 0; i < num_clbits; ++i) {
        Grow(last_clbit_, clbits[i]);
        inst.preds.push_back(last_clbit_[clbits[i]]);
    }
    for (std::size_t i = 0; i < num_qubits; ++i) {
        last_qubit_[qubits[i]] = ref;
    }
    for (std::size_t i = 0; i < num_clbits; ++i) {
        last_clbit_[clbits[i]] = ref;
    }
    insts_.push_back(std::move(inst));
    succ_valid_ = false;
    return ref;
}

void Circuit::clear() {
    insts_.clear();
    last_qubit_.clear();
    last_clbit_.clear();
    names_.clear();
    name_ids_.clear();
    succ_valid_ = false;
}

std::uint32_t Circuit::intern_name(std::string_view name) {
    auto it = name_ids_.find(std::string(name));
    if (it != name_ids_.end()) {
        return it->second;
    }
    const auto id = static_cast<std::uint32_t>(names_.size());
    names_.emplace_back(name);
    name_ids_.emplace(names_.back(), id);
    return id;
}

std::string_view Circuit::gate_name(const Instruction& inst) const {
    if (inst.op.kind == OpKind::kOpaque) {
        return inst.op.has_ext() ? std::string_view(names_.at(inst.op.ext)) : std::string_view();
    }
    return GetOpInfo(inst.op.kind).name;
}

void Circuit::ensure_successors() const {
    if (succ_valid_) {
        return;
    }
    const std::size_t n = insts_.size();
    succ_offsets_.assign(n + 1, 0);
    for (InstRef i = 0; i < n; ++i) {
        foreach_predecessor(i, [this](InstRef p) { ++succ_offsets_[p + 1]; });
    }
    for (std::size_t i = 0; i < n; ++i) {
        succ_offsets_[i + 1] += succ_offsets_[i];
    }
    succ_.assign(succ_offsets_[n], kNoInst);
    std::vector<std::uint32_t> fill(succ_offsets_.begin(), succ_offsets_.end() - 1);
    for (InstRef i = 0; i < n; ++i) {
        foreach_predecessor(i, [&](InstRef p) { succ_[fill[p]++] = i; });
    }
    succ_valid_ = true;
}

std::vector<InstRef> Circuit::qubit_chain(std::uint32_t q) const {
    std::vector<InstRef> chain;
    InstRef cur = last_on_qubit(q);
    while (cur != kNoInst) {
        chain.push_back(cur);
        const Instruction& inst = insts_[cur];
        InstRef prev = kNoInst;
        for (std::size_t k = 0; k < inst.qubits.size(); ++k) {
            if (inst.qubits[k] == q) {
                prev = inst.preds[k];
                break;
            }
        }
        cur = prev;
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

std::vector<std::uint32_t> Circuit::asap_layers(std::uint32_t* depth) const {
    std::vector<std::uint32_t> layer(insts_.size(), 0);
    std::uint32_t max_layer = 0;
    bool any = false;
    for (InstRef i = 0; i < insts_.size(); ++i) {
        std::uint32_t l = 0;
        foreach_predecessor(i, [&](InstRef p) { l = std::max(l, layer[p] + 1); });
        layer[i] = l;
        max_layer = std::max(max_layer, l);
        any = true;
    }
    if (depth != nullptr) {
        *depth = any ? max_layer + 1 : 0;
    }
    return layer;
}

}  // namespace quspect::native::ir
