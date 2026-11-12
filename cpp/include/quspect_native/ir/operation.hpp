#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>

namespace quspect::native::ir {

// 门 / 指令的种类。与 Python 端 quspect.gates.standard、quspect.gates.parameterized 中的
// 门一一对应（名字相同），另有测量、屏障、噪声通道，以及名字不在表中的 "其他门"（kOpaque，
// 名字保存在线路的名字表中）。新增种类只能追加在 kOpaque 之前并同步 kOpTable。
enum class OpKind : std::uint8_t {
    // 固定门
    kId,
    kH,
    kX,
    kY,
    kZ,
    kS,
    kSdg,
    kT,
    kTdg,
    kSX,
    kSXdg,
    kCX,
    kCZ,
    kSwap,
    kCCX,
    // 参数门
    kRX,
    kRY,
    kRZ,
    kP,
    kU1,
    kU2,
    kU3,
    kCRX,
    kCRY,
    kCRZ,
    kCP,
    kCU3,
    kRXX,
    kRYY,
    kRZZ,
    // 名字不在上表中的门
    kOpaque,
    // 非门指令
    kMeasure,
    kBarrier,
    kChannel,
};

constexpr std::size_t kNumOpKinds = static_cast<std::size_t>(OpKind::kChannel) + 1;

// 每种操作的静态信息。num_qubits / num_params 为 0xff 表示可变（屏障、测量、其他门、通道）。
struct OpInfo {
    std::string_view name;
    std::uint8_t num_qubits;
    std::uint8_t num_params;
};

constexpr std::uint8_t kVariable = 0xff;

inline constexpr std::array<OpInfo, kNumOpKinds> kOpTable = {{
    {"id", 1, 0},    {"h", 1, 0},     {"x", 1, 0},
    {"y", 1, 0},     {"z", 1, 0},     {"s", 1, 0},
    {"sdg", 1, 0},   {"t", 1, 0},     {"tdg", 1, 0},
    {"sx", 1, 0},    {"sxdg", 1, 0},  {"cx", 2, 0},
    {"cz", 2, 0},    {"swap", 2, 0},  {"ccx", 3, 0},
    {"rx", 1, 1},    {"ry", 1, 1},    {"rz", 1, 1},
    {"p", 1, 1},     {"u1", 1, 1},    {"u2", 1, 2},
    {"u3", 1, 3},    {"crx", 2, 1},   {"cry", 2, 1},
    {"crz", 2, 1},   {"cp", 2, 1},    {"cu3", 2, 3},
    {"rxx", 2, 1},   {"ryy", 2, 1},   {"rzz", 2, 1},
    {"", kVariable, kVariable},       {"measure", kVariable, 0},
    {"barrier", kVariable, 0},        {"channel", kVariable, 0},
}};

constexpr const OpInfo& GetOpInfo(OpKind kind) noexcept {
    return kOpTable[static_cast<std::size_t>(kind)];
}

constexpr bool IsGate(OpKind kind) noexcept {
    return kind <= OpKind::kOpaque;
}

// 门名 -> 种类（只查标准门与参数门，其余返回空）。
std::optional<OpKind> GateKindFromName(std::string_view name) noexcept;

// 操作（类型擦除的值类型，tweedledum 的小缓冲思路）：最多 3 个数值参数内联存放，32 字节，
// 可平凡复制。不能内联表示的部分（其他门的名字、符号参数、通道对象、标签等）通过 ext 引用
// 线路的侧表，ext == kNoExt 表示没有。
struct Operation {
    static constexpr std::uint32_t kNoExt = 0xffffffffu;
    static constexpr std::size_t kMaxInlineParams = 3;

    // flags 位
    static constexpr std::uint8_t kSymbolic = 1;  // 参数含符号表达式（由宿主语言保存在侧表）
    static constexpr std::uint8_t kLabeled = 2;   // 指令带标签（保存在侧表）

    OpKind kind = OpKind::kId;
    std::uint8_t num_params = 0;
    std::uint8_t flags = 0;
    std::uint8_t reserved = 0;
    std::uint32_t ext = kNoExt;
    std::array<double, kMaxInlineParams> params{};

    static Operation Gate(OpKind kind) noexcept {
        Operation op;
        op.kind = kind;
        return op;
    }
    static Operation Gate(OpKind kind, std::initializer_list<double> values) noexcept {
        Operation op;
        op.kind = kind;
        for (double v : values) {
            if (op.num_params < kMaxInlineParams) {
                op.params[op.num_params++] = v;
            }
        }
        return op;
    }
    bool is_gate() const noexcept {
        return IsGate(kind);
    }
    bool has_ext() const noexcept {
        return ext != kNoExt;
    }
};

static_assert(sizeof(Operation) == 32, "Operation is meant to fit in half a cache line");

}  // namespace quspect::native::ir
