#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "quspect_native/ir/dag_store.hpp"
#include "quspect_native/ir/small_vector.hpp"

namespace quspect::native::ir {

// quspect.dag.passes 中各优化 pass 的 C++ 实现，直接在 DagStore 上运行。
//
// 每个函数逐条复刻对应 Python pass 的遍历顺序、匹配规则与 DAG 调用序列（同样的 remove_edge /
// add_edge / remove_node 顺序，从而得到同样的边序号、拓扑序平局与事务日志），浮点运算也与
// Python 相同（单次加法 / 取模 / 比较，不涉及 FMA）。节点上的指令信息由 DagPassHost 解码为
// PassNode；需要创建新门对象的步骤（旋转合并、基矢重综合、矩阵恒等判断）回调宿主完成。

// ---- 门名驻留 ----

// 常用门名的固定编号；其他名字按首次出现分配编号（进程内唯一）。相同编号 <=> 相同名字。
enum GateNameId : std::int32_t {
    kNameNone = -1,
    kNameId = 0,
    kNameH,
    kNameX,
    kNameY,
    kNameZ,
    kNameS,
    kNameSdg,
    kNameT,
    kNameTdg,
    kNameRx,
    kNameRy,
    kNameRz,
    kNameP,
    kNameU1,
    kNameU2,
    kNameU3,
    kNameCx,
    kNameCz,
    kNameSwap,
    kNameCrx,
    kNameCry,
    kNameCrz,
    kNameCp,
    kNameRxx,
    kNameRyy,
    kNameRzz,
    kNumFixedGateNames
};

static_assert(kNumFixedGateNames <= 32, "gate families are 32-bit masks over the fixed names");

std::int32_t InternGateName(std::string_view name);

// ---- 节点信息 ----

// CircuitInstruction.type（指令为 None 时为 kNone）。
enum class PassInstrType : std::uint8_t { kNone, kGate, kMeasure, kBarrier, kChannel };

// 一个 DAG 节点在 pass 中用到的全部信息（DAGNode + CircuitInstruction + Gate 的字段）。
struct PassNode {
    bool operation = false;                     // node_type 为 OPERATION
    PassInstrType type = PassInstrType::kNone;  // instruction.type
    bool has_gate = false;                      // instruction.gate is not None
    std::int32_t name = kNameNone;              // gate.name
    std::int64_t num_qubits = 0;                // gate.num_qubits
    SmallVector<std::int64_t, 3> qubits;        // instruction.qubits（有序）
    SmallVector<std::int64_t, 2> clbits;        // instruction.clbits（有序）
    SmallVector<double, 3> params;              // float(p)（仅 concrete[i] 为真时有意义）
    SmallVector<std::uint8_t, 3> concrete;      // isinstance(p, (int, float))

    bool is_gate() const noexcept {
        return operation && type == PassInstrType::kGate;
    }
    bool is_measure() const noexcept {
        return operation && type == PassInstrType::kMeasure;
    }
    bool all_concrete() const noexcept {
        for (std::uint8_t c : concrete) {
            if (c == 0) {
                return false;
            }
        }
        return true;
    }
};

// 结构性错误（对应 Python 端抛出的 RuntimeError，例如 "Wire q0 of node 3/4 is broken."）。
class DagPassError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// pass 与宿主（Python 绑定或测试）之间的接口。
class DagPassHost {
public:
    virtual ~DagPassHost() = default;
    // 节点信息；节点不存在时抛出 DagKeyError(id)（对应 dag.get_node 的 KeyError）。
    virtual const PassNode& node(NodeId id) = 0;
    // 节点已从 DagStore 删除（宿主同步删除节点对象）。
    virtual void node_removed(NodeId id) = 0;
    // 节点 id 的指令换成 gate_name（rx / ry / rz / p 之一）的单参数门 angle，类型与比特不变，
    // 并记为一次修改（substitute_node_instruction）。
    virtual void substitute_rotation(NodeId id, std::int32_t gate_name, double angle) = 0;
    // u2 / u3 门是否在全局相位意义下为恒等（is_identity_gate 的矩阵分支）。
    virtual bool matrix_identity(NodeId id) = 0;
    // 基矢模式下单比特串 block 的重综合结果：替换门的个数，std::nullopt 表示保留原串。
    virtual std::optional<std::size_t> resynthesize(const std::vector<NodeId>& block) = 0;
    // 用最近一次 resynthesize 结果中的第 k 个门替换节点 id 的指令（记为一次修改）。
    virtual void apply_replacement(NodeId id, std::size_t k) = 0;
};

// ---- pass ----

// GateCancellation.run
void RunGateCancellation(DagStore& dag, DagPassHost& host);

// CommutationAnalysis.run（max_lookahead 为 MAX_LOOKAHEAD）
void RunCommutationAnalysis(DagStore& dag, DagPassHost& host, std::int64_t max_lookahead);

// RotationMerging.run
void RunRotationMerging(DagStore& dag, DagPassHost& host);

// SingleQubitGateFusion.run（basis_gates 不为 None 的基矢重综合模式）
void RunSingleQubitResynthesis(DagStore& dag, DagPassHost& host);

// EliminateDiagonalBeforeMeasurement.run
void RunEliminateDiagonalBeforeMeasurement(DagStore& dag, DagPassHost& host);

// RemoveIdentityGates.run。若某个旋转角为 ±inf（Python 的 math.fmod 会抛 ValueError），
// 在修改 DAG 之前返回 false，由调用方改走 Python 实现以得到相同的异常。
bool RunRemoveIdentityGates(DagStore& dag, DagPassHost& host);

}  // namespace quspect::native::ir
