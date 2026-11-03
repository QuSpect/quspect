#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::synthesis {

// Pauli 旋转序列在耦合图上的综合（quspect.hamsim 的原生核心）。
//
// 输入是按时间顺序的 Pauli 旋转 exp(-iθ_k P_k)（只给出 P_k；角度由 Python 端填入），输出是一串
// 单比特 Clifford、耦合边上的 CX 与"第 k 个旋转作用在比特 q 的 σ 轴上"的标记，整体精确实现
//
//     Π_k exp(-iθ_k P_k)   （只交换相互对易的旋转，乘积不变）。
//
// 方法（Clifford 框架贪心）：维护一个 Clifford 框架 F，使已输出的线路 E 满足 E = F·(已实现的旋转)。
// 下一个旋转在框架中的像 Q = F P F† 用耦合边上的两比特 Clifford（每个 1 个 CX）逐步约化到单比特，
// 然后输出单比特旋转；框架不还原，留给后续旋转复用（相邻旋转共享 Clifford 前缀）。
//   * 约化门：边 (a, b) 两端都在 Q 的支集上时，C(σ_a, σ_b) 消去一端（4 种选择）；
//     支集在图上不连通时，用两个这样的门把一端的 Pauli 移到更靠近其余支集的邻居上（"跳跃"，
//     2 个 CX，SWAP 需要 3 个 CX；位置的变化记在框架里，不需要换回）。
//   * 选择：在可选门中取使 Q 与之后 window 个旋转的估计代价（支集在图上的 Steiner 树估计，
//     相邻 1 个 CX、距离 d 记 2d − 1）加权和最小者；下一个旋转从"对易前沿"（与之前所有未实现
//     旋转都对易的旋转，限前 front 个）中取代价最小者。
//   * 块（block_mode = 1）：当前旋转的像是一条耦合边 (a, b) 上的两比特 Pauli 时，把对易前沿中像也在
//     {a, b} 上、可以提前的旋转并成一个两比特块，在原位输出（不改变框架，交给后端合成为 ≤ 3 个 CX 的
//     SU(4) 块）；或者在块后再加一个 SWAP 并把它记进框架（"镜像门"：泛型 SU(4) 块乘 SWAP 仍是 3 个 CX，
//     等于免费移动两个比特上的 Pauli）。三种做法（框架约化 / 原位块 / 块 + SWAP）按同一个前瞻代价比较。
//   * 收尾（finalize）：最后把框架 F 本身在同一张图上综合掉（逐个取活动子图的非割点 v，把一对行
//     F X_q F†、F Z_q F† 约化回 X_v、Z_v），线路因此精确等于旋转之积（不留 Clifford 尾巴）。
//     finalize_permutation 时 q 可以不等于 v（取约化代价最小的行对），框架只还原到一个比特置换，
//     由输出布局吸收（final_perm）。
// 全部计分为整数运算，结果与平台无关（逐位一致）。

struct PauliNetworkOptions {
    int window = 12;           // 计分时向后看的旋转数
    int front = 64;            // 对易前沿的搜索范围（未实现旋转的前 front 个）
    int next_weight = 512;     // 第一个向后看旋转的权重（单位 1/1024；当前旋转为 1024）
    int decay = 614;           // 之后每个位置权重乘 decay/1024
    bool finalize = true;      // 是否把最终框架综合掉（false 时输出 E = F·U，框架见 frame）
    int block_mode = 0;        // 0：只用框架约化；1：边上的两比特块（原位 / 加镜像 SWAP）参与比较
    int block_bias = 1024;     // 块代价的权重（单位 1/1024；> 1024 偏向框架约化）
    bool finalize_permutation = false;  // 收尾只还原到比特置换（final_perm），由布局吸收
    int cx_limit = 0;          // > 0：CX 数超过它时提前停止（aborted，输出不完整）
};

enum class NetGateKind : std::uint8_t { kH = 0, kS = 1, kSdg = 2, kCX = 3, kX = 4, kY = 5, kZ = 6, kRotation = 7, kSwap = 8 };

// 输出门。kRotation：第 rotation 个输入旋转在比特 q0 上沿 axis（'X'/'Y'/'Z'）实现，
// negate 为真时角度取反（框架像带负号）。kSwap：(q0, q1) 上的 SWAP（只出现在块之后，由后端与块合并）。
struct NetGate {
    NetGateKind kind;
    int q0;
    int q1;
    int rotation;
    char axis;
    bool negate;
};

struct PauliNetworkResult {
    std::vector<NetGate> gates;
    std::vector<int> order;  // 旋转的实际实现顺序（输入编号）
    int num_cx = 0;       // CX 数；块按估计的合成代价计（≤ 3，块 + SWAP 记 3）
    int finalize_cx = 0;  // 其中收尾框架综合的 CX 数
    int blocks = 0;       // 原位输出的块数（其中 swap_blocks 个带镜像 SWAP）
    int swap_blocks = 0;
    // 收尾后的置换：初始在物理比特 q 上的态最终在 final_perm[q] 上（finalize_permutation 为假时恒等）
    std::vector<int> final_perm;
    bool aborted = false;  // 超过 cx_limit 提前停止
};

// paulis[k] = (x 位, z 位)，比特编号 < num_qubits ≤ 64·words。所有旋转的支集必须在 active 内；
// active 在 edges 上的导出子图必须连通（只用这张子图上的边）。
//
// 抛出 std::invalid_argument：比特越界、支集不在 active 内或 active 不连通。
PauliNetworkResult SynthesizePauliNetwork(int num_qubits, const std::vector<std::pair<int, int>>& edges,
                                          const std::vector<int>& active,
                                          const std::vector<std::vector<int>>& x_qubits,
                                          const std::vector<std::vector<int>>& z_qubits,
                                          const PauliNetworkOptions& options);

// 精确合并：把与之前某个相同 Pauli 之间只隔着与之对易的旋转的旋转并入前者（向前最多看 span 个
// 合并后的旋转）。返回 slot[k]（并入的输出编号）；输出旋转的 Pauli 是其第一个输入。
std::vector<int> MergeCommutingRotations(int num_qubits, const std::vector<std::vector<int>>& x_qubits,
                                         const std::vector<std::vector<int>>& z_qubits, int span);

}  // namespace quspect::native::synthesis
