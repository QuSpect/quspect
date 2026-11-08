#pragma once

// 受耦合约束的 CNOT-相位线路最优综合（quspect.transpiler._phase_poly 的搜索内核）。
//
// CNOT-相位线路（只含 CX 与对角单比特门）作用为 |x⟩ ↦ e^{i f(x)} |A x⟩：A ∈ GL(n, 2) 是 CX 网络的
// 线性可逆映射，f(x) = Σ_y θ_y · (y·x) 是相位多项式（y 取遍出现过的奇偶项）。由路径和表示，
// 一条线路实现它当且仅当 (1) 其 CX 网络实现 A，(2) 每个 θ_y ≠ 0 的奇偶 y 在某一时刻出现在
// 某根线上（对角门放在那里）。本内核在给定的有向耦合边上搜索满足 (1)(2) 的最少 CX 序列：
// 状态 = (各线当前的奇偶行, 已出现的必需项集合)，一步 = 一条 CX（行 t ^= 行 c），A* 的启发式
//     h = max(尚未出现的必需项数, 与目标不同的行数)
// 是可采纳的（一条 CX 只改一行、只产生一个新奇偶）。所以在 max_cx 之内找到的解是最优的；
// 超过展开预算则放弃（返回 found = false）。确定性：同 f 值按 g 大者优先，再按入队顺序。
//
// 精确最优的 CNOT-相位综合问题见 Amy, Azimzadeh, Mosca, "On the CNOT-complexity of CNOT-phase
// circuits"（Quantum Sci. Technol. 4, 015002, 2018），本内核是其受耦合约束、小规模（n ≤ 6）的
// 直接搜索版本。

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

struct CnotPhaseResult {
    // 是否在 max_cx 与预算之内找到解。
    bool found = false;
    // CX 序列：(控制, 目标)，比特为 0..n-1。
    std::vector<std::pair<int, int>> moves;
    // 展开的状态数。
    std::int64_t expanded = 0;
};

// n：比特数（1..6）；edges：允许的有向 CX（控制, 目标）；target_rows[i]：结束时第 i 根线的奇偶
// （n 位掩码，初始为 1 << i）；required：必须出现的奇偶掩码（重复与单位向量均可）；
// max_cx：CX 数上限（含）；node_budget：最多展开的状态数。
CnotPhaseResult SynthesizeCnotPhase(int n, const std::vector<std::pair<int, int>>& edges,
                                    const std::vector<std::uint32_t>& target_rows,
                                    const std::vector<std::uint32_t>& required, int max_cx, std::int64_t node_budget);

}  // namespace quspect::native::transpiler
