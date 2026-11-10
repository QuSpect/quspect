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
    // 结束时第 i 根线持有目标行 perm[i]（只有 free_mask 中的线可以与目标不同；没有自由线时为恒等）。
    std::vector<int> perm;
};

// n：比特数（1..6）；edges：允许的有向 CX（控制, 目标）；target_rows[i]：结束时第 i 根线的奇偶
// （n 位掩码，初始为 1 << i）；required：必须出现的奇偶掩码（重复与单位向量均可）；
// max_cx：CX 数上限（含）；node_budget：最多展开的状态数。
// free_mask：结束时可以互换的线（线路末端的线：输出相差一个置换由最终布局吸收）；这些线的终态只需
// 作为多重集等于对应的目标行。启发式中“不同的行数”相应地改为“自由目标行中尚未出现的个数”，仍可采纳。
CnotPhaseResult SynthesizeCnotPhase(int n, const std::vector<std::pair<int, int>>& edges,
                                    const std::vector<std::uint32_t>& target_rows,
                                    const std::vector<std::uint32_t>& required, int max_cx, std::int64_t node_budget,
                                    std::uint32_t free_mask = 0);

// 大区域（n ≤ 32）的启发式综合：逐层束搜索。状态 = (各线的奇偶行, 已出现的必需项)，一步 = 一条允许的 CX。
// 同一层的状态按估计的剩余代价排序，保留 beam_width 个（去掉以前出现过的状态）：
//     h = 2·缺的必需项数 + 2·Σ_{缺的 y} min_i |row_i ⊕ y| / P + 0.5·与目标不同的行数 + 0.5·Σ_i |E_i ⊕ e_i| / n，
// 其中 E = R·T⁻¹ 是当前行矩阵 R 相对目标 T 的“剩余线性映射”（目标是单位阵；自由线的 E_i 只需是自由线的单位向量）。
// 第一个满足全部条件的状态即返回（层数 = CX 数）；超过 max_cx 层则 found = false。不保证最优，
// 调用方只在严格更省时采用。结果与线程无关（确定性的排序与去重）。
CnotPhaseResult SynthesizeCnotPhaseBeam(int n, const std::vector<std::pair<int, int>>& edges,
                                        const std::vector<std::uint32_t>& target_rows,
                                        const std::vector<std::uint32_t>& required, int max_cx, int beam_width,
                                        std::uint32_t free_mask = 0);

}  // namespace quspect::native::transpiler
