#pragma once

// 逐层完美匹配线路（如量子体积 QV：每层是一组互不相交的一般 SU(4) 块）在路径（line）上的路由。
//
// 模型：一般 SU(4) 块 U 与其同对 SWAP 合并后仍是一个两比特酉（≤ 3 个 CZ），所以每个块执行时两个比特
// 可以免费交换。若每层的块都在同一时刻处于相邻位置（"对齐"调度），第 l 层的排列 σ_l 就只由块的顺序
// 决定（块内两个比特的顺序免费），层间的最少 SWAP 数（逆序数，Kendall-τ 距离）是
//     d(σ_{l-1}, σ_l) = #{(u, v)：u、v 在 σ_{l-1} 和 σ_l 中都不同块，且相对顺序不同}，
// 块内的顺序在离开上一层时（U·SWAP）和到达这一层时（SWAP·U）都免费调整。小规模穷举表明对齐调度
// 不损失最优性（与任意异步调度的最少付费 SWAP 数相同），于是路由化为一条"有序划分链"的优化：
//     min_{σ_1..σ_L} Σ_l d(σ_{l-1}, σ_l)，
// 类似 Kemeny 中位排列。固定两侧邻居时 σ_l 的子问题是线性排序问题（LOP，权 w(A, B) = A 排在 B
// 之前时与两侧邻居的逆序对数），用插入式局部搜索求解；整条链做前后交替的块坐标下降，再加扰动 +
// 局部重优化（迭代局部搜索）。层间的 SWAP 用奇偶换位排序（odd-even transposition sort）生成：
// 只交换逆序的相邻对，SWAP 数恰为逆序数，轮数 ≤ n（深度小）。

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

struct LayerChainOptions {
    // 块坐标下降的最大扫描轮数（某轮总代价不降即停）。
    int max_sweeps = 64;
    // 迭代局部搜索的扰动次数（0：不做）。
    int perturbations = 0;
    // 每次扰动重排的连续层数。
    int perturb_span = 6;
    std::uint64_t seed = 1;
};

struct LayerChainResult {
    // 付费 SWAP 总数（= Σ_l d(σ_{l-1}, σ_l)）。
    std::int64_t cost = 0;
    // 初始排列：位置 -> 比特。
    std::vector<int> initial;
    // swaps[l]：第 l 层执行之后、第 l + 1 层之前依次做的相邻交换（位置 i 表示交换位置 i 与 i + 1），
    // 先是上一层块内的免费翻转，再是奇偶换位排序（l = 0 .. L - 2）。
    std::vector<std::vector<int>> swaps;
    // 每次转移中免费翻转的个数（不计入 cost）。
    std::vector<int> flips;
    int sweeps = 0;
};

// ``layers[l]``：第 l 层的比特对（比特 0..n-1，每个比特每层至多出现一次）。
//
// 抛出 std::invalid_argument：比特越界或同层重复。
LayerChainResult RouteLayersOnPath(int n, const std::vector<std::vector<std::pair<int, int>>>& layers,
                                   const LayerChainOptions& options);

// 有序划分链的总代价 Σ_l d(σ_{l-1}, σ_l)（orders[l] 为第 l 层的块序列，块为 1 或 2 个比特）。
std::int64_t LayerChainCost(int n, const std::vector<std::vector<std::vector<int>>>& orders);

}  // namespace quspect::native::transpiler
