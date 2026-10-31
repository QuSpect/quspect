#pragma once

// 一般 SU(4) 块线路（量子体积 QV 一类：所有两比特块的镜像代价为 0）在任意耦合图上的
// "截断 rollout" 路由。
//
// 这类线路里每个块执行时都可以免费交换它的两个比特（U·SWAP 仍是一个两比特酉），所以路由只为
// "让块的两个比特相遇"付费。SABRE 每步只看单个 SWAP 的代价函数
//     H = Σ_{前沿块}(d − 1) / |F| + W · Σ_{扩展块} d / |X|
// 是一个贪心基策略；这里按 Bertsekas 的 rollout（近似动态规划中的策略改进）思路，对每个候选
// SWAP 再向前模拟一步基策略（截断 rollout，深度 2）：候选 s 的分数 = H(s) + min_{s'} H(s, s')，
// 若 s 已使某个前沿块相邻（下一步即可执行）则第二项为 H(s) − exe_bonus。只对 H(s) 最小的
// ``width`` 个候选做第二层展开（beam）。执行块时按后两个块的伙伴距离决定是否免费翻转。
//
// 布局：从给定布局出发做 ``layout_iters`` 次"正向 → 反向"路由（SABRE 布局迭代），连同给定布局
// 本身作为正向路由的起点，取付费 SWAP 最少的一次。

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

struct Su4RouteOptions {
    // 第二层展开的候选数（≤ 1：退化为单步贪心）。
    int width = 16;
    // 扩展集权重 W。
    double ext_weight = 0.5;
    // 第一步就让前沿块相邻时第二项的奖励。
    double exe_bonus = 0.5;
    double decay_delta = 0.001;
    int decay_reset_interval = 5;
    // 正向 / 反向布局迭代次数。
    int layout_iters = 1;
    // 分数并列的候选中优先选两个物理比特最早空闲的（减小深度，不改变分数）。
    bool depth_tiebreak = true;
    std::uint64_t seed = 1;
};

struct Su4RouteResult {
    // 付费 SWAP 数（不含执行块后的免费翻转）。
    std::int64_t paid = 0;
    std::int64_t flips = 0;
    // 深度代理：块与付费 SWAP 各占一个时间层时的关键路径长度（多个起点付费数相同时取较小者）。
    std::int64_t depth = 0;
    // 正向路由的起始布局：layout[比特] = 物理比特。
    std::vector<int> layout;
    // 事件：>= 0 为块编号（输入 blocks 的下标），< 0 为 SWAP，编码 -1 - (x * N + y)（x < y）。
    std::vector<std::int64_t> events;
};

// ``adjacency``：N 个物理比特的邻接表（连通）；``blocks``：按拓扑序排列的两比特块（比特编号
// 0..N-1，同一比特的块按出现顺序执行）；``layout``：初始布局（长度 N 的排列）。
//
// 抛出 std::invalid_argument：布局不是排列、比特越界或耦合图不连通。
Su4RouteResult RouteSu4Blocks(const std::vector<std::vector<int>>& adjacency,
                              const std::vector<std::pair<int, int>>& blocks, const std::vector<int>& layout,
                              const Su4RouteOptions& options);

}  // namespace quspect::native::transpiler
