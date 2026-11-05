#pragma once

// 全局束搜索（beam search）路由：只插 SWAP、按 SWAP 数优化的两比特门线路路由。
//
// 状态 = 布局 + 每个比特上"下一个两比特块"的指针（块按输入顺序排在各自两个比特的链上，链头
// 同为一个块时它才在前沿上），复制代价是 O(比特数)。每一步，束中每个状态对每条与前沿块比特
// 相邻的耦合边各插一个 SWAP（不立即撤销上一个 SWAP），随后贪心执行所有已相邻的前沿块；
// 子状态按代价
//     0.5 · F + ext_weight · 0.5 · X − exec_bonus · 已执行块数
// 加上衰减惩罚（自上次执行块以来反复使用同一物理比特的 SWAP 越来越贵，作用同 SABRE 的衰减）
// 排序，去掉此前已进入过束的状态（布局与指针都相同）后保留 ``beam`` 个——同一状态晚到只会多用
// SWAP，这也让束不会在等价状态之间绕圈。F 是前沿块的距离减一之和，X 是
// 每个比特往后 ``ext_depth`` 个块的距离减一按 ``gamma^k`` 衰减之和（两项都从两端各计一次，
// 故乘 0.5）。所有状态每步都恰好多插一个 SWAP，所以先执行完全部块的状态 SWAP 最少。
// 若 4 · N 步内束中最好的已执行块数没有增加，就把最好状态里距离最近的前沿块沿最短路直接接通
// （release valve），排除活锁。
//
// 布局：从给定布局出发路由，再从正向路由的终态布局反向路由得到新的起始布局，共
// ``layout_iters`` 轮（SABRE 布局迭代，但每次路由都用束搜索），取 SWAP 最少的正向路由。
//
// 与 SimpleTES（arXiv 2604.19341）的进化贪心策略不同，这里靠搜索而不是打分函数。束搜索做
// 量子比特路由已有前人工作（Qiu, Zhu, Wei, Entropy 27:232, 2025；Nishio et al., ACM JETC
// 16:3, 2020；Qiskit LookaheadSwap），代价的结构与前者相近。

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

struct BeamRouteOptions {
    // 束宽（≥ 1；1 即带去重的单步贪心）。
    int beam = 128;
    // 扩展集：每个比特往后看的块数。
    int ext_depth = 8;
    // 扩展集的逐块衰减系数。
    double gamma = 0.7;
    // 扩展集权重。
    double ext_weight = 0.5;
    // 每执行一个块的代价奖励。
    double exec_bonus = 3.0;
    // 衰减：自上次执行块以来，一个 SWAP 的两个物理比特每已参与过一次 SWAP，代价加 decay
    // （执行块后清零；0 关闭）。
    double decay = 1.0;
    // 正向 / 反向布局迭代轮数（0：只从给定布局正向路由一次）。
    int layout_iters = 2;
    // 并列打破用的随机种子。
    std::uint64_t seed = 1;
    // 每次路由（正向或反向）的 SWAP 上限，超过即放弃这次路由（0：不限）。
    std::int64_t max_swaps = 0;
};

struct BeamRouteResult {
    // 插入的 SWAP 数（-1：第一次正向路由就超过了 max_swaps，没有结果）。
    std::int64_t swaps = 0;
    // 正向路由的起始布局：layout[比特] = 物理比特（长度 N 的排列）。
    std::vector<int> layout;
    // 事件：>= 0 为块编号（输入 blocks 的下标），< 0 为 SWAP，编码 -1 - (x * N + y)（x < y）。
    std::vector<std::int64_t> events;
};

// ``adjacency``：N 个物理比特的邻接表（连通）；``blocks``：按拓扑序排列的两比特块（比特编号
// 0..N-1，同一比特上的块按出现顺序执行）；``layout``：初始布局（长度 N 的排列）。结果只依赖
// 输入与 ``options``（与线程数、平台无关）。
//
// 抛出 std::invalid_argument：布局不是排列、比特越界、选项越界或耦合图不连通。
BeamRouteResult RouteBeam(const std::vector<std::vector<int>>& adjacency,
                          const std::vector<std::pair<int, int>>& blocks, const std::vector<int>& layout,
                          const BeamRouteOptions& options);

}  // namespace quspect::native::transpiler
