#pragma once

// 路由的精确窗口搜索与大邻域搜索（LNS）。
//
// 一次路由是事件序列：执行路由 DAG 节点（>= 0）或物理 SWAP（-1 - (x * n + y)，x < y）。
// 本模块给出：
//   * 事件序列的 SWAP 代价模型（RouteSwapCost）：普通 SWAP 计 3 个 CZ；与同一物理比特对上相邻的
//     块（两比特节点）合并的 SWAP 只计该块的镜像代价 cz(U·SWAP) - cz(U)。
//   * ExactRoute：从给定初始布局出发，用 A*（可采纳下界）求整条线路在该代价模型下的最优路由
//     （小线路上估计启发式路由离最优的距离）。
//   * RefineRouting：以已有路由为起点，反复取一个窗口（连续若干个 SWAP 及其间的节点），固定窗口
//     入口布局与出口处"之后还要用到的"逻辑比特的位置，在窗口内用同一 A* 精确重路由；代价下降就替换。
//   * 深度次级目标（RefineOptions::depth_ties）：每个物理比特带一个"忙到何时"的时间（按双比特门层计，
//     SWAP 计其代价，节点计其权重），窗口出口后的其余事件固定，所以整条路由的（代理）深度恰为
//     max_q (出口时间[q] + 尾长[q])，尾长 = 出口后 q 上首个操作起的最长路径。A* 按 (代价, 深度) 字典序
//     搜索：代价下界不变（主目标的最优性不受影响），深度部分只用当前时间（时间只增不减，可采纳）。

#include <cstdint>
#include <vector>

#include "quspect_native/transpiler/sabre_routing.hpp"

namespace quspect::native::transpiler {

// 节点的代价信息（与 RoutingDag 一一对应）。
//
// ``mirror_cost[n]``：两比特节点 n 与紧邻的同对 SWAP 合并后多出的 CZ 数（0..3）；-1 表示不合并（按 3 计）。
// ``measure_token[n]``：测量节点所测的逻辑比特（测量打断其物理比特上的块），其他节点为 -1。
// ``node_weight[n]``：节点 n 的双比特门层数（深度代理的时间权重）；为空 / 越界时两比特节点计 1、
// 三比特节点计 6、测量计 0。
struct RouteCostModel {
    std::vector<int> mirror_cost;
    std::vector<int> measure_token;
    std::vector<int> node_weight = {};
};

// 精确搜索的选项。
struct ExactSearchOptions {
    // A* 展开的状态数上限（达到后返回目前找到的最好解，optimal = false）。
    std::int64_t max_expansions = 200000;
    // false：所有 SWAP 计 3（纯 SWAP 数最优）；true：按镜像代价模型。
    bool merge_costs = true;
};

struct ExactRouteResult {
    // 是否找到解（预算内）。
    bool found = false;
    // 是否证明最优（A* 在预算内弹出目标状态）。
    bool optimal = false;
    std::int64_t cost = 0;
    std::int64_t num_swaps = 0;
    std::int64_t expansions = 0;
    std::vector<std::int64_t> events;
    // 已证明的最优代价下界（optimal 时等于 cost）。
    std::int64_t lower_bound = 0;
};

// LNS 选项。
struct RefineOptions {
    // 依次尝试的窗口大小（窗口内原有的 SWAP 个数）。
    std::vector<int> window_swaps = {1, 2, 3, 4};
    // 对全部窗口大小的扫描轮数（某一轮没有任何改进就提前结束）。
    int passes = 2;
    // 每个窗口的 A* 展开上限与全局展开预算。
    std::int64_t window_expansions = 4000;
    std::int64_t total_expansions = 4000000;
    // 窗口内最多的节点数（超过则跳过该窗口）。
    int max_window_nodes = 64;
    // 出口之后为合并判定而向后扫描的事件数上限。
    int lookahead_events = 4096;
    bool merge_costs = true;
    // 窗口搜索的顺序权重（g + weight * h；> 1 为加权 A*）与"找到第一个改进即停"。
    double weight = 1.0;
    bool stop_at_first = false;
    // 主扫描以深度为次级目标：代价下降时在最便宜的窗口解中取（代理）深度最小的。
    bool depth_ties = false;
    // 主扫描之后再做的"代价相同、深度更小也替换"的扫描轮数及其 A* 展开总预算（与 depth_ties 无关）。
    int depth_passes = 0;
    std::int64_t depth_expansions = 0;
};

struct RefineResult {
    std::vector<std::int64_t> events;
    std::int64_t old_cost = 0;
    std::int64_t new_cost = 0;
    std::int64_t old_swaps = 0;
    std::int64_t new_swaps = 0;
    std::int64_t windows = 0;
    std::int64_t improved = 0;
    std::int64_t expansions = 0;
    // 整条路由的深度代理（双比特门层，见文件头）精炼前后的值（不跟踪深度时为 0）。
    std::int64_t old_depth = 0;
    std::int64_t new_depth = 0;
};

// 事件序列在代价模型下的 SWAP 总代价（见文件头）。规则：SWAP 的代价 = min(前一操作合并, 后一操作合并)，
// 其中"前 / 后一操作合并"指两个物理比特上紧邻的前 / 后一个操作是同一个落在这对比特上的两比特节点 m
// （单比特门不打断），代价为 m 的镜像代价，否则为 3。
//
// 抛出 std::invalid_argument：布局 / 事件不合法（节点不可执行、依赖未满足、SWAP 不在耦合边上）。
std::int64_t RouteSwapCost(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                           const std::vector<int>& layout, const std::vector<std::int64_t>& events,
                           bool merge_costs = true);

// 从 ``layout`` 出发执行整个 DAG 的最优路由（A*）。``upper_bound`` > 0 时只找代价严格小于它的解。
//
// 抛出 std::invalid_argument：布局不合法或拓扑不连通。
ExactRouteResult ExactRoute(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                            const std::vector<int>& layout, const ExactSearchOptions& options,
                            std::int64_t upper_bound = 0);

// 大邻域搜索精炼 ``events``（从 ``layout`` 出发的合法路由）。返回的事件序列同样合法、执行同一组节点，
// 代价（RouteSwapCost）不增。含三比特节点的窗口跳过。
RefineResult RefineRouting(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                           const std::vector<int>& layout, const std::vector<std::int64_t>& events,
                           const RefineOptions& options);

}  // namespace quspect::native::transpiler
