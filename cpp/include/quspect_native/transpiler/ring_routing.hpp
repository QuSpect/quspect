#pragma once

// 逐层匹配线路（量子体积 QV：每层是一组互不相交的一般 SU(4) 块）在环（ring）上的分层路由。
//
// 路径上的分层路由（layer_routing.hpp）把层间付费 SWAP 数化为块序之间的 Kendall-τ 距离。环上的相应
// 量是"循环排列的最少相邻交换数"：把环展开成它的万有覆盖（整数直线，每个比特在 x + k·n 处各有一个
// 副本），每个比特 t 选一个提升位移 δ_t ≡ y_t − x_t (mod n)，满足 Σ δ_t = 0（每个 SWAP 一个比特 +1、
// 一个 −1）。给定提升后，最少交换数恰为覆盖上副本的交叉数（Jerrum 1985）
//     C(δ) = Σ_{t<u} | ⌊(x_t + δ_t − x_u − δ_u) / n⌋ − ⌊(x_t − x_u) / n⌋ |，
// 并由"周期性"奇偶换位排序实现：相邻两个比特 a（左）、b（右）的剩余位移满足 r_a − r_b ≥ 2 时交换，
// 每次交换恰好消去一个交叉。转移代价 = min 于提升、上一层块内的免费翻转（离开时 U·SWAP）与本层块内
// 的朝向（到达时 SWAP·U）。每层的布局 = 块的循环顺序 + 旋转偏移；整条链 Σ_l C_l 用块坐标下降（插入
// 移动 + 旋转）与迭代局部搜索优化。转移内部的提升 / 翻转选择用局部搜索（单块翻转、成对提升交换）。

#include <cstdint>
#include <utility>
#include <vector>

#include "quspect_native/transpiler/layer_routing.hpp"

namespace quspect::native::transpiler {

// 环上一次转移的提升解（测试 / 诊断用）。
struct RingTransition {
    // 起点（提升坐标，已含免费翻转）与终点（提升坐标，已含本层块的朝向选择），按比特编号。
    std::vector<std::int64_t> x;
    std::vector<std::int64_t> t;
    std::int64_t crossings = 0;
};

// 给定提升坐标的交叉数 C（x、t 按比特编号，长度相同；n 为环长）。
std::int64_t RingCrossings(int n, const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& t);

// ``layers[l]``：第 l 层的比特对（比特 0..n-1，每层每个比特至多出现一次），n ≥ 3。
//
// 返回值的字段含义同 RouteLayersOnPath，但位置 i 的交换指环上的边 (i, (i + 1) mod n)。
//
// 抛出 std::invalid_argument：n < 3、比特越界或同层重复。
LayerChainResult RouteLayersOnRing(int n, const std::vector<std::vector<std::pair<int, int>>>& layers,
                                   const LayerChainOptions& options);

}  // namespace quspect::native::transpiler
