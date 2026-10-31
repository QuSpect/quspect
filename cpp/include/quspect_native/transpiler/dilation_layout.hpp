#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

// 交互图的一条带权边 (a, b, weight)，a < b，weight > 0。
struct WeightedEdge {
    int a;
    int b;
    std::int64_t weight;
};

// 一个低伸缩布局：cost = Σ w · (d(π(a), π(b)) − 1)，layout[q] 为物理比特（无相互作用的
// 逻辑比特为 −1）。
struct DilationLayout {
    std::int64_t cost;
    std::vector<int> layout;
};

// 与 Python 端 quspect.transpiler._embedding.dilation_layouts_python 逐位一致：
// 最大邻接序贪心生长（Warnsdorff 规则破平）+ 二交换模拟退火（SplitMix64 随机源），
// 返回至多 count 个互不相同的布局（代价从小到大）。
//
// 抛出 std::invalid_argument：尺寸不一致、边越界或权重非正。
std::vector<DilationLayout> DilationLayouts(int num_logical, const std::vector<WeightedEdge>& edges,
                                            const std::vector<std::vector<int>>& adjacency,
                                            const std::vector<std::vector<std::int64_t>>& distance, std::uint64_t seed,
                                            int count);

}  // namespace quspect::native::transpiler
