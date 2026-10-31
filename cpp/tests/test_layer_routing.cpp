// 逐层匹配线路在路径上的路由（有序划分链 + 奇偶换位排序）的 gtest。
#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/layer_routing.hpp"

namespace {

using quspect::native::transpiler::LayerChainCost;
using quspect::native::transpiler::LayerChainOptions;
using quspect::native::transpiler::RouteLayersOnPath;

using Layers = std::vector<std::vector<std::pair<int, int>>>;

Layers RandomLayers(int n, int num, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Layers layers;
    for (int l = 0; l < num; ++l) {
        std::vector<int> p(n);
        for (int i = 0; i < n; ++i) {
            p[i] = i;
        }
        std::shuffle(p.begin(), p.end(), rng);
        std::vector<std::pair<int, int>> layer;
        for (int i = 0; i + 1 < n; i += 2) {
            layer.emplace_back(p[i], p[i + 1]);
        }
        layers.push_back(layer);
    }
    return layers;
}

// 重放：每层开始时该层的每一对都必须相邻；付费 SWAP 数 = 总交换数 - 免费翻转数 = cost；
// 免费翻转紧跟上一层的块（交换的正是上一层的一对）。
void CheckReplay(int n, const Layers& layers, const LayerChainOptions& options) {
    const auto res = RouteLayersOnPath(n, layers, options);
    ASSERT_EQ(static_cast<int>(res.initial.size()), n);
    std::vector<int> arr = res.initial;
    std::vector<int> where(n);
    std::int64_t paid = 0;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (int i = 0; i < n; ++i) {
            where[arr[i]] = i;
        }
        for (auto [a, b] : layers[l]) {
            EXPECT_EQ(std::abs(where[a] - where[b]), 1) << "layer " << l;
        }
        if (l + 1 == layers.size()) {
            break;
        }
        const auto& sw = res.swaps[l];
        const int flips = res.flips[l];
        ASSERT_LE(flips, static_cast<int>(sw.size()));
        std::set<std::pair<int, int>> prev;
        for (auto [a, b] : layers[l]) {
            prev.insert({std::min(a, b), std::max(a, b)});
        }
        for (int j = 0; j < static_cast<int>(sw.size()); ++j) {
            const int i = sw[j];
            ASSERT_GE(i, 0);
            ASSERT_LT(i + 1, n);
            if (j < flips) {
                EXPECT_TRUE(prev.count({std::min(arr[i], arr[i + 1]), std::max(arr[i], arr[i + 1])}));
            }
            std::swap(arr[i], arr[i + 1]);
        }
        paid += static_cast<std::int64_t>(sw.size()) - flips;
    }
    EXPECT_EQ(paid, res.cost);
}

TEST(LayerRouting, ReplayIsLegalAndCostMatches) {
    LayerChainOptions options;
    for (std::uint32_t seed = 0; seed < 6; ++seed) {
        CheckReplay(10, RandomLayers(10, 8, seed), options);
        CheckReplay(9, RandomLayers(9, 5, seed + 100), options);
    }
    options.perturbations = 50;
    CheckReplay(12, RandomLayers(12, 10, 7), options);
}

TEST(LayerRouting, RepeatedLayerIsFree) {
    // 两层相同：第二层不需要任何付费 SWAP
    Layers layers = {{{0, 5}, {1, 3}, {2, 4}}, {{0, 5}, {1, 3}, {2, 4}}};
    const auto res = RouteLayersOnPath(6, layers, LayerChainOptions{});
    EXPECT_EQ(res.cost, 0);
}

TEST(LayerRouting, ChainCostMatchesBruteForceDefinition) {
    // 两层 n = 4：{(0,1),(2,3)} → {(0,2),(1,3)}：最少 1 个付费 SWAP
    Layers layers = {{{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}};
    const auto res = RouteLayersOnPath(4, layers, LayerChainOptions{});
    EXPECT_EQ(res.cost, 1);
    std::vector<std::vector<std::vector<int>>> orders = {{{0, 1}, {2, 3}}, {{0, 2}, {1, 3}}};
    EXPECT_EQ(LayerChainCost(4, orders), 1);
}

TEST(LayerRouting, PerturbationsNeverIncreaseCost) {
    const auto layers = RandomLayers(16, 12, 3);
    LayerChainOptions plain;
    LayerChainOptions ils;
    ils.perturbations = 200;
    EXPECT_LE(RouteLayersOnPath(16, layers, ils).cost, RouteLayersOnPath(16, layers, plain).cost);
}

TEST(LayerRouting, RejectsOverlappingPairs) {
    Layers bad = {{{0, 1}, {1, 2}}};
    EXPECT_THROW(RouteLayersOnPath(3, bad, LayerChainOptions{}), std::invalid_argument);
}

}  // namespace
