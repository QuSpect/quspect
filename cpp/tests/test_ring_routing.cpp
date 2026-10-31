// 逐层匹配线路在环上的分层路由（循环块序链 + 万有覆盖交叉数）的 gtest。
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/ring_routing.hpp"

namespace {

using quspect::native::transpiler::LayerChainOptions;
using quspect::native::transpiler::RingCrossings;
using quspect::native::transpiler::RouteLayersOnRing;

using Layers = std::vector<std::vector<std::pair<int, int>>>;

Layers RandomLayers(int n, int num, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Layers layers;
    for (int l = 0; l < num; ++l) {
        std::vector<int> p(n);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        std::vector<std::pair<int, int>> layer;
        for (int i = 0; i + 1 < n; i += 2) {
            layer.emplace_back(p[i], p[i + 1]);
        }
        layers.push_back(layer);
    }
    return layers;
}

// 环上所有排列的精确最少相邻交换数（BFS，从恒等排列出发；arr[位置] = 比特）。
std::map<std::vector<int>, int> RingDistances(int n) {
    std::map<std::vector<int>, int> dist;
    std::vector<int> start(n);
    std::iota(start.begin(), start.end(), 0);
    dist[start] = 0;
    std::deque<std::vector<int>> queue{start};
    while (!queue.empty()) {
        const auto cur = queue.front();
        queue.pop_front();
        const int d = dist[cur];
        for (int i = 0; i < n; ++i) {
            auto next = cur;
            std::swap(next[i], next[(i + 1) % n]);
            if (dist.emplace(next, d + 1).second) {
                queue.push_back(next);
            }
        }
    }
    return dist;
}

TEST(RingRouting, CrossingFormulaMatchesExactCyclicSortingDistance) {
    // Jerrum：最少相邻交换数 = min 于 Σδ = 0 的提升的交叉数（这里在 δ ∈ (y − x) + n·{−1, 0, 1} 上穷举）
    const int n = 6;
    const auto dist = RingDistances(n);
    for (const auto& [arr, d] : dist) {
        std::vector<std::int64_t> x(n);
        std::vector<int> y(n);
        for (int t = 0; t < n; ++t) {
            x[t] = t;
        }
        for (int i = 0; i < n; ++i) {
            y[arr[i]] = i;
        }
        std::int64_t best = -1;
        std::vector<int> k(n, -1);
        while (true) {
            std::vector<std::int64_t> t(n);
            std::int64_t sum = 0;
            for (int i = 0; i < n; ++i) {
                t[i] = y[i] + static_cast<std::int64_t>(n) * k[i];
                sum += t[i] - x[i];
            }
            if (sum == 0) {
                const std::int64_t c = RingCrossings(n, x, t);
                best = best < 0 ? c : std::min(best, c);
            }
            int i = 0;
            while (i < n && k[i] == 1) {
                k[i++] = -1;
            }
            if (i == n) {
                break;
            }
            ++k[i];
        }
        EXPECT_EQ(best, d);
    }
}

// 重放：每层开始时该层的每一对都在环上相邻；付费 SWAP 数 = 总交换数 − 免费翻转数 = cost；
// 免费翻转紧跟上一层的块（交换的正是上一层的一对）。
void CheckReplay(int n, const Layers& layers, const LayerChainOptions& options) {
    const auto res = RouteLayersOnRing(n, layers, options);
    ASSERT_EQ(static_cast<int>(res.initial.size()), n);
    std::vector<int> arr = res.initial;
    std::vector<int> where(n);
    std::int64_t paid = 0;
    for (std::size_t l = 0; l < layers.size(); ++l) {
        for (int i = 0; i < n; ++i) {
            where[arr[i]] = i;
        }
        for (auto [a, b] : layers[l]) {
            const int d = std::abs(where[a] - where[b]);
            EXPECT_TRUE(d == 1 || d == n - 1) << "layer " << l;
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
            ASSERT_LT(i, n);
            const int k = (i + 1) % n;
            if (j < flips) {
                EXPECT_TRUE(prev.count({std::min(arr[i], arr[k]), std::max(arr[i], arr[k])}));
            }
            std::swap(arr[i], arr[k]);
        }
        paid += static_cast<std::int64_t>(sw.size()) - flips;
    }
    EXPECT_EQ(paid, res.cost);
}

TEST(RingRouting, ReplayIsLegalAndCostMatches) {
    LayerChainOptions options;
    for (std::uint32_t seed = 0; seed < 6; ++seed) {
        CheckReplay(10, RandomLayers(10, 8, seed), options);
        CheckReplay(9, RandomLayers(9, 5, seed + 100), options);
    }
    options.perturbations = 50;
    CheckReplay(12, RandomLayers(12, 10, 7), options);
    CheckReplay(3, RandomLayers(3, 4, 1), options);
}

TEST(RingRouting, RepeatedLayerIsFree) {
    Layers layers = {{{0, 5}, {1, 3}, {2, 4}}, {{0, 5}, {1, 3}, {2, 4}}};
    EXPECT_EQ(RouteLayersOnRing(6, layers, LayerChainOptions{}).cost, 0);
}

TEST(RingRouting, WrapAroundIsUsed) {
    // n = 4：{(0,1),(2,3)} → {(1,2),(3,0)}：只需旋转（环上 0 个付费 SWAP，路径上不行）
    Layers layers = {{{0, 1}, {2, 3}}, {{1, 2}, {3, 0}}};
    EXPECT_EQ(RouteLayersOnRing(4, layers, LayerChainOptions{}).cost, 0);
}

TEST(RingRouting, PerturbationsNeverIncreaseCost) {
    const auto layers = RandomLayers(12, 10, 3);
    LayerChainOptions plain;
    LayerChainOptions ils;
    ils.perturbations = 200;
    EXPECT_LE(RouteLayersOnRing(12, layers, ils).cost, RouteLayersOnRing(12, layers, plain).cost);
}

TEST(RingRouting, RejectsBadInput) {
    Layers bad = {{{0, 1}, {1, 2}}};
    EXPECT_THROW(RouteLayersOnRing(3, bad, LayerChainOptions{}), std::invalid_argument);
    EXPECT_THROW(RouteLayersOnRing(2, {{{0, 1}}}, LayerChainOptions{}), std::invalid_argument);
}

}  // namespace
