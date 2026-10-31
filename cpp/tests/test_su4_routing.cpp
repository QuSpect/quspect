// 一般 SU(4) 块线路的截断 rollout 路由的 gtest。
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/su4_routing.hpp"

namespace {

using quspect::native::transpiler::RouteSu4Blocks;
using quspect::native::transpiler::Su4RouteOptions;

using Adjacency = std::vector<std::vector<int>>;
using Blocks = std::vector<std::pair<int, int>>;

Adjacency Grid(int rows, int cols) {
    Adjacency adj(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int q = r * cols + c;
            if (c + 1 < cols) {
                adj[q].push_back(q + 1);
                adj[q + 1].push_back(q);
            }
            if (r + 1 < rows) {
                adj[q].push_back(q + cols);
                adj[q + cols].push_back(q);
            }
        }
    }
    return adj;
}

// 每层把 n 个比特随机配对（量子体积式）。
Blocks RandomLayers(int n, int num, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Blocks blocks;
    for (int l = 0; l < num; ++l) {
        std::vector<int> p(n);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), rng);
        for (int i = 0; i + 1 < n; i += 2) {
            blocks.emplace_back(p[i], p[i + 1]);
        }
    }
    return blocks;
}

// 重放事件：块按每个比特的顺序执行且执行时相邻；SWAP 在耦合边上；付费数 = SWAP 数 − 翻转数，
// 翻转紧跟在同一对比特的块之后。
void CheckReplay(const Adjacency& adj, const Blocks& blocks, const Su4RouteOptions& options) {
    const int n = static_cast<int>(adj.size());
    std::vector<int> layout(n);
    std::iota(layout.begin(), layout.end(), 0);
    std::mt19937 rng(11);
    std::shuffle(layout.begin(), layout.end(), rng);
    const auto res = RouteSu4Blocks(adj, blocks, layout, options);
    ASSERT_EQ(static_cast<int>(res.layout.size()), n);
    std::vector<int> pos = res.layout;
    std::vector<int> tok(n);
    for (int t = 0; t < n; ++t) {
        tok[pos[t]] = t;
    }
    std::vector<std::vector<int>> queue(n);
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        queue[blocks[b].first].push_back(static_cast<int>(b));
        queue[blocks[b].second].push_back(static_cast<int>(b));
    }
    std::vector<std::size_t> ptr(n, 0);
    std::int64_t swaps = 0;
    std::int64_t flips = 0;
    int last = -1;
    std::size_t executed = 0;
    // 深度代理：块与付费 SWAP 各占一层，免费翻转并入块
    std::vector<std::int64_t> busy(n, 0);
    std::int64_t depth = 0;
    auto occupy = [&](int u, int v) {
        const std::int64_t t = std::max(busy[u], busy[v]) + 1;
        busy[u] = busy[v] = t;
        depth = std::max(depth, t);
    };
    for (std::int64_t e : res.events) {
        if (e >= 0) {
            const auto [a, b] = blocks[e];
            ASSERT_EQ(queue[a][ptr[a]], e);
            ASSERT_EQ(queue[b][ptr[b]], e);
            ++ptr[a];
            ++ptr[b];
            const int pa = pos[a];
            EXPECT_NE(std::find(adj[pa].begin(), adj[pa].end(), pos[b]), adj[pa].end());
            occupy(pa, pos[b]);
            last = static_cast<int>(e);
            ++executed;
            continue;
        }
        const std::int64_t code = -1 - e;
        const int x = static_cast<int>(code / n);
        const int y = static_cast<int>(code % n);
        ASSERT_LT(x, y);
        EXPECT_NE(std::find(adj[x].begin(), adj[x].end(), y), adj[x].end());
        bool flip = false;
        if (last >= 0) {
            const auto [a, b] = blocks[last];
            flip = (pos[a] == x && pos[b] == y) || (pos[a] == y && pos[b] == x);
        }
        if (flip) {
            ++flips;
        } else {
            occupy(x, y);
        }
        last = -1;
        std::swap(tok[x], tok[y]);
        pos[tok[x]] = x;
        pos[tok[y]] = y;
        ++swaps;
    }
    EXPECT_EQ(executed, blocks.size());
    EXPECT_EQ(flips, res.flips);
    EXPECT_EQ(swaps - flips, res.paid);
    EXPECT_EQ(depth, res.depth);
}

TEST(Su4Routing, ReplayIsLegalAndCountsMatch) {
    Su4RouteOptions options;
    CheckReplay(Grid(3, 3), RandomLayers(9, 6, 1), options);
    CheckReplay(Grid(4, 4), RandomLayers(16, 8, 2), options);
    options.width = 1;
    options.layout_iters = 0;
    CheckReplay(Grid(2, 5), RandomLayers(10, 7, 3), options);
    // 物理比特多于逻辑比特（空闲比特也参与交换）
    options.width = 4;
    options.layout_iters = 2;
    CheckReplay(Grid(3, 4), RandomLayers(9, 6, 4), options);
}

TEST(Su4Routing, AlreadyAdjacentNeedsNoSwap) {
    // 每个块都在一条边上，布局恒等：0 个付费 SWAP
    const Blocks blocks = {{0, 1}, {2, 3}, {0, 1}};
    std::vector<int> layout = {0, 1, 2, 3};
    const auto res = RouteSu4Blocks(Grid(2, 2), blocks, layout, Su4RouteOptions{});
    EXPECT_EQ(res.paid, 0);
}

TEST(Su4Routing, RolloutHelpsOnAverage) {
    // 大网格上截断 rollout 比单步贪心少付费 SWAP（小规模上两者相当）
    const auto adj = Grid(10, 10);
    std::int64_t greedy = 0;
    std::int64_t rollout = 0;
    for (std::uint32_t seed = 0; seed < 2; ++seed) {
        const auto blocks = RandomLayers(100, 20, seed + 20);
        std::vector<int> layout(100);
        std::iota(layout.begin(), layout.end(), 0);
        Su4RouteOptions g;
        g.width = 1;
        Su4RouteOptions r;
        r.width = 16;
        greedy += RouteSu4Blocks(adj, blocks, layout, g).paid;
        rollout += RouteSu4Blocks(adj, blocks, layout, r).paid;
    }
    EXPECT_LT(rollout, greedy);
}

TEST(Su4Routing, EarliestFreeTieBreakKeepsRoutingShallow) {
    const auto adj = Grid(8, 8);
    std::int64_t plain = 0;
    std::int64_t shallow = 0;
    for (std::uint32_t seed = 0; seed < 2; ++seed) {
        const auto blocks = RandomLayers(64, 16, seed + 40);
        std::vector<int> layout(64);
        std::iota(layout.begin(), layout.end(), 0);
        Su4RouteOptions a;
        a.depth_tiebreak = false;
        Su4RouteOptions b;
        plain += RouteSu4Blocks(adj, blocks, layout, a).depth;
        shallow += RouteSu4Blocks(adj, blocks, layout, b).depth;
    }
    EXPECT_LT(shallow, plain);
}

TEST(Su4Routing, RejectsBadInput) {
    const auto adj = Grid(2, 2);
    EXPECT_THROW(RouteSu4Blocks(adj, {{0, 1}}, {0, 1, 2}, Su4RouteOptions{}), std::invalid_argument);
    EXPECT_THROW(RouteSu4Blocks(adj, {{0, 1}}, {0, 0, 1, 2}, Su4RouteOptions{}), std::invalid_argument);
    EXPECT_THROW(RouteSu4Blocks(adj, {{0, 0}}, {0, 1, 2, 3}, Su4RouteOptions{}), std::invalid_argument);
    const Adjacency split = {{1}, {0}, {3}, {2}};
    EXPECT_THROW(RouteSu4Blocks(split, {{0, 2}}, {0, 1, 2, 3}, Su4RouteOptions{}), std::invalid_argument);
}

}  // namespace
