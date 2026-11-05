// 全局束搜索路由的 gtest。
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/beam_routing.hpp"

namespace {

using quspect::native::transpiler::BeamRouteOptions;
using quspect::native::transpiler::BeamRouteResult;
using quspect::native::transpiler::RouteBeam;

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

Adjacency Line(int n) {
    Adjacency adj(n);
    for (int q = 0; q + 1 < n; ++q) {
        adj[q].push_back(q + 1);
        adj[q + 1].push_back(q);
    }
    return adj;
}

Blocks RandomBlocks(int n_logical, int count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Blocks blocks;
    while (static_cast<int>(blocks.size()) < count) {
        const int a = static_cast<int>(rng() % n_logical);
        const int b = static_cast<int>(rng() % n_logical);
        if (a != b) {
            blocks.emplace_back(a, b);
        }
    }
    return blocks;
}

std::vector<int> Identity(int n) {
    std::vector<int> layout(n);
    std::iota(layout.begin(), layout.end(), 0);
    return layout;
}

// 重放事件：每个块恰好执行一次、按各比特上的顺序、执行时两个比特相邻；SWAP 都在耦合边上。
// 返回 SWAP 数（不合法时 -1）。
std::int64_t Replay(const Adjacency& adj, const Blocks& blocks, const BeamRouteResult& r) {
    const int n = static_cast<int>(adj.size());
    std::vector<int> l2p = r.layout;
    std::vector<int> p2l(n);
    for (int q = 0; q < n; ++q) {
        p2l[l2p[q]] = q;
    }
    auto adjacent = [&](int x, int y) {
        for (int v : adj[x]) {
            if (v == y) {
                return true;
            }
        }
        return false;
    };
    std::vector<int> next_on(n, 0);  // 每个比特上已执行的块数
    std::vector<std::vector<int>> chain(n);
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        chain[blocks[b].first].push_back(static_cast<int>(b));
        chain[blocks[b].second].push_back(static_cast<int>(b));
    }
    std::vector<char> done(blocks.size(), 0);
    std::int64_t swaps = 0;
    for (std::int64_t e : r.events) {
        if (e < 0) {
            const std::int64_t code = -1 - e;
            const int x = static_cast<int>(code / n);
            const int y = static_cast<int>(code % n);
            if (x >= y || !adjacent(x, y)) {
                return -1;
            }
            std::swap(p2l[x], p2l[y]);
            l2p[p2l[x]] = x;
            l2p[p2l[y]] = y;
            ++swaps;
            continue;
        }
        const auto [a, b] = blocks[e];
        if (done[e] || chain[a][next_on[a]] != e || chain[b][next_on[b]] != e || !adjacent(l2p[a], l2p[b])) {
            return -1;
        }
        done[e] = 1;
        ++next_on[a];
        ++next_on[b];
    }
    for (char d : done) {
        if (!d) {
            return -1;
        }
    }
    return swaps;
}

TEST(BeamRouting, AdjacentBlocksNeedNoSwap) {
    const Adjacency adj = Line(4);
    const Blocks blocks = {{0, 1}, {1, 2}, {2, 3}, {0, 1}};
    const BeamRouteResult r = RouteBeam(adj, blocks, Identity(4), BeamRouteOptions{});
    EXPECT_EQ(r.swaps, 0);
    EXPECT_EQ(Replay(adj, blocks, r), 0);
}

TEST(BeamRouting, OneSwapOnALine) {
    const Adjacency adj = Line(3);
    const Blocks blocks = {{0, 2}};
    BeamRouteOptions opt;
    opt.layout_iters = 0;
    const BeamRouteResult r = RouteBeam(adj, blocks, Identity(3), opt);
    EXPECT_EQ(r.swaps, 1);
    EXPECT_EQ(Replay(adj, blocks, r), 1);
}

TEST(BeamRouting, RandomCircuitsAreValidAndDeterministic) {
    const Adjacency adj = Grid(4, 5);
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        const Blocks blocks = RandomBlocks(16, 200, seed);
        BeamRouteOptions opt;
        opt.beam = 16;
        opt.seed = seed;
        const BeamRouteResult r = RouteBeam(adj, blocks, Identity(20), opt);
        EXPECT_GT(r.swaps, 0);
        EXPECT_EQ(Replay(adj, blocks, r), r.swaps);
        const BeamRouteResult again = RouteBeam(adj, blocks, Identity(20), opt);
        EXPECT_EQ(again.swaps, r.swaps);
        EXPECT_EQ(again.layout, r.layout);
        EXPECT_EQ(again.events, r.events);
    }
}

TEST(BeamRouting, WiderBeamIsNotWorseOnAverage) {
    const Adjacency adj = Grid(4, 5);
    std::int64_t narrow = 0;
    std::int64_t wide = 0;
    for (std::uint64_t seed = 11; seed <= 16; ++seed) {
        const Blocks blocks = RandomBlocks(20, 300, seed);
        BeamRouteOptions opt;
        opt.beam = 1;
        narrow += RouteBeam(adj, blocks, Identity(20), opt).swaps;
        opt.beam = 32;
        wide += RouteBeam(adj, blocks, Identity(20), opt).swaps;
    }
    EXPECT_LT(wide, narrow);
}

TEST(BeamRouting, RejectsInvalidInput) {
    const Adjacency adj = Line(3);
    EXPECT_THROW(RouteBeam(adj, {{0, 0}}, Identity(3), BeamRouteOptions{}), std::invalid_argument);
    EXPECT_THROW(RouteBeam(adj, {{0, 1}}, {0, 0, 1}, BeamRouteOptions{}), std::invalid_argument);
    EXPECT_THROW(RouteBeam(adj, {{0, 1}}, {0, 1}, BeamRouteOptions{}), std::invalid_argument);
    Adjacency split = {{1}, {0}, {}};
    EXPECT_THROW(RouteBeam(split, {{0, 1}}, Identity(3), BeamRouteOptions{}), std::invalid_argument);
    BeamRouteOptions bad;
    bad.beam = 0;
    EXPECT_THROW(RouteBeam(adj, {{0, 1}}, Identity(3), bad), std::invalid_argument);
}

}  // namespace
