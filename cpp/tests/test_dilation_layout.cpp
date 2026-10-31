// 低伸缩初始布局的 gtest：链、星形与不可嵌入的情形；期望值与 Python 参考实现
// （quspect.transpiler._embedding.dilation_layouts_python，同一种子）一致。
#include <cstdint>
#include <deque>
#include <set>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/dilation_layout.hpp"

namespace {

using quspect::native::transpiler::DilationLayout;
using quspect::native::transpiler::DilationLayouts;
using quspect::native::transpiler::WeightedEdge;

std::vector<std::vector<std::int64_t>> BfsDistances(const std::vector<std::vector<int>>& adjacency) {
    const int n = static_cast<int>(adjacency.size());
    std::vector<std::vector<std::int64_t>> dist(n, std::vector<std::int64_t>(n, -1));
    for (int s = 0; s < n; ++s) {
        dist[s][s] = 0;
        std::deque<int> queue = {s};
        while (!queue.empty()) {
            const int node = queue.front();
            queue.pop_front();
            for (int z : adjacency[node]) {
                if (dist[s][z] < 0) {
                    dist[s][z] = dist[s][node] + 1;
                    queue.push_back(z);
                }
            }
        }
    }
    return dist;
}

std::vector<std::vector<int>> Grid(int rows, int cols) {
    std::vector<std::vector<int>> adj(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int q = r * cols + c;
            if (r > 0)
                adj[q].push_back(q - cols);
            if (c > 0)
                adj[q].push_back(q - 1);
            if (c + 1 < cols)
                adj[q].push_back(q + 1);
            if (r + 1 < rows)
                adj[q].push_back(q + cols);
        }
    }
    return adj;
}

std::vector<WeightedEdge> Chain(int n) {
    std::vector<WeightedEdge> edges;
    for (int i = 0; i + 1 < n; ++i) {
        edges.push_back({i, i + 1, 3});
    }
    return edges;
}

std::int64_t Dilation(const std::vector<WeightedEdge>& edges, const std::vector<std::vector<std::int64_t>>& dist,
                      const std::vector<int>& layout) {
    std::int64_t total = 0;
    for (const auto& e : edges) {
        total += e.weight * (dist[layout[e.a]][layout[e.b]] - 1);
    }
    return total;
}

TEST(DilationLayoutTest, ChainEmbedsInGridWithoutDilation) {
    const auto adj = Grid(4, 5);
    const auto dist = BfsDistances(adj);
    const auto edges = Chain(20);  // 哈密顿路径
    const auto found = DilationLayouts(20, edges, adj, dist, 7, 2);
    ASSERT_FALSE(found.empty());
    EXPECT_EQ(found[0].cost, 0);
    EXPECT_EQ(Dilation(edges, dist, found[0].layout), 0);
    const std::set<int> used(found[0].layout.begin(), found[0].layout.end());
    EXPECT_EQ(used.size(), 20u);
}

TEST(DilationLayoutTest, CostMatchesLayoutAndIsSorted) {
    const auto adj = Grid(3, 3);
    const auto dist = BfsDistances(adj);
    // K4 加一条悬挂边：网格里不可无伸缩嵌入
    const std::vector<WeightedEdge> edges = {{0, 1, 2}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1},
                                             {1, 3, 1}, {2, 3, 4}, {3, 5, 1}};
    const auto found = DilationLayouts(6, edges, adj, dist, 1, 3);
    ASSERT_FALSE(found.empty());
    for (std::size_t i = 0; i < found.size(); ++i) {
        EXPECT_EQ(found[i].cost, Dilation(edges, dist, found[i].layout));
        EXPECT_EQ(found[i].layout[4], -1);  // 无相互作用的逻辑比特不放置
        if (i > 0) {
            EXPECT_LE(found[i - 1].cost, found[i].cost);
            EXPECT_NE(found[i - 1].layout, found[i].layout);
        }
    }
    EXPECT_GT(found[0].cost, 0);
}

TEST(DilationLayoutTest, StarCentreOnHighDegreeQubit) {
    const auto adj = Grid(3, 3);
    const auto dist = BfsDistances(adj);
    const std::vector<WeightedEdge> edges = {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {0, 4, 1}};
    const auto found = DilationLayouts(5, edges, adj, dist, 0, 1);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].cost, 0);
    EXPECT_EQ(found[0].layout[0], 4);  // 网格中心
}

TEST(DilationLayoutTest, DeterministicForSeed) {
    const auto adj = Grid(4, 4);
    const auto dist = BfsDistances(adj);
    std::vector<WeightedEdge> edges;
    for (int i = 0; i < 12; ++i) {
        for (int j = i + 1; j < 12; j += 3) {
            edges.push_back({i, j, 1 + (i + j) % 3});
        }
    }
    const auto a = DilationLayouts(12, edges, adj, dist, 42, 2);
    const auto b = DilationLayouts(12, edges, adj, dist, 42, 2);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].cost, b[i].cost);
        EXPECT_EQ(a[i].layout, b[i].layout);
    }
}

TEST(DilationLayoutTest, RejectsInvalidInput) {
    const auto adj = Grid(2, 2);
    const auto dist = BfsDistances(adj);
    EXPECT_THROW(DilationLayouts(3, {{1, 0, 1}}, adj, dist, 0, 1), std::invalid_argument);
    EXPECT_THROW(DilationLayouts(3, {{0, 1, 0}}, adj, dist, 0, 1), std::invalid_argument);
    EXPECT_THROW(DilationLayouts(5, {{0, 1, 1}}, adj, dist, 0, 1), std::invalid_argument);
    EXPECT_TRUE(DilationLayouts(3, {}, adj, dist, 0, 1).empty());
}

}  // namespace
