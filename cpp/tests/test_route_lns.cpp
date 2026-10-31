// 路由精确搜索（A*）与大邻域搜索（LNS）的 gtest。
#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/route_lns.hpp"
#include "quspect_native/transpiler/sabre_routing.hpp"

namespace {

using quspect::native::transpiler::CPythonRandom;
using quspect::native::transpiler::ExactRoute;
using quspect::native::transpiler::ExactSearchOptions;
using quspect::native::transpiler::RefineOptions;
using quspect::native::transpiler::RefineRouting;
using quspect::native::transpiler::RouteCostModel;
using quspect::native::transpiler::RouteDag;
using quspect::native::transpiler::RouteOptions;
using quspect::native::transpiler::RouteSwapCost;
using quspect::native::transpiler::RoutingDag;
using quspect::native::transpiler::RoutingGraph;

constexpr std::int64_t kInf = 1000000000;

std::vector<std::vector<std::int64_t>> BfsDistances(const std::vector<std::vector<int>>& adjacency) {
    const int n = static_cast<int>(adjacency.size());
    std::vector<std::vector<std::int64_t>> dist(n, std::vector<std::int64_t>(n, kInf));
    for (int s = 0; s < n; ++s) {
        dist[s][s] = 0;
        std::deque<int> queue = {s};
        while (!queue.empty()) {
            const int node = queue.front();
            queue.pop_front();
            for (int z : adjacency[node]) {
                if (dist[s][z] == kInf) {
                    dist[s][z] = dist[s][node] + 1;
                    queue.push_back(z);
                }
            }
        }
    }
    return dist;
}

std::vector<std::vector<int>> Adjacency(int n, const std::vector<std::pair<int, int>>& edges) {
    std::vector<std::vector<int>> adj(n);
    for (auto [a, b] : edges) {
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    return adj;
}

RoutingGraph MakeGraph(int n, const std::vector<std::pair<int, int>>& edges) {
    auto adj = Adjacency(n, edges);
    auto dist = BfsDistances(adj);
    return RoutingGraph(std::move(adj), dist);
}

RoutingGraph Line(int n) {
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i + 1 < n; ++i) {
        edges.emplace_back(i, i + 1);
    }
    return MakeGraph(n, edges);
}

// 两比特门序列的依赖 DAG（每个比特上前后相继）。
RoutingDag SequentialDag(const std::vector<std::pair<int, int>>& gates, int num_qubits) {
    std::vector<int> nq1;
    std::vector<int> nq2;
    std::vector<int> nq3;
    std::vector<std::vector<int>> succ(gates.size());
    std::vector<std::vector<int>> pred(gates.size());
    std::vector<int> last(num_qubits, -1);
    for (std::size_t i = 0; i < gates.size(); ++i) {
        auto [a, b] = gates[i];
        nq1.push_back(a);
        nq2.push_back(b);
        nq3.push_back(-1);
        std::set<int> ps;
        for (int q : {a, b}) {
            if (last[q] >= 0) {
                ps.insert(last[q]);
            }
            last[q] = static_cast<int>(i);
        }
        for (int p : ps) {
            pred[i].push_back(p);
            succ[p].push_back(static_cast<int>(i));
        }
    }
    return RoutingDag(nq1, nq2, nq3, succ, pred, false);
}

std::int64_t Swap(int x, int y, int n) {
    if (x > y) {
        std::swap(x, y);
    }
    return -1 - (static_cast<std::int64_t>(x) * n + y);
}

// 参照：纯 SWAP 数下的广度优先搜索（状态 = 布局 + 已执行节点集；可执行节点立即执行）。
int BruteForceMinSwaps(const RoutingGraph& graph, const RoutingDag& dag, const std::vector<int>& layout) {
    const int n = graph.num_physical();
    const int m = dag.num_nodes();
    auto drain = [&](std::vector<int>& pi, std::vector<char>& done) {
        bool progress = true;
        while (progress) {
            progress = false;
            for (int i = 0; i < m; ++i) {
                if (done[i]) {
                    continue;
                }
                bool ready = true;
                for (int p : dag.pred()[i]) {
                    ready = ready && done[p];
                }
                if (ready && graph.Adjacent(pi[dag.nq1()[i]], pi[dag.nq2()[i]])) {
                    done[i] = 1;
                    progress = true;
                }
            }
        }
    };
    using State = std::pair<std::vector<int>, std::vector<char>>;
    std::map<State, int> seen;
    std::deque<State> queue;
    State root{layout, std::vector<char>(m, 0)};
    drain(root.first, root.second);
    seen[root] = 0;
    queue.push_back(root);
    while (!queue.empty()) {
        State s = queue.front();
        queue.pop_front();
        const int d = seen[s];
        if (std::all_of(s.second.begin(), s.second.end(), [](char c) { return c != 0; })) {
            return d;
        }
        for (int e = 0; e < graph.num_edges(); ++e) {
            State t = s;
            const int x = graph.EdgeX(e);
            const int y = graph.EdgeY(e);
            for (int& p : t.first) {
                if (p == x) {
                    p = y;
                } else if (p == y) {
                    p = x;
                }
            }
            drain(t.first, t.second);
            if (seen.emplace(t, d + 1).second) {
                queue.push_back(t);
            }
        }
    }
    return -1;
}

std::vector<int> Identity(int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }
    return v;
}

std::vector<std::int64_t> SabreEvents(const RoutingGraph& graph, const RoutingDag& dag, const std::vector<int>& layout,
                                      std::uint32_t seed) {
    RouteOptions options;
    CPythonRandom rng(seed);
    return RouteDag(graph, dag, options, layout, true, true, rng).events;
}

TEST(RouteSwapCost, PlainSwapsCostThree) {
    const RoutingGraph graph = Line(3);
    const RoutingDag dag = SequentialDag({{0, 2}}, 3);
    const RouteCostModel model;
    const std::vector<std::int64_t> events = {Swap(0, 1, 3), 0};
    EXPECT_EQ(RouteSwapCost(graph, dag, model, Identity(3), events), 3);
}

TEST(RouteSwapCost, SwapAfterBlockOnSamePairUsesMirrorCost) {
    // 节点 0 在 (0, 1) 上执行后紧跟同对 SWAP：只计镜像代价 1；节点 1 随后在 (1, 2) 上执行。
    const RoutingGraph graph = Line(3);
    const RoutingDag dag = SequentialDag({{0, 1}, {0, 2}}, 3);
    RouteCostModel model;
    model.mirror_cost = {1, 1};
    const std::vector<std::int64_t> events = {0, Swap(0, 1, 3), 1};
    EXPECT_EQ(RouteSwapCost(graph, dag, model, Identity(3), events), 1);
    EXPECT_EQ(RouteSwapCost(graph, dag, model, Identity(3), events, false), 3);
}

TEST(RouteSwapCost, SwapBeforeBlockOnSamePairIsRefunded) {
    const RoutingGraph graph = Line(3);
    const RoutingDag dag = SequentialDag({{0, 1}}, 3);
    RouteCostModel model;
    model.mirror_cost = {0};
    const std::vector<std::int64_t> events = {Swap(0, 1, 3), 0};
    EXPECT_EQ(RouteSwapCost(graph, dag, model, Identity(3), events), 0);
}

TEST(RouteSwapCost, RejectsIllegalSequences) {
    const RoutingGraph graph = Line(3);
    const RoutingDag dag = SequentialDag({{0, 2}}, 3);
    const RouteCostModel model;
    EXPECT_THROW(RouteSwapCost(graph, dag, model, Identity(3), {0}), std::invalid_argument);
    EXPECT_THROW(RouteSwapCost(graph, dag, model, Identity(3), {Swap(0, 2, 3), 0}), std::invalid_argument);
}

TEST(ExactRoute, MatchesBruteForceSwapCount) {
    std::mt19937 gen(7);
    const std::vector<std::vector<std::pair<int, int>>> topologies = {
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}},
        {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}},
        {{0, 1}, {1, 2}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}},
    };
    for (const auto& edges : topologies) {
        int n = 0;
        for (auto [a, b] : edges) {
            n = std::max({n, a + 1, b + 1});
        }
        const RoutingGraph graph = MakeGraph(n, edges);
        for (int trial = 0; trial < 12; ++trial) {
            std::vector<std::pair<int, int>> gates;
            std::uniform_int_distribution<int> pick(0, n - 1);
            for (int g = 0; g < 7; ++g) {
                int a = pick(gen);
                int b = pick(gen);
                while (b == a) {
                    b = pick(gen);
                }
                gates.emplace_back(a, b);
            }
            const RoutingDag dag = SequentialDag(gates, n);
            std::vector<int> layout = Identity(n);
            std::shuffle(layout.begin(), layout.end(), gen);
            const RouteCostModel model;
            ExactSearchOptions options;
            options.merge_costs = false;
            const auto result = ExactRoute(graph, dag, model, layout, options);
            ASSERT_TRUE(result.found);
            ASSERT_TRUE(result.optimal);
            EXPECT_EQ(result.num_swaps, BruteForceMinSwaps(graph, dag, layout));
            EXPECT_EQ(result.cost, 3 * result.num_swaps);
            EXPECT_EQ(RouteSwapCost(graph, dag, model, layout, result.events, false), result.cost);
        }
    }
}

TEST(ExactRoute, MergeCostIsConsistentAndNotWorseThanSabre) {
    std::mt19937 gen(11);
    const RoutingGraph graph = Line(5);
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<std::pair<int, int>> gates;
        std::uniform_int_distribution<int> pick(0, 4);
        for (int g = 0; g < 8; ++g) {
            int a = pick(gen);
            int b = pick(gen);
            while (b == a) {
                b = pick(gen);
            }
            gates.emplace_back(a, b);
        }
        const RoutingDag dag = SequentialDag(gates, 5);
        RouteCostModel model;
        model.mirror_cost.assign(gates.size(), 0);
        for (std::size_t i = 0; i < gates.size(); ++i) {
            model.mirror_cost[i] = static_cast<int>(i % 3);
        }
        const std::vector<int> layout = Identity(5);
        const auto sabre = SabreEvents(graph, dag, layout, 3 + trial);
        const std::int64_t sabre_cost = RouteSwapCost(graph, dag, model, layout, sabre);
        const auto result = ExactRoute(graph, dag, model, layout, ExactSearchOptions{});
        ASSERT_TRUE(result.found);
        EXPECT_LE(result.cost, sabre_cost);
        EXPECT_EQ(RouteSwapCost(graph, dag, model, layout, result.events), result.cost);
    }
}

TEST(ExactRoute, UpperBoundPrunesEverything) {
    const RoutingGraph graph = Line(3);
    const RoutingDag dag = SequentialDag({{0, 2}}, 3);
    ExactSearchOptions options;
    options.merge_costs = false;
    const auto result = ExactRoute(graph, dag, RouteCostModel{}, Identity(3), options, 3);
    EXPECT_FALSE(result.found);
}

TEST(RefineRouting, RemovesWastefulSwaps) {
    // 正确但浪费的路由：先在 (2, 3) 上来回交换一次，再用一个 SWAP 让两个门都可执行。
    const int n = 4;
    const RoutingGraph graph = Line(n);
    const RoutingDag dag = SequentialDag({{0, 2}, {1, 3}}, n);
    const RouteCostModel model;
    const std::vector<std::int64_t> events = {Swap(2, 3, n), Swap(2, 3, n), Swap(1, 2, n), 0, 1};
    RefineOptions options;
    options.merge_costs = false;
    const auto res = RefineRouting(graph, dag, model, Identity(n), events, options);
    EXPECT_EQ(res.old_cost, 9);
    EXPECT_EQ(res.new_cost, 3);
    EXPECT_EQ(RouteSwapCost(graph, dag, model, Identity(n), res.events, false), res.new_cost);
}

TEST(RefineRouting, KeepsLegalityAndNodeSetOnRandomCircuits) {
    std::mt19937 gen(5);
    const std::vector<std::pair<int, int>> grid = {{0, 1}, {1, 2}, {3, 4}, {4, 5}, {6, 7}, {7, 8},
                                                   {0, 3}, {3, 6}, {1, 4}, {4, 7}, {2, 5}, {5, 8}};
    const RoutingGraph graph = MakeGraph(9, grid);
    for (int trial = 0; trial < 8; ++trial) {
        std::vector<std::pair<int, int>> gates;
        std::uniform_int_distribution<int> pick(0, 8);
        for (int g = 0; g < 40; ++g) {
            int a = pick(gen);
            int b = pick(gen);
            while (b == a) {
                b = pick(gen);
            }
            gates.emplace_back(a, b);
        }
        const RoutingDag dag = SequentialDag(gates, 9);
        RouteCostModel model;
        for (std::size_t i = 0; i < gates.size(); ++i) {
            model.mirror_cost.push_back(static_cast<int>(gen() % 4) - 1);
        }
        const std::vector<int> layout = Identity(9);
        const auto events = SabreEvents(graph, dag, layout, 17 + trial);
        const auto res = RefineRouting(graph, dag, model, layout, events, RefineOptions{});
        EXPECT_LE(res.new_cost, res.old_cost);
        EXPECT_EQ(RouteSwapCost(graph, dag, model, layout, res.events), res.new_cost);
        std::vector<std::int64_t> a;
        std::vector<std::int64_t> b;
        for (auto e : events) {
            if (e >= 0) {
                a.push_back(e);
            }
        }
        for (auto e : res.events) {
            if (e >= 0) {
                b.push_back(e);
            }
        }
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        EXPECT_EQ(a, b);
    }
}

TEST(RefineRouting, DepthSweepsAreLexicographicInCostAndDepth) {
    std::mt19937 gen(11);
    const RoutingGraph graph = Line(7);
    for (int trial = 0; trial < 6; ++trial) {
        std::vector<std::pair<int, int>> gates;
        std::uniform_int_distribution<int> pick(0, 6);
        for (int g = 0; g < 30; ++g) {
            int a = pick(gen);
            int b = pick(gen);
            while (b == a) {
                b = pick(gen);
            }
            gates.emplace_back(a, b);
        }
        const RoutingDag dag = SequentialDag(gates, 7);
        RouteCostModel model;
        for (std::size_t i = 0; i < gates.size(); ++i) {
            model.mirror_cost.push_back(static_cast<int>(gen() % 3));
            model.node_weight.push_back(1 + static_cast<int>(gen() % 3));
        }
        const std::vector<int> layout = Identity(7);
        const auto events = SabreEvents(graph, dag, layout, 31 + trial);
        RefineOptions depth_only;
        depth_only.passes = 0;
        depth_only.depth_passes = 2;
        depth_only.depth_expansions = 200000;
        const auto res = RefineRouting(graph, dag, model, layout, events, depth_only);
        EXPECT_LE(res.new_cost, res.old_cost);
        if (res.new_cost == res.old_cost) {
            EXPECT_LE(res.new_depth, res.old_depth);
        }
        EXPECT_EQ(RouteSwapCost(graph, dag, model, layout, res.events), res.new_cost);
        RefineOptions lex;
        lex.depth_ties = true;
        const auto res2 = RefineRouting(graph, dag, model, layout, events, lex);
        EXPECT_LE(res2.new_cost, res2.old_cost);
        EXPECT_EQ(RouteSwapCost(graph, dag, model, layout, res2.events), res2.new_cost);
        EXPECT_GT(res2.old_depth, 0);
    }
}

TEST(RefineRouting, DepthSweepPrefersParallelSwaps) {
    // 两个门 (0, 2)、(3, 5) 在 line 6 上各需一个 SWAP；旧路由把两个 SWAP 都放在同一侧比特上串行
    // （(1, 2) 再 (2, 3) 依赖链），深度扫描应找到代价相同、深度不增的路由。
    const int n = 6;
    const RoutingGraph graph = Line(n);
    const RoutingDag dag = SequentialDag({{0, 2}, {3, 5}}, n);
    RouteCostModel model;
    model.node_weight = {1, 1};
    const std::vector<std::int64_t> events = {Swap(1, 2, n), 0, Swap(3, 4, n), 1};
    RefineOptions options;
    options.passes = 0;
    options.depth_passes = 1;
    options.depth_expansions = 10000;
    const auto res = RefineRouting(graph, dag, model, Identity(n), events, options);
    EXPECT_EQ(res.new_cost, res.old_cost);
    EXPECT_LE(res.new_depth, res.old_depth);
    EXPECT_EQ(res.old_depth, 4);
}

}  // namespace
