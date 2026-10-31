// SABRE 路由核心的 gtest：小规模手工 DAG，期望输出由 Python 参考实现
// （quspect.transpiler._routing._route_dag_python，同一随机种子）生成。
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/cpython_random.hpp"
#include "quspect_native/transpiler/sabre_routing.hpp"

namespace {

using quspect::native::transpiler::CPythonRandom;
using quspect::native::transpiler::RouteDag;
using quspect::native::transpiler::RouteOptions;
using quspect::native::transpiler::RouteResult;
using quspect::native::transpiler::RoutingDag;
using quspect::native::transpiler::RoutingGraph;

constexpr std::int64_t kInf = 1000000000;

// 与 Python _bfs_distances 相同：不连通为 10**9。
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

struct Case {
    std::vector<std::vector<int>> adjacency;
    std::vector<int> nq1;
    std::vector<int> nq2;
    std::vector<int> nq3;
    std::vector<std::vector<int>> succ;
    std::vector<std::vector<int>> pred;
    bool has_three_qubit = false;
    RouteOptions options;
    std::vector<int> layout;
    bool forward = true;
    std::uint64_t seed = 0;
    std::vector<std::int64_t> expected_events;
    std::vector<int> expected_final_pi;
    std::int64_t expected_num_swaps = 0;
    int expected_depth = 0;
    // 路由后调用方随机源的下一个 getrandbits(32)（验证随机源状态的推进）。
    std::uint32_t expected_next_bits = 0;
};

void RunCase(const Case& c) {
    RoutingGraph graph(c.adjacency, BfsDistances(c.adjacency));
    RoutingDag dag(c.nq1, c.nq2, c.nq3, c.succ, c.pred, c.has_three_qubit);
    CPythonRandom rng(c.seed);
    RouteResult result = RouteDag(graph, dag, c.options, c.layout, c.forward, /*record=*/true, rng);
    EXPECT_EQ(result.events, c.expected_events);
    EXPECT_EQ(result.final_pi, c.expected_final_pi);
    EXPECT_EQ(result.num_swaps, c.expected_num_swaps);
    EXPECT_EQ(result.depth, c.expected_depth);
    EXPECT_EQ(rng.GetRandBits(32), c.expected_next_bits);

    // 不记录事件时 SWAP 数与最终排列不变。
    CPythonRandom rng2(c.seed);
    RouteResult quiet = RouteDag(graph, dag, c.options, c.layout, c.forward, /*record=*/false, rng2);
    EXPECT_TRUE(quiet.events.empty());
    EXPECT_EQ(quiet.final_pi, c.expected_final_pi);
    EXPECT_EQ(quiet.num_swaps, c.expected_num_swaps);
    EXPECT_EQ(quiet.depth, 0);
}

TEST(SabreRoutingTest, Cost3MatchesTreeSize) {
    std::vector<std::vector<int>> line = {{1}, {0, 2}, {1, 3}, {2}};
    RoutingGraph graph(line, BfsDistances(line));
    EXPECT_EQ(graph.Cost3(0, 2, 1), 1);
    EXPECT_EQ(graph.Cost3(0, 3, 1), 2);
    EXPECT_TRUE(graph.Adjacent(1, 2));
    EXPECT_FALSE(graph.Adjacent(0, 2));
    // incident_edges：物理比特 1 的边按邻居顺序为 (0,1)、(1,2)
    ASSERT_EQ(graph.EdgesAt(1).size(), 2U);
    EXPECT_EQ(graph.EdgeX(graph.EdgesAt(1)[0]), 0);
    EXPECT_EQ(graph.EdgeY(graph.EdgesAt(1)[1]), 2);
}

TEST(SabreRoutingTest, DisconnectedTopologyThrows) {
    std::vector<std::vector<int>> two_pairs = {{1}, {0}, {3}, {2}};
    RoutingGraph graph(two_pairs, BfsDistances(two_pairs));
    RoutingDag dag({0}, {1}, {-1}, {{}}, {{}}, false);
    CPythonRandom rng(0);
    // 逻辑比特 0、1 分别放在两个连通分量上
    EXPECT_THROW(RouteDag(graph, dag, RouteOptions{}, {0, 2, 1, 3}, true, true, rng), std::invalid_argument);
}

TEST(SabreRoutingTest, RejectsInvalidLayout) {
    std::vector<std::vector<int>> line = {{1}, {0, 2}, {1}};
    RoutingGraph graph(line, BfsDistances(line));
    RoutingDag dag({0}, {2}, {-1}, {{}}, {{}}, false);
    CPythonRandom rng(0);
    EXPECT_THROW(RouteDag(graph, dag, RouteOptions{}, {0, 1}, true, true, rng), std::invalid_argument);
    EXPECT_THROW(RouteDag(graph, dag, RouteOptions{}, {0, 0, 1}, true, true, rng), std::invalid_argument);
    EXPECT_THROW(RoutingDag({0}, {1}, {-1}, {{3}}, {{}}, false), std::invalid_argument);
}

TEST(SabreRoutingTest, EmptyDagIsNoop) {
    std::vector<std::vector<int>> line = {{1}, {0, 2}, {1}};
    RoutingGraph graph(line, BfsDistances(line));
    RoutingDag dag({}, {}, {}, {}, {}, false);
    CPythonRandom rng(3);
    RouteResult result = RouteDag(graph, dag, RouteOptions{}, {2, 0, 1}, true, true, rng);
    EXPECT_EQ(result.num_swaps, 0);
    EXPECT_EQ(result.final_pi, (std::vector<int>{2, 0, 1}));
    EXPECT_FALSE(rng.used());
}

TEST(SabreRoutingTest, Line3SingleCx) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1}};
    c.nq1 = {0};
    c.nq2 = {2};
    c.nq3 = {-1};
    c.succ = {{}};
    c.pred = {{}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {0, 1, 2};
    c.forward = true;
    c.seed = 0;
    c.expected_events = {-6, 0};
    c.expected_final_pi = {0, 2, 1};
    c.expected_num_swaps = 1;
    c.expected_depth = 2;
    c.expected_next_bits = 3255389356U;
    RunCase(c);
}

TEST(SabreRoutingTest, Line5Random) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
    c.nq1 = {0, 3, 3, 4, 4, 3, 4, 2, 0, 2, 2, 0, 1, 2, 4, 2, 0, 4, 2, 4, 0, 4, 4, 1, 2};
    c.nq2 = {2, 1, 0, 0, 0, 0, 1, 0, 1, 3, 3, 3, 2, 0, 3, 3, 1, 2, 0, 1, 3, 3, 2, 4, 0};
    c.nq3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    c.succ = {{2, 7},   {2, 6},   {3, 5},   {4},      {5, 6},   {7, 9},   {8, 14},  {8, 9},   {11, 12},
              {10},     {11, 12}, {13, 14}, {13, 16}, {15, 16}, {15, 17}, {17, 20}, {18, 19}, {18, 19},
              {20, 22}, {21, 23}, {21, 24}, {22},     {23, 24}, {},       {}};
    c.pred = {{},       {},       {0, 1},   {2},      {3},      {2, 4},   {1, 4},   {0, 5},   {6, 7},
              {5, 7},   {9},      {8, 10},  {8, 10},  {11, 12}, {6, 11},  {13, 14}, {12, 13}, {14, 15},
              {16, 17}, {16, 17}, {15, 18}, {19, 20}, {18, 21}, {19, 22}, {20, 22}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {3, 0, 4, 1, 2};
    c.forward = true;
    c.seed = 11;
    c.expected_events = {1,  0,   -8,  2,  -14, 3,  4,   6,  5,  -20, 7,  9,  10,  -2,  8,  -14, 12, 11, 13,  -14, 16,
                         -2, -20, -14, 14, -20, 15, -14, 17, 19, -20, 18, 20, -20, -14, 21, -14, 22, 23, -14, 24};
    c.expected_final_pi = {4, 0, 3, 2, 1};
    c.expected_num_swaps = 16;
    c.expected_depth = 30;
    c.expected_next_bits = 793110136U;
    RunCase(c);
}

TEST(SabreRoutingTest, Line5Backward) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
    c.nq1 = {0, 2, 1, 4, 0, 3, 2, 4, 4, 3, 4, 2, 4, 1, 4, 4, 1, 1, 1, 2, 0, 1, 4, 0, 2};
    c.nq2 = {4, 4, 3, 2, 4, 1, 1, 1, 2, 1, 1, 3, 3, 2, 3, 0, 0, 0, 0, 4, 4, 4, 0, 2, 4};
    c.nq3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    c.succ = {{1, 4},   {3},      {5},      {4, 6}, {7, 15},  {6, 9}, {7, 8},   {8, 9}, {10, 11},
              {10, 11}, {12, 13}, {12, 13}, {14},   {16, 19}, {15},   {16, 19}, {17},   {18},
              {20, 21}, {20, 23}, {21, 22}, {22},   {23, 24}, {24},   {}};
    c.pred = {{},     {0},      {},       {1},      {0, 3},   {2},      {3, 5},  {4, 6},   {6, 7},
              {5, 7}, {8, 9},   {8, 9},   {10, 11}, {10, 11}, {12},     {4, 14}, {13, 15}, {16},
              {17},   {13, 15}, {18, 19}, {18, 20}, {20, 21}, {19, 22}, {22, 23}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {1, 4, 0, 2, 3};
    c.forward = false;
    c.seed = 12;
    c.expected_events = {-14, -8, 24, -8,  23, 22, -20, 21,  20, -8, 19, 18, 17, 16, 15, -8, -14, 14, 12, -2,
                         13,  10, -8, -14, 11, 8,  -8,  -14, 9,  -8, 7,  -8, 4,  6,  5,  2,  3,   1,  0};
    c.expected_final_pi = {0, 3, 2, 4, 1};
    c.expected_num_swaps = 14;
    c.expected_depth = 31;
    c.expected_next_bits = 2038265566U;
    RunCase(c);
}

TEST(SabreRoutingTest, Grid23Toffoli) {
    Case c;
    c.adjacency = {{1, 3}, {0, 2, 4}, {1, 5}, {4, 0}, {3, 5, 1}, {4, 2}};
    c.nq1 = {4, 3, 2, 4, 1, 5, 0, 3, 1, 5, 3, 2, 1, 5, 0, 2, 3, 5, 2, 0};
    c.nq2 = {1, 4, 4, 3, 4, 0, 2, 5, 2, 3, 4, 0, 2, 1, 5, 0, 0, 4, 4, 4};
    c.nq3 = {2, -1, -1, -1, 3, -1, 3, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, 2, -1, 5};
    c.succ = {{1, 2, 4}, {2, 3},   {3, 6},   {4},  {6, 8, 10}, {6, 7},   {7, 8, 11}, {9},      {9, 12}, {10, 11, 13},
              {16, 17},  {12, 14}, {13, 15}, {14}, {15, 17},   {16, 17}, {19},       {18, 19}, {19},    {}};
    c.pred = {{},     {0},    {0, 1},  {1, 2},  {0, 3},   {},       {2, 4, 5}, {5, 6},       {4, 6}, {7, 8},
              {4, 9}, {6, 9}, {8, 11}, {9, 12}, {11, 13}, {12, 14}, {10, 15},  {10, 14, 15}, {17},   {16, 17, 18}};
    c.has_three_qubit = true;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {5, 0, 3, 1, 4, 2};
    c.forward = true;
    c.seed = 13;
    c.expected_events = {5,   0,  1,  2,  3,  4,   -30, 6,   8,  7,  -23, 9,  11,
                         -11, 12, 10, -9, 13, -23, 14,  -18, 15, 17, 18,  16, 19};
    c.expected_final_pi = {4, 0, 5, 3, 2, 1};
    c.expected_num_swaps = 6;
    c.expected_depth = 20;
    c.expected_next_bits = 1112433019U;
    RunCase(c);
}

TEST(SabreRoutingTest, Grid23CommutationFusion) {
    Case c;
    c.adjacency = {{1, 3}, {0, 2, 4}, {1, 5}, {4, 0}, {3, 5, 1}, {4, 2}};
    c.nq1 = {0, 4, 4, 1, 1, 2, 1, 4, 2, 3, 0, 5, 0, 1, 1, 4, 2, 2, 5, 0, 2, 2, 2, 1, 5, 2, 3, 0, 1, 4};
    c.nq2 = {3, 2, 2, 0, 5, 0, 5, 2, 4, 1, 5, 3, 3, 2, 2, 5, 3, 5, 3, 2, 3, 0, 4, 0, 1, 0, 1, 1, 3, 1};
    c.nq3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    c.succ = {{3, 9},
              {2, 7},
              {5, 7},
              {5, 9, 10, 12, 19, 21, 23, 25},
              {9, 11},
              {7, 21, 23, 25},
              {9, 11},
              {8},
              {13, 14, 15},
              {11, 13, 14},
              {11, 21, 23, 25},
              {12, 15, 16},
              {16, 21, 23, 25},
              {16, 17, 23},
              {16, 17, 23},
              {17, 22},
              {18, 19, 20, 26},
              {18, 19, 24},
              {28},
              {20, 21, 23, 25},
              {21, 28},
              {22, 25, 27},
              {29},
              {24, 27},
              {26, 27},
              {27},
              {28},
              {28},
              {29},
              {}};
    c.pred = {{},
              {},
              {1},
              {0},
              {},
              {2, 3},
              {},
              {1, 2, 5},
              {7},
              {0, 3, 4, 6},
              {3},
              {4, 6, 9, 10},
              {3, 11},
              {8, 9},
              {8, 9},
              {8, 11},
              {11, 12, 13, 14},
              {13, 14, 15},
              {16, 17},
              {3, 16, 17},
              {16, 19},
              {3, 5, 10, 12, 19, 20},
              {15, 21},
              {3, 5, 10, 12, 13, 14, 19},
              {17, 23},
              {3, 5, 10, 12, 19, 21},
              {16, 24},
              {21, 23, 24, 25},
              {18, 20, 26, 27},
              {22, 28}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.3;
    c.layout = {2, 5, 0, 4, 1, 3};
    c.forward = true;
    c.seed = 14;
    c.expected_events = {1,  2,   -30, 4,  6,  0,  -18, 3,  -30, 10, 9,  -23, 5,   7,  8,  -18, 11, 15, -9, 13,  14, -2,
                         17, -30, 12,  16, 18, -4, 19,  23, 20,  21, 25, 22,  -23, 24, 26, -23, 27, 28, -9, -23, 29};
    c.expected_final_pi = {0, 4, 2, 3, 1, 5};
    c.expected_num_swaps = 13;
    c.expected_depth = 29;
    c.expected_next_bits = 2645178341U;
    RunCase(c);
}

TEST(SabreRoutingTest, Line5ReleaseValve) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
    c.nq1 = {2, 0, 1, 1, 3, 4, 0, 1, 1, 3, 2, 4, 2, 0, 3, 0, 1, 2, 4, 0};
    c.nq2 = {0, 1, 3, 3, 1, 3, 1, 4, 4, 2, 0, 2, 1, 2, 2, 3, 0, 4, 2, 1};
    c.nq3 = {-1, 4, -1, 4, 0, 0, 4, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1};
    c.succ = {{1, 9},   {2, 3, 4}, {3},  {4, 5},   {5, 6},       {6, 9}, {7, 10}, {8},  {11, 12}, {10, 14},
              {11, 13}, {12, 17},  {13}, {14, 15}, {15, 16, 17}, {16},   {19},    {18}, {},       {}};
    c.pred = {{},     {0},     {1},     {1, 2},   {1, 3},  {3, 4},   {4, 5},   {6},      {7},  {0, 5},
              {6, 9}, {8, 10}, {8, 11}, {10, 12}, {9, 13}, {13, 14}, {14, 15}, {11, 14}, {17}, {16}};
    c.has_three_qubit = true;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 1;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {4, 3, 2, 1, 0};
    c.forward = true;
    c.seed = 15;
    c.expected_events = {-14, 0, -20, -2, 1,  -8,  2,  3,   -14, 4,  -2, 5,  -2,  6,  -8, 7,  8,  -2, -8,
                         -20, 9, -14, -2, 10, -14, 11, -14, -2,  12, 13, 14, -14, 17, 18, -2, 15, 16, 19};
    c.expected_final_pi = {1, 0, 3, 2, 4};
    c.expected_num_swaps = 18;
    c.expected_depth = 30;
    c.expected_next_bits = 4145683452U;
    RunCase(c);
}

TEST(SabreRoutingTest, Line5Measure) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
    c.nq1 = {0, -1, 2, 3, 2, -1, 3, 0, 2, 4, 2, 0, 1, 1, 4, -1, -1, 3, 4, 4, 0, 1, 2, 2, 4};
    c.nq2 = {3, -1, 4, 0, 0, -1, 0, 4, 3, 0, 0, 2, 4, 4, 2, -1, -1, 0, 3, 3, 1, 0, 1, 3, 3};
    c.nq3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    c.succ = {{1},  {2, 3, 5}, {4, 5}, {4, 5},   {5},  {6, 7, 8, 12}, {7, 8},           {9},      {10, 15}, {10, 12},
              {11}, {14, 15},  {13},   {14, 15}, {15}, {16},          {17, 18, 20, 22}, {18, 20}, {19},     {23, 24},
              {21}, {22},      {23},   {24},     {}};
    c.pred = {{},       {0},  {1},      {1},      {2, 3},   {1, 2, 3, 4},    {5},  {5, 6}, {5, 6},   {7},
              {8, 9},   {10}, {5, 9},   {12},     {11, 13}, {8, 11, 13, 14}, {15}, {16},   {16, 17}, {18},
              {16, 17}, {20}, {16, 21}, {19, 22}, {19, 23}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 1.3;
    c.options.extended_set_size = 20;
    c.options.decay_delta = 0.05;
    c.options.decay_reset_interval = 2;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {0, 2, 4, 1, 3};
    c.forward = true;
    c.seed = 16;
    c.expected_events = {0,  1,   3,  2,  -2, -20, -14, 4,  5,  6,  -8, 8,  -20, 7,  9,  12, 13, 10,
                         11, -14, 14, 15, 16, -2,  -8,  17, 20, 21, 18, 19, -20, -2, -8, 22, 23, 24};
    c.expected_final_pi = {4, 3, 2, 1, 0};
    c.expected_num_swaps = 11;
    c.expected_depth = 22;
    c.expected_next_bits = 2015295599U;
    RunCase(c);
}

TEST(SabreRoutingTest, Line5NoLookahead) {
    Case c;
    c.adjacency = {{1}, {0, 2}, {1, 3}, {2, 4}, {3}};
    c.nq1 = {1, 2, 3, 4, 0, 1, 4, 0, 4, 2, 0, 2, 1, 4, 3, 0, 3, 3, 2, 3, 1, 2, 1, 2, 2};
    c.nq2 = {3, 0, 0, 0, 3, 2, 1, 4, 3, 4, 2, 0, 3, 2, 0, 4, 2, 2, 1, 0, 3, 3, 4, 0, 1};
    c.nq3 = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    c.succ = {{2, 5},   {2, 5},   {3, 4},   {4, 6},   {7, 8},   {6, 9},   {7, 12},  {8, 10}, {9, 12},
              {10, 13}, {11},     {13, 14}, {14, 18}, {15, 16}, {15, 16}, {19, 22}, {17},    {18, 19},
              {20, 21}, {20, 23}, {21, 22}, {23},     {24},     {24},     {}};
    c.pred = {{},       {},       {0, 1},   {2},      {2, 3},   {0, 1},   {3, 5},   {4, 6},   {4, 7},
              {5, 8},   {7, 9},   {10},     {6, 8},   {9, 11},  {11, 12}, {13, 14}, {13, 14}, {16},
              {12, 17}, {15, 17}, {18, 19}, {18, 20}, {15, 20}, {19, 21}, {22, 23}};
    c.has_three_qubit = false;
    c.options.lookahead_weight = 0.5;
    c.options.extended_set_size = 0;
    c.options.decay_delta = 0.001;
    c.options.decay_reset_interval = 5;
    c.options.release_valve_threshold = 0;
    c.options.swap_fusion_bonus = 0.0;
    c.layout = {2, 0, 4, 3, 1};
    c.forward = true;
    c.seed = 17;
    c.expected_events = {-14, 1,   -8, 0,  -14, 2,  3,   4,  -20, -2, -14, 5,  -20, -8,  6,  7,
                         -2,  -14, 8,  -8, 9,   12, -20, -2, -14, 10, 11,  14, 13,  -8,  15, 16,
                         17,  -14, 19, 18, -20, 20, -14, 21, -8,  22, -14, 23, -8,  -20, 24};
    c.expected_final_pi = {4, 2, 3, 1, 0};
    c.expected_num_swaps = 22;
    c.expected_depth = 33;
    c.expected_next_bits = 1083980018U;
    RunCase(c);
}

// ---- 镜像门感知路由（期望值由 Python 参考实现生成）----

// 线 0-1-2-3，恒等布局：节点 0 = (0,1)，节点 1 = (0,2) 依赖节点 0。
struct MirrorFixture {
    std::vector<std::vector<int>> line = {{1}, {0, 2}, {1, 3}, {2}};
    RoutingGraph graph{line, BfsDistances(line)};

    RouteResult Route(const std::vector<int>& mirror_fwd, const std::vector<int>& mirror_bwd,
                      const RouteOptions& options, bool forward, std::uint32_t* next_bits) const {
        RoutingDag dag({0, 0}, {1, 2}, {-1, -1}, {{1}, {}}, {{}, {0}}, false, mirror_fwd, mirror_bwd);
        CPythonRandom rng(5);
        RouteResult result = RouteDag(graph, dag, options, {0, 1, 2, 3}, forward, /*record=*/true, rng);
        *next_bits = rng.GetRandBits(32);
        return result;
    }
};

TEST(SabreMirrorTest, FreeMirrorReplacesRoutingSwap) {
    MirrorFixture f;
    RouteOptions options;
    options.mirror_gates = true;
    std::uint32_t bits = 0;
    RouteResult r = f.Route({0, 0}, {0, 0}, options, true, &bits);
    // 执行节点 0 后在同一对 (0,1) 上镜像，节点 1 随即可执行，无需路由 SWAP
    EXPECT_EQ(r.events, (std::vector<std::int64_t>{0, -2, 1}));
    EXPECT_EQ(r.final_pi, (std::vector<int>{1, 0, 2, 3}));
    EXPECT_EQ(r.num_swaps, 1);
    EXPECT_EQ(r.num_mirrors, 1);
    EXPECT_EQ(r.swap_cost, 0);
    EXPECT_EQ(r.depth, 2);
    EXPECT_EQ(bits, 2675342405U);
}

TEST(SabreMirrorTest, OptionOffOrExpensiveMirrorKeepsPlainRouting) {
    MirrorFixture f;
    std::uint32_t bits = 0;
    RouteOptions off;
    RouteOptions on;
    on.mirror_gates = true;
    // 选项关闭 / 镜像代价 3（收益 1 低于 0 + 3 * 0.5）：与普通 SABRE 一致
    for (const auto& [options, cost] : {std::pair{off, 0}, std::pair{on, 3}}) {
        RouteResult r = f.Route({cost, cost}, {cost, cost}, options, true, &bits);
        EXPECT_EQ(r.events, (std::vector<std::int64_t>{0, -7, 1}));
        EXPECT_EQ(r.final_pi, (std::vector<int>{0, 2, 1, 3}));
        EXPECT_EQ(r.num_mirrors, 0);
        EXPECT_EQ(r.swap_cost, 3);
        EXPECT_EQ(bits, 3185950873U);
    }
    // 未标注镜像代价的 DAG：选项打开也不做镜像
    RouteResult r = f.Route({}, {}, on, true, &bits);
    EXPECT_EQ(r.num_mirrors, 0);
    EXPECT_EQ(r.swap_cost, 3);
}

TEST(SabreMirrorTest, FusedSearchSwapCountsMirrorCost) {
    MirrorFixture f;
    RouteOptions options;
    options.mirror_gates = true;
    options.mirror_cz_penalty = 2.0;  // 执行时拒绝镜像（收益 1 < 2）
    options.swap_fusion_bonus = 0.3;  // 搜索时 (0,1) 得到 1 - 0.3 * 2 / 3 的折扣
    std::uint32_t bits = 0;
    RouteResult r = f.Route({1, -1}, {-1, 1}, options, true, &bits);
    EXPECT_EQ(r.events, (std::vector<std::int64_t>{0, -2, 1}));
    EXPECT_EQ(r.num_mirrors, 0);
    EXPECT_EQ(r.swap_cost, 1);
    EXPECT_EQ(r.depth, 3);
    EXPECT_EQ(bits, 2675342405U);
    // 无融合奖励时与普通 SABRE 一致
    options.swap_fusion_bonus = 0.0;
    r = f.Route({1, -1}, {-1, 1}, options, true, &bits);
    EXPECT_EQ(r.events, (std::vector<std::int64_t>{0, -7, 1}));
    EXPECT_EQ(r.swap_cost, 3);
}

TEST(SabreMirrorTest, BackwardUsesBlockStartCosts) {
    MirrorFixture f;
    RouteOptions options;
    options.mirror_gates = true;
    std::uint32_t bits = 0;
    RouteResult r = f.Route({1, -1}, {-1, 1}, options, false, &bits);
    EXPECT_EQ(r.events, (std::vector<std::int64_t>{-2, 1, 0}));
    EXPECT_EQ(r.final_pi, (std::vector<int>{1, 0, 2, 3}));
    EXPECT_EQ(r.swap_cost, 3);
    EXPECT_EQ(bits, 2675342405U);
}

TEST(SabreMirrorTest, RejectsInvalidMirrorArrays) {
    EXPECT_THROW(RoutingDag({0}, {1}, {-1}, {{}}, {{}}, false, {0}, {}), std::invalid_argument);
    EXPECT_THROW(RoutingDag({0}, {1}, {-1}, {{}}, {{}}, false, {0, 0}, {0, 0}), std::invalid_argument);
    EXPECT_THROW(RoutingDag({0}, {1}, {-1}, {{}}, {{}}, false, {4}, {0}), std::invalid_argument);
    EXPECT_THROW(RoutingDag({0}, {1}, {-1}, {{}}, {{}}, false, {0}, {-2}), std::invalid_argument);
    EXPECT_NO_THROW(RoutingDag({0}, {1}, {-1}, {{}}, {{}}, false, {-1}, {3}));
}

}  // namespace
