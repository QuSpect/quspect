#pragma once

#include <cstdint>
#include <vector>

#include "quspect_native/transpiler/cpython_random.hpp"

namespace quspect::native::transpiler {

// 物理耦合图：邻接表（邻居顺序决定候选 SWAP 顺序与最短路径的选择）、距离矩阵与候选边表。
//
// 与 Python 端 ``SABRE._G`` / ``SABRE._D`` / ``incident_edges`` 一一对应；距离矩阵由调用方
// 传入（不连通的比特对为 10**9），保证两端使用完全相同的整数距离。
class RoutingGraph {
 public:
    // 抛出 std::invalid_argument：尺寸不一致或邻居越界。
    RoutingGraph(std::vector<std::vector<int>> adjacency, const std::vector<std::vector<std::int64_t>>& distance);

    int num_physical() const {
        return num_physical_;
    }
    std::int64_t Distance(int a, int b) const {
        return distance_[static_cast<std::size_t>(a) * num_physical_ + b];
    }
    bool Adjacent(int a, int b) const {
        return adjacent_[static_cast<std::size_t>(a) * num_physical_ + b] != 0;
    }
    const std::vector<int>& Neighbors(int p) const {
        return adjacency_[p];
    }
    // ``incident_edges(G)[p]`` 的边编号（去重、保持邻居顺序）。
    const std::vector<int>& EdgesAt(int p) const {
        return edges_at_[p];
    }
    int num_edges() const {
        return static_cast<int>(edge_x_.size());
    }
    int EdgeX(int e) const {
        return edge_x_[e];
    }
    int EdgeY(int e) const {
        return edge_y_[e];
    }
    // y 在 Neighbors(x) 中（首次出现）的下标；x 在 Neighbors(y) 中的下标。
    int EdgeIndexOfYAtX(int e) const {
        return edge_y_at_x_[e];
    }
    int EdgeIndexOfXAtY(int e) const {
        return edge_x_at_y_[e];
    }

    // 三比特节点代价：min(dab + dac, dab + dbc, dac + dbc) - 1（1 = 可执行）。
    std::int64_t Cost3(int a, int b, int c) const;

 private:
    int num_physical_;
    std::vector<std::vector<int>> adjacency_;
    std::vector<std::int64_t> distance_;
    std::vector<std::uint8_t> adjacent_;
    std::vector<std::vector<int>> edges_at_;
    std::vector<int> edge_x_;
    std::vector<int> edge_y_;
    std::vector<int> edge_y_at_x_;
    std::vector<int> edge_x_at_y_;
};

// 路由 DAG（Python ``_RoutingDag`` 的路由相关字段）。
//
// ``nq1[n] < 0``：测量节点（总是可执行）；``nq3[n] >= 0``：三比特 ``ccx`` 节点。
//
// ``mirror_fwd`` / ``mirror_bwd``（可为空，表示未标注）：节点是其比特对块在正向 / 反向路由中的
// 末节点时为镜像代价 cz(U·SWAP) - cz(U)（0..3），否则为 -1（Python ``_annotate_mirror_costs``）。
class RoutingDag {
 public:
    // 抛出 std::invalid_argument：数组长度不一致、后继 / 前驱越界或镜像代价超出 [-1, 3]。
    RoutingDag(std::vector<int> nq1, std::vector<int> nq2, std::vector<int> nq3, std::vector<std::vector<int>> succ,
               std::vector<std::vector<int>> pred, bool has_three_qubit, std::vector<int> mirror_fwd = {},
               std::vector<int> mirror_bwd = {});

    int num_nodes() const {
        return static_cast<int>(nq1_.size());
    }
    const std::vector<int>& nq1() const {
        return nq1_;
    }
    const std::vector<int>& nq2() const {
        return nq2_;
    }
    const std::vector<int>& nq3() const {
        return nq3_;
    }
    const std::vector<std::vector<int>>& succ() const {
        return succ_;
    }
    const std::vector<std::vector<int>>& pred() const {
        return pred_;
    }
    bool has_three_qubit() const {
        return has_three_qubit_;
    }
    bool has_mirror_costs() const {
        return !mirror_fwd_.empty();
    }
    const std::vector<int>& mirror_fwd() const {
        return mirror_fwd_;
    }
    const std::vector<int>& mirror_bwd() const {
        return mirror_bwd_;
    }

 private:
    std::vector<int> nq1_;
    std::vector<int> nq2_;
    std::vector<int> nq3_;
    std::vector<std::vector<int>> succ_;
    std::vector<std::vector<int>> pred_;
    bool has_three_qubit_;
    std::vector<int> mirror_fwd_;
    std::vector<int> mirror_bwd_;
};

// SABRE 参数（与 Python ``SABRE`` 的同名属性一致）。
struct RouteOptions {
    double lookahead_weight = 0.5;
    int extended_set_size = 20;
    double decay_delta = 0.001;
    int decay_reset_interval = 5;
    // <= 0 表示默认值 10 * max(num_physical, 1)。
    int release_valve_threshold = 0;
    double swap_fusion_bonus = 0.0;
    // 镜像门感知路由（需要 DAG 带镜像代价）：刚执行的块末节点之后若交换其两个逻辑比特让前沿 +
    // 扩展集代价下降超过 mirror_threshold + mirror_cz_penalty * 镜像代价，就插入该 SWAP。
    bool mirror_gates = false;
    double mirror_threshold = 0.0;
    double mirror_cz_penalty = 0.5;
};

struct RouteResult {
    // 所有 SWAP 事件数（含镜像 SWAP）。
    std::int64_t num_swaps = 0;
    // SWAP 的估计 CZ 代价（普通 SWAP 3，与块合并的 SWAP 计镜像代价）。
    std::int64_t swap_cost = 0;
    std::int64_t num_mirrors = 0;
    std::vector<int> final_pi;
    // 执行的节点编号（>= 0）或编码后的 SWAP -1 - (x * n + y)；只在 record 时记录。
    std::vector<std::int64_t> events;
    int depth = 0;
};

// 一次 SABRE 路由（Python ``quspect.transpiler._routing._route_dag_python`` 的逐位等价实现）。
//
// ``layout[token] = physical`` 必须是 0..num_physical-1 的排列。并列最优的候选 SWAP 用 ``rng``
// 按 CPython ``Random.choice`` 的语义打破平局，因此给定相同的随机源状态，结果（事件序列、
// 最终排列、SWAP 数、深度以及随机源的后续状态）与 Python 实现完全一致。
//
// 抛出 std::invalid_argument：布局不合法，或当前映射下没有可用的路由 SWAP（拓扑不连通）。
RouteResult RouteDag(const RoutingGraph& graph, const RoutingDag& dag, const RouteOptions& options,
                     const std::vector<int>& layout, bool forward, bool record, CPythonRandom& rng);

// 不连通错误的消息（与 Python ``_routing._DISCONNECTED`` 一致）。
extern const char kDisconnectedMessage[];

}  // namespace quspect::native::transpiler
