// SABRE 路由主循环的原生实现。
//
// 逐行对应 quspect/transpiler/_routing.py 中的 Python 实现（_route_dag_python、_extended_set、
// _release_valve、_shortest_path、_delta3），包括：
//   * 前沿节点的插入顺序（Python dict 的插入顺序）、候选 SWAP 的合并顺序（dict.update）；
//   * 惰性代价增量表的失效规则（只影响缓存，但为了与参考实现一一对应仍原样保留）；
//   * 浮点计分表达式的运算顺序（本文件以 -ffp-contract=off 编译，禁止 FMA 合并）；
//   * 平局用 CPython Random.choice 语义抽样。
// 距离相关的量都是整数（int64），与求和顺序无关。

#include "quspect_native/transpiler/sabre_routing.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace quspect::native::transpiler {

const char kDisconnectedMessage[]
    = "No routing SWAP is available for the current mapping. "
      "The selected physical-qubit topology may be disconnected.";

namespace {

constexpr double kTieEps = 1e-10;

int FirstIndexOf(const std::vector<int>& values, int value) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool Contains(const std::vector<int>& values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void RemoveFirst(std::vector<int>& values, int value) {
    auto it = std::find(values.begin(), values.end(), value);
    if (it != values.end()) {
        values.erase(it);
    }
}

}  // namespace

// =============================================================================
// RoutingGraph / RoutingDag
// =============================================================================

RoutingGraph::RoutingGraph(std::vector<std::vector<int>> adjacency,
                           const std::vector<std::vector<std::int64_t>>& distance)
    : num_physical_(static_cast<int>(adjacency.size())), adjacency_(std::move(adjacency)) {
    const int n = num_physical_;
    if (distance.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("distance matrix must have one row per physical qubit");
    }
    distance_.assign(static_cast<std::size_t>(n) * n, 0);
    adjacent_.assign(static_cast<std::size_t>(n) * n, 0);
    for (int a = 0; a < n; ++a) {
        if (distance[a].size() != static_cast<std::size_t>(n)) {
            throw std::invalid_argument("distance matrix must be square");
        }
        for (int b = 0; b < n; ++b) {
            distance_[static_cast<std::size_t>(a) * n + b] = distance[a][b];
        }
        for (int z : adjacency_[a]) {
            if (z < 0 || z >= n) {
                throw std::invalid_argument("adjacency entry out of range");
            }
            adjacent_[static_cast<std::size_t>(a) * n + z] = 1;
        }
    }
    // incident_edges：code = x * n + y（x < y），每个物理比特按邻居顺序列出其边（dict 去重）。
    std::unordered_map<std::int64_t, int> edge_of_code;
    edges_at_.assign(n, {});
    for (int p = 0; p < n; ++p) {
        for (int z : adjacency_[p]) {
            const int x = std::min(p, z);
            const int y = std::max(p, z);
            const std::int64_t code = static_cast<std::int64_t>(x) * n + y;
            auto [it, inserted] = edge_of_code.emplace(code, static_cast<int>(edge_x_.size()));
            if (inserted) {
                edge_x_.push_back(x);
                edge_y_.push_back(y);
            }
            if (!Contains(edges_at_[p], it->second)) {
                edges_at_[p].push_back(it->second);
            }
        }
    }
    edge_y_at_x_.resize(edge_x_.size());
    edge_x_at_y_.resize(edge_x_.size());
    for (std::size_t e = 0; e < edge_x_.size(); ++e) {
        edge_y_at_x_[e] = FirstIndexOf(adjacency_[edge_x_[e]], edge_y_[e]);
        edge_x_at_y_[e] = FirstIndexOf(adjacency_[edge_y_[e]], edge_x_[e]);
    }
}

std::int64_t RoutingGraph::Cost3(int a, int b, int c) const {
    const std::int64_t dab = Distance(a, b);
    const std::int64_t dac = Distance(a, c);
    const std::int64_t dbc = Distance(b, c);
    return std::min({dab + dac, dab + dbc, dac + dbc}) - 1;
}

RoutingDag::RoutingDag(std::vector<int> nq1, std::vector<int> nq2, std::vector<int> nq3,
                       std::vector<std::vector<int>> succ, std::vector<std::vector<int>> pred, bool has_three_qubit,
                       std::vector<int> mirror_fwd, std::vector<int> mirror_bwd)
    : nq1_(std::move(nq1))
    , nq2_(std::move(nq2))
    , nq3_(std::move(nq3))
    , succ_(std::move(succ))
    , pred_(std::move(pred))
    , has_three_qubit_(has_three_qubit)
    , mirror_fwd_(std::move(mirror_fwd))
    , mirror_bwd_(std::move(mirror_bwd)) {
    const std::size_t n = nq1_.size();
    if (nq2_.size() != n || nq3_.size() != n || succ_.size() != n || pred_.size() != n) {
        throw std::invalid_argument("routing DAG arrays must have the same length");
    }
    if (mirror_fwd_.size() != mirror_bwd_.size() || (!mirror_fwd_.empty() && mirror_fwd_.size() != n)) {
        throw std::invalid_argument("mirror cost arrays must be empty or have one entry per node");
    }
    for (const auto* costs : {&mirror_fwd_, &mirror_bwd_}) {
        for (int c : *costs) {
            if (c < -1 || c > 3) {
                throw std::invalid_argument("mirror cost out of range");
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (const auto* lists : {&succ_, &pred_}) {
            for (int s : (*lists)[i]) {
                if (s < 0 || static_cast<std::size_t>(s) >= n) {
                    throw std::invalid_argument("routing DAG edge out of range");
                }
            }
        }
    }
}

// =============================================================================
// Routing pass
// =============================================================================

namespace {

class Router {
 public:
    Router(const RoutingGraph& graph, const RoutingDag& dag, const RouteOptions& options,
           const std::vector<int>& layout, bool forward, bool record, CPythonRandom& rng)
        : g_(graph)
        , n_(graph.num_physical())
        , nq1_(dag.nq1())
        , nq2_(dag.nq2())
        , nq3_(dag.nq3())
        , has3_(dag.has_three_qubit())
        , succ_(forward ? dag.succ() : dag.pred())
        , record_(record)
        , rng_(rng)
        , w_(options.lookahead_weight)
        , ext_size_(options.extended_set_size)
        , delta_(options.decay_delta)
        , reset_interval_(options.decay_reset_interval)
        , valve_(options.release_valve_threshold > 0 ? options.release_valve_threshold : 10 * std::max(n_, 1))
        , fusion_(1.0 - options.swap_fusion_bonus)
        , use_fusion_(fusion_ != 1.0 && !has3_)
        , mirror_on_(options.mirror_gates && dag.has_mirror_costs())
        , mcost_(forward ? dag.mirror_fwd() : dag.mirror_bwd())
        , track_last_(use_fusion_ || mirror_on_) {
        for (int d = 0; d < 4; ++d) {
            // Python：1.0 - bonus * (3 - d) / 3；threshold + d * penalty + eps（相同运算顺序）
            mfusion_[d] = 1.0 - options.swap_fusion_bonus * static_cast<double>(3 - d) / 3.0;
            mlimit_[d] = options.mirror_threshold + static_cast<double>(d) * options.mirror_cz_penalty + kTieEps;
        }
        if (layout.size() != static_cast<std::size_t>(n_)) {
            throw std::invalid_argument("layout must be a full permutation of the physical qubits");
        }
        if (reset_interval_ < 1) {
            throw std::invalid_argument("decay_reset_interval must be at least 1");
        }
        pi_ = layout;
        rpi_.assign(n_, -1);
        for (int token = 0; token < n_; ++token) {
            const int p = pi_[token];
            if (p < 0 || p >= n_ || rpi_[p] != -1) {
                throw std::invalid_argument("layout must be a full permutation of the physical qubits");
            }
            rpi_[p] = token;
        }
        const auto& in_lists = forward ? dag.pred() : dag.succ();
        indeg_.resize(in_lists.size());
        for (std::size_t i = 0; i < in_lists.size(); ++i) {
            indeg_[i] = static_cast<int>(in_lists[i].size());
        }
        in_front_.assign(in_lists.size(), 0);
        front2_.assign(n_, {});
        front3_.assign(n_, {});
        if (record_) {
            level_.assign(n_, 0);
        }
        last_op_.assign(n_, -1);
        last_exec_.assign(n_, -1);
        fpart_.assign(n_, {});
        partners_.assign(n_, {});
        ext3_of_.assign(n_, {});
        tables_.assign(n_, {});
        table_valid_.assign(n_, 0);
        has_part_.assign(n_, 0);
        decay_.assign(n_, 1.0);
        edge_stamp_.assign(g_.num_edges(), 0);
        in_old_partners_.assign(n_, 0);
        in_old_fpart_.assign(n_, 0);
        old_partner_lists_.assign(n_, {});
        old_fpart_lists_.assign(n_, {});
    }

    RouteResult Run();

 private:
    std::int64_t D(int a, int b) const {
        return g_.Distance(a, b);
    }
    std::int64_t NodeCost(int node) const {
        const int c = nq3_[node];
        if (c >= 0) {
            return g_.Cost3(pi_[nq1_[node]], pi_[nq2_[node]], pi_[c]);
        }
        return D(pi_[nq1_[node]], pi_[nq2_[node]]);
    }
    void AddFront(int node) {
        front_.push_back(node);
        in_front_[node] = 1;
    }
    void Invalidate(int token) {
        table_valid_[token] = 0;
    }

    void Drain(std::vector<int>& stack);
    // 返回该 SWAP 的估计 CZ 代价（3，或紧跟块末节点时的镜像代价）。
    int ApplySwap(int x, int y, std::vector<int>& ready);
    // 把可执行节点移出前沿并 drain。
    void Retire(std::vector<int>& ready);
    // 评估本轮 drain 中执行的块末节点的镜像；可执行的前沿节点追加到 ready。
    void TryMirrors(std::vector<int>& ready);
    const std::vector<std::pair<std::int64_t, std::int64_t>>& Table(int token);
    void RebuildExtendedSet();
    void ExtendedSet();
    std::int64_t Delta3(const std::vector<int>& nodes_x, const std::vector<int>& nodes_y, int tx, int ty, int x,
                        int y) const;
    std::vector<std::pair<int, int>> ReleaseValve() const;
    std::vector<int> ShortestPath(int source, int dest) const;

    const RoutingGraph& g_;
    const int n_;
    const std::vector<int>& nq1_;
    const std::vector<int>& nq2_;
    const std::vector<int>& nq3_;
    const bool has3_;
    const std::vector<std::vector<int>>& succ_;
    const bool record_;
    CPythonRandom& rng_;
    const double w_;
    const int ext_size_;
    const double delta_;
    const int reset_interval_;
    const int valve_;
    const double fusion_;
    const bool use_fusion_;
    const bool mirror_on_;
    const std::vector<int>& mcost_;
    const bool track_last_;
    double mfusion_[4];
    double mlimit_[4];
    std::vector<int> mcands_;
    std::vector<int> last_exec_;
    std::int64_t swap_cost_ = 0;
    std::int64_t num_mirrors_ = 0;

    std::vector<int> pi_;
    std::vector<int> rpi_;
    std::vector<int> indeg_;
    // 前沿：插入顺序 + 成员标记（删除时稳定压缩）。
    std::vector<int> front_;
    std::vector<std::uint8_t> in_front_;
    std::vector<std::vector<int>> front2_;
    std::vector<std::vector<int>> front3_;
    std::vector<std::int64_t> events_;
    std::int64_t num_swaps_ = 0;
    std::vector<int> level_;
    std::vector<int> last_op_;
    int op_counter_ = -2;

    std::vector<std::vector<int>> fpart_;
    std::vector<std::vector<int>> partners_;
    std::vector<std::vector<int>> ext3_of_;
    std::vector<int> front_tokens_;
    std::vector<int> touched_;
    std::vector<std::vector<std::pair<std::int64_t, std::int64_t>>> tables_;
    std::vector<std::uint8_t> table_valid_;
    std::vector<std::uint8_t> has_part_;

    std::vector<int> ext_;
    std::vector<int> ext2_;
    std::vector<int> ext3_;
    std::vector<double> decay_;

    std::vector<int> edge_stamp_;
    int stamp_ = 0;
    std::vector<int> cands_;
    std::vector<int> best_;

    // 扩展集重建时的旧伙伴列表（Python 中的 old_partners / old_fpart 字典）。
    std::vector<std::uint8_t> in_old_partners_;
    std::vector<std::uint8_t> in_old_fpart_;
    std::vector<std::vector<int>> old_partner_lists_;
    std::vector<std::vector<int>> old_fpart_lists_;
    std::vector<int> old_partner_tokens_;
    std::vector<int> old_fpart_tokens_;
    std::vector<int> visit_;
    std::vector<int> decremented_;
};

void Router::Drain(std::vector<int>& stack) {
    while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        const int a = nq1_[node];
        if (a >= 0) {
            const int c = nq3_[node];
            const int b = nq2_[node];
            if (c < 0) {
                const int pa = pi_[a];
                const int pb = pi_[b];
                if (!g_.Adjacent(pa, pb)) {
                    AddFront(node);
                    front2_[a].push_back(node);
                    front2_[b].push_back(node);
                    continue;
                }
                if (record_) {
                    const int lv = std::max(level_[pa], level_[pb]) + 1;
                    level_[pa] = level_[pb] = lv;
                }
                if (track_last_) {
                    last_op_[pa] = last_op_[pb] = node;
                }
                if (mirror_on_) {
                    last_exec_[a] = last_exec_[b] = node;
                    if (mcost_[node] >= 0) {
                        mcands_.push_back(node);
                    }
                }
            } else {
                const int pa = pi_[a];
                const int pb = pi_[b];
                const int pc = pi_[c];
                if (g_.Cost3(pa, pb, pc) != 1) {
                    AddFront(node);
                    front3_[a].push_back(node);
                    front3_[b].push_back(node);
                    front3_[c].push_back(node);
                    continue;
                }
                if (record_) {
                    const int lv = std::max({level_[pa], level_[pb], level_[pc]}) + 1;
                    level_[pa] = level_[pb] = level_[pc] = lv;
                }
                if (mirror_on_) {
                    --op_counter_;
                    last_op_[pa] = last_op_[pb] = last_op_[pc] = op_counter_;
                    last_exec_[a] = last_exec_[b] = last_exec_[c] = node;
                }
            }
        } else if (mirror_on_) {
            // 测量是调度栅栏：之前的块都不再与之后的 SWAP 相邻
            mcands_.clear();
            std::fill(last_op_.begin(), last_op_.end(), -1);
        }
        if (record_) {
            events_.push_back(node);
        }
        for (int s : succ_[node]) {
            if (--indeg_[s] == 0) {
                stack.push_back(s);
            }
        }
    }
}

int Router::ApplySwap(int x, int y, std::vector<int>& ready) {
    int cost = 3;
    if (mirror_on_) {
        const int op = last_op_[x];
        if (op >= 0 && op == last_op_[y] && mcost_[op] >= 0) {
            cost = mcost_[op];
        }
    }
    const int tx = rpi_[x];
    const int ty = rpi_[y];
    pi_[tx] = y;
    pi_[ty] = x;
    rpi_[x] = ty;
    rpi_[y] = tx;
    // tx、ty 移动了：它们自己以及以它们为伙伴的比特的增量表失效
    Invalidate(tx);
    Invalidate(ty);
    for (int t : {tx, ty}) {
        for (int u : fpart_[t]) {
            Invalidate(u);
        }
        for (int u : partners_[t]) {
            Invalidate(u);
        }
    }
    if (track_last_) {
        --op_counter_;
        last_op_[x] = last_op_[y] = op_counter_;
    }
    for (int token : {tx, ty}) {
        for (int gate : front2_[token]) {
            if (!Contains(ready, gate) && g_.Adjacent(pi_[nq1_[gate]], pi_[nq2_[gate]])) {
                ready.push_back(gate);
            }
        }
        if (has3_) {
            for (int gate : front3_[token]) {
                if (!Contains(ready, gate) && NodeCost(gate) == 1) {
                    ready.push_back(gate);
                }
            }
        }
    }
    return cost;
}

void Router::Retire(std::vector<int>& ready) {
    for (int gate : ready) {
        in_front_[gate] = 0;
        const int c = nq3_[gate];
        if (c < 0) {
            RemoveFirst(front2_[nq1_[gate]], gate);
            RemoveFirst(front2_[nq2_[gate]], gate);
        } else {
            for (int token : {nq1_[gate], nq2_[gate], c}) {
                RemoveFirst(front3_[token], gate);
            }
        }
    }
    front_.erase(std::remove_if(front_.begin(), front_.end(), [&](int gate) { return in_front_[gate] == 0; }),
                 front_.end());
    Drain(ready);
}

void Router::TryMirrors(std::vector<int>& ready) {
    // 镜像门：刚执行的块末节点 n（其比特上之后没有再执行别的节点）后紧跟同对 SWAP 的代价只有
    // mcost_[n]；若交换 n 的两个逻辑比特让前沿 + 扩展集代价下降超过 mlimit_[mcost_[n]]，就执行它。
    const double e_scale = ext_.empty() ? 0.0 : w_ / static_cast<double>(ext_.size());
    for (int node : mcands_) {
        const int a = nq1_[node];
        const int b = nq2_[node];
        if (last_exec_[a] != node || last_exec_[b] != node) {
            continue;
        }
        const int pa = pi_[a];
        const int pb = pi_[b];
        auto moved = [&](const std::vector<int>& others, int self_partner, int from, int to) {
            std::int64_t sum = 0;
            for (int o : others) {
                if (o != self_partner) {
                    const int po = pi_[o];
                    sum += D(to, po) - D(from, po);
                }
            }
            return sum;
        };
        std::int64_t df = moved(fpart_[a], b, pa, pb) + moved(fpart_[b], a, pb, pa);
        std::int64_t de = moved(partners_[a], b, pa, pb) + moved(partners_[b], a, pb, pa);
        if (has3_) {
            if (!front3_[a].empty() || !front3_[b].empty()) {
                df += Delta3(front3_[a], front3_[b], a, b, pa, pb);
            }
            if (!ext3_of_[a].empty() || !ext3_of_[b].empty()) {
                de += Delta3(ext3_of_[a], ext3_of_[b], a, b, pa, pb);
            }
        }
        const int d = mcost_[node];
        // Python：-(df + de * e_scale) > mlimit[d]
        const double gain = -(static_cast<double>(df) + static_cast<double>(de) * e_scale);
        if (gain > mlimit_[d]) {
            const int x = std::min(pa, pb);
            const int y = std::max(pa, pb);
            ApplySwap(x, y, ready);
            swap_cost_ += d;
            ++num_mirrors_;
            ++num_swaps_;
            if (record_) {
                events_.push_back(-1 - (static_cast<std::int64_t>(x) * n_ + y));
            }
            last_exec_[a] = last_exec_[b] = -1;
        }
    }
    mcands_.clear();
}

const std::vector<std::pair<std::int64_t, std::int64_t>>& Router::Table(int token) {
    auto& table = tables_[token];
    if (table_valid_[token] && !table.empty()) {
        return table;
    }
    const int p = pi_[token];
    const auto& neighbors = g_.Neighbors(p);
    table.resize(neighbors.size());
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const int z = neighbors[i];
        std::int64_t df = 0;
        for (int o : fpart_[token]) {
            const int po = pi_[o];
            if (po != z) {
                df += D(z, po) - D(p, po);
            }
        }
        std::int64_t de = 0;
        for (int o : partners_[token]) {
            const int po = pi_[o];
            if (po != z) {
                de += D(z, po) - D(p, po);
            }
        }
        table[i] = {df, de};
    }
    table_valid_[token] = 1;
    return table;
}

void Router::ExtendedSet() {
    ext_.clear();
    if (ext_size_ == 0) {
        return;
    }
    visit_.assign(front_.begin(), front_.end());
    decremented_.clear();
    int found = 0;
    // 遍历时向 visit_ 追加：按下标访问，继续访问新追加的节点
    for (std::size_t vi = 0; vi < visit_.size(); ++vi) {
        const int v = visit_[vi];
        for (int s : succ_[v]) {
            const int d = indeg_[s] - 1;
            indeg_[s] = d;
            decremented_.push_back(s);
            if (d == 0) {
                if (nq1_[s] >= 0) {
                    ext_.push_back(s);
                    ++found;
                    if (found >= ext_size_) {
                        break;
                    }
                }
                visit_.push_back(s);
            }
        }
        if (found >= ext_size_) {
            break;
        }
    }
    for (int s : decremented_) {
        ++indeg_[s];
    }
}

void Router::RebuildExtendedSet() {
    // old_partners = {token: partners[token] for token in touched}
    for (int t : old_partner_tokens_) {
        in_old_partners_[t] = 0;
    }
    old_partner_tokens_.clear();
    for (int t : touched_) {
        if (!in_old_partners_[t]) {
            in_old_partners_[t] = 1;
            old_partner_tokens_.push_back(t);
            old_partner_lists_[t].swap(partners_[t]);
        }
        partners_[t].clear();
        ext3_of_[t].clear();
    }
    ExtendedSet();
    touched_.clear();
    ext2_.clear();
    ext3_.clear();
    for (int gate : ext_) {
        const int a = nq1_[gate];
        const int b = nq2_[gate];
        const int c = nq3_[gate];
        if (c >= 0) {
            ext3_.push_back(gate);
            for (int token : {a, b, c}) {
                ext3_of_[token].push_back(gate);
                touched_.push_back(token);
            }
            continue;
        }
        ext2_.push_back(gate);
        partners_[a].push_back(b);
        partners_[b].push_back(a);
        touched_.push_back(a);
        touched_.push_back(b);
    }
    // 前沿只在外层迭代之间变化：重建每个比特的前沿伙伴表
    for (int t : old_fpart_tokens_) {
        in_old_fpart_[t] = 0;
    }
    old_fpart_tokens_.clear();
    for (int t : front_tokens_) {
        if (!in_old_fpart_[t]) {
            in_old_fpart_[t] = 1;
            old_fpart_tokens_.push_back(t);
            old_fpart_lists_[t].swap(fpart_[t]);
        }
        fpart_[t].clear();
    }
    front_tokens_.clear();
    for (int gate : front_) {
        if (nq3_[gate] < 0) {
            const int a = nq1_[gate];
            const int b = nq2_[gate];
            fpart_[a].push_back(b);
            fpart_[b].push_back(a);
            front_tokens_.push_back(a);
            front_tokens_.push_back(b);
        }
    }
    // 伙伴列表变化的比特：增量表失效
    for (int t : old_partner_tokens_) {
        if (partners_[t] != old_partner_lists_[t]) {
            Invalidate(t);
        }
    }
    for (int t : old_fpart_tokens_) {
        if (fpart_[t] != old_fpart_lists_[t]) {
            Invalidate(t);
        }
    }
    for (int t : touched_) {
        if (!in_old_partners_[t]) {
            Invalidate(t);
        }
        has_part_[t] = 1;
    }
    for (int t : front_tokens_) {
        if (!in_old_fpart_[t]) {
            Invalidate(t);
        }
        has_part_[t] = 1;
    }
    for (int t : old_partner_tokens_) {
        has_part_[t] = (!fpart_[t].empty() || !partners_[t].empty()) ? 1 : 0;
    }
    for (int t : old_fpart_tokens_) {
        has_part_[t] = (!fpart_[t].empty() || !partners_[t].empty()) ? 1 : 0;
    }
}

std::int64_t Router::Delta3(const std::vector<int>& nodes_x, const std::vector<int>& nodes_y, int tx, int ty, int x,
                            int y) const {
    // 每个节点只计一次（dict.fromkeys(nodes_x + nodes_y)）；整数求和与顺序无关。
    std::int64_t total = 0;
    auto moved = [&](int token) { return token == tx ? y : (token == ty ? x : pi_[token]); };
    auto add = [&](int gate) {
        const int a = nq1_[gate];
        const int b = nq2_[gate];
        const int c = nq3_[gate];
        total += g_.Cost3(moved(a), moved(b), moved(c)) - g_.Cost3(pi_[a], pi_[b], pi_[c]);
    };
    for (std::size_t i = 0; i < nodes_x.size(); ++i) {
        const int gate = nodes_x[i];
        if (std::find(nodes_x.begin(), nodes_x.begin() + static_cast<std::ptrdiff_t>(i), gate)
            == nodes_x.begin() + static_cast<std::ptrdiff_t>(i)) {
            add(gate);
        }
    }
    for (std::size_t i = 0; i < nodes_y.size(); ++i) {
        const int gate = nodes_y[i];
        if (!Contains(nodes_x, gate)
            && std::find(nodes_y.begin(), nodes_y.begin() + static_cast<std::ptrdiff_t>(i), gate)
                   == nodes_y.begin() + static_cast<std::ptrdiff_t>(i)) {
            add(gate);
        }
    }
    return total;
}

std::vector<int> Router::ShortestPath(int source, int dest) const {
    std::vector<int> parent(n_, -1);
    std::vector<int> queue;
    queue.reserve(n_);
    parent[source] = source;
    queue.push_back(source);
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const int node = queue[head];
        if (node == dest) {
            break;
        }
        for (int neighbor : g_.Neighbors(node)) {
            if (parent[neighbor] < 0) {
                parent[neighbor] = node;
                queue.push_back(neighbor);
            }
        }
    }
    if (parent[dest] < 0) {
        return {};
    }
    std::vector<int> path = {dest};
    while (path.back() != source) {
        path.push_back(parent[path.back()]);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::pair<int, int>> Router::ReleaseValve() const {
    auto normalized = [](int x, int y) { return x < y ? std::make_pair(x, y) : std::make_pair(y, x); };
    int target = -1;
    std::int64_t target_cost = 0;
    for (int gate : front_) {
        const std::int64_t cost = NodeCost(gate);
        if (target < 0 || cost < target_cost) {
            target = gate;
            target_cost = cost;
        }
    }
    std::vector<std::pair<int, int>> out;
    if (nq3_[target] < 0) {
        const auto path = ShortestPath(pi_[nq1_[target]], pi_[nq2_[target]]);
        if (path.empty()) {
            throw std::invalid_argument(kDisconnectedMessage);
        }
        const int k = static_cast<int>(path.size()) - 1;
        const int left = k / 2;
        const int right = (k - 1) - left;
        for (int i = 0; i < left; ++i) {
            out.push_back(normalized(path[i], path[i + 1]));
        }
        for (int i = 0; i < right; ++i) {
            out.push_back(normalized(path[k - i], path[k - i - 1]));
        }
        return out;
    }

    const int toks[3] = {nq1_[target], nq2_[target], nq3_[target]};
    int pos[3] = {pi_[toks[0]], pi_[toks[1]], pi_[toks[2]]};
    // middle：三者中到另外两个距离和最小者（首个最小值）
    int middle = 0;
    std::int64_t middle_sum = 0;
    for (int m = 0; m < 3; ++m) {
        std::int64_t sum = 0;
        for (int o = 0; o < 3; ++o) {
            if (o != m) {
                sum += D(pos[m], pos[o]);
            }
        }
        if (m == 0 || sum < middle_sum) {
            middle = m;
            middle_sum = sum;
        }
    }
    std::vector<int> ends;
    for (int o = 0; o < 3; ++o) {
        if (o != middle) {
            ends.push_back(o);
        }
    }
    std::stable_sort(ends.begin(), ends.end(),
                     [&](int a, int b) { return D(pos[a], pos[middle]) < D(pos[b], pos[middle]); });
    std::vector<int> occupant = rpi_;
    auto connected = [&]() { return g_.Cost3(pos[0], pos[1], pos[2]) == 1; };
    for (int end : ends) {
        while (!connected() && D(pos[end], pos[middle]) > 1) {
            const auto path = ShortestPath(pos[end], pos[middle]);
            if (path.empty()) {
                throw std::invalid_argument(kDisconnectedMessage);
            }
            const int x = path[0];
            const int y = path[1];
            const int tx = occupant[x];
            const int ty = occupant[y];
            occupant[x] = ty;
            occupant[y] = tx;
            pos[end] = y;
            for (int i = 0; i < 3; ++i) {
                if (toks[i] == ty) {
                    pos[i] = x;
                }
            }
            out.push_back(normalized(x, y));
        }
    }
    return out;
}

RouteResult Router::Run() {
    {
        std::vector<int> initial;
        for (int node = 0; node < static_cast<int>(indeg_.size()); ++node) {
            if (indeg_[node] == 0) {
                initial.push_back(node);
            }
        }
        Drain(initial);
    }

    std::vector<std::pair<int, int>> pending;
    std::vector<int> ready;
    while (!front_.empty()) {
        RebuildExtendedSet();

        ready.clear();
        if (!mcands_.empty()) {
            TryMirrors(ready);
            if (!ready.empty()) {
                Retire(ready);
                std::fill(decay_.begin(), decay_.end(), 1.0);
                continue;
            }
        }

        pending.clear();
        std::int64_t pending_cost = 0;
        int steps = 0;
        while (ready.empty()) {
            if (static_cast<int>(pending.size()) >= valve_) {
                for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
                    ApplySwap(it->first, it->second, ready);
                }
                ready.clear();
                pending = ReleaseValve();
                pending_cost = 0;
                for (const auto& [x, y] : pending) {
                    pending_cost += ApplySwap(x, y, ready);
                }
                break;
            }

            std::int64_t f_sum = 0;
            bool front3_nodes = false;
            ++stamp_;
            cands_.clear();
            auto add_edges = [&](int p) {
                for (int e : g_.EdgesAt(p)) {
                    if (edge_stamp_[e] != stamp_) {
                        edge_stamp_[e] = stamp_;
                        cands_.push_back(e);
                    }
                }
            };
            for (int gate : front_) {
                const int c = nq3_[gate];
                const int pa = pi_[nq1_[gate]];
                const int pb = pi_[nq2_[gate]];
                if (c < 0) {
                    f_sum += D(pa, pb);
                    add_edges(pa);
                    add_edges(pb);
                } else {
                    const int pc = pi_[c];
                    f_sum += g_.Cost3(pa, pb, pc);
                    add_edges(pa);
                    add_edges(pb);
                    add_edges(pc);
                    front3_nodes = true;
                }
            }
            if (cands_.empty()) {
                throw std::invalid_argument(kDisconnectedMessage);
            }
            std::int64_t e_sum = 0;
            for (int gate : ext2_) {
                e_sum += D(pi_[nq1_[gate]], pi_[nq2_[gate]]);
            }
            for (int gate : ext3_) {
                e_sum += g_.Cost3(pi_[nq1_[gate]], pi_[nq2_[gate]], pi_[nq3_[gate]]);
            }
            const double e_scale = ext_.empty() ? 0.0 : w_ / static_cast<double>(ext_.size());

            double best_score = std::numeric_limits<double>::infinity();
            best_.clear();
            for (int e : cands_) {
                const int x = g_.EdgeX(e);
                const int y = g_.EdgeY(e);
                const int tx = rpi_[x];
                const int ty = rpi_[y];
                std::int64_t df = 0;
                std::int64_t de = 0;
                if (has_part_[tx]) {
                    const auto& entry = Table(tx)[g_.EdgeIndexOfYAtX(e)];
                    df = entry.first;
                    de = entry.second;
                    if (has_part_[ty]) {
                        const auto& entry2 = Table(ty)[g_.EdgeIndexOfXAtY(e)];
                        df += entry2.first;
                        de += entry2.second;
                    }
                } else if (has_part_[ty]) {
                    const auto& entry = Table(ty)[g_.EdgeIndexOfXAtY(e)];
                    df = entry.first;
                    de = entry.second;
                }
                if (front3_nodes && (!front3_[tx].empty() || !front3_[ty].empty())) {
                    df += Delta3(front3_[tx], front3_[ty], tx, ty, x, y);
                }
                if (!ext3_.empty() && (!ext3_of_[tx].empty() || !ext3_of_[ty].empty())) {
                    de += Delta3(ext3_of_[tx], ext3_of_[ty], tx, ty, x, y);
                }
                // Python：score = f_sum + df + (e_sum + de) * e_scale（整数部分精确，再转为 double）
                const double front_part = static_cast<double>(f_sum + df);
                const double ext_part = static_cast<double>(e_sum + de) * e_scale;
                double score = front_part + ext_part;
                const double dec = decay_[x] > decay_[y] ? decay_[x] : decay_[y];
                score *= dec;
                const int op = last_op_[x];
                if (use_fusion_ && op >= 0 && op == last_op_[y]) {
                    if (!mirror_on_) {
                        score *= fusion_;
                    } else if (mcost_[op] >= 0) {
                        // 镜像感知：按该块的精确镜像代价折扣
                        score *= mfusion_[mcost_[op]];
                    }
                }
                if (score < best_score - kTieEps) {
                    best_score = score;
                    best_.clear();
                    best_.push_back(e);
                } else if (score <= best_score + kTieEps) {
                    best_.push_back(e);
                }
            }

            const int chosen = best_.size() == 1 ? best_[0] : best_[rng_.RandBelow(best_.size())];
            const int x = g_.EdgeX(chosen);
            const int y = g_.EdgeY(chosen);
            pending_cost += ApplySwap(x, y, ready);
            pending.emplace_back(x, y);
            ++steps;
            if (steps % reset_interval_ == 0) {
                std::fill(decay_.begin(), decay_.end(), 1.0);
            } else {
                decay_[x] += delta_;
                decay_[y] += delta_;
            }
        }

        num_swaps_ += static_cast<std::int64_t>(pending.size());
        swap_cost_ += pending_cost;
        if (record_) {
            for (const auto& [x, y] : pending) {
                events_.push_back(-1 - (static_cast<std::int64_t>(x) * n_ + y));
                const int lv = std::max(level_[x], level_[y]) + 1;
                level_[x] = level_[y] = lv;
            }
        }
        Retire(ready);
        std::fill(decay_.begin(), decay_.end(), 1.0);
    }

    RouteResult result;
    result.num_swaps = num_swaps_;
    result.swap_cost = swap_cost_;
    result.num_mirrors = num_mirrors_;
    result.final_pi = pi_;
    result.events = std::move(events_);
    result.depth = level_.empty() ? 0 : *std::max_element(level_.begin(), level_.end());
    return result;
}

}  // namespace

RouteResult RouteDag(const RoutingGraph& graph, const RoutingDag& dag, const RouteOptions& options,
                     const std::vector<int>& layout, bool forward, bool record, CPythonRandom& rng) {
    if (dag.num_nodes() > 0) {
        for (int i = 0; i < dag.num_nodes(); ++i) {
            for (int q : {dag.nq1()[i], dag.nq2()[i], dag.nq3()[i]}) {
                if (q >= graph.num_physical()) {
                    throw std::invalid_argument("routing DAG qubit out of range");
                }
            }
        }
    }
    Router router(graph, dag, options, layout, forward, record, rng);
    return router.Run();
}

}  // namespace quspect::native::transpiler
