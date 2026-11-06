// 全局束搜索路由，见 beam_routing.hpp。

#include "quspect_native/transpiler/beam_routing.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace quspect::native::transpiler {

namespace {

struct Graph {
    int n = 0;
    std::vector<std::vector<int>> adj;
    std::vector<int> dist;  // n * n

    int D(int a, int b) const {
        return dist[static_cast<std::size_t>(a) * n + b];
    }
};

Graph MakeGraph(const std::vector<std::vector<int>>& adjacency) {
    Graph g;
    g.n = static_cast<int>(adjacency.size());
    const int n = g.n;
    g.adj.resize(n);
    for (int s = 0; s < n; ++s) {
        for (int v : adjacency[s]) {
            if (v < 0 || v >= n || v == s) {
                throw std::invalid_argument("adjacency entry out of range");
            }
            g.adj[s].push_back(v);
        }
        std::sort(g.adj[s].begin(), g.adj[s].end());
        g.adj[s].erase(std::unique(g.adj[s].begin(), g.adj[s].end()), g.adj[s].end());
    }
    constexpr int kInf = std::numeric_limits<int>::max();
    g.dist.assign(static_cast<std::size_t>(n) * n, kInf);
    for (int s = 0; s < n; ++s) {
        std::deque<int> queue{s};
        g.dist[static_cast<std::size_t>(s) * n + s] = 0;
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop_front();
            const int du = g.dist[static_cast<std::size_t>(s) * n + u];
            for (int v : g.adj[u]) {
                int& dv = g.dist[static_cast<std::size_t>(s) * n + v];
                if (dv == kInf) {
                    dv = du + 1;
                    queue.push_back(v);
                }
            }
        }
        for (int v = 0; v < n; ++v) {
            if (g.dist[static_cast<std::size_t>(s) * n + v] == kInf) {
                throw std::invalid_argument("coupling graph must be connected");
            }
        }
    }
    return g;
}

// 回溯树的一个节点：父节点与这一步插入的 SWAP。
struct TraceNode {
    int parent;
    int x, y;
};

struct State {
    std::vector<int> l2p, p2l, ptr;
    // 衰减：自上次执行块以来每个物理比特参与的 SWAP 数（封顶 255），与累计惩罚
    std::vector<std::uint8_t> uses;
    double penalty = 0.0;
    int executed = 0;
    int node = -1;  // 回溯树中的节点（-1：根）
    int last_x = -1, last_y = -1;
    double score = 0.0;
    std::uint64_t hash = 0;
    std::size_t order = 0;  // 生成顺序，排序并列时的确定性次序
};

class BeamRouter {
 public:
    BeamRouter(const Graph& g, const std::vector<std::pair<int, int>>& blocks, const BeamRouteOptions& opt)
        : g_(g), blocks_(blocks), opt_(opt), chain_(g.n), pos_(blocks.size()), rng_(opt.seed) {
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            const auto [a, c] = blocks_[b];
            pos_[b] = {static_cast<int>(chain_[a].size()), static_cast<int>(chain_[c].size())};
            chain_[a].push_back(static_cast<int>(b));
            chain_[c].push_back(static_cast<int>(b));
        }
        for (int q = 0; q < g_.n; ++q) {
            if (!chain_[q].empty()) {
                active_.push_back(q);
            }
        }
        gpow_.assign(static_cast<std::size_t>(opt_.ext_depth) + 2, 1.0);
        for (std::size_t k = 1; k < gpow_.size(); ++k) {
            gpow_[k] = gpow_[k - 1] * opt_.gamma;
        }
    }

    // 从 ``start`` 路由全部块；返回 SWAP 数（超过 max_swaps 时返回 -1），填写事件与终态布局。
    std::int64_t Route(const std::vector<int>& start, std::vector<std::int64_t>* events, std::vector<int>* final_l2p) {
        const int total = static_cast<int>(blocks_.size());
        std::vector<TraceNode> trace;
        State s0 = Initial(start);
        Execute(s0, active_, nullptr);
        std::vector<State> beam{s0};
        int done = -1;
        int stall = 0;
        int best_exec = s0.executed;
        std::int64_t steps = 0;
        std::size_t order = 0;
        std::vector<State> kids;
        std::vector<std::pair<int, int>> cands;
        // 已进入过束的状态（布局 + 进度）：之后再到达同一状态只会多用 SWAP，直接丢弃，
        // 这样束不会在等价状态之间来回绕圈
        std::unordered_set<std::uint64_t> visited{Hash(s0)};
        std::vector<char> seen(g_.n, 0);
        while (true) {
            for (std::size_t i = 0; i < beam.size(); ++i) {
                if (beam[i].executed == total) {
                    done = static_cast<int>(i);
                    break;
                }
            }
            if (done >= 0) {
                break;
            }
            kids.clear();
            for (const State& s : beam) {
                cands.clear();
                for (int q : active_) {
                    const int b = Next(s, q);
                    if (b < 0 || !IsFront(s, b)) {
                        continue;
                    }
                    const int p = s.l2p[q];
                    if (seen[p]) {
                        continue;
                    }
                    seen[p] = 1;
                    for (int nb : g_.adj[p]) {
                        const int x = std::min(p, nb);
                        const int y = std::max(p, nb);
                        if (x == s.last_x && y == s.last_y) {
                            continue;  // 不立即撤销上一个 SWAP
                        }
                        cands.emplace_back(x, y);
                    }
                }
                for (int q : active_) {
                    seen[s.l2p[q]] = 0;
                }
                std::sort(cands.begin(), cands.end());
                cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
                for (const auto [x, y] : cands) {
                    State c = s;
                    std::array<int, 2> touched{};
                    const int nt = Swap(c, x, y, &touched);
                    c.last_x = x;
                    c.last_y = y;
                    const int before = c.executed;
                    Execute(c, std::vector<int>(touched.begin(), touched.begin() + nt), nullptr);
                    if (opt_.decay > 0.0) {
                        if (c.executed > before) {
                            std::fill(c.uses.begin(), c.uses.end(), std::uint8_t{0});
                            c.penalty = 0.0;
                        } else {
                            // 没有进展时反复动同一批比特要付出代价（SABRE 的衰减），否则在拥挤的
                            // 耦合图上束会在代价不变的 SWAP 之间来回漂移
                            c.penalty += opt_.decay * static_cast<double>(c.uses[x] + c.uses[y]);
                            c.uses[x] = static_cast<std::uint8_t>(std::min(255, c.uses[x] + 1));
                            c.uses[y] = static_cast<std::uint8_t>(std::min(255, c.uses[y] + 1));
                        }
                    }
                    c.score = Cost(c) + c.penalty + 1e-9 * static_cast<double>(rng_() % 1000);
                    c.hash = Hash(c);
                    c.order = order++;
                    kids.push_back(std::move(c));
                }
            }
            if (kids.empty()) {
                throw std::invalid_argument("no routing SWAP is available");
            }
            std::sort(kids.begin(), kids.end(), [](const State& a, const State& b) {
                return a.score != b.score ? a.score < b.score : a.order < b.order;
            });
            std::vector<State> next;
            next.reserve(static_cast<std::size_t>(opt_.beam));
            for (State& k : kids) {
                if (static_cast<int>(next.size()) >= opt_.beam) {
                    break;
                }
                if (!visited.insert(k.hash).second) {
                    continue;
                }
                // 只有留在束里的状态进入回溯树
                trace.push_back({k.node, k.last_x, k.last_y});
                k.node = static_cast<int>(trace.size()) - 1;
                next.push_back(std::move(k));
            }
            if (next.empty()) {
                // 所有子状态都已到达过：从当前最好的状态强制接通一个前沿块
                State s = beam[0];
                steps += ReleaseValve(s, &trace);
                visited.insert(Hash(s));
                beam.assign(1, std::move(s));
                stall = 0;
                best_exec = beam[0].executed;
                if (opt_.max_swaps > 0 && steps > opt_.max_swaps) {
                    return -1;
                }
                continue;
            }
            beam.swap(next);
            ++steps;
            if (opt_.max_swaps > 0 && steps > opt_.max_swaps) {
                return -1;  // 超过 SWAP 上限：放弃
            }
            int most = 0;
            for (const State& s : beam) {
                most = std::max(most, s.executed);
            }
            if (most > best_exec) {
                best_exec = most;
                stall = 0;
            } else if (++stall > 4 * g_.n) {
                State s = beam[0];
                steps += ReleaseValve(s, &trace);
                visited.insert(Hash(s));
                beam.assign(1, std::move(s));
                stall = 0;
                best_exec = beam[0].executed;
                if (opt_.max_swaps > 0 && steps > opt_.max_swaps) {
                    return -1;
                }
            }
        }
        // 回溯 SWAP 序列，重放一遍得到事件
        std::vector<std::pair<int, int>> swaps;
        for (int n = beam[done].node; n >= 0; n = trace[n].parent) {
            swaps.emplace_back(trace[n].x, trace[n].y);
        }
        std::reverse(swaps.begin(), swaps.end());
        State s = Initial(start);
        events->clear();
        std::vector<int> rec;
        Execute(s, active_, &rec);
        events->insert(events->end(), rec.begin(), rec.end());
        for (const auto [x, y] : swaps) {
            rec.clear();
            std::array<int, 2> touched{};
            const int nt = Swap(s, x, y, &touched);
            events->push_back(-1 - (static_cast<std::int64_t>(x) * g_.n + y));
            Execute(s, std::vector<int>(touched.begin(), touched.begin() + nt), &rec);
            events->insert(events->end(), rec.begin(), rec.end());
        }
        if (s.executed != total) {
            throw std::logic_error("beam routing replay did not execute every block");
        }
        *final_l2p = s.l2p;
        return static_cast<std::int64_t>(swaps.size());
    }

 private:
    State Initial(const std::vector<int>& start) const {
        State s;
        s.l2p = start;
        s.p2l.assign(g_.n, -1);
        for (int q = 0; q < g_.n; ++q) {
            s.p2l[start[q]] = q;
        }
        s.ptr.assign(g_.n, 0);
        if (opt_.decay > 0.0) {
            s.uses.assign(g_.n, 0);
        }
        return s;
    }

    int Next(const State& s, int q) const {
        return s.ptr[q] < static_cast<int>(chain_[q].size()) ? chain_[q][s.ptr[q]] : -1;
    }

    bool IsFront(const State& s, int b) const {
        return s.ptr[blocks_[b].first] == pos_[b][0] && s.ptr[blocks_[b].second] == pos_[b][1];
    }

    // 交换物理比特 x、y 上的内容；返回受影响的有块比特数（写入 touched）。
    int Swap(State& s, int x, int y, std::array<int, 2>* touched) const {
        const int lx = s.p2l[x];
        const int ly = s.p2l[y];
        s.p2l[x] = ly;
        s.p2l[y] = lx;
        int nt = 0;
        if (lx >= 0) {
            s.l2p[lx] = y;
            if (!chain_[lx].empty()) {
                (*touched)[nt++] = lx;
            }
        }
        if (ly >= 0) {
            s.l2p[ly] = x;
            if (!chain_[ly].empty()) {
                (*touched)[nt++] = ly;
            }
        }
        return nt;
    }

    // 从 ``seeds`` 出发贪心执行所有已相邻的前沿块（rec 非空时按执行顺序记录块编号）。
    void Execute(State& s, std::vector<int> work, std::vector<int>* rec) const {
        while (!work.empty()) {
            const int q = work.back();
            work.pop_back();
            const int b = Next(s, q);
            if (b < 0 || !IsFront(s, b)) {
                continue;
            }
            const auto [a, c] = blocks_[b];
            if (g_.D(s.l2p[a], s.l2p[c]) != 1) {
                continue;
            }
            ++s.ptr[a];
            ++s.ptr[c];
            ++s.executed;
            if (rec != nullptr) {
                rec->push_back(b);
            }
            work.push_back(a);
            work.push_back(c);
        }
    }

    double Cost(const State& s) const {
        // 距离是整数：F、X 的整数部分与求和顺序无关；衰减项按固定顺序累加
        std::int64_t front = 0;
        double ext = 0.0;
        for (int q : active_) {
            const auto& ch = chain_[q];
            const int start = s.ptr[q];
            const int end = std::min(static_cast<int>(ch.size()), start + opt_.ext_depth + 1);
            for (int k = start; k < end; ++k) {
                const int b = ch[k];
                const int r = blocks_[b].first == q ? blocks_[b].second : blocks_[b].first;
                const int d = g_.D(s.l2p[q], s.l2p[r]) - 1;
                if (k == start && IsFront(s, b)) {
                    front += d;
                } else {
                    ext += gpow_[k - start] * d;
                }
            }
        }
        return 0.5 * static_cast<double>(front) + opt_.ext_weight * 0.5 * ext
               - opt_.exec_bonus * static_cast<double>(s.executed);
    }

    std::uint64_t Hash(const State& s) const {
        std::uint64_t h = 1469598103934665603ULL;
        for (int x : s.l2p) {
            h ^= static_cast<std::uint64_t>(x) + 0x9e37;
            h *= 1099511628211ULL;
        }
        for (int q : active_) {
            h ^= static_cast<std::uint64_t>(s.ptr[q]) + 0x7f4a;
            h *= 1099511628211ULL;
        }
        return h;
    }

    // 把最近的前沿块沿最短路接通并执行（排除活锁）；返回插入的 SWAP 数。
    int ReleaseValve(State& s, std::vector<TraceNode>* trace) const {
        int inserted = 0;
        int best = -1;
        int best_d = INT_MAX;
        for (int q : active_) {
            const int b = Next(s, q);
            if (b < 0 || !IsFront(s, b)) {
                continue;
            }
            const int d = g_.D(s.l2p[blocks_[b].first], s.l2p[blocks_[b].second]);
            if (d < best_d) {
                best_d = d;
                best = b;
            }
        }
        const int qa = blocks_[best].first;
        const int qb = blocks_[best].second;
        while (g_.D(s.l2p[qa], s.l2p[qb]) > 1) {
            const int pa = s.l2p[qa];
            const int pb = s.l2p[qb];
            int step = -1;
            for (int nb : g_.adj[pa]) {
                if (g_.D(nb, pb) == g_.D(pa, pb) - 1) {
                    step = nb;
                    break;
                }
            }
            std::array<int, 2> touched{};
            Swap(s, pa, step, &touched);
            trace->push_back({s.node, std::min(pa, step), std::max(pa, step)});
            s.node = static_cast<int>(trace->size()) - 1;
            s.last_x = std::min(pa, step);
            s.last_y = std::max(pa, step);
            ++inserted;
        }
        Execute(s, {qa, qb}, nullptr);
        std::fill(s.uses.begin(), s.uses.end(), std::uint8_t{0});
        s.penalty = 0.0;
        return inserted;
    }

    const Graph& g_;
    const std::vector<std::pair<int, int>>& blocks_;
    BeamRouteOptions opt_;
    std::vector<std::vector<int>> chain_;  // chain_[q] = q 上的块，按顺序
    std::vector<std::array<int, 2>> pos_;  // pos_[b] = b 在两个比特链上的位置
    std::vector<int> active_;              // 有块的比特
    std::vector<double> gpow_;
    std::mt19937_64 rng_;
};

}  // namespace

BeamRouteResult RouteBeam(const std::vector<std::vector<int>>& adjacency,
                          const std::vector<std::pair<int, int>>& blocks, const std::vector<int>& layout,
                          const BeamRouteOptions& options) {
    if (options.beam < 1 || options.ext_depth < 0 || options.layout_iters < 0 || options.max_swaps < 0
        || !(options.decay >= 0.0)) {
        throw std::invalid_argument("beam must be >= 1, ext_depth, layout_iters, max_swaps and decay >= 0");
    }
    const Graph g = MakeGraph(adjacency);
    const int n = g.n;
    if (static_cast<int>(layout.size()) != n) {
        throw std::invalid_argument("layout must assign every physical qubit");
    }
    std::vector<char> used(n, 0);
    for (int p : layout) {
        if (p < 0 || p >= n || used[p]) {
            throw std::invalid_argument("layout must be a permutation");
        }
        used[p] = 1;
    }
    for (const auto& [a, b] : blocks) {
        if (a < 0 || b < 0 || a >= n || b >= n || a == b) {
            throw std::invalid_argument("block qubits must be distinct and in range");
        }
    }
    std::vector<std::pair<int, int>> reversed(blocks.rbegin(), blocks.rend());
    BeamRouter forward(g, blocks, options);
    BeamRouter backward(g, reversed, options);

    BeamRouteResult best;
    best.swaps = -1;
    std::vector<int> start = layout;
    for (int it = 0; it <= options.layout_iters; ++it) {
        std::vector<std::int64_t> events;
        std::vector<int> final_l2p;
        const std::int64_t swaps = forward.Route(start, &events, &final_l2p);
        if (swaps < 0) {
            break;
        }
        if (best.swaps < 0 || swaps < best.swaps) {
            best.swaps = swaps;
            best.layout = start;
            best.events = std::move(events);
        }
        if (it == options.layout_iters || swaps == 0) {
            break;
        }
        std::vector<std::int64_t> back_events;
        std::vector<int> back_final;
        if (backward.Route(final_l2p, &back_events, &back_final) < 0) {
            break;
        }
        start = std::move(back_final);
    }
    return best;
}

}  // namespace quspect::native::transpiler
