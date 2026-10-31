// 一般 SU(4) 块线路的截断 rollout 路由，见 su4_routing.hpp。

#include "quspect_native/transpiler/su4_routing.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>

namespace quspect::native::transpiler {

namespace {

std::uint64_t SplitMix(std::uint64_t& state) {
    std::uint64_t x = (state += 0x9E3779B97F4A7C15ULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

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
    g.adj = adjacency;
    const int n = g.n;
    constexpr int kInf = std::numeric_limits<int>::max();
    g.dist.assign(static_cast<std::size_t>(n) * n, kInf);
    for (int s = 0; s < n; ++s) {
        for (int v : g.adj[s]) {
            if (v < 0 || v >= n || v == s) {
                throw std::invalid_argument("adjacency entry out of range");
            }
        }
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

class Su4Router {
 public:
    Su4Router(const Graph& g, const std::vector<std::pair<int, int>>& blocks, const Su4RouteOptions& options)
        : g_(g), blocks_(blocks), opt_(options), tq_(g.n) {
        for (std::size_t b = 0; b < blocks_.size(); ++b) {
            tq_[blocks_[b].first].push_back(static_cast<int>(b));
            tq_[blocks_[b].second].push_back(static_cast<int>(b));
        }
        const std::size_t nb = blocks_.size();
        in_f_.assign(nb, 0);
        in_x_.assign(nb, 0);
    }

    // 从 pos（比特 -> 物理）出发路由整条线路，pos 变为最终布局；返回付费 SWAP 数。
    std::int64_t Run(std::vector<int>& pos, std::uint64_t seed, std::vector<std::int64_t>* events,
                     std::int64_t* flips) {
        const int n = g_.n;
        pos_ = &pos;
        tok_.assign(n, -1);
        for (int t = 0; t < n; ++t) {
            tok_[pos[t]] = t;
        }
        ptr_.assign(n, 0);
        decay_.assign(n, 1.0);
        events_ = events;
        flips_ = 0;
        paid_ = 0;
        rng_ = seed;
        done_ = 0;
        last_pair_ = {-1, -1};
        busy_.assign(n, 0);
        depth_ = 0;
        for (int t = 0; t < n; ++t) {
            TryExecute(t);
        }
        Rebuild();
        int since_reset = 0;
        int no_progress = 0;
        const int total = static_cast<int>(blocks_.size());
        std::vector<std::pair<int, int>> cands;
        std::vector<std::pair<int, int>> cands2;
        std::vector<std::pair<double, int>> scored;
        std::vector<int> ties;
        while (done_ < total) {
            if (no_progress > 10 * n) {
                ReleaseValve();
                no_progress = 0;
                continue;
            }
            Candidates(&cands);
            scored.clear();
            for (std::size_t i = 0; i < cands.size(); ++i) {
                const auto [u, v] = cands[i];
                const auto [df, dx] = Delta(u, v);
                const double h = std::max(decay_[u], decay_[v]) * Score(sum_f_ + df, sum_x_ + dx);
                scored.emplace_back(h, static_cast<int>(i));
            }
            std::vector<double> total_score(cands.size(), std::numeric_limits<double>::infinity());
            if (opt_.width <= 1) {
                for (const auto& [h, i] : scored) {
                    total_score[i] = h;
                }
            } else {
                const int k = std::min<int>(opt_.width, static_cast<int>(scored.size()));
                std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
                for (int r = 0; r < k; ++r) {
                    const auto [h1, i] = scored[r];
                    const auto [u, v] = cands[i];
                    const int x = tok_[u];
                    const int y = tok_[v];
                    const auto [df, dx] = Delta(u, v);
                    sum_f_ += df;
                    sum_x_ += dx;
                    SwapTokens(u, v);
                    double b2;
                    if (Adjacent(tok_f_[x]) || Adjacent(tok_f_[y])) {
                        b2 = Score(sum_f_, sum_x_) - opt_.exe_bonus;
                    } else {
                        b2 = std::numeric_limits<double>::infinity();
                        Candidates(&cands2);
                        for (const auto [u2, v2] : cands2) {
                            const auto [df2, dx2] = Delta(u2, v2);
                            b2 = std::min(b2, Score(sum_f_ + df2, sum_x_ + dx2));
                        }
                    }
                    SwapTokens(u, v);
                    sum_f_ -= df;
                    sum_x_ -= dx;
                    total_score[i] = h1 + b2;
                }
            }
            double best = std::numeric_limits<double>::infinity();
            for (double s : total_score) {
                best = std::min(best, s);
            }
            ties.clear();
            std::int64_t earliest = std::numeric_limits<std::int64_t>::max();
            for (std::size_t i = 0; i < cands.size(); ++i) {
                if (total_score[i] <= best + 1e-12) {
                    const auto [u, v] = cands[i];
                    const std::int64_t start = opt_.depth_tiebreak ? std::max(busy_[u], busy_[v]) : 0;
                    if (start < earliest) {
                        earliest = start;
                        ties.clear();
                    }
                    if (start == earliest) {
                        ties.push_back(static_cast<int>(i));
                    }
                }
            }
            const auto [u, v] = cands[ties[SplitMix(rng_) % ties.size()]];
            PaidSwap(u, v);
            decay_[u] += opt_.decay_delta;
            decay_[v] += opt_.decay_delta;
            if (++since_reset % std::max(1, opt_.decay_reset_interval) == 0) {
                std::fill(decay_.begin(), decay_.end(), 1.0);
            }
            const int before = done_;
            const int x = tok_[u];
            const int y = tok_[v];
            TryExecute(x);
            TryExecute(y);
            if (done_ != before) {
                Rebuild();
                std::fill(decay_.begin(), decay_.end(), 1.0);
                since_reset = 0;
                no_progress = 0;
            } else {
                ++no_progress;
            }
        }
        if (flips != nullptr) {
            *flips = flips_;
        }
        return paid_;
    }

    std::int64_t depth() const {
        return depth_;
    }

 private:
    using Pos = std::vector<int>;

    int Front(int t) const {
        return ptr_[t] < static_cast<int>(tq_[t].size()) ? tq_[t][ptr_[t]] : -1;
    }
    int Next(int t, int k) const {
        const int i = ptr_[t] + k;
        return i < static_cast<int>(tq_[t].size()) ? tq_[t][i] : -1;
    }
    bool Ready(int b) const {
        return b >= 0 && Front(blocks_[b].first) == b && Front(blocks_[b].second) == b;
    }
    bool Adjacent(int b) const {
        return b >= 0 && g_.D((*pos_)[blocks_[b].first], (*pos_)[blocks_[b].second]) == 1;
    }
    double Score(std::int64_t sf, std::int64_t sx) const {
        double h = static_cast<double>(sf) / static_cast<double>(std::max<std::size_t>(1, f_.size()));
        if (!x_.empty()) {
            h += opt_.ext_weight * static_cast<double>(sx) / static_cast<double>(x_.size());
        }
        return h;
    }

    void SwapTokens(int u, int v) {
        const int a = tok_[u];
        const int b = tok_[v];
        tok_[u] = b;
        tok_[v] = a;
        (*pos_)[a] = v;
        (*pos_)[b] = u;
    }

    void Emit(std::int64_t code) {
        last_pair_ = {-1, -1};
        if (events_ != nullptr) {
            events_->push_back(code);
        }
    }
    void EmitSwap(int u, int v) {
        const int x = std::min(u, v);
        const int y = std::max(u, v);
        Emit(-1 - (static_cast<std::int64_t>(x) * g_.n + y));
    }
    // 路由 SWAP；紧跟在同一对比特的块之后（中间没有其他事件）时它被块吸收，计为免费翻转。
    void PaidSwap(int u, int v) {
        const bool free = last_pair_ == std::array<int, 2>{std::min(u, v), std::max(u, v)};
        SwapTokens(u, v);
        EmitSwap(u, v);
        if (free) {
            ++flips_;
        } else {
            ++paid_;
            Occupy(u, v);
        }
    }

    // 物理比特 u、v 上的一个两比特操作（块或付费 SWAP）占用一个时间层。
    void Occupy(int u, int v) {
        const std::int64_t t = std::max(busy_[u], busy_[v]) + 1;
        busy_[u] = t;
        busy_[v] = t;
        depth_ = std::max(depth_, t);
    }

    // 后两个块的伙伴距离（加权），用于执行后的免费翻转决策。
    std::int64_t FlipCost(int a, int b) const {
        std::int64_t s = 0;
        for (int t : {a, b}) {
            for (int k = 0; k < 2; ++k) {
                const int nb = Next(t, k);
                if (nb < 0) {
                    break;
                }
                const int o = blocks_[nb].first == t ? blocks_[nb].second : blocks_[nb].first;
                s += (k == 0 ? 4 : 1) * g_.D((*pos_)[t], (*pos_)[o]);
            }
        }
        return s;
    }

    // 若比特 t 的前沿块就绪且相邻就执行（连锁执行随后就绪的块）。
    void TryExecute(int t) {
        std::vector<int> work{t};
        while (!work.empty()) {
            const int s = work.back();
            work.pop_back();
            const int b = Front(s);
            if (!Ready(b) || !Adjacent(b)) {
                continue;
            }
            const auto [a, c] = blocks_[b];
            Emit(b);
            Occupy((*pos_)[a], (*pos_)[c]);
            ++ptr_[a];
            ++ptr_[c];
            ++done_;
            const std::int64_t keep = FlipCost(a, c);
            const int pa = (*pos_)[a];
            const int pc = (*pos_)[c];
            SwapTokens(pa, pc);
            if (FlipCost(a, c) < keep) {
                EmitSwap(pa, pc);
                ++flips_;
            } else {
                SwapTokens(pa, pc);
                last_pair_ = {std::min(pa, pc), std::max(pa, pc)};
            }
            work.push_back(a);
            work.push_back(c);
        }
    }

    void Rebuild() {
        const int n = g_.n;
        for (int b : f_) {
            in_f_[b] = 0;
        }
        for (int b : x_) {
            in_x_[b] = 0;
        }
        f_.clear();
        x_.clear();
        tok_f_.assign(n, -1);
        tok_x_.assign(n, {-1, -1});
        for (int t = 0; t < n; ++t) {
            const int b = Front(t);
            if (Ready(b)) {
                tok_f_[t] = b;
                if (!in_f_[b]) {
                    in_f_[b] = 1;
                    f_.push_back(b);
                }
            }
        }
        for (int t = 0; t < n; ++t) {
            const int b = Front(t);
            if (b < 0) {
                continue;
            }
            const int e = Ready(b) ? Next(t, 1) : b;
            if (e >= 0 && !in_f_[e] && !in_x_[e]) {
                in_x_[e] = 1;
                x_.push_back(e);
            }
        }
        for (int e : x_) {
            for (int t : {blocks_[e].first, blocks_[e].second}) {
                auto& slot = tok_x_[t];
                if (slot[0] < 0) {
                    slot[0] = e;
                } else if (slot[1] < 0 && slot[0] != e) {
                    slot[1] = e;
                }
            }
        }
        sum_f_ = 0;
        sum_x_ = 0;
        for (int b : f_) {
            sum_f_ += g_.D((*pos_)[blocks_[b].first], (*pos_)[blocks_[b].second]) - 1;
        }
        for (int e : x_) {
            sum_x_ += g_.D((*pos_)[blocks_[e].first], (*pos_)[blocks_[e].second]);
        }
    }

    // 交换物理比特 u、v 上的比特后 (Σ_F, Σ_X) 的变化。
    std::pair<std::int64_t, std::int64_t> Delta(int u, int v) const {
        const int x = tok_[u];
        const int y = tok_[v];
        std::array<int, 6> aff{};
        int na = 0;
        auto add = [&](int b) {
            if (b < 0) {
                return;
            }
            for (int i = 0; i < na; ++i) {
                if (aff[i] == b) {
                    return;
                }
            }
            aff[na++] = b;
        };
        add(tok_f_[x]);
        add(tok_f_[y]);
        add(tok_x_[x][0]);
        add(tok_x_[x][1]);
        add(tok_x_[y][0]);
        add(tok_x_[y][1]);
        auto at = [&](int t) {
            if (t == x) {
                return v;
            }
            if (t == y) {
                return u;
            }
            return (*pos_)[t];
        };
        std::int64_t df = 0;
        std::int64_t dx = 0;
        for (int i = 0; i < na; ++i) {
            const auto [p, q] = blocks_[aff[i]];
            const std::int64_t d = g_.D(at(p), at(q)) - g_.D((*pos_)[p], (*pos_)[q]);
            if (in_f_[aff[i]]) {
                df += d;
            } else {
                dx += d;
            }
        }
        return {df, dx};
    }

    void Candidates(std::vector<std::pair<int, int>>* out) const {
        out->clear();
        for (int b : f_) {
            for (int t : {blocks_[b].first, blocks_[b].second}) {
                const int u = (*pos_)[t];
                for (int v : g_.adj[u]) {
                    out->emplace_back(std::min(u, v), std::max(u, v));
                }
            }
        }
        std::sort(out->begin(), out->end());
        out->erase(std::unique(out->begin(), out->end()), out->end());
    }

    // 长时间没有块执行：把最近的前沿块沿最短路径移到相邻。
    void ReleaseValve() {
        int best = -1;
        int bd = std::numeric_limits<int>::max();
        for (int b : f_) {
            const int d = g_.D((*pos_)[blocks_[b].first], (*pos_)[blocks_[b].second]);
            if (d < bd) {
                bd = d;
                best = b;
            }
        }
        const auto [a, c] = blocks_[best];
        while (g_.D((*pos_)[a], (*pos_)[c]) > 1) {
            const int u = (*pos_)[a];
            int next = -1;
            for (int w : g_.adj[u]) {
                if (g_.D(w, (*pos_)[c]) < g_.D(u, (*pos_)[c])) {
                    next = w;
                    break;
                }
            }
            PaidSwap(u, next);
        }
        TryExecute(a);
        Rebuild();
        std::fill(decay_.begin(), decay_.end(), 1.0);
    }

    const Graph& g_;
    std::vector<std::pair<int, int>> blocks_;
    Su4RouteOptions opt_;
    std::vector<std::vector<int>> tq_;
    Pos* pos_ = nullptr;
    std::vector<int> tok_;
    std::vector<int> ptr_;
    std::vector<double> decay_;
    std::vector<std::int64_t>* events_ = nullptr;
    std::int64_t flips_ = 0;
    std::int64_t paid_ = 0;
    std::uint64_t rng_ = 1;
    int done_ = 0;
    // 最后一个事件是这对物理比特上的块执行时为该对，否则 {-1, -1}。
    std::array<int, 2> last_pair_{-1, -1};
    // 每个物理比特最后一个两比特操作结束的时间层，与其最大值（深度代理）。
    std::vector<std::int64_t> busy_;
    std::int64_t depth_ = 0;
    std::vector<int> f_;
    std::vector<int> x_;
    std::vector<char> in_f_;
    std::vector<char> in_x_;
    std::vector<int> tok_f_;
    std::vector<std::array<int, 2>> tok_x_;
    std::int64_t sum_f_ = 0;
    std::int64_t sum_x_ = 0;
};

}  // namespace

Su4RouteResult RouteSu4Blocks(const std::vector<std::vector<int>>& adjacency,
                              const std::vector<std::pair<int, int>>& blocks, const std::vector<int>& layout,
                              const Su4RouteOptions& options) {
    const Graph g = MakeGraph(adjacency);
    const int n = g.n;
    if (static_cast<int>(layout.size()) != n) {
        throw std::invalid_argument("layout must assign every physical qubit");
    }
    std::vector<char> seen(n, 0);
    for (int p : layout) {
        if (p < 0 || p >= n || seen[p]) {
            throw std::invalid_argument("layout must be a permutation");
        }
        seen[p] = 1;
    }
    for (auto [a, b] : blocks) {
        if (a < 0 || b < 0 || a >= n || b >= n || a == b) {
            throw std::invalid_argument("block qubits must be distinct and in range");
        }
    }
    std::vector<std::pair<int, int>> reversed(blocks.rbegin(), blocks.rend());
    Su4Router forward(g, blocks, options);
    Su4Router backward(g, reversed, options);
    std::uint64_t rng = options.seed;
    std::vector<std::vector<int>> starts{layout};
    std::vector<int> cur = layout;
    for (int it = 0; it < options.layout_iters; ++it) {
        forward.Run(cur, SplitMix(rng), nullptr, nullptr);
        backward.Run(cur, SplitMix(rng), nullptr, nullptr);
        starts.push_back(cur);
    }
    Su4RouteResult best;
    best.paid = -1;
    for (const auto& start : starts) {
        std::vector<int> pos = start;
        std::vector<std::int64_t> events;
        std::int64_t flips = 0;
        const std::int64_t paid = forward.Run(pos, SplitMix(rng), &events, &flips);
        const std::int64_t depth = forward.depth();
        if (best.paid < 0 || paid < best.paid || (paid == best.paid && depth < best.depth)) {
            best.paid = paid;
            best.depth = depth;
            best.flips = flips;
            best.layout = start;
            best.events = std::move(events);
        }
    }
    return best;
}

}  // namespace quspect::native::transpiler
