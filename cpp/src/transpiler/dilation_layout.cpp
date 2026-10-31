#include "quspect_native/transpiler/dilation_layout.hpp"

#include <cmath>

#include <algorithm>
#include <deque>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace quspect::native::transpiler {

namespace {

// 与 Python 端 _embedding 的常量一致。
constexpr int kSaStarts = 3;
constexpr std::int64_t kSaMovesPerQubit = 400;
constexpr std::uint64_t kLocalMove = 820;

// SplitMix64（与 Python 端 _SplitMix64 一致）。
class SplitMix64 {
 public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {
    }
    std::uint64_t Next() {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    std::uint64_t Below(std::uint64_t n) {
        return Next() % n;
    }
    double Unit() {
        return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0);
    }

 private:
    std::uint64_t state_;
};

using Neighbours = std::vector<std::vector<std::pair<int, std::int64_t>>>;

class Embedder {
 public:
    Embedder(int num_logical, const std::vector<WeightedEdge>& edges, const std::vector<std::vector<int>>& adjacency,
             const std::vector<std::vector<std::int64_t>>& distance)
        : num_logical_(num_logical)
        , num_physical_(static_cast<int>(adjacency.size()))
        , adj_(adjacency)
        , nbr_(num_logical) {
        dist_.resize(static_cast<std::size_t>(num_physical_) * num_physical_);
        for (int p = 0; p < num_physical_; ++p) {
            for (int q = 0; q < num_physical_; ++q) {
                dist_[static_cast<std::size_t>(p) * num_physical_ + q] = distance[p][q];
            }
        }
        for (const auto& e : edges) {
            nbr_[e.a].emplace_back(e.b, e.weight);
            nbr_[e.b].emplace_back(e.a, e.weight);
        }
        for (auto& row : nbr_) {
            std::sort(row.begin(), row.end());
        }
        for (int q = 0; q < num_logical_; ++q) {
            if (!nbr_[q].empty()) {
                active_.push_back(q);
            }
        }
        wdeg_.assign(num_logical_, 0);
        for (int q = 0; q < num_logical_; ++q) {
            for (const auto& [o, w] : nbr_[q]) {
                wdeg_[q] += w;
            }
        }
    }

    std::vector<DilationLayout> Run(const std::vector<WeightedEdge>& edges, std::uint64_t seed, int count) {
        if (active_.empty()) {
            return {};
        }
        std::vector<int> logical_starts;
        std::vector<int> physical_starts;
        Starts(&logical_starts, &physical_starts);
        std::vector<DilationLayout> greedy;
        for (int sl : logical_starts) {
            for (int sp : physical_starts) {
                std::vector<int> pos = Greedy(sl, sp);
                const std::int64_t cost = Cost(pos);
                greedy.push_back({cost, std::move(pos)});
            }
        }
        greedy = DistinctBest(std::move(greedy), kSaStarts);
        std::int64_t total_w = 0;
        for (const auto& e : edges) {
            total_w += e.weight;
        }
        const double t0 = std::max(1.0, static_cast<double>(total_w) / static_cast<double>(edges.size()));
        const std::int64_t moves = kSaMovesPerQubit * static_cast<std::int64_t>(active_.size());
        SplitMix64 rng(seed);
        std::vector<DilationLayout> results = greedy;
        for (const auto& g : greedy) {
            results.push_back(Anneal(g.layout, &rng, moves, t0));
        }
        return DistinctBest(std::move(results), count);
    }

 private:
    std::int64_t D(int p, int q) const {
        return dist_[static_cast<std::size_t>(p) * num_physical_ + q];
    }

    // BFS 最后到达的顶点（邻接表顺序）。
    static int BfsFar(const std::vector<std::vector<int>>& adj, int start) {
        std::vector<char> seen(adj.size(), 0);
        std::deque<int> queue = {start};
        seen[start] = 1;
        int last = start;
        while (!queue.empty()) {
            const int u = queue.front();
            queue.pop_front();
            last = u;
            for (int v : adj[u]) {
                if (!seen[v]) {
                    seen[v] = 1;
                    queue.push_back(v);
                }
            }
        }
        return last;
    }

    static int PseudoPeripheral(const std::vector<std::vector<int>>& adj, int start) {
        return BfsFar(adj, BfsFar(adj, start));
    }

    static void AddUnique(std::vector<int>* list, int value) {
        if (std::find(list->begin(), list->end(), value) == list->end()) {
            list->push_back(value);
        }
    }

    void Starts(std::vector<int>* logical, std::vector<int>* physical) const {
        std::vector<std::vector<int>> log_adj(num_logical_);
        for (int q = 0; q < num_logical_; ++q) {
            for (const auto& [o, w] : nbr_[q]) {
                log_adj[q].push_back(o);
            }
        }
        int min_q = active_[0];
        int max_q = active_[0];
        for (int q : active_) {
            if (std::make_pair(wdeg_[q], q) < std::make_pair(wdeg_[min_q], min_q)) {
                min_q = q;
            }
            if (std::make_pair(wdeg_[q], -q) > std::make_pair(wdeg_[max_q], -max_q)) {
                max_q = q;
            }
        }
        AddUnique(logical, PseudoPeripheral(log_adj, active_[0]));
        AddUnique(logical, min_q);
        AddUnique(logical, max_q);

        int min_p = 0;
        int max_p = 0;
        int centre = 0;
        std::int64_t centre_ecc = -1;
        for (int p = 0; p < num_physical_; ++p) {
            const int deg = static_cast<int>(adj_[p].size());
            if (deg < static_cast<int>(adj_[min_p].size())) {
                min_p = p;
            }
            if (deg > static_cast<int>(adj_[max_p].size())) {
                max_p = p;
            }
            std::int64_t ecc = 0;
            for (int q = 0; q < num_physical_; ++q) {
                ecc = std::max(ecc, D(p, q));
            }
            if (centre_ecc < 0 || ecc < centre_ecc) {
                centre_ecc = ecc;
                centre = p;
            }
        }
        AddUnique(physical, PseudoPeripheral(adj_, 0));
        AddUnique(physical, min_p);
        AddUnique(physical, centre);
        AddUnique(physical, max_p);
    }

    std::vector<int> Greedy(int start_logical, int start_physical) const {
        std::vector<int> pos(num_logical_, -1);
        std::vector<char> free(num_physical_, 1);
        std::vector<int> free_deg(num_physical_);
        for (int p = 0; p < num_physical_; ++p) {
            free_deg[p] = static_cast<int>(adj_[p].size());
        }
        std::vector<int> open_deg(num_logical_);
        for (int q = 0; q < num_logical_; ++q) {
            open_deg[q] = static_cast<int>(nbr_[q].size());
        }
        std::vector<std::int64_t> conn(num_logical_, 0);
        auto place = [&](int q, int p) {
            pos[q] = p;
            free[p] = 0;
            for (int r : adj_[p]) {
                --free_deg[r];
            }
            for (const auto& [o, w] : nbr_[q]) {
                --open_deg[o];
                conn[o] += w;
            }
        };
        place(start_logical, start_physical);
        std::vector<int> remaining;
        for (int q : active_) {
            if (q != start_logical) {
                remaining.push_back(q);
            }
        }
        std::vector<std::pair<int, std::int64_t>> placed;
        while (!remaining.empty()) {
            std::size_t best_i = 0;
            for (std::size_t i = 1; i < remaining.size(); ++i) {
                const int q = remaining[i];
                const int b = remaining[best_i];
                if (std::make_tuple(conn[q], wdeg_[q], -q) > std::make_tuple(conn[b], wdeg_[b], -b)) {
                    best_i = i;
                }
            }
            const int q = remaining[best_i];
            remaining.erase(remaining.begin() + static_cast<std::ptrdiff_t>(best_i));
            bool have = false;
            std::tuple<std::int64_t, std::int64_t, int> best_key{0, 0, 0};
            int best_p = -1;
            if (conn[q] == 0) {
                // 新的相互作用连通分量：放在空闲邻居最多的空闲比特上
                for (int p = 0; p < num_physical_; ++p) {
                    if (!free[p]) {
                        continue;
                    }
                    const std::tuple<std::int64_t, std::int64_t, int> key{-free_deg[p], 0, p};
                    if (!have || key < best_key) {
                        have = true;
                        best_key = key;
                        best_p = p;
                    }
                }
            } else {
                placed.clear();
                for (const auto& [o, w] : nbr_[q]) {
                    if (pos[o] >= 0) {
                        placed.emplace_back(pos[o], w);
                    }
                }
                const int need = open_deg[q];
                for (int p = 0; p < num_physical_; ++p) {
                    if (!free[p]) {
                        continue;
                    }
                    std::int64_t cost = 0;
                    for (const auto& [po, w] : placed) {
                        cost += w * (D(p, po) - 1);
                    }
                    if (have && cost > std::get<0>(best_key)) {
                        continue;
                    }
                    const int short_by = need - free_deg[p];
                    const std::tuple<std::int64_t, std::int64_t, int> key{cost + (short_by > 0 ? short_by : 0),
                                                                          free_deg[p], p};
                    if (!have || key < best_key) {
                        have = true;
                        best_key = key;
                        best_p = p;
                    }
                }
            }
            place(q, best_p);
        }
        return pos;
    }

    std::int64_t Cost(const std::vector<int>& pos) const {
        std::int64_t total = 0;
        for (int a : active_) {
            for (const auto& [b, w] : nbr_[a]) {
                if (a < b) {
                    total += w * (D(pos[a], pos[b]) - 1);
                }
            }
        }
        return total;
    }

    DilationLayout Anneal(const std::vector<int>& start, SplitMix64* rng, std::int64_t moves, double t0) const {
        std::vector<int> occ(num_physical_, -1);
        std::vector<int> cur_pos = start;
        for (int q : active_) {
            occ[cur_pos[q]] = q;
        }
        auto part = [&](int t, int p, int excl) -> std::int64_t {
            if (t < 0) {
                return 0;
            }
            std::int64_t s = 0;
            for (const auto& [o, w] : nbr_[t]) {
                if (o != excl) {
                    s += w * (D(p, cur_pos[o]) - 1);
                }
            }
            return s;
        };
        std::int64_t cur = Cost(cur_pos);
        DilationLayout best{cur, cur_pos};
        const auto n_active = static_cast<std::uint64_t>(active_.size());
        for (std::int64_t k = 0; k < moves; ++k) {
            const int t = active_[rng->Below(n_active)];
            const int p = cur_pos[t];
            const auto& edges = nbr_[t];
            int q;
            if (rng->Below(1024) < kLocalMove) {
                const int o = edges[rng->Below(edges.size())].first;
                const auto& around = adj_[cur_pos[o]];
                q = around[rng->Below(around.size())];
            } else {
                q = static_cast<int>(rng->Below(static_cast<std::uint64_t>(num_physical_)));
            }
            if (q == p) {
                continue;
            }
            const int u = occ[q];
            const std::int64_t delta = part(t, q, u) + part(u, p, t) - part(t, p, u) - part(u, q, t);
            if (delta > 0) {
                const double temp = t0 * static_cast<double>(moves - k) / static_cast<double>(moves);
                if (!(rng->Unit() < std::exp(-static_cast<double>(delta) / temp))) {
                    continue;
                }
            }
            cur_pos[t] = q;
            occ[q] = t;
            occ[p] = u;
            if (u >= 0) {
                cur_pos[u] = p;
            }
            cur += delta;
            if (cur < best.cost) {
                best.cost = cur;
                best.layout = cur_pos;
            }
        }
        return best;
    }

    static std::vector<DilationLayout> DistinctBest(std::vector<DilationLayout> items, int count) {
        std::stable_sort(items.begin(), items.end(),
                         [](const DilationLayout& x, const DilationLayout& y) { return x.cost < y.cost; });
        std::vector<DilationLayout> out;
        std::set<std::vector<int>> seen;
        for (auto& item : items) {
            if (!seen.insert(item.layout).second) {
                continue;
            }
            out.push_back(std::move(item));
            if (static_cast<int>(out.size()) == count) {
                break;
            }
        }
        return out;
    }

    int num_logical_;
    int num_physical_;
    const std::vector<std::vector<int>>& adj_;
    std::vector<std::int64_t> dist_;
    Neighbours nbr_;
    std::vector<int> active_;
    std::vector<std::int64_t> wdeg_;
};

}  // namespace

std::vector<DilationLayout> DilationLayouts(int num_logical, const std::vector<WeightedEdge>& edges,
                                            const std::vector<std::vector<int>>& adjacency,
                                            const std::vector<std::vector<std::int64_t>>& distance, std::uint64_t seed,
                                            int count) {
    const int n_phys = static_cast<int>(adjacency.size());
    if (num_logical < 0 || num_logical > n_phys || count < 0) {
        throw std::invalid_argument("DilationLayouts: invalid num_logical or count");
    }
    if (static_cast<int>(distance.size()) != n_phys) {
        throw std::invalid_argument("DilationLayouts: distance matrix size mismatch");
    }
    for (int p = 0; p < n_phys; ++p) {
        if (static_cast<int>(distance[p].size()) != n_phys) {
            throw std::invalid_argument("DilationLayouts: distance matrix size mismatch");
        }
        for (int q : adjacency[p]) {
            if (q < 0 || q >= n_phys) {
                throw std::invalid_argument("DilationLayouts: neighbour out of range");
            }
        }
    }
    for (const auto& e : edges) {
        if (e.a < 0 || e.b >= num_logical || e.a >= e.b || e.weight <= 0) {
            throw std::invalid_argument("DilationLayouts: invalid edge");
        }
    }
    if (count == 0 || edges.empty()) {
        return {};
    }
    Embedder embedder(num_logical, edges, adjacency, distance);
    return embedder.Run(edges, seed, count);
}

}  // namespace quspect::native::transpiler
