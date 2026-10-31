// 逐层匹配线路在路径上的路由：有序划分链的块坐标下降 + 奇偶换位排序，见 layer_routing.hpp。

#include "quspect_native/transpiler/layer_routing.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
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

using Block = std::array<int, 2>;  // 比特对，或 {t, -1}

class ChainSolver {
 public:
    ChainSolver(int n, const std::vector<std::vector<std::pair<int, int>>>& layers) : n_(n) {
        const int num = static_cast<int>(layers.size());
        blocks_.resize(num);
        orders_.resize(num);
        pos_.assign(num, std::vector<int>(n, 0));
        std::vector<int> seen(n, -1);
        for (int l = 0; l < num; ++l) {
            for (auto [a, b] : layers[l]) {
                if (a < 0 || b < 0 || a >= n || b >= n || a == b || seen[a] == l || seen[b] == l) {
                    throw std::invalid_argument("layer pairs must be disjoint qubit pairs in range");
                }
                seen[a] = seen[b] = l;
                blocks_[l].push_back({a, b});
            }
            for (int t = 0; t < n; ++t) {
                if (seen[t] != l) {
                    blocks_[l].push_back({t, -1});
                }
            }
            orders_[l].resize(blocks_[l].size());
            std::iota(orders_[l].begin(), orders_[l].end(), 0);
        }
    }

    int num_layers() const {
        return static_cast<int>(blocks_.size());
    }
    const std::vector<int>& order(int l) const {
        return orders_[l];
    }
    const Block& block(int l, int b) const {
        return blocks_[l][b];
    }

    void UpdatePos(int l) {
        const auto& ord = orders_[l];
        for (std::size_t i = 0; i < ord.size(); ++i) {
            for (int t : blocks_[l][ord[i]]) {
                if (t >= 0) {
                    pos_[l][t] = static_cast<int>(i);
                }
            }
        }
    }

    // 第 l - 1 → l 次转移的逆序数。
    std::int64_t TransCost(int l) const {
        if (l <= 0 || l >= num_layers()) {
            return 0;
        }
        const auto& prev = pos_[l - 1];
        const auto& ord = orders_[l];
        // 按顺序扫描 σ_l，统计已出现的 prev 秩比当前大的个数（树状数组），同块不计
        std::vector<int> bit(n_ + 1, 0);
        std::int64_t inv = 0;
        int seen = 0;
        for (int b : ord) {
            const Block& blk = blocks_[l][b];
            for (int t : blk) {
                if (t < 0) {
                    continue;
                }
                // 已出现、prev 秩严格更大的个数
                int le = 0;
                for (int i = prev[t] + 1; i > 0; i -= i & -i) {
                    le += bit[i];
                }
                inv += seen - le;
            }
            for (int t : blk) {
                if (t < 0) {
                    continue;
                }
                for (int i = prev[t] + 1; i <= n_; i += i & -i) {
                    ++bit[i];
                }
                ++seen;
            }
        }
        return inv;
    }

    std::int64_t Total() const {
        std::int64_t s = 0;
        for (int l = 1; l < num_layers(); ++l) {
            s += TransCost(l);
        }
        return s;
    }

    // 以两侧邻居为准重排第 l 层（插入式局部搜索）。
    void Reoptimize(int l, bool sort_first) {
        const int m = static_cast<int>(blocks_[l].size());
        const bool left = l > 0;
        const bool right = l + 1 < num_layers();
        w_.assign(static_cast<std::size_t>(m) * m, 0);
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) {
                if (a == b) {
                    continue;
                }
                int c = 0;
                for (int u : blocks_[l][a]) {
                    if (u < 0) {
                        continue;
                    }
                    for (int v : blocks_[l][b]) {
                        if (v < 0) {
                            continue;
                        }
                        if (left && pos_[l - 1][u] > pos_[l - 1][v]) {
                            ++c;
                        }
                        if (right && pos_[l + 1][u] > pos_[l + 1][v]) {
                            ++c;
                        }
                    }
                }
                w_[static_cast<std::size_t>(a) * m + b] = c;
            }
        }
        auto& ord = orders_[l];
        if (sort_first && (left || right)) {
            // 按两侧邻居中的平均秩排序作为起点
            std::vector<double> key(m, 0.0);
            for (int a = 0; a < m; ++a) {
                double s = 0;
                int cnt = 0;
                for (int u : blocks_[l][a]) {
                    if (u < 0) {
                        continue;
                    }
                    if (left) {
                        s += pos_[l - 1][u];
                        ++cnt;
                    }
                    if (right) {
                        s += pos_[l + 1][u];
                        ++cnt;
                    }
                }
                key[a] = cnt > 0 ? s / cnt : 0.0;
            }
            std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return key[x] < key[y]; });
        }
        InsertionSearch(ord, m);
        UpdatePos(l);
    }

    // 一轮前向或后向扫描。
    void Sweep(bool forward, int lo, int hi) {
        if (forward) {
            for (int l = lo; l < hi; ++l) {
                Reoptimize(l, false);
            }
        } else {
            for (int l = hi - 1; l >= lo; --l) {
                Reoptimize(l, false);
            }
        }
    }

    // 贪心初值：逐层只看左邻居。
    void GreedyInit() {
        UpdatePos(0);
        const int num = num_layers();
        for (int l = 1; l < num; ++l) {
            // 只用左邻居：暂时把右邻居当作不存在
            const int m = static_cast<int>(blocks_[l].size());
            w_.assign(static_cast<std::size_t>(m) * m, 0);
            std::vector<double> key(m, 0.0);
            for (int a = 0; a < m; ++a) {
                double s = 0;
                int cnt = 0;
                for (int u : blocks_[l][a]) {
                    if (u >= 0) {
                        s += pos_[l - 1][u];
                        ++cnt;
                    }
                }
                key[a] = s / std::max(1, cnt);
                for (int b = 0; b < m; ++b) {
                    if (a == b) {
                        continue;
                    }
                    int c = 0;
                    for (int u : blocks_[l][a]) {
                        for (int v : blocks_[l][b]) {
                            if (u >= 0 && v >= 0 && pos_[l - 1][u] > pos_[l - 1][v]) {
                                ++c;
                            }
                        }
                    }
                    w_[static_cast<std::size_t>(a) * m + b] = c;
                }
            }
            auto& ord = orders_[l];
            std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return key[x] < key[y]; });
            InsertionSearch(ord, m);
            UpdatePos(l);
        }
    }

    // 扰动第 [lo, hi) 层：每层随机交换若干对相邻块。
    void Kick(int lo, int hi, std::uint64_t& rng) {
        for (int l = lo; l < hi; ++l) {
            auto& ord = orders_[l];
            const int m = static_cast<int>(ord.size());
            if (m < 2) {
                continue;
            }
            const int kicks = 1 + static_cast<int>(SplitMix(rng) % 4);
            for (int r = 0; r < kicks; ++r) {
                const int i = static_cast<int>(SplitMix(rng) % static_cast<std::uint64_t>(m - 1));
                std::swap(ord[i], ord[i + 1]);
            }
            UpdatePos(l);
        }
    }

    std::vector<std::vector<int>> SaveOrders(int lo, int hi) const {
        return {orders_.begin() + lo, orders_.begin() + hi};
    }
    void RestoreOrders(int lo, const std::vector<std::vector<int>>& saved) {
        for (std::size_t i = 0; i < saved.size(); ++i) {
            orders_[lo + i] = saved[i];
            UpdatePos(lo + static_cast<int>(i));
        }
    }

 private:
    void InsertionSearch(std::vector<int>& ord, int m) {
        bool improved = true;
        std::vector<int> rest;
        rest.reserve(m);
        int guard = 0;
        while (improved && guard++ < 64) {
            improved = false;
            for (int i = 0; i < m; ++i) {
                const int x = ord[i];
                rest.clear();
                for (int j = 0; j < m; ++j) {
                    if (j != i) {
                        rest.push_back(ord[j]);
                    }
                }
                // c(j)：把 x 插在 rest[j] 之前的代价；c(0) = Σ_y w[x][y]，右移越过 y：+ w[y][x] - w[x][y]
                std::int64_t c = 0;
                for (int y : rest) {
                    c += w_[static_cast<std::size_t>(x) * m + y];
                }
                std::int64_t cur = 0;
                std::int64_t best = c;
                int best_j = 0;
                for (int j = 0; j <= m - 1; ++j) {
                    if (j == i) {
                        cur = c;
                    }
                    if (c < best) {
                        best = c;
                        best_j = j;
                    }
                    if (j < m - 1) {
                        const int y = rest[j];
                        c += w_[static_cast<std::size_t>(y) * m + x] - w_[static_cast<std::size_t>(x) * m + y];
                    }
                }
                if (best < cur) {
                    rest.insert(rest.begin() + best_j, x);
                    ord = rest;
                    improved = true;
                }
            }
        }
    }

    int n_;
    std::vector<std::vector<Block>> blocks_;
    std::vector<std::vector<int>> orders_;
    std::vector<std::vector<int>> pos_;
    std::vector<std::int64_t> w_;
};

}  // namespace

std::int64_t LayerChainCost(int n, const std::vector<std::vector<std::vector<int>>>& orders) {
    std::int64_t total = 0;
    std::vector<int> prev(n, -1);
    for (std::size_t l = 0; l < orders.size(); ++l) {
        std::vector<int> cur(n, -1);
        for (std::size_t i = 0; i < orders[l].size(); ++i) {
            for (int t : orders[l][i]) {
                cur[t] = static_cast<int>(i);
            }
        }
        if (l > 0) {
            for (int u = 0; u < n; ++u) {
                for (int v = 0; v < n; ++v) {
                    if (u != v && cur[u] < cur[v] && prev[u] > prev[v]) {
                        ++total;
                    }
                }
            }
        }
        prev = std::move(cur);
    }
    return total;
}

LayerChainResult RouteLayersOnPath(int n, const std::vector<std::vector<std::pair<int, int>>>& layers,
                                   const LayerChainOptions& options) {
    LayerChainResult res;
    if (layers.empty()) {
        res.initial.resize(n);
        std::iota(res.initial.begin(), res.initial.end(), 0);
        return res;
    }
    ChainSolver solver(n, layers);
    const int num = solver.num_layers();
    solver.GreedyInit();
    std::int64_t total = solver.Total();
    for (int s = 0; s < options.max_sweeps; ++s) {
        solver.Sweep(s % 2 == 0, 0, num);
        ++res.sweeps;
        const std::int64_t t = solver.Total();
        if (t >= total) {
            total = t;
            break;
        }
        total = t;
    }
    // 迭代局部搜索：扰动一段连续层，局部块坐标下降，变好就保留
    std::uint64_t rng = options.seed;
    const int span = std::max(1, std::min(options.perturb_span, num));
    for (int r = 0; r < options.perturbations && num > 1; ++r) {
        const int lo = static_cast<int>(SplitMix(rng) % static_cast<std::uint64_t>(num - span + 1));
        const int hi = lo + span;
        const int a = std::max(0, lo - 1);
        const int b = std::min(num, hi + 1);
        auto local = [&] {
            std::int64_t s = 0;
            for (int l = a; l <= std::min(num - 1, b); ++l) {
                s += solver.TransCost(l);
            }
            return s;
        };
        const std::int64_t before = local();
        const auto saved = solver.SaveOrders(a, b);
        solver.Kick(lo, hi, rng);
        std::int64_t cur = local();
        for (int s = 0; s < 4; ++s) {
            solver.Sweep(s % 2 == 0, a, b);
            const std::int64_t t = local();
            if (t >= cur) {
                break;
            }
            cur = t;
        }
        if (cur < before) {
            total += cur - before;
        } else {
            solver.RestoreOrders(a, saved);
        }
    }
    res.cost = solver.Total();

    // 生成 SWAP：arr 为位置 -> 比特
    std::vector<int> arr;
    arr.reserve(n);
    for (int b : solver.order(0)) {
        for (int t : solver.block(0, b)) {
            if (t >= 0) {
                arr.push_back(t);
            }
        }
    }
    res.initial = arr;
    std::vector<int> where(n);
    std::vector<int> rank(n);
    std::vector<int> target(n);
    res.swaps.assign(num > 0 ? num - 1 : 0, {});
    res.flips.assign(num > 0 ? num - 1 : 0, 0);
    for (int l = 0; l + 1 < num; ++l) {
        auto& out = res.swaps[l];
        const auto& next = solver.order(l + 1);
        for (std::size_t i = 0; i < next.size(); ++i) {
            for (int t : solver.block(l + 1, next[i])) {
                if (t >= 0) {
                    rank[t] = static_cast<int>(i);
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            where[arr[i]] = i;
        }
        // 上一层块内的免费翻转（紧跟块之后，并入块）
        for (int b : solver.order(l)) {
            const Block& blk = solver.block(l, b);
            if (blk[1] < 0) {
                continue;
            }
            const int i = std::min(where[blk[0]], where[blk[1]]);
            if (rank[arr[i]] > rank[arr[i + 1]]) {
                std::swap(arr[i], arr[i + 1]);
                out.push_back(i);
                ++res.flips[l];
            }
        }
        for (int i = 0; i < n; ++i) {
            where[arr[i]] = i;
        }
        // 目标位置：块序，块内保持当前顺序
        int k = 0;
        for (int b : next) {
            Block blk = solver.block(l + 1, b);
            if (blk[1] >= 0 && where[blk[0]] > where[blk[1]]) {
                std::swap(blk[0], blk[1]);
            }
            for (int t : blk) {
                if (t >= 0) {
                    target[t] = k++;
                }
            }
        }
        // 奇偶换位排序：只交换逆序的相邻对
        bool moved = true;
        for (int round = 0; moved || round % 2 == 1; ++round) {
            if (round % 2 == 0) {
                moved = false;
            }
            for (int i = round % 2; i + 1 < n; i += 2) {
                if (target[arr[i]] > target[arr[i + 1]]) {
                    std::swap(arr[i], arr[i + 1]);
                    out.push_back(i);
                    moved = true;
                }
            }
            if (round > 4 * n + 4) {
                break;  // 不会发生（≤ n 轮即有序）
            }
        }
    }
    return res;
}

}  // namespace quspect::native::transpiler
