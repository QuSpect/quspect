// 逐层匹配线路在环上的分层路由：循环块序链 + 万有覆盖上的交叉数，见 ring_routing.hpp。

#include "quspect_native/transpiler/ring_routing.hpp"

#include <cmath>

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

std::int64_t FloorDiv(std::int64_t a, std::int64_t b) {
    std::int64_t q = a / b;
    if (a % b != 0 && a < 0) {
        --q;
    }
    return q;
}

std::int64_t Mod(std::int64_t a, std::int64_t n) {
    a %= n;
    return a < 0 ? a + n : a;
}

using Block = std::array<int, 2>;  // 比特对，或 {t, -1}

// 一次转移（上一层布局 → 本层布局）的最少付费交换数。
//
// 给定两端的块内朝向时提升是确定的：记前向位移 d_t = (y_t − x_t) mod n，Σ δ = 0 要求恰有
// K = Σ d_t / n 个比特反向走（δ_t = d_t − n），最优解取 d_t 最大的 K 个（阈值规则；小规模穷举
// 验证与全部提升的最优一致）。两端的朝向（上一层块的免费翻转、本层块的到达朝向）用单块翻转的
// 首次改进局部搜索选择。
class TransitionSolver {
 public:
    explicit TransitionSolver(int n) : n_(n), x_(n), t_(n), y_(n), d_(n), order_(n) {
    }

    // xs / ys：每个比特在上一层 / 本层的位置（按存储的块内朝向）；prev / next：两层的两比特块。
    std::int64_t Solve(const std::vector<int>& xs, const std::vector<int>& ys, const std::vector<Block>& prev,
                       const std::vector<Block>& next) {
        for (int t = 0; t < n_; ++t) {
            x_[t] = xs[t];
            y_[t] = ys[t];
        }
        std::int64_t cost = Evaluate();
        for (int pass = 0; pass < 16; ++pass) {
            bool improved = false;
            for (const Block& blk : prev) {
                std::swap(x_[blk[0]], x_[blk[1]]);
                const std::int64_t c = Evaluate();
                if (c < cost) {
                    cost = c;
                    improved = true;
                } else {
                    std::swap(x_[blk[0]], x_[blk[1]]);
                }
            }
            for (const Block& blk : next) {
                std::swap(y_[blk[0]], y_[blk[1]]);
                const std::int64_t c = Evaluate();
                if (c < cost) {
                    cost = c;
                    improved = true;
                } else {
                    std::swap(y_[blk[0]], y_[blk[1]]);
                }
            }
            if (!improved) {
                break;
            }
        }
        Evaluate();  // t_ 与最终朝向一致
        return cost;
    }

    // 起点（位置，含免费翻转）与终点（提升坐标）。
    const std::vector<std::int64_t>& x() const {
        return x_;
    }
    const std::vector<std::int64_t>& t() const {
        return t_;
    }

 private:
    // 当前朝向下的阈值提升与交叉数。
    std::int64_t Evaluate() {
        const int n = n_;
        std::int64_t sum = 0;
        for (int t = 0; t < n; ++t) {
            d_[t] = Mod(y_[t] - x_[t], n);
            sum += d_[t];
            order_[t] = t;
        }
        const std::int64_t k = sum / n;
        std::stable_sort(order_.begin(), order_.end(), [&](int a, int b) { return d_[a] > d_[b]; });
        for (int t = 0; t < n; ++t) {
            t_[t] = x_[t] + d_[t];
        }
        for (std::int64_t i = 0; i < k; ++i) {
            t_[order_[i]] -= n;
        }
        // x ∈ [0, n)、t ∈ [x − n, x + n)：差值的整除商只有几种取值，用比较代替除法
        std::int64_t cost = 0;
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                cost += std::abs(SmallFloorDiv(t_[a] - t_[b]) - (x_[a] < x_[b] ? -1 : 0));
            }
        }
        return cost;
    }

    // ⌊v / n⌋，|v| < 3n。
    int SmallFloorDiv(std::int64_t v) const {
        const std::int64_t n = n_;
        if (v >= 0) {
            return v < n ? 0 : (v < 2 * n ? 1 : 2);
        }
        return v >= -n ? -1 : (v >= -2 * n ? -2 : -3);
    }

    int n_;
    std::vector<std::int64_t> x_;
    std::vector<std::int64_t> t_;
    std::vector<std::int64_t> y_;
    std::vector<std::int64_t> d_;
    std::vector<int> order_;
};

class RingChain {
 public:
    RingChain(int n, const std::vector<std::vector<std::pair<int, int>>>& layers) : n_(n), ts_(n) {
        const int num = static_cast<int>(layers.size());
        blocks_.resize(num);
        two_.resize(num);
        orders_.resize(num);
        rots_.assign(num, 0);
        sites_.assign(num, std::vector<int>(n, 0));
        tc_.assign(num, 0);
        std::vector<int> seen(n, -1);
        for (int l = 0; l < num; ++l) {
            for (auto [a, b] : layers[l]) {
                if (a < 0 || b < 0 || a >= n || b >= n || a == b || seen[a] == l || seen[b] == l) {
                    throw std::invalid_argument("layer pairs must be disjoint qubit pairs in range");
                }
                seen[a] = seen[b] = l;
                blocks_[l].push_back({a, b});
                two_[l].push_back({a, b});
            }
            for (int t = 0; t < n; ++t) {
                if (seen[t] != l) {
                    blocks_[l].push_back({t, -1});
                }
            }
            orders_[l].resize(blocks_[l].size());
            std::iota(orders_[l].begin(), orders_[l].end(), 0);
            UpdateSites(l);
        }
    }

    int num_layers() const {
        return static_cast<int>(blocks_.size());
    }
    const std::vector<int>& sites(int l) const {
        return sites_[l];
    }
    const std::vector<Block>& two(int l) const {
        return two_[l];
    }

    void UpdateSites(int l) {
        int k = rots_[l];
        for (int b : orders_[l]) {
            for (int t : blocks_[l][b]) {
                if (t >= 0) {
                    sites_[l][t] = k % n_;
                    ++k;
                }
            }
        }
    }

    // 第 l - 1 → l 次转移的最少付费交换数（局部搜索的提升解）。
    std::int64_t Trans(int l) {
        return ts_.Solve(sites_[l - 1], sites_[l], two_[l - 1], two_[l]);
    }

    // 解出第 l - 1 → l 次转移，返回其提升解。
    RingTransition Solution(int l) {
        RingTransition out;
        out.crossings = Trans(l);
        out.x = ts_.x();
        out.t = ts_.t();
        return out;
    }

    std::int64_t Total() const {
        std::int64_t s = 0;
        for (int l = 1; l < num_layers(); ++l) {
            s += tc_[l];
        }
        return s;
    }

    void Recompute(int l, int built) {
        if (l > 0 && l < built) {
            tc_[l] = Trans(l);
        }
    }

    // 以两侧邻居（只考虑前 built 层）为准重排第 l 层：旋转与插入移动的首次改进局部搜索。
    void Reoptimize(int l, int built, int max_passes) {
        auto local = [&] {
            std::int64_t s = 0;
            if (l > 0) {
                s += Trans(l);
            }
            if (l + 1 < built) {
                s += Trans(l + 1);
            }
            return s;
        };
        std::int64_t best = local();
        auto& ord = orders_[l];
        const int m = static_cast<int>(ord.size());
        for (int pass = 0; pass < max_passes; ++pass) {
            bool improved = false;
            const int rot0 = rots_[l];
            for (int r = 1; r < n_; ++r) {
                rots_[l] = (rot0 + r) % n_;
                UpdateSites(l);
                const std::int64_t c = local();
                if (c < best) {
                    best = c;
                    improved = true;
                    break;
                }
                rots_[l] = rot0;
            }
            UpdateSites(l);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (j == i) {
                        continue;
                    }
                    const std::vector<int> saved = ord;
                    const int b = ord[i];
                    ord.erase(ord.begin() + i);
                    ord.insert(ord.begin() + j, b);
                    UpdateSites(l);
                    const std::int64_t c = local();
                    if (c < best) {
                        best = c;
                        improved = true;
                    } else {
                        ord = saved;
                    }
                }
            }
            UpdateSites(l);
            if (!improved) {
                break;
            }
        }
        Recompute(l, built);
        Recompute(l + 1, built);
    }

    // 贪心初值：第 l 层按上一层位置的圆周平均排序，再只对左邻居局部搜索。
    void GreedyInit(int passes) {
        const int num = num_layers();
        const double tau = 2.0 * std::acos(-1.0);
        for (int l = 1; l < num; ++l) {
            const int m = static_cast<int>(blocks_[l].size());
            std::vector<double> key(m, 0.0);
            for (int b = 0; b < m; ++b) {
                double cx = 0;
                double cy = 0;
                for (int t : blocks_[l][b]) {
                    if (t >= 0) {
                        const double ang = tau * sites_[l - 1][t] / n_;
                        cx += std::cos(ang);
                        cy += std::sin(ang);
                    }
                }
                double a = std::atan2(cy, cx);
                if (a < 0) {
                    a += tau;
                }
                key[b] = a;
            }
            auto& ord = orders_[l];
            std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return key[x] < key[y]; });
            UpdateSites(l);
            Reoptimize(l, l + 1, passes);
        }
        for (int l = 1; l < num; ++l) {
            tc_[l] = Trans(l);
        }
    }

    void Kick(int lo, int hi, std::uint64_t& rng) {
        for (int l = lo; l < hi; ++l) {
            auto& ord = orders_[l];
            const int m = static_cast<int>(ord.size());
            if (m >= 2) {
                const int kicks = 1 + static_cast<int>(SplitMix(rng) % 3);
                for (int r = 0; r < kicks; ++r) {
                    const int i = static_cast<int>(SplitMix(rng) % static_cast<std::uint64_t>(m - 1));
                    std::swap(ord[i], ord[i + 1]);
                }
            }
            if (SplitMix(rng) % 2 == 0) {
                rots_[l] = (rots_[l] + 1 + static_cast<int>(SplitMix(rng) % 2) * (n_ - 2)) % n_;
            }
            UpdateSites(l);
        }
    }

    struct Saved {
        std::vector<std::vector<int>> orders;
        std::vector<int> rots;
        std::vector<std::int64_t> tc;
    };
    Saved Save(int lo, int hi) const {
        Saved s;
        s.orders.assign(orders_.begin() + lo, orders_.begin() + hi);
        s.rots.assign(rots_.begin() + lo, rots_.begin() + hi);
        s.tc = tc_;
        return s;
    }
    void Restore(int lo, const Saved& s) {
        for (std::size_t i = 0; i < s.orders.size(); ++i) {
            orders_[lo + i] = s.orders[i];
            rots_[lo + i] = s.rots[i];
            UpdateSites(lo + static_cast<int>(i));
        }
        tc_ = s.tc;
    }

 private:
    int n_;
    TransitionSolver ts_;
    std::vector<std::vector<Block>> blocks_;
    std::vector<std::vector<Block>> two_;
    std::vector<std::vector<int>> orders_;
    std::vector<int> rots_;
    std::vector<std::vector<int>> sites_;
    std::vector<std::int64_t> tc_;
};

}  // namespace

std::int64_t RingCrossings(int n, const std::vector<std::int64_t>& x, const std::vector<std::int64_t>& t) {
    if (n <= 0 || x.size() != t.size()) {
        throw std::invalid_argument("ring crossings need n > 0 and matching x / t");
    }
    std::int64_t s = 0;
    for (std::size_t a = 0; a < x.size(); ++a) {
        for (std::size_t b = a + 1; b < x.size(); ++b) {
            s += std::abs(FloorDiv(t[a] - t[b], n) - FloorDiv(x[a] - x[b], n));
        }
    }
    return s;
}

LayerChainResult RouteLayersOnRing(int n, const std::vector<std::vector<std::pair<int, int>>>& layers,
                                   const LayerChainOptions& options) {
    if (n < 3) {
        throw std::invalid_argument("a ring needs at least 3 sites");
    }
    LayerChainResult res;
    if (layers.empty()) {
        res.initial.resize(n);
        std::iota(res.initial.begin(), res.initial.end(), 0);
        return res;
    }
    RingChain chain(n, layers);
    const int num = chain.num_layers();
    constexpr int kPasses = 8;
    chain.GreedyInit(kPasses);
    std::int64_t total = chain.Total();
    for (int s = 0; s < options.max_sweeps; ++s) {
        if (s % 2 == 0) {
            for (int l = 0; l < num; ++l) {
                chain.Reoptimize(l, num, kPasses);
            }
        } else {
            for (int l = num - 1; l >= 0; --l) {
                chain.Reoptimize(l, num, kPasses);
            }
        }
        ++res.sweeps;
        const std::int64_t t = chain.Total();
        if (t >= total) {
            total = t;
            break;
        }
        total = t;
    }
    // 迭代局部搜索：扰动一段连续层，局部重优化，变好就保留
    std::uint64_t rng = options.seed;
    const int span = std::max(1, std::min(options.perturb_span, num));
    for (int r = 0; r < options.perturbations && num > 1; ++r) {
        const int lo = static_cast<int>(SplitMix(rng) % static_cast<std::uint64_t>(num - span + 1));
        const int hi = lo + span;
        const auto saved = chain.Save(lo, hi);
        const std::int64_t before = chain.Total();
        chain.Kick(lo, hi, rng);
        for (int l = lo; l <= std::min(num - 1, hi); ++l) {
            chain.Recompute(l, num);
        }
        if (r % 2 == 0) {
            for (int l = lo; l < hi; ++l) {
                chain.Reoptimize(l, num, 2);
            }
        } else {
            for (int l = hi - 1; l >= lo; --l) {
                chain.Reoptimize(l, num, 2);
            }
        }
        if (chain.Total() >= before) {
            chain.Restore(lo, saved);
        }
    }
    res.cost = chain.Total();

    // 生成 SWAP：arr 为位置 -> 比特（第 0 层按存储朝向）
    std::vector<int> arr(n);
    for (int t = 0; t < n; ++t) {
        arr[chain.sites(0)[t]] = t;
    }
    res.initial = arr;
    res.swaps.assign(num - 1, {});
    res.flips.assign(num - 1, 0);
    std::vector<int> where(n);
    std::vector<std::int64_t> rem(n);
    // 边分类：偶数环两类（偶 / 奇起点），奇数环最后一条边单独一类
    const int classes = n % 2 == 0 ? 2 : 3;
    auto edge_class = [&](int i) { return (n % 2 == 1 && i == n - 1) ? 2 : i % 2; };
    std::int64_t paid = 0;
    for (int l = 1; l < num; ++l) {
        auto& out = res.swaps[l - 1];
        const RingTransition sol = chain.Solution(l);
        for (int i = 0; i < n; ++i) {
            where[arr[i]] = i;
        }
        // 免费翻转：让上一层的块内朝向与解的起点一致（交换的正是上一层的一对）
        for (const Block& blk : chain.two(l - 1)) {
            const int a = blk[0];
            const int b = blk[1];
            const int want_a = static_cast<int>(Mod(sol.x[a], n));
            if (where[a] != want_a) {
                const int i = Mod(where[b] - where[a], n) == 1 ? where[a] : where[b];
                std::swap(arr[i], arr[(i + 1) % n]);
                where[arr[i]] = i;
                where[arr[(i + 1) % n]] = (i + 1) % n;
                out.push_back(i);
                ++res.flips[l - 1];
            }
        }
        for (int t = 0; t < n; ++t) {
            rem[t] = sol.t[t] - sol.x[t];
        }
        // 周期性奇偶换位排序：相邻 a（左）、b（右）满足 rem[a] − rem[b] ≥ 2 时交换
        bool moved = true;
        int idle = 0;
        for (int round = 0; idle < classes; ++round) {
            const int cls = round % classes;
            moved = false;
            for (int i = 0; i < n; ++i) {
                if (edge_class(i) != cls) {
                    continue;
                }
                const int j = (i + 1) % n;
                const int a = arr[i];
                const int b = arr[j];
                if (rem[a] - rem[b] >= 2) {
                    std::swap(arr[i], arr[j]);
                    --rem[a];
                    ++rem[b];
                    out.push_back(i);
                    moved = true;
                    ++paid;
                }
            }
            idle = moved ? 0 : idle + 1;
            if (round > 8 * n * n + 16) {
                throw std::logic_error("ring sort did not converge");
            }
        }
        for (int t = 0; t < n; ++t) {
            if (rem[t] != 0 || arr[Mod(sol.t[t], n)] != t) {
                throw std::logic_error("ring sort ended off target");
            }
        }
    }
    if (paid != res.cost) {
        throw std::logic_error("ring sort swap count differs from the chain cost");
    }
    return res;
}

}  // namespace quspect::native::transpiler
