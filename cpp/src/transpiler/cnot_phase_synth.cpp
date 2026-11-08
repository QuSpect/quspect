#include "quspect_native/transpiler/cnot_phase_synth.hpp"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace quspect::native::transpiler {

namespace {

constexpr int kMaxQubits = 6;
constexpr int kRowBits = 6;  // 每行占 6 位，n ≤ 6 时 36 位

struct Key {
    std::uint64_t rows;
    std::uint64_t seen;
    bool operator==(const Key& o) const {
        return rows == o.rows && seen == o.seen;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const {
        std::uint64_t h = k.rows * 0x9E3779B97F4A7C15ULL;
        h ^= k.seen + 0x7F4A7C159E3779B9ULL + (h << 6) + (h >> 2);
        return static_cast<std::size_t>(h);
    }
};

struct Info {
    int g;
    Key parent;
    int move;  // edges 下标；-1 为起点
};

struct Node {
    int f;
    int g;
    std::uint64_t order;
    Key key;
};

struct NodeCmp {
    bool operator()(const Node& a, const Node& b) const {
        if (a.f != b.f)
            return a.f > b.f;
        if (a.g != b.g)
            return a.g < b.g;
        return a.order > b.order;
    }
};

inline int popcount64(std::uint64_t x) {
    int c = 0;
    while (x != 0) {
        x &= x - 1;
        ++c;
    }
    return c;
}

inline std::uint32_t row_of(std::uint64_t rows, int i) {
    return static_cast<std::uint32_t>((rows >> (kRowBits * i)) & ((1ULL << kRowBits) - 1));
}

inline std::uint64_t with_row(std::uint64_t rows, int i, std::uint32_t v) {
    const std::uint64_t mask = ((1ULL << kRowBits) - 1) << (kRowBits * i);
    return (rows & ~mask) | (static_cast<std::uint64_t>(v) << (kRowBits * i));
}

}  // namespace

CnotPhaseResult SynthesizeCnotPhase(int n, const std::vector<std::pair<int, int>>& edges,
                                    const std::vector<std::uint32_t>& target_rows,
                                    const std::vector<std::uint32_t>& required, int max_cx, std::int64_t node_budget,
                                    std::uint32_t free_mask) {
    if (n < 1 || n > kMaxQubits)
        throw std::invalid_argument("SynthesizeCnotPhase: n must be in 1..6");
    if (static_cast<int>(target_rows.size()) != n)
        throw std::invalid_argument("SynthesizeCnotPhase: target_rows must have n entries");
    for (const auto& e : edges) {
        if (e.first < 0 || e.first >= n || e.second < 0 || e.second >= n || e.first == e.second)
            throw std::invalid_argument("SynthesizeCnotPhase: bad edge");
    }
    // 必需项：去重、去 0，编号 → 位
    std::vector<std::uint32_t> req;
    for (std::uint32_t m : required) {
        if (m != 0 && m < (1U << n))
            req.push_back(m);
    }
    std::sort(req.begin(), req.end());
    req.erase(std::unique(req.begin(), req.end()), req.end());
    if (req.size() > 63)
        throw std::invalid_argument("SynthesizeCnotPhase: too many required terms");
    std::vector<int> term_bit(1U << n, -1);
    for (std::size_t k = 0; k < req.size(); ++k)
        term_bit[req[k]] = static_cast<int>(k);
    const std::uint64_t full = (1ULL << req.size()) - 1;

    std::uint64_t target = 0;
    std::uint64_t start = 0;
    std::uint64_t seen0 = 0;
    for (int i = 0; i < n; ++i) {
        target = with_row(target, i, target_rows[static_cast<std::size_t>(i)]);
        start = with_row(start, i, 1U << i);
        const int b = term_bit[1U << i];
        if (b >= 0)
            seen0 |= 1ULL << b;
    }

    free_mask &= (1U << n) - 1;
    // 与目标不同的行数：固定线逐行比较；自由线按集合比较（目标行线性无关，互不相同）
    auto row_misses = [&](std::uint64_t rows) {
        int h2 = 0;
        for (int i = 0; i < n; ++i) {
            if ((free_mask >> i) & 1U) {
                const std::uint32_t want = row_of(target, i);
                bool hit = false;
                for (int j = 0; j < n && !hit; ++j)
                    hit = ((free_mask >> j) & 1U) != 0 && row_of(rows, j) == want;
                h2 += hit ? 0 : 1;
            } else {
                h2 += row_of(rows, i) != row_of(target, i) ? 1 : 0;
            }
        }
        return h2;
    };
    // 可采纳的下界：每条 CX 只产生一个新的行值。缺的必需项各要一条 CX；与目标不同的行各要一条“最后写入”
    // 该行目标值的 CX，目标值本身不是缺的必需项时，这条 CX 与前者互不相同。所以
    //     h = 缺的必需项数 + #{与目标不同、且目标值不是缺的必需项的行}
    // （自由线：自由目标行中尚未出现、且不是缺的必需项的个数）。
    auto missing_target = [&](std::uint32_t want, std::uint64_t seen) {
        const int b = term_bit[want];
        return b >= 0 && ((seen >> b) & 1ULL) == 0;
    };
    auto heuristic = [&](std::uint64_t rows, std::uint64_t seen) {
        int h = popcount64(full & ~seen);
        for (int i = 0; i < n; ++i) {
            const std::uint32_t want = row_of(target, i);
            if ((free_mask >> i) & 1U) {
                bool hit = false;
                for (int j = 0; j < n && !hit; ++j)
                    hit = ((free_mask >> j) & 1U) != 0 && row_of(rows, j) == want;
                h += (!hit && !missing_target(want, seen)) ? 1 : 0;
            } else {
                h += (row_of(rows, i) != want && !missing_target(want, seen)) ? 1 : 0;
            }
        }
        return h;
    };

    CnotPhaseResult result;
    std::unordered_map<Key, Info, KeyHash> info;
    std::priority_queue<Node, std::vector<Node>, NodeCmp> open;
    std::uint64_t order = 0;
    const Key k0{start, seen0};
    const int h0 = heuristic(start, seen0);
    if (h0 > max_cx)
        return result;
    info.emplace(k0, Info{0, k0, -1});
    open.push(Node{h0, 0, order++, k0});
    while (!open.empty()) {
        const Node cur = open.top();
        open.pop();
        const auto it = info.find(cur.key);
        if (it->second.g < cur.g)
            continue;  // 过期项
        if (cur.key.seen == full && (cur.key.rows == target || (free_mask != 0 && row_misses(cur.key.rows) == 0))) {
            Key k = cur.key;
            while (true) {
                const Info& in = info.at(k);
                if (in.move < 0)
                    break;
                result.moves.push_back(edges[static_cast<std::size_t>(in.move)]);
                k = in.parent;
            }
            std::reverse(result.moves.begin(), result.moves.end());
            result.perm.resize(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                result.perm[static_cast<std::size_t>(i)] = i;
                for (int j = 0; j < n; ++j) {
                    if (row_of(cur.key.rows, i) == row_of(target, j)) {
                        result.perm[static_cast<std::size_t>(i)] = j;
                        break;
                    }
                }
            }
            result.found = true;
            return result;
        }
        if (++result.expanded > node_budget)
            return result;
        const int ng = cur.g + 1;
        if (ng > max_cx)
            continue;
        for (std::size_t e = 0; e < edges.size(); ++e) {
            const int c = edges[e].first;
            const int t = edges[e].second;
            const std::uint32_t nt = row_of(cur.key.rows, t) ^ row_of(cur.key.rows, c);
            const std::uint64_t nrows = with_row(cur.key.rows, t, nt);
            std::uint64_t nseen = cur.key.seen;
            const int b = term_bit[nt];
            if (b >= 0)
                nseen |= 1ULL << b;
            const int nf = ng + heuristic(nrows, nseen);
            if (nf > max_cx)
                continue;
            const Key nk{nrows, nseen};
            auto found = info.find(nk);
            if (found != info.end() && found->second.g <= ng)
                continue;
            info[nk] = Info{ng, cur.key, static_cast<int>(e)};
            open.push(Node{nf, ng, order++, nk});
        }
    }
    return result;
}

namespace {

// GF(2) 上 n×n 矩阵（行 = 位掩码）求逆；奇异时返回 false。
bool InvertGf2(int n, const std::vector<std::uint32_t>& m, std::vector<std::uint32_t>& inv) {
    std::vector<std::uint32_t> a(m);
    inv.assign(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        inv[static_cast<std::size_t>(i)] = 1U << i;
    for (int col = 0; col < n; ++col) {
        int piv = -1;
        for (int r = col; r < n; ++r) {
            if ((a[static_cast<std::size_t>(r)] >> col) & 1U) {
                piv = r;
                break;
            }
        }
        if (piv < 0)
            return false;
        std::swap(a[static_cast<std::size_t>(piv)], a[static_cast<std::size_t>(col)]);
        std::swap(inv[static_cast<std::size_t>(piv)], inv[static_cast<std::size_t>(col)]);
        for (int r = 0; r < n; ++r) {
            if (r != col && ((a[static_cast<std::size_t>(r)] >> col) & 1U)) {
                a[static_cast<std::size_t>(r)] ^= a[static_cast<std::size_t>(col)];
                inv[static_cast<std::size_t>(r)] ^= inv[static_cast<std::size_t>(col)];
            }
        }
    }
    return true;
}

inline int popcount32(std::uint32_t x) {
    return popcount64(x);
}

std::uint64_t Mix(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    return h * 0xBF58476D1CE4E5B9ULL;
}

struct BeamState {
    std::vector<std::uint32_t> rows;   // 当前奇偶行 R_i
    std::vector<std::uint32_t> resid;  // E_i = R_i · T⁻¹
    std::vector<std::uint64_t> seen;   // 已出现的必需项（位集）
    std::vector<std::uint8_t> best1;   // 每个必需项：min_i |R_i ⊕ y|
    std::vector<std::uint8_t> best2;   // 次小（另一根线上）
    std::vector<std::int8_t> who1;     // 取到 best1 的线
    int missing = 0;
    int history = -1;  // 在 hist 中的下标
};

struct Child {
    double score;
    int parent;  // beam 中的下标
    int edge;
    std::uint64_t hash;
};

}  // namespace

CnotPhaseResult SynthesizeCnotPhaseBeam(int n, const std::vector<std::pair<int, int>>& edges,
                                        const std::vector<std::uint32_t>& target_rows,
                                        const std::vector<std::uint32_t>& required, int max_cx, int beam_width,
                                        std::uint32_t free_mask) {
    if (n < 1 || n > 32)
        throw std::invalid_argument("SynthesizeCnotPhaseBeam: n must be in 1..32");
    if (static_cast<int>(target_rows.size()) != n)
        throw std::invalid_argument("SynthesizeCnotPhaseBeam: target_rows must have n entries");
    for (const auto& e : edges) {
        if (e.first < 0 || e.first >= n || e.second < 0 || e.second >= n || e.first == e.second)
            throw std::invalid_argument("SynthesizeCnotPhaseBeam: bad edge");
    }
    const std::uint32_t all = n == 32 ? 0xFFFFFFFFU : ((1U << n) - 1);
    free_mask &= all;
    CnotPhaseResult result;
    std::vector<std::uint32_t> tinv;
    if (!InvertGf2(n, target_rows, tinv))
        throw std::invalid_argument("SynthesizeCnotPhaseBeam: target is singular");
    std::vector<std::uint32_t> req;
    for (std::uint32_t m : required) {
        if (m != 0 && (m & ~all) == 0)
            req.push_back(m);
    }
    std::sort(req.begin(), req.end());
    req.erase(std::unique(req.begin(), req.end()), req.end());
    const int P = static_cast<int>(req.size());
    const std::size_t words = (static_cast<std::size_t>(P) + 63) / 64;
    std::unordered_map<std::uint32_t, int> term_index;
    term_index.reserve(req.size() * 2 + 1);
    for (int k = 0; k < P; ++k)
        term_index.emplace(req[static_cast<std::size_t>(k)], k);
    auto term_of = [&](std::uint32_t m) {
        const auto it = term_index.find(m);
        return it == term_index.end() ? -1 : it->second;
    };
    // 每行的剩余线性距离：固定线 |E_i ⊕ e_i|；自由线 |E_i| − [E_i 含自由位]（为 0 当且仅当是自由线的单位向量）
    auto row_dist = [&](int i, std::uint32_t e) {
        if ((free_mask >> i) & 1U)
            return popcount32(e) - ((e & free_mask) != 0 ? 1 : 0);
        return popcount32(e ^ (1U << i));
    };
    const double inv_p = P > 0 ? 1.0 / P : 0.0;
    const double inv_n = 1.0 / n;
    // 估计的剩余代价（越小越好）
    // 权重：缺的必需项、相位距离（按项数平均）、与目标不同的行、线性距离（按线数平均）；
    // 在合成问题（全连接 / 环上的 ZZ 层、随机三体奇偶、全连接 CX 梯子）上选定
    auto score_of = [&](int missing, int phase_dist, int lin_rows, int lin_bits) {
        return 2.0 * missing + 2.0 * phase_dist * inv_p + 0.5 * lin_rows + 0.5 * lin_bits * inv_n;
    };

    auto fill_best = [&](BeamState& s) {
        s.missing = 0;
        for (int k = 0; k < P; ++k) {
            const std::uint32_t y = req[static_cast<std::size_t>(k)];
            int b1 = 99;
            int b2 = 99;
            int w1 = -1;
            for (int i = 0; i < n; ++i) {
                const int d = popcount32(s.rows[static_cast<std::size_t>(i)] ^ y);
                if (d < b1) {
                    b2 = b1;
                    b1 = d;
                    w1 = i;
                } else if (d < b2) {
                    b2 = d;
                }
            }
            s.best1[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(b1);
            s.best2[static_cast<std::size_t>(k)] = static_cast<std::uint8_t>(b2);
            s.who1[static_cast<std::size_t>(k)] = static_cast<std::int8_t>(w1);
            if (((s.seen[static_cast<std::size_t>(k) / 64] >> (k % 64)) & 1ULL) == 0)
                ++s.missing;
        }
    };
    auto state_hash = [&](const std::vector<std::uint32_t>& rows, const std::vector<std::uint64_t>& seen) {
        std::uint64_t h = 0x1234567ULL;
        for (std::uint32_t r : rows)
            h = Mix(h, r);
        for (std::uint64_t w : seen)
            h = Mix(h, w);
        return h;
    };
    auto is_goal = [&](const BeamState& s) {
        if (s.missing != 0)
            return false;
        for (int i = 0; i < n; ++i) {
            if (row_dist(i, s.resid[static_cast<std::size_t>(i)]) != 0)
                return false;
        }
        return true;
    };

    // 走过的步：(上一步下标, 边)
    std::vector<std::pair<int, int>> hist;
    BeamState s0;
    s0.rows.resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        s0.rows[static_cast<std::size_t>(i)] = 1U << i;
    s0.resid = tinv;  // E = I · T⁻¹
    s0.seen.assign(words, 0);
    for (int i = 0; i < n; ++i) {
        const int k = term_of(1U << i);
        if (k >= 0)
            s0.seen[static_cast<std::size_t>(k) / 64] |= 1ULL << (k % 64);
    }
    s0.best1.resize(static_cast<std::size_t>(P));
    s0.best2.resize(static_cast<std::size_t>(P));
    s0.who1.resize(static_cast<std::size_t>(P));
    fill_best(s0);
    auto finish = [&](const BeamState& s) {
        for (int h = s.history; h >= 0; h = hist[static_cast<std::size_t>(h)].first)
            result.moves.push_back(edges[static_cast<std::size_t>(hist[static_cast<std::size_t>(h)].second)]);
        std::reverse(result.moves.begin(), result.moves.end());
        result.perm.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const std::uint32_t e = s.resid[static_cast<std::size_t>(i)];
            int j = i;
            if ((free_mask >> i) & 1U) {
                for (int b = 0; b < n; ++b) {
                    if (e == (1U << b)) {
                        j = b;
                        break;
                    }
                }
            }
            result.perm[static_cast<std::size_t>(i)] = j;
        }
        result.found = true;
    };
    if (is_goal(s0)) {
        finish(s0);
        return result;
    }
    std::unordered_set<std::uint64_t> visited;
    visited.insert(state_hash(s0.rows, s0.seen));
    std::vector<BeamState> beam{s0};
    std::vector<Child> children;
    beam_width = std::max(1, beam_width);
    for (int layer = 1; layer <= max_cx; ++layer) {
        children.clear();
        for (int bi = 0; bi < static_cast<int>(beam.size()); ++bi) {
            const BeamState& s = beam[static_cast<std::size_t>(bi)];
            int lin_rows = 0;
            int lin_bits = 0;
            for (int i = 0; i < n; ++i) {
                const int d = row_dist(i, s.resid[static_cast<std::size_t>(i)]);
                lin_rows += d != 0 ? 1 : 0;
                lin_bits += d;
            }
            for (int e = 0; e < static_cast<int>(edges.size()); ++e) {
                const int c = edges[static_cast<std::size_t>(e)].first;
                const int t = edges[static_cast<std::size_t>(e)].second;
                const std::uint32_t nr = s.rows[static_cast<std::size_t>(t)] ^ s.rows[static_cast<std::size_t>(c)];
                const std::uint32_t ne = s.resid[static_cast<std::size_t>(t)] ^ s.resid[static_cast<std::size_t>(c)];
                const int k_new = term_of(nr);
                const bool fresh = k_new >= 0
                                   && ((s.seen[static_cast<std::size_t>(k_new) / 64] >> (k_new % 64)) & 1ULL) == 0;
                const int d_old = row_dist(t, s.resid[static_cast<std::size_t>(t)]);
                const int d_new = row_dist(t, ne);
                const int nl_rows = lin_rows - (d_old != 0 ? 1 : 0) + (d_new != 0 ? 1 : 0);
                const int nl_bits = lin_bits - d_old + d_new;
                // 可采纳的下界（同 A*）：缺的必需项数 + 固定线中与目标不同、且目标值不是缺的必需项的行数；
                // 层数加下界超过上限的子状态不可能在上限内完成
                int lb = s.missing - (fresh ? 1 : 0);
                for (int i = 0; i < n; ++i) {
                    if ((free_mask >> i) & 1U)
                        continue;
                    const std::uint32_t ei = i == t ? ne : s.resid[static_cast<std::size_t>(i)];
                    if (ei == (1U << i))
                        continue;
                    const int kt = term_of(target_rows[static_cast<std::size_t>(i)]);
                    const bool tmiss = kt >= 0 && kt != k_new
                                       && ((s.seen[static_cast<std::size_t>(kt) / 64] >> (kt % 64)) & 1ULL) == 0;
                    lb += tmiss ? 0 : 1;
                }
                if (layer + lb > max_cx)
                    continue;
                int phase_dist = 0;
                for (int k = 0; k < P; ++k) {
                    if ((s.seen[static_cast<std::size_t>(k) / 64] >> (k % 64)) & 1ULL)
                        continue;
                    if (k == k_new)
                        continue;
                    const int other = s.who1[static_cast<std::size_t>(k)] == t ? s.best2[static_cast<std::size_t>(k)]
                                                                               : s.best1[static_cast<std::size_t>(k)];
                    phase_dist += std::min(other, popcount32(nr ^ req[static_cast<std::size_t>(k)]));
                }
                const double sc = score_of(s.missing - (fresh ? 1 : 0), phase_dist, nl_rows, nl_bits);
                children.push_back(Child{sc, bi, e, 0});
            }
        }
        // 确定性排序：分数，再按 (父, 边) 顺序
        std::stable_sort(children.begin(), children.end(), [](const Child& a, const Child& b) {
            if (a.score != b.score)
                return a.score < b.score;
            if (a.parent != b.parent)
                return a.parent < b.parent;
            return a.edge < b.edge;
        });
        std::vector<BeamState> next;
        next.reserve(static_cast<std::size_t>(beam_width));
        for (const Child& ch : children) {
            if (static_cast<int>(next.size()) >= beam_width)
                break;
            const BeamState& s = beam[static_cast<std::size_t>(ch.parent)];
            const int c = edges[static_cast<std::size_t>(ch.edge)].first;
            const int t = edges[static_cast<std::size_t>(ch.edge)].second;
            BeamState ns;
            ns.rows = s.rows;
            ns.resid = s.resid;
            ns.seen = s.seen;
            ns.rows[static_cast<std::size_t>(t)] ^= ns.rows[static_cast<std::size_t>(c)];
            ns.resid[static_cast<std::size_t>(t)] ^= ns.resid[static_cast<std::size_t>(c)];
            const int k_new = term_of(ns.rows[static_cast<std::size_t>(t)]);
            if (k_new >= 0)
                ns.seen[static_cast<std::size_t>(k_new) / 64] |= 1ULL << (k_new % 64);
            const std::uint64_t h = state_hash(ns.rows, ns.seen);
            if (!visited.insert(h).second)
                continue;
            ns.best1.resize(static_cast<std::size_t>(P));
            ns.best2.resize(static_cast<std::size_t>(P));
            ns.who1.resize(static_cast<std::size_t>(P));
            fill_best(ns);
            hist.emplace_back(s.history, ch.edge);
            ns.history = static_cast<int>(hist.size()) - 1;
            result.expanded += 1;
            if (is_goal(ns)) {
                finish(ns);
                return result;
            }
            next.push_back(std::move(ns));
        }
        if (next.empty())
            break;
        beam = std::move(next);
    }
    return result;
}

}  // namespace quspect::native::transpiler
