#include "quspect_native/transpiler/cnot_phase_synth.hpp"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <stdexcept>
#include <unordered_map>

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
                                    const std::vector<std::uint32_t>& required, int max_cx, std::int64_t node_budget) {
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

    auto heuristic = [&](std::uint64_t rows, std::uint64_t seen) {
        const int h1 = popcount64(full & ~seen);
        int h2 = 0;
        for (int i = 0; i < n; ++i)
            h2 += row_of(rows, i) != row_of(target, i) ? 1 : 0;
        return std::max(h1, h2);
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
        if (cur.key.rows == target && cur.key.seen == full) {
            Key k = cur.key;
            while (true) {
                const Info& in = info.at(k);
                if (in.move < 0)
                    break;
                result.moves.push_back(edges[static_cast<std::size_t>(in.move)]);
                k = in.parent;
            }
            std::reverse(result.moves.begin(), result.moves.end());
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

}  // namespace quspect::native::transpiler
