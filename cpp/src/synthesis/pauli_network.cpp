// Pauli 旋转序列在耦合图上的 Clifford 框架贪心综合（见 pauli_network.hpp）。
//
// 只用整数运算：Pauli 为 (x, z) 位串 + 符号位（Aaronson–Gottesman 的符号形式，Y 记作 x = z = 1），
// 计分为 int64，结果与平台无关。
#include "quspect_native/synthesis/pauli_network.hpp"

#include "quspect_native/common/bit_ops.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quspect::native::synthesis {
namespace {

using Word = std::uint64_t;

// 单比特 Pauli 编码：bit0 = x，bit1 = z。I = 0, X = 1, Z = 2, Y = 3。
constexpr int kI = 0;
constexpr int kX = 1;
constexpr int kZ = 2;
constexpr int kY = 3;

inline bool Anti1(int p, int s) { return (((p & 1) & (s >> 1)) ^ ((p >> 1) & (s & 1))) != 0; }

// 与 p 反对易的两个 Pauli（固定顺序，保证确定性）。
inline std::array<int, 2> AntiOf(int p) {
    switch (p) {
        case kX:
            return {kY, kZ};
        case kY:
            return {kX, kZ};
        default:
            return {kX, kY};
    }
}

// 两比特 Clifford C(σa, σb) 作用在 (a, b) 上（"Pauli 控制 Pauli"，1 个 CX）。
struct Tqe {
    int sa;
    int sb;
    int a;
    int b;
};

// 支集上的快速规则（不计符号）：P_a 与 σa 反对易 → b 乘 σb；P_b 与 σb 反对易 → a 乘 σa。
inline void TqeLocal(const Tqe& g, int& pa, int& pb) {
    const bool anti_a = pa != kI && Anti1(pa, g.sa);
    const bool anti_b = pb != kI && Anti1(pb, g.sb);
    const int na = anti_b ? (pa ^ g.sa) : pa;
    const int nb = anti_a ? (pb ^ g.sb) : pb;
    pa = na;
    pb = nb;
}

class PauliTable {
public:
    PauliTable(int words, int count) : words_(words), x_(static_cast<std::size_t>(words) * count, 0),
                                       z_(static_cast<std::size_t>(words) * count, 0), neg_(count, 0) {}

    int words() const { return words_; }
    int size() const { return static_cast<int>(neg_.size()); }
    int AddRow() {
        x_.resize(x_.size() + words_, 0);
        z_.resize(z_.size() + words_, 0);
        neg_.push_back(0);
        return size() - 1;
    }
    Word* x(int i) { return x_.data() + static_cast<std::size_t>(i) * words_; }
    Word* z(int i) { return z_.data() + static_cast<std::size_t>(i) * words_; }
    const Word* x(int i) const { return x_.data() + static_cast<std::size_t>(i) * words_; }
    const Word* z(int i) const { return z_.data() + static_cast<std::size_t>(i) * words_; }
    std::uint8_t& neg(int i) { return neg_[i]; }
    std::uint8_t neg(int i) const { return neg_[i]; }

    int Local(int i, int q) const {
        const int w = q >> 6;
        const Word bit = Word{1} << (q & 63);
        return ((x(i)[w] & bit) ? 1 : 0) | ((z(i)[w] & bit) ? 2 : 0);
    }
    void SetLocal(int i, int q, int p) {
        const int w = q >> 6;
        const Word bit = Word{1} << (q & 63);
        x(i)[w] = (p & 1) ? (x(i)[w] | bit) : (x(i)[w] & ~bit);
        z(i)[w] = (p & 2) ? (z(i)[w] | bit) : (z(i)[w] & ~bit);
    }

    // 单比特 / CX 共轭（带符号）。
    void H(int i, int q) {
        const int p = Local(i, q);
        if (p == kY) {
            neg(i) ^= 1;
        }
        SetLocal(i, q, p == kX ? kZ : p == kZ ? kX : p);
    }
    void S(int i, int q) {  // X → Y, Y → −X
        const int p = Local(i, q);
        if (p & 1) {
            if (p & 2) {
                neg(i) ^= 1;
            }
            SetLocal(i, q, p ^ 2);
        }
    }
    void Sdg(int i, int q) {  // X → −Y, Y → X
        const int p = Local(i, q);
        if (p & 1) {
            if (!(p & 2)) {
                neg(i) ^= 1;
            }
            SetLocal(i, q, p ^ 2);
        }
    }
    void Cx(int i, int c, int t) {
        const int pc = Local(i, c);
        const int pt = Local(i, t);
        const bool xc = pc & 1, zc = pc & 2, xt = pt & 1, zt = pt & 2;
        if (xc && zt && (xt == zc)) {
            neg(i) ^= 1;
        }
        const int nt = pt ^ (xc ? 1 : 0);
        const int nc = pc ^ (zt ? 2 : 0);
        SetLocal(i, c, nc);
        SetLocal(i, t, nt);
    }
    void PauliGate(int i, int q, int g) {  // 用 Pauli g 共轭：与 g 反对易时变号
        const int p = Local(i, q);
        if (p != kI && Anti1(p, g)) {
            neg(i) ^= 1;
        }
    }

    bool Anticommute(int i, int j) const {
        int par = 0;
        for (int w = 0; w < words_; ++w) {
            par ^= common::Popcount64((x(i)[w] & z(j)[w]) ^ (z(i)[w] & x(j)[w])) & 1;
        }
        return par != 0;
    }
    int Weight(int i) const {
        int c = 0;
        for (int w = 0; w < words_; ++w) {
            c += common::Popcount64(x(i)[w] | z(i)[w]);
        }
        return c;
    }
    bool Touches(int i, int q) const {
        const int w = q >> 6;
        const Word bit = Word{1} << (q & 63);
        return ((x(i)[w] | z(i)[w]) & bit) != 0;
    }
    void Support(int i, std::vector<int>& out) const {
        out.clear();
        for (int w = 0; w < words_; ++w) {
            Word s = x(i)[w] | z(i)[w];
            while (s) {
                const int b = common::Ctz64(s);
                out.push_back(w * 64 + b);
                s &= s - 1;
            }
        }
    }
    void Copy(int dst, const PauliTable& src, int i) {
        std::copy(src.x(i), src.x(i) + words_, x(dst));
        std::copy(src.z(i), src.z(i) + words_, z(dst));
        neg(dst) = src.neg(i);
    }

private:
    int words_;
    std::vector<Word> x_;
    std::vector<Word> z_;
    std::vector<std::uint8_t> neg_;
};

// 基本门（作用在线路末尾）。
struct Elem {
    NetGateKind kind;
    int q0;
    int q1;
};

void ToZ(int s, int q, std::vector<Elem>& out) {  // σ → Z（作为控制）
    if (s == kX) {
        out.push_back({NetGateKind::kH, q, -1});
    } else if (s == kY) {
        out.push_back({NetGateKind::kSdg, q, -1});
        out.push_back({NetGateKind::kH, q, -1});
    }
}

void ToX(int s, int q, std::vector<Elem>& out) {  // σ → X（作为目标）
    if (s == kZ) {
        out.push_back({NetGateKind::kH, q, -1});
    } else if (s == kY) {
        out.push_back({NetGateKind::kSdg, q, -1});
    }
}

void AppendTqe(const Tqe& g, std::vector<Elem>& out) {
    std::vector<Elem> pre;
    ToZ(g.sa, g.a, pre);
    ToX(g.sb, g.b, pre);
    out.insert(out.end(), pre.begin(), pre.end());
    out.push_back({NetGateKind::kCX, g.a, g.b});
    for (auto it = pre.rbegin(); it != pre.rend(); ++it) {
        Elem e = *it;
        if (e.kind == NetGateKind::kS) {
            e.kind = NetGateKind::kSdg;
        } else if (e.kind == NetGateKind::kSdg) {
            e.kind = NetGateKind::kS;
        }
        out.push_back(e);
    }
}

void ApplyElem(PauliTable& t, int i, const Elem& e) {
    switch (e.kind) {
        case NetGateKind::kH:
            t.H(i, e.q0);
            break;
        case NetGateKind::kS:
            t.S(i, e.q0);
            break;
        case NetGateKind::kSdg:
            t.Sdg(i, e.q0);
            break;
        case NetGateKind::kCX:
            t.Cx(i, e.q0, e.q1);
            break;
        case NetGateKind::kX:
            t.PauliGate(i, e.q0, kX);
            break;
        case NetGateKind::kY:
            t.PauliGate(i, e.q0, kY);
            break;
        case NetGateKind::kZ:
            t.PauliGate(i, e.q0, kZ);
            break;
        default:
            break;
    }
}

// 一个候选：1 个（约化）或 2 个（跳跃）TQE。
struct Candidate {
    std::array<Tqe, 2> g;
    int count;
};

class Graph {
public:
    Graph(int n, const std::vector<std::pair<int, int>>& edges, const std::vector<int>& active)
        : n_(n), adj_(n), in_(n, 0) {
        for (int q : active) {
            if (q < 0 || q >= n) {
                throw std::invalid_argument("active qubit out of range");
            }
            in_[q] = 1;
        }
        for (const auto& [a, b] : edges) {
            if (a < 0 || b < 0 || a >= n || b >= n) {
                throw std::invalid_argument("edge out of range");
            }
            if (a == b || !in_[a] || !in_[b]) {
                continue;
            }
            adj_[a].push_back(b);
            adj_[b].push_back(a);
        }
        for (auto& v : adj_) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        Recompute();
        for (int q : active) {
            if (dist_[active.front()][q] >= kFar) {
                throw std::invalid_argument("active qubits are not connected by the coupling edges");
            }
        }
    }

    static constexpr int kFar = 1 << 20;

    int n() const { return n_; }
    bool active(int q) const { return in_[q] != 0; }
    const std::vector<int>& adj(int q) const { return adj_[q]; }
    int dist(int a, int b) const { return dist_[a][b]; }

    void Deactivate(int q) {
        in_[q] = 0;
        Recompute();
    }

    // 活动子图中的非割点（升序）。
    std::vector<int> NonCutVertices() const {
        std::vector<int> nodes;
        for (int q = 0; q < n_; ++q) {
            if (in_[q]) {
                nodes.push_back(q);
            }
        }
        if (nodes.size() <= 2) {
            return nodes;
        }
        std::vector<int> disc(n_, -1), low(n_, 0), parent(n_, -1);
        std::vector<std::uint8_t> art(n_, 0);
        int timer = 0;
        // 迭代 DFS（Tarjan）
        const int root = nodes.front();
        std::vector<std::pair<int, std::size_t>> stack;
        stack.push_back({root, 0});
        disc[root] = low[root] = timer++;
        int root_children = 0;
        while (!stack.empty()) {
            auto& [u, idx] = stack.back();
            const auto& nb = adj_[u];
            if (idx < nb.size()) {
                const int v = nb[idx++];
                if (!in_[v]) {
                    continue;
                }
                if (disc[v] < 0) {
                    parent[v] = u;
                    disc[v] = low[v] = timer++;
                    if (u == root) {
                        ++root_children;
                    }
                    stack.push_back({v, 0});
                } else if (v != parent[u]) {
                    low[u] = std::min(low[u], disc[v]);
                }
            } else {
                const int done = u;
                stack.pop_back();
                if (!stack.empty()) {
                    const int p = stack.back().first;
                    low[p] = std::min(low[p], low[done]);
                    if (p != root && low[done] >= disc[p]) {
                        art[p] = 1;
                    }
                }
            }
        }
        if (root_children > 1) {
            art[root] = 1;
        }
        std::vector<int> out;
        for (int q : nodes) {
            if (!art[q]) {
                out.push_back(q);
            }
        }
        return out;
    }

private:
    void Recompute() {
        dist_.assign(n_, std::vector<int>(n_, kFar));
        std::deque<int> queue;
        for (int s = 0; s < n_; ++s) {
            if (!in_[s]) {
                continue;
            }
            auto& d = dist_[s];
            d[s] = 0;
            queue.assign(1, s);
            while (!queue.empty()) {
                const int u = queue.front();
                queue.pop_front();
                for (int v : adj_[u]) {
                    if (in_[v] && d[v] > d[u] + 1) {
                        d[v] = d[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
    }

    int n_;
    std::vector<std::vector<int>> adj_;
    std::vector<std::uint8_t> in_;
    std::vector<std::vector<int>> dist_;
};

// 支集的 Steiner 树代价估计：度量闭包上的最小生成树，边权 2d − 1（相邻 1 个 CX，每多一跳 2 个）。
std::int64_t SupportCost(const Graph& g, const std::vector<int>& s) {
    const std::size_t k = s.size();
    if (k <= 1) {
        return 0;
    }
    std::vector<std::int64_t> best(k, std::numeric_limits<std::int64_t>::max());
    std::vector<std::uint8_t> used(k, 0);
    used[0] = 1;
    std::size_t u = 0;
    std::int64_t total = 0;
    for (std::size_t step = 1; step < k; ++step) {
        std::size_t pick = k;
        for (std::size_t v = 0; v < k; ++v) {
            if (used[v]) {
                continue;
            }
            const std::int64_t w = 2 * static_cast<std::int64_t>(g.dist(s[u], s[v])) - 1;
            if (w < best[v]) {
                best[v] = w;
            }
            if (pick == k || best[v] < best[pick]) {
                pick = v;
            }
        }
        used[pick] = 1;
        total += best[pick];
        u = pick;
    }
    return total;
}

class Synthesizer {
public:
    Synthesizer(int n, const std::vector<std::pair<int, int>>& edges, const std::vector<int>& active,
                const PauliNetworkOptions& options)
        : graph_(n, edges, active), options_(options), words_((n + 63) / 64), frame_(words_, 2 * n), scratch_(words_, 1) {
        for (int q = 0; q < n; ++q) {
            frame_.SetLocal(2 * q, q, kX);
            frame_.SetLocal(2 * q + 1, q, kZ);
        }
    }

    PauliNetworkResult Run(const std::vector<std::vector<int>>& xs, const std::vector<std::vector<int>>& zs) {
        const int m = static_cast<int>(xs.size());
        PauliTable orig(words_, m);
        for (int k = 0; k < m; ++k) {
            for (int q : xs[k]) {
                Check(q);
                orig.SetLocal(k, q, orig.Local(k, q) | 1);
            }
            for (int q : zs[k]) {
                Check(q);
                orig.SetLocal(k, q, orig.Local(k, q) | 2);
            }
        }
        // 当前框架下的像：只为前 span 个未实现的旋转维护
        const int span = std::max(options_.front, options_.window + 1);
        cur_ = PauliTable(words_, m);
        std::vector<int> remaining;
        remaining.reserve(m);
        for (int k = 0; k < m; ++k) {
            if (orig.Weight(k) > 0) {
                remaining.push_back(k);
            }
        }
        materialized_.assign(m, 0);
        auto materialize = [&]() {
            const int upto = std::min(span, static_cast<int>(remaining.size()));
            for (int pos = 0; pos < upto; ++pos) {
                const int k = remaining[pos];
                if (!materialized_[k]) {
                    Image(orig, k, cur_, k);
                    materialized_[k] = 1;
                    live_.push_back(k);
                }
            }
        };
        materialize();
        std::vector<int> supp;
        std::vector<int> window;
        std::vector<std::int64_t> wcost;
        while (!remaining.empty()) {
            // 对易前沿中代价最小者
            const int lim = std::min(options_.front, static_cast<int>(remaining.size()));
            int pick_pos = -1;
            std::int64_t pick_cost = 0;
            for (int pos = 0; pos < lim; ++pos) {
                const int k = remaining[pos];
                bool ok = true;
                for (int prev = 0; prev < pos && ok; ++prev) {
                    ok = !cur_.Anticommute(k, remaining[prev]);
                }
                if (!ok) {
                    continue;
                }
                cur_.Support(k, supp);
                const std::int64_t c = SupportCost(graph_, supp);
                if (pick_pos < 0 || c < pick_cost) {
                    pick_pos = pos;
                    pick_cost = c;
                }
                if (c == 0) {
                    break;
                }
            }
            const int k = remaining[pick_pos];
            window.clear();
            wcost.clear();
            for (int pos = 0; pos < static_cast<int>(remaining.size()) &&
                              static_cast<int>(window.size()) < options_.window;
                 ++pos) {
                if (pos == pick_pos) {
                    continue;
                }
                window.push_back(remaining[pos]);
                cur_.Support(remaining[pos], supp);
                wcost.push_back(SupportCost(graph_, supp));
            }
            ReduceRotation(k, window, wcost);
            cur_.Support(k, supp);
            const int q = supp.front();
            NetGate rot{NetGateKind::kRotation, q, -1, k, "?XZY"[cur_.Local(k, q)], cur_.neg(k) != 0};
            result_.gates.push_back(rot);
            result_.order.push_back(k);
            remaining.erase(remaining.begin() + pick_pos);
            live_.erase(std::find(live_.begin(), live_.end(), k));
            materialize();
        }
        if (options_.finalize) {
            const int before = result_.num_cx;
            Finalize();
            result_.finalize_cx = result_.num_cx - before;
        }
        return std::move(result_);
    }

private:
    void Check(int q) const {
        if (q < 0 || q >= graph_.n()) {
            throw std::invalid_argument("rotation qubit out of range");
        }
        if (!graph_.active(q)) {
            throw std::invalid_argument("rotation acts on a qubit outside the active set");
        }
    }

    // src[i] 在当前框架下的像写入 dst[j]：P = (−1)^neg Π_q σ_q，Y = i X Z。
    void Image(const PauliTable& src, int i, PauliTable& dst, int j) {
        // e 形式：i^e X^x Z^z
        std::vector<Word> ax(words_, 0), az(words_, 0);
        int e = 0;
        auto mul = [&](int row) {
            // 行（符号形式）→ e 形式
            const Word* rx = frame_.x(row);
            const Word* rz = frame_.z(row);
            int er = 2 * frame_.neg(row);
            int cross = 0;
            for (int w = 0; w < words_; ++w) {
                er += common::Popcount64(rx[w] & rz[w]);
                cross += common::Popcount64(az[w] & rx[w]);
            }
            e += er + 2 * cross;
            for (int w = 0; w < words_; ++w) {
                ax[w] ^= rx[w];
                az[w] ^= rz[w];
            }
        };
        std::vector<int> supp;
        src.Support(i, supp);
        for (int q : supp) {
            const int p = src.Local(i, q);
            if (p & 1) {
                mul(2 * q);
            }
            if (p & 2) {
                mul(2 * q + 1);
            }
            if (p == kY) {
                e += 1;
            }
        }
        e += 2 * src.neg(i);
        int ny = 0;
        for (int w = 0; w < words_; ++w) {
            ny += common::Popcount64(ax[w] & az[w]);
        }
        const int s = ((e - ny) % 4 + 4) % 4;
        if (s & 1) {
            throw std::logic_error("non-Hermitian Pauli image");
        }
        std::copy(ax.begin(), ax.end(), dst.x(j));
        std::copy(az.begin(), az.end(), dst.z(j));
        dst.neg(j) = static_cast<std::uint8_t>(s >> 1);
    }

    void Emit(const Candidate& c) {
        std::vector<Elem> elems;
        for (int t = 0; t < c.count; ++t) {
            AppendTqe(c.g[t], elems);
        }
        for (const Elem& e : elems) {
            for (int k : live_) {
                ApplyElem(cur_, k, e);
            }
            for (int r = 0; r < frame_.size(); ++r) {
                ApplyElem(frame_, r, e);
            }
            result_.gates.push_back({e.kind, e.q0, e.q1, -1, 0, false});
            if (e.kind == NetGateKind::kCX) {
                ++result_.num_cx;
            }
        }
    }

    // 候选对支集的作用（不计符号），写入 scratch_ 的第 0 行。
    void Preview(const PauliTable& t, int i, const Candidate& c) {
        scratch_.Copy(0, t, i);
        for (int s = 0; s < c.count; ++s) {
            const Tqe& g = c.g[s];
            int pa = scratch_.Local(0, g.a);
            int pb = scratch_.Local(0, g.b);
            TqeLocal(g, pa, pb);
            scratch_.SetLocal(0, g.a, pa);
            scratch_.SetLocal(0, g.b, pb);
        }
    }

    // 把 t 的第 i 行约化到 root（root < 0：任意单比特）的候选门。
    //   keep_x ≥ 0：frame_ 的第 keep_x 行已是 root 上的单比特，触及 root 的门必须保持它不变。
    void Candidates(const PauliTable& t, int i, int root, int keep_x, std::vector<Candidate>& out) {
        out.clear();
        std::vector<int> supp;
        t.Support(i, supp);
        auto in_supp = [&](int q) { return t.Touches(i, q); };
        const int xl = keep_x >= 0 ? frame_.Local(keep_x, root) : kI;
        for (int a : supp) {
            for (int b : graph_.adj(a)) {
                if (b <= a || !in_supp(b)) {
                    continue;
                }
                const int pa = t.Local(i, a);
                const int pb = t.Local(i, b);
                if (keep_x >= 0 && (a == root || b == root)) {
                    if (a == root) {
                        out.push_back({{Tqe{xl, pb, a, b}, Tqe{}}, 1});
                    } else {
                        out.push_back({{Tqe{pa, xl, a, b}, Tqe{}}, 1});
                    }
                    continue;
                }
                if (b != root) {  // 消去 b
                    for (int sa : AntiOf(pa)) {
                        out.push_back({{Tqe{sa, pb, a, b}, Tqe{}}, 1});
                    }
                }
                if (a != root) {  // 消去 a
                    for (int sb : AntiOf(pb)) {
                        out.push_back({{Tqe{pa, sb, a, b}, Tqe{}}, 1});
                    }
                }
            }
        }
        if (!out.empty()) {
            return;
        }
        // 跳跃：把 a 上的 Pauli 移到更靠近其余支集（及 root）的邻居 b 上
        std::vector<int> targets = supp;
        if (root >= 0 && !in_supp(root)) {
            targets.push_back(root);
        }
        for (int a : supp) {
            if (a == root) {
                continue;
            }
            int da = Graph::kFar;
            for (int s : targets) {
                if (s != a) {
                    da = std::min(da, graph_.dist(a, s));
                }
            }
            const int pa = t.Local(i, a);
            for (int b : graph_.adj(a)) {
                if (in_supp(b) || (keep_x >= 0 && b == root)) {
                    continue;
                }
                int db = Graph::kFar;
                for (int s : targets) {
                    if (s != a) {
                        db = std::min(db, graph_.dist(b, s));
                    }
                }
                if (db >= da) {
                    continue;
                }
                for (int sa : AntiOf(pa)) {
                    for (int sb : {kX, kY, kZ}) {
                        for (int sb2 : AntiOf(sb)) {
                            out.push_back({{Tqe{sa, sb, a, b}, Tqe{pa, sb2, a, b}}, 2});
                        }
                    }
                }
            }
        }
    }

    void ReduceRotation(int k, const std::vector<int>& window, std::vector<std::int64_t>& wcost) {
        std::vector<Candidate> cands;
        std::vector<int> supp;
        std::vector<std::int64_t> weights(window.size());
        std::int64_t w = options_.next_weight;
        for (std::size_t j = 0; j < window.size(); ++j) {
            weights[j] = w;
            w = (w * options_.decay) / 1024;
        }
        while (cur_.Weight(k) > 1) {
            Candidates(cur_, k, -1, -1, cands);
            if (cands.empty()) {
                throw std::logic_error("pauli network: no reducing gate (disconnected support)");
            }
            std::size_t best = 0;
            std::int64_t best_score = 0;
            for (std::size_t c = 0; c < cands.size(); ++c) {
                Preview(cur_, k, cands[c]);
                scratch_.Support(0, supp);
                std::int64_t score = 1024 * SupportCost(graph_, supp);
                for (std::size_t j = 0; j < window.size(); ++j) {
                    const int r = window[j];
                    bool touched = false;
                    for (int s = 0; s < cands[c].count && !touched; ++s) {
                        touched = cur_.Touches(r, cands[c].g[s].a) || cur_.Touches(r, cands[c].g[s].b);
                    }
                    if (!touched) {
                        continue;
                    }
                    Preview(cur_, r, cands[c]);
                    scratch_.Support(0, supp);
                    score += weights[j] * (SupportCost(graph_, supp) - wcost[j]);
                }
                if (c == 0 || score < best_score) {
                    best = c;
                    best_score = score;
                }
            }
            Emit(cands[best]);
            for (std::size_t j = 0; j < window.size(); ++j) {
                cur_.Support(window[j], supp);
                wcost[j] = SupportCost(graph_, supp);
            }
        }
    }

    std::int64_t RootCost(const PauliTable& t, int i, int root) {
        std::vector<int> supp;
        t.Support(i, supp);
        const bool has = t.Touches(i, root);
        if (!has) {
            supp.insert(std::lower_bound(supp.begin(), supp.end(), root), root);
        }
        return SupportCost(graph_, supp) + (has ? 0 : 1);
    }

    // frame_ 的第 row 行约化到 root 上的单比特（keep_x ≥ 0 时保持该行不变）。
    void ReduceRow(int row, int root, int keep_x) {
        std::vector<Candidate> cands;
        std::vector<int> supp;
        for (;;) {
            frame_.Support(row, supp);
            if (supp.size() == 1 && supp.front() == root) {
                return;
            }
            Candidates(frame_, row, root, keep_x, cands);
            if (cands.empty()) {
                throw std::logic_error("pauli network: frame finalization got stuck");
            }
            std::size_t best = 0;
            std::int64_t best_score = 0;
            for (std::size_t c = 0; c < cands.size(); ++c) {
                Preview(frame_, row, cands[c]);
                const std::int64_t score = RootCost(scratch_, 0, root);
                if (c == 0 || score < best_score) {
                    best = c;
                    best_score = score;
                }
            }
            Emit(cands[best]);
        }
    }

    // 单比特 Clifford（H/S 字 + Pauli）把 (行 X, 行 Z) 在 q 上变回 (+X, +Z)。
    void FixSingle(int q) {
        const int rx = 2 * q;
        const int rz = 2 * q + 1;
        for (int len = 0; len <= 6; ++len) {
            for (int word = 0; word < (1 << len); ++word) {
                for (int pg = 0; pg < 4; ++pg) {
                    std::vector<Elem> elems;
                    for (int b = 0; b < len; ++b) {
                        elems.push_back({((word >> b) & 1) ? NetGateKind::kS : NetGateKind::kH, q, -1});
                    }
                    if (pg > 0) {
                        elems.push_back({pg == 1 ? NetGateKind::kX : pg == 2 ? NetGateKind::kZ : NetGateKind::kY, q, -1});
                    }
                    PauliTable t(words_, 2);
                    t.Copy(0, frame_, rx);
                    t.Copy(1, frame_, rz);
                    for (const Elem& e : elems) {
                        ApplyElem(t, 0, e);
                        ApplyElem(t, 1, e);
                    }
                    if (t.Weight(0) == 1 && t.Local(0, q) == kX && !t.neg(0) && t.Weight(1) == 1 &&
                        t.Local(1, q) == kZ && !t.neg(1)) {
                        for (const Elem& e : elems) {
                            for (int r = 0; r < frame_.size(); ++r) {
                                ApplyElem(frame_, r, e);
                            }
                            result_.gates.push_back({e.kind, e.q0, e.q1, -1, 0, false});
                        }
                        return;
                    }
                }
            }
        }
        throw std::logic_error("pauli network: single-qubit fix failed");
    }

    void Finalize() {
        live_.clear();
        int left = 0;
        for (int q = 0; q < graph_.n(); ++q) {
            left += graph_.active(q) ? 1 : 0;
        }
        while (left > 0) {
            const std::vector<int> cand = graph_.NonCutVertices();
            int q = -1;
            std::int64_t qc = 0;
            for (int v : cand) {
                const std::int64_t c = RootCost(frame_, 2 * v, v) + RootCost(frame_, 2 * v + 1, v);
                if (q < 0 || c < qc) {
                    q = v;
                    qc = c;
                }
            }
            ReduceRow(2 * q, q, -1);
            ReduceRow(2 * q + 1, q, 2 * q);
            FixSingle(q);
            graph_.Deactivate(q);
            --left;
        }
        // 检查：框架应为恒等
        for (int q = 0; q < graph_.n(); ++q) {
            if (frame_.Weight(2 * q) != 1 || frame_.Local(2 * q, q) != kX || frame_.neg(2 * q) ||
                frame_.Weight(2 * q + 1) != 1 || frame_.Local(2 * q + 1, q) != kZ || frame_.neg(2 * q + 1)) {
                throw std::logic_error("pauli network: frame is not the identity after finalization");
            }
        }
    }

    Graph graph_;
    PauliNetworkOptions options_;
    int words_;
    PauliTable frame_;    // 2q：F X_q F†，2q+1：F Z_q F†
    PauliTable scratch_;  // 计分用
    PauliTable cur_{1, 0};
    std::vector<std::uint8_t> materialized_;
    std::vector<int> live_;  // 已在当前框架下维护的未实现旋转
    PauliNetworkResult result_;
};

}  // namespace

PauliNetworkResult SynthesizePauliNetwork(int num_qubits, const std::vector<std::pair<int, int>>& edges,
                                          const std::vector<int>& active,
                                          const std::vector<std::vector<int>>& x_qubits,
                                          const std::vector<std::vector<int>>& z_qubits,
                                          const PauliNetworkOptions& options) {
    if (num_qubits <= 0) {
        throw std::invalid_argument("num_qubits must be positive");
    }
    if (active.empty()) {
        throw std::invalid_argument("active set is empty");
    }
    if (x_qubits.size() != z_qubits.size()) {
        throw std::invalid_argument("x_qubits and z_qubits differ in length");
    }
    if (options.window < 0 || options.front < 1) {
        throw std::invalid_argument("window must be >= 0 and front >= 1");
    }
    Synthesizer s(num_qubits, edges, active, options);
    return s.Run(x_qubits, z_qubits);
}

std::vector<int> MergeCommutingRotations(int num_qubits, const std::vector<std::vector<int>>& x_qubits,
                                         const std::vector<std::vector<int>>& z_qubits, int span) {
    if (x_qubits.size() != z_qubits.size()) {
        throw std::invalid_argument("x_qubits and z_qubits differ in length");
    }
    const int words = (num_qubits + 63) / 64;
    const int m = static_cast<int>(x_qubits.size());
    PauliTable in(words, m);
    for (int k = 0; k < m; ++k) {
        for (int q : x_qubits[k]) {
            if (q < 0 || q >= num_qubits) {
                throw std::invalid_argument("rotation qubit out of range");
            }
            in.SetLocal(k, q, in.Local(k, q) | 1);
        }
        for (int q : z_qubits[k]) {
            if (q < 0 || q >= num_qubits) {
                throw std::invalid_argument("rotation qubit out of range");
            }
            in.SetLocal(k, q, in.Local(k, q) | 2);
        }
    }
    std::vector<int> slot(m, -1);
    std::vector<int> out;  // 输出编号 → 代表输入
    for (int k = 0; k < m; ++k) {
        int found = -1;
        const int stop = std::max(0, static_cast<int>(out.size()) - span);
        for (int j = static_cast<int>(out.size()) - 1; j >= stop; --j) {
            const int r = out[j];
            bool same = true;
            for (int w = 0; w < words && same; ++w) {
                same = in.x(r)[w] == in.x(k)[w] && in.z(r)[w] == in.z(k)[w];
            }
            if (same) {
                found = j;
                break;
            }
            if (in.Anticommute(r, k)) {
                break;
            }
        }
        if (found < 0) {
            found = static_cast<int>(out.size());
            out.push_back(k);
        }
        slot[k] = found;
    }
    return slot;
}

}  // namespace quspect::native::synthesis
