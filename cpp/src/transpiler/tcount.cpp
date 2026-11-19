// T 门数优化的原生搜索内核，见 tcount.hpp。
//
// 每个函数都与 quspect/transpiler/_tcount_poly.py 中的同名 Python 参考实现逐步对应：相同的
// 迭代顺序、相同的择优规则（严格更优才替换，平局保留先出现的）、相同的随机数抽取序列。
// 修改其中一边时必须同步修改另一边；tests/transpiler/test_tcount.py 比较两者的输出。

#include "quspect_native/transpiler/tcount.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/common/parallel.hpp"
#include "quspect_native/transpiler/cpython_random.hpp"

namespace quspect::native::transpiler {

namespace {

// ---------------------------------------------------------------------------------------------
// 列类型：64 位掩码（std::uint64_t）或 256 位掩码（TCountWide）。内核对列类型是泛型的。
// ---------------------------------------------------------------------------------------------
inline bool IsZero(std::uint64_t c) {
    return c == 0;
}
inline bool IsZero(const TCountWide& c) {
    return (c.w[0] | c.w[1] | c.w[2] | c.w[3]) == 0;
}
inline bool BitOf(std::uint64_t c, int i) {
    return ((c >> i) & 1u) != 0;
}
inline bool BitOf(const TCountWide& c, int i) {
    return ((c.w[static_cast<std::size_t>(i >> 6)] >> (i & 63)) & 1u) != 0;
}
inline int TopBit(std::uint64_t c) {  // c != 0
    int h = 63;
    while (!((c >> h) & 1u))
        --h;
    return h;
}
inline int TopBit(const TCountWide& c) {  // c != 0
    for (int word = 3; word >= 0; --word)
        if (c.w[static_cast<std::size_t>(word)] != 0)
            return word * 64 + TopBit(c.w[static_cast<std::size_t>(word)]);
    return -1;
}
template <class Col>
struct ColumnBits;
template <>
struct ColumnBits<std::uint64_t> {
    static constexpr int value = 64;
};
template <>
struct ColumnBits<TCountWide> {
    static constexpr int value = 256;
};
using Word = std::uint64_t;

// ---------------------------------------------------------------------------------------------
// m 位的位向量（列下标上的 GF(2) 向量），按 64 位字存放。
// ---------------------------------------------------------------------------------------------
struct BitRows {
    int words = 0;
    std::vector<Word> data;  // 行连续存放

    int Count() const {
        return words == 0 ? 0 : static_cast<int>(data.size() / static_cast<std::size_t>(words));
    }
    Word* Row(int r) {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(words);
    }
    const Word* Row(int r) const {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(words);
    }
};

inline bool TestBit(const Word* v, int bit) {
    return (v[bit >> 6] >> (bit & 63)) & 1u;
}

inline void SetBit(Word* v, int bit) {
    v[bit >> 6] |= Word{1} << (bit & 63);
}

inline int HighBit(const Word* v, int words) {
    for (int w = words - 1; w >= 0; --w) {
        if (v[w] != 0) {
            Word x = v[w];
            int b = 63;
            while (!((x >> b) & 1u))
                --b;
            return w * 64 + b;
        }
    }
    return -1;
}

// 以最高位为主元的消元器。行空间确定之后，完全约化的形式唯一，所以插入顺序不影响结果。
struct Eliminator {
    int m = 0;
    int words = 0;
    std::vector<int> pivot_row;  // pivot_row[bit] = 行号或 -1
    BitRows rows;

    void Init(int m_) {
        m = m_;
        words = (m_ + 63) / 64;
        pivot_row.assign(static_cast<std::size_t>(m_), -1);
        rows.words = words;
        rows.data.clear();
    }

    // 插入一行（会被修改）。
    void Insert(Word* r) {
        while (true) {
            const int h = HighBit(r, words);
            if (h < 0)
                return;
            const int pr = pivot_row[static_cast<std::size_t>(h)];
            if (pr < 0) {
                pivot_row[static_cast<std::size_t>(h)] = rows.Count();
                rows.data.insert(rows.data.end(), r, r + words);
                return;
            }
            const Word* p = rows.Row(pr);
            for (int w = 0; w < words; ++w)
                r[w] ^= p[w];
        }
    }

    // 零空间的基（与 Python 的 _nullspace 相同的顺序：自由列升序）。
    std::vector<std::vector<Word>> Nullspace() {
        for (int p = 0; p < m; ++p) {  // 主元位升序做完全约化
            const int pr = pivot_row[static_cast<std::size_t>(p)];
            if (pr < 0)
                continue;
            const Word* prow = rows.Row(pr);
            const int count = rows.Count();
            for (int q = 0; q < count; ++q) {
                if (q == pr)
                    continue;
                Word* qrow = rows.Row(q);
                if (TestBit(qrow, p))
                    for (int w = 0; w < words; ++w)
                        qrow[w] ^= prow[w];
            }
        }
        std::vector<std::vector<Word>> basis;
        for (int f = 0; f < m; ++f) {
            if (pivot_row[static_cast<std::size_t>(f)] >= 0)
                continue;
            std::vector<Word> v(static_cast<std::size_t>(words), 0);
            SetBit(v.data(), f);
            for (int p = 0; p < m; ++p) {
                const int pr = pivot_row[static_cast<std::size_t>(p)];
                if (pr >= 0 && TestBit(rows.Row(pr), f))
                    SetBit(v.data(), p);
            }
            basis.push_back(std::move(v));
        }
        return basis;
    }
};

// 去掉零列并成对消去相同的列，保持首次出现的顺序（Python 的 normalise）。
template <class Col>
std::vector<Col> Normalise(const std::vector<Col>& cols) {
    std::unordered_map<Col, int> count;
    count.reserve(cols.size() * 2);
    for (Col c : cols)
        if (!IsZero(c))
            ++count[c];
    // Python 的字典语义：出现奇数次的值保留；删除后重新插入的值排到末尾。一个值出现三次以上时
    // 它的位置是最后一次"从无到有"的插入处，这里按该语义重放。
    bool simple = true;
    for (const auto& kv : count)
        if (kv.second > 2) {
            simple = false;
            break;
        }
    std::vector<Col> out;
    if (simple) {
        out.reserve(cols.size());
        for (Col c : cols)
            if (!IsZero(c) && count[c] == 1)
                out.push_back(c);
        return out;
    }
    std::vector<std::pair<Col, bool>> order;  // (值, 是否仍然存在)
    std::unordered_map<Col, std::size_t> where;
    for (Col c : cols) {
        if (IsZero(c))
            continue;
        auto it = where.find(c);
        if (it != where.end()) {
            order[it->second].second = false;
            where.erase(it);
        } else {
            where[c] = order.size();
            order.emplace_back(c, true);
        }
    }
    for (const auto& item : order)
        if (item.second)
            out.push_back(item.first);
    return out;
}

// A 的各行（变量 a 在哪些列中出现）。
template <class Col>
BitRows MatrixRows(const std::vector<Col>& cols, int k) {
    const int m = static_cast<int>(cols.size());
    BitRows rows;
    rows.words = (m + 63) / 64;
    rows.data.assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(rows.words), 0);
    for (int j = 0; j < m; ++j) {
        const Col& c = cols[static_cast<std::size_t>(j)];
        for (int a = 0; a < k; ++a)
            if (BitOf(c, a))
                SetBit(rows.Row(a), j);
    }
    return rows;
}

// 把依赖 z 的约束行插入消元器（Python 的 _constraints 中的二次、三次行）。
template <class Col>
void InsertZConstraints(Eliminator& el, const BitRows& rows, int k, Col z) {
    const int words = rows.words;
    std::vector<Word> r(static_cast<std::size_t>(words));
    std::vector<int> in_z(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i)
        in_z[static_cast<std::size_t>(i)] = BitOf(z, i) ? 1 : 0;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            const int zi = in_z[static_cast<std::size_t>(i)], zj = in_z[static_cast<std::size_t>(j)];
            if (!zi && !zj)
                continue;
            const Word* ri = rows.Row(i);
            const Word* rj = rows.Row(j);
            for (int w = 0; w < words; ++w)
                r[static_cast<std::size_t>(w)] = (zi ? rj[w] : 0) ^ (zj ? ri[w] : 0);
            el.Insert(r.data());
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            for (int l = j + 1; l < k; ++l) {
                const int zi = in_z[static_cast<std::size_t>(i)], zj = in_z[static_cast<std::size_t>(j)],
                          zl = in_z[static_cast<std::size_t>(l)];
                if (!zi && !zj && !zl)
                    continue;
                const Word* ri = rows.Row(i);
                const Word* rj = rows.Row(j);
                const Word* rl = rows.Row(l);
                for (int w = 0; w < words; ++w)
                    r[static_cast<std::size_t>(w)] =
                        (zi ? (rj[w] & rl[w]) : 0) ^ (zj ? (ri[w] & rl[w]) : 0) ^ (zl ? (ri[w] & rj[w]) : 0);
                el.Insert(r.data());
            }
        }
    }
}

// {y : A + z y^T 保持三次部分} 的基。base 是只含 A 的各行的消元器（对所有 z 相同）。
template <class Col>
std::vector<std::vector<Word>> MoveBasis(const Eliminator& base, const BitRows& rows, int k, Col z) {
    Eliminator el = base;
    InsertZConstraints(el, rows, k, z);
    return el.Nullspace();
}

Eliminator BaseEliminator(const BitRows& rows, int k, int m) {
    Eliminator el;
    el.Init(m);
    std::vector<Word> r(static_cast<std::size_t>(rows.words));
    for (int a = 0; a < k; ++a) {
        std::copy(rows.Row(a), rows.Row(a) + rows.words, r.begin());
        el.Insert(r.data());
    }
    return el;
}

// A + z y^T 之后的列（Python 的 _move）。
template <class Col>
std::vector<Col> ApplyMove(const std::vector<Col>& cols, Col z, const std::vector<Word>& y) {
    const int m = static_cast<int>(cols.size());
    std::vector<Col> next;
    next.reserve(cols.size() + 1);
    int weight = 0;
    for (int j = 0; j < m; ++j) {
        const bool flip = TestBit(y.data(), j);
        weight += flip ? 1 : 0;
        next.push_back(flip ? (cols[static_cast<std::size_t>(j)] ^ z) : cols[static_cast<std::size_t>(j)]);
    }
    if (weight & 1)
        next.push_back(z);
    return Normalise(next);
}

// 一个 z 的最优移动：列数与对应的 y（没有减少列数的移动时 size = m）。
struct MoveChoice {
    int size = 0;
    std::vector<Word> y;
};

template <class Col>
MoveChoice BestMoveForZ(const std::vector<Col>& cols, const std::unordered_map<Col, int>& index,
                        const Eliminator& base, const BitRows& rows, int k, Col z) {
    const int m = static_cast<int>(cols.size());
    const int words = rows.words;
    MoveChoice best;
    best.size = m;
    const std::vector<std::vector<Word>> basis = MoveBasis(base, rows, k, z);
    if (basis.empty())
        return best;
    // partner[j]：cols[j] ^ z 所在的列（没有则 -1）；jz：z 本身所在的列。
    std::vector<int> partner(static_cast<std::size_t>(m), -1);
    for (int j = 0; j < m; ++j) {
        auto it = index.find(cols[static_cast<std::size_t>(j)] ^ z);
        if (it != index.end())
            partner[static_cast<std::size_t>(j)] = it->second;
    }
    int jz = -1;
    {
        auto it = index.find(z);
        if (it != index.end())
            jz = it->second;
    }
    auto size_after = [&](const Word* y) {
        int size = m;
        int weight = 0;
        for (int w = 0; w < words; ++w) {
            Word bits = y[w];
            while (bits != 0) {
                const int j = w * 64 + common::Ctz64(bits);
                bits &= bits - 1;
                ++weight;
                const int pj = partner[static_cast<std::size_t>(j)];
                if (pj >= 0 && !TestBit(y, pj))
                    size -= 2;  // 翻转后的列与一个未翻转的列相同：成对消去
            }
        }
        const bool z_flipped = jz >= 0 && TestBit(y, jz);
        if (z_flipped)
            size -= 1;  // 列 z 变成零列
        if (weight & 1)
            size += (jz >= 0 && !z_flipped) ? -1 : 1;  // 新增的列 z（与未翻转的列 z 消去）
        return size;
    };
    auto consider = [&](const std::vector<Word>& y) {
        const int size = size_after(y.data());
        if (size < best.size) {
            best.size = size;
            best.y = y;
        }
    };
    for (const auto& y : basis)
        consider(y);
    const std::size_t head = std::min<std::size_t>(basis.size(), 12);
    std::vector<Word> y(static_cast<std::size_t>(words));
    for (std::size_t a = 0; a < head; ++a) {
        for (std::size_t b = a + 1; b < head; ++b) {
            for (int w = 0; w < words; ++w)
                y[static_cast<std::size_t>(w)] = basis[a][static_cast<std::size_t>(w)] ^ basis[b][static_cast<std::size_t>(w)];
            consider(y);
        }
    }
    return best;
}

// 贪心 TODD 到局部极小（Python 的 _todd_once）。
template <class Col>
std::vector<Col> ToddOnce(std::vector<Col> cols, int k, int threads) {
    while (true) {
        const int m = static_cast<int>(cols.size());
        if (m < 4)
            return cols;
        std::unordered_map<Col, int> index;
        index.reserve(cols.size() * 2);
        for (int j = 0; j < m; ++j)
            index[cols[static_cast<std::size_t>(j)]] = j;
        const BitRows rows = MatrixRows(cols, k);
        const Eliminator base = BaseEliminator(rows, k, m);
        std::vector<Col> zs;
        {
            std::unordered_set<Col> seen;
            seen.reserve(static_cast<std::size_t>(m) * static_cast<std::size_t>(m) / 2 + 1);
            for (int a = 0; a < m; ++a)
                for (int b = a + 1; b < m; ++b) {
                    const Col z = cols[static_cast<std::size_t>(a)] ^ cols[static_cast<std::size_t>(b)];
                    if (seen.insert(z).second)
                        zs.push_back(z);
                }
        }
        std::vector<MoveChoice> choice(zs.size());
        const int workers = std::max(1, std::min<int>(threads, static_cast<int>(zs.size() / 16)));
        if (workers <= 1) {
            for (std::size_t i = 0; i < zs.size(); ++i)
                choice[i] = BestMoveForZ(cols, index, base, rows, k, zs[i]);
        } else {
            std::vector<std::function<void()>> tasks;
            for (int t = 0; t < workers; ++t)
                tasks.emplace_back([&, t]() {
                    for (std::size_t i = static_cast<std::size_t>(t); i < zs.size(); i += static_cast<std::size_t>(workers))
                        choice[i] = BestMoveForZ(cols, index, base, rows, k, zs[i]);
                });
            common::RunTasks(tasks, workers);
        }
        int best = -1;
        for (std::size_t i = 0; i < zs.size(); ++i)
            if (choice[i].size < m && (best < 0 || choice[i].size < choice[static_cast<std::size_t>(best)].size))
                best = static_cast<int>(i);
        if (best < 0)
            return cols;
        cols = ApplyMove(cols, zs[static_cast<std::size_t>(best)], choice[static_cast<std::size_t>(best)].y);
    }
}

std::pair<std::uint64_t, std::uint64_t> TwoDistinct(CPythonRandom& rng, std::uint64_t m) {
    const std::uint64_t a = rng.RandBelow(m);
    std::uint64_t b = rng.RandBelow(m - 1);
    if (b >= a)
        ++b;
    return {a, b};
}

// TODD 加平台移动（Python 的 _todd_python）。
template <class Col>
std::vector<Col> Todd(const std::vector<Col>& input, int k, int plateau, std::uint64_t seed, int threads) {
    std::vector<Col> best = ToddOnce(Normalise(input), k, threads);
    if (plateau <= 0 || best.size() < 4)
        return best;
    CPythonRandom rng(seed);
    std::vector<Col> cur = best;
    for (int step = 0; step < plateau; ++step) {
        const int m = static_cast<int>(cur.size());
        if (m < 4)
            break;
        const auto [a, b] = TwoDistinct(rng, static_cast<std::uint64_t>(m));
        const Col z = rng.RandBelow(10) < 7 ? Col(cur[a] ^ cur[b]) : cur[a];
        const BitRows rows = MatrixRows(cur, k);
        const Eliminator base = BaseEliminator(rows, k, m);
        const std::vector<std::vector<Word>> basis = MoveBasis(base, rows, k, z);
        if (basis.empty())
            continue;
        std::vector<Word> y(static_cast<std::size_t>(rows.words), 0);
        for (const auto& v : basis)
            if (rng.GetRandBits(1))
                for (int w = 0; w < rows.words; ++w)
                    y[static_cast<std::size_t>(w)] ^= v[static_cast<std::size_t>(w)];
        if (std::all_of(y.begin(), y.end(), [](Word w) { return w == 0; }))
            y = basis[rng.RandBelow(basis.size())];
        std::vector<Col> next = ApplyMove(cur, z, y);
        if (next.empty() || static_cast<int>(next.size()) > m)
            continue;
        cur = ToddOnce(std::move(next), k, threads);
        if (cur.size() < best.size())
            best = cur;
        else if (cur.size() > best.size() + 1)
            cur = best;
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// Reed–Muller 平面搜索
// ---------------------------------------------------------------------------------------------
template <class Col>
bool Independent(const std::array<Col, 4>& dirs) {
    Col piv[ColumnBits<Col>::value];
    bool has[ColumnBits<Col>::value] = {false};
    for (Col d : dirs) {
        while (!IsZero(d)) {
            const int h = TopBit(d);
            if (!has[h]) {
                has[h] = true;
                piv[h] = d;
                break;
            }
            d ^= piv[h];
        }
        if (IsZero(d))
            return false;
    }
    return true;
}

// 过五个不同的随机列的 4 维平面的点（去掉 0）；五列相关时返回 false。
template <class Col>
bool PickFlat(const std::vector<Col>& cs, CPythonRandom& rng, std::vector<Col>& pts) {
    std::array<std::uint64_t, 5> picked{};
    int count = 0;
    while (count < 5) {
        const std::uint64_t i = rng.RandBelow(cs.size());
        bool dup = false;
        for (int t = 0; t < count; ++t)
            dup = dup || picked[static_cast<std::size_t>(t)] == i;
        if (!dup)
            picked[static_cast<std::size_t>(count++)] = i;
    }
    const Col base = cs[picked[0]];
    std::array<Col, 4> dirs{};
    for (int t = 0; t < 4; ++t)
        dirs[static_cast<std::size_t>(t)] = cs[picked[static_cast<std::size_t>(t + 1)]] ^ base;
    if (!Independent(dirs))
        return false;
    pts.clear();
    for (unsigned mask = 0; mask < 16; ++mask) {
        Col p = base;
        for (int t = 0; t < 4; ++t)
            if ((mask >> t) & 1u)
                p ^= dirs[static_cast<std::size_t>(t)];
        if (!IsZero(p))
            pts.push_back(p);
    }
    return true;
}

// 升序的 cs 与点集的对称差，升序。
template <class Col>
std::vector<Col> SymmetricDifference(const std::vector<Col>& cs, const std::vector<Col>& pts) {
    std::vector<Col> a = cs;
    std::vector<Col> b = pts;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<Col> out;
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

template <class Col>
bool BestFlatMove(const std::vector<Col>& cs, CPythonRandom& rng, int samples, std::vector<Col>& out) {
    if (cs.size() < 5)
        return false;
    std::unordered_set<Col> sset(cs.begin(), cs.end());
    std::vector<Col> pts, best;
    int gain_best = 0;
    for (int s = 0; s < samples; ++s) {
        if (!PickFlat(cs, rng, pts))
            continue;
        int inside = 0;
        for (Col p : pts)
            inside += sset.count(p) ? 1 : 0;
        const int gain = 2 * inside - static_cast<int>(pts.size());
        if (gain > gain_best) {
            best = pts;
            gain_best = gain;
        }
    }
    if (gain_best == 0)
        return false;
    out = SymmetricDifference(cs, best);
    return true;
}

template <class Col>
std::vector<Col> RmSearch(const TCountRmProblemT<Col>& prob, int samples) {
    CPythonRandom rng(prob.seed);
    std::vector<Col> cur = Todd(prob.cols, prob.k, 0, 0, 1);
    std::vector<Col> best = cur;
    std::vector<Col> pts;
    std::vector<Col> next;
    for (int it = 0; it < prob.iterations; ++it) {
        if (BestFlatMove(cur, rng, samples, next)) {
            cur = Todd(next, prob.k, 0, 0, 1);
        } else {
            if (cur.size() < 5)
                break;
            bool found = false;
            for (int t = 0; t < 50 && !found; ++t)
                found = PickFlat(cur, rng, pts);
            if (!found)
                continue;
            std::vector<Col> cand = Todd(SymmetricDifference(cur, pts), prob.k, 0, 0, 1);
            const std::int64_t d = static_cast<std::int64_t>(cand.size()) - static_cast<std::int64_t>(cur.size());
            if (d <= 0 || (d <= 60 && rng.GetRandBits(static_cast<int>(d)) == 0))
                cur = std::move(cand);
        }
        if (cur.size() < best.size())
            best = cur;
    }
    return best;
}

// ---------------------------------------------------------------------------------------------
// 翻转图随机游走
// ---------------------------------------------------------------------------------------------
using Terms = std::vector<TCountTerm>;

// 去掉含零因子的项，合并有两个公共因子的项，直到不动（Python 的 _reduce）。
void Reduce(Terms& dec) {
    bool changed = true;
    while (changed) {
        changed = false;
        dec.erase(std::remove_if(dec.begin(), dec.end(),
                                 [](const TCountTerm& t) { return t[0] == 0 || t[1] == 0 || t[2] == 0; }),
                  dec.end());
        std::map<std::tuple<int, std::uint64_t, std::uint64_t>, std::size_t> seen;
        for (std::size_t idx = 0; idx < dec.size() && !changed; ++idx) {
            const TCountTerm t = dec[idx];
            for (int p = 0; p < 3; ++p) {
                const auto key = std::make_tuple(p, t[static_cast<std::size_t>((p + 1) % 3)],
                                                 t[static_cast<std::size_t>((p + 2) % 3)]);
                auto it = seen.find(key);
                if (it != seen.end()) {
                    dec[it->second][static_cast<std::size_t>(p)] ^= t[static_cast<std::size_t>(p)];
                    dec.erase(dec.begin() + static_cast<std::ptrdiff_t>(idx));
                    changed = true;
                    break;
                }
                seen.emplace(key, idx);
            }
        }
    }
}

Terms FlipWalk(const TCountFlipProblem& prob, int plus_after) {
    CPythonRandom rng(prob.seed);
    Terms dec = prob.start;
    Reduce(dec);
    Terms best = dec;
    std::int64_t since = 0;
    std::vector<std::vector<int>> groups;
    std::unordered_map<std::uint64_t, std::size_t> group_of;
    std::vector<std::size_t> cands;
    for (std::int64_t step = 0; step < prob.steps; ++step) {
        const int p = static_cast<int>(rng.RandBelow(3));
        groups.clear();
        group_of.clear();
        for (std::size_t i = 0; i < dec.size(); ++i) {
            const std::uint64_t key = dec[i][static_cast<std::size_t>(p)];
            auto it = group_of.find(key);
            if (it == group_of.end()) {
                group_of.emplace(key, groups.size());
                groups.emplace_back();
                groups.back().push_back(static_cast<int>(i));
            } else {
                groups[it->second].push_back(static_cast<int>(i));
            }
        }
        cands.clear();
        for (std::size_t g = 0; g < groups.size(); ++g)
            if (groups[g].size() >= 2)
                cands.push_back(g);
        if (cands.empty() || since > plus_after) {
            if (dec.size() >= 2) {
                const auto [i, j] = TwoDistinct(rng, dec.size());
                const TCountTerm s = dec[i], t = dec[j];
                Terms next;
                next.reserve(dec.size() + 1);
                for (std::size_t idx = 0; idx < dec.size(); ++idx)
                    if (idx != i && idx != j)
                        next.push_back(dec[idx]);
                next.push_back({s[0] ^ t[0], s[1], s[2]});
                next.push_back({t[0], s[1] ^ t[1], s[2]});
                next.push_back({t[0], t[1], s[2] ^ t[2]});
                dec = std::move(next);
                Reduce(dec);
            }
            since = 0;
            continue;
        }
        const std::vector<int>& g = groups[cands[rng.RandBelow(cands.size())]];
        const auto [gi, gj] = TwoDistinct(rng, g.size());
        const std::size_t i = static_cast<std::size_t>(g[gi]), j = static_cast<std::size_t>(g[gj]);
        std::size_t q = static_cast<std::size_t>((p + 1) % 3), r = static_cast<std::size_t>((p + 2) % 3);
        if (rng.GetRandBits(1))
            std::swap(q, r);
        dec[i][q] ^= dec[j][q];
        dec[j][r] ^= dec[i][r];
        const std::size_t n0 = dec.size();
        Reduce(dec);
        since = dec.size() < n0 ? 0 : since + 1;
        if (dec.size() < best.size())
            best = dec;
    }
    return best;
}

template <class Col>
void CheckK(int k) {
    if (k < 0 || k > ColumnBits<Col>::value)
        throw std::invalid_argument("tcount: too many variables for this column type");
}

template <class Col>
std::vector<std::vector<Col>> ToddBatch(const std::vector<TCountToddProblemT<Col>>& problems, int threads) {
    for (const auto& p : problems)
        CheckK<Col>(p.k);
    std::vector<std::vector<Col>> out(problems.size());
    if (problems.size() == 1) {
        out[0] = Todd(problems[0].cols, problems[0].k, problems[0].plateau, problems[0].seed, threads);
        return out;
    }
    std::vector<std::function<void()>> tasks;
    tasks.reserve(problems.size());
    for (std::size_t i = 0; i < problems.size(); ++i)
        tasks.emplace_back([&, i]() {
            out[i] = Todd(problems[i].cols, problems[i].k, problems[i].plateau, problems[i].seed, 1);
        });
    common::RunTasks(tasks, threads);
    return out;
}

template <class Col>
std::vector<std::vector<Col>> RmSearchBatch(const std::vector<TCountRmProblemT<Col>>& problems, int samples,
                                            int threads) {
    for (const auto& p : problems)
        CheckK<Col>(p.k);
    std::vector<std::vector<Col>> out(problems.size());
    std::vector<std::function<void()>> tasks;
    tasks.reserve(problems.size());
    for (std::size_t i = 0; i < problems.size(); ++i)
        tasks.emplace_back([&, i]() { out[i] = RmSearch(problems[i], samples); });
    common::RunTasks(tasks, threads);
    return out;
}

}  // namespace

std::vector<TCountColumns> TCountToddBatch(const std::vector<TCountToddProblem>& problems, int threads) {
    return ToddBatch(problems, threads);
}

std::vector<TCountWideColumns> TCountToddBatchWide(const std::vector<TCountToddProblemWide>& problems, int threads) {
    return ToddBatch(problems, threads);
}

std::vector<TCountColumns> TCountRmSearch(const std::vector<TCountRmProblem>& problems, int samples, int threads) {
    return RmSearchBatch(problems, samples, threads);
}

std::vector<TCountWideColumns> TCountRmSearchWide(const std::vector<TCountRmProblemWide>& problems, int samples,
                                                  int threads) {
    return RmSearchBatch(problems, samples, threads);
}

std::vector<std::vector<TCountTerm>> TCountFlipWalks(const std::vector<TCountFlipProblem>& problems,
                                                     int plus_after, int threads) {
    std::vector<Terms> out(problems.size());
    std::vector<std::function<void()>> tasks;
    tasks.reserve(problems.size());
    for (std::size_t i = 0; i < problems.size(); ++i)
        tasks.emplace_back([&, i]() { out[i] = FlipWalk(problems[i], plus_after); });
    common::RunTasks(tasks, threads);
    return out;
}

}  // namespace quspect::native::transpiler
