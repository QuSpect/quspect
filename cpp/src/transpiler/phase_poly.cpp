#include "quspect_native/transpiler/phase_poly.hpp"

#include <cmath>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "quspect_native/common/parallel.hpp"
#include "quspect_native/detmath/scalar.hpp"
#include "quspect_native/transpiler/cnot_phase_synth.hpp"

namespace quspect::native::transpiler {

namespace {

namespace dm = quspect::native::detmath;

using M8 = std::array<double, 8>;

constexpr double kSqrtHalf = 0.7071067811865476;  // 1/√2，正确舍入
constexpr double kMonoAtol = 1e-10;
constexpr double kZeroAngle = 1e-12;
constexpr double kVerifyAtol = 1e-9;
constexpr double kPi = 3.141592653589793;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int kExactWires = 6;    // ≤：A*
constexpr int kBeamMinWires = 5;  // ≥：束搜索
constexpr int kMaxWires = 16;
constexpr int kDenseWires = 6;   // ≤：比较整个酉矩阵
constexpr int kStateWires = 14;  // ≤：再比较两个伪随机态

const M8 kH = {kSqrtHalf, 0.0, kSqrtHalf, 0.0, kSqrtHalf, 0.0, -kSqrtHalf, 0.0};
const M8 kI2 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
const M8 kX = {0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0};

// 可能不透明的 2×2 矩阵
struct Mat {
    M8 m = kI2;
    bool opaque = false;
};

Mat Mul(const Mat& x, const Mat& y) {
    Mat out;
    if (x.opaque || y.opaque) {
        out.opaque = true;
        return out;
    }
    const double ar = x.m[0], ai = x.m[1], br = x.m[2], bi = x.m[3], cr = x.m[4], ci = x.m[5], dr = x.m[6], di = x.m[7];
    const double er = y.m[0], ei = y.m[1], fr = y.m[2], fi = y.m[3], gr = y.m[4], gi = y.m[5], hr = y.m[6], hi = y.m[7];
    out.m = {(ar * er - ai * ei) + (br * gr - bi * gi), (ar * ei + ai * er) + (br * gi + bi * gr),
             (ar * fr - ai * fi) + (br * hr - bi * hi), (ar * fi + ai * fr) + (br * hi + bi * hr),
             (cr * er - ci * ei) + (dr * gr - di * gi), (cr * ei + ci * er) + (dr * gi + di * gr),
             (cr * fr - ci * fi) + (dr * hr - di * hi), (cr * fi + ci * fr) + (dr * hi + di * hr)};
    return out;
}

Mat Fixed(const M8& m) {
    Mat out;
    out.m = m;
    return out;
}

bool Small(double re, double im) {
    const double atol2 = kMonoAtol * kMonoAtol;
    return re * re + im * im <= atol2;
}

// 0：对角，1：反对角，-1：都不是（或不透明）
int Monomial(const Mat& x) {
    if (x.opaque)
        return -1;
    const M8& m = x.m;
    if (Small(m[2], m[3]) && Small(m[4], m[5]))
        return 0;
    if (Small(m[0], m[1]) && Small(m[6], m[7]))
        return 1;
    return -1;
}

double RelativePhase(double u0, double u1, double v0, double v1) {
    const double re = v0 * u0 + v1 * u1;
    const double im = v1 * u0 - v0 * u1;
    return dm::Atan2(im, re);
}

Mat RzMat(double theta) {
    const double c = dm::Cos(0.5 * theta);
    const double s = dm::Sin(0.5 * theta);
    Mat out;
    out.m = {c, -s, 0.0, 0.0, 0.0, 0.0, c, s};
    return out;
}

// 定向后的项
enum ItemKind : int { kU = 0, kCX = 1, kCZ = 2, kRz = 3, kXg = 4 };

struct Item {
    int kind = kU;
    int q0 = 0;
    int q1 = -1;
    Mat mat;                         // kU
    std::vector<PhasePolyOp> gates;  // kU：段内的门
    double angle = 0.0;              // kRz
};

struct Ctx {
    const std::vector<std::array<double, 8>>& matrices;
    const std::vector<std::uint8_t>& opaque;
};

Mat OpMat(const Ctx& ctx, const PhasePolyOp& op) {
    switch (op.kind) {
        case kPPOrig: {
            Mat out;
            const auto idx = static_cast<std::size_t>(op.index);
            if (ctx.opaque[idx] != 0) {
                out.opaque = true;
            } else {
                out.m = ctx.matrices[idx];
            }
            return out;
        }
        case kPPH:
            return Fixed(kH);
        case kPPX:
            return Fixed(kX);
        case kPPRz:
            return RzMat(op.angle);
        default:
            throw std::logic_error("OpMat: not a single-qubit op");
    }
}

PhasePolyOp HOp(int q) {
    PhasePolyOp op;
    op.kind = kPPH;
    op.a = q;
    return op;
}

// ---------------------------------------------------------------------------
// 区域划分

struct Piece {
    bool is_region = false;
    int item = -1;           // 单个项
    std::vector<int> wires;  // 区域的线（按加入顺序）
    std::vector<int> items;  // 区域的项（按线路顺序）
};

void ItemQubits(const Item& it, int& n, int qs[2]) {
    qs[0] = it.q0;
    if (it.kind == kCX || it.kind == kCZ) {
        qs[1] = it.q1;
        n = 2;
    } else {
        n = 1;
    }
}

// 从 items（下标序列）的开头生长一个凸区域
void TakeRegion(const std::vector<Item>& all, const std::vector<int>& items, int cap, std::vector<int>& pre,
                std::vector<int>& wires, std::vector<int>& reg, std::vector<int>& post) {
    std::set<int> inside;
    std::set<int> closed;
    int n_open = 0;
    for (std::size_t pos = 0; pos < items.size(); ++pos) {
        const Item& it = all[static_cast<std::size_t>(items[pos])];
        int nq = 0;
        int qs[2];
        ItemQubits(it, nq, qs);
        bool close = false;
        bool any_closed = false;
        for (int k = 0; k < nq; ++k)
            any_closed = any_closed || closed.count(qs[k]) != 0;
        if (any_closed) {
            post.push_back(items[pos]);
            close = true;
        } else if (it.kind == kU) {
            if (inside.count(it.q0) == 0) {
                pre.push_back(items[pos]);
            } else if (Monomial(it.mat) >= 0) {
                reg.push_back(items[pos]);
            } else {
                post.push_back(items[pos]);
                close = true;
            }
        } else {
            int n_new = 0;
            for (int k = 0; k < nq; ++k)
                n_new += inside.count(qs[k]) == 0 ? 1 : 0;
            if (static_cast<int>(inside.size()) + n_new <= cap) {
                for (int k = 0; k < nq; ++k) {
                    if (inside.insert(qs[k]).second) {
                        wires.push_back(qs[k]);
                        ++n_open;
                    }
                }
                reg.push_back(items[pos]);
            } else {
                post.push_back(items[pos]);
                close = true;
            }
        }
        if (close) {
            for (int k = 0; k < nq; ++k) {
                if (closed.insert(qs[k]).second && inside.count(qs[k]) != 0)
                    --n_open;
            }
        }
        if (!reg.empty() && n_open == 0) {
            post.insert(post.end(), items.begin() + static_cast<std::ptrdiff_t>(pos) + 1, items.end());
            return;
        }
    }
}

std::vector<Piece> SplitRegions(const std::vector<Item>& all, std::vector<int> rest, int cap) {
    std::vector<Piece> out;
    while (!rest.empty()) {
        std::vector<int> pre, wires, reg, post;
        TakeRegion(all, rest, cap, pre, wires, reg, post);
        for (int i : pre) {
            Piece p;
            p.item = i;
            out.push_back(std::move(p));
        }
        if (reg.empty()) {
            for (int i : post) {
                Piece p;
                p.item = i;
                out.push_back(std::move(p));
            }
            break;
        }
        Piece p;
        p.is_region = true;
        p.wires = std::move(wires);
        p.items = std::move(reg);
        out.push_back(std::move(p));
        rest = std::move(post);
    }
    return out;
}

std::vector<Piece> SplitRegionsDir(const std::vector<Item>& all, int cap, bool reverse) {
    std::vector<int> order(all.size());
    for (std::size_t i = 0; i < all.size(); ++i)
        order[i] = static_cast<int>(i);
    if (!reverse)
        return SplitRegions(all, std::move(order), cap);
    std::reverse(order.begin(), order.end());
    std::vector<Piece> back = SplitRegions(all, std::move(order), cap);
    std::reverse(back.begin(), back.end());
    for (Piece& p : back)
        std::reverse(p.items.begin(), p.items.end());
    return back;
}

// ---------------------------------------------------------------------------
// 相位多项式

struct PhasePoly {
    std::vector<std::uint32_t> rows;
    std::vector<int> consts;
    std::map<std::uint32_t, double> terms;  // 约化后非零的项
    int n2 = 0;
};

// 区域（或新发出的项）的 (终态行, 常数, {奇偶: 角}, 两比特门数)；含非单项式段时返回空
std::optional<PhasePoly> ComputePhasePoly(const std::vector<int>& wires, const std::vector<const Item*>& reg) {
    std::unordered_map<int, int> local;
    for (std::size_t i = 0; i < wires.size(); ++i)
        local[wires[i]] = static_cast<int>(i);
    PhasePoly pp;
    const std::size_t k = wires.size();
    pp.rows.resize(k);
    pp.consts.assign(k, 0);
    for (std::size_t i = 0; i < k; ++i)
        pp.rows[i] = 1U << i;
    // 插入顺序无关：同一键按项的顺序累加
    std::map<std::uint32_t, double> terms;
    auto add = [&](std::uint32_t mask, int c, double angle) {
        auto it = terms.find(mask);
        const double base = it == terms.end() ? 0.0 : it->second;
        terms[mask] = base + (c != 0 ? -angle : angle);
    };
    for (const Item* itp : reg) {
        const Item& it = *itp;
        if (it.kind == kRz) {
            const auto i = static_cast<std::size_t>(local.at(it.q0));
            add(pp.rows[i], pp.consts[i], it.angle);
        } else if (it.kind == kXg) {
            pp.consts[static_cast<std::size_t>(local.at(it.q0))] ^= 1;
        } else if (it.kind == kU) {
            const auto i = static_cast<std::size_t>(local.at(it.q0));
            const int kind = Monomial(it.mat);
            const M8& m = it.mat.m;
            if (kind == 0) {
                add(pp.rows[i], pp.consts[i], RelativePhase(m[0], m[1], m[6], m[7]));
            } else if (kind == 1) {
                add(pp.rows[i], pp.consts[i], RelativePhase(m[4], m[5], m[2], m[3]));
                pp.consts[i] ^= 1;
            } else {
                return std::nullopt;
            }
        } else if (it.kind == kCX) {
            const auto c = static_cast<std::size_t>(local.at(it.q0));
            const auto t = static_cast<std::size_t>(local.at(it.q1));
            pp.rows[t] ^= pp.rows[c];
            pp.consts[t] ^= pp.consts[c];
            ++pp.n2;
        } else {  // 对角 cz：π·va·vb = π/2·(va + vb − va⊕vb)
            const auto a = static_cast<std::size_t>(local.at(it.q0));
            const auto b = static_cast<std::size_t>(local.at(it.q1));
            add(pp.rows[a], pp.consts[a], 0.5 * kPi);
            add(pp.rows[b], pp.consts[b], 0.5 * kPi);
            add(pp.rows[a] ^ pp.rows[b], pp.consts[a] ^ pp.consts[b], -0.5 * kPi);
            ++pp.n2;
        }
    }
    for (const auto& [mask, angle] : terms) {
        const double reduced = dm::Remainder(angle, kTwoPi);
        if (std::fabs(reduced) > kZeroAngle)
            pp.terms[mask] = reduced;
    }
    return pp;
}

// ---------------------------------------------------------------------------
// 求解（按问题缓存：同一编译内各路由候选、各轮的区域问题大量重复）

struct Problem {
    int n;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::uint32_t> rows;
    std::vector<std::uint32_t> terms;
    int max_cx;
    std::uint32_t free_mask;
    int beam;
    std::int64_t budget;
};

using ProblemKey = std::vector<std::int64_t>;

ProblemKey KeyOf(const Problem& p) {
    ProblemKey k;
    k.reserve(8 + 2 * p.edges.size() + p.rows.size() + p.terms.size());
    k.push_back(p.n);
    k.push_back(p.max_cx);
    k.push_back(p.free_mask);
    k.push_back(p.beam);
    k.push_back(p.budget);
    k.push_back(static_cast<std::int64_t>(p.edges.size()));
    for (const auto& e : p.edges)
        k.push_back(e.first * 64 + e.second);
    k.push_back(static_cast<std::int64_t>(p.rows.size()));
    for (std::uint32_t r : p.rows)
        k.push_back(r);
    for (std::uint32_t t : p.terms)
        k.push_back(t);
    return k;
}

struct KeyHash {
    std::size_t operator()(const ProblemKey& k) const {
        std::uint64_t h = 0x84222325CBF29CE4ULL;
        for (std::int64_t v : k) {
            h ^= static_cast<std::uint64_t>(v) + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::size_t>(h);
    }
};

std::mutex g_cache_mutex;
std::unordered_map<ProblemKey, std::optional<CnotPhaseResult>, KeyHash> g_cache;
constexpr std::size_t kCacheMax = 200000;

std::vector<std::optional<CnotPhaseResult>> Solve(const std::vector<Problem>& problems) {
    std::vector<std::optional<CnotPhaseResult>> out(problems.size());
    std::vector<ProblemKey> keys(problems.size());
    std::vector<std::size_t> todo;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (std::size_t i = 0; i < problems.size(); ++i) {
            keys[i] = KeyOf(problems[i]);
            const auto it = g_cache.find(keys[i]);
            if (it != g_cache.end()) {
                out[i] = it->second;
            } else {
                todo.push_back(i);
            }
        }
    }
    if (todo.empty())
        return out;
    std::vector<CnotPhaseResult> found(todo.size());
    std::vector<std::function<void()>> tasks;
    tasks.reserve(todo.size());
    for (std::size_t j = 0; j < todo.size(); ++j) {
        tasks.emplace_back([&, j] {
            const Problem& p = problems[todo[j]];
            if (p.beam > 0) {
                found[j] = SynthesizeCnotPhaseBeam(p.n, p.edges, p.rows, p.terms, p.max_cx, p.beam, p.free_mask);
            } else {
                found[j] = SynthesizeCnotPhase(p.n, p.edges, p.rows, p.terms, p.max_cx, p.budget, p.free_mask);
            }
        });
    }
    common::RunTasks(tasks, common::CompileThreads());
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    if (g_cache.size() + todo.size() > kCacheMax)
        g_cache.clear();
    for (std::size_t j = 0; j < todo.size(); ++j) {
        std::optional<CnotPhaseResult> r;
        if (found[j].found)
            r = std::move(found[j]);
        out[todo[j]] = r;
        g_cache[keys[todo[j]]] = out[todo[j]];
    }
    return out;
}

// ---------------------------------------------------------------------------
// 发出与验证

std::vector<Item> Emit(const std::vector<int>& wires, const std::vector<std::pair<int, int>>& moves,
                       const PhasePoly& pp) {
    const std::size_t k = wires.size();
    std::vector<std::uint32_t> rows(k);
    for (std::size_t i = 0; i < k; ++i)
        rows[i] = 1U << i;
    std::vector<Item> out;
    std::set<std::uint32_t> placed;
    auto place = [&](std::size_t i) {
        const std::uint32_t m = rows[i];
        const auto it = pp.terms.find(m);
        if (it != pp.terms.end() && placed.insert(m).second) {
            Item r;
            r.kind = kRz;
            r.q0 = wires[i];
            r.angle = it->second;
            out.push_back(std::move(r));
        }
    };
    for (std::size_t i = 0; i < k; ++i)
        place(i);
    for (const auto& [c, t] : moves) {
        rows[static_cast<std::size_t>(t)] ^= rows[static_cast<std::size_t>(c)];
        Item cx;
        cx.kind = kCX;
        cx.q0 = wires[static_cast<std::size_t>(c)];
        cx.q1 = wires[static_cast<std::size_t>(t)];
        out.push_back(std::move(cx));
        place(static_cast<std::size_t>(t));
    }
    for (std::size_t i = 0; i < k; ++i) {
        if (pp.consts[i] != 0) {
            Item x;
            x.kind = kXg;
            x.q0 = wires[i];
            out.push_back(std::move(x));
        }
    }
    return out;
}

// 列主序的 dim × cols 复矩阵（实部、虚部分开），左乘一个项
struct State {
    std::size_t dim = 0;
    std::size_t cols = 0;
    std::vector<double> re, im;  // re[col * dim + row]
};

void ApplyItem(State& s, const std::unordered_map<int, int>& local, const Item& it) {
    const std::size_t dim = s.dim;
    if (it.kind == kU || it.kind == kRz || it.kind == kXg) {
        M8 m;
        if (it.kind == kU) {
            m = it.mat.m;
        } else if (it.kind == kRz) {
            m = RzMat(it.angle).m;
        } else {
            m = kX;
        }
        const std::size_t bit = std::size_t{1} << local.at(it.q0);
        for (std::size_t c = 0; c < s.cols; ++c) {
            double* re = s.re.data() + c * dim;
            double* im = s.im.data() + c * dim;
            for (std::size_t r = 0; r < dim; ++r) {
                if (r & bit)
                    continue;
                const std::size_t r1 = r | bit;
                const double ar = re[r], ai = im[r], br = re[r1], bi = im[r1];
                // 新 |0⟩ = m00·a + m01·b，新 |1⟩ = m10·a + m11·b
                re[r] = (m[0] * ar - m[1] * ai) + (m[2] * br - m[3] * bi);
                im[r] = (m[0] * ai + m[1] * ar) + (m[2] * bi + m[3] * br);
                re[r1] = (m[4] * ar - m[5] * ai) + (m[6] * br - m[7] * bi);
                im[r1] = (m[4] * ai + m[5] * ar) + (m[6] * bi + m[7] * br);
            }
        }
    } else if (it.kind == kCX) {
        const std::size_t cb = std::size_t{1} << local.at(it.q0);
        const std::size_t tb = std::size_t{1} << local.at(it.q1);
        for (std::size_t c = 0; c < s.cols; ++c) {
            double* re = s.re.data() + c * dim;
            double* im = s.im.data() + c * dim;
            for (std::size_t r = 0; r < dim; ++r) {
                if ((r & cb) && !(r & tb)) {
                    std::swap(re[r], re[r | tb]);
                    std::swap(im[r], im[r | tb]);
                }
            }
        }
    } else {  // kCZ
        const std::size_t ab = std::size_t{1} << local.at(it.q0);
        const std::size_t bb = std::size_t{1} << local.at(it.q1);
        for (std::size_t c = 0; c < s.cols; ++c) {
            double* re = s.re.data() + c * dim;
            double* im = s.im.data() + c * dim;
            for (std::size_t r = 0; r < dim; ++r) {
                if ((r & ab) && (r & bb)) {
                    re[r] = -re[r];
                    im[r] = -im[r];
                }
            }
        }
    }
}

bool PhaseAlignedClose(const State& a, const State& b) {
    std::size_t best = 0;
    double best_mag = -1.0;
    for (std::size_t i = 0; i < a.re.size(); ++i) {
        const double mag = a.re[i] * a.re[i] + a.im[i] * a.im[i];
        if (mag > best_mag) {
            best_mag = mag;
            best = i;
        }
    }
    const double br = b.re[best], bi = b.im[best];
    if (br * br + bi * bi < 1e-12)
        return false;
    const double ar = a.re[best], ai = a.im[best];
    const double re = ar * br + ai * bi;
    const double im = ai * br - ar * bi;
    const double norm = std::sqrt(re * re + im * im);
    const double pr = re / norm, pi = im / norm;
    const double tol = kVerifyAtol * kVerifyAtol;
    for (std::size_t i = 0; i < a.re.size(); ++i) {
        const double dr = a.re[i] - (pr * b.re[i] - pi * b.im[i]);
        const double di = a.im[i] - (pr * b.im[i] + pi * b.re[i]);
        if (dr * dr + di * di > tol)
            return false;
    }
    return true;
}

std::uint64_t SplitMix(std::uint64_t& x) {
    std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

bool SamePhasePoly(const std::vector<int>& wires, const std::vector<const Item*>& old_items,
                   const std::vector<const Item*>& new_items) {
    const auto a = ComputePhasePoly(wires, old_items);
    const auto b = ComputePhasePoly(wires, new_items);
    if (!a || !b || a->rows != b->rows || a->consts != b->consts || a->terms.size() != b->terms.size())
        return false;
    for (const auto& [mask, angle] : a->terms) {
        const auto it = b->terms.find(mask);
        if (it == b->terms.end())
            return false;
        if (std::fabs(dm::Remainder(angle - it->second, kTwoPi)) > kVerifyAtol)
            return false;
    }
    return true;
}

bool Verify(const std::vector<int>& wires, const std::vector<const Item*>& old_items,
            const std::vector<const Item*>& new_items) {
    const std::size_t k = wires.size();
    if (k > static_cast<std::size_t>(kDenseWires)) {
        if (!SamePhasePoly(wires, old_items, new_items))
            return false;
        if (k > static_cast<std::size_t>(kStateWires))
            return true;
    }
    std::unordered_map<int, int> local;
    for (std::size_t i = 0; i < k; ++i)
        local[wires[i]] = static_cast<int>(i);
    State a;
    a.dim = std::size_t{1} << k;
    if (k <= static_cast<std::size_t>(kDenseWires)) {
        a.cols = a.dim;
        a.re.assign(a.dim * a.cols, 0.0);
        a.im.assign(a.dim * a.cols, 0.0);
        for (std::size_t i = 0; i < a.dim; ++i)
            a.re[i * a.dim + i] = 1.0;
    } else {
        a.cols = 2;
        a.re.resize(a.dim * 2);
        a.im.resize(a.dim * 2);
        std::uint64_t seed = 20261107;
        for (std::size_t i = 0; i < a.re.size(); ++i) {
            a.re[i] = static_cast<double>(SplitMix(seed) >> 11) * 0x1.0p-53 - 0.5;
            a.im[i] = static_cast<double>(SplitMix(seed) >> 11) * 0x1.0p-53 - 0.5;
        }
    }
    State b = a;
    for (const Item* it : old_items)
        ApplyItem(a, local, *it);
    for (const Item* it : new_items)
        ApplyItem(b, local, *it);
    return PhaseAlignedClose(a, b);
}

// ---------------------------------------------------------------------------
// 一轮

struct RoundOut {
    int n_two;
    std::vector<PhasePolyOp> ops;
    std::int64_t replaced;
    std::int64_t saved;
};

void EmitItem(const Item& it, std::vector<PhasePolyOp>& out, int& n_two) {
    switch (it.kind) {
        case kU:
            out.insert(out.end(), it.gates.begin(), it.gates.end());
            break;
        case kCZ: {
            PhasePolyOp op;
            op.kind = kPPCz;
            op.a = it.q0;
            op.b = it.q1;
            out.push_back(op);
            ++n_two;
            break;
        }
        case kCX: {
            out.push_back(HOp(it.q1));
            PhasePolyOp op;
            op.kind = kPPCz;
            op.a = it.q0;
            op.b = it.q1;
            out.push_back(op);
            out.push_back(HOp(it.q1));
            ++n_two;
            break;
        }
        case kRz: {
            PhasePolyOp op;
            op.kind = kPPRz;
            op.a = it.q0;
            op.angle = it.angle;
            out.push_back(op);
            break;
        }
        case kXg: {
            PhasePolyOp op;
            op.kind = kPPX;
            op.a = it.q0;
            out.push_back(op);
            break;
        }
        default:
            break;
    }
}

std::optional<RoundOut> Resynthesize(const std::vector<Item>& oriented, const std::set<std::pair<int, int>>& pairs,
                                     const PhasePolySetting& setting, int beam_width, PhasePolyStats& stats) {
    const std::vector<Piece> pieces = SplitRegionsDir(oriented, setting.max_wires, setting.reverse);
    std::vector<Problem> problems;
    struct Meta {
        std::size_t piece;
        int n2;
    };
    std::vector<Meta> meta;
    std::vector<std::optional<PhasePoly>> polys(pieces.size());
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
        const Piece& p = pieces[pi];
        if (!p.is_region)
            continue;
        std::vector<const Item*> reg;
        reg.reserve(p.items.size());
        for (int i : p.items)
            reg.push_back(&oriented[static_cast<std::size_t>(i)]);
        auto pp = ComputePhasePoly(p.wires, reg);
        if (!pp || pp->n2 < 2)
            continue;
        ++stats.regions;
        std::unordered_map<int, int> local;
        for (std::size_t i = 0; i < p.wires.size(); ++i)
            local[p.wires[i]] = static_cast<int>(i);
        std::vector<std::pair<int, int>> edges;
        for (const auto& [a, b] : pairs) {
            const auto ia = local.find(a);
            const auto ib = local.find(b);
            if (ia != local.end() && ib != local.end())
                edges.emplace_back(ia->second, ib->second);
        }
        const std::size_t ne = edges.size();
        for (std::size_t e = 0; e < ne; ++e)
            edges.emplace_back(edges[e].second, edges[e].first);
        Problem base;
        base.n = static_cast<int>(p.wires.size());
        base.edges = std::move(edges);
        base.rows = pp->rows;
        for (const auto& kv : pp->terms)
            base.terms.push_back(kv.first);
        base.max_cx = pp->n2 - 1;
        base.free_mask = 0;
        if (base.n <= kExactWires) {
            Problem a = base;
            a.beam = 0;
            a.budget = setting.node_budget;
            problems.push_back(std::move(a));
            meta.push_back(Meta{pi, pp->n2});
        }
        if (base.n >= kBeamMinWires) {
            Problem b = base;
            b.beam = beam_width;
            b.budget = 0;
            problems.push_back(std::move(b));
            meta.push_back(Meta{pi, pp->n2});
        }
        polys[pi] = std::move(pp);
    }
    if (problems.empty())
        return std::nullopt;
    const auto solutions = Solve(problems);
    std::map<std::size_t, std::vector<std::pair<int, int>>> best_sol;
    for (std::size_t j = 0; j < problems.size(); ++j) {
        const auto& sol = solutions[j];
        if (!sol || static_cast<int>(sol->moves.size()) >= meta[j].n2)
            continue;
        auto it = best_sol.find(meta[j].piece);
        if (it == best_sol.end() || sol->moves.size() < it->second.size())
            best_sol[meta[j].piece] = sol->moves;
    }
    std::map<std::size_t, std::vector<Item>> replaced;
    std::int64_t saved = 0;
    for (const auto& [pi, moves] : best_sol) {
        const Piece& p = pieces[pi];
        const PhasePoly& pp = *polys[pi];
        std::vector<Item> fresh = Emit(p.wires, moves, pp);
        std::vector<const Item*> old_items, new_items;
        for (int i : p.items)
            old_items.push_back(&oriented[static_cast<std::size_t>(i)]);
        for (const Item& it : fresh)
            new_items.push_back(&it);
        if (!Verify(p.wires, old_items, new_items))
            continue;
        saved += pp.n2 - static_cast<std::int64_t>(moves.size());
        replaced.emplace(pi, std::move(fresh));
    }
    if (replaced.empty())
        return std::nullopt;
    RoundOut out;
    out.n_two = 0;
    out.replaced = static_cast<std::int64_t>(replaced.size());
    out.saved = saved;
    for (std::size_t pi = 0; pi < pieces.size(); ++pi) {
        const Piece& p = pieces[pi];
        if (!p.is_region) {
            EmitItem(oriented[static_cast<std::size_t>(p.item)], out.ops, out.n_two);
            continue;
        }
        const auto it = replaced.find(pi);
        if (it != replaced.end()) {
            for (const Item& x : it->second)
                EmitItem(x, out.ops, out.n_two);
        } else {
            for (int i : p.items)
                EmitItem(oriented[static_cast<std::size_t>(i)], out.ops, out.n_two);
        }
    }
    return out;
}

struct RunInfo {
    int qubit;
    std::vector<PhasePolyOp> gates;
    Mat mat;
    int before_g = -1, before_k = -1;  // 段前的门 (g, k)
    int after_g = -1, after_k = -1;    // 段后的门
};

std::optional<RoundOut> Round(const Ctx& ctx, const std::vector<PhasePolyOp>& ops,
                              const std::vector<std::pair<int, int>>& extra_pairs,
                              const std::vector<PhasePolySetting>& settings, int beam_width, PhasePolyStats& stats) {
    std::set<std::pair<int, int>> pairs;
    for (const auto& [a, b] : extra_pairs)
        pairs.emplace(std::min(a, b), std::max(a, b));
    int n_two = 0;
    for (const PhasePolyOp& op : ops) {
        if (op.kind == kPPCz) {
            pairs.emplace(std::min(op.a, op.b), std::max(op.a, op.b));
            ++n_two;
        }
    }
    if (n_two < 2)
        return std::nullopt;

    // 1. 各线上的单比特段与两比特门的相邻关系
    std::vector<RunInfo> runs;
    std::vector<std::pair<int, int>> gate_qubits;
    std::vector<int> hints;
    std::vector<std::pair<int, int>> seq;  // (0: 段, 1: 门), 下标
    std::unordered_map<int, int> open_run;
    std::vector<std::array<int, 2>> run_prev, run_next;
    std::unordered_map<int, std::pair<int, int>> last_gate;
    auto run_of = [&](int q) {
        const auto it = open_run.find(q);
        if (it != open_run.end())
            return it->second;
        const int r = static_cast<int>(runs.size());
        RunInfo ri;
        ri.qubit = q;
        const auto lg = last_gate.find(q);
        if (lg != last_gate.end()) {
            ri.before_g = lg->second.first;
            ri.before_k = lg->second.second;
        }
        runs.push_back(std::move(ri));
        open_run[q] = r;
        seq.emplace_back(0, r);
        return r;
    };
    for (const PhasePolyOp& op : ops) {
        if (op.kind != kPPCz) {
            const int r = run_of(op.a);
            runs[static_cast<std::size_t>(r)].gates.push_back(op);
            runs[static_cast<std::size_t>(r)].mat = Mul(OpMat(ctx, op), runs[static_cast<std::size_t>(r)].mat);
            continue;
        }
        const int g = static_cast<int>(gate_qubits.size());
        gate_qubits.emplace_back(op.a, op.b);
        hints.push_back(op.hint);
        const int ra = run_of(op.a);
        const int rb = run_of(op.b);
        const std::array<int, 2> rp = {ra, rb};
        const int qs[2] = {op.a, op.b};
        for (int k = 0; k < 2; ++k) {
            runs[static_cast<std::size_t>(rp[k])].after_g = g;
            runs[static_cast<std::size_t>(rp[k])].after_k = k;
            open_run.erase(qs[k]);
            last_gate[qs[k]] = {g, k};
        }
        run_prev.push_back(rp);
        run_next.push_back({-1, -1});
        seq.emplace_back(1, g);
    }
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (runs[r].before_g >= 0)
            run_next[static_cast<std::size_t>(runs[r].before_g)][static_cast<std::size_t>(runs[r].before_k)]
                = static_cast<int>(r);
    }
    const int G = static_cast<int>(gate_qubits.size());

    // 2. 读法：o[g] ∈ {0: 目标 b, 1: 目标 a, 2: 对角}
    auto is_target = [](int o, int k) { return (o == 0 && k == 1) || (o == 1 && k == 0) ? 1 : 0; };
    // 每段的 4 种 (前门目标?, 后门目标?) 是否单项式；-1 表示线首 / 线尾段（不计）
    std::vector<std::array<int, 4>> mono(runs.size());
    const Mat H = Fixed(kH);
    for (std::size_t r = 0; r < runs.size(); ++r) {
        if (runs[r].before_g < 0 || runs[r].after_g < 0) {
            mono[r] = {-1, -1, -1, -1};
            continue;
        }
        const Mat& m = runs[r].mat;
        const Mat hm = Mul(H, m);
        const Mat mh = Mul(m, H);
        const Mat hmh = Mul(hm, H);
        mono[r] = {Monomial(m) >= 0 ? 1 : 0, Monomial(mh) >= 0 ? 1 : 0, Monomial(hm) >= 0 ? 1 : 0,
                   Monomial(hmh) >= 0 ? 1 : 0};
    }
    std::vector<int> orient(static_cast<std::size_t>(G), 0);
    auto run_cost = [&](int r) {
        const auto& tab = mono[static_cast<std::size_t>(r)];
        if (tab[0] < 0)
            return 0;
        const RunInfo& ri = runs[static_cast<std::size_t>(r)];
        const int tp = is_target(orient[static_cast<std::size_t>(ri.before_g)], ri.before_k);
        const int tn = is_target(orient[static_cast<std::size_t>(ri.after_g)], ri.after_k);
        return tab[static_cast<std::size_t>(tp + 2 * tn)] != 0 ? 0 : 1;
    };
    auto local_cost = [&](int g) {
        int c = 0;
        for (int r : run_prev[static_cast<std::size_t>(g)])
            if (r >= 0)
                c += run_cost(r);
        for (int r : run_next[static_cast<std::size_t>(g)])
            if (r >= 0)
                c += run_cost(r);
        return c;
    };
    for (int g = 0; g < G; ++g) {
        int best_o = 0;
        int best_c = -1;
        std::vector<int> cands;
        if (hints[static_cast<std::size_t>(g)] >= 0)
            cands.push_back(hints[static_cast<std::size_t>(g)]);
        cands.push_back(0);
        cands.push_back(1);
        cands.push_back(2);
        for (int o : cands) {
            orient[static_cast<std::size_t>(g)] = o;
            const int c = local_cost(g);
            if (best_c < 0 || c < best_c) {
                best_o = o;
                best_c = c;
            }
        }
        orient[static_cast<std::size_t>(g)] = best_o;
    }
    std::vector<std::vector<int>> neighbours(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g) {
        auto& near = neighbours[static_cast<std::size_t>(g)];
        auto visit = [&](int r) {
            if (r < 0)
                return;
            const RunInfo& ri = runs[static_cast<std::size_t>(r)];
            if (ri.before_g >= 0 && ri.before_g != g)
                near.push_back(ri.before_g);
            if (ri.after_g >= 0 && ri.after_g != g)
                near.push_back(ri.after_g);
        };
        for (int r : run_prev[static_cast<std::size_t>(g)])
            visit(r);
        for (int r : run_next[static_cast<std::size_t>(g)])
            visit(r);
    }
    std::vector<int> pending(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        pending[static_cast<std::size_t>(g)] = g;
    std::vector<char> queued(static_cast<std::size_t>(G), 1);
    std::int64_t steps = 0;
    const std::int64_t limit = 8LL * G;
    while (!pending.empty() && steps < limit) {
        const int g = pending.back();
        pending.pop_back();
        queued[static_cast<std::size_t>(g)] = 0;
        ++steps;
        const int cur = orient[static_cast<std::size_t>(g)];
        int best_c = local_cost(g);
        int best_o = cur;
        for (int o = 0; o < 3; ++o) {
            if (o == cur)
                continue;
            orient[static_cast<std::size_t>(g)] = o;
            const int c = local_cost(g);
            if (c < best_c) {
                best_o = o;
                best_c = c;
            }
        }
        orient[static_cast<std::size_t>(g)] = best_o;
        if (best_o != cur) {
            for (int h : neighbours[static_cast<std::size_t>(g)]) {
                if (!queued[static_cast<std::size_t>(h)]) {
                    queued[static_cast<std::size_t>(h)] = 1;
                    pending.push_back(h);
                }
            }
        }
    }

    // 3. 定向后的序列
    std::vector<Item> oriented;
    oriented.reserve(seq.size() + 8);
    std::unordered_map<int, bool> pre_h;
    for (const auto& [kind, idx] : seq) {
        if (kind == 0) {
            const RunInfo& run = runs[static_cast<std::size_t>(idx)];
            Item it;
            it.kind = kU;
            it.q0 = run.qubit;
            it.gates = run.gates;
            it.mat = run.mat;
            const auto ph = pre_h.find(run.qubit);
            if (ph != pre_h.end()) {
                const bool had = ph->second;
                pre_h.erase(ph);
                if (had) {
                    it.gates.insert(it.gates.begin(), HOp(run.qubit));
                    it.mat = Mul(it.mat, H);
                }
            }
            if (run.after_g >= 0 && is_target(orient[static_cast<std::size_t>(run.after_g)], run.after_k)) {
                it.gates.push_back(HOp(run.qubit));
                it.mat = Mul(H, it.mat);
            }
            oriented.push_back(std::move(it));
            continue;
        }
        const auto [a, b] = gate_qubits[static_cast<std::size_t>(idx)];
        const int o = orient[static_cast<std::size_t>(idx)];
        if (o == 2) {
            Item it;
            it.kind = kCZ;
            it.q0 = a;
            it.q1 = b;
            oriented.push_back(std::move(it));
        } else {
            const int c = o == 0 ? a : b;
            const int t = o == 0 ? b : a;
            Item it;
            it.kind = kCX;
            it.q0 = c;
            it.q1 = t;
            oriented.push_back(std::move(it));
            if (run_next[static_cast<std::size_t>(idx)][t == a ? 0 : 1] >= 0) {
                pre_h[t] = true;
            } else {  // 线尾：补一个 H
                Item h;
                h.kind = kU;
                h.q0 = t;
                h.gates.push_back(HOp(t));
                h.mat = H;
                oriented.push_back(std::move(h));
            }
        }
    }

    // 4. 各种划分各做一次，取两比特门最少的
    std::optional<RoundOut> best;
    for (const PhasePolySetting& s : settings) {
        auto r = Resynthesize(oriented, pairs, s, beam_width, stats);
        if (r && (!best || r->n_two < best->n_two))
            best = std::move(r);
    }
    return best;
}

}  // namespace

PhasePolyResult RunPhasePolynomial(const std::vector<PhasePolyOp>& ops,
                                   const std::vector<std::array<double, 8>>& matrices,
                                   const std::vector<std::uint8_t>& opaque,
                                   const std::vector<std::pair<int, int>>& extra_pairs,
                                   const std::vector<PhasePolySetting>& settings, int beam_width, int max_rounds) {
    if (matrices.size() != opaque.size())
        throw std::invalid_argument("RunPhasePolynomial: matrices and opaque differ in length");
    for (const PhasePolySetting& s : settings) {
        if (s.max_wires < 2 || s.max_wires > kMaxWires)
            throw std::invalid_argument("RunPhasePolynomial: max_wires must be in 2..16");
    }
    for (const PhasePolyOp& op : ops) {
        if (op.kind == kPPOrig && (op.index < 0 || static_cast<std::size_t>(op.index) >= matrices.size()))
            throw std::invalid_argument("RunPhasePolynomial: bad gate index");
    }
    const Ctx ctx{matrices, opaque};
    PhasePolyResult result;
    std::vector<PhasePolyOp> current = ops;
    for (int round = 0; round < max_rounds; ++round) {
        auto r = Round(ctx, current, extra_pairs, settings, beam_width, result.stats);
        ++result.stats.rounds;
        if (!r)
            break;
        result.stats.replaced += r->replaced;
        result.stats.saved += r->saved;
        current = std::move(r->ops);
        result.improved = true;
    }
    if (result.improved)
        result.ops = std::move(current);
    return result;
}

}  // namespace quspect::native::transpiler
