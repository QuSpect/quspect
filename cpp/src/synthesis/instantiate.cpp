#include "quspect_native/synthesis/instantiate.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::synthesis {

namespace dm = quspect::native::detmath;

namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kTwoPi = 6.283185307179586;
constexpr int kRes = 128;  // 64 个复元素的实部与虚部

enum OpKind : int { kRz = 0, kRy = 1, kCz = 2 };

struct Op {
    int kind;
    int a;
    int b;
    int p;  // 参数下标（CZ 为 −1）
};

std::vector<Op> BuildOps(const std::vector<std::array<int, 2>>& czs) {
    std::vector<Op> ops;
    ops.reserve(9 + 5 * czs.size());
    for (int q = 0; q < 3; ++q) {
        ops.push_back({kRz, q, -1, 3 * q});
        ops.push_back({kRy, q, -1, 3 * q + 1});
        ops.push_back({kRz, q, -1, 3 * q + 2});
    }
    int p = 9;
    for (const auto& e : czs) {
        ops.push_back({kCz, e[0], e[1], -1});
        ops.push_back({kRy, e[0], -1, p});
        ops.push_back({kRz, e[0], -1, p + 1});
        ops.push_back({kRy, e[1], -1, p + 2});
        ops.push_back({kRz, e[1], -1, p + 3});
        p += 4;
    }
    return ops;
}

M8 Identity() {
    M8 u{};
    for (int i = 0; i < 8; ++i) {
        u[static_cast<std::size_t>(i * 9)] = Cx{1.0, 0.0};
    }
    return u;
}

// 单比特门矩阵 (g00, g01, g10, g11)。
std::array<Cx, 4> Gate1(int kind, double theta) {
    const double h = 0.5 * theta;
    const double c = dm::Cos(h);
    const double s = dm::Sin(h);
    if (kind == kRz) {
        return {Cx{c, -s}, Cx{0.0, 0.0}, Cx{0.0, 0.0}, Cx{c, s}};
    }
    return {Cx{c, 0.0}, Cx{-s, 0.0}, Cx{s, 0.0}, Cx{c, 0.0}};
}

int Bit(int q) {
    return 1 << (2 - q);
}

// u ← G(q) · u
void Left1(M8& u, int q, const std::array<Cx, 4>& g) {
    const int bit = Bit(q);
    for (int r = 0; r < 8; ++r) {
        if (r & bit) {
            continue;
        }
        const int r1 = r | bit;
        for (int c = 0; c < 8; ++c) {
            const Cx x = u[static_cast<std::size_t>(r * 8 + c)];
            const Cx y = u[static_cast<std::size_t>(r1 * 8 + c)];
            u[static_cast<std::size_t>(r * 8 + c)] = dm::Add(dm::Mul(g[0], x), dm::Mul(g[1], y));
            u[static_cast<std::size_t>(r1 * 8 + c)] = dm::Add(dm::Mul(g[2], x), dm::Mul(g[3], y));
        }
    }
}

// l ← l · G(q)
void Right1(M8& l, int q, const std::array<Cx, 4>& g) {
    const int bit = Bit(q);
    for (int c = 0; c < 8; ++c) {
        if (c & bit) {
            continue;
        }
        const int c1 = c | bit;
        for (int r = 0; r < 8; ++r) {
            const Cx x = l[static_cast<std::size_t>(r * 8 + c)];
            const Cx y = l[static_cast<std::size_t>(r * 8 + c1)];
            l[static_cast<std::size_t>(r * 8 + c)] = dm::Add(dm::Mul(x, g[0]), dm::Mul(y, g[2]));
            l[static_cast<std::size_t>(r * 8 + c1)] = dm::Add(dm::Mul(x, g[1]), dm::Mul(y, g[3]));
        }
    }
}

void LeftCz(M8& u, int a, int b) {
    const int m = Bit(a) | Bit(b);
    for (int r = 0; r < 8; ++r) {
        if ((r & m) == m) {
            for (int c = 0; c < 8; ++c) {
                u[static_cast<std::size_t>(r * 8 + c)] = dm::Neg(u[static_cast<std::size_t>(r * 8 + c)]);
            }
        }
    }
}

void RightCz(M8& l, int a, int b) {
    const int m = Bit(a) | Bit(b);
    for (int c = 0; c < 8; ++c) {
        if ((c & m) == m) {
            for (int r = 0; r < 8; ++r) {
                l[static_cast<std::size_t>(r * 8 + c)] = dm::Neg(l[static_cast<std::size_t>(r * 8 + c)]);
            }
        }
    }
}

M8 Forward(const std::vector<Op>& ops, const double* x) {
    M8 u = Identity();
    for (const Op& op : ops) {
        if (op.kind == kCz) {
            LeftCz(u, op.a, op.b);
        } else {
            Left1(u, op.a, Gate1(op.kind, x[op.p]));
        }
    }
    return u;
}

// tr(U†V) = Σ conj(U)·V（MulConj(a, b) = conj(a)·b）
Cx OverlapConjU(const M8& u, const M8& v) {
    Cx s{0.0, 0.0};
    for (int e = 0; e < 64; ++e) {
        s = dm::Add(s, dm::MulConj(u[static_cast<std::size_t>(e)], v[static_cast<std::size_t>(e)]));
    }
    return s;
}

double MaxResidual(const M8& v, const M8& u, const Cx& w) {
    double m = 0.0;
    for (int e = 0; e < 64; ++e) {
        const Cx d = dm::Sub(v[static_cast<std::size_t>(e)], dm::Mul(w, u[static_cast<std::size_t>(e)]));
        m = std::max(m, std::max(d.re < 0 ? -d.re : d.re, d.im < 0 ? -d.im : d.im));
    }
    return m;
}

double HalfCost(const M8& v, const M8& u, const Cx& w) {
    double s = 0.0;
    for (int e = 0; e < 64; ++e) {
        const Cx d = dm::Sub(v[static_cast<std::size_t>(e)], dm::Mul(w, u[static_cast<std::size_t>(e)]));
        s += d.re * d.re + d.im * d.im;
    }
    return 0.5 * s;
}

// 相位无关的距离 √(1 − |tr(U†V)|²/64)。
double Heuristic(const M8& v, const M8& u) {
    const Cx t = OverlapConjU(u, v);
    const double f = (t.re * t.re + t.im * t.im) / 64.0;
    return f >= 1.0 ? 0.0 : dm::Sqrt(1.0 - f);
}

// 残差与雅可比：j 为 kRes × n（行主序），n = P + 1（最后一列是相位 φ）。
void Jacobian(const std::vector<Op>& ops, const double* x, int num_params, const M8& u, double phi,
              std::vector<double>& jac, std::vector<double>& res, M8& v_out) {
    const int n = num_params + 1;
    const std::size_t m = ops.size();
    std::vector<M8> rs(m);
    M8 cur = Identity();
    std::vector<std::array<Cx, 4>> gates(m);
    for (std::size_t j = 0; j < m; ++j) {
        const Op& op = ops[j];
        if (op.kind == kCz) {
            LeftCz(cur, op.a, op.b);
        } else {
            gates[j] = Gate1(op.kind, x[op.p]);
            Left1(cur, op.a, gates[j]);
        }
        rs[j] = cur;
    }
    v_out = cur;
    const Cx w = dm::ExpI(phi);
    jac.assign(static_cast<std::size_t>(kRes * n), 0.0);
    res.assign(kRes, 0.0);
    for (int e = 0; e < 64; ++e) {
        const Cx d = dm::Sub(cur[static_cast<std::size_t>(e)], dm::Mul(w, u[static_cast<std::size_t>(e)]));
        res[static_cast<std::size_t>(e)] = d.re;
        res[static_cast<std::size_t>(64 + e)] = d.im;
        // ∂r/∂φ = −i·w·U
        const Cx wu = dm::Mul(w, u[static_cast<std::size_t>(e)]);
        jac[static_cast<std::size_t>(e * n + num_params)] = wu.im;
        jac[static_cast<std::size_t>((64 + e) * n + num_params)] = -wu.re;
    }
    M8 left = Identity();
    for (std::size_t jj = m; jj-- > 0;) {
        const Op& op = ops[jj];
        if (op.kind != kCz) {
            // ∂V/∂θ = L · (−i/2) σ · R_j，σ = Z（Rz）或 Y（Ry）作用在 op.a 上
            M8 xs = rs[jj];
            const int bit = Bit(op.a);
            if (op.kind == kRz) {
                for (int r = 0; r < 8; ++r) {
                    if (r & bit) {
                        for (int c = 0; c < 8; ++c) {
                            xs[static_cast<std::size_t>(r * 8 + c)] = dm::Neg(xs[static_cast<std::size_t>(r * 8 + c)]);
                        }
                    }
                }
            } else {
                for (int r = 0; r < 8; ++r) {
                    if (r & bit) {
                        continue;
                    }
                    const int r1 = r | bit;
                    for (int c = 0; c < 8; ++c) {
                        const Cx a0 = rs[jj][static_cast<std::size_t>(r * 8 + c)];
                        const Cx a1 = rs[jj][static_cast<std::size_t>(r1 * 8 + c)];
                        // Y = [[0, −i], [i, 0]]
                        xs[static_cast<std::size_t>(r * 8 + c)] = Cx{a1.im, -a1.re};
                        xs[static_cast<std::size_t>(r1 * 8 + c)] = Cx{-a0.im, a0.re};
                    }
                }
            }
            const int col = op.p;
            for (int r = 0; r < 8; ++r) {
                for (int c = 0; c < 8; ++c) {
                    Cx s{0.0, 0.0};
                    for (int k = 0; k < 8; ++k) {
                        s = dm::Add(s, dm::Mul(left[static_cast<std::size_t>(r * 8 + k)],
                                               xs[static_cast<std::size_t>(k * 8 + c)]));
                    }
                    // (−i/2)·s
                    const int e = r * 8 + c;
                    jac[static_cast<std::size_t>(e * n + col)] = 0.5 * s.im;
                    jac[static_cast<std::size_t>((64 + e) * n + col)] = -0.5 * s.re;
                }
            }
            Right1(left, op.a, gates[jj]);
        } else {
            RightCz(left, op.a, op.b);
        }
    }
}

// 解 (A) δ = b（A 对称正定，就地 Cholesky）。失败返回 false。
bool CholeskySolve(std::vector<double>& a, std::vector<double>& b, int n) {
    for (int j = 0; j < n; ++j) {
        double d = a[static_cast<std::size_t>(j * n + j)];
        for (int k = 0; k < j; ++k) {
            const double l = a[static_cast<std::size_t>(j * n + k)];
            d -= l * l;
        }
        if (!(d > 0.0)) {
            return false;
        }
        const double ljj = dm::Sqrt(d);
        a[static_cast<std::size_t>(j * n + j)] = ljj;
        for (int i = j + 1; i < n; ++i) {
            double s = a[static_cast<std::size_t>(i * n + j)];
            for (int k = 0; k < j; ++k) {
                s -= a[static_cast<std::size_t>(i * n + k)] * a[static_cast<std::size_t>(j * n + k)];
            }
            a[static_cast<std::size_t>(i * n + j)] = s / ljj;
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k) {
            s -= a[static_cast<std::size_t>(i * n + k)] * b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] = s / a[static_cast<std::size_t>(i * n + i)];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k) {
            s -= a[static_cast<std::size_t>(k * n + i)] * b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] = s / a[static_cast<std::size_t>(i * n + i)];
    }
    return true;
}

struct Fit {
    std::vector<double> x;  // P + 1（末尾为 φ）
    double max_res = 1e300;
    double heuristic = 1e300;
};

// 一个起点的 LM。iterations 累加用掉的迭代数。
Fit Optimize(const std::vector<Op>& ops, int num_params, const M8& u, std::vector<double> x,
             const InstantiateOptions& opt, std::int64_t& iterations) {
    const int n = num_params + 1;
    {
        const M8 v0 = Forward(ops, x.data());
        x[static_cast<std::size_t>(num_params)] = dm::Arg(OverlapConjU(u, v0));
    }
    std::vector<double> jac, res, a(static_cast<std::size_t>(n * n)), g(static_cast<std::size_t>(n));
    std::vector<double> trial(static_cast<std::size_t>(n)), sys, rhs;
    M8 v{};
    double lambda = 1e-3;
    double cost = -1.0;
    int limit = opt.max_iters;
    int stall = 0;
    for (int it = 0; it < limit; ++it) {
        Jacobian(ops, x.data(), num_params, u, x[static_cast<std::size_t>(num_params)], jac, res, v);
        ++iterations;
        double c0 = 0.0;
        for (double r : res) {
            c0 += r * r;
        }
        c0 *= 0.5;
        cost = c0;
        const double mres = MaxResidual(v, u, dm::ExpI(x[static_cast<std::size_t>(num_params)]));
        if (mres <= opt.tol) {
            break;
        }
        if (mres <= 1e-6 && limit == opt.max_iters) {
            limit = opt.max_iters + opt.polish_iters;  // 接近解：允许多几步把残差压到 tol
        }
        // A = JᵀJ, g = Jᵀr
        for (int i = 0; i < n; ++i) {
            for (int k = i; k < n; ++k) {
                double s = 0.0;
                for (int row = 0; row < kRes; ++row) {
                    s += jac[static_cast<std::size_t>(row * n + i)] * jac[static_cast<std::size_t>(row * n + k)];
                }
                a[static_cast<std::size_t>(i * n + k)] = s;
                a[static_cast<std::size_t>(k * n + i)] = s;
            }
            double s = 0.0;
            for (int row = 0; row < kRes; ++row) {
                s += jac[static_cast<std::size_t>(row * n + i)] * res[static_cast<std::size_t>(row)];
            }
            g[static_cast<std::size_t>(i)] = s;
        }
        bool accepted = false;
        for (int tries = 0; tries < 12 && !accepted; ++tries) {
            sys = a;
            rhs.assign(static_cast<std::size_t>(n), 0.0);
            for (int i = 0; i < n; ++i) {
                const double d = a[static_cast<std::size_t>(i * n + i)];
                sys[static_cast<std::size_t>(i * n + i)] = d + lambda * (d > 1e-12 ? d : 1e-12);
                rhs[static_cast<std::size_t>(i)] = -g[static_cast<std::size_t>(i)];
            }
            if (!CholeskySolve(sys, rhs, n)) {
                lambda *= 4.0;
                continue;
            }
            for (int i = 0; i < n; ++i) {
                trial[static_cast<std::size_t>(i)] = x[static_cast<std::size_t>(i)] + rhs[static_cast<std::size_t>(i)];
            }
            const M8 vt = Forward(ops, trial.data());
            const double ct = HalfCost(vt, u, dm::ExpI(trial[static_cast<std::size_t>(num_params)]));
            if (ct < c0) {
                accepted = true;
                x = trial;
                lambda = std::max(lambda * 0.3, 1e-15);
                stall = (ct > c0 * (1.0 - 1e-6)) ? stall + 1 : 0;
            } else {
                lambda *= 4.0;
            }
        }
        if (!accepted || stall >= 3 || lambda > 1e12) {
            break;
        }
    }
    Fit f;
    const M8 vf = Forward(ops, x.data());
    f.max_res = MaxResidual(vf, u, dm::ExpI(x[static_cast<std::size_t>(num_params)]));
    f.heuristic = Heuristic(vf, u);
    f.x = std::move(x);
    (void)cost;
    return f;
}

struct Node {
    int target;
    std::vector<std::array<int, 2>> czs;
    std::vector<double> x;  // 含 φ
    double f;
    std::int64_t seq;
};

std::uint64_t NextRand(std::uint64_t& s) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}

double RandAngle(std::uint64_t& s) {
    const std::uint64_t r = NextRand(s) >> 11;  // 53 位
    return static_cast<double>(r) * (kTwoPi / 9007199254740992.0) - kPi;
}

double Wrap(double t) {
    return dm::Remainder(t, kTwoPi);
}

}  // namespace

M8 InstantiateUnitary(const std::vector<std::array<int, 2>>& czs, const std::vector<double>& params) {
    if (params.size() != 9 + 4 * czs.size()) {
        throw std::invalid_argument("expected 9 + 4·len(czs) parameters");
    }
    return Forward(BuildOps(czs), params.data());
}

InstantiateResult InstantiateSearch(const std::vector<M8>& targets, const std::vector<std::array<int, 2>>& edges,
                                    const InstantiateOptions& opt) {
    for (const auto& e : edges) {
        if (e[0] < 0 || e[0] > 2 || e[1] < 0 || e[1] > 2 || e[0] == e[1]) {
            throw std::invalid_argument("edge on local qubits 0..2 expected");
        }
    }
    InstantiateResult out;
    std::uint64_t rng = opt.seed;
    std::int64_t seq = 0;
    int kbound = opt.max_cz;
    std::vector<Node> heap;
    auto cmp = [](const Node& p, const Node& q) {
        // std::push_heap 维护最大堆：f 小、seq 小的优先
        if (p.f != q.f) {
            return p.f > q.f;
        }
        return p.seq > q.seq;
    };

    // 实例化一个结构：起点 = seed_x（可为空）+ restarts 个随机起点；返回最好的拟合。
    auto instantiate = [&](int t, const std::vector<std::array<int, 2>>& czs, const std::vector<double>* seed_x) {
        const std::vector<Op> ops = BuildOps(czs);
        const int num_params = 9 + 4 * static_cast<int>(czs.size());
        Fit best;
        const int starts = (seed_x != nullptr ? 1 : 0) + std::max(0, opt.restarts) + (seed_x == nullptr ? 1 : 0);
        for (int s = 0; s < starts && out.iterations < opt.budget; ++s) {
            std::vector<double> x(static_cast<std::size_t>(num_params + 1), 0.0);
            if (s == 0 && seed_x != nullptr) {
                const std::size_t keep = std::min(seed_x->size() - 1, static_cast<std::size_t>(num_params));
                std::copy(seed_x->begin(), seed_x->begin() + static_cast<std::ptrdiff_t>(keep), x.begin());
            } else {
                for (int i = 0; i < num_params; ++i) {
                    x[static_cast<std::size_t>(i)] = RandAngle(rng);
                }
            }
            Fit f = Optimize(ops, num_params, targets[static_cast<std::size_t>(t)], std::move(x), opt, out.iterations);
            const bool better = best.x.empty() || (f.max_res <= opt.tol && best.max_res > opt.tol) ||
                                (best.max_res > opt.tol && f.heuristic < best.heuristic);
            if (better) {
                best = std::move(f);
            }
            if (best.max_res <= opt.tol) {
                break;
            }
        }
        ++out.nodes;
        return best;
    };

    auto record = [&](int t, const std::vector<std::array<int, 2>>& czs, const Fit& f) {
        const int k = static_cast<int>(czs.size());
        if (f.max_res > opt.tol || k > kbound) {
            return false;
        }
        std::vector<double> p(f.x.begin(), f.x.end() - 1);
        for (double& v : p) {
            v = Wrap(v);
        }
        const M8 v = InstantiateUnitary(czs, p);
        const M8& u = targets[static_cast<std::size_t>(t)];
        const Cx ov = OverlapConjU(u, v);
        const double mag = dm::Abs(ov);
        if (!(mag > 0.0)) {
            return false;
        }
        const Cx w = dm::DivReal(ov, mag);
        const double res = MaxResidual(v, u, w);
        if (res > opt.tol) {
            return false;
        }
        out.found = true;
        out.target = t;
        out.czs = czs;
        out.params = std::move(p);
        out.residual = res;
        kbound = k - 1;
        return true;
    };

    for (int t = 0; t < static_cast<int>(targets.size()); ++t) {
        if (out.iterations >= opt.budget) {
            break;
        }
        std::vector<std::array<int, 2>> none;
        Fit f = instantiate(t, none, nullptr);
        if (record(t, none, f)) {
            return out;  // 0 个 CZ：不可能更少
        }
        heap.push_back({t, none, std::move(f.x), opt.heuristic_factor * f.heuristic, seq++});
        std::push_heap(heap.begin(), heap.end(), cmp);
    }

    while (!heap.empty() && out.iterations < opt.budget) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        Node node = std::move(heap.back());
        heap.pop_back();
        const int k = static_cast<int>(node.czs.size());
        if (k + 1 > kbound) {
            continue;
        }
        for (const auto& e : edges) {
            if (out.iterations >= opt.budget || k + 1 > kbound) {
                break;
            }
            // 同一对上连续 4 个 CZ 是多余的（任意两比特门 ≤ 3 个 CZ）
            if (k >= 3) {
                bool run = true;
                for (int i = k - 3; i < k; ++i) {
                    const auto& c = node.czs[static_cast<std::size_t>(i)];
                    if (!((c[0] == e[0] && c[1] == e[1]) || (c[0] == e[1] && c[1] == e[0]))) {
                        run = false;
                        break;
                    }
                }
                if (run) {
                    continue;
                }
            }
            std::vector<std::array<int, 2>> czs = node.czs;
            czs.push_back(e);
            Fit f = instantiate(node.target, czs, &node.x);
            if (record(node.target, czs, f)) {
                continue;
            }
            heap.push_back({node.target, std::move(czs), std::move(f.x),
                            static_cast<double>(k + 1) + opt.heuristic_factor * f.heuristic, seq++});
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }
    return out;
}

}  // namespace quspect::native::synthesis
