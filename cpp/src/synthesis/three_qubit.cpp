#include "quspect_native/synthesis/three_qubit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::synthesis {

namespace dm = quspect::native::detmath;

namespace {

M8 Identity8() {
    M8 u{};
    for (int i = 0; i < 8; ++i) {
        u[static_cast<std::size_t>(i * 9)] = Cx{1.0, 0.0};
    }
    return u;
}

// out = G(local) · u：行 r 的局部比特取值 g(r)，新行 = Σ_{g'} G[g(r), g'] · u[r(g')]（g' 升序）。
M8 Apply(const M8& u, const LocalGate& gate) {
    const int k = static_cast<int>(gate.qubits.size());
    if (k < 1 || k > 3) {
        throw std::invalid_argument("three-qubit gate on 1..3 local qubits expected");
    }
    for (int q : gate.qubits) {
        if (q < 0 || q > 2) {
            throw std::invalid_argument("local qubit out of range");
        }
    }
    const int dim = 1 << k;
    M8 out{};
    for (int r = 0; r < 8; ++r) {
        int g = 0;
        for (int j = 0; j < k; ++j) {
            g = (g << 1) | ((r >> (2 - gate.qubits[static_cast<std::size_t>(j)])) & 1);
        }
        int base = r;
        for (int j = 0; j < k; ++j) {
            base &= ~(1 << (2 - gate.qubits[static_cast<std::size_t>(j)]));
        }
        for (int c = 0; c < 8; ++c) {
            Cx s{0.0, 0.0};
            for (int gp = 0; gp < dim; ++gp) {
                int src = base;
                for (int j = 0; j < k; ++j) {
                    if ((gp >> (k - 1 - j)) & 1) {
                        src |= 1 << (2 - gate.qubits[static_cast<std::size_t>(j)]);
                    }
                }
                s = dm::Add(s, dm::Mul(gate.matrix[g * dim + gp], u[static_cast<std::size_t>(src * 8 + c)]));
            }
            out[static_cast<std::size_t>(r * 8 + c)] = s;
        }
    }
    return out;
}

double OffDiagonal(const M4& d) {
    double err = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                const double a = dm::Abs(d[static_cast<std::size_t>(i * 4 + j)]);
                if (!(a == a)) {
                    return a;
                }
                err = std::max(err, a);
            }
        }
    }
    return err;
}

M4 Block(const M8& u, int row, int col) {
    M4 out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[static_cast<std::size_t>(i * 4 + j)] = u[static_cast<std::size_t>((row + i) * 8 + col + j)];
        }
    }
    return out;
}

}  // namespace

M8 ThreeQubitUnitary(const std::vector<LocalGate>& gates) {
    M8 u = Identity8();
    for (const LocalGate& g : gates) {
        u = Apply(u, g);
    }
    return u;
}

double ThreeQubitDistance(const M8& u, const M8& v) {
    const Cx overlap = dm::Vdot(64, v.data(), u.data());
    const double ab = dm::Abs(overlap);
    if (ab < 1e-300) {
        return std::numeric_limits<double>::infinity();
    }
    const Cx phase = dm::DivReal(overlap, ab);
    double dist = 0.0;
    for (std::size_t e = 0; e < 64; ++e) {
        const double a = dm::Abs(dm::Sub(u[e], dm::Mul(phase, v[e])));
        if (!(a == a)) {
            return a;
        }
        dist = std::max(dist, a);
    }
    return dist;
}

std::optional<std::pair<M4, std::array<Cx, 4>>> UnitaryEig(const M4& m, double tol) {
    // H(t) = (e^{−it}·m + e^{it}·m†)/2 是厄米矩阵，特征向量与 m 相同（特征值 cos(θ_k − t)）；
    // 对几个固定的 t 求 H(t) 的特征分解，取使 V†·m·V 最接近对角的一个（简并的 H(t) 换下一个 t）。
    M4 best_v{};
    std::array<Cx, 4> best_lam{};
    double best_err = std::numeric_limits<double>::infinity();
    for (int attempt = 0; attempt < 12 && !(best_err <= 1e-13); ++attempt) {
        const double t = 0.5 + 0.7 * attempt;
        const Cx e = dm::ExpI(-t);
        M4 h;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const Cx x = dm::Mul(e, m[static_cast<std::size_t>(i * 4 + j)]);
                const Cx y = dm::Conj(dm::Mul(e, m[static_cast<std::size_t>(j * 4 + i)]));
                h[static_cast<std::size_t>(i * 4 + j)] = dm::Scale(dm::Add(x, y), 0.5);
            }
        }
        double w[4];
        M4 v;
        if (!dm::EighHerm(4, h.data(), w, v.data())) {
            continue;
        }
        M4 mv;
        M4 d;
        dm::MatMul(4, m.data(), v.data(), mv.data());
        dm::MatMulAdjA(4, v.data(), mv.data(), d.data());
        const double err = OffDiagonal(d);
        if (err < best_err) {
            best_err = err;
            best_v = v;
            for (int k = 0; k < 4; ++k) {
                best_lam[static_cast<std::size_t>(k)] = d[static_cast<std::size_t>(k * 5)];
            }
        }
    }
    if (!(best_err <= tol)) {
        return std::nullopt;
    }
    // 规范化特征基：特征值相同（|λ_i − λ_j| ≤ 1e-8）的特征向量张成的子空间里改取"最接近计算基"的
    // 一组基，并按计算基的下标排列列（列 i 由 e_i 在所属特征子空间上的投影正交化得到）。这样已经
    // 是对角（或块对角、带简并）的 m 得到 V = I 或局部的 V，后续两比特合成需要的纠缠门最少——
    // 特征向量基的取法不影响分解的正确性，只影响 V、W 的合成代价。
    M4 v = best_v;
    std::array<Cx, 4> lam = best_lam;
    int cluster[4];
    for (int i = 0; i < 4; ++i) {
        cluster[i] = i;
        for (int j = 0; j < i; ++j) {
            if (dm::Abs(dm::Sub(lam[static_cast<std::size_t>(i)], lam[static_cast<std::size_t>(j)])) <= 1e-8) {
                cluster[i] = cluster[j];
                break;
            }
        }
    }
    M4 out{};
    std::array<Cx, 4> out_lam{};
    bool used[4] = {false, false, false, false};
    // 每个簇的投影 P_c = Σ v_j v_j^†；依次（簇按首个特征向量的下标）为簇里的每个向量选一个未用的
    // 计算基下标 i（使 P_c e_i 对本簇已选向量正交化后的剩余最大，相等时取下标小者）。
    for (int c = 0; c < 4; ++c) {
        int members[4];
        int k = 0;
        for (int j = 0; j < 4; ++j) {
            if (cluster[j] == c) {
                members[k++] = j;
            }
        }
        if (k == 0) {
            continue;
        }
        int chosen[4];
        for (int step = 0; step < k; ++step) {
            int best_i = -1;
            double best_n = -1.0;
            Cx best_z[4];
            for (int i = 0; i < 4; ++i) {
                if (used[i]) {
                    continue;
                }
                Cx z[4];
                for (int r = 0; r < 4; ++r) {
                    Cx acc{0.0, 0.0};
                    for (int t = 0; t < k; ++t) {
                        const std::size_t col = static_cast<std::size_t>(members[t]);
                        acc = dm::Add(acc, dm::Mul(v[static_cast<std::size_t>(r * 4) + col],
                                                   dm::Conj(v[static_cast<std::size_t>(i * 4) + col])));
                    }
                    z[r] = acc;  // (P_c e_i)_r
                }
                for (int pass = 0; pass < 2; ++pass) {
                    for (int t = 0; t < step; ++t) {
                        const std::size_t col = static_cast<std::size_t>(chosen[t]);
                        Cx dot{0.0, 0.0};
                        for (int r = 0; r < 4; ++r) {
                            dot = dm::Add(dot, dm::MulConj(out[static_cast<std::size_t>(r * 4) + col], z[r]));
                        }
                        for (int r = 0; r < 4; ++r) {
                            z[r] = dm::Sub(z[r], dm::Mul(dot, out[static_cast<std::size_t>(r * 4) + col]));
                        }
                    }
                }
                double n2 = 0.0;
                for (int r = 0; r < 4; ++r) {
                    n2 += dm::Norm2(z[r]);
                }
                if (n2 > best_n) {
                    best_n = n2;
                    best_i = i;
                    std::copy(z, z + 4, best_z);
                }
            }
            if (best_i < 0 || !(best_n > 0.0)) {
                return std::make_pair(best_v, best_lam);  // 不应发生：保留原来的特征基
            }
            used[best_i] = true;
            chosen[step] = best_i;
            // 相位：第 best_i 个分量为正实数
            const double inv = 1.0 / dm::Sqrt(best_n);
            const double ab = dm::Abs(best_z[best_i]);
            const Cx ph = ab > 0.0 ? dm::Conj(dm::DivReal(best_z[best_i], ab)) : Cx{1.0, 0.0};
            for (int r = 0; r < 4; ++r) {
                out[static_cast<std::size_t>(r * 4 + best_i)] = dm::Scale(dm::Mul(best_z[r], ph), inv);
            }
            out_lam[static_cast<std::size_t>(best_i)] = lam[static_cast<std::size_t>(members[0])];
        }
    }
    // 新基的残差（特征值取 V^† m V 的对角元）不超过容差才采用
    M4 mv;
    M4 d;
    dm::MatMul(4, m.data(), out.data(), mv.data());
    dm::MatMulAdjA(4, out.data(), mv.data(), d.data());
    if (!(OffDiagonal(d) <= tol)) {
        return std::make_pair(best_v, best_lam);
    }
    for (int k = 0; k < 4; ++k) {
        out_lam[static_cast<std::size_t>(k)] = d[static_cast<std::size_t>(k * 5)];
    }
    return std::make_pair(out, out_lam);
}

std::optional<Demultiplexed> Demultiplex(const M4& a, const M4& b) {
    M4 m;
    dm::MatMulAdjB(4, a.data(), b.data(), m.data());  // a @ b†
    const auto eig = UnitaryEig(m, 1e-9);
    if (!eig) {
        return std::nullopt;
    }
    Demultiplexed out;
    out.v = eig->first;
    M4 vb;
    dm::MatMulAdjA(4, out.v.data(), b.data(), vb.data());  // v† @ b
    for (int k = 0; k < 4; ++k) {
        const double phase = dm::Arg(eig->second[static_cast<std::size_t>(k)]) / 2.0;
        out.phases[static_cast<std::size_t>(k)] = phase;
        const Cx d = dm::ExpI(phase);
        for (int j = 0; j < 4; ++j) {
            out.w[static_cast<std::size_t>(k * 4 + j)] = dm::Mul(d, vb[static_cast<std::size_t>(k * 4 + j)]);
        }
    }
    return out;
}

std::optional<CosSin> CosSinDecompose(const M8& u) {
    const M4 u11 = Block(u, 0, 0);
    const M4 u12 = Block(u, 0, 4);
    const M4 u21 = Block(u, 4, 0);
    const M4 u22 = Block(u, 4, 4);
    CosSin out;
    double c[4];
    if (!dm::Svd(4, u11.data(), out.u1.data(), c, out.v1h.data())) {
        return std::nullopt;
    }
    // X = u21 · v1h† = u2 · S：列 i 的模为 s_i。按模降序（相等时下标小者优先）对列做两遍复
    // Gram–Schmidt 得 u2 的列；模几乎为零的列（s_i ≈ 0）用标准基向量的正交补补齐。
    M4 x;
    dm::MatMulAdjB(4, u21.data(), out.v1h.data(), x.data());
    double norm2[4];
    int order[4] = {0, 1, 2, 3};
    for (int i = 0; i < 4; ++i) {
        norm2[i] = 0.0;
        for (int r = 0; r < 4; ++r) {
            norm2[i] += dm::Norm2(x[static_cast<std::size_t>(r * 4 + i)]);
        }
    }
    std::stable_sort(order, order + 4, [&norm2](int p, int q) { return norm2[p] > norm2[q]; });
    M4 u2{};
    int done[4];
    auto orthogonalize = [&](Cx* z, int count) {
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < count; ++j) {
                const int col = done[j];
                Cx dot{0.0, 0.0};
                for (int r = 0; r < 4; ++r) {
                    dot = dm::Add(dot, dm::MulConj(u2[static_cast<std::size_t>(r * 4 + col)], z[r]));
                }
                for (int r = 0; r < 4; ++r) {
                    z[r] = dm::Sub(z[r], dm::Mul(dot, u2[static_cast<std::size_t>(r * 4 + col)]));
                }
            }
        }
        double n2 = 0.0;
        for (int r = 0; r < 4; ++r) {
            n2 += dm::Norm2(z[r]);
        }
        return n2;
    };
    for (int step = 0; step < 4; ++step) {
        const int i = order[step];
        Cx z[4];
        for (int r = 0; r < 4; ++r) {
            z[r] = x[static_cast<std::size_t>(r * 4 + i)];
        }
        double n2 = orthogonalize(z, step);
        if (!(n2 > 1e-24)) {
            double best = -1.0;
            Cx best_z[4];
            for (int e = 0; e < 4; ++e) {
                Cx t[4];
                for (int r = 0; r < 4; ++r) {
                    t[r] = Cx{r == e ? 1.0 : 0.0, 0.0};
                }
                const double tn = orthogonalize(t, step);
                if (tn > best) {
                    best = tn;
                    std::copy(t, t + 4, best_z);
                }
            }
            std::copy(best_z, best_z + 4, z);
            n2 = best;
        }
        if (!(n2 > 0.0)) {
            return std::nullopt;
        }
        const double inv = 1.0 / dm::Sqrt(n2);
        for (int r = 0; r < 4; ++r) {
            u2[static_cast<std::size_t>(r * 4 + i)] = dm::Scale(z[r], inv);
        }
        // 相位：使 u2_i† X_i 为非负实数，s_i = |u2_i† X_i|
        Cx proj{0.0, 0.0};
        for (int r = 0; r < 4; ++r) {
            const std::size_t e = static_cast<std::size_t>(r * 4 + i);
            proj = dm::Add(proj, dm::MulConj(u2[e], x[e]));
        }
        const double s = dm::Abs(proj);
        if (s > 0.0) {
            const Cx ph = dm::DivReal(proj, s);
            for (int r = 0; r < 4; ++r) {
                u2[static_cast<std::size_t>(r * 4 + i)] = dm::Mul(u2[static_cast<std::size_t>(r * 4 + i)], ph);
            }
        }
        out.theta[static_cast<std::size_t>(i)] = dm::Atan2(s, std::min(c[i], 1.0));
        done[step] = i;
    }
    out.u2 = u2;
    // v2h = C·(u2† u22) − S·(u1† u12)
    M4 a;
    M4 b;
    dm::MatMulAdjA(4, out.u2.data(), u22.data(), a.data());
    dm::MatMulAdjA(4, out.u1.data(), u12.data(), b.data());
    for (int i = 0; i < 4; ++i) {
        const double ci = dm::Cos(out.theta[static_cast<std::size_t>(i)]);
        const double si = dm::Sin(out.theta[static_cast<std::size_t>(i)]);
        for (int j = 0; j < 4; ++j) {
            const std::size_t e = static_cast<std::size_t>(i * 4 + j);
            out.v2h[e] = dm::Sub(dm::Scale(a[e], ci), dm::Scale(b[e], si));
        }
    }
    return out;
}

}  // namespace quspect::native::synthesis
