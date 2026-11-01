#include "quspect_native/detmath/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace quspect::native::detmath {

namespace {

void CheckDim(int n) {
    if (n < 1 || n > kMaxDim) {
        throw std::invalid_argument("detmath: matrix dimension out of range");
    }
}

// 有限值判断（NaN、±Inf 为 false）。
bool Finite(double x) {
    return x - x == 0.0;
}

// 向量 x（步长 stride）中模最大的分量的下标（模相等时取下标最小者）。
int ArgMaxAbs(int n, const double* x, int stride) {
    int best = 0;
    double best_abs = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const double a = std::fabs(x[i * stride]);
        if (a > best_abs) {
            best_abs = a;
            best = i;
        }
    }
    return best;
}

int ArgMaxNorm2(int n, const Cx* x, int stride) {
    int best = 0;
    double best_abs = Norm2(x[0]);
    for (int i = 1; i < n; ++i) {
        const double a = Norm2(x[i * stride]);
        if (a > best_abs) {
            best_abs = a;
            best = i;
        }
    }
    return best;
}

// 实矩阵 v（n×n，行主序）每列的符号规范化：模最大的分量为正。
void NormalizeColumnSigns(int n, double* v) {
    for (int j = 0; j < n; ++j) {
        const int k = ArgMaxAbs(n, v + j, n);
        if (v[k * n + j] < 0.0) {
            for (int i = 0; i < n; ++i) {
                v[i * n + j] = -v[i * n + j];
            }
        }
    }
}

// 交换实矩阵 v 的第 i、j 列。
void SwapColumns(int n, double* v, int i, int j) {
    for (int r = 0; r < n; ++r) {
        std::swap(v[r * n + i], v[r * n + j]);
    }
}

void SwapColumns(int n, Cx* v, int i, int j) {
    for (int r = 0; r < n; ++r) {
        std::swap(v[r * n + i], v[r * n + j]);
    }
}

// Σ |x_k|²（k 升序）。
double ColumnNorm2(int n, const Cx* a, int col) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) {
        s += Norm2(a[k * n + col]);
    }
    return s;
}

// Σ conj(a_ki)·a_kj（k 升序）。
Cx ColumnDot(int n, const Cx* a, int i, int j) {
    Cx s{0.0, 0.0};
    for (int k = 0; k < n; ++k) {
        s = Add(s, MulConj(a[k * n + i], a[k * n + j]));
    }
    return s;
}

}  // namespace

// ---- 乘法 ----------------------------------------------------------------------

void MatMul(int n, const Cx* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Mul(a[i * n + k], b[k * n + j]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulAdjA(int n, const Cx* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, MulConj(a[k * n + i], b[k * n + j]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulAdjB(int n, const Cx* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Mul(a[i * n + k], Conj(b[j * n + k])));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulTransA(int n, const Cx* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Mul(a[k * n + i], b[k * n + j]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulRC(int n, const double* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Scale(b[k * n + j], a[i * n + k]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulCR(int n, const Cx* a, const double* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Scale(a[i * n + k], b[k * n + j]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulRTransC(int n, const double* a, const Cx* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Scale(b[k * n + j], a[k * n + i]));
            }
            out[i * n + j] = s;
        }
    }
}

void MatMulCRTrans(int n, const Cx* a, const double* b, Cx* out) {
    CheckDim(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Cx s{0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                s = Add(s, Scale(a[i * n + k], b[j * n + k]));
            }
            out[i * n + j] = s;
        }
    }
}

M2 MatMul(const M2& a, const M2& b) {
    M2 out;
    MatMul(2, a.data(), b.data(), out.data());
    return out;
}

M4 MatMul(const M4& a, const M4& b) {
    M4 out;
    MatMul(4, a.data(), b.data(), out.data());
    return out;
}

M2 Adjoint(const M2& a) {
    return {Conj(a[0]), Conj(a[2]), Conj(a[1]), Conj(a[3])};
}

M4 Adjoint(const M4& a) {
    M4 out;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i * 4 + j] = Conj(a[j * 4 + i]);
        }
    }
    return out;
}

Cx Vdot(int len, const Cx* a, const Cx* b) {
    Cx s{0.0, 0.0};
    for (int k = 0; k < len; ++k) {
        s = Add(s, MulConj(a[k], b[k]));
    }
    return s;
}

// ---- 行列式 ----------------------------------------------------------------------

Cx Det(int n, const Cx* a) {
    CheckDim(n);
    Cx m[kMaxDim * kMaxDim];
    for (int i = 0; i < n * n; ++i) {
        m[i] = a[i];
    }
    Cx det{1.0, 0.0};
    bool negate = false;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = std::fabs(m[k * n + k].re) + std::fabs(m[k * n + k].im);
        for (int r = k + 1; r < n; ++r) {
            const double v = std::fabs(m[r * n + k].re) + std::fabs(m[r * n + k].im);
            if (v > best) {
                best = v;
                p = r;
            }
        }
        if (best == 0.0) {
            return {0.0, 0.0};
        }
        if (p != k) {
            for (int c = 0; c < n; ++c) {
                std::swap(m[k * n + c], m[p * n + c]);
            }
            negate = !negate;
        }
        const Cx piv = m[k * n + k];
        det = Mul(det, piv);
        for (int r = k + 1; r < n; ++r) {
            const Cx f = Div(m[r * n + k], piv);
            for (int c = k + 1; c < n; ++c) {
                m[r * n + c] = Sub(m[r * n + c], Mul(f, m[k * n + c]));
            }
        }
    }
    return negate ? Neg(det) : det;
}

double DetReal(int n, const double* a) {
    CheckDim(n);
    double m[kMaxDim * kMaxDim];
    for (int i = 0; i < n * n; ++i) {
        m[i] = a[i];
    }
    double det = 1.0;
    bool negate = false;
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = std::fabs(m[k * n + k]);
        for (int r = k + 1; r < n; ++r) {
            const double v = std::fabs(m[r * n + k]);
            if (v > best) {
                best = v;
                p = r;
            }
        }
        if (best == 0.0) {
            return 0.0;
        }
        if (p != k) {
            for (int c = 0; c < n; ++c) {
                std::swap(m[k * n + c], m[p * n + c]);
            }
            negate = !negate;
        }
        const double piv = m[k * n + k];
        det *= piv;
        for (int r = k + 1; r < n; ++r) {
            const double f = m[r * n + k] / piv;
            for (int c = k + 1; c < n; ++c) {
                m[r * n + c] -= f * m[k * n + c];
            }
        }
    }
    return negate ? -det : det;
}

// ---- 实对称特征分解（循环 Jacobi）--------------------------------------------------

bool EighSym(int n, const double* a_in, double* w, double* v) {
    CheckDim(n);
    double a[kMaxDim * kMaxDim];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = i <= j ? a_in[i * n + j] : a_in[j * n + i];
            if (!Finite(a[i * n + j])) {
                return false;
            }
        }
    }
    for (int i = 0; i < n * n; ++i) {
        v[i] = 0.0;
    }
    double b[kMaxDim];
    double z[kMaxDim];
    for (int i = 0; i < n; ++i) {
        v[i * n + i] = 1.0;
        b[i] = a[i * n + i];
        w[i] = b[i];
        z[i] = 0.0;
    }
    // 只用上三角（p < q）。旋转 P：P_pp = P_qq = c，P_pq = s，P_qp = −s；a ← P^T a P，v ← v P。
    // 对角元的增量先累加到 z，每轮结束后再并入 b（Rutishauser，减少舍入误差的累积）。
    auto rotate = [](double* m, int i, int j, int k, int l, int dim, double s, double tau) {
        const double g = m[i * dim + j];
        const double h = m[k * dim + l];
        m[i * dim + j] = g - s * (h + g * tau);
        m[k * dim + l] = h + s * (g - h * tau);
    };
    bool converged = false;
    for (int sweep = 1; sweep <= 64; ++sweep) {
        double sm = 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                sm += std::fabs(a[p * n + q]);
            }
        }
        if (sm == 0.0) {
            converged = true;
            break;
        }
        const double tresh = sweep < 4 ? 0.2 * sm / static_cast<double>(n * n) : 0.0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                const double g = 100.0 * std::fabs(apq);
                if (sweep > 4 && std::fabs(w[p]) + g == std::fabs(w[p]) && std::fabs(w[q]) + g == std::fabs(w[q])) {
                    a[p * n + q] = 0.0;
                } else if (std::fabs(apq) > tresh) {
                    double h = w[q] - w[p];
                    double t;
                    if (std::fabs(h) + g == std::fabs(h)) {
                        t = apq / h;
                    } else {
                        const double theta = 0.5 * h / apq;
                        t = 1.0 / (std::fabs(theta) + Sqrt(1.0 + theta * theta));
                        if (theta < 0.0) {
                            t = -t;
                        }
                    }
                    const double c = 1.0 / Sqrt(1.0 + t * t);
                    const double s = t * c;
                    const double tau = s / (1.0 + c);
                    h = t * apq;
                    z[p] -= h;
                    z[q] += h;
                    w[p] -= h;
                    w[q] += h;
                    a[p * n + q] = 0.0;
                    for (int j = 0; j < p; ++j) {
                        rotate(a, j, p, j, q, n, s, tau);
                    }
                    for (int j = p + 1; j < q; ++j) {
                        rotate(a, p, j, j, q, n, s, tau);
                    }
                    for (int j = q + 1; j < n; ++j) {
                        rotate(a, p, j, q, j, n, s, tau);
                    }
                    for (int j = 0; j < n; ++j) {
                        rotate(v, j, p, j, q, n, s, tau);
                    }
                }
            }
        }
        for (int p = 0; p < n; ++p) {
            b[p] += z[p];
            w[p] = b[p];
            z[p] = 0.0;
        }
    }
    if (!converged) {
        return false;
    }
    // 特征值升序（选择排序：相等时保持原顺序）
    for (int i = 0; i < n - 1; ++i) {
        int k = i;
        for (int j = i + 1; j < n; ++j) {
            if (w[j] < w[k]) {
                k = j;
            }
        }
        if (k != i) {
            std::swap(w[i], w[k]);
            SwapColumns(n, v, i, k);
        }
    }
    NormalizeColumnSigns(n, v);
    return true;
}

// ---- 两个可交换实对称矩阵的同时对角化 ----------------------------------------------

bool JointDiagSym(int n, const double* a_in, const double* b_in, double* v) {
    CheckDim(n);
    double a[kMaxDim * kMaxDim];
    double b[kMaxDim * kMaxDim];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = i <= j ? a_in[i * n + j] : a_in[j * n + i];
            b[i * n + j] = i <= j ? b_in[i * n + j] : b_in[j * n + i];
            if (!Finite(a[i * n + j]) || !Finite(b[i * n + j])) {
                return false;
            }
        }
    }
    for (int i = 0; i < n * n; ++i) {
        v[i] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        v[i * n + i] = 1.0;
    }
    // 旋转 R：R_pp = R_qq = c，R_pq = s，R_qp = −s。(R^T m R)_pq = cos2θ·m_pq + sin2θ·(m_pp − m_qq)/2，
    // 即 (cos2θ, sin2θ) 与 g_m = (m_pq, (m_pp − m_qq)/2) 的内积；使 Σ_m 内积² 最小的单位向量是
    // G = Σ_m g_m g_m^T 的最小特征值对应的特征向量（取 cos2θ ≥ 0，即 |θ| ≤ π/4）。
    auto apply = [n](double* m, double c, double s, int p, int q) {
        for (int r = 0; r < n; ++r) {
            const double mp = m[r * n + p];
            const double mq = m[r * n + q];
            m[r * n + p] = c * mp - s * mq;
            m[r * n + q] = s * mp + c * mq;
        }
        for (int j = 0; j < n; ++j) {
            const double mp = m[p * n + j];
            const double mq = m[q * n + j];
            m[p * n + j] = c * mp - s * mq;
            m[q * n + j] = s * mp + c * mq;
        }
    };
    for (int sweep = 0; sweep < 100; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                const double bpq = b[p * n + q];
                const double off = apq * apq + bpq * bpq;
                const double app = a[p * n + p];
                const double aqq = a[q * n + q];
                const double bpp = b[p * n + p];
                const double bqq = b[q * n + q];
                const double scale = app * app + aqq * aqq + bpp * bpp + bqq * bqq;
                // 非对角元相对于对角元已在舍入误差以内（约 1e-15）时跳过
                if (!(off > 1e-30 * scale) || !(off > 1e-300)) {
                    continue;
                }
                const double da = 0.5 * (app - aqq);
                const double db = 0.5 * (bpp - bqq);
                const double g11 = off;
                const double g12 = apq * da + bpq * db;
                const double g22 = da * da + db * db;
                const double t = 0.5 * (g11 - g22);
                const double r = Hypot(t, g12);
                if (r == 0.0) {
                    continue;
                }
                double vx;
                double vy;
                if (t >= 0.0) {
                    vx = -g12;
                    vy = t + r;
                } else {
                    vx = r - t;
                    vy = -g12;
                }
                const double nv = Hypot(vx, vy);
                vx /= nv;
                vy /= nv;
                if (vx < 0.0) {
                    vx = -vx;
                    vy = -vy;
                }
                const double c = Sqrt(0.5 * (1.0 + vx));
                const double s = vy / (2.0 * c);
                if (s == 0.0) {
                    continue;
                }
                rotated = true;
                apply(a, c, s, p, q);
                apply(b, c, s, p, q);
                for (int row = 0; row < n; ++row) {
                    const double vp = v[row * n + p];
                    const double vq = v[row * n + q];
                    v[row * n + p] = c * vp - s * vq;
                    v[row * n + q] = s * vp + c * vq;
                }
            }
        }
        if (!rotated) {
            NormalizeColumnSigns(n, v);
            return true;
        }
    }
    NormalizeColumnSigns(n, v);
    return false;
}

// ---- 奇异值分解（单边 Jacobi）------------------------------------------------------

bool Svd(int n, const Cx* a_in, Cx* u, double* s, Cx* vh) {
    CheckDim(n);
    Cx a[kMaxDim * kMaxDim];
    Cx v[kMaxDim * kMaxDim];
    for (int i = 0; i < n * n; ++i) {
        a[i] = a_in[i];
        if (!Finite(a[i].re) || !Finite(a[i].im)) {
            return false;
        }
        v[i] = Cx{0.0, 0.0};
    }
    for (int i = 0; i < n; ++i) {
        v[i * n + i] = Cx{1.0, 0.0};
    }
    // 对列对 (i, j) 右乘 2×2 酉矩阵 J = [[c, s·e], [−s·conj(e), c]]（e = γ / |γ|，γ = a_i^† a_j），
    // 使两列正交：a_i ← c·a_i − s·conj(e)·a_j，a_j ← s·e·a_i + c·a_j。
    bool converged = false;
    for (int sweep = 0; sweep < 64 && !converged; ++sweep) {
        converged = true;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double alpha = ColumnNorm2(n, a, i);
                const double beta = ColumnNorm2(n, a, j);
                // 可忽略的列（相对于另一列小于 1e-18 倍，或接近下溢）不再旋转：它们在 ε 以下，
                // 继续正交化只会在次正规数范围内反复舍入；对应的 U 列最后用 Gram-Schmidt 补齐
                if (!(alpha > 1e-280) || !(beta > 1e-280) || std::min(alpha, beta) <= 1e-36 * std::max(alpha, beta)) {
                    continue;
                }
                const Cx gamma = ColumnDot(n, a, i, j);
                const double g = Abs(gamma);
                if (!(g > 1e-15 * (Sqrt(alpha) * Sqrt(beta))) || !(g > 1e-300)) {
                    continue;
                }
                converged = false;
                const double zeta = (beta - alpha) / (2.0 * g);
                const double t = (zeta >= 0.0 ? 1.0 : -1.0) / (std::fabs(zeta) + Sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / Sqrt(1.0 + t * t);
                const double sn = c * t;
                const Cx e = DivReal(gamma, g);
                const Cx se = Scale(e, sn);
                const Cx sec = Conj(se);
                for (Cx* m : {a, v}) {
                    for (int r = 0; r < n; ++r) {
                        const Cx mi = m[r * n + i];
                        const Cx mj = m[r * n + j];
                        m[r * n + i] = Sub(Scale(mi, c), Mul(sec, mj));
                        m[r * n + j] = Add(Mul(se, mi), Scale(mj, c));
                    }
                }
            }
        }
    }
    if (!converged) {
        return false;
    }
    double sigma[kMaxDim];
    int order[kMaxDim];
    for (int j = 0; j < n; ++j) {
        sigma[j] = Sqrt(ColumnNorm2(n, a, j));
        order[j] = j;
    }
    // 降序（稳定的插入排序）
    for (int i = 1; i < n; ++i) {
        const int cur = order[i];
        int k = i;
        while (k > 0 && sigma[order[k - 1]] < sigma[cur]) {
            order[k] = order[k - 1];
            --k;
        }
        order[k] = cur;
    }
    // 相位规范化：V 的每列中模最大的分量为正实数（a 的对应列乘同一相位，A·V = U·Σ 不变）
    for (int j = 0; j < n; ++j) {
        const int k = ArgMaxNorm2(n, v + j, n);
        const Cx top = v[k * n + j];
        const double m = Abs(top);
        if (m > 0.0) {
            const Cx ph = DivReal(Conj(top), m);
            for (int r = 0; r < n; ++r) {
                v[r * n + j] = Mul(v[r * n + j], ph);
                a[r * n + j] = Mul(a[r * n + j], ph);
            }
            v[k * n + j] = Cx{m, 0.0};
        }
    }
    const double smax = sigma[order[0]];
    for (int jj = 0; jj < n; ++jj) {
        const int j = order[jj];
        s[jj] = sigma[j];
        for (int r = 0; r < n; ++r) {
            vh[jj * n + r] = Conj(v[r * n + j]);
        }
        const bool tiny = !(sigma[j] > 1e-8 * smax) || !(sigma[j] > 1e-300);
        if (!tiny) {
            for (int r = 0; r < n; ++r) {
                u[r * n + jj] = DivReal(a[r * n + j], sigma[j]);
            }
            continue;
        }
        // 奇异值相对很小：a_j / σ_j 的正交性只有 ε·σ_max / σ_j。改为对前面各列做两遍 Gram-Schmidt：
        // 先试 a_j 本身（投影后至少保留一半长度才接受），否则取投影后最长的标准基向量
        auto project = [&](Cx* cand) {
            for (int pass = 0; pass < 2; ++pass) {
                for (int kk = 0; kk < jj; ++kk) {
                    Cx d{0.0, 0.0};
                    for (int r = 0; r < n; ++r) {
                        d = Add(d, MulConj(u[r * n + kk], cand[r]));
                    }
                    for (int r = 0; r < n; ++r) {
                        cand[r] = Sub(cand[r], Mul(u[r * n + kk], d));
                    }
                }
            }
            double nrm = 0.0;
            for (int r = 0; r < n; ++r) {
                nrm += Norm2(cand[r]);
            }
            return Sqrt(nrm);
        };
        Cx best[kMaxDim];
        double best_norm = 0.0;
        Cx cand[kMaxDim];
        for (int r = 0; r < n; ++r) {
            cand[r] = a[r * n + j];
        }
        const double own = project(cand);
        if (own > 1e-300 && own >= 0.5 * sigma[j]) {
            for (int r = 0; r < n; ++r) {
                best[r] = cand[r];
            }
            best_norm = own;
        } else {
            for (int e = 0; e < n; ++e) {
                for (int r = 0; r < n; ++r) {
                    cand[r] = Cx{r == e ? 1.0 : 0.0, 0.0};
                }
                const double nrm = project(cand);
                if (nrm > best_norm) {
                    best_norm = nrm;
                    for (int r = 0; r < n; ++r) {
                        best[r] = cand[r];
                    }
                }
            }
        }
        if (!(best_norm > 0.0)) {
            return false;
        }
        for (int r = 0; r < n; ++r) {
            u[r * n + jj] = DivReal(best[r], best_norm);
        }
    }
    return true;
}

// ---- QR（Householder）-------------------------------------------------------------

void Qr(int n, const Cx* a, Cx* q, Cx* r) {
    CheckDim(n);
    for (int i = 0; i < n * n; ++i) {
        r[i] = a[i];
        q[i] = Cx{0.0, 0.0};
    }
    for (int i = 0; i < n; ++i) {
        q[i * n + i] = Cx{1.0, 0.0};
    }
    Cx vec[kMaxDim];
    for (int k = 0; k < n - 1; ++k) {
        double alpha2 = 0.0;
        for (int i = k; i < n; ++i) {
            alpha2 += Norm2(r[i * n + k]);
        }
        const double alpha = Sqrt(alpha2);
        if (alpha == 0.0) {
            continue;
        }
        const Cx x0 = r[k * n + k];
        const double a0 = Abs(x0);
        const Cx phase = a0 > 0.0 ? DivReal(x0, a0) : Cx{1.0, 0.0};
        // v = x + phase·α·e_k：H = I − 2 v v^† / (v^† v) 把 x 变成 −phase·α·e_k（无相消）
        double vnorm2 = 0.0;
        for (int i = k; i < n; ++i) {
            vec[i] = i == k ? Add(x0, Scale(phase, alpha)) : r[i * n + k];
            vnorm2 += Norm2(vec[i]);
        }
        const double f = 2.0 / vnorm2;
        for (int c = k; c < n; ++c) {
            Cx d{0.0, 0.0};
            for (int i = k; i < n; ++i) {
                d = Add(d, MulConj(vec[i], r[i * n + c]));
            }
            const Cx fd = Scale(d, f);
            for (int i = k; i < n; ++i) {
                r[i * n + c] = Sub(r[i * n + c], Mul(vec[i], fd));
            }
        }
        for (int row = 0; row < n; ++row) {
            Cx d{0.0, 0.0};
            for (int i = k; i < n; ++i) {
                d = Add(d, Mul(q[row * n + i], vec[i]));
            }
            const Cx fd = Scale(d, f);
            for (int i = k; i < n; ++i) {
                q[row * n + i] = Sub(q[row * n + i], Mul(fd, Conj(vec[i])));
            }
        }
        for (int i = k + 1; i < n; ++i) {
            r[i * n + k] = Cx{0.0, 0.0};
        }
    }
    // R 的对角元化为非负实数：R 第 k 行乘 conj(φ)，Q 第 k 列乘 φ
    for (int k = 0; k < n; ++k) {
        const Cx d = r[k * n + k];
        const double m = Abs(d);
        if (!(m > 0.0)) {
            continue;
        }
        const Cx ph = DivReal(d, m);
        const Cx phc = Conj(ph);
        for (int c = k; c < n; ++c) {
            r[k * n + c] = Mul(r[k * n + c], phc);
        }
        r[k * n + k] = Cx{m, 0.0};
        for (int row = 0; row < n; ++row) {
            q[row * n + k] = Mul(q[row * n + k], ph);
        }
    }
}

}  // namespace quspect::native::detmath
