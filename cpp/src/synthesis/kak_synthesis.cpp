#include "quspect_native/synthesis/kak_synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace quspect::native::synthesis {

namespace {

// 与 two_qubit.py 的模块常量相同（math.pi 及其按 2 的幂缩放，均为精确运算）。
constexpr double kPi = 3.141592653589793;
constexpr double kPi4 = kPi / 4;
constexpr double kPi2 = kPi / 2;
constexpr double kTwoPi = 2 * kPi;
constexpr double kStabilizerTol = 1e-9;   // _STABILIZER_TOL
constexpr double kExactEulerAtol = 1e-12;  // _EXACT_EULER_ATOL
constexpr double kDefaultAtol = 1e-9;      // DEFAULT_ATOL
constexpr std::size_t kStabilizerEmits = 4;  // _STABILIZER_EMITS
// 只用于比较的量（numpy np.abs 等）与阈值的相对距离小于该值时退回 Python。
constexpr double kDecisionMargin = 1e-9;

M2 Conj(const M2& m) {
    return {Cx{m[0].re, -m[0].im}, Cx{m[1].re, -m[1].im}, Cx{m[2].re, -m[2].im}, Cx{m[3].re, -m[3].im}};
}

M4 Conj(const M4& m) {
    M4 out;
    for (int i = 0; i < 16; ++i) {
        out[i] = Cx{m[i].re, -m[i].im};
    }
    return out;
}

// 材料化的共轭转置（``np.array([s.conj().T])`` 的值）。
M2 ConjT(const M2& m) {
    return {Cx{m[0].re, -m[0].im}, Cx{m[2].re, -m[2].im}, Cx{m[1].re, -m[1].im}, Cx{m[3].re, -m[3].im}};
}

Cx Add(const Cx& a, const Cx& b) {
    return {a.re + b.re, a.im + b.im};
}

Cx Sub(const Cx& a, const Cx& b) {
    return {a.re - b.re, a.im - b.im};
}

bool IsZeroAngle(double angle, double atol) {
    // _is_zero_angle：abs(math.remainder(angle, 2π)) <= atol
    return std::fabs(std::remainder(angle, kTwoPi)) <= atol;
}

double Wrap(double angle) {
    // _wrap：math.remainder 落在 (−π, π]
    const double w = std::remainder(angle, kTwoPi);
    return w <= -kPi ? kPi : w;
}

bool IsSwapFace(const std::array<double, 3>& v) {
    return std::fabs(v[0] - kPi4) <= 1e-9 && std::fabs(v[1] - kPi4) <= 1e-9;
}

// 只用于比较的量 x 与阈值 t：x < t 的结果；两者太接近时退回 Python。
bool ClearlyLess(double x, double t, const char* what) {
    if (!(x == x) || !(t == t)) {
        throw Fallback(what);
    }
    if (x == t) {
        // 两个量算法相同、输入相同时逐位相同；此处不同算法的结果恰好相等，无法判断
        throw Fallback(what);
    }
    const double scale = std::max(std::fabs(x), std::fabs(t));
    if (std::isfinite(scale) && std::fabs(x - t) <= kDecisionMargin * scale) {
        throw Fallback(what);
    }
    return x < t;
}

}  // namespace

// _try_synthesize 中只取决于 u 的局部因子（_factor_local(k1u)、_factor_local(k2u)）。
struct KakSynthesizer::UFactors {
    Pair a;  // k1u
    Pair d;  // k2u
};

KakSynthesizer::KakSynthesizer(const LinalgFunctions& fns, const ArithModes& modes, KakConstants constants)
    : np_(fns, modes), c_(std::move(constants)) {
    if (!c_.normals.empty() && c_.normals.size() < 196) {
        throw std::invalid_argument("need 196 precomputed normal deviates");
    }
}

void KakSynthesizer::SetNormals(std::vector<double> normals) {
    if (normals.size() < 196) {
        throw std::invalid_argument("need 196 precomputed normal deviates");
    }
    c_.normals = std::move(normals);
}

int KakSynthesizer::AddEntangler(EntanglerSpec spec) {
    // Entangler.oriented(True)：_SWAP4 @ matrix @ _SWAP4
    M4 reversed = Mat4(Mat4(c_.swap4, spec.matrix), c_.swap4);
    entanglers_.push_back(std::move(spec));
    entangler_reversed_.push_back(reversed);
    return static_cast<int>(entanglers_.size()) - 1;
}

// ---- 小工具 --------------------------------------------------------------------

M2 KakSynthesizer::Mat2(const M2& a, const M2& b) const {
    M2 out;
    np_.MatMul(2, View(a), View(b), out.data());
    return out;
}

M2 KakSynthesizer::Mat2(const CView& a, const CView& b) const {
    M2 out;
    np_.MatMul(2, a, b, out.data());
    return out;
}

M4 KakSynthesizer::Mat4(const M4& a, const M4& b) const {
    M4 out;
    np_.MatMul(4, View(a), View(b), out.data());
    return out;
}

M4 KakSynthesizer::Kron2(const M2& left, const M2& right) const {
    // (left[:, None, :, None] * right[None, :, None, :]).reshape(4, 4)
    M4 out;
    const MulMode mode = np_.modes().kron;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    out[(i * 2 + j) * 4 + k * 2 + l] = Mul(mode, left[i * 2 + k], right[j * 2 + l]);
                }
            }
        }
    }
    return out;
}

M2 KakSynthesizer::Rx(double theta) const {
    // _rx：c = cos(θ/2), s = sin(θ/2)；[[c, -1j*s], [-1j*s, c]]
    const double c = SeparateCos(theta / 2);
    const double s = SeparateSin(theta / 2);
    const Cx off = PyMulReal(np_.modes(), Cx{-0.0, -1.0}, s);
    return {Cx{c, 0.0}, off, off, Cx{c, 0.0}};
}

M2 KakSynthesizer::Rz(double theta) const {
    // _rz：[[cmath.exp(-0.5j*θ), 0], [0, cmath.exp(0.5j*θ)]]
    const ArithModes& m = np_.modes();
    const Cx d0 = PyExp(m, PyMulReal(m, Cx{-0.0, -0.5}, theta));
    const Cx d1 = PyExp(m, PyMulReal(m, Cx{0.0, 0.5}, theta));
    return {d0, Cx{0.0, 0.0}, Cx{0.0, 0.0}, d1};
}

M2 KakSynthesizer::AxisRot(const M2& pauli, double phi) const {
    // math.cos(phi/2) * _I2 - 1j * math.sin(phi/2) * pauli
    const ArithModes& m = np_.modes();
    const double c = SeparateCos(phi / 2);
    const Cx js = PyMulReal(m, Cx{0.0, 1.0}, SeparateSin(phi / 2));
    M2 out;
    for (int e = 0; e < 4; ++e) {
        const Cx a = Mul(m.scalar_arr, Cx{c, 0.0}, c_.i2[e]);
        const Cx b = Mul(m.scalar_arr, js, pauli[e]);
        out[e] = Sub(a, b);
    }
    return out;
}

double KakSynthesizer::AxisAngle(const M2& w, const M2& pauli) const {
    // x = complex(np.trace(w)); y = 1j * complex(np.trace(pauli @ w))
    // atan2(2.0 * (x * y.conjugate()).real, abs(x) ** 2 - abs(y) ** 2)
    const ArithModes& m = np_.modes();
    const Cx x = Add(w[0], w[3]);
    const M2 pw = Mat2(pauli, w);
    const Cx y = Mul(m.py, Cx{0.0, 1.0}, Add(pw[0], pw[3]));
    const Cx xy = Mul(m.py, x, Cx{y.re, -y.im});
    const double ax = PyAbs(x);
    const double ay = PyAbs(y);
    return PyAtan2(2.0 * xy.re, PyPow(ax, 2.0) - PyPow(ay, 2.0));
}

M4 KakSynthesizer::LayersUnitary(const std::vector<Layer>& layers) const {
    M4 u = Kron2(layers[0].left, layers[0].right);
    for (std::size_t i = 1; i < layers.size(); ++i) {
        u = Mat4(Mat4(Kron2(layers[i].left, layers[i].right), c_.cz4), u);
    }
    return u;
}

std::vector<KakSynthesizer::Layer> KakSynthesizer::Template(int k, const std::array<double, 3>& v) const {
    const double a = v[0];
    const double b = v[1];
    const double c = v[2];
    const M2& eye = c_.i2;
    switch (k) {
        case 0:
            return {Layer{eye, eye}};
        case 1:
            return {Layer{eye, eye}, Layer{eye, eye}};
        case 2:
            return {Layer{eye, eye}, Layer{Rx(2 * a), Rx(-2 * b)}, Layer{eye, eye}};
        default:
            return {Layer{eye, eye}, Layer{c_.h, Rx(2 * a)}, Layer{Mat2(c_.h, Rz(2 * b + kPi2)), Rx(2 * c)},
                    Layer{eye, eye}};
    }
}

std::vector<KakSynthesizer::Layer> KakSynthesizer::SwapTemplate(double c) const {
    const double theta = kPi2 - 2.0 * std::fabs(c);
    return {Layer{c_.i2, c_.h}, Layer{c_.h, Mat2(Rz(theta), c_.h)}, Layer{c_.h, c_.h}, Layer{c_.i2, c_.h}};
}

int KakSynthesizer::CountFromCoords(const std::array<double, 3>& v, double atol) {
    const double a = v[0];
    const double b = v[1];
    const double c = v[2];
    if (a <= atol) {
        return 0;
    }
    if (std::fabs(a - kPi4) <= atol && b <= atol) {
        return 1;
    }
    if (std::fabs(c) <= atol) {
        return 2;
    }
    return 3;
}

// ---- KAK -----------------------------------------------------------------------

void KakSynthesizer::CheckUnitary(const M4& u) const {
    // np.allclose(arr.conj().T @ arr, np.eye(4), atol=1e-8)
    const M4 uc = Conj(u);
    M4 g;
    np_.MatMul(4, ViewT(uc), View(u), g.data());
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const Cx e = Sub(g[i * 4 + j], Cx{i == j ? 1.0 : 0.0, 0.0});
            const double dist = std::hypot(e.re, e.im);
            const double thr = 1e-8 + 1e-5 * (i == j ? 1.0 : 0.0);
            if (!ClearlyLess(dist, thr, "unitarity check")) {
                throw Fallback("not unitary");
            }
        }
    }
}

std::array<double, 16> KakSynthesizer::RealOrthogonalDiagonalize(const M4& m2_in) const {
    M4 m2 = m2_in;
    for (Cx& z : m2) {
        if (std::fabs(z.re) < 1e-14) {
            z.re = 0.0;
        }
        if (std::fabs(z.im) < 1e-14) {
            z.im = 0.0;
        }
    }
    // Python 保留误差最小（严格更小才替换）的尝试。误差用 numpy 的 np.abs 求得，这里的 hypot 可能差
    // 最后几位：彼此太接近、无法判断先后的尝试都留作候选，只要之后有一次尝试明显更好就不影响结果；
    // 最后仍有多个不同的候选时退回 Python。
    struct Attempt {
        std::array<double, 16> p;
        M4 d;
        double err;
    };
    enum class Rel { kLess, kGreater, kEqual, kUnknown };
    auto relation = [](const Attempt& x, const Attempt& y) {
        if (std::memcmp(x.d.data(), y.d.data(), sizeof(M4)) == 0) {
            return Rel::kEqual;  // 输入逐位相同，numpy 的误差也相同
        }
        const double scale = std::max(x.err, y.err);
        if (x.err == y.err || std::fabs(x.err - y.err) <= kDecisionMargin * scale) {
            return Rel::kUnknown;
        }
        return x.err < y.err ? Rel::kLess : Rel::kGreater;
    };
    std::vector<Attempt> best;
    for (int attempt = 0; attempt < 100; ++attempt) {
        double w_re;
        double w_im;
        if (attempt == 0) {
            w_re = 1.0;
            w_im = 0.0;
        } else if (attempt == 1) {
            w_re = 0.0;
            w_im = 1.0;
        } else {
            if (!HasNormals()) {
                throw NeedNormals();
            }
            w_re = c_.normals[static_cast<std::size_t>(2 * (attempt - 2))];
            w_im = c_.normals[static_cast<std::size_t>(2 * (attempt - 2) + 1)];
        }
        std::array<double, 16> mix;
        for (int e = 0; e < 16; ++e) {
            const double x = w_re * m2[e].re;
            const double y = w_im * m2[e].im;
            mix[e] = x + y;
        }
        Attempt cur;
        std::array<double, 4> w;
        if (!np_.Eigh4(mix, &w, &cur.p)) {
            throw Fallback("eigh failed");
        }
        // d = p.T @ m2 @ p；err = max |offdiag(d)|
        M4 t;
        np_.MatMul(4, RView{cur.p.data(), 1, 4}, View(m2), t.data());
        np_.MatMul(4, View(t), RView{cur.p.data(), 4, 1}, cur.d.data());
        cur.err = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (i != j) {
                    const double a = std::hypot(cur.d[i * 4 + j].re, cur.d[i * 4 + j].im);
                    if (!(a == a)) {
                        throw Fallback("non-finite eigenvector residual");
                    }
                    cur.err = std::max(cur.err, a);
                }
            }
        }
        bool all_less = true;
        bool none_less = true;
        for (const Attempt& b : best) {
            const Rel r = relation(cur, b);
            all_less = all_less && r == Rel::kLess;
            none_less = none_less && (r == Rel::kGreater || r == Rel::kEqual);
        }
        if (best.empty() || all_less) {
            best.assign(1, cur);
        } else if (!none_less) {
            best.push_back(cur);
        }
        if (ClearlyLess(cur.err, 1e-13, "eigenvector residual vs 1e-13")) {
            break;
        }
    }
    for (const Attempt& b : best) {
        if (std::memcmp(b.p.data(), best[0].p.data(), sizeof(b.p)) != 0) {
            throw Fallback("eigenvector residuals too close to order");
        }
    }
    std::array<double, 16> p = best[0].p;
    if (np_.DetNegative4(RView{p.data(), 4, 1})) {
        for (int i = 0; i < 4; ++i) {
            p[i * 4] = -p[i * 4];
        }
    }
    return p;
}

void KakSynthesizer::RawKak(const M4& u, Kak* out) const {
    const ArithModes& m = np_.modes();
    const Cx det = np_.Det(4, View(u));
    const double phase0 = PyPhase(det) / 4.0;
    const Cx scale = PyExp(m, PyMulReal(m, Cx{-0.0, -1.0}, phase0));
    M4 u_su;
    for (int e = 0; e < 16; ++e) {
        u_su[e] = Mul(m.arr_scalar, u[e], scale);
    }
    // up = _MAGIC_DAG @ u_su @ _MAGIC；m2 = up.T @ up（syrk）
    M4 t;
    M4 up;
    M4 m2;
    np_.MatMul(4, ViewT(c_.magic_conj), View(u_su), t.data());
    np_.MatMul(4, View(t), View(c_.magic), up.data());
    np_.MatMul(4, ViewT(up), View(up), m2.data());
    std::array<double, 16> p = RealOrthogonalDiagonalize(m2);
    // eig = np.diag(p.T @ m2 @ p)；theta = np.angle(eig) / 2
    M4 d;
    np_.MatMul(4, RView{p.data(), 1, 4}, View(m2), t.data());
    np_.MatMul(4, View(t), RView{p.data(), 4, 1}, d.data());
    std::array<double, 4> theta;
    for (int i = 0; i < 4; ++i) {
        theta[i] = std::atan2(d[i * 5].im, d[i * 5].re) / 2.0;
    }
    // d = np.exp(1j * theta)；o1c = up @ p @ np.diag(d.conj())
    M4 dd{};
    for (int i = 0; i < 4; ++i) {
        const Cx z = Mul(m.scalar_arr, Cx{0.0, 1.0}, Cx{theta[i], 0.0});
        const Cx e = NpExpImag(m, z);
        dd[i * 5] = Cx{e.re, -e.im};
    }
    M4 o1c;
    np_.MatMul(4, View(up), RView{p.data(), 4, 1}, t.data());
    np_.MatMul(4, View(t), View(dd), o1c.data());
    std::array<double, 16> o1;
    for (int e = 0; e < 16; ++e) {
        o1[e] = o1c[e].re;
    }
    // np.linalg.det(o1c.real) < 0（行列式只取决于数值，与视图步长无关）
    if (np_.DetNegative4(RView{o1.data(), 4, 1})) {
        theta[0] += kPi;
        for (int i = 0; i < 4; ++i) {
            o1[i * 4] = -o1[i * 4];
        }
    }
    double g[4];
    np_.MatVecT4(c_.phase_map.data(), theta.data(), g);
    // k1 = _MAGIC @ o1 @ _MAGIC_DAG；k2 = _MAGIC @ p.T @ _MAGIC_DAG
    np_.MatMul(4, View(c_.magic), RView{o1.data(), 4, 1}, t.data());
    np_.MatMul(4, View(t), ViewT(c_.magic_conj), out->k1.data());
    np_.MatMul(4, View(c_.magic), RView{p.data(), 1, 4}, t.data());
    np_.MatMul(4, View(t), ViewT(c_.magic_conj), out->k2.data());
    out->phase = phase0 + g[0] / 4.0;
    out->v = {g[1] / 4.0, g[2] / 4.0, g[3] / 4.0};
}

void KakSynthesizer::Canonicalize(Kak* kak) const {
    std::array<double, 3>& v = kak->v;
    // 1. 每个坐标移到 (−π/4, π/4]
    for (int i = 0; i < 3; ++i) {
        // Python：k 是 int（math.floor 的结果），参与浮点运算时转成 float（0 为 +0.0）
        const double fl = std::floor((v[i] + kPi4) / kPi2);
        if (!(std::fabs(fl) < 1e15)) {
            throw Fallback("Weyl coordinate out of range");
        }
        auto k = -static_cast<long long>(fl);
        double reduced = v[i] + static_cast<double>(k) * kPi2;
        if (reduced <= -kPi4 + 1e-12) {
            k += 1;
            reduced += kPi2;
        }
        v[i] = reduced;
        if (k % 2 != 0) {
            kak->k2 = Mat4(c_.pauli_pairs[i], kak->k2);
        }
        kak->phase += -kPi2 * static_cast<double>(k);
    }
    auto conj = [this, kak](const M4& c_mat) {
        const M4 cc = Conj(c_mat);
        M4 k1;
        np_.MatMul(4, View(kak->k1), ViewT(cc), k1.data());
        kak->k1 = k1;
        kak->k2 = Mat4(c_mat, kak->k2);
    };
    // 2. 按绝对值降序排列
    const std::pair<int, int> order[3] = {{0, 1}, {1, 2}, {0, 1}};
    for (const auto& [i, j] : order) {
        if (std::fabs(v[i]) < std::fabs(v[j])) {
            conj(i == 0 ? c_.swap_ab : c_.swap_bc);
            std::swap(v[i], v[j]);
        }
    }
    // 3. a、b 非负
    if (v[0] < 0 && v[1] < 0) {
        conj(c_.flip_ab);
        v[0] = -v[0];
        v[1] = -v[1];
    } else if (v[0] < 0) {
        conj(c_.flip_ac);
        v[0] = -v[0];
        v[2] = -v[2];
    }
    if (v[1] < 0) {
        conj(c_.flip_bc);
        v[1] = -v[1];
        v[2] = -v[2];
    }
    // 4. a = π/4 面上取 c ≥ 0
    if (v[2] < 0 && std::fabs(v[0] - kPi4) <= 1e-11) {
        conj(c_.flip_ac);
        v[0] = -v[0] + kPi2;
        v[2] = -v[2];
        kak->k2 = Mat4(c_.pauli_pairs[0], kak->k2);
        kak->phase += -kPi2;
    }
}

Kak KakSynthesizer::Kak4(const M4& u) const {
    Kak kak;
    RawKak(u, &kak);
    Canonicalize(&kak);
    return kak;
}

std::array<M2, 2> KakSynthesizer::FactorLocal(const M4& k) const {
    // realigned = k.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
    M4 realigned;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int l = 0; l < 2; ++l) {
                for (int mm = 0; mm < 2; ++mm) {
                    realigned[(i * 2 + l) * 4 + j * 2 + mm] = k[(i * 2 + j) * 4 + l * 2 + mm];
                }
            }
        }
    }
    M4 u;
    M4 vh;
    std::array<double, 4> s;
    if (!np_.Svd4(realigned, &u, &s, &vh)) {
        throw Fallback("svd failed");
    }
    const ArithModes& m = np_.modes();
    const double scale = std::sqrt(s[0]);
    M2 a_mat;
    M2 c_mat;
    for (int e = 0; e < 4; ++e) {
        a_mat[e] = Mul(m.arr_scalar, u[e * 4], Cx{scale, 0.0});
        c_mat[e] = Mul(m.arr_scalar, vh[e], Cx{scale, 0.0});
    }
    const Cx det_a = np_.Det(2, View(a_mat));
    if (PyAbs(det_a) > 1e-300) {
        const Cx root = PySqrt(det_a);
        for (int e = 0; e < 4; ++e) {
            a_mat[e] = NpDiv(m.np_div, a_mat[e], root);
            c_mat[e] = Mul(m.arr_scalar, c_mat[e], root);
        }
    }
    return {a_mat, c_mat};
}

// ---- 单比特欧拉合成 ------------------------------------------------------------

void KakSynthesizer::EulerOps(const M2& matrix, int qubit, int basis, double atol, std::vector<Op>* out) const {
    const ArithModes& md = np_.modes();
    // _zxz_angles
    const Cx det = Sub(Mul(md.np_scalar, matrix[0], matrix[3]), Mul(md.np_scalar, matrix[1], matrix[2]));
    const Cx root = PySqrt(det);
    const Cx m00 = NpDiv(md.np_div, matrix[0], root);
    const Cx m10 = NpDiv(md.np_div, matrix[2], root);
    const double a00 = PyAbs(m00);
    const double a10 = PyAbs(m10);
    const double theta = 2.0 * PyAtan2(a10, a00);
    const double plus = a00 > 1e-12 ? -2.0 * PyPhase(m00) : 0.0;
    const double minus = a10 > 1e-12 ? 2.0 * PyPhase(m10) + kPi : 0.0;
    const double phi = (plus + minus) / 2.0;
    const double lam = (plus - minus) / 2.0;
    if (!std::isfinite(theta) || !std::isfinite(phi) || !std::isfinite(lam)) {
        throw Fallback("non-finite Euler angles");
    }
    const auto q = static_cast<std::uint8_t>(qubit);
    auto rz = [&](double angle) { out->push_back(Op{OpKind::kRz, q, 0, Wrap(angle)}); };
    auto gate = [&](OpKind kind) { out->push_back(Op{kind, q, 0, 0.0}); };
    auto rz_nonzero = [&](double angle) {
        if (!IsZeroAngle(angle, atol)) {
            rz(angle);
        }
    };
    if (basis == 0) {
        // euler_zxz_ops
        if (IsZeroAngle(theta, atol)) {
            rz_nonzero(phi + lam);
            return;
        }
        if (std::fabs(theta - kPi) <= atol) {
            rz_nonzero(lam - phi);
            gate(OpKind::kX);
            return;
        }
        if (std::fabs(theta - kPi2) <= atol) {
            rz_nonzero(lam - kPi2);
            gate(OpKind::kH);
            rz_nonzero(phi - kPi2);
            return;
        }
        rz_nonzero(lam);
        gate(OpKind::kH);
        rz(theta);
        gate(OpKind::kH);
        rz_nonzero(phi);
        return;
    }
    // euler_zsx_ops
    if (IsZeroAngle(theta, atol)) {
        rz_nonzero(phi + lam);
        return;
    }
    if (std::fabs(theta - kPi) <= atol) {
        rz_nonzero(lam - phi);
        gate(OpKind::kX);
        return;
    }
    if (std::fabs(theta - kPi2) <= atol) {
        rz_nonzero(lam);
        gate(OpKind::kSX);
        rz_nonzero(phi);
        return;
    }
    rz_nonzero(lam + kPi2);
    gate(OpKind::kSX);
    rz_nonzero(theta + kPi);
    gate(OpKind::kSX);
    rz_nonzero(phi + kPi2);
}

M2 KakSynthesizer::OpsLayerMatrix(const std::vector<Op>& ops, std::size_t begin, std::size_t end) const {
    M2 m = c_.i2;
    for (std::size_t i = begin; i < end; ++i) {
        const Op& op = ops[i];
        switch (op.kind) {
            case OpKind::kRz:
                m = Mat2(Rz(op.param), m);
                break;
            case OpKind::kH:
                m = Mat2(c_.h, m);
                break;
            case OpKind::kSX:
                m = Mat2(c_.sx, m);
                break;
            default:
                m = Mat2(c_.x, m);
                break;
        }
    }
    return m;
}

double KakSynthesizer::DistanceUpToPhase(const M4& u, const M4& v) const {
    // overlap = np.vdot(v, u)；phase = overlap / abs(overlap)；max |u − phase·v|
    const Cx overlap = np_.Vdot16(v.data(), u.data());
    const double ab = PyAbs(overlap);
    Cx phase{1.0, 0.0};
    if (ab > 1e-300) {
        phase = Cx{overlap.re / ab, overlap.im / ab};
    }
    double dist = 0.0;
    for (int e = 0; e < 16; ++e) {
        const Cx diff = Sub(u[e], Mul(np_.modes().scalar_arr, phase, v[e]));
        const double a = std::hypot(diff.re, diff.im);
        if (!(a == a)) {
            return a;
        }
        dist = std::max(dist, a);
    }
    return dist;
}

bool KakSynthesizer::Verified(double err, double verify_atol) {
    if (!(err == err)) {
        throw Fallback("non-finite verification distance");
    }
    if (verify_atol == 0.0 || std::isinf(verify_atol)) {
        return err <= verify_atol;  // err 为 0 / 有限值的判断与 numpy 一致
    }
    if (std::fabs(err - verify_atol) <= kDecisionMargin * verify_atol) {
        throw Fallback("verification distance near the tolerance");
    }
    return err <= verify_atol;
}

KakSynthesizer::TryResult KakSynthesizer::EmitLayers(const M4& u, const std::vector<Layer>& layers,
                                                     const std::vector<const M4*>& gates,
                                                     const std::vector<bool>& reversed, double atol,
                                                     const SynthOptions& opt) const {
    TryResult result;
    M4 rebuilt{};
    for (std::size_t idx = 0; idx < layers.size(); ++idx) {
        std::vector<Op> layer_ops;
        EulerOps(layers[idx].left, 0, opt.euler, atol, &layer_ops);
        const std::size_t split = layer_ops.size();
        EulerOps(layers[idx].right, 1, opt.euler, atol, &layer_ops);
        const M4 layer = Kron2(OpsLayerMatrix(layer_ops, 0, split), OpsLayerMatrix(layer_ops, split, layer_ops.size()));
        if (idx == 0) {
            rebuilt = layer;
        } else {
            const bool rev = reversed[idx - 1];
            result.ops.push_back(Op{OpKind::kEntangler, static_cast<std::uint8_t>(rev ? 1 : 0),
                                    static_cast<std::uint8_t>(rev ? 0 : 1), 0.0});
            rebuilt = Mat4(Mat4(layer, *gates[idx - 1]), rebuilt);
        }
        result.ops.insert(result.ops.end(), layer_ops.begin(), layer_ops.end());
    }
    result.err = DistanceUpToPhase(u, rebuilt);
    return result;
}

KakSynthesizer::TryResult KakSynthesizer::EmitFixedLayers(const M4& u, const std::vector<Layer>& factored, int k,
                                                          double atol, const SynthOptions& opt) const {
    const EntanglerSpec& ent = entanglers_.at(static_cast<std::size_t>(opt.entangler));
    if (ent.name == "cz") {
        std::vector<const M4*> gates(static_cast<std::size_t>(k), &c_.cz4);
        std::vector<bool> reversed(static_cast<std::size_t>(k), false);
        return EmitLayers(u, factored, gates, reversed, atol, opt);
    }
    const M4& rev_matrix = entangler_reversed_.at(static_cast<std::size_t>(opt.entangler));
    const int choices = ent.symmetric ? 1 : 2;
    int combos = 1;
    for (int i = 0; i < k; ++i) {
        combos *= choices;
    }
    TryResult best;
    bool have_best = false;
    bool best_failed = false;
    for (int combo = 0; combo < combos; ++combo) {
        std::vector<Layer> dressed = factored;
        std::vector<const M4*> gates;
        std::vector<bool> reversed;
        for (int i = 0; i < k; ++i) {
            // itertools.product：第一个位置变化最慢
            const bool rev = choices == 2 && ((combo >> (k - 1 - i)) & 1) != 0;
            const M2& b0 = rev ? ent.before[1] : ent.before[0];
            const M2& b1 = rev ? ent.before[0] : ent.before[1];
            const M2& a0 = rev ? ent.after[1] : ent.after[0];
            const M2& a1 = rev ? ent.after[0] : ent.after[1];
            Layer& cur = dressed[static_cast<std::size_t>(i)];
            cur = Layer{Mat2(b0, cur.left), Mat2(b1, cur.right)};
            Layer& next = dressed[static_cast<std::size_t>(i) + 1];
            next = Layer{Mat2(next.left, a0), Mat2(next.right, a1)};
            gates.push_back(rev ? &rev_matrix : &ent.matrix);
            reversed.push_back(rev);
        }
        TryResult result = EmitLayers(u, dressed, gates, reversed, atol, opt);
        const bool failed = !Verified(result.err, opt.verify_atol);
        if (!have_best || failed < best_failed || (failed == best_failed && result.ops.size() < best.ops.size())) {
            best = std::move(result);
            best_failed = failed;
            have_best = true;
        }
    }
    return best;
}

std::optional<std::vector<Op>> KakSynthesizer::VerifiedEuler(const M2& u, double atol, int basis) const {
    for (double tol : {atol, std::min(atol, 1e-12)}) {
        std::vector<Op> ops;
        EulerOps(u, 0, basis, tol, &ops);
        const M2 v = OpsLayerMatrix(ops, 0, ops.size());
        // overlap = np.vdot(v, u)；phase = overlap / abs(overlap)；max |u − phase·v| <= 1e-9
        Cx overlap{0.0, 0.0};
        for (int e = 0; e < 4; ++e) {
            overlap = Cx{overlap.re + (v[e].re * u[e].re + v[e].im * u[e].im),
                         overlap.im + (v[e].re * u[e].im - v[e].im * u[e].re)};
        }
        const double ab = PyAbs(overlap);
        Cx phase{1.0, 0.0};
        if (ab > 1e-300) {
            phase = Cx{overlap.re / ab, overlap.im / ab};
        }
        double dist = 0.0;
        for (int e = 0; e < 4; ++e) {
            const Cx diff = Sub(u[e], Mul(np_.modes().scalar_arr, phase, v[e]));
            dist = std::max(dist, std::hypot(diff.re, diff.im));
        }
        if (Verified(dist, 1e-9)) {
            return ops;
        }
    }
    return std::nullopt;
}

M4 KakSynthesizer::BlockUnitary(const std::vector<BlockMember>& members, const M4& swap4) const {
    M4 u{};
    for (int i = 0; i < 4; ++i) {
        u[i * 5] = Cx{1.0, 0.0};
    }
    for (const BlockMember& member : members) {
        M4 full;
        switch (member.slot) {
            case 0:
                std::copy(member.matrix, member.matrix + 16, full.begin());
                break;
            case 1:
            case 2: {
                M2 m;
                std::copy(member.matrix, member.matrix + 4, m.begin());
                full = member.slot == 1 ? Kron2(m, c_.i2) : Kron2(c_.i2, m);
                break;
            }
            case 3: {
                constexpr int kRev[4] = {0, 2, 1, 3};
                for (int r = 0; r < 4; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        full[r * 4 + c] = member.matrix[kRev[r] * 4 + kRev[c]];
                    }
                }
                break;
            }
            default: {
                M4 m;
                std::copy(member.matrix, member.matrix + 16, m.begin());
                full = Mat4(Mat4(swap4, m), swap4);
                break;
            }
        }
        u = Mat4(full, u);
    }
    return u;
}

// ---- 稳定子候选 ----------------------------------------------------------------

std::vector<KakSynthesizer::Pair> KakSynthesizer::StabilizerFamily(
    double a, double b, double c, const std::vector<std::pair<int, M2>>& targets, bool fit_cosets) const {
    const double tol = kStabilizerTol;
    const Pair eye{c_.i2, c_.i2};
    const Pair px{c_.x, c_.x};
    const Pair py{c_.y, c_.y};
    const Pair pz{c_.z, c_.z};
    if (std::fabs(a - b) > tol && std::fabs(b - c) > tol && std::fabs(b) > tol) {
        return {eye, px, py, pz};
    }
    std::vector<Pair> cands{eye};
    std::vector<Pair> paulis;
    std::vector<Pair> fitted;
    if (std::fabs(a - b) <= tol && std::fabs(b - c) <= tol) {
        for (const auto& t : targets) {
            cands.push_back(Pair{t.second, t.second});
        }
        paulis = {px, py, pz};
    } else if (std::fabs(b) <= tol && std::fabs(c) <= tol) {
        paulis = {pz};
        std::vector<double> phis[2] = {{0.0}, {0.0}};
        for (const auto& t : targets) {
            phis[t.first].push_back(AxisAngle(t.second, c_.x));
        }
        for (double p0 : phis[0]) {
            for (double p1 : phis[1]) {
                if (p0 != 0.0 || p1 != 0.0) {
                    cands.push_back(Pair{AxisRot(c_.x, p0), AxisRot(c_.x, p1)});
                }
            }
        }
    } else {
        const bool ab = std::fabs(a - b) <= tol;
        const M2& pauli = ab ? c_.z : c_.x;
        for (const auto& t : targets) {
            const M2 r = AxisRot(pauli, AxisAngle(t.second, pauli));
            cands.push_back(Pair{r, r});
        }
        paulis = {ab ? px : pz};
        if (fit_cosets) {
            const M2& p = paulis[0][0];
            for (const auto& t : targets) {
                const M2 r = Mat2(AxisRot(pauli, AxisAngle(Mat2(t.second, p), pauli)), p);
                fitted.push_back(Pair{r, r});
            }
        }
    }
    std::vector<Pair> out = cands;
    for (const Pair& s : cands) {
        for (const Pair& p : paulis) {
            out.push_back(Pair{Mat2(s[0], p[0]), Mat2(s[1], p[1])});
        }
    }
    out.insert(out.end(), fitted.begin(), fitted.end());
    return out;
}

std::vector<KakSynthesizer::Pair> KakSynthesizer::StabilizerCandidates(int k, const std::array<double, 3>& v,
                                                                       const Pair& k1u, const Pair& k1v,
                                                                       const Pair& k2v, const Pair& k2u,
                                                                       bool fit_cosets,
                                                                       bool mirrored_stabilizers) const {
    if (k == 0) {
        return {Pair{c_.i2, c_.i2}};
    }
    const double a = v[0];
    const double b = v[1];
    const double c = v[2];
    // targets：k1u_i^† k1v_i 与 k2u_i k2v_i^†
    const M2 c1u0 = Conj(k1u[0]);
    const M2 c1u1 = Conj(k1u[1]);
    const M2 c2v0 = Conj(k2v[0]);
    const M2 c2v1 = Conj(k2v[1]);
    std::vector<std::pair<int, M2>> targets = {
        {0, Mat2(ViewT(c1u0), View(k1v[0]))},
        {1, Mat2(ViewT(c1u1), View(k1v[1]))},
        {0, Mat2(View(k2u[0]), ViewT(c2v0))},
        {1, Mat2(View(k2u[1]), ViewT(c2v1))},
    };
    std::vector<Pair> out = StabilizerFamily(a, b, c, targets, fit_cosets);
    if (mirrored_stabilizers && c < -kStabilizerTol && std::fabs(b + c) <= kStabilizerTol) {
        std::vector<std::pair<int, M2>> mirrored;
        for (const auto& t : targets) {
            mirrored.emplace_back(t.first, t.first == 1 ? Mat2(Mat2(c_.z, t.second), c_.z) : t.second);
        }
        const std::vector<Pair> fam = StabilizerFamily(a, b, -c, mirrored, fit_cosets);
        for (std::size_t i = 1; i < fam.size(); ++i) {
            out.push_back(Pair{fam[i][0], Mat2(Mat2(c_.z, fam[i][1]), c_.z)});
        }
    }
    return out;
}

std::vector<std::pair<KakSynthesizer::Pair, KakSynthesizer::Pair>> KakSynthesizer::SwapFaceGauges(
    const Pair& k1u, const Pair& k1v, const Pair& k2v, const Pair& k2u) const {
    Pair lw;
    Pair rw;
    for (int i = 0; i < 2; ++i) {
        const M2 cu = Conj(k1u[static_cast<std::size_t>(i)]);
        lw[static_cast<std::size_t>(i)] = Mat2(ViewT(cu), View(k1v[static_cast<std::size_t>(i)]));
        const M2 cd = Conj(k2u[static_cast<std::size_t>(i)]);
        rw[static_cast<std::size_t>(i)] = Mat2(View(k2v[static_cast<std::size_t>(i)]), ViewT(cd));
    }
    // Python 的 {0.0, x, y}：按 == 去重、保留先插入的值，再升序排列
    auto sorted_set = [](double x, double y) {
        std::vector<double> vals{0.0};
        for (double t : {x, y}) {
            if (!(t == t)) {
                throw Fallback("non-finite gauge angle");
            }
            if (std::find(vals.begin(), vals.end(), t) == vals.end()) {
                vals.push_back(t);
            }
        }
        std::sort(vals.begin(), vals.end());
        return vals;
    };
    std::vector<std::pair<Pair, Pair>> out;
    for (int pi = 0; pi < 2; ++pi) {
        const M2& p = pi == 0 ? c_.i2 : c_.x;
        const std::vector<double> phis =
            sorted_set(-AxisAngle(Mat2(lw[1], p), c_.z), AxisAngle(Mat2(p, rw[0]), c_.z));
        const std::vector<double> psis =
            sorted_set(-AxisAngle(Mat2(lw[0], p), c_.z), AxisAngle(Mat2(p, rw[1]), c_.z));
        for (double phi : phis) {
            for (double psi : psis) {
                if (pi == 0 && phi == 0.0 && psi == 0.0) {
                    continue;
                }
                Pair left{Mat2(AxisRot(c_.z, -psi), p), Mat2(AxisRot(c_.z, -phi), p)};
                Pair right{Mat2(p, AxisRot(c_.z, phi)), Mat2(p, AxisRot(c_.z, psi))};
                out.emplace_back(left, right);
            }
        }
    }
    return out;
}

// ---- _try_synthesize ------------------------------------------------------------

KakSynthesizer::TryResult KakSynthesizer::TrySynthesize(const M4& u, const Kak& kak_u, const UFactors& uf, int k,
                                                        double atol, const std::array<double, 3>* template_coords,
                                                        const SynthOptions& opt, bool swap_face) const {
    const std::array<double, 3>& coords = template_coords != nullptr ? *template_coords : kak_u.v;
    const std::vector<Layer> layers = swap_face ? SwapTemplate(coords[2]) : Template(k, coords);
    const Kak kv = Kak4(LayersUnitary(layers));
    const Pair& a = uf.a;
    const Pair b = FactorLocal(kv.k1);
    const Pair& d = uf.d;
    const Pair c = FactorLocal(kv.k2);
    const Pair b_conj{Conj(b[0]), Conj(b[1])};  // b_dag[i] = ViewT(b_conj[i])
    const Pair c_conj{Conj(c[0]), Conj(c[1])};

    std::vector<Pair> cands =
        StabilizerCandidates(k, kv.v, a, b, c, d, /*fit_cosets=*/swap_face, opt.mirrored_stabilizers);
    std::vector<std::pair<Pair, Pair>> gauges;
    if (swap_face) {
        gauges = SwapFaceGauges(a, b, c, d);
    }
    const std::size_t nc = cands.size();
    const std::size_t n = nc + gauges.size();
    auto left_stab = [&](std::size_t j, int i) -> const M2& {
        return j < nc ? cands[j][static_cast<std::size_t>(i)] : gauges[j - nc].first[static_cast<std::size_t>(i)];
    };

    if (n == 1) {
        // fixed(cands[0])（只有 k = 0）
        const Pair& stab = cands[0];
        Pair lefts;
        Pair rights;
        for (int i = 0; i < 2; ++i) {
            const auto si = static_cast<std::size_t>(i);
            lefts[si] = Mat2(View(Mat2(a[si], stab[si])), ViewT(b_conj[si]));
            const M2 sc = Conj(stab[si]);
            rights[si] = Mat2(View(Mat2(ViewT(c_conj[si]), ViewT(sc))), View(d[si]));
        }
        std::vector<Layer> out = layers;
        out[0] = Layer{Mat2(layers[0].left, rights[0]), Mat2(layers[0].right, rights[1])};
        Layer& last = out.back();
        last = Layer{Mat2(lefts[0], last.left), Mat2(lefts[1], last.right)};
        return EmitFixedLayers(u, out, k, atol, opt);
    }

    // 分批计算每个候选的首末层（与 numpy 的逐项 matmul 相同），按欧拉长度打分
    std::vector<M2> firsts[2];
    std::vector<M2> lasts[2];
    for (int i = 0; i < 2; ++i) {
        const auto si = static_cast<std::size_t>(i);
        firsts[i].resize(n);
        lasts[i].resize(n);
        const M2& first_layer = si == 0 ? layers[0].left : layers[0].right;
        const M2& last_layer = si == 0 ? layers.back().left : layers.back().right;
        for (std::size_t j = 0; j < n; ++j) {
            const M2& s = left_stab(j, i);
            const M2 left = Mat2(View(Mat2(a[si], s)), ViewT(b_conj[si]));
            M2 right;
            if (!gauges.empty()) {
                // right_stack = np.array([...])：候选部分是 s.conj().T 的值，规范对部分是右因子
                const M2 r = j < nc ? ConjT(cands[j][si]) : gauges[j - nc].second[si];
                right = Mat2(View(Mat2(ViewT(c_conj[si]), View(r))), View(d[si]));
            } else {
                // right_stack = stack.conj().transpose(0, 2, 1)
                const M2 sc = Conj(s);
                right = Mat2(View(Mat2(ViewT(c_conj[si]), ViewT(sc))), View(d[si]));
            }
            firsts[i][j] = Mat2(first_layer, right);
            lasts[i][j] = Mat2(left, last_layer);
        }
    }
    std::vector<int> scores(n, 0);
    std::vector<Op> scratch;
    for (const std::vector<M2>* group : {&firsts[0], &firsts[1], &lasts[0], &lasts[1]}) {
        for (std::size_t j = 0; j < n; ++j) {
            scratch.clear();
            EulerOps((*group)[j], 0, opt.euler, atol, &scratch);
            scores[j] += static_cast<int>(scratch.size());
        }
    }
    auto factored_of = [&](std::size_t j) {
        std::vector<Layer> out;
        out.push_back(Layer{firsts[0][j], firsts[1][j]});
        for (std::size_t l = 1; l + 1 < layers.size(); ++l) {
            out.push_back(layers[l]);
        }
        out.push_back(Layer{lasts[0][j], lasts[1][j]});
        return out;
    };
    std::vector<std::size_t> better;
    for (std::size_t j = 1; j < n; ++j) {
        if (gauges.empty() ? scores[j] < scores[0] : scores[j] <= scores[0]) {
            better.push_back(j);
        }
    }
    std::stable_sort(better.begin(), better.end(),
                     [&](std::size_t x, std::size_t y) { return scores[x] < scores[y]; });
    if (gauges.empty() && better.size() > kStabilizerEmits) {
        better.resize(kStabilizerEmits);
    }
    const EntanglerSpec& ent = entanglers_.at(static_cast<std::size_t>(opt.entangler));
    if (ent.name == "cz") {
        std::optional<TryResult> fallback;
        for (std::size_t j : better) {
            TryResult result = EmitFixedLayers(u, factored_of(j), k, atol, opt);
            if (Verified(result.err, opt.verify_atol)) {
                return result;
            }
            if (!fallback) {
                fallback = std::move(result);
            }
        }
        TryResult result = EmitFixedLayers(u, factored_of(0), k, atol, opt);
        if (Verified(result.err, opt.verify_atol) || !fallback) {
            return result;
        }
        return std::move(*fallback);
    }
    TryResult best;
    bool have_best = false;
    bool best_failed = false;
    std::vector<std::size_t> order{0};
    order.insert(order.end(), better.begin(), better.end());
    for (std::size_t j : order) {
        TryResult result = EmitFixedLayers(u, factored_of(j), k, atol, opt);
        const bool failed = !Verified(result.err, opt.verify_atol);
        if (!have_best || failed < best_failed || (failed == best_failed && result.ops.size() < best.ops.size())) {
            best = std::move(result);
            best_failed = failed;
            have_best = true;
        }
    }
    return best;
}

// ---- 入口 --------------------------------------------------------------------------

std::vector<Op> KakSynthesizer::SynthesizeExact(const M4& u, const SynthOptions& opt, bool* uses_flag) const {
    return SynthesizeExact(u, Kak4(u), opt, uses_flag);
}

std::vector<Op> KakSynthesizer::SynthesizeExact(const M4& u, const Kak& kak, const SynthOptions& opt,
                                                bool* uses_flag) const {
    const int k_min = CountFromCoords(kak.v, opt.atol);
    const bool swap_face = k_min == 3 && IsSwapFace(kak.v);
    if (uses_flag != nullptr) {
        *uses_flag = swap_face;
    }
    const UFactors uf{FactorLocal(kak.k1), FactorLocal(kak.k2)};
    std::vector<double> euler_tols{opt.atol};
    if (opt.atol > kExactEulerAtol) {
        euler_tols.push_back(kExactEulerAtol);
    }
    if (opt.swap_face_template && swap_face) {
        for (double tol : euler_tols) {
            TryResult r = TrySynthesize(u, kak, uf, 3, tol, nullptr, opt, true);
            if (Verified(r.err, opt.verify_atol)) {
                return std::move(r.ops);
            }
        }
    }
    for (int k = k_min; k < 4; ++k) {
        std::array<double, 3> mirrored = kak.v;
        mirrored[2] = -mirrored[2];
        const std::array<double, 3>* candidates[2] = {nullptr, &mirrored};
        const int num_candidates = k == 3 ? 2 : 1;
        for (int ci = 0; ci < num_candidates; ++ci) {
            for (double tol : euler_tols) {
                TryResult r = TrySynthesize(u, kak, uf, k, tol, candidates[ci], opt, false);
                if (Verified(r.err, opt.verify_atol)) {
                    return std::move(r.ops);
                }
            }
        }
    }
    // Python 端抛出 RuntimeError（附最优误差）
    throw Fallback("two-qubit synthesis failed");
}

std::optional<std::vector<Op>> KakSynthesizer::SynthesizeSwapFace(const M4& u, const SynthOptions& opt) const {
    const Kak kak = Kak4(u);
    if (!IsSwapFace(kak.v) || CountFromCoords(kak.v, kDefaultAtol) < 3) {
        return std::nullopt;
    }
    const UFactors uf{FactorLocal(kak.k1), FactorLocal(kak.k2)};
    for (double tol : {kDefaultAtol, kExactEulerAtol}) {
        TryResult r = TrySynthesize(u, kak, uf, 3, tol, nullptr, opt, true);
        if (Verified(r.err, opt.verify_atol)) {
            return std::move(r.ops);
        }
    }
    return std::nullopt;
}

bool KakSynthesizer::ApproximationImpossible(const Kak& kak, double max_infidelity, double atol,
                                             int max_entanglers) const {
    const int k_exact = CountFromCoords(kak.v, atol);
    const int k_end = std::min(k_exact, max_entanglers + 1);
    const double budget = max_infidelity * (1 + 1e-6);
    for (int k = 0; k < k_end; ++k) {
        // _projected_infidelity
        const double a = kak.v[0];
        const double b = kak.v[1];
        const double c = kak.v[2];
        double delta[3];
        if (k == 0) {
            delta[0] = a;
            delta[1] = b;
            delta[2] = c;
        } else if (k == 1) {
            delta[0] = a - kPi4;
            delta[1] = b;
            delta[2] = c;
        } else {
            delta[0] = 0.0;
            delta[1] = 0.0;
            delta[2] = c;
        }
        double phases[4];
        for (int r = 0; r < 4; ++r) {
            phases[r] = c_.can_phases[static_cast<std::size_t>(r * 3)] * delta[0]
                        + c_.can_phases[static_cast<std::size_t>(r * 3 + 1)] * delta[1]
                        + c_.can_phases[static_cast<std::size_t>(r * 3 + 2)] * delta[2];
        }
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const double s = std::sin(0.5 * (phases[i] - phases[j]));
                sum += s * s;
            }
        }
        const double predicted = 4.0 * (2.0 * sum / 16.0) / 5.0;
        // predicted > budget 时 Python 跳过该 k；否则它会尝试合成（交给 Python）
        if (!(predicted > budget)) {
            return false;
        }
        if (predicted - budget <= kDecisionMargin * budget + 1e-300) {
            throw Fallback("predicted infidelity near the budget");
        }
    }
    return true;
}

}  // namespace quspect::native::synthesis
