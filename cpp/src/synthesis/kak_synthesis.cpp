#include "quspect_native/synthesis/kak_synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::synthesis {

namespace dm = quspect::native::detmath;

namespace {

// 与 two_qubit.py 的模块常量相同（math.pi 及其按 2 的幂缩放，均为精确运算）。
constexpr double kPi = 3.141592653589793;
constexpr double kPi4 = kPi / 4;
constexpr double kPi2 = kPi / 2;
constexpr double kTwoPi = 2 * kPi;
constexpr double kStabilizerTol = 1e-9;      // _STABILIZER_TOL
constexpr double kExactEulerAtol = 1e-12;    // _EXACT_EULER_ATOL
constexpr double kDefaultAtol = 1e-9;        // DEFAULT_ATOL
constexpr std::size_t kStabilizerEmits = 4;  // _STABILIZER_EMITS
// 同时对角化的残差（max |offdiag(P^T M P)|）小于该值即接受（与 Python 的重试阈值相同）。
constexpr double kDiagonalizeTol = 1e-13;
// Weyl 坐标与室边界（a = b、b = |c|、b = 0、c = 0、a = π/4）的距离不超过该值时，SynthesizeExact
// 再用第二种特征向量基合成一次。
constexpr double kBoundaryTol = 1e-6;

M2 Conj(const M2& m) {
    return {dm::Conj(m[0]), dm::Conj(m[1]), dm::Conj(m[2]), dm::Conj(m[3])};
}

M2 Mat2(const M2& a, const M2& b) {
    return dm::MatMul(a, b);
}

M4 Mat4(const M4& a, const M4& b) {
    return dm::MatMul(a, b);
}

// a^† b（2×2）。
M2 Mat2AdjA(const M2& a, const M2& b) {
    M2 out;
    dm::MatMulAdjA(2, a.data(), b.data(), out.data());
    return out;
}

// a b^†（2×2）。
M2 Mat2AdjB(const M2& a, const M2& b) {
    M2 out;
    dm::MatMulAdjB(2, a.data(), b.data(), out.data());
    return out;
}

bool IsZeroAngle(double angle, double atol) {
    // _is_zero_angle：abs(math.remainder(angle, 2π)) <= atol（remainder 是精确运算）
    return std::fabs(dm::Remainder(angle, kTwoPi)) <= atol;
}

double Wrap(double angle) {
    // _wrap：math.remainder 落在 (−π, π]
    const double w = dm::Remainder(angle, kTwoPi);
    return w <= -kPi ? kPi : w;
}

bool IsSwapFace(const std::array<double, 3>& v) {
    return std::fabs(v[0] - kPi4) <= 1e-9 && std::fabs(v[1] - kPi4) <= 1e-9;
}

// ``err <= verify_atol``（NaN 视为未通过）。
bool Verified(double err, double verify_atol) {
    return err <= verify_atol;
}

// max |offdiag(P^T M P)|（P 实正交，M 复对称）。
double OffDiagonalResidual(const std::array<double, 16>& p, const M4& m) {
    M4 t;
    M4 d;
    dm::MatMulRTransC(4, p.data(), m.data(), t.data());
    dm::MatMulCR(4, t.data(), p.data(), d.data());
    double err = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                const double a = dm::Abs(d[i * 4 + j]);
                if (!(a == a)) {
                    return a;
                }
                err = std::max(err, a);
            }
        }
    }
    return err;
}

// 两个 KAK 结果逐位相同（相位、坐标与局部因子）。
bool SameKak(const Kak& a, const Kak& b) {
    auto same = [](double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; };
    if (!same(a.phase, b.phase)) {
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (!same(a.v[static_cast<std::size_t>(i)], b.v[static_cast<std::size_t>(i)])) {
            return false;
        }
    }
    for (std::size_t e = 0; e < 16; ++e) {
        if (!same(a.k1[e].re, b.k1[e].re) || !same(a.k1[e].im, b.k1[e].im) || !same(a.k2[e].re, b.k2[e].re)
            || !same(a.k2[e].im, b.k2[e].im)) {
            return false;
        }
    }
    return true;
}

std::string FailureMessage(double best) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3e", best);
    return std::string("Two-qubit synthesis failed to reproduce the unitary (best error ") + buf + ").";
}

}  // namespace

// ---- 常量 ----------------------------------------------------------------------

KakConstants KakConstants::Make() {
    KakConstants c;
    const double s = 1.0 / dm::Sqrt(2.0);  // _SQ2 = 1.0 / math.sqrt(2.0)
    const Cx zero{0.0, 0.0};
    const Cx one{1.0, 0.0};
    const Cx neg{-1.0, 0.0};
    const Cx im{0.0, 1.0};
    const Cx neg_im{0.0, -1.0};
    // _MAGIC = [[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]] · _SQ2
    const Cx magic[16] = {one, zero, zero, im, zero, im, one, zero, zero, im, neg, zero, one, zero, zero, neg_im};
    for (int e = 0; e < 16; ++e) {
        c.magic[e] = dm::Scale(magic[e], s);
    }
    c.magic_dag = dm::Adjoint(c.magic);
    c.i2 = {one, zero, zero, one};
    c.x = {zero, one, one, zero};
    c.y = {zero, neg_im, im, zero};
    c.z = {one, zero, zero, neg};
    c.h = {Cx{s, 0.0}, Cx{s, 0.0}, Cx{s, 0.0}, Cx{-s, 0.0}};
    c.sx = {Cx{0.5, 0.5}, Cx{0.5, -0.5}, Cx{0.5, -0.5}, Cx{0.5, 0.5}};
    c.cz4 = M4{};
    c.swap4 = M4{};
    constexpr int kSwap[4] = {0, 2, 1, 3};
    for (int i = 0; i < 4; ++i) {
        c.cz4[i * 5] = i == 3 ? neg : one;
        c.swap4[i * 4 + kSwap[i]] = one;
    }
    c.pauli_pairs = {KakSynthesizer::Kron2(c.x, c.x), KakSynthesizer::Kron2(c.y, c.y),
                     KakSynthesizer::Kron2(c.z, c.z)};
    const M2 sgate{one, zero, zero, im};
    const M2 rx90{Cx{s, 0.0}, Cx{0.0, -s}, Cx{0.0, -s}, Cx{s, 0.0}};
    c.swap_ab = KakSynthesizer::Kron2(sgate, sgate);
    c.swap_bc = KakSynthesizer::Kron2(rx90, rx90);
    c.flip_ab = KakSynthesizer::Kron2(c.z, c.i2);
    c.flip_ac = KakSynthesizer::Kron2(c.y, c.i2);
    c.flip_bc = KakSynthesizer::Kron2(c.x, c.i2);
    c.can_phases = {1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0};
    for (int r = 0; r < 4; ++r) {
        c.phase_map[static_cast<std::size_t>(r * 4)] = 1.0;
        for (int j = 0; j < 3; ++j) {
            c.phase_map[static_cast<std::size_t>(r * 4 + 1 + j)] = c.can_phases[static_cast<std::size_t>(r * 3 + j)];
        }
    }
    return c;
}

// _try_synthesize 中只取决于 u 的局部因子（_factor_local(k1u)、_factor_local(k2u)）。
struct KakSynthesizer::UFactors {
    Pair a;  // k1u
    Pair d;  // k2u
};

KakSynthesizer::KakSynthesizer() : c_(KakConstants::Make()) {}

int KakSynthesizer::AddEntangler(EntanglerSpec spec) {
    // Entangler.oriented(True)：_SWAP4 @ matrix @ _SWAP4
    M4 reversed = Mat4(Mat4(c_.swap4, spec.matrix), c_.swap4);
    entanglers_.push_back(std::move(spec));
    entangler_reversed_.push_back(reversed);
    return static_cast<int>(entanglers_.size()) - 1;
}

// ---- 小工具 --------------------------------------------------------------------

M4 KakSynthesizer::Kron2(const M2& left, const M2& right) {
    // (left[:, None, :, None] * right[None, :, None, :]).reshape(4, 4)
    M4 out;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    out[(i * 2 + j) * 4 + k * 2 + l] = dm::Mul(left[i * 2 + k], right[j * 2 + l]);
                }
            }
        }
    }
    return out;
}

M2 KakSynthesizer::Rx(double theta) {
    // _rx：c = cos(θ/2), s = sin(θ/2)；[[c, -1j*s], [-1j*s, c]]
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    const Cx off{0.0, -s};
    return {Cx{c, 0.0}, off, off, Cx{c, 0.0}};
}

M2 KakSynthesizer::Rz(double theta) {
    // _rz：diag(e^{−iθ/2}, e^{iθ/2})
    return {dm::ExpI(-0.5 * theta), Cx{0.0, 0.0}, Cx{0.0, 0.0}, dm::ExpI(0.5 * theta)};
}

M2 KakSynthesizer::AxisRot(const M2& pauli, double phi) const {
    // cos(φ/2)·I − i·sin(φ/2)·P
    const double c = dm::Cos(phi / 2);
    const Cx js{0.0, dm::Sin(phi / 2)};
    M2 out;
    for (int e = 0; e < 4; ++e) {
        out[e] = dm::Sub(dm::Scale(c_.i2[e], c), dm::Mul(js, pauli[e]));
    }
    return out;
}

double KakSynthesizer::AxisAngle(const M2& w, const M2& pauli) {
    // x = tr(w)；y = i·tr(P w)；atan2(2·Re(x·conj(y)), |x|² − |y|²)
    const Cx x = dm::Add(w[0], w[3]);
    const M2 pw = Mat2(pauli, w);
    const Cx y = dm::Mul(Cx{0.0, 1.0}, dm::Add(pw[0], pw[3]));
    const Cx xy = dm::Mul(x, dm::Conj(y));
    const double ax = dm::Abs(x);
    const double ay = dm::Abs(y);
    return dm::Atan2(2.0 * xy.re, ax * ax - ay * ay);
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
    // np.allclose(u^† u, I, atol=1e-8)：|g_ij − δ_ij| <= 1e-8 + 1e-5·δ_ij（NaN 不通过）
    M4 g;
    dm::MatMulAdjA(4, u.data(), u.data(), g.data());
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const double dist = dm::Abs(dm::Sub(g[i * 4 + j], Cx{i == j ? 1.0 : 0.0, 0.0}));
            const double thr = 1e-8 + 1e-5 * (i == j ? 1.0 : 0.0);
            if (!(dist <= thr)) {
                throw NotUnitary();
            }
        }
    }
}

std::array<double, 16> KakSynthesizer::RealOrthogonalDiagonalize(const M4& m2_in, int variant) const {
    // m2 = U_B^T U_B 是复对称酉矩阵，Re m2 与 Im m2 是可交换的实对称矩阵：它们的同时特征向量
    // 构成所求的 P ∈ SO(4)。
    M4 m2 = m2_in;
    std::array<double, 16> re;
    std::array<double, 16> imag;
    for (int e = 0; e < 16; ++e) {
        Cx& z = m2[e];
        if (std::fabs(z.re) < 1e-14) {
            z.re = 0.0;
        }
        if (std::fabs(z.im) < 1e-14) {
            z.im = 0.0;
        }
        re[e] = z.re;
        imag[e] = z.im;
    }
    std::array<double, 16> best{};
    double best_err = std::numeric_limits<double>::infinity();
    bool have = false;
    std::array<double, 16> p;
    std::array<double, 4> w;
    auto consider = [&](const std::array<double, 16>& cand) {
        const double err = OffDiagonalResidual(cand, m2);
        if (err < best_err) {
            best = cand;
            best_err = err;
            have = true;
        }
    };
    if (variant == 1) {
        // Re M、Im M 的特征分解（与 Python 的前两次尝试相同）
        for (const std::array<double, 16>* mat : {&re, &imag}) {
            if (!(best_err < kDiagonalizeTol) && dm::EighSym(4, mat->data(), w.data(), p.data())) {
                consider(p);
            }
        }
    }
    if (!(best_err < kDiagonalizeTol)) {
        dm::JointDiagSym(4, re.data(), imag.data(), p.data());
        consider(p);
    }
    // 都不够精确时（输入只是近似酉矩阵等）：对 cos t·Re + sin t·Im 做特征分解，取残差最小者
    for (int attempt = 0; attempt < 32 && !(best_err < kDiagonalizeTol); ++attempt) {
        const double t = 0.5 + 0.7 * attempt;
        const double wr = dm::Cos(t);
        const double wi = dm::Sin(t);
        std::array<double, 16> mix;
        for (int e = 0; e < 16; ++e) {
            mix[e] = wr * re[e] + wi * imag[e];
        }
        if (dm::EighSym(4, mix.data(), w.data(), p.data())) {
            consider(p);
        }
    }
    if (!have) {
        throw SynthesisFailed("KAK: eigenvector computation failed");
    }
    if (dm::DetReal(4, best.data()) < 0) {
        for (int i = 0; i < 4; ++i) {
            best[i * 4] = -best[i * 4];
        }
    }
    return best;
}

void KakSynthesizer::RawKak(const M4& u, int variant, Kak* out) const {
    const Cx det = dm::Det(u);
    const double phase0 = dm::Arg(det) / 4.0;
    const Cx scale = dm::ExpI(-phase0);
    M4 u_su;
    for (int e = 0; e < 16; ++e) {
        u_su[e] = dm::Mul(u[e], scale);
    }
    // up = _MAGIC_DAG @ u_su @ _MAGIC；m2 = up.T @ up
    const M4 up = Mat4(Mat4(c_.magic_dag, u_su), c_.magic);
    M4 m2;
    dm::MatMulTransA(4, up.data(), up.data(), m2.data());
    std::array<double, 16> p = RealOrthogonalDiagonalize(m2, variant);
    // eig = diag(p.T @ m2 @ p)；theta = angle(eig) / 2
    M4 t;
    M4 d;
    dm::MatMulRTransC(4, p.data(), m2.data(), t.data());
    dm::MatMulCR(4, t.data(), p.data(), d.data());
    std::array<double, 4> theta;
    for (int i = 0; i < 4; ++i) {
        theta[i] = dm::Arg(d[i * 5]) / 2.0;
    }
    // o1c = up @ p @ diag(exp(−iθ))；o1 = Re(o1c)
    dm::MatMulCR(4, up.data(), p.data(), t.data());
    std::array<double, 16> o1;
    for (int j = 0; j < 4; ++j) {
        const Cx dd = dm::ExpI(-theta[j]);
        for (int i = 0; i < 4; ++i) {
            o1[i * 4 + j] = dm::Mul(t[i * 4 + j], dd).re;
        }
    }
    if (dm::DetReal(4, o1.data()) < 0) {
        theta[0] += kPi;
        for (int i = 0; i < 4; ++i) {
            o1[i * 4] = -o1[i * 4];
        }
    }
    // g = _PHASE_MAP.T @ theta
    double g[4];
    for (int j = 0; j < 4; ++j) {
        double s = 0.0;
        for (int i = 0; i < 4; ++i) {
            s += c_.phase_map[static_cast<std::size_t>(i * 4 + j)] * theta[i];
        }
        g[j] = s;
    }
    // k1 = _MAGIC @ o1 @ _MAGIC_DAG；k2 = _MAGIC @ p.T @ _MAGIC_DAG
    dm::MatMulCR(4, c_.magic.data(), o1.data(), t.data());
    out->k1 = Mat4(t, c_.magic_dag);
    dm::MatMulCRTrans(4, c_.magic.data(), p.data(), t.data());
    out->k2 = Mat4(t, c_.magic_dag);
    out->phase = phase0 + g[0] / 4.0;
    out->v = {g[1] / 4.0, g[2] / 4.0, g[3] / 4.0};
}

void KakSynthesizer::Canonicalize(Kak* kak) const {
    std::array<double, 3>& v = kak->v;
    // 1. 每个坐标移到 (−π/4, π/4]
    for (int i = 0; i < 3; ++i) {
        const double fl = std::floor((v[i] + kPi4) / kPi2);
        if (!(std::fabs(fl) < 1e15)) {
            throw SynthesisFailed("Weyl coordinate out of range");
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
        // Can(v_old) = C^† Can(v_new) C：k1 ← k1 C^†，k2 ← C k2
        M4 k1;
        dm::MatMulAdjB(4, kak->k1.data(), c_mat.data(), k1.data());
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

Kak KakSynthesizer::Kak4(const M4& u, int variant) const {
    Kak kak;
    RawKak(u, variant, &kak);
    Canonicalize(&kak);
    return kak;
}

std::array<M2, 2> KakSynthesizer::FactorLocal(const M4& k) const {
    // realigned = k.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4) = vec(A) vec(C)^T（k = A ⊗ C）
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
    if (!dm::Svd(4, realigned.data(), u.data(), s.data(), vh.data())) {
        throw SynthesisFailed("KAK: singular value decomposition failed");
    }
    const double scale = dm::Sqrt(s[0]);
    M2 a_mat;
    M2 c_mat;
    for (int e = 0; e < 4; ++e) {
        a_mat[e] = dm::Scale(u[e * 4], scale);
        c_mat[e] = dm::Scale(vh[e], scale);
    }
    const Cx det_a = dm::Det(a_mat);
    if (dm::Abs(det_a) > 1e-300) {
        const Cx root = dm::Sqrt(det_a);
        for (int e = 0; e < 4; ++e) {
            a_mat[e] = dm::Div(a_mat[e], root);
            c_mat[e] = dm::Mul(c_mat[e], root);
        }
    }
    return {a_mat, c_mat};
}

// ---- 单比特欧拉合成 ------------------------------------------------------------

void KakSynthesizer::EulerOps(const M2& matrix, int qubit, int basis, double atol, std::vector<Op>* out) const {
    // _zxz_angles
    const Cx det = dm::Sub(dm::Mul(matrix[0], matrix[3]), dm::Mul(matrix[1], matrix[2]));
    const Cx root = dm::Sqrt(det);
    const Cx m00 = dm::Div(matrix[0], root);
    const Cx m10 = dm::Div(matrix[2], root);
    const double a00 = dm::Abs(m00);
    const double a10 = dm::Abs(m10);
    const double theta = 2.0 * dm::Atan2(a10, a00);
    const double plus = a00 > 1e-12 ? -2.0 * dm::Arg(m00) : 0.0;
    const double minus = a10 > 1e-12 ? 2.0 * dm::Arg(m10) + kPi : 0.0;
    const double phi = (plus + minus) / 2.0;
    const double lam = (plus - minus) / 2.0;
    if (!std::isfinite(theta) || !std::isfinite(phi) || !std::isfinite(lam)) {
        throw SynthesisFailed("non-finite Euler angles");
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

std::optional<std::vector<Op>> KakSynthesizer::VerifiedEuler(const M2& u, double atol, int basis) const {
    for (double tol : {atol, std::min(atol, kExactEulerAtol)}) {
        std::vector<Op> ops;
        EulerOps(u, 0, basis, tol, &ops);
        const M2 v = OpsLayerMatrix(ops, 0, ops.size());
        // overlap = np.vdot(v, u)；phase = overlap / abs(overlap)；max |u − phase·v| <= 1e-9
        const Cx overlap = dm::Vdot(4, v.data(), u.data());
        const double ab = dm::Abs(overlap);
        Cx phase{1.0, 0.0};
        if (ab > 1e-300) {
            phase = dm::DivReal(overlap, ab);
        }
        double dist = 0.0;
        for (int e = 0; e < 4; ++e) {
            const double a = dm::Abs(dm::Sub(u[e], dm::Mul(phase, v[e])));
            dist = (a == a) ? std::max(dist, a) : a;
            if (!(a == a)) {
                break;
            }
        }
        if (Verified(dist, 1e-9)) {
            return ops;
        }
    }
    return std::nullopt;
}

M4 KakSynthesizer::BlockUnitary(const std::vector<BlockMember>& members) const {
    M4 u{};
    for (int i = 0; i < 4; ++i) {
        u[static_cast<std::size_t>(i * 5)] = Cx{1.0, 0.0};
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
                        full[static_cast<std::size_t>(r * 4 + c)] = member.matrix[kRev[r] * 4 + kRev[c]];
                    }
                }
                break;
            }
            default: {
                M4 m;
                std::copy(member.matrix, member.matrix + 16, m.begin());
                full = Mat4(Mat4(c_.swap4, m), c_.swap4);
                break;
            }
        }
        u = Mat4(full, u);
    }
    return u;
}

std::optional<std::array<M2, 2>> KakSynthesizer::SwapResidual(const M4& u, double tol) {
    // w = SWAP @ u（行置换，精确）
    constexpr int kSwap[4] = {0, 2, 1, 3};
    M4 w;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            w[static_cast<std::size_t>(i * 4 + j)] = u[static_cast<std::size_t>(kSwap[i] * 4 + j)];
        }
    }
    // r = w.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)：r[(a, c), (b, d)] = w[(a, b), (c, d)]
    M4 r;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    r[static_cast<std::size_t>((a * 2 + c) * 4 + b * 2 + d)] =
                        w[static_cast<std::size_t>((a * 2 + b) * 4 + c * 2 + d)];
                }
            }
        }
    }
    M4 left;
    M4 right;
    double sv[4];
    if (!dm::Svd(4, r.data(), left.data(), sv, right.data())) {
        return std::nullopt;
    }
    if (!(sv[1] <= tol * sv[0])) {
        return std::nullopt;
    }
    const double scale = dm::Sqrt(sv[0]);
    M2 ma;
    M2 mb;
    for (int k = 0; k < 4; ++k) {
        ma[static_cast<std::size_t>(k)] = dm::Scale(left[static_cast<std::size_t>(k * 4)], scale);  // left[:, 0]
        mb[static_cast<std::size_t>(k)] = dm::Scale(right[static_cast<std::size_t>(k)], scale);     // right[0, :]
    }
    const M4 prod = Kron2(ma, mb);
    for (int e = 0; e < 16; ++e) {
        const double diff = dm::Abs(dm::Sub(prod[static_cast<std::size_t>(e)], w[static_cast<std::size_t>(e)]));
        if (!(diff <= tol + 1e-5 * dm::Abs(w[static_cast<std::size_t>(e)]))) {
            return std::nullopt;
        }
    }
    const Cx ra = dm::Sqrt(dm::Det(ma));
    const Cx rb = dm::Sqrt(dm::Det(mb));
    for (int k = 0; k < 4; ++k) {
        ma[static_cast<std::size_t>(k)] = dm::Div(ma[static_cast<std::size_t>(k)], ra);
        mb[static_cast<std::size_t>(k)] = dm::Div(mb[static_cast<std::size_t>(k)], rb);
    }
    return std::array<M2, 2>{ma, mb};
}

double KakSynthesizer::DistanceUpToPhase(const M4& u, const M4& v) {
    // overlap = vdot(v, u)；phase = overlap / |overlap|；max |u − phase·v|
    const Cx overlap = dm::Vdot(16, v.data(), u.data());
    const double ab = dm::Abs(overlap);
    Cx phase{1.0, 0.0};
    if (ab > 1e-300) {
        phase = dm::DivReal(overlap, ab);
    }
    double dist = 0.0;
    for (int e = 0; e < 16; ++e) {
        const double a = dm::Abs(dm::Sub(u[e], dm::Mul(phase, v[e])));
        if (!(a == a)) {
            return a;
        }
        dist = std::max(dist, a);
    }
    return dist;
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
    std::vector<std::pair<int, M2>> targets = {
        {0, Mat2AdjA(k1u[0], k1v[0])},
        {1, Mat2AdjA(k1u[1], k1v[1])},
        {0, Mat2AdjB(k2u[0], k2v[0])},
        {1, Mat2AdjB(k2u[1], k2v[1])},
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
    for (std::size_t i = 0; i < 2; ++i) {
        lw[i] = Mat2AdjA(k1u[i], k1v[i]);
        rw[i] = Mat2AdjB(k2v[i], k2u[i]);
    }
    // Python 的 {0.0, x, y}：按 == 去重，再升序排列
    auto sorted_set = [](double x, double y) {
        std::vector<double> vals{0.0};
        for (double t : {x, y}) {
            if (!(t == t)) {
                throw SynthesisFailed("non-finite gauge angle");
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

    // u ∝ (k1u S k1v^†) V (k2v^† S^† k2u)：lefts = a_i S_i b_i^†，rights = c_i^† R_i d_i（R = S^† 或规范对的右因子）
    std::vector<Pair> cands =
        StabilizerCandidates(k, kv.v, a, b, c, d, /*fit_cosets=*/swap_face, opt.mirrored_stabilizers);
    std::vector<std::pair<Pair, Pair>> gauges;
    if (swap_face) {
        gauges = SwapFaceGauges(a, b, c, d);
    }
    const std::size_t nc = cands.size();
    const std::size_t n = nc + gauges.size();
    auto left_stab = [&](std::size_t j, std::size_t i) -> const M2& {
        return j < nc ? cands[j][i] : gauges[j - nc].first[i];
    };
    auto right_stab = [&](std::size_t j, std::size_t i) -> M2 {
        return j < nc ? dm::Adjoint(cands[j][i]) : gauges[j - nc].second[i];
    };
    auto outer = [&](std::size_t j, std::size_t i, M2* first, M2* last) {
        const M2 left = Mat2AdjB(Mat2(a[i], left_stab(j, i)), b[i]);
        const M2 right = Mat2(Mat2AdjA(c[i], right_stab(j, i)), d[i]);
        const M2& first_layer = i == 0 ? layers[0].left : layers[0].right;
        const M2& last_layer = i == 0 ? layers.back().left : layers.back().right;
        *first = Mat2(first_layer, right);
        *last = Mat2(left, last_layer);
    };

    if (n == 1) {
        // fixed(cands[0])（只有 k = 0）
        std::vector<Layer> out = layers;
        M2 f0;
        M2 f1;
        M2 l0;
        M2 l1;
        outer(0, 0, &f0, &l0);
        outer(0, 1, &f1, &l1);
        out[0] = Layer{f0, f1};
        if (out.size() == 1) {
            // 第一层与最后一层是同一层：先右因子，再左因子
            out[0] = Layer{Mat2(Mat2AdjB(Mat2(a[0], cands[0][0]), b[0]), f0),
                           Mat2(Mat2AdjB(Mat2(a[1], cands[0][1]), b[1]), f1)};
        } else {
            out.back() = Layer{l0, l1};
        }
        return EmitFixedLayers(u, out, k, atol, opt);
    }

    // 每个候选的首末层，按欧拉长度打分
    std::vector<M2> firsts[2];
    std::vector<M2> lasts[2];
    for (std::size_t i = 0; i < 2; ++i) {
        firsts[i].resize(n);
        lasts[i].resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            outer(j, i, &firsts[i][j], &lasts[i][j]);
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
    double best0 = std::numeric_limits<double>::infinity();
    double best1 = std::numeric_limits<double>::infinity();
    bool flag0 = false;
    bool flag1 = false;
    std::vector<Op> ops0;
    std::vector<Op> ops1;
    bool ok0 = true;
    bool ok1 = true;
    try {
        ops0 = SynthesizeExactOnce(u, kak, opt, &flag0, &best0);
    } catch (const SynthesisFailed&) {
        ok0 = false;
    }
    // 第二种基只在 Weyl 室的边界附近（k1、k2 有连续或离散的额外自由度）才可能给出不同的电路
    const double a = kak.v[0];
    const double b = kak.v[1];
    const double c = kak.v[2];
    const bool boundary = std::fabs(a - b) <= kBoundaryTol || std::fabs(b - std::fabs(c)) <= kBoundaryTol ||
                          std::fabs(b) <= kBoundaryTol || std::fabs(c) <= kBoundaryTol ||
                          std::fabs(a - kPi4) <= kBoundaryTol;
    // 第二种基给出逐位相同的 KAK（约三分之一的边界块）时合成结果相同，不必再合成
    const Kak kak1 = boundary ? Kak4(u, 1) : Kak{};
    if (boundary && !SameKak(kak1, kak)) {
        try {
            ops1 = SynthesizeExactOnce(u, kak1, opt, &flag1, &best1);
        } catch (const SynthesisFailed&) {
            ok1 = false;
        }
    } else {
        ok1 = false;
    }
    if (!ok0 && !ok1) {
        throw SynthesisFailed(FailureMessage(std::min(best0, best1)));
    }
    auto entanglers = [](const std::vector<Op>& ops) {
        return std::count_if(ops.begin(), ops.end(), [](const Op& o) { return o.kind == OpKind::kEntangler; });
    };
    const bool take1 = !ok0 || (ok1 && (entanglers(ops1) < entanglers(ops0) ||
                                        (entanglers(ops1) == entanglers(ops0) && ops1.size() < ops0.size())));
    if (uses_flag != nullptr) {
        *uses_flag = take1 ? flag1 : flag0;
    }
    return take1 ? std::move(ops1) : std::move(ops0);
}

std::vector<Op> KakSynthesizer::SynthesizeExactOnce(const M4& u, const Kak& kak, const SynthOptions& opt,
                                                    bool* uses_flag, double* best_err) const {
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
    double best = std::numeric_limits<double>::infinity();
    auto note = [&best](double err) {
        if (err < best) {
            best = err;
        }
    };
    if (opt.swap_face_template && swap_face) {
        for (double tol : euler_tols) {
            TryResult r = TrySynthesize(u, kak, uf, 3, tol, nullptr, opt, true);
            if (Verified(r.err, opt.verify_atol)) {
                return std::move(r.ops);
            }
            note(r.err);
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
                note(r.err);
            }
        }
    }
    *best_err = best;
    throw SynthesisFailed(FailureMessage(best));
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
                const double s = dm::Sin(0.5 * (phases[i] - phases[j]));
                sum += s * s;
            }
        }
        const double predicted = 4.0 * (2.0 * sum / 16.0) / 5.0;
        // predicted > budget 时 Python 跳过该 k；否则可能存在近似
        if (!(predicted > budget)) {
            return false;
        }
    }
    return true;
}

}  // namespace quspect::native::synthesis
