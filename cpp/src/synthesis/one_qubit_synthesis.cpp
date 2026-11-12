#include "quspect_native/synthesis/one_qubit_synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/detmath/scalar.hpp"
#include "quspect_native/synthesis/gate_matrices.hpp"


namespace quspect::native::synthesis {

namespace dm = quspect::native::detmath;

namespace {

constexpr double kPi = 3.141592653589793;  // math.pi / np.pi
constexpr double kTwoPi = 2.0 * kPi;       // _TWO_PI
constexpr double kAtol = 1e-9;              // _ATOL
constexpr double kKeepAtol = 1e-12;         // _KEEP_ATOL

using dm::Abs;

double Angle(const Cx& z) {
    // np.angle：atan2(imag, real)
    return dm::Arg(z);
}

// CPython float % float（float_rem）。
double PyMod(double x, double w) {
    double mod = std::fmod(x, w);
    if (mod != 0.0) {
        if ((w < 0.0) != (mod < 0.0)) {
            mod += w;
        }
    } else {
        mod = std::copysign(0.0, w);
    }
    return mod;
}

// CPython round(float) → int（半数取偶）。
std::int64_t PyRound(double x) {
    double rounded = std::round(x);
    if (std::fabs(x - rounded) == 0.5) {
        rounded = 2.0 * std::round(x / 2.0);
    }
    return static_cast<std::int64_t>(rounded);
}

double NormalizeAngle(double theta) {
    double t = std::fmod(theta, kTwoPi);
    if (t <= -kPi) {
        t += kTwoPi;
    } else if (t > kPi) {
        t -= kTwoPi;
    }
    return t;
}

void Append(std::vector<Gate1>* out, const std::vector<Gate1>& part) {
    out->insert(out->end(), part.begin(), part.end());
}

Gate1 Fixed(int index) {
    return Gate1{0, index, 0.0, 0.0, 0.0};
}

}  // namespace

OneQubitSynthesizer::OneQubitSynthesizer() {
    static const char* const kNames[kNumFixedGates] = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"};
    for (int i = 0; i < kNumFixedGates; ++i) {
        GateMatrix2(kNames[i], nullptr, 0, &fixed_[static_cast<std::size_t>(i)]);
    }
}

double OneQubitSynthesizer::AngleOfQuotient(const Cx& num, const Cx& den) {
    // np.angle(u_a / u_b)
    return Angle(dm::Div(num, den));
}

std::optional<std::vector<Gate1>> OneQubitSynthesizer::Rz(double theta, const OneQubitBasis& basis,
                                                          double atol) const {
    // _Emitter.rz
    const double t = NormalizeAngle(theta);
    if (std::fabs(t) < atol || std::fabs(std::fabs(t) - kTwoPi) < atol) {
        return std::vector<Gate1>{};
    }
    if (basis.rz) {
        return std::vector<Gate1>{Gate1{1, 0, t, 0.0, 0.0}};
    }
    const double k = t / (kPi / 4.0);
    std::int64_t kr = PyRound(k);
    if (std::fabs(k - static_cast<double>(kr)) < 1e-9) {
        if (kr == -4) {
            kr = 4;
        }
        if (kr >= -4 && kr <= 4 && basis.has_word[static_cast<std::size_t>(kr + 4)]) {
            std::vector<Gate1> word;
            for (int g : basis.words[static_cast<std::size_t>(kr + 4)]) {
                word.push_back(Fixed(g));
            }
            return word;
        }
    }
    return std::nullopt;
}

std::array<double, 3> OneQubitSynthesizer::ZyzAngles(const M2& m) const {
    // quspect.dag.passes.single_qubit_fusion._decompose_u3
    Cx u00 = m[0];
    Cx u01 = m[1];
    Cx u10 = m[2];
    Cx u11 = m[3];
    double c = Abs(u00);
    double s = Abs(u10);
    c = std::min(c, 1.0);
    s = std::min(s, 1.0);
    const double theta = 2.0 * dm::Atan2(s, c);
    constexpr double eps = 1e-10;
    if (Abs(u00) >= eps) {
        // phase_correction = np.exp(-1j * np.angle(u00))
        const Cx pc = dm::ExpI(-Angle(u00));
        u00 = dm::Mul(u00, pc);
        u01 = dm::Mul(u01, pc);
        u10 = dm::Mul(u10, pc);
        u11 = dm::Mul(u11, pc);
    }
    double phi;
    double lam;
    if (theta < eps) {
        phi = 0.0;
        lam = Angle(u11);
    } else if (std::fabs(theta - kPi) < eps) {
        phi = Angle(u10);
        lam = Angle(Cx{-u01.re, -u01.im});
    } else if (c >= s) {
        const double total = Angle(u11);
        // np.conj(-u01)
        const double diff = Angle(dm::Mul(u10, Cx{-u01.re, u01.im}));
        phi = 0.5 * (total + diff);
        lam = 0.5 * (total - diff);
        const Cx e = dm::ExpI(-phi);
        if (dm::Mul(e, u10).re < 0.0) {
            phi += kPi;
            lam += kPi;
        }
    } else {
        phi = Angle(u10);
        lam = Angle(Cx{-u01.re, -u01.im});
    }
    phi = PyMod(phi, 2.0 * kPi);
    lam = PyMod(lam, 2.0 * kPi);
    return {theta, phi, lam};
}

bool OneQubitSynthesizer::NamedPrefilterSkips(const M2& gate, const M2& m) const {
    // _named_match 的预筛选：d = |m| − |G|，d.max() > limit or d.min() < −limit 时跳过
    constexpr double limit = kAtol + 1e-12;  // _ATOL + _NAMED_SLACK
    bool skip = false;
    for (int i = 0; i < 4; ++i) {
        const double d = Abs(m[i]) - Abs(gate[i]);
        skip = skip || d > limit || d < -limit;
    }
    return skip;
}

bool OneQubitSynthesizer::EqualUpToPhase(const M2& a, const M2& b) const {
    // _equal_up_to_phase(a, b)：相位按 a 中模最大的元素（第一个最大值）对齐，再比较
    // max|a − e^{iφ}·b| < _ATOL。
    int k = 0;
    double top = Abs(a[0]);
    for (int i = 1; i < 4; ++i) {
        const double v = Abs(a[static_cast<std::size_t>(i)]);
        if (v > top) {
            top = v;
            k = i;
        }
    }
    const Cx& ak = a[static_cast<std::size_t>(k)];
    const Cx& bk = b[static_cast<std::size_t>(k)];
    if (Abs(bk) < 0.5 * top) {
        return false;
    }
    const Cx ratio = dm::Div(ak, bk);
    const Cx phase = dm::DivReal(ratio, Abs(ratio));
    double dist = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double d = Abs(dm::Sub(a[static_cast<std::size_t>(i)], dm::Mul(phase, b[static_cast<std::size_t>(i)])));
        if (!(d == d)) {
            return false;
        }
        dist = std::max(dist, d);
    }
    return dist < kAtol;
}

M2 OneQubitSynthesizer::SequenceMatrix(const std::vector<Gate1>& gates) const {
    // _sequence_matrix：m = I；对每个门 m = gate_matrix(g) @ m
    M2 m{Cx{1.0, 0.0}, Cx{0.0, 0.0}, Cx{0.0, 0.0}, Cx{1.0, 0.0}};
    for (const Gate1& g : gates) {
        M2 u;
        switch (g.kind) {
            case 0:
                u = fixed_[static_cast<std::size_t>(g.index)];
                break;
            case 1:  // rz / p / u1（相差全局相位，不影响验证）
                GateMatrix2("rz", &g.p0, 1, &u);
                break;
            case 2:
                GateMatrix2("rx", &g.p0, 1, &u);
                break;
            case 3:
                GateMatrix2("ry", &g.p0, 1, &u);
                break;
            case 5: {
                const double p[2] = {g.p0, g.p1};
                GateMatrix2("u2", p, 2, &u);
                break;
            }
            default: {
                const double p[3] = {g.p0, g.p1, g.p2};
                GateMatrix2("u3", p, 3, &u);
                break;
            }
        }
        m = dm::MatMul(u, m);
    }
    return m;
}

std::vector<std::optional<std::vector<Gate1>>> OneQubitSynthesizer::Candidates(const M2& m,
                                                                               const OneQubitBasis& basis,
                                                                               double atol) const {
    std::vector<std::optional<std::vector<Gate1>>> cands;
    const Cx& u00 = m[0];
    const Cx& u01 = m[1];
    const Cx& u10 = m[2];
    const Cx& u11 = m[3];
    // is_identity_up_to_phase(m, 1e-9)；atol 小于默认容差时（空序列已验证失败）继续生成其余形式
    if (atol == kAtol && Abs(u01) < 1e-9 && Abs(u10) < 1e-9 && Abs(Cx{u00.re - u11.re, u00.im - u11.im}) < 1e-9
        && std::fabs(Abs(u00) - 1.0) < 1e-9) {
        cands.emplace_back(std::vector<Gate1>{});
        return cands;
    }
    // 单个具名门
    for (int g : basis.named) {
        const M2& gm = fixed_[static_cast<std::size_t>(g)];
        if (NamedPrefilterSkips(gm, m)) {
            continue;
        }
        if (EqualUpToPhase(gm, m)) {
            cands.emplace_back(std::vector<Gate1>{Fixed(g)});
            return cands;
        }
    }
    // 对角：RZ(arg(u11/u00))
    if (Abs(u01) < kAtol && Abs(u10) < kAtol) {
        cands.push_back(Rz(AngleOfQuotient(u11, u00), basis, atol));
        return cands;
    }
    const std::array<double, 3> zyz = ZyzAngles(m);
    const double theta = zyz[0];
    const double phi = zyz[1];
    const double lam = zyz[2];
    if (basis.u3) {
        cands.emplace_back(std::vector<Gate1>{Gate1{4, 0, theta, phi, lam}});
    }
    // 反对角：X 后接 RZ(arg(u10/u01))
    if (Abs(u00) < kAtol && Abs(u11) < kAtol) {
        const double a = AngleOfQuotient(u10, u01);
        const auto tail = Rz(a, basis, atol);
        if (basis.x && tail) {
            std::vector<Gate1> c{Fixed(kGateX)};
            Append(&c, *tail);
            cands.emplace_back(std::move(c));
        }
        if (basis.sx && tail) {
            std::vector<Gate1> c{Fixed(kGateSX), Fixed(kGateSX)};
            Append(&c, *tail);
            cands.emplace_back(std::move(c));
        }
        if (basis.h) {
            const auto mid = Rz(kPi, basis, atol);
            if (mid && tail) {
                std::vector<Gate1> c{Fixed(kGateH)};
                Append(&c, *mid);
                c.push_back(Fixed(kGateH));
                Append(&c, *tail);
                cands.emplace_back(std::move(c));
            }
        }
    }
    // 半旋转：rz(c), h, rz(a)
    if (basis.h && std::fabs(Abs(u00) - dm::Sqrt(0.5)) < 1e-9) {
        const double a = AngleOfQuotient(u10, u00);
        const double c = AngleOfQuotient(u01, u00);
        const auto first = Rz(c, basis, atol);
        const auto last = Rz(a, basis, atol);
        if (first && last) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Fixed(kGateH));
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
    }
    // ZYZ
    if (basis.ry) {
        const auto first = Rz(lam, basis, atol);
        const auto last = Rz(phi, basis, atol);
        if (first && last) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Gate1{3, 0, theta, 0.0, 0.0});
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
    }
    // ZXZ（两种等价写法）
    const double zxz[2][3] = {{lam - kPi / 2.0, theta, phi + kPi / 2.0}, {lam + kPi / 2.0, -theta, phi - kPi / 2.0}};
    for (const auto& row : zxz) {
        const double c = row[0];
        const double b = row[1];
        const double a = row[2];
        const auto first = Rz(c, basis, atol);
        const auto last = Rz(a, basis, atol);
        if (!first || !last) {
            continue;
        }
        if (basis.rx) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Gate1{2, 0, b, 0.0, 0.0});
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
        if (basis.h) {
            const auto mid = Rz(b, basis, atol);
            if (mid) {
                std::vector<Gate1> cand = *first;
                cand.push_back(Fixed(kGateH));
                Append(&cand, *mid);
                cand.push_back(Fixed(kGateH));
                Append(&cand, *last);
                cands.emplace_back(std::move(cand));
            }
        }
        if (basis.sx && std::fabs(b - kPi / 2.0) < 1e-9) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Fixed(kGateSX));
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
        if (basis.sxdg && std::fabs(b + kPi / 2.0) < 1e-9) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Fixed(kGateSXdg));
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
    }
    // ZSX：rz(lam), sx, rz(theta + π), sx, rz(phi + π)
    if (basis.sx) {
        const auto first = Rz(lam, basis, atol);
        const auto mid = Rz(theta + kPi, basis, atol);
        const auto last = Rz(phi + kPi, basis, atol);
        if (first && mid && last) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Fixed(kGateSX));
            Append(&cand, *mid);
            cand.push_back(Fixed(kGateSX));
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
    }
    return cands;
}

std::vector<std::optional<std::vector<Gate1>>> OneQubitSynthesizer::FallbackCandidates(
    const M2& m, const OneQubitBasis& basis) const {
    // _fallback_candidates：先是保留容差以下旋转的常规形式
    std::vector<std::optional<std::vector<Gate1>>> cands = Candidates(m, basis, kKeepAtol);
    const Cx& u00 = m[0];
    const Cx& u01 = m[1];
    const Cx& u10 = m[2];
    const Cx& u11 = m[3];
    const std::array<double, 3> zyz = ZyzAngles(m);
    const double theta = zyz[0];
    const double phi = zyz[1];
    const double lam = zyz[2];
    // U3（Candidates 的对角分支在它之前返回）
    if (basis.u3) {
        cands.emplace_back(std::vector<Gate1>{Gate1{4, 0, theta, phi, lam}});
    }
    // 半旋转是一个 U2
    if (basis.u2 && std::fabs(theta - kPi / 2.0) < 1e-9) {
        cands.emplace_back(std::vector<Gate1>{Gate1{5, 0, phi, lam, 0.0}});
    }
    // XYX：W = H·U·H ∝ RZ(a)·RY(t)·RZ(c)，U ∝ RX(a)·RY(−t)·RX(c)（线路顺序 rx(c), ry(−t), rx(a)）
    if (basis.rx && basis.ry) {
        const Cx sum0 = dm::Add(u00, u01);
        const Cx dif0 = dm::Sub(u00, u01);
        const Cx sum1 = dm::Add(u10, u11);
        const Cx dif1 = dm::Sub(u10, u11);
        const auto half = [](const Cx& z) { return Cx{0.5 * z.re, 0.5 * z.im}; };
        const M2 w{half(dm::Add(sum0, sum1)), half(dm::Add(dif0, dif1)), half(dm::Sub(sum0, sum1)),
                   half(dm::Sub(dif0, dif1))};
        const std::array<double, 3> xyx = ZyzAngles(w);
        const double t = xyx[0];
        const double a = xyx[1];
        const double c = xyx[2];
        const Gate1 gates[3] = {Gate1{2, 0, NormalizeAngle(c), 0.0, 0.0}, Gate1{3, 0, -t, 0.0, 0.0},
                                Gate1{2, 0, NormalizeAngle(a), 0.0, 0.0}};
        const double angles[3] = {c, t, a};
        std::vector<Gate1> cand;
        for (int i = 0; i < 3; ++i) {
            // _is_zero_rotation
            const double n = NormalizeAngle(angles[i]);
            if (!(std::fabs(n) < kKeepAtol || std::fabs(std::fabs(n) - kTwoPi) < kKeepAtol)) {
                cand.push_back(gates[i]);
            }
        }
        cands.emplace_back(std::move(cand));
    }
    // U2·U2：U ∝ RZ(φ+π/2)·RY(π/2)·RZ(θ+π)·RY(π/2)·RZ(λ+π/2)
    if (basis.u2) {
        cands.emplace_back(std::vector<Gate1>{
            Gate1{5, 0, 0.0, NormalizeAngle(lam + kPi / 2.0), 0.0},
            Gate1{5, 0, NormalizeAngle(phi + kPi / 2.0), NormalizeAngle(theta + kPi), 0.0}});
    }
    return cands;
}

std::optional<std::vector<Gate1>> OneQubitSynthesizer::ShortestValid(
    const std::vector<std::optional<std::vector<Gate1>>>& cands, const M2& m) const {
    std::optional<std::vector<Gate1>> best;
    for (const auto& cand : cands) {
        if (!cand) {
            continue;
        }
        if (best && cand->size() >= best->size()) {
            continue;
        }
        if (EqualUpToPhase(SequenceMatrix(*cand), m)) {
            best = cand;
        }
    }
    return best;
}

std::optional<std::vector<Gate1>> OneQubitSynthesizer::Synthesize(const M2& m, const OneQubitBasis& basis) const {
    for (const Cx& z : m) {
        if (!std::isfinite(z.re) || !std::isfinite(z.im)) {
            throw std::domain_error("non-finite single-qubit matrix");
        }
    }
    std::optional<std::vector<Gate1>> best = ShortestValid(Candidates(m, basis, kAtol), m);
    if (!best) {
        best = ShortestValid(FallbackCandidates(m, basis), m);
    }
    return best;
}

}  // namespace quspect::native::synthesis
