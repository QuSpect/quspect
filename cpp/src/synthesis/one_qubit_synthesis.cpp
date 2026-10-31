#include "quspect_native/synthesis/one_qubit_synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

#include "quspect_native/synthesis/kak_synthesis.hpp"

namespace quspect::native::synthesis {

namespace {

constexpr double kPi = 3.141592653589793;  // math.pi / np.pi
constexpr double kTwoPi = 2.0 * kPi;       // _TWO_PI
constexpr double kAtol = 1e-9;              // _ATOL
// 只用于比较的量（numpy 数组的 np.abs、验证用的矩阵乘积）与阈值的距离小于该值时退回 Python。
constexpr double kMargin = 1e-12;

using C = std::complex<double>;

C ToC(const Cx& z) {
    return {z.re, z.im};
}

double Abs(const Cx& z) {
    // numpy 标量 abs（npy_cabs）与 CPython abs：hypot（探测确认）
    return std::hypot(z.re, z.im);
}

double Angle(const Cx& z) {
    // np.angle：arctan2(imag, real)
    return std::atan2(z.im, z.re);
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

OneQubitSynthesizer::OneQubitSynthesizer(const ArithModes& modes, const std::array<M2, kNumFixedGates>& fixed)
    : modes_(modes), fixed_(fixed) {}

double OneQubitSynthesizer::NpAngleOfQuotient(const Cx& num, const Cx& den) const {
    // np.angle(u_a / u_b)：numpy 标量除法（CDOUBLE_divide）
    return Angle(NpDiv(modes_.np_div, num, den));
}

std::optional<std::vector<Gate1>> OneQubitSynthesizer::Rz(double theta, const OneQubitBasis& basis) const {
    // _Emitter.rz
    const double t = NormalizeAngle(theta);
    if (std::fabs(t) < kAtol || std::fabs(std::fabs(t) - kTwoPi) < kAtol) {
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
    const double theta = 2.0 * std::atan2(s, c);
    constexpr double eps = 1e-10;
    if (Abs(u00) >= eps) {
        // phase_correction = np.exp(-1j * np.angle(u00))
        const Cx pc = NpExpImag(modes_, PyMulReal(modes_, Cx{-0.0, -1.0}, Angle(u00)));
        u00 = Mul(modes_.np_scalar, u00, pc);
        u01 = Mul(modes_.np_scalar, u01, pc);
        u10 = Mul(modes_.np_scalar, u10, pc);
        u11 = Mul(modes_.np_scalar, u11, pc);
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
        const double diff = Angle(Mul(modes_.np_scalar, u10, Cx{-u01.re, u01.im}));
        phi = 0.5 * (total + diff);
        lam = 0.5 * (total - diff);
        const Cx e = NpExpImag(modes_, PyMulReal(modes_, Cx{-0.0, -1.0}, phi));
        if (Mul(modes_.np_scalar, e, u10).re < 0.0) {
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
        if (std::fabs(std::fabs(d) - limit) <= 1e-14) {
            throw Fallback("named-gate prefilter near its limit");
        }
        skip = skip || d > limit || d < -limit;
    }
    return skip;
}

bool OneQubitSynthesizer::EqualUpToPhase(const M2& a, const M2& b) const {
    // _equal_up_to_phase(a, b)：相位按 a 中模最大的元素对齐，再比较 max|a − e^{iφ}·b| < _ATOL。
    // numpy 的 argmax 取第一个最大值；模几乎相等的元素都试一遍，结论必须一致。
    std::array<double, 4> mag;
    double top = 0.0;
    for (int i = 0; i < 4; ++i) {
        mag[i] = Abs(a[i]);
        top = std::max(top, mag[i]);
    }
    int verdict = -1;
    for (int k = 0; k < 4; ++k) {
        if (mag[k] < top * (1.0 - 1e-12)) {
            continue;
        }
        const C ak = ToC(a[k]);
        const C bk = ToC(b[k]);
        bool equal;
        const double half = 0.5 * std::abs(ak);
        if (std::fabs(std::abs(bk) - half) <= kMargin * std::max(1.0, half)) {
            throw Fallback("phase reference entry near the cut-off");
        }
        if (std::abs(bk) < half) {
            equal = false;
        } else {
            const C ratio = ak / bk;
            const C phase = ratio / std::abs(ratio);
            double dist = 0.0;
            for (int i = 0; i < 4; ++i) {
                dist = std::max(dist, std::abs(ToC(a[i]) - phase * ToC(b[i])));
            }
            if (!(dist == dist) || std::fabs(dist - kAtol) <= kMargin) {
                throw Fallback("one-qubit verification distance near the tolerance");
            }
            equal = dist < kAtol;
        }
        if (verdict >= 0 && verdict != static_cast<int>(equal)) {
            throw Fallback("phase reference entry ambiguous");
        }
        verdict = static_cast<int>(equal);
    }
    return verdict == 1;
}

M2 OneQubitSynthesizer::SequenceMatrix(const std::vector<Gate1>& gates) const {
    // 只用于验证（与 numpy 的乘积可能差最后几位，比较带余量）
    C m[4] = {1.0, 0.0, 0.0, 1.0};
    for (const Gate1& g : gates) {
        C u[4];
        switch (g.kind) {
            case 0:
                for (int i = 0; i < 4; ++i) {
                    u[i] = ToC(fixed_[static_cast<std::size_t>(g.index)][static_cast<std::size_t>(i)]);
                }
                break;
            case 1:  // rz / p / u1（相差全局相位，不影响验证）
                u[0] = std::exp(C(0.0, -g.p0 / 2));
                u[1] = 0.0;
                u[2] = 0.0;
                u[3] = std::exp(C(0.0, g.p0 / 2));
                break;
            case 2: {
                const double c = std::cos(g.p0 / 2);
                const double s = std::sin(g.p0 / 2);
                u[0] = c;
                u[1] = C(0.0, -s);
                u[2] = C(0.0, -s);
                u[3] = c;
                break;
            }
            case 3: {
                const double c = std::cos(g.p0 / 2);
                const double s = std::sin(g.p0 / 2);
                u[0] = c;
                u[1] = -s;
                u[2] = s;
                u[3] = c;
                break;
            }
            default: {
                const double c = std::cos(g.p0 / 2);
                const double s = std::sin(g.p0 / 2);
                const C e_ip = std::exp(C(0.0, g.p1));
                const C e_il = std::exp(C(0.0, g.p2));
                u[0] = c;
                u[1] = -s * e_il;
                u[2] = s * e_ip;
                u[3] = c * e_ip * e_il;
                break;
            }
        }
        const C r[4] = {u[0] * m[0] + u[1] * m[2], u[0] * m[1] + u[1] * m[3], u[2] * m[0] + u[3] * m[2],
                        u[2] * m[1] + u[3] * m[3]};
        for (int i = 0; i < 4; ++i) {
            m[i] = r[i];
        }
    }
    return {Cx{m[0].real(), m[0].imag()}, Cx{m[1].real(), m[1].imag()}, Cx{m[2].real(), m[2].imag()},
            Cx{m[3].real(), m[3].imag()}};
}

std::vector<std::optional<std::vector<Gate1>>> OneQubitSynthesizer::Candidates(const M2& m,
                                                                               const OneQubitBasis& basis) const {
    std::vector<std::optional<std::vector<Gate1>>> cands;
    const Cx& u00 = m[0];
    const Cx& u01 = m[1];
    const Cx& u10 = m[2];
    const Cx& u11 = m[3];
    // is_identity_up_to_phase(m, 1e-9)
    if (Abs(u01) < 1e-9 && Abs(u10) < 1e-9 && Abs(Cx{u00.re - u11.re, u00.im - u11.im}) < 1e-9
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
        cands.push_back(Rz(NpAngleOfQuotient(u11, u00), basis));
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
        const double a = NpAngleOfQuotient(u10, u01);
        const auto tail = Rz(a, basis);
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
            const auto mid = Rz(kPi, basis);
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
    if (basis.h && std::fabs(Abs(u00) - std::sqrt(0.5)) < 1e-9) {
        const double a = NpAngleOfQuotient(u10, u00);
        const double c = NpAngleOfQuotient(u01, u00);
        const auto first = Rz(c, basis);
        const auto last = Rz(a, basis);
        if (first && last) {
            std::vector<Gate1> cand = *first;
            cand.push_back(Fixed(kGateH));
            Append(&cand, *last);
            cands.emplace_back(std::move(cand));
        }
    }
    // ZYZ
    if (basis.ry) {
        const auto first = Rz(lam, basis);
        const auto last = Rz(phi, basis);
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
        const auto first = Rz(c, basis);
        const auto last = Rz(a, basis);
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
            const auto mid = Rz(b, basis);
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
        const auto first = Rz(lam, basis);
        const auto mid = Rz(theta + kPi, basis);
        const auto last = Rz(phi + kPi, basis);
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

std::optional<std::vector<Gate1>> OneQubitSynthesizer::Synthesize(const M2& m, const OneQubitBasis& basis) const {
    for (const Cx& z : m) {
        if (!std::isfinite(z.re) || !std::isfinite(z.im)) {
            throw Fallback("non-finite single-qubit matrix");
        }
    }
    std::optional<std::vector<Gate1>> best;
    for (const auto& cand : Candidates(m, basis)) {
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

}  // namespace quspect::native::synthesis
