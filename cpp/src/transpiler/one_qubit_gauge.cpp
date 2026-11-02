#include "quspect_native/transpiler/one_qubit_gauge.hpp"

#include <cmath>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::transpiler {

namespace dm = quspect::native::detmath;

namespace {

constexpr double kPi = 3.141592653589793;
constexpr double kHalfPi = kPi / 2.0;
constexpr double kTwoPi = 2.0 * kPi;
constexpr int kInf = 1000000;

bool Finite(const Complex& z) {
    return std::isfinite(z.re) && std::isfinite(z.im);
}

// 复数运算（Mul、Sub、Neg、Conj、Abs 等）直接用 detmath 的（Complex 即 detmath::Cx）。
using dm::Abs;
using dm::Conj;
using dm::Mul;
using dm::Neg;
using dm::Sub;

constexpr const Complex& D(const Complex& z) {
    return z;
}

constexpr const Complex& C(const Complex& z) {
    return z;
}

// arg a（atan2(im, re)）。
double Phase(const Complex& a) {
    return dm::Arg(D(a));
}

// e^{−iθ}。
Complex ExpMinusI(double theta) {
    return C(dm::ExpI(-theta));
}

// a / b（Smith 算法）；除以零返回 nullopt（Python 端抛异常）。
std::optional<Complex> DivChecked(const Complex& a, const Complex& b) {
    if (b.re == 0.0 && b.im == 0.0) {
        return std::nullopt;
    }
    return C(dm::Div(D(a), D(b)));
}

// Python ``float % float``（同号余数，精确）。
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

// ``_nz``：RZ(theta) 是否不是恒等（容差 1e-9）。
int Nz(double theta) {
    // |θ| < 2π 时 fmod(θ, 2π) = θ（精确），免去库调用
    double t = std::fabs(theta) < kTwoPi ? theta : std::fmod(theta, kTwoPi);
    if (t <= -kPi) {
        t += kTwoPi;
    } else if (t > kPi) {
        t -= kTwoPi;
    }
    return (std::fabs(t) < 1e-9 || std::fabs(std::fabs(t) - kTwoPi) < 1e-9) ? 0 : 1;
}

// ``_gauge._zyz``。
std::optional<std::array<double, 3>> PyZyz(Complex u00, Complex u01, Complex u10, Complex u11) {
    const double a00 = Abs(u00);
    const double c = std::min(a00, 1.0);
    const double s = std::min(Abs(u10), 1.0);
    const double theta = 2.0 * dm::Atan2(s, c);
    if (a00 >= 1e-10) {
        const Complex corr = ExpMinusI(Phase(u00));
        u00 = Mul(u00, corr);
        u01 = Mul(u01, corr);
        u10 = Mul(u10, corr);
        u11 = Mul(u11, corr);
    }
    double phi;
    double lam;
    if (theta < 1e-10) {
        phi = 0.0;
        lam = Phase(u11);
    } else if (std::fabs(theta - kPi) < 1e-10) {
        phi = Phase(u10);
        lam = Phase(Neg(u01));
    } else if (c >= s) {
        const double total = Phase(u11);
        const double diff = Phase(Mul(u10, Conj(Neg(u01))));
        phi = 0.5 * (total + diff);
        lam = 0.5 * (total - diff);
        if (Mul(ExpMinusI(phi), u10).re < 0.0) {
            phi += kPi;
            lam += kPi;
        }
    } else {
        phi = Phase(u10);
        lam = Phase(Neg(u01));
    }
    return std::array<double, 3>{theta, PyMod(phi, kTwoPi), PyMod(lam, kTwoPi)};
}

// ``_LengthModel._named``。
std::optional<bool> Named(const Matrix2& a, const LengthModel& model) {
    int k = 0;
    double best = Norm2(a[0]);
    for (int i = 1; i < 4; ++i) {
        const double v = Norm2(a[i]);
        if (v > best) {  // Python max(key=...) 保留第一个最大值
            best = v;
            k = i;
        }
    }
    for (const Matrix2& g : model.named) {
        const Complex& gk = g[k];
        if (Norm2(gk) < 0.25 * best) {
            continue;
        }
        auto ph = DivChecked(a[k], gk);
        if (!ph) {
            return std::nullopt;
        }
        ph = DivChecked(*ph, Complex{Abs(*ph), 0.0});
        if (!ph) {
            return std::nullopt;
        }
        bool all = true;
        for (int i = 0; i < 4 && all; ++i) {
            all = Norm2(Sub(a[i], Mul(*ph, g[i]))) < 1e-18;
        }
        if (all) {
            return true;
        }
    }
    return false;
}

}  // namespace

std::optional<int> FastLength(const Matrix2& m, const LengthModel& model) {
    for (const Complex& z : m) {
        if (!Finite(z)) {
            return std::nullopt;
        }
    }
    const Complex u00 = m[0];
    const Complex u01 = m[1];
    const Complex u10 = m[2];
    const Complex u11 = m[3];
    const double sqrt_half = dm::Sqrt(0.5);
    const double a00 = Abs(u00);
    if (Norm2(u10) < 1e-18 && Norm2(u01) < 1e-18) {
        if (Norm2(Sub(u00, u11)) < 1e-18) {
            return 0;
        }
        return 1;
    }
    if (a00 < 1e-9 || std::fabs(a00 - sqrt_half) < 1e-9) {
        const auto named = Named(m, model);
        if (!named) {
            return std::nullopt;
        }
        if (*named) {
            return 1;
        }
    }
    int best = kInf;
    if (a00 < 1e-9 && Norm2(u11) < 1e-18) {
        const auto q = DivChecked(u10, u01);
        if (!q) {
            return std::nullopt;
        }
        const int tail = Nz(Phase(*q));
        if (model.x) {
            best = 1 + tail;
        } else if (model.sx) {
            best = 2 + tail;
        } else if (model.h) {
            best = 3 + tail;
        }
    }
    if (model.h && std::fabs(a00 - sqrt_half) < 1e-9) {
        const auto q1 = DivChecked(u10, u00);
        const auto q2 = DivChecked(u01, u00);
        if (!q1 || !q2) {
            return std::nullopt;
        }
        best = std::min(best, 1 + Nz(Phase(*q1)) + Nz(Phase(*q2)));
    }
    const auto angles = PyZyz(u00, u01, u10, u11);
    if (!angles) {
        return std::nullopt;
    }
    const double theta = (*angles)[0];
    const double phi = (*angles)[1];
    const double lam = (*angles)[2];
    const double cands[2][3] = {{lam - kHalfPi, theta, phi + kHalfPi}, {lam + kHalfPi, -theta, phi - kHalfPi}};
    for (const auto& cba : cands) {
        const double c = cba[0];
        const double b = cba[1];
        const double a = cba[2];
        const int ends = Nz(c) + Nz(a);
        if (model.h) {
            best = std::min(best, 2 + ends + Nz(b));
        }
        if (model.sx && std::fabs(b - kHalfPi) < 1e-9) {
            best = std::min(best, 1 + ends);
        }
        if (model.sxdg && std::fabs(b + kHalfPi) < 1e-9) {
            best = std::min(best, 1 + ends);
        }
    }
    if (model.sx) {
        best = std::min(best, 2 + Nz(lam) + Nz(theta + kPi) + Nz(phi + kPi));
    }
    return best;
}

std::optional<LeadingAngles> LeadingAnglesInZFrame(const Matrix2& z) {
    for (const Complex& v : z) {
        if (!Finite(v)) {
            return std::nullopt;
        }
    }
    LeadingAngles out;
    if (Norm2(z[1]) < 1e-24 && Norm2(z[2]) < 1e-24) {
        // [float(np.angle(z[1, 1] / z[0, 0]))]
        const auto q = DivChecked(z[3], z[0]);
        if (!q) {
            return std::nullopt;
        }
        out.count = 1;
        out.angles[0] = Phase(*q);
        return out;
    }
    // single_qubit_fusion._decompose_u3 的 phi
    Complex u00 = z[0];
    Complex u01 = z[1];
    Complex u10 = z[2];
    Complex u11 = z[3];
    const double a00 = Abs(u00);
    const double c = std::min(a00, 1.0);
    const double s = std::min(Abs(u10), 1.0);
    const double theta = 2.0 * dm::Atan2(s, c);
    constexpr double kEps = 1e-10;
    if (a00 >= kEps) {
        const Complex corr = ExpMinusI(Phase(u00));
        u00 = Mul(u00, corr);
        u01 = Mul(u01, corr);
        u10 = Mul(u10, corr);
        u11 = Mul(u11, corr);
    }
    double phi;
    if (theta < kEps) {
        phi = 0.0;
    } else if (std::fabs(theta - kPi) < kEps) {
        phi = Phase(u10);
    } else if (c >= s) {
        const double total = Phase(u11);
        const double diff = Phase(Mul(u10, Conj(Neg(u01))));
        phi = 0.5 * (total + diff);
        if (Mul(ExpMinusI(phi), u10).re < 0.0) {
            phi += kPi;
        }
    } else {
        phi = Phase(u10);
    }
    phi = PyMod(phi, 2.0 * kPi);
    out.count = 4;
    out.angles = {phi, phi + kHalfPi, phi - kHalfPi, phi + kPi};
    return out;
}

Matrix2 AxisRotation(int axis, double phi) {
    static const Matrix2 kIdentity = {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}};
    static const Matrix2 kPaulis[3] = {
        {Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{-1.0, 0.0}},   // z
        {Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}},    // x
        {Complex{0.0, 0.0}, Complex{-0.0, -1.0}, Complex{0.0, 1.0}, Complex{0.0, 0.0}},  // y
    };
    const Matrix2& p = kPaulis[axis];
    const double c = dm::Cos(phi / 2.0);
    const double s = dm::Sin(phi / 2.0);
    // cos(φ/2)·I − (i·sin(φ/2))·P
    const Complex is{0.0, s};
    Matrix2 out;
    for (int i = 0; i < 4; ++i) {
        out[i] = Sub(C(dm::Scale(D(kIdentity[i]), c)), Mul(is, p[i]));
    }
    return out;
}

std::optional<int> EulerLength(const Matrix2& m, int basis, double atol) {
    for (const Complex& v : m) {
        if (!Finite(v)) {
            return std::nullopt;
        }
    }
    // _zxz_angles（与 KakSynthesizer::EulerOps 的运算相同，长度与之一致）
    const dm::Cx det = dm::Sub(dm::Mul(D(m[0]), D(m[3])), dm::Mul(D(m[1]), D(m[2])));
    const dm::Cx root = dm::Sqrt(det);
    const dm::Cx m00 = dm::Div(D(m[0]), root);
    const dm::Cx m10 = dm::Div(D(m[2]), root);
    const double a00 = dm::Abs(m00);
    const double a10 = dm::Abs(m10);
    const double theta = 2.0 * dm::Atan2(a10, a00);
    const double plus = a00 > 1e-12 ? -2.0 * dm::Arg(m00) : 0.0;
    const double minus = a10 > 1e-12 ? 2.0 * dm::Arg(m10) + kPi : 0.0;
    const double phi = (plus + minus) / 2.0;
    const double lam = (plus - minus) / 2.0;
    if (!std::isfinite(theta) || !std::isfinite(phi) || !std::isfinite(lam)) {
        return std::nullopt;
    }
    const auto within = [atol](double dist) { return dist <= atol; };
    const auto zero = [&within](double angle) { return within(std::fabs(dm::Remainder(angle, kTwoPi))); };
    const auto rz = [&zero](double angle) { return zero(angle) ? 0 : 1; };
    std::optional<int> length;
    if (basis == 0) {
        if (zero(theta)) {
            length = rz(phi + lam);
        } else if (within(std::fabs(theta - kPi))) {
            length = rz(lam - phi) + 1;
        } else if (within(std::fabs(theta - kHalfPi))) {
            length = rz(lam - kHalfPi) + 1 + rz(phi - kHalfPi);
        } else {
            length = rz(lam) + 3 + rz(phi);
        }
    } else if (basis == 1) {
        if (zero(theta)) {
            length = rz(phi + lam);
        } else if (within(std::fabs(theta - kPi))) {
            length = rz(lam - phi) + 1;
        } else if (within(std::fabs(theta - kHalfPi))) {
            length = rz(lam) + 1 + rz(phi);
        } else {
            length = rz(lam + kHalfPi) + 1 + rz(theta + kPi) + 1 + rz(phi + kHalfPi);
        }
    }
    return length;
}

std::int64_t PyRound(double x) {
    double rounded = std::round(x);
    if (std::fabs(x - rounded) == 0.5) {
        rounded = 2.0 * std::round(x / 2.0);
    }
    return static_cast<std::int64_t>(rounded);
}

namespace {

std::uint64_t Bits(double x) {
    std::uint64_t b;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

}  // namespace

std::size_t GaugeBeamSearch::MemoKeyHash::operator()(const MemoKey& k) const {
    std::uint64_t h = static_cast<std::uint64_t>(k.rid) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<std::uint64_t>(k.key) + 0x7F4A7C159E3779B9ULL + (h << 6) + (h >> 2);
    h ^= k.phi_e_bits + (h << 6) + (h >> 2);
    h ^= (static_cast<std::uint64_t>(k.fp) << 1) | (static_cast<std::uint64_t>(k.fn) << 2)
         | (static_cast<std::uint64_t>(k.last) << 3);
    return static_cast<std::size_t>(h);
}

GaugeBeamSearch::GaugeBeamSearch(int num_slots, int beam, int gate_weight, int run_weight, LengthModel model)
    : beam_(beam), gate_weight_(gate_weight), run_weight_(run_weight), model_(std::move(model)) {
    if (num_slots < 0 || beam < 1) {
        throw std::invalid_argument("GaugeBeamSearch: num_slots >= 0 and beam >= 1 required");
    }
    Reset(std::vector<double>(static_cast<std::size_t>(num_slots), 0.0));
}

void GaugeBeamSearch::Reset(const std::vector<double>& start_phi) {
    current_.assign(start_phi.size(), {});
    for (std::size_t i = 0; i < start_phi.size(); ++i) {
        current_[i].push_back(Entry{0, start_phi[i], -1, 0});
    }
    layers_.assign(start_phi.size(), {});
    pending_.assign(start_phi.size(), {});
}

int GaugeBeamSearch::Prepare(const std::vector<int>& slots, const std::vector<int>& rids,
                             const std::vector<std::uint8_t>& fp, const std::vector<std::uint8_t>& fn,
                             const std::vector<int>& ax_in, const std::vector<int>& ax_out,
                             const std::vector<std::uint8_t>& last, const std::vector<double>& phi_fixed,
                             const std::vector<int>& run_length, const std::vector<std::uint8_t>& base_changed) {
    const std::size_t n = slots.size();
    if (rids.size() != n || fp.size() != n || fn.size() != n || ax_in.size() != n || ax_out.size() != n
        || last.size() != n || phi_fixed.size() != n || run_length.size() != n || base_changed.size() != n) {
        throw std::invalid_argument("GaugeBeamSearch::Prepare: inconsistent lengths");
    }
    for (const int slot : slots) {
        if (slot < 0 || static_cast<std::size_t>(slot) >= current_.size()) {
            throw std::invalid_argument("GaugeBeamSearch::Prepare: slot out of range");
        }
    }
    slots_ = slots;
    ax_out_ = ax_out;
    last_ = last;
    phi_fixed_ = phi_fixed;
    run_length_ = run_length;
    base_changed_ = base_changed;
    entry_unique_.clear();
    entry_begin_.assign(1, 0);
    unique_.clear();
    compute_row_.clear();
    compute_unique_.clear();
    compute_rid_.clear();
    compute_ax_out_.clear();
    rotated_.clear();
    rotate_in_.clear();
    rotate_in_mats_.clear();
    need_leading_.clear();
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t first = unique_.size();
        const bool is_last = last[r] != 0;
        for (const Entry& e : current_[static_cast<std::size_t>(slots[r])]) {
            const MemoKey key{
                rids[r], fp[r] != 0, fn[r] != 0, PyRound(e.phi * 1e10), is_last, Bits(is_last ? phi_fixed[r] : 0.0)};
            int found = -1;
            for (std::size_t u = first; u < unique_.size(); ++u) {
                if (unique_[u].key == key) {
                    found = static_cast<int>(u);
                    break;
                }
            }
            if (found < 0) {
                found = static_cast<int>(unique_.size());
                const MemoOptions* hit = MemoGet(key);
                int compute = -1;
                if (hit == nullptr) {
                    compute = static_cast<int>(compute_row_.size());
                    compute_row_.push_back(static_cast<int>(r));
                    compute_unique_.push_back(found);
                    compute_rid_.push_back(rids[r]);
                    compute_ax_out_.push_back(ax_out[r]);
                    const bool rotate = ax_in[r] >= 0 && e.phi != 0.0;
                    rotated_.push_back(rotate ? 1 : 0);
                    if (rotate) {
                        rotate_in_.push_back(compute);
                        rotate_in_mats_.push_back(AxisRotation(ax_in[r], e.phi));
                    }
                    if (!is_last && ax_out[r] >= 0) {
                        need_leading_.push_back(compute);
                    }
                }
                unique_.push_back(Unique{static_cast<int>(r), e.phi, key, hit, compute});
            }
            entry_unique_.push_back(found);
        }
        entry_begin_.push_back(static_cast<int>(entry_unique_.size()));
    }
    return static_cast<int>(compute_row_.size());
}

void GaugeBeamSearch::SetWires(std::vector<std::vector<int>> rids, std::vector<std::vector<int>> axes,
                               std::vector<int> run_length) {
    if (rids.size() != current_.size() || axes.size() != rids.size()) {
        throw std::invalid_argument("GaugeBeamSearch::SetWires: one run / axis list per slot required");
    }
    for (std::size_t w = 0; w < rids.size(); ++w) {
        if (rids[w].empty() || axes[w].size() + 1 != rids[w].size()) {
            throw std::invalid_argument("GaugeBeamSearch::SetWires: k anchors between k + 1 runs required");
        }
        for (const int rid : rids[w]) {
            if (rid < 0 || static_cast<std::size_t>(rid) >= run_length.size()) {
                throw std::invalid_argument("GaugeBeamSearch::SetWires: run id out of range");
            }
        }
    }
    wire_rids_ = std::move(rids);
    wire_axes_ = std::move(axes);
    wire_run_length_ = std::move(run_length);
}

int GaugeBeamSearch::PrepareLayer(int layer) {
    std::vector<int> slots, rids, ax_in, ax_out, run_length;
    std::vector<std::uint8_t> last;
    for (std::size_t w = 0; w < wire_rids_.size(); ++w) {
        const auto& wr = wire_rids_[w];
        const int k = static_cast<int>(wr.size()) - 1;
        if (layer < 0 || layer > k) {
            continue;
        }
        const auto& ax = wire_axes_[w];
        slots.push_back(static_cast<int>(w));
        rids.push_back(wr[static_cast<std::size_t>(layer)]);
        ax_in.push_back(layer > 0 ? ax[static_cast<std::size_t>(layer - 1)] : -1);
        ax_out.push_back(layer < k ? ax[static_cast<std::size_t>(layer)] : -1);
        last.push_back(layer == k ? 1 : 0);
        run_length.push_back(wire_run_length_[static_cast<std::size_t>(wr[static_cast<std::size_t>(layer)])]);
    }
    const std::size_t n = slots.size();
    const std::vector<std::uint8_t> zeros(n, 0);
    return Prepare(slots, rids, zeros, zeros, ax_in, ax_out, last, std::vector<double>(n, 0.0), run_length, zeros);
}

void GaugeBeamSearch::Options(const std::vector<Matrix2>& z) {
    if (z.size() != need_leading_.size()) {
        throw std::invalid_argument("GaugeBeamSearch::Options: one z matrix per leading-angle entry required");
    }
    option_offsets_.assign(1, 0);
    option_phis_.clear();
    option_mm_.clear();
    option_rows_.clear();
    option_mats_.clear();
    std::size_t lead = 0;
    for (std::size_t c = 0; c < compute_row_.size(); ++c) {
        const std::size_t r = static_cast<std::size_t>(compute_row_[c]);
        const int axis = ax_out_[r];
        if (last_[r] != 0) {
            option_phis_.push_back(phi_fixed_[r]);  // opts = [phi_fixed]
        } else {
            option_phis_.push_back(0.0);
            if (lead < need_leading_.size() && need_leading_[lead] == static_cast<int>(c)) {
                const auto angles = LeadingAnglesInZFrame(z[lead]);
                ++lead;
                if (!angles) {
                    throw std::domain_error("GaugeBeamSearch: leading angles not reproducible");
                }
                for (int j = 0; j < angles->count; ++j) {
                    option_phis_.push_back(angles->angles[static_cast<std::size_t>(j)]);
                }
            }
        }
        for (std::size_t j = static_cast<std::size_t>(option_offsets_.back()); j < option_phis_.size(); ++j) {
            const double phi = option_phis_[j];
            if (phi != 0.0 && axis >= 0) {
                option_mm_.push_back(static_cast<int>(option_mats_.size()));
                option_rows_.push_back(static_cast<int>(c));
                option_mats_.push_back(AxisRotation(axis, -phi));
            } else {
                option_mm_.push_back(-1);
            }
        }
        option_offsets_.push_back(static_cast<int>(option_phis_.size()));
    }
}

void GaugeBeamSearch::Finish(const std::vector<Matrix2>& m, const std::vector<Matrix2>& mm, bool insert_memo) {
    if (m.size() != compute_row_.size() || mm.size() != option_mats_.size()) {
        throw std::invalid_argument("GaugeBeamSearch::Finish: inconsistent matrix counts");
    }
    option_costs_.assign(option_phis_.size(), 0);
    std::vector<MemoOptions> computed(compute_row_.size());
    for (std::size_t c = 0; c < compute_row_.size(); ++c) {
        const std::size_t r = static_cast<std::size_t>(compute_row_[c]);
        const int run_length = run_length_[r];
        MemoOptions& opts = computed[c];
        opts.phi_in = unique_[static_cast<std::size_t>(compute_unique_[c])].phi_in;
        for (int j = option_offsets_[c]; j < option_offsets_[c + 1]; ++j) {
            const int idx = option_mm_[static_cast<std::size_t>(j)];
            const bool changed = idx >= 0 || rotated_[c] != 0 || base_changed_[r] != 0;
            const auto len = FastLength(idx >= 0 ? mm[static_cast<std::size_t>(idx)] : m[c], model_);
            if (!len) {
                throw std::domain_error("GaugeBeamSearch: run length not reproducible");
            }
            int cost = *len;
            if (!changed && run_length >= 0) {
                cost = std::min(cost, run_length);
            }
            cost = gate_weight_ * cost + (cost != 0 ? run_weight_ : 0);
            option_costs_[static_cast<std::size_t>(j)] = cost;
            opts.phis.push_back(option_phis_[static_cast<std::size_t>(j)]);
            opts.costs.push_back(cost);
        }
    }
    std::vector<std::pair<std::int64_t, Entry>> nxt;
    for (std::size_t r = 0; r < slots_.size(); ++r) {
        const std::size_t slot = static_cast<std::size_t>(slots_[r]);
        auto& prev = current_[slot];
        nxt.clear();
        for (std::size_t pi = 0; pi < prev.size(); ++pi) {
            const std::int64_t total = prev[pi].total;
            const Unique& u = unique_[static_cast<std::size_t>(entry_unique_[static_cast<std::size_t>(entry_begin_[r]) + pi])];
            const MemoOptions& opts = u.hit != nullptr ? *u.hit : computed[static_cast<std::size_t>(u.compute)];
            for (std::size_t j = 0; j < opts.phis.size(); ++j) {
                const double phi = opts.phis[j];
                const int c = opts.costs[j];
                const std::int64_t akey = PyRound(dm::Remainder(phi, kTwoPi) * 1e8);
                auto it = std::find_if(nxt.begin(), nxt.end(), [akey](const auto& kv) { return kv.first == akey; });
                const Entry entry{total + c, phi, static_cast<int>(pi), c};
                if (it == nxt.end()) {
                    nxt.emplace_back(akey, entry);
                } else if (total + c < it->second.total) {
                    it->second = entry;  // dict 赋值保留原插入位置
                }
            }
        }
        std::vector<Entry> beam;
        beam.reserve(nxt.size());
        for (const auto& kv : nxt) {
            beam.push_back(kv.second);
        }
        std::stable_sort(beam.begin(), beam.end(), [](const Entry& a, const Entry& b) {
            if (a.total != b.total) {
                return a.total < b.total;
            }
            return std::fabs(a.phi) < std::fabs(b.phi);
        });
        if (beam.size() > static_cast<std::size_t>(beam_)) {
            beam.resize(static_cast<std::size_t>(beam_));
        }
        prev = beam;
        layers_[slot].push_back(std::move(beam));
    }
    // 先更新束（命中项的指针指向 memo 内部，写入新项可能使 unordered_map 重新散列，但不会使
    // 已有元素的引用失效），再写入 / 挂起本层算出的选项。
    for (std::size_t c = 0; c < compute_row_.size(); ++c) {
        const Unique& u = unique_[static_cast<std::size_t>(compute_unique_[c])];
        if (insert_memo) {
            MemoPut(u.key, std::move(computed[c]));
        } else {
            pending_[static_cast<std::size_t>(slots_[static_cast<std::size_t>(u.row)])].emplace_back(
                u.key, std::move(computed[c]));
        }
    }
}

bool GaugeBeamSearch::CommitSlots(const std::vector<int>& slots) {
    for (const int slot : slots) {
        if (slot < 0 || static_cast<std::size_t>(slot) >= pending_.size()) {
            throw std::invalid_argument("GaugeBeamSearch::CommitSlots: slot out of range");
        }
        for (const auto& kv : pending_[static_cast<std::size_t>(slot)]) {
            const MemoOptions* old = MemoGet(kv.first);
            if (old != nullptr && Bits(old->phi_in) != Bits(kv.second.phi_in)) {
                return false;
            }
        }
    }
    for (const int slot : slots) {
        for (auto& kv : pending_[static_cast<std::size_t>(slot)]) {
            MemoPut(kv.first, std::move(kv.second));
        }
        pending_[static_cast<std::size_t>(slot)].clear();
    }
    return true;
}

const GaugeBeamSearch::MemoOptions* GaugeBeamSearch::MemoGet(const MemoKey& key) const {
    const auto it = memo_.find(key);
    return it == memo_.end() ? nullptr : &it->second;
}

void GaugeBeamSearch::MemoPut(const MemoKey& key, MemoOptions options) {
    memo_.emplace(key, std::move(options));  // 已有的键保持不变（首次写入为准）
}

void GaugeBeamSearch::AbsorbMemo(GaugeBeamSearch& other) {
    if (memo_.empty()) {
        memo_.swap(other.memo_);
        return;
    }
    memo_.reserve(memo_.size() + other.memo_.size());
    for (auto& kv : other.memo_) {
        memo_.emplace(kv.first, std::move(kv.second));
    }
    other.memo_.clear();
}

std::int64_t GaugeBeamSearch::Backtrack(int slot, std::vector<double>* angles, std::vector<int>* costs) const {
    if (slot < 0 || static_cast<std::size_t>(slot) >= layers_.size()
        || layers_[static_cast<std::size_t>(slot)].empty()) {
        throw std::invalid_argument("GaugeBeamSearch::Backtrack: no layers for this slot");
    }
    const auto& layers = layers_[static_cast<std::size_t>(slot)];
    const std::size_t n = layers.size();
    angles->assign(n, 0.0);
    costs->assign(n, 0);
    int best = 0;
    for (std::size_t k = n; k-- > 0;) {
        const Entry& e = layers[k][static_cast<std::size_t>(best)];
        (*angles)[k] = e.phi;
        (*costs)[k] = e.cost;
        best = e.parent;
    }
    return layers[n - 1][0].total;
}

}  // namespace quspect::native::transpiler
