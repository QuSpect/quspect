#include "quspect_native/synthesis/numpy_euler.hpp"

#include <algorithm>
#include <cmath>

namespace quspect::native::synthesis {

namespace {

// 与 two_qubit.py 的模块常量相同（math.pi 及其按 2 的幂缩放，均为精确运算）。
constexpr double kPi = 3.141592653589793;
constexpr double kPi2 = kPi / 2;
constexpr double kTwoPi = 2 * kPi;
// 只用于比较的量（numpy np.abs 等）与阈值的相对距离小于该值时退回 Python。
constexpr double kDecisionMargin = 1e-9;

Cx Minus(const Cx& a, const Cx& b) {
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

// ``err <= verify_atol``（err 的最后几位可能与 numpy 不同：落在边界附近时抛出 Fallback）。
bool Verified(double err, double verify_atol) {
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

}  // namespace

NumpyEuler::NumpyEuler(const LinalgFunctions& fns, const ArithModes& modes, const KakConstants& constants)
    : np_(fns, modes), i2_(constants.i2), h_(constants.h), x_(constants.x), sx_(constants.sx) {}

M2 NumpyEuler::Mat2(const M2& a, const M2& b) const {
    M2 out;
    np_.MatMul(2, View(a), View(b), out.data());
    return out;
}

M4 NumpyEuler::Mat4(const M4& a, const M4& b) const {
    M4 out;
    np_.MatMul(4, View(a), View(b), out.data());
    return out;
}

M4 NumpyEuler::Kron2(const M2& left, const M2& right) const {
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

M2 NumpyEuler::Rz(double theta) const {
    // _rz：[[cmath.exp(-0.5j*θ), 0], [0, cmath.exp(0.5j*θ)]]
    const ArithModes& m = np_.modes();
    const Cx d0 = PyExp(m, PyMulReal(m, Cx{-0.0, -0.5}, theta));
    const Cx d1 = PyExp(m, PyMulReal(m, Cx{0.0, 0.5}, theta));
    return {d0, Cx{0.0, 0.0}, Cx{0.0, 0.0}, d1};
}

void NumpyEuler::EulerOps(const M2& matrix, int qubit, int basis, double atol, std::vector<Op>* out) const {
    const ArithModes& md = np_.modes();
    // _zxz_angles
    const Cx det = Minus(Mul(md.np_scalar, matrix[0], matrix[3]), Mul(md.np_scalar, matrix[1], matrix[2]));
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

M2 NumpyEuler::OpsLayerMatrix(const std::vector<Op>& ops) const {
    M2 m = i2_;
    for (const Op& op : ops) {
        switch (op.kind) {
            case OpKind::kRz:
                m = Mat2(Rz(op.param), m);
                break;
            case OpKind::kH:
                m = Mat2(h_, m);
                break;
            case OpKind::kSX:
                m = Mat2(sx_, m);
                break;
            default:
                m = Mat2(x_, m);
                break;
        }
    }
    return m;
}

std::optional<std::vector<Op>> NumpyEuler::VerifiedEuler(const M2& u, double atol, int basis) const {
    for (double tol : {atol, std::min(atol, 1e-12)}) {
        std::vector<Op> ops;
        EulerOps(u, 0, basis, tol, &ops);
        const M2 v = OpsLayerMatrix(ops);
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
            const Cx diff = Minus(u[e], Mul(np_.modes().scalar_arr, phase, v[e]));
            dist = std::max(dist, std::hypot(diff.re, diff.im));
        }
        if (Verified(dist, 1e-9)) {
            return ops;
        }
    }
    return std::nullopt;
}

M4 NumpyEuler::BlockUnitary(const std::vector<BlockMember>& members, const M4& swap4) const {
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
                full = member.slot == 1 ? Kron2(m, i2_) : Kron2(i2_, m);
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

}  // namespace quspect::native::synthesis
