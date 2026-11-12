#include "quspect_native/synthesis/gate_matrices.hpp"

#include <algorithm>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/scalar.hpp"

namespace quspect::native::synthesis {

namespace dm = quspect::native::detmath;

namespace {

using M2 = std::array<Cx, 4>;

constexpr double kPi = 3.141592653589793;  // math.pi
// cos(π/4) 与 sin(π/4)（math.pi / 4 的正确舍入值；与各平台 libm 的结果相同）
constexpr double kCosPi4 = 0x1.6a09e667f3bcdp-1;
constexpr double kSinPi4 = 0x1.6a09e667f3bccp-1;

constexpr Cx kZero{0.0, 0.0};
constexpr Cx kOne{1.0, 0.0};

M2 Rx(double theta) {
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    return {Cx{c, 0.0}, Cx{0.0, -s}, Cx{0.0, -s}, Cx{c, 0.0}};
}

M2 Ry(double theta) {
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    return {Cx{c, 0.0}, Cx{-s, 0.0}, Cx{s, 0.0}, Cx{c, 0.0}};
}

M2 Rz(double theta) {
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    return {Cx{c, -s}, kZero, kZero, Cx{c, s}};
}

M2 Phase(double theta) {
    return {kOne, kZero, kZero, Cx{dm::Cos(theta), dm::Sin(theta)}};
}

M2 U3(double theta, double phi, double lam) {
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    const Cx e_ip{dm::Cos(phi), dm::Sin(phi)};
    const Cx e_il{dm::Cos(lam), dm::Sin(lam)};
    return {Cx{c, 0.0}, dm::Scale(e_il, -s), dm::Scale(e_ip, s), dm::Mul(dm::Scale(e_ip, c), e_il)};
}

bool Fixed2(std::string_view name, M2* m) {
    const double h = 1.0 / dm::Sqrt(2.0);
    if (name == "id") {
        *m = {kOne, kZero, kZero, kOne};
    } else if (name == "x") {
        *m = {kZero, kOne, kOne, kZero};
    } else if (name == "y") {
        *m = {kZero, Cx{0.0, -1.0}, Cx{0.0, 1.0}, kZero};
    } else if (name == "z") {
        *m = {kOne, kZero, kZero, Cx{-1.0, 0.0}};
    } else if (name == "h") {
        // np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        *m = {Cx{h, 0.0}, Cx{h, 0.0}, Cx{h, 0.0}, Cx{-h, 0.0}};
    } else if (name == "s") {
        *m = {kOne, kZero, kZero, Cx{0.0, 1.0}};
    } else if (name == "sdg") {
        *m = {kOne, kZero, kZero, Cx{0.0, -1.0}};
    } else if (name == "t") {
        *m = {kOne, kZero, kZero, Cx{kCosPi4, kSinPi4}};
    } else if (name == "tdg") {
        *m = {kOne, kZero, kZero, Cx{kCosPi4, -kSinPi4}};
    } else if (name == "sx") {
        *m = {Cx{0.5, 0.5}, Cx{0.5, -0.5}, Cx{0.5, -0.5}, Cx{0.5, 0.5}};
    } else if (name == "sxdg") {
        *m = {Cx{0.5, -0.5}, Cx{0.5, 0.5}, Cx{0.5, 0.5}, Cx{0.5, -0.5}};
    } else {
        return false;
    }
    return true;
}

bool Param2(std::string_view name, const double* p, int n, M2* m) {
    auto need = [n](int k) { return n == k; };
    if (name == "rx") {
        if (!need(1)) return false;
        *m = Rx(p[0]);
    } else if (name == "ry") {
        if (!need(1)) return false;
        *m = Ry(p[0]);
    } else if (name == "rz") {
        if (!need(1)) return false;
        *m = Rz(p[0]);
    } else if (name == "p" || name == "u1") {
        if (!need(1)) return false;
        *m = Phase(p[0]);
    } else if (name == "u2") {
        if (!need(2)) return false;
        *m = U3(kPi / 2, p[0], p[1]);
    } else if (name == "u3") {
        if (!need(3)) return false;
        *m = U3(p[0], p[1], p[2]);
    } else {
        return false;
    }
    return true;
}

void Identity(int dim, Cx* out) {
    std::fill(out, out + dim * dim, kZero);
    for (int i = 0; i < dim; ++i) {
        out[i * dim + i] = kOne;
    }
}

// exp(−i·θ/2·P⊗P) = c·I − i·s·P⊗P（P = X / Y / Z，按 axis 选择）。
void PauliPairRotation(char axis, double theta, Cx* out) {
    const double c = dm::Cos(theta / 2);
    const double s = dm::Sin(theta / 2);
    std::fill(out, out + 16, kZero);
    if (axis == 'z') {
        // Z⊗Z = diag(1, −1, −1, 1)
        out[0] = Cx{c, -s};
        out[5] = Cx{c, s};
        out[10] = Cx{c, s};
        out[15] = Cx{c, -s};
        return;
    }
    // X⊗X 与 Y⊗Y 都只翻转两个比特：反对角元为 ±1（Y⊗Y 的 |00⟩↔|11⟩ 为 −1）
    const double corner = axis == 'x' ? -s : s;
    out[0] = out[5] = out[10] = out[15] = Cx{c, 0.0};
    out[3] = out[12] = Cx{0.0, corner};
    out[6] = out[9] = Cx{0.0, -s};
}

void Controlled(const M2& u, Cx* out) {
    Identity(4, out);
    out[10] = u[0];
    out[11] = u[1];
    out[14] = u[2];
    out[15] = u[3];
}

}  // namespace

bool GateMatrix2(std::string_view name, const double* params, int num_params, std::array<Cx, 4>* out) {
    if (Fixed2(name, out)) {
        return num_params == 0;
    }
    return Param2(name, params, num_params, out);
}

bool GateMatrix(std::string_view name, const double* params, int num_params, Cx* out, int* dim) {
    M2 m;
    if (GateMatrix2(name, params, num_params, &m)) {
        *dim = 2;
        std::copy(m.begin(), m.end(), out);
        return true;
    }
    if (name == "cx" || name == "cz" || name == "swap") {
        if (num_params != 0) {
            return false;
        }
        *dim = 4;
        Identity(4, out);
        if (name == "cx") {
            out[10] = kZero;
            out[11] = kOne;
            out[14] = kOne;
            out[15] = kZero;
        } else if (name == "cz") {
            out[15] = Cx{-1.0, 0.0};
        } else {
            out[5] = kZero;
            out[6] = kOne;
            out[9] = kOne;
            out[10] = kZero;
        }
        return true;
    }
    if (name == "ccx") {
        if (num_params != 0) {
            return false;
        }
        *dim = 8;
        Identity(8, out);
        out[54] = kZero;
        out[55] = kOne;
        out[62] = kOne;
        out[63] = kZero;
        return true;
    }
    if (name == "cp") {
        if (num_params != 1) {
            return false;
        }
        *dim = 4;
        Identity(4, out);
        out[15] = Cx{dm::Cos(params[0]), dm::Sin(params[0])};
        return true;
    }
    if (name == "rxx" || name == "ryy" || name == "rzz") {
        if (num_params != 1) {
            return false;
        }
        *dim = 4;
        PauliPairRotation(name[1], params[0], out);
        return true;
    }
    std::string_view inner;
    if (name == "crx") {
        inner = "rx";
    } else if (name == "cry") {
        inner = "ry";
    } else if (name == "crz") {
        inner = "rz";
    } else if (name == "cu3") {
        inner = "u3";
    } else {
        return false;
    }
    if (!Param2(inner, params, num_params, &m)) {
        return false;
    }
    *dim = 4;
    Controlled(m, out);
    return true;
}

}  // namespace quspect::native::synthesis
