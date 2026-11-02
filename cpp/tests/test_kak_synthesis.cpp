// 合成（synthesis/：两比特 KAK、单比特合成、门矩阵、块合并的辅助函数，均在确定性数值层上）的 gtest。
//
// 这里直接检查算法：KAK 分解能重建输入、坐标落在 Weyl 室内、合成出的门序列等于输入（相差全局相位）
// 且纠缠门数最少、欧拉合成正确、门矩阵与定义一致。跨平台逐位一致性由 tools/det_fingerprint.py 与
// Python 测试中的金标准摘要检查。
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/synthesis/kak_synthesis.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/synthesis/gate_matrices.hpp"
#include "quspect_native/synthesis/one_qubit_synthesis.hpp"
#include "quspect_native/synthesis/three_qubit.hpp"

namespace {

using namespace quspect::native::synthesis;
using I = std::int64_t;
using C = std::complex<double>;

C ToC(const Cx& z) {
    return {z.re, z.im};
}
Cx FromC(const C& z) {
    return {z.real(), z.imag()};
}

// ---- 常量（与 two_qubit.py 相同的定义）--------------------------------------------

using CM = std::vector<C>;  // 行主序方阵

CM MatMul(const CM& a, const CM& b) {
    const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.size()))));
    CM out(a.size(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                out[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    return out;
}

CM Kron(const CM& a, const CM& b) {
    CM out(16);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    out[(i * 2 + k) * 4 + j * 2 + l] = a[i * 2 + j] * b[k * 2 + l];
                }
            }
        }
    }
    return out;
}

CM Dagger(const CM& a) {
    const auto n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(a.size()))));
    CM out(a.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out[j * n + i] = std::conj(a[i * n + j]);
        }
    }
    return out;
}

template <std::size_t N>
std::array<Cx, N> ToArr(const CM& m) {
    std::array<Cx, N> out;
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = FromC(m[i]);
    }
    return out;
}

template <std::size_t N>
CM FromArr(const std::array<Cx, N>& m) {
    CM out(N);
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = ToC(m[i]);
    }
    return out;
}

const double kS = 1.0 / std::sqrt(2.0);
const CM kI2{1, 0, 0, 1};
const CM kX{0, 1, 1, 0};
const CM kY{0, C(0, -1), C(0, 1), 0};
const CM kZ{1, 0, 0, -1};
const CM kH{kS, kS, kS, -kS};
const CM kSg{1, 0, 0, C(0, 1)};
const CM kRx90{kS, C(0, -kS), C(0, -kS), kS};
const CM kSX{C(0.5, 0.5), C(0.5, -0.5), C(0.5, -0.5), C(0.5, 0.5)};
const CM kMagic{kS, 0, 0, C(0, kS), 0, C(0, kS), kS, 0, 0, C(0, kS), -kS, 0, kS, 0, 0, C(0, -kS)};
const CM kCZ{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1};
const CM kCX{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
const CM kSwap{1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

// KakConstants::Make() 与 two_qubit.py 的常量矩阵数值相同。
TEST(KakConstants, MatchTheModuleConstants) {
    const KakConstants c = KakConstants::Make();
    auto same = [](const auto& got, const CM& want) {
        for (std::size_t i = 0; i < want.size(); ++i) {
            EXPECT_EQ(ToC(got[i]), want[i]) << i;
        }
    };
    same(c.magic, kMagic);
    same(c.magic_dag, Dagger(kMagic));
    same(c.h, kH);
    same(c.sx, kSX);
    same(c.cz4, kCZ);
    same(c.swap4, kSwap);
    same(c.pauli_pairs[1], Kron(kY, kY));
    same(c.swap_ab, Kron(kSg, kSg));
    same(c.swap_bc, Kron(kRx90, kRx90));
    same(c.flip_ac, Kron(kY, kI2));
    const double phases[4][3] = {{1, -1, 1}, {1, 1, -1}, {-1, -1, -1}, {-1, 1, 1}};
    for (int r = 0; r < 4; ++r) {
        EXPECT_EQ(c.phase_map[static_cast<std::size_t>(r * 4)], 1.0);
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(c.phase_map[static_cast<std::size_t>(r * 4 + 1 + k)], phases[r][k]);
            EXPECT_EQ(c.can_phases[static_cast<std::size_t>(r * 3 + k)], phases[r][k]);
        }
    }
}

KakSynthesizer MakeSynth() {
    KakSynthesizer s;
    EntanglerSpec cz;
    cz.name = "cz";
    cz.matrix = ToArr<16>(kCZ);
    cz.before = {ToArr<4>(kI2), ToArr<4>(kI2)};
    cz.after = {ToArr<4>(kI2), ToArr<4>(kI2)};
    cz.symmetric = true;
    EXPECT_EQ(s.AddEntangler(cz), 0);
    EntanglerSpec cx;
    cx.name = "cx";
    cx.matrix = ToArr<16>(kCX);
    cx.before = {ToArr<4>(kI2), ToArr<4>(kH)};
    cx.after = {ToArr<4>(kI2), ToArr<4>(kH)};
    cx.symmetric = false;
    EXPECT_EQ(s.AddEntangler(cx), 1);
    return s;
}

CM RandomUnitary(std::mt19937_64& rng, int n) {
    std::normal_distribution<double> normal;
    CM a(static_cast<std::size_t>(n * n));
    for (C& z : a) {
        z = C(normal(rng), normal(rng));
    }
    // Gram–Schmidt（按列）
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            C d = 0;
            for (int i = 0; i < n; ++i) {
                d += std::conj(a[i * n + k]) * a[i * n + j];
            }
            for (int i = 0; i < n; ++i) {
                a[i * n + j] -= d * a[i * n + k];
            }
        }
        double norm = 0;
        for (int i = 0; i < n; ++i) {
            norm += std::norm(a[i * n + j]);
        }
        for (int i = 0; i < n; ++i) {
            a[i * n + j] /= std::sqrt(norm);
        }
    }
    return a;
}

CM Canonical(double a, double b, double c) {
    // exp(i(a XX + b YY + c ZZ))：在 magic 基下为对角阵
    const double phases[4][3] = {{1, -1, 1}, {1, 1, -1}, {-1, -1, -1}, {-1, 1, 1}};
    CM d(16, 0.0);
    for (int r = 0; r < 4; ++r) {
        d[r * 5] = std::exp(C(0, phases[r][0] * a + phases[r][1] * b + phases[r][2] * c));
    }
    return MatMul(MatMul(kMagic, d), Dagger(kMagic));
}

double DistanceUpToPhase(const CM& u, const CM& v) {
    C overlap = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        overlap += std::conj(v[i]) * u[i];
    }
    const C phase = overlap / std::abs(overlap);
    double d = 0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        d = std::max(d, std::abs(u[i] - phase * v[i]));
    }
    return d;
}

CM OpsUnitary(const std::vector<Op>& ops, const CM& entangler) {
    CM u = Kron(kI2, kI2);
    for (const Op& op : ops) {
        CM g;
        if (op.kind == OpKind::kEntangler) {
            g = op.q0 == 0 ? entangler : MatMul(MatMul(kSwap, entangler), kSwap);
        } else {
            CM m;
            switch (op.kind) {
                case OpKind::kRz:
                    m = {std::exp(C(0, -op.param / 2)), 0, 0, std::exp(C(0, op.param / 2))};
                    break;
                case OpKind::kH:
                    m = kH;
                    break;
                case OpKind::kSX:
                    m = kSX;
                    break;
                default:
                    m = kX;
                    break;
            }
            g = op.q0 == 0 ? Kron(m, kI2) : Kron(kI2, m);
        }
        u = MatMul(g, u);
    }
    return u;
}

int CountEntanglers(const std::vector<Op>& ops) {
    return static_cast<int>(std::count_if(ops.begin(), ops.end(), [](const Op& o) { return o.kind == OpKind::kEntangler; }));
}

// ---- 测试 ----------------------------------------------------------------------

TEST(GateMatrices, MatchTheDefinitions) {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> ang(-7.0, 7.0);
    auto close = [](const Cx* got, const CM& want) {
        for (std::size_t i = 0; i < want.size(); ++i) {
            EXPECT_LT(std::abs(ToC(got[i]) - want[i]), 1e-15) << i;
        }
    };
    Cx out[64];
    int dim = 0;
    for (int trial = 0; trial < 20; ++trial) {
        const double t = ang(rng);
        const double p = ang(rng);
        const double l = ang(rng);
        const double c = std::cos(t / 2);
        const double s = std::sin(t / 2);
        ASSERT_TRUE(GateMatrix("rx", &t, 1, out, &dim));
        EXPECT_EQ(dim, 2);
        close(out, {c, C(0, -s), C(0, -s), c});
        ASSERT_TRUE(GateMatrix("ry", &t, 1, out, &dim));
        close(out, {c, -s, s, c});
        ASSERT_TRUE(GateMatrix("rz", &t, 1, out, &dim));
        close(out, {std::exp(C(0, -t / 2)), 0, 0, std::exp(C(0, t / 2))});
        ASSERT_TRUE(GateMatrix("p", &t, 1, out, &dim));
        close(out, {1, 0, 0, std::exp(C(0, t))});
        const double u3p[3] = {t, p, l};
        const CM u3{c, -s * std::exp(C(0, l)), s * std::exp(C(0, p)), c * std::exp(C(0, p + l))};
        ASSERT_TRUE(GateMatrix("u3", u3p, 3, out, &dim));
        close(out, u3);
        ASSERT_TRUE(GateMatrix("cu3", u3p, 3, out, &dim));
        EXPECT_EQ(dim, 4);
        close(out, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, u3[0], u3[1], 0, 0, u3[2], u3[3]});
        ASSERT_TRUE(GateMatrix("cp", &t, 1, out, &dim));
        close(out, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, std::exp(C(0, t))});
    }
    ASSERT_TRUE(GateMatrix("h", nullptr, 0, out, &dim));
    close(out, kH);
    ASSERT_TRUE(GateMatrix("sx", nullptr, 0, out, &dim));
    close(out, kSX);
    ASSERT_TRUE(GateMatrix("cx", nullptr, 0, out, &dim));
    close(out, kCX);
    ASSERT_TRUE(GateMatrix("ccx", nullptr, 0, out, &dim));
    EXPECT_EQ(dim, 8);
    EXPECT_EQ(out[55].re, 1.0);
    EXPECT_EQ(out[63].re, 0.0);
    // T = diag(1, e^{iπ/4})：正确舍入的常数
    ASSERT_TRUE(GateMatrix("t", nullptr, 0, out, &dim));
    EXPECT_EQ(out[3].re, 0x1.6a09e667f3bcdp-1);
    EXPECT_EQ(out[3].im, 0x1.6a09e667f3bccp-1);
    // 未知门名与参数个数不对
    EXPECT_FALSE(GateMatrix("foo", nullptr, 0, out, &dim));
    EXPECT_FALSE(GateMatrix("rx", nullptr, 0, out, &dim));
    const double one = 1.0;
    EXPECT_FALSE(GateMatrix("h", &one, 1, out, &dim));
}

TEST(KakSynthesizer, VerifiedEulerAndBlockUnitary) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(31);
    for (int i = 0; i < 20; ++i) {
        const CM m = RandomUnitary(rng, 2);
        for (int basis : {0, 1}) {
            const auto ops = s.VerifiedEuler(ToArr<4>(m), 1e-9, basis);
            ASSERT_TRUE(ops.has_value());
            EXPECT_LT(DistanceUpToPhase(FromArr(s.OpsLayerMatrix(*ops, 0, ops->size())), m), 1e-9);
        }
    }
    // 块：4×4 门、两个单比特门、反向的 4×4 门、swap4 共轭
    const CM g4 = RandomUnitary(rng, 4);
    const CM a = RandomUnitary(rng, 2);
    const CM b = RandomUnitary(rng, 2);
    const M4 g4a = ToArr<16>(g4);
    const M2 aa = ToArr<4>(a);
    const M2 ba = ToArr<4>(b);
    const std::vector<KakSynthesizer::BlockMember> members = {
        {g4a.data(), 0}, {aa.data(), 1}, {ba.data(), 2}, {g4a.data(), 3}, {g4a.data(), 4}};
    const CM swapped = MatMul(MatMul(kSwap, g4), kSwap);
    CM ref = MatMul(Kron(kI2, b), MatMul(Kron(a, kI2), g4));
    ref = MatMul(swapped, ref);
    ref = MatMul(swapped, ref);
    const M4 got = s.BlockUnitary(members);
    for (int i = 0; i < 16; ++i) {
        EXPECT_LT(std::abs(ToC(got[i]) - ref[i]), 1e-14);
    }
}

TEST(KakSynthesizer, KakReconstructsAndLiesInTheChamber) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 20; ++trial) {
        const CM u = trial == 0 ? kCZ : (trial == 1 ? kSwap : RandomUnitary(rng, 4));
        const Kak kak = s.Kak4(ToArr<16>(u));
        const double a = kak.v[0];
        const double b = kak.v[1];
        const double c = kak.v[2];
        EXPECT_LE(a, 3.141592653589793 / 4 + 1e-12);
        EXPECT_GE(a + 1e-12, b);
        EXPECT_GE(b + 1e-12, std::fabs(c));
        const CM rebuilt = MatMul(MatMul(FromArr(kak.k1), Canonical(a, b, c)), FromArr(kak.k2));
        CM scaled = rebuilt;
        for (C& z : scaled) {
            z *= std::exp(C(0, kak.phase));
        }
        double err = 0;
        for (int i = 0; i < 16; ++i) {
            err = std::max(err, std::abs(scaled[i] - u[i]));
        }
        EXPECT_LT(err, 1e-9) << "trial " << trial;
        // 局部因子可分解为张量积
        const auto f = s.FactorLocal(kak.k1);
        EXPECT_LT(DistanceUpToPhase(Kron(FromArr(f[0]), FromArr(f[1])), FromArr(kak.k1)), 1e-9);
    }
}

TEST(KakSynthesizer, ExactSynthesisUsesTheMinimalEntanglerCount) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(5);
    struct Case {
        CM u;
        int expected;
    };
    const CM local = Kron(RandomUnitary(rng, 2), RandomUnitary(rng, 2));
    const CM local2 = Kron(RandomUnitary(rng, 2), RandomUnitary(rng, 2));
    std::vector<Case> cases = {
        {local, 0},
        {MatMul(MatMul(local, kCZ), local2), 1},
        {MatMul(MatMul(local, Canonical(0.4, 0.2, 0.0)), local2), 2},
        {MatMul(MatMul(local, Canonical(0.5, 0.3, 0.1)), local2), 3},
        {kSwap, 3},
        {RandomUnitary(rng, 4), 3},
    };
    for (int ent = 0; ent < 2; ++ent) {
        for (int euler = 0; euler < 2; ++euler) {
            for (const Case& c : cases) {
                SynthOptions opt;
                opt.entangler = ent;
                opt.euler = euler;
                bool uses_flag = false;
                const std::vector<Op> ops = s.SynthesizeExact(ToArr<16>(c.u), opt, &uses_flag);
                EXPECT_EQ(CountEntanglers(ops), c.expected);
                EXPECT_LT(DistanceUpToPhase(OpsUnitary(ops, ent == 0 ? kCZ : kCX), c.u), 1e-8);
                for (const Op& op : ops) {
                    if (op.kind == OpKind::kH) {
                        EXPECT_EQ(euler, 0);
                    }
                    if (op.kind == OpKind::kSX) {
                        EXPECT_EQ(euler, 1);
                    }
                }
            }
        }
    }
}

TEST(KakSynthesizer, SwapFaceTemplateHasOneInnerRotation) {
    const KakSynthesizer s = MakeSynth();
    CM zz(16, 0.0);
    const double d[4] = {1, -1, -1, 1};
    for (int i = 0; i < 4; ++i) {
        zz[i * 5] = std::exp(C(0, -0.15 * d[i]));
    }
    const CM u = MatMul(kSwap, zz);
    SynthOptions opt;
    opt.entangler = 1;  // cx
    const auto ops = s.SynthesizeSwapFace(ToArr<16>(u), opt);
    ASSERT_TRUE(ops.has_value());
    EXPECT_EQ(CountEntanglers(*ops), 3);
    EXPECT_LT(DistanceUpToPhase(OpsUnitary(*ops, kCX), u), 1e-8);
    // CZ 不在 SWAP 面上
    EXPECT_FALSE(s.SynthesizeSwapFace(ToArr<16>(kCZ), opt).has_value());
    // 模块开关打开时普通合成也用 SWAP 形模板
    opt.swap_face_template = true;
    bool uses_flag = false;
    const auto shaped = s.SynthesizeExact(ToArr<16>(u), opt, &uses_flag);
    EXPECT_TRUE(uses_flag);
    EXPECT_EQ(CountEntanglers(shaped), 3);
}

TEST(KakSynthesizer, EulerOpsReproduceTheMatrix) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(3);
    std::vector<CM> mats = {kI2, kH, kX, kZ, kSX};
    for (int i = 0; i < 10; ++i) {
        mats.push_back(RandomUnitary(rng, 2));
    }
    for (int basis = 0; basis < 2; ++basis) {
        for (const CM& m : mats) {
            std::vector<Op> ops;
            s.EulerOps(ToArr<4>(m), 0, basis, 1e-9, &ops);
            CM rebuilt = kI2;
            for (const Op& op : ops) {
                CM g = op.kind == OpKind::kRz ? CM{std::exp(C(0, -op.param / 2)), 0, 0, std::exp(C(0, op.param / 2))}
                       : op.kind == OpKind::kH ? kH
                       : op.kind == OpKind::kSX ? kSX
                                                : kX;
                rebuilt = MatMul(g, rebuilt);
            }
            EXPECT_LT(DistanceUpToPhase(rebuilt, m), 1e-9);
            EXPECT_LE(ops.size(), 5u);
        }
    }
    std::vector<Op> ops;
    s.EulerOps(ToArr<4>(kI2), 0, 0, 1e-9, &ops);
    EXPECT_TRUE(ops.empty());
}

TEST(KakSynthesizer, ApproximationFilterAndUnitarityCheck) {
    const KakSynthesizer s = MakeSynth();
    std::mt19937_64 rng(9);
    const Kak kak = s.Kak4(ToArr<16>(RandomUnitary(rng, 4)));
    EXPECT_TRUE(s.ApproximationImpossible(kak, 1e-12, 1e-9, 2));
    // CP(1e-6)：去掉相互作用的不保真度约 5e-14，预算 1e-12 时可以近似
    CM cp(16, 0.0);
    cp[0] = cp[5] = cp[10] = 1.0;
    cp[15] = std::exp(C(0, 2e-6));
    EXPECT_FALSE(s.ApproximationImpossible(s.Kak4(ToArr<16>(cp)), 1e-12, 1e-9, 2));
    CM bad = kCZ;
    bad[0] = 2.0;
    EXPECT_THROW(s.CheckUnitary(ToArr<16>(bad)), NotUnitary);
    EXPECT_NO_THROW(s.CheckUnitary(ToArr<16>(kCZ)));
}

// ---- 单比特合成 ------------------------------------------------------------------

std::array<M2, kNumFixedGates> FixedMatrices() {
    const C t = std::exp(C(0, 3.141592653589793 / 4));
    const std::vector<CM> mats = {kH, kX, kY, kZ, kSg, {1, 0, 0, C(0, -1)}, {1, 0, 0, t},
                                  {1, 0, 0, std::conj(t)}, kSX, Dagger(kSX)};
    std::array<M2, kNumFixedGates> out;
    for (int i = 0; i < kNumFixedGates; ++i) {
        out[static_cast<std::size_t>(i)] = ToArr<4>(mats[static_cast<std::size_t>(i)]);
    }
    return out;
}

CM Gate1Matrix(const Gate1& g, const std::array<M2, kNumFixedGates>& fixed) {
    switch (g.kind) {
        case 0:
            return FromArr(fixed[static_cast<std::size_t>(g.index)]);
        case 1:
            return {std::exp(C(0, -g.p0 / 2)), 0, 0, std::exp(C(0, g.p0 / 2))};
        case 2:
            return {std::cos(g.p0 / 2), C(0, -std::sin(g.p0 / 2)), C(0, -std::sin(g.p0 / 2)), std::cos(g.p0 / 2)};
        case 3:
            return {std::cos(g.p0 / 2), -std::sin(g.p0 / 2), std::sin(g.p0 / 2), std::cos(g.p0 / 2)};
        default: {
            const double c = std::cos(g.p0 / 2);
            const double s = std::sin(g.p0 / 2);
            return {c, -s * std::exp(C(0, g.p2)), s * std::exp(C(0, g.p1)), c * std::exp(C(0, g.p1 + g.p2))};
        }
    }
}

TEST(OneQubitSynthesizer, SynthesizesInSeveralBases) {
    const auto fixed = FixedMatrices();
    const OneQubitSynthesizer synth;
    OneQubitBasis zxz;  // {rz, h, x}
    zxz.rz = zxz.h = zxz.x = true;
    zxz.named = {kGateH, kGateX};
    OneQubitBasis zsx;  // {rz, sx, x}
    zsx.rz = zsx.sx = zsx.x = true;
    zsx.named = {kGateX, kGateSX};
    OneQubitBasis u3;
    u3.u3 = true;
    OneQubitBasis clifford_t;  // {h, s, t}：只能表示 Clifford+T
    clifford_t.h = true;
    clifford_t.named = {kGateH, kGateS, kGateT};
    clifford_t.has_word[1 + 4] = true;
    clifford_t.words[1 + 4] = {kGateT};
    clifford_t.has_word[2 + 4] = true;
    clifford_t.words[2 + 4] = {kGateS};
    clifford_t.has_word[3 + 4] = true;
    clifford_t.words[3 + 4] = {kGateS, kGateT};
    std::mt19937_64 rng(17);
    std::vector<CM> mats = {kI2, kH, kX, kSX, kSg};
    for (int i = 0; i < 8; ++i) {
        mats.push_back(RandomUnitary(rng, 2));
    }
    for (const OneQubitBasis* basis : {&zxz, &zsx, &u3}) {
        for (const CM& m : mats) {
            if (basis == &u3 && std::abs(m[1]) < 1e-12 && std::abs(m[0] - m[3]) > 1e-9) {
                // 没有 rz 类门时对角矩阵只能走 rz（与 Python 相同：无法表示）
                EXPECT_FALSE(synth.Synthesize(ToArr<4>(m), *basis).has_value());
                continue;
            }
            const auto gates = synth.Synthesize(ToArr<4>(m), *basis);
            ASSERT_TRUE(gates.has_value());
            CM rebuilt = kI2;
            for (const Gate1& g : *gates) {
                rebuilt = MatMul(Gate1Matrix(g, fixed), rebuilt);
            }
            EXPECT_LT(DistanceUpToPhase(rebuilt, m), 1e-9);
            EXPECT_LE(gates->size(), 5u);
        }
    }
    // 恒等、具名门
    EXPECT_TRUE(synth.Synthesize(ToArr<4>(kI2), zxz)->empty());
    const auto h = synth.Synthesize(ToArr<4>(kH), zxz);
    ASSERT_EQ(h->size(), 1u);
    EXPECT_EQ((*h)[0].index, kGateH);
    // Clifford+T：S·T 可以表示，随机酉矩阵不行
    const CM st = MatMul(FromArr(fixed[kGateT]), FromArr(fixed[kGateS]));
    EXPECT_TRUE(synth.Synthesize(ToArr<4>(st), clifford_t).has_value());
    EXPECT_FALSE(synth.Synthesize(ToArr<4>(mats.back()), clifford_t).has_value());
}

TEST(OneQubitSynthesizer, ZyzAnglesRebuildTheMatrix) {
    const OneQubitSynthesizer synth;
    std::mt19937_64 rng(23);
    for (int i = 0; i < 20; ++i) {
        const CM m = RandomUnitary(rng, 2);
        const auto a = synth.ZyzAngles(ToArr<4>(m));
        const Gate1 g{4, 0, a[0], a[1], a[2]};
        EXPECT_LT(DistanceUpToPhase(Gate1Matrix(g, FixedMatrices()), m), 1e-12);
        EXPECT_GE(a[1], 0.0);
        EXPECT_LT(a[1], 2 * 3.141592653589793);
    }
}

// 厄米特征分解：随机与简并（重特征值）输入，重构误差与正交性。
TEST(ThreeQubit, HermitianEigAndUnitaryEig) {
    namespace dm = quspect::native::detmath;
    std::mt19937_64 rng(53);
    for (int trial = 0; trial < 60; ++trial) {
        const CM q = RandomUnitary(rng, 4);
        std::vector<double> lam = {0.3, -1.2, 0.7, 2.0};
        if (trial % 3 == 1) {
            lam = {0.5, 0.5, -0.25, -0.25};
        } else if (trial % 3 == 2) {
            lam = {1.0, 1.0, 1.0, -1.0};
        }
        CM d(16, 0.0);
        for (int i = 0; i < 4; ++i) {
            d[i * 5] = lam[static_cast<std::size_t>(i)];
        }
        const CM h = MatMul(MatMul(q, d), Dagger(q));
        const M4 ha = ToArr<16>(h);
        double w[4];
        M4 v;
        ASSERT_TRUE(dm::EighHerm(4, ha.data(), w, v.data()));
        const CM vv = FromArr(v);
        const CM vhv = MatMul(MatMul(Dagger(vv), h), vv);
        const CM vdv = MatMul(Dagger(vv), vv);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                EXPECT_LT(std::abs(vhv[i * 4 + j] - (i == j ? C(w[i]) : C(0.0))), 1e-13);
                EXPECT_LT(std::abs(vdv[i * 4 + j] - (i == j ? C(1.0) : C(0.0))), 1e-13);
            }
        }
        // 酉矩阵 Q·diag(e^{iλ})·Q† 的特征分解
        CM e(16, 0.0);
        for (int i = 0; i < 4; ++i) {
            e[i * 5] = std::exp(C(0.0, lam[static_cast<std::size_t>(i)]));
        }
        const CM m = MatMul(MatMul(q, e), Dagger(q));
        const auto eig = UnitaryEig(ToArr<16>(m), 1e-9);
        ASSERT_TRUE(eig.has_value());
        const CM ev = FromArr(eig->first);
        const CM evm = MatMul(MatMul(Dagger(ev), m), ev);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const C want = i == j ? ToC(eig->second[static_cast<std::size_t>(i)]) : C(0.0);
                EXPECT_LT(std::abs(evm[i * 4 + j] - want), 1e-13);
            }
        }
    }
}

// 多路复用分解与余弦–正弦分解重建输入。
TEST(ThreeQubit, DemultiplexAndCosSinReconstruct) {
    std::mt19937_64 rng(59);
    for (int trial = 0; trial < 40; ++trial) {
        const CM a = RandomUnitary(rng, 4);
        const CM b = trial % 4 == 0 ? a : RandomUnitary(rng, 4);
        const auto dmx = Demultiplex(ToArr<16>(a), ToArr<16>(b));
        ASSERT_TRUE(dmx.has_value());
        CM d(16, 0.0);
        CM dd(16, 0.0);
        for (int k = 0; k < 4; ++k) {
            d[k * 5] = std::exp(C(0.0, dmx->phases[static_cast<std::size_t>(k)]));
            dd[k * 5] = std::conj(d[k * 5]);
        }
        const CM v = FromArr(dmx->v);
        const CM w = FromArr(dmx->w);
        const CM ra = MatMul(MatMul(v, d), w);
        const CM rb = MatMul(MatMul(v, dd), w);
        for (int i = 0; i < 16; ++i) {
            EXPECT_LT(std::abs(ra[i] - a[i]), 1e-12);
            EXPECT_LT(std::abs(rb[i] - b[i]), 1e-12);
        }
        const CM u8 = RandomUnitary(rng, 8);
        M8 u;
        for (int i = 0; i < 64; ++i) {
            u[static_cast<std::size_t>(i)] = Cx{u8[i].real(), u8[i].imag()};
        }
        const auto cs = CosSinDecompose(u);
        ASSERT_TRUE(cs.has_value());
        CM left(64, 0.0);
        CM mid(64, 0.0);
        CM right(64, 0.0);
        for (int i = 0; i < 4; ++i) {
            const double t = cs->theta[static_cast<std::size_t>(i)];
            mid[i * 8 + i] = std::cos(t);
            mid[(i + 4) * 8 + i + 4] = std::cos(t);
            mid[i * 8 + i + 4] = -std::sin(t);
            mid[(i + 4) * 8 + i] = std::sin(t);
            for (int j = 0; j < 4; ++j) {
                left[i * 8 + j] = ToC(cs->u1[static_cast<std::size_t>(i * 4 + j)]);
                left[(i + 4) * 8 + j + 4] = ToC(cs->u2[static_cast<std::size_t>(i * 4 + j)]);
                right[i * 8 + j] = ToC(cs->v1h[static_cast<std::size_t>(i * 4 + j)]);
                right[(i + 4) * 8 + j + 4] = ToC(cs->v2h[static_cast<std::size_t>(i * 4 + j)]);
            }
        }
        const CM rebuilt = MatMul(MatMul(left, mid), right);
        for (int i = 0; i < 64; ++i) {
            EXPECT_LT(std::abs(rebuilt[i] - u8[i]), 1e-12);
        }
    }
}

}  // namespace
