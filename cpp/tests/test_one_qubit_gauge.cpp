// 单比特门最小化标量内核的 gtest：期望值由 Python 参考实现生成（十六进制浮点，逐位比较），
// 生成方式见 tests/transpiler/test_gauge_differential.py（同一批函数的 Python 版本）。
#include <cmath>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/one_qubit_gauge.hpp"

namespace {

using quspect::native::transpiler::AxisRotation;
using quspect::native::transpiler::Complex;
using quspect::native::transpiler::EulerLength;
using quspect::native::transpiler::FastLength;
using quspect::native::transpiler::GaugeBeamSearch;
using quspect::native::transpiler::LeadingAnglesInZFrame;
using quspect::native::transpiler::LengthModel;
using quspect::native::transpiler::Matrix2;
using quspect::native::transpiler::PyRound;

// 由 Python 参考实现生成（quspect.transpiler._gauge / quspect.synthesis.two_qubit）。
const Matrix2 kMats[] = {
    {{{-0x1.73b6d414d9980p-5, 0x1.eebfcf4157f12p-1},
      {0x1.2d092b97a2ed8p-3, -0x1.a6a1650bd3242p-3},
      {-0x1.c73c44e59c082p-6, 0x1.01e05ba5c9b96p-2},
      {-0x1.382a8263b2f28p-1, 0x1.80898abed788cp-1}}},
    {{{-0x1.55f61217a6a4ap-1, 0x1.23b366944ddd6p-1},
      {-0x1.d113f8eee6502p-2, 0x1.36f10982ef1dfp-3},
      {0x1.b8ecd432e687dp-2, 0x1.ad33ed2b4fb05p-3},
      {-0x1.2d255b5e122b6p-1, -0x1.4dac6530b0e93p-1}}},
    {{{-0x1.46e3bc56f82c0p-1, 0x1.0ccd7d85febcap-1},
      {0x1.9a8051e065758p-2, 0x1.947f4513afc7ap-2},
      {0x1.1c79d83eb2c15p-2, 0x1.f533823ac9eb8p-2},
      {-0x1.583ec27dfb861p-1, 0x1.ec5ea182dfd98p-2}}},
    {{{-0x1.56537e145b058p-2, -0x1.079780b247b24p-3},
      {-0x1.c821e35b850d6p-1, 0x1.1dfe1097d51dep-2},
      {0x1.3e56d1eeb3eb2p-1, -0x1.649a108a9edfcp-1},
      {-0x1.51165ce6cb430p-6, 0x1.6e370fe02030ap-2}}},
    {{{-0x1.2372e7ca70140p-6, -0x1.32359e292807ap-1},
      {0x1.bc3ca15bbb8bap-2, -0x1.58e7f79289dacp-1},
      {0x1.7df4e131af6f5p-1, 0x1.2b5f6c15f1c73p-2},
      {-0x1.57a1755f7e3b3p-2, -0x1.fb4107cc776d4p-2}}},
    {{{-0x1.a64791dd44810p-2, 0x1.31e7929369592p-2},
      {-0x1.2cc58c0442b7cp-2, -0x1.9e30be31899b7p-1},
      {-0x1.87060d3d5df90p-1, -0x1.964d06919f3c9p-2},
      {-0x1.666518fdd0f71p-2, 0x1.7abfb84b11e31p-2}}},
    {{{0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {-0x1.6a09e667f3bccp-1, 0x0.0p+0}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}}},
    {{{0x0.0p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}}},
    {{{0x1.65f92ea41b73cp-1, -0x1.b0d1ae309adc8p-4},
      {0x1.65f92ea41b73cp-1, -0x1.b0d1ae309adc8p-4},
      {0x1.65f92ea41b73cp-1, 0x1.b0d1ae309adc8p-4},
      {-0x1.65f92ea41b73cp-1, -0x1.b0d1ae309adc8p-4}}},
    {{{0x1.0000000000000p-1, -0x1.ffffffffffffep-2},
      {0x1.0000000000000p-1, -0x1.ffffffffffffep-2},
      {0x1.0000000000000p-1, 0x1.ffffffffffffep-2},
      {-0x1.0000000000000p-1, -0x1.ffffffffffffep-2}}},
    {{{0x1.0000000000000p+0, -0x1.b7cdfd9d7bdbbp-35},
      {0x0.0p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x1.0000000000000p+0, 0x1.b7cdfd9d7bdbbp-35}}},
    {{{0x1.18f93c949f54cp-54, -0x1.c30ff123ac82ap-58},
      {0x1.98eaecb8bcb2cp-4, -0x1.fd712f9a817c1p-1},
      {-0x1.98eaecb8bcb2cp-4, -0x1.fd712f9a817c1p-1},
      {0x1.18f93c949f54cp-54, 0x1.c30ff123ac82ap-58}}},
};
const int kLenCz[] = {5, 5, 5, 5, 5, 5, 1, 0, 1, 2, 2, 0, 2};
const int kLenIbm[] = {5, 5, 5, 5, 5, 5, 3, 0, 1, 3, 3, 0, 2};
const int kEulerZxz[] = {5, 5, 5, 5, 5, 5, 1, 0, 1, 2, 2, 0, 2};
const int kEulerZsx[] = {5, 5, 5, 5, 5, 5, 3, 0, 1, 3, 3, 0, 2};
const Matrix2 kRotX07 = {{{0x1.e0f575d0de5b7p-1, 0x0.0p+0},
                          {0x0.0p+0, -0x1.5f209a5390a8bp-2},
                          {0x0.0p+0, -0x1.5f209a5390a8bp-2},
                          {0x1.e0f575d0de5b7p-1, 0x0.0p+0}}};
const Matrix2 kRotYm2 = {{{0x1.14a280fb5068cp-1, 0x0.0p+0},
                          {0x1.aed548f090ceep-1, 0x0.0p+0},
                          {-0x1.aed548f090ceep-1, 0x0.0p+0},
                          {0x1.14a280fb5068cp-1, 0x0.0p+0}}};
const Matrix2 kRotZ0 = {
    {{0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}}};
const std::vector<Matrix2> kNamedCZ = {
    {{{0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {0x1.6a09e667f3bccp-1, 0x0.0p+0},
      {-0x1.6a09e667f3bccp-1, 0x0.0p+0}}},
    {{{0x0.0p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}}},
    {{{0x0.0p+0, 0x0.0p+0},
      {-0x0.0p+0, -0x1.0000000000000p+0},
      {0x0.0p+0, 0x1.0000000000000p+0},
      {0x0.0p+0, 0x0.0p+0}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {-0x1.0000000000000p+0, 0x0.0p+0}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}, {0x0.0p+0, 0x1.0000000000000p+0}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {-0x0.0p+0, -0x1.0000000000000p+0}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bccp-1}}},
    {{{0x1.0000000000000p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x0.0p+0, 0x0.0p+0},
      {0x1.6a09e667f3bcdp-1, -0x1.6a09e667f3bccp-1}}}};
const std::vector<Matrix2> kNamedIBM = {
    {{{0x0.0p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x1.0000000000000p+0, 0x0.0p+0}, {0x0.0p+0, 0x0.0p+0}}},
    {{{0x1.0000000000000p-1, 0x1.0000000000000p-1},
      {0x1.0000000000000p-1, -0x1.0000000000000p-1},
      {0x1.0000000000000p-1, -0x1.0000000000000p-1},
      {0x1.0000000000000p-1, 0x1.0000000000000p-1}}}};
// x 轴前导角：z = V†·m·V 与 _leading_angles(m, 'x')
const Matrix2 kFramesX[] = {
    {{{-0x1.125d86e644b0dp-2, 0x1.c34897480fc6bp-1},
      {0x1.8ee5d02a9f7abp-3, 0x1.58cecb985a0e2p-2},
      {0x1.7a6b422f7af49p-2, -0x1.f1890a4d64f5ap-4},
      {-0x1.8c6e5863bc671p-2, 0x1.ac00c2b81fb30p-1}}},
    {{{-0x1.47977fe9dc5a0p-1, 0x1.1e207e1e594f5p-3},
      {0x1.9c2fafd751f2ap-2, 0x1.477842578b758p-1},
      {-0x1.edd11d4a7ae53p-2, 0x1.29e7896d7350ep-1},
      {-0x1.3b83ed8bdc75fp-1, -0x1.c604788fe57ebp-3}}},
    {{{-0x1.43a569c56796ap-2, 0x1.e3eb18f755c16p-1},
      {-0x1.6d41b54eb0003p-5, 0x1.1be12d606f873p-4},
      {0x1.41790bdf72d0ap-4, -0x1.9bbf1cefe420ep-6},
      {-0x1.fd4fc9f23fe6ap-1, 0x1.f11b55018e7e5p-5}}},
    {{{-0x1.3f7d835e34ff0p-2, -0x1.819581ecd8941p-4},
      {0x1.32ebd493a0e7dp-1, -0x1.764d407995b9bp-1},
      {-0x1.d38ce0b698108p-1, 0x1.f52f6173ced38p-3},
      {-0x1.5f3b042495d45p-5, 0x1.4ad0b002327c8p-2}}},
    {{{0x1.a726dff17a96dp-2, -0x1.798731cb7a5aap-1},
      {0x1.428bb3f53d465p-2, 0x1.ba02935a966d5p-2},
      {0x1.6f4976ccf9bb1p-9, -0x1.119663f037879p-1},
      {-0x1.887fc1e6cfe9ap-1, -0x1.6c9de087d2c6fp-2}}},
    {{{-0x1.d1df945685086p-1, -0x1.13039c0b1b9bap-2},
      {-0x1.109483aaf6621p-2, 0x1.5d3c5019cb707p-3},
      {0x1.a164159705b03p-3, -0x1.eeec9b891c844p-3},
      {0x1.2e24fba3e9319p-3, 0x1.dfd57374cb6bdp-1}}},
    {{{0x1.6a09e667f3bcbp-1, 0x0.0p+0},
      {0x1.6a09e667f3bcbp-1, 0x0.0p+0},
      {0x1.6a09e667f3bcap-1, 0x0.0p+0},
      {-0x1.6a09e667f3bcbp-1, 0x0.0p+0}}},
    {{{0x1.ffffffffffffep-1, 0x0.0p+0},
      {-0x1.9cacbc97eaa40p-56, 0x0.0p+0},
      {-0x1.9cacbc97eaa40p-56, 0x0.0p+0},
      {0x1.ffffffffffffep-1, 0x0.0p+0}}},
    {{{0x1.ffffffffffffep-1, 0x0.0p+0},
      {-0x1.9cacbc97eaa40p-56, 0x0.0p+0},
      {0x1.9cacbc97eaa40p-56, 0x0.0p+0},
      {-0x1.ffffffffffffep-1, 0x0.0p+0}}},
    {{{0x1.65f92ea41b73bp-1, -0x1.b0d1ae309adc6p-4},
      {0x1.65f92ea41b73bp-1, 0x1.b0d1ae309adc7p-4},
      {0x1.65f92ea41b73bp-1, -0x1.b0d1ae309adc6p-4},
      {-0x1.65f92ea41b73bp-1, -0x1.b0d1ae309adc6p-4}}},
    {{{0x1.ffffffffffffep-2, -0x1.ffffffffffffdp-2},
      {0x1.ffffffffffffep-2, 0x1.ffffffffffffdp-2},
      {0x1.ffffffffffffep-2, -0x1.ffffffffffffdp-2},
      {-0x1.ffffffffffffep-2, -0x1.ffffffffffffdp-2}}},
    {{{0x1.ffffffffffffep-1, 0x1.0c6e992014040p-92},
      {-0x1.9cacbc97eaa40p-56, -0x1.b7cdfd9d7bdbap-35},
      {-0x1.9cacbc97eaa40p-56, -0x1.b7cdfd9d7bdbap-35},
      {0x1.ffffffffffffep-1, 0x1.0c6e992014040p-92}}},
    {{{0x1.053b822f1716ap-54, -0x1.fd712f9a817bfp-1},
      {-0x1.98eaecb8bcb2bp-4, 0x1.7d4fdaed94980p-56},
      {0x1.98eaecb8bcb2bp-4, -0x1.7d4fdaed94980p-56},
      {0x1.0a4bea9edfb6fp-54, 0x1.fd712f9a817bfp-1}}},
};
const double kLeadX[][4] = {
    {0x1.0661b2726d188p+2, 0x1.6ae99fc37dccep+2, 0x1.43b38a42b8c84p+1, 0x1.cf718d148e814p+2},
    {0x1.67a4e0325ffc6p+2, 0x1.cc2ccd8370b0cp+2, 0x1.031cf2e14f480p+2, 0x1.185a5d6a40b29p+3},
    {0x1.051b479e2420dp+2, 0x1.69a334ef34d53p+2, 0x1.4126b49a26d8ep+1, 0x1.ce2b224045899p+2},
    {0x1.6e9cce119a44cp+2, 0x1.d324bb62aaf92p+2, 0x1.0a14e0c089906p+2, 0x1.1bd65459ddd6cp+3},
    {0x1.71c47e8cc626ep+2, 0x1.d64c6bddd6db4p+2, 0x1.0d3c913bb5728p+2, 0x1.1d6a2c9773c7dp+3},
    {0x1.fc01d1354628cp+0, 0x1.c710c33cc47d2p+1, 0x1.a7886fc40d5d0p-2, 0x1.48104eef72f2fp+2},
    {0x0.0p+0, 0x1.921fb54442d18p+0, -0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1},
    {0x0.0p+0, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0x1.921fb54442d18p+1, 0x0.0p+0, 0x0.0p+0, 0x0.0p+0},
    {0x0.0p+0, 0x1.921fb54442d18p+0, -0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1},
    {0x0.0p+0, 0x1.921fb54442d18p+0, -0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1},
    {0x1.2d97c7f3321d2p+2, 0x1.921fb54442d18p+2, 0x1.921fb54442d18p+1, 0x1.f6a7a2955385ep+2},
    {0x1.921fb54442d18p+0, 0x1.921fb54442d18p+1, 0x0.0p+0, 0x1.2d97c7f3321d2p+2},
};
const int kLeadXCount[] = {4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4};

constexpr int kNumMats = static_cast<int>(sizeof(kMats) / sizeof(kMats[0]));

bool SameBits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

void ExpectSameMatrix(const Matrix2& got, const Matrix2& want) {
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(SameBits(got[i].re, want[i].re)) << "entry " << i;
        EXPECT_TRUE(SameBits(got[i].im, want[i].im)) << "entry " << i;
    }
}

LengthModel CzModel() {
    LengthModel model;
    model.h = true;
    model.x = true;
    model.named = kNamedCZ;
    return model;
}

LengthModel IbmModel() {
    LengthModel model;
    model.x = true;
    model.sx = true;
    model.named = kNamedIBM;
    return model;
}

TEST(OneQubitGauge, FastLengthMatchesPython) {
    const LengthModel cz = CzModel();
    const LengthModel ibm = IbmModel();
    for (int i = 0; i < kNumMats; ++i) {
        EXPECT_EQ(FastLength(kMats[i], cz).value(), kLenCz[i]) << "matrix " << i;
        EXPECT_EQ(FastLength(kMats[i], ibm).value(), kLenIbm[i]) << "matrix " << i;
    }
}

TEST(OneQubitGauge, EulerLengthMatchesPython) {
    for (int i = 0; i < kNumMats; ++i) {
        EXPECT_EQ(EulerLength(kMats[i], 0, 1e-9).value(), kEulerZxz[i]) << "matrix " << i;
        EXPECT_EQ(EulerLength(kMats[i], 1, 1e-9).value(), kEulerZsx[i]) << "matrix " << i;
    }
    EXPECT_FALSE(EulerLength(kMats[0], 7, 1e-9).has_value());
}

// Rx(θ)：θ 恰好等于 atol 时 θ 的最后几位取决于 atan2 的实现，必须交回 Python。
Matrix2 RxMatrix(double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    return {Complex{c, 0.0}, Complex{0.0, -s}, Complex{0.0, -s}, Complex{c, 0.0}};
}

TEST(OneQubitGauge, EulerLengthDefersNearThreshold) {
    for (int basis = 0; basis < 2; ++basis) {
        EXPECT_FALSE(EulerLength(RxMatrix(1e-9), basis, 1e-9).has_value());
        EXPECT_FALSE(EulerLength(RxMatrix(1e-12), basis, 1e-12).has_value());
        EXPECT_FALSE(EulerLength(RxMatrix(std::acos(-1.0) / 2.0 + 1e-9), basis, 1e-9).has_value());
        // 远离阈值：照常给出长度（与 len(euler_ops(...)) 相同）
        EXPECT_EQ(EulerLength(RxMatrix(1e-7), basis, 1e-9).value(), basis == 0 ? 3 : 5);
        EXPECT_EQ(EulerLength(RxMatrix(1e-11), basis, 1e-9).value(), 0);
    }
}

TEST(OneQubitGauge, LeadingAnglesMatchPython) {
    for (int i = 0; i < kNumMats; ++i) {
        const auto lead = LeadingAnglesInZFrame(kFramesX[i]);
        ASSERT_TRUE(lead.has_value());
        ASSERT_EQ(lead->count, kLeadXCount[i]) << "matrix " << i;
        for (int j = 0; j < lead->count; ++j) {
            EXPECT_TRUE(SameBits(lead->angles[j], kLeadX[i][j])) << "matrix " << i << " angle " << j;
        }
    }
}

TEST(OneQubitGauge, AxisRotationMatchesNumpy) {
    ExpectSameMatrix(AxisRotation(1, 0.7), kRotX07);
    ExpectSameMatrix(AxisRotation(2, -2.0), kRotYm2);
    ExpectSameMatrix(AxisRotation(0, 0.0), kRotZ0);
}

TEST(OneQubitGauge, NonFiniteInputIsNotReproduced) {
    Matrix2 m = kMats[0];
    m[1].re = std::nan("");
    EXPECT_FALSE(FastLength(m, CzModel()).has_value());
    EXPECT_FALSE(LeadingAnglesInZFrame(m).has_value());
    EXPECT_FALSE(EulerLength(m, 0, 1e-9).has_value());
}

TEST(OneQubitGauge, PyRoundIsHalfEven) {
    EXPECT_EQ(PyRound(0.5), 0);
    EXPECT_EQ(PyRound(1.5), 2);
    EXPECT_EQ(PyRound(2.5), 2);
    EXPECT_EQ(PyRound(-2.5), -2);
    EXPECT_EQ(PyRound(-0.4), 0);
    EXPECT_EQ(PyRound(31415926535.7), 31415926536);
}

// 一条线、两个运行（中间一个 CZ，Z 轴可穿过）：rz(0.3) | CZ | rz(-0.3) 的最优解是把 rz(0.3)
// 推过 CZ 与后面的 rz(-0.3) 抵消（两个运行都变成空）。
Matrix2 DiagProduct(const Matrix2& a, const Matrix2& b) {
    // 两个对角矩阵相乘（只用于测试，不要求与 numpy 逐位一致）
    const auto mul = [](const Complex& x, const Complex& y) {
        return Complex{x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re};
    };
    return {mul(a[0], b[0]), Complex{0.0, 0.0}, Complex{0.0, 0.0}, mul(a[3], b[3])};
}

void RunTwoRunChain(GaugeBeamSearch* search, bool insert_memo) {
    const Matrix2 rz = AxisRotation(0, 0.3);
    search->Prepare({0}, {0}, {0}, {0}, {-1}, {0}, {0}, {0.0}, {1}, {0});
    ASSERT_EQ(search->ComputeRow().size(), 1u);
    EXPECT_TRUE(search->RotateIn().empty());
    ASSERT_EQ(search->NeedLeading().size(), 1u);
    search->Options({rz});  // z 轴：V = I，z = m
    std::vector<Matrix2> mm;
    for (const Matrix2& r : search->OptionMatrices()) {
        mm.push_back(DiagProduct(r, rz));
    }
    search->Finish({rz}, mm, insert_memo);
    const Matrix2 rz_back = AxisRotation(0, -0.3);
    search->Prepare({0}, {1}, {0}, {0}, {0}, {-1}, {1}, {0.0}, {1}, {0});
    std::vector<Matrix2> m;
    const auto& rows = search->ComputeRow();
    const auto& rot = search->RotateInMatrices();
    std::size_t k = 0;
    for (std::size_t c = 0; c < rows.size(); ++c) {
        const bool rotated = k < search->RotateIn().size() && search->RotateIn()[k] == static_cast<int>(c);
        m.push_back(rotated ? DiagProduct(rz_back, rot[k++]) : rz_back);
    }
    search->Options({});
    search->Finish(m, {}, insert_memo);
}

TEST(OneQubitGauge, BeamSearchMovesRotationThroughAnchor) {
    GaugeBeamSearch search(1, 4, 2, 1, CzModel());
    RunTwoRunChain(&search, true);
    std::vector<double> angles;
    std::vector<int> costs;
    EXPECT_EQ(search.Backtrack(0, &angles, &costs), 0);
    ASSERT_EQ(angles.size(), 2u);
    EXPECT_NEAR(angles[0], 0.3, 1e-12);
    EXPECT_EQ(costs[0], 0);
    EXPECT_EQ(costs[1], 0);
    // 第二遍全部命中 memo：没有计算项
    search.Reset({0.0});
    search.Prepare({0}, {0}, {0}, {0}, {-1}, {0}, {0}, {0.0}, {1}, {0});
    EXPECT_TRUE(search.ComputeRow().empty());
}

TEST(OneQubitGauge, PendingMemoEntriesCommitInOrder) {
    GaugeBeamSearch search(1, 4, 2, 1, CzModel());
    RunTwoRunChain(&search, false);
    const GaugeBeamSearch::MemoKey key{0, false, false, 0, false, 0};
    EXPECT_EQ(search.MemoGet(key), nullptr);  // 挂起，尚未写入
    EXPECT_TRUE(search.CommitSlots({0}));
    ASSERT_NE(search.MemoGet(key), nullptr);
    EXPECT_EQ(search.MemoGet(key)->phis.size(), 2u);  // 对角：选项 [0, 前导角]
    // 投机求值之后，同一个键先由别的求值以不同的 phi_in 写入：提交失败，且不写入
    GaugeBeamSearch other(1, 4, 2, 1, CzModel());
    RunTwoRunChain(&other, false);
    other.MemoPut(key, GaugeBeamSearch::MemoOptions{1e-12, {0.0}, {0}});
    EXPECT_FALSE(other.CommitSlots({0}));
    EXPECT_EQ(other.MemoGet(GaugeBeamSearch::MemoKey{1, false, false, 3000000000, true, 0}), nullptr);
}

TEST(OneQubitGauge, BeamSearchRejectsBadInput) {
    GaugeBeamSearch search(1, 4, 2, 1, CzModel());
    EXPECT_THROW(search.Prepare({1}, {0}, {0}, {0}, {-1}, {-1}, {1}, {0.0}, {0}, {0}), std::invalid_argument);
    EXPECT_THROW(search.Prepare({0}, {0, 1}, {0}, {0}, {-1}, {-1}, {1}, {0.0}, {0}, {0}), std::invalid_argument);
    EXPECT_THROW(search.CommitSlots({3}), std::invalid_argument);
    std::vector<double> angles;
    std::vector<int> costs;
    EXPECT_THROW(search.Backtrack(0, &angles, &costs), std::invalid_argument);
}

}  // namespace
