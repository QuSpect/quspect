// CX 朝向全局选择（对偶分解）的 gtest。
#include <cmath>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/cx_orientation.hpp"
#include "quspect_native/transpiler/one_qubit_gauge.hpp"

namespace {

using quspect::native::transpiler::AxisRotation;
using quspect::native::transpiler::Complex;
using quspect::native::transpiler::LengthModel;
using quspect::native::transpiler::Matrix2;
using quspect::native::transpiler::OrientationEnergy;
using quspect::native::transpiler::OrientationOptions;
using quspect::native::transpiler::OrientationProblem;
using quspect::native::transpiler::SolveCxOrientation;

const double kR = std::sqrt(0.5);
const Matrix2 kI = {Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, 0}};
const Matrix2 kH = {Complex{kR, 0}, Complex{kR, 0}, Complex{kR, 0}, Complex{-kR, 0}};

// IBM {rz, sx, x}
LengthModel IbmModel() {
    LengthModel m;
    m.x = true;
    m.sx = true;
    m.named = {
        {Complex{0, 0}, Complex{1, 0}, Complex{1, 0}, Complex{0, 0}},
        {Complex{0.5, 0.5}, Complex{0.5, -0.5}, Complex{0.5, -0.5}, Complex{0.5, 0.5}},
    };
    return m;
}

// q0 是两个 CX 的控制线：H·CX·RX(θ)·CX·H 在 X 坐标系里；两个 CX 一起反向后 H 全部抵消，
// RX 变成目标线上的 RZ。
OrientationProblem XFrameZz(double theta) {
    OrientationProblem p;
    p.num_gates = 2;
    p.wire_gates = {{0, 1}, {0, 1}};
    p.wire_is_control = {{1, 1}, {0, 0}};
    p.wire_runs = {{kH, AxisRotation(1, theta), kH}, {kH, kI, kH}};
    return p;
}

TEST(CxOrientationTest, FlipsWholeXFrameBlock) {
    const OrientationProblem p = XFrameZz(0.3);
    const OrientationOptions opt;
    const auto r = SolveCxOrientation(p, IbmModel(), opt);
    EXPECT_EQ(r.flip, (std::vector<std::uint8_t>{1, 1}));
    EXPECT_LT(r.energy, r.initial_energy);
    // 反向后只剩一个 rz：代价 2·1 + 1
    EXPECT_DOUBLE_EQ(r.energy, 3.0);
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, r.flip, IbmModel(), opt), r.energy);
}

TEST(CxOrientationTest, KeepsOrientationWithoutGain) {
    OrientationProblem p;
    p.num_gates = 2;
    p.wire_gates = {{0, 1}, {0, 1}};
    p.wire_is_control = {{1, 1}, {0, 0}};
    p.wire_runs = {{kI, kI, kI}, {kI, AxisRotation(0, 0.7), kI}};
    const auto r = SolveCxOrientation(p, IbmModel(), OrientationOptions{});
    EXPECT_EQ(r.flip, (std::vector<std::uint8_t>{0, 0}));
    EXPECT_DOUBLE_EQ(r.energy, r.initial_energy);
    EXPECT_DOUBLE_EQ(r.energy, 3.0);
}

TEST(CxOrientationTest, RotationsSlideThroughControls) {
    // RZ 在控制线上可以穿过 CX：两个 RZ 并成一个运行（代价 3），而不是两个（代价 6）。
    OrientationProblem p;
    p.num_gates = 1;
    p.wire_gates = {{0}, {0}};
    p.wire_is_control = {{1}, {0}};
    p.wire_runs = {{AxisRotation(0, 0.4), AxisRotation(0, 0.5)}, {kI, kI}};
    const OrientationOptions opt;
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, {0}, IbmModel(), opt), 3.0);
}

TEST(CxOrientationTest, NeverWorseOnChains) {
    // 交错的 CX 链：结果的代价不超过初始代价，且与重新求值一致。
    OrientationProblem p;
    p.num_gates = 4;
    p.wire_gates = {{0, 2}, {0, 1, 3}, {1, 2, 3}};
    p.wire_is_control = {{1, 0}, {0, 1, 0}, {0, 1, 1}};
    p.wire_runs = {
        {kH, AxisRotation(1, 0.2), kH}, {kI, kH, AxisRotation(2, 0.9), kH}, {AxisRotation(1, 1.1), kI, kH, kI}};
    const OrientationOptions opt;
    const auto r = SolveCxOrientation(p, IbmModel(), opt);
    EXPECT_LE(r.energy, r.initial_energy);
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, r.flip, IbmModel(), opt), r.energy);
}

TEST(CxOrientationTest, RejectsMalformedProblems) {
    OrientationProblem p = XFrameZz(0.3);
    p.wire_gates[1] = {0};
    p.wire_is_control[1] = {0};
    p.wire_runs[1] = {kI, kI};
    EXPECT_THROW(SolveCxOrientation(p, IbmModel(), OrientationOptions{}), std::invalid_argument);
}

}  // namespace
