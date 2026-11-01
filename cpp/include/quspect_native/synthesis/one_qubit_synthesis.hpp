#pragma once

#include <array>
#include <optional>
#include <vector>

#include "quspect_native/detmath/cx.hpp"

namespace quspect::native::synthesis {

// 单比特合成（quspect.gates.synthesis._synthesize_one_qubit）的 C++ 实现。
//
// 算法（候选门序列、验证、取最短者）与 Python 参考实现相同；全部数值运算走确定性数值层
// （detmath：固定顺序的复数运算、CORE-MATH 的正确舍入 atan2 / hypot / sin / cos），候选的验证矩阵由
// gate_matrices.hpp 的门矩阵相乘得到，所以同一输入在所有平台上得到逐位相同的门与参数。Python 参考
// 实现只在原生模块缺失时使用（数值上等价，不要求逐位一致）。

// 不带参数的标准门（按 quspect.gates.synthesis._Emitter.named 的顺序）。
enum FixedGate : int { kGateH = 0, kGateX, kGateY, kGateZ, kGateS, kGateSdg, kGateT, kGateTdg, kGateSX, kGateSXdg };
constexpr int kNumFixedGates = 10;

// 一个合成出的门：kind 0 = 标准门（index 为 FixedGate），1 = rz 类门（rz / p / u1，参数 p0），
// 2 = rx(p0)，3 = ry(p0)，4 = u3(p0, p1, p2)。
struct Gate1 {
    int kind;
    int index;
    double p0;
    double p1;
    double p2;
};

// _Emitter 的门集信息。
struct OneQubitBasis {
    bool rz = false;  // 有 rz / p / u1 之一
    bool h = false;
    bool x = false;
    bool sx = false;
    bool sxdg = false;
    bool rx = false;
    bool ry = false;
    bool u3 = false;
    std::array<bool, 9> has_word{};               // 下标 k + 4（k = −4…4）
    std::array<std::vector<int>, 9> words{};      // k·π/4 的 Clifford+T 词（FixedGate 序列）
    std::vector<int> named;                       // 门集中的具名门（候选顺序）
};

using Cx = detmath::Cx;
using M2 = std::array<Cx, 4>;  // 2×2，行主序

class OneQubitSynthesizer {
 public:
    OneQubitSynthesizer();

    // ``_synthesize_one_qubit(m, basis)``：nullopt = Python 返回 None（门集无法表示）。输入含非有限值
    // 时抛出 std::domain_error（交给 Python 参考实现处理）。
    std::optional<std::vector<Gate1>> Synthesize(const M2& m, const OneQubitBasis& basis) const;

    // ``_zyz_angles(m)``（公开以便对拍）：(theta, phi, lam)。
    std::array<double, 3> ZyzAngles(const M2& m) const;

 private:
    std::optional<std::vector<Gate1>> Rz(double theta, const OneQubitBasis& basis) const;
    std::vector<std::optional<std::vector<Gate1>>> Candidates(const M2& m, const OneQubitBasis& basis) const;
    bool EqualUpToPhase(const M2& a, const M2& b) const;
    bool NamedPrefilterSkips(const M2& gate, const M2& m) const;
    M2 SequenceMatrix(const std::vector<Gate1>& gates) const;
    static double AngleOfQuotient(const Cx& num, const Cx& den);

    std::array<M2, kNumFixedGates> fixed_;
};

}  // namespace quspect::native::synthesis
