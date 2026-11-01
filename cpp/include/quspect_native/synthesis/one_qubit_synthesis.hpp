#pragma once

#include <array>
#include <optional>
#include <vector>

#include "quspect_native/synthesis/numpy_compat.hpp"

namespace quspect::native::synthesis {

// 单比特合成（quspect.gates.synthesis._synthesize_one_qubit）的 C++ 实现。
//
// 候选门序列的参数（旋转角）由 numpy / CPython 标量运算求得，这里按 numpy_compat.hpp 中探测到的
// 运算顺序逐位复刻，所以输出的门与参数和 Python 完全相同。候选的验证（``_equal_up_to_phase``）
// 与具名门的预筛选用到 numpy 数组的 ``np.abs``（SIMD 实现与 hypot 可能差最后一位），只用于和
// 阈值比较：落在阈值附近时抛出 Fallback，由调用方改走 Python。

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

class OneQubitSynthesizer {
 public:
    // fixed：各 FixedGate 的矩阵（Python 端 gate_matrix 的结果）。
    OneQubitSynthesizer(const ArithModes& modes, const std::array<M2, kNumFixedGates>& fixed);

    // ``_synthesize_one_qubit(m, basis)``：nullopt = Python 返回 None（门集无法表示）。
    std::optional<std::vector<Gate1>> Synthesize(const M2& m, const OneQubitBasis& basis) const;

    // ``_zyz_angles(m)``（公开以便对拍）：(theta, phi, lam)。
    std::array<double, 3> ZyzAngles(const M2& m) const;

 private:
    std::optional<std::vector<Gate1>> Rz(double theta, const OneQubitBasis& basis) const;
    std::vector<std::optional<std::vector<Gate1>>> Candidates(const M2& m, const OneQubitBasis& basis) const;
    bool EqualUpToPhase(const M2& a, const M2& b) const;
    bool NamedPrefilterSkips(const M2& gate, const M2& m) const;
    M2 SequenceMatrix(const std::vector<Gate1>& gates) const;
    double NpAngleOfQuotient(const Cx& num, const Cx& den) const;

    ArithModes modes_;
    std::array<M2, kNumFixedGates> fixed_;
};

}  // namespace quspect::native::synthesis
