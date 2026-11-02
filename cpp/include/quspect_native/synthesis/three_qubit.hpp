#pragma once

#include <array>
#include <optional>
#include <vector>

#include "quspect_native/detmath/cx.hpp"

namespace quspect::native::synthesis {

// 三比特块重综合（quspect.transpiler._three_qubit）的数值部分，在确定性数值层上实现：块的 8×8
// 酉矩阵、验证距离、多路复用两比特酉矩阵的分解（demultiplex）与余弦–正弦分解（cossin）。
// Python 端原来用 numpy 的 tensordot、scipy.linalg.schur / cossin（LAPACK），结果随平台的
// BLAS / LAPACK 而不同；这里的结果在所有平台上逐位相同。

using Cx = detmath::Cx;
using M4 = std::array<Cx, 16>;
using M8 = std::array<Cx, 64>;

// 一个门：矩阵（2^k × 2^k，行主序）与它作用的局部比特（0 为最高位，k = 1、2 或 3）。
struct LocalGate {
    const Cx* matrix;
    std::vector<int> qubits;
};

// u = I；对每个门 u = G(local) · u（_three_qubit._apply）。
M8 ThreeQubitUnitary(const std::vector<LocalGate>& gates);

// _three_qubit._distance：overlap = vdot(v, u)，|overlap| < 1e-300 时为 +∞，否则
// max |u − (overlap/|overlap|)·v|。
double ThreeQubitDistance(const M8& u, const M8& v);

// _demultiplex 的数值部分：|0⟩⟨0| ⊗ A + |1⟩⟨1| ⊗ B = (I ⊗ V)·(D ⊕ D†)·(I ⊗ W)，其中
// A·B† = V·D²·V†（酉矩阵的特征分解）、phases = arg(D²) / 2、W = D·V†·B。A·B† 的特征分解
// 残差（max |offdiag(V† A B† V)|）超过 1e-9 时返回 nullopt（Python：Schur 形式不是对角的）。
struct Demultiplexed {
    M4 v;
    M4 w;
    std::array<double, 4> phases;
};
std::optional<Demultiplexed> Demultiplex(const M4& a, const M4& b);

// 4×4 酉矩阵 m 的特征分解 m = V·diag(λ)·V†。简并特征子空间里取最接近计算基的一组基，列 i 由
// e_i 的投影得到（m 已是对角 / 块对角时 V 为单位阵或局部矩阵，后续合成的纠缠门最少）。残差
// 超过 tol 时返回 nullopt。
std::optional<std::pair<M4, std::array<Cx, 4>>> UnitaryEig(const M4& m, double tol);

// scipy.linalg.cossin(u, p=4, q=4, separate=True) 的约定：
// u = [[u1, 0], [0, u2]] · [[C, −S], [S, C]] · [[v1h, 0], [0, v2h]]，C = diag(cos θ)，S = diag(sin θ)。
// 由 u11 的奇异值分解与对 u21·v1h^† 的列做复 Gram–Schmidt 得到（θ 的顺序与 u11 的奇异值降序相同）。
struct CosSin {
    M4 u1, u2, v1h, v2h;
    std::array<double, 4> theta;
};
std::optional<CosSin> CosSinDecompose(const M8& u);

}  // namespace quspect::native::synthesis
