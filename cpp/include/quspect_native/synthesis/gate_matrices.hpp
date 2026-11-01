#pragma once

#include <array>
#include <string_view>

#include "quspect_native/detmath/cx.hpp"

namespace quspect::native::synthesis {

// 门矩阵（quspect.gates._matrix_utils.gate_matrix）的确定性实现。
//
// 标准门是精确常量（1/√2 由正确舍入的 sqrt 求得，T / Tdg 的 cos(π/4)、sin(π/4) 写成正确舍入的
// 字面值）；参数化门的 sin / cos 用 CORE-MATH（正确舍入），复数乘法按固定顺序、不含 FMA，所以
// 同一参数在所有平台上得到逐位相同的矩阵。运算形式与 Python 参考实现相同：
//
//   rx(θ)  = [[c, −i·s], [−i·s, c]]                 c = cos(θ/2), s = sin(θ/2)
//   ry(θ)  = [[c, −s], [s, c]]
//   rz(θ)  = diag(c − i·s, c + i·s)
//   p(θ) = u1(θ) = diag(1, cos θ + i·sin θ)
//   u2(φ, λ) = u3(π/2, φ, λ)
//   u3(θ, φ, λ) = [[c, −s·e^{iλ}], [s·e^{iφ}, (c·e^{iφ})·e^{iλ}]]，e^{ix} = cos x + i·sin x
//   crx / cry / crz / cu3：右下角 2×2 块为对应的单比特门；cp(θ) = diag(1, 1, 1, e^{iθ})
//
// 虚部的符号零与 Python（CPython 3.13 的复数运算）的结果可能不同（只影响 −0.0 / +0.0）。

using Cx = detmath::Cx;

// 门名 → 矩阵（行主序，dim × dim）。返回 false：未知门名或参数个数不对（调用方按 Python 语义报错）。
// out 至少能容纳 64 个元素（ccx 为 8×8）。
bool GateMatrix(std::string_view name, const double* params, int num_params, Cx* out, int* dim);

// 单比特门矩阵的便捷形式（门名须为单比特门）。
bool GateMatrix2(std::string_view name, const double* params, int num_params, std::array<Cx, 4>* out);

}  // namespace quspect::native::synthesis
