#pragma once

#include <array>

#include "quspect_native/detmath/complex.hpp"
#include "quspect_native/detmath/config.hpp"

namespace quspect::native::detmath {

// 小矩阵线性代数（n ≤ kMaxDim，行主序）。
//
// 编译路径上只出现 2×2、4×4（两比特块）与 8×8（三比特块）的复矩阵，所以这里不追求渐近复杂度，
// 而是固定的、可逐位复现的运算步骤：
//
//   * 乘法、内积：下标升序累加，复数乘法见 complex.hpp；
//   * 行列式：部分主元 LU（主元取 |re| + |im| 最大者，相等时取下标最小者）；
//   * 实对称特征分解：循环 Jacobi（Rutishauser 的阈值规则），特征值升序；
//   * 两个可交换实对称矩阵的同时对角化：循环 Jacobi，每对 (p, q) 的旋转角使两个矩阵的
//     (p, q) 元平方和最小（Bunse-Gerstner, Byers & Mehrmann 1993；Cardoso & Souloumiac 1996）；
//   * 奇异值分解：单边 Jacobi（Hestenes），奇异值降序；
//   * QR：Householder，R 的对角元为非负实数（输入列满秩时 QR 唯一）。
//
// Jacobi 类算法的精度（正交性、重构误差）不劣于 LAPACK 的 dsyevd / zgesdd，对重特征值、
// 重奇异值与秩亏输入同样稳定；固定的循环顺序与收敛判据保证结果与平台无关。
//
// 特征向量 / 奇异向量的符号（相位）约定：每个向量中模最大（相等时下标最小）的分量为正实数
// （同时对角化与特征分解的实向量）；奇异值分解中 V 的每列同样规范化，U 随之确定。

constexpr int kMaxDim = 8;

using M2 = std::array<Cx, 4>;
using M4 = std::array<Cx, 16>;
using R4 = std::array<double, 16>;

// ---- 乘法 ----------------------------------------------------------------------

// out = a @ b（n×n 复矩阵；out 不得与 a、b 重叠）。
void MatMul(int n, const Cx* a, const Cx* b, Cx* out);
// out = a^† @ b。
void MatMulAdjA(int n, const Cx* a, const Cx* b, Cx* out);
// out = a @ b^†。
void MatMulAdjB(int n, const Cx* a, const Cx* b, Cx* out);
// out = a^T @ b（numpy 的 ``a.T @ b``）。
void MatMulTransA(int n, const Cx* a, const Cx* b, Cx* out);
// 实矩阵与复矩阵的乘法（实数乘复数：(r·re, r·im)）。
void MatMulRC(int n, const double* a, const Cx* b, Cx* out);
void MatMulCR(int n, const Cx* a, const double* b, Cx* out);
// out = a^T @ b，a 为实矩阵。
void MatMulRTransC(int n, const double* a, const Cx* b, Cx* out);
// out = a @ b^T，b 为实矩阵。
void MatMulCRTrans(int n, const Cx* a, const double* b, Cx* out);

M2 MatMul(const M2& a, const M2& b);
M4 MatMul(const M4& a, const M4& b);
M2 Adjoint(const M2& a);
M4 Adjoint(const M4& a);

// Σ conj(a_k)·b_k（k 升序）。
Cx Vdot(int len, const Cx* a, const Cx* b);

// ---- 行列式 ----------------------------------------------------------------------

Cx Det(int n, const Cx* a);
double DetReal(int n, const double* a);
inline Cx Det(const M2& a) {
    return Det(2, a.data());
}
inline Cx Det(const M4& a) {
    return Det(4, a.data());
}

// ---- 分解 ----------------------------------------------------------------------

// 实对称矩阵 a（只读取上三角）的特征分解：w 升序，v 的第 j 列是 w[j] 的特征向量。
// 不收敛（输入含 NaN / Inf）时返回 false。
bool EighSym(int n, const double* a, double* w, double* v);

// 两个可交换实对称矩阵 a、b 的同时对角化：返回正交矩阵 v，使 v^T a v 与 v^T b v 都（近似）为
// 对角矩阵。列的顺序为 Jacobi 迭代的自然顺序（不排序）。输入不可交换时结果是近似的同时对角化
// （两个矩阵非对角元平方和的局部极小）。不收敛时返回 false。
bool JointDiagSym(int n, const double* a, const double* b, double* v);

// 复厄米矩阵 a（n ≤ kMaxDim / 2，只读取上三角）的特征分解：v 的第 j 列是特征值 w[j] 的单位特征
// 向量（列的顺序不排序；每列模最大的分量为正实数）。经由实嵌入 [[Re, −Im], [Im, Re]] 的
// Jacobi 分解求得。不收敛时返回 false。
bool EighHerm(int n, const Cx* a, double* w, Cx* v);

// 复矩阵的奇异值分解 a = u · diag(s) · vh：s 降序，u、vh 为酉矩阵。不收敛时返回 false。
bool Svd(int n, const Cx* a, Cx* u, double* s, Cx* vh);

// 复矩阵的 QR 分解 a = q · r：q 为酉矩阵，r 为上三角且对角元为非负实数。
void Qr(int n, const Cx* a, Cx* q, Cx* r);

}  // namespace quspect::native::detmath
