#pragma once

#include <cstdint>
#include <vector>

#include "quspect_native/transpiler/one_qubit_gauge.hpp"

namespace quspect::native::transpiler {

// CX 朝向的全局选择（quspect.transpiler._gauge 的 ``global_orientation``）。
//
// ``CX(a, b) = (H⊗H)·CX(b, a)·(H⊗H)``：每个 CX 可以独立反向，H 并进它两侧四个单比特运行。
// 第 g 个 CX 的朝向是一个二值变量 s_g；一条线上的单比特门代价只依赖这条线上相邻 CX 的朝向
// （以及穿过 CX 推移的旋转），所以总代价是"每条线一条链"的和，而每个 CX 恰好出现在两条链上
// （控制线与目标线）。这正是对偶分解（Lagrange 松弛）的标准形式：把每个 CX 的朝向复制成两份，
// 每条链在给定乘子下用动态规划（Viterbi + 旋转束搜索）精确求解，用次梯度更新乘子使两份一致；
// 每次迭代用两份副本给出原问题的可行解，取代价最小者。局部搜索（逐个 / 成对反向）在需要整片
// 一起反向的情形（如 QAOA 的 ZZ 链被换到 X 坐标系）里卡在局部极小，对偶分解没有这个问题。
//
// 一条线的链：运行 R_0, g_1, R_1, …, g_n, R_n。g_j 反向时 R_{j-1} 末尾、R_j 开头各乘一个 H；
// 与 g_j 在这条线上对易的轴（控制线 Z、目标线 X，反向后互换）上的旋转可以从 R_{j-1} 末尾移到
// R_j 开头，候选角取 R_{j-1} 的前导角（与 _gauge 的束搜索相同）。运行的代价为
// ``gate_weight·门数 + run_weight·[非空]``，门数由 ``FastLength`` 给出。
struct OrientationProblem {
    int num_gates = 0;
    // 每条线：沿线的 CX 编号、这条线是否为该 CX 的控制线、运行矩阵（CX 数 + 1 个）。
    std::vector<std::vector<int>> wire_gates;
    std::vector<std::vector<std::uint8_t>> wire_is_control;
    std::vector<std::vector<Matrix2>> wire_runs;
};

struct OrientationOptions {
    int beam = 4;
    int gate_weight = 2;
    int run_weight = 1;
    int iterations = 40;
    // 连续这么多次迭代没有找到更好的原问题解就停止。默认等于 iterations（不提前停止）：对偶分解的
    // 原问题解常在乘子收敛后期才出现（如 bmw_quark_cardinality_16 在 15 次无改进之后）。
    int patience = 40;
};

struct OrientationResult {
    std::vector<std::uint8_t> flip;  // 每个 CX 是否反向
    double energy = 0.0;             // 选中朝向的代价
    double initial_energy = 0.0;     // 全不反向的代价
    double dual = 0.0;               // 最后一次迭代的对偶值（束搜索下为近似下界）
    int iterations = 0;
};

// 一条线在给定朝向（``flip`` 按 CX 编号）下的代价。
double WireCost(const OrientationProblem& problem, int wire, const std::vector<std::uint8_t>& flip,
                const LengthModel& model, const OrientationOptions& options);

// 所有线的代价之和。
double OrientationEnergy(const OrientationProblem& problem, const std::vector<std::uint8_t>& flip,
                         const LengthModel& model, const OrientationOptions& options);

// 对偶分解求 CX 朝向；结果的代价不超过全不反向的代价。
OrientationResult SolveCxOrientation(const OrientationProblem& problem, const LengthModel& model,
                                     const OrientationOptions& options);

}  // namespace quspect::native::transpiler
