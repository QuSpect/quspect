#pragma once

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

// 待调度的指令序列（CSR）：节点 g 占用资源 resources[offsets[g] .. offsets[g + 1])
// （比特与经典比特，互不相同），axes 为每个资源上的对易轴（−1 无，0 Z，1 X，2 Y）；
// two[g] 为双比特门（计入双比特门深度），counted[g] 为计入总深度的指令（屏障不计）。
struct ScheduleProblem {
    std::vector<int> offsets;
    std::vector<int> resources;
    std::vector<int> axes;
    std::vector<std::uint8_t> two;
    std::vector<std::uint8_t> counted;
};

// 与 Python 参考实现 quspect.transpiler._schedule._best_order 逐位一致：对易组上的串行
// 调度生成（插入式、关键路径优先级）+ 双向调整（double justification），在总深度、
// 保持双比特门顺序、先压双比特门深度三种调度中取双比特门深度不增时总深度最小者。
// 返回节点的新顺序（输入顺序即 0..n−1）。
//
// 抛出 std::invalid_argument：数组尺寸不一致或资源编号为负。
std::vector<int> BestDepthOrder(const ScheduleProblem& problem);

// 按 order 重排节点后的问题（order 为 0..n−1 的排列）。
//
// 抛出 std::invalid_argument：order 不是排列。
ScheduleProblem Permuted(const ScheduleProblem& problem, const std::vector<int>& order);

// 按 order 排列时 transpile 报告的 (深度, 两比特门深度)：只看编号小于 num_qubits 的资源
// （比特），总深度只计 counted 节点，两比特门深度只计 two 节点。
std::pair<int, int> ReportedDepths(const ScheduleProblem& problem, const std::vector<int>& order, int num_qubits);

// 两比特块层面的调度输入：每条指令的比特（CSR）、种类（0 其他，1 单比特门，2 两比特门，
// 3 不带比特的屏障 = 已出现的全部比特）与矩阵（matrix_offset 为 matrices 中的起点，
// 单比特门 4 个、两比特门 16 个元素，行主序、按指令自身的比特顺序；−1 为未知矩阵）。
struct BlockInput {
    std::vector<int> qubit_offsets;
    std::vector<int> qubits;
    std::vector<int> kind;
    std::vector<int> matrix_offset;
    std::vector<std::complex<double>> matrices;
};

// 与 Python 参考实现 quspect.transpiler._schedule._block_order 一致：把线路划分为极大两比特块
// （同一比特对上的两比特门及其间的单比特门）与单条指令，块按其酉矩阵在各比特上的对易 Pauli
// 轴参与调度（单条指令用 problem 中的资源与轴）。返回展开后的指令顺序；没有可用的块轴或
// 块顺序不变时返回空。
std::vector<int> BlockOrder(const ScheduleProblem& problem, const BlockInput& blocks);

}  // namespace quspect::native::transpiler
