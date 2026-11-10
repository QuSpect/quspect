#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "quspect_native/synthesis/three_qubit.hpp"

namespace quspect::native::synthesis {

// 三比特酉矩阵的数值实例化搜索（QSearch 式，Davis 等 2020）：在固定结构的模板上
// 数值优化参数，按 CZ 数从小到大搜索结构，找到与目标相差全局相位、逐元素残差
// ≤ tol 的线路。
//
// 结构：开头每根线一个 U3 = Rz·Ry·Rz，之后每个 CZ(a, b) 后面在 a、b 上各一个 Rz·Ry
// （U3 中先作用的 Rz 与 CZ 对易，并进前一个门的 Rz，所以只需两个参数）。参数顺序：
// 开头 q = 0, 1, 2 各 (Rz 先作用, Ry, Rz 后作用)，然后每个 CZ (a 的 Ry, a 的 Rz, b 的 Ry,
// b 的 Rz)。局部比特 0 为最高位（与 ThreeQubitUnitary 相同）。
//
// 优化：残差 r = V(θ) − e^{iφ}·U（128 个实数，φ 也是参数），Levenberg–Marquardt，
// 解析雅可比。搜索：A*（f = CZ 数 + heuristic_factor · √(1 − |tr(U†V)|²/64)），
// 子结点在末尾加一个耦合边上的 CZ（同一对上连续 ≤ 3 个）；子结点从父结点的最优参数
// （新层为恒等）出发，另有 restarts 个固定伪随机起点。找到一个解后继续只展开更少
// CZ 的结点，直到队列空或预算（LM 迭代次数）用完。
//
// 确定性：只用 IEEE 基本运算、sqrt 与 detmath 的正确舍入三角函数（本文件以
// QSPECT_EXACT_FP 编译，不收缩 FMA），伪随机数是整数 LCG，优先队列按 (f, 插入序号)
// 全序：结果与平台、线程无关，逐位一致。预算按迭代次数计，不按墙钟。
struct InstantiateOptions {
    int max_cz = 0;                 // 只接受 ≤ max_cz 个 CZ 的解
    std::int64_t budget = 20000;    // LM 迭代总数（每次迭代 = 一次雅可比 + 线性求解）
    int restarts = 1;               // 每个结点额外的伪随机起点数
    int max_iters = 80;             // 每个起点的迭代上限（接近解时另加 polish_iters）
    int polish_iters = 40;
    double tol = 1e-12;             // 成功：max |V − e^{iφ}U| ≤ tol
    double heuristic_factor = 10.0;
    std::uint64_t seed = 0x9E3779B97F4A7C15ULL;
};

struct InstantiateResult {
    bool found = false;
    int target = -1;                          // 命中的目标下标
    std::vector<std::array<int, 2>> czs;      // 按作用顺序
    std::vector<double> params;               // 9 + 4·len(czs)，已规约到 (−π, π]
    double residual = 0.0;                    // max |V − e^{iφ}U|
    std::int64_t iterations = 0;              // 用掉的 LM 迭代数
    std::int64_t nodes = 0;                   // 实例化过的结构数
};

// 依次以 targets 中的每个酉矩阵为目标（例如同一线路的不同输出置换），返回 CZ 数最少的解
// （CZ 数相同时先找到的）。edges 为允许的局部比特对（无向）。
InstantiateResult InstantiateSearch(const std::vector<M8>& targets, const std::vector<std::array<int, 2>>& edges,
                                    const InstantiateOptions& options);

// 由结构与参数算出 V（用于测试与验证）。
M8 InstantiateUnitary(const std::vector<std::array<int, 2>>& czs, const std::vector<double>& params);

}  // namespace quspect::native::synthesis
