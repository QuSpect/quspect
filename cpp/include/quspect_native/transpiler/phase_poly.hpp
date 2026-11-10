#pragma once

// 路由后 CNOT-相位区域重综合的整轮流程（quspect.transpiler._phase_poly.PhasePolynomialPass 的原生实现）。
//
// 输入是只含单比特门与 cz / cx 的线路（单比特门给出确定性计算的 2×2 矩阵，cx 已展开为 h·cz·h 并带朝向
// 提示）。每一轮：
//   1. 读法：每个 cz 读作 CX(a→b)、CX(b→a) 或对角门；两比特门之间一根线上的单比特段在读法确定后为
//      H^{t'} R H^{t}，局部搜索使尽量多的段是单项式（对角或反对角）。
//   2. 区域：按 (最多几根线, A* 预算, 方向) 的每种设置把定向后的线路贪心划分为凸的 CNOT-相位区域
//      （方向为反向时从线路末尾开始生长）。
//   3. 综合：每个区域算出线性映射 A、常数 c 与相位多项式 f；≤ 6 根线用 A*（最优），≥ 5 根线用束搜索
//      （见 cnot_phase_synth.hpp），取严格更少 CX 的解。
//   4. 验证：每个替换都在发出的门上验证（≤ 6 根线比较整个酉矩阵；更大的区域比较重新计算的相位多项式、
//      线性映射与常数，≤ 14 根线时再比较两个伪随机态），不通过则保留原区域。
//   5. 各设置中两比特门最少的划分胜出（平局取靠前的设置）；有改进则进入下一轮。
// 所有浮点运算逐次舍入（本文件按 QSPECT_EXACT_FP 编译，不收缩成 FMA），三角函数用 detmath 的正确舍入
// 实现，输出与平台无关；与 Python 参考实现（_phase_poly.py）逐位相同。

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

// 输出 / 内部的门：kOrig 为输入中的第 index 个单比特门，其余为新发出的门。
enum PhasePolyOpKind : int { kPPOrig = 0, kPPH = 1, kPPCz = 2, kPPRz = 3, kPPX = 4 };

struct PhasePolyOp {
    int kind = kPPOrig;
    int a = 0;           // 单比特门的比特；cz 的第一个比特
    int b = -1;          // cz 的第二个比特
    int hint = -1;       // cz：0 表示来自 cx(a→b)（读法提示），否则 -1
    int index = -1;      // kOrig：输入单比特门的编号
    double angle = 0.0;  // kRz：角度
};

struct PhasePolySetting {
    int max_wires = 4;
    std::int64_t node_budget = 200000;
    bool reverse = false;
};

struct PhasePolyStats {
    std::int64_t regions = 0;
    std::int64_t replaced = 0;
    std::int64_t saved = 0;
    std::int64_t rounds = 0;
};

struct PhasePolyResult {
    bool improved = false;
    std::vector<PhasePolyOp> ops;
    PhasePolyStats stats;
};

// ops：输入线路（只含 kOrig 单比特门与 kCz）；matrices[index]：第 index 个单比特门的矩阵
// (a.re, a.im, b.re, b.im, c.re, c.im, d.re, d.im)，opaque[index] 非 0 表示不透明（符号参数等）；
// extra_pairs：除线路中已有的比特对以外允许使用的无序比特对。
PhasePolyResult RunPhasePolynomial(const std::vector<PhasePolyOp>& ops,
                                   const std::vector<std::array<double, 8>>& matrices,
                                   const std::vector<std::uint8_t>& opaque,
                                   const std::vector<std::pair<int, int>>& extra_pairs,
                                   const std::vector<PhasePolySetting>& settings, int beam_width, int max_rounds);

}  // namespace quspect::native::transpiler
