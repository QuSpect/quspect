// 对易感知深度调度的 gtest：对易 CZ 环的边着色、非对易门保持顺序、双比特门深度不增，
// 以及与 Python 参考实现（quspect.transpiler._schedule._best_order）一致的顺序。
#include <algorithm>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/depth_schedule.hpp"

namespace {

using quspect::native::transpiler::BestDepthOrder;
using quspect::native::transpiler::BlockInput;
using quspect::native::transpiler::BlockOrder;
using quspect::native::transpiler::ScheduleProblem;

constexpr int kZ = 0;
constexpr int kX = 1;
constexpr int kNone = -1;

struct Op {
    std::vector<int> qubits;
    std::vector<int> axes;
};

ScheduleProblem Build(const std::vector<Op>& ops) {
    ScheduleProblem p;
    p.offsets.push_back(0);
    for (const Op& op : ops) {
        p.resources.insert(p.resources.end(), op.qubits.begin(), op.qubits.end());
        p.axes.insert(p.axes.end(), op.axes.begin(), op.axes.end());
        p.offsets.push_back(static_cast<int>(p.resources.size()));
        p.two.push_back(op.qubits.size() == 2 ? 1 : 0);
        p.counted.push_back(1);
    }
    return p;
}

int Depth(const std::vector<Op>& ops, const std::vector<int>& order, bool two_only) {
    std::vector<int> level(64, 0);
    int deepest = 0;
    for (int g : order) {
        if (two_only && ops[g].qubits.size() != 2) {
            continue;
        }
        int lv = 0;
        for (int q : ops[g].qubits) {
            lv = std::max(lv, level[q]);
        }
        ++lv;
        for (int q : ops[g].qubits) {
            level[q] = lv;
        }
        deepest = std::max(deepest, lv);
    }
    return deepest;
}

Op Cz(int a, int b) {
    return {{a, b}, {kZ, kZ}};
}
Op Cx(int c, int t) {
    return {{c, t}, {kZ, kX}};
}
Op Rz(int q) {
    return {{q}, {kZ}};
}
Op H(int q) {
    return {{q}, {kNone}};
}

std::vector<int> Identity(int n) {
    std::vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }
    return v;
}

TEST(DepthScheduleTest, CommutingRingIsEdgeColored) {
    // cz(0,1) cz(1,2) cz(2,3) cz(3,0)：原顺序深度 4，两种颜色即可（两层）
    const std::vector<Op> ops = {Cz(0, 1), Cz(1, 2), Cz(2, 3), Cz(3, 0)};
    const std::vector<int> order = BestDepthOrder(Build(ops));
    EXPECT_EQ(Depth(ops, Identity(4), false), 4);
    EXPECT_EQ(Depth(ops, order, false), 2);
}

TEST(DepthScheduleTest, NonCommutingGatesKeepTheirOrder) {
    // h 不与任何门对易：cx 链保持原顺序
    const std::vector<Op> ops = {Cx(0, 1), H(1), Cx(1, 2), H(2), Cx(2, 3)};
    const std::vector<int> order = BestDepthOrder(Build(ops));
    EXPECT_EQ(order, Identity(5));
}

TEST(DepthScheduleTest, RotationFloatsThroughControl) {
    // rz(0) 与 cx(0,1)、cx(0,3) 的控制位对易：三者可任意排列，cx(0,3) 提前与 cx(1,2)
    // 并行，深度 4 → 3（比特 0 上有三个门，3 是下界）
    const std::vector<Op> ops = {Cx(1, 2), Cx(0, 1), Rz(0), Cx(0, 3)};
    const std::vector<int> order = BestDepthOrder(Build(ops));
    EXPECT_EQ(Depth(ops, Identity(4), false), 4);
    EXPECT_EQ(Depth(ops, order, false), 3);
    EXPECT_LE(Depth(ops, order, true), Depth(ops, Identity(4), true));
    // 与 Python 参考实现的顺序一致
    EXPECT_EQ(order, (std::vector<int>{0, 3, 1, 2}));
}

TEST(DepthScheduleTest, TwoQubitDepthNeverIncreases) {
    const std::vector<Op> ops = {Cx(0, 1), Rz(0), Rz(0), Rz(0), Cx(0, 2), Cx(2, 1), H(1), Cx(1, 0), Rz(1), Cz(1, 2)};
    const std::vector<int> order = BestDepthOrder(Build(ops));
    std::vector<int> sorted = order;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(sorted, Identity(static_cast<int>(ops.size())));
    EXPECT_LE(Depth(ops, order, true), Depth(ops, Identity(static_cast<int>(ops.size())), true));
    EXPECT_LE(Depth(ops, order, false), Depth(ops, Identity(static_cast<int>(ops.size())), false));
}

TEST(DepthScheduleTest, RejectsInconsistentInput) {
    ScheduleProblem p = Build({Cz(0, 1)});
    p.two.push_back(0);
    EXPECT_THROW(BestDepthOrder(p), std::invalid_argument);
    ScheduleProblem q = Build({Cz(0, 1)});
    q.resources[0] = -1;
    EXPECT_THROW(BestDepthOrder(q), std::invalid_argument);
    ScheduleProblem r;
    EXPECT_THROW(BestDepthOrder(r), std::invalid_argument);
}

TEST(DepthScheduleTest, EmptyProblem) {
    ScheduleProblem p;
    p.offsets.push_back(0);
    EXPECT_TRUE(BestDepthOrder(p).empty());
}

// cx(a,b)·rz(b)·cx(a,b) 块（ZZ 旋转）的指令与矩阵：cx 16 个元素、rz 4 个元素。
void AddZz(int a, int b, std::vector<Op>* ops, BlockInput* in) {
    using C = std::complex<double>;
    const std::vector<C> cx = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
    const std::vector<C> rz = {std::polar(1.0, -0.15), 0, 0, std::polar(1.0, 0.15)};
    auto add = [&](const Op& op, int kind, const std::vector<C>& m) {
        ops->push_back(op);
        in->qubits.insert(in->qubits.end(), op.qubits.begin(), op.qubits.end());
        in->qubit_offsets.push_back(static_cast<int>(in->qubits.size()));
        in->kind.push_back(kind);
        in->matrix_offset.push_back(static_cast<int>(in->matrices.size()));
        in->matrices.insert(in->matrices.end(), m.begin(), m.end());
    };
    add(Cx(a, b), 2, cx);
    add(Rz(b), 1, rz);
    add(Cx(a, b), 2, cx);
}

TEST(DepthScheduleTest, ZzBlocksCommuteOnTheBlockLevel) {
    // ZZ(0,1)·ZZ(1,2)·ZZ(3,2)：逐门不对易（CX 目标位对控制位），按块都是对角的
    std::vector<Op> ops;
    BlockInput in;
    in.qubit_offsets.push_back(0);
    AddZz(0, 1, &ops, &in);
    AddZz(1, 2, &ops, &in);
    AddZz(3, 2, &ops, &in);
    const ScheduleProblem p = Build(ops);
    EXPECT_EQ(BestDepthOrder(p), Identity(9));
    const std::vector<int> order = BlockOrder(p, in);
    ASSERT_EQ(order.size(), 9U);
    EXPECT_EQ(Depth(ops, Identity(9), false), 9);
    EXPECT_EQ(Depth(ops, order, false), 6);
    // 块内顺序不变
    EXPECT_EQ(order, (std::vector<int>{0, 1, 2, 6, 7, 8, 3, 4, 5}));
}

TEST(DepthScheduleTest, BlockOrderWithoutCommutingBlocksIsEmpty) {
    // 未知矩阵的块没有对易轴：不适用
    std::vector<Op> ops;
    BlockInput in;
    in.qubit_offsets.push_back(0);
    AddZz(0, 1, &ops, &in);
    AddZz(1, 2, &ops, &in);
    for (int& off : in.matrix_offset) {
        off = -1;
    }
    EXPECT_TRUE(BlockOrder(Build(ops), in).empty());
    in.kind[0] = 1;  // 与比特数不符
    EXPECT_THROW(BlockOrder(Build(ops), in), std::invalid_argument);
}

// 宽线路上的长对易组（大量对角门与 CZ 混排）：调度中会出现成段已占用的时隙。结果必须是排列、
// 保持非对易对的先后、两比特门深度不增，且对同一输入确定。
TEST(DepthScheduleTest, WideCommutingGroupsKeepPrecedence) {
    std::mt19937 rng(7);
    std::vector<Op> ops;
    const int width = 24;
    for (int k = 0; k < 3000; ++k) {
        const int r = static_cast<int>(rng() % 10);
        const int a = static_cast<int>(rng() % width);
        int b = static_cast<int>(rng() % (width - 1));
        b += b >= a ? 1 : 0;
        if (r < 4) {
            ops.push_back(Cz(a, b));
        } else if (r < 6) {
            ops.push_back(Cx(a, b));
        } else if (r < 9) {
            ops.push_back(Rz(a));
        } else {
            ops.push_back(H(a));
        }
    }
    const ScheduleProblem p = Build(ops);
    const std::vector<int> order = BestDepthOrder(p);
    ASSERT_EQ(order.size(), ops.size());
    std::vector<int> pos(ops.size(), -1);
    for (std::size_t k = 0; k < order.size(); ++k) {
        ASSERT_EQ(pos[order[k]], -1);
        pos[order[k]] = static_cast<int>(k);
    }
    // 同一比特上：轴不同（或无轴）的两条指令保持原来的先后
    for (int q = 0; q < width; ++q) {
        std::vector<std::pair<int, int>> on;  // (指令, 轴)
        for (int g = 0; g < static_cast<int>(ops.size()); ++g) {
            for (std::size_t s = 0; s < ops[g].qubits.size(); ++s) {
                if (ops[g].qubits[s] == q) {
                    on.emplace_back(g, ops[g].axes[s]);
                }
            }
        }
        for (std::size_t i = 0; i < on.size(); ++i) {
            for (std::size_t j = i + 1; j < on.size(); ++j) {
                if (on[i].second == kNone || on[i].second != on[j].second) {
                    EXPECT_LT(pos[on[i].first], pos[on[j].first]);
                    break;  // 之后的指令由传递性保证
                }
            }
        }
    }
    EXPECT_LE(Depth(ops, order, true), Depth(ops, Identity(static_cast<int>(ops.size())), true));
    EXPECT_LE(Depth(ops, order, false), Depth(ops, Identity(static_cast<int>(ops.size())), false));
    EXPECT_EQ(order, BestDepthOrder(p));
}

}  // namespace
