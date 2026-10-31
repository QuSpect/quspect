#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {

// 单比特门最小化（quspect.transpiler._gauge）的标量内核。
//
// 这些函数与 Python 参考实现逐位一致：Python 端用 CPython 复数（``complex``、``cmath``）或
// numpy 复数标量计算，这里按各自的浮点运算顺序复刻——CPython / numpy 的复数乘除在 arm64 上由
// clang 收缩成 FMA，这里用显式 ``std::fma`` 写出同样的收缩，并以 ``-ffp-contract=off``
// 编译，禁止编译器再做其他合并。超出复刻范围的输入（非有限值、除以零）返回 ``std::nullopt``，
// 由调用方退回 Python。
//
// 复数以 (re, im) 对表示。
struct Complex {
    double re;
    double im;
};

// 2×2 矩阵（行主序：m00, m01, m10, m11）。
using Matrix2 = std::array<Complex, 4>;

// ``_LengthModel`` 的门集标志与具名门矩阵（按 Python 端 ``named_entries`` 的顺序）。
struct LengthModel {
    bool h = false;
    bool x = false;
    bool sx = false;
    bool sxdg = false;
    std::vector<Matrix2> named;
};

// ``_LengthModel._fast(m)``：rz 系门集下 ``synthesize_one_qubit`` 的门数（CPython 复数语义）。
std::optional<int> FastLength(const Matrix2& m, const LengthModel& model);

// ``_leading_angles(m, axis)`` 在已经换到 Z 坐标系的 ``z = V†·m·V`` 上的部分（numpy 标量语义）：
// 对角时返回 1 个角度，否则返回 4 个；不能复刻时返回 std::nullopt。
struct LeadingAngles {
    int count = 0;
    std::array<double, 4> angles{};
};
std::optional<LeadingAngles> LeadingAnglesInZFrame(const Matrix2& z);

// ``_rot(axis, phi)`` = cos(phi/2)·I − (1j·sin(phi/2))·P（numpy 数组运算语义）。axis：0 = z, 1 = x, 2 = y。
Matrix2 AxisRotation(int axis, double phi);

// ``len(quspect.synthesis.two_qubit.euler_ops(m, q, basis=..., atol=atol))``（numpy / CPython 语义）：
// basis 0 = "zxz"（rz·h·rz·h·rz），1 = "zsx"（rz·sx·rz·sx·rz）。不能复刻时返回 std::nullopt。
std::optional<int> EulerLength(const Matrix2& m, int basis, double atol);

// CPython ``round(x)``（float → int，半数取偶）。
std::int64_t PyRound(double x);

// ``minimize_single_qubit_gates`` 的分层束搜索与 memo。
//
// 与 Python 端 ``solve(q, s, e, phi_s, phi_e)`` 逐位一致。一个"槽位"是一条独立的运行链：初始搜索时
// 每条线一个槽位（``solve(q, 0, k, 0.0, 0.0)``，此时没有 CX 反向，各条线互不相关），CX 朝向局部
// 搜索时每个候选反向的每条线一个槽位（两个运行的窗口）。所有槽位按层同时推进；矩阵乘法仍由
// Python 端用 numpy 批量完成（numpy 的 2×2 复数矩阵乘法走 BLAS，这里无法复刻），每层三步：
//
//   1. ``Prepare``：每个槽位的束条目按 memo 键 ``(rid, fp, fn, round(phi_in·1e10), last, phi_e)``
//      去重并查 memo；未命中的条目进入"计算表"，给出其中需要右乘 ``rot(ax_in, phi_in)`` 的条目的
//      旋转矩阵与需要前导角的条目；
//   2. ``Options``：由 Python 端算好的 ``z = V†·m·V`` 求前导角，列出每个计算条目的选项角，给出非零
//      选项需要左乘的 ``rot(ax_out, −phi)``；
//   3. ``Finish``：由运行矩阵与选项矩阵求长度与代价，按 Python 端的规则更新束；未命中的选项写入
//      memo（初始搜索），或先挂起（局部搜索的投机求值，``CommitSlots`` 时按顺序提交）。
//
// 不能复刻的输入抛出 std::domain_error（调用方退回 Python 实现）。
class GaugeBeamSearch {
 public:
    struct Entry {
        std::int64_t total;
        double phi;
        int parent;
        int cost;
    };

    struct MemoKey {
        int rid;
        bool fp;
        bool fn;
        std::int64_t key;
        bool last;
        std::uint64_t phi_e_bits;  // ``phi_e if last else 0.0`` 的位模式
        bool operator==(const MemoKey& o) const {
            return rid == o.rid && fp == o.fp && fn == o.fn && key == o.key && last == o.last
                   && phi_e_bits == o.phi_e_bits;
        }
    };

    struct MemoOptions {
        double phi_in;  // 首次写入时的 phi_in（同键的选项由它算出）
        std::vector<double> phis;
        std::vector<int> costs;
    };

    GaugeBeamSearch(int num_slots, int beam, int gate_weight, int run_weight, LengthModel model);

    // 重新开始：每个槽位的初始束为 [(0, start_phi[slot], −1, 0)]；清空层与挂起的 memo 项（memo 保留）。
    void Reset(const std::vector<double>& start_phi);

    // 本层的活动槽位与每个槽位当前运行的属性（每行一个）：运行编号、两侧 CX 是否反向、ax_in /
    // ax_out（−1 = 无）、是否最后一个运行及其固定带出角、``len(run.gates)``（非原生为 −1）、
    // ``run_base`` 是否改动了运行矩阵（乘了 H）。返回计算表的长度。
    int Prepare(const std::vector<int>& slots, const std::vector<int>& rids, const std::vector<std::uint8_t>& fp,
                const std::vector<std::uint8_t>& fn, const std::vector<int>& ax_in, const std::vector<int>& ax_out,
                const std::vector<std::uint8_t>& last, const std::vector<double>& phi_fixed,
                const std::vector<int>& run_length, const std::vector<std::uint8_t>& base_changed);

    // 初始搜索的线表：每个槽位（线）的运行编号与锚点轴（−1 = 无；锚点 i 在运行 i 之后），以及每个
    // 运行的 ``len(run.gates)``（按运行编号；非原生为 −1）。之后 ``PrepareLayer(i)`` 等价于对所有
    // 还有第 i 个运行的线调用 ``Prepare``（无反向、phi_e = 0）。
    void SetWires(std::vector<std::vector<int>> rids, std::vector<std::vector<int>> axes, std::vector<int> run_length);
    int PrepareLayer(int layer);
    // 计算表每项的运行编号与 ax_out（Prepare / PrepareLayer 之后有效）。
    const std::vector<int>& ComputeRid() const {
        return compute_rid_;
    }
    const std::vector<int>& ComputeAxOut() const {
        return compute_ax_out_;
    }

    // 计算表：每项所属的行（slots 中的位置）、phi_in；其中需要右乘旋转的项（计算表下标，升序）与
    // 旋转矩阵；需要前导角的项（计算表下标，升序；Options 的 z 按此顺序给出）。
    const std::vector<int>& ComputeRow() const {
        return compute_row_;
    }
    const std::vector<int>& RotateIn() const {
        return rotate_in_;
    }
    const std::vector<Matrix2>& RotateInMatrices() const {
        return rotate_in_mats_;
    }
    const std::vector<int>& NeedLeading() const {
        return need_leading_;
    }

    // 给出非零选项所属的计算表下标（升序）与左乘矩阵。
    void Options(const std::vector<Matrix2>& z);
    const std::vector<int>& OptionRows() const {
        return option_rows_;
    }
    const std::vector<Matrix2>& OptionMatrices() const {
        return option_mats_;
    }

    // m：每个计算项的运行矩阵（含 run_base 与右乘旋转）；mm：每个非零选项的矩阵（与 OptionRows 对齐）。
    // insert_memo：未命中项立即写入 memo；否则挂到所属槽位上。
    void Finish(const std::vector<Matrix2>& m, const std::vector<Matrix2>& mm, bool insert_memo);

    // 按顺序提交槽位挂起的 memo 项：若某个键此时已在 memo 中而首次写入的 phi_in 不同（顺序求值会
    // 用到不同的选项），返回 false 且不写入；否则写入所有缺失项并返回 true。
    bool CommitSlots(const std::vector<int>& slots);

    // 回溯一个槽位：(总代价, 每个运行带出的角度, 每个运行的代价)。
    std::int64_t Backtrack(int slot, std::vector<double>* angles, std::vector<int>* costs) const;

    // Python 端顺序求值使用的 memo 接口（与上面共用同一张表；Put 只在键缺失时写入）。
    const MemoOptions* MemoGet(const MemoKey& key) const;
    void MemoPut(const MemoKey& key, MemoOptions options);

 private:
    struct MemoKeyHash {
        std::size_t operator()(const MemoKey& k) const;
    };
    struct Unique {
        int row;
        double phi_in;
        MemoKey key;
        const MemoOptions* hit;  // memo 命中（否则为 nullptr）
        int compute;             // 计算表下标（命中时为 −1）
    };

    int beam_;
    int gate_weight_;
    int run_weight_;
    LengthModel model_;
    std::vector<std::vector<Entry>> current_;
    std::vector<std::vector<std::vector<Entry>>> layers_;
    std::unordered_map<MemoKey, MemoOptions, MemoKeyHash> memo_;
    std::vector<std::vector<std::pair<MemoKey, MemoOptions>>> pending_;
    // 本层
    std::vector<int> slots_;
    std::vector<int> ax_out_;
    std::vector<std::uint8_t> last_;
    std::vector<double> phi_fixed_;
    std::vector<int> run_length_;
    std::vector<std::uint8_t> base_changed_;
    std::vector<std::vector<int>> entry_unique_;  // 每行：束条目 → 唯一条目
    std::vector<Unique> unique_;
    std::vector<int> compute_row_;
    std::vector<int> compute_unique_;
    std::vector<int> compute_rid_;
    std::vector<int> compute_ax_out_;
    std::vector<std::vector<int>> wire_rids_;
    std::vector<std::vector<int>> wire_axes_;
    std::vector<int> wire_run_length_;
    std::vector<std::uint8_t> rotated_;  // 按计算表
    std::vector<int> rotate_in_;
    std::vector<Matrix2> rotate_in_mats_;
    std::vector<int> need_leading_;
    std::vector<int> option_offsets_;  // 按计算表
    std::vector<double> option_phis_;
    std::vector<int> option_costs_;
    std::vector<int> option_mm_;  // 选项 → mm 下标（−1：用运行矩阵 m）
    std::vector<int> option_rows_;
    std::vector<Matrix2> option_mats_;
};

}  // namespace quspect::native::transpiler
