// 单比特门最小化（quspect.transpiler._gauge._minimize）的整体原生实现。
//
// Python 端的原生路径（_minimize(..., native)）逐条遍历指令、用 numpy 做 2×2 乘法、逐个运行调用
// 合成函数并构造指令；这里把同一流程整体搬到 C++：
//
//   1. 划分：每条线上的单比特门运行与两比特门锚点（与 Python 相同的运行 / 锚点编号与顺序）；
//      门矩阵与对易轴由 Python 端的 matrix_of / axes_of 给出（每个门对象只调用一次）；
//   2. 束搜索（solve_all_native）、CX 朝向局部搜索（sweep_native / evaluate / solve）与输出，
//      全部按 Python 端的运算顺序执行：2×2 乘法走 numpy 自己的 BLAS（NumpyLinalg，与 numpy
//      matmul 的 BLAS 调用相同，逐位一致），束搜索的内核与 memo 是同一个 GaugeBeamSearch；
//   3. 输出运行的合成回调 Python 的 synthesize_one_qubit（同一矩阵只调用一次，结果相同），
//      指令按数据类 __init__ 的字段顺序直接构造（常见情形）。
//
// 遇到不能逐项复刻的输入（指令 / 比特类型不是普通的、矩阵布局不同、内核不能复刻的数值、
// 需要抛出异常的情形）返回 None，由 Python 端按原实现处理（结果与异常都相同）。
#include "gauge_bindings.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_interned.hpp"
#include "run_helpers.hpp"
#include "quspect_native/common/parallel.hpp"
#include "quspect_native/synthesis/numpy_compat.hpp"
#include "quspect_native/transpiler/one_qubit_gauge.hpp"
#include "synthesis_bindings.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using transpiler::Complex;
using transpiler::GaugeBeamSearch;
using transpiler::Matrix2;

constexpr double kTwoPi = 2.0 * 3.141592653589793;

using runs::Attr;
using runs::Decline;
using runs::ReadBits;
using runs::ReadMatrix2;
using runs::Truthy;
using GaugeNames = runs::RunNames;
using runs::Names;

struct GateRun {
    std::vector<PyObject*> gates;   // 借用：由输入指令持有
    std::vector<PyObject*> instrs;  // 借用：输入指令
    Matrix2 matrix;
    bool native = true;
    bool plain = true;
};

struct Anchor {
    int axis;  // 原朝向下的对易轴（−1 无）
    int gate;  // 可反向的门编号，或 −1
};

// 一个线槽中的比特位置：(线, 锚点下标)。
struct WirePos {
    int wire;
    int anchor;
};

struct Solved {
    std::int64_t total;
    std::vector<double> angles;
    std::vector<int> costs;
};

// 与 _gauge._SWAP_AXIS.get 相同：z ↔ x，y → 无。
int SwapAxis(int axis) {
    if (axis == 0) {
        return 1;
    }
    if (axis == 1) {
        return 0;
    }
    return -1;
}

// 初始束搜索按线分组并行的最小工作量（运行数）：更小的输入在调用线程中计算。
constexpr std::size_t kMinParallelRuns = 8192;

int SearchThreads(std::size_t runs) {
    if (runs < kMinParallelRuns) {
        return 1;
    }
    const auto useful = static_cast<int>(std::min<std::size_t>(runs / (kMinParallelRuns / 2), 64));
    return std::max(1, std::min(common::CompileThreads(), useful));
}

struct Options {
    py::object instr_cls;
    py::object type_gate;
    py::object type_barrier;
    py::object gate_cls;
    py::object one_qubit;
    py::object matrix_of;
    py::object axes_of;
    py::object synthesize;
    Matrix2 h;
    std::array<Matrix2, 3> to_z;
    std::array<Matrix2, 3> to_z_conj;
    int beam;
    int gate_weight;
    int run_weight;
    int sweeps;
    double atol;
};

class Minimizer {
 public:
    Minimizer(const synthesis::NumpyLinalg& la, const transpiler::LengthModel& model, const Options& opt)
        : mat_(la),
          model_(model),
          opt_(opt),
          synth_(opt.synthesize),
          factory_(opt.instr_cls, opt.type_gate, opt.gate_cls) {}

    // 结果：新指令列表；py::none() 表示原线路不变（单比特门不减少）。Decline：交回 Python。
    py::object Run(const py::sequence& instrs) {
        py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(instrs.ptr(), "instructions"));
        if (!fast) {
            throw py::error_already_set();
        }
        const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast.ptr());
        PyObject** items = PySequence_Fast_ITEMS(fast.ptr());
        Split(items, count);
        flipped_.assign(flip_instr_.size(), 0);
        SolveAll();
        for (int sweep = 0; sweep < opt_.sweeps; ++sweep) {
            if (!Sweep()) {
                break;
            }
        }
        return Emit(items, count);
    }

 private:
    // ---- 矩阵运算（numpy 语义） ----
    Matrix2 MatMul(const Matrix2& a, const Matrix2& b) const {
        return mat_.Mul(a, b);
    }
    // ``V†·m·V``：numpy 的 ``v.conj().T @ m @ v``（左操作数是 C 连续数组的转置视图）。
    Matrix2 ZFrame(const Matrix2& m, int axis) const {
        const Matrix2 tmp = mat_.MulT(opt_.to_z_conj[static_cast<std::size_t>(axis)], m);
        return mat_.Mul(tmp, opt_.to_z[static_cast<std::size_t>(axis)]);
    }

    int Length(const Matrix2& m) const {
        const auto len = transpiler::FastLength(m, model_);
        if (!len) {
            throw Decline{};
        }
        return *len;
    }

    // ---- 1. 划分 ----
    int WireOf(long long q) {
        const auto it = wire_of_.find(q);
        if (it != wire_of_.end()) {
            return it->second;
        }
        const int w = static_cast<int>(all_wires_.size());
        wire_of_.emplace(q, w);
        all_wires_.push_back(q);
        wire_runs_.emplace_back();
        wire_anchors_.emplace_back();
        current_.push_back(-1);
        return w;
    }

    int NewRun(int w) {
        runs_.emplace_back();
        runs_.back().matrix = Identity();
        const int rid = static_cast<int>(runs_.size()) - 1;
        current_[static_cast<std::size_t>(w)] = rid;
        wire_runs_[static_cast<std::size_t>(w)].push_back(rid);
        return rid;
    }

    int CurrentOrNew(int w) {
        const int rid = current_[static_cast<std::size_t>(w)];
        return rid >= 0 ? rid : NewRun(w);
    }

    static Matrix2 Identity() {
        return runs::Identity2();
    }

    // matrix_of(gate)（每个门对象只调用一次；None → −1）。
    int MatrixIndex(PyObject* gate) {
        const auto it = matrix_index_.find(gate);
        if (it != matrix_index_.end()) {
            return it->second;
        }
        py::object m = opt_.matrix_of(py::handle(gate));
        int index = -1;
        if (!m.is_none()) {
            index = static_cast<int>(gate_matrices_.size());
            gate_matrices_.push_back(ReadMatrix2(m.ptr()));
        }
        keep_.push_back(py::reinterpret_borrow<py::object>(gate));
        matrix_index_.emplace(gate, index);
        return index;
    }

    std::pair<int, int> AxesOf(PyObject* gate) {
        const auto it = axes_.find(gate);
        if (it != axes_.end()) {
            return it->second;
        }
        py::object ax = opt_.axes_of(py::handle(gate));
        if (!PyTuple_CheckExact(ax.ptr()) || PyTuple_GET_SIZE(ax.ptr()) != 2) {
            throw Decline{};
        }
        auto code = [](PyObject* a) -> int {
            if (a == Py_None) {
                return -1;
            }
            if (!PyUnicode_CheckExact(a)) {
                throw Decline{};
            }
            const std::string s = py::reinterpret_borrow<py::str>(a);
            if (s == "z") {
                return 0;
            }
            if (s == "x") {
                return 1;
            }
            if (s == "y") {
                return 2;
            }
            throw Decline{};
        };
        const std::pair<int, int> out{code(PyTuple_GET_ITEM(ax.ptr(), 0)), code(PyTuple_GET_ITEM(ax.ptr(), 1))};
        keep_.push_back(py::reinterpret_borrow<py::object>(gate));
        axes_.emplace(gate, out);
        return out;
    }

    void Split(PyObject** items, Py_ssize_t count) {
        GaugeNames& n = Names();
        auto* instr_type = reinterpret_cast<PyTypeObject*>(opt_.instr_cls.ptr());
        instr_runs_.assign(static_cast<std::size_t>(count), {});
        anchored_.assign(static_cast<std::size_t>(count), 0);
        std::vector<long long> qubits;
        std::vector<long long> unique;
        for (Py_ssize_t idx = 0; idx < count; ++idx) {
            PyObject* ins = items[idx];
            if (Py_TYPE(ins) != instr_type) {
                throw Decline{};
            }
            py::object type = Attr(ins, n.type);
            const bool is_gate = type.is(opt_.type_gate);
            const bool is_barrier = type.is(opt_.type_barrier);
            py::object gate = Attr(ins, n.gate);
            ReadBits(Attr(ins, n.qubits).ptr(), &qubits);
            if (is_gate && !gate.is_none() && qubits.size() == 1) {
                const int mi = MatrixIndex(gate.ptr());
                if (mi >= 0) {
                    const int w = WireOf(qubits[0]);
                    GateRun& run = runs_[static_cast<std::size_t>(CurrentOrNew(w))];
                    run.gates.push_back(gate.ptr());
                    run.instrs.push_back(ins);
                    if (Truthy(Attr(ins, n.label).ptr()) || Truthy(Attr(ins, n.clbits).ptr())) {
                        run.plain = false;
                    }
                    run.matrix = MatMul(gate_matrices_[static_cast<std::size_t>(mi)], run.matrix);
                    const int contains = PySequence_Contains(opt_.one_qubit.ptr(), Attr(gate.ptr(), n.name).ptr());
                    if (contains < 0) {
                        throw py::error_already_set();
                    }
                    if (contains == 0) {
                        run.native = false;
                    }
                    ++original_1q_;
                    continue;
                }
            }
            // list(dict.fromkeys(ins.qubits))
            unique.clear();
            for (long long q : qubits) {
                if (std::find(unique.begin(), unique.end(), q) == unique.end()) {
                    unique.push_back(q);
                }
            }
            if (unique.empty() && is_barrier) {
                unique = all_wires_;
            }
            std::vector<int> axes(unique.size(), -1);
            int flip = -1;
            if (is_gate && !gate.is_none() && qubits.size() == 2) {
                const auto ax = AxesOf(gate.ptr());
                if (unique.size() != 2) {
                    throw Decline{};  // zip(..., strict=True) 会抛出 ValueError
                }
                axes = {ax.first, ax.second};
                py::object name = Attr(gate.ptr(), n.name);
                if (PyUnicode_CheckExact(name.ptr()) && PyUnicode_CompareWithASCIIString(name.ptr(), "cx") == 0
                    && !Truthy(Attr(gate.ptr(), n.params).ptr())) {
                    flip = static_cast<int>(flip_instr_.size());
                    flip_instr_.push_back(static_cast<int>(idx));
                }
            }
            std::vector<std::pair<int, int>>& entries = instr_runs_[static_cast<std::size_t>(idx)];
            std::array<WirePos, 2> pos{};
            for (std::size_t k = 0; k < unique.size(); ++k) {
                const int w = WireOf(unique[k]);
                const int rid = CurrentOrNew(w);
                entries.emplace_back(w, rid);
                auto& anchors = wire_anchors_[static_cast<std::size_t>(w)];
                anchors.push_back(Anchor{axes[k], flip});
                if (k < 2) {
                    pos[k] = WirePos{w, static_cast<int>(anchors.size()) - 1};
                }
                NewRun(w);
            }
            if (flip >= 0) {
                flip_pos_.push_back(pos);
            }
            anchored_[static_cast<std::size_t>(idx)] = 1;
        }
        run_len_.resize(runs_.size());
        for (std::size_t r = 0; r < runs_.size(); ++r) {
            run_len_[r] = runs_[r].native ? static_cast<int>(runs_[r].gates.size()) : -1;
        }
    }

    // ---- 锚点（当前朝向） ----
    bool IsFlipped(const Anchor& a) const {
        return a.gate >= 0 && flipped_[static_cast<std::size_t>(a.gate)] != 0;
    }
    int AnchorAxis(const Anchor& a) const {
        if (IsFlipped(a) && a.axis >= 0) {
            return SwapAxis(a.axis);
        }
        return a.axis;
    }

    // run_base：运行 i 乘上两侧反向 CX 的 Hadamard。
    std::pair<Matrix2, bool> RunBase(int w, int i) const {
        const auto& anchors = wire_anchors_[static_cast<std::size_t>(w)];
        Matrix2 m = runs_[static_cast<std::size_t>(wire_runs_[static_cast<std::size_t>(w)][static_cast<std::size_t>(i)])]
                        .matrix;
        bool changed = false;
        if (i > 0 && IsFlipped(anchors[static_cast<std::size_t>(i - 1)])) {
            m = MatMul(m, opt_.h);
            changed = true;
        }
        if (i < static_cast<int>(anchors.size()) && IsFlipped(anchors[static_cast<std::size_t>(i)])) {
            m = MatMul(opt_.h, m);
            changed = true;
        }
        return {m, changed};
    }

    // ---- 2. 初始束搜索（solve_all_native） ----
    //
    // 反向任何 CX 之前各条线互不相关：memo 键含运行编号（每个运行只属于一条线），一条线的计算
    // 从不读到另一条线写入的项。所以可以把线分成几组，在各自的 GaugeBeamSearch 上按层推进
    // （可以并行，见 SearchThreads），最后把各组的 memo 并入主对象——结果与所有线一起分层
    // 推进完全相同。
    void SolveAll() {
        const std::size_t num_wires = all_wires_.size();
        search_ = std::make_unique<GaugeBeamSearch>(0, opt_.beam, opt_.gate_weight, opt_.run_weight, model_);
        outs_.assign(num_wires, {});
        costs_.assign(num_wires, {});
        // 按运行数大致均分的连续线组
        const int threads = num_wires == 0 ? 1 : std::min<int>(SearchThreads(runs_.size()), static_cast<int>(num_wires));
        std::vector<std::size_t> bounds{0};
        {
            const double share = static_cast<double>(runs_.size()) / threads;
            std::size_t acc = 0;
            for (std::size_t w = 0; w < num_wires; ++w) {
                acc += wire_runs_[w].size();
                if (static_cast<int>(bounds.size()) < threads && acc >= share * static_cast<double>(bounds.size())
                    && w + 1 < num_wires) {
                    bounds.push_back(w + 1);
                }
            }
            bounds.push_back(num_wires);
        }
        const std::size_t groups = bounds.size() - 1;
        std::vector<std::unique_ptr<GaugeBeamSearch>> parts(groups);
        std::vector<std::function<void()>> tasks;
        for (std::size_t k = 0; k < groups; ++k) {
            tasks.emplace_back([&, k] { parts[k] = SolveWires(bounds[k], bounds[k + 1]); });
        }
        if (groups <= 1) {
            common::RunTasks(tasks, 1);
        } else {
            py::gil_scoped_release release;
            common::RunTasks(tasks, static_cast<int>(groups));
        }
        for (std::size_t k = 0; k < groups; ++k) {
            for (std::size_t w = bounds[k]; w < bounds[k + 1]; ++w) {
                parts[k]->Backtrack(static_cast<int>(w - bounds[k]), &outs_[w], &costs_[w]);
            }
            search_->AbsorbMemo(*parts[k]);
        }
    }

    // 线 [begin, end) 的分层束搜索（不调用 Python，可在释放 GIL 的线程中运行）。
    std::unique_ptr<GaugeBeamSearch> SolveWires(std::size_t begin, std::size_t end) const {
        auto search = std::make_unique<GaugeBeamSearch>(static_cast<int>(end - begin), opt_.beam, opt_.gate_weight,
                                                        opt_.run_weight, model_);
        std::vector<std::vector<int>> rids(wire_runs_.begin() + static_cast<std::ptrdiff_t>(begin),
                                           wire_runs_.begin() + static_cast<std::ptrdiff_t>(end));
        std::vector<std::vector<int>> axes(end - begin);
        std::size_t depth = 0;
        for (std::size_t w = begin; w < end; ++w) {
            for (const Anchor& a : wire_anchors_[w]) {
                axes[w - begin].push_back(a.axis);
            }
            depth = std::max(depth, wire_runs_[w].size());
        }
        search->SetWires(std::move(rids), std::move(axes), run_len_);
        std::vector<Matrix2> m;
        std::vector<Matrix2> z;
        std::vector<Matrix2> mm;
        for (std::size_t i = 0; i < depth; ++i) {
            search->PrepareLayer(static_cast<int>(i));
            const auto& crid = search->ComputeRid();
            m.resize(crid.size());
            for (std::size_t c = 0; c < crid.size(); ++c) {
                m[c] = runs_[static_cast<std::size_t>(crid[c])].matrix;
            }
            Layer(*search, &m, search->ComputeAxOut(), &z, &mm, true);
        }
        return search;
    }

    // 一层的矩阵乘法与 Options / Finish（m：计算表每项的运行矩阵，已含 run_base；ax_out 按计算表）。
    void Layer(GaugeBeamSearch& search, std::vector<Matrix2>* m, const std::vector<int>& ax_out,
               std::vector<Matrix2>* z, std::vector<Matrix2>* mm, bool insert_memo) const {
        const auto& rot_in = search.RotateIn();
        const auto& rot_mats = search.RotateInMatrices();
        for (std::size_t k = 0; k < rot_in.size(); ++k) {
            Matrix2& x = (*m)[static_cast<std::size_t>(rot_in[k])];
            x = MatMul(x, rot_mats[k]);
        }
        const auto& lead = search.NeedLeading();
        z->resize(lead.size());
        for (std::size_t k = 0; k < lead.size(); ++k) {
            const auto c = static_cast<std::size_t>(lead[k]);
            (*z)[k] = ZFrame((*m)[c], ax_out[c]);
        }
        search.Options(*z);
        const auto& rows = search.OptionRows();
        const auto& rot_out = search.OptionMatrices();
        mm->resize(rows.size());
        for (std::size_t k = 0; k < rows.size(); ++k) {
            (*mm)[k] = MatMul(rot_out[k], (*m)[static_cast<std::size_t>(rows[k])]);
        }
        search.Finish(*m, *mm, insert_memo);
    }

    // ---- 3. CX 朝向局部搜索 ----
    bool FlippedFor(const Anchor& a, int g) const {
        return a.gate >= 0 && ((flipped_[static_cast<std::size_t>(a.gate)] != 0) != (a.gate == g));
    }
    int AxisFor(const Anchor& a, int g) const {
        int ax = a.axis;
        if (ax >= 0 && FlippedFor(a, g)) {
            ax = SwapAxis(ax);
        }
        return ax;
    }

    // sweep_native：所有门先按本轮开始时的状态投机求值，再按门的顺序提交。
    bool Sweep() {
        struct Chain {
            int g;
            int w;
            int j;
        };
        std::vector<Chain> chains;
        std::vector<double> start;
        for (std::size_t g = 0; g < flip_pos_.size(); ++g) {
            for (const WirePos& p : flip_pos_[g]) {
                chains.push_back(Chain{static_cast<int>(g), p.wire, p.anchor});
                start.push_back(p.anchor > 0 ? outs_[static_cast<std::size_t>(p.wire)][static_cast<std::size_t>(p.anchor - 1)]
                                             : 0.0);
            }
        }
        search_->Reset(start);
        std::vector<Matrix2> m;
        std::vector<Matrix2> z;
        std::vector<Matrix2> mm;
        for (int layer = 0; layer < 2; ++layer) {
            const std::size_t nrows = chains.size();
            std::vector<int> slots(nrows), rids(nrows), ax_in(nrows), ax_out(nrows), run_len(nrows);
            std::vector<std::uint8_t> fp(nrows), fn(nrows), last(nrows), changed(nrows);
            std::vector<double> phi_fixed(nrows);
            for (std::size_t s = 0; s < nrows; ++s) {
                const Chain& c = chains[s];
                const auto& anchors = wire_anchors_[static_cast<std::size_t>(c.w)];
                const int i = c.j + layer;
                const bool has_prev = i > 0;
                const bool has_next = i < static_cast<int>(anchors.size());
                const bool f_prev = has_prev && FlippedFor(anchors[static_cast<std::size_t>(i - 1)], c.g);
                const bool f_next = has_next && FlippedFor(anchors[static_cast<std::size_t>(i)], c.g);
                const int rid = wire_runs_[static_cast<std::size_t>(c.w)][static_cast<std::size_t>(i)];
                slots[s] = static_cast<int>(s);
                rids[s] = rid;
                fp[s] = f_prev ? 1 : 0;
                fn[s] = f_next ? 1 : 0;
                ax_in[s] = has_prev ? AxisFor(anchors[static_cast<std::size_t>(i - 1)], c.g) : -1;
                ax_out[s] = has_next ? AxisFor(anchors[static_cast<std::size_t>(i)], c.g) : -1;
                last[s] = static_cast<std::uint8_t>(layer);
                phi_fixed[s] =
                    layer != 0 ? outs_[static_cast<std::size_t>(c.w)][static_cast<std::size_t>(c.j + 1)] : 0.0;
                run_len[s] = run_len_[static_cast<std::size_t>(rid)];
                changed[s] = (f_prev || f_next) ? 1 : 0;
            }
            search_->Prepare(slots, rids, fp, fn, ax_in, ax_out, last, phi_fixed, run_len, changed);
            // native_layer：m = matrices[rids[crow]]；fp 行右乘 H，fn 行左乘 H
            const auto& crow = search_->ComputeRow();
            m.resize(crow.size());
            std::vector<int> cax(crow.size());
            for (std::size_t c = 0; c < crow.size(); ++c) {
                const auto r = static_cast<std::size_t>(crow[c]);
                m[c] = runs_[static_cast<std::size_t>(rids[r])].matrix;
                cax[c] = ax_out[r];
            }
            for (std::size_t c = 0; c < crow.size(); ++c) {
                if (fp[static_cast<std::size_t>(crow[c])] != 0) {
                    m[c] = MatMul(m[c], opt_.h);
                }
            }
            for (std::size_t c = 0; c < crow.size(); ++c) {
                if (fn[static_cast<std::size_t>(crow[c])] != 0) {
                    m[c] = MatMul(opt_.h, m[c]);
                }
            }
            Layer(*search_, &m, cax, &z, &mm, false);
        }
        std::vector<Solved> results(chains.size());
        for (std::size_t s = 0; s < chains.size(); ++s) {
            results[s].total = search_->Backtrack(static_cast<int>(s), &results[s].angles, &results[s].costs);
        }
        bool improved = false;
        std::unordered_set<std::int64_t> dirty;
        auto dkey = [](int w, int a) { return (static_cast<std::int64_t>(w) << 32) ^ static_cast<std::uint32_t>(a); };
        for (std::size_t g = 0; g < flip_pos_.size(); ++g) {
            const WirePos pa = flip_pos_[g][0];
            const WirePos pb = flip_pos_[g][1];
            bool conflict = false;
            for (const WirePos& p : {pa, pb}) {
                for (int a : {p.anchor - 2, p.anchor - 1, p.anchor + 1}) {
                    conflict = conflict || dirty.count(dkey(p.wire, a)) != 0;
                }
            }
            bool accepted;
            if (conflict || !search_->CommitSlots({static_cast<int>(2 * g), static_cast<int>(2 * g + 1)})) {
                accepted = Evaluate(static_cast<int>(g));
            } else {
                const Solved& ra = results[2 * g];
                const Solved& rb = results[2 * g + 1];
                const std::int64_t old = CostSum(pa.wire, pa.anchor, pa.anchor + 1) + CostSum(pb.wire, pb.anchor, pb.anchor + 1);
                accepted = ra.total + rb.total < old;
                if (accepted) {
                    Store(pa.wire, pa.anchor, ra);
                    Store(pb.wire, pb.anchor, rb);
                    flipped_[g] ^= 1;
                }
            }
            if (accepted) {
                dirty.insert(dkey(pa.wire, pa.anchor));
                dirty.insert(dkey(pb.wire, pb.anchor));
                improved = true;
            }
        }
        return improved;
    }

    std::int64_t CostSum(int w, int s, int e) const {
        std::int64_t sum = 0;
        for (int i = s; i <= e; ++i) {
            sum += costs_[static_cast<std::size_t>(w)][static_cast<std::size_t>(i)];
        }
        return sum;
    }

    void Store(int w, int s, const Solved& r) {
        for (std::size_t k = 0; k < r.angles.size(); ++k) {
            outs_[static_cast<std::size_t>(w)][static_cast<std::size_t>(s) + k] = r.angles[k];
            costs_[static_cast<std::size_t>(w)][static_cast<std::size_t>(s) + k] = r.costs[k];
        }
    }

    // evaluate(g)：反向门 g 并重解两条线上的窗口（_WINDOW = 0：锚点两侧各一个运行）。
    bool Evaluate(int g) {
        const WirePos pa = flip_pos_[static_cast<std::size_t>(g)][0];
        const WirePos pb = flip_pos_[static_cast<std::size_t>(g)][1];
        auto window = [&](const WirePos& p) {
            const int k = static_cast<int>(wire_runs_[static_cast<std::size_t>(p.wire)].size()) - 1;
            const int s = std::max(0, p.anchor);
            const int e = std::min(k, p.anchor + 1);
            const auto& o = outs_[static_cast<std::size_t>(p.wire)];
            const double phi_s = s > 0 ? o[static_cast<std::size_t>(s - 1)] : 0.0;
            return std::array<double, 4>{static_cast<double>(s), static_cast<double>(e), phi_s,
                                         o[static_cast<std::size_t>(e)]};
        };
        const auto wa = window(pa);
        const auto wb = window(pb);
        const auto sa = static_cast<int>(wa[0]);
        const auto ea = static_cast<int>(wa[1]);
        const auto sb = static_cast<int>(wb[0]);
        const auto eb = static_cast<int>(wb[1]);
        const std::int64_t old = CostSum(pa.wire, sa, ea) + CostSum(pb.wire, sb, eb);
        flipped_[static_cast<std::size_t>(g)] ^= 1;
        const Solved ra = Solve(pa.wire, sa, ea, wa[2], wa[3]);
        const Solved rb = Solve(pb.wire, sb, eb, wb[2], wb[3]);
        if (ra.total + rb.total < old) {
            Store(pa.wire, sa, ra);
            Store(pb.wire, sb, rb);
            return true;
        }
        flipped_[static_cast<std::size_t>(g)] ^= 1;
        return false;
    }

    // solve(q, s, e, phi_s, phi_e)：固定两端角度时运行 s..e 的最优带出角（memo 与束搜索共用）。
    Solved Solve(int w, int s, int e, double phi_s, double phi_e) {
        struct Item {
            std::int64_t total;
            double phi;
            int parent;
            int cost;
        };
        const auto& anchors = wire_anchors_[static_cast<std::size_t>(w)];
        const auto& rids = wire_runs_[static_cast<std::size_t>(w)];
        std::vector<std::vector<Item>> layers;
        std::vector<Item> prev{Item{0, phi_s, -1, 0}};
        for (int i = s; i <= e; ++i) {
            const int rid = rids[static_cast<std::size_t>(i)];
            const bool fp = i > 0 && IsFlipped(anchors[static_cast<std::size_t>(i - 1)]);
            const bool fn = i < static_cast<int>(anchors.size()) && IsFlipped(anchors[static_cast<std::size_t>(i)]);
            const int ax_in = i > 0 ? AnchorAxis(anchors[static_cast<std::size_t>(i - 1)]) : -1;
            const int ax_out = i < static_cast<int>(anchors.size()) ? AnchorAxis(anchors[static_cast<std::size_t>(i)]) : -1;
            const bool last = i == e;
            std::vector<std::pair<std::int64_t, Item>> nxt;  // 按插入顺序的 dict
            for (std::size_t pi = 0; pi < prev.size(); ++pi) {
                const std::int64_t total = prev[pi].total;
                const double phi_in = prev[pi].phi;
                if (!std::isfinite(phi_in * 1e10)) {
                    throw Decline{};  // Python 的 round() 抛出异常
                }
                const GaugeBeamSearch::MemoKey key{rid,  fp,   fn, transpiler::PyRound(phi_in * 1e10),
                                                   last, Bits(last ? phi_e : 0.0)};
                const GaugeBeamSearch::MemoOptions* options = search_->MemoGet(key);
                GaugeBeamSearch::MemoOptions computed;
                if (options == nullptr) {
                    computed = RunOptions(w, i, phi_in, ax_in, ax_out, last, phi_e);
                    search_->MemoPut(key, computed);
                    options = search_->MemoGet(key);
                }
                for (std::size_t j = 0; j < options->phis.size(); ++j) {
                    const double phi = options->phis[j];
                    const int c = options->costs[j];
                    if (!std::isfinite(phi)) {
                        throw Decline{};
                    }
                    const std::int64_t akey = transpiler::PyRound(std::remainder(phi, kTwoPi) * 1e8);
                    auto it = std::find_if(nxt.begin(), nxt.end(), [akey](const auto& kv) { return kv.first == akey; });
                    const Item item{total + c, phi, static_cast<int>(pi), c};
                    if (it == nxt.end()) {
                        nxt.emplace_back(akey, item);
                    } else if (total + c < it->second.total) {
                        it->second = item;
                    }
                }
            }
            std::vector<Item> beam;
            beam.reserve(nxt.size());
            for (const auto& kv : nxt) {
                beam.push_back(kv.second);
            }
            std::stable_sort(beam.begin(), beam.end(), [](const Item& a, const Item& b) {
                if (a.total != b.total) {
                    return a.total < b.total;
                }
                return std::fabs(a.phi) < std::fabs(b.phi);
            });
            if (beam.size() > static_cast<std::size_t>(opt_.beam)) {
                beam.resize(static_cast<std::size_t>(opt_.beam));
            }
            prev = beam;
            layers.push_back(std::move(beam));
        }
        Solved out;
        out.total = layers.back()[0].total;
        const std::size_t n = layers.size();
        out.angles.assign(n, 0.0);
        out.costs.assign(n, 0);
        int best = 0;
        for (std::size_t k = n; k-- > 0;) {
            const Item& it = layers[k][static_cast<std::size_t>(best)];
            out.angles[k] = it.phi;
            out.costs[k] = it.cost;
            best = it.parent;
        }
        return out;
    }

    static std::uint64_t Bits(double x) {
        std::uint64_t b;
        std::memcpy(&b, &x, sizeof(b));
        return b;
    }

    // run_options_native：运行 i 的（带出角, 代价）候选。
    GaugeBeamSearch::MemoOptions RunOptions(int w, int i, double phi_in, int ax_in, int ax_out, bool last,
                                            double phi_e) {
        const int rid = wire_runs_[static_cast<std::size_t>(w)][static_cast<std::size_t>(i)];
        auto [m, changed] = RunBase(w, i);
        if (ax_in >= 0 && phi_in != 0.0) {
            m = MatMul(m, transpiler::AxisRotation(ax_in, phi_in));
            changed = true;
        }
        std::vector<double> opts;
        if (last) {
            opts.push_back(phi_e);
        } else if (ax_out < 0) {
            opts.push_back(0.0);
        } else {
            opts.push_back(0.0);
            const auto lead = transpiler::LeadingAnglesInZFrame(ZFrame(m, ax_out));
            if (!lead) {
                throw Decline{};
            }
            for (int j = 0; j < lead->count; ++j) {
                opts.push_back(lead->angles[static_cast<std::size_t>(j)]);
            }
        }
        const int base_len = Length(m);
        GaugeBeamSearch::MemoOptions out;
        out.phi_in = phi_in;
        const int run_len = run_len_[static_cast<std::size_t>(rid)];
        for (double phi : opts) {
            int c;
            bool ch;
            if (phi != 0.0 && ax_out >= 0) {
                c = Length(MatMul(transpiler::AxisRotation(ax_out, -phi), m));
                ch = true;
            } else {
                c = base_len;
                ch = changed;
            }
            if (!ch && run_len >= 0) {
                c = std::min(c, run_len);
            }
            out.phis.push_back(phi);
            out.costs.push_back(opt_.gate_weight * c + (c != 0 ? opt_.run_weight : 0));
        }
        return out;
    }

    // ---- 4. 输出 ----
    // abs(m[k]) < atol（numpy 复数标量 abs）；离阈值太近时交回 Python。
    bool Below(const Complex& v) const {
        const double a = std::hypot(v.re, v.im);
        if (std::fabs(a - opt_.atol) <= opt_.atol * 1e-12) {
            throw Decline{};
        }
        return a < opt_.atol;
    }

    py::object Emit(PyObject** items, Py_ssize_t count) {
        // final：每个运行的输出门序列（keep：保留原门）
        std::vector<py::object> final_seq(runs_.size());
        std::vector<std::uint8_t> keep(runs_.size(), 0);
        long long total = 0;
        for (std::size_t w = 0; w < all_wires_.size(); ++w) {
            const auto& anchors = wire_anchors_[w];
            const auto& rids = wire_runs_[w];
            for (std::size_t i = 0; i < rids.size(); ++i) {
                const auto rid = static_cast<std::size_t>(rids[i]);
                const GateRun& run = runs_[rid];
                auto [m, changed] = RunBase(static_cast<int>(w), static_cast<int>(i));
                const double phi_in = i > 0 ? outs_[w][i - 1] : 0.0;
                const double phi_out = outs_[w][i];
                if (phi_in != 0.0) {
                    const int ax = AnchorAxis(anchors[i - 1]);
                    if (ax < 0) {
                        throw Decline{};
                    }
                    m = MatMul(m, transpiler::AxisRotation(ax, phi_in));
                    changed = true;
                }
                if (phi_out != 0.0) {
                    const int ax = AnchorAxis(anchors[i]);
                    if (ax < 0) {
                        throw Decline{};
                    }
                    m = MatMul(transpiler::AxisRotation(ax, -phi_out), m);
                    changed = true;
                }
                if (!changed && run.native
                    && (run.gates.empty() || (run.gates.size() == 1 && !(Below(m[1]) && Below(m[2]))))) {
                    keep[rid] = 1;
                    total += static_cast<long long>(run.gates.size());
                    continue;
                }
                py::object seq = synth_.Get(m);
                if (!changed && run.native
                    && (seq.is_none() || static_cast<Py_ssize_t>(run.gates.size()) <= PyList_GET_SIZE(seq.ptr()))) {
                    keep[rid] = 1;
                    total += static_cast<long long>(run.gates.size());
                    continue;
                }
                if (seq.is_none()) {
                    if (!run.gates.empty() || changed) {
                        throw Decline{};  // Python 端抛出 ValueError
                    }
                    final_seq[rid] = py::list();
                    continue;
                }
                total += PyList_GET_SIZE(seq.ptr());
                final_seq[rid] = seq;
            }
        }
        if (total > original_1q_) {
            return py::none();
        }
        py::list out;
        auto emit = [&](int w, int rid) {
            const auto r = static_cast<std::size_t>(rid);
            const GateRun& run = runs_[r];
            const long long q = all_wires_[static_cast<std::size_t>(w)];
            if (keep[r] != 0) {
                if (run.plain) {
                    for (PyObject* ins : run.instrs) {
                        out.append(py::handle(ins));
                    }
                } else {
                    for (PyObject* g : run.gates) {
                        out.append(factory_.OneQubit(g, q));
                    }
                }
                return;
            }
            const py::object& seq = final_seq[r];
            for (Py_ssize_t k = 0; k < PyList_GET_SIZE(seq.ptr()); ++k) {
                out.append(factory_.OneQubit(PyList_GET_ITEM(seq.ptr(), k), q));
            }
        };
        std::unordered_map<int, int> flip_of;
        for (std::size_t g = 0; g < flip_instr_.size(); ++g) {
            flip_of.emplace(flip_instr_[g], static_cast<int>(g));
        }
        GaugeNames& n = Names();
        for (Py_ssize_t idx = 0; idx < count; ++idx) {
            if (anchored_[static_cast<std::size_t>(idx)] == 0) {
                continue;  // 并入了某个运行
            }
            for (const auto& [w, rid] : instr_runs_[static_cast<std::size_t>(idx)]) {
                emit(w, rid);
            }
            PyObject* ins = items[idx];
            const auto it = flip_of.find(static_cast<int>(idx));
            if (it != flip_of.end() && flipped_[static_cast<std::size_t>(it->second)] != 0) {
                py::object qubits = Attr(ins, n.qubits);  // 普通 int 组成的 tuple（划分时已检查）
                py::tuple rq(PyTuple_GET_SIZE(qubits.ptr()));
                const Py_ssize_t nq = PyTuple_GET_SIZE(qubits.ptr());
                for (Py_ssize_t k = 0; k < nq; ++k) {
                    rq[k] = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(qubits.ptr(), nq - 1 - k));
                }
                py::dict kw;
                kw[n.type] = Attr(ins, n.type);
                kw[n.gate] = Attr(ins, n.gate);
                kw[n.qubits] = rq;
                kw[n.label] = Attr(ins, n.label);
                out.append(opt_.instr_cls(**kw));
            } else {
                out.append(py::handle(ins));
            }
        }
        for (std::size_t w = 0; w < all_wires_.size(); ++w) {
            emit(static_cast<int>(w), current_[w]);
        }
        return std::move(out);
    }

    runs::Mat2 mat_;
    const transpiler::LengthModel& model_;
    const Options& opt_;
    runs::SynthCache synth_;
    runs::InstructionFactory factory_;

    std::vector<GateRun> runs_;
    std::vector<int> run_len_;
    std::unordered_map<long long, int> wire_of_;
    std::vector<long long> all_wires_;
    std::vector<std::vector<int>> wire_runs_;
    std::vector<std::vector<Anchor>> wire_anchors_;
    std::vector<int> current_;
    std::vector<std::vector<std::pair<int, int>>> instr_runs_;  // 指令 → [(线, 运行)]
    std::vector<std::uint8_t> anchored_;
    std::vector<int> flip_instr_;
    std::vector<std::array<WirePos, 2>> flip_pos_;
    std::vector<std::uint8_t> flipped_;
    long long original_1q_ = 0;

    std::unordered_map<PyObject*, int> matrix_index_;
    std::vector<Matrix2> gate_matrices_;
    std::unordered_map<PyObject*, std::pair<int, int>> axes_;
    std::vector<py::object> keep_;  // 缓存键（门对象）的强引用

    std::unique_ptr<GaugeBeamSearch> search_;
    std::vector<std::vector<double>> outs_;
    std::vector<std::vector<int>> costs_;

};

Matrix2 ArgMatrix(const py::handle& obj) {
    return ReadMatrix2(obj.ptr());
}

}  // namespace

void BindGauge(py::module_& m) {
    m.def(
        "_gauge_arithmetic",
        [](const py::object& synth) {
            transpiler::GaugeArithmetic a;  // 默认（arm64 上 clang 的形式）
            if (!synth.is_none()) {
                const synthesis::NumpyLinalg* la = KakLinalg(synth);
                if (la == nullptr) {
                    throw std::invalid_argument("synth must be a _KakSynthesizer or None");
                }
                const synthesis::ArithModes& modes = la->modes();
                a.py_mul = static_cast<int>(modes.py);
                a.np_scalar_mul = static_cast<int>(modes.np_scalar);
                a.scalar_arr_mul = static_cast<int>(modes.scalar_arr);
                // CPython 的复数除法与乘法由同一个编译器、同样的选项编译：收缩与否相同
                a.py_div_fma = modes.py != synthesis::MulMode::kPlain;
                a.np_div_fma = modes.np_div == synthesis::DivMode::kFma;
            }
            transpiler::SetGaugeArithmetic(a);
            return py::make_tuple(a.py_mul, a.np_scalar_mul, a.scalar_arr_mul, a.py_div_fma, a.np_div_fma);
        },
        py::arg("synth"),
        "Set the complex-arithmetic semantics of the single-qubit minimization kernels: the defaults (None) "
        "or the orders probed by a _KakSynthesizer. Returns (py_mul, np_scalar_mul, scalar_arr_mul, "
        "py_div_fma, np_div_fma).");

    m.def(
        "_gauge_minimize",
        [](const py::sequence& instrs, const transpiler::LengthModel& model, const py::object& synth,
           const py::object& one_qubit, const py::object& matrix_of, const py::object& axes_of,
           const py::object& synthesize, const py::object& instr_cls, const py::object& type_gate,
           const py::object& type_barrier, const py::object& gate_cls, const py::object& h, const py::sequence& to_z,
           const py::sequence& to_z_conj, int beam, int gate_weight, int run_weight, int sweeps,
           double atol) -> py::object {
            const synthesis::NumpyLinalg* la = KakLinalg(synth);
            if (la == nullptr) {
                return py::make_tuple(false, py::none());
            }
            Options opt;
            opt.instr_cls = instr_cls;
            opt.type_gate = type_gate;
            opt.type_barrier = type_barrier;
            opt.gate_cls = gate_cls;
            opt.one_qubit = one_qubit;
            opt.matrix_of = matrix_of;
            opt.axes_of = axes_of;
            opt.synthesize = synthesize;
            opt.beam = beam;
            opt.gate_weight = gate_weight;
            opt.run_weight = run_weight;
            opt.sweeps = sweeps;
            opt.atol = atol;
            try {
                opt.h = ArgMatrix(h);
                if (py::len(to_z) != 3 || py::len(to_z_conj) != 3) {
                    throw std::invalid_argument("three frame matrices required");
                }
                for (std::size_t a = 0; a < 3; ++a) {
                    opt.to_z[a] = ArgMatrix(to_z[a]);
                    opt.to_z_conj[a] = ArgMatrix(to_z_conj[a]);
                }
                Minimizer minimizer(*la, model, opt);
                return py::make_tuple(true, minimizer.Run(instrs));
            } catch (const Decline&) {
                return py::make_tuple(false, py::none());
            } catch (const std::domain_error&) {
                return py::make_tuple(false, py::none());
            }
        },
        py::arg("instrs"), py::arg("model"), py::arg("synth"), py::arg("one_qubit"), py::arg("matrix_of"),
        py::arg("axes_of"), py::arg("synthesize"), py::arg("instr_cls"), py::arg("type_gate"),
        py::arg("type_barrier"), py::arg("gate_cls"), py::arg("h"), py::arg("to_z"), py::arg("to_z_conj"),
        py::arg("beam"), py::arg("gate_weight"), py::arg("run_weight"), py::arg("sweeps"), py::arg("atol"),
        "Native body of _gauge._minimize: (True, instructions | None) or (False, None) to use Python.");
}

}  // namespace quspect::native::pybind
