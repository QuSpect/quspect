// 目标门集翻译（quspect.transpiler._basis.translate_to_target）的原生实现。
//
// 与 Python 实现逐步相同：沿指令顺序收集每条线上的单比特门运行（乘积为确定性数值层的 2×2 乘法，
// 见 run_helpers.hpp，全平台逐位一致），每个 cz 换成目标纠缠门并把修饰门并入两侧的运行（非对称纠缠门按同样的
// 代价比较选朝向，向后看的矩阵同样逐门相乘），需要重综合的运行回调 Python 的
// synthesize_one_qubit（同一矩阵只调用一次），输出指令按数据类 __init__ 的字段顺序直接构造。
// 门矩阵来自 Python 端同一个按门缓存的 matrix_of（原生模块的确定性门矩阵）。不支持的输入（以及
// Python 端会抛出异常的情形）返回 (False, None)，由 Python 实现处理。
#include "basis_bindings.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "run_helpers.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using runs::Attr;
using runs::Decline;
using runs::Matrix2;
using runs::ReadBits;
using runs::ReadMatrix2;
using runs::Truthy;

constexpr int kNoSynthesis = 1000000;  // synth_len 的 10**6

// 纠缠门的一个朝向：修饰门 before / after（按 (q0, q1)）与它们是否为单位阵（Python 端 _is_identity）。
struct Dressing {
    std::array<Matrix2, 2> before;
    std::array<Matrix2, 2> after;
    std::array<bool, 2> before_identity;
    std::array<bool, 2> after_identity;
};

struct PendingRun {
    std::vector<PyObject*> gates;  // 借用：输入指令
    Matrix2 matrix = runs::Identity2();
    bool dirty = false;
};

class Translator {
 public:
    Translator(py::object native, py::object matrix_of, py::object synthesize,
               runs::InstructionFactory factory, py::object type_gate, py::object type_barrier, bool has_ent,
               bool symmetric, py::object ent_gate, std::array<Dressing, 2> dress, std::vector<Matrix2> far_ends)
        : native_(std::move(native)),
          matrix_of_(std::move(matrix_of)),
          synth_(std::move(synthesize)),
          factory_(std::move(factory)),
          type_gate_(std::move(type_gate)),
          type_barrier_(std::move(type_barrier)),
          has_ent_(has_ent),
          symmetric_(symmetric),
          ent_gate_(std::move(ent_gate)),
          dress_(std::move(dress)),
          far_ends_(std::move(far_ends)) {}

    py::list Run(PyObject** items, Py_ssize_t count, PyTypeObject* instr_type) {
        runs::RunNames& n = Names();
        decoded_.resize(static_cast<std::size_t>(count));
        // 解码（并检查类型）；wire_pos：每条线上指令的位置
        for (Py_ssize_t i = 0; i < count; ++i) {
            PyObject* ins = items[i];
            if (Py_TYPE(ins) != instr_type) {
                throw Decline{};
            }
            Decoded& d = decoded_[static_cast<std::size_t>(i)];
            d.obj = ins;
            py::object type = Attr(ins, n.type);
            d.is_gate = type.is(type_gate_);
            d.is_barrier = type.is(type_barrier_);
            d.gate = Attr(ins, n.gate);
            ReadBits(Attr(ins, n.qubits).ptr(), &d.qubits);
            if (d.is_gate && !d.gate.is_none()) {
                py::object name = Attr(d.gate.ptr(), n.name);
                d.name = name;
                d.cz = PyUnicode_CheckExact(name.ptr()) && PyUnicode_CompareWithASCIIString(name.ptr(), "cz") == 0
                       && !Truthy(Attr(d.gate.ptr(), n.params).ptr());
            }
            for (long long q : d.qubits) {
                wire_pos_[q].push_back(static_cast<int>(i));
            }
        }
        for (auto& kv : wire_pos_) {
            cursor_[kv.first] = 0;
        }
        for (Py_ssize_t i = 0; i < count; ++i) {
            Step(decoded_[static_cast<std::size_t>(i)]);
        }
        std::vector<long long> remaining(order_.begin(), order_.end());
        std::sort(remaining.begin(), remaining.end());
        for (long long q : remaining) {
            Flush(q);
        }
        return out_;
    }

 private:
    struct Decoded {
        PyObject* obj = nullptr;
        bool is_gate = false;
        bool is_barrier = false;
        bool cz = false;
        py::object gate;
        py::object name;
        std::vector<long long> qubits;
    };

    static runs::RunNames& Names() {
        return runs::Names();
    }

    bool NameIn(const py::object& name, const py::object& set) const {
        const int r = PySequence_Contains(set.ptr(), name.ptr());
        if (r < 0) {
            throw py::error_already_set();
        }
        return r != 0;
    }

    // matrix_of(gate)：每个门对象只调用一次；None → nullptr。
    const Matrix2* MatrixOf(PyObject* gate) {
        const auto it = matrix_index_.find(gate);
        int index;
        if (it != matrix_index_.end()) {
            index = it->second;
        } else {
            py::object m = matrix_of_(py::handle(gate));
            index = -1;
            if (!m.is_none()) {
                index = static_cast<int>(gate_matrices_.size());
                gate_matrices_.push_back(ReadMatrix2(m.ptr()));
            }
            keep_.push_back(py::reinterpret_borrow<py::object>(gate));
            matrix_index_.emplace(gate, index);
        }
        return index < 0 ? nullptr : &gate_matrices_[static_cast<std::size_t>(index)];
    }

    bool IsTranslated(const Decoded& d) const {
        return has_ent_ && d.is_gate && !d.gate.is_none() && d.cz && d.qubits.size() == 2;
    }

    PendingRun& RunOf(long long q) {
        auto it = runs_.find(q);
        if (it == runs_.end()) {
            it = runs_.emplace(q, PendingRun{}).first;
            order_.push_back(q);
        }
        return it->second;
    }

    int SynthLen(const Matrix2& m) {
        py::object seq = synth_.Get(m);
        return seq.is_none() ? kNoSynthesis : static_cast<int>(PyList_GET_SIZE(seq.ptr()));
    }

    void Flush(long long q) {
        const auto it = runs_.find(q);
        if (it == runs_.end()) {
            return;
        }
        PendingRun run = std::move(it->second);
        runs_.erase(it);
        order_.erase(std::find(order_.begin(), order_.end(), q));
        if (!run.dirty) {
            for (PyObject* ins : run.gates) {
                out_.append(py::handle(ins));
            }
            return;
        }
        py::object seq = synth_.Get(run.matrix);
        if (seq.is_none()) {
            throw Decline{};  // Python 端抛出 ValueError
        }
        for (Py_ssize_t k = 0; k < PyList_GET_SIZE(seq.ptr()); ++k) {
            out_.append(factory_.OneQubit(PyList_GET_ITEM(seq.ptr(), k), q));
        }
    }

    // lookahead(q)：当前指令之后 q 上的单比特门乘积，以及这段是否止于另一个要翻译的 cz。
    std::pair<Matrix2, bool> Lookahead(long long q) {
        Matrix2 m = runs::Identity2();
        const std::vector<int>& positions = wire_pos_[q];
        for (std::size_t pos = static_cast<std::size_t>(cursor_[q]); pos < positions.size(); ++pos) {
            const Decoded& d = decoded_[static_cast<std::size_t>(positions[pos])];
            if (!(d.is_gate && d.qubits.size() == 1 && !d.gate.is_none())) {
                return {m, IsTranslated(d)};
            }
            const Matrix2* g = MatrixOf(d.gate.ptr());
            if (g == nullptr) {
                return {m, false};
            }
            m = mat_.Mul(*g, m);
        }
        return {m, false};
    }

    void Step(const Decoded& d) {
        for (long long q : d.qubits) {
            ++cursor_[q];
        }
        if (d.is_gate && !d.gate.is_none() && d.qubits.size() == 1) {
            const long long q = d.qubits[0];
            const Matrix2* m = MatrixOf(d.gate.ptr());
            if (m == nullptr) {
                if (!NameIn(d.name, native_)) {
                    throw Decline{};  // Python 端抛出 ValueError
                }
                Flush(q);
                out_.append(py::handle(d.obj));
                return;
            }
            PendingRun& run = RunOf(q);
            run.gates.push_back(d.obj);
            run.matrix = mat_.Mul(*m, run.matrix);
            if (!NameIn(d.name, native_)) {
                run.dirty = true;
            }
            return;
        }
        if (IsTranslated(d)) {
            const long long a = d.qubits[0];
            const long long b = d.qubits[1];
            if (a == b) {
                throw Decline{};
            }
            bool rev = false;
            if (!symmetric_) {
                // pre：已有运行的矩阵或 float64 单位阵；post：向后看的乘积
                std::array<bool, 2> pre_eye{};
                std::array<Matrix2, 2> pre{};
                std::array<std::pair<Matrix2, bool>, 2> post{};
                const std::array<long long, 2> ab{a, b};
                for (std::size_t s = 0; s < 2; ++s) {
                    const auto it = runs_.find(ab[s]);
                    pre_eye[s] = it == runs_.end();
                    if (!pre_eye[s]) {
                        pre[s] = it->second.matrix;
                    }
                }
                for (std::size_t s = 0; s < 2; ++s) {
                    post[s] = Lookahead(ab[s]);
                }
                long long best_cost = 0;
                bool have_best = false;
                for (int r = 0; r < 2; ++r) {
                    const Dressing& dr = dress_[static_cast<std::size_t>(r)];
                    long long cost = 0;
                    for (std::size_t s = 0; s < 2; ++s) {
                        cost += SynthLen(pre_eye[s] ? mat_.MulRealIdentity(dr.before[s])
                                                    : mat_.Mul(dr.before[s], pre[s]));
                        const Matrix2 m_post = mat_.Mul(post[s].first, dr.after[s]);
                        int best_end = SynthLen(m_post);
                        if (post[s].second) {
                            for (const Matrix2& f : far_ends_) {
                                best_end = std::min(best_end, SynthLen(mat_.Mul(f, m_post)));
                            }
                        }
                        cost += best_end;
                    }
                    if (!have_best || cost < best_cost) {
                        best_cost = cost;
                        rev = r == 1;
                        have_best = true;
                    }
                }
            }
            const Dressing& dr = dress_[rev ? 1 : 0];
            const std::array<long long, 2> ab{a, b};
            for (std::size_t s = 0; s < 2; ++s) {
                if (!dr.before_identity[s]) {
                    PendingRun& run = RunOf(ab[s]);
                    run.matrix = mat_.Mul(dr.before[s], run.matrix);
                    run.dirty = true;
                }
                Flush(ab[s]);
            }
            out_.append(rev ? factory_.TwoQubit(ent_gate_.ptr(), b, a) : factory_.TwoQubit(ent_gate_.ptr(), a, b));
            for (std::size_t s = 0; s < 2; ++s) {
                if (!dr.after_identity[s]) {
                    PendingRun& run = RunOf(ab[s]);
                    run.matrix = dr.after[s];
                    run.dirty = true;
                }
            }
            return;
        }
        if (d.is_gate && !d.gate.is_none() && !NameIn(d.name, native_)) {
            throw Decline{};  // Python 端抛出 ValueError
        }
        if (d.qubits.empty() && d.is_barrier) {
            const std::vector<long long> all(order_.begin(), order_.end());
            for (long long q : all) {
                Flush(q);
            }
        }
        for (long long q : d.qubits) {
            Flush(q);
        }
        out_.append(py::handle(d.obj));
    }

    runs::Mat2 mat_;
    py::object native_;
    py::object matrix_of_;
    runs::SynthCache synth_;
    runs::InstructionFactory factory_;
    py::object type_gate_;
    py::object type_barrier_;
    bool has_ent_;
    bool symmetric_;
    py::object ent_gate_;
    std::array<Dressing, 2> dress_;
    std::vector<Matrix2> far_ends_;

    std::vector<Decoded> decoded_;
    std::unordered_map<long long, std::vector<int>> wire_pos_;
    std::unordered_map<long long, int> cursor_;
    std::unordered_map<long long, PendingRun> runs_;
    std::vector<long long> order_;  // runs_ 的插入顺序（与 Python dict 相同）
    std::unordered_map<PyObject*, int> matrix_index_;
    std::vector<Matrix2> gate_matrices_;
    std::vector<py::object> keep_;
    py::list out_;
};

}  // namespace

void BindBasis(py::module_& m) {
    m.def(
        "_translate_to_target",
        [](const py::sequence& instrs, const py::object& native,
           const py::object& matrix_of, const py::object& synthesize, const py::object& instr_cls,
           const py::object& type_gate, const py::object& type_barrier, const py::object& gate_cls,
           const py::object& ent) -> py::object {
            try {
                // ent：None，或 (symmetric, gate, [(before0, before1, after0, after1, flags) × 2], far_ends)
                bool has_ent = !ent.is_none();
                bool symmetric = true;
                py::object ent_gate = py::none();
                std::array<Dressing, 2> dress{};
                std::vector<Matrix2> far_ends;
                if (has_ent) {
                    auto t = ent.cast<py::tuple>();
                    symmetric = t[0].cast<bool>();
                    ent_gate = t[1];
                    auto ds = t[2].cast<py::sequence>();
                    for (std::size_t r = 0; r < 2; ++r) {
                        auto one = ds[r].cast<py::tuple>();
                        Dressing& dr = dress[r];
                        dr.before = {ReadMatrix2(one[0].ptr()), ReadMatrix2(one[1].ptr())};
                        dr.after = {ReadMatrix2(one[2].ptr()), ReadMatrix2(one[3].ptr())};
                        auto flags = one[4].cast<py::tuple>();
                        dr.before_identity = {flags[0].cast<bool>(), flags[1].cast<bool>()};
                        dr.after_identity = {flags[2].cast<bool>(), flags[3].cast<bool>()};
                    }
                    for (const auto item : t[3].cast<py::sequence>()) {
                        far_ends.push_back(ReadMatrix2(item.ptr()));
                    }
                }
                py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(instrs.ptr(), "instructions"));
                if (!fast) {
                    throw py::error_already_set();
                }
                Translator tr(native, matrix_of, synthesize,
                              runs::InstructionFactory(instr_cls, type_gate, gate_cls), type_gate, type_barrier,
                              has_ent, symmetric, ent_gate, dress, std::move(far_ends));
                py::list out = tr.Run(PySequence_Fast_ITEMS(fast.ptr()), PySequence_Fast_GET_SIZE(fast.ptr()),
                                      reinterpret_cast<PyTypeObject*>(instr_cls.ptr()));
                return py::make_tuple(true, out);
            } catch (const Decline&) {
                return py::make_tuple(false, py::none());
            }
        },
        py::arg("instrs"), py::arg("native"), py::arg("matrix_of"), py::arg("synthesize"),
        py::arg("instr_cls"), py::arg("type_gate"), py::arg("type_barrier"), py::arg("gate_cls"), py::arg("ent"),
        "Native body of _basis.translate_to_target: (True, instructions) or (False, None) to use Python.");

    m.def(
        "_gate_instructions",
        [](const py::object& instr_cls, const py::object& type_gate, const py::object& gate_cls,
           const py::sequence& items, const py::object& label) {
            const runs::InstructionFactory factory(instr_cls, type_gate, gate_cls);
            py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(items.ptr(), "items"));
            if (!fast) {
                throw py::error_already_set();
            }
            const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast.ptr());
            PyObject** data = PySequence_Fast_ITEMS(fast.ptr());
            py::list out(count);
            for (Py_ssize_t i = 0; i < count; ++i) {
                py::object pair = py::reinterpret_borrow<py::object>(data[i]);
                if (!PyTuple_CheckExact(pair.ptr())) {
                    pair = py::reinterpret_steal<py::object>(PySequence_Tuple(pair.ptr()));
                    if (!pair) {
                        throw py::error_already_set();
                    }
                }
                if (PyTuple_GET_SIZE(pair.ptr()) != 2) {
                    throw py::value_error("expected (gate, qubits) pairs");
                }
                py::object ins =
                    factory.Gate(PyTuple_GET_ITEM(pair.ptr(), 0), PyTuple_GET_ITEM(pair.ptr(), 1), label.ptr());
                PyList_SET_ITEM(out.ptr(), i, ins.release().ptr());
            }
            return out;
        },
        py::arg("instr_cls"), py::arg("type_gate"), py::arg("gate_cls"), py::arg("items"), py::arg("label"),
        "[CircuitInstruction(type=GATE, gate=g, qubits=qs, clbits=(), label=label) for g, qs in items], "
        "plain cases built field by field.");
}

}  // namespace quspect::native::pybind
