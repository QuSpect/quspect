// 相位折叠（quspect.transpiler._phase_fold.PhaseFoldingPass.run）的原生实现。
//
// 与 Python 实现逐步相同：同样的线性扫描、同样的路径变量编号顺序、同样的角度累加顺序与
// math.remainder 归约（IEEE 余数是精确运算，std::remainder 与 CPython 的实现结果相同），
// 遇到非门指令时同样清空全部状态。Python 端用整数位集表示奇偶函数（长线路上位数随新变量数
// 增长，每次异或与哈希都是整段运算）；这里每个奇偶函数是排好序的变量编号表（通常只有几个
// 变量），驻留为编号后比较与查表都是 O(1)。合并后的 rz 由 Python 端的 _rz 构造。
// 不能逐项复刻的输入（指令子类、非 int 比特、门名不是 str、参数不是 tuple、Real 子类的参数、
// 无穷大角度等）返回 None，由 Python 实现处理。
#include "phase_fold_bindings.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "py_interned.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

struct FoldNames {
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str gate = InternedStr("gate");
    py::str name = InternedStr("name");
    py::str params = InternedStr("params");
};

FoldNames& Names() {
    static FoldNames* names = new FoldNames();
    return *names;
}

py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

// 不能复刻时抛出，整个调用返回 None。
struct Decline {};

// 奇偶函数（路径变量的集合）的驻留表：编号 -> 排好序的变量表。
class ParityPool {
public:
    int Single(std::uint32_t var) {
        buf_.assign(1, var);
        return Intern();
    }
    // 对称差（位集的异或）。
    int Xor(int a, int b) {
        const auto& x = sets_[static_cast<std::size_t>(a)];
        const auto& y = sets_[static_cast<std::size_t>(b)];
        buf_.clear();
        std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(buf_));
        return Intern();
    }

private:
    struct Hash {
        std::size_t operator()(const std::vector<std::uint32_t>& v) const noexcept {
            std::uint64_t h = 1469598103934665603ULL ^ v.size();
            for (std::uint32_t x : v) {
                h = (h ^ x) * 1099511628211ULL;
            }
            return static_cast<std::size_t>(h);
        }
    };
    int Intern() {
        auto it = ids_.find(buf_);
        if (it != ids_.end()) {
            return it->second;
        }
        const int id = static_cast<int>(sets_.size());
        sets_.push_back(buf_);
        ids_.emplace(buf_, id);
        return id;
    }
    std::vector<std::vector<std::uint32_t>> sets_;
    std::unordered_map<std::vector<std::uint32_t>, int, Hash> ids_;
    std::vector<std::uint32_t> buf_;
};

struct Term {
    std::size_t slot;
    long long qubit;
    int sign;
    double angle;
    long long count;
};

class Folder {
public:
    Folder(const py::object& instr_cls, const py::object& type_gate, const py::dict& fixed, const py::object& rotation,
           const py::object& diagonal, const py::object& real)
        : instr_type_(reinterpret_cast<PyTypeObject*>(instr_cls.ptr())),
          type_gate_(type_gate.ptr()),
          fixed_(fixed),
          rotation_(rotation),
          diagonal_(diagonal),
          real_(real) {}

    // 返回 (输出指令, rotations_in, rotations_out)；合并后的 rz 用 make_rz(angle, qubit) 构造。
    py::tuple Run(const py::list& instrs, const py::object& make_rz) {
        FoldNames& n = Names();
        std::vector<PyObject*> out;  // 借用：输入指令（nullptr 为删除的位置）
        long long rotations_in = 0;
        make_rz_ = &make_rz;
        merged_.assign(PyList_GET_SIZE(instrs.ptr()), py::object());
        dropped_.assign(PyList_GET_SIZE(instrs.ptr()), 0);
        rotations_out_ = 0;
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(instrs.ptr()); ++i) {
            PyObject* ins = PyList_GET_ITEM(instrs.ptr(), i);
            if (Py_TYPE(ins) != instr_type_) {
                throw Decline();
            }
            py::object gate = Attr(ins, n.gate);
            if (Attr(ins, n.type).ptr() != type_gate_ || gate.is_none()) {
                Flush();
                out.push_back(ins);
                continue;
            }
            ReadQubits(Attr(ins, n.qubits).ptr());
            py::object name = Attr(gate.ptr(), n.name);
            if (!PyUnicode_CheckExact(name.ptr())) {
                throw Decline();
            }
            auto is = [&name](const char* text) { return PyUnicode_CompareWithASCIIString(name.ptr(), text) == 0; };
            if (qs_.size() == 1) {
                const long long q = qs_[0];
                double phase = 0.0;
                if (PhaseOf(name.ptr(), gate.ptr(), phase)) {
                    ++rotations_in;
                    const int key = Wire(q);
                    const int sign = const_[q] != 0 ? -1 : 1;
                    auto it = term_index_.find(key);
                    if (it == term_index_.end()) {
                        term_index_.emplace(key, terms_.size());
                        terms_.push_back(Term{out.size(), q, sign, sign == -1 ? -phase : phase, 1});
                        out.push_back(ins);
                    } else {
                        Term& t = terms_[it->second];
                        t.angle += sign == -1 ? -phase : phase;
                        t.count += 1;
                    }
                    continue;
                }
                if (is("x") || is("y")) {
                    Wire(q);
                    const_[q] ^= 1;
                } else if (!Contains(rotation_, name.ptr()) && !is("id")) {
                    Refresh(q);
                }
                out.push_back(ins);
                continue;
            }
            if (is("cx") && qs_.size() == 2) {
                const long long c = qs_[0];
                const long long t = qs_[1];
                const int pt = Wire(t);
                const int pc = Wire(c);
                parity_[t] = pool_.Xor(pt, pc);
                const_[t] ^= const_[c];
            } else if (is("swap") && qs_.size() == 2) {
                const long long a = qs_[0];
                const long long b = qs_[1];
                Wire(a);
                Wire(b);
                std::swap(parity_[a], parity_[b]);
                std::swap(const_[a], const_[b]);
            } else if (Contains(diagonal_, name.ptr()) && Numeric(gate.ptr())) {
                // 对角门不改变任何比特的奇偶函数
            } else if (is("ccx") && qs_.size() == 3) {
                Refresh(qs_[2]);
            } else {
                for (long long q : qs_) {
                    Refresh(q);
                }
            }
            out.push_back(ins);
        }

        // 收尾：剩余的项按插入顺序（与 Python dict 相同）写回。
        EmitTerms();
        py::list result;
        for (std::size_t k = 0; k < out.size(); ++k) {
            if (dropped_[k] != 0) {
                continue;
            }
            PyObject* item = merged_[k] ? merged_[k].ptr() : out[k];
            if (PyList_Append(result.ptr(), item) != 0) {
                throw py::error_already_set();
            }
        }
        return py::make_tuple(result, rotations_in, rotations_out_);
    }

private:
    // 把各项合并后的角度写回其第一次出现的位置（后续出现的旋转已从输出中删去）。
    void EmitTerms() {
        constexpr double kTwoPi = 2.0 * 3.141592653589793;
        constexpr double kZeroAtol = 1e-12;
        for (const Term& t : terms_) {
            const double x = t.sign == -1 ? -t.angle : t.angle;
            if (std::isinf(x)) {
                throw Decline();  // math.remainder(inf, ...) 抛 ValueError
            }
            const double angle = std::remainder(x, kTwoPi);
            if (std::fabs(angle) <= kZeroAtol) {
                dropped_[t.slot] = 1;
                continue;
            }
            ++rotations_out_;
            if (t.count > 1) {
                merged_[t.slot] = (*make_rz_)(angle, t.qubit);
            }
        }
    }

    void Flush() {
        EmitTerms();
        terms_.clear();
        term_index_.clear();
        parity_.clear();
        const_.clear();
    }

    int Wire(long long q) {
        auto it = parity_.find(q);
        if (it != parity_.end()) {
            return it->second;
        }
        const int p = pool_.Single(next_var_++);
        parity_[q] = p;
        const_[q] = 0;
        return p;
    }

    void Refresh(long long q) {
        parity_[q] = pool_.Single(next_var_++);
        const_[q] = 0;
    }

    void ReadQubits(PyObject* tuple) {
        qs_.clear();
        if (!PyTuple_CheckExact(tuple)) {
            throw Decline();
        }
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
            PyObject* item = PyTuple_GET_ITEM(tuple, i);
            if (!PyLong_CheckExact(item)) {
                throw Decline();
            }
            int overflow = 0;
            const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
            if (overflow != 0) {
                throw Decline();
            }
            qs_.push_back(v);
        }
    }

    static bool Contains(const py::object& set, PyObject* key) {
        const int r = PySequence_Contains(set.ptr(), key);
        if (r < 0) {
            throw py::error_already_set();
        }
        return r != 0;
    }

    bool IsReal(PyObject* value) const {
        const int r = PyObject_IsInstance(value, real_.ptr());
        if (r < 0) {
            throw py::error_already_set();
        }
        return r != 0;
    }

    // _phase_of(gate)：固定相位门查表；rz / p / u1 的唯一参数为实数时取 float(param)。
    bool PhaseOf(PyObject* name, PyObject* gate, double& phase) const {
        PyObject* fixed = PyDict_GetItemWithError(fixed_.ptr(), name);
        if (fixed == nullptr && PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (fixed != nullptr && fixed != Py_None) {
            if (!PyFloat_CheckExact(fixed)) {
                throw Decline();
            }
            phase = PyFloat_AS_DOUBLE(fixed);
            return true;
        }
        if (!Contains(rotation_, name)) {
            return false;
        }
        py::object params = Attr(gate, Names().params);
        if (!PyTuple_CheckExact(params.ptr())) {
            throw Decline();
        }
        if (PyTuple_GET_SIZE(params.ptr()) != 1) {
            return false;
        }
        PyObject* p = PyTuple_GET_ITEM(params.ptr(), 0);
        if (!IsReal(p)) {
            return false;
        }
        if (PyFloat_CheckExact(p)) {
            phase = PyFloat_AS_DOUBLE(p);
            return true;
        }
        if (PyLong_CheckExact(p) || PyBool_Check(p)) {
            phase = PyLong_AsDouble(p);
            if (phase == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                throw Decline();  // float(int) 溢出
            }
            return true;
        }
        throw Decline();  // 其他 Real：float() 可能被重载
    }

    // _is_numeric(gate)：全部参数都是实数。
    bool Numeric(PyObject* gate) const {
        py::object params = Attr(gate, Names().params);
        py::object it = py::reinterpret_steal<py::object>(PyObject_GetIter(params.ptr()));
        if (!it) {
            throw py::error_already_set();
        }
        while (PyObject* item = PyIter_Next(it.ptr())) {
            py::object owned = py::reinterpret_steal<py::object>(item);
            if (!IsReal(item)) {
                return false;
            }
        }
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        return true;
    }

    PyTypeObject* instr_type_;
    PyObject* type_gate_;
    py::dict fixed_;
    py::object rotation_;
    py::object diagonal_;
    py::object real_;

    ParityPool pool_;
    std::uint32_t next_var_ = 0;
    std::unordered_map<long long, int> parity_;
    std::unordered_map<long long, int> const_;
    std::vector<Term> terms_;
    std::unordered_map<int, std::size_t> term_index_;
    std::vector<long long> qs_;
    const py::object* make_rz_ = nullptr;
    std::vector<py::object> merged_;
    std::vector<std::uint8_t> dropped_;
    long long rotations_out_ = 0;
};

}  // namespace

void BindPhaseFold(py::module_& m) {
    m.def(
        "_phase_fold",
        [](const py::list& instrs, const py::object& instr_cls, const py::object& type_gate, const py::dict& fixed,
           const py::object& rotation, const py::object& diagonal, const py::object& real,
           const py::object& make_rz) -> py::object {
            try {
                Folder folder(instr_cls, type_gate, fixed, rotation, diagonal, real);
                return folder.Run(instrs, make_rz);
            } catch (const Decline&) {
                return py::none();
            }
        },
        py::arg("instrs"), py::arg("instr_cls"), py::arg("type_gate"), py::arg("fixed"), py::arg("rotation"),
        py::arg("diagonal"), py::arg("real"), py::arg("make_rz"),
        "Native PhaseFoldingPass.run: (instructions, rotations_in, rotations_out), or None to use Python.");
}

}  // namespace quspect::native::pybind
