// 块合并（quspect.transpiler._consolidate.ConsolidateBlocksPass）的两次扫描：收集两比特块、收集单比特串。
//
// 扫描逻辑与 Python 实现逐步相同（open_block 按 dict 的插入顺序维护，块按关闭的先后输出）；门矩阵
// 先查 pass 的矩阵缓存 dict（与 _matrix 相同的键），未命中时调用 pass 的 _matrix；两比特门的代价先查
// _KNOWN_2Q_COST，否则调用 _cost。块的重综合（_finalize）仍由 Python 按输出顺序完成：它只写各自的
// 槽位与统计量，与扫描状态无关，所以推迟到扫描之后执行结果相同。cost < 2 的块 _finalize 不做任何事，
// 不输出。指令不是普通 CircuitInstruction 时返回 None，由 Python 端按原实现处理。
#include "consolidate_bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include "py_interned.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

struct ConsolidateNames {
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str clbits = InternedStr("clbits");
    py::str gate = InternedStr("gate");
    py::str name = InternedStr("name");
    py::str params = InternedStr("params");
};

ConsolidateNames& Names() {
    static ConsolidateNames* names = new ConsolidateNames();
    return *names;
}

py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

struct Instr {
    bool is_gate = false;
    bool is_barrier = false;
    py::object gate;
    std::vector<long long> qubits;
};

bool ReadQubits(PyObject* t, std::vector<long long>& out) {
    out.clear();
    if (!PyTuple_CheckExact(t)) {
        return false;
    }
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(t); ++i) {
        PyObject* item = PyTuple_GET_ITEM(t, i);
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            return false;
        }
        out.push_back(v);
    }
    return true;
}

bool Decode(const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
            const py::object& type_barrier, std::vector<Instr>& out) {
    ConsolidateNames& n = Names();
    py::object fast = py::reinterpret_steal<py::object>(PySequence_Fast(instrs.ptr(), "instructions"));
    if (!fast) {
        throw py::error_already_set();
    }
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(fast.ptr());
    PyObject** items = PySequence_Fast_ITEMS(fast.ptr());
    out.resize(static_cast<std::size_t>(count));
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* ins = items[i];
        if (Py_TYPE(ins) != reinterpret_cast<PyTypeObject*>(instr_cls.ptr())) {
            return false;
        }
        Instr& r = out[static_cast<std::size_t>(i)];
        py::object type = Attr(ins, n.type);
        r.is_gate = type.is(type_gate);
        r.is_barrier = type.is(type_barrier);
        r.gate = Attr(ins, n.gate);
        if (!ReadQubits(Attr(ins, n.qubits).ptr(), r.qubits)) {
            return false;
        }
    }
    return true;
}

// pass._matrix(gate)：先按门对象身份复用本次扫描中的结果，再查 pass 的矩阵缓存 dict，最后调用 _matrix。
class MatrixLookup {
public:
    MatrixLookup(py::dict cache, py::object matrix_fn) : cache_(std::move(cache)), fn_(std::move(matrix_fn)) {
    }
    py::object operator()(const py::object& gate) {
        auto it = by_id_.find(gate.ptr());
        if (it != by_id_.end()) {
            return it->second.second;
        }
        py::object mat;
        PyObject* hit = PyDict_GetItemWithError(cache_.ptr(), gate.ptr());
        if (hit != nullptr) {
            mat = py::reinterpret_borrow<py::object>(hit);
        } else {
            if (PyErr_Occurred()) {
                if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
                    throw py::error_already_set();
                }
                PyErr_Clear();  // 不可哈希的门：_matrix 自己处理
            }
            mat = fn_(gate);
        }
        by_id_.emplace(gate.ptr(), std::make_pair(gate, mat));
        return mat;
    }

private:
    py::dict cache_;
    py::object fn_;
    std::unordered_map<PyObject*, std::pair<py::object, py::object>> by_id_;  // 持有门对象，身份有效
};

struct Block {
    long long qa;
    long long qb;
    std::vector<long> members;
    long first_2q;
    long long cost;
    long long num_2q;
};

// 以 dict 的插入顺序维护 qubit -> 块（删除后重新插入排到末尾）。
class OpenBlocks {
public:
    long get(long long q) const {
        auto it = pos_.find(q);
        return it == pos_.end() ? -1 : entries_[it->second].second;
    }
    void set(long long q, long block) {
        auto it = pos_.find(q);
        if (it != pos_.end()) {
            entries_[it->second].second = block;
            return;
        }
        pos_.emplace(q, entries_.size());
        entries_.emplace_back(q, block);
        alive_.push_back(1);
    }
    void erase(long long q) {
        auto it = pos_.find(q);
        if (it == pos_.end()) {
            return;
        }
        alive_[it->second] = 0;
        pos_.erase(it);
    }
    // list({id(b): b for b in open_block.values()}.values())
    std::vector<long> distinct_values() const {
        std::vector<long> out;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            if (alive_[i] != 0 && std::find(out.begin(), out.end(), entries_[i].second) == out.end()) {
                out.push_back(entries_[i].second);
            }
        }
        return out;
    }

private:
    std::unordered_map<long long, std::size_t> pos_;
    std::vector<std::pair<long long, long>> entries_;
    std::vector<std::uint8_t> alive_;
};

py::object LongList(const std::vector<long>& v) {
    PyObject* out = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (out == nullptr) {
        throw py::error_already_set();
    }
    for (std::size_t i = 0; i < v.size(); ++i) {
        PyList_SET_ITEM(out, static_cast<Py_ssize_t>(i), PyLong_FromLong(v[i]));
    }
    return py::reinterpret_steal<py::object>(out);
}

}  // namespace

void BindConsolidate(py::module_& m) {
    m.def(
        "_consolidate_blocks",
        [](const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
           const py::object& type_barrier, const py::dict& matrix_cache, const py::object& matrix_fn,
           const py::dict& known_cost, const py::object& cost_fn) -> py::object {
            std::vector<Instr> decoded;
            if (!Decode(instrs, instr_cls, type_gate, type_barrier, decoded)) {
                return py::none();
            }
            ConsolidateNames& n = Names();
            MatrixLookup matrix(matrix_cache, matrix_fn);
            const std::size_t count = decoded.size();
            py::list mats(count);
            for (std::size_t i = 0; i < count; ++i) {
                mats[i] = py::none();
            }
            std::vector<Block> blocks;
            std::vector<long> closed;  // 关闭顺序（只记 cost >= 2 的块）
            OpenBlocks open_block;
            std::unordered_map<long long, std::vector<long>> pending;

            auto close = [&](long b) {
                const Block& blk = blocks[static_cast<std::size_t>(b)];
                for (long long q : {blk.qa, blk.qb}) {
                    if (open_block.get(q) == b) {
                        open_block.erase(q);
                    }
                }
                if (blk.cost >= 2) {
                    closed.push_back(b);
                }
            };
            auto take_pending = [&](long long q, std::vector<long>& out) {
                auto it = pending.find(q);
                if (it != pending.end()) {
                    out.insert(out.end(), it->second.begin(), it->second.end());
                    pending.erase(it);
                }
            };

            for (std::size_t idx = 0; idx < count; ++idx) {
                const Instr& ins = decoded[idx];
                py::object mat = py::none();
                if (ins.is_gate && !ins.gate.is_none() && ins.qubits.size() <= 2) {
                    mat = matrix(ins.gate);
                }
                if (mat.is_none()) {
                    // 不透明的操作：关闭它所在比特上的块
                    if (ins.qubits.empty() && ins.is_barrier) {
                        for (long b : open_block.distinct_values()) {
                            close(b);
                        }
                        pending.clear();
                        continue;
                    }
                    for (long long q : ins.qubits) {
                        const long hit = open_block.get(q);
                        if (hit >= 0) {
                            close(hit);
                        }
                        pending.erase(q);
                    }
                    continue;
                }
                mats[idx] = mat;
                if (ins.qubits.size() == 1) {
                    const long long q = ins.qubits[0];
                    const long owner = open_block.get(q);
                    if (owner >= 0) {
                        blocks[static_cast<std::size_t>(owner)].members.push_back(static_cast<long>(idx));
                    } else {
                        pending[q].push_back(static_cast<long>(idx));
                    }
                    continue;
                }
                const long long a = ins.qubits[0];
                const long long b = ins.qubits[1];
                // self._cost(gate, mat)
                long long cost = 0;
                py::object name = Attr(ins.gate.ptr(), n.name);
                PyObject* known = PyDict_GetItemWithError(known_cost.ptr(), name.ptr());
                if (known == nullptr && PyErr_Occurred()) {
                    throw py::error_already_set();
                }
                bool have = false;
                if (known != nullptr && known != Py_None) {
                    py::object params = Attr(ins.gate.ptr(), n.params);
                    const int truth = PyObject_IsTrue(params.ptr());
                    if (truth < 0) {
                        throw py::error_already_set();
                    }
                    if (truth == 0 && PyLong_CheckExact(known)) {
                        cost = PyLong_AsLongLong(known);
                        have = true;
                    }
                }
                if (!have) {
                    cost = cost_fn(ins.gate, mat).cast<long long>();
                }
                const long cur = open_block.get(a);
                if (cur >= 0 && open_block.get(b) == cur) {
                    Block& blk = blocks[static_cast<std::size_t>(cur)];
                    blk.members.push_back(static_cast<long>(idx));
                    blk.cost += cost;
                    blk.num_2q += 1;
                    continue;
                }
                for (long long q : {a, b}) {
                    const long old = open_block.get(q);
                    if (old >= 0) {
                        close(old);
                    }
                }
                std::vector<long> members;
                take_pending(a, members);
                take_pending(b, members);
                std::sort(members.begin(), members.end());
                members.push_back(static_cast<long>(idx));
                const long id = static_cast<long>(blocks.size());
                blocks.push_back(Block{a, b, std::move(members), static_cast<long>(idx), cost, 1});
                open_block.set(a, id);
                open_block.set(b, id);
            }
            for (long b : open_block.distinct_values()) {
                close(b);
            }

            py::list out(closed.size());
            for (std::size_t k = 0; k < closed.size(); ++k) {
                const Block& blk = blocks[static_cast<std::size_t>(closed[k])];
                out[k] = py::make_tuple(py::make_tuple(blk.qa, blk.qb), LongList(blk.members), blk.first_2q, blk.cost,
                                        blk.num_2q);
            }
            return py::make_tuple(mats, out);
        },
        py::arg("instructions"), py::arg("instr_cls"), py::arg("type_gate"), py::arg("type_barrier"),
        py::arg("matrix_cache"), py::arg("matrix_fn"), py::arg("known_cost"), py::arg("cost_fn"),
        "Block sweep of ConsolidateBlocksPass.run: (matrices, closed blocks with cost >= 2), or None.");

    m.def(
        "_single_qubit_runs",
        [](const py::sequence& instrs, const py::object& instr_cls, const py::object& type_gate,
           const py::object& type_barrier, const py::dict& matrix_cache, const py::object& matrix_fn) -> py::object {
            std::vector<Instr> decoded;
            if (!Decode(instrs, instr_cls, type_gate, type_barrier, decoded)) {
                return py::none();
            }
            MatrixLookup matrix(matrix_cache, matrix_fn);
            std::unordered_map<long long, std::vector<long>> runs;
            std::vector<long long> order;  // runs 的插入顺序
            py::list out;
            auto flush = [&](long long q) {
                auto it = runs.find(q);
                if (it == runs.end()) {
                    return;
                }
                if (it->second.size() >= 2) {
                    out.append(py::make_tuple(q, LongList(it->second)));
                }
                runs.erase(it);
                order.erase(std::find(order.begin(), order.end(), q));
            };
            for (std::size_t idx = 0; idx < decoded.size(); ++idx) {
                const Instr& ins = decoded[idx];
                if (ins.is_gate && !ins.gate.is_none() && ins.qubits.size() == 1 && !matrix(ins.gate).is_none()) {
                    const long long q = ins.qubits[0];
                    auto [it, inserted] = runs.try_emplace(q);
                    if (inserted) {
                        order.push_back(q);
                    }
                    it->second.push_back(static_cast<long>(idx));
                    continue;
                }
                if (ins.qubits.empty() && ins.is_barrier) {
                    const std::vector<long long> all = order;
                    for (long long q : all) {
                        flush(q);
                    }
                    continue;
                }
                for (long long q : ins.qubits) {
                    flush(q);
                }
            }
            const std::vector<long long> all = order;
            for (long long q : all) {
                flush(q);
            }
            return std::move(out);
        },
        py::arg("instructions"), py::arg("instr_cls"), py::arg("type_gate"), py::arg("type_barrier"),
        py::arg("matrix_cache"), py::arg("matrix_fn"),
        "Single-qubit runs (qubit, indices) of ConsolidateBlocksPass._fuse_single_qubit_runs, or None.");
}

}  // namespace quspect::native::pybind
