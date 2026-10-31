// DAG 优化 pass（quspect.dag.passes）在 _DagCore 上的原生实现与 PassManager 的收敛签名。
//
// pass 本身在 ir/dag_passes.cpp 中，直接操作 DagStore；这里的宿主负责把 DAGNode / CircuitInstruction /
// Gate 解码成 ir::PassNode（按 instruction 对象身份缓存，跨 pass 复用），并在需要新门对象时调用
// Python（Gate.with_params、CircuitInstruction 构造、单比特重综合），所以新建的对象与 Python 实现
// 完全相同。解码时遇到参考实现语义难以逐项复刻的输入（非规范线标签、DAGNode / CircuitInstruction
// 的子类、非 tuple 的比特、int / float 的子类参数等）时不修改 DAG、返回 False，由 Python 端改走原实现。
#include "dag_pass_bindings.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "py_interned.hpp"

#include "quspect_native/ir/dag_passes.hpp"
#include "quspect_native/ir/dag_store.hpp"

#include "dag_core.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

using dag_core::PyDagCore;
using ir::DagNodeKind;
using ir::NodeId;
using ir::PassInstrType;
using ir::PassNode;
using ir::WireId;

// 由 quspect.dag.passes._native 登记的类型与门模板。
struct PassRegistry {
    py::object instr_cls;
    py::object gate_cls;
    py::object type_gate;
    py::object type_measure;
    py::object type_barrier;
    py::object type_channel;
    py::object rx;
    py::object ry;
    py::object rz;
    py::object p;
    bool ready = false;
};

PassRegistry& Reg() {
    static PassRegistry* reg = new PassRegistry();  // 进程结束时不析构 Python 对象
    return *reg;
}

struct PassNames {
    py::str instruction = InternedStr("instruction");
    py::str type = InternedStr("type");
    py::str qubits = InternedStr("qubits");
    py::str clbits = InternedStr("clbits");
    py::str gate = InternedStr("gate");
    py::str name = InternedStr("name");
    py::str num_qubits = InternedStr("num_qubits");
    py::str params = InternedStr("params");
    py::str with_params = InternedStr("with_params");
    py::str channel = InternedStr("channel");
    py::str label = InternedStr("label");
};

PassNames& Names() {
    static PassNames* names = new PassNames();
    return *names;
}

PyObject* EmptyStr() {
    static PyObject* empty = PyUnicode_FromString("");  // 进程内常驻
    return empty;
}

PyObject* EmptyTuple() {
    static PyObject* empty = PyTuple_New(0);  // 进程内常驻
    return empty;
}

py::object Attr(PyObject* obj, const py::str& name) {
    PyObject* v = PyObject_GetAttr(obj, name.ptr());
    if (v == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(v);
}

// 比特元组：必须是由非负 int 组成的 tuple（与 Python 端的元组相等比较逐项一致）。
template <typename Vec>
bool DecodeBits(PyObject* tuple, Vec& out) {
    out.clear();
    if (!PyTuple_CheckExact(tuple)) {
        return false;
    }
    const Py_ssize_t n = PyTuple_GET_SIZE(tuple);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GET_ITEM(tuple, i);
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0 || v < 0 || v > (std::int64_t{1} << 40)) {
            return false;
        }
        out.push_back(static_cast<std::int64_t>(v));
    }
    return true;
}

// 解码一条指令（None 表示没有指令）。不能精确表示时返回 false（不抛异常）。
bool DecodeInstruction(PyObject* instr, DagNodeKind kind, PyDagCore::PassEntry& entry) {
    PassRegistry& reg = Reg();
    PassNames& n = Names();
    PassNode& info = entry.info;
    info = PassNode{};
    info.operation = kind == DagNodeKind::kOperation;
    entry.instr = py::reinterpret_borrow<py::object>(instr);
    entry.gate = py::none();
    if (instr == Py_None) {
        return true;
    }
    if (Py_TYPE(instr) != reinterpret_cast<PyTypeObject*>(reg.instr_cls.ptr())) {
        return false;
    }
    py::object type = Attr(instr, n.type);
    if (type.is(reg.type_gate)) {
        info.type = PassInstrType::kGate;
    } else if (type.is(reg.type_measure)) {
        info.type = PassInstrType::kMeasure;
    } else if (type.is(reg.type_barrier)) {
        info.type = PassInstrType::kBarrier;
    } else if (type.is(reg.type_channel)) {
        info.type = PassInstrType::kChannel;
    } else {
        return false;
    }
    if (!DecodeBits(Attr(instr, n.qubits).ptr(), info.qubits) || !DecodeBits(Attr(instr, n.clbits).ptr(), info.clbits)) {
        return false;
    }
    py::object gate = Attr(instr, n.gate);
    if (gate.is_none()) {
        return true;
    }
    info.has_gate = true;
    entry.gate = gate;
    py::object name = Attr(gate.ptr(), n.name);
    if (!PyUnicode_Check(name.ptr())) {
        return false;
    }
    Py_ssize_t len = 0;
    const char* s = PyUnicode_AsUTF8AndSize(name.ptr(), &len);
    if (s == nullptr) {
        PyErr_Clear();
        return false;
    }
    info.name = ir::InternGateName(std::string_view(s, static_cast<std::size_t>(len)));
    py::object nq = Attr(gate.ptr(), n.num_qubits);
    if (!PyLong_CheckExact(nq.ptr())) {
        return false;
    }
    int overflow = 0;
    info.num_qubits = PyLong_AsLongLongAndOverflow(nq.ptr(), &overflow);
    if (overflow != 0) {
        return false;
    }
    py::object params = Attr(gate.ptr(), n.params);
    if (!PyTuple_CheckExact(params.ptr())) {
        return false;
    }
    const Py_ssize_t np = PyTuple_GET_SIZE(params.ptr());
    for (Py_ssize_t i = 0; i < np; ++i) {
        PyObject* p = PyTuple_GET_ITEM(params.ptr(), i);
        double value = 0.0;
        std::uint8_t concrete = 1;
        if (PyFloat_CheckExact(p)) {
            value = PyFloat_AS_DOUBLE(p);
        } else if (PyBool_Check(p)) {
            value = p == Py_True ? 1.0 : 0.0;
        } else if (PyLong_CheckExact(p)) {
            value = PyLong_AsDouble(p);
            if (value == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();  // float(int) 溢出：交给 Python 抛出同样的异常
                return false;
            }
        } else if (PyFloat_Check(p) || PyLong_Check(p)) {
            return false;  // int / float 的子类：float(p) 可能被重载
        } else {
            concrete = 0;  // 符号参数
            value = std::nan("");
        }
        info.params.push_back(value);
        info.concrete.push_back(concrete);
    }
    return true;
}

// 单比特重综合结果的缓存（键：门名、比特数与参数的位模式，与 Python 端以 (gate, repr(params))
// 为键的 _run_cache 判等一致；含 NaN 或非 float 参数的串不进缓存）。
class FusionRunCache {
public:
    std::unordered_map<std::string, py::object> map;
};

// 宿主：节点信息来自 PyDagCore 的缓存；新指令与 Python 实现以同样的方式构造。
class PyPassHost final : public ir::DagPassHost {
public:
    explicit PyPassHost(PyDagCore& core) : core_(core) {
    }

    void set_fusion(py::handle pass_obj, FusionRunCache* cache) {
        fusion_pass_ = py::reinterpret_borrow<py::object>(pass_obj);
        fusion_cache_ = cache;
    }
    void set_identity_fn(py::handle fn) {
        identity_fn_ = py::reinterpret_borrow<py::object>(fn);
    }

    // 解码全部节点（按 instruction 对象身份复用缓存）。有无法精确处理的节点时返回 false。
    bool prepare() {
        if (!Reg().ready || !dag_core::Registry().ready || core_.has_noncanonical_wires()) {
            return false;
        }
        if (core_.pass_cache_current()) {
            return true;
        }
        core_.pass_validated_version = -1;
        auto* node_cls = reinterpret_cast<PyTypeObject*>(dag_core::Registry().dag_node_cls.ptr());
        core_.store.node_ids(ids_);
        auto& entries = core_.pass_entries();
        for (NodeId id : ids_) {
            PyObject* obj = core_.node_object(id);
            if (obj == nullptr || Py_TYPE(obj) != node_cls) {
                return false;
            }
            py::object instr = Attr(obj, Names().instruction);
            if (PyDagCore::PassEntry* e = entries.find(id); e != nullptr && e->instr.ptr() == instr.ptr()) {
                continue;
            }
            PyDagCore::PassEntry fresh;
            if (!DecodeInstruction(instr.ptr(), core_.store.kind(id), fresh)) {
                return false;
            }
            entries[id] = std::move(fresh);
        }
        return true;
    }

    const PassNode& node(NodeId id) override {
        PyDagCore::PassEntry* e = core_.pass_entries().find(id);
        if (e == nullptr) {
            throw ir::DagKeyError(id);
        }
        return e->info;
    }

    void node_removed(NodeId id) override {
        core_.forget_node(id);
    }

    void substitute_rotation(NodeId id, std::int32_t gate_name, double angle) override {
        PassRegistry& reg = Reg();
        const py::object* tmpl = &reg.p;
        if (gate_name == ir::kNameRx) {
            tmpl = &reg.rx;
        } else if (gate_name == ir::kNameRy) {
            tmpl = &reg.ry;
        } else if (gate_name == ir::kNameRz) {
            tmpl = &reg.rz;
        }
        py::object value = py::reinterpret_steal<py::object>(PyFloat_FromDouble(angle));
        if (!value) {
            throw py::error_already_set();
        }
        py::object gate = NewRotation(*tmpl, value);
        PyObject* old = entry(id).instr.ptr();
        py::object instr = NewInstruction(Attr(old, Names().type), gate, old);
        substitute(id, instr);
    }

    bool matrix_identity(NodeId id) override {
        py::object r = identity_fn_(entry(id).gate);
        const int truth = PyObject_IsTrue(r.ptr());
        if (truth < 0) {
            throw py::error_already_set();
        }
        return truth != 0;
    }

    std::optional<std::size_t> resynthesize(const std::vector<NodeId>& block) override {
        py::list gates(block.size());
        bool keyable = fusion_cache_ != nullptr;
        key_.clear();
        for (std::size_t i = 0; i < block.size(); ++i) {
            const PyDagCore::PassEntry& e = entry(block[i]);
            gates[i] = e.gate;
            if (keyable) {
                keyable = AppendKey(e);
            }
        }
        py::object result;
        if (keyable) {
            auto it = fusion_cache_->map.find(key_);
            if (it != fusion_cache_->map.end()) {
                result = it->second;
            }
        }
        if (!result) {
            result = fusion_pass_.attr("_resynthesize_block")(gates);
            if (keyable) {
                fusion_cache_->map.emplace(key_, result);
            }
        }
        if (result.is_none()) {
            pending_ = py::object();
            return std::nullopt;
        }
        pending_ = py::reinterpret_steal<py::object>(PySequence_Fast(result.ptr(), "replacement must be a sequence"));
        if (!pending_) {
            throw py::error_already_set();
        }
        return static_cast<std::size_t>(PySequence_Fast_GET_SIZE(pending_.ptr()));
    }

    void apply_replacement(NodeId id, std::size_t k) override {
        py::object gate = py::reinterpret_borrow<py::object>(
            PySequence_Fast_GET_ITEM(pending_.ptr(), static_cast<Py_ssize_t>(k)));
        PyObject* old = entry(id).instr.ptr();
        py::object instr = NewInstruction(Reg().type_gate, gate, old);
        substitute(id, instr);
    }

private:
    PyDagCore::PassEntry& entry(NodeId id) {
        PyDagCore::PassEntry* e = core_.pass_entries().find(id);
        if (e == nullptr) {
            throw ir::DagKeyError(id);
        }
        return *e;
    }

    // 不经 __init__、按字段顺序写入（与冻结数据类 __init__ 的效果相同，见 ir_bindings.cpp）。
    static py::object NewDataclass(const py::object& cls) {
        auto* type = reinterpret_cast<PyTypeObject*>(cls.ptr());
        PyObject* obj = PyBaseObject_Type.tp_new(type, EmptyTuple(), nullptr);
        if (obj == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(obj);
    }
    static void SetField(PyObject* obj, const py::str& name, PyObject* value) {
        if (PyObject_GenericSetAttr(obj, name.ptr(), value) != 0) {
            throw py::error_already_set();
        }
    }

    // template.with_params(value)：Gate(name, num_qubits, params=(value,), label=template.label)。
    // 模板不是普通 Gate 时调用 with_params 本身。
    static py::object NewRotation(const py::object& tmpl, const py::object& value) {
        PassRegistry& reg = Reg();
        PassNames& n = Names();
        if (Py_TYPE(tmpl.ptr()) != reinterpret_cast<PyTypeObject*>(reg.gate_cls.ptr())) {
            return tmpl.attr(n.with_params)(value);
        }
        py::object params = py::reinterpret_steal<py::object>(PyTuple_Pack(1, value.ptr()));
        if (!params) {
            throw py::error_already_set();
        }
        py::object gate = NewDataclass(reg.gate_cls);
        SetField(gate.ptr(), n.name, Attr(tmpl.ptr(), n.name).ptr());
        SetField(gate.ptr(), n.num_qubits, Attr(tmpl.ptr(), n.num_qubits).ptr());
        SetField(gate.ptr(), n.params, params.ptr());
        SetField(gate.ptr(), n.label, Attr(tmpl.ptr(), n.label).ptr());
        return gate;
    }

    // CircuitInstruction(type=type, gate=gate, qubits=old.qubits, clbits=old.clbits)。
    // 常见情形（GATE、单比特普通门、比特为非负 int）直接写字段，__post_init__ 的校验在这里
    // 必然通过；其他情形调用构造函数（同样的校验与异常）。
    static py::object NewInstruction(const py::object& type, const py::object& gate, PyObject* old) {
        PassRegistry& reg = Reg();
        PassNames& n = Names();
        py::object qubits = Attr(old, n.qubits);
        py::object clbits = Attr(old, n.clbits);
        if (type.is(reg.type_gate) && Py_TYPE(gate.ptr()) == reinterpret_cast<PyTypeObject*>(reg.gate_cls.ptr())
            && SimpleOneQubit(gate.ptr(), qubits.ptr(), clbits.ptr())) {
            py::object instr = NewDataclass(reg.instr_cls);
            SetField(instr.ptr(), n.type, type.ptr());
            SetField(instr.ptr(), n.qubits, qubits.ptr());
            SetField(instr.ptr(), n.clbits, clbits.ptr());
            SetField(instr.ptr(), n.gate, gate.ptr());
            SetField(instr.ptr(), n.channel, Py_None);
            SetField(instr.ptr(), n.label, EmptyStr());
            return instr;
        }
        return CallInstruction(type, gate, qubits, clbits);
    }

    // gate.num_qubits == 1 == len(qubits)，比特都是非负 int 组成的 tuple。
    static bool SimpleOneQubit(PyObject* gate, PyObject* qubits, PyObject* clbits) {
        if (!PyTuple_CheckExact(qubits) || PyTuple_GET_SIZE(qubits) != 1 || !PyTuple_CheckExact(clbits)) {
            return false;
        }
        py::object nq = Attr(gate, Names().num_qubits);
        if (!PyLong_CheckExact(nq.ptr()) || PyLong_AsLong(nq.ptr()) != 1) {
            PyErr_Clear();
            return false;
        }
        auto nonneg_int = [](PyObject* v) {
            if (!PyLong_CheckExact(v)) {
                return false;
            }
            int overflow = 0;
            const long long x = PyLong_AsLongLongAndOverflow(v, &overflow);
            return overflow > 0 || (overflow == 0 && x >= 0);
        };
        if (!nonneg_int(PyTuple_GET_ITEM(qubits, 0))) {
            return false;
        }
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(clbits); ++i) {
            if (!nonneg_int(PyTuple_GET_ITEM(clbits, i))) {
                return false;
            }
        }
        return true;
    }

    static py::object CallInstruction(const py::object& type, const py::object& gate, const py::object& qubits,
                                      const py::object& clbits) {
        PassNames& n = Names();
        py::dict kwargs;
        kwargs[n.type] = type;
        kwargs[n.gate] = gate;
        kwargs[n.qubits] = qubits;
        kwargs[n.clbits] = clbits;
        PyObject* r = PyObject_Call(Reg().instr_cls.ptr(), EmptyTuple(), kwargs.ptr());
        if (r == nullptr) {
            throw py::error_already_set();
        }
        return py::reinterpret_steal<py::object>(r);
    }

    // dag.substitute_node_instruction(id, instr)
    void substitute(NodeId id, const py::object& instr) {
        PyObject* obj = core_.node_object(id);
        if (obj == nullptr) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(id));
            PyErr_SetObject(PyExc_KeyError, key.ptr());
            throw py::error_already_set();
        }
        if (PyObject_SetAttr(obj, Names().instruction.ptr(), instr.ptr()) != 0) {
            throw py::error_already_set();
        }
        core_.store.bump_version();
        PyDagCore::PassEntry fresh;
        if (!DecodeInstruction(instr.ptr(), core_.store.kind(id), fresh)) {
            throw py::type_error("unsupported instruction produced during a native DAG pass");
        }
        core_.pass_entries()[id] = std::move(fresh);
    }

    // 门的缓存键；不能精确判等时返回 false。
    bool AppendKey(const PyDagCore::PassEntry& e) {
        PyObject* gate = e.gate.ptr();
        if (Py_TYPE(gate) != reinterpret_cast<PyTypeObject*>(Reg().gate_cls.ptr())) {
            return false;
        }
        py::object params = Attr(gate, Names().params);
        const Py_ssize_t np = PyTuple_GET_SIZE(params.ptr());
        auto put = [this](const void* p, std::size_t n) { key_.append(static_cast<const char*>(p), n); };
        const std::int32_t name = e.info.name;
        const std::int64_t nq = e.info.num_qubits;
        const auto count = static_cast<std::int32_t>(np);
        put(&name, sizeof(name));
        put(&nq, sizeof(nq));
        put(&count, sizeof(count));
        for (Py_ssize_t i = 0; i < np; ++i) {
            PyObject* p = PyTuple_GET_ITEM(params.ptr(), i);
            if (!PyFloat_CheckExact(p)) {
                return false;
            }
            const double v = PyFloat_AS_DOUBLE(p);
            if (std::isnan(v)) {
                return false;
            }
            std::uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            put(&bits, sizeof(bits));
        }
        return true;
    }

    PyDagCore& core_;
    std::vector<NodeId> ids_;
    py::object fusion_pass_;
    FusionRunCache* fusion_cache_ = nullptr;
    py::object identity_fn_;
    py::object pending_;
    std::string key_;
};

// 把 pass 中的 C++ 异常翻译成 Python 端参考实现会抛出的异常。
template <typename F>
void Translate(F&& f);

// 运行一个 pass：解码节点（失败时不修改 DAG、返回 false），运行，完成后记下缓存有效的版本号。
template <typename F>
bool RunPass(PyDagCore& d, PyPassHost& host, F&& f) {
    if (!host.prepare()) {
        return false;
    }
    bool done = true;
    Translate([&] { done = f(); });
    if (done) {
        d.pass_validated_version = d.store.version();
    }
    return done;
}

template <typename F>
void Translate(F&& f) {
    try {
        f();
    } catch (const ir::DagKeyError& e) {
        if (e.has_int_key()) {
            py::object key = py::reinterpret_steal<py::object>(PyLong_FromLongLong(e.key()));
            PyErr_SetObject(PyExc_KeyError, key.ptr());
            throw py::error_already_set();
        }
        throw py::key_error(e.what());
    } catch (const ir::DagCycleError& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    } catch (const ir::DagPassError& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    } catch (const std::logic_error& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
}

// ---------------------------------------------------------------------------
// PassManager 的收敛签名
// ---------------------------------------------------------------------------

// 常见指令的 repr 在 C++ 中直接拼出（与数据类生成的 CircuitInstruction.__repr__ 和 Gate.__repr__
// 逐字相同：float 用 float.__repr__ 的算法，str 用其自身的 repr）；其他指令调用 repr()。
// 登记时用一组样例与 repr() 逐字比较，不一致就不启用。
bool& FastReprEnabled() {
    static bool enabled = false;
    return enabled;
}

bool AppendStrRepr(PyObject* s, std::string& out) {
    if (!PyUnicode_CheckExact(s)) {
        return false;
    }
    py::object r = py::reinterpret_steal<py::object>(PyObject_Repr(s));
    if (!r) {
        throw py::error_already_set();
    }
    Py_ssize_t n = 0;
    const char* c = PyUnicode_AsUTF8AndSize(r.ptr(), &n);
    if (c == nullptr) {
        throw py::error_already_set();
    }
    out.append(c, static_cast<std::size_t>(n));
    return true;
}

bool AppendIntTuple(PyObject* t, std::string& out) {
    if (!PyTuple_CheckExact(t)) {
        return false;
    }
    const Py_ssize_t n = PyTuple_GET_SIZE(t);
    out.push_back('(');
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_GET_ITEM(t, i);
        if (!PyLong_CheckExact(item)) {
            return false;
        }
        int overflow = 0;
        const long long v = PyLong_AsLongLongAndOverflow(item, &overflow);
        if (overflow != 0) {
            return false;
        }
        if (i > 0) {
            out.append(", ");
        }
        out.append(std::to_string(v));
    }
    if (n == 1) {
        out.push_back(',');
    }
    out.push_back(')');
    return true;
}

// 类型枚举成员的 repr（登记时缓存）。
struct TypeReprs {
    std::string gate, measure, barrier, channel;
};

TypeReprs& TypeRepr() {
    static TypeReprs* t = new TypeReprs();
    return *t;
}

bool FastRepr(PyObject* instr, std::string& out) {
    PassRegistry& reg = Reg();
    PassNames& n = Names();
    out.clear();
    if (Py_TYPE(instr) != reinterpret_cast<PyTypeObject*>(reg.instr_cls.ptr())) {
        return false;
    }
    py::object type = Attr(instr, n.type);
    const std::string* type_repr = nullptr;
    TypeReprs& tr = TypeRepr();
    if (type.is(reg.type_gate)) {
        type_repr = &tr.gate;
    } else if (type.is(reg.type_measure)) {
        type_repr = &tr.measure;
    } else if (type.is(reg.type_barrier)) {
        type_repr = &tr.barrier;
    } else if (type.is(reg.type_channel)) {
        type_repr = &tr.channel;
    } else {
        return false;
    }
    if (!Attr(instr, n.channel).is_none()) {
        return false;
    }
    out.append("CircuitInstruction(type=");
    out.append(*type_repr);
    out.append(", qubits=");
    if (!AppendIntTuple(Attr(instr, n.qubits).ptr(), out)) {
        return false;
    }
    out.append(", clbits=");
    if (!AppendIntTuple(Attr(instr, n.clbits).ptr(), out)) {
        return false;
    }
    out.append(", gate=");
    py::object gate = Attr(instr, n.gate);
    if (gate.is_none()) {
        out.append("None");
    } else {
        if (Py_TYPE(gate.ptr()) != reinterpret_cast<PyTypeObject*>(reg.gate_cls.ptr())) {
            return false;
        }
        out.append("Gate(name=");
        if (!AppendStrRepr(Attr(gate.ptr(), n.name).ptr(), out)) {
            return false;
        }
        out.append(", num_qubits=");
        py::object nq = Attr(gate.ptr(), n.num_qubits);
        if (!PyLong_CheckExact(nq.ptr())) {
            return false;
        }
        int overflow = 0;
        const long long nqv = PyLong_AsLongLongAndOverflow(nq.ptr(), &overflow);
        if (overflow != 0) {
            return false;
        }
        out.append(std::to_string(nqv));
        py::object params = Attr(gate.ptr(), n.params);
        if (!PyTuple_CheckExact(params.ptr())) {
            return false;
        }
        const Py_ssize_t np = PyTuple_GET_SIZE(params.ptr());
        if (np > 0) {
            out.append(", params=(");
            for (Py_ssize_t i = 0; i < np; ++i) {
                PyObject* p = PyTuple_GET_ITEM(params.ptr(), i);
                if (!PyFloat_CheckExact(p)) {
                    return false;
                }
                char* text = PyOS_double_to_string(PyFloat_AS_DOUBLE(p), 'r', 0, Py_DTSF_ADD_DOT_0, nullptr);
                if (text == nullptr) {
                    throw py::error_already_set();
                }
                if (i > 0) {
                    out.append(", ");
                }
                out.append(text);
                PyMem_Free(text);
            }
            if (np == 1) {
                out.push_back(',');
            }
            out.push_back(')');
        }
        py::object glabel = Attr(gate.ptr(), n.label);
        if (!PyUnicode_CheckExact(glabel.ptr())) {
            return false;
        }
        if (PyUnicode_GET_LENGTH(glabel.ptr()) > 0) {
            out.append(", label=");
            AppendStrRepr(glabel.ptr(), out);
        }
        out.push_back(')');
    }
    out.append(", channel=None, label=");
    if (!AppendStrRepr(Attr(instr, n.label).ptr(), out)) {
        return false;
    }
    out.push_back(')');
    return true;
}

// id(instruction) -> repr 的缓存（与 Python 端的 repr_cache 相同：持有指令的强引用，按身份命中），
// repr 字符串驻留为编号，签名只比较编号。
class ReprCache {
public:
    std::int32_t id_of(PyObject* instr) {
        auto it = by_instr_.find(instr);
        if (it != by_instr_.end()) {
            return it->second.second;
        }
        if (!FastReprEnabled() || !FastRepr(instr, text_)) {
            py::object r = py::reinterpret_steal<py::object>(PyObject_Repr(instr));
            if (!r) {
                throw py::error_already_set();
            }
            Py_ssize_t n = 0;
            const char* s = PyUnicode_AsUTF8AndSize(r.ptr(), &n);
            if (s == nullptr) {
                throw py::error_already_set();
            }
            text_.assign(s, static_cast<std::size_t>(n));
        }
        auto found = ids_.find(text_);
        std::int32_t id = 0;
        if (found != ids_.end()) {
            id = found->second;
        } else {
            id = static_cast<std::int32_t>(reprs_.size());
            ids_.emplace(text_, id);
            reprs_.push_back(text_);
        }
        by_instr_.emplace(instr, std::make_pair(py::reinterpret_borrow<py::object>(instr), id));
        return id;
    }
    const std::string& text(std::int32_t id) const {
        return reprs_[static_cast<std::size_t>(id)];
    }
    py::str repr(std::int32_t id) const {
        return py::str(reprs_[static_cast<std::size_t>(id)]);
    }

private:
    std::unordered_map<PyObject*, std::pair<py::object, std::int32_t>> by_instr_;
    std::unordered_map<std::string, std::int32_t> ids_;
    std::vector<std::string> reprs_;
    std::string text_;
};

// 每条线上依次出现的指令；与 _wire_sequences 返回的 dict 判等关系相同：同一个指令对象的 repr
// 必然相同（Python 端的 repr_cache 同样按身份复用），不同对象才比较 repr（经缓存，只算一次）。
class WireSignature {
public:
    std::vector<std::vector<PyObject*>> seqs;  // 下标为线编号（qubit q -> 2q，clbit c -> 2c+1）
    std::vector<py::object> keep;              // 持有指令对象，保证身份比较有效
    std::shared_ptr<ReprCache> cache;

    void trim() {
        while (!seqs.empty() && seqs.back().empty()) {
            seqs.pop_back();
        }
    }
    bool equals(const WireSignature& other) const {
        if (seqs.size() != other.seqs.size()) {
            return false;
        }
        for (std::size_t w = 0; w < seqs.size(); ++w) {
            const auto& a = seqs[w];
            const auto& b = other.seqs[w];
            if (a.size() != b.size()) {
                return false;
            }
            for (std::size_t i = 0; i < a.size(); ++i) {
                if (a[i] == b[i]) {
                    continue;
                }
                if (cache == other.cache ? cache->id_of(a[i]) != cache->id_of(b[i])
                                         : cache->text(cache->id_of(a[i])) != other.cache->text(other.cache->id_of(b[i]))) {
                    return false;
                }
            }
        }
        return true;
    }

    // _wire_sequences 的 dict 形式（qubit q 的键为 q，clbit c 的键为 -1 - c）。
    py::dict to_dict() const {
        py::dict out;
        for (std::size_t w = 0; w < seqs.size(); ++w) {
            if (seqs[w].empty()) {
                continue;
            }
            const auto wire = static_cast<std::int64_t>(w);
            const std::int64_t key = wire % 2 == 0 ? wire / 2 : -1 - wire / 2;
            py::list seq(seqs[w].size());
            for (std::size_t i = 0; i < seqs[w].size(); ++i) {
                seq[i] = cache->repr(cache->id_of(seqs[w][i]));
            }
            out[py::int_(key)] = seq;
        }
        return out;
    }
};

py::object ComputeSignature(PyDagCore& core, const std::shared_ptr<ReprCache>& cache) {
    if (core.has_noncanonical_wires()) {
        return py::none();
    }
    // 每条线上依次出现的操作节点 = nodes_on_wire(线) 中的操作节点（与按拓扑序逐节点追加的结果相同；
    // 沿线遍历按线缓存，未改动的线不必重走）。
    ir::DagStore& store = core.store;
    const bool trusted = core.pass_cache_current();
    std::vector<NodeId> ids;
    store.node_ids(ids);
    auto sig = std::make_unique<WireSignature>();
    sig->cache = cache;
    sig->keep.reserve(ids.size());
    ir::IdMap<PyObject*> instrs;
    std::vector<std::uint8_t> present;
    for (NodeId id : ids) {
        if (store.kind(id) != DagNodeKind::kOperation) {
            continue;
        }
        PyDagCore::PassEntry* e = trusted ? core.pass_entries().find(id) : nullptr;
        if (e != nullptr) {
            sig->keep.push_back(e->instr);
        } else {
            PyObject* obj = core.node_object(id);
            if (obj == nullptr) {
                continue;
            }
            sig->keep.push_back(Attr(obj, Names().instruction));
        }
        instrs[id] = sig->keep.back().ptr();
        for (WireId w : *store.node_wires(id)) {
            const auto idx = static_cast<std::size_t>(w);
            if (idx >= present.size()) {
                present.resize(idx + 1, 0);
            }
            present[idx] = 1;
        }
    }
    sig->seqs.resize(present.size());
    std::vector<NodeId> wire_nodes;
    Translate([&] {
        for (std::size_t w = 0; w < present.size(); ++w) {
            if (present[w] == 0) {
                continue;
            }
            store.nodes_on_wire(static_cast<WireId>(w), wire_nodes);
            auto& seq = sig->seqs[w];
            for (NodeId id : wire_nodes) {
                if (PyObject* const* k = instrs.find(id)) {
                    seq.push_back(*k);
                }
            }
        }
    });
    sig->trim();
    return py::cast(std::move(sig));
}

}  // namespace

void BindDagPasses(py::module_& m, py::class_<PyDagCore>& core) {
    m.def(
        "_register_dag_pass_types",
        [](py::object instr_cls, py::object gate_cls, py::object type_gate, py::object type_measure,
           py::object type_barrier, py::object type_channel, py::object rx, py::object ry, py::object rz,
           py::object p) {
            PassRegistry& reg = Reg();
            reg.instr_cls = std::move(instr_cls);
            reg.gate_cls = std::move(gate_cls);
            reg.type_gate = std::move(type_gate);
            reg.type_measure = std::move(type_measure);
            reg.type_barrier = std::move(type_barrier);
            reg.type_channel = std::move(type_channel);
            reg.rx = std::move(rx);
            reg.ry = std::move(ry);
            reg.rz = std::move(rz);
            reg.p = std::move(p);
            auto type_repr = [](const py::object& t) { return py::repr(t).cast<std::string>(); };
            TypeReprs& tr = TypeRepr();
            tr.gate = type_repr(reg.type_gate);
            tr.measure = type_repr(reg.type_measure);
            tr.barrier = type_repr(reg.type_barrier);
            tr.channel = type_repr(reg.type_channel);
            reg.ready = true;
        },
        "Register the Python types used by the native DAG passes (called by quspect.dag.passes._native).");

    m.def(
        "_enable_fast_instruction_repr",
        [](const py::sequence& samples) {
            // 样例中每条能直接拼出 repr 的指令都必须与 repr() 逐字相同，才启用。
            FastReprEnabled() = false;
            std::string text;
            std::size_t formatted = 0;
            for (py::handle item : samples) {
                if (!FastRepr(item.ptr(), text)) {
                    continue;
                }
                if (text != py::repr(item).cast<std::string>()) {
                    return false;
                }
                ++formatted;
            }
            FastReprEnabled() = formatted > 0;
            return FastReprEnabled();
        },
        py::arg("samples"), "Enable the native instruction repr after checking it against repr() on samples.");

    py::class_<FusionRunCache>(m, "_FusionRunCache", py::module_local(),
                               "Memo of single-qubit resynthesis decisions for one SingleQubitGateFusion.")
        .def(py::init<>())
        .def("__len__", [](const FusionRunCache& c) { return c.map.size(); });

    py::class_<ReprCache, std::shared_ptr<ReprCache>>(m, "_ReprCache", py::module_local(),
                                                       "Instruction repr cache shared by the wire signatures "
                                                       "of one PassManager run.")
        .def(py::init<>());

    py::class_<WireSignature>(m, "_WireSignature", py::module_local(),
                              "Per-wire instruction sequence of a DAG (PassManager convergence check).")
        .def("__eq__", [](const WireSignature& a, const WireSignature& b) { return a.equals(b); }, py::is_operator())
        .def("__ne__", [](const WireSignature& a, const WireSignature& b) { return !a.equals(b); }, py::is_operator())
        .def("to_dict", &WireSignature::to_dict, "Return the signature in the form of _wire_sequences.");

    core.def(
            "_set_pass_trust", [](PyDagCore& d, bool trusted) { d.pass_trusted = trusted; }, py::arg("trusted"))
        .def(
            "_wire_signature",
            [](PyDagCore& d, const std::shared_ptr<ReprCache>& cache) { return ComputeSignature(d, cache); },
            py::arg("cache"))
        .def("_native_gate_cancellation",
             [](PyDagCore& d) {
                 PyPassHost host(d);
                 return RunPass(d, host, [&] {
                     ir::RunGateCancellation(d.store, host);
                     return true;
                 });
             })
        .def(
            "_native_commutation",
            [](PyDagCore& d, std::int64_t max_lookahead) {
                PyPassHost host(d);
                return RunPass(d, host, [&] {
                    ir::RunCommutationAnalysis(d.store, host, max_lookahead);
                    return true;
                });
            },
            py::arg("max_lookahead"))
        .def("_native_rotation_merging",
             [](PyDagCore& d) {
                 PyPassHost host(d);
                 return RunPass(d, host, [&] {
                     ir::RunRotationMerging(d.store, host);
                     return true;
                 });
             })
        .def(
            "_native_single_qubit_resynthesis",
            [](PyDagCore& d, py::object pass_obj, FusionRunCache* cache) {
                PyPassHost host(d);
                host.set_fusion(pass_obj, cache);
                return RunPass(d, host, [&] {
                    ir::RunSingleQubitResynthesis(d.store, host);
                    return true;
                });
            },
            py::arg("pass_obj"), py::arg("cache"))
        .def("_native_diagonal_before_measure",
             [](PyDagCore& d) {
                 PyPassHost host(d);
                 return RunPass(d, host, [&] {
                     ir::RunEliminateDiagonalBeforeMeasurement(d.store, host);
                     return true;
                 });
             })
        .def(
            "_native_identity_removal",
            [](PyDagCore& d, py::object is_identity) {
                PyPassHost host(d);
                host.set_identity_fn(is_identity);
                return RunPass(d, host, [&] { return ir::RunRemoveIdentityGates(d.store, host); });
            },
            py::arg("is_identity"));
}

}  // namespace quspect::native::pybind
