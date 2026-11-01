// 路由事件回放（quspect.transpiler._mapping.SABRE._finish）与 CZ 估计（estimate_two_qubit_cost）的原生实现。
//
// SABRE 候选搜索对每条路由都要回放事件、（输入有排列时）去掉末尾的路由 SWAP，再估计 CZ 数；
// Python 端每个门都要构造一个 _MappingGate 并多次遍历整个门表。这里先把路由 DAG 中的门展平成一张
// 源门表（_ReplayTable，每个 DAG 建一次），回放、末尾 SWAP 消除与估计都在紧凑的记录上完成，
// 最后一次性构造出与 Python 逐字段相同的 _MappingGate 列表。
//
// 规则与 Python 逐步相同：
//   * _replay：前缀单比特门按当前布局落位；两比特节点及其块内后续门（extra）带 gate、不带 clbit；
//     三比特节点只带比特（gate / clbit 为 None）；SWAP 记为 (True, x, y, -1, "swap")，首次被 SWAP
//     触及且没有逻辑比特的物理比特按 (x, y) 的顺序追加为工作线。
//   * _elide_trailing_swaps：从后往前，两端之后都没有非测量操作的路由 SWAP 被丢弃，其后的测量改到
//     状态最终所在的物理比特；工作线与最终映射由保留的 SWAP 重新计算；再按 sigma 还原输入排列。
//   * estimate_two_qubit_cost：同一物理比特对上的极大段合并计价 min(3, 和)（一个 SWAP 加一个代价 1 的
//     门计 2），测量或别的比特对上的两比特门结束一段。
// 含三比特节点的 DAG 只做回放（Toffoli 展开仍由 Python 完成），由调用方继续后面的步骤。
#include "replay_bindings.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "py_interned.hpp"

namespace py = pybind11;

namespace quspect::native::pybind {

namespace {

struct ReplayNames {
    py::str is_two_qubit = InternedStr("is_two_qubit");
    py::str q1 = InternedStr("q1");
    py::str q2 = InternedStr("q2");
    py::str index = InternedStr("index");
    py::str name = InternedStr("name");
    py::str clbit = InternedStr("clbit");
    py::str gate = InternedStr("gate");
    py::str q3 = InternedStr("q3");
    py::str swap = InternedStr("swap");
};

ReplayNames& Names() {
    static ReplayNames* names = new ReplayNames();
    return *names;
}

// 源门（DAG 中的一个 _MappingGate）。
struct Source {
    bool two = false;
    int q1 = 0;
    int q2 = 0;
    int q3 = -1;
    bool swap_name = false;     // name == "swap"
    bool measure_name = false;  // name == "measure"
    bool index_negative = false;
    int weight = 1;  // 估计中两比特门的代价（"swap" 为 3）
    py::object index;
    py::object name;
    py::object clbit;
    py::object gate;
};

// 回放出的一个门。form：0 单比特（复制 clbit / gate），1 两比特（复制 gate），2 三比特，3 路由 SWAP。
struct Record {
    int src;
    int q1;
    int q2;
    int q3;
    std::uint8_t form;
};

class ReplayTable {
 public:
    ReplayTable(const py::list& sources, std::vector<int> node_src, std::vector<std::vector<int>> prefix,
                std::vector<std::vector<std::pair<std::vector<int>, int>>> extra, std::vector<int> trailing)
        : node_src_(std::move(node_src)),
          prefix_(std::move(prefix)),
          extra_(std::move(extra)),
          trailing_(std::move(trailing)) {
        src_.reserve(sources.size());
        for (const py::handle item : sources) {
            const py::tuple t = item.cast<py::tuple>();
            if (t.size() != 11) {
                throw py::value_error("_ReplayTable: every source takes 11 fields");
            }
            Source s;
            s.two = t[0].cast<bool>();
            s.q1 = t[1].cast<int>();
            s.q2 = t[2].cast<int>();
            s.q3 = t[3].cast<int>();
            s.index = t[4];
            s.index_negative = t[4].cast<long long>() < 0;
            s.name = t[5];
            s.clbit = t[6];
            s.gate = t[7];
            s.weight = t[8].cast<int>();
            s.swap_name = t[9].cast<bool>();
            s.measure_name = t[10].cast<bool>();
            max_token_ = std::max({max_token_, s.q1, s.q2, s.q3});
            min_token_ = std::min({min_token_, s.q1, s.q2});
            has_three_ = has_three_ || s.q3 >= 0;
            src_.push_back(std::move(s));
        }
        const int count = static_cast<int>(src_.size());
        auto check = [count](int id) {
            if (id < 0 || id >= count) {
                throw py::value_error("_ReplayTable: source index out of range");
            }
        };
        if (prefix_.size() != node_src_.size() || (!extra_.empty() && extra_.size() != node_src_.size())) {
            throw py::value_error("_ReplayTable: per-node lists differ in length");
        }
        for (int id : node_src_) {
            check(id);
        }
        for (const auto& ids : prefix_) {
            for (int id : ids) {
                check(id);
            }
        }
        for (const auto& members : extra_) {
            for (const auto& [pre, g2] : members) {
                for (int id : pre) {
                    check(id);
                }
                check(g2);
            }
        }
        for (int id : trailing_) {
            check(id);
        }
    }

    bool has_three_qubit() const {
        return has_three_;
    }

    // SABRE._finish（+ 估计）。full = false 或 DAG 含三比特节点时只回放，返回 (gates, initial, final, -1, False)；
    // 否则 (gates, initial, final, 估计, True)。输入不满足快速路径的前提时返回 None。
    py::object Finish(const std::vector<std::int64_t>& events, const std::vector<int>& layout, int num_logical,
                      int num_physical, bool elide, const std::vector<int>& sigma, const py::object& gate_cls,
                      bool full) const {
        const int n = num_physical;
        if (n <= 0 || static_cast<int>(layout.size()) != n || num_logical < 0 || num_logical > n
            || max_token_ >= n || min_token_ < 0) {
            return py::none();
        }
        std::vector<int> pi = layout;
        std::vector<int> rpi(n, -1);
        for (int t = 0; t < n; ++t) {
            const int p = pi[t];
            if (p < 0 || p >= n || rpi[p] >= 0) {
                return py::none();
            }
            rpi[p] = t;
        }
        std::vector<int> initial(layout.begin(), layout.begin() + num_logical);
        std::vector<char> has_wire(n, 0);
        for (int p : initial) {
            has_wire[p] = 1;
        }
        const std::int64_t num_nodes = static_cast<std::int64_t>(node_src_.size());
        std::vector<Record> recs;
        recs.reserve(events.size() * 2 + trailing_.size());
        auto emit1 = [&](int id) {
            const int p = pi[src_[id].q1];
            recs.push_back({id, p, p, -1, 0});
        };
        for (const std::int64_t event : events) {
            if (event >= 0) {
                if (event >= num_nodes) {
                    return py::none();
                }
                for (int id : prefix_[event]) {
                    emit1(id);
                }
                const int node = node_src_[event];
                const Source& s = src_[node];
                if (s.two) {
                    recs.push_back({node, pi[s.q1], pi[s.q2], -1, 1});
                    if (!extra_.empty()) {
                        for (const auto& [pre, g2] : extra_[event]) {
                            for (int id : pre) {
                                emit1(id);
                            }
                            const Source& t = src_[g2];
                            recs.push_back({g2, pi[t.q1], pi[t.q2], -1, 1});
                        }
                    }
                } else if (s.q3 >= 0) {
                    recs.push_back({node, pi[s.q1], pi[s.q2], pi[s.q3], 2});
                } else {
                    emit1(node);
                }
            } else {
                const std::int64_t code = -1 - event;
                const std::int64_t xx = code / n;
                const std::int64_t yy = code % n;
                if (xx >= n) {
                    return py::none();
                }
                const int x = static_cast<int>(xx);
                const int y = static_cast<int>(yy);
                for (int p : {x, y}) {
                    if (!has_wire[p]) {
                        has_wire[p] = 1;
                        initial.push_back(p);
                    }
                }
                const int tx = rpi[x];
                const int ty = rpi[y];
                pi[tx] = y;
                pi[ty] = x;
                rpi[x] = ty;
                rpi[y] = tx;
                recs.push_back({-1, x, y, -1, 3});
            }
        }
        for (int id : trailing_) {
            emit1(id);
        }
        std::vector<int> initial_rpi(n, 0);
        for (int t = 0; t < n; ++t) {
            initial_rpi[layout[t]] = t;
        }
        std::vector<int> final_map;
        final_map.reserve(initial.size());
        for (int p : initial) {
            final_map.push_back(pi[initial_rpi[p]]);
        }
        if (!full || has_three_) {
            return py::make_tuple(Materialize(recs, gate_cls), initial, final_map, -1, false);
        }
        if (elide) {
            if (static_cast<int>(sigma.size()) < num_logical) {
                return py::none();
            }
            for (int w = 0; w < num_logical; ++w) {
                if (sigma[w] < 0 || sigma[w] >= static_cast<int>(final_map.size())) {
                    return py::none();
                }
            }
            ElideTrailingSwaps(recs, initial, final_map, num_logical, n);
            std::vector<int> mapped(final_map.size());
            for (std::size_t w = 0; w < final_map.size(); ++w) {
                mapped[w] = static_cast<int>(w) < num_logical ? final_map[static_cast<std::size_t>(sigma[w])]
                                                              : final_map[w];
            }
            final_map = std::move(mapped);
        }
        const std::int64_t est = Estimate(recs, n);
        return py::make_tuple(Materialize(recs, gate_cls), initial, final_map, est, true);
    }

 private:
    bool SwapName(const Record& r) const {
        return r.form == 3 || src_[r.src].swap_name;
    }
    bool RoutingSwap(const Record& r) const {
        return r.form == 3 || (src_[r.src].swap_name && src_[r.src].index_negative);
    }
    bool MeasureName(const Record& r) const {
        return r.form != 3 && src_[r.src].measure_name;
    }

    void ElideTrailingSwaps(std::vector<Record>& recs, std::vector<int>& wires, std::vector<int>& position,
                            int num_logical, int n) const {
        std::vector<char> touched(n, 0);
        // measures_at：物理比特 -> 之后要改到它上面的测量（记录下标）；present 区分"无"与"空表"
        std::vector<std::vector<std::size_t>> measures_at(n);
        std::vector<char> present(n, 0);
        std::vector<char> dropped(recs.size(), 0);
        bool any = false;
        for (std::size_t k = recs.size(); k-- > 0;) {
            const Record& g = recs[k];
            if (RoutingSwap(g)) {
                const int x = g.q1;
                const int y = g.q2;
                if (!touched[x] && !touched[y]) {
                    dropped[k] = 1;
                    any = true;
                    std::vector<std::size_t> mx = std::move(measures_at[x]);
                    const bool hx = present[x] != 0;
                    std::vector<std::size_t> my = std::move(measures_at[y]);
                    const bool hy = present[y] != 0;
                    measures_at[x].clear();
                    measures_at[y].clear();
                    present[x] = present[y] = 0;
                    if (hy) {
                        measures_at[x] = std::move(my);
                        present[x] = 1;
                    }
                    if (hx) {
                        measures_at[y] = std::move(mx);
                        present[y] = 1;
                    }
                    continue;
                }
                touched[x] = touched[y] = 1;
            } else if (MeasureName(g)) {
                measures_at[g.q1].push_back(k);
                present[g.q1] = 1;
            } else {
                touched[g.q1] = touched[g.q2] = 1;
            }
        }
        if (any) {
            std::vector<int> target_of(recs.size(), -1);
            for (int p = 0; p < n; ++p) {
                if (present[p]) {
                    for (std::size_t k : measures_at[p]) {
                        target_of[k] = p;
                    }
                }
            }
            std::vector<Record> kept;
            kept.reserve(recs.size());
            for (std::size_t k = 0; k < recs.size(); ++k) {
                if (dropped[k]) {
                    continue;
                }
                Record g = recs[k];
                const int p = target_of[k];
                if (p >= 0 && p != g.q1) {
                    g.q1 = p;
                    g.q2 = p;
                }
                kept.push_back(g);
            }
            recs = std::move(kept);
        }
        wires.resize(static_cast<std::size_t>(num_logical));
        std::vector<int> occupant(n, -1);
        for (int w = 0; w < num_logical; ++w) {
            occupant[wires[w]] = w;
        }
        position = wires;
        for (const Record& g : recs) {
            if (!RoutingSwap(g)) {
                continue;
            }
            for (int p : {g.q1, g.q2}) {
                if (occupant[p] < 0) {
                    occupant[p] = static_cast<int>(wires.size());
                    wires.push_back(p);
                    position.push_back(p);
                }
            }
            const int wx = occupant[g.q1];
            const int wy = occupant[g.q2];
            occupant[g.q1] = wy;
            occupant[g.q2] = wx;
            position[wx] = g.q2;
            position[wy] = g.q1;
        }
    }

    std::int64_t Estimate(const std::vector<Record>& recs, int n) const {
        struct Block {
            int a;
            int b;
            int total;
            int swaps;
            bool open;
        };
        auto block_cost = [](const Block& blk) -> std::int64_t {
            if (blk.swaps == 1 && blk.total == 4) {
                return 2;
            }
            return std::min(3, blk.total);
        };
        std::vector<int> owner(n, -1);
        std::vector<Block> blocks;
        std::int64_t cost = 0;
        auto close = [&](int b) {
            Block& blk = blocks[static_cast<std::size_t>(b)];
            cost += block_cost(blk);
            owner[blk.a] = owner[blk.b] = -1;
            blk.open = false;
        };
        for (const Record& g : recs) {
            if (g.q3 >= 0) {
                for (int q : {g.q1, g.q2, g.q3}) {
                    if (owner[q] >= 0) {
                        close(owner[q]);
                    }
                }
                cost += 6;
            } else if (g.form == 1 || g.form == 3) {
                const int a = g.q1;
                const int b = g.q2;
                const bool is_swap = SwapName(g);
                const int weight = g.form == 3 ? 3 : src_[g.src].weight;
                const int oa = owner[a];
                if (oa >= 0 && oa == owner[b]) {
                    Block& blk = blocks[static_cast<std::size_t>(oa)];
                    blk.total += weight;
                    blk.swaps += is_swap ? 1 : 0;
                    continue;
                }
                if (oa >= 0) {
                    close(oa);
                }
                if (owner[b] >= 0) {
                    close(owner[b]);
                }
                const int idx = static_cast<int>(blocks.size());
                blocks.push_back({a, b, weight, is_swap ? 1 : 0, true});
                owner[a] = owner[b] = idx;
            } else if (MeasureName(g) && owner[g.q1] >= 0) {
                close(owner[g.q1]);
            }
        }
        for (const Block& blk : blocks) {
            if (blk.open) {
                cost += block_cost(blk);
            }
        }
        return cost;
    }

    // 与 _MappingGate(is_two_qubit, q1, q2, index, name, clbit, gate, q3) 逐字段相同的对象（同样的属性顺序）。
    py::list Materialize(const std::vector<Record>& recs, const py::object& gate_cls) const {
        ReplayNames& nm = Names();
        auto* cls = reinterpret_cast<PyTypeObject*>(gate_cls.ptr());
        py::object empty = py::reinterpret_steal<py::object>(PyTuple_New(0));
        if (!empty) {
            throw py::error_already_set();
        }
        py::object minus_one = py::int_(-1);
        PyObject* list = PyList_New(static_cast<Py_ssize_t>(recs.size()));
        if (list == nullptr) {
            throw py::error_already_set();
        }
        py::list out = py::reinterpret_steal<py::list>(list);
        for (std::size_t k = 0; k < recs.size(); ++k) {
            const Record& g = recs[k];
            PyObject* raw = cls->tp_new(cls, empty.ptr(), nullptr);
            if (raw == nullptr) {
                throw py::error_already_set();
            }
            py::object obj = py::reinterpret_steal<py::object>(raw);
            auto set = [&](const py::str& name, PyObject* value) {
                if (PyObject_GenericSetAttr(obj.ptr(), name.ptr(), value) != 0) {
                    throw py::error_already_set();
                }
            };
            auto set_int = [&](const py::str& name, int v) {
                py::object o = py::reinterpret_steal<py::object>(PyLong_FromLong(v));
                if (!o) {
                    throw py::error_already_set();
                }
                set(name, o.ptr());
            };
            const bool two = g.form == 1 || g.form == 3;
            set(nm.is_two_qubit, two ? Py_True : Py_False);
            set_int(nm.q1, g.q1);
            set_int(nm.q2, g.q2);
            if (g.form == 3) {
                set(nm.index, minus_one.ptr());
                set(nm.name, nm.swap.ptr());
                set(nm.clbit, Py_None);
                set(nm.gate, Py_None);
                set(nm.q3, minus_one.ptr());
            } else {
                const Source& s = src_[g.src];
                set(nm.index, s.index.ptr());
                set(nm.name, s.name.ptr());
                set(nm.clbit, g.form == 0 ? s.clbit.ptr() : Py_None);
                set(nm.gate, g.form <= 1 ? s.gate.ptr() : Py_None);
                if (g.form == 2) {
                    set_int(nm.q3, g.q3);
                } else {
                    set(nm.q3, minus_one.ptr());
                }
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(k), obj.release().ptr());
        }
        return out;
    }

    std::vector<Source> src_;
    std::vector<int> node_src_;
    std::vector<std::vector<int>> prefix_;
    std::vector<std::vector<std::pair<std::vector<int>, int>>> extra_;
    std::vector<int> trailing_;
    int max_token_ = -1;
    int min_token_ = 0;
    bool has_three_ = false;
};

}  // namespace

void BindReplay(py::module_& m) {
    py::class_<ReplayTable>(m, "_ReplayTable", py::module_local())
        .def(py::init<const py::list&, std::vector<int>, std::vector<std::vector<int>>,
                      std::vector<std::vector<std::pair<std::vector<int>, int>>>, std::vector<int>>(),
             py::arg("sources"), py::arg("node_src"), py::arg("prefix"), py::arg("extra"), py::arg("trailing"))
        .def_property_readonly("has_three_qubit", &ReplayTable::has_three_qubit);
    m.def(
        "_replay_finish",
        [](const ReplayTable& table, const std::vector<std::int64_t>& events, const std::vector<int>& layout,
           int num_logical, int num_physical, bool elide, const std::vector<int>& sigma, const py::object& gate_cls,
           bool full) { return table.Finish(events, layout, num_logical, num_physical, elide, sigma, gate_cls, full); },
        py::arg("table"), py::arg("events"), py::arg("layout"), py::arg("num_logical"), py::arg("num_physical"),
        py::arg("elide"), py::arg("sigma"), py::arg("gate_cls"), py::arg("full") = true,
        "Replay routing events (SABRE._finish) and estimate the CZ count; returns (gates, initial, final, "
        "estimate, finished) or None.");
}

}  // namespace quspect::native::pybind
