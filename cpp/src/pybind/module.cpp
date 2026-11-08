#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quspect_native/common/parallel.hpp"
#include "quspect_native/detmath/linalg.hpp"
#include "quspect_native/executor/executor.hpp"
#include "quspect_native/executor/gradient_executor.hpp"
#include "quspect_native/ir/program_builder.hpp"
#include "quspect_native/ir/qsal_decoder.hpp"
#include "quspect_native/ir/validation.hpp"
#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/kernels/noise_kernels.hpp"
#include "quspect_native/optimization/ising_solver.hpp"
#include "quspect_native/runtime/classical_register.hpp"
#include "quspect_native/runtime/density_matrix.hpp"
#include "quspect_native/runtime/program.hpp"
#include "quspect_native/runtime/state_vector.hpp"
#include "quspect_native/transpiler/cpython_random.hpp"
#include "quspect_native/transpiler/cx_orientation.hpp"
#include "quspect_native/transpiler/depth_schedule.hpp"
#include "quspect_native/transpiler/dilation_layout.hpp"
#include "quspect_native/transpiler/layer_routing.hpp"
#include "quspect_native/transpiler/one_qubit_gauge.hpp"
#include "quspect_native/transpiler/ring_routing.hpp"
#include "quspect_native/transpiler/route_lns.hpp"
#include "quspect_native/transpiler/sabre_routing.hpp"
#include "quspect_native/transpiler/su4_routing.hpp"
#include "quspect_native/transpiler/beam_routing.hpp"
#include "quspect_native/transpiler/cnot_phase_synth.hpp"
#include "quspect_native/transpiler/phase_poly.hpp"

#include "dag_bindings.hpp"
#include "detmath_bindings.hpp"
#include "ir_bindings.hpp"
#include "consolidate_bindings.hpp"
#include "gauge_bindings.hpp"
#include "basis_bindings.hpp"
#include "circuit_bindings.hpp"
#include "phase_fold_bindings.hpp"
#include "replay_bindings.hpp"
#include "schedule_bindings.hpp"
#include "synthesis_bindings.hpp"

namespace py = pybind11;
using namespace quspect::native;

namespace {

using ComplexArray = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

// (N, 2, 2) 复数数组 → 2×2 矩阵列表（行主序）。
std::vector<transpiler::Matrix2> ToMatrices(const ComplexArray& arr) {
    if (arr.ndim() != 3 || arr.shape(1) != 2 || arr.shape(2) != 2) {
        throw std::invalid_argument("expected an (N, 2, 2) complex array");
    }
    const auto n = static_cast<std::size_t>(arr.shape(0));
    const std::complex<double>* data = arr.data();
    std::vector<transpiler::Matrix2> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (int j = 0; j < 4; ++j) {
            const std::complex<double>& z = data[i * 4 + static_cast<std::size_t>(j)];
            out[i][j] = transpiler::Complex{z.real(), z.imag()};
        }
    }
    return out;
}

// 2×2 矩阵列表 → (N, 2, 2) 复数数组。
py::array_t<std::complex<double>> FromMatrices(const std::vector<transpiler::Matrix2>& ms) {
    const auto n = static_cast<py::ssize_t>(ms.size());
    py::array_t<std::complex<double>> out({n, static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
    std::complex<double>* data = out.mutable_data();
    for (std::size_t i = 0; i < ms.size(); ++i) {
        for (int j = 0; j < 4; ++j) {
            data[i * 4 + static_cast<std::size_t>(j)] = std::complex<double>(ms[i][j].re, ms[i][j].im);
        }
    }
    return out;
}

template <typename T>
py::array_t<T> ToArray(const std::vector<T>& v) {
    py::array_t<T> out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// memo 键（``phi_e`` 按位比较）。
transpiler::GaugeBeamSearch::MemoKey MemoKeyOf(int rid, bool fp, bool fn, std::int64_t key, bool last, double phi_e) {
    std::uint64_t bits;
    std::memcpy(&bits, &phi_e, sizeof(bits));
    return transpiler::GaugeBeamSearch::MemoKey{rid, fp, fn, key, last, bits};
}

executor::PrecisionKind ParsePrecision(const std::string& precision) {
    if (precision == "single") {
        return executor::PrecisionKind::kFloat32;
    }
    if (precision == "double") {
        return executor::PrecisionKind::kFloat64;
    }
    throw std::invalid_argument("precision must be 'single' or 'double'");
}

/// Owns a decoded variational program so repeated objective evaluations only
/// transfer parameter values across the Python/C++ boundary.
class AdjointPlan {
 public:
    AdjointPlan(const std::string& qsal_json, const std::string& precision) {
        ir::QSALDecoder decoder;
        auto program = decoder.Decode(qsal_json);
        if (!program) {
            throw std::runtime_error("QSAL decode error: " + decoder.LastError());
        }

        const auto errors = ir::Validator::Validate(*program);
        if (!errors.empty()) {
            std::string message = "QSAL validation error(s):";
            for (const auto& error : errors) {
                message += "\n  - " + error;
            }
            throw std::runtime_error(message);
        }

        precision_ = ParsePrecision(precision);
        prepared_ = std::make_unique<executor::PreparedAdjointGradient>(*program);
    }

    executor::GradientResult Evaluate(const std::unordered_map<std::string, double>& parameter_values) const {
        return prepared_->Run(parameter_values, precision_);
    }

    const std::vector<std::string>& ParameterNames() const {
        return prepared_->parameter_names();
    }

    const std::vector<std::string>& ObservableNames() const {
        return prepared_->observable_labels();
    }

 private:
    std::unique_ptr<executor::PreparedAdjointGradient> prepared_;
    executor::PrecisionKind precision_ = executor::PrecisionKind::kFloat64;
};

/// 获取运行时能力信息
py::dict get_runtime_capabilities() {
    py::dict caps;
    caps["avx2_available"] = kernels::CpuSupportsAVX2();
    caps["backend_f32"] = kernels::GetDispatchF32().backend_name;
    caps["backend_f64"] = kernels::GetDispatchF64().backend_name;
    caps["version"] = "0.2.0";
    caps["phase"] = "state-vector objective plans";
    caps["objective_plans"] = true;
    caps["differentiation_methods"] = py::make_tuple("adjoint");
    caps["differentiable_gates"] = py::make_tuple("rx", "ry", "rz");
    caps["noise_channels"] = true;
    caps["noisy_trajectories"] = true;
    caps["openmp"] = common::OpenMPEnabled();
    caps["num_threads"] = common::GetNumThreads();
    return caps;
}

/// Helper: get data from DensityMatrix as a mutable numpy VIEW.
///
/// A bare data pointer with no base handle makes pybind allocate an owning
/// copy; writes through it (as DensityMatrix.copy()/from_array do via
/// `_data[...] = ...`) then never reach the C++ storage, so those methods
/// silently return an all-zero |0><0| state. Using the Python ``self`` object
/// as the array base makes the array a non-owning VIEW into ``amplitudes_``
/// (reads and writes hit the underlying vector) AND keeps the DensityMatrix
/// alive for as long as the view exists (a bare capsule would dangle once a
/// temporary DensityMatrix is garbage-collected).
py::array_t<std::complex<double>> _dm_data_py(py::object self) {
    auto& dm = self.cast<runtime::DensityMatrix<double>&>();
    const auto dim = static_cast<py::ssize_t>(dm.dim());
    const auto item_size = static_cast<py::ssize_t>(sizeof(std::complex<double>));
    return py::array_t<std::complex<double>>({dim, dim}, {dim * item_size, item_size}, dm.data(), self);
}

/// Helper: apply unitary to DensityMatrix from Python objects.
void _dm_apply_unitary_py(runtime::DensityMatrix<double>& dm, const py::object& umat_obj,
                          const py::object& qubits_obj) {
    auto umat = py::array_t<std::complex<double>, py::array::c_style>::ensure(umat_obj);
    auto qubits = py::array_t<uint32_t, py::array::c_style>::ensure(qubits_obj);
    if (!umat || !qubits) {
        throw std::runtime_error("Invalid array arguments for DensityMatrix.apply_unitary");
    }
    dm.ApplyUnitary(umat.data(), qubits.data(), static_cast<uint32_t>(qubits.size()));
}

/// Helper: apply Kraus channel to DensityMatrix from Python objects.
void _dm_apply_kraus_py(runtime::DensityMatrix<double>& dm, const py::object& kraus_obj, uint32_t num_ops,
                        const py::object& qubits_obj) {
    auto kraus = py::array_t<std::complex<double>, py::array::c_style>::ensure(kraus_obj);
    auto qubits = py::array_t<uint32_t, py::array::c_style>::ensure(qubits_obj);
    if (!kraus || !qubits) {
        throw std::runtime_error("Invalid array arguments for DensityMatrix.apply_kraus_channel");
    }
    dm.ApplyKrausChannel(kraus.data(), num_ops, qubits.data(), static_cast<uint32_t>(qubits.size()));
}

/// Kraus Monte Carlo wrapper for pybind.
uint32_t apply_kraus_mc_py(py::array_t<std::complex<double>, py::array::c_style> state,
                           py::array_t<std::complex<double>, py::array::c_style> kraus_ops, uint32_t num_ops,
                           py::array_t<uint32_t, py::array::c_style> qubits, uint64_t seed) {
    uint64_t dim = static_cast<uint64_t>(state.size());
    std::mt19937_64 rng(seed);
    std::complex<double>* state_data = state.mutable_data();
    const std::complex<double>* kraus_data = kraus_ops.data();
    const uint32_t* qubit_data = qubits.data();
    const auto num_qubits = static_cast<uint32_t>(qubits.size());
    // 只访问已取出的原始缓冲区指针，不触碰 Python 对象，可以安全释放 GIL。
    py::gil_scoped_release release;
    return kernels::apply_kraus_monte_carlo<double>(state_data, dim, kraus_data, num_ops, qubit_data, num_qubits, rng);
}

using DenseMatrixArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

optimization::DenseIsingModel DenseIsingModelFromArray(const DenseMatrixArray& matrix) {
    if (matrix.ndim() != 2 || matrix.shape(0) != matrix.shape(1)) {
        throw std::invalid_argument("Ising matrix must be a square two-dimensional array");
    }
    std::vector<double> coefficients;
    if (matrix.size() > 0) {
        coefficients.assign(matrix.data(), matrix.data() + matrix.size());
    }
    return optimization::DenseIsingModel(std::move(coefficients), static_cast<std::size_t>(matrix.shape(0)));
}

py::array_t<std::int8_t> IsingSamplesToArray(const optimization::IsingSolverResult& result) {
    const auto num_reads = static_cast<py::ssize_t>(result.samples.size());
    const auto num_spins = static_cast<py::ssize_t>(result.num_spins);
    py::array_t<std::int8_t> output({num_reads, num_spins});
    auto view = output.mutable_unchecked<2>();
    for (py::ssize_t read = 0; read < num_reads; ++read) {
        const auto& spins = result.samples[static_cast<std::size_t>(read)].spins;
        for (py::ssize_t spin = 0; spin < num_spins; ++spin) {
            view(read, spin) = spins[static_cast<std::size_t>(spin)];
        }
    }
    return output;
}

py::array_t<double> IsingEnergiesToArray(const optimization::IsingSolverResult& result) {
    const auto num_reads = static_cast<py::ssize_t>(result.samples.size());
    py::array_t<double> output(num_reads);
    auto view = output.mutable_unchecked<1>();
    for (py::ssize_t read = 0; read < num_reads; ++read) {
        view(read) = result.samples[static_cast<std::size_t>(read)].energy;
    }
    return output;
}

}  // namespace

// 同一份源码编译为两个扩展模块：quspect._core_scalar 与 quspect._core_avx2。
// 模块名由 CMake 通过 QSPECT_MODULE_NAME 传入，必须与输出文件名一致。
#ifndef QSPECT_MODULE_NAME
#    error "QSPECT_MODULE_NAME must be defined by the build system"
#endif

// 两个模块可能被同一进程加载（加载器先导入 scalar 模块探测 CPU），因此所有绑定类型
// 都使用 py::module_local()，避免 pybind11 全局类型注册表冲突。
PYBIND11_MODULE(QSPECT_MODULE_NAME, m) {
    m.doc() = "QuSpect C++ native simulation backend";

    // ---- 线路 IR：连续存储的线路（隐式 DAG）、QASM 快速路径与 DAGCircuit 的图存储内核 ----
    quspect::native::pybind::BindIr(m);
    quspect::native::pybind::BindDagCore(m);
    quspect::native::pybind::BindSynthesis(m);
    quspect::native::pybind::BindDetmath(m);
    quspect::native::pybind::BindSchedule(m);
    quspect::native::pybind::BindConsolidate(m);
    quspect::native::pybind::BindGauge(m);
    quspect::native::pybind::BindBasis(m);
    quspect::native::pybind::BindCircuitHelpers(m);
    quspect::native::pybind::BindPhaseFold(m);
    quspect::native::pybind::BindReplay(m);

    // ---- CPU 特性探测（供 Python 加载器决定是否导入 AVX2 模块）----
    m.def("cpu_supports_avx2", &kernels::CpuSupportsAVX2,
          "Return True if the CPU and OS support AVX2 + FMA (required by the AVX2 kernels)");

    // ---- 运行时能力 ----
    m.def("get_runtime_capabilities", &get_runtime_capabilities,
          "Return dict of runtime capabilities and active backend info");

    // ---- OpenMP 线程数（进程内对本扩展模块的所有并行内核生效）----
    m.def("get_num_threads", &common::GetNumThreads,
          "Return the number of threads used by parallel kernels (1 when built without OpenMP)");
    m.def("set_num_threads", &common::SetNumThreads, py::arg("n"),
          "Set the number of threads used by parallel kernels; raises ValueError when n < 1");

    // ---- 稠密 Ising 优化 ----
    py::class_<optimization::IsingSolverResult>(m, "_NativeIsingResult", py::module_local())
        .def_property_readonly("samples", &IsingSamplesToArray)
        .def_property_readonly("energies", &IsingEnergiesToArray);

    m.def(
        "_solve_dense_ising",
        [](const DenseMatrixArray& matrix, std::uint64_t seed, std::size_t num_reads, std::size_t sweeps,
           double beta_start, double beta_end) {
            auto model = DenseIsingModelFromArray(matrix);
            optimization::IsingSolverOptions options;
            options.seed = seed;
            options.num_reads = num_reads;
            options.sweeps = sweeps;
            options.beta_start = beta_start;
            options.beta_end = beta_end;
            py::gil_scoped_release release;
            return optimization::SolveIsingSimulatedAnnealing(model, options);
        },
        py::arg("matrix"), py::arg("seed") = 0, py::arg("num_reads") = 1, py::arg("sweeps") = 1000,
        py::arg("beta_start") = 0.1, py::arg("beta_end") = 5.0,
        "Minimize E(s) = -s.T @ matrix @ s with simulated annealing.");

    m.def(
        "_solve_dense_ising_exact",
        [](const DenseMatrixArray& matrix, std::size_t max_spins) {
            auto model = DenseIsingModelFromArray(matrix);
            optimization::IsingSolverResult result;
            result.num_spins = model.num_spins();
            {
                py::gil_scoped_release release;
                result.samples.push_back(optimization::SolveIsingExact(model, max_spins));
            }
            return result;
        },
        py::arg("matrix"), py::arg("max_spins") = 24,
        "Exactly minimize a small dense Ising matrix with Gray-code enumeration.");

    // ---- SABRE 路由核心（quspect.transpiler._routing 的原生实现）----
    py::class_<transpiler::RoutingGraph>(m, "_RoutingGraph", py::module_local())
        .def(py::init<std::vector<std::vector<int>>, const std::vector<std::vector<std::int64_t>>&>(),
             py::arg("adjacency"), py::arg("distance"))
        .def_property_readonly("num_physical", &transpiler::RoutingGraph::num_physical);

    py::class_<transpiler::RoutingDag>(m, "_RoutingDag", py::module_local())
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<int>, std::vector<std::vector<int>>,
                      std::vector<std::vector<int>>, bool, std::vector<int>, std::vector<int>>(),
             py::arg("nq1"), py::arg("nq2"), py::arg("nq3"), py::arg("succ"), py::arg("pred"),
             py::arg("has_three_qubit"), py::arg("mirror_fwd") = std::vector<int>{},
             py::arg("mirror_bwd") = std::vector<int>{})
        .def_property_readonly("num_nodes", &transpiler::RoutingDag::num_nodes);

    m.def(
        "_route_dag",
        [](const transpiler::RoutingGraph& graph, const transpiler::RoutingDag& dag, const std::vector<int>& layout,
           bool forward, bool record, double lookahead_weight, int extended_set_size, double decay_delta,
           int decay_reset_interval, int release_valve_threshold, double swap_fusion_bonus, bool mirror_gates,
           double mirror_threshold, double mirror_cz_penalty, const std::vector<std::uint32_t>& rng_state) {
            auto rng = transpiler::CPythonRandom::FromState(rng_state);
            transpiler::RouteOptions options;
            options.lookahead_weight = lookahead_weight;
            options.extended_set_size = extended_set_size;
            options.decay_delta = decay_delta;
            options.decay_reset_interval = decay_reset_interval;
            options.release_valve_threshold = release_valve_threshold;
            options.swap_fusion_bonus = swap_fusion_bonus;
            options.mirror_gates = mirror_gates;
            options.mirror_threshold = mirror_threshold;
            options.mirror_cz_penalty = mirror_cz_penalty;
            transpiler::RouteResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::RouteDag(graph, dag, options, layout, forward, record, rng);
            }
            py::object state = py::none();
            if (rng.used()) {
                state = py::tuple(py::cast(rng.State()));
            }
            return py::make_tuple(result.num_swaps, std::move(result.final_pi), std::move(result.events), result.depth,
                                  result.swap_cost, result.num_mirrors, state);
        },
        py::arg("graph"), py::arg("dag"), py::arg("layout"), py::arg("forward"), py::arg("record"),
        py::arg("lookahead_weight"), py::arg("extended_set_size"), py::arg("decay_delta"),
        py::arg("decay_reset_interval"), py::arg("release_valve_threshold"), py::arg("swap_fusion_bonus"),
        py::arg("mirror_gates"), py::arg("mirror_threshold"), py::arg("mirror_cz_penalty"), py::arg("rng_state"),
        "Run one SABRE routing pass; returns (num_swaps, final_pi, events, depth, swap_cost, num_mirrors, "
        "rng_state or None).");

    // 多条互不相关的路由链（SABRE 候选的各次试验）：每项 (layout, iters, rng_state) 先做 iters 次
    // "正向 + 反向"的布局精炼（不记录事件），再从精炼后的布局正向路由一次（记录事件），整条链用同一个
    // 随机源，与依次调用 _route_dag 相同。各项在一个任务中计算，按输入顺序返回
    // (精炼后的布局, 每次路由的 SWAP 数, num_swaps, final_pi, events, depth, swap_cost, num_mirrors)；
    // 出错（例如不连通）的项返回 None，由调用方按原路径重算以得到同样的异常。
    m.def(
        "_route_dag_chains",
        [](const transpiler::RoutingGraph& graph, const transpiler::RoutingDag& dag, const py::list& jobs,
           double lookahead_weight, int extended_set_size, double decay_delta, int decay_reset_interval,
           int release_valve_threshold, double swap_fusion_bonus, bool mirror_gates, double mirror_threshold,
           double mirror_cz_penalty) {
            transpiler::RouteOptions options;
            options.lookahead_weight = lookahead_weight;
            options.extended_set_size = extended_set_size;
            options.decay_delta = decay_delta;
            options.decay_reset_interval = decay_reset_interval;
            options.release_valve_threshold = release_valve_threshold;
            options.swap_fusion_bonus = swap_fusion_bonus;
            options.mirror_gates = mirror_gates;
            options.mirror_threshold = mirror_threshold;
            options.mirror_cz_penalty = mirror_cz_penalty;
            struct Chain {
                std::vector<int> layout;
                int iters = 0;
                std::vector<std::uint32_t> state;
                std::vector<int> swaps;
                transpiler::RouteResult result;
                bool ok = false;
            };
            std::vector<Chain> chains(jobs.size());
            for (std::size_t i = 0; i < chains.size(); ++i) {
                const py::tuple job = jobs[i].cast<py::tuple>();
                chains[i].layout = job[0].cast<std::vector<int>>();
                chains[i].iters = job[1].cast<int>();
                chains[i].state = job[2].cast<std::vector<std::uint32_t>>();
            }
            {
                py::gil_scoped_release release;
                std::vector<std::function<void()>> tasks;
                tasks.reserve(chains.size());
                for (Chain& chain : chains) {
                    tasks.emplace_back([&graph, &dag, &options, &chain] {
                        try {
                            auto rng = transpiler::CPythonRandom::FromState(chain.state);
                            for (int it = 0; it < chain.iters; ++it) {
                                const auto fwd = transpiler::RouteDag(graph, dag, options, chain.layout, true, false, rng);
                                chain.swaps.push_back(fwd.num_swaps);
                                const auto bwd = transpiler::RouteDag(graph, dag, options, fwd.final_pi, false, false, rng);
                                chain.swaps.push_back(bwd.num_swaps);
                                chain.layout = bwd.final_pi;
                            }
                            chain.result = transpiler::RouteDag(graph, dag, options, chain.layout, true, true, rng);
                            chain.swaps.push_back(chain.result.num_swaps);
                            chain.ok = true;
                        } catch (const std::exception&) {
                            chain.ok = false;
                        }
                    });
                }
                common::RunTasks(tasks, common::CompileThreads());
            }
            py::list out;
            for (Chain& chain : chains) {
                if (!chain.ok) {
                    out.append(py::none());
                    continue;
                }
                transpiler::RouteResult& r = chain.result;
                out.append(py::make_tuple(std::move(chain.layout), std::move(chain.swaps), r.num_swaps,
                                          std::move(r.final_pi), std::move(r.events), r.depth, r.swap_cost,
                                          r.num_mirrors));
            }
            return out;
        },
        py::arg("graph"), py::arg("dag"), py::arg("jobs"), py::arg("lookahead_weight"), py::arg("extended_set_size"),
        py::arg("decay_delta"), py::arg("decay_reset_interval"), py::arg("release_valve_threshold"),
        py::arg("swap_fusion_bonus"), py::arg("mirror_gates"), py::arg("mirror_threshold"),
        py::arg("mirror_cz_penalty"),
        "Independent routing chains (layout refinement + recorded forward pass), computed concurrently.");

    // 路由的精确窗口搜索与大邻域搜索（quspect.transpiler._route_lns 的原生核心）
    m.def(
        "_route_swap_cost",
        [](const transpiler::RoutingGraph& graph, const transpiler::RoutingDag& dag,
           const std::vector<int>& mirror_cost, const std::vector<int>& measure_token, const std::vector<int>& layout,
           const std::vector<std::int64_t>& events, bool merge_costs) {
            transpiler::RouteCostModel model{mirror_cost, measure_token};
            py::gil_scoped_release release;
            return transpiler::RouteSwapCost(graph, dag, model, layout, events, merge_costs);
        },
        py::arg("graph"), py::arg("dag"), py::arg("mirror_cost"), py::arg("measure_token"), py::arg("layout"),
        py::arg("events"), py::arg("merge_costs") = true,
        "SWAP cost of a routing event sequence under the block-merge cost model.");

    m.def(
        "_exact_route",
        [](const transpiler::RoutingGraph& graph, const transpiler::RoutingDag& dag,
           const std::vector<int>& mirror_cost, const std::vector<int>& measure_token, const std::vector<int>& layout,
           std::int64_t max_expansions, bool merge_costs, std::int64_t upper_bound) {
            transpiler::RouteCostModel model{mirror_cost, measure_token};
            transpiler::ExactSearchOptions options;
            options.max_expansions = max_expansions;
            options.merge_costs = merge_costs;
            transpiler::ExactRouteResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::ExactRoute(graph, dag, model, layout, options, upper_bound);
            }
            return py::make_tuple(result.found, result.optimal, result.cost, result.num_swaps, result.expansions,
                                  result.lower_bound, std::move(result.events));
        },
        py::arg("graph"), py::arg("dag"), py::arg("mirror_cost"), py::arg("measure_token"), py::arg("layout"),
        py::arg("max_expansions"), py::arg("merge_costs"), py::arg("upper_bound") = 0,
        "A* routing of the whole DAG; returns (found, optimal, cost, num_swaps, expansions, lower_bound, events).");

    m.def(
        "_refine_routing",
        [](const transpiler::RoutingGraph& graph, const transpiler::RoutingDag& dag,
           const std::vector<int>& mirror_cost, const std::vector<int>& measure_token, const std::vector<int>& layout,
           const std::vector<std::int64_t>& events, const std::vector<int>& window_swaps, int passes,
           std::int64_t window_expansions, std::int64_t total_expansions, int max_window_nodes, bool merge_costs,
           double weight, bool stop_at_first, const std::vector<int>& node_weight, bool depth_ties, int depth_passes,
           std::int64_t depth_expansions) {
            transpiler::RouteCostModel model{mirror_cost, measure_token, node_weight};
            transpiler::RefineOptions options;
            options.window_swaps = window_swaps;
            options.passes = passes;
            options.window_expansions = window_expansions;
            options.total_expansions = total_expansions;
            options.max_window_nodes = max_window_nodes;
            options.merge_costs = merge_costs;
            options.weight = weight;
            options.stop_at_first = stop_at_first;
            options.depth_ties = depth_ties;
            options.depth_passes = depth_passes;
            options.depth_expansions = depth_expansions;
            options.threads = common::CompileThreads();
            transpiler::RefineResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::RefineRouting(graph, dag, model, layout, events, options);
            }
            return py::make_tuple(std::move(result.events), result.old_cost, result.new_cost, result.old_swaps,
                                  result.new_swaps, result.windows, result.improved, result.expansions,
                                  result.old_depth, result.new_depth);
        },
        py::arg("graph"), py::arg("dag"), py::arg("mirror_cost"), py::arg("measure_token"), py::arg("layout"),
        py::arg("events"), py::arg("window_swaps"), py::arg("passes"), py::arg("window_expansions"),
        py::arg("total_expansions"), py::arg("max_window_nodes"), py::arg("merge_costs"), py::arg("weight") = 1.0,
        py::arg("stop_at_first") = false, py::arg("node_weight") = std::vector<int>{}, py::arg("depth_ties") = false,
        py::arg("depth_passes") = 0, py::arg("depth_expansions") = 0,
        "Large-neighbourhood search over routing windows; returns (events, old_cost, new_cost, old_swaps, "
        "new_swaps, windows, improved, expansions, old_depth, new_depth).");

    // 多个互不相关的 _refine_routing（各候选路由的精炼）：每项的参数元组与 _refine_routing 的位置参数相同，
    // 各自在一个任务中计算，按输入顺序返回与依次调用相同的结果
    m.def(
        "_refine_routing_many",
        [](const py::list& jobs) {
            struct Job {
                const transpiler::RoutingGraph* graph;
                const transpiler::RoutingDag* dag;
                std::vector<int> mirror_cost;
                std::vector<int> measure_token;
                std::vector<int> node_weight;
                std::vector<int> layout;
                std::vector<std::int64_t> events;
                transpiler::RefineOptions options;
                transpiler::RefineResult result;
            };
            std::vector<Job> work(jobs.size());
            for (std::size_t i = 0; i < work.size(); ++i) {
                const py::tuple args = jobs[i].cast<py::tuple>();
                if (args.size() != 18) {
                    throw py::value_error("_refine_routing_many: every job takes the 18 arguments of _refine_routing");
                }
                Job& job = work[i];
                job.graph = &args[0].cast<const transpiler::RoutingGraph&>();
                job.dag = &args[1].cast<const transpiler::RoutingDag&>();
                job.mirror_cost = args[2].cast<std::vector<int>>();
                job.measure_token = args[3].cast<std::vector<int>>();
                job.layout = args[4].cast<std::vector<int>>();
                job.events = args[5].cast<std::vector<std::int64_t>>();
                transpiler::RefineOptions& options = job.options;
                options.window_swaps = args[6].cast<std::vector<int>>();
                options.passes = args[7].cast<int>();
                options.window_expansions = args[8].cast<std::int64_t>();
                options.total_expansions = args[9].cast<std::int64_t>();
                options.max_window_nodes = args[10].cast<int>();
                options.merge_costs = args[11].cast<bool>();
                options.weight = args[12].cast<double>();
                options.stop_at_first = args[13].cast<bool>();
                job.node_weight = args[14].cast<std::vector<int>>();
                options.depth_ties = args[15].cast<bool>();
                options.depth_passes = args[16].cast<int>();
                options.depth_expansions = args[17].cast<std::int64_t>();
            }
            {
                py::gil_scoped_release release;
                // 线程分给各项：任务之间并行，每项内部再用分到的线程做窗口投机（结果与线程数无关）
                const int threads = common::CompileThreads();
                const int count = static_cast<int>(std::max<std::size_t>(work.size(), 1));
                const int inner = std::max(1, threads / count);
                std::vector<std::function<void()>> tasks;
                tasks.reserve(work.size());
                for (Job& job : work) {
                    job.options.threads = inner;
                    tasks.emplace_back([&job] {
                        transpiler::RouteCostModel model{job.mirror_cost, job.measure_token, job.node_weight};
                        job.result = transpiler::RefineRouting(*job.graph, *job.dag, model, job.layout, job.events,
                                                               job.options);
                    });
                }
                common::RunTasks(tasks, threads);
            }
            py::list out;
            for (Job& job : work) {
                transpiler::RefineResult& result = job.result;
                out.append(py::make_tuple(std::move(result.events), result.old_cost, result.new_cost,
                                          result.old_swaps, result.new_swaps, result.windows, result.improved,
                                          result.expansions, result.old_depth, result.new_depth));
            }
            return out;
        },
        py::arg("jobs"),
        "_refine_routing for each argument tuple of jobs, computed concurrently; list of its results.");

    // 逐层匹配线路在路径上的路由（quspect.transpiler._layer_route 的原生核心）
    m.def(
        "_route_layers_on_path",
        [](int n, const std::vector<std::vector<std::pair<int, int>>>& layers, int max_sweeps, int perturbations,
           int perturb_span, std::uint64_t seed) {
            transpiler::LayerChainOptions options;
            options.max_sweeps = max_sweeps;
            options.perturbations = perturbations;
            options.perturb_span = perturb_span;
            options.seed = seed;
            transpiler::LayerChainResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::RouteLayersOnPath(n, layers, options);
            }
            return py::make_tuple(result.cost, std::move(result.initial), std::move(result.swaps),
                                  std::move(result.flips), result.sweeps);
        },
        py::arg("n"), py::arg("layers"), py::arg("max_sweeps") = 64, py::arg("perturbations") = 0,
        py::arg("perturb_span") = 6, py::arg("seed") = 1,
        "Route a sequence of qubit-pair layers on a path; returns (cost, initial, swaps, flips, sweeps).");

    m.def(
        "_route_layers_on_ring",
        [](int n, const std::vector<std::vector<std::pair<int, int>>>& layers, int max_sweeps, int perturbations,
           int perturb_span, std::uint64_t seed) {
            transpiler::LayerChainOptions options;
            options.max_sweeps = max_sweeps;
            options.perturbations = perturbations;
            options.perturb_span = perturb_span;
            options.seed = seed;
            transpiler::LayerChainResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::RouteLayersOnRing(n, layers, options);
            }
            return py::make_tuple(result.cost, std::move(result.initial), std::move(result.swaps),
                                  std::move(result.flips), result.sweeps);
        },
        py::arg("n"), py::arg("layers"), py::arg("max_sweeps") = 64, py::arg("perturbations") = 0,
        py::arg("perturb_span") = 6, py::arg("seed") = 1,
        "Route a sequence of qubit-pair layers on a ring; returns (cost, initial, swaps, flips, sweeps); "
        "swap i exchanges ring sites i and (i + 1) mod n.");

    m.def(
        "_route_su4_blocks",
        [](const std::vector<std::vector<int>>& adjacency, const std::vector<std::pair<int, int>>& blocks,
           const std::vector<int>& layout, int width, double ext_weight, double exe_bonus, int layout_iters,
           std::uint64_t seed, bool depth_tiebreak) {
            transpiler::Su4RouteOptions options;
            options.depth_tiebreak = depth_tiebreak;
            options.width = width;
            options.ext_weight = ext_weight;
            options.exe_bonus = exe_bonus;
            options.layout_iters = layout_iters;
            options.seed = seed;
            transpiler::Su4RouteResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::RouteSu4Blocks(adjacency, blocks, layout, options);
            }
            return py::make_tuple(result.paid, result.flips, std::move(result.layout), std::move(result.events),
                                  result.depth);
        },
        py::arg("adjacency"), py::arg("blocks"), py::arg("layout"), py::arg("width") = 16, py::arg("ext_weight") = 0.5,
        py::arg("exe_bonus") = 0.5, py::arg("layout_iters") = 1, py::arg("seed") = 1, py::arg("depth_tiebreak") = true,
        "Truncated-rollout routing of generic SU(4) blocks; returns (paid, flips, layout, events, depth).");

    // 同一输入、不同 (width, seed) 的多次截断展开路由：互不相关，各自在一个任务中计算（结果与依次
    // 调用 _route_su4_blocks 相同，按 configs 的顺序返回）
    m.def(
        "_route_su4_blocks_many",
        [](const std::vector<std::vector<int>>& adjacency, const std::vector<std::pair<int, int>>& blocks,
           const std::vector<int>& layout, const std::vector<std::pair<int, std::uint64_t>>& configs,
           double ext_weight, double exe_bonus, int layout_iters, bool depth_tiebreak) {
            std::vector<transpiler::Su4RouteResult> results(configs.size());
            {
                py::gil_scoped_release release;
                std::vector<std::function<void()>> tasks;
                tasks.reserve(configs.size());
                for (std::size_t i = 0; i < configs.size(); ++i) {
                    tasks.emplace_back([&, i] {
                        transpiler::Su4RouteOptions options;
                        options.depth_tiebreak = depth_tiebreak;
                        options.width = configs[i].first;
                        options.ext_weight = ext_weight;
                        options.exe_bonus = exe_bonus;
                        options.layout_iters = layout_iters;
                        options.seed = configs[i].second;
                        results[i] = transpiler::RouteSu4Blocks(adjacency, blocks, layout, options);
                    });
                }
                common::RunTasks(tasks, common::CompileThreads());
            }
            py::list out;
            for (auto& result : results) {
                out.append(py::make_tuple(result.paid, result.flips, std::move(result.layout), std::move(result.events),
                                          result.depth));
            }
            return out;
        },
        py::arg("adjacency"), py::arg("blocks"), py::arg("layout"), py::arg("configs"), py::arg("ext_weight") = 0.5,
        py::arg("exe_bonus") = 0.5, py::arg("layout_iters") = 1, py::arg("depth_tiebreak") = true,
        "_route_su4_blocks for each (width, seed) of configs, computed concurrently; list of its results.");

    // 全局束搜索路由（beam_routing.hpp）：每个起始布局各一个任务，互不相关（结果与线程数无关，
    // 按 layouts 的顺序返回 (swaps, layout, events)）
    m.def(
        "_route_beam_many",
        [](const std::vector<std::vector<int>>& adjacency, const std::vector<std::pair<int, int>>& blocks,
           const std::vector<std::vector<int>>& layouts, const std::vector<std::uint64_t>& seeds, int beam,
           int ext_depth, double gamma, double ext_weight, double exec_bonus, int layout_iters, std::int64_t max_swaps,
           double decay) {
            if (seeds.size() != layouts.size()) {
                throw std::invalid_argument("seeds and layouts must have the same length");
            }
            std::vector<transpiler::BeamRouteResult> results(layouts.size());
            {
                py::gil_scoped_release release;
                std::vector<std::function<void()>> tasks;
                tasks.reserve(layouts.size());
                for (std::size_t i = 0; i < layouts.size(); ++i) {
                    tasks.emplace_back([&, i] {
                        transpiler::BeamRouteOptions options;
                        options.beam = beam;
                        options.ext_depth = ext_depth;
                        options.gamma = gamma;
                        options.ext_weight = ext_weight;
                        options.exec_bonus = exec_bonus;
                        options.layout_iters = layout_iters;
                        options.seed = seeds[i];
                        options.max_swaps = max_swaps;
                        options.decay = decay;
                        results[i] = transpiler::RouteBeam(adjacency, blocks, layouts[i], options);
                    });
                }
                common::RunTasks(tasks, common::CompileThreads());
            }
            py::list out;
            for (auto& result : results) {
                out.append(py::make_tuple(result.swaps, std::move(result.layout), std::move(result.events)));
            }
            return out;
        },
        py::arg("adjacency"), py::arg("blocks"), py::arg("layouts"), py::arg("seeds"), py::arg("beam") = 128,
        py::arg("ext_depth") = 8, py::arg("gamma") = 0.7, py::arg("ext_weight") = 0.5, py::arg("exec_bonus") = 3.0,
        py::arg("layout_iters") = 2, py::arg("max_swaps") = 0, py::arg("decay") = 1.0,
        "Beam-search SWAP routing from each start layout, computed concurrently; list of (swaps, layout, events) "
        "(swaps = -1: over max_swaps).");

    // 路由后 CNOT-相位区域重综合的整轮流程（phase_poly.hpp）。输入线路按门给出 (种类, a, b, 提示, 编号)：
    // 种类 0 = 单比特门（编号指向 matrices / opaque），2 = cz（提示 0 表示来自 cx(a→b)），1 = h。
    // 返回 None（没有改进）或 ((种类, a, b, 编号, 角度) 各列, (regions, replaced, saved, rounds))，
    // 种类：0 原单比特门，1 h，2 cz，3 rz，4 x。
    m.def(
        "_phase_poly_run",
        [](const std::vector<int>& kinds, const std::vector<int>& qa, const std::vector<int>& qb,
           const std::vector<int>& hint, const std::vector<int>& index, const std::vector<std::array<double, 8>>& mats,
           const std::vector<std::uint8_t>& opaque, const std::vector<std::pair<int, int>>& extra_pairs,
           const std::vector<std::tuple<int, std::int64_t, bool>>& settings, int beam_width, int max_rounds) -> py::object {
            const std::size_t n = kinds.size();
            if (qa.size() != n || qb.size() != n || hint.size() != n || index.size() != n)
                throw std::invalid_argument("_phase_poly_run: column lengths differ");
            std::vector<transpiler::PhasePolyOp> ops(n);
            for (std::size_t i = 0; i < n; ++i) {
                ops[i].kind = kinds[i];
                ops[i].a = qa[i];
                ops[i].b = qb[i];
                ops[i].hint = hint[i];
                ops[i].index = index[i];
                if (kinds[i] != transpiler::kPPOrig && kinds[i] != transpiler::kPPH && kinds[i] != transpiler::kPPCz)
                    throw std::invalid_argument("_phase_poly_run: bad op kind");
            }
            std::vector<transpiler::PhasePolySetting> sets;
            for (const auto& [cap, budget, reverse] : settings)
                sets.push_back(transpiler::PhasePolySetting{cap, budget, reverse});
            transpiler::PhasePolyResult res;
            {
                py::gil_scoped_release release;
                res = transpiler::RunPhasePolynomial(ops, mats, opaque, extra_pairs, sets, beam_width, max_rounds);
            }
            const auto stats = py::make_tuple(res.stats.regions, res.stats.replaced, res.stats.saved, res.stats.rounds);
            if (!res.improved)
                return py::make_tuple(py::none(), stats);
            std::vector<int> ok, oa, ob, oi;
            std::vector<double> ang;
            ok.reserve(res.ops.size());
            for (const auto& op : res.ops) {
                ok.push_back(op.kind);
                oa.push_back(op.a);
                ob.push_back(op.b);
                oi.push_back(op.index);
                ang.push_back(op.angle);
            }
            return py::make_tuple(py::make_tuple(ok, oa, ob, oi, ang), stats);
        },
        py::arg("kinds"), py::arg("a"), py::arg("b"), py::arg("hint"), py::arg("index"), py::arg("matrices"),
        py::arg("opaque"), py::arg("extra_pairs"), py::arg("settings"), py::arg("beam_width"), py::arg("max_rounds"),
        "Post-routing CNOT-phase region resynthesis (all rounds); ((kinds, a, b, index, angle) or None, stats).");

    // 同上，问题带自由线与方法：(n, 有向边, 目标行, 必需奇偶, CX 上限, 自由线掩码, 束宽, 展开预算)。
    // 束宽 0：A*（n ≤ 6，最优，预算为展开数）；束宽 > 0：逐层束搜索（n ≤ 32，启发式）。
    // 按输入顺序返回 (CX 序列, 终态置换) 或 None
    m.def(
        "_cnot_phase_synth_batch",
        [](const std::vector<std::tuple<int, std::vector<std::pair<int, int>>, std::vector<std::uint32_t>,
                                        std::vector<std::uint32_t>, int, std::uint32_t, int, std::int64_t>>& problems) {
            std::vector<transpiler::CnotPhaseResult> results(problems.size());
            {
                py::gil_scoped_release release;
                std::vector<std::function<void()>> tasks;
                tasks.reserve(problems.size());
                for (std::size_t i = 0; i < problems.size(); ++i) {
                    tasks.emplace_back([&, i] {
                        const auto& [n, edges, rows, required, max_cx, free_mask, beam, budget] = problems[i];
                        if (beam > 0) {
                            results[i] =
                                transpiler::SynthesizeCnotPhaseBeam(n, edges, rows, required, max_cx, beam, free_mask);
                        } else {
                            results[i] =
                                transpiler::SynthesizeCnotPhase(n, edges, rows, required, max_cx, budget, free_mask);
                        }
                    });
                }
                common::RunTasks(tasks, common::CompileThreads());
            }
            py::list out;
            for (auto& r : results) {
                if (r.found) {
                    out.append(py::make_tuple(py::cast(std::move(r.moves)), py::cast(std::move(r.perm))));
                } else {
                    out.append(py::none());
                }
            }
            return out;
        },
        py::arg("problems"),
        "CNOT-phase synthesis for each problem (n, edges, target_rows, required, max_cx, free_mask, beam_width, "
        "node_budget): A* (optimal, n <= 6) when beam_width == 0, else layer-wise beam search (n <= 32); "
        "list of (moves, perm) or None.");

    // 受耦合约束的 CNOT-相位线路最优综合（cnot_phase_synth.hpp），一批互不相关的问题并行求解；
    // 每个问题 (n, 有向边, 目标行, 必需奇偶, CX 上限)，按输入顺序返回 CX 序列或 None
    m.def(
        "_cnot_phase_synth_many",
        [](const std::vector<std::tuple<int, std::vector<std::pair<int, int>>, std::vector<std::uint32_t>,
                                        std::vector<std::uint32_t>, int>>& problems,
           std::int64_t node_budget) {
            std::vector<transpiler::CnotPhaseResult> results(problems.size());
            {
                py::gil_scoped_release release;
                std::vector<std::function<void()>> tasks;
                tasks.reserve(problems.size());
                for (std::size_t i = 0; i < problems.size(); ++i) {
                    tasks.emplace_back([&, i] {
                        const auto& [n, edges, rows, required, max_cx] = problems[i];
                        results[i] = transpiler::SynthesizeCnotPhase(n, edges, rows, required, max_cx, node_budget);
                    });
                }
                common::RunTasks(tasks, common::CompileThreads());
            }
            py::list out;
            for (auto& r : results) {
                if (r.found) {
                    out.append(py::cast(std::move(r.moves)));
                } else {
                    out.append(py::none());
                }
            }
            return out;
        },
        py::arg("problems"), py::arg("node_budget") = 200000,
        "Minimum-CX CNOT-phase synthesis on directed edges for each problem (n, edges, target_rows, required, max_cx); "
        "list of [(control, target), ...] or None (no solution within max_cx or the node budget).");

    // 低伸缩初始布局（quspect.transpiler._embedding 的原生实现）
    m.def(
        "_dilation_layouts",
        [](int num_logical, const std::vector<std::tuple<int, int, std::int64_t>>& edges,
           const std::vector<std::vector<int>>& adjacency, const std::vector<std::vector<std::int64_t>>& distance,
           std::uint64_t seed, int count) {
            std::vector<transpiler::WeightedEdge> weighted;
            weighted.reserve(edges.size());
            for (const auto& [a, b, w] : edges) {
                weighted.push_back({a, b, w});
            }
            std::vector<transpiler::DilationLayout> found;
            {
                py::gil_scoped_release release;
                found = transpiler::DilationLayouts(num_logical, weighted, adjacency, distance, seed, count);
            }
            py::list out;
            for (auto& item : found) {
                out.append(py::make_tuple(item.cost, py::cast(std::move(item.layout))));
            }
            return out;
        },
        py::arg("num_logical"), py::arg("edges"), py::arg("adjacency"), py::arg("distance"), py::arg("seed"),
        py::arg("count"), "Low-dilation layouts of a weighted interaction graph; returns [(cost, layout)].");

    // 对易感知的深度调度（quspect.transpiler._schedule 的原生实现）
    m.def(
        "_depth_schedule",
        [](std::vector<int> offsets, std::vector<int> resources, std::vector<int> axes, std::vector<std::uint8_t> two,
           std::vector<std::uint8_t> counted, int num_qubits, std::optional<std::vector<int>> start) {
            transpiler::ScheduleProblem problem{std::move(offsets), std::move(resources), std::move(axes),
                                                std::move(two), std::move(counted)};
            std::vector<int> order;
            std::pair<int, int> before;
            std::pair<int, int> after;
            {
                py::gil_scoped_release release;
                if (start.has_value()) {
                    // 从给定顺序出发：在重排后的问题上调度，再映射回原编号
                    const transpiler::ScheduleProblem moved = transpiler::Permuted(problem, *start);
                    const std::vector<int> local = transpiler::BestDepthOrder(moved);
                    order.reserve(local.size());
                    for (int g : local) {
                        order.push_back((*start)[g]);
                    }
                    before = transpiler::ReportedDepths(problem, *start, num_qubits);
                } else {
                    order = transpiler::BestDepthOrder(problem);
                    std::vector<int> identity(order.size());
                    for (std::size_t g = 0; g < identity.size(); ++g) {
                        identity[g] = static_cast<int>(g);
                    }
                    before = transpiler::ReportedDepths(problem, identity, num_qubits);
                }
                after = transpiler::ReportedDepths(problem, order, num_qubits);
            }
            return py::make_tuple(order, py::make_tuple(before.first, before.second),
                                  py::make_tuple(after.first, after.second));
        },
        py::arg("offsets"), py::arg("resources"), py::arg("axes"), py::arg("two"), py::arg("counted"),
        py::arg("num_qubits"), py::arg("start") = py::none(),
        "Depth-minimizing order of commuting instructions (see quspect.transpiler._schedule); returns "
        "(order, (depth, 2q depth) of the start order, (depth, 2q depth) of the result).");

    // 两比特块层面的调度（quspect.transpiler._schedule._block_order 的原生实现）
    m.def(
        "_block_schedule",
        [](std::vector<int> offsets, std::vector<int> resources, std::vector<int> axes, std::vector<std::uint8_t> two,
           std::vector<std::uint8_t> counted, std::vector<int> qubit_offsets, std::vector<int> qubits,
           std::vector<int> kind, std::vector<int> matrix_offset,
           py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> matrices) {
            transpiler::ScheduleProblem problem{std::move(offsets), std::move(resources), std::move(axes),
                                                std::move(two), std::move(counted)};
            transpiler::BlockInput input;
            input.qubit_offsets = std::move(qubit_offsets);
            input.qubits = std::move(qubits);
            input.kind = std::move(kind);
            input.matrix_offset = std::move(matrix_offset);
            input.matrices.assign(matrices.data(), matrices.data() + matrices.size());
            std::vector<int> order;
            {
                py::gil_scoped_release release;
                order = transpiler::BlockOrder(problem, input);
            }
            return order;
        },
        py::arg("offsets"), py::arg("resources"), py::arg("axes"), py::arg("two"), py::arg("counted"),
        py::arg("qubit_offsets"), py::arg("qubits"), py::arg("kind"), py::arg("matrix_offset"), py::arg("matrices"),
        "Instruction order after scheduling whole two-qubit blocks (empty: not applicable / unchanged).");

    // CPython random.Random 的复刻（供测试比对）：从 getstate()[1] 抽取 count 个 getrandbits(k) / _randbelow(n)。
    m.def(
        "_cpython_random_draws",
        [](const std::vector<std::uint32_t>& rng_state, int k, std::uint64_t below, int count) {
            auto rng = transpiler::CPythonRandom::FromState(rng_state);
            std::vector<std::uint64_t> out;
            out.reserve(count > 0 ? count : 0);
            for (int i = 0; i < count; ++i) {
                out.push_back(below > 0 ? rng.RandBelow(below) : rng.GetRandBits(k));
            }
            return py::make_tuple(out, py::tuple(py::cast(rng.State())));
        },
        py::arg("rng_state"), py::arg("k"), py::arg("below"), py::arg("count"),
        "Draw count getrandbits(k) (below == 0) or _randbelow(below) values; returns (values, new_state).");

    m.def(
        "_cpython_random_seed_state",
        [](std::uint64_t seed) { return py::tuple(py::cast(transpiler::CPythonRandom(seed).State())); },
        py::arg("seed"), "State words (getstate()[1]) of random.Random(seed).");

    // ---- 单比特门最小化的标量内核（quspect.transpiler._gauge 的原生实现）----
    py::class_<transpiler::LengthModel>(m, "_GaugeLengthModel", py::module_local())
        .def(py::init([](bool h, bool x, bool sx, bool sxdg, const ComplexArray& named) {
                 transpiler::LengthModel model;
                 model.h = h;
                 model.x = x;
                 model.sx = sx;
                 model.sxdg = sxdg;
                 model.named = ToMatrices(named);
                 return model;
             }),
             py::arg("h"), py::arg("x"), py::arg("sx"), py::arg("sxdg"), py::arg("named"));

    py::class_<transpiler::GaugeBeamSearch>(m, "_GaugeBeamSearch", py::module_local())
        .def(py::init<int, int, int, int, transpiler::LengthModel>(), py::arg("num_slots"), py::arg("beam"),
             py::arg("gate_weight"), py::arg("run_weight"), py::arg("model"))
        .def("reset", &transpiler::GaugeBeamSearch::Reset, py::arg("start_phi"))
        .def(
            "prepare",
            [](transpiler::GaugeBeamSearch& self, const std::vector<int>& slots, const std::vector<int>& rids,
               const std::vector<std::uint8_t>& fp, const std::vector<std::uint8_t>& fn, const std::vector<int>& ax_in,
               const std::vector<int>& ax_out, const std::vector<std::uint8_t>& last,
               const std::vector<double>& phi_fixed, const std::vector<int>& run_length,
               const std::vector<std::uint8_t>& base_changed) {
                self.Prepare(slots, rids, fp, fn, ax_in, ax_out, last, phi_fixed, run_length, base_changed);
                return py::make_tuple(ToArray(self.ComputeRow()), ToArray(self.RotateIn()),
                                      FromMatrices(self.RotateInMatrices()), ToArray(self.NeedLeading()));
            },
            py::arg("slots"), py::arg("rids"), py::arg("fp"), py::arg("fn"), py::arg("ax_in"), py::arg("ax_out"),
            py::arg("last"), py::arg("phi_fixed"), py::arg("run_length"), py::arg("base_changed"),
            "Returns (compute_row, rotate_in, rotate_in_matrices, need_leading).")
        .def(
            "options",
            [](transpiler::GaugeBeamSearch& self, const ComplexArray& z) {
                self.Options(ToMatrices(z));
                return py::make_tuple(ToArray(self.OptionRows()), FromMatrices(self.OptionMatrices()));
            },
            py::arg("z"), "Returns (option_rows, option_matrices).")
        .def(
            "finish",
            [](transpiler::GaugeBeamSearch& self, const ComplexArray& mats, const ComplexArray& option_mats,
               bool insert_memo) { self.Finish(ToMatrices(mats), ToMatrices(option_mats), insert_memo); },
            py::arg("mats"), py::arg("option_mats"), py::arg("insert_memo"))
        .def("commit_slots", &transpiler::GaugeBeamSearch::CommitSlots, py::arg("slots"))
        .def("set_wires", &transpiler::GaugeBeamSearch::SetWires, py::arg("rids"), py::arg("axes"),
             py::arg("run_length"))
        .def(
            "prepare_layer",
            [](transpiler::GaugeBeamSearch& self, int layer) {
                self.PrepareLayer(layer);
                return py::make_tuple(ToArray(self.ComputeRid()), ToArray(self.ComputeAxOut()),
                                      ToArray(self.RotateIn()), FromMatrices(self.RotateInMatrices()),
                                      ToArray(self.NeedLeading()));
            },
            py::arg("layer"), "Returns (compute_rid, compute_ax_out, rotate_in, rotate_in_matrices, need_leading).")
        .def(
            "backtrack",
            [](const transpiler::GaugeBeamSearch& self, int slot) {
                std::vector<double> angles;
                std::vector<int> costs;
                const std::int64_t total = self.Backtrack(slot, &angles, &costs);
                return py::make_tuple(total, angles, costs);
            },
            py::arg("slot"))
        .def(
            "memo_get",
            [](const transpiler::GaugeBeamSearch& self, int rid, bool fp, bool fn, std::int64_t key, bool last,
               double phi_e) -> py::object {
                const auto* opts = self.MemoGet(MemoKeyOf(rid, fp, fn, key, last, phi_e));
                if (opts == nullptr) {
                    return py::none();
                }
                py::list out;
                for (std::size_t j = 0; j < opts->phis.size(); ++j) {
                    out.append(py::make_tuple(opts->phis[j], opts->costs[j]));
                }
                return out;
            },
            py::arg("rid"), py::arg("fp"), py::arg("fn"), py::arg("key"), py::arg("last"), py::arg("phi_e"),
            "Options [(phi, cost), ...] stored under the memo key, or None.")
        .def(
            "memo_put",
            [](transpiler::GaugeBeamSearch& self, int rid, bool fp, bool fn, std::int64_t key, bool last, double phi_e,
               double phi_in, const std::vector<std::pair<double, int>>& options) {
                transpiler::GaugeBeamSearch::MemoOptions opts;
                opts.phi_in = phi_in;
                for (const auto& [phi, cost] : options) {
                    opts.phis.push_back(phi);
                    opts.costs.push_back(cost);
                }
                self.MemoPut(MemoKeyOf(rid, fp, fn, key, last, phi_e), std::move(opts));
            },
            py::arg("rid"), py::arg("fp"), py::arg("fn"), py::arg("key"), py::arg("last"), py::arg("phi_e"),
            py::arg("phi_in"), py::arg("options"), "Store options under the memo key unless it is present.");

    m.def(
        "_cx_orientation",
        [](const transpiler::LengthModel& model, int num_gates, std::vector<std::vector<int>> wire_gates,
           std::vector<std::vector<std::uint8_t>> wire_is_control, const ComplexArray& runs, int beam, int gate_weight,
           int run_weight, int iterations) {
            transpiler::OrientationProblem problem;
            problem.num_gates = num_gates;
            const auto mats = ToMatrices(runs);
            std::size_t pos = 0;
            for (const auto& gates : wire_gates) {
                const std::size_t count = gates.size() + 1;
                if (pos + count > mats.size()) {
                    throw std::invalid_argument("runs: need len(gates) + 1 matrices per wire");
                }
                problem.wire_runs.emplace_back(mats.begin() + static_cast<std::ptrdiff_t>(pos),
                                               mats.begin() + static_cast<std::ptrdiff_t>(pos + count));
                pos += count;
            }
            if (pos != mats.size()) {
                throw std::invalid_argument("runs: need len(gates) + 1 matrices per wire");
            }
            problem.wire_gates = std::move(wire_gates);
            problem.wire_is_control = std::move(wire_is_control);
            transpiler::OrientationOptions options;
            options.beam = beam;
            options.gate_weight = gate_weight;
            options.run_weight = run_weight;
            options.iterations = iterations;
            transpiler::OrientationResult result;
            {
                py::gil_scoped_release release;
                result = transpiler::SolveCxOrientation(problem, model, options);
            }
            return py::make_tuple(result.flip, result.energy, result.initial_energy, result.dual, result.iterations);
        },
        py::arg("model"), py::arg("num_gates"), py::arg("wire_gates"), py::arg("wire_is_control"), py::arg("runs"),
        py::arg("beam"), py::arg("gate_weight"), py::arg("run_weight"), py::arg("iterations"),
        "Global CX orientation by dual decomposition: (flip, energy, initial_energy, dual, iterations).");

    m.def(
        "_euler_lengths",
        [](const ComplexArray& mats, int basis, double atol) {
            const auto ms = ToMatrices(mats);
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(ms.size()));
            auto o = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < ms.size(); ++i) {
                const auto r = transpiler::EulerLength(ms[i], basis, atol);
                o(static_cast<py::ssize_t>(i)) = r ? *r : -1;
            }
            return out;
        },
        py::arg("mats"), py::arg("basis"), py::arg("atol"),
        "len(euler_ops(m, 0, basis=('zxz', 'zsx')[basis], atol=atol)) of each matrix (-1: use Python).");

    m.def(
        "_gauge_lengths",
        [](const transpiler::LengthModel& model, const ComplexArray& mats) {
            const auto ms = ToMatrices(mats);
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(ms.size()));
            auto o = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < ms.size(); ++i) {
                const auto r = transpiler::FastLength(ms[i], model);
                o(static_cast<py::ssize_t>(i)) = r ? *r : -1;
            }
            return out;
        },
        py::arg("model"), py::arg("mats"),
        "_LengthModel._fast of each 2x2 matrix of an (N, 2, 2) stack (-1: not reproducible, use Python).");

    m.def(
        "_gauge_leading_angles",
        [](const ComplexArray& mats) {
            const auto ms = ToMatrices(mats);
            const auto n = static_cast<py::ssize_t>(ms.size());
            py::array_t<std::int64_t> counts(n);
            py::array_t<double> angles({n, static_cast<py::ssize_t>(4)});
            auto c = counts.mutable_unchecked<1>();
            auto a = angles.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < n; ++i) {
                const auto r = transpiler::LeadingAnglesInZFrame(ms[static_cast<std::size_t>(i)]);
                c(i) = r ? r->count : 0;
                for (int j = 0; j < 4; ++j) {
                    a(i, j) = r ? r->angles[j] : 0.0;
                }
            }
            return py::make_tuple(counts, angles);
        },
        py::arg("mats"), "_leading_angles in the Z frame for each matrix: (counts, angles); count 0: use Python.");

    m.def(
        "_gauge_rotations",
        [](const std::vector<int>& axes, const std::vector<double>& phis) {
            if (axes.size() != phis.size()) {
                throw std::invalid_argument("axes and phis must have the same length");
            }
            const auto n = static_cast<py::ssize_t>(axes.size());
            py::array_t<std::complex<double>> out({n, static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
            auto o = out.mutable_unchecked<3>();
            for (py::ssize_t i = 0; i < n; ++i) {
                const int axis = axes[static_cast<std::size_t>(i)];
                if (axis < 0 || axis > 2) {
                    throw std::invalid_argument("axis must be 0 (z), 1 (x) or 2 (y)");
                }
                const auto r = transpiler::AxisRotation(axis, phis[static_cast<std::size_t>(i)]);
                for (int j = 0; j < 4; ++j) {
                    o(i, j / 2, j % 2) = std::complex<double>(r[j].re, r[j].im);
                }
            }
            return out;
        },
        py::arg("axes"), py::arg("phis"), "_rot(axis, phi) for each pair (axis 0 = z, 1 = x, 2 = y).");

    m.def(
        "_gauge_matmul",
        [](const ComplexArray& a, const ComplexArray& b) {
            // a @ b for (2, 2) / (N, 2, 2) stacks (broadcast like numpy), on the deterministic numerics layer
            auto count = [](const ComplexArray& x) -> py::ssize_t {
                if (x.ndim() == 2 && x.shape(0) == 2 && x.shape(1) == 2) {
                    return -1;
                }
                if (x.ndim() == 3 && x.shape(1) == 2 && x.shape(2) == 2) {
                    return x.shape(0);
                }
                throw std::invalid_argument("expected a (2, 2) matrix or an (N, 2, 2) stack");
            };
            const py::ssize_t na = count(a);
            const py::ssize_t nb = count(b);
            if (na >= 0 && nb >= 0 && na != nb) {
                throw std::invalid_argument("stack sizes differ");
            }
            const py::ssize_t n = std::max(na, nb);
            // numpy 缓冲区与 Cx 之间按字节拷贝（不做类型双关）
            const auto copy = [](const ComplexArray& x) {
                std::vector<detmath::Cx> v(static_cast<std::size_t>(x.size()));
                std::memcpy(v.data(), x.data(), v.size() * sizeof(detmath::Cx));
                return v;
            };
            const std::vector<detmath::Cx> va = copy(a);
            const std::vector<detmath::Cx> vb = copy(b);
            const std::size_t stacks = static_cast<std::size_t>(n < 0 ? 1 : n);
            std::vector<detmath::Cx> vo(4 * stacks);
            for (std::size_t i = 0; i < stacks; ++i) {
                detmath::MatMul(2, va.data() + (na < 0 ? 0 : 4 * i), vb.data() + (nb < 0 ? 0 : 4 * i),
                                vo.data() + 4 * i);
            }
            py::array_t<std::complex<double>> out =
                n < 0 ? py::array_t<std::complex<double>>({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)})
                      : py::array_t<std::complex<double>>(
                            {n, static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
            std::memcpy(out.mutable_data(), vo.data(), vo.size() * sizeof(detmath::Cx));
            return out;
        },
        py::arg("a"), py::arg("b"),
        "a @ b for 2x2 matrices or (N, 2, 2) stacks (deterministic products, bit-identical on all platforms).");

    // ---- 精度枚举 ----
    py::enum_<executor::PrecisionKind>(m, "PrecisionKind", py::module_local())
        .value("FLOAT32", executor::PrecisionKind::kFloat32)
        .value("FLOAT64", executor::PrecisionKind::kFloat64)
        .export_values();

    // ---- 后端类型 ----
    py::enum_<executor::BackendKind>(m, "BackendKind", py::module_local())
        .value("SIMULATOR", executor::BackendKind::kSimulator)
        .value("HARDWARE", executor::BackendKind::kHardware)
        .export_values();

    // ---- ExecutionRequest ----
    py::class_<executor::ExecutionRequest>(m, "ExecutionRequest", py::module_local())
        .def(py::init<>())
        .def_readwrite("shots", &executor::ExecutionRequest::shots)
        .def_readwrite("return_statevector", &executor::ExecutionRequest::return_statevector)
        .def_readwrite("return_expectations", &executor::ExecutionRequest::return_expectations)
        .def_readwrite("return_samples", &executor::ExecutionRequest::return_samples);

    // ---- ExecutionResult ----
    py::class_<executor::ExecutionResult>(m, "ExecutionResult", py::module_local())
        .def(py::init<>())
        .def_readonly("statevector_f64", &executor::ExecutionResult::statevector_f64)
        .def_readonly("statevector_f32", &executor::ExecutionResult::statevector_f32)
        .def_readonly("counts", &executor::ExecutionResult::counts)
        .def_readonly("packed_samples", &executor::ExecutionResult::packed_samples)
        .def_readonly("expectations_real", &executor::ExecutionResult::expectations_real)
        .def_readonly("expectations_complex", &executor::ExecutionResult::expectations_complex)
        .def_readonly("metadata", &executor::ExecutionResult::metadata);

    // ---- Internal objective-plan result and prepared program ----
    py::class_<executor::GradientResult>(m, "_AdjointEvaluation", py::module_local())
        .def_readonly("parameter_names", &executor::GradientResult::parameter_names)
        .def_readonly("observable_names", &executor::GradientResult::observable_labels)
        .def_readonly("values", &executor::GradientResult::expectations)
        .def_readonly("jacobian", &executor::GradientResult::gradients);

    py::class_<AdjointPlan>(m, "_AdjointPlan", py::module_local())
        .def(py::init<const std::string&, const std::string&>(), py::arg("qsal_json"), py::arg("precision") = "double")
        .def_property_readonly("parameter_names", &AdjointPlan::ParameterNames)
        .def_property_readonly("observable_names", &AdjointPlan::ObservableNames)
        .def("evaluate", &AdjointPlan::Evaluate, py::arg("parameter_values"), py::call_guard<py::gil_scoped_release>());

    // ---- QSALProgram types ----
    py::class_<runtime::ParameterExpression>(m, "ParameterExpression", py::module_local())
        .def(py::init<>())
        .def_readwrite("constant", &runtime::ParameterExpression::constant)
        .def_readwrite("terms", &runtime::ParameterExpression::terms)
        .def_property_readonly("is_symbolic", &runtime::ParameterExpression::IsSymbolic);

    py::class_<runtime::GateOperation>(m, "GateOperation", py::module_local())
        .def(py::init<>())
        .def_readwrite("name", &runtime::GateOperation::name)
        .def_readwrite("qubits", &runtime::GateOperation::qubits)
        .def_readwrite("params_f64", &runtime::GateOperation::params_f64)
        .def_readwrite("params_f32", &runtime::GateOperation::params_f32)
        .def_readwrite("parameter_expressions", &runtime::GateOperation::parameter_expressions)
        .def_readwrite("control_mask", &runtime::GateOperation::control_mask)
        .def_readwrite("control_value", &runtime::GateOperation::control_value);

    py::class_<runtime::MeasureOperation>(m, "MeasureOperation", py::module_local())
        .def(py::init<>())
        .def_readwrite("qubits", &runtime::MeasureOperation::qubits)
        .def_readwrite("clbits", &runtime::MeasureOperation::clbits);

    py::class_<runtime::ChannelOperation>(m, "ChannelOperation", py::module_local())
        .def(py::init<>())
        .def_readwrite("channel_kind", &runtime::ChannelOperation::channel_kind)
        .def_readwrite("qubits", &runtime::ChannelOperation::qubits)
        .def_readwrite("params_f64", &runtime::ChannelOperation::params_f64)
        .def_readwrite("params_f32", &runtime::ChannelOperation::params_f32)
        .def_readonly("use_pauli_fast_path", &runtime::ChannelOperation::use_pauli_fast_path)
        .def_readwrite("num_kraus_ops", &runtime::ChannelOperation::num_kraus_ops)
        .def_readwrite("kraus_operators_f64", &runtime::ChannelOperation::kraus_operators_f64)
        .def_readwrite("kraus_operators_f32", &runtime::ChannelOperation::kraus_operators_f32);

    py::class_<runtime::ObservableTerm>(m, "ObservableTerm", py::module_local())
        .def(py::init<>())
        .def_readwrite("label", &runtime::ObservableTerm::label)
        .def_readwrite("coefficient", &runtime::ObservableTerm::coefficient)
        .def_readwrite("paulis", &runtime::ObservableTerm::paulis);

    py::class_<runtime::QSALProgram>(m, "QSALProgram", py::module_local())
        .def(py::init<>())
        .def_readwrite("num_qubits", &runtime::QSALProgram::num_qubits)
        .def_readwrite("num_clbits", &runtime::QSALProgram::num_clbits)
        .def_readwrite("observables", &runtime::QSALProgram::observables)
        .def_readwrite("parameter_names", &runtime::QSALProgram::parameter_names);

    // ---- SimulationExecutor ----
    py::class_<executor::SimulationExecutor>(m, "SimulationExecutor", py::module_local())
        .def(py::init([](const std::string& precision, uint64_t seed) {
                 executor::SimulationExecutor::Options opts;
                 opts.precision = ParsePrecision(precision);
                 opts.seed = seed;
                 return std::make_unique<executor::SimulationExecutor>(opts);
             }),
             py::arg("precision") = "double", py::arg("seed") = 0)
        .def("run", &executor::SimulationExecutor::Run, py::call_guard<py::gil_scoped_release>());

    // ---- HardwareExecutor (占位) ----
    py::class_<executor::HardwareExecutor>(m, "HardwareExecutor", py::module_local())
        .def(py::init<>())
        .def("run", &executor::HardwareExecutor::Run);

    // ---- QSALDecoder ----
    py::class_<ir::QSALDecoder>(m, "QSALDecoder", py::module_local())
        .def(py::init<>())
        .def("decode", &ir::QSALDecoder::Decode)
        .def("last_error", &ir::QSALDecoder::LastError);

    // ---- Validator ----
    py::class_<ir::Validator>(m, "Validator", py::module_local())
        .def_static("validate", &ir::Validator::Validate)
        .def_static("check_index_bounds", &ir::Validator::CheckIndexBounds)
        .def_static("check_gate_names", &ir::Validator::CheckGateNames)
        .def_static("check_parameter_count", &ir::Validator::CheckParameterCount)
        .def_static("check_measure_consistency", &ir::Validator::CheckMeasureConsistency)
        .def_static("check_channels", &ir::Validator::CheckChannels);

    // ---- ProgramBuilder ----
    py::class_<ir::ProgramBuilder>(m, "ProgramBuilder", py::module_local())
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t>(), py::arg("num_qubits"), py::arg("num_clbits") = 0)
        .def("set_num_qubits", &ir::ProgramBuilder::SetNumQubits, py::return_value_policy::reference)
        .def("set_num_clbits", &ir::ProgramBuilder::SetNumClbits, py::return_value_policy::reference)
        .def("add_gate", &ir::ProgramBuilder::AddGate, py::return_value_policy::reference)
        .def("add_barrier", &ir::ProgramBuilder::AddBarrier, py::return_value_policy::reference)
        .def("add_measure", &ir::ProgramBuilder::AddMeasure, py::return_value_policy::reference)
        .def("add_observable", &ir::ProgramBuilder::AddObservable, py::return_value_policy::reference)
        .def("build", &ir::ProgramBuilder::Build);

    // ---- 便捷函数：decode_qsal — 解码 QSAL JSON 并运行 ----
    m.def(
        "decode_qsal",
        [](const std::string& qsal_json) {
            ir::QSALDecoder decoder;
            auto prog = decoder.Decode(qsal_json);
            if (!prog) {
                throw std::runtime_error("QSAL decode error: " + decoder.LastError());
            }
            return prog;
        },
        py::arg("qsal_json"), "Decode a QSAL JSON string into a QSALProgram. Throws on decode error.");

    // ---- 便捷函数：decode_qsal_and_run — 一站式 decode + validate + run ----
    m.def(
        "decode_qsal_and_run",
        [](const std::string& qsal_json, const std::string& precision, uint64_t seed, uint64_t shots,
           bool return_statevector, bool return_expectations, bool return_samples) {
            // Step 1: Decode
            ir::QSALDecoder decoder;
            auto prog = decoder.Decode(qsal_json);
            if (!prog) {
                throw std::runtime_error("QSAL decode error: " + decoder.LastError());
            }

            // Step 2: Validate
            auto errors = ir::Validator::Validate(*prog);
            if (!errors.empty()) {
                std::string msg = "QSAL validation error(s):";
                for (const auto& e : errors) {
                    msg += "\n  - " + e;
                }
                throw std::runtime_error(msg);
            }

            // Step 3: Execute
            executor::SimulationExecutor::Options opts;
            opts.precision = ParsePrecision(precision);
            opts.seed = seed;
            executor::SimulationExecutor exec(opts);

            executor::ExecutionRequest req;
            req.shots = shots;
            req.seed = seed;
            req.return_statevector = return_statevector;
            req.return_expectations = return_expectations;
            req.return_samples = return_samples;

            py::gil_scoped_release release;
            return exec.Run(*prog, req);
        },
        py::arg("qsal_json"), py::arg("precision") = "double", py::arg("seed") = 0, py::arg("shots") = 0,
        py::arg("return_statevector") = false, py::arg("return_expectations") = false,
        py::arg("return_samples") = false, "Decode QSAL JSON, validate, and run in one call. Returns ExecutionResult.");

    // ---- DensityMatrix (header-only, explicitly instantiated) ----
    py::class_<runtime::DensityMatrix<double>>(m, "DensityMatrix64", py::module_local())
        .def(py::init<uint32_t>(), py::arg("num_qubits"))
        .def_property_readonly("num_qubits", &runtime::DensityMatrix<double>::num_qubits)
        .def_property_readonly("dim", &runtime::DensityMatrix<double>::dim)
        .def("trace", &runtime::DensityMatrix<double>::Trace)
        .def("normalize", &runtime::DensityMatrix<double>::Normalize)
        .def("set_zero", &runtime::DensityMatrix<double>::SetZero)
        .def("set_pure_zero", &runtime::DensityMatrix<double>::SetPureZero)
        .def("apply_unitary", &_dm_apply_unitary_py)
        .def("apply_kraus_channel", &_dm_apply_kraus_py)
        .def("data", &_dm_data_py);

    py::class_<runtime::DensityMatrix<float>>(m, "DensityMatrix32", py::module_local())
        .def(py::init<uint32_t>(), py::arg("num_qubits"))
        .def_property_readonly("num_qubits", &runtime::DensityMatrix<float>::num_qubits)
        .def_property_readonly("dim", &runtime::DensityMatrix<float>::dim)
        .def("trace", &runtime::DensityMatrix<float>::Trace)
        .def("normalize", &runtime::DensityMatrix<float>::Normalize)
        .def("set_zero", &runtime::DensityMatrix<float>::SetZero)
        .def("set_pure_zero", &runtime::DensityMatrix<float>::SetPureZero);

    // ---- Noise kernel functions ----
    // apply_dense_unitary_f64: apply a k-qubit dense matrix to a state vector.
    m.def("apply_dense_unitary",
          [](const py::object& state_obj, const py::object& matrix_obj, const py::object& qubits_obj) {
              auto state = py::array_t<std::complex<double>, py::array::c_style>::ensure(state_obj);
              auto matrix = py::array_t<std::complex<double>, py::array::c_style>::ensure(matrix_obj);
              auto qubits = py::array_t<uint32_t, py::array::c_style>::ensure(qubits_obj);
              if (!state || !matrix || !qubits) {
                  throw std::runtime_error("Invalid array arguments");
              }
              uint64_t dim = static_cast<uint64_t>(state.size());
              std::complex<double>* state_data = state.mutable_data();
              const std::complex<double>* matrix_data = matrix.data();
              const uint32_t* qubit_data = qubits.data();
              const auto num_qubits = static_cast<uint32_t>(qubits.size());
              py::gil_scoped_release release;
              kernels::apply_dense_unitary<double>(state_data, dim, qubit_data, num_qubits, matrix_data);
          });

    // apply_kraus_mc: Monte Carlo Kraus selection on a state vector.
    m.def("apply_kraus_mc", &apply_kraus_mc_py, py::arg("state"), py::arg("kraus_ops"), py::arg("num_ops"),
          py::arg("qubits"), py::arg("seed"));
}
