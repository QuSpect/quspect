// OpenMP 并行内核测试
//
// 对每个并行化的内核，分别用 1 个线程和 kParallelThreads 个线程运行，比较结果。
// 状态向量维度同时覆盖阈值以下（单线程路径）和阈值以上（并行路径）。
//
// 门内核按元素独立计算，归约使用与线程数无关的 BlockedReduce，因此两次结果应逐位一致；
// 这里仍使用极小的相对容差，避免把实现细节（例如编译器对不同实例的 FMA 收缩）写死进测试。
// 未启用 OpenMP 时 SetNumThreads 为空操作，测试依然成立。

#include <cmath>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/common/parallel.hpp"
#include "quspect_native/executor/executor.hpp"
#include "quspect_native/executor/gradient_executor.hpp"
#include "quspect_native/ir/program_builder.hpp"
#include "quspect_native/kernels/dispatch_table.hpp"
#include "quspect_native/kernels/noise_kernels.hpp"
#include "quspect_native/optimization/ising_solver.hpp"
#include "quspect_native/runtime/program.hpp"
#include "quspect_native/runtime/state_vector.hpp"

namespace quspect::native::kernels::scalar {

// scalar 内核在两个静态库变体中都存在，用作 AVX2 内核的参照实现。
template <typename FP>
void apply_z(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_single_qubit_gate(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_cz(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);
template <typename FP>
void apply_dense_gate(std::complex<FP>*, uint64_t, const uint32_t*, uint32_t, const std::complex<FP>*);
template <typename FP>
std::complex<FP> compute_pauli_expectation(const std::complex<FP>*, uint64_t, uint64_t, uint64_t, uint32_t, int);

}  // namespace quspect::native::kernels::scalar

namespace {

using quspect::native::common::BlockedReduce;
using quspect::native::common::GetNumThreads;
using quspect::native::common::kParallelMinDim;
using quspect::native::common::kReductionBlockSize;
using quspect::native::common::OpenMPEnabled;
using quspect::native::common::ParallelForRange;
using quspect::native::common::SetNumThreads;
using quspect::native::common::ThreadsForDim;

namespace kernels = quspect::native::kernels;
namespace runtime = quspect::native::runtime;
namespace executor = quspect::native::executor;
namespace optimization = quspect::native::optimization;
namespace ir = quspect::native::ir;

constexpr int kParallelThreads = 4;
constexpr uint32_t kSmallQubits = 10;  // 2^10 < kParallelMinDim：串行路径
constexpr uint32_t kLargeQubits = 16;  // 2^16 = 4 * kParallelMinDim：4 线程并行路径

static_assert((uint64_t{1} << kSmallQubits) < kParallelMinDim, "small state must stay below the threshold");
static_assert((uint64_t{1} << kLargeQubits) >= kParallelMinDim * kParallelThreads,
              "large state must use all kParallelThreads threads");

/// 在作用域内设置线程数，结束时恢复原值
class ScopedNumThreads {
 public:
    explicit ScopedNumThreads(int num_threads) : previous_(GetNumThreads()) {
        SetNumThreads(num_threads);
    }
    ~ScopedNumThreads() {
        SetNumThreads(previous_);
    }
    ScopedNumThreads(const ScopedNumThreads&) = delete;
    ScopedNumThreads& operator=(const ScopedNumThreads&) = delete;

 private:
    int previous_;
};

template <typename FP>
std::vector<std::complex<FP>> RandomState(uint32_t num_qubits, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::complex<FP>> state(uint64_t{1} << num_qubits);
    double norm = 0.0;
    for (auto& amplitude : state) {
        const double re = normal(rng);
        const double im = normal(rng);
        amplitude = std::complex<FP>(static_cast<FP>(re), static_cast<FP>(im));
        norm += re * re + im * im;
    }
    const FP scale = static_cast<FP>(1.0 / std::sqrt(norm));
    for (auto& amplitude : state) {
        amplitude *= scale;
    }
    return state;
}

template <typename FP>
std::vector<std::complex<FP>> RandomMatrix(uint32_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<std::complex<FP>> matrix(static_cast<std::size_t>(dim) * dim);
    for (auto& value : matrix) {
        value = std::complex<FP>(static_cast<FP>(uniform(rng)), static_cast<FP>(uniform(rng)));
    }
    return matrix;
}

template <typename FP>
double Tolerance() {
    return std::is_same_v<FP, float> ? 1e-6 : 1e-13;
}

template <typename FP>
void ExpectStatesClose(const std::vector<std::complex<FP>>& expected, const std::vector<std::complex<FP>>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    double max_diff = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(expected[i] - actual[i])));
    }
    EXPECT_LE(max_diff, Tolerance<FP>());
}

void ExpectRelativeClose(double expected, double actual, double rel_tol) {
    EXPECT_LE(std::abs(expected - actual), rel_tol * std::max(1.0, std::abs(expected)))
        << "expected " << expected << " actual " << actual;
}

/// 用 1 线程与 kParallelThreads 线程分别对同一初始状态执行 op，比较结果状态
template <typename FP, typename Op>
void CompareThreadCounts(uint32_t num_qubits, Op op) {
    const auto initial = RandomState<FP>(num_qubits, 1234 + num_qubits);
    auto serial = initial;
    auto parallel = initial;
    {
        ScopedNumThreads threads(1);
        op(serial.data(), static_cast<uint64_t>(serial.size()));
    }
    {
        ScopedNumThreads threads(kParallelThreads);
        op(parallel.data(), static_cast<uint64_t>(parallel.size()));
    }
    ExpectStatesClose(serial, parallel);
    // 必须真的改变了状态，否则比较没有意义
    EXPECT_NE(initial, serial);
}

template <typename FP>
const kernels::KernelDispatchTable<FP>& Dispatch() {
    if constexpr (std::is_same_v<FP, float>) {
        return kernels::GetDispatchF32();
    } else {
        return kernels::GetDispatchF64();
    }
}

// ============================================================
// 并行辅助工具
// ============================================================

TEST(ParallelHelperTest, SetAndGetNumThreads) {
    const int original = GetNumThreads();
    EXPECT_GE(original, 1);

    SetNumThreads(3);
    EXPECT_EQ(GetNumThreads(), OpenMPEnabled() ? 3 : 1);
    SetNumThreads(1);
    EXPECT_EQ(GetNumThreads(), 1);

    SetNumThreads(original);
    EXPECT_EQ(GetNumThreads(), original);
}

TEST(ParallelHelperTest, RejectsNonPositiveThreadCounts) {
    const int original = GetNumThreads();
    EXPECT_THROW(SetNumThreads(0), std::invalid_argument);
    EXPECT_THROW(SetNumThreads(-2), std::invalid_argument);
    EXPECT_EQ(GetNumThreads(), original);
}

TEST(ParallelHelperTest, ThreadsForDimRespectsThreshold) {
    ScopedNumThreads threads(kParallelThreads);
    EXPECT_EQ(ThreadsForDim(16), 1);
    EXPECT_EQ(ThreadsForDim(kParallelMinDim), 1);
    EXPECT_EQ(ThreadsForDim(2 * kParallelMinDim - 1), 1);
    // 每个线程至少 kParallelMinDim 个振幅
    EXPECT_EQ(ThreadsForDim(2 * kParallelMinDim), OpenMPEnabled() ? 2 : 1);
    EXPECT_EQ(ThreadsForDim(kParallelMinDim * kParallelThreads), GetNumThreads());
    EXPECT_EQ(ThreadsForDim(kParallelMinDim * 1024), GetNumThreads());
}

TEST(ParallelHelperTest, ParallelForRangeCoversEveryIndexOnce) {
    for (const int64_t count : {int64_t{0}, int64_t{1}, int64_t{2}, int64_t{7}, int64_t{1000}, int64_t{100003}}) {
        for (const int num_threads : {1, 2, 3, 4, 7, 64}) {
            std::vector<int> hits(static_cast<std::size_t>(count), 0);
            ParallelForRange(count, num_threads, [&](int64_t begin, int64_t end) {
                ASSERT_LE(0, begin);
                ASSERT_LT(begin, end);
                ASSERT_LE(end, count);
                for (int64_t i = begin; i < end; ++i) {
                    ++hits[static_cast<std::size_t>(i)];
                }
            });
            EXPECT_TRUE(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }))
                << "count=" << count << " threads=" << num_threads;
        }
    }
}

TEST(ParallelHelperTest, BlockedReduceIsIndependentOfThreadCount) {
    const int64_t count = 5 * kReductionBlockSize + 123;
    std::vector<double> values(static_cast<std::size_t>(count));
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (auto& v : values) {
        v = uniform(rng);
    }
    auto body = [&](int64_t begin, int64_t end, double* acc) {
        for (int64_t i = begin; i < end; ++i) {
            acc[0] += values[static_cast<std::size_t>(i)];
            acc[1] += values[static_cast<std::size_t>(i)] * values[static_cast<std::size_t>(i)];
        }
    };
    const auto serial = BlockedReduce<2>(count, 1, body);
    for (const int num_threads : {2, 3, 4, 8}) {
        const auto parallel = BlockedReduce<2>(count, num_threads, body);
        EXPECT_EQ(serial[0], parallel[0]);  // 逐位一致
        EXPECT_EQ(serial[1], parallel[1]);
    }
    double reference = 0.0;
    for (double v : values) {
        reference += v;
    }
    ExpectRelativeClose(reference, serial[0], 1e-12);
    EXPECT_EQ(BlockedReduce<1>(0, 4, body)[0], 0.0);
}

// ============================================================
// dispatch 内核：1 线程 vs N 线程
// ============================================================

template <typename FP>
class ParallelKernelTest : public ::testing::Test {};

using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ParallelKernelTest, Precisions);

TYPED_TEST(ParallelKernelTest, SingleQubitNamedGates) {
    using FP = TypeParam;
    using Complex = std::complex<FP>;
    const auto& table = Dispatch<FP>();
    const FP c = std::cos(FP(0.37));
    const FP s = std::sin(FP(0.37));
    const Complex rx[4] = {Complex(c, 0), Complex(0, -s), Complex(0, -s), Complex(c, 0)};
    const Complex ry[4] = {Complex(c, 0), Complex(-s, 0), Complex(s, 0), Complex(c, 0)};
    const Complex rz[4] = {Complex(c, -s), Complex(0, 0), Complex(0, 0), Complex(c, s)};

    struct NamedKernel {
        const char* name;
        kernels::SingleQubitKernel<FP> kernel;
        const Complex* matrix;
    };
    const NamedKernel named[] = {
        {"x", table.x, nullptr}, {"y", table.y, nullptr},     {"z", table.z, nullptr}, {"h", table.h, nullptr},
        {"s", table.s, nullptr}, {"sdg", table.sdg, nullptr}, {"t", table.t, nullptr}, {"tdg", table.tdg, nullptr},
        {"rx", table.rx, rx},    {"ry", table.ry, ry},        {"rz", table.rz, rz},
    };
    for (const uint32_t num_qubits : {kSmallQubits, kLargeQubits}) {
        for (const auto& gate : named) {
            // 覆盖最低位、跨 SIMD 块的位以及最高位
            for (const uint32_t qubit : {0u, 1u, 2u, num_qubits - 1}) {
                SCOPED_TRACE(std::string(gate.name) + " n=" + std::to_string(num_qubits)
                             + " q=" + std::to_string(qubit));
                CompareThreadCounts<FP>(
                    num_qubits, [&](Complex* state, uint64_t dim) { gate.kernel(state, dim, qubit, gate.matrix); });
            }
        }
    }
}

TYPED_TEST(ParallelKernelTest, ControlledAndSwapGates) {
    using FP = TypeParam;
    using Complex = std::complex<FP>;
    const auto& table = Dispatch<FP>();
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const uint64_t c0 = uint64_t{1} << 0;
        const uint64_t c_high = uint64_t{1} << (n - 1);
        // CNOT（control 低位 / 高位）、Toffoli（双控制位）
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.cnot(st, dim, n - 1, c0, c0, nullptr); });
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.cnot(st, dim, 0, c_high, c_high, nullptr); });
        CompareThreadCounts<FP>(
            n, [&](Complex* st, uint64_t dim) { table.cnot(st, dim, 3, c0 | c_high, c0 | c_high, nullptr); });
        // CZ
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.cz(st, dim, 1, c0, c0, nullptr); });
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.cz(st, dim, 0, c_high, c_high, nullptr); });
        // SWAP
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.swap(st, dim, 0, n - 1); });
        CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.swap(st, dim, 2, 1); });
    }
}

TYPED_TEST(ParallelKernelTest, DenseAndDiagonalGates) {
    using FP = TypeParam;
    using Complex = std::complex<FP>;
    const auto& table = Dispatch<FP>();
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        const std::vector<std::vector<uint32_t>> target_sets = {{3}, {0, n - 1}, {5, 1, 2}, {0, 2, 4, 6, 8}};
        for (const auto& targets : target_sets) {
            SCOPED_TRACE("n=" + std::to_string(n) + " k=" + std::to_string(targets.size()));
            const auto k = static_cast<uint32_t>(targets.size());
            const auto matrix = RandomMatrix<FP>(1u << k, 99 + k);
            CompareThreadCounts<FP>(
                n, [&](Complex* st, uint64_t dim) { table.dense_k(st, dim, targets.data(), k, matrix.data()); });
            const auto diag = RandomMatrix<FP>(1u << k, 7 + k);  // 只使用前 2^k 个元素
            CompareThreadCounts<FP>(
                n, [&](Complex* st, uint64_t dim) { table.diagonal_k(st, dim, targets.data(), k, diag.data()); });
        }
    }
}

TYPED_TEST(ParallelKernelTest, ProbabilityAndCollapse) {
    using FP = TypeParam;
    using Complex = std::complex<FP>;
    const auto& table = Dispatch<FP>();
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        const auto state = RandomState<FP>(n, 42 + n);
        for (const uint32_t qubit : {0u, 3u, n - 1}) {
            SCOPED_TRACE("n=" + std::to_string(n) + " q=" + std::to_string(qubit));
            FP serial = 0;
            FP parallel = 0;
            {
                ScopedNumThreads threads(1);
                serial = table.prob0(state.data(), state.size(), qubit);
            }
            {
                ScopedNumThreads threads(kParallelThreads);
                parallel = table.prob0(state.data(), state.size(), qubit);
            }
            ExpectRelativeClose(serial, parallel, Tolerance<FP>());

            double reference = 0.0;
            for (uint64_t i = 0; i < state.size(); ++i) {
                if (((i >> qubit) & 1) == 0) {
                    reference += static_cast<double>(std::norm(state[i]));
                }
            }
            ExpectRelativeClose(reference, serial, std::is_same_v<FP, float> ? 1e-5 : 1e-12);

            for (const int outcome : {0, 1}) {
                CompareThreadCounts<FP>(n, [&](Complex* st, uint64_t dim) { table.collapse(st, dim, qubit, outcome); });
            }
        }
    }
}

TYPED_TEST(ParallelKernelTest, PauliExpectation) {
    using FP = TypeParam;
    const auto& table = Dispatch<FP>();
    struct PauliCase {
        uint64_t bit_flip;
        uint64_t phase_flip;
        uint32_t pivot;
        int y_count;
    };
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        const auto state = RandomState<FP>(n, 5 + n);
        const uint64_t high = uint64_t{1} << (n - 1);
        const PauliCase cases[] = {
            {0, 0b101, 0, 0},                  // Z0 Z2
            {0b1, 0, 0, 0},                    // X0
            {0b10, 0b10, 1, 1},                // Y1
            {0b11 | high, 0b10 | high, 0, 2},  // X0 Y1 Y_{n-1}
            {high, 0b1100, n - 1, 0},          // Z2 Z3 X_{n-1}
        };
        for (const auto& c : cases) {
            SCOPED_TRACE("n=" + std::to_string(n) + " bfm=" + std::to_string(c.bit_flip));
            std::complex<FP> serial;
            std::complex<FP> parallel;
            {
                ScopedNumThreads threads(1);
                serial = table.pauli_expect(state.data(), state.size(), c.bit_flip, c.phase_flip, c.pivot, c.y_count);
            }
            {
                ScopedNumThreads threads(kParallelThreads);
                parallel = table.pauli_expect(state.data(), state.size(), c.bit_flip, c.phase_flip, c.pivot, c.y_count);
            }
            ExpectRelativeClose(serial.real(), parallel.real(), Tolerance<FP>());
            ExpectRelativeClose(serial.imag(), parallel.imag(), Tolerance<FP>());

            // 与 scalar 参照实现一致（AVX2 构建时验证 AVX2 内核）
            const auto reference = kernels::scalar::compute_pauli_expectation<FP>(
                state.data(), state.size(), c.bit_flip, c.phase_flip, c.pivot, c.y_count);
            ExpectRelativeClose(reference.real(), serial.real(), Tolerance<FP>());
        }
    }
}

TYPED_TEST(ParallelKernelTest, ActiveBackendMatchesScalarOnLargeStates) {
    using FP = TypeParam;
    using Complex = std::complex<FP>;
    const auto& table = Dispatch<FP>();
    const auto initial = RandomState<FP>(kLargeQubits, 77);
    const Complex u[4] = {Complex(FP(0.6), 0), Complex(0, FP(0.8)), Complex(0, FP(0.8)), Complex(FP(0.6), 0)};
    const std::vector<uint32_t> five_targets = {0, 3, 6, 9, 12};
    const auto dense5 = RandomMatrix<FP>(32, 5);

    ScopedNumThreads threads(kParallelThreads);
    for (const uint32_t qubit : {0u, 1u, 2u, kLargeQubits - 1}) {
        auto active = initial;
        auto reference = initial;
        table.z(active.data(), active.size(), qubit, nullptr);
        kernels::scalar::apply_z<FP>(reference.data(), reference.size(), qubit, nullptr);
        ExpectStatesClose(reference, active);

        active = initial;
        reference = initial;
        table.rx(active.data(), active.size(), qubit, u);
        kernels::scalar::apply_single_qubit_gate<FP>(reference.data(), reference.size(), qubit, u);
        ExpectStatesClose(reference, active);

        active = initial;
        reference = initial;
        const uint64_t control = uint64_t{1} << ((qubit + 1) % kLargeQubits);
        table.cz(active.data(), active.size(), qubit, control, control, nullptr);
        kernels::scalar::apply_cz<FP>(reference.data(), reference.size(), qubit, control, control, nullptr);
        ExpectStatesClose(reference, active);
    }

    // 5 比特稠密门：AVX2 实现回退到 scalar
    auto active = initial;
    auto reference = initial;
    table.dense_k(active.data(), active.size(), five_targets.data(), 5, dense5.data());
    kernels::scalar::apply_dense_gate<FP>(reference.data(), reference.size(), five_targets.data(), 5, dense5.data());
    ExpectStatesClose(reference, active);
}

// ============================================================
// runtime / executor / 噪声 / Ising
// ============================================================

TYPED_TEST(ParallelKernelTest, StateVectorNormAndNormalize) {
    using FP = TypeParam;
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        runtime::StateVector<FP> serial(n);
        runtime::StateVector<FP> parallel(n);
        const auto values = RandomState<FP>(n, 3);
        for (uint64_t i = 0; i < serial.dim(); ++i) {
            serial.data()[i] = values[i] * FP(3);
            parallel.data()[i] = values[i] * FP(3);
        }
        FP serial_norm = 0;
        FP parallel_norm = 0;
        {
            ScopedNumThreads threads(1);
            serial_norm = serial.SquaredNorm();
            serial.Normalize();
        }
        {
            ScopedNumThreads threads(kParallelThreads);
            parallel_norm = parallel.SquaredNorm();
            parallel.Normalize();
        }
        ExpectRelativeClose(serial_norm, parallel_norm, Tolerance<FP>());
        ExpectRelativeClose(9.0, serial_norm, std::is_same_v<FP, float> ? 1e-5 : 1e-12);
        ExpectRelativeClose(1.0, parallel.SquaredNorm(), std::is_same_v<FP, float> ? 1e-5 : 1e-12);
        for (uint64_t i = 0; i < serial.dim(); ++i) {
            ASSERT_LE(std::abs(serial.data()[i] - parallel.data()[i]), Tolerance<FP>());
        }

        ScopedNumThreads threads(kParallelThreads);
        parallel.SetComputationalBasis(serial.dim() - 1);
        EXPECT_EQ(parallel.data()[serial.dim() - 1], std::complex<FP>(1, 0));
        EXPECT_EQ(parallel.SquaredNorm(), FP(1));
        parallel.SetZeroState();
        EXPECT_EQ(parallel.data()[0], std::complex<FP>(1, 0));
        EXPECT_EQ(parallel.SquaredNorm(), FP(1));
    }
}

TEST(ParallelExecutorTest, NoiseKernelsMatchAcrossThreadCounts) {
    using Complex = std::complex<double>;
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        const std::vector<uint32_t> targets = {1, n - 2};
        const auto unitary = RandomMatrix<double>(4, 11);
        CompareThreadCounts<double>(n, [&](Complex* st, uint64_t dim) {
            kernels::apply_dense_unitary<double>(st, dim, targets.data(), 2, unitary.data());
        });

        // 振幅阻尼风格的两个 Kraus 算子（单比特）
        const double gamma = 0.3;
        const Complex kraus[8] = {Complex(1, 0), Complex(0, 0),
                                  Complex(0, 0), Complex(std::sqrt(1 - gamma), 0),
                                  Complex(0, 0), Complex(std::sqrt(gamma), 0),
                                  Complex(0, 0), Complex(0, 0)};
        const uint32_t target = n - 1;
        for (const uint64_t seed : {1ULL, 2ULL, 3ULL}) {
            std::vector<uint32_t> choices;  // [1 线程的选择, N 线程的选择]
            CompareThreadCounts<double>(n, [&](Complex* st, uint64_t dim) {
                std::mt19937_64 rng(seed);
                choices.push_back(kernels::apply_kraus_monte_carlo<double>(st, dim, kraus, 2, &target, 1, rng));
            });
            ASSERT_EQ(choices.size(), 2u);
            EXPECT_EQ(choices[0], choices[1]);
        }
    }
}

TEST(ParallelExecutorTest, SimulationResultsMatchAcrossThreadCounts) {
    for (const uint32_t n : {kSmallQubits, kLargeQubits}) {
        ir::ProgramBuilder builder(n, n);
        for (uint32_t q = 0; q < n; ++q) {
            builder.AddGate("h", {q});
            builder.AddGate("ry", {q}, {0.1 * (q + 1)});
        }
        for (uint32_t q = 0; q + 1 < n; ++q) {
            builder.AddGate("cx", {q, q + 1});
            builder.AddGate("rz", {q + 1}, {0.05 * (q + 2)});
        }
        builder.AddGate("swap", {0, n - 1});
        builder.AddObservable("Z0", {{0, 'Z'}});
        builder.AddObservable("XY", {{1, 'X'}, {n - 1, 'Y'}});
        // 中途测量 + 后续门：触发逐 shot 轨迹（prob0 / collapse 参与随机决策）
        builder.AddMeasure({0}, {0});
        builder.AddGate("h", {0});
        std::vector<uint32_t> all(n);
        for (uint32_t q = 0; q < n; ++q) {
            all[q] = q;
        }
        builder.AddMeasure(all, all);
        auto program = builder.Build();

        for (const auto precision : {executor::PrecisionKind::kFloat32, executor::PrecisionKind::kFloat64}) {
            SCOPED_TRACE("n=" + std::to_string(n));
            executor::SimulationExecutor::Options options;
            options.precision = precision;
            options.seed = 2024;
            executor::ExecutionRequest request;
            request.shots = 64;
            request.return_expectations = true;

            executor::ExecutionResult serial;
            executor::ExecutionResult parallel;
            {
                ScopedNumThreads threads(1);
                serial = executor::SimulationExecutor(options).Run(*program, request);
            }
            {
                ScopedNumThreads threads(kParallelThreads);
                parallel = executor::SimulationExecutor(options).Run(*program, request);
            }
            EXPECT_EQ(serial.counts, parallel.counts);
            ASSERT_EQ(serial.expectations_real.size(), parallel.expectations_real.size());
            for (const auto& [label, value] : serial.expectations_real) {
                ExpectRelativeClose(value, parallel.expectations_real.at(label), 1e-6);
            }
        }
    }
}

TEST(ParallelExecutorTest, AdjointGradientMatchesAcrossThreadCounts) {
    const uint32_t n = kLargeQubits;
    runtime::QSALProgram program;
    program.num_qubits = n;
    program.parameter_names = {"theta0", "theta1", "theta2", "theta3"};
    for (uint32_t q = 0; q < n; ++q) {
        runtime::GateOperation h;
        h.opcode = runtime::OpCode::kNamedGate;
        h.name = "h";
        h.qubits = {q};
        program.ops.push_back(h);

        runtime::GateOperation rotation;
        rotation.opcode = runtime::OpCode::kNamedGate;
        rotation.name = (q % 3 == 0) ? "rx" : (q % 3 == 1 ? "ry" : "rz");
        rotation.qubits = {q};
        runtime::ParameterExpression expression;
        expression.constant = 0.1;
        expression.terms = {{"theta" + std::to_string(q % 4), 1.0 + 0.1 * q}};
        rotation.parameter_expressions.push_back(expression);
        program.ops.push_back(rotation);
    }
    for (uint32_t q = 0; q + 1 < n; ++q) {
        runtime::GateOperation cx;
        cx.opcode = runtime::OpCode::kNamedGate;
        cx.name = "cx";
        cx.qubits = {q, q + 1};
        program.ops.push_back(cx);
    }
    runtime::ObservableTerm zz;
    zz.label = "H";
    zz.coefficient = {0.7, 0.0};
    zz.paulis = {{0, 'Z'}, {n - 1, 'Z'}};
    program.observables.push_back(zz);
    runtime::ObservableTerm xy = zz;
    xy.coefficient = {-0.4, 0.0};
    xy.paulis = {{2, 'X'}, {5, 'Y'}};
    program.observables.push_back(xy);

    const std::unordered_map<std::string, double> values = {
        {"theta0", 0.3}, {"theta1", -0.8}, {"theta2", 1.1}, {"theta3", 0.25}};
    executor::GradientResult serial;
    executor::GradientResult parallel;
    {
        ScopedNumThreads threads(1);
        serial = executor::AdjointGradientExecutor().Run(program, values, executor::PrecisionKind::kFloat64);
    }
    {
        ScopedNumThreads threads(kParallelThreads);
        parallel = executor::AdjointGradientExecutor().Run(program, values, executor::PrecisionKind::kFloat64);
    }
    ASSERT_EQ(serial.expectations.size(), parallel.expectations.size());
    for (std::size_t i = 0; i < serial.expectations.size(); ++i) {
        ExpectRelativeClose(serial.expectations[i], parallel.expectations[i], 1e-12);
        ASSERT_EQ(serial.gradients[i].size(), parallel.gradients[i].size());
        for (std::size_t j = 0; j < serial.gradients[i].size(); ++j) {
            ExpectRelativeClose(serial.gradients[i][j], parallel.gradients[i][j], 1e-12);
        }
    }
}

TEST(ParallelExecutorTest, IsingAnnealingIsIndependentOfThreadCount) {
    const std::size_t num_spins = 12;
    std::vector<double> matrix(num_spins * num_spins, 0.0);
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (std::size_t r = 0; r < num_spins; ++r) {
        for (std::size_t c = r + 1; c < num_spins; ++c) {
            matrix[r * num_spins + c] = matrix[c * num_spins + r] = uniform(rng);
        }
    }
    const optimization::DenseIsingModel model(matrix, num_spins);
    optimization::IsingSolverOptions options;
    options.seed = 17;
    options.num_reads = 9;
    options.sweeps = 50;

    optimization::IsingSolverResult serial;
    optimization::IsingSolverResult parallel;
    {
        ScopedNumThreads threads(1);
        serial = optimization::SolveIsingSimulatedAnnealing(model, options);
    }
    {
        ScopedNumThreads threads(kParallelThreads);
        parallel = optimization::SolveIsingSimulatedAnnealing(model, options);
    }
    ASSERT_EQ(serial.samples.size(), parallel.samples.size());
    for (std::size_t i = 0; i < serial.samples.size(); ++i) {
        EXPECT_EQ(serial.samples[i].spins, parallel.samples[i].spins);
        EXPECT_EQ(serial.samples[i].energy, parallel.samples[i].energy);
    }
}

}  // namespace
