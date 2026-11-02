// CX 朝向全局选择（对偶分解）的 gtest。
#include <cmath>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/transpiler/cx_orientation.hpp"
#include "quspect_native/transpiler/one_qubit_gauge.hpp"

namespace {

using quspect::native::transpiler::AxisRotation;
using quspect::native::transpiler::Complex;
using quspect::native::transpiler::LengthModel;
using quspect::native::transpiler::Matrix2;
using quspect::native::transpiler::OrientationEnergy;
using quspect::native::transpiler::OrientationOptions;
using quspect::native::transpiler::OrientationProblem;
using quspect::native::transpiler::SolveCxOrientation;

const double kR = std::sqrt(0.5);
const Matrix2 kI = {Complex{1, 0}, Complex{0, 0}, Complex{0, 0}, Complex{1, 0}};
const Matrix2 kH = {Complex{kR, 0}, Complex{kR, 0}, Complex{kR, 0}, Complex{-kR, 0}};

// IBM {rz, sx, x}
LengthModel IbmModel() {
    LengthModel m;
    m.x = true;
    m.sx = true;
    m.named = {
        {Complex{0, 0}, Complex{1, 0}, Complex{1, 0}, Complex{0, 0}},
        {Complex{0.5, 0.5}, Complex{0.5, -0.5}, Complex{0.5, -0.5}, Complex{0.5, 0.5}},
    };
    return m;
}

// q0 是两个 CX 的控制线：H·CX·RX(θ)·CX·H 在 X 坐标系里；两个 CX 一起反向后 H 全部抵消，
// RX 变成目标线上的 RZ。
OrientationProblem XFrameZz(double theta) {
    OrientationProblem p;
    p.num_gates = 2;
    p.wire_gates = {{0, 1}, {0, 1}};
    p.wire_is_control = {{1, 1}, {0, 0}};
    p.wire_runs = {{kH, AxisRotation(1, theta), kH}, {kH, kI, kH}};
    return p;
}

TEST(CxOrientationTest, FlipsWholeXFrameBlock) {
    const OrientationProblem p = XFrameZz(0.3);
    const OrientationOptions opt;
    const auto r = SolveCxOrientation(p, IbmModel(), opt);
    EXPECT_EQ(r.flip, (std::vector<std::uint8_t>{1, 1}));
    EXPECT_LT(r.energy, r.initial_energy);
    // 反向后只剩一个 rz：代价 2·1 + 1
    EXPECT_DOUBLE_EQ(r.energy, 3.0);
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, r.flip, IbmModel(), opt), r.energy);
}

TEST(CxOrientationTest, KeepsOrientationWithoutGain) {
    OrientationProblem p;
    p.num_gates = 2;
    p.wire_gates = {{0, 1}, {0, 1}};
    p.wire_is_control = {{1, 1}, {0, 0}};
    p.wire_runs = {{kI, kI, kI}, {kI, AxisRotation(0, 0.7), kI}};
    const auto r = SolveCxOrientation(p, IbmModel(), OrientationOptions{});
    EXPECT_EQ(r.flip, (std::vector<std::uint8_t>{0, 0}));
    EXPECT_DOUBLE_EQ(r.energy, r.initial_energy);
    EXPECT_DOUBLE_EQ(r.energy, 3.0);
}

TEST(CxOrientationTest, RotationsSlideThroughControls) {
    // RZ 在控制线上可以穿过 CX：两个 RZ 并成一个运行（代价 3），而不是两个（代价 6）。
    OrientationProblem p;
    p.num_gates = 1;
    p.wire_gates = {{0}, {0}};
    p.wire_is_control = {{1}, {0}};
    p.wire_runs = {{AxisRotation(0, 0.4), AxisRotation(0, 0.5)}, {kI, kI}};
    const OrientationOptions opt;
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, {0}, IbmModel(), opt), 3.0);
}

TEST(CxOrientationTest, NeverWorseOnChains) {
    // 交错的 CX 链：结果的代价不超过初始代价，且与重新求值一致。
    OrientationProblem p;
    p.num_gates = 4;
    p.wire_gates = {{0, 2}, {0, 1, 3}, {1, 2, 3}};
    p.wire_is_control = {{1, 0}, {0, 1, 0}, {0, 1, 1}};
    p.wire_runs = {
        {kH, AxisRotation(1, 0.2), kH}, {kI, kH, AxisRotation(2, 0.9), kH}, {AxisRotation(1, 1.1), kI, kH, kI}};
    const OrientationOptions opt;
    const auto r = SolveCxOrientation(p, IbmModel(), opt);
    EXPECT_LE(r.energy, r.initial_energy);
    EXPECT_DOUBLE_EQ(OrientationEnergy(p, r.flip, IbmModel(), opt), r.energy);
}

// 优化前（逐层排序 + 按运行编号的 memo）的对偶分解实现，原样保留作对照：新实现必须给出逐位相同的
// 结果（朝向、代价、对偶值、迭代次数）。
namespace reference {

using quspect::native::transpiler::FastLength;
using quspect::native::transpiler::LeadingAngles;
using quspect::native::transpiler::LeadingAnglesInZFrame;
using quspect::native::transpiler::OrientationResult;

constexpr int kAxisZ = 0;
constexpr int kAxisX = 1;
constexpr int kNoAxis = -1;
// 不能复刻（非有限值）的运行按一般旋转的门数计分。
constexpr int kFallbackLength = 5;
constexpr double kAngleTol = 1e-9;

Matrix2 Mul(const Matrix2& a, const Matrix2& b) {
    auto mul = [](const Complex& x, const Complex& y) {
        return Complex{x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re};
    };
    auto add = [](const Complex& x, const Complex& y) { return Complex{x.re + y.re, x.im + y.im}; };
    return {add(mul(a[0], b[0]), mul(a[1], b[2])), add(mul(a[0], b[1]), mul(a[1], b[3])),
            add(mul(a[2], b[0]), mul(a[3], b[2])), add(mul(a[2], b[1]), mul(a[3], b[3]))};
}

Matrix2 Hadamard() {
    const double r = std::sqrt(0.5);
    return {Complex{r, 0.0}, Complex{r, 0.0}, Complex{r, 0.0}, Complex{-r, 0.0}};
}

// 线 ``is_control`` 上、朝向 ``flip`` 的 CX 允许穿过的旋转轴。
int CommutingAxis(bool is_control, bool flip) {
    return is_control != flip ? kAxisZ : kAxisX;
}

std::uint64_t Bits(double x) {
    std::uint64_t b = 0;
    std::memcpy(&b, &x, sizeof(b));
    return b;
}

struct MemoKey {
    int run;
    int left;  // 左侧 CX 朝向（-1：无）
    int axis_in;
    std::uint64_t phi_in;
    int right;  // 右侧 CX 朝向（-1：无）
    bool operator==(const MemoKey& o) const {
        return run == o.run && left == o.left && axis_in == o.axis_in && phi_in == o.phi_in && right == o.right;
    }
};

struct MemoKeyHash {
    std::size_t operator()(const MemoKey& k) const {
        std::size_t h = std::hash<std::uint64_t>()(k.phi_in);
        h ^= static_cast<std::size_t>(k.run) * 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>((k.left + 1) * 9 + (k.axis_in + 1) * 3 + (k.right + 1)) + (h << 6) + (h >> 2);
        return h;
    }
};

// 一个运行在给定入射旋转与两侧朝向下的选项：(带出角, 代价)；带出角 0 表示不带出。
using Options = std::vector<std::pair<double, int>>;

class ChainSolver {
 public:
    ChainSolver(const OrientationProblem& problem, const LengthModel& model, const OrientationOptions& options)
        : problem_(problem), model_(model), options_(options), h_(Hadamard()) {
        runs_offset_.resize(problem.wire_runs.size() + 1, 0);
        for (std::size_t w = 0; w < problem.wire_runs.size(); ++w) {
            runs_offset_[w + 1] = runs_offset_[w] + static_cast<int>(problem.wire_runs[w].size());
        }
    }

    // 线 ``wire`` 的最小代价。``fixed`` 非空时朝向取 fixed[g]；否则自由，``linear[j]`` 为第 j 个 CX
    // 反向时附加的代价。``chosen`` 返回沿线各 CX 的朝向。
    double Solve(int wire, const std::vector<std::uint8_t>* fixed, const std::vector<double>* linear,
                 std::vector<std::uint8_t>* chosen) {
        const auto& gates = problem_.wire_gates[wire];
        const auto& is_control = problem_.wire_is_control[wire];
        const auto& runs = problem_.wire_runs[wire];
        const int n = static_cast<int>(gates.size());
        if (static_cast<int>(runs.size()) != n + 1) {
            throw std::invalid_argument("each wire needs one more run than gates");
        }
        layers_.assign(static_cast<std::size_t>(n) + 1, {});
        std::vector<State> current = {State{0.0, -1, kNoAxis, 0.0, -1}};
        for (int j = 0; j <= n; ++j) {
            std::vector<State> next;
            int choices[2] = {0, 1};
            int num_choices = 2;
            if (j == n) {
                choices[0] = -1;
                num_choices = 1;
            } else if (fixed != nullptr) {
                choices[0] = (*fixed)[static_cast<std::size_t>(gates[j])] ? 1 : 0;
                num_choices = 1;
            }
            for (int si = 0; si < static_cast<int>(current.size()); ++si) {
                const State& st = current[static_cast<std::size_t>(si)];
                for (int c = 0; c < num_choices; ++c) {
                    const int right = choices[c];
                    const int axis_out = right < 0 ? kNoAxis : CommutingAxis(is_control[j] != 0, right == 1);
                    const Options& opts = RunOptions(runs_offset_[wire] + j, runs[j], st, right, axis_out);
                    const double extra = (right == 1 && linear != nullptr) ? (*linear)[static_cast<std::size_t>(j)]
                                                                           : 0.0;
                    for (const auto& [phi, cost] : opts) {
                        next.push_back(State{st.cost + cost + extra, right, axis_out, phi, si});
                    }
                }
            }
            current = Prune(std::move(next));
            layers_[static_cast<std::size_t>(j)] = current;
        }
        int best = 0;
        for (int i = 1; i < static_cast<int>(current.size()); ++i) {
            if (current[static_cast<std::size_t>(i)].cost < current[static_cast<std::size_t>(best)].cost) {
                best = i;
            }
        }
        const double value = current[static_cast<std::size_t>(best)].cost;
        if (chosen != nullptr) {
            chosen->assign(static_cast<std::size_t>(n), 0);
            int idx = best;
            for (int j = n; j >= 0; --j) {
                const State& st = layers_[static_cast<std::size_t>(j)][static_cast<std::size_t>(idx)];
                if (j < n) {
                    (*chosen)[static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(st.flip == 1);
                }
                idx = st.parent;
            }
        }
        return value;
    }

 private:
    struct State {
        double cost;
        int flip;  // 本状态之前（左侧）CX 的朝向；-1：链首
        int axis;  // 带入下一个运行的旋转轴
        double phi;
        int parent;
    };

    int Score(const Matrix2& m) const {
        const std::optional<int> len = FastLength(m, model_);
        const int n = len.has_value() ? *len : kFallbackLength;
        return options_.gate_weight * n + (n > 0 ? options_.run_weight : 0);
    }

    const Options& RunOptions(int run_id, const Matrix2& run, const State& st, int right, int axis_out) {
        const MemoKey key{run_id, st.flip, st.axis, Bits(st.phi), right};
        auto it = memo_.find(key);
        if (it != memo_.end()) {
            return it->second;
        }
        Matrix2 base = run;
        if (st.flip == 1) {
            base = Mul(base, h_);
        }
        if (st.axis != kNoAxis && st.phi != 0.0) {
            base = Mul(base, AxisRotation(st.axis, st.phi));
        }
        if (right == 1) {
            base = Mul(h_, base);
        }
        Options opts;
        opts.emplace_back(0.0, Score(base));
        if (axis_out != kNoAxis) {
            const Matrix2 z = axis_out == kAxisZ ? base : Mul(h_, Mul(base, h_));
            const std::optional<LeadingAngles> lead = LeadingAnglesInZFrame(z);
            if (lead.has_value()) {
                for (int k = 0; k < lead->count; ++k) {
                    const double phi = lead->angles[static_cast<std::size_t>(k)];
                    if (!std::isfinite(phi) || std::abs(phi) < kAngleTol) {
                        continue;
                    }
                    const Matrix2 rest = Mul(AxisRotation(axis_out, -phi), base);
                    opts.emplace_back(phi, Score(rest));
                }
            }
        }
        return memo_.emplace(key, std::move(opts)).first->second;
    }

    // 每个朝向保留代价最低的 beam 个（角度去重）状态。
    std::vector<State> Prune(std::vector<State> states) const {
        std::stable_sort(states.begin(), states.end(), [](const State& a, const State& b) { return a.cost < b.cost; });
        std::vector<State> out;
        int kept[3] = {0, 0, 0};
        for (const State& st : states) {
            int& count = kept[st.flip + 1];
            if (count >= options_.beam) {
                continue;
            }
            bool dup = false;
            for (const State& o : out) {
                if (o.flip == st.flip && o.axis == st.axis && std::abs(o.phi - st.phi) < kAngleTol) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            out.push_back(st);
            ++count;
        }
        return out;
    }

    const OrientationProblem& problem_;
    const LengthModel& model_;
    const OrientationOptions& options_;
    Matrix2 h_;
    std::vector<int> runs_offset_;
    std::vector<std::vector<State>> layers_;
    std::unordered_map<MemoKey, Options, MemoKeyHash> memo_;
};

OrientationResult ReferenceSolve(const OrientationProblem& problem, const LengthModel& model,
                                 const OrientationOptions& options) {
    const int num_wires = static_cast<int>(problem.wire_gates.size());
    const auto ng = static_cast<std::size_t>(problem.num_gates);
    ChainSolver solver(problem, model, options);

    // 一条线的固定朝向代价只依赖这条线上的朝向位：每条线缓存最近几组位向量的代价
    // （相邻迭代的两份副本大多只在少数线上不同）。
    constexpr std::size_t kWireCache = 4;
    std::vector<std::vector<std::pair<std::vector<std::uint8_t>, double>>> wire_cache(
        static_cast<std::size_t>(num_wires));
    std::vector<std::uint8_t> bits;
    auto energy = [&](const std::vector<std::uint8_t>& flip) {
        double total = 0.0;
        for (int w = 0; w < num_wires; ++w) {
            const auto& gates = problem.wire_gates[static_cast<std::size_t>(w)];
            bits.resize(gates.size());
            for (std::size_t j = 0; j < gates.size(); ++j) {
                bits[j] = flip[static_cast<std::size_t>(gates[j])];
            }
            auto& cache = wire_cache[static_cast<std::size_t>(w)];
            bool hit = false;
            for (const auto& [key, cost] : cache) {
                if (key == bits) {
                    total += cost;
                    hit = true;
                    break;
                }
            }
            if (hit) {
                continue;
            }
            const double cost = solver.Solve(w, &flip, nullptr, nullptr);
            if (cache.size() >= kWireCache) {
                cache.erase(cache.begin());
            }
            cache.emplace_back(bits, cost);
            total += cost;
        }
        return total;
    };

    OrientationResult result;
    result.flip.assign(ng, 0);
    result.initial_energy = energy(result.flip);
    result.energy = result.initial_energy;
    if (ng == 0) {
        return result;
    }
    {
        const std::vector<std::uint8_t> all(ng, 1);
        const double e = energy(all);
        if (e < result.energy) {
            result.energy = e;
            result.flip = all;
        }
    }

    // 每条线的对偶子问题只依赖它上面的乘子：乘子没变的线沿用上次的解。
    std::vector<std::uint8_t> dirty(static_cast<std::size_t>(num_wires), 1);
    std::vector<double> wire_dual(static_cast<std::size_t>(num_wires), 0.0);
    std::vector<std::vector<std::uint8_t>> wire_chosen(static_cast<std::size_t>(num_wires));
    std::vector<std::vector<int>> gate_wires(ng);
    for (int w = 0; w < num_wires; ++w) {
        for (int g : problem.wire_gates[static_cast<std::size_t>(w)]) {
            gate_wires[static_cast<std::size_t>(g)].push_back(w);
        }
    }

    std::vector<double> lambda(ng, 0.0);
    std::vector<std::uint8_t> ctrl_copy(ng, 0);
    std::vector<std::uint8_t> tgt_copy(ng, 0);
    std::vector<double> linear;
    double theta = 1.0;
    double best_dual = -std::numeric_limits<double>::infinity();
    int stall = 0;
    int no_primal_gain = 0;
    for (int it = 0; it < options.iterations; ++it) {
        double dual = 0.0;
        for (int w = 0; w < num_wires; ++w) {
            const auto wi = static_cast<std::size_t>(w);
            const auto& gates = problem.wire_gates[wi];
            const auto& is_control = problem.wire_is_control[wi];
            if (dirty[wi]) {
                linear.resize(gates.size());
                for (std::size_t j = 0; j < gates.size(); ++j) {
                    const double l = lambda[static_cast<std::size_t>(gates[j])];
                    linear[j] = is_control[j] ? l : -l;
                }
                wire_dual[wi] = solver.Solve(w, nullptr, &linear, &wire_chosen[wi]);
                dirty[wi] = 0;
            }
            dual += wire_dual[wi];
            for (std::size_t j = 0; j < gates.size(); ++j) {
                (is_control[j] ? ctrl_copy : tgt_copy)[static_cast<std::size_t>(gates[j])] = wire_chosen[wi][j];
            }
        }
        result.dual = dual;
        result.iterations = it + 1;
        double norm2 = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
            const double d = static_cast<double>(ctrl_copy[g]) - static_cast<double>(tgt_copy[g]);
            norm2 += d * d;
        }
        bool improved = false;
        for (const auto* cand : {&ctrl_copy, &tgt_copy}) {
            const double e = energy(*cand);
            if (e < result.energy) {
                result.energy = e;
                result.flip = *cand;
                improved = true;
            }
            if (norm2 == 0.0) {
                break;
            }
        }
        if (norm2 == 0.0) {
            break;  // 两份副本一致：链的解拼成了原问题的解
        }
        no_primal_gain = improved ? 0 : no_primal_gain + 1;
        if (no_primal_gain >= options.patience) {
            break;
        }
        if (dual > best_dual + 1e-9) {
            best_dual = dual;
            stall = 0;
        } else if (++stall >= 3) {
            theta *= 0.5;
            stall = 0;
        }
        const double gap = std::max(result.energy - dual, 1.0);
        const double step = theta * gap / norm2;
        for (std::size_t g = 0; g < ng; ++g) {
            if (ctrl_copy[g] == tgt_copy[g]) {
                continue;
            }
            const double d = static_cast<double>(ctrl_copy[g]) - static_cast<double>(tgt_copy[g]);
            lambda[g] += step * d;
            for (int w : gate_wires[g]) {
                dirty[static_cast<std::size_t>(w)] = 1;
            }
        }
    }
    return result;
}

}  // namespace reference

// 随机问题（多数运行是单位阵、矩阵大量重复，与布线后的线路相同）上与优化前的实现逐位比较。
// 只用随机引擎的原始输出（不用实现相关的分布）。
class ProblemGenerator {
 public:
    explicit ProblemGenerator(std::uint64_t seed) : rng_(seed) {
        pool_ = {kH,
                 AxisRotation(0, 0.25 * kPi),
                 AxisRotation(1, 0.5 * kPi),
                 AxisRotation(0, kPi),
                 Mul(kH, AxisRotation(0, 0.75 * kPi)),
                 Mul(AxisRotation(1, 0.3), AxisRotation(0, -1.1))};
    }

    std::uint64_t Next() {
        return rng_();
    }
    int Below(int n) {
        return static_cast<int>(Next() % static_cast<std::uint64_t>(n));
    }
    double Angle() {
        return static_cast<double>(Next() >> 11) * 0x1.0p-53 * 2.0 * kPi - kPi;
    }

    Matrix2 Run() {
        const int r = Below(100);
        if (r < 55) {
            return kI;
        }
        if (r < 90) {
            return pool_[static_cast<std::size_t>(Below(static_cast<int>(pool_.size())))];
        }
        if (r < 95) {
            return AxisRotation(Below(2), Angle());
        }
        return Mul(Mul(AxisRotation(0, Angle()), AxisRotation(1, Angle())), AxisRotation(0, Angle()));
    }

    OrientationProblem Problem() {
        OrientationProblem p;
        const int wires = 2 + Below(6);
        p.num_gates = 1 + Below(120);
        p.wire_gates.resize(static_cast<std::size_t>(wires));
        p.wire_is_control.resize(static_cast<std::size_t>(wires));
        p.wire_runs.resize(static_cast<std::size_t>(wires));
        for (int g = 0; g < p.num_gates; ++g) {
            const int c = Below(wires);
            const int t = (c + 1 + Below(wires - 1)) % wires;
            for (const int w : {c, t}) {
                const auto wi = static_cast<std::size_t>(w);
                p.wire_gates[wi].push_back(g);
                p.wire_is_control[wi].push_back(w == c ? 1 : 0);
                p.wire_runs[wi].push_back(Run());
            }
        }
        for (auto& runs : p.wire_runs) {
            runs.push_back(Run());
        }
        // 不参与任何 CX 的线去掉
        OrientationProblem q;
        q.num_gates = p.num_gates;
        for (std::size_t w = 0; w < p.wire_gates.size(); ++w) {
            if (!p.wire_gates[w].empty()) {
                q.wire_gates.push_back(p.wire_gates[w]);
                q.wire_is_control.push_back(p.wire_is_control[w]);
                q.wire_runs.push_back(p.wire_runs[w]);
            }
        }
        return q;
    }

 private:
    static constexpr double kPi = 3.141592653589793;

    static Matrix2 Mul(const Matrix2& a, const Matrix2& b) {
        auto mul = [](const Complex& x, const Complex& y) {
            return Complex{x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re};
        };
        auto add = [](const Complex& x, const Complex& y) { return Complex{x.re + y.re, x.im + y.im}; };
        return {add(mul(a[0], b[0]), mul(a[1], b[2])), add(mul(a[0], b[1]), mul(a[1], b[3])),
                add(mul(a[2], b[0]), mul(a[3], b[2])), add(mul(a[2], b[1]), mul(a[3], b[3]))};
    }

    std::mt19937_64 rng_;
    std::vector<Matrix2> pool_;
};

TEST(CxOrientationTest, MatchesReferenceImplementationBitwise) {
    ProblemGenerator gen(20261101);
    const LengthModel model = IbmModel();
    const int weights[][2] = {{2, 1}, {1, 0}, {3, 2}, {0, -1}};
    int flipped = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const OrientationProblem p = gen.Problem();
        OrientationOptions opt;
        opt.beam = 1 + gen.Below(6);
        const auto& w = weights[gen.Below(4)];
        opt.gate_weight = w[0];
        opt.run_weight = w[1];
        opt.iterations = 1 + gen.Below(50);
        opt.patience = 1 + gen.Below(opt.iterations + 5);
        // 固定朝向的代价（随机朝向）
        std::vector<std::uint8_t> flip(static_cast<std::size_t>(p.num_gates));
        for (auto& f : flip) {
            f = static_cast<std::uint8_t>(gen.Next() & 1u);
        }
        reference::ChainSolver solver(p, model, opt);
        double total = 0.0;
        for (int wire = 0; wire < static_cast<int>(p.wire_gates.size()); ++wire) {
            total += solver.Solve(wire, &flip, nullptr, nullptr);
        }
        const double energy = OrientationEnergy(p, flip, model, opt);
        ASSERT_EQ(std::memcmp(&energy, &total, sizeof(double)), 0) << "trial " << trial;
        const auto got = SolveCxOrientation(p, model, opt);
        const auto want = reference::ReferenceSolve(p, model, opt);
        ASSERT_EQ(got.flip, want.flip) << "trial " << trial;
        ASSERT_EQ(std::memcmp(&got.energy, &want.energy, sizeof(double)), 0) << "trial " << trial;
        ASSERT_EQ(std::memcmp(&got.initial_energy, &want.initial_energy, sizeof(double)), 0) << "trial " << trial;
        ASSERT_EQ(std::memcmp(&got.dual, &want.dual, sizeof(double)), 0) << "trial " << trial;
        ASSERT_EQ(got.iterations, want.iterations) << "trial " << trial;
        for (const std::uint8_t f : got.flip) {
            flipped += f;
        }
    }
    EXPECT_GT(flipped, 0);  // 确实走到了反向的情形
}

TEST(CxOrientationTest, RejectsMalformedProblems) {
    OrientationProblem p = XFrameZz(0.3);
    p.wire_gates[1] = {0};
    p.wire_is_control[1] = {0};
    p.wire_runs[1] = {kI, kI};
    EXPECT_THROW(SolveCxOrientation(p, IbmModel(), OrientationOptions{}), std::invalid_argument);
}

}  // namespace
