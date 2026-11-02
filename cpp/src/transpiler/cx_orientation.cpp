#include "quspect_native/transpiler/cx_orientation.hpp"

#include <cmath>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {
namespace {

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

// 链求解器的选项表。一个"条目"是一个运行在给定上下文（入射朝向 / 轴 / 角、本运行右侧 CX 的
// 朝向与带出轴）下的全部选项：(带出角, 代价)，带出角 0 表示不带出。条目只依赖运行矩阵的内容与
// 上下文，按内容去重（布线后的线路里大多数运行是单位阵）；束中的状态指向它来自的选项，下一个
// 运行的条目由 (选项, 运行矩阵, 朝向, 带出轴) 经转移表直接找到。
struct Option {
    double phi;
    int cost;
    std::int8_t right;     // 产生这个选项的条目右侧 CX 的朝向（-1：链尾 / 链首的虚拟选项）
    std::int8_t axis_out;  // 带入下一个运行的旋转轴
};

struct EntryKey {
    int matrix;
    int left;
    int axis_in;
    std::uint64_t phi_in;
    int right;
    int axis_out;
    bool operator==(const EntryKey& o) const {
        return matrix == o.matrix && left == o.left && axis_in == o.axis_in && phi_in == o.phi_in && right == o.right
               && axis_out == o.axis_out;
    }
};

struct EntryKeyHash {
    std::size_t operator()(const EntryKey& k) const {
        std::uint64_t h = k.phi_in * 0x9E3779B97F4A7C15ULL;
        h ^= (static_cast<std::uint64_t>(static_cast<std::uint32_t>(k.matrix)) << 8)
             ^ static_cast<std::uint64_t>((k.left + 1) * 27 + (k.axis_in + 1) * 9 + (k.right + 1) * 3
                                          + (k.axis_out + 1));
        h *= 0xBF58476D1CE4E5B9ULL;
        return static_cast<std::size_t>(h ^ (h >> 31));
    }
};

struct MatrixKeyHash {
    std::size_t operator()(const std::array<std::uint64_t, 8>& k) const {
        std::uint64_t h = 0;
        for (std::uint64_t v : k) {
            h = (h ^ v) * 0x9E3779B97F4A7C15ULL;
            h ^= h >> 29;
        }
        return static_cast<std::size_t>(h);
    }
};

// (选项, 运行矩阵, 朝向, 带出轴) → 条目的开放寻址表（键打包成 64 位）。
class TransitionTable {
 public:
    static constexpr std::uint32_t kMissing = 0xFFFFFFFFu;

    TransitionTable() {
        Rehash(1024);
    }

    static std::uint64_t Key(std::uint32_t option, int matrix, int right, int axis_out) {
        return (static_cast<std::uint64_t>(option) << 32) | (static_cast<std::uint64_t>(matrix) << 4)
               | static_cast<std::uint64_t>((right + 1) * 3 + (axis_out + 1));
    }

    std::uint32_t Find(std::uint64_t key) const {
        for (std::size_t i = Slot(key);; i = (i + 1) & mask_) {
            if (keys_[i] == key) {
                return values_[i];
            }
            if (keys_[i] == kEmpty) {
                return kMissing;
            }
        }
    }

    void Insert(std::uint64_t key, std::uint32_t value) {
        if (2 * (size_ + 1) > keys_.size()) {
            Rehash(2 * keys_.size());
        }
        std::size_t i = Slot(key);
        while (keys_[i] != kEmpty) {
            i = (i + 1) & mask_;
        }
        keys_[i] = key;
        values_[i] = value;
        ++size_;
    }

 private:
    static constexpr std::uint64_t kEmpty = ~std::uint64_t{0};

    std::size_t Slot(std::uint64_t key) const {
        return static_cast<std::size_t>((key * 0x9E3779B97F4A7C15ULL) >> shift_) & mask_;
    }

    void Rehash(std::size_t capacity) {
        std::vector<std::uint64_t> keys(capacity, kEmpty);
        std::vector<std::uint32_t> values(capacity, 0);
        keys.swap(keys_);
        values.swap(values_);
        mask_ = capacity - 1;
        shift_ = 64;
        for (std::size_t c = capacity; c > 1; c >>= 1) {
            --shift_;
        }
        size_ = 0;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i] != kEmpty) {
                std::size_t s = Slot(keys[i]);
                while (keys_[s] != kEmpty) {
                    s = (s + 1) & mask_;
                }
                keys_[s] = keys[i];
                values_[s] = values[i];
                ++size_;
            }
        }
    }

    std::vector<std::uint64_t> keys_;
    std::vector<std::uint32_t> values_;
    std::size_t mask_ = 0;
    int shift_ = 64;
    std::size_t size_ = 0;
};

class ChainSolver {
 public:
    ChainSolver(const OrientationProblem& problem, const LengthModel& model, const OrientationOptions& options)
        : problem_(problem)
        , model_(model)
        , options_(options)
        , h_(Hadamard())
        , monotone_(options.gate_weight >= 0 && options.run_weight >= 0) {
        // 运行矩阵按内容（逐位）编号（转移表的键里占 28 位）
        std::size_t num_runs = 0;
        for (const auto& runs : problem.wire_runs) {
            num_runs += runs.size();
        }
        if (num_runs >= (std::size_t{1} << 28)) {
            throw std::invalid_argument("too many single-qubit runs");
        }
        std::unordered_map<std::array<std::uint64_t, 8>, int, MatrixKeyHash> ids;
        wire_matrix_.resize(problem.wire_runs.size());
        for (std::size_t w = 0; w < problem.wire_runs.size(); ++w) {
            for (const Matrix2& m : problem.wire_runs[w]) {
                std::array<std::uint64_t, 8> key{};
                for (int i = 0; i < 4; ++i) {
                    key[static_cast<std::size_t>(2 * i)] = Bits(m[static_cast<std::size_t>(i)].re);
                    key[static_cast<std::size_t>(2 * i + 1)] = Bits(m[static_cast<std::size_t>(i)].im);
                }
                const auto [it, added] = ids.emplace(key, static_cast<int>(matrices_.size()));
                if (added) {
                    matrices_.push_back(m);
                }
                wire_matrix_[w].push_back(it->second);
            }
        }
        // 链首的虚拟选项：无朝向、不带旋转
        options_list_.push_back(Option{0.0, 0, -1, static_cast<std::int8_t>(kNoAxis)});
        sorted_.push_back(0);
    }

    // 线 ``wire`` 的最小代价。``fixed`` 非空时朝向取 fixed[g]；否则自由，``linear[j]`` 为第 j 个 CX
    // 反向时附加的代价。``chosen`` 返回沿线各 CX 的朝向。
    //
    // ``budget``（只用于固定朝向、代价非负时）：某一层的最低代价已经不低于它时提前停止，返回这个
    // 下界（状态代价沿链不减，最终代价不会更低），``Aborted()`` 为真。
    double Solve(int wire, const std::vector<std::uint8_t>* fixed, const std::vector<double>* linear,
                 std::vector<std::uint8_t>* chosen, double budget = std::numeric_limits<double>::infinity()) {
        aborted_ = false;
        const bool bounded = budget < std::numeric_limits<double>::infinity();
        const auto& gates = problem_.wire_gates[static_cast<std::size_t>(wire)];
        const auto& is_control = problem_.wire_is_control[static_cast<std::size_t>(wire)];
        const auto& matrix = wire_matrix_[static_cast<std::size_t>(wire)];
        const int n = static_cast<int>(gates.size());
        if (static_cast<int>(matrix.size()) != n + 1) {
            throw std::invalid_argument("each wire needs one more run than gates");
        }
        // 第 j 层（运行 j 之后）的状态是 states_[offset_[j + 1], offset_[j + 2])；第 0 段是链首。
        states_.clear();
        offset_.assign(static_cast<std::size_t>(n) + 3, 0);
        states_.push_back(State{0.0, -1, 0});
        offset_[1] = 1;
        for (int j = 0; j <= n; ++j) {
            int choices[2] = {0, 1};
            int num_choices = 2;
            if (j == n) {
                choices[0] = -1;
                num_choices = 1;
            } else if (fixed != nullptr) {
                choices[0] = (*fixed)[static_cast<std::size_t>(gates[static_cast<std::size_t>(j)])] ? 1 : 0;
                num_choices = 1;
            }
            const std::size_t begin = offset_[static_cast<std::size_t>(j)];
            const std::size_t end = offset_[static_cast<std::size_t>(j) + 1];
            Layer layer{begin, end, matrix[static_cast<std::size_t>(j)], num_choices, {}, {}, {}};
            bool ordinary = true;
            for (int c = 0; c < num_choices; ++c) {
                const int right = choices[c];
                layer.right[c] = right;
                layer.axis_out[c] = right < 0 ? kNoAxis
                                              : CommutingAxis(is_control[static_cast<std::size_t>(j)] != 0, right == 1);
                layer.extra[c] = (right == 1 && linear != nullptr) ? (*linear)[static_cast<std::size_t>(j)] : 0.0;
                ordinary = ordinary && std::fabs(layer.extra[c]) < kOrdinary;
            }
            for (std::size_t si = begin; si < end; ++si) {
                ordinary = ordinary && std::fabs(states_[si].cost) < kOrdinary;
            }
            const std::size_t first = states_.size();
            if (ordinary) {
                Prune(layer);
            } else {
                PruneSorted(layer);
            }
            offset_[static_cast<std::size_t>(j) + 2] = states_.size();
            // 层内状态按代价升序，第一个就是这一层的最低代价
            if (bounded && states_[first].cost >= budget) {
                aborted_ = true;
                return states_[first].cost;
            }
        }
        const std::size_t last = offset_[static_cast<std::size_t>(n) + 1];
        std::size_t best = last;
        for (std::size_t i = last + 1; i < states_.size(); ++i) {
            if (states_[i].cost < states_[best].cost) {
                best = i;
            }
        }
        const double value = states_[best].cost;
        if (chosen != nullptr) {
            chosen->assign(static_cast<std::size_t>(n), 0);
            int idx = static_cast<int>(best - last);
            for (int j = n; j >= 0; --j) {
                const State& st = states_[offset_[static_cast<std::size_t>(j) + 1] + static_cast<std::size_t>(idx)];
                if (j < n) {
                    (*chosen)[static_cast<std::size_t>(j)] = static_cast<std::uint8_t>(options_list_[st.option].right
                                                                                       == 1);
                }
                idx = st.parent;
            }
        }
        return value;
    }

    bool Aborted() const {
        return aborted_;
    }

 private:
    struct State {
        double cost;
        int parent;            // 上一层状态的下标（层内）
        std::uint32_t option;  // 状态来自的选项（朝向、带出轴与带出角）
    };
    struct Candidate {
        double cost;
        int parent;
        std::uint32_t option;
    };
    struct Entry {
        std::uint32_t begin;
        std::uint32_t count;
    };
    struct Cursor {
        double head;        // 列表当前位置的代价
        double base;        // 状态的代价
        std::uint64_t key;  // 当前位置的生成序号：(层内状态 << 32) | (朝向选择 << 16) | 条目内选项
        std::uint32_t begin;
        std::uint32_t count;
        std::uint32_t pos;
    };
    struct Accepted {
        double cost;
        std::uint64_t key;
        int parent;
        std::uint32_t option;
    };
    // 候选代价的"正常"量级：低于它时 (base + 选项代价) + extra 随选项代价严格单调。
    static constexpr double kOrdinary = 1099511627776.0;  // 2^40

    int Score(const Matrix2& m) const {
        const std::optional<int> len = FastLength(m, model_);
        const int n = len.has_value() ? *len : kFallbackLength;
        return options_.gate_weight * n + (n > 0 ? options_.run_weight : 0);
    }

    std::uint32_t Transition(std::uint32_t option, int mat, int right, int axis_out) {
        const std::uint64_t key = TransitionTable::Key(option, mat, right, axis_out);
        std::uint32_t e = transitions_.Find(key);
        if (e == TransitionTable::kMissing) {
            const Option& o = options_list_[option];
            e = FindEntry(mat, o.right, o.axis_out, o.phi, right, axis_out);
            transitions_.Insert(key, e);
        }
        return e;
    }

    std::uint32_t FindEntry(int mat, int left, int axis_in, double phi_in, int right, int axis_out) {
        const EntryKey key{mat, left, axis_in, Bits(phi_in), right, axis_out};
        auto it = entry_ids_.find(key);
        if (it != entry_ids_.end()) {
            return it->second;
        }
        Matrix2 base = matrices_[static_cast<std::size_t>(mat)];
        if (left == 1) {
            base = Mul(base, h_);
        }
        if (axis_in != kNoAxis && phi_in != 0.0) {
            base = Mul(base, AxisRotation(axis_in, phi_in));
        }
        if (right == 1) {
            base = Mul(h_, base);
        }
        const auto r = static_cast<std::int8_t>(right);
        const auto a = static_cast<std::int8_t>(axis_out);
        Entry entry{static_cast<std::uint32_t>(options_list_.size()), 0};
        options_list_.push_back(Option{0.0, Score(base), r, a});
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
                    options_list_.push_back(Option{phi, Score(rest), r, a});
                }
            }
        }
        entry.count = static_cast<std::uint32_t>(options_list_.size()) - entry.begin;
        sorted_.resize(options_list_.size());
        for (std::uint32_t k = 0; k < entry.count; ++k) {
            sorted_[entry.begin + k] = static_cast<std::uint8_t>(k);
        }
        std::stable_sort(sorted_.begin() + entry.begin, sorted_.end(), [&](std::uint8_t a, std::uint8_t b) {
            return options_list_[entry.begin + a].cost < options_list_[entry.begin + b].cost;
        });
        const auto id = static_cast<std::uint32_t>(entries_.size());
        entries_.push_back(entry);
        entry_ids_.emplace(key, id);
        return id;
    }

    // 一层的输入：上一层状态 states_[begin, end)（按代价升序）、运行矩阵、各朝向选择的朝向 / 带出轴 /
    // 反向附加代价。
    struct Layer {
        std::size_t begin;
        std::size_t end;
        int matrix;
        int num_choices;
        std::array<int, 2> right;
        std::array<int, 2> axis_out;
        std::array<double, 2> extra;
    };

    // 每个朝向保留代价最低的 beam 个（角度去重）状态，追加到 states_。参考做法（PruneSorted）是按
    // (状态, 朝向, 选项) 的生成顺序列出全部候选 (st.cost + 选项代价) + extra，按代价稳定排序后依次
    // 取用；这里给出相同的结果：
    //   - 稳定排序 = 按 (代价, 生成序号) 排序，生成序号用 (状态, 朝向, 选项) 的字典序表示；
    //   - 取用只与同一朝向里排在前面的候选有关，所以各朝向分别处理，最后按 (代价, 生成序号) 合并；
    //   - 同一 (状态, 朝向) 的候选代价随选项代价严格单调（量级远小于 2^52 时），按条目内预排的选项
    //     顺序就是有序列表，各列表多路归并、取满即停；
    //   - 选项代价非负时，状态 s 的候选不低于 st.cost + extra（随 s 不减）：只有它不高于当前最小的
    //     列表头时才需要展开这个状态（查转移表），其余状态的候选不会被取到。
    void Prune(const Layer& layer) {
        accepted_.clear();
        for (int c = 0; c < layer.num_choices; ++c) {
            const int right = layer.right[static_cast<std::size_t>(c)];
            const int axis_out = layer.axis_out[static_cast<std::size_t>(c)];
            const double extra = layer.extra[static_cast<std::size_t>(c)];
            cursors_.clear();
            std::size_t next = layer.begin;
            const std::size_t group_first = accepted_.size();
            int count = 0;
            while (count < options_.beam) {
                std::size_t best = cursors_.size();
                for (std::size_t i = 0; i < cursors_.size(); ++i) {
                    if (best == cursors_.size() || cursors_[i].head < cursors_[best].head
                        || (cursors_[i].head == cursors_[best].head && cursors_[i].key < cursors_[best].key)) {
                        best = i;
                    }
                }
                if (next < layer.end
                    && (best == cursors_.size() || !monotone_ || states_[next].cost + extra <= cursors_[best].head)) {
                    const State& st = states_[next];
                    const std::uint32_t e = Transition(st.option, layer.matrix, right, axis_out);
                    const Entry& entry = entries_[e];
                    const std::uint8_t k = sorted_[entry.begin];
                    const auto si = static_cast<std::uint64_t>(next - layer.begin);
                    cursors_.push_back(Cursor{st.cost + options_list_[entry.begin + k].cost + extra, st.cost,
                                              (si << 32) | (static_cast<std::uint64_t>(c) << 16) | k, entry.begin,
                                              entry.count, 0});
                    ++next;
                    continue;
                }
                if (best == cursors_.size()) {
                    break;
                }
                Cursor& cur = cursors_[best];
                const double cost = cur.head;
                const std::uint64_t key = cur.key;
                const std::uint32_t option = cur.begin + static_cast<std::uint32_t>(key & 0xFFFFu);
                if (++cur.pos < cur.count) {
                    const std::uint8_t k = sorted_[cur.begin + cur.pos];
                    cur.head = cur.base + options_list_[cur.begin + k].cost + extra;
                    cur.key = (key & ~std::uint64_t{0xFFFF}) | k;
                } else {
                    cur = cursors_.back();
                    cursors_.pop_back();
                }
                const Option& o = options_list_[option];
                bool dup = false;
                for (std::size_t i = group_first; i < accepted_.size(); ++i) {
                    const Option& p = options_list_[accepted_[i].option];
                    if (p.axis_out == o.axis_out && std::abs(p.phi - o.phi) < kAngleTol) {
                        dup = true;
                        break;
                    }
                }
                if (dup) {
                    continue;
                }
                accepted_.push_back(Accepted{cost, key, static_cast<int>(key >> 32), option});
                ++count;
            }
        }
        // 各朝向的结果按 (代价, 生成序号) 合并（插入排序，最多 2·beam 个）
        for (std::size_t i = 1; i < accepted_.size(); ++i) {
            const Accepted x = accepted_[i];
            std::size_t k = i;
            while (k > 0
                   && (x.cost < accepted_[k - 1].cost
                       || (x.cost == accepted_[k - 1].cost && x.key < accepted_[k - 1].key))) {
                accepted_[k] = accepted_[k - 1];
                --k;
            }
            accepted_[k] = x;
        }
        for (const Accepted& x : accepted_) {
            states_.push_back(State{x.cost, x.parent, x.option});
        }
    }

    // 参考做法：全部候选按生成顺序列出，按代价稳定排序，依次取用（代价量级异常或 NaN 时使用）。
    void PruneSorted(const Layer& layer) {
        candidates_.clear();
        for (std::size_t si = layer.begin; si < layer.end; ++si) {
            const State& st = states_[si];
            for (int c = 0; c < layer.num_choices; ++c) {
                const std::uint32_t e = Transition(st.option, layer.matrix, layer.right[static_cast<std::size_t>(c)],
                                                   layer.axis_out[static_cast<std::size_t>(c)]);
                const Entry& entry = entries_[e];
                for (std::uint32_t k = entry.begin; k < entry.begin + entry.count; ++k) {
                    candidates_.push_back(
                        Candidate{st.cost + options_list_[k].cost + layer.extra[static_cast<std::size_t>(c)],
                                  static_cast<int>(si - layer.begin), k});
                }
            }
        }
        std::stable_sort(candidates_.begin(), candidates_.end(),
                         [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
        const std::size_t first = states_.size();
        int kept[3] = {0, 0, 0};
        for (const Candidate& x : candidates_) {
            const Option& o = options_list_[x.option];
            int& count = kept[o.right + 1];
            if (count >= options_.beam) {
                continue;
            }
            bool dup = false;
            for (std::size_t i = first; i < states_.size(); ++i) {
                const Option& p = options_list_[states_[i].option];
                if (p.right == o.right && p.axis_out == o.axis_out && std::abs(p.phi - o.phi) < kAngleTol) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            states_.push_back(State{x.cost, x.parent, x.option});
            ++count;
        }
    }

    const OrientationProblem& problem_;
    const LengthModel& model_;
    const OrientationOptions& options_;
    Matrix2 h_;
    std::vector<Matrix2> matrices_;
    std::vector<std::vector<int>> wire_matrix_;
    std::vector<Option> options_list_;
    std::vector<Entry> entries_;
    std::unordered_map<EntryKey, std::uint32_t, EntryKeyHash> entry_ids_;
    TransitionTable transitions_;
    std::vector<State> states_;
    std::vector<std::size_t> offset_;
    std::vector<Candidate> candidates_;
    std::vector<Cursor> cursors_;
    std::vector<Accepted> accepted_;
    bool aborted_ = false;
    bool monotone_ = true;              // 选项代价非负（权重非负）
    std::vector<std::uint8_t> sorted_;  // 与 options_list_ 平行：条目内按代价稳定排序后的选项序号
};

void CheckProblem(const OrientationProblem& problem) {
    const std::size_t wires = problem.wire_gates.size();
    if (problem.wire_is_control.size() != wires || problem.wire_runs.size() != wires) {
        throw std::invalid_argument("wire_gates, wire_is_control and wire_runs must have the same length");
    }
    std::vector<int> seen(static_cast<std::size_t>(problem.num_gates), 0);
    for (std::size_t w = 0; w < wires; ++w) {
        if (problem.wire_is_control[w].size() != problem.wire_gates[w].size()) {
            throw std::invalid_argument("wire_is_control must match wire_gates");
        }
        for (int g : problem.wire_gates[w]) {
            if (g < 0 || g >= problem.num_gates) {
                throw std::invalid_argument("gate index out of range");
            }
            ++seen[static_cast<std::size_t>(g)];
        }
    }
    for (int c : seen) {
        if (c != 2) {
            throw std::invalid_argument("every CX must appear on exactly two wires");
        }
    }
}

}  // namespace

double WireCost(const OrientationProblem& problem, int wire, const std::vector<std::uint8_t>& flip,
                const LengthModel& model, const OrientationOptions& options) {
    ChainSolver solver(problem, model, options);
    return solver.Solve(wire, &flip, nullptr, nullptr);
}

double OrientationEnergy(const OrientationProblem& problem, const std::vector<std::uint8_t>& flip,
                         const LengthModel& model, const OrientationOptions& options) {
    ChainSolver solver(problem, model, options);
    double total = 0.0;
    for (int w = 0; w < static_cast<int>(problem.wire_gates.size()); ++w) {
        total += solver.Solve(w, &flip, nullptr, nullptr);
    }
    return total;
}

OrientationResult SolveCxOrientation(const OrientationProblem& problem, const LengthModel& model,
                                     const OrientationOptions& options) {
    CheckProblem(problem);
    const int num_wires = static_cast<int>(problem.wire_gates.size());
    const auto ng = static_cast<std::size_t>(problem.num_gates);
    ChainSolver solver(problem, model, options);

    // 一条线的固定朝向代价只依赖这条线上的朝向位：每条线缓存最近几组位向量的代价
    // （相邻迭代的两份副本大多只在少数线上不同）。
    constexpr std::size_t kWireCache = 4;
    std::vector<std::vector<std::pair<std::vector<std::uint8_t>, double>>> wire_cache(
        static_cast<std::size_t>(num_wires));
    // 固定朝向下的代价都是非负整数（double 表示，求和精确、与次序无关）：候选朝向只需判断是否
    // 低于当前最优，``energy(flip, bound)`` 在总代价低于 bound 时返回精确值，否则返回某个不低于
    // bound 的值（先累加缓存命中的线，其余的线逐条求解，部分和或链内某层的下界达到 bound 即停）。
    const bool monotone = options.gate_weight >= 0 && options.run_weight >= 0;
    std::vector<std::uint8_t> bits;
    std::vector<int> pending;
    auto wire_bits = [&](int w, const std::vector<std::uint8_t>& flip) {
        const auto& gates = problem.wire_gates[static_cast<std::size_t>(w)];
        bits.resize(gates.size());
        for (std::size_t j = 0; j < gates.size(); ++j) {
            bits[j] = flip[static_cast<std::size_t>(gates[j])];
        }
    };
    auto energy = [&](const std::vector<std::uint8_t>& flip, double bound) {
        if (!monotone) {
            bound = std::numeric_limits<double>::infinity();
        }
        double total = 0.0;
        pending.clear();
        for (int w = 0; w < num_wires; ++w) {
            wire_bits(w, flip);
            bool hit = false;
            for (const auto& [key, cost] : wire_cache[static_cast<std::size_t>(w)]) {
                if (key == bits) {
                    total += cost;
                    hit = true;
                    break;
                }
            }
            if (!hit) {
                pending.push_back(w);
            }
        }
        for (int w : pending) {
            if (total >= bound) {
                return total;
            }
            const double cost = solver.Solve(w, &flip, nullptr, nullptr, bound - total);
            if (solver.Aborted()) {
                return total + cost;
            }
            wire_bits(w, flip);
            auto& cache = wire_cache[static_cast<std::size_t>(w)];
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
    result.initial_energy = energy(result.flip, std::numeric_limits<double>::infinity());
    result.energy = result.initial_energy;
    if (ng == 0) {
        return result;
    }
    {
        const std::vector<std::uint8_t> all(ng, 1);
        const double e = energy(all, result.energy);
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
            const double e = energy(*cand, result.energy);
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

}  // namespace quspect::native::transpiler
