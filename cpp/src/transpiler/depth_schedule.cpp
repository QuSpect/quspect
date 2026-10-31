// 对易感知的深度调度（quspect.transpiler._schedule 的原生实现，逐位一致）。
#include "quspect_native/transpiler/depth_schedule.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quspect::native::transpiler {
namespace {

constexpr int kNoAxis = -1;
// 每个调度的双向调整轮数（与 Python 的 _ROUNDS 相同）
constexpr int kRounds = 2;

using Score = std::pair<int, int>;

// Sgs 的最早空闲时隙查询：每个资源一张"下一个空闲时隙"并查集（占用的时隙 t 指向 t + 1，
// 查询时路径减半），与逐个时隙试探得到同一个时隙，但跳过成段已占用的时隙。
//
// 表项带代编号：Reset 只把代编号加一（旧代的表项视为空闲），表在同一次 BestDepthOrder 的
// 各次 Sgs 之间复用，不必每次重新填充整个时间范围。
class FreeSlots {
 public:
    void Reset(int num_resources) {
        if (static_cast<int>(slots_.size()) < num_resources) {
            slots_.resize(num_resources);
        }
        if (++gen_ == 0) {  // 代编号回绕：清空所有表项
            for (auto& v : slots_) {
                v.clear();
            }
            gen_ = 1;
        }
    }

    // 资源 q 上不早于 t 的第一个空闲时隙。
    std::int64_t Find(int q, std::int64_t t) {
        std::vector<Slot>& e = slots_[q];
        const auto size = static_cast<std::int64_t>(e.size());
        std::int64_t x = t;
        for (;;) {
            if (x >= size || e[x].gen != gen_) {
                return x;
            }
            const std::int64_t y = e[x].next;
            if (y >= size || e[y].gen != gen_) {
                return y;
            }
            const std::int32_t z = e[y].next;
            e[x].next = z;
            x = z;
        }
    }

    void Occupy(int q, std::int64_t t) {
        std::vector<Slot>& e = slots_[q];
        if (static_cast<std::int64_t>(e.size()) <= t) {
            e.resize(static_cast<std::size_t>(std::max<std::int64_t>(t + 1, 2 * static_cast<std::int64_t>(e.size()))));
        }
        e[t].next = static_cast<std::int32_t>(t + 1);
        e[t].gen = gen_;
    }

 private:
    struct Slot {
        std::int32_t next = 0;
        std::uint32_t gen = 0;
    };
    std::vector<std::vector<Slot>> slots_;
    std::uint32_t gen_ = 0;
};

// Sgs 的就绪堆：按 (−prio, 节点) 取最小。−prio 的取值范围在 2^32 以内时打包成一个 64 位整数
// 放进四叉堆（键互不相同，出堆次序与按对比较的二叉堆相同），否则按对比较。
class ReadyHeap {
 public:
    ReadyHeap(const std::vector<std::int64_t>& prio, const std::vector<int>& nodes) {
        std::int64_t lo = 0;
        std::int64_t hi = 0;
        bool first = true;
        for (int g : nodes) {
            const std::int64_t k = -prio[g];
            if (first || k < lo) {
                lo = k;
            }
            if (first || k > hi) {
                hi = k;
            }
            first = false;
        }
        // hi ≥ lo，无符号差即真实差值（不会溢出）
        packed_ = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) < (std::uint64_t{1} << 32)
                  && nodes.size() < (std::size_t{1} << 31);
        base_ = lo;
        if (packed_) {
            keys_.reserve(nodes.size());
        }
    }

    void Push(std::int64_t neg_prio, int node) {
        if (packed_) {
            // 四叉最小堆：上浮
            const std::uint64_t key =
                (static_cast<std::uint64_t>(neg_prio - base_) << 32) | static_cast<std::uint32_t>(node);
            std::size_t i = keys_.size();
            keys_.push_back(key);
            while (i > 0) {
                const std::size_t parent = (i - 1) / 4;
                if (keys_[parent] <= key) {
                    break;
                }
                keys_[i] = keys_[parent];
                i = parent;
            }
            keys_[i] = key;
        } else {
            pairs_.emplace_back(neg_prio, node);
            std::push_heap(pairs_.begin(), pairs_.end(), std::greater<std::pair<std::int64_t, int>>());
        }
    }

    bool Empty() const {
        return packed_ ? keys_.empty() : pairs_.empty();
    }

    int Pop() {
        if (packed_) {
            const int node = static_cast<int>(keys_.front() & 0xffffffffu);
            const std::uint64_t key = keys_.back();
            keys_.pop_back();
            const std::size_t size = keys_.size();
            if (size > 0) {
                // 下沉
                std::size_t i = 0;
                for (;;) {
                    const std::size_t first = 4 * i + 1;
                    if (first >= size) {
                        break;
                    }
                    std::size_t best = first;
                    const std::size_t last = std::min(first + 4, size);
                    for (std::size_t c = first + 1; c < last; ++c) {
                        if (keys_[c] < keys_[best]) {
                            best = c;
                        }
                    }
                    if (keys_[best] >= key) {
                        break;
                    }
                    keys_[i] = keys_[best];
                    i = best;
                }
                keys_[i] = key;
            }
            return node;
        }
        std::pop_heap(pairs_.begin(), pairs_.end(), std::greater<std::pair<std::int64_t, int>>());
        const int node = pairs_.back().second;
        pairs_.pop_back();
        return node;
    }

 private:
    bool packed_ = false;
    std::int64_t base_ = 0;
    std::vector<std::uint64_t> keys_;
    std::vector<std::pair<std::int64_t, int>> pairs_;
};

// 一个输入序列上各比特的对易组与（可选）双比特门链的先后约束。
//
// 组员与链后继按 CSR 存放（每个组 / 节点占一段连续下标，顺序与逐个追加相同）。
class Groups {
 public:
    Groups(const ScheduleProblem& p, int num_resources, const std::vector<int>& seq, bool chain)
        : p_(p), seq_(seq), num_resources_(num_resources) {
        const int n = static_cast<int>(p.offsets.size()) - 1;
        const int slots = p.offsets.back();
        group_.assign(slots, 0);
        chain_pred_.assign(slots, -1);
        std::vector<int> last_group(num_resources, -1);
        std::vector<int> last_axis(num_resources, kNoAxis);
        std::vector<int> last_two(num_resources, -1);
        std::vector<int> group_size;
        std::vector<int> succ_count(n, 0);
        for (int g : seq_) {
            for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
                const int q = p.resources[s];
                const int a = p.axes[s];
                int gid;
                if (last_group[q] >= 0 && a != kNoAxis && last_axis[q] == a) {
                    gid = last_group[q];
                    ++group_size[gid];
                } else {
                    gid = static_cast<int>(group_size.size());
                    group_size.push_back(1);
                    next_.push_back(-1);
                    if (last_group[q] >= 0) {
                        next_[last_group[q]] = gid;
                    }
                    last_group[q] = gid;
                    last_axis[q] = a;
                }
                group_[s] = gid;
                if (chain && p.two[g]) {
                    const int prev = last_two[q];
                    if (prev >= 0 && group_[SlotOf(prev, q)] == gid) {
                        chain_pred_[s] = prev;
                        ++succ_count[prev];
                    }
                    last_two[q] = g;
                }
            }
        }
        const int num_groups = static_cast<int>(group_size.size());
        member_begin_.assign(num_groups + 1, 0);
        for (int gid = 0; gid < num_groups; ++gid) {
            member_begin_[gid + 1] = member_begin_[gid] + group_size[gid];
        }
        succ_begin_.assign(n + 1, 0);
        for (int g = 0; g < n; ++g) {
            succ_begin_[g + 1] = succ_begin_[g] + succ_count[g];
        }
        members_.resize(member_begin_.back());
        succ_.resize(succ_begin_.back());
        std::vector<int> member_fill(member_begin_.begin(), member_begin_.end() - 1);
        std::vector<int> succ_fill(succ_begin_.begin(), succ_begin_.end() - 1);
        for (int g : seq_) {
            for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
                members_[member_fill[group_[s]]++] = g;
                if (chain_pred_[s] >= 0) {
                    succ_[succ_fill[chain_pred_[s]]++] = g;
                }
            }
        }
        prev_.assign(num_groups, -1);
        for (int gid = 0; gid < num_groups; ++gid) {
            if (next_[gid] >= 0) {
                prev_[next_[gid]] = gid;
            }
        }
    }

    // 关键路径优先级（剩余时间的下界；组的尾部还受同一比特上组员总时长的约束）。
    std::vector<std::int64_t> BottomLevels(const std::vector<int>& dur) const {
        std::vector<std::int64_t> tail(prev_.size(), 0);
        std::vector<std::int64_t> bl(succ_begin_.size() - 1, 0);
        for (auto it = seq_.rbegin(); it != seq_.rend(); ++it) {
            const int g = *it;
            std::int64_t b = 0;
            for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                const int nx = next_[group_[s]];
                if (nx >= 0 && tail[nx] > b) {
                    b = tail[nx];
                }
            }
            for (int k = succ_begin_[g]; k < succ_begin_[g + 1]; ++k) {
                const int succ = succ_[k];
                if (bl[succ] > b) {
                    b = bl[succ];
                }
            }
            bl[g] = dur[g] + b;
            for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                const int gid = group_[s];
                if (members_[member_begin_[gid]] != g) {
                    continue;
                }
                const int nx = next_[gid];
                std::int64_t t = nx >= 0 ? tail[nx] : 0;
                std::int64_t worst = 0;
                for (int k = member_begin_[gid]; k < member_begin_[gid + 1]; ++k) {
                    const int h = members_[k];
                    t += dur[h];
                    worst = std::max(worst, bl[h]);
                }
                tail[gid] = std::max(t, worst);
            }
        }
        return bl;
    }

    // 插入式串行调度生成；返回按（开始时间、时长、调度次序）排列的节点顺序。
    std::vector<int> Sgs(const std::vector<std::int64_t>& prio, const std::vector<int>& dur, FreeSlots* free) const {
        const int n = static_cast<int>(succ_begin_.size()) - 1;
        const int num_groups = static_cast<int>(prev_.size());
        std::vector<int> remaining(num_groups);
        for (int gid = 0; gid < num_groups; ++gid) {
            remaining[gid] = member_begin_[gid + 1] - member_begin_[gid];
        }
        std::vector<std::int64_t> group_fin(num_groups, 0);
        std::vector<int> pending(n, 0);
        ReadyHeap heap(prio, seq_);
        for (int g : seq_) {
            int c = 0;
            for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                if (prev_[group_[s]] >= 0) {
                    ++c;
                }
                if (chain_pred_[s] >= 0) {
                    ++c;
                }
            }
            pending[g] = c;
            if (c == 0) {
                heap.Push(-prio[g], g);
            }
        }
        free->Reset(num_resources_);
        std::vector<std::int64_t> start(n, 0);
        std::vector<std::int64_t> fin(n, 0);
        std::vector<int> popped;
        popped.reserve(n);
        while (!heap.Empty()) {
            const int g = heap.Pop();
            std::int64_t est = 0;
            for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                const int pg = prev_[group_[s]];
                if (pg >= 0 && group_fin[pg] > est) {
                    est = group_fin[pg];
                }
                const int c = chain_pred_[s];
                if (c >= 0 && fin[c] > est) {
                    est = fin[c];
                }
            }
            std::int64_t t = est;
            if (dur[g] != 0) {
                // 所有资源都空闲的最早时隙 ≥ est
                for (bool moved = true; moved;) {
                    moved = false;
                    for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                        const std::int64_t f = free->Find(p_.resources[s], t);
                        if (f != t) {
                            t = f;
                            moved = true;
                        }
                    }
                }
                for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                    free->Occupy(p_.resources[s], t);
                }
            }
            start[g] = t;
            fin[g] = t + dur[g];
            popped.push_back(g);
            for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
                const int gid = group_[s];
                if (fin[g] > group_fin[gid]) {
                    group_fin[gid] = fin[g];
                }
                if (--remaining[gid] == 0) {
                    const int nx = next_[gid];
                    if (nx >= 0) {
                        for (int k = member_begin_[nx]; k < member_begin_[nx + 1]; ++k) {
                            const int h = members_[k];
                            if (--pending[h] == 0) {
                                heap.Push(-prio[h], h);
                            }
                        }
                    }
                }
            }
            for (int k = succ_begin_[g]; k < succ_begin_[g + 1]; ++k) {
                const int succ = succ_[k];
                if (--pending[succ] == 0) {
                    heap.Push(-prio[succ], succ);
                }
            }
        }
        if (static_cast<int>(popped.size()) != n) {
            throw std::logic_error("BestDepthOrder: cyclic precedence");
        }
        return SortedBySchedule(popped, start, dur);
    }

 private:
    int SlotOf(int g, int q) const {
        for (int s = p_.offsets[g]; s < p_.offsets[g + 1]; ++s) {
            if (p_.resources[s] == q) {
                return s;
            }
        }
        throw std::logic_error("BestDepthOrder: resource not found");
    }

    // popped（调度次序）按（开始时间、时长、调度次序）排序。键的取值范围不大时用稳定的
    // 计数排序（popped 已按调度次序排列），否则比较排序；两者结果相同（键互不相同）。
    static std::vector<int> SortedBySchedule(const std::vector<int>& popped, const std::vector<std::int64_t>& start,
                                             const std::vector<int>& dur) {
        const auto n = static_cast<std::int64_t>(popped.size());
        std::int64_t max_start = 0;
        int max_dur = 0;
        bool small = true;
        for (int g : popped) {
            if (start[g] < 0 || dur[g] < 0) {
                small = false;
                break;
            }
            max_start = std::max(max_start, start[g]);
            max_dur = std::max(max_dur, dur[g]);
        }
        const std::int64_t width = static_cast<std::int64_t>(max_dur) + 1;
        small = small && max_dur < (1 << 20) && max_start < (std::int64_t{1} << 40)
                && (max_start + 1) * width <= 8 * n + 1024;
        if (!small) {
            std::vector<int> order(popped);
            std::vector<int> pos(start.size(), 0);
            for (std::size_t k = 0; k < popped.size(); ++k) {
                pos[popped[k]] = static_cast<int>(k);
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                if (start[a] != start[b]) {
                    return start[a] < start[b];
                }
                if (dur[a] != dur[b]) {
                    return dur[a] < dur[b];
                }
                return pos[a] < pos[b];
            });
            return order;
        }
        const std::int64_t keys = (max_start + 1) * width;
        std::vector<int> count(static_cast<std::size_t>(keys) + 1, 0);
        for (int g : popped) {
            ++count[static_cast<std::size_t>(start[g] * width + dur[g]) + 1];
        }
        for (std::int64_t k = 0; k < keys; ++k) {
            count[k + 1] += count[k];
        }
        std::vector<int> order(popped.size());
        for (int g : popped) {
            order[count[static_cast<std::size_t>(start[g] * width + dur[g])]++] = g;
        }
        return order;
    }

    const ScheduleProblem& p_;
    std::vector<int> seq_;
    int num_resources_;
    std::vector<int> group_;
    std::vector<int> member_begin_;
    std::vector<int> members_;
    std::vector<int> next_;
    std::vector<int> prev_;
    std::vector<int> chain_pred_;
    std::vector<int> succ_begin_;
    std::vector<int> succ_;
};

// 按顺序 ASAP 排布的开始 / 结束时间。
void Asap(const ScheduleProblem& p, int num_resources, const std::vector<int>& order, const std::vector<int>& dur,
          std::vector<std::int64_t>* start, std::vector<std::int64_t>* fin) {
    const int n = static_cast<int>(p.offsets.size()) - 1;
    std::vector<std::int64_t> avail(num_resources, 0);
    start->assign(n, 0);
    fin->assign(n, 0);
    for (int g : order) {
        std::int64_t t = 0;
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            t = std::max(t, avail[p.resources[s]]);
        }
        (*start)[g] = t;
        (*fin)[g] = t + dur[g];
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            avail[p.resources[s]] = (*fin)[g];
        }
    }
}

// 只计 weight 为 1 的节点的深度。
int Metric(const ScheduleProblem& p, int num_resources, const std::vector<int>& order, const std::vector<int>& weight) {
    std::vector<int> level(num_resources, 0);
    int deepest = 0;
    for (int g : order) {
        if (weight[g] == 0) {
            continue;
        }
        int lv = 0;
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            lv = std::max(lv, level[p.resources[s]]);
        }
        ++lv;
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            level[p.resources[s]] = lv;
        }
        deepest = std::max(deepest, lv);
    }
    return deepest;
}

}  // namespace

std::vector<int> BestDepthOrder(const ScheduleProblem& p) {
    if (p.offsets.empty() || p.offsets.front() != 0) {
        throw std::invalid_argument("BestDepthOrder: offsets must start with 0.");
    }
    const int n = static_cast<int>(p.offsets.size()) - 1;
    if (p.two.size() != static_cast<std::size_t>(n) || p.counted.size() != static_cast<std::size_t>(n)
        || p.resources.size() != static_cast<std::size_t>(p.offsets.back()) || p.axes.size() != p.resources.size()) {
        throw std::invalid_argument("BestDepthOrder: inconsistent array sizes.");
    }
    for (int g = 0; g < n; ++g) {
        if (p.offsets[g + 1] < p.offsets[g]) {
            throw std::invalid_argument("BestDepthOrder: offsets must be non-decreasing.");
        }
    }
    int num_resources = 0;
    for (int q : p.resources) {
        if (q < 0) {
            throw std::invalid_argument("BestDepthOrder: negative resource.");
        }
        num_resources = std::max(num_resources, q + 1);
    }
    std::vector<int> dur_all(n);
    std::vector<int> dur_two(n);
    for (int g = 0; g < n; ++g) {
        dur_all[g] = p.counted[g] ? 1 : 0;
        dur_two[g] = p.two[g] ? 1 : 0;
    }
    // 各次 Sgs 共用的空闲时隙表
    FreeSlots free_slots;
    auto depth_score = [&](const std::vector<int>& order) -> Score {
        return {Metric(p, num_resources, order, dur_all), Metric(p, num_resources, order, dur_two)};
    };
    auto two_score = [&](const std::vector<int>& order) -> Score {
        const Score s = depth_score(order);
        return {s.second, s.first};
    };
    auto make = [&](const std::vector<int>& seq, bool chain) { return Groups(p, num_resources, seq, chain); };
    auto justify = [&](const Groups& fwd, const Groups& bwd, std::vector<int> order, const std::vector<int>& dur,
                       const std::function<Score(const std::vector<int>&)>& score) {
        std::vector<int> best = order;
        Score best_score = score(order);
        std::vector<int> cur = std::move(order);
        std::vector<std::int64_t> start;
        std::vector<std::int64_t> fin;
        for (int round = 0; round < kRounds; ++round) {
            Asap(p, num_resources, cur, dur, &start, &fin);
            cur = bwd.Sgs(fin, dur, &free_slots);
            std::reverse(cur.begin(), cur.end());
            Asap(p, num_resources, cur, dur, &start, &fin);
            for (auto& s : start) {
                s = -s;
            }
            cur = fwd.Sgs(start, dur, &free_slots);
            const Score sc = score(cur);
            if (sc < best_score) {
                best = cur;
                best_score = sc;
            } else {
                break;
            }
        }
        return best;
    };
    auto reversed = [](std::vector<int> v) {
        std::reverse(v.begin(), v.end());
        return v;
    };
    auto total = [&](const std::vector<int>& seq, bool chain) {
        const Groups fwd = make(seq, chain);
        const Groups bwd = make(reversed(seq), chain);
        std::vector<int> first = fwd.Sgs(fwd.BottomLevels(dur_all), dur_all, &free_slots);
        return justify(fwd, bwd, std::move(first), dur_all, depth_score);
    };

    std::vector<int> seq(n);
    for (int g = 0; g < n; ++g) {
        seq[g] = g;
    }
    const Score base = depth_score(seq);
    std::vector<int> best = seq;
    Score best_score = base;
    std::vector<std::vector<int>> cands;
    cands.push_back(total(seq, false));
    cands.push_back(total(seq, true));
    {
        const Groups free = make(seq, false);
        const Groups free_r = make(reversed(seq), false);
        std::vector<int> by_two = free.Sgs(free.BottomLevels(dur_two), dur_two, &free_slots);
        by_two = justify(free, free_r, std::move(by_two), dur_two, two_score);
        cands.push_back(total(by_two, true));
    }
    for (const auto& cand : cands) {
        const Score sc = depth_score(cand);
        if (sc.second <= base.second && sc < best_score) {
            best = cand;
            best_score = sc;
        }
    }
    return best;
}

ScheduleProblem Permuted(const ScheduleProblem& p, const std::vector<int>& order) {
    const int n = static_cast<int>(p.offsets.size()) - 1;
    if (static_cast<int>(order.size()) != n) {
        throw std::invalid_argument("Permuted: order must be a permutation.");
    }
    std::vector<std::uint8_t> seen(n, 0);
    ScheduleProblem out;
    out.offsets.reserve(p.offsets.size());
    out.offsets.push_back(0);
    out.resources.reserve(p.resources.size());
    out.axes.reserve(p.axes.size());
    for (int g : order) {
        if (g < 0 || g >= n || seen[g] != 0) {
            throw std::invalid_argument("Permuted: order must be a permutation.");
        }
        seen[g] = 1;
        out.resources.insert(out.resources.end(), p.resources.begin() + p.offsets[g],
                             p.resources.begin() + p.offsets[g + 1]);
        out.axes.insert(out.axes.end(), p.axes.begin() + p.offsets[g], p.axes.begin() + p.offsets[g + 1]);
        out.offsets.push_back(static_cast<int>(out.resources.size()));
        out.two.push_back(p.two[g]);
        out.counted.push_back(p.counted[g]);
    }
    return out;
}

std::pair<int, int> ReportedDepths(const ScheduleProblem& p, const std::vector<int>& order, int num_qubits) {
    std::vector<int> level(std::max(num_qubits, 0), 0);
    std::vector<int> level2(std::max(num_qubits, 0), 0);
    int deepest = 0;
    int deepest2 = 0;
    for (int g : order) {
        if (p.counted[g] == 0) {
            continue;
        }
        int lv = 0;
        int lv2 = 0;
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            const int q = p.resources[s];
            if (q < num_qubits) {
                lv = std::max(lv, level[q]);
                lv2 = std::max(lv2, level2[q]);
            }
        }
        ++lv;
        ++lv2;
        for (int s = p.offsets[g]; s < p.offsets[g + 1]; ++s) {
            const int q = p.resources[s];
            if (q < num_qubits) {
                level[q] = lv;
                if (p.two[g] != 0) {
                    level2[q] = lv2;
                }
            }
        }
        deepest = std::max(deepest, lv);
        if (p.two[g] != 0) {
            deepest2 = std::max(deepest2, lv2);
        }
    }
    return {deepest, deepest2};
}

namespace {

using Complex = std::complex<double>;
using Mat4 = std::array<Complex, 16>;

Mat4 Multiply(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Complex acc = 0.0;
            for (int k = 0; k < 4; ++k) {
                acc += a[4 * i + k] * b[4 * k + j];
            }
            out[4 * i + j] = acc;
        }
    }
    return out;
}

// 4×4 嵌入：单比特门在块的第 slot 个比特上（0 为高位），两比特门 slot 2 表示比特顺序相反。
Mat4 Embedded(const BlockInput& in, int instr, int slot) {
    const Complex* m = in.matrices.data() + in.matrix_offset[instr];
    Mat4 out{};
    if (in.kind[instr] == 1) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int hi_i = i >> 1;
                const int lo_i = i & 1;
                const int hi_j = j >> 1;
                const int lo_j = j & 1;
                if (slot == 0) {
                    out[4 * i + j] = lo_i == lo_j ? m[2 * hi_i + hi_j] : Complex(0.0);
                } else {
                    out[4 * i + j] = hi_i == hi_j ? m[2 * lo_i + lo_j] : Complex(0.0);
                }
            }
        }
        return out;
    }
    static constexpr int kSwap[4] = {0, 2, 1, 3};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[4 * i + j] = slot == 2 ? m[4 * kSwap[i] + kSwap[j]] : m[4 * i + j];
        }
    }
    return out;
}

// 块酉矩阵在两个比特上的对易 Pauli 轴（Z、X、Y 中第一个对易者，−1 为无）。
std::pair<int, int> BlockAxes(const Mat4& u) {
    static const Complex kI(0.0, 1.0);
    const std::array<std::array<Complex, 4>, 3> paulis = {{
        {Complex(1.0), Complex(0.0), Complex(0.0), Complex(-1.0)},
        {Complex(0.0), Complex(1.0), Complex(1.0), Complex(0.0)},
        {Complex(0.0), -kI, kI, Complex(0.0)},
    }};
    int found[2] = {kNoAxis, kNoAxis};
    for (int slot = 0; slot < 2; ++slot) {
        for (int code = 0; code < 3; ++code) {
            Mat4 op{};
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    const int hi_i = i >> 1;
                    const int lo_i = i & 1;
                    const int hi_j = j >> 1;
                    const int lo_j = j & 1;
                    const auto& p = paulis[code];
                    if (slot == 0) {
                        op[4 * i + j] = lo_i == lo_j ? p[2 * hi_i + hi_j] : Complex(0.0);
                    } else {
                        op[4 * i + j] = hi_i == hi_j ? p[2 * lo_i + lo_j] : Complex(0.0);
                    }
                }
            }
            const Mat4 left = Multiply(op, u);
            const Mat4 right = Multiply(u, op);
            double err = 0.0;
            for (int k = 0; k < 16; ++k) {
                err = std::max(err, std::abs(left[k] - right[k]));
            }
            if (err < 1e-12) {
                found[slot] = code;
                break;
            }
        }
    }
    return {found[0], found[1]};
}

}  // namespace

std::vector<int> BlockOrder(const ScheduleProblem& p, const BlockInput& in) {
    const int n = static_cast<int>(in.kind.size());
    if (static_cast<int>(in.qubit_offsets.size()) != n + 1 || static_cast<int>(in.matrix_offset.size()) != n
        || static_cast<int>(p.offsets.size()) != n + 1) {
        throw std::invalid_argument("BlockOrder: inconsistent array sizes.");
    }
    for (int i = 0; i < n; ++i) {
        const int nq = in.qubit_offsets[i + 1] - in.qubit_offsets[i];
        if ((in.kind[i] == 1 && nq != 1) || (in.kind[i] == 2 && nq != 2)) {
            throw std::invalid_argument("BlockOrder: gate kind does not match its qubits.");
        }
        const int size = in.kind[i] == 1 ? 4 : 16;
        if ((in.kind[i] == 1 || in.kind[i] == 2) && in.matrix_offset[i] >= 0
            && in.matrix_offset[i] + size > static_cast<int>(in.matrices.size())) {
            throw std::invalid_argument("BlockOrder: matrix offset out of range.");
        }
    }
    int width = 0;
    for (int q : in.qubits) {
        if (q < 0) {
            throw std::invalid_argument("BlockOrder: negative qubit.");
        }
        width = std::max(width, q + 1);
    }
    // --- 划分（与 Python 的 _blocks 相同） ---
    std::vector<std::vector<int>> nodes;
    std::vector<std::pair<int, int>> pair;  // 块的比特对；单条指令为 (−1, −1)
    std::vector<int> open_block(width, -1);
    std::vector<std::vector<int>> pending(width);
    std::vector<std::uint8_t> seen(width, 0);
    auto close = [&](int q) {
        const int b = open_block[q];
        if (b < 0) {
            return;
        }
        open_block[q] = -1;
        const int other = pair[b].first == q ? pair[b].second : pair[b].first;
        if (open_block[other] == b) {
            open_block[other] = -1;
        }
    };
    auto flush = [&](int q) {
        for (int i : pending[q]) {
            nodes.push_back({i});
            pair.emplace_back(-1, -1);
        }
        pending[q].clear();
    };
    for (int idx = 0; idx < n; ++idx) {
        const int* qs = in.qubits.data() + in.qubit_offsets[idx];
        const int nq = in.qubit_offsets[idx + 1] - in.qubit_offsets[idx];
        for (int k = 0; k < nq; ++k) {
            seen[qs[k]] = 1;
        }
        if (in.kind[idx] == 1) {
            pending[qs[0]].push_back(idx);
            continue;
        }
        if (in.kind[idx] == 2) {
            const int a = qs[0];
            const int b = qs[1];
            const int blk = open_block[a];
            if (blk >= 0 && open_block[b] == blk) {
                std::vector<int> absorbed = pending[a];
                absorbed.insert(absorbed.end(), pending[b].begin(), pending[b].end());
                std::sort(absorbed.begin(), absorbed.end());
                pending[a].clear();
                pending[b].clear();
                nodes[blk].insert(nodes[blk].end(), absorbed.begin(), absorbed.end());
                nodes[blk].push_back(idx);
                continue;
            }
            for (int q : {a, b}) {
                close(q);
                flush(q);
            }
            nodes.push_back({idx});
            pair.emplace_back(a, b);
            open_block[a] = open_block[b] = static_cast<int>(nodes.size()) - 1;
            continue;
        }
        if (in.kind[idx] == 3) {
            for (int q = 0; q < width; ++q) {
                if (seen[q] != 0) {
                    close(q);
                    flush(q);
                }
            }
        } else {
            for (int k = 0; k < nq; ++k) {
                close(qs[k]);
                flush(qs[k]);
            }
        }
        nodes.push_back({idx});
        pair.emplace_back(-1, -1);
    }
    for (int q = 0; q < width; ++q) {
        flush(q);
    }
    // --- 节点问题 ---
    ScheduleProblem np;
    np.offsets.push_back(0);
    bool useful = false;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
        const std::vector<int>& members = nodes[k];
        if (members.size() == 1) {
            const int g = members[0];
            np.resources.insert(np.resources.end(), p.resources.begin() + p.offsets[g],
                                p.resources.begin() + p.offsets[g + 1]);
            np.axes.insert(np.axes.end(), p.axes.begin() + p.offsets[g], p.axes.begin() + p.offsets[g + 1]);
            np.two.push_back(p.two[g]);
            np.counted.push_back(p.counted[g]);
        } else {
            const int a = pair[k].first;
            const int b = pair[k].second;
            bool known = true;
            Mat4 u{};
            bool first = true;
            for (int i : members) {
                if (in.matrix_offset[i] < 0) {
                    known = false;
                    break;
                }
                const int* qs = in.qubits.data() + in.qubit_offsets[i];
                int slot;
                if (in.kind[i] == 1) {
                    slot = qs[0] == a ? 0 : 1;
                } else {
                    slot = qs[0] == a && qs[1] == b ? 1 : 2;
                }
                const Mat4 m = Embedded(in, i, slot);
                u = first ? m : Multiply(m, u);
                first = false;
            }
            std::pair<int, int> axes(kNoAxis, kNoAxis);
            if (known) {
                axes = BlockAxes(u);
            }
            useful = useful || axes.first != kNoAxis || axes.second != kNoAxis;
            np.resources.push_back(a);
            np.resources.push_back(b);
            np.axes.push_back(axes.first);
            np.axes.push_back(axes.second);
            np.two.push_back(1);
            np.counted.push_back(1);
        }
        np.offsets.push_back(static_cast<int>(np.resources.size()));
    }
    if (!useful) {
        return {};
    }
    const std::vector<int> order = BestDepthOrder(np);
    bool same = true;
    for (std::size_t k = 0; k < order.size(); ++k) {
        if (order[k] != static_cast<int>(k)) {
            same = false;
            break;
        }
    }
    if (same) {
        return {};
    }
    std::vector<int> out;
    out.reserve(n);
    for (int k : order) {
        out.insert(out.end(), nodes[k].begin(), nodes[k].end());
    }
    return out;
}

}  // namespace quspect::native::transpiler
