// 路由的精确窗口搜索（A*）与大邻域搜索（LNS），见 route_lns.hpp。
//
// 代价模型（"合并项"）：
//   每个物理比特上最后一个操作若与另一个物理比特上的最后一个操作是同一个操作 o（o 落在这对比特上），
//   就记一个"开放项"(x, y)：o 是两比特节点时记它的镜像代价（之后紧跟的同对 SWAP 只计这个代价），
//   o 是 SWAP 时记它目前的代价 p（之后紧跟的同对两比特节点 m 把它降到 min(p, mirror(m))）。
//   任何别的操作触及 x 或 y 时该项关闭。这样每个 SWAP 的代价恰好是 min(前合并, 后合并, 3)。
//
// A* 状态 = (布局, 已执行的窗口节点, 开放项)；节点一旦可执行就立即执行（贪心排干）。
// 下界：
//   * SWAP 数下界 hs：一次 SWAP 只移动两个逻辑比特各一步，一个比特的移动至多让它所在的一个节点的
//     距离减 1。取剩余节点中比特互不相交的一组 M（按拓扑序贪心），再加上出口受约束且不在 M 中的
//     比特到出口的距离：hs >= ceil(Σ / 2)；单个节点 / 单个比特的需求也是下界。
//   * 代价下界：hs 个 SWAP 至多各用掉一个"合并机会"（开放节点项、剩余节点、出口后的首个节点），
//     取最便宜的 hs 个；已开放 SWAP 项可能的退款、出口后首个 SWAP 的代价下界也计入。

#include "quspect_native/transpiler/route_lns.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace quspect::native::transpiler {

namespace {

constexpr std::int64_t kInfCost = std::numeric_limits<std::int64_t>::max() / 4;
constexpr std::uint8_t kNodeItem = 1;
constexpr std::uint8_t kSwapItem = 2;
constexpr int kSwapCost = 3;
constexpr int kThreeQubitWeight = 6;
constexpr std::int64_t kNoSecondary = std::numeric_limits<std::int64_t>::min();

std::uint64_t SplitMix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// 每个物理比特上的开放项（对称存储在两端）。
struct ItemTable {
    std::vector<int> partner;
    std::vector<std::uint8_t> kind;
    std::vector<std::uint8_t> val;

    void Init(int n) {
        partner.assign(n, -1);
        kind.assign(n, 0);
        val.assign(n, 0);
    }
    void Close(int q) {
        const int p = partner[q];
        if (p >= 0) {
            partner[p] = -1;
            partner[q] = -1;
        }
    }
    void Set(int x, int y, std::uint8_t k, std::uint8_t v) {
        partner[x] = y;
        partner[y] = x;
        kind[x] = kind[y] = k;
        val[x] = val[y] = v;
    }
};

// SWAP (x, y)：返回其暂定代价。
int ApplySwap(ItemTable& items, int x, int y, bool merge) {
    int prov = kSwapCost;
    if (merge && items.partner[x] == y && items.kind[x] == kNodeItem) {
        prov = items.val[x];
    }
    items.Close(x);
    items.Close(y);
    items.Set(x, y, kSwapItem, static_cast<std::uint8_t>(prov));
    return prov;
}

// 两比特节点落在 (x, y)，镜像代价 mc（-1 = 不合并）：返回代价变化（<= 0，前一个同对 SWAP 的退款）。
int ApplyNode(ItemTable& items, int x, int y, int mc, bool merge) {
    int delta = 0;
    if (merge && mc >= 0 && items.partner[x] == y && items.kind[x] == kSwapItem && mc < items.val[x]) {
        delta = mc - items.val[x];
    }
    items.Close(x);
    items.Close(y);
    if (merge && mc >= 0 && mc < kSwapCost) {
        items.Set(x, y, kNodeItem, static_cast<std::uint8_t>(mc));
    }
    return delta;
}

std::int64_t EncodeSwap(int x, int y, int n) {
    if (x > y) {
        std::swap(x, y);
    }
    return -1 - (static_cast<std::int64_t>(x) * n + y);
}

void DecodeSwap(std::int64_t event, int n, int& x, int& y) {
    const std::int64_t code = -1 - event;
    x = static_cast<int>(code / n);
    y = static_cast<int>(code % n);
}

enum class NodeKind : std::uint8_t { kTwo, kThree, kMeasure, kOther };

NodeKind KindOf(const RoutingDag& dag, int node) {
    if (dag.nq1()[node] < 0) {
        return NodeKind::kMeasure;
    }
    if (dag.nq3()[node] >= 0) {
        return NodeKind::kThree;
    }
    return NodeKind::kTwo;
}

int MirrorOf(const RouteCostModel& model, int node) {
    if (node < static_cast<int>(model.mirror_cost.size())) {
        return model.mirror_cost[node];
    }
    return -1;
}

int MeasureTokenOf(const RouteCostModel& model, int node) {
    if (node < static_cast<int>(model.measure_token.size())) {
        return model.measure_token[node];
    }
    return -1;
}

// 节点的时间权重（深度代理，见 route_lns.hpp）。
int WeightOf(const RouteCostModel& model, const RoutingDag& dag, int node) {
    if (node < static_cast<int>(model.node_weight.size())) {
        return std::max(0, model.node_weight[node]);
    }
    switch (KindOf(dag, node)) {
        case NodeKind::kTwo:
            return 1;
        case NodeKind::kThree:
            return kThreeQubitWeight;
        default:
            return 0;
    }
}

// 深度代理的次级键：(max_q (t[q] + tail[q])) << 32 | min(Σ_q (t[q] + tail[q]), 2^32 - 1)。
// 时间只增不减，所以部分路由的键不超过它任何完成的键（可采纳）。
std::int64_t SecondaryKey(const std::vector<std::int32_t>& t, const std::vector<std::int32_t>& tail) {
    std::int64_t mx = 0;
    std::int64_t sum = 0;
    const std::size_t n = t.size();
    for (std::size_t q = 0; q < n; ++q) {
        const std::int64_t v = static_cast<std::int64_t>(t[q]) + (tail.empty() ? 0 : tail[q]);
        mx = std::max(mx, v);
        sum += v;
    }
    return (mx << 32) | std::min<std::int64_t>(sum, 0xFFFFFFFFLL);
}

// 两比特操作结束后两个物理比特的时间：从两者较晚的那个起再加 dt（dt < 0 记 0）。
void Advance2(std::vector<std::int32_t>& t, int x, int y, int dt) {
    const std::int32_t v = std::max(t[x], t[y]) + std::max(0, dt);
    t[x] = v;
    t[y] = v;
}

void CheckLayout(const std::vector<int>& layout, int n) {
    if (static_cast<int>(layout.size()) != n) {
        throw std::invalid_argument("layout must be a full permutation of the physical qubits");
    }
    std::vector<char> seen(n, 0);
    for (int p : layout) {
        if (p < 0 || p >= n || seen[p]) {
            throw std::invalid_argument("layout must be a permutation of 0..num_physical-1");
        }
        seen[p] = 1;
    }
}

// 出口之后每个物理比特上的第一个操作。
struct PostOp {
    // 0：无（超出扫描范围），1：打断（测量 / 三比特节点 / 其他），2：SWAP，3：两比特节点。
    int kind = 0;
    int partner = -1;
    // SWAP：其"后合并"代价（下一个操作是同对节点时为该节点的镜像代价，否则 3）；节点：镜像代价（-1 = 无）。
    int cost = kSwapCost;
    // 操作的唯一编号（事件下标），用来判断两个比特上的首个操作是否是同一个。
    std::int64_t id = -1;
    // 两比特节点的 DAG 编号。
    int node = -1;
};

struct PostSwap {
    int x;
    int y;
    int next_cost;
};

// 一个窗口子问题。
struct WindowSpec {
    std::vector<int> nodes;  // DAG 节点（拓扑序）
    std::vector<int> entry_layout;
    std::vector<int> exit_pos;  // 每个逻辑比特：出口必须所在的物理比特，-1 = 自由
    ItemTable entry_items;
    std::vector<PostOp> post;  // 空 = 出口后没有操作
    std::vector<PostSwap> post_swaps;
    std::int64_t upper_bound = kInfCost;  // 只接受代价严格小于它的解
    std::int64_t max_expansions = 0;
    bool merge = true;
    // 搜索顺序 g + weight * h（weight > 1：加权 A*，更快找到改进；剪枝仍用可采纳的 g + h）。
    double weight = 1.0;
    // 找到第一个比上界便宜的解就停止。
    bool stop_at_first = false;
    // 深度次级目标：入口时间与出口后各物理比特的尾长（见 route_lns.hpp）；``upper_secondary`` 是
    // 与 upper_bound 同代价的解要严格优于的次级键（kNoSecondary：只接受更便宜的解）。
    bool depth = false;
    std::vector<std::int32_t> entry_times;
    std::vector<std::int32_t> tail;
    std::int64_t upper_secondary = kNoSecondary;
};

struct WindowSolution {
    bool found = false;
    bool optimal = false;
    std::int64_t cost = 0;
    std::int64_t secondary = 0;
    std::int64_t lower_bound = 0;
    std::int64_t expansions = 0;
    std::vector<std::int64_t> events;
};

// 出口处的合并结算：出口后首个操作对开放项的影响，以及出口后首个 SWAP 的代价。
std::int64_t FinishAtExit(const ItemTable& items, const WindowSpec& spec) {
    std::int64_t adj = 0;
    if (spec.post.empty()) {
        return 0;
    }
    const int n = static_cast<int>(items.partner.size());
    if (spec.merge) {
        for (int x = 0; x < n; ++x) {
            const int y = items.partner[x];
            if (y <= x || items.kind[x] != kSwapItem) {
                continue;
            }
            const PostOp& px = spec.post[x];
            const PostOp& py = spec.post[y];
            if (px.kind == 3 && py.kind == 3 && px.id == py.id && px.cost >= 0 && px.cost < items.val[x]) {
                adj += px.cost - items.val[x];
            }
        }
    }
    for (const PostSwap& s : spec.post_swaps) {
        int prev = kSwapCost;
        if (spec.merge && items.partner[s.x] == s.y && items.kind[s.x] == kNodeItem) {
            prev = items.val[s.x];
        }
        adj += std::min(prev, s.next_cost);
    }
    return adj;
}

// 窗口 A* 求解器。
class WindowSolver {
 public:
    WindowSolver(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model)
        : graph_(graph), dag_(dag), model_(model), n_(graph.num_physical()) {
        local_of_.assign(dag.num_nodes(), -1);
        zpos_.resize(static_cast<std::size_t>(n_) * n_);
        std::uint64_t seed = 0x5DEECE66DULL;
        for (auto& z : zpos_) {
            seed = SplitMix(seed);
            z = seed;
        }
        edge_stamp_.assign(graph.num_edges(), 0);
        token_stamp_.assign(n_, 0);
        rem_stamp_.assign(n_, 0);
        rem_tok_.assign(n_, 0);
        chance_.assign(n_, 0);
        chance_mc_.assign(n_, kSwapCost);
        mark_stamp_.assign(n_, 0);
        chain_f_.assign(n_, 0);
        chain_g_.assign(n_, 0);
        chain_c_.assign(n_, 0);
    }

    WindowSolution Solve(const WindowSpec& spec);

    // 用同一套规则计算一段旧事件（从 spec 的入口状态开始）的窗口代价；不合法时抛出异常。
    // ``secondary``（可为空）：旧事件的次级键（spec.depth 时）。
    std::int64_t EvaluateEvents(const WindowSpec& spec, const std::vector<std::int64_t>& events, std::size_t begin,
                                std::size_t end, std::int64_t* secondary = nullptr);

 private:
    struct Work {
        std::vector<int> pos;
        std::vector<int> rpi;
        std::vector<char> exec;
        std::vector<int> indeg;
        ItemTable items;
        std::vector<std::int32_t> t;  // 深度代理：每个物理比特忙到何时（spec.depth 时）
        int remaining = 0;
        std::int64_t g = 0;
    };

    struct StateRec {
        int parent;
        int edge;  // 物理边编号，根为 -1
        std::int64_t g;
        std::int64_t h;
        std::int64_t sec;  // 次级键（深度代理）；关闭时为 0
        int last_edge;
        bool node_since;
        bool goal;
        bool full;
    };

    void Setup(const WindowSpec& spec);
    void Teardown();
    // 窗口入口状态。
    void InitWork(const WindowSpec& spec, Work& w) const;
    bool Executable(const Work& w, int i) const;
    void Execute(Work& w, int i, std::vector<std::int64_t>* out) const;
    bool Drain(Work& w, std::vector<std::int64_t>* out) const;
    void ApplySwapWork(Work& w, int x, int y, std::vector<std::int64_t>* out) const;
    // full = false：只用便宜的部分（逐节点相遇距离 / 出口距离的匹配）；true：再加相遇后去出口的联合下界与枢纽链下界。
    std::int64_t LowerBound(const Work& w, const WindowSpec& spec, bool& goal, bool full) const;
    std::uint64_t Hash(const Work& w) const;
    void Store(const Work& w, std::int64_t h, std::int64_t sec, int parent, int edge, int last_edge, bool node_since,
               bool goal);
    std::int64_t Secondary(const Work& w) const {
        return depth_ ? SecondaryKey(w.t, spec_->tail) : 0;
    }
    void Load(int s, Work& w) const;

    const RoutingGraph& graph_;
    const RoutingDag& dag_;
    const RouteCostModel& model_;
    int n_;

    // 窗口节点信息（局部下标）
    std::vector<int> local_of_;
    std::vector<int> wnode_;
    std::vector<NodeKind> wkind_;
    std::vector<int> wa_, wb_, wc_;
    std::vector<int> wmc_;
    std::vector<int> wwt_;  // 时间权重
    std::vector<std::vector<int>> wsucc_;
    std::vector<int> windeg0_;
    std::vector<std::uint64_t> znode_;
    std::vector<std::uint64_t> zpos_;
    std::vector<int> constrained_;  // 受约束的逻辑比特
    std::vector<char> is_constrained_;
    std::vector<int> wnodes_of_token_count_;
    int boundary_cnt_[3] = {0, 0, 0};
    bool merge_ = true;
    bool depth_ = false;
    const WindowSpec* spec_ = nullptr;

    // 状态仓库
    int words_ = 0;
    std::vector<StateRec> states_;
    std::vector<std::uint16_t> pos_store_;
    std::vector<std::uint64_t> exec_store_;
    std::vector<std::uint64_t> item_store_;
    std::vector<std::size_t> item_off_;
    std::vector<std::int32_t> time_store_;

    mutable std::vector<std::uint32_t> edge_stamp_;
    mutable std::uint32_t edge_clock_ = 0;
    mutable std::vector<std::uint32_t> token_stamp_;
    mutable std::uint32_t token_clock_ = 0;
    mutable std::vector<std::uint32_t> rem_stamp_;
    mutable std::vector<int> rem_tok_;
    mutable std::vector<int> chance_;
    mutable std::vector<int> chance_mc_;
    // 出口后首个节点（同对、两个比特上都是首个操作）的逻辑比特及其镜像代价：窗口末尾的 SWAP 可与之合并
    std::vector<std::pair<int, int>> boundary_tokens_;
    struct Requirement {
        std::int64_t moves;
        int node;
    };
    mutable std::vector<Requirement> req_;
    mutable std::vector<std::uint32_t> mark_stamp_;
    mutable std::uint32_t mark_clock_ = 0;
    mutable std::vector<std::int64_t> chain_f_;
    mutable std::vector<std::int64_t> chain_g_;
    mutable std::vector<std::int64_t> chain_c_;
    mutable std::vector<std::vector<int>> buckets_;

    // 窗口节点 j 是否（传递地）依赖 i。
    bool Reaches(int i, int j) const {
        return ((reach_[static_cast<std::size_t>(i) * words_ + j / 64] >> (j % 64)) & 1U) != 0;
    }
    std::vector<std::uint64_t> reach_;

    // g(x) = min_u f(u) + D(u, x)（单位边权图上的多源 BFS / 小图直接用距离矩阵）。
    void DistanceTransform(const std::vector<std::int64_t>& f, std::vector<std::int64_t>& g) const;
};

void WindowSolver::DistanceTransform(const std::vector<std::int64_t>& f, std::vector<std::int64_t>& g) const {
    if (n_ <= 32) {
        for (int x = 0; x < n_; ++x) {
            std::int64_t best = kInfCost;
            for (int u = 0; u < n_; ++u) {
                best = std::min(best, f[u] + graph_.Distance(u, x));
            }
            g[x] = best;
        }
        return;
    }
    // Dial：按值分桶的 Dijkstra
    std::int64_t lo = kInfCost;
    std::int64_t hi = 0;
    for (int u = 0; u < n_; ++u) {
        lo = std::min(lo, f[u]);
        hi = std::max(hi, f[u]);
    }
    const std::int64_t span = std::min<std::int64_t>(hi - lo, n_) + n_ + 1;
    if (static_cast<std::int64_t>(buckets_.size()) < span) {
        buckets_.resize(static_cast<std::size_t>(span));
    }
    for (int x = 0; x < n_; ++x) {
        g[x] = std::min(f[x], lo + span - 1);
        buckets_[static_cast<std::size_t>(g[x] - lo)].push_back(x);
    }
    for (std::int64_t b = 0; b < span; ++b) {
        auto& bucket = buckets_[static_cast<std::size_t>(b)];
        for (std::size_t j = 0; j < bucket.size(); ++j) {
            const int x = bucket[j];
            if (g[x] != lo + b) {
                continue;
            }
            for (int z : graph_.Neighbors(x)) {
                if (g[z] > g[x] + 1) {
                    g[z] = g[x] + 1;
                    buckets_[static_cast<std::size_t>(g[z] - lo)].push_back(z);
                }
            }
        }
        bucket.clear();
    }
}

void WindowSolver::Setup(const WindowSpec& spec) {
    spec_ = &spec;
    merge_ = spec.merge;
    depth_ = spec.depth;
    const int k = static_cast<int>(spec.nodes.size());
    wnode_ = spec.nodes;
    wkind_.assign(k, NodeKind::kOther);
    wa_.assign(k, -1);
    wb_.assign(k, -1);
    wc_.assign(k, -1);
    wmc_.assign(k, -1);
    wwt_.assign(k, 0);
    wsucc_.assign(k, {});
    windeg0_.assign(k, 0);
    znode_.resize(k);
    for (int i = 0; i < k; ++i) {
        local_of_[wnode_[i]] = i;
    }
    for (int i = 0; i < k; ++i) {
        const int node = wnode_[i];
        wkind_[i] = KindOf(dag_, node);
        if (wkind_[i] == NodeKind::kMeasure) {
            wa_[i] = MeasureTokenOf(model_, node);
        } else {
            wa_[i] = dag_.nq1()[node];
            wb_[i] = dag_.nq2()[node];
            wc_[i] = dag_.nq3()[node];
        }
        if (wkind_[i] == NodeKind::kTwo && merge_) {
            const int mc = MirrorOf(model_, node);
            wmc_[i] = (mc >= 0 && mc < kSwapCost) ? mc : -1;
        }
        wwt_[i] = WeightOf(model_, dag_, node);
        for (int p : dag_.pred()[node]) {
            const int lp = local_of_[p];
            if (lp >= 0) {
                if (lp >= i) {
                    throw std::invalid_argument("window nodes must be in topological order");
                }
                wsucc_[lp].push_back(i);
                ++windeg0_[i];
            }
        }
        znode_[i] = SplitMix(0xA5A5A5A5ULL + static_cast<std::uint64_t>(node) * 0x10001ULL);
    }
    is_constrained_.assign(n_, 0);
    constrained_.clear();
    for (int t = 0; t < n_; ++t) {
        if (!spec.exit_pos.empty() && spec.exit_pos[t] >= 0) {
            is_constrained_[t] = 1;
            constrained_.push_back(t);
        }
    }
    boundary_cnt_[0] = boundary_cnt_[1] = boundary_cnt_[2] = 0;
    boundary_tokens_.clear();
    if (merge_ && !spec.post.empty()) {
        for (int x = 0; x < n_; ++x) {
            const PostOp& px = spec.post[x];
            if (px.kind == 3 && px.partner > x) {
                const PostOp& py = spec.post[px.partner];
                if (py.kind == 3 && py.id == px.id && px.cost >= 0 && px.cost < kSwapCost) {
                    ++boundary_cnt_[px.cost];
                    boundary_tokens_.emplace_back(dag_.nq1()[px.node], px.cost);
                    boundary_tokens_.emplace_back(dag_.nq2()[px.node], px.cost);
                }
            }
        }
    }
    words_ = (k + 63) / 64;
    // 传递闭包（逆拓扑序）
    reach_.assign(static_cast<std::size_t>(k) * words_, 0);
    for (int i = k - 1; i >= 0; --i) {
        std::uint64_t* row = &reach_[static_cast<std::size_t>(i) * words_];
        for (int sc : wsucc_[i]) {
            row[sc / 64] |= 1ULL << (sc % 64);
            const std::uint64_t* other = &reach_[static_cast<std::size_t>(sc) * words_];
            for (int wd = 0; wd < words_; ++wd) {
                row[wd] |= other[wd];
            }
        }
    }
    states_.clear();
    pos_store_.clear();
    exec_store_.clear();
    item_store_.clear();
    item_off_.clear();
    item_off_.push_back(0);
    time_store_.clear();
}

void WindowSolver::InitWork(const WindowSpec& spec, Work& w) const {
    const int k = static_cast<int>(wnode_.size());
    w.pos = spec.entry_layout;
    w.rpi.assign(n_, 0);
    for (int t = 0; t < n_; ++t) {
        w.rpi[w.pos[t]] = t;
    }
    w.exec.assign(k, 0);
    w.indeg = windeg0_;
    w.items = spec.entry_items;
    if (spec.entry_times.empty()) {
        w.t.assign(n_, 0);
    } else {
        w.t = spec.entry_times;
    }
    w.remaining = k;
    w.g = 0;
}

void WindowSolver::Teardown() {
    for (int node : wnode_) {
        local_of_[node] = -1;
    }
    spec_ = nullptr;
}

bool WindowSolver::Executable(const Work& w, int i) const {
    switch (wkind_[i]) {
        case NodeKind::kTwo:
            return graph_.Adjacent(w.pos[wa_[i]], w.pos[wb_[i]]);
        case NodeKind::kThree:
            return graph_.Cost3(w.pos[wa_[i]], w.pos[wb_[i]], w.pos[wc_[i]]) == 1;
        default:
            return true;
    }
}

void WindowSolver::Execute(Work& w, int i, std::vector<std::int64_t>* out) const {
    switch (wkind_[i]) {
        case NodeKind::kTwo: {
            const int px = w.pos[wa_[i]];
            const int py = w.pos[wb_[i]];
            const int delta = ApplyNode(w.items, px, py, wmc_[i], merge_);
            w.g += delta;
            if (depth_) {
                // 与前一个同对 SWAP 合并时，合并块只比 SWAP 多出 wt + delta 层
                Advance2(w.t, px, py, wwt_[i] + delta);
            }
            break;
        }
        case NodeKind::kThree: {
            const int px = w.pos[wa_[i]];
            const int py = w.pos[wb_[i]];
            const int pz = w.pos[wc_[i]];
            w.items.Close(px);
            w.items.Close(py);
            w.items.Close(pz);
            if (depth_) {
                const std::int32_t v = std::max({w.t[px], w.t[py], w.t[pz]}) + wwt_[i];
                w.t[px] = w.t[py] = w.t[pz] = v;
            }
            break;
        }
        default:
            if (wa_[i] >= 0) {
                w.items.Close(w.pos[wa_[i]]);
            }
            break;
    }
    w.exec[i] = 1;
    --w.remaining;
    for (int s : wsucc_[i]) {
        --w.indeg[s];
    }
    if (out != nullptr) {
        out->push_back(wnode_[i]);
    }
}

bool WindowSolver::Drain(Work& w, std::vector<std::int64_t>* out) const {
    bool any = false;
    bool progress = true;
    const int k = static_cast<int>(wnode_.size());
    while (progress && w.remaining > 0) {
        progress = false;
        for (int i = 0; i < k; ++i) {
            if (!w.exec[i] && w.indeg[i] == 0 && Executable(w, i)) {
                Execute(w, i, out);
                progress = true;
                any = true;
            }
        }
    }
    return any;
}

void WindowSolver::ApplySwapWork(Work& w, int x, int y, std::vector<std::int64_t>* out) const {
    const int prov = ApplySwap(w.items, x, y, merge_);
    w.g += prov;
    if (depth_) {
        Advance2(w.t, x, y, prov);
    }
    const int tx = w.rpi[x];
    const int ty = w.rpi[y];
    w.pos[tx] = y;
    w.pos[ty] = x;
    w.rpi[x] = ty;
    w.rpi[y] = tx;
    if (out != nullptr) {
        out->push_back(EncodeSwap(x, y, n_));
    }
}

std::int64_t WindowSolver::LowerBound(const Work& w, const WindowSpec& spec, bool& goal, bool full) const {
    goal = false;
    const int k = static_cast<int>(wnode_.size());
    ++token_clock_;
    // 每个逻辑比特：剩余窗口节点数、"便宜移动"机会数（紧跟其某个节点的合并 SWAP）及其最低代价
    auto touch = [&](int t) {
        if (rem_stamp_[t] != token_clock_) {
            rem_stamp_[t] = token_clock_;
            rem_tok_[t] = 0;
            chance_[t] = 0;
            chance_mc_[t] = kSwapCost;
        }
    };
    int cnt[3] = {boundary_cnt_[0], boundary_cnt_[1], boundary_cnt_[2]};
    int min_r = kSwapCost;
    for (int i = 0; i < k; ++i) {
        if (w.exec[i] || wkind_[i] == NodeKind::kMeasure) {
            continue;
        }
        for (int t : {wa_[i], wb_[i], wc_[i]}) {
            if (t >= 0) {
                touch(t);
                ++rem_tok_[t];
                if (wmc_[i] >= 0) {
                    ++chance_[t];
                    chance_mc_[t] = std::min(chance_mc_[t], wmc_[i]);
                }
            }
        }
        if (wmc_[i] >= 0) {
            ++cnt[wmc_[i]];
            min_r = std::min(min_r, wmc_[i]);
        }
    }
    if (merge_) {
        for (int x = 0; x < n_; ++x) {
            const int y = w.items.partner[x];
            if (y > x && w.items.kind[x] == kNodeItem) {
                ++cnt[w.items.val[x]];
                for (int t : {w.rpi[x], w.rpi[y]}) {
                    touch(t);
                    ++chance_[t];
                    chance_mc_[t] = std::min<int>(chance_mc_[t], w.items.val[x]);
                }
            }
        }
        for (const auto& [t, mc] : boundary_tokens_) {
            touch(t);
            ++chance_[t];
            chance_mc_[t] = std::min(chance_mc_[t], mc);
        }
    }
    // 移动代价（半个 CZ 为单位）：一次 SWAP 的代价平摊给它移动的两个比特；一个比特的便宜移动
    // 不超过它的机会数，每次至少 mc，其余每次 3。
    auto move_cost = [&](std::int64_t moves, int chances, int mc) -> std::int64_t {
        if (!merge_) {
            return kSwapCost * moves;
        }
        const std::int64_t cheap = std::min<std::int64_t>(moves, chances);
        return cheap * mc + (moves - cheap) * kSwapCost;
    };
    auto chance_of = [&](int t) { return rem_stamp_[t] == token_clock_ ? chance_[t] : 0; };
    auto mc_of = [&](int t) { return rem_stamp_[t] == token_clock_ ? chance_mc_[t] : kSwapCost; };
    auto rem_of = [&](int t) { return rem_stamp_[t] == token_clock_ ? rem_tok_[t] : 0; };

    std::int64_t mx = 0;
    const int num_edges = graph_.num_edges();
    // 每个剩余节点的移动量需求：两个比特先在某条边 (m, m') 上相遇；受约束的比特之后还要走到出口
    // （不论它之后还有没有别的节点，出口段总是必需的）。
    req_.clear();
    for (int i = 0; i < k; ++i) {
        if (w.exec[i]) {
            continue;
        }
        if (wkind_[i] == NodeKind::kTwo) {
            const int a = wa_[i];
            const int b = wb_[i];
            const int pa = w.pos[a];
            const int pb = w.pos[b];
            const std::int64_t d = graph_.Distance(pa, pb) - 1;
            mx = std::max(mx, d);
            const int ea = is_constrained_[a] ? spec.exit_pos[a] : -1;
            const int eb = is_constrained_[b] ? spec.exit_pos[b] : -1;
            const std::int64_t alone = (ea >= 0 ? graph_.Distance(pa, ea) : 0)
                                       + (eb >= 0 ? graph_.Distance(pb, eb) : 0);
            std::int64_t joint = std::max(d, alone);
            if (full && (ea >= 0 || eb >= 0)) {
                joint = kInfCost;
                for (int e = 0; e < num_edges && joint > d; ++e) {
                    const int u = graph_.EdgeX(e);
                    const int v = graph_.EdgeY(e);
                    const std::int64_t au = graph_.Distance(pa, u) + (ea >= 0 ? graph_.Distance(u, ea) : 0);
                    const std::int64_t av = graph_.Distance(pa, v) + (ea >= 0 ? graph_.Distance(v, ea) : 0);
                    const std::int64_t bu = graph_.Distance(pb, u) + (eb >= 0 ? graph_.Distance(u, eb) : 0);
                    const std::int64_t bv = graph_.Distance(pb, v) + (eb >= 0 ? graph_.Distance(v, eb) : 0);
                    joint = std::min(joint, std::min(au + bv, av + bu));
                }
                joint = std::max(joint, d);
            }
            mx = std::max(mx, (joint + 1) / 2);
            // 只在相遇需求超过两个比特各自的出口距离之和时才值得占用它们
            if (joint > alone) {
                req_.push_back({joint, i});
            }
        } else if (wkind_[i] == NodeKind::kThree) {
            const std::int64_t d = graph_.Cost3(w.pos[wa_[i]], w.pos[wb_[i]], w.pos[wc_[i]]) - 1;
            mx = std::max(mx, d);
            if (d > 0) {
                req_.push_back({d, i});
            }
        }
    }
    std::sort(req_.begin(), req_.end(), [](const Requirement& x, const Requirement& y) {
        return x.moves > y.moves || (x.moves == y.moves && x.node < y.node);
    });
    // 比特互不相交的一组需求可以相加（每次移动只服务一个比特，至多减少它所在的一项需求）：
    // 在未被标记的比特上按需求从大到小贪心选取，再加上未被选中的受约束比特到出口的距离。
    auto marked = [&](int t) { return t >= 0 && mark_stamp_[t] == mark_clock_; };
    auto mark = [&](int t) {
        if (t >= 0) {
            mark_stamp_[t] = mark_clock_;
        }
    };
    auto matching = [&](std::int64_t& tot, std::int64_t& half) {
        for (const Requirement& r : req_) {
            const int i = r.node;
            const int a = wa_[i];
            const int b = wb_[i];
            const int c = wc_[i];
            if (marked(a) || marked(b) || marked(c)) {
                continue;
            }
            mark(a);
            mark(b);
            mark(c);
            int chances = chance_of(a) + chance_of(b);
            int mc = std::min(mc_of(a), mc_of(b));
            if (c >= 0) {
                chances += chance_of(c);
                mc = std::min(mc, mc_of(c));
            }
            tot += r.moves;
            half += move_cost(r.moves, chances, mc);
        }
        for (int t : constrained_) {
            if (!marked(t)) {
                const std::int64_t d = graph_.Distance(w.pos[t], spec.exit_pos[t]);
                tot += d;
                half += move_cost(d, chance_of(t), mc_of(t));
            }
        }
    };
    std::int64_t total = 0;
    std::int64_t half_cost = 0;
    ++mark_clock_;
    matching(total, half_cost);

    // 枢纽链下界：剩余节点最多的比特 a 按顺序与伙伴 b1, b2, … 相遇。DP：f_i(x) = a 在第 i 次相遇
    // 时位于 x 的最小总移动量（a 的路程 + 各伙伴到相遇点再到出口的路程，伙伴各只计一次）：
    //   f_i(x) = min_u [f_{i-1}(u) + D(u, x)] + min_{y ∈ N(x)} [D(p_b, y) + D(y, e_b)]，
    // 第一项是图上的距离变换（多源 BFS）。最后加上 a 到出口的距离。
    int hub = -1;
    int hub_rem = 1;
    for (int i = 0; i < k; ++i) {
        if (!w.exec[i] && wkind_[i] == NodeKind::kTwo) {
            for (int t : {wa_[i], wb_[i]}) {
                const int r = rem_of(t);
                if (r > hub_rem || (r == hub_rem && hub >= 0 && t < hub)) {
                    hub = t;
                    hub_rem = r;
                }
            }
        }
    }
    if (hub >= 0 && full) {
        ++mark_clock_;
        mark(hub);
        int chances = chance_of(hub);
        int mc = mc_of(hub);
        const int ph = w.pos[hub];
        for (int x = 0; x < n_; ++x) {
            chain_f_[x] = graph_.Distance(ph, x);
        }
        int steps = 0;
        int prev = -1;
        for (int i = 0; i < k; ++i) {
            if (w.exec[i] || wkind_[i] != NodeKind::kTwo || (wa_[i] != hub && wb_[i] != hub)) {
                continue;
            }
            // 只取依赖链上的节点（对易感知 DAG 中同一比特上的节点不一定有先后）
            if (prev >= 0 && !Reaches(prev, i)) {
                continue;
            }
            prev = i;
            const int b = wa_[i] == hub ? wb_[i] : wa_[i];
            if (marked(b)) {
                continue;  // 同一个伙伴再次出现：它的移动已计入，这次相遇放松掉
            }
            mark(b);
            chances += chance_of(b);
            mc = std::min(mc, mc_of(b));
            ++steps;
            DistanceTransform(chain_f_, chain_g_);
            const int pb = w.pos[b];
            const int eb = is_constrained_[b] ? spec.exit_pos[b] : -1;
            for (int y = 0; y < n_; ++y) {
                chain_c_[y] = graph_.Distance(pb, y) + (eb >= 0 ? graph_.Distance(y, eb) : 0);
            }
            for (int x = 0; x < n_; ++x) {
                std::int64_t best = kInfCost;
                for (int y : graph_.Neighbors(x)) {
                    best = std::min(best, chain_c_[y]);
                }
                chain_f_[x] = chain_g_[x] + best;
            }
        }
        if (steps >= 2) {
            const int eh = is_constrained_[hub] ? spec.exit_pos[hub] : -1;
            std::int64_t walk = kInfCost;
            for (int x = 0; x < n_; ++x) {
                walk = std::min(walk, chain_f_[x] + (eh >= 0 ? graph_.Distance(x, eh) : 0));
            }
            mx = std::max(mx, (walk + 1) / 2);
            std::int64_t tot_b = walk;
            std::int64_t half_b = move_cost(walk, chances, mc);
            matching(tot_b, half_b);
            total = std::max(total, tot_b);
            half_cost = std::max(half_cost, half_b);
        }
    }
    bool at_exit = true;
    for (int t : constrained_) {
        const std::int64_t d = graph_.Distance(w.pos[t], spec.exit_pos[t]);
        if (d > 0) {
            at_exit = false;
        }
        mx = std::max(mx, d);
    }
    if (w.remaining == 0 && at_exit) {
        goal = true;
        return FinishAtExit(w.items, spec);
    }
    const std::int64_t hs = std::max(mx, (total + 1) / 2);
    std::int64_t h = 0;
    if (!merge_) {
        h = kSwapCost * hs;
    } else {
        // hs 个 SWAP 各至多用掉一个合并机会
        std::int64_t left = hs;
        for (int v = 0; v < kSwapCost && left > 0; ++v) {
            const std::int64_t take = std::min<std::int64_t>(left, cnt[v]);
            h += take * v;
            left -= take;
        }
        h += left * kSwapCost;
        h = std::max(h, (half_cost + 1) / 2);
        // 开放 SWAP 项的可能退款
        for (int x = 0; x < n_; ++x) {
            const int y = w.items.partner[x];
            if (y <= x || w.items.kind[x] != kSwapItem || w.items.val[x] == 0) {
                continue;
            }
            const int p = w.items.val[x];
            int best = p;
            const int tx = w.rpi[x];
            const int ty = w.rpi[y];
            for (int i = 0; i < k && best > 0; ++i) {
                if (!w.exec[i] && wmc_[i] >= 0 && ((wa_[i] == tx && wb_[i] == ty) || (wa_[i] == ty && wb_[i] == tx))) {
                    best = std::min(best, wmc_[i]);
                }
            }
            if (!spec.post.empty()) {
                const PostOp& px = spec.post[x];
                const PostOp& py = spec.post[y];
                if (px.kind == 3 && py.kind == 3 && px.id == py.id && px.cost >= 0) {
                    best = std::min(best, px.cost);
                }
            }
            h -= p - best;
        }
    }
    // 出口后首个 SWAP 的代价下界
    for (const PostSwap& ps : spec.post_swaps) {
        int prev = kSwapCost;
        if (merge_) {
            if (w.items.partner[ps.x] == ps.y && w.items.kind[ps.x] == kNodeItem) {
                prev = w.items.val[ps.x];
            }
            prev = std::min(prev, min_r);
        }
        h += std::min(prev, ps.next_cost);
    }
    return h;
}

std::uint64_t WindowSolver::Hash(const Work& w) const {
    std::uint64_t h = 0;
    // 位置只对仍有剩余节点或出口受约束的逻辑比特有意义
    ++token_clock_;
    const int k = static_cast<int>(wnode_.size());
    for (int i = 0; i < k; ++i) {
        if (w.exec[i]) {
            h ^= znode_[i];
            continue;
        }
        for (int t : {wa_[i], wb_[i], wc_[i]}) {
            if (t >= 0 && token_stamp_[t] != token_clock_) {
                token_stamp_[t] = token_clock_;
                h ^= zpos_[static_cast<std::size_t>(t) * n_ + w.pos[t]];
            }
        }
    }
    for (int t : constrained_) {
        if (token_stamp_[t] != token_clock_) {
            token_stamp_[t] = token_clock_;
            h ^= zpos_[static_cast<std::size_t>(t) * n_ + w.pos[t]];
        }
    }
    for (int x = 0; x < n_; ++x) {
        const int y = w.items.partner[x];
        if (y > x) {
            const std::uint64_t code = (static_cast<std::uint64_t>(x) << 32) | (static_cast<std::uint64_t>(y) << 12)
                                       | (static_cast<std::uint64_t>(w.items.kind[x]) << 4) | w.items.val[x];
            h ^= SplitMix(code ^ 0xC0FFEEULL);
        }
    }
    return h;
}

void WindowSolver::Store(const Work& w, std::int64_t h, std::int64_t sec, int parent, int edge, int last_edge,
                         bool node_since, bool goal) {
    states_.push_back({parent, edge, w.g, h, sec, last_edge, node_since, goal, goal});
    if (depth_) {
        time_store_.insert(time_store_.end(), w.t.begin(), w.t.end());
    }
    for (int t = 0; t < n_; ++t) {
        pos_store_.push_back(static_cast<std::uint16_t>(w.pos[t]));
    }
    const int k = static_cast<int>(wnode_.size());
    for (int word = 0; word < words_; ++word) {
        std::uint64_t bits = 0;
        for (int b = 0; b < 64; ++b) {
            const int i = word * 64 + b;
            if (i < k && w.exec[i]) {
                bits |= 1ULL << b;
            }
        }
        exec_store_.push_back(bits);
    }
    for (int x = 0; x < n_; ++x) {
        const int y = w.items.partner[x];
        if (y > x) {
            item_store_.push_back(static_cast<std::uint64_t>(x) | (static_cast<std::uint64_t>(y) << 20)
                                  | (static_cast<std::uint64_t>(w.items.kind[x]) << 40)
                                  | (static_cast<std::uint64_t>(w.items.val[x]) << 48));
        }
    }
    item_off_.push_back(item_store_.size());
}

void WindowSolver::Load(int s, Work& w) const {
    const std::size_t base = static_cast<std::size_t>(s) * n_;
    for (int t = 0; t < n_; ++t) {
        const int p = pos_store_[base + t];
        w.pos[t] = p;
        w.rpi[p] = t;
    }
    const int k = static_cast<int>(wnode_.size());
    w.remaining = 0;
    for (int i = 0; i < k; ++i) {
        const std::uint64_t bits = exec_store_[static_cast<std::size_t>(s) * words_ + i / 64];
        w.exec[i] = static_cast<char>((bits >> (i % 64)) & 1U);
        w.indeg[i] = windeg0_[i];
    }
    for (int i = 0; i < k; ++i) {
        if (w.exec[i]) {
            for (int sc : wsucc_[i]) {
                --w.indeg[sc];
            }
        } else {
            ++w.remaining;
        }
    }
    std::fill(w.items.partner.begin(), w.items.partner.end(), -1);
    for (std::size_t j = item_off_[s]; j < item_off_[s + 1]; ++j) {
        const std::uint64_t code = item_store_[j];
        const int x = static_cast<int>(code & 0xFFFFFU);
        const int y = static_cast<int>((code >> 20) & 0xFFFFFU);
        w.items.Set(x, y, static_cast<std::uint8_t>((code >> 40) & 0xFFU),
                    static_cast<std::uint8_t>((code >> 48) & 0xFFU));
    }
    w.g = states_[s].g;
    if (depth_) {
        const auto off = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(s) * n_);
        std::copy(time_store_.begin() + off, time_store_.begin() + off + n_, w.t.begin());
    }
}

std::int64_t WindowSolver::EvaluateEvents(const WindowSpec& spec, const std::vector<std::int64_t>& events,
                                          std::size_t begin, std::size_t end, std::int64_t* secondary) {
    Setup(spec);
    Work w;
    InitWork(spec, w);
    for (std::size_t j = begin; j < end; ++j) {
        const std::int64_t e = events[j];
        if (e >= 0) {
            const int i = local_of_[e];
            if (i < 0 || w.exec[i] || w.indeg[i] != 0 || !Executable(w, i)) {
                Teardown();
                throw std::invalid_argument("invalid routing event sequence");
            }
            Execute(w, i, nullptr);
        } else {
            int x = 0;
            int y = 0;
            DecodeSwap(e, n_, x, y);
            ApplySwapWork(w, x, y, nullptr);
        }
    }
    std::int64_t cost = w.g + FinishAtExit(w.items, spec);
    if (secondary != nullptr) {
        *secondary = Secondary(w);
    }
    Teardown();
    return cost;
}

WindowSolution WindowSolver::Solve(const WindowSpec& spec) {
    Setup(spec);
    WindowSolution sol;
    const int k = static_cast<int>(wnode_.size());
    Work root;
    InitWork(spec, root);
    Drain(root, nullptr);

    // (g + weight * h, -g, 次级键, state)：f 相同时先走更深的状态（尽快找到解）；次级键（深度代理，关闭时
    // 为 0）在代价最优的解找到之后继续在同代价的平台上搜索更浅的解
    using Entry = std::tuple<double, std::int64_t, std::int64_t, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<>> open;
    std::unordered_map<std::uint64_t, std::pair<std::int64_t, std::int64_t>> best_g;
    best_g.reserve(1024);

    std::int64_t incumbent = spec.upper_bound;
    std::int64_t incumbent_sec = depth_ ? spec.upper_secondary : 0;
    int incumbent_state = -1;
    // (f, sec) 不优于现有最好解（字典序）：剪掉
    auto dominated = [&](std::int64_t f, std::int64_t sec) {
        return f > incumbent || (f == incumbent && sec >= incumbent_sec);
    };

    auto push = [&](const Work& w, int parent, int edge, int last_edge, bool node_since) {
        // 先查置换表（哈希含已执行节点、相关比特位置与开放项，目标状态因此也唯一），重复状态不必算下界
        const std::uint64_t key = Hash(w);
        const std::int64_t sec = Secondary(w);
        auto it = best_g.find(key);
        if (it != best_g.end() && (it->second.first < w.g || (it->second.first == w.g && it->second.second <= sec))) {
            return;
        }
        bool goal = false;
        const std::int64_t h = LowerBound(w, spec, goal, false);
        const std::int64_t f = w.g + h;
        if (dominated(f, sec)) {
            return;
        }
        best_g[key] = {w.g, sec};
        Store(w, goal ? 0 : h, sec, parent, edge, last_edge, node_since, goal);
        const int idx = static_cast<int>(states_.size()) - 1;
        if (goal) {
            // 目标状态的 f 就是精确代价
            states_[idx].g = f;
            incumbent = f;
            incumbent_sec = sec;
            incumbent_state = idx;
            open.emplace(static_cast<double>(f), -f, sec, idx);
            return;
        }
        open.emplace(static_cast<double>(w.g) + spec.weight * static_cast<double>(h), -w.g, sec, idx);
    };

    push(root, -1, -1, -1, true);
    Work cur;
    cur.pos.assign(n_, 0);
    cur.rpi.assign(n_, 0);
    cur.exec.assign(k, 0);
    cur.indeg.assign(k, 0);
    cur.items.Init(n_);
    cur.t.assign(n_, 0);
    Work child = cur;
    std::vector<int> cand;
    std::int64_t lower = spec.upper_bound;
    bool exhausted = false;
    const bool exact_order = spec.weight <= 1.0;
    while (!open.empty()) {
        const int s = std::get<3>(open.top());
        const std::int64_t f = states_[s].g + states_[s].h;
        if (states_[s].goal) {
            if (s == incumbent_state && exact_order) {
                sol.optimal = true;
                lower = f;
                if (!depth_) {
                    break;
                }
                // 代价已证明最优：继续在 f 相同、次级键更小的状态上找更浅的解（被支配的状态弹出即丢弃）
            }
            open.pop();
            continue;
        }
        if (dominated(f, states_[s].sec)) {
            open.pop();
            continue;
        }
        if (incumbent_state >= 0 && spec.stop_at_first) {
            break;
        }
        if (sol.expansions >= spec.max_expansions) {
            exhausted = true;
            lower = std::min(lower, f);
            break;
        }
        open.pop();
        Load(s, cur);
        if (!states_[s].full) {
            // 惰性 A*：弹出时才算完整下界，变大则重新入队
            bool g1 = false;
            const std::int64_t h2 = LowerBound(cur, spec, g1, true);
            states_[s].full = true;
            if (h2 > states_[s].h) {
                states_[s].h = h2;
                if (!dominated(cur.g + h2, states_[s].sec)) {
                    open.emplace(static_cast<double>(cur.g) + spec.weight * static_cast<double>(h2), -cur.g,
                                 states_[s].sec, s);
                }
                continue;
            }
        }
        ++sol.expansions;
        const StateRec rec = states_[s];
        // 候选边：触及仍有剩余节点的比特，或尚未到出口的受约束比特
        ++edge_clock_;
        cand.clear();
        auto add_token = [&](int t) {
            const int p = cur.pos[t];
            for (int e : graph_.EdgesAt(p)) {
                if (edge_stamp_[e] != edge_clock_) {
                    edge_stamp_[e] = edge_clock_;
                    cand.push_back(e);
                }
            }
        };
        for (int i = 0; i < k; ++i) {
            if (!cur.exec[i]) {
                for (int t : {wa_[i], wb_[i], wc_[i]}) {
                    if (t >= 0 && wkind_[i] != NodeKind::kMeasure) {
                        add_token(t);
                    }
                }
            }
        }
        for (int t : constrained_) {
            if (cur.pos[t] != spec.exit_pos[t]) {
                add_token(t);
            }
        }
        std::sort(cand.begin(), cand.end());
        for (int e : cand) {
            const int x = graph_.EdgeX(e);
            const int y = graph_.EdgeY(e);
            if (!rec.node_since && rec.last_edge >= 0) {
                if (e == rec.last_edge) {
                    continue;
                }
                const int lx = graph_.EdgeX(rec.last_edge);
                const int ly = graph_.EdgeY(rec.last_edge);
                const bool disjoint = x != lx && x != ly && y != lx && y != ly;
                if (disjoint && e < rec.last_edge) {
                    continue;
                }
            }
            if (cur.items.partner[x] == y && cur.items.kind[x] == kSwapItem) {
                continue;  // 同一条边上连续两次 SWAP（中间没有触及它们的操作）= 撤销
            }
            child.pos = cur.pos;
            child.rpi = cur.rpi;
            child.exec = cur.exec;
            child.indeg = cur.indeg;
            child.items.partner = cur.items.partner;
            child.items.kind = cur.items.kind;
            child.items.val = cur.items.val;
            child.remaining = cur.remaining;
            child.g = cur.g;
            if (depth_) {
                child.t = cur.t;
            }
            ApplySwapWork(child, x, y, nullptr);
            const bool ran = Drain(child, nullptr);
            push(child, s, e, e, ran);
        }
    }
    if (!exhausted && open.empty()) {
        // 搜索空间穷尽：现有最好解（或上界）就是最优
        sol.optimal = incumbent_state >= 0;
        lower = incumbent;
    } else if (!exact_order) {
        lower = 0;
    }
    sol.lower_bound = lower;
    if (incumbent_state >= 0) {
        sol.found = true;
        sol.cost = states_[incumbent_state].g;
        sol.secondary = states_[incumbent_state].sec;
        if (sol.optimal) {
            sol.lower_bound = sol.cost;
        }
        // 重建事件：沿父指针取边，从根重放
        std::vector<int> path;
        for (int s = incumbent_state; s >= 0; s = states_[s].parent) {
            if (states_[s].edge >= 0) {
                path.push_back(states_[s].edge);
            }
        }
        std::reverse(path.begin(), path.end());
        Work w;
        InitWork(spec, w);
        Drain(w, &sol.events);
        for (int e : path) {
            ApplySwapWork(w, graph_.EdgeX(e), graph_.EdgeY(e), &sol.events);
            Drain(w, &sol.events);
        }
    }
    Teardown();
    return sol;
}

// 沿事件序列推进的全局状态（布局 + 开放项 + 每个逻辑比特已执行的节点数 + 深度代理的时间）。
struct ScanState {
    std::vector<int> pi;
    std::vector<int> rpi;
    ItemTable items;
    std::vector<int> used;
    std::vector<std::int32_t> t;  // 为空：不跟踪时间

    void Init(const std::vector<int>& layout, bool track_time) {
        const int n = static_cast<int>(layout.size());
        pi = layout;
        rpi.assign(n, 0);
        for (int q = 0; q < n; ++q) {
            rpi[layout[q]] = q;
        }
        items.Init(n);
        used.assign(n, 0);
        t.assign(track_time ? n : 0, 0);
    }
};

// 一个事件落在哪些物理比特上、它的时间增量（与 WindowSolver 的规则一致）。
struct EventSpan {
    int q[3] = {-1, -1, -1};
    int nq = 0;
    int dt = 0;
};

void ScanApply(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model, ScanState& st,
               std::int64_t event, bool merge, std::int64_t* cost, EventSpan* span = nullptr) {
    const int n = graph.num_physical();
    const bool timed = !st.t.empty();
    if (event >= 0) {
        const int node = static_cast<int>(event);
        switch (KindOf(dag, node)) {
            case NodeKind::kTwo: {
                const int a = dag.nq1()[node];
                const int b = dag.nq2()[node];
                const int px = st.pi[a];
                const int py = st.pi[b];
                const int d = ApplyNode(st.items, px, py, merge ? MirrorOf(model, node) : -1, merge);
                if (cost != nullptr) {
                    *cost += d;
                }
                const int dt = std::max(0, WeightOf(model, dag, node) + d);
                if (timed) {
                    Advance2(st.t, px, py, dt);
                }
                if (span != nullptr) {
                    *span = EventSpan{{px, py, -1}, 2, dt};
                }
                ++st.used[a];
                ++st.used[b];
                break;
            }
            case NodeKind::kThree: {
                const int w = WeightOf(model, dag, node);
                std::int32_t v = 0;
                int j = 0;
                for (int t : {dag.nq1()[node], dag.nq2()[node], dag.nq3()[node]}) {
                    st.items.Close(st.pi[t]);
                    ++st.used[t];
                    if (timed) {
                        v = std::max(v, st.t[st.pi[t]]);
                    }
                    if (span != nullptr) {
                        span->q[j] = st.pi[t];
                    }
                    ++j;
                }
                if (timed) {
                    for (int t : {dag.nq1()[node], dag.nq2()[node], dag.nq3()[node]}) {
                        st.t[st.pi[t]] = v + w;
                    }
                }
                if (span != nullptr) {
                    span->nq = 3;
                    span->dt = w;
                }
                break;
            }
            default: {
                const int t = MeasureTokenOf(model, node);
                if (t >= 0) {
                    st.items.Close(st.pi[t]);
                }
                if (span != nullptr) {
                    *span = EventSpan{};
                    if (t >= 0) {
                        span->q[0] = st.pi[t];
                        span->nq = 1;
                    }
                }
                break;
            }
        }
        return;
    }
    int x = 0;
    int y = 0;
    DecodeSwap(event, n, x, y);
    const int c = ApplySwap(st.items, x, y, merge);
    if (cost != nullptr) {
        *cost += c;
    }
    if (timed) {
        Advance2(st.t, x, y, c);
    }
    if (span != nullptr) {
        *span = EventSpan{{x, y, -1}, 2, c};
    }
    const int tx = st.rpi[x];
    const int ty = st.rpi[y];
    st.pi[tx] = y;
    st.pi[ty] = x;
    st.rpi[x] = ty;
    st.rpi[y] = tx;
}

// 校验并计算整条事件序列的代价。
std::int64_t SequenceCost(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                          const std::vector<int>& layout, const std::vector<std::int64_t>& events, bool merge) {
    const int n = graph.num_physical();
    CheckLayout(layout, n);
    ScanState st;
    st.Init(layout, false);
    std::vector<int> indeg(dag.num_nodes());
    for (int i = 0; i < dag.num_nodes(); ++i) {
        indeg[i] = static_cast<int>(dag.pred()[i].size());
    }
    std::vector<char> done(dag.num_nodes(), 0);
    std::int64_t cost = 0;
    for (std::int64_t e : events) {
        if (e >= 0) {
            if (e >= dag.num_nodes() || done[e] || indeg[e] != 0) {
                throw std::invalid_argument("invalid routing event sequence: dependency violated");
            }
            const int node = static_cast<int>(e);
            const NodeKind kind = KindOf(dag, node);
            if (kind == NodeKind::kTwo && !graph.Adjacent(st.pi[dag.nq1()[node]], st.pi[dag.nq2()[node]])) {
                throw std::invalid_argument("invalid routing event sequence: node not executable");
            }
            if (kind == NodeKind::kThree
                && graph.Cost3(st.pi[dag.nq1()[node]], st.pi[dag.nq2()[node]], st.pi[dag.nq3()[node]]) != 1) {
                throw std::invalid_argument("invalid routing event sequence: node not executable");
            }
            done[e] = 1;
            for (int s : dag.succ()[node]) {
                --indeg[s];
            }
        } else {
            int x = 0;
            int y = 0;
            DecodeSwap(e, n, x, y);
            if (x < 0 || y >= n || !graph.Adjacent(x, y)) {
                throw std::invalid_argument("invalid routing event sequence: SWAP not on a coupling edge");
            }
        }
        ScanApply(graph, dag, model, st, e, merge, &cost);
    }
    return cost;
}

// 由每个事件的比特与时间增量倒推：tail[j] = dt[j] + max_{q ∈ 事件 j} next[q]（next 初值为出口之后的尾长）。
void BackwardTails(const std::vector<EventSpan>& spans, std::vector<std::int32_t>& next,
                   std::vector<std::int32_t>& out) {
    for (std::size_t jj = spans.size(); jj-- > 0;) {
        const EventSpan& sp = spans[jj];
        std::int32_t v = 0;
        for (int a = 0; a < sp.nq; ++a) {
            v = std::max(v, next[sp.q[a]]);
        }
        v += sp.dt;
        out[jj] = v;
        for (int a = 0; a < sp.nq; ++a) {
            next[sp.q[a]] = v;
        }
    }
}

// 每个事件起（含它自己）到结尾的最长路径（深度代理）。
std::vector<std::int32_t> EventTails(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                                     const std::vector<int>& layout, const std::vector<std::int64_t>& events,
                                     bool merge) {
    ScanState st;
    st.Init(layout, false);
    std::vector<EventSpan> spans(events.size());
    for (std::size_t j = 0; j < events.size(); ++j) {
        ScanApply(graph, dag, model, st, events[j], merge, nullptr, &spans[j]);
    }
    std::vector<std::int32_t> next(graph.num_physical(), 0);
    std::vector<std::int32_t> out(events.size(), 0);
    BackwardTails(spans, next, out);
    return out;
}

// 整条路由的深度代理（所有物理比特时间的最大值）。
std::int64_t Makespan(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                      const std::vector<int>& layout, const std::vector<std::int64_t>& events, bool merge) {
    ScanState st;
    st.Init(layout, true);
    for (std::int64_t e : events) {
        ScanApply(graph, dag, model, st, e, merge, nullptr);
    }
    std::int64_t mx = 0;
    for (std::int32_t v : st.t) {
        mx = std::max<std::int64_t>(mx, v);
    }
    return mx;
}

// 出口（事件下标 e，出口布局 pi/rpi）之后每个物理比特上的首个操作与出口后首个 SWAP（填入 spec.post / post_swaps）。
void ScanPost(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
              const std::vector<std::int64_t>& ev, std::size_t e, const std::vector<int>& exit_pi,
              const std::vector<int>& exit_rpi, bool merge, int lookahead_events, WindowSpec& spec) {
    const int n = graph.num_physical();
    spec.post.assign(n, PostOp{});
    std::vector<int> pi2 = exit_pi;
    std::vector<int> rpi2 = exit_rpi;
    int found = 0;
    const std::size_t limit = std::min(ev.size(), e + static_cast<std::size_t>(lookahead_events));
    auto mark = [&](int q, int kind, int partner, int cost, std::int64_t id, int node = -1) {
        if (spec.post[q].kind == 0) {
            spec.post[q] = PostOp{kind, partner, cost, id, node};
            ++found;
        }
    };
    for (std::size_t j = e; j < limit && found < n; ++j) {
        const std::int64_t x = ev[j];
        const auto id = static_cast<std::int64_t>(j);
        if (x >= 0) {
            const int node = static_cast<int>(x);
            switch (KindOf(dag, node)) {
                case NodeKind::kTwo: {
                    const int pa = pi2[dag.nq1()[node]];
                    const int pb = pi2[dag.nq2()[node]];
                    const int mc = merge ? MirrorOf(model, node) : -1;
                    mark(pa, 3, pb, mc, id, node);
                    mark(pb, 3, pa, mc, id, node);
                    break;
                }
                case NodeKind::kThree:
                    for (int t : {dag.nq1()[node], dag.nq2()[node], dag.nq3()[node]}) {
                        mark(pi2[t], 1, -1, 0, id);
                    }
                    break;
                default: {
                    const int t = MeasureTokenOf(model, node);
                    if (t >= 0) {
                        mark(pi2[t], 1, -1, 0, id);
                    }
                    break;
                }
            }
        } else {
            int a = 0;
            int b = 0;
            DecodeSwap(x, n, a, b);
            const bool first = spec.post[a].kind == 0 && spec.post[b].kind == 0;
            if (first) {
                // 这个 SWAP 的"后合并"代价：其后两个比特上的首个操作是同一个同对节点
                int next_cost = kSwapCost;
                if (merge) {
                    // 交换后 a、b 上的逻辑比特；它们只能被触及 a 或 b 的操作移走
                    const int tok_a = rpi2[b];
                    const int tok_b = rpi2[a];
                    auto touches = [&](int t) { return t == tok_a || t == tok_b; };
                    const std::size_t limit2 = std::min(ev.size(), j + 1 + static_cast<std::size_t>(lookahead_events));
                    for (std::size_t jj = j + 1; jj < limit2; ++jj) {
                        const std::int64_t y = ev[jj];
                        if (y < 0) {
                            int c = 0;
                            int d = 0;
                            DecodeSwap(y, n, c, d);
                            if (c == a || c == b || d == a || d == b) {
                                break;
                            }
                            continue;
                        }
                        const int node = static_cast<int>(y);
                        const NodeKind kind = KindOf(dag, node);
                        if (kind == NodeKind::kTwo) {
                            const bool on1 = touches(dag.nq1()[node]);
                            const bool on2 = touches(dag.nq2()[node]);
                            if (on1 && on2) {
                                const int mc = MirrorOf(model, node);
                                if (mc >= 0) {
                                    next_cost = std::min(kSwapCost, mc);
                                }
                                break;
                            }
                            if (on1 || on2) {
                                break;
                            }
                        } else if (kind == NodeKind::kThree) {
                            if (touches(dag.nq1()[node]) || touches(dag.nq2()[node]) || touches(dag.nq3()[node])) {
                                break;
                            }
                        } else if (touches(MeasureTokenOf(model, node))) {
                            break;
                        }
                    }
                }
                spec.post_swaps.push_back({a, b, next_cost});
            }
            mark(a, 2, b, 0, id);
            mark(b, 2, a, 0, id);
            const int ta = rpi2[a];
            const int tb = rpi2[b];
            pi2[ta] = b;
            pi2[tb] = a;
            rpi2[a] = tb;
            rpi2[b] = ta;
        }
    }
}

}  // namespace

std::int64_t RouteSwapCost(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                           const std::vector<int>& layout, const std::vector<std::int64_t>& events, bool merge_costs) {
    return SequenceCost(graph, dag, model, layout, events, merge_costs);
}

ExactRouteResult ExactRoute(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                            const std::vector<int>& layout, const ExactSearchOptions& options,
                            std::int64_t upper_bound) {
    const int n = graph.num_physical();
    CheckLayout(layout, n);
    WindowSpec spec;
    spec.nodes.resize(dag.num_nodes());
    for (int i = 0; i < dag.num_nodes(); ++i) {
        spec.nodes[i] = i;
    }
    spec.entry_layout = layout;
    spec.entry_items.Init(n);
    spec.upper_bound = upper_bound > 0 ? upper_bound : kInfCost;
    spec.max_expansions = options.max_expansions;
    spec.merge = options.merge_costs;
    WindowSolver solver(graph, dag, model);
    WindowSolution sol = solver.Solve(spec);
    ExactRouteResult out;
    out.found = sol.found;
    out.optimal = sol.optimal;
    out.cost = sol.cost;
    out.expansions = sol.expansions;
    out.lower_bound = sol.lower_bound >= kInfCost ? -1 : sol.lower_bound;
    out.events = std::move(sol.events);
    for (std::int64_t e : out.events) {
        out.num_swaps += e < 0 ? 1 : 0;
    }
    return out;
}

RefineResult RefineRouting(const RoutingGraph& graph, const RoutingDag& dag, const RouteCostModel& model,
                           const std::vector<int>& layout, const std::vector<std::int64_t>& events,
                           const RefineOptions& options) {
    const int n = graph.num_physical();
    const bool merge = options.merge_costs;
    // 主扫描是否按 (代价, 深度) 字典序；深度扫描总是跟踪时间
    const bool depth_main = options.depth_ties;
    const bool depth = options.depth_ties || options.depth_passes > 0;
    RefineResult res;
    res.events = events;
    res.old_cost = SequenceCost(graph, dag, model, layout, events, merge);
    for (std::int64_t e : events) {
        res.old_swaps += e < 0 ? 1 : 0;
    }
    if (depth) {
        res.old_depth = Makespan(graph, dag, model, layout, events, merge);
    }
    // 每个逻辑比特上的节点总数（决定出口约束）
    std::vector<int> total_uses(n, 0);
    for (int i = 0; i < dag.num_nodes(); ++i) {
        const NodeKind kind = KindOf(dag, i);
        if (kind == NodeKind::kTwo || kind == NodeKind::kThree) {
            ++total_uses[dag.nq1()[i]];
            ++total_uses[dag.nq2()[i]];
            if (kind == NodeKind::kThree) {
                ++total_uses[dag.nq3()[i]];
            }
        }
    }
    WindowSolver solver(graph, dag, model);
    std::vector<std::int64_t>& ev = res.events;
    std::vector<int> wcount(n, 0);
    // 深度代理：每个事件起的最长路径（与 ev 对齐；替换窗口时只更新窗口段，窗口之前的值本次扫描不再用到）
    std::vector<std::int32_t> tails;

    // 一次窗口大小为 k 的扫描；equal：代价相同而次级键更小也替换。返回是否有改进。
    auto sweep = [&](int k, bool equal, std::int64_t& used_exp, std::int64_t budget) {
        bool any = false;
        const bool timed = depth_main || equal;
        ScanState st;
        st.Init(layout, timed);
        if (timed) {
            tails = EventTails(graph, dag, model, layout, ev, merge);
        }
        std::size_t p = 0;
        bool tail_done = false;
        while (p < ev.size() && used_exp < budget) {
            // 窗口 [p, e)：从 p 起的 k 个 SWAP 以及直到下一个 SWAP 之前的所有节点
            std::size_t e = p;
            int swaps = 0;
            bool has3 = false;
            int nodes = 0;
            while (e < ev.size()) {
                if (ev[e] < 0) {
                    if (swaps == k) {
                        break;
                    }
                    ++swaps;
                } else {
                    ++nodes;
                    if (KindOf(dag, static_cast<int>(ev[e])) == NodeKind::kThree) {
                        has3 = true;
                    }
                }
                ++e;
            }
            const bool tail = e == ev.size();
            const bool eligible = swaps > 0 && !has3 && nodes <= options.max_window_nodes
                                  && (swaps == k || (tail && !tail_done));
            if (tail) {
                tail_done = true;
            }
            if (eligible) {
                ++res.windows;
                WindowSpec spec;
                spec.entry_layout = st.pi;
                spec.entry_items = st.items;
                spec.merge = merge;
                spec.weight = options.weight;
                spec.stop_at_first = options.stop_at_first;
                spec.max_expansions = std::min(options.window_expansions, budget - used_exp);
                spec.depth = timed;
                if (timed) {
                    spec.entry_times = st.t;
                    spec.tail.assign(n, 0);
                }
                std::fill(wcount.begin(), wcount.end(), 0);
                std::vector<int> exit_pi = st.pi;
                std::vector<int> exit_rpi = st.rpi;
                for (std::size_t j = p; j < e; ++j) {
                    const std::int64_t x = ev[j];
                    if (x >= 0) {
                        const int node = static_cast<int>(x);
                        spec.nodes.push_back(node);
                        if (KindOf(dag, node) == NodeKind::kTwo) {
                            ++wcount[dag.nq1()[node]];
                            ++wcount[dag.nq2()[node]];
                        }
                    } else {
                        int a = 0;
                        int b = 0;
                        DecodeSwap(x, n, a, b);
                        const int ta = exit_rpi[a];
                        const int tb = exit_rpi[b];
                        exit_pi[ta] = b;
                        exit_pi[tb] = a;
                        exit_rpi[a] = tb;
                        exit_rpi[b] = ta;
                    }
                }
                // 窗口节点要按拓扑序排列：事件顺序就是一个拓扑序，但求解器要求前驱下标更小
                spec.exit_pos.assign(n, -1);
                for (int t = 0; t < n; ++t) {
                    if (st.used[t] + wcount[t] < total_uses[t]) {
                        spec.exit_pos[t] = exit_pi[t];
                    }
                }
                // 出口之后每个物理比特上的首个操作
                if (e < ev.size()) {
                    ScanPost(graph, dag, model, ev, e, exit_pi, exit_rpi, merge, options.lookahead_events, spec);
                    if (timed) {
                        for (int q = 0; q < n; ++q) {
                            if (spec.post[q].kind != 0) {
                                spec.tail[q] = tails[static_cast<std::size_t>(spec.post[q].id)];
                            }
                        }
                    }
                }
                std::int64_t old_sec = 0;
                const std::int64_t old_cost = solver.EvaluateEvents(spec, ev, p, e, &old_sec);
                spec.upper_bound = old_cost;
                spec.upper_secondary = equal ? old_sec : kNoSecondary;
                WindowSolution sol = solver.Solve(spec);
                used_exp += sol.expansions;
                if (sol.found && (sol.cost < old_cost || (equal && sol.cost == old_cost && sol.secondary < old_sec))) {
                    if (timed) {
                        // 新窗口段的尾长：从出口后的尾长倒推
                        ScanState sc = st;
                        std::vector<EventSpan> spans(sol.events.size());
                        for (std::size_t j = 0; j < sol.events.size(); ++j) {
                            ScanApply(graph, dag, model, sc, sol.events[j], merge, nullptr, &spans[j]);
                        }
                        std::vector<std::int32_t> next(spec.tail);
                        std::vector<std::int32_t> seg(sol.events.size(), 0);
                        BackwardTails(spans, next, seg);
                        tails.erase(tails.begin() + static_cast<std::ptrdiff_t>(p),
                                    tails.begin() + static_cast<std::ptrdiff_t>(e));
                        tails.insert(tails.begin() + static_cast<std::ptrdiff_t>(p), seg.begin(), seg.end());
                    }
                    ev.erase(ev.begin() + static_cast<std::ptrdiff_t>(p), ev.begin() + static_cast<std::ptrdiff_t>(e));
                    ev.insert(ev.begin() + static_cast<std::ptrdiff_t>(p), sol.events.begin(), sol.events.end());
                    ++res.improved;
                    any = true;
                }
            }
            // 前进到下一个 SWAP 之后
            bool advanced = false;
            while (p < ev.size()) {
                const std::int64_t x = ev[p];
                ScanApply(graph, dag, model, st, x, merge, nullptr);
                ++p;
                if (x < 0) {
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                break;
            }
        }
        return any;
    };

    for (int pass = 0; pass < options.passes; ++pass) {
        bool any_pass = false;
        for (int k : options.window_swaps) {
            if (k >= 1) {
                any_pass = sweep(k, false, res.expansions, options.total_expansions) || any_pass;
            }
        }
        if (!any_pass) {
            break;
        }
    }
    if (depth) {
        std::int64_t used = 0;
        for (int pass = 0; pass < options.depth_passes; ++pass) {
            bool any_pass = false;
            for (int k : options.window_swaps) {
                if (k >= 1) {
                    any_pass = sweep(k, true, used, options.depth_expansions) || any_pass;
                }
            }
            if (!any_pass) {
                break;
            }
        }
        res.expansions += used;
    }
    res.new_cost = SequenceCost(graph, dag, model, layout, ev, merge);
    if (res.new_cost > res.old_cost) {
        // 出口后扫描范围截断造成的模型偏差：退回原路由
        res.events = events;
        res.new_cost = res.old_cost;
        res.improved = 0;
    }
    for (std::int64_t e : res.events) {
        res.new_swaps += e < 0 ? 1 : 0;
    }
    if (depth) {
        res.new_depth = Makespan(graph, dag, model, layout, res.events, merge);
    }
    return res;
}

}  // namespace quspect::native::transpiler
