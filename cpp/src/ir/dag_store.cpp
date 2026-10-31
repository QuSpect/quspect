#include "quspect_native/ir/dag_store.hpp"

#include <algorithm>
#include <unordered_set>

namespace quspect::native::ir {

namespace {

const DagStore::AdjList& EmptyAdj() {
    static const DagStore::AdjList kEmpty;
    return kEmpty;
}

bool Contains(const DagStore::WireList& wires, WireId w) noexcept {
    for (WireId x : wires) {
        if (x == w) {
            return true;
        }
    }
    return false;
}

void InsertBySeq(DagStore::AdjList& list, const DagAdj& entry) {
    auto it = std::lower_bound(list.begin(), list.end(), entry.seq,
                               [](const DagAdj& a, EdgeSeq seq) { return a.seq < seq; });
    list.insert(it, entry);
}

void EraseBySeq(DagStore::AdjList& list, EdgeSeq seq) noexcept {
    for (auto it = list.begin(); it != list.end(); ++it) {
        if (it->seq == seq) {
            list.erase(it);
            return;
        }
    }
}

// 搜索中的访问标记集合。
class SeenSet {
public:
    void insert(NodeId id) {
        set_.insert(id);
    }
    bool contains(NodeId id) const {
        return set_.count(id) > 0;
    }

private:
    std::unordered_set<NodeId> set_;
};

}  // namespace

// ---------------------------------------------------------------------------
// 节点
// ---------------------------------------------------------------------------

const DagStore::NodeSlot* DagStore::slot(NodeId id) const noexcept {
    const std::int32_t* idx = slot_of_.find(id);
    return idx == nullptr ? nullptr : &slots_[static_cast<std::size_t>(*idx)];
}

void DagStore::add_node(NodeId id, DagNodeKind kind, const WireId* wires, std::size_t num_wires) {
    if (slot_of_.contains(id)) {
        throw std::invalid_argument("Node with id=" + std::to_string(id) + " already exists in DAG.");
    }
    NodeSlot s;
    s.id = id;
    s.kind = kind;
    s.alive = true;
    s.wires.assign(wires, wires + num_wires);
    slot_of_[id] = static_cast<std::int32_t>(slots_.size());
    slots_.push_back(std::move(s));
    ++num_alive_;
    for (std::size_t i = 0; i < num_wires; ++i) {
        const WireId w = wires[i];
        wire_count_[w] += 1;
        if (kind == DagNodeKind::kInput) {
            if (wire_input_.contains(w)) {
                ambiguous_wires_[w] = 1;
            }
            wire_input_[w] = id;
        }
    }
    if (pos_) {
        (*pos_)[id] = next_pos_;
        ++next_pos_;
    }
    touch();
}

void DagStore::remove_node(NodeId id) {
    const std::int32_t* idx_ptr = slot_of_.find(id);
    if (idx_ptr == nullptr) {
        throw DagKeyError("Node id=" + std::to_string(id) + " not found in DAG.");
    }
    const auto idx = static_cast<std::size_t>(*idx_ptr);
    NodeSlot& s = slots_[idx];
    for (WireId w : s.wires) {
        wire_count_[w] -= 1;
        if (s.kind == DagNodeKind::kInput) {
            ambiguous_wires_[w] = 1;
        }
    }
    s.alive = false;
    s.wires.clear();
    slot_of_.erase(id);
    --num_alive_;

    std::vector<EdgeSeq> incident;
    if (const Adjacency* a = adj_.find(id)) {
        incident.reserve(a->out.size() + a->in.size());
        for (const DagAdj& e : a->out) {
            incident.push_back(e.seq);
        }
        for (const DagAdj& e : a->in) {
            incident.push_back(e.seq);
        }
    }
    std::sort(incident.begin(), incident.end());
    incident.erase(std::unique(incident.begin(), incident.end()), incident.end());
    for (EdgeSeq seq : incident) {
        DagEdge edge = unlink(seq);
        if (journal_open_) {
            journal_.push_back(JournalEntry{JournalKind::kRemove, seq, edge, 0, nullptr});
        }
    }
    adj_.erase(id);
    if (pos_) {
        pos_->erase(id);
    }
    touch();
    if (slots_.size() > 64 && slots_.size() > 2 * num_alive_) {
        compact_order();
    }
}

void DagStore::compact_order() {
    std::size_t k = 0;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (!slots_[i].alive) {
            continue;
        }
        if (k != i) {
            slots_[k] = std::move(slots_[i]);
        }
        slot_of_[slots_[k].id] = static_cast<std::int32_t>(k);
        ++k;
    }
    slots_.resize(k);
}

void DagStore::remove_node_and_reconnect(NodeId id, const WireId* wires, std::size_t num_wires) {
    struct Reconnect {
        NodeId pred;
        NodeId succ;
        WireId wire;
    };
    SmallVector<Reconnect, 4> reconnect;
    for (std::size_t i = 0; i < num_wires; ++i) {
        const auto pred = wire_predecessor(id, wires[i]);
        const auto succ = wire_successor(id, wires[i]);
        if (pred && succ) {
            reconnect.push_back(Reconnect{*pred, *succ, wires[i]});
        }
    }
    remove_node(id);
    for (const Reconnect& r : reconnect) {
        add_edge(r.pred, r.succ, r.wire);
    }
}

DagNodeKind DagStore::kind(NodeId id) const {
    const NodeSlot* s = slot(id);
    if (s == nullptr) {
        throw DagKeyError(id);
    }
    return s->kind;
}

void DagStore::node_ids(std::vector<NodeId>& out) const {
    out.clear();
    out.reserve(num_alive_);
    for (const NodeSlot& s : slots_) {
        if (s.alive) {
            out.push_back(s.id);
        }
    }
}

const DagStore::WireList* DagStore::node_wires(NodeId id) const noexcept {
    const NodeSlot* s = slot(id);
    return s == nullptr ? nullptr : &s->wires;
}

bool DagStore::on_member_wire(NodeId id, WireId member) const noexcept {
    const NodeSlot* s = slot(id);
    return s != nullptr && Contains(s->wires, member);
}

// ---------------------------------------------------------------------------
// 边
// ---------------------------------------------------------------------------

void DagStore::link(EdgeSeq seq, const DagEdge& edge) {
    const auto k = static_cast<std::size_t>(seq);
    if (k >= edges_.size()) {
        edges_.resize(std::max(k + 1, edges_.size() * 2));
    }
    edges_[k] = EdgeRec{edge, true};
    ++num_edges_;
    const DagAdj out_entry{seq, edge.target, edge.wire};
    const DagAdj in_entry{seq, edge.source, edge.wire};
    if (seq >= next_seq_) {
        next_seq_ = seq + 1;
        adj_[edge.source].out.push_back(out_entry);
        adj_[edge.target].in.push_back(in_entry);
    } else {
        // 恢复较早的边：保持邻接表按边序号有序。
        InsertBySeq(adj_[edge.source].out, out_entry);
        InsertBySeq(adj_[edge.target].in, in_entry);
    }
    touch();
}

DagEdge DagStore::unlink(EdgeSeq seq) {
    EdgeRec& rec = edges_[static_cast<std::size_t>(seq)];
    rec.alive = false;
    --num_edges_;
    const DagEdge edge = rec.edge;
    if (Adjacency* a = adj_.find(edge.source)) {
        EraseBySeq(a->out, seq);
    }
    if (Adjacency* a = adj_.find(edge.target)) {
        EraseBySeq(a->in, seq);
    }
    touch();
    return edge;
}

std::optional<EdgeSeq> DagStore::find_edge_seq(NodeId source, NodeId target, WireId wire) const noexcept {
    for (const DagAdj& e : out_adj(source)) {
        if (e.nbr == target && e.wire == wire) {
            return e.seq;
        }
    }
    return std::nullopt;
}

void DagStore::add_edge(NodeId source, NodeId target, WireId wire, bool defer_order_check) {
    const EdgeSeq seq = next_seq_;
    link(seq, DagEdge{source, target, wire});
    if (journal_open_) {
        journal_.push_back(JournalEntry{JournalKind::kAdd, seq, DagEdge{}, 0, nullptr});
    }
    if (defer_order_check || !pos_) {
        return;
    }
    const std::int64_t* ps = pos_->find(source);
    const std::int64_t* pt = pos_->find(target);
    if (ps == nullptr || pt == nullptr || *ps >= *pt) {
        invalidate_positions();
    }
}

void DagStore::remove_edge(NodeId source, NodeId target, WireId wire) {
    const auto seq = find_edge_seq(source, target, wire);
    if (!seq) {
        throw DagEdgeNotFound();
    }
    const DagEdge edge = unlink(*seq);
    if (journal_open_) {
        journal_.push_back(JournalEntry{JournalKind::kRemove, *seq, edge, 0, nullptr});
    }
}

void DagStore::edges(std::vector<DagEdge>& out) const {
    out.clear();
    out.reserve(num_edges_);
    for (const EdgeRec& rec : edges_) {
        if (rec.alive) {
            out.push_back(rec.edge);
        }
    }
}

const DagStore::AdjList& DagStore::out_adj(NodeId id) const noexcept {
    const Adjacency* a = adj_.find(id);
    return a == nullptr ? EmptyAdj() : a->out;
}

const DagStore::AdjList& DagStore::in_adj(NodeId id) const noexcept {
    const Adjacency* a = adj_.find(id);
    return a == nullptr ? EmptyAdj() : a->in;
}

bool DagStore::has_edge(NodeId source, NodeId target, std::optional<WireId> wire) const noexcept {
    for (const DagAdj& e : out_adj(source)) {
        if (e.nbr == target && (!wire || e.wire == *wire)) {
            return true;
        }
    }
    return false;
}

std::optional<NodeId> DagStore::wire_predecessor(NodeId id, WireId wire) const noexcept {
    std::optional<NodeId> result;
    for (const DagAdj& e : in_adj(id)) {
        if (e.wire == wire) {
            result = e.nbr;
        }
    }
    return result;
}

std::optional<NodeId> DagStore::wire_successor(NodeId id, WireId wire) const noexcept {
    std::optional<NodeId> result;
    for (const DagAdj& e : out_adj(id)) {
        if (e.wire == wire) {
            result = e.nbr;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// 事务
// ---------------------------------------------------------------------------

std::int64_t DagStore::begin_transaction() {
    if (!journal_open_) {
        journal_open_ = true;
        journal_.clear();
    }
    ++txn_depth_;
    return static_cast<std::int64_t>(journal_.size());
}

void DagStore::rollback(std::int64_t savepoint) {
    if (!journal_open_) {
        throw std::logic_error("rollback() called without an open transaction.");
    }
    while (static_cast<std::int64_t>(journal_.size()) > savepoint) {
        JournalEntry entry = std::move(journal_.back());
        journal_.pop_back();
        switch (entry.kind) {
            case JournalKind::kAdd: {
                const auto k = static_cast<std::size_t>(entry.key);
                if (k < edges_.size() && edges_[k].alive) {
                    unlink(entry.key);
                }
                break;
            }
            case JournalKind::kRemove:
                link(entry.key, entry.edge);
                break;
            case JournalKind::kPos:
                if (pos_) {
                    (*pos_)[entry.key] = entry.old_pos;
                }
                break;
            case JournalKind::kPosInvalidate:
                pos_ = std::move(entry.positions);
                break;
        }
    }
}

void DagStore::end_transaction() noexcept {
    if (txn_depth_ > 0) {
        --txn_depth_;
    }
    if (txn_depth_ == 0) {
        journal_open_ = false;
        journal_.clear();
    }
}

// ---------------------------------------------------------------------------
// 拓扑
// ---------------------------------------------------------------------------

const std::vector<NodeId>& DagStore::topological_order() {
    if (topo_valid_) {
        return topo_cache_;
    }
    const std::size_t n = slots_.size();
    std::vector<std::int64_t> in_degree(n, 0);
    std::vector<std::int32_t> queue;
    queue.reserve(num_alive_);
    for (std::size_t i = 0; i < n; ++i) {
        if (!slots_[i].alive) {
            continue;
        }
        in_degree[i] = static_cast<std::int64_t>(in_adj(slots_[i].id).size());
        if (in_degree[i] == 0) {
            queue.push_back(static_cast<std::int32_t>(i));
        }
    }
    std::vector<NodeId> result;
    result.reserve(num_alive_);
    for (std::size_t head = 0; head < queue.size(); ++head) {
        const NodeId id = slots_[static_cast<std::size_t>(queue[head])].id;
        result.push_back(id);
        for (const DagAdj& e : out_adj(id)) {
            const std::int32_t* succ = slot_of_.find(e.nbr);
            if (succ == nullptr) {
                throw DagKeyError(e.nbr);
            }
            if (--in_degree[static_cast<std::size_t>(*succ)] == 0) {
                queue.push_back(*succ);
            }
        }
    }
    if (result.size() != num_alive_) {
        throw DagCycleError();
    }
    topo_cache_ = std::move(result);
    topo_valid_ = true;
    return topo_cache_;
}

void DagStore::invalidate_positions() {
    if (!pos_) {
        return;
    }
    if (journal_open_) {
        journal_.push_back(JournalEntry{JournalKind::kPosInvalidate, 0, DagEdge{}, 0, pos_});
    }
    pos_.reset();
}

DagStore::PosMap& DagStore::ensure_positions() {
    if (!pos_) {
        const std::vector<NodeId>& order = topological_order();
        auto fresh = std::make_shared<PosMap>();
        for (std::size_t k = 0; k < order.size(); ++k) {
            (*fresh)[order[k]] = static_cast<std::int64_t>(k);
        }
        pos_ = std::move(fresh);
        next_pos_ = static_cast<std::int64_t>(order.size());
    }
    return *pos_;
}

void DagStore::ensure_topological_positions() {
    ensure_positions();
}

bool DagStore::reverse_adjacent_edge_order(NodeId first, NodeId second) {
    if (!pos_ || !pos_->contains(first) || !pos_->contains(second)) {
        // 没有可用的编号：对重连后的图做一次完整（带环检测）的拓扑排序并采用它。
        const std::vector<NodeId>* order = nullptr;
        try {
            order = &topological_order();
        } catch (const DagCycleError&) {
            return false;
        }
        if (journal_open_) {
            journal_.push_back(JournalEntry{JournalKind::kPosInvalidate, 0, DagEdge{}, 0, pos_});
        }
        auto fresh = std::make_shared<PosMap>();
        for (std::size_t k = 0; k < order->size(); ++k) {
            (*fresh)[(*order)[k]] = static_cast<std::int64_t>(k);
        }
        pos_ = std::move(fresh);
        next_pos_ = static_cast<std::int64_t>(order->size());
        return true;
    }
    PosMap& pos = *pos_;
    const std::int64_t lb = *pos.find(first);
    const std::int64_t ub = *pos.find(second);
    if (lb > ub) {
        return true;  // 已经与新顺序一致
    }

    // 从 first 向前搜索，以 second 的编号为界。
    std::vector<NodeId> forward{first};
    SeenSet seen_f;
    seen_f.insert(first);
    std::vector<NodeId> stack{first};
    while (!stack.empty()) {
        const NodeId nid = stack.back();
        stack.pop_back();
        for (const DagAdj& e : out_adj(nid)) {
            if (e.nbr == second) {
                return false;
            }
            if (!seen_f.contains(e.nbr)) {
                const std::int64_t* p = pos.find(e.nbr);
                if (p != nullptr && *p < ub) {
                    seen_f.insert(e.nbr);
                    forward.push_back(e.nbr);
                    stack.push_back(e.nbr);
                }
            }
        }
    }
    // 从 second 向后搜索，以 first 的编号为界。
    std::vector<NodeId> backward{second};
    SeenSet seen_b;
    seen_b.insert(second);
    stack.assign(1, second);
    while (!stack.empty()) {
        const NodeId nid = stack.back();
        stack.pop_back();
        for (const DagAdj& e : in_adj(nid)) {
            if (!seen_b.contains(e.nbr)) {
                const std::int64_t* p = pos.find(e.nbr);
                if (p != nullptr && *p > lb) {
                    seen_b.insert(e.nbr);
                    backward.push_back(e.nbr);
                    stack.push_back(e.nbr);
                }
            }
        }
    }

    auto by_pos = [&pos](NodeId a, NodeId b) { return *pos.find(a) < *pos.find(b); };
    std::stable_sort(backward.begin(), backward.end(), by_pos);
    std::stable_sort(forward.begin(), forward.end(), by_pos);
    std::vector<NodeId> moved = std::move(backward);
    moved.insert(moved.end(), forward.begin(), forward.end());
    std::vector<std::int64_t> slots;
    slots.reserve(moved.size());
    for (NodeId nid : moved) {
        slots.push_back(*pos.find(nid));
    }
    std::sort(slots.begin(), slots.end());
    for (std::size_t k = 0; k < moved.size(); ++k) {
        std::int64_t& p = *pos.find(moved[k]);
        if (journal_open_) {
            journal_.push_back(JournalEntry{JournalKind::kPos, moved[k], DagEdge{}, p, nullptr});
        }
        p = slots[k];
    }
    return true;
}

std::vector<std::vector<NodeId>> DagStore::layers() {
    const std::vector<NodeId>& topo = topological_order();
    std::vector<std::int64_t> depth(slots_.size(), 0);
    std::int64_t max_depth = 0;
    for (NodeId nid : topo) {
        const AdjList& preds = in_adj(nid);
        std::int64_t d = 0;
        if (!preds.empty()) {
            std::int64_t best = -1;
            for (const DagAdj& e : preds) {
                const std::int32_t* ps = slot_of_.find(e.nbr);
                if (ps == nullptr) {
                    throw DagKeyError(e.nbr);
                }
                best = std::max(best, depth[static_cast<std::size_t>(*ps)]);
            }
            d = 1 + best;
        }
        depth[static_cast<std::size_t>(*slot_of_.find(nid))] = d;
        max_depth = std::max(max_depth, d);
    }
    std::vector<std::vector<NodeId>> result(static_cast<std::size_t>(max_depth + 1));
    for (NodeId nid : topo) {
        result[static_cast<std::size_t>(depth[static_cast<std::size_t>(*slot_of_.find(nid))])].push_back(nid);
    }
    return result;
}

std::int64_t DagStore::critical_path_length() {
    const auto l = layers();
    if (l.size() <= 2) {
        return 0;
    }
    return static_cast<std::int64_t>(l.size()) - 2;
}

// ---------------------------------------------------------------------------
// 沿线查询
// ---------------------------------------------------------------------------

bool DagStore::walk_wire(WireId wire, std::optional<WireId> member, std::vector<NodeId>& out) const {
    out.clear();
    if (const std::uint8_t* amb = ambiguous_wires_.find(wire); amb != nullptr) {
        return false;
    }
    const NodeId* start = wire_input_.find(wire);
    if (start == nullptr || !has_node(*start) || !member) {
        return false;
    }
    const std::int64_t* count = wire_count_.find(wire);
    const std::size_t expected = count == nullptr || *count < 0 ? 0 : static_cast<std::size_t>(*count);
    out.push_back(*start);
    NodeId nid = *start;
    while (true) {
        std::optional<NodeId> nxt;
        for (const DagAdj& e : out_adj(nid)) {
            if (e.wire == wire) {
                if (nxt) {
                    return false;
                }
                nxt = e.nbr;
            }
        }
        if (!nxt) {
            break;
        }
        const NodeSlot* s = slot(*nxt);
        if (s == nullptr || !Contains(s->wires, *member)) {
            return false;
        }
        out.push_back(*nxt);
        if (out.size() > expected) {
            return false;
        }
        nid = *nxt;
    }
    return out.size() == expected;
}

void DagStore::scan_wire(WireId member, std::vector<NodeId>& out) {
    out.clear();
    for (NodeId nid : topological_order()) {
        if (on_member_wire(nid, member)) {
            out.push_back(nid);
        }
    }
}

// ---------------------------------------------------------------------------
// 状态导出 / 导入
// ---------------------------------------------------------------------------

namespace {

std::vector<std::pair<NodeId, std::int64_t>> PositionsOf(const IdMap<std::int64_t>& map) {
    std::vector<std::pair<NodeId, std::int64_t>> out;
    map.for_each([&](std::int64_t k, std::int64_t v) { out.emplace_back(k, v); });
    std::sort(out.begin(), out.end());
    return out;
}

std::shared_ptr<IdMap<std::int64_t>> PositionMap(const std::vector<std::pair<NodeId, std::int64_t>>& entries) {
    auto map = std::make_shared<IdMap<std::int64_t>>();
    for (const auto& [k, v] : entries) {
        (*map)[k] = v;
    }
    return map;
}

}  // namespace

DagState DagStore::export_state() const {
    DagState st;
    for (const NodeSlot& s : slots_) {
        if (s.alive) {
            st.nodes.push_back(DagState::Node{s.id, s.kind, std::vector<WireId>(s.wires.begin(), s.wires.end())});
        }
    }
    for (std::size_t seq = 0; seq < edges_.size(); ++seq) {
        if (edges_[seq].alive) {
            st.edges.push_back(DagState::Edge{static_cast<EdgeSeq>(seq), edges_[seq].edge});
        }
    }
    st.next_seq = next_seq_;
    wire_input_.for_each([&](std::int64_t w, NodeId n) { st.wire_input.emplace_back(w, n); });
    ambiguous_wires_.for_each([&](std::int64_t w, std::uint8_t) { st.ambiguous_wires.push_back(w); });
    wire_count_.for_each([&](std::int64_t w, std::int64_t c) { st.wire_count.emplace_back(w, c); });
    std::sort(st.wire_input.begin(), st.wire_input.end());
    std::sort(st.ambiguous_wires.begin(), st.ambiguous_wires.end());
    std::sort(st.wire_count.begin(), st.wire_count.end());
    st.num_qubits = num_qubits_;
    st.num_clbits = num_clbits_;
    st.version = version_;
    st.has_positions = static_cast<bool>(pos_);
    if (pos_) {
        st.positions = PositionsOf(*pos_);
    }
    st.next_pos = next_pos_;
    st.journal_open = journal_open_;
    for (const JournalEntry& e : journal_) {
        DagState::JournalEntry je{static_cast<std::uint8_t>(e.kind), e.key, e.edge, e.old_pos,
                                  static_cast<bool>(e.positions), {}};
        if (e.positions) {
            je.positions = PositionsOf(*e.positions);
        }
        st.journal.push_back(std::move(je));
    }
    st.txn_depth = txn_depth_;
    return st;
}

void DagStore::import_state(const DagState& st) {
    DagStore fresh;
    for (const DagState::Node& n : st.nodes) {
        if (fresh.slot_of_.contains(n.id)) {
            throw std::invalid_argument("duplicate node id in DAG state");
        }
        NodeSlot s;
        s.id = n.id;
        s.kind = n.kind;
        s.alive = true;
        s.wires.assign(n.wires.begin(), n.wires.end());
        fresh.slot_of_[n.id] = static_cast<std::int32_t>(fresh.slots_.size());
        fresh.slots_.push_back(std::move(s));
        ++fresh.num_alive_;
    }
    for (const DagState::Edge& e : st.edges) {
        if (e.seq < 0 || e.seq >= st.next_seq
            || (static_cast<std::size_t>(e.seq) < fresh.edges_.size() && fresh.edges_[e.seq].alive)) {
            throw std::invalid_argument("invalid edge sequence number in DAG state");
        }
        fresh.next_seq_ = std::max(fresh.next_seq_, e.seq);
        fresh.link(e.seq, e.edge);  // 按序号递增加入：邻接表自然有序
    }
    fresh.next_seq_ = st.next_seq;
    for (const auto& [w, n] : st.wire_input) {
        fresh.wire_input_[w] = n;
    }
    for (WireId w : st.ambiguous_wires) {
        fresh.ambiguous_wires_[w] = 1;
    }
    for (const auto& [w, c] : st.wire_count) {
        fresh.wire_count_[w] = c;
    }
    fresh.num_qubits_ = st.num_qubits;
    fresh.num_clbits_ = st.num_clbits;
    fresh.version_ = st.version;
    fresh.topo_valid_ = false;
    if (st.has_positions) {
        fresh.pos_ = PositionMap(st.positions);
    }
    fresh.next_pos_ = st.next_pos;
    fresh.journal_open_ = st.journal_open;
    for (const DagState::JournalEntry& je : st.journal) {
        if (je.kind > 3) {
            throw std::invalid_argument("invalid journal entry in DAG state");
        }
        fresh.journal_.push_back(JournalEntry{static_cast<JournalKind>(je.kind), je.key, je.edge, je.old_pos,
                                              je.has_positions ? PositionMap(je.positions) : nullptr});
    }
    fresh.txn_depth_ = st.txn_depth;
    *this = std::move(fresh);
}

}  // namespace quspect::native::ir
