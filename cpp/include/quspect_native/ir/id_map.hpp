#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspect::native::ir {

// 以 64 位整数为键的映射：非负且不太大的键（线路中节点编号、比特编号几乎总是 0..N 连续）
// 存放在按键下标的稠密数组中，O(1) 访问且缓存友好；负数或过大的键退回哈希表。
// 语义与 std::unordered_map 相同（存在 / 不存在），只是不保证遍历顺序。
template <typename V>
class IdMap {
public:
    // 稠密部分的键上限：超过的键放进哈希表，避免一个离群的大编号撑爆数组。
    static constexpr std::int64_t kDenseLimit = std::int64_t{1} << 26;

    V* find(std::int64_t key) noexcept {
        if (is_dense(key)) {
            const auto k = static_cast<std::size_t>(key);
            return k < present_.size() && present_[k] ? &dense_[k] : nullptr;
        }
        auto it = sparse_.find(key);
        return it == sparse_.end() ? nullptr : &it->second;
    }
    const V* find(std::int64_t key) const noexcept {
        return const_cast<IdMap*>(this)->find(key);
    }

    bool contains(std::int64_t key) const noexcept {
        return find(key) != nullptr;
    }

    // 返回 key 对应的值，不存在时先以 V{} 插入。
    V& operator[](std::int64_t key) {
        if (is_dense(key)) {
            const auto k = static_cast<std::size_t>(key);
            if (k >= present_.size()) {
                std::size_t n = std::max<std::size_t>(k + 1, present_.size() * 2);
                n = std::max<std::size_t>(n, 16);
                present_.resize(n, 0);
                dense_.resize(n);
            }
            if (!present_[k]) {
                present_[k] = 1;
                dense_[k] = V{};
                ++size_;
            }
            return dense_[k];
        }
        auto [it, inserted] = sparse_.try_emplace(key);
        if (inserted) {
            ++size_;
        }
        return it->second;
    }

    // 删除 key；返回是否存在。
    bool erase(std::int64_t key) {
        if (is_dense(key)) {
            const auto k = static_cast<std::size_t>(key);
            if (k < present_.size() && present_[k]) {
                present_[k] = 0;
                dense_[k] = V{};
                --size_;
                return true;
            }
            return false;
        }
        if (sparse_.erase(key) > 0) {
            --size_;
            return true;
        }
        return false;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    void clear() {
        dense_.clear();
        present_.clear();
        sparse_.clear();
        size_ = 0;
    }

    // 预留稠密部分的容量（键 0..n-1）。
    void reserve_dense(std::size_t n) {
        if (n > present_.size() && static_cast<std::int64_t>(n) <= kDenseLimit) {
            present_.resize(n, 0);
            dense_.resize(n);
        }
    }

private:
    static bool is_dense(std::int64_t key) noexcept {
        return key >= 0 && key < kDenseLimit;
    }

    std::vector<V> dense_;
    std::vector<std::uint8_t> present_;
    std::unordered_map<std::int64_t, V> sparse_;
    std::size_t size_ = 0;
};

}  // namespace quspect::native::ir
