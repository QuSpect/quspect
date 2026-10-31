#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <type_traits>

namespace quspect::native::ir {

// 带内联缓冲区的小向量（只用于平凡可复制类型）：元素不超过 N 个时存放在对象内部，
// 不做堆分配；超过后整体搬到堆上，按 2 倍扩容。量子线路中指令的比特列表、DAG 节点
// 的邻接表几乎总是 1–3 个元素，用它可以避免逐节点的小块堆分配并保持数据局部性。
template <typename T, std::size_t N>
class SmallVector {
    static_assert(std::is_trivially_copyable<T>::value, "SmallVector requires a trivially copyable type");
    static_assert(N > 0, "SmallVector requires a non-empty inline buffer");

public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    SmallVector() noexcept {  // NOLINT(modernize-use-equals-default): 内联缓冲区不需要初始化
    }

    SmallVector(std::initializer_list<T> init) {
        assign(init.begin(), init.end());
    }

    template <typename It>
    SmallVector(It first, It last) {
        assign(first, last);
    }

    SmallVector(const SmallVector& other) {
        assign(other.begin(), other.end());
    }

    SmallVector(SmallVector&& other) noexcept {
        steal(other);
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            release();
            steal(other);
        }
        return *this;
    }

    ~SmallVector() {
        release();
    }

    template <typename It>
    void assign(It first, It last) {
        const auto n = static_cast<size_type>(std::distance(first, last));
        size_ = 0;
        reserve(n);
        std::copy(first, last, data());
        size_ = n;
    }

    T* data() noexcept {
        return heap_ != nullptr ? heap_ : inline_ptr();
    }
    const T* data() const noexcept {
        return heap_ != nullptr ? heap_ : inline_ptr();
    }

    size_type size() const noexcept {
        return size_;
    }
    size_type capacity() const noexcept {
        return heap_ != nullptr ? capacity_ : N;
    }
    bool empty() const noexcept {
        return size_ == 0;
    }
    bool is_inline() const noexcept {
        return heap_ == nullptr;
    }

    iterator begin() noexcept {
        return data();
    }
    iterator end() noexcept {
        return data() + size_;
    }
    const_iterator begin() const noexcept {
        return data();
    }
    const_iterator end() const noexcept {
        return data() + size_;
    }

    T& operator[](size_type i) noexcept {
        return data()[i];
    }
    const T& operator[](size_type i) const noexcept {
        return data()[i];
    }
    T& back() noexcept {
        return data()[size_ - 1];
    }
    const T& back() const noexcept {
        return data()[size_ - 1];
    }

    void reserve(size_type n) {
        if (n <= capacity()) {
            return;
        }
        size_type cap = std::max<size_type>(n, capacity() * 2);
        T* fresh = static_cast<T*>(std::malloc(cap * sizeof(T)));
        if (fresh == nullptr) {
            throw std::bad_alloc();
        }
        if (size_ > 0) {
            std::memcpy(static_cast<void*>(fresh), static_cast<const void*>(data()), size_ * sizeof(T));
        }
        if (heap_ != nullptr) {
            std::free(heap_);
        }
        heap_ = fresh;
        capacity_ = cap;
    }

    void push_back(const T& value) {
        if (size_ == capacity()) {
            T copy = value;  // value 可能指向自身缓冲区
            reserve(size_ + 1);
            data()[size_++] = copy;
            return;
        }
        data()[size_++] = value;
    }

    void pop_back() noexcept {
        --size_;
    }

    void clear() noexcept {
        size_ = 0;
    }

    void resize(size_type n, const T& value = T()) {
        reserve(n);
        for (size_type i = size_; i < n; ++i) {
            data()[i] = value;
        }
        size_ = n;
    }

    // 在 pos 处插入一个元素（pos 之后的元素后移）。
    iterator insert(const_iterator pos, const T& value) {
        const auto idx = static_cast<size_type>(pos - begin());
        T copy = value;
        reserve(size_ + 1);
        T* d = data();
        std::memmove(static_cast<void*>(d + idx + 1), static_cast<const void*>(d + idx), (size_ - idx) * sizeof(T));
        d[idx] = copy;
        ++size_;
        return d + idx;
    }

    // 删除 pos 处的元素（保持其余元素的相对顺序）。
    iterator erase(const_iterator pos) noexcept {
        const auto idx = static_cast<size_type>(pos - begin());
        T* d = data();
        std::memmove(static_cast<void*>(d + idx), static_cast<const void*>(d + idx + 1),
                     (size_ - idx - 1) * sizeof(T));
        --size_;
        return d + idx;
    }

    friend bool operator==(const SmallVector& a, const SmallVector& b) {
        return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
    }
    friend bool operator!=(const SmallVector& a, const SmallVector& b) {
        return !(a == b);
    }

private:
    T* inline_ptr() noexcept {
        return reinterpret_cast<T*>(inline_);
    }
    const T* inline_ptr() const noexcept {
        return reinterpret_cast<const T*>(inline_);
    }

    void release() noexcept {
        if (heap_ != nullptr) {
            std::free(heap_);
            heap_ = nullptr;
        }
        size_ = 0;
    }

    void steal(SmallVector& other) noexcept {
        size_ = other.size_;
        if (other.heap_ != nullptr) {
            heap_ = other.heap_;
            capacity_ = other.capacity_;
            other.heap_ = nullptr;
        } else {
            heap_ = nullptr;
            std::memcpy(static_cast<void*>(inline_), static_cast<const void*>(other.inline_), size_ * sizeof(T));
        }
        other.size_ = 0;
    }

    T* heap_ = nullptr;
    size_type size_ = 0;
    size_type capacity_ = 0;
    alignas(T) unsigned char inline_[N * sizeof(T)];
};

}  // namespace quspect::native::ir
