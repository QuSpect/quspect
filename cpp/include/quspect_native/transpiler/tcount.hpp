#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// T 门数优化（quspect.transpiler.TCountOptimizePass）的原生搜索内核。
//
// 一个相位多项式写成一组奇偶列（k <= 64 个变量上的位掩码），T 门数即列数。这里实现三种
// 搜索，与 quspect/transpiler/_tcount_poly.py 中的 Python 参考实现逐步一致（相同的输入与
// 种子给出相同的输出）：
//   * TODD（Heyfron & Campbell 2018）：A -> A + z y^T，y 取自保持三次部分的约束的零空间，
//     每一步对所有 z = c_a + c_b 取删去列数最多的移动，可选的平台移动；
//   * Reed–Muller 平面搜索：与 4 维仿射平面的点集做异或，再接 TODD；
//   * 翻转图随机游走（Kauers & Moosbauer 2022）：GF(2) 上三线性张量的低秩分解。
// 随机数只以整数形式抽取（CPythonRandom 的 RandBelow / GetRandBits），不涉及浮点运算。
// 结果的正确性由调用方用精确的相位多项式恒等检查确认。

namespace quspect::native::transpiler {

using TCountColumns = std::vector<std::uint64_t>;
using TCountTerm = std::array<std::uint64_t, 3>;

// 256 位的列（w[0] 为最低的 64 位），用于多于 64 个变量的相位多项式。大小比较按整数值，
// 与 Python 中对整数排序的结果一致。
struct TCountWide {
    std::array<std::uint64_t, 4> w{};

    TCountWide operator^(const TCountWide& o) const {
        TCountWide r;
        for (std::size_t i = 0; i < 4; ++i)
            r.w[i] = w[i] ^ o.w[i];
        return r;
    }
    TCountWide& operator^=(const TCountWide& o) {
        for (std::size_t i = 0; i < 4; ++i)
            w[i] ^= o.w[i];
        return *this;
    }
    bool operator==(const TCountWide& o) const {
        return w == o.w;
    }
    bool operator!=(const TCountWide& o) const {
        return w != o.w;
    }
    bool operator<(const TCountWide& o) const {
        for (std::size_t i = 4; i-- > 0;)
            if (w[i] != o.w[i])
                return w[i] < o.w[i];
        return false;
    }
};
using TCountWideColumns = std::vector<TCountWide>;

template <class Col>
struct TCountToddProblemT {
    std::vector<Col> cols;
    int k = 0;
    int plateau = 0;
    std::uint64_t seed = 0;
};
using TCountToddProblem = TCountToddProblemT<std::uint64_t>;
using TCountToddProblemWide = TCountToddProblemT<TCountWide>;

template <class Col>
struct TCountRmProblemT {
    std::vector<Col> cols;
    int k = 0;
    int iterations = 0;
    std::uint64_t seed = 0;
};
using TCountRmProblem = TCountRmProblemT<std::uint64_t>;
using TCountRmProblemWide = TCountRmProblemT<TCountWide>;

struct TCountFlipProblem {
    std::vector<TCountTerm> start;
    std::int64_t steps = 0;
    std::uint64_t seed = 0;
};

// 对每个问题做 TODD；问题之间并行（只有一个问题时在候选 z 上并行）。threads <= 1 时单线程。
// 结果与线程数无关。k 必须在 [0, 64]，否则抛出 std::invalid_argument。
std::vector<TCountColumns> TCountToddBatch(const std::vector<TCountToddProblem>& problems, int threads);

// Reed–Muller 平面搜索：每步抽样 samples 个 4 维平面，取改进最大的；没有改进时做一次随机
// 平面扰动，以 2^-d 的概率接受（d 为列数的增加）。问题之间并行。
std::vector<TCountColumns> TCountRmSearch(const std::vector<TCountRmProblem>& problems, int samples, int threads);

// 同上，列为 256 位（k <= 256）。结果与 64 位版本在 k <= 64 时相同。
std::vector<TCountWideColumns> TCountToddBatchWide(const std::vector<TCountToddProblemWide>& problems, int threads);
std::vector<TCountWideColumns> TCountRmSearchWide(const std::vector<TCountRmProblemWide>& problems, int samples,
                                                  int threads);

// 翻转图随机游走；返回每个问题见到的秩最低的分解。连续 plus_after 步没有降秩时做一次 plus 移动。
std::vector<std::vector<TCountTerm>> TCountFlipWalks(const std::vector<TCountFlipProblem>& problems,
                                                     int plus_after, int threads);

}  // namespace quspect::native::transpiler

namespace std {
template <>
struct hash<quspect::native::transpiler::TCountWide> {
    size_t operator()(const quspect::native::transpiler::TCountWide& c) const noexcept {
        std::uint64_t h = 0x9E3779B97F4A7C15ull;
        for (std::uint64_t word : c.w) {
            h ^= word + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
        }
        return static_cast<size_t>(h);
    }
};
}  // namespace std
