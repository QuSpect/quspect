// SmallVector / IdMap 的 gtest：内联存储与堆迁移、插入删除保持顺序、拷贝移动语义，
// 以及 IdMap 稠密 / 稀疏两部分的存在性语义。
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/ir/id_map.hpp"
#include "quspect_native/ir/small_vector.hpp"

namespace {

using quspect::native::ir::IdMap;
using quspect::native::ir::SmallVector;

TEST(SmallVector, StaysInlineUpToCapacity) {
    SmallVector<int, 3> v;
    EXPECT_TRUE(v.empty());
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    EXPECT_TRUE(v.is_inline());
    EXPECT_EQ(v.size(), 3u);
    v.push_back(4);
    EXPECT_FALSE(v.is_inline());
    EXPECT_EQ(std::vector<int>(v.begin(), v.end()), (std::vector<int>{1, 2, 3, 4}));
}

TEST(SmallVector, InsertAndEraseKeepOrder) {
    SmallVector<int, 2> v{1, 3, 5};
    v.insert(v.begin() + 1, 2);
    v.insert(v.end(), 6);
    v.insert(v.begin() + 4, 4);
    EXPECT_EQ(std::vector<int>(v.begin(), v.end()), (std::vector<int>{1, 2, 3, 5, 4, 6}));
    v.erase(v.begin() + 4);
    v.erase(v.begin());
    EXPECT_EQ(std::vector<int>(v.begin(), v.end()), (std::vector<int>{2, 3, 5, 6}));
}

TEST(SmallVector, PushBackOfOwnElementDuringGrowth) {
    SmallVector<int, 2> v{7, 8};
    v.push_back(v[0]);  // 引用自身缓冲区，扩容时不能失效
    EXPECT_EQ(std::vector<int>(v.begin(), v.end()), (std::vector<int>{7, 8, 7}));
}

TEST(SmallVector, CopyAndMove) {
    SmallVector<int, 2> a{1, 2, 3, 4};
    SmallVector<int, 2> b = a;
    EXPECT_EQ(a, b);
    SmallVector<int, 2> c = std::move(a);
    EXPECT_EQ(c, b);
    EXPECT_TRUE(a.empty());
    SmallVector<int, 2> d{9};
    d = c;
    EXPECT_EQ(d, b);
    SmallVector<int, 2> e{5};
    SmallVector<int, 2> f = std::move(e);
    EXPECT_EQ(std::vector<int>(f.begin(), f.end()), (std::vector<int>{5}));
    EXPECT_TRUE(f.is_inline());
}

TEST(IdMap, DenseAndSparseKeys) {
    IdMap<int> m;
    EXPECT_FALSE(m.contains(0));
    m[0] = 10;
    m[5] = 50;
    m[-3] = -30;
    m[IdMap<int>::kDenseLimit + 7] = 70;
    EXPECT_EQ(m.size(), 4u);
    EXPECT_EQ(*m.find(5), 50);
    EXPECT_EQ(*m.find(-3), -30);
    EXPECT_EQ(*m.find(IdMap<int>::kDenseLimit + 7), 70);
    EXPECT_EQ(m.find(4), nullptr);
    EXPECT_TRUE(m.erase(5));
    EXPECT_FALSE(m.erase(5));
    EXPECT_TRUE(m.erase(-3));
    EXPECT_EQ(m.size(), 2u);
    // 删除后重新插入得到默认值
    EXPECT_EQ(m[5], 0);
}

}  // namespace
