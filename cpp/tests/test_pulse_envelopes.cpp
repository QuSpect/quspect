// quspect::native::pulse 包络核心的单元测试。
//
// 期望值来自闭式公式，或来自 envelopes-qc 0.1.5 原实现的实测结果（序列化字节、Custom 指纹、
// 解码窗口长度等）。与原实现的逐位比对见 tools/pulse_compat/。

#include <cmath>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/pulse/envelopes.hpp"

namespace quspect::native::pulse {
namespace {

std::string Hex(const std::string& bytes) {
    static const char* kDigits = "0123456789abcdef";
    std::string out;
    for (unsigned char c : bytes) {
        out += kDigits[c >> 4];
        out += kDigits[c & 0xf];
    }
    return out;
}

EnvPtr MakeGaussian() {
    return std::make_shared<Gaussian>(10.0, 4.0, 0.7f);
}

EnvPtr MakeDrag() {
    return std::make_shared<GaussianDrag>(15.0, 10.0, 1.0f, 0.5, 0.1, 0.2);
}

TEST(PulseEnvelopes, GaussianMatchesClosedForm) {
    WaveCache wc(0.5);
    Decoded d = DecodeEnvelope(MakeGaussian(), wc);
    ASSERT_TRUE(d.real);
    EXPECT_FALSE(d.complex);
    EXPECT_DOUBLE_EQ(d.t, -2.0);
    // 24 ns / 0.5 ns + 2 个尾部填充采样
    ASSERT_EQ(d.size, 50);
    const double sigma = 4.0 / std::sqrt(8.0 * std::log(2.0));
    for (long long k = 0; k < d.size; ++k) {
        const double t = d.t + 0.5 * static_cast<double>(k);
        const double expected = 0.7 * std::exp(-(t - 10.0) * (t - 10.0) / (2.0 * sigma * sigma));
        EXPECT_NEAR((*d.real)[k], expected, 1e-6) << "k=" << k;
    }
}

TEST(PulseEnvelopes, DragIsComplexAndCarriesPhase) {
    WaveCache wc(0.5);
    Decoded d = DecodeEnvelope(MakeDrag(), wc);
    ASSERT_TRUE(d.complex);
    // 峰值处 (t = 15) 只剩载波相位 -2*pi*df*t + phase
    const long long peak = static_cast<long long>((15.0 - d.t) / 0.5);
    const double phase = -2.0 * 3.141592653589793 * 0.1 * 15.0 + 0.2;
    EXPECT_NEAR((*d.complex)[peak].real(), std::cos(phase), 1e-5);
    EXPECT_NEAR((*d.complex)[peak].imag(), std::sin(phase), 1e-5);
}

TEST(PulseEnvelopes, RectAndTriangleEdges) {
    WaveCache wc(0.5);
    Decoded rect = DecodeEnvelope(std::make_shared<Rect>(0.0, 5.0, 1.0f), wc);
    ASSERT_EQ(rect.size, 12);
    for (int k = 0; k < 10; ++k)
        EXPECT_EQ((*rect.real)[k], 1.0f);
    EXPECT_EQ((*rect.real)[10], 0.0f);
    Decoded tri = DecodeEnvelope(std::make_shared<Triangle>(0.0, 4.0, 1.0f, true), wc);
    EXPECT_FLOAT_EQ((*tri.real)[0], 1.0f);
    EXPECT_FLOAT_EQ((*tri.real)[4], 0.5f);
    EXPECT_FLOAT_EQ((*tri.real)[8], 0.0f);
}

TEST(PulseEnvelopes, RippleRectWithoutRipplesIsSmoothedRect) {
    WaveCache wc(0.5);
    Decoded ripple = DecodeEnvelope(std::make_shared<RippleRect>(0.0, 20.0, 1.0f, 2.0, 0, 0, 0, 0), wc);
    const double sigma = (2.0 + 1e-10) / 2.0 / std::sqrt(std::log(2.0));
    for (long long k = 0; k < ripple.size; ++k) {
        const double t = ripple.t + 0.5 * static_cast<double>(k);
        const double expected = 0.5 * (std::erf(t / sigma) - std::erf((t - 20.0) / sigma));
        EXPECT_NEAR((*ripple.real)[k], expected, 1e-6) << "k=" << k;
    }
}

TEST(PulseEnvelopes, GaussianSerializationLayout) {
    // u64 节点数 | u8 类型标签 | t0 | w | amp(float32)，与原实现字节一致
    EXPECT_EQ(Hex(Serialize(MakeGaussian())),
              "0100000000000000"
              "00"
              "0000000000002440"
              "0000000000001040"
              "3333333f");
}

TEST(PulseEnvelopes, SerializationRoundTripKeepsSharing) {
    EnvPtr g = MakeGaussian();
    EnvPtr tree = Add(Mix(Add(Multiply(Shift(g, 1.0), 2.0f), MakeDrag()), 0.05, 0.0, true), Add(g, g));
    const std::string bytes = Serialize(tree);
    EXPECT_EQ(Serialize(Deserialize(bytes)), bytes);
    // 共享的叶子节点只序列化一次
    const std::string shared = Serialize(Add(g, g));
    EXPECT_EQ(Hex(shared.substr(0, 8)), "0200000000000000");
}

TEST(PulseEnvelopes, CustomIdMatchesReference) {
    auto data = std::make_shared<std::vector<float>>(std::vector<float>{1.0f, 2.0f, 3.0f});
    Custom custom(1.0, 0.5, data);
    EXPECT_EQ(custom.ToString(), "Custom_4ddde61976de0fa92d30b63d5604fa42dd60f06ef8e03e9bcc04327c56f9fba7");
    EXPECT_DOUBLE_EQ(custom.End(), 2.5);
}

TEST(PulseEnvelopes, CustomRejectsNonPositiveDt) {
    auto data = std::make_shared<std::vector<float>>(3, 1.0f);
    EXPECT_THROW(Custom(0.0, 0.0, data), std::runtime_error);
}

TEST(PulseEnvelopes, DecodeWindowLengths) {
    EnvPtr g = MakeGaussian();
    WaveCache wc(0.5);
    struct Case {
        double start, end, t;
        long long size;
    };
    const Case cases[] = {{-5.3, 30.7, -5.3, 72},    {-5.25, 30.25, -5.25, 72}, {-5.24, 30.26, -5.24, 71},
                          {-5.26, 30.24, -5.26, 71}, {3.3, 17.1, 3.3, 27},      {5.0, 5.0, 5.0, 0}};
    for (const Case& c : cases) {
        Decoded d = DecodeEnvelope(g, wc, c.start, c.end);
        EXPECT_DOUBLE_EQ(d.t, c.t);
        EXPECT_EQ(d.size, c.size) << c.start << ".." << c.end;
    }
    // 反向窗口得到负长度（Python 层抛出 ValueError）
    EXPECT_LT(DecodeEnvelope(g, wc, 10.0, 5.0).size, 0);
    EXPECT_THROW(DecodeEnvelope(g, wc, 30.0, 40.0), std::runtime_error);
    EXPECT_THROW(DecodeEnvelope(g, wc, -40.0, -30.0), std::runtime_error);
}

TEST(PulseEnvelopes, CacheKeysUseTruncatingModulo) {
    WaveCache wc(0.5);
    Decoded shifted = DecodeEnvelope(Shift(MakeGaussian(), 0.25), wc);
    // start = -1.75：截断除法得到 shift_t = -1.5、base = -25000（单位 1e-5）
    EXPECT_DOUBLE_EQ(shifted.t, -1.5);
    const std::string repr = wc.ToString();
    EXPECT_NE(repr.find("    -25000:\n      [400000 ]"), std::string::npos) << repr;
}

TEST(PulseEnvelopes, WaveCacheRejectsBadResolution) {
    EXPECT_THROW(WaveCache(0.0), std::invalid_argument);
    EXPECT_THROW(WaveCache(1e-7), std::invalid_argument);
}

TEST(PulseEnvelopes, RewritingRules) {
    EnvPtr g = MakeGaussian();
    EnvPtr d = MakeDrag();
    EXPECT_EQ(Shift(Multiply(g, 2.0f), 3.0)->ToString(), "EnvProd(EnvShift(Gaussian))");
    EXPECT_EQ(Mix(Add(g, d), 0.1, 0.0, true)->ToString(), "EnvSum(EnvMix(Gaussian), EnvMix(GaussianDRAG))");
    EXPECT_EQ(Align(Add(g, d), {0.0, 1.0})->ToString(),
              "EnvSum(EnvShift(Gaussian), EnvShift(Gaussian), EnvShift(GaussianDRAG), EnvShift(GaussianDRAG))");
    EXPECT_TRUE(Multiply(Multiply(g, 2.0f), CFloat(0.0f, 1.0f))->IsComplexProd());
    EXPECT_EQ(Add(g, 0.0), g);
    EXPECT_THROW(Add(g, 1.5), std::invalid_argument);
    EXPECT_THROW(Align(g, {0.0, 1.0}, std::vector<float>{1.0f}), std::runtime_error);
}

TEST(PulseEnvelopes, SplitAssignsItemsToBlocks) {
    EnvPtr seq = Align(MakeGaussian(), {0.0, 100.0});
    std::vector<EnvPtr> blocks = Split(seq, {-50.0, 60.0}, {50.0, 200.0});
    ASSERT_EQ(blocks.size(), 2u);
    EXPECT_EQ(blocks[0]->ToString(), "EnvSum(EnvShift(Gaussian))");
    EXPECT_THROW(Split(seq, {-50.0, 60.0}, {50.0, 105.0}), std::runtime_error);
    EXPECT_THROW(Split(seq, {-50.0, 200.0}, {50.0, 300.0}), std::runtime_error);
}

// ---- 原实现会段错误或卡死的输入 ----

TEST(PulseEnvelopes, EmptySumRaises) {
    EnvPtr empty = Align(MakeGaussian(), {});
    EXPECT_THROW(empty->Start(), std::runtime_error);
    WaveCache wc(0.5);
    EXPECT_THROW(DecodeEnvelope(empty, wc), std::runtime_error);
}

TEST(PulseEnvelopes, TruncatedDataRaises) {
    const std::string bytes = Serialize(Add(MakeGaussian(), MakeDrag()));
    for (std::size_t cut = 0; cut < bytes.size(); ++cut) {
        EXPECT_THROW(Deserialize(bytes.substr(0, cut)), std::runtime_error) << "cut=" << cut;
    }
    EXPECT_THROW(Deserialize(std::string("\x01\0\0\0\0\0\0\0\x63", 9)), std::runtime_error);
}

TEST(PulseEnvelopes, MixAndCopyOfRealScaledComplexEnvelope) {
    EnvPtr scaled = Multiply(MakeDrag(), 2.0f);
    EnvPtr mixed = Mix(scaled, 0.05, 0.3, true);
    EXPECT_EQ(mixed->ToString(), "EnvProd(EnvMix(GaussianDRAG))");
    WaveCache wc(0.5);
    Decoded a = DecodeEnvelope(mixed, wc);
    Decoded b = DecodeEnvelope(Multiply(Mix(MakeDrag(), 0.05, 0.3, true), 2.0f), wc);
    ASSERT_EQ(a.size, b.size);
    EXPECT_EQ(std::memcmp(a.complex->data(), b.complex->data(), a.complex->size() * sizeof(CFloat)), 0);
    EXPECT_EQ(Serialize(Copy(mixed)), Serialize(mixed));
}

TEST(PulseEnvelopes, SelfAppend) {
    auto sum = std::make_shared<EnvSum>(std::vector<EnvPtr>{MakeGaussian(), MakeDrag()});
    EnvPtr self = sum;
    InplaceAdd(self, self);
    EXPECT_EQ(sum->ToString(), "EnvSum(Gaussian, GaussianDRAG, Gaussian, GaussianDRAG)");
}

TEST(PulseEnvelopes, CopySharesLeavesButNotComposites) {
    EnvPtr g = MakeGaussian();
    EnvPtr shift = Shift(g, 1.0);
    EnvPtr tree = Add(Multiply(shift, 2.0f), Multiply(shift, 3.0f));
    // 原实现：叶子保持共享，组合节点按引用次数复制（序列化节点数 5 -> 6）
    EXPECT_EQ(Hex(Serialize(tree).substr(0, 8)), "0500000000000000");
    EXPECT_EQ(Hex(Serialize(Copy(tree)).substr(0, 8)), "0600000000000000");
}

}  // namespace
}  // namespace quspect::native::pulse
