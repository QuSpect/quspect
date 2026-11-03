// Pauli 旋转网络综合的 gtest：每个输出旋转在原始坐标下实现的 Pauli（含符号）等于输入，
// 实现顺序只交换对易的旋转，CX 只在耦合边上，收尾后框架为恒等；以及精确合并与参数检查。
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "quspect_native/common/bit_ops.hpp"
#include "quspect_native/synthesis/pauli_network.hpp"

namespace {

using quspect::native::synthesis::MergeCommutingRotations;
using quspect::native::synthesis::NetGate;
using quspect::native::synthesis::NetGateKind;
using quspect::native::synthesis::PauliNetworkOptions;
using quspect::native::synthesis::PauliNetworkResult;
using quspect::native::synthesis::SynthesizePauliNetwork;

// 小规模 Pauli（≤ 64 比特）：x、z 位与符号。
struct P {
    std::uint64_t x = 0;
    std::uint64_t z = 0;
    bool neg = false;
};

int Local(const P& p, int q) { return static_cast<int>(((p.x >> q) & 1U) | (((p.z >> q) & 1U) << 1)); }

void Set(P& p, int q, int v) {
    p.x = (p.x & ~(1ULL << q)) | (static_cast<std::uint64_t>(v & 1) << q);
    p.z = (p.z & ~(1ULL << q)) | (static_cast<std::uint64_t>((v >> 1) & 1) << q);
}

// 用门 g 共轭（g P g†）；inverse 为真时用 g†。
void Conj(P& p, const NetGate& g, bool inverse) {
    const int q = g.q0;
    int v = Local(p, q);
    switch (g.kind) {
        case NetGateKind::kH:
            if (v == 3) {
                p.neg = !p.neg;
            }
            Set(p, q, v == 1 ? 2 : v == 2 ? 1 : v);
            break;
        case NetGateKind::kS:
        case NetGateKind::kSdg: {
            const bool s = (g.kind == NetGateKind::kS) != inverse;
            if (v & 1) {
                if (s ? (v & 2) : !(v & 2)) {
                    p.neg = !p.neg;
                }
                Set(p, q, v ^ 2);
            }
            break;
        }
        case NetGateKind::kCX: {
            const int c = g.q0, t = g.q1;
            const int pc = Local(p, c), pt = Local(p, t);
            const bool xc = pc & 1, zc = pc & 2, xt = pt & 1, zt = pt & 2;
            if (xc && zt && (xt == zc)) {
                p.neg = !p.neg;
            }
            Set(p, t, pt ^ (xc ? 1 : 0));
            Set(p, c, pc ^ (zt ? 2 : 0));
            break;
        }
        case NetGateKind::kSwap: {
            const int a = g.q0, b = g.q1;
            const int pa = Local(p, a);
            Set(p, a, Local(p, b));
            Set(p, b, pa);
            break;
        }
        case NetGateKind::kX:
        case NetGateKind::kY:
        case NetGateKind::kZ: {
            const int s = g.kind == NetGateKind::kX ? 1 : g.kind == NetGateKind::kZ ? 2 : 3;
            const bool anti = v != 0 && ((((v & 1) & (s >> 1)) ^ ((v >> 1) & (s & 1))) != 0);
            if (anti) {
                p.neg = !p.neg;
            }
            break;
        }
        default:
            break;
    }
}

bool Anti(const P& a, const P& b) { return (quspect::native::common::Popcount64((a.x & b.z) ^ (a.z & b.x)) & 1) != 0; }

P Parse(const std::string& s) {  // 字符 i 作用在比特 i 上
    P p;
    for (int q = 0; q < static_cast<int>(s.size()); ++q) {
        Set(p, q, s[q] == 'X' ? 1 : s[q] == 'Z' ? 2 : s[q] == 'Y' ? 3 : 0);
    }
    return p;
}

PauliNetworkResult RunNet(int n, const std::vector<std::pair<int, int>>& edges, const std::vector<int>& active,
                       const std::vector<P>& rots, const PauliNetworkOptions& opt = {}) {
    std::vector<std::vector<int>> xs, zs;
    for (const P& p : rots) {
        std::vector<int> x, z;
        for (int q = 0; q < n; ++q) {
            if ((p.x >> q) & 1U) {
                x.push_back(q);
            }
            if ((p.z >> q) & 1U) {
                z.push_back(q);
            }
        }
        xs.push_back(x);
        zs.push_back(z);
    }
    return SynthesizePauliNetwork(n, edges, active, xs, zs, opt);
}

// 验证：每个旋转实现的是输入 Pauli；顺序合法；CX 在边上；收尾后框架恒等。
void Verify(int n, const std::vector<std::pair<int, int>>& edges, const std::vector<P>& rots,
            const PauliNetworkResult& res, bool finalized) {
    std::set<std::pair<int, int>> eset;
    for (auto [a, b] : edges) {
        eset.insert({std::min(a, b), std::max(a, b)});
    }
    std::vector<int> pos(rots.size(), -1);
    int count = 0;
    for (std::size_t i = 0; i < res.gates.size(); ++i) {
        const NetGate& g = res.gates[i];
        if (g.kind == NetGateKind::kCX || g.kind == NetGateKind::kSwap) {
            EXPECT_TRUE(eset.count({std::min(g.q0, g.q1), std::max(g.q0, g.q1)})) << g.q0 << "-" << g.q1;
            continue;
        }
        if (g.kind != NetGateKind::kRotation) {
            continue;
        }
        // 实现的 Pauli：F† σ_q F，F 为之前的全部 Clifford 门
        P s;
        Set(s, g.q0, g.axis == 'X' ? 1 : g.axis == 'Z' ? 2 : 3);
        s.neg = g.negate;
        for (std::size_t j = i; j-- > 0;) {
            Conj(s, res.gates[j], true);
        }
        const P& want = rots[g.rotation];
        EXPECT_EQ(s.x, want.x) << "rotation " << g.rotation;
        EXPECT_EQ(s.z, want.z) << "rotation " << g.rotation;
        EXPECT_EQ(s.neg, want.neg) << "rotation " << g.rotation;
        pos[g.rotation] = count++;
    }
    for (std::size_t i = 0; i < rots.size(); ++i) {
        if (rots[i].x == 0 && rots[i].z == 0) {
            continue;
        }
        ASSERT_GE(pos[i], 0) << "rotation " << i << " missing";
        for (std::size_t j = 0; j < i; ++j) {
            if (pos[j] >= 0 && Anti(rots[i], rots[j])) {
                EXPECT_LT(pos[j], pos[i]) << "anticommuting rotations " << j << ", " << i << " swapped";
            }
        }
    }
    if (finalized) {
        // 收尾后框架为置换 final_perm（默认恒等）：q 上的 X / Z 变成 final_perm[q] 上的 X / Z
        ASSERT_EQ(res.final_perm.size(), static_cast<std::size_t>(n));
        for (int q = 0; q < n; ++q) {
            for (int v : {1, 2}) {
                P s;
                Set(s, q, v);
                for (const NetGate& g : res.gates) {
                    Conj(s, g, false);
                }
                P want;
                Set(want, res.final_perm[q], v);
                EXPECT_EQ(s.x, want.x);
                EXPECT_EQ(s.z, want.z);
                EXPECT_FALSE(s.neg);
            }
        }
    }
}

std::vector<std::pair<int, int>> Line(int n) {
    std::vector<std::pair<int, int>> e;
    for (int q = 0; q + 1 < n; ++q) {
        e.push_back({q, q + 1});
    }
    return e;
}

std::vector<std::pair<int, int>> Full(int n) {
    std::vector<std::pair<int, int>> e;
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            e.push_back({a, b});
        }
    }
    return e;
}

std::vector<int> Range(int n) {
    std::vector<int> v(n);
    for (int q = 0; q < n; ++q) {
        v[q] = q;
    }
    return v;
}

// 4 比特 H2（JW）风格的项与 Hubbard 风格的长 Z 串
std::vector<P> Molecule() {
    std::vector<P> r;
    for (const char* s : {"ZIII", "IZII", "ZZII", "XXYY", "YYXX", "XYYX", "YXXY", "IIZZ", "ZIZI", "IZIZ", "IIIZ"}) {
        r.push_back(Parse(s));
    }
    return r;
}

std::vector<P> Hopping(int n) {
    std::vector<P> r;
    for (int a = 0; a + 2 < n; ++a) {
        for (char c : {'X', 'Y'}) {
            std::string s(n, 'I');
            s[a] = c;
            for (int k = a + 1; k < a + 2; ++k) {
                s[k] = 'Z';
            }
            s[a + 2] = c;
            r.push_back(Parse(s));
        }
        std::string zz(n, 'I');
        zz[a] = zz[a + 1] = 'Z';
        r.push_back(Parse(zz));
    }
    return r;
}

TEST(PauliNetwork, FullConnectivityExact) {
    const auto rots = Molecule();
    const auto res = RunNet(4, Full(4), Range(4), rots);
    Verify(4, Full(4), rots, res, true);
    EXPECT_GT(res.num_cx, 0);
}

TEST(PauliNetwork, LineWithHopsExact) {
    // 支集在链上不相邻：需要跳跃
    std::vector<P> rots = {Parse("XIIIZ"), Parse("ZIYII"), Parse("IXIIY"), Parse("YZZZX"), Parse("IIZIZ")};
    const auto res = RunNet(5, Line(5), Range(5), rots);
    Verify(5, Line(5), rots, res, true);
}

TEST(PauliNetwork, HoppingChainExactAndDeterministic) {
    const auto rots = Hopping(7);
    std::vector<P> seq;
    for (int rep = 0; rep < 3; ++rep) {
        seq.insert(seq.end(), rots.begin(), rots.end());
        seq.insert(seq.end(), rots.rbegin(), rots.rend());
    }
    const auto a = RunNet(7, Line(7), Range(7), seq);
    const auto b = RunNet(7, Line(7), Range(7), seq);
    Verify(7, Line(7), seq, a, true);
    ASSERT_EQ(a.gates.size(), b.gates.size());
    EXPECT_EQ(a.num_cx, b.num_cx);
    EXPECT_EQ(a.order, b.order);
}

TEST(PauliNetwork, WorkCounterDeterministicAndGrowing) {
    // 计分工作量只计数、不影响结果：同一输入相同；序列更长工作量更大
    const auto rots = Hopping(7);
    std::vector<P> seq;
    for (int rep = 0; rep < 3; ++rep) {
        seq.insert(seq.end(), rots.begin(), rots.end());
    }
    const auto a = RunNet(7, Line(7), Range(7), seq);
    const auto b = RunNet(7, Line(7), Range(7), seq);
    EXPECT_GT(a.work, 0);
    EXPECT_EQ(a.work, b.work);
    const std::vector<P> half(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(seq.size() / 2));
    const auto c = RunNet(7, Line(7), Range(7), half);
    EXPECT_LT(c.work, a.work);
}

TEST(PauliNetwork, AncillaInActiveSetAndUnfinalized) {
    // 比特 1 不在任何旋转上，但在活动集中（连接 0 与 2）
    std::vector<P> rots = {Parse("ZIZ"), Parse("XIX"), Parse("YIY"), Parse("ZIX")};
    PauliNetworkOptions opt;
    const auto res = RunNet(3, Line(3), Range(3), rots, opt);
    Verify(3, Line(3), rots, res, true);
    opt.finalize = false;
    const auto open = RunNet(3, Line(3), Range(3), rots, opt);
    Verify(3, Line(3), rots, open, false);
    EXPECT_EQ(open.finalize_cx, 0);
    EXPECT_LE(open.num_cx, res.num_cx);
}

TEST(PauliNetwork, IdentityRotationsSkipped) {
    std::vector<P> rots = {Parse("II"), Parse("ZZ"), Parse("II")};
    const auto res = RunNet(2, Line(2), Range(2), rots);
    Verify(2, Line(2), rots, res, true);
    EXPECT_EQ(res.order.size(), 1U);
}

TEST(PauliNetwork, RejectsBadInput) {
    std::vector<P> rots = {Parse("ZIZ")};
    // 活动集不连通（缺比特 1）
    EXPECT_THROW(RunNet(3, Line(3), {0, 2}, rots), std::invalid_argument);
    // 旋转作用在活动集之外
    EXPECT_THROW(RunNet(3, Line(3), {0, 1}, rots), std::invalid_argument);
    EXPECT_THROW(SynthesizePauliNetwork(0, {}, {}, {}, {}, {}), std::invalid_argument);
}

// 链上的 Heisenberg（XX + YY + ZZ，加纵场）两个 Strang 步：块模式下同一条边上的项原位成块
std::vector<P> HeisenbergSteps(int n, int steps) {
    std::vector<P> one;
    for (int a = 0; a + 1 < n; ++a) {
        for (char c : {'X', 'Y', 'Z'}) {
            std::string s(n, 'I');
            s[a] = s[a + 1] = c;
            one.push_back(Parse(s));
        }
    }
    for (int a = 0; a < n; ++a) {
        std::string s(n, 'I');
        s[a] = 'Z';
        one.push_back(Parse(s));
    }
    std::vector<P> seq;
    for (int k = 0; k < steps; ++k) {
        seq.insert(seq.end(), one.begin(), one.end());
        seq.insert(seq.end(), one.rbegin(), one.rend());
    }
    return seq;
}

TEST(PauliNetwork, BlockModeExact) {
    const auto rots = HeisenbergSteps(6, 2);
    PauliNetworkOptions opt;
    opt.block_mode = 1;
    const auto res = RunNet(6, Line(6), Range(6), rots, opt);
    Verify(6, Line(6), rots, res, true);
    EXPECT_GT(res.blocks, 0);
    // 偏向块（bias 很小）时也必须精确，且出现带镜像 SWAP 的块（支集不相邻的项）
    std::vector<P> hop = HeisenbergSteps(5, 1);
    hop.push_back(Parse("XIIIX"));
    hop.push_back(Parse("ZIZII"));
    opt.block_bias = 200;
    const auto res2 = RunNet(5, Line(5), Range(5), hop, opt);
    Verify(5, Line(5), hop, res2, true);
    EXPECT_GT(res2.blocks, 0);
}

TEST(PauliNetwork, FinalizePermutationExact) {
    const auto rots = Hopping(7);
    PauliNetworkOptions opt;
    opt.finalize_permutation = true;
    for (int mode : {0, 1}) {
        opt.block_mode = mode;
        const auto res = RunNet(7, Line(7), Range(7), rots, opt);
        Verify(7, Line(7), rots, res, true);
        std::vector<int> sorted = res.final_perm;
        std::sort(sorted.begin(), sorted.end());
        EXPECT_EQ(sorted, Range(7));
    }
    const auto mol = Molecule();
    const auto full = RunNet(4, Full(4), Range(4), mol, opt);
    Verify(4, Full(4), mol, full, true);
}

TEST(PauliNetwork, CxLimitAborts) {
    const auto rots = Hopping(7);
    PauliNetworkOptions opt;
    const auto ref = RunNet(7, Line(7), Range(7), rots, opt);
    EXPECT_FALSE(ref.aborted);
    opt.cx_limit = ref.num_cx;  // 恰好等于：不中止，结果相同
    const auto same = RunNet(7, Line(7), Range(7), rots, opt);
    EXPECT_FALSE(same.aborted);
    EXPECT_EQ(same.num_cx, ref.num_cx);
    opt.cx_limit = ref.num_cx / 2;
    EXPECT_TRUE(RunNet(7, Line(7), Range(7), rots, opt).aborted);
}

TEST(MergeCommuting, MergesAcrossCommutingOnly) {
    // ZZ · XI · ZZ：XI 与 ZZ 反对易，不能合并；ZZ · ZI · ZZ：可以
    std::vector<std::vector<int>> xs = {{}, {0}, {}, {}, {}, {}};
    std::vector<std::vector<int>> zs = {{0, 1}, {}, {0, 1}, {0}, {0, 1}, {1}};
    const auto slot = MergeCommutingRotations(2, xs, zs, 64);
    EXPECT_EQ(slot, (std::vector<int>{0, 1, 2, 3, 2, 4}));
    // span 限制：只往回看 1 个
    const auto slot1 = MergeCommutingRotations(2, xs, zs, 1);
    EXPECT_EQ(slot1, (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

}  // namespace
