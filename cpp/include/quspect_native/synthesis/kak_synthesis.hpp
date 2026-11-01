#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "quspect_native/detmath/linalg.hpp"

namespace quspect::native::synthesis {

// 两比特 KAK 合成（quspect.synthesis.two_qubit 的 _kak4 / _factor_local / _try_synthesize /
// _synthesize_two_qubit_ops / _synthesize_swap_face_block 的算法）的确定性实现。
//
// 全部数值运算走确定性数值层（quspect_native/detmath）：固定顺序的小矩阵乘法、部分主元 LU 行列式、
// Jacobi 同时对角化与单边 Jacobi 奇异值分解、CORE-MATH 的正确舍入 sin / cos / atan2 / hypot；
// 不调用 BLAS / LAPACK / 平台 libm。同一输入在所有平台上得到逐位相同的门序列与参数，这是编译
// 结果的标准答案；Python 参考实现只在原生模块缺失时使用，与本实现不要求逐位一致（算法相同，
// 数值细节不同）。
//
// 与 Python 参考实现的算法差异：
//
//   * _real_orthogonal_diagonalize：Python 对 Re M、Im M 的随机实组合做 eigh 并验证、失败重试；
//     这里直接对 (Re M, Im M) 做 Jacobi 同时对角化（简并谱同样稳定，不需要随机数）。同时对角化的
//     残差超过 1e-13 时（输入不是酉矩阵等）才退回固定系数的组合 + Jacobi 特征分解，取残差最小者；
//   * _factor_local：秩 1 分解用单边 Jacobi 奇异值分解的最大奇异三元组。

using Cx = detmath::Cx;
using M2 = std::array<Cx, 4>;   // 2×2，行主序
using M4 = std::array<Cx, 16>;  // 4×4，行主序

// 合成出的一个门（局部比特 0 / 1）。
enum class OpKind : std::uint8_t { kEntangler, kRz, kH, kX, kSX };
struct Op {
    OpKind kind;
    std::uint8_t q0;
    std::uint8_t q1;  // 只对纠缠门有效
    double param;     // 只对 rz 有效
};

// 输入不是酉矩阵（与 Python 的 ``ValueError("Matrix is not unitary.")`` 对应）。
class NotUnitary : public std::invalid_argument {
 public:
    NotUnitary() : std::invalid_argument("Matrix is not unitary.") {}
};

// 没有模板能在 verify_atol 内重现输入（与 Python 的 RuntimeError 对应；对有效输入不会发生）。
class SynthesisFailed : public std::runtime_error {
 public:
    explicit SynthesisFailed(const std::string& what) : std::runtime_error(what) {}
};

// 常量矩阵（与 two_qubit.py 的模块常量数值相同；全部是精确值或 1/sqrt(2) 的乘积，各平台相同）。
struct KakConstants {
    M4 magic{};       // _MAGIC
    M4 magic_dag{};   // _MAGIC_DAG
    M2 i2{}, x{}, y{}, z{}, h{}, sx{};
    M4 cz4{};
    M4 swap4{};
    std::array<M4, 3> pauli_pairs{};  // _PAULI_PAIRS
    M4 swap_ab{}, swap_bc{};          // _SWAP_AB, _SWAP_BC
    M4 flip_ab{}, flip_ac{}, flip_bc{};
    std::array<double, 16> phase_map{};   // _PHASE_MAP（4×4）
    std::array<double, 12> can_phases{};  // _CAN_PHASES（4×3）

    static KakConstants Make();
};

// 本地纠缠门（quspect.synthesis.two_qubit.Entangler）。
struct EntanglerSpec {
    std::string name;
    M4 matrix{};
    std::array<M2, 2> before{};
    std::array<M2, 2> after{};
    bool symmetric = true;
};

// 合成选项（对应 TwoQubitBasis 与模块级开关）。
struct SynthOptions {
    int entangler = 0;           // KakSynthesizer::AddEntangler 返回的编号
    int euler = 0;               // 0 = "zxz"，1 = "zsx"
    double atol = 1e-9;
    double verify_atol = 1e-9;
    bool swap_face_template = false;  // _SWAP_FACE_TEMPLATE
    bool mirrored_stabilizers = true; // _MIRRORED_STABILIZERS
};

// _kak4 的结果：u = e^{i·phase} · k1 · Can(v) · k2。
struct Kak {
    double phase = 0.0;
    M4 k1{};
    std::array<double, 3> v{};
    M4 k2{};
};

class KakSynthesizer {
 public:
    KakSynthesizer();

    int AddEntangler(EntanglerSpec spec);

    // ``_check_unitary_4`` 的判断（np.allclose(u^† u, I, atol=1e-8)）：不是酉矩阵时抛出 NotUnitary。
    void CheckUnitary(const M4& u) const;

    // ``_kak4(u)``。简并的 Weyl 点上 k1、k2 不唯一（取决于特征向量的基）；variant 选择基的取法：
    // 0 = (Re M, Im M) 的同时对角化；1 = 依次尝试 Re M、Im M 与固定系数组合的特征分解（特征值
    // 升序，与 Python 参考实现的尝试顺序相同），都不满足精度时同 0。
    Kak Kak4(const M4& u, int variant = 0) const;
    // ``_factor_local(k)``。
    std::array<M2, 2> FactorLocal(const M4& k) const;
    // ``_count_from_coords(v, atol)``。
    static int CountFromCoords(const std::array<double, 3>& v, double atol);

    // ``_synthesize_two_qubit_ops(u, atol, verify_atol, basis)``；uses_flag：结果是否取决于
    // ``_SWAP_FACE_TEMPLATE``（三个纠缠门且在 SWAP 面上）。失败时抛出 SynthesisFailed。
    //
    // 与 Python 参考实现的区别：分别用 Kak4 的两种基（variant 0、1）合成，取（纠缠门数，门数）
    // 较小者（相同时取 variant 0）。简并点上两种基给出的单比特层长度不同，取较短者使结果不依赖
    // 特征向量基的偶然选择。
    std::vector<Op> SynthesizeExact(const M4& u, const SynthOptions& opt, bool* uses_flag) const;
    // 同上，但复用已算好的 KAK。
    std::vector<Op> SynthesizeExact(const M4& u, const Kak& kak, const SynthOptions& opt, bool* uses_flag) const;

    // ``_synthesize_swap_face_block(u, basis, verify_atol)``（nullopt：Python 返回 None）。
    std::optional<std::vector<Op>> SynthesizeSwapFace(const M4& u, const SynthOptions& opt) const;

    // ``_approximate_two_qubit_ops`` 是否一定返回 None（每个更少的纠缠门数的预测不保真度都超出预算）。
    bool ApproximationImpossible(const Kak& kak, double max_infidelity, double atol, int max_entanglers) const;

    // 单比特欧拉合成 ``euler_ops(m, qubit, basis, atol)``（公开以便测试）。
    void EulerOps(const M2& m, int qubit, int basis, double atol, std::vector<Op>* out) const;
    // ``_kron2``、``_axis_rot``、``_axis_angle``、门序列的 2×2 矩阵（公开以便测试）。
    static M4 Kron2(const M2& left, const M2& right);
    M2 AxisRot(const M2& pauli, double phi) const;
    static double AxisAngle(const M2& w, const M2& pauli);
    M2 OpsLayerMatrix(const std::vector<Op>& ops, std::size_t begin, std::size_t end) const;

    const KakConstants& constants() const {
        return c_;
    }
    const EntanglerSpec& entangler(int index) const {
        return entanglers_.at(static_cast<std::size_t>(index));
    }

 private:
    struct Layer {
        M2 left;
        M2 right;
    };
    using Pair = std::array<M2, 2>;
    struct TryResult {
        std::vector<Op> ops;
        double err;
    };
    struct UFactors;  // _try_synthesize 中只取决于 u 的部分（同一次合成中复用）

    static M2 Rx(double theta);
    static M2 Rz(double theta);
    M4 LayersUnitary(const std::vector<Layer>& layers) const;
    std::vector<Layer> Template(int k, const std::array<double, 3>& v) const;
    std::vector<Layer> SwapTemplate(double c) const;

    void RawKak(const M4& u, int variant, Kak* out) const;
    void Canonicalize(Kak* kak) const;
    std::array<double, 16> RealOrthogonalDiagonalize(const M4& m2, int variant) const;
    std::vector<Op> SynthesizeExactOnce(const M4& u, const Kak& kak, const SynthOptions& opt, bool* uses_flag,
                                        double* best_err) const;

    std::vector<Pair> StabilizerCandidates(int k, const std::array<double, 3>& v, const Pair& k1u, const Pair& k1v,
                                           const Pair& k2v, const Pair& k2u, bool fit_cosets,
                                           bool mirrored_stabilizers) const;
    std::vector<Pair> StabilizerFamily(double a, double b, double c, const std::vector<std::pair<int, M2>>& targets,
                                       bool fit_cosets) const;
    std::vector<std::pair<Pair, Pair>> SwapFaceGauges(const Pair& k1u, const Pair& k1v, const Pair& k2v,
                                                      const Pair& k2u) const;

    TryResult TrySynthesize(const M4& u, const Kak& kak_u, const UFactors& uf, int k, double atol,
                            const std::array<double, 3>* template_coords, const SynthOptions& opt,
                            bool swap_face) const;
    TryResult EmitFixedLayers(const M4& u, const std::vector<Layer>& factored, int k, double atol,
                              const SynthOptions& opt) const;
    TryResult EmitLayers(const M4& u, const std::vector<Layer>& layers, const std::vector<const M4*>& gates,
                         const std::vector<bool>& reversed, double atol, const SynthOptions& opt) const;
    static double DistanceUpToPhase(const M4& u, const M4& v);

    KakConstants c_;
    std::vector<EntanglerSpec> entanglers_;
    std::vector<M4> entangler_reversed_;  // _SWAP4 @ E @ _SWAP4
};

}  // namespace quspect::native::synthesis
