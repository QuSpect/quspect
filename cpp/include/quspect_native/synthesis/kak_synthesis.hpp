#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "quspect_native/synthesis/numpy_compat.hpp"

namespace quspect::native::synthesis {

// 两比特 KAK 合成（quspect.synthesis.two_qubit 的 _kak4 / _factor_local / _try_synthesize /
// _synthesize_two_qubit_ops / _synthesize_swap_face_block）的 C++ 实现。
//
// 逐个函数复刻 Python 参考实现：每一步 numpy / CPython 运算都换成 numpy_compat.hpp 中语义相同的
// 运算（同一个 BLAS / LAPACK、同样的调用方式、探测得到的逐元素运算顺序），因此输出的门序列与参数
// 和 Python 逐位一致。只有两类量不能保证逐位一致，它们只用于比较（不进入输出）：
//
//   * numpy 的复数 ``np.abs``（各平台的 SIMD 实现不同）——用于 ``_distance_up_to_phase`` 与
//     ``_real_orthogonal_diagonalize`` 的误差；
//   * 近似合成的预测不保真度。
//
// 比较落在阈值附近（相对误差 1e-9 以内）时抛出 ``Fallback``，由调用方改走 Python 参考实现；
// LAPACK 失败、输入不是有限值等 Python 会报错的情形同样抛出 ``Fallback``。

// 需要退回 Python 参考实现。
class Fallback : public std::runtime_error {
 public:
    explicit Fallback(const char* what) : std::runtime_error(what) {}
};

// _real_orthogonal_diagonalize 需要随机组合系数，但尚未提供（见 KakConstants::normals）。
class NeedNormals : public Fallback {
 public:
    NeedNormals() : Fallback("need normals") {}
};

// 合成出的一个门（局部比特 0 / 1）。
enum class OpKind : std::uint8_t { kEntangler, kRz, kH, kX, kSX };
struct Op {
    OpKind kind;
    std::uint8_t q0;
    std::uint8_t q1;  // 只对纠缠门有效
    double param;     // 只对 rz 有效
};

// Python 端的常量矩阵（直接取自 two_qubit.py 的模块常量，保证逐位相同）。
struct KakConstants {
    M4 magic{};       // _MAGIC
    M4 magic_conj{};  // _MAGIC.conj()（_MAGIC_DAG 是它的转置视图）
    M2 i2{}, x{}, y{}, z{}, h{}, sx{};
    M4 cz4{};
    M4 swap4{};
    std::array<M4, 3> pauli_pairs{};  // _PAULI_PAIRS
    M4 swap_ab{}, swap_bc{};          // _SWAP_AB, _SWAP_BC
    M4 flip_ab{}, flip_ac{}, flip_bc{};
    std::array<double, 16> phase_map{};    // _PHASE_MAP（4×4，C 连续）
    std::array<double, 12> can_phases{};   // _CAN_PHASES（4×3）
    // default_rng(1234567).normal() 的前 2·98 个值（只有前两次尝试都失败时才用到；为空时抛出
    // NeedNormals，由调用方补上后重试）
    std::vector<double> normals;
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
    KakSynthesizer(const LinalgFunctions& fns, const ArithModes& modes, KakConstants constants);

    int AddEntangler(EntanglerSpec spec);
    // 补上 KakConstants::normals（至少 196 个）。
    void SetNormals(std::vector<double> normals);
    bool HasNormals() const {
        return c_.normals.size() >= 196;
    }

    // ``_check_unitary_4`` 的判断：true = 是酉矩阵；不是酉矩阵或落在容差边界上时抛出 Fallback。
    void CheckUnitary(const M4& u) const;

    // ``_kak4(u)``。
    Kak Kak4(const M4& u) const;
    // ``_factor_local(k)``。
    std::array<M2, 2> FactorLocal(const M4& k) const;
    // ``_count_from_coords(v, atol)``。
    static int CountFromCoords(const std::array<double, 3>& v, double atol);

    // ``_synthesize_two_qubit_ops(u, atol, verify_atol, basis)``；uses_flag：结果是否取决于
    // ``_SWAP_FACE_TEMPLATE``（三个纠缠门且在 SWAP 面上）。
    std::vector<Op> SynthesizeExact(const M4& u, const SynthOptions& opt, bool* uses_flag) const;
    // 同上，但复用已算好的 KAK。
    std::vector<Op> SynthesizeExact(const M4& u, const Kak& kak, const SynthOptions& opt, bool* uses_flag) const;

    // ``_synthesize_swap_face_block(u, basis, verify_atol)``（nullopt：Python 返回 None）。
    std::optional<std::vector<Op>> SynthesizeSwapFace(const M4& u, const SynthOptions& opt) const;

    // ``_approximate_two_qubit_ops`` 是否一定返回 None（每个更少的纠缠门数的预测不保真度都明显
    // 超出预算）。不能确定时抛出 Fallback。
    bool ApproximationImpossible(const Kak& kak, double max_infidelity, double atol, int max_entanglers) const;

    // quspect.transpiler._consolidate._verified_euler_ops_uncached：先按 atol 去掉零旋转，误差超出
    // 1e-9 时改用精确角度；都不满足时返回 nullopt（Python 返回 None）。
    std::optional<std::vector<Op>> VerifiedEuler(const M2& u, double atol, int basis) const;

    // 块的 4×4 酉矩阵：u = I；对每个成员 u = full @ u（numpy 的乘法）。slot 0：4×4 原样；1：
    // np.kron(m, I2)；2：np.kron(I2, m)；3：交换两个比特的置换副本（m[np.ix_(REV, REV)]）；
    // 4：swap4 @ m @ swap4。
    struct BlockMember {
        const Cx* matrix;
        int slot;
    };
    M4 BlockUnitary(const std::vector<BlockMember>& members, const M4& swap4) const;

    // 单比特欧拉合成 ``euler_ops(m, qubit, basis, atol)``（公开以便测试）。
    void EulerOps(const M2& m, int qubit, int basis, double atol, std::vector<Op>* out) const;
    // ``_kron2``、``_axis_rot``、``_axis_angle``（公开以便与 Python 对拍）。
    M4 Kron2(const M2& left, const M2& right) const;
    M2 AxisRot(const M2& pauli, double phi) const;
    double AxisAngle(const M2& w, const M2& pauli) const;

    const NumpyLinalg& linalg() const {
        return np_;
    }
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

    M2 Mat2(const M2& a, const M2& b) const;
    M2 Mat2(const CView& a, const CView& b) const;
    M4 Mat4(const M4& a, const M4& b) const;
    M2 Rx(double theta) const;
    M2 Rz(double theta) const;
    M4 LayersUnitary(const std::vector<Layer>& layers) const;
    std::vector<Layer> Template(int k, const std::array<double, 3>& v) const;
    std::vector<Layer> SwapTemplate(double c) const;

    void RawKak(const M4& u, Kak* out) const;
    void Canonicalize(Kak* kak) const;
    std::array<double, 16> RealOrthogonalDiagonalize(const M4& m2) const;

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
    M2 OpsLayerMatrix(const std::vector<Op>& ops, std::size_t begin, std::size_t end) const;
    double DistanceUpToPhase(const M4& u, const M4& v) const;
    // ``err <= verify_atol``（err 的最后几位可能与 numpy 不同：落在边界附近时抛出 Fallback）。
    static bool Verified(double err, double verify_atol);

    NumpyLinalg np_;
    KakConstants c_;
    std::vector<EntanglerSpec> entanglers_;
    std::vector<M4> entangler_reversed_;  // _SWAP4 @ E @ _SWAP4
};

}  // namespace quspect::native::synthesis
