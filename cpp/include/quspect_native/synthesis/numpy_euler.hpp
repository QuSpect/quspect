#pragma once

#include <optional>
#include <vector>

#include "quspect_native/synthesis/kak_synthesis.hpp"
#include "quspect_native/synthesis/numpy_compat.hpp"

namespace quspect::native::synthesis {

// 块合并（quspect.transpiler._consolidate）中仍以 Python / numpy 为准的部分：单比特门串的欧拉
// 重合成（``_verified_euler_ops_uncached``）与块的 4×4 酉矩阵乘积。它们按 numpy_compat.hpp 的
// numpy 语义运算（numpy 的 BLAS、探测到的逐元素运算顺序），与 Python 参考实现逐位一致；比较落在
// 阈值附近时抛出 Fallback。两比特 KAK 合成本身见 kak_synthesis.hpp（确定性数值层）。
class NumpyEuler {
 public:
    NumpyEuler(const LinalgFunctions& fns, const ArithModes& modes, const KakConstants& constants);

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

    // numpy 语义的欧拉合成 ``euler_ops(m, qubit, basis, atol)``（公开以便测试）。
    void EulerOps(const M2& m, int qubit, int basis, double atol, std::vector<Op>* out) const;
    M4 Kron2(const M2& left, const M2& right) const;

    const NumpyLinalg& linalg() const {
        return np_;
    }

 private:
    M2 Mat2(const M2& a, const M2& b) const;
    M4 Mat4(const M4& a, const M4& b) const;
    M2 Rz(double theta) const;
    M2 OpsLayerMatrix(const std::vector<Op>& ops) const;

    NumpyLinalg np_;
    M2 i2_, h_, x_, sx_;
};

}  // namespace quspect::native::synthesis
