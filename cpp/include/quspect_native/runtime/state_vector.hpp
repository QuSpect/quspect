#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace quspect::native::runtime {

/// 量子态向量 — little-endian basis 编码
///
/// qubit 0 对应 basis index 的 bit 0，qubit 1 对应 bit 1，以此类推。
/// 对于 n 个 qubit，状态向量维度为 2^n。
template <typename FP>
class StateVector {
 public:
    using Complex = std::complex<FP>;

    /// 构造 n 比特状态向量，初始化为 |0...0>
    explicit StateVector(uint32_t num_qubits);

    /// 从指定计算基构造
    explicit StateVector(uint32_t num_qubits, uint64_t basis);

    /// 重置为 |0...0>
    void SetZeroState();

    /// 设置到指定计算基
    void SetComputationalBasis(uint64_t basis);

    /// 归一化状态向量
    void Normalize();

    /// 返回平方模长
    FP SquaredNorm() const;

    /// 获取振幅数据指针
    Complex* data();
    const Complex* data() const;

    /// qubit 数
    uint32_t num_qubits() const;

    /// 状态向量维度 (2^num_qubits)
    uint64_t dim() const;

 private:
    uint32_t num_qubits_;
    std::vector<Complex> amplitudes_;
};

// 显式实例化声明
extern template class StateVector<float>;
extern template class StateVector<double>;

}  // namespace quspect::native::runtime
