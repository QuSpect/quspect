#pragma once

#include <complex>
#include <cstdint>
#include <functional>

namespace quspect::native::kernels {

// ============================================================
// 函数签名类型别名
// ============================================================

/// 单比特门 kernel: 对 state[0..dim) 上的 qubit 施加 2x2 unitary
template <typename FP>
using SingleQubitKernel = void (*)(std::complex<FP>* state, uint64_t dim, uint32_t qubit,
                                   const std::complex<FP>* matrix  // 2x2 row-major
);

/// 受控单比特门 kernel
template <typename FP>
using ControlledSingleQubitKernel = void (*)(std::complex<FP>* state, uint64_t dim, uint32_t target,
                                             uint64_t control_mask, uint64_t control_value,
                                             const std::complex<FP>* matrix);

/// SWAP kernel
template <typename FP>
using SwapKernel = void (*)(std::complex<FP>* state, uint64_t dim, uint32_t qubit0, uint32_t qubit1);

/// 通用 k 比特稠密门 kernel
template <typename FP>
using DenseKernel = void (*)(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits, uint32_t num_targets,
                             const std::complex<FP>* matrix  // 2^k x 2^k row-major
);

/// 对角门 kernel
template <typename FP>
using DiagonalKernel = void (*)(std::complex<FP>* state, uint64_t dim, const uint32_t* target_qubits,
                                uint32_t num_targets,
                                const std::complex<FP>* diag  // 2^k entries
);

/// 单比特测量概率 kernel: 返回 qubit=0 的概率
template <typename FP>
using ProbabilityKernel = FP (*)(const std::complex<FP>* state, uint64_t dim, uint32_t qubit);

/// Pauli 期望值 kernel
template <typename FP>
using ExpectationKernel = std::complex<FP> (*)(const std::complex<FP>* state, uint64_t dim, uint64_t bit_flip_mask,
                                               uint64_t phase_flip_mask, uint32_t pivot_qubit, int y_count);

/// 终端采样 kernel: 从概率分布中生成一个样本
template <typename FP>
using SamplingKernel = uint64_t (*)(const std::complex<FP>* state, uint64_t dim, double random01);

/// 单比特测量 collapse kernel
template <typename FP>
using CollapseKernel = void (*)(std::complex<FP>* state, uint64_t dim, uint32_t qubit,
                                int outcome  // 0 或 1
);

// ============================================================
// KernelDispatchTable — 运行时 dispatch 的核心结构
// ============================================================

template <typename FP>
struct KernelDispatchTable {
    // ---- 单比特 named gate 快路径 ----
    SingleQubitKernel<FP> x;
    SingleQubitKernel<FP> y;
    SingleQubitKernel<FP> z;
    SingleQubitKernel<FP> h;
    SingleQubitKernel<FP> s;
    SingleQubitKernel<FP> sdg;
    SingleQubitKernel<FP> t;
    SingleQubitKernel<FP> tdg;
    SingleQubitKernel<FP> rx;  // RX(theta) — 参数已在调用前数值化
    SingleQubitKernel<FP> ry;
    SingleQubitKernel<FP> rz;

    // ---- 双比特 named gate 快路径 ----
    ControlledSingleQubitKernel<FP> cnot;
    ControlledSingleQubitKernel<FP> cz;
    SwapKernel<FP> swap;

    // ---- 通用路径 ----
    DenseKernel<FP> dense_k;
    DiagonalKernel<FP> diagonal_k;

    // ---- 测量与统计 ----
    ProbabilityKernel<FP> prob0;
    CollapseKernel<FP> collapse;

    // ---- 期望值与采样 ----
    ExpectationKernel<FP> pauli_expect;
    SamplingKernel<FP> sampling;

    // ---- 元信息 ----
    const char* backend_name;
};

// ============================================================
// 全局 dispatch 获取函数
// ============================================================

/// 运行时自动选择最优 kernel table (scalar 或 avx2)
const KernelDispatchTable<float>& GetDispatchF32();

/// 运行时自动选择最优 kernel table (scalar 或 avx2)
const KernelDispatchTable<double>& GetDispatchF64();

/// CPU 特性探测
bool CpuSupportsAVX2();

}  // namespace quspect::native::kernels
