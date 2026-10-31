#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace quspect::native::kernels {
template <typename FP>
struct KernelDispatchTable;
}  // namespace quspect::native::kernels

namespace quspect::native::runtime {

template <typename FP>
class StateVector;
class ClassicalRegisterFile;
struct QSALProgram;

/// RuntimeEngine — 线路执行引擎
///
/// 职责：
/// - 遍历 QSALProgram 中的每条指令
/// - 将 named gate / dense gate / diagonal gate 分发到对应的 kernel
/// - 管理 StateVector 和 ClassicalRegisterFile 的生命周期
/// - Barrier 直接跳过
/// - Phase 4: 支持 mid-circuit measurement（概率计算 + 坍缩 + 经典寄存器写入）
///
/// 模板参数 FP = float 或 double
template <typename FP>
class RuntimeEngine {
 public:
    using Complex = std::complex<FP>;

    /// 构造引擎
    ///
    /// Args:
    ///   program: 待执行的线路程序
    ///   table: kernel dispatch table (scalar 或 avx2)
    ///   seed: 随机数种子，用于测量采样
    explicit RuntimeEngine(const QSALProgram& program, const ::quspect::native::kernels::KernelDispatchTable<FP>& table,
                           uint64_t seed = 0);

    ~RuntimeEngine();

    /// 执行整条线路（包括测量操作）
    ///
    /// 若 execute_measurements 为 false，则跳过所有测量门
    /// （用于终端测量快速路径，在此模式下仅执行量子门）
    void Run(bool execute_measurements = true);

    /// 重置状态向量为 |0...0⟩ 并清零经典寄存器
    ///
    /// 用于 per-shot 中途测量循环中每次 shot 前重置引擎状态。
    void Reset();

    /// 获取末态 statevector
    StateVector<FP>& GetStateVector();
    const StateVector<FP>& GetStateVector() const;

    /// 获取经典寄存器
    ClassicalRegisterFile& GetClassicalRegister();
    const ClassicalRegisterFile& GetClassicalRegister() const;

    /// Sample one computational-basis state from the current state vector.
    ///
    /// The engine-owned RNG is used, so a terminal measurement consumes the
    /// same random stream as all preceding noise channels in this trajectory.
    uint64_t SampleBasis();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 显式实例化声明
extern template class RuntimeEngine<float>;
extern template class RuntimeEngine<double>;

}  // namespace quspect::native::runtime
