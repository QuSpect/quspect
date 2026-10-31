#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace quspect::native::runtime {
struct QSALProgram;
struct ObservableTerm;
}  // namespace quspect::native::runtime

namespace quspect::native::ir {

/// 编程式 QSALProgram 构建器
///
/// 提供便捷的方法链式构建 QSALProgram，避免手动填充字段。
/// 适用于 C++ 测试代码和 Python 桥接层。
///
/// Example:
///   auto prog = ProgramBuilder(2, 1)
///       .AddGate("h", {0})
///       .AddGate("cx", {0, 1})
///       .AddMeasure({0, 1}, {0, 1})
///       .AddObservable("ZZ", {{0, 'Z'}, {1, 'Z'}}, 1.0)
///       .Build();
class ProgramBuilder {
 public:
    ProgramBuilder();
    explicit ProgramBuilder(uint32_t num_qubits, uint32_t num_clbits = 0);

    /// 设置 qubit/clbit 数量
    ProgramBuilder& SetNumQubits(uint32_t n);
    ProgramBuilder& SetNumClbits(uint32_t n);

    /// 添加 named gate（标准门或参数化门）
    ProgramBuilder& AddGate(const std::string& name, const std::vector<uint32_t>& qubits,
                            const std::vector<double>& params = {});

    /// 添加 barrier
    ProgramBuilder& AddBarrier(const std::vector<uint32_t>& qubits);

    /// 添加测量操作
    ProgramBuilder& AddMeasure(const std::vector<uint32_t>& qubits, const std::vector<uint32_t>& clbits);

    /// 添加可观测量
    ProgramBuilder& AddObservable(const std::string& label, const std::vector<std::pair<uint32_t, char>>& paulis,
                                  std::complex<double> coefficient = std::complex<double>(1.0, 0));

    /// 构建并返回 QSALProgram
    std::unique_ptr<runtime::QSALProgram> Build();

 private:
    std::unique_ptr<runtime::QSALProgram> prog_;
};

}  // namespace quspect::native::ir
