#pragma once

#include <string>
#include <vector>

namespace quspect::native::runtime {
struct QSALProgram;
}  // namespace quspect::native::runtime

namespace quspect::native::ir {

/// QSALProgram 输入校验器
///
/// 在设计文档第 6.4 节定义的校验规则集合。
/// 所有方法返回错误信息列表；空列表表示校验通过。
class Validator {
 public:
    /// 对 QSALProgram 执行全部校验
    /// 返回错误信息列表，空列表表示校验通过
    static std::vector<std::string> Validate(const runtime::QSALProgram& prog);

    /// qubit/clbit 索引越界检查
    static std::vector<std::string> CheckIndexBounds(const runtime::QSALProgram& prog);

    /// 门名称已知性检查
    static std::vector<std::string> CheckGateNames(const runtime::QSALProgram& prog);

    /// 参数数量匹配检查
    static std::vector<std::string> CheckParameterCount(const runtime::QSALProgram& prog);

    /// 测量操作 qubit/clbit 一致性检查
    static std::vector<std::string> CheckMeasureConsistency(const runtime::QSALProgram& prog);

    /// Noise-channel shape, target, and trace-preserving checks.
    static std::vector<std::string> CheckChannels(const runtime::QSALProgram& prog);
};

}  // namespace quspect::native::ir
