#pragma once

#include <memory>
#include <string>

namespace quspect::native::runtime {
struct QSALProgram;
}  // namespace quspect::native::runtime

namespace quspect::native::ir {

/// QSALDecoder — 将 QSAL JSON 字符串解码为本地程序对象
///
/// 当前 Phase 1 仅定义接口和校验骨架。
/// 完整解码逻辑在 Phase 5 实现。
class QSALDecoder {
 public:
    QSALDecoder();
    ~QSALDecoder();

    /// 从 QSAL JSON 字符串解码
    /// 返回 nullptr 表示解码失败（可通过 LastError() 获取错误信息）
    std::unique_ptr<runtime::QSALProgram> Decode(const std::string& qsal_json);

    /// 最后错误信息
    const std::string& LastError() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace quspect::native::ir
