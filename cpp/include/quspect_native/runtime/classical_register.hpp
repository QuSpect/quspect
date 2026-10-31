#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace quspect::native::runtime {

/// 经典寄存器文件 — 独立于状态向量的测量结果存储
///
/// 与 qulacs 不同：经典寄存器不塞入状态对象。
/// 原因：
/// - 状态与测量结果是两个不同语义层
/// - 便于做 per-shot 结果收集
/// - 便于未来硬件后端复用相同结果对象
class ClassicalRegisterFile {
 public:
    /// 构造 size 位经典寄存器，全部初始化为 0
    explicit ClassicalRegisterFile(uint32_t num_clbits);

    /// 写入单比特值
    void Set(uint32_t index, uint8_t value);

    /// 读取单比特值
    uint8_t Get(uint32_t index) const;

    /// 将寄存器内容导出为 bitstring (index 0 = 最低位)
    std::string ToBitstring() const;

    /// 获取寄存器容量
    uint32_t num_clbits() const;

 private:
    std::vector<uint8_t> bits_;
};

}  // namespace quspect::native::runtime
