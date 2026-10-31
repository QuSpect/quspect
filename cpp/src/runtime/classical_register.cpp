#include "quspect_native/runtime/classical_register.hpp"

#include <algorithm>
#include <stdexcept>

namespace quspect::native::runtime {

ClassicalRegisterFile::ClassicalRegisterFile(uint32_t num_clbits) : bits_(num_clbits, 0) {
}

void ClassicalRegisterFile::Set(uint32_t index, uint8_t value) {
    if (index >= bits_.size()) {
        throw std::out_of_range("classical register index out of range");
    }
    if (value > 1) {
        throw std::invalid_argument("classical register value must be 0 or 1");
    }
    bits_[index] = value;
}

uint8_t ClassicalRegisterFile::Get(uint32_t index) const {
    if (index >= bits_.size()) {
        throw std::out_of_range("classical register index out of range");
    }
    return bits_[index];
}

std::string ClassicalRegisterFile::ToBitstring() const {
    // index 0 = 最低位（设计中约定）
    // 返回字符串时，通常用户期望从左到右读高位到低位
    // 所以将 bits_[num_clbits-1]..bits_[0] 按序输出
    std::string result;
    result.reserve(bits_.size());
    for (int i = static_cast<int>(bits_.size()) - 1; i >= 0; --i) {
        result.push_back(bits_[i] ? '1' : '0');
    }
    return result;
}

uint32_t ClassicalRegisterFile::num_clbits() const {
    return static_cast<uint32_t>(bits_.size());
}

}  // namespace quspect::native::runtime
