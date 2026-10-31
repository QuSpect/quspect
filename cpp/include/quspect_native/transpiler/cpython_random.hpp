#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace quspect::native::transpiler {

// 与 CPython ``random.Random``（Modules/_randommodule.c + Lib/random.py）逐位一致的随机源。
//
// 只实现路由需要的部分：MT19937 核心、``seed(int)``、``getrandbits(k)``（k <= 64）与
// ``_randbelow(n)``（即 ``choice`` 的下标抽样）。内部状态与 ``Random.getstate()[1]``
// （624 个 32 位字 + 下标）一一对应，可以在 Python 与 C++ 之间来回传递，保证调用方的
// Python 随机源在原生路由之后继续产生完全相同的序列。
class CPythonRandom {
 public:
    static constexpr int kStateSize = 624;
    // ``getstate()[1]`` 的长度：624 个状态字加一个下标。
    static constexpr int kPackedStateSize = kStateSize + 1;

    // 等价于 ``random.Random(0)``。
    CPythonRandom();

    // 等价于 ``random.Random(seed)``（非负整数种子，按 32 位小端字切分后 init_by_array）。
    explicit CPythonRandom(std::uint64_t seed);

    // 从 ``Random.getstate()[1]`` 恢复；长度必须是 625，下标必须在 [0, 624]。
    // 抛出 std::invalid_argument。
    static CPythonRandom FromState(const std::vector<std::uint32_t>& packed_state);

    // ``Random.getstate()[1]`` 的内容（625 个字）。
    std::vector<std::uint32_t> State() const;

    // MT19937 的下一个 32 位输出（CPython ``genrand_uint32``）。
    std::uint32_t GenRandUint32();

    // ``Random.getrandbits(k)``，0 <= k <= 64；抛出 std::invalid_argument。
    std::uint64_t GetRandBits(int k);

    // ``Random._randbelow(n)``（``_randbelow_with_getrandbits``），n > 0；抛出 std::invalid_argument。
    std::uint64_t RandBelow(std::uint64_t n);

    // 自构造（或 FromState）以来是否抽取过随机数。
    bool used() const {
        return used_;
    }

 private:
    void InitGenRand(std::uint32_t s);
    void InitByArray(const std::uint32_t* key, std::size_t key_length);

    std::array<std::uint32_t, kStateSize> mt_{};
    int index_ = kStateSize;
    bool used_ = false;
};

}  // namespace quspect::native::transpiler
