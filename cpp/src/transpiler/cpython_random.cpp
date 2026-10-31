#include "quspect_native/transpiler/cpython_random.hpp"

#include <stdexcept>

namespace quspect::native::transpiler {

namespace {

constexpr int kN = CPythonRandom::kStateSize;
constexpr int kM = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
constexpr std::uint32_t kUpperMask = 0x80000000U;
constexpr std::uint32_t kLowerMask = 0x7fffffffU;

int BitLength(std::uint64_t n) {
    int bits = 0;
    while (n != 0) {
        ++bits;
        n >>= 1;
    }
    return bits;
}

}  // namespace

CPythonRandom::CPythonRandom() : CPythonRandom(0) {
}

CPythonRandom::CPythonRandom(std::uint64_t seed) {
    // CPython random_seed：abs(seed) 按 32 位字从低到高切分；seed == 0 时 key = [0]。
    std::uint32_t key[2] = {static_cast<std::uint32_t>(seed & 0xffffffffU), static_cast<std::uint32_t>(seed >> 32)};
    const std::size_t key_length = (seed >> 32) != 0 ? 2 : 1;
    InitByArray(key, key_length);
}

CPythonRandom CPythonRandom::FromState(const std::vector<std::uint32_t>& packed_state) {
    if (packed_state.size() != static_cast<std::size_t>(kPackedStateSize)) {
        throw std::invalid_argument("CPythonRandom state must have 625 entries");
    }
    const std::uint32_t index = packed_state[kStateSize];
    if (index > static_cast<std::uint32_t>(kStateSize)) {
        throw std::invalid_argument("CPythonRandom state index out of range");
    }
    CPythonRandom rng;
    for (int i = 0; i < kStateSize; ++i) {
        rng.mt_[i] = packed_state[i];
    }
    rng.index_ = static_cast<int>(index);
    rng.used_ = false;
    return rng;
}

std::vector<std::uint32_t> CPythonRandom::State() const {
    std::vector<std::uint32_t> out(mt_.begin(), mt_.end());
    out.push_back(static_cast<std::uint32_t>(index_));
    return out;
}

void CPythonRandom::InitGenRand(std::uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < kN; ++i) {
        mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<std::uint32_t>(i);
    }
    index_ = kN;
}

void CPythonRandom::InitByArray(const std::uint32_t* key, std::size_t key_length) {
    InitGenRand(19650218U);
    std::size_t i = 1;
    std::size_t j = 0;
    std::size_t k = (static_cast<std::size_t>(kN) > key_length ? static_cast<std::size_t>(kN) : key_length);
    for (; k != 0; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] + static_cast<std::uint32_t>(j);
        ++i;
        ++j;
        if (i >= static_cast<std::size_t>(kN)) {
            mt_[0] = mt_[kN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = kN - 1; k != 0; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - static_cast<std::uint32_t>(i);
        ++i;
        if (i >= static_cast<std::size_t>(kN)) {
            mt_[0] = mt_[kN - 1];
            i = 1;
        }
    }
    mt_[0] = 0x80000000U;  // 保证初始状态非零
    index_ = kN;
    used_ = false;
}

std::uint32_t CPythonRandom::GenRandUint32() {
    static constexpr std::uint32_t kMag01[2] = {0x0U, kMatrixA};
    used_ = true;
    std::uint32_t y;
    if (index_ >= kN) {
        int kk = 0;
        for (; kk < kN - kM; ++kk) {
            y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
            mt_[kk] = mt_[kk + kM] ^ (y >> 1) ^ kMag01[y & 0x1U];
        }
        for (; kk < kN - 1; ++kk) {
            y = (mt_[kk] & kUpperMask) | (mt_[kk + 1] & kLowerMask);
            mt_[kk] = mt_[kk + (kM - kN)] ^ (y >> 1) ^ kMag01[y & 0x1U];
        }
        y = (mt_[kN - 1] & kUpperMask) | (mt_[0] & kLowerMask);
        mt_[kN - 1] = mt_[kM - 1] ^ (y >> 1) ^ kMag01[y & 0x1U];
        index_ = 0;
    }
    y = mt_[index_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

std::uint64_t CPythonRandom::GetRandBits(int k) {
    if (k < 0 || k > 64) {
        throw std::invalid_argument("GetRandBits supports 0 <= k <= 64");
    }
    if (k == 0) {
        return 0;
    }
    if (k <= 32) {
        return GenRandUint32() >> (32 - k);
    }
    // CPython：按 32 位字从最低位到最高位依次填充，最后一个字丢弃多余的低位。
    const std::uint64_t low = GenRandUint32();
    std::uint64_t high = GenRandUint32();
    if (k < 64) {
        high >>= (64 - k);
    }
    return low | (high << 32);
}

std::uint64_t CPythonRandom::RandBelow(std::uint64_t n) {
    if (n == 0) {
        throw std::invalid_argument("RandBelow requires n > 0");
    }
    const int k = BitLength(n);
    std::uint64_t r = GetRandBits(k);
    while (r >= n) {
        r = GetRandBits(k);
    }
    return r;
}

}  // namespace quspect::native::transpiler
