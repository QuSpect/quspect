// Minimal SHA-256 (FIPS 180-4), used to identify Custom envelope data.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace quspect::native::pulse {

class SHA256 {
 public:
    SHA256() {
        Reset();
    }

    void Update(const void* data, size_t len) {
        const auto* p = static_cast<const unsigned char*>(data);
        bit_len_ += static_cast<uint64_t>(len) * 8;
        while (len > 0) {
            size_t take = 64 - buf_len_;
            if (take > len)
                take = len;
            std::memcpy(buf_ + buf_len_, p, take);
            buf_len_ += take;
            p += take;
            len -= take;
            if (buf_len_ == 64) {
                Transform();
                buf_len_ = 0;
            }
        }
    }

    void Final(unsigned char out[32]) {
        uint64_t bits = bit_len_;
        unsigned char pad = 0x80;
        Update(&pad, 1);
        unsigned char zero = 0;
        while (buf_len_ != 56)
            Update(&zero, 1);
        unsigned char len_be[8];
        for (int i = 0; i < 8; ++i)
            len_be[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        Update(len_be, 8);
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = static_cast<unsigned char>(h_[i] >> 24);
            out[4 * i + 1] = static_cast<unsigned char>(h_[i] >> 16);
            out[4 * i + 2] = static_cast<unsigned char>(h_[i] >> 8);
            out[4 * i + 3] = static_cast<unsigned char>(h_[i]);
        }
    }

 private:
    static uint32_t RotR(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    void Reset() {
        static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, init, sizeof(h_));
        buf_len_ = 0;
        bit_len_ = 0;
    }

    void Transform() {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(buf_[4 * i]) << 24) | (uint32_t(buf_[4 * i + 1]) << 16) | (uint32_t(buf_[4 * i + 2]) << 8)
                   | uint32_t(buf_[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t s1 = RotR(e, 6) ^ RotR(e, 11) ^ RotR(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + s1 + ch + k[i] + w[i];
            uint32_t s0 = RotR(a, 2) ^ RotR(a, 13) ^ RotR(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    uint32_t h_[8];
    unsigned char buf_[64];
    size_t buf_len_;
    uint64_t bit_len_;
};

}  // namespace quspect::native::pulse
