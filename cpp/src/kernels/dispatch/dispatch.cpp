#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#    define QSPECT_X86_MSVC 1
#    include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
#    define QSPECT_X86_GNU 1
#    include <cpuid.h>
#endif

#include "quspect_native/kernels/dispatch_table.hpp"

namespace quspect::native::kernels {

// ============================================================
// CPU 特性检测
// ============================================================

namespace {

#if defined(QSPECT_X86_MSVC) || defined(QSPECT_X86_GNU)

// CPUID 寄存器结果
struct CpuidRegs {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
};

/// 执行 CPUID(leaf, subleaf)。leaf 超出 CPU 支持的最大值时返回 false。
bool QueryCpuid(uint32_t leaf, uint32_t subleaf, CpuidRegs* out) {
#    if defined(QSPECT_X86_MSVC)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (static_cast<uint32_t>(regs[0]) < leaf) {
        return false;
    }
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    out->eax = static_cast<uint32_t>(regs[0]);
    out->ebx = static_cast<uint32_t>(regs[1]);
    out->ecx = static_cast<uint32_t>(regs[2]);
    out->edx = static_cast<uint32_t>(regs[3]);
    return true;
#    else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx)) {
        return false;
    }
    out->eax = eax;
    out->ebx = ebx;
    out->ecx = ecx;
    out->edx = edx;
    return true;
#    endif
}

/// 读取 XCR0 低 32 位（调用前必须确认 OSXSAVE 已置位）。
uint32_t ReadXcr0Low() {
#    if defined(QSPECT_X86_MSVC)
    return static_cast<uint32_t>(_xgetbv(0) & 0xFFFFFFFFULL);
#    else
    uint32_t xcr0_lo = 0;
    uint32_t xcr0_hi = 0;
    __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void) xcr0_hi;
    return xcr0_lo;
#    endif
}

#endif  // QSPECT_X86_MSVC || QSPECT_X86_GNU

}  // namespace

// AVX2 内核以 -mavx2 -mfma（MSVC: /arch:AVX2）编译，因此除 AVX2 外还必须要求 FMA。
bool CpuSupportsAVX2() {
#if defined(QSPECT_X86_MSVC) || defined(QSPECT_X86_GNU)
    CpuidRegs regs;

    // CPUID.01H:ECX — FMA (bit 12)、OSXSAVE (bit 27)、AVX (bit 28)
    if (!QueryCpuid(1, 0, &regs)) {
        return false;
    }
    constexpr uint32_t kFmaBit = 1u << 12;
    constexpr uint32_t kOsxsaveBit = 1u << 27;
    constexpr uint32_t kAvxBit = 1u << 28;
    if ((regs.ecx & kFmaBit) == 0 || (regs.ecx & kOsxsaveBit) == 0 || (regs.ecx & kAvxBit) == 0) {
        return false;
    }

    // XGETBV (XCR0): 确认 OS 已启用 SSE (bit 1) 和 AVX (bit 2) 状态保存
    if ((ReadXcr0Low() & 0x6u) != 0x6u) {
        return false;
    }

    // CPUID.07H.00H:EBX — AVX2 (bit 5)
    if (!QueryCpuid(7, 0, &regs)) {
        return false;
    }
    constexpr uint32_t kAvx2Bit = 1u << 5;
    return (regs.ebx & kAvx2Bit) != 0;
#else
    return false;
#endif
}

// ============================================================
// Scalar kernel 函数声明（定义在各自的 .cpp 文件中）
// ============================================================

namespace scalar {

// 单比特门
template <typename FP>
void apply_x(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_y(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_z(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_h(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_s(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_sdg(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_t(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_tdg(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_rx(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_ry(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_rz(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);

// 双比特门
template <typename FP>
void apply_cnot(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);
template <typename FP>
void apply_cz(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);
template <typename FP>
void apply_swap(std::complex<FP>*, uint64_t, uint32_t, uint32_t);

// 通用门
template <typename FP>
void apply_dense_gate(std::complex<FP>*, uint64_t, const uint32_t*, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_diagonal_gate(std::complex<FP>*, uint64_t, const uint32_t*, uint32_t, const std::complex<FP>*);

// 测量与采样
template <typename FP>
FP compute_probability_zero(const std::complex<FP>*, uint64_t, uint32_t);
template <typename FP>
void collapse_qubit(std::complex<FP>*, uint64_t, uint32_t, int);
template <typename FP>
uint64_t sample_from_statevector(const std::complex<FP>*, uint64_t, double);
template <typename FP>
std::complex<FP> compute_pauli_expectation(const std::complex<FP>*, uint64_t, uint64_t, uint64_t, uint32_t, int);

}  // namespace scalar

// ============================================================
// 填充 dispatch table — scalar
// ============================================================

template <typename FP>
static KernelDispatchTable<FP> BuildScalarTable() {
    KernelDispatchTable<FP> t;
    std::memset(&t, 0, sizeof(t));

    t.backend_name = "scalar";

    // 单比特门
    t.x = scalar::apply_x<FP>;
    t.y = scalar::apply_y<FP>;
    t.z = scalar::apply_z<FP>;
    t.h = scalar::apply_h<FP>;
    t.s = scalar::apply_s<FP>;
    t.sdg = scalar::apply_sdg<FP>;
    t.t = scalar::apply_t<FP>;
    t.tdg = scalar::apply_tdg<FP>;
    t.rx = scalar::apply_rx<FP>;
    t.ry = scalar::apply_ry<FP>;
    t.rz = scalar::apply_rz<FP>;

    // 双比特门
    t.cnot = scalar::apply_cnot<FP>;
    t.cz = scalar::apply_cz<FP>;
    t.swap = scalar::apply_swap<FP>;

    // 通用门
    t.dense_k = scalar::apply_dense_gate<FP>;
    t.diagonal_k = scalar::apply_diagonal_gate<FP>;

    // 测量与采样
    t.prob0 = scalar::compute_probability_zero<FP>;
    t.collapse = scalar::collapse_qubit<FP>;
    t.sampling = scalar::sample_from_statevector<FP>;
    t.pauli_expect = scalar::compute_pauli_expectation<FP>;

    return t;
}

// ============================================================
// AVX2 kernel 函数声明（定义在各自的 .cpp 文件中）
// ============================================================

#ifdef QSPECT_USE_AVX2
namespace avx2 {

// 单比特门
template <typename FP>
void apply_x(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_y(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_z(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_h(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_s(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_sdg(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_t(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_tdg(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_rx(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_ry(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_rz(std::complex<FP>*, uint64_t, uint32_t, const std::complex<FP>*);

// 双比特门
template <typename FP>
void apply_cnot(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);
template <typename FP>
void apply_cz(std::complex<FP>*, uint64_t, uint32_t, uint64_t, uint64_t, const std::complex<FP>*);
template <typename FP>
void apply_swap(std::complex<FP>*, uint64_t, uint32_t, uint32_t);

// 通用门
template <typename FP>
void apply_dense_gate(std::complex<FP>*, uint64_t, const uint32_t*, uint32_t, const std::complex<FP>*);
template <typename FP>
void apply_diagonal_gate(std::complex<FP>*, uint64_t, const uint32_t*, uint32_t, const std::complex<FP>*);

// 期望值
template <typename FP>
std::complex<FP> compute_pauli_expectation(const std::complex<FP>*, uint64_t, uint64_t, uint64_t, uint32_t, int);

}  // namespace avx2

/// 填充 AVX2 dispatch table
template <typename FP>
static KernelDispatchTable<FP> BuildAVX2Table() {
    KernelDispatchTable<FP> t;
    std::memset(&t, 0, sizeof(t));

    t.backend_name = "avx2";

    // 单比特门
    t.x = avx2::apply_x<FP>;
    t.y = avx2::apply_y<FP>;
    t.z = avx2::apply_z<FP>;
    t.h = avx2::apply_h<FP>;
    t.s = avx2::apply_s<FP>;
    t.sdg = avx2::apply_sdg<FP>;
    t.t = avx2::apply_t<FP>;
    t.tdg = avx2::apply_tdg<FP>;
    t.rx = avx2::apply_rx<FP>;
    t.ry = avx2::apply_ry<FP>;
    t.rz = avx2::apply_rz<FP>;

    // 双比特门
    t.cnot = avx2::apply_cnot<FP>;
    t.cz = avx2::apply_cz<FP>;
    t.swap = avx2::apply_swap<FP>;

    // 通用门
    t.dense_k = avx2::apply_dense_gate<FP>;
    t.diagonal_k = avx2::apply_diagonal_gate<FP>;

    // 测量与采样 — 回退到 scalar（非向量化友好操作）
    t.prob0 = scalar::compute_probability_zero<FP>;
    t.collapse = scalar::collapse_qubit<FP>;
    t.sampling = scalar::sample_from_statevector<FP>;

    // 期望值
    t.pauli_expect = avx2::compute_pauli_expectation<FP>;

    return t;
}
#endif  // QSPECT_USE_AVX2

// ============================================================
// 全局 dispatch 实例
// ============================================================

static const KernelDispatchTable<float> kScalarTableF32 = BuildScalarTable<float>();
static const KernelDispatchTable<double> kScalarTableF64 = BuildScalarTable<double>();

#ifdef QSPECT_USE_AVX2
static const KernelDispatchTable<float> kAVX2TableF32 = BuildAVX2Table<float>();
static const KernelDispatchTable<double> kAVX2TableF64 = BuildAVX2Table<double>();
#endif

const KernelDispatchTable<float>& GetDispatchF32() {
#ifdef QSPECT_USE_AVX2
    if (CpuSupportsAVX2())
        return kAVX2TableF32;
#endif
    return kScalarTableF32;
}

const KernelDispatchTable<double>& GetDispatchF64() {
#ifdef QSPECT_USE_AVX2
    if (CpuSupportsAVX2())
        return kAVX2TableF64;
#endif
    return kScalarTableF64;
}

}  // namespace quspect::native::kernels
