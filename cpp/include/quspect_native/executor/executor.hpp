#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace quspect::native::runtime {
struct QSALProgram;
}  // namespace quspect::native::runtime

namespace quspect::native::executor {

// ============================================================
// 枚举类型
// ============================================================

enum class BackendKind {
    kSimulator,
    kHardware,
};

enum class PrecisionKind {
    kFloat32,
    kFloat64,
};

// ============================================================
// 执行请求
// ============================================================

struct ExecutionRequest {
    uint64_t shots = 0;
    bool return_statevector = false;
    bool return_expectations = false;
    bool return_samples = false;
    std::optional<uint64_t> seed;
};

// ============================================================
// 执行结果
// ============================================================

struct ExecutionResult {
    // 末态 statevector（根据精度选一个）
    std::vector<std::complex<double>> statevector_f64;
    std::vector<std::complex<float>> statevector_f32;

    // 采样原始结果（packed: 每个 uint64 的低 clbits 位是测量结果）
    std::vector<uint64_t> packed_samples;

    // 计数聚合 {bitstring: count}
    std::unordered_map<std::string, uint64_t> counts;

    // 可观测量期望值
    std::unordered_map<std::string, double> expectations_real;
    std::unordered_map<std::string, std::complex<double>> expectations_complex;

    // 元信息
    std::unordered_map<std::string, std::string> metadata;
};

// ============================================================
// Executor 抽象接口
// ============================================================

class IExecutor {
 public:
    virtual ~IExecutor() = default;

    virtual ExecutionResult Run(const runtime::QSALProgram& program, const ExecutionRequest& request) = 0;
};

// ============================================================
// SimulationExecutor — 状态向量模拟器
// ============================================================

class SimulationExecutor : public IExecutor {
 public:
    struct Options {
        PrecisionKind precision = PrecisionKind::kFloat64;
        uint64_t seed = 0;
    };

    explicit SimulationExecutor(Options options);
    ~SimulationExecutor() override;

    ExecutionResult Run(const runtime::QSALProgram& program, const ExecutionRequest& request) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================
// HardwareExecutor — 真机占位
// ============================================================

class HardwareExecutor : public IExecutor {
 public:
    HardwareExecutor();
    ~HardwareExecutor() override;

    ExecutionResult Run(const runtime::QSALProgram& program, const ExecutionRequest& request) override;
};

}  // namespace quspect::native::executor
