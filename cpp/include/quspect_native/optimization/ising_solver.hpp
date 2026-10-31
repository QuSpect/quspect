#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace quspect::native::optimization {

// 稠密 Ising 模型采用全矩阵硬件层约定 E(s) = -s^T M s。
class DenseIsingModel {
 public:
    DenseIsingModel(std::vector<double> matrix, std::size_t num_spins);

    std::size_t num_spins() const;
    const std::vector<double>& matrix() const;

    double Energy(const std::vector<std::int8_t>& spins) const;
    double FlipEnergyDelta(const std::vector<std::int8_t>& spins, std::size_t spin_index) const;

 private:
    std::size_t num_spins_;
    std::vector<double> matrix_;
};

struct IsingSolverOptions {
    std::uint64_t seed = 0;
    std::size_t num_reads = 1;
    std::size_t sweeps = 1000;
    double beta_start = 0.1;
    double beta_end = 5.0;
};

struct IsingSample {
    std::vector<std::int8_t> spins;
    double energy = 0.0;
};

struct IsingSolverResult {
    std::size_t num_spins = 0;
    std::vector<IsingSample> samples;
};

// 对每个 read 独立执行 Metropolis 模拟退火，并按能量稳定排序。
IsingSolverResult SolveIsingSimulatedAnnealing(const DenseIsingModel& model, const IsingSolverOptions& options = {});

// 使用 Gray code 遍历全部状态，作为小规模问题的精确正确性基准。
IsingSample SolveIsingExact(const DenseIsingModel& model, std::size_t max_spins = 24);

}  // namespace quspect::native::optimization
