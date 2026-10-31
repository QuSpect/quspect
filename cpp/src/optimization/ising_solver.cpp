#include "quspect_native/optimization/ising_solver.hpp"

#include <cmath>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "quspect_native/common/parallel.hpp"

namespace quspect::native::optimization {
namespace {

constexpr double kMatrixTolerance = 1e-12;

void ValidateSpinVector(const std::vector<std::int8_t>& spins, std::size_t expected_size) {
    if (spins.size() != expected_size) {
        throw std::invalid_argument("spin vector size does not match the Ising matrix");
    }
    for (const std::int8_t spin : spins) {
        if (spin != -1 && spin != 1) {
            throw std::invalid_argument("spins must contain only -1 and +1");
        }
    }
}

void ValidateSolverOptions(const IsingSolverOptions& options) {
    if (options.num_reads == 0) {
        throw std::invalid_argument("num_reads must be greater than zero");
    }
    if (options.sweeps == 0) {
        throw std::invalid_argument("sweeps must be greater than zero");
    }
    if (!std::isfinite(options.beta_start) || options.beta_start <= 0.0) {
        throw std::invalid_argument("beta_start must be finite and greater than zero");
    }
    if (!std::isfinite(options.beta_end) || options.beta_end <= 0.0) {
        throw std::invalid_argument("beta_end must be finite and greater than zero");
    }
    if (options.beta_end < options.beta_start) {
        throw std::invalid_argument("beta_end must be greater than or equal to beta_start");
    }
    if (options.num_reads > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument("num_reads is too large");
    }
}

double EnergyUnchecked(const std::vector<double>& matrix, std::size_t num_spins,
                       const std::vector<std::int8_t>& spins) {
    double energy = 0.0;
    for (std::size_t row = 0; row < num_spins; ++row) {
        const double row_spin = static_cast<double>(spins[row]);
        for (std::size_t column = row + 1; column < num_spins; ++column) {
            energy -= 2.0 * matrix[row * num_spins + column] * row_spin * static_cast<double>(spins[column]);
        }
    }
    return energy;
}

std::vector<double> ComputeLocalFields(const DenseIsingModel& model, const std::vector<std::int8_t>& spins) {
    const std::size_t num_spins = model.num_spins();
    const auto& matrix = model.matrix();
    std::vector<double> fields(num_spins, 0.0);
    for (std::size_t row = 0; row < num_spins; ++row) {
        double field = 0.0;
        for (std::size_t column = 0; column < num_spins; ++column) {
            field += matrix[row * num_spins + column] * static_cast<double>(spins[column]);
        }
        fields[row] = field;
    }
    return fields;
}

void ApplyAcceptedFlip(const DenseIsingModel& model, std::size_t spin_index, std::vector<std::int8_t>* spins,
                       std::vector<double>* fields) {
    const std::size_t num_spins = model.num_spins();
    const auto& matrix = model.matrix();
    const std::int8_t old_spin = (*spins)[spin_index];
    const double field_change_factor = -2.0 * static_cast<double>(old_spin);
    for (std::size_t row = 0; row < num_spins; ++row) {
        (*fields)[row] += matrix[row * num_spins + spin_index] * field_change_factor;
    }
    (*spins)[spin_index] = static_cast<std::int8_t>(-old_spin);
}

std::uint64_t SplitMix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t SeedForRead(std::uint64_t base_seed, std::size_t read_index) {
    return SplitMix64(base_seed + static_cast<std::uint64_t>(read_index));
}

double BetaAtSweep(const IsingSolverOptions& options, std::size_t sweep) {
    if (options.sweeps == 1) {
        return options.beta_end;
    }
    const double fraction = static_cast<double>(sweep) / static_cast<double>(options.sweeps - 1);
    return options.beta_start + fraction * (options.beta_end - options.beta_start);
}

IsingSample RunOneRead(const DenseIsingModel& model, const IsingSolverOptions& options, std::size_t read_index) {
    const std::size_t num_spins = model.num_spins();
    std::mt19937_64 random_engine(SeedForRead(options.seed, read_index));
    std::uniform_real_distribution<double> uniform_probability(0.0, 1.0);

    std::vector<std::int8_t> spins(num_spins, -1);
    for (std::int8_t& spin : spins) {
        spin = (random_engine() & 1ULL) == 0ULL ? -1 : 1;
    }

    std::vector<std::size_t> update_order(num_spins);
    std::iota(update_order.begin(), update_order.end(), 0);
    std::vector<double> fields = ComputeLocalFields(model, spins);
    double energy = EnergyUnchecked(model.matrix(), num_spins, spins);
    IsingSample best{spins, energy};

    for (std::size_t sweep = 0; sweep < options.sweeps; ++sweep) {
        std::shuffle(update_order.begin(), update_order.end(), random_engine);
        const double beta = BetaAtSweep(options, sweep);
        for (const std::size_t spin_index : update_order) {
            const double delta = 4.0 * static_cast<double>(spins[spin_index]) * fields[spin_index];
            const bool accept = delta <= 0.0 || uniform_probability(random_engine) < std::exp(-beta * delta);
            if (!accept) {
                continue;
            }
            energy += delta;
            ApplyAcceptedFlip(model, spin_index, &spins, &fields);
            if (energy < best.energy) {
                best.spins = spins;
                best.energy = energy;
            }
        }
    }

    // 增量累加可能产生微小舍入漂移，跨语言返回前重新计算一次。
    best.energy = EnergyUnchecked(model.matrix(), num_spins, best.spins);
    return best;
}

std::size_t CountTrailingZeros(std::uint64_t value) {
    std::size_t count = 0;
    while ((value & 1ULL) == 0ULL) {
        value >>= 1U;
        ++count;
    }
    return count;
}

}  // namespace

DenseIsingModel::DenseIsingModel(std::vector<double> matrix, std::size_t num_spins)
    : num_spins_(num_spins), matrix_(std::move(matrix)) {
    if (num_spins_ != 0 && num_spins_ > std::numeric_limits<std::size_t>::max() / num_spins_) {
        throw std::invalid_argument("Ising matrix dimension is too large");
    }
    if (matrix_.size() != num_spins_ * num_spins_) {
        throw std::invalid_argument("Ising matrix must be square");
    }

    for (std::size_t row = 0; row < num_spins_; ++row) {
        for (std::size_t column = 0; column < num_spins_; ++column) {
            const double coefficient = matrix_[row * num_spins_ + column];
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument("Ising matrix must contain only finite values");
            }
            if (row == column && std::abs(coefficient) > kMatrixTolerance) {
                throw std::invalid_argument("Ising matrix diagonal must be zero");
            }
            if (column > row) {
                const double opposing = matrix_[column * num_spins_ + row];
                if (std::abs(coefficient - opposing) > kMatrixTolerance) {
                    throw std::invalid_argument("Ising matrix must be symmetric");
                }
                // Averaging preserves s^T M s and makes the exact symmetry
                // required by cached local-field updates explicit.
                const double canonical = 0.5 * (coefficient + opposing);
                matrix_[row * num_spins_ + column] = canonical;
                matrix_[column * num_spins_ + row] = canonical;
            }
        }
        matrix_[row * num_spins_ + row] = 0.0;
    }
}

std::size_t DenseIsingModel::num_spins() const {
    return num_spins_;
}

const std::vector<double>& DenseIsingModel::matrix() const {
    return matrix_;
}

double DenseIsingModel::Energy(const std::vector<std::int8_t>& spins) const {
    ValidateSpinVector(spins, num_spins_);
    return EnergyUnchecked(matrix_, num_spins_, spins);
}

double DenseIsingModel::FlipEnergyDelta(const std::vector<std::int8_t>& spins, std::size_t spin_index) const {
    ValidateSpinVector(spins, num_spins_);
    if (spin_index >= num_spins_) {
        throw std::out_of_range("spin_index is outside the Ising model");
    }
    double field = 0.0;
    for (std::size_t column = 0; column < num_spins_; ++column) {
        field += matrix_[spin_index * num_spins_ + column] * static_cast<double>(spins[column]);
    }
    return 4.0 * static_cast<double>(spins[spin_index]) * field;
}

IsingSolverResult SolveIsingSimulatedAnnealing(const DenseIsingModel& model, const IsingSolverOptions& options) {
    ValidateSolverOptions(options);

    IsingSolverResult result;
    result.num_spins = model.num_spins();
    result.samples.resize(options.num_reads);

    // 每个 read 使用由 (seed, read_index) 派生的独立随机数流，并写入自己的结果槽位，
    // 因此结果与线程数及调度无关。
    const auto num_reads = static_cast<std::int64_t>(options.num_reads);
    common::ParallelForRange(num_reads, num_reads > 1 ? common::GetNumThreads() : 1,
                             [&](std::int64_t begin, std::int64_t end) {
                                 for (std::int64_t read_index = begin; read_index < end; ++read_index) {
                                     result.samples[static_cast<std::size_t>(read_index)] = RunOneRead(
                                         model, options, static_cast<std::size_t>(read_index));
                                 }
                             });

    // 保留每个 read（包括重复样本），等能量时保持 read 索引顺序。
    std::stable_sort(result.samples.begin(), result.samples.end(),
                     [](const IsingSample& left, const IsingSample& right) { return left.energy < right.energy; });
    return result;
}

IsingSample SolveIsingExact(const DenseIsingModel& model, std::size_t max_spins) {
    const std::size_t num_spins = model.num_spins();
    if (num_spins > max_spins) {
        throw std::invalid_argument("exact Ising solver exceeds max_spins");
    }
    if (num_spins >= std::numeric_limits<std::uint64_t>::digits) {
        throw std::invalid_argument("exact Ising solver supports fewer than 64 spins");
    }

    std::vector<std::int8_t> spins(num_spins, -1);
    std::vector<double> fields = ComputeLocalFields(model, spins);
    double energy = EnergyUnchecked(model.matrix(), num_spins, spins);
    IsingSample best{spins, energy};

    const std::uint64_t state_count = 1ULL << num_spins;
    for (std::uint64_t state_index = 1; state_index < state_count; ++state_index) {
        const std::size_t spin_index = CountTrailingZeros(state_index);
        const double delta = 4.0 * static_cast<double>(spins[spin_index]) * fields[spin_index];
        energy += delta;
        ApplyAcceptedFlip(model, spin_index, &spins, &fields);
        if (energy < best.energy) {
            best.spins = spins;
            best.energy = energy;
        }
    }

    best.energy = EnergyUnchecked(model.matrix(), num_spins, best.spins);
    return best;
}

}  // namespace quspect::native::optimization
