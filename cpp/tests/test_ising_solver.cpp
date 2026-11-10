#include <cmath>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include "quspect_native/optimization/ising_solver.hpp"

namespace {

using quspect::native::optimization::DenseIsingModel;
using quspect::native::optimization::IsingSample;
using quspect::native::optimization::IsingSolverOptions;
using quspect::native::optimization::SolveIsingExact;
using quspect::native::optimization::SolveIsingSimulatedAnnealing;

IsingSample BruteForce(const DenseIsingModel& model) {
    const std::size_t num_spins = model.num_spins();
    std::vector<std::int8_t> spins(num_spins, -1);
    IsingSample best{spins, model.Energy(spins)};
    const std::uint64_t state_count = 1ULL << num_spins;
    for (std::uint64_t state = 0; state < state_count; ++state) {
        for (std::size_t spin = 0; spin < num_spins; ++spin) {
            spins[spin] = ((state >> spin) & 1ULL) == 0ULL ? -1 : 1;
        }
        const double energy = model.Energy(spins);
        if (energy < best.energy) {
            best = {spins, energy};
        }
    }
    return best;
}

DenseIsingModel MakeDenseFerromagnet(std::size_t num_spins) {
    std::vector<double> matrix(num_spins * num_spins, 0.0);
    for (std::size_t row = 0; row < num_spins; ++row) {
        for (std::size_t column = row + 1; column < num_spins; ++column) {
            const double coupling = 0.25 + static_cast<double>((row + column) % 5) * 0.1;
            matrix[row * num_spins + column] = coupling;
            matrix[column * num_spins + row] = coupling;
        }
    }
    return DenseIsingModel(std::move(matrix), num_spins);
}

void ExpectSameResult(const quspect::native::optimization::IsingSolverResult& left,
                      const quspect::native::optimization::IsingSolverResult& right) {
    ASSERT_EQ(left.num_spins, right.num_spins);
    ASSERT_EQ(left.samples.size(), right.samples.size());
    for (std::size_t index = 0; index < left.samples.size(); ++index) {
        EXPECT_EQ(left.samples[index].spins, right.samples[index].spins);
        EXPECT_DOUBLE_EQ(left.samples[index].energy, right.samples[index].energy);
    }
}

}  // namespace

TEST(DenseIsingModelTest, UsesHardwareEnergyConvention) {
    const DenseIsingModel model(
        {
            0.0,
            2.0,
            -1.0,
            2.0,
            0.0,
            3.0,
            -1.0,
            3.0,
            0.0,
        },
        3);
    EXPECT_DOUBLE_EQ(model.Energy({1, -1, 1}), 12.0);
}

TEST(DenseIsingModelTest, FlipDeltaMatchesRecomputedEnergy) {
    const DenseIsingModel model(
        {
            0.0,
            1.5,
            -0.25,
            1.5,
            0.0,
            2.0,
            -0.25,
            2.0,
            0.0,
        },
        3);
    const std::vector<std::int8_t> spins{1, -1, 1};
    const double original_energy = model.Energy(spins);

    for (std::size_t spin_index = 0; spin_index < spins.size(); ++spin_index) {
        auto flipped = spins;
        flipped[spin_index] = static_cast<std::int8_t>(-flipped[spin_index]);
        EXPECT_NEAR(model.FlipEnergyDelta(spins, spin_index), model.Energy(flipped) - original_energy, 1e-12);
    }
}

TEST(DenseIsingModelTest, RejectsInvalidMatrices) {
    EXPECT_THROW(DenseIsingModel({0.0, 1.0, 1.0}, 2), std::invalid_argument);
    EXPECT_THROW(DenseIsingModel({1.0, 0.0, 0.0, 0.0}, 2), std::invalid_argument);
    EXPECT_THROW(DenseIsingModel({0.0, 1.0, 2.0, 0.0}, 2), std::invalid_argument);
    EXPECT_THROW(DenseIsingModel({0.0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}, 2), std::invalid_argument);
    EXPECT_THROW(DenseIsingModel({0.0, std::numeric_limits<double>::infinity(), 0.0, 0.0}, 2), std::invalid_argument);
}

TEST(DenseIsingModelTest, RejectsInvalidSpinVectors) {
    const DenseIsingModel model({0.0, 1.0, 1.0, 0.0}, 2);
    EXPECT_THROW(model.Energy({1}), std::invalid_argument);
    EXPECT_THROW(model.Energy({0, 1}), std::invalid_argument);
    EXPECT_THROW(model.FlipEnergyDelta({1, -1}, 2), std::out_of_range);
}

TEST(DenseIsingModelTest, CanonicalizesSymmetryWithinTolerance) {
    const DenseIsingModel model({0.0, 1.0, 1.0 + 5e-13, 0.0}, 2);
    EXPECT_DOUBLE_EQ(model.matrix()[1], model.matrix()[2]);
    EXPECT_NEAR(model.Energy({1, 1}), -(2.0 + 5e-13), 1e-15);

    const std::vector<std::int8_t> spins{1, -1};
    const double original_energy = model.Energy(spins);
    auto flipped = spins;
    flipped[0] = -flipped[0];
    EXPECT_NEAR(model.FlipEnergyDelta(spins, 0), model.Energy(flipped) - original_energy, 1e-15);
}

TEST(ExactIsingSolverTest, SolvesFerromagneticAndAntiferromagneticPairs) {
    const DenseIsingModel ferromagnetic({0.0, 2.0, 2.0, 0.0}, 2);
    const IsingSample ferro_result = SolveIsingExact(ferromagnetic);
    EXPECT_DOUBLE_EQ(ferro_result.energy, -4.0);
    EXPECT_EQ(ferro_result.spins[0], ferro_result.spins[1]);

    const DenseIsingModel antiferromagnetic({0.0, -3.0, -3.0, 0.0}, 2);
    const IsingSample antiferro_result = SolveIsingExact(antiferromagnetic);
    EXPECT_DOUBLE_EQ(antiferro_result.energy, -6.0);
    EXPECT_EQ(antiferro_result.spins[0], -antiferro_result.spins[1]);
}

TEST(ExactIsingSolverTest, GrayCodeMatchesIndependentEnumeration) {
    const DenseIsingModel model(
        {
            0.0, 0.3,  -1.2, 0.7,  -0.4, 0.3, 0.0, 0.8,  -0.2, 1.1,  -1.2, 0.8, 0.0,
            0.9, -0.6, 0.7,  -0.2, 0.9,  0.0, 0.5, -0.4, 1.1,  -0.6, 0.5,  0.0,
        },
        5);
    const IsingSample exact = SolveIsingExact(model);
    const IsingSample brute_force = BruteForce(model);
    EXPECT_NEAR(exact.energy, brute_force.energy, 1e-12);
    EXPECT_NEAR(model.Energy(exact.spins), exact.energy, 1e-12);
}

TEST(ExactIsingSolverTest, HandlesEmptyModelAndHonorsLimit) {
    const DenseIsingModel empty({}, 0);
    const IsingSample empty_result = SolveIsingExact(empty);
    EXPECT_TRUE(empty_result.spins.empty());
    EXPECT_DOUBLE_EQ(empty_result.energy, 0.0);

    const DenseIsingModel three_spins(std::vector<double>(9, 0.0), 3);
    EXPECT_THROW(SolveIsingExact(three_spins, 2), std::invalid_argument);
}

TEST(SimulatedAnnealingTest, ReachesKnownDenseGroundState) {
    const DenseIsingModel model = MakeDenseFerromagnet(6);
    const IsingSample exact = SolveIsingExact(model);
    IsingSolverOptions options;
    options.seed = 982451653;
    options.num_reads = 16;
    options.sweeps = 80;
    options.beta_start = 0.05;
    options.beta_end = 8.0;

    const auto result = SolveIsingSimulatedAnnealing(model, options);
    ASSERT_EQ(result.samples.size(), options.num_reads);
    EXPECT_NEAR(result.samples.front().energy, exact.energy, 1e-12);
    for (const auto& sample : result.samples) {
        EXPECT_NEAR(model.Energy(sample.spins), sample.energy, 1e-12);
    }
    EXPECT_TRUE(
        std::is_sorted(result.samples.begin(), result.samples.end(),
                       [](const IsingSample& left, const IsingSample& right) { return left.energy < right.energy; }));
}

TEST(SimulatedAnnealingTest, IsReproducibleForFixedSeed) {
    const DenseIsingModel model(
        {
            0.0,
            0.9,
            -0.2,
            0.4,
            0.9,
            0.0,
            0.8,
            -1.1,
            -0.2,
            0.8,
            0.0,
            0.3,
            0.4,
            -1.1,
            0.3,
            0.0,
        },
        4);
    IsingSolverOptions options;
    options.seed = 123456789;
    options.num_reads = 12;
    options.sweeps = 40;
    options.beta_start = 0.1;
    options.beta_end = 4.0;

    ExpectSameResult(SolveIsingSimulatedAnnealing(model, options), SolveIsingSimulatedAnnealing(model, options));
}

// On a flat landscape every proposed flip is accepted, so a read is a pure function of its random
// stream and two reads are equal only when their streams are.  Forty spins make a chance match
// (probability 2^-40 per pair) irrelevant.
std::size_t CountSharedReads(const quspect::native::optimization::IsingSolverResult& left,
                             const quspect::native::optimization::IsingSolverResult& right) {
    std::size_t shared = 0;
    for (const auto& left_sample : left.samples) {
        for (const auto& right_sample : right.samples) {
            if (left_sample.spins == right_sample.spins) {
                ++shared;
            }
        }
    }
    return shared;
}

TEST(SimulatedAnnealingTest, AdjacentSeedsUseUnrelatedReadStreams) {
    constexpr std::size_t kNumSpins = 40;
    const DenseIsingModel flat(std::vector<double>(kNumSpins * kNumSpins, 0.0), kNumSpins);
    IsingSolverOptions options;
    options.num_reads = 32;
    options.sweeps = 3;

    for (const std::uint64_t base_seed : {std::uint64_t{0}, std::uint64_t{7}, std::uint64_t{982451653},
                                          std::numeric_limits<std::uint64_t>::max() - 40}) {
        options.seed = base_seed;
        const auto base = SolveIsingSimulatedAnnealing(flat, options);
        // Reads of one run are distinct: only the diagonal pairs match.
        EXPECT_EQ(CountSharedReads(base, base), options.num_reads) << "seed " << base_seed;
        for (std::uint64_t offset = 1; offset <= 40; ++offset) {
            options.seed = base_seed + offset;
            EXPECT_EQ(CountSharedReads(base, SolveIsingSimulatedAnnealing(flat, options)), 0U)
                << "seeds " << base_seed << " and " << base_seed + offset;
        }
    }
}

TEST(SimulatedAnnealingTest, ReadOfOneSeedIsNotFirstReadOfAnotherSeed) {
    constexpr std::size_t kNumSpins = 40;
    const DenseIsingModel flat(std::vector<double>(kNumSpins * kNumSpins, 0.0), kNumSpins);
    IsingSolverOptions options;
    options.seed = 123456789;
    options.num_reads = 8;
    options.sweeps = 3;
    const auto many = SolveIsingSimulatedAnnealing(flat, options);

    options.num_reads = 1;
    for (std::size_t read_index = 1; read_index < many.samples.size(); ++read_index) {
        options.seed = 123456789 + read_index;
        const auto single = SolveIsingSimulatedAnnealing(flat, options);
        EXPECT_NE(single.samples.front().spins, many.samples[read_index].spins) << "read " << read_index;
    }
}

TEST(SimulatedAnnealingTest, IsIndependentOfOpenMpThreadCount) {
#ifdef _OPENMP
    const DenseIsingModel model = MakeDenseFerromagnet(8);
    IsingSolverOptions options;
    options.seed = 777;
    options.num_reads = 24;
    options.sweeps = 30;
    options.beta_start = 0.1;
    options.beta_end = 4.0;

    omp_set_num_threads(1);
    const auto serial_result = SolveIsingSimulatedAnnealing(model, options);
    omp_set_num_threads(std::min(4, omp_get_num_procs()));
    const auto parallel_result = SolveIsingSimulatedAnnealing(model, options);
    ExpectSameResult(serial_result, parallel_result);
#else
    GTEST_SKIP() << "OpenMP is not available in this build";
#endif
}

TEST(SimulatedAnnealingTest, RejectsInvalidOptions) {
    const DenseIsingModel model({0.0}, 1);
    IsingSolverOptions options;

    options.num_reads = 0;
    EXPECT_THROW(SolveIsingSimulatedAnnealing(model, options), std::invalid_argument);
    options.num_reads = 1;
    options.sweeps = 0;
    EXPECT_THROW(SolveIsingSimulatedAnnealing(model, options), std::invalid_argument);
    options.sweeps = 1;
    options.beta_start = 0.0;
    EXPECT_THROW(SolveIsingSimulatedAnnealing(model, options), std::invalid_argument);
    options.beta_start = 2.0;
    options.beta_end = 1.0;
    EXPECT_THROW(SolveIsingSimulatedAnnealing(model, options), std::invalid_argument);
    options.beta_start = 0.1;
    options.beta_end = std::numeric_limits<double>::infinity();
    EXPECT_THROW(SolveIsingSimulatedAnnealing(model, options), std::invalid_argument);
}
