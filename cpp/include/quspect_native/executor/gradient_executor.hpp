#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "quspect_native/executor/executor.hpp"

namespace quspect::native::runtime {
struct QSALProgram;
}  // namespace quspect::native::runtime

namespace quspect::native::executor {

/// Exact expectation values and adjoint gradients for a parameterized circuit.
///
/// ``gradients[observable][parameter]`` follows the stable orders stored in
/// ``observable_labels`` and ``parameter_names`` respectively.
struct GradientResult {
    std::vector<std::string> parameter_names;
    std::vector<std::string> observable_labels;
    std::vector<double> expectations;
    std::vector<std::vector<double>> gradients;
};

/// A reusable, immutable adjoint-gradient execution plan.
///
/// Construction copies and validates the program, compiles named gates into a
/// static tape, resolves affine-expression terms to parameter-column indices,
/// and groups Pauli terms while precomputing their masks.  ``Run`` owns no
/// mutable shared state, so concurrent calls on the same prepared object are
/// safe.
class PreparedAdjointGradient {
 public:
    explicit PreparedAdjointGradient(const runtime::QSALProgram& program);
    ~PreparedAdjointGradient();

    PreparedAdjointGradient(const PreparedAdjointGradient&) noexcept = default;
    PreparedAdjointGradient& operator=(const PreparedAdjointGradient&) noexcept = default;
    PreparedAdjointGradient(PreparedAdjointGradient&&) noexcept = default;
    PreparedAdjointGradient& operator=(PreparedAdjointGradient&&) noexcept = default;

    GradientResult Run(const std::unordered_map<std::string, double>& parameter_values,
                       PrecisionKind precision = PrecisionKind::kFloat64) const;

    const std::vector<std::string>& parameter_names() const noexcept;
    const std::vector<std::string>& observable_labels() const noexcept;

 private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

/// Exact state-vector adjoint differentiation.
///
/// The first implementation intentionally supports only reversible named
/// gates and Hermitian Pauli-sum observables.  It performs one forward circuit
/// traversal and one reverse traversal, without storing a state per gate.
class AdjointGradientExecutor {
 public:
    GradientResult Run(const runtime::QSALProgram& program,
                       const std::unordered_map<std::string, double>& parameter_values,
                       PrecisionKind precision = PrecisionKind::kFloat64) const;
};

}  // namespace quspect::native::executor
