"""Quadratic optimization intermediate representations and conversions."""

from quspect.optimization.codec import encode_ising_matrix_csv
from quspect.optimization.compiler import (
    CompilationResult,
    CompiledQuadraticProgram,
    compile_program,
    compile_quadratic_program,
)
from quspect.optimization.conversion import (
    binary_samples_to_spins,
    binary_to_ising,
    bqm_from_qubo_matrix,
    bqm_to_ising,
    ising_to_binary,
    qubo_matrix_from_bqm,
    spin_samples_to_binary,
)
from quspect.optimization.expression import (
    BinaryVariable,
    QuadraticExpression,
    as_expression,
    binary_array,
)
from quspect.optimization.hardware import (
    HardwareIsingProgram,
    compile_hardware_ising,
)
from quspect.optimization.isingq import (
    IsingQClient,
    IsingQError,
    IsingQProgram,
    compile_isingq,
)
from quspect.optimization.model import (
    BinaryQuadraticModel,
    IsingProblem,
    OptimizationSense,
    VariableType,
)
from quspect.optimization.native_solver import (
    NativeIsingResult,
    NativeIsingSolver,
    NativeIsingSolverError,
)
from quspect.optimization.program import (
    LinearEqualityConstraint,
    NonnegativeQuadraticZeroConstraint,
    QuadraticObjective,
    QuadraticProgram,
)

__all__ = [
    "IsingQClient",
    "IsingQError",
    "IsingQProgram",
    "compile_isingq",
    "BinaryVariable",
    "QuadraticExpression",
    "as_expression",
    "binary_array",
    "QuadraticObjective",
    "LinearEqualityConstraint",
    "NonnegativeQuadraticZeroConstraint",
    "QuadraticProgram",
    "CompilationResult",
    "CompiledQuadraticProgram",
    "compile_program",
    "compile_quadratic_program",
    "VariableType",
    "OptimizationSense",
    "BinaryQuadraticModel",
    "IsingProblem",
    "HardwareIsingProgram",
    "binary_samples_to_spins",
    "spin_samples_to_binary",
    "binary_to_ising",
    "ising_to_binary",
    "bqm_to_ising",
    "bqm_from_qubo_matrix",
    "qubo_matrix_from_bqm",
    "compile_hardware_ising",
    "encode_ising_matrix_csv",
    "NativeIsingResult",
    "NativeIsingSolver",
    "NativeIsingSolverError",
]
