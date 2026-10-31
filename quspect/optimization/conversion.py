"""Exact conversions between QUBO, binary quadratic, and Ising models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from quspect.optimization.model import (
    BinaryQuadraticModel,
    FloatArray,
    IsingProblem,
    OptimizationSense,
    VariableType,
)


def binary_to_ising(model: BinaryQuadraticModel) -> IsingProblem:
    """Convert a binary-domain quadratic model with ``s = 2x - 1``.

    The transformation is energy preserving, including the constant offset
    and optimization direction.

    Args:
        model: Binary-domain quadratic model.

    Returns:
        Equivalent logical Ising problem.

    Raises:
        ValueError: If ``model`` is not binary-domain.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import (
        ...     BinaryQuadraticModel,
        ...     VariableType,
        ...     binary_to_ising,
        ... )
        >>> bqm = BinaryQuadraticModel(
        ...     ("x",),
        ...     np.array([2.0]),
        ...     np.zeros((1, 1)),
        ...     variable_type=VariableType.BINARY,
        ... )
        >>> binary_to_ising(bqm).evaluate([-1])
        0.0
    """
    if model.variable_type is not VariableType.BINARY:
        raise ValueError("binary_to_ising requires a binary-domain model.")
    pair_sums = np.sum(model.quadratic, axis=1)
    linear = 0.5 * model.linear + 0.25 * pair_sums
    quadratic = 0.25 * model.quadratic
    offset = (
        model.offset
        + 0.5 * float(np.sum(model.linear))
        + 0.25 * float(np.sum(np.triu(model.quadratic, k=1)))
    )
    return IsingProblem(
        variables=model.variables,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        sense=model.sense,
    )


def ising_to_binary(problem: IsingProblem) -> BinaryQuadraticModel:
    """Convert an Ising problem to binary variables with ``s = 2x - 1``.

    Args:
        problem: Logical Ising problem.

    Returns:
        Energy-equivalent binary-domain quadratic model.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import IsingProblem, ising_to_binary
        >>> problem = IsingProblem(
        ...     ("s",), np.array([1.0]), np.zeros((1, 1))
        ... )
        >>> ising_to_binary(problem).evaluate([0])
        -1.0
    """
    pair_sums = np.sum(problem.quadratic, axis=1)
    linear = 2.0 * problem.linear - 2.0 * pair_sums
    quadratic = 4.0 * problem.quadratic
    offset = (
        problem.offset
        - float(np.sum(problem.linear))
        + float(np.sum(np.triu(problem.quadratic, k=1)))
    )
    return BinaryQuadraticModel(
        variables=problem.variables,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        variable_type=VariableType.BINARY,
        sense=problem.sense,
    )


def bqm_to_ising(model: BinaryQuadraticModel) -> IsingProblem:
    """Return an Ising view of a binary or spin quadratic model.

    Args:
        model: Binary- or spin-domain quadratic model.

    Returns:
        Equivalent logical Ising problem.
    """
    if model.variable_type is VariableType.BINARY:
        return binary_to_ising(model)
    return IsingProblem(
        variables=model.variables,
        linear=model.linear,
        quadratic=model.quadratic,
        offset=model.offset,
        sense=model.sense,
    )


def binary_samples_to_spins(samples: object) -> NDArray[np.int8]:
    """Convert one or more binary samples with ``s = 2x - 1``.

    The input shape is retained and the returned array is read-only.

    Args:
        samples: One- or two-dimensional array containing only ``0`` and ``1``.

    Returns:
        Spin samples containing only ``-1`` and ``+1``.

    Raises:
        ValueError: If the input is not one- or two-dimensional or is outside
            the binary domain.
    """
    binary = np.asarray(samples)
    if binary.ndim not in (1, 2):
        raise ValueError("Binary samples must be a one- or two-dimensional array.")
    if not bool(np.all(np.logical_or(binary == 0, binary == 1))):
        raise ValueError("Binary samples must contain only 0 and 1.")
    spins = np.asarray(2 * binary.astype(np.int8) - 1, dtype=np.int8)
    spins.setflags(write=False)
    return spins


def spin_samples_to_binary(samples: object) -> NDArray[np.int8]:
    """Convert one or more spin samples with ``x = (s + 1) / 2``.

    The input shape is retained and the returned array is read-only.

    Args:
        samples: One- or two-dimensional array containing only ``-1`` and
            ``+1``.

    Returns:
        Binary samples containing only ``0`` and ``1``.

    Raises:
        ValueError: If the input is not one- or two-dimensional or is outside
            the spin domain.
    """
    spins = np.asarray(samples)
    if spins.ndim not in (1, 2):
        raise ValueError("Spin samples must be a one- or two-dimensional array.")
    if not bool(np.all(np.logical_or(spins == -1, spins == 1))):
        raise ValueError("Spin samples must contain only -1 and +1.")
    binary = np.asarray((spins.astype(np.int8) + 1) // 2, dtype=np.int8)
    binary.setflags(write=False)
    return binary


def bqm_from_qubo_matrix(
    matrix: object,
    *,
    variables: Sequence[str] | None = None,
    offset: float = 0.0,
    sense: OptimizationSense = OptimizationSense.MINIMIZE,
) -> BinaryQuadraticModel:
    """Build a binary quadratic model from ``x.T @ Q @ x + offset``.

    Both upper-triangular and symmetric QUBO matrices are accepted.  Opposing
    off-diagonal entries are added into one canonical pair coefficient.

    Args:
        matrix: Finite square QUBO matrix.
        variables: Optional variable names. Defaults to ``x[0]``, ``x[1]``, ...
        offset: Constant QUBO term.
        sense: Optimization direction.

    Returns:
        Canonical binary quadratic model.

    Raises:
        ValueError: If the matrix is not finite and square or names mismatch.

    Example:
        >>> import numpy as np
        >>> from quspect.optimization import bqm_from_qubo_matrix
        >>> model = bqm_from_qubo_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))
        >>> model.evaluate([1, 1])
        6.0
    """
    qubo = np.asarray(matrix, dtype=np.float64)
    if qubo.ndim != 2 or qubo.shape[0] != qubo.shape[1]:
        raise ValueError(f"QUBO matrix must be square, got {qubo.shape}.")
    if not np.all(np.isfinite(qubo)):
        raise ValueError("QUBO matrix must contain only finite values.")
    size = int(qubo.shape[0])
    names = (
        tuple(variables)
        if variables is not None
        else tuple(f"x[{index}]" for index in range(size))
    )
    if len(names) != size:
        raise ValueError(f"variables must contain {size} names, got {len(names)}.")
    linear = np.diag(qubo).copy()
    quadratic = qubo + qubo.T
    np.fill_diagonal(quadratic, 0.0)
    return BinaryQuadraticModel(
        variables=names,
        linear=linear,
        quadratic=quadratic,
        offset=offset,
        variable_type=VariableType.BINARY,
        sense=sense,
    )


def qubo_matrix_from_bqm(model: BinaryQuadraticModel) -> FloatArray:
    """Return a canonical upper-triangular QUBO matrix.

    The result satisfies ``model.evaluate(x) == x.T @ Q @ x + offset``.
    The constant offset remains available on ``model.offset`` and is not
    embedded in the returned matrix.

    Args:
        model: Binary-domain quadratic model.

    Returns:
        Read-only upper-triangular QUBO matrix.

    Raises:
        ValueError: If the model uses spin variables.
    """
    if model.variable_type is not VariableType.BINARY:
        raise ValueError("qubo_matrix_from_bqm requires a binary-domain model.")
    matrix = np.zeros(
        (model.num_variables, model.num_variables),
        dtype=np.float64,
    )
    np.fill_diagonal(matrix, model.linear)
    upper_indices = np.triu_indices(model.num_variables, k=1)
    matrix[upper_indices] = model.quadratic[upper_indices]
    matrix.setflags(write=False)
    return cast(NDArray[np.float64], matrix)
