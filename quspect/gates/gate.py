"""Core gate data model for QuSpect.

This module defines the :class:`Gate` dataclass — a lightweight,
immutable value object representing a quantum gate's metadata.
Matrix computation is handled separately in :mod:`quspect.gates._matrix_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any


@dataclass(frozen=True, kw_only=True)
class Gate:
    """A quantum gate definition — lightweight, hashable, immutable.

    ``Gate`` stores only metadata: name, qubit count, parameters, and
    an optional human-readable label. It does **not** carry its own
    unitary matrix. Use :func:`quspect.gates._matrix_utils.gate_matrix`
    to compute the matrix for a given gate.

    Attributes:
        name: Short canonical name, e.g. ``"h"``, ``"cx"``, ``"rx"``.
        num_qubits: Number of qubits this gate acts on.
        params: Gate parameters. Empty for fixed gates; one or more
            floats or symbolic Parameters for parameterized gates. Any
            real numeric value (``int``, ``numpy.float32``, ``numpy.int64``,
            ...) is normalized to a Python ``float`` on construction so
            that gates serialize and compare consistently.
        label: Optional human-readable label for display purposes.

    Example:
        >>> import numpy as np
        >>> Gate(name="rx", num_qubits=1, params=(np.float32(0.5),)).params
        (0.5,)
    """

    name: str
    num_qubits: int
    params: tuple[Any, ...] = field(default_factory=tuple)
    label: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"Gate num_qubits must be >= 1, got {self.num_qubits}")
        # Normalize real numeric scalars (including numpy scalars) to Python
        # floats. Symbolic parameters are kept unchanged.
        normalized = tuple(
            float(param) if isinstance(param, Real) else param for param in self.params
        )
        object.__setattr__(self, "params", normalized)

    def __repr__(self) -> str:
        label_part = f", label={self.label!r}" if self.label else ""
        params_part = f", params={self.params!r}" if self.params else ""
        return (
            f"Gate(name={self.name!r}, num_qubits={self.num_qubits}"
            f"{params_part}{label_part})"
        )

    def with_params(self, *params: float) -> Gate:
        """Return a new ``Gate`` with the given parameters.

        This is the primary way to create a parameterized gate instance
        from a gate template.

        Args:
            *params: Numeric parameter values.

        Returns:
            A new ``Gate`` instance with ``params`` set.

        Raises:
            ValueError: If ``params`` is empty.
        """
        if not params:
            raise ValueError("with_params() requires at least one parameter")
        return Gate(
            name=self.name,
            num_qubits=self.num_qubits,
            params=params,
            label=self.label,
        )
