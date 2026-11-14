"""Registry of gates defined outside QuSpect's built-in gate set.

A registered gate is a fixed (parameter-free) gate defined by its unitary
and by an exact decomposition into built-in gates.  Registration makes the
gate a first-class gate of the SDK:

* :func:`~quspect.gates.gate_matrix` returns its matrix (so unitaries,
  equivalence checks and the density-matrix simulator know it);
* the decomposition rules of :mod:`quspect.ir.compilation` lower it to its
  decomposition, so :class:`~quspect.transpiler.DecomposePass` and the
  OpenQASM 2.0 exporter can express it in other gates;
* the state-vector simulators replace it by its decomposition before they
  hand a circuit to the native core, which knows only built-in gates.

:func:`~quspect.synthesis.register_entangler` registers every new
entangler here, with the decomposition given by its CZ dressing.  QSAL
documents are unchanged: a registered gate is written under its own name.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from quspect.gates.gate import Gate

#: Largest ``|matrix - phase · product|`` entry accepted for a decomposition.
DECOMPOSITION_ATOL = 1e-9


@dataclass(frozen=True)
class GateDefinition:
    """A registered gate: its unitary and its decomposition.

    Attributes:
        name: Gate name.
        num_qubits: Number of qubits the gate acts on.
        matrix: Its ``2^n × 2^n`` unitary, qubit 0 the most significant (the
            convention of :func:`~quspect.gates.gate_matrix`).  Read-only.
        decomposition: ``(gate, qubits)`` pairs in time order, ``qubits``
            relative to the registered gate's own qubits.  Every gate is a
            built-in gate with numeric parameters, and the product equals
            ``matrix`` up to a global phase.

    Example:
        >>> definition = gate_definition("my_cx")  # see register_gate
        >>> [(gate.name, qubits) for gate, qubits in definition.lower((3, 5))]
        [('h', (5,)), ('cz', (3, 5)), ('h', (5,))]
    """

    name: str
    num_qubits: int
    matrix: np.ndarray
    decomposition: tuple[tuple[Gate, tuple[int, ...]], ...]

    @property
    def gate(self) -> Gate:
        """The gate to append to circuits: ``Gate(name, num_qubits)``."""
        return Gate(name=self.name, num_qubits=self.num_qubits)

    def lower(self, qubits: Sequence[int]) -> list[tuple[Gate, tuple[int, ...]]]:
        """The decomposition on absolute ``qubits`` (the gate's qubits in order).

        Args:
            qubits: The qubits the registered gate is applied to.

        Returns:
            ``(gate, qubits)`` pairs in time order.

        Raises:
            ValueError: If ``qubits`` does not hold one qubit per gate qubit.
        """
        if len(qubits) != self.num_qubits:
            raise ValueError(
                f"Gate {self.name!r} acts on {self.num_qubits} qubit(s), "
                f"got {len(qubits)}."
            )
        return [
            (gate, tuple(qubits[index] for index in relative))
            for gate, relative in self.decomposition
        ]


_REGISTRY: dict[str, GateDefinition] = {}


def register_gate(
    name: str,
    matrix: Any,
    decomposition: Iterable[tuple[Gate, Sequence[int]]],
) -> Gate:
    """Register a fixed gate by its unitary and a decomposition into built-ins.

    Registering a name again replaces its definition.

    Args:
        name: Gate name; must not be a built-in gate or instruction name.
        matrix: The gate's ``2^n × 2^n`` unitary with qubit 0 the most
            significant, as :func:`~quspect.gates.gate_matrix` returns it.
        decomposition: ``(gate, qubits)`` pairs in time order whose product is
            ``matrix`` up to a global phase.  Each gate is a built-in gate
            with numeric parameters; ``qubits`` are relative to the
            registered gate (``0`` is its first qubit).

    Returns:
        The gate to append to circuits, ``Gate(name=name, num_qubits=n)``.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty or built-in, ``matrix`` is not a
            unitary on whole qubits, or the decomposition is malformed or
            does not implement ``matrix``.

    Example:
        >>> from quspect.gates import gate_matrix
        >>> from quspect.gates.standard import CX, CZ, H
        >>> # CX = H(1) · CZ · H(1), under a name of its own.
        >>> steps = [(H, (1,)), (CZ, (0, 1)), (H, (1,))]
        >>> register_gate("my_cx", gate_matrix(CX), steps)
        Gate(name='my_cx', num_qubits=2)
    """
    if not isinstance(name, str):
        raise TypeError(f"Gate name must be a string, got {name!r}.")
    if not name:
        raise ValueError("Gate name must not be empty.")
    if name in _builtin_names():
        raise ValueError(
            f"{name!r} is a built-in gate or instruction name and cannot be registered."
        )
    unitary = _checked_unitary(name, matrix)
    num_qubits = unitary.shape[0].bit_length() - 1
    steps = _checked_decomposition(name, num_qubits, decomposition)
    product = _product(steps, num_qubits)
    if not _equal_up_to_phase(unitary, product):
        raise ValueError(
            f"The decomposition of {name!r} does not implement its matrix "
            f"(up to a global phase, to {DECOMPOSITION_ATOL})."
        )
    unitary.setflags(write=False)
    _REGISTRY[name] = GateDefinition(name, num_qubits, unitary, steps)
    return Gate(name=name, num_qubits=num_qubits)


def gate_definition(name: str) -> GateDefinition | None:
    """Return the :class:`GateDefinition` registered as ``name``, if any.

    Args:
        name: Gate name.

    Returns:
        The definition, or ``None`` for a name that is not registered
        (built-in gates are never registered).

    Example:
        >>> gate_definition("h") is None
        True
    """
    return _REGISTRY.get(name)


def _builtin_names() -> frozenset[str]:
    from quspect.gates.parameterized import ALL_PARAMETERIZED
    from quspect.gates.standard import ALL_STANDARD

    names = {gate.name for gate in (*ALL_STANDARD, *ALL_PARAMETERIZED)}
    return frozenset(names | {"measure", "barrier", "reset"})


def _checked_unitary(name: str, matrix: Any) -> np.ndarray:
    try:
        unitary = np.array(matrix, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The matrix of {name!r} is not numeric: {exc}") from exc
    dim = unitary.shape[0] if unitary.ndim == 2 else 0
    if (
        unitary.ndim != 2
        or unitary.shape != (dim, dim)
        or dim < 2
        or dim & (dim - 1)
        or not np.all(np.isfinite(unitary))
    ):
        raise ValueError(
            f"The matrix of {name!r} must be a finite 2^n × 2^n array (n >= 1), "
            f"got shape {unitary.shape}."
        )
    if not np.allclose(
        unitary.conj().T @ unitary, np.eye(dim), atol=DECOMPOSITION_ATOL
    ):
        raise ValueError(f"The matrix of {name!r} is not unitary.")
    return unitary


def _checked_decomposition(
    name: str, num_qubits: int, decomposition: Iterable[tuple[Gate, Sequence[int]]]
) -> tuple[tuple[Gate, tuple[int, ...]], ...]:
    from quspect.gates._matrix_utils import gate_matrix

    builtin = _builtin_names()
    steps: list[tuple[Gate, tuple[int, ...]]] = []
    for item in decomposition:
        try:
            gate, qubits = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The decomposition of {name!r} must hold (gate, qubits) pairs, "
                f"got {item!r}."
            ) from exc
        if not isinstance(gate, Gate) or gate.name not in builtin:
            raise ValueError(
                f"The decomposition of {name!r} may only use built-in gates, "
                f"got {gate!r}."
            )
        relative = tuple(qubits)
        if (
            len(relative) != gate.num_qubits
            or len(set(relative)) != len(relative)
            or not all(
                isinstance(q, (int, np.integer))
                and not isinstance(q, bool)
                and 0 <= q < num_qubits
                for q in relative
            )
        ):
            raise ValueError(
                f"The decomposition of {name!r} applies {gate.name!r} to "
                f"{relative!r}; expected {gate.num_qubits} distinct qubit(s) in "
                f"range({num_qubits})."
            )
        try:
            gate_matrix(gate)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"The decomposition of {name!r} uses {gate!r}, which has no "
                f"numeric matrix: {exc}"
            ) from exc
        steps.append((gate, tuple(int(q) for q in relative)))
    return tuple(steps)


def _product(
    steps: Sequence[tuple[Gate, tuple[int, ...]]], num_qubits: int
) -> np.ndarray:
    """Unitary of ``steps`` on ``num_qubits`` qubits (qubit 0 most significant)."""
    from quspect.gates._matrix_utils import gate_matrix

    dim = 1 << num_qubits
    total = np.eye(dim, dtype=complex)
    for gate, qubits in steps:
        k = len(qubits)
        factor = gate_matrix(gate).reshape([2] * (2 * k))
        tensor = total.reshape([2] * num_qubits + [dim])
        tensor = np.tensordot(factor, tensor, axes=(list(range(k, 2 * k)), qubits))
        total = np.moveaxis(tensor, list(range(k)), list(qubits)).reshape(dim, dim)
    return total


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    overlap = np.vdot(b, a)
    if abs(overlap) < DECOMPOSITION_ATOL:
        return False
    phase = overlap / abs(overlap)
    return bool(np.allclose(a, phase * b, atol=DECOMPOSITION_ATOL, rtol=0.0))
