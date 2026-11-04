"""QuSpect Hamiltonian simulation — product formulas compiled for a target.

:func:`compile_evolution` turns ``exp(-i H t)`` for a Pauli sum ``H`` into a
circuit for a :class:`~quspect.transpiler.CompilationTarget`: it fixes a
product formula (order, steps, term order; optionally the fewest steps that
meet an error target) and compiles its rotations exactly, by Pauli-network
synthesis on the coupling graph (:func:`synthesize_network`) or by the
textbook ladder circuit through :func:`~quspect.transpiler.transpile`,
keeping the cheaper one.

Example:
    >>> from quspect.hamsim import compile_evolution
    >>> from quspect.transpiler import CompilationTarget
    >>> h = [("XX", (0, 1), 1.0), ("YY", (0, 1), 1.0), ("ZZ", (1, 2), 1.0)]
    >>> line = CompilationTarget.from_coupling_map(
    ...     3, [(0, 1), (1, 2)], native_gates={"cz", "h", "rz", "s", "sdg", "x"}
    ... )
    >>> res = compile_evolution(h, 1.0, line, steps=2, order=2)
    >>> res.steps
    2
"""

from quspect.hamsim._compile import EvolutionResult, compile_evolution
from quspect.hamsim._formulas import (
    Rotation,
    commutator_bound,
    one_step,
    order_terms,
    product_formula,
    state_error,
)
from quspect.hamsim._network import (
    NetworkCircuit,
    connected_active_set,
    ladder_circuit,
    merge_commuting,
    synthesize_network,
)
from quspect.hamsim._terms import PauliTerm, pauli_terms

__all__ = [
    "EvolutionResult",
    "NetworkCircuit",
    "PauliTerm",
    "Rotation",
    "commutator_bound",
    "compile_evolution",
    "connected_active_set",
    "ladder_circuit",
    "merge_commuting",
    "one_step",
    "order_terms",
    "pauli_terms",
    "product_formula",
    "state_error",
    "synthesize_network",
]
