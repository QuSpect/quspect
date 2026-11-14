"""Gate compilation and decomposition for IR export.

This module provides a rule-based framework for decomposing quantum
gates into simpler primitives that are supported by target formats
(e.g., OpenQASM 2.0 ``qelib1.inc``).

Built-in decompositions cover common cases (SWAP → 3× CX, CCX →
CX + single-qubit gates).  Users can register custom decomposition
rules for application-specific gates.  A gate registered with
:func:`quspect.gates.register_gate` (or
:func:`quspect.synthesis.register_entangler`) is lowered to its registered
decomposition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.gates.gate import Gate


# =============================================================================
# Decomposition rule
# =============================================================================


@dataclass(frozen=True)
class GateDecompositionRule:
    """A rule for decomposing one gate into a sequence of simpler gates.

    Each rule describes how to replace a specific gate with an ordered
    sequence of ``(gate, qubit_indices)`` pairs that together implement
    the same unitary on the given qubits.

    The ``qubit_indices`` in the decomposition are **relative** to the
    original gate's qubits — ``0`` means the first qubit of the original,
    ``1`` means the second, and so on.

    Attributes:
        gate_name: Name of the gate this rule decomposes.
        num_qubits: Number of qubits this rule applies to.  Must match
            ``original_gate.num_qubits``.
        decompose: Callable taking ``(gate, qubits_tuple)`` and returning
            a list of ``(Gate, tuple[int, ...])`` pairs.
        condition: Optional predicate — decompose only if this returns
            ``True`` when called with the original gate.
        priority: Higher-priority rules are tried first.
        description: Human-readable description of the decomposition.
    """

    gate_name: str
    num_qubits: int
    decompose: Callable[..., list[tuple[Gate, tuple[int, ...]]]]
    condition: Callable[[Gate], bool] | None = None
    priority: int = 0
    description: str = ""


# =============================================================================
# Built-in decompositions
# =============================================================================


def _decompose_swap(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose SWAP into three CX gates.

    SWAP(q0, q1) = CX(q0, q1); CX(q1, q0); CX(q0, q1)
    """
    from quspect.gates.standard import CX

    q0, q1 = qubits[0], qubits[1]
    return [
        (CX, (q0, q1)),
        (CX, (q1, q0)),
        (CX, (q0, q1)),
    ]


def _decompose_ccx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose Toffoli (CCX) using the standard Nielsen & Chuang circuit.

    This is the textbook decomposition using H, T, T†, and CX gates.
    Produces 6 CX gates + single-qubit gates.

    Reference:
        Nielsen & Chuang, *Quantum Computation and Quantum Information*,
        §4.3 (Figure 4.9).
    """
    from quspect.gates.standard import CX, TDG, H, T

    c1, c2, t = qubits[0], qubits[1], qubits[2]

    # The standard Toffoli decomposition (Nielsen & Chuang Fig 4.9):
    return [
        (H, (t,)),
        (CX, (c2, t)),
        (TDG, (t,)),
        (CX, (c1, t)),
        (T, (t,)),
        (CX, (c2, t)),
        (TDG, (t,)),
        (CX, (c1, t)),
        (T, (c2,)),
        (T, (t,)),
        (CX, (c1, c2)),
        (H, (t,)),
        (T, (c1,)),
        (TDG, (c2,)),
        (CX, (c1, c2)),
    ]


def _decompose_cx_to_cz(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CX into CZ + H gates.

    CX(c, t) = H(t); CZ(c, t); H(t)
    """
    from quspect.gates.standard import CZ, H

    c, t = qubits[0], qubits[1]
    return [
        (H, (t,)),
        (CZ, (c, t)),
        (H, (t,)),
    ]


def _decompose_cz_to_cx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CZ into CX + H gates (for CX-native platforms).

    CZ(a, b) = H(b); CX(a, b); H(b)
    """
    from quspect.gates.standard import CX, H

    a, b = qubits[0], qubits[1]
    return [
        (H, (b,)),
        (CX, (a, b)),
        (H, (b,)),
    ]


def _decompose_sx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose SX (or SXdg) into H + S (or Sdg) + H, exactly.

    SX = H; S; H and SXdg = H; Sdg; H (no global-phase difference).
    """
    from quspect.gates.standard import SDG, H, S

    q = qubits[0]
    middle = S if gate.name == "sx" else SDG
    return [(H, (q,)), (middle, (q,)), (H, (q,))]


def _decompose_rx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose RX(θ) into H + RZ(θ) + H.

    RX(θ) = H; RZ(θ); H
    """
    from quspect.gates.parameterized import RZ
    from quspect.gates.standard import H

    theta = _get_param(gate, 0)
    q = qubits[0]
    return [
        (H, (q,)),
        (RZ.with_params(theta), (q,)),
        (H, (q,)),
    ]


def _decompose_ry(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose RY(θ) into SDG + RX(θ) + S.

    RY(θ) = S†; RX(θ); S
    """
    from quspect.gates.parameterized import RX
    from quspect.gates.standard import SDG, S

    theta = _get_param(gate, 0)
    q = qubits[0]
    return [
        (SDG, (q,)),
        (RX.with_params(theta), (q,)),
        (S, (q,)),
    ]


def _decompose_p(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose P(θ) into RZ(θ).

    P(θ) = RZ(θ)  (same unitary up to global phase).
    """
    from quspect.gates.parameterized import RZ

    theta = _get_param(gate, 0)
    return [(RZ.with_params(theta), (qubits[0],))]


def _decompose_u1(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose U1(λ) into RZ(λ)."""
    from quspect.gates.parameterized import RZ

    lam = _get_param(gate, 0)
    return [(RZ.with_params(lam), (qubits[0],))]


def _decompose_u2(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose U2(φ, λ) = U3(π/2, φ, λ)."""
    from quspect.gates.parameterized import U3

    phi = _get_param(gate, 0)
    lam = _get_param(gate, 1)
    import numpy as np

    return [(U3.with_params(np.pi / 2, phi, lam), (qubits[0],))]


def _decompose_u3(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose U3(θ, φ, λ) into RZ + RY + RZ.

    Circuit order is RZ(λ); RY(θ); RZ(φ), equivalent to U3 up to
    a global phase.
    """
    from quspect.gates.parameterized import RY, RZ

    theta = _get_param(gate, 0)
    phi = _get_param(gate, 1)
    lam = _get_param(gate, 2)
    q = qubits[0]
    return [
        (RZ.with_params(lam), (q,)),
        (RY.with_params(theta), (q,)),
        (RZ.with_params(phi), (q,)),
    ]


def _decompose_crz(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CRZ(θ) into RZ + CX + RZ + CX.

    CRZ(θ) = [I ⊗ RZ(θ/2)]; CX; [I ⊗ RZ(−θ/2)]; CX
    """
    from quspect.gates.parameterized import RZ
    from quspect.gates.standard import CX

    theta = _get_param(gate, 0)
    half = theta / 2.0
    c, t = qubits[0], qubits[1]
    return [
        (RZ.with_params(half), (t,)),
        (CX, (c, t)),
        (RZ.with_params(-half), (t,)),
        (CX, (c, t)),
    ]


def _decompose_crx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CRX(θ) into H + CRZ(θ) + H on target.

    CRX(θ) = [I ⊗ H]; CRZ(θ); [I ⊗ H]
    """
    from quspect.gates.parameterized import CRZ
    from quspect.gates.standard import H

    theta = _get_param(gate, 0)
    c, t = qubits[0], qubits[1]
    return [
        (H, (t,)),
        (CRZ.with_params(theta), (c, t)),
        (H, (t,)),
    ]


def _decompose_cry(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CRY(θ) into SDG + CRX(θ) + S on target.

    CRY(θ) = [I ⊗ S†]; CRX(θ); [I ⊗ S]
    """
    from quspect.gates.parameterized import CRX
    from quspect.gates.standard import SDG, S

    theta = _get_param(gate, 0)
    c, t = qubits[0], qubits[1]
    return [
        (SDG, (t,)),
        (CRX.with_params(theta), (c, t)),
        (S, (t,)),
    ]


def _decompose_cp(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CP(θ) into P(θ/2) on control followed by CRZ(θ)."""
    from quspect.gates.parameterized import CRZ, P

    theta = _get_param(gate, 0)
    c, t = qubits[0], qubits[1]
    return [
        (P.with_params(theta / 2.0), (c,)),
        (CRZ.with_params(theta), (c, t)),
    ]


def _decompose_cu3(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose CU3(θ, φ, λ) into single-qubit gates + 2×CX.

    Standard decomposition from Nielsen & Chuang §4.3.
    """
    from quspect.gates.parameterized import U3, P
    from quspect.gates.standard import CX

    theta = _get_param(gate, 0)
    phi = _get_param(gate, 1)
    lam = _get_param(gate, 2)
    c, t = qubits[0], qubits[1]
    return [
        (P.with_params((lam + phi) / 2.0), (c,)),
        (P.with_params((lam - phi) / 2.0), (t,)),
        (CX, (c, t)),
        (U3.with_params(-theta / 2.0, 0.0, -(phi + lam) / 2.0), (t,)),
        (CX, (c, t)),
        (U3.with_params(theta / 2.0, phi, 0.0), (t,)),
    ]


def _decompose_rzz(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose RZZ(θ) into CX + RZ(θ) + CX, exactly (no global phase).

    RZZ(θ) = CX(a, b); [I ⊗ RZ(θ)]; CX(a, b) — the CX pair writes the
    parity ``a ⊕ b`` onto ``b`` for the RZ.  This is the ``qelib1.inc``
    definition of ``rzz``.
    """
    from quspect.gates.parameterized import RZ
    from quspect.gates.standard import CX

    theta = _get_param(gate, 0)
    a, b = qubits[0], qubits[1]
    return [
        (CX, (a, b)),
        (RZ.with_params(theta), (b,)),
        (CX, (a, b)),
    ]


def _decompose_rxx(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose RXX(θ) into H ⊗ H + RZZ(θ) + H ⊗ H, exactly.

    RXX(θ) = [H ⊗ H]; RZZ(θ); [H ⊗ H], since H Z H = X.
    """
    from quspect.gates.parameterized import RZZ
    from quspect.gates.standard import H

    theta = _get_param(gate, 0)
    a, b = qubits[0], qubits[1]
    return [
        (H, (a,)),
        (H, (b,)),
        (RZZ.with_params(theta), (a, b)),
        (H, (a,)),
        (H, (b,)),
    ]


def _decompose_ryy(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose RYY(θ) into RX(π/2) ⊗ RX(π/2) + RZZ(θ) + RX(-π/2) ⊗ RX(-π/2).

    RX(-π/2) Z RX(π/2) = Y, so the RX pair rotates the ZZ interaction into
    YY exactly (qiskit's definition of ``ryy``).
    """
    import math

    from quspect.gates.parameterized import RX, RZZ

    theta = _get_param(gate, 0)
    a, b = qubits[0], qubits[1]
    quarter = RX.with_params(math.pi / 2)
    back = RX.with_params(-math.pi / 2)
    return [
        (quarter, (a,)),
        (quarter, (b,)),
        (RZZ.with_params(theta), (a, b)),
        (back, (a,)),
        (back, (b,)),
    ]


def _get_param(gate: Gate, idx: int) -> Any:
    """Extract a numeric or symbolic parameter without discarding symbols."""
    return gate.params[idx] if gate.params else 0.0


# =============================================================================
# Rule registry
# =============================================================================

BUILTIN_DECOMPOSITIONS: list[GateDecompositionRule] = [
    GateDecompositionRule(
        gate_name="swap",
        num_qubits=2,
        decompose=_decompose_swap,
        priority=10,
        description="SWAP → 3× CX",
    ),
    GateDecompositionRule(
        gate_name="ccx",
        num_qubits=3,
        decompose=_decompose_ccx,
        priority=10,
        description="Toffoli → H + T + T† + CX",
    ),
    # Single-qubit → RZ basis
    GateDecompositionRule(
        gate_name="rx",
        num_qubits=1,
        decompose=_decompose_rx,
        priority=8,
        description="RX(θ) → H + RZ(θ) + H",
    ),
    GateDecompositionRule(
        gate_name="ry",
        num_qubits=1,
        decompose=_decompose_ry,
        priority=8,
        description="RY(θ) → SDG + RX(θ) + S",
    ),
    GateDecompositionRule(
        gate_name="p",
        num_qubits=1,
        decompose=_decompose_p,
        priority=8,
        description="P(θ) → RZ(θ)",
    ),
    GateDecompositionRule(
        gate_name="u1",
        num_qubits=1,
        decompose=_decompose_u1,
        priority=8,
        description="U1(λ) → RZ(λ)",
    ),
    GateDecompositionRule(
        gate_name="u2",
        num_qubits=1,
        decompose=_decompose_u2,
        priority=8,
        description="U2(φ,λ) → U3(π/2,φ,λ)",
    ),
    GateDecompositionRule(
        gate_name="u3",
        num_qubits=1,
        decompose=_decompose_u3,
        priority=8,
        description="U3(θ,φ,λ) → RZ + RY + RZ",
    ),
    # Two-qubit controlled rotations
    GateDecompositionRule(
        gate_name="crz",
        num_qubits=2,
        decompose=_decompose_crz,
        priority=8,
        description="CRZ(θ) → RZ + CX + RZ + CX",
    ),
    GateDecompositionRule(
        gate_name="crx",
        num_qubits=2,
        decompose=_decompose_crx,
        priority=8,
        description="CRX(θ) → H + CRZ(θ) + H",
    ),
    GateDecompositionRule(
        gate_name="cry",
        num_qubits=2,
        decompose=_decompose_cry,
        priority=8,
        description="CRY(θ) → SDG + CRX(θ) + S",
    ),
    GateDecompositionRule(
        gate_name="cp",
        num_qubits=2,
        decompose=_decompose_cp,
        priority=8,
        description="CP(θ) → P(θ/2) + CRZ(θ)",
    ),
    GateDecompositionRule(
        gate_name="cu3",
        num_qubits=2,
        decompose=_decompose_cu3,
        priority=8,
        description="CU3 → U3 + CX + U3 + CX + ...",
    ),
    # Two-qubit Pauli rotations
    GateDecompositionRule(
        gate_name="rzz",
        num_qubits=2,
        decompose=_decompose_rzz,
        priority=8,
        description="RZZ(θ) → CX + RZ(θ) + CX",
    ),
    GateDecompositionRule(
        gate_name="rxx",
        num_qubits=2,
        decompose=_decompose_rxx,
        priority=8,
        description="RXX(θ) → H⊗H + RZZ(θ) + H⊗H",
    ),
    GateDecompositionRule(
        gate_name="ryy",
        num_qubits=2,
        decompose=_decompose_ryy,
        priority=8,
        description="RYY(θ) → RX(π/2)⊗RX(π/2) + RZZ(θ) + RX(-π/2)⊗RX(-π/2)",
    ),
    # CX → CZ + H (for CZ-native platforms)
    GateDecompositionRule(
        gate_name="cx",
        num_qubits=2,
        decompose=_decompose_cx_to_cz,
        priority=5,
        description="CX → H + CZ + H",
    ),
    # CZ → CX + H (for CX-native platforms)
    GateDecompositionRule(
        gate_name="cz",
        num_qubits=2,
        decompose=_decompose_cz_to_cx,
        priority=5,
        description="CZ → H + CX + H",
    ),
    GateDecompositionRule(
        gate_name="sx",
        num_qubits=1,
        decompose=_decompose_sx,
        priority=8,
        description="SX → H + S + H",
    ),
    GateDecompositionRule(
        gate_name="sxdg",
        num_qubits=1,
        decompose=_decompose_sx,
        priority=8,
        description="SXdg → H + Sdg + H",
    ),
]
"""Built-in gate decomposition rules."""


# Global mutable registry for user-defined rules.
_EXTRA_RULES: list[GateDecompositionRule] = []


# =============================================================================
# Public API
# =============================================================================


def register_decomposition(rule: GateDecompositionRule) -> None:
    """Register a custom gate decomposition rule.

    Custom rules are tried **after** built-in rules with the same
    priority level.

    Args:
        rule: The decomposition rule to register.

    Example:
        >>> rule = GateDecompositionRule(
        ...     gate_name="my_gate",
        ...     num_qubits=2,
        ...     decompose=lambda g, qs: [(CX, qs[:2])],
        ... )
        >>> register_decomposition(rule)
    """
    _EXTRA_RULES.append(rule)


def clear_extra_rules() -> None:
    """Remove all user-registered decomposition rules.

    Useful in test cleanup.
    """
    _EXTRA_RULES.clear()


def get_rules_for(gate_name: str, num_qubits: int) -> list[GateDecompositionRule]:
    """Return all applicable rules for a given gate, sorted by priority.

    A gate registered with :func:`quspect.gates.register_gate` has one more
    rule, its registered decomposition, tried after the user-registered
    rules of the same priority.

    Args:
        gate_name: Name of the gate to decompose.
        num_qubits: Number of qubits the gate acts on.

    Returns:
        List of matching rules (highest priority first), or an empty
        list if no rule matches.
    """
    candidates = [
        r
        for r in BUILTIN_DECOMPOSITIONS + _EXTRA_RULES
        if r.gate_name == gate_name and r.num_qubits == num_qubits
    ]
    registered = _registered_rule(gate_name, num_qubits)
    if registered is not None:
        candidates.append(registered)
    candidates.sort(key=lambda r: -r.priority)
    return candidates


def _registered_rule(gate_name: str, num_qubits: int) -> GateDecompositionRule | None:
    """The rule lowering a registered gate to its registered decomposition."""
    from quspect.gates.registry import gate_definition

    definition = gate_definition(gate_name)
    if definition is None or definition.num_qubits != num_qubits:
        return None
    return GateDecompositionRule(
        gate_name=gate_name,
        num_qubits=num_qubits,
        decompose=lambda gate, qubits: definition.lower(qubits),
        condition=lambda gate: not gate.params,
        description=f"{gate_name} → its registered decomposition",
    )


def _lower_registered_gates(circuit: QuantumCircuit) -> QuantumCircuit:
    """Replace every registered gate of ``circuit`` by its decomposition.

    The decompositions hold built-in gates only, so the result is a circuit
    the native core and the OpenQASM 2.0 writer know.  A circuit without
    registered gates is returned itself (not a copy), so it is serialized
    and exported exactly as before.
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import _gate_instructions
    from quspect.gates.registry import _REGISTRY

    if not _REGISTRY:
        return circuit
    instructions = circuit.instructions
    if not any(
        instr.gate is not None and instr.is_gate and instr.gate.name in _REGISTRY
        for instr in instructions
    ):
        return circuit
    out: list[CircuitInstruction] = []
    for instr in instructions:
        gate = instr.gate
        definition = (
            _REGISTRY.get(gate.name) if gate is not None and instr.is_gate else None
        )
        if definition is None:
            out.append(instr)
            continue
        if gate is not None and gate.params:
            raise ValueError(
                f"Registered gate {gate.name!r} takes no parameters, got {gate!r}."
            )
        out.extend(_gate_instructions(definition.lower(instr.qubits), instr.label))
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)


def decompose_gate(
    gate: Gate, qubits: tuple[int, ...]
) -> list[tuple[Gate, tuple[int, ...]]]:
    """Decompose a single gate into simpler gates.

    Looks up the best-matching :class:`GateDecompositionRule` and applies
    it.  The returned ``(gate, qubits)`` pairs use **absolute** qubit
    indices (the rule's relative indices are remapped).

    Args:
        gate: The gate to decompose.
        qubits: The absolute qubit indices the gate was applied to.

    Returns:
        A list of ``(Gate, tuple[int, ...])`` pairs — the decomposed
        gate sequence.

    Raises:
        ValueError: If no decomposition rule matches the gate.
    """
    rules = get_rules_for(gate.name, gate.num_qubits)

    for rule in rules:
        if rule.condition is not None and not rule.condition(gate):
            continue
        # The decomposition returns (gate, relative_qubits) pairs.
        # Remap relative qubit indices to absolute ones.
        raw = rule.decompose(gate, qubits)
        # raw already uses absolute qubit indices from qubits parameter
        return raw

    raise ValueError(
        f"No decomposition rule found for gate {gate.name!r} "
        f"({gate.num_qubits} qubit(s))."
    )


def decompose_circuit(
    circuit: QuantumCircuit,
    *,
    supported_gates: set[str] | None = None,
) -> QuantumCircuit:
    """Walk a circuit and decompose unsupported gates in place of the
    originals.

    For each instruction that is a gate not in ``supported_gates``, look
    up a decomposition rule and insert the resulting gate sequence in its
    place.  Non-gate instructions (measure, barrier) pass through unchanged.

    Gate decomposition is **not recursive** — if a decomposition produces
    gates that are themselves unsupported, the caller should iterate.

    Args:
        circuit: The circuit to process.
        supported_gates: Set of gate names that do NOT need decomposition.
            If ``None``, uses the QASM 2.0 gate registry as the supported
            set.

    Returns:
        A new :class:`QuantumCircuit` with unsupported gates replaced.
    """
    from quspect.circuit._native import unsupported_gate_indices
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.ir.qasm2 import get_supported_gates

    if supported_gates is None:
        supported_gates = get_supported_gates()

    new_circuit = QuantumCircuit(name=circuit.name)
    out = new_circuit._instructions

    indices = (
        unsupported_gate_indices(circuit._instructions, supported_gates)
        if type(circuit) is QuantumCircuit
        else None
    )
    if indices is not None:
        # Only the listed instructions can need a decomposition; the runs
        # between them pass through unchanged.
        instrs = circuit._instructions
        start = 0
        for i in indices:
            out.extend(instrs[start:i])
            _decompose_instruction(instrs[i], supported_gates, out)
            start = i + 1
        out.extend(instrs[start:])
        return new_circuit

    for instr in circuit:
        _decompose_instruction(instr, supported_gates, out)
    return new_circuit


def _decompose_instruction(
    instr: CircuitInstruction, supported_gates: set[str], out: list[CircuitInstruction]
) -> None:
    """Append ``instr`` or its decomposition to ``out`` (see decompose_circuit)."""
    from quspect.circuit.instruction import InstructionType, _gate_instructions

    if instr.type != InstructionType.GATE or instr.gate is None:
        # Pass through non-gate instructions
        out.append(instr)
        return

    gate = instr.gate
    if gate.name in supported_gates:
        # Already supported — pass through
        out.append(instr)
        return

    # Try decomposition
    try:
        direct = _synthesize_direct(gate, instr.qubits, supported_gates)
        substitutions = (
            direct if direct is not None else decompose_gate(gate, instr.qubits)
        )
    except ValueError:
        # No rule available — keep the original gate and let the
        # exporter raise the appropriate error.
        out.append(instr)
        return

    out.extend(_gate_instructions(substitutions, instr.label if instr.label else ""))


#: Single-qubit gates whose rule chain is long on ``{h, rz}`` bases
#: (``u3 → rz·ry·rz → rz·sdg·h·rz·h·s·rz``) and which are therefore
#: synthesized directly when ``ry`` is not supported.
_DIRECT_SYNTHESIS_GATES: frozenset[str] = frozenset({"u3", "u2"})


def _synthesize_direct(
    gate: Gate, qubits: tuple[int, ...], supported_gates: set[str]
) -> list[tuple[Gate, tuple[int, ...]]] | None:
    """Synthesize ``u3``/``u2`` straight into an ``rz``/``h`` style basis.

    Returns the basis-aware ZXZ form ``rz·h·rz·h·rz`` (at most five gates,
    identity rotations dropped, special angles shortened) when the
    supported set lacks ``ry`` but can express the gate, otherwise
    ``None`` so that the regular rule chain is used.
    """
    if (
        gate.name not in _DIRECT_SYNTHESIS_GATES
        or gate.num_qubits != 1
        or "ry" in supported_gates
        or not all(isinstance(p, (int, float)) for p in gate.params)
    ):
        return None
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.synthesis import synthesize_one_qubit

    try:
        seq = synthesize_one_qubit(gate_matrix(gate), supported_gates)
    except (ValueError, TypeError):
        return None
    if seq is None:
        return None
    return [(g, (qubits[0],)) for g in seq]


def decompose_deep(
    circuit: QuantumCircuit,
    *,
    supported_gates: set[str],
    max_passes: int = 20,
) -> QuantumCircuit:
    """Recursively decompose unsupported gates until convergence.

    Unlike :func:`decompose_circuit` which does a single pass, this
    function iterates until all gates are in ``supported_gates`` or
    no further decompositions are possible.

    Args:
        circuit: The circuit to decompose.
        supported_gates: Set of gate names that are considered native.
        max_passes: Safety limit on decomposition iterations.

    Returns:
        A new circuit with as many gates decomposed as possible.

    Raises:
        RecursionError: If decomposition does not converge within
            *max_passes*.
    """
    from quspect.circuit._native import unsupported_gate_indices
    from quspect.circuit.instruction import InstructionType

    current = circuit
    for _ in range(max_passes):
        decomposed = decompose_circuit(current, supported_gates=supported_gates)
        indices = unsupported_gate_indices(decomposed._instructions, supported_gates)
        # Only the listed instructions can be unsupported gates.
        candidates = (
            decomposed
            if indices is None
            else [decomposed._instructions[i] for i in indices]
        )
        all_supported = True
        for instr in candidates:
            if (
                instr.type == InstructionType.GATE
                and instr.gate is not None
                and instr.gate.name not in supported_gates
            ):
                all_supported = False
                break
        if all_supported:
            return decomposed
        if decomposed.instructions == current.instructions:
            return decomposed
        current = decomposed

    raise RecursionError(
        f"Gate decomposition did not converge after {max_passes} passes. "
        f"Remaining unsupported gates: "
        f"{_unsupported_gates(current, supported_gates)}"
    )


def _unsupported_gates(circuit: QuantumCircuit, supported_gates: set[str]) -> list[str]:
    """Return names of gates in *circuit* that are not in *supported_gates*."""
    from quspect.circuit.instruction import InstructionType

    result: list[str] = []
    for instr in circuit:
        if (
            instr.type == InstructionType.GATE
            and instr.gate is not None
            and instr.gate.name not in supported_gates
        ):
            result.append(instr.gate.name)
    return sorted(set(result))
