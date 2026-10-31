"""Semantic preservation of the ``quspect.dag.passes`` optimization passes.

Every pass (and the full :class:`PassManager` pipeline) is run on many
random circuits and on circuits densely seeded with cancellable / mergeable /
commutable motifs.  The output unitary must equal the input unitary up to a
global phase (``atol=1e-9``); both unitaries are computed by qiskit
``Operator`` on ``to_qiskit`` conversions.  Circuits with terminal
measurements must keep the exact measurement distribution, and passes run on
symbolic (parameterized) circuits must commute with parameter binding.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import quspect
from qiskit_tests._dag_helpers import apply_pass, apply_passes
from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    qiskit_probabilities,
    random_circuit_pair,
)
from quspect.circuit.parameter import Parameter
from quspect.dag.passes import (
    BasePass,
    CommutationAnalysis,
    EliminateDiagonalBeforeMeasurement,
    GateCancellation,
    RotationMerging,
    SingleQubitGateFusion,
)

pytestmark = pytest.mark.qiskit_ref

PASSES: dict[str, Callable[[], BasePass]] = {
    "cancel": GateCancellation,
    "commute": CommutationAnalysis,
    "rotmerge": RotationMerging,
    "fuse": SingleQubitGateFusion,
    "diagmeas": EliminateDiagonalBeforeMeasurement,
}
PIPELINE: list[Callable[[], BasePass]] = list(PASSES.values())

_CLIFFORD_T_ROT = [
    "id", "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "p", "u1", "cx", "cz", "swap", "crz", "crx", "cp",
]  # fmt: skip


def _run(name: str, circ: quspect.QuantumCircuit) -> quspect.QuantumCircuit:
    """Apply a named pass, or the full pipeline for ``"pipeline"``."""
    if name == "pipeline":
        return apply_passes(PIPELINE, circ)[0]
    return apply_pass(PASSES[name], circ)


def _assert_same_unitary(
    before: quspect.QuantumCircuit, after: quspect.QuantumCircuit, n: int
) -> None:
    """Unitaries equal up to global phase (qiskit reference)."""
    assert_unitary_equal(
        qiskit_circuit_unitary(after, num_qubits=n),
        qiskit_circuit_unitary(before, num_qubits=n),
        up_to_phase=True,
        atol=1e-9,
    )


def _motif_circuit(n: int, length: int, seed: int) -> quspect.QuantumCircuit:
    """Random circuit densely seeded with optimizable motifs.

    Motifs: self-inverse pairs, S/Sdg and T/Tdg pairs, opposite rotations,
    same-axis rotation chains, mixed rz/p/u1 chains, CX sandwiches around
    gates commuting with control/target, CX pairs sharing control/target,
    single-qubit runs, diagonal gates before (future) measurements, barriers
    separating would-be cancellations.
    """
    rng = np.random.default_rng(seed)
    c = quspect.QuantumCircuit().id(n - 1)

    def ang() -> float:
        return float(rng.uniform(-2 * math.pi, 2 * math.pi))

    def pick(k: int) -> list[int]:
        return [int(q) for q in rng.permutation(n)[:k]]

    for _ in range(length):
        m = int(rng.integers(12))
        a, b = pick(2) if n > 1 else (0, 0)
        if m == 0:
            g = str(rng.choice(["h", "x", "y", "z", "id"]))
            getattr(c, g)(a)
            getattr(c, g)(a)
        elif m == 1:
            g, gi = [("s", "sdg"), ("t", "tdg"), ("sdg", "s"), ("tdg", "t")][
                int(rng.integers(4))
            ]
            getattr(c, g)(a)
            getattr(c, gi)(a)
        elif m == 2:
            g = str(rng.choice(["rx", "ry", "rz", "p"]))
            t = ang()
            getattr(c, g)(t, a)
            getattr(c, g)(-t, a)
        elif m == 3 and n > 1:
            g = str(rng.choice(["cx", "cz", "swap"]))
            getattr(c, g)(a, b)
            getattr(c, g)(a, b)
        elif m == 4 and n > 1:
            g = str(rng.choice(["crx", "cry", "crz", "cp"]))
            t = ang()
            getattr(c, g)(t, a, b)
            getattr(c, g)(-t, a, b)
        elif m == 5:
            for _ in range(int(rng.integers(2, 5))):
                g = str(rng.choice(["rz", "p", "u1"]))
                getattr(c, g)(ang(), a)
        elif m == 6:
            g = str(rng.choice(["rx", "ry"]))
            for _ in range(int(rng.integers(2, 4))):
                getattr(c, g)(ang(), a)
        elif m == 7 and n > 1:  # CX - (Z-family on control) - CX
            c.cx(a, b)
            g = str(rng.choice(["z", "s", "t", "rz", "p", "tdg"]))
            if g in ("rz", "p"):
                getattr(c, g)(ang(), a)
            else:
                getattr(c, g)(a)
            c.cx(a, b)
        elif m == 8 and n > 1:  # CX - (X-family on target) - CX
            c.cx(a, b)
            if rng.random() < 0.5:
                c.x(b)
            else:
                c.rx(ang(), b)
            c.cx(a, b)
        elif m == 9 and n > 2:  # CX pairs sharing a control or a target
            d = next(q for q in range(n) if q not in (a, b))
            if rng.random() < 0.5:
                c.cx(a, b).cx(a, d).cx(a, b)
            else:
                c.cx(a, b).cx(d, b).cx(a, b)
        elif m == 10:
            for _ in range(int(rng.integers(2, 6))):
                spec = GATE_SPECS[
                    str(rng.choice(["h", "s", "t", "rx", "ry", "u2", "u3", "y"]))
                ]
                spec.apply_quspect(c, [ang() for _ in range(spec.num_params)], [a])
        else:
            g = str(rng.choice(["h", "rz", "barrier"]))
            if g == "barrier":
                c.h(a).barrier(a).h(a)
            elif g == "h":
                c.h(a)
            else:
                c.rz(ang(), a)
    return c


_NAMES = [*PASSES, "pipeline"]


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in range(2, 6) for s in range(4)])
def test_random_circuit_unitary_preserved(name: str, n: int, seed: int) -> None:
    """Random circuits over all 25 gates: pass output == input up to phase."""
    circ, _ = random_circuit_pair(n, 40, 1000 * n + seed)
    _assert_same_unitary(circ, _run(name, circ), n)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in range(2, 6) for s in range(4)])
def test_random_clifford_rotation_circuit_preserved(
    name: str, n: int, seed: int
) -> None:
    """Random circuits over gates the passes know about: unitary preserved."""
    circ, _ = random_circuit_pair(n, 50, 77 * n + seed, gate_set=_CLIFFORD_T_ROT)
    _assert_same_unitary(circ, _run(name, circ), n)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize(("n", "seed"), [(n, s) for n in range(1, 6) for s in range(5)])
def test_motif_circuit_preserved(name: str, n: int, seed: int) -> None:
    """Circuits dense in cancellable/mergeable/commutable motifs: preserved."""
    circ = _motif_circuit(n, 25, 500 * n + seed)
    out = _run(name, circ)
    _assert_same_unitary(circ, out, n)
    if name in ("cancel", "rotmerge", "fuse", "pipeline"):
        assert len(out.instructions) <= len(circ.instructions)
    if name == "commute":
        assert len(out.instructions) == len(circ.instructions)


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("seed", range(6))
def test_terminal_measurement_distribution_preserved(name: str, seed: int) -> None:
    """With permuted terminal measurements the exact distribution is unchanged."""
    n = 2 + seed % 3
    circ = _motif_circuit(n, 20, 9000 + seed)
    rng = np.random.default_rng(seed)
    for q in range(n):  # trailing diagonals: eliminable before measurement
        circ.rz(float(rng.uniform(-3, 3)), q).t(q).s(q)
    circ.measure(list(range(n)), [int(c) for c in rng.permutation(n)])
    out = _run(name, circ)
    p_in, p_out = qiskit_probabilities(circ), qiskit_probabilities(out)
    keys = set(p_in) | set(p_out)
    for k in keys:
        assert p_out.get(k, 0.0) == pytest.approx(p_in.get(k, 0.0), abs=1e-9)


@pytest.mark.parametrize("name", _NAMES)
def test_mid_circuit_measurement_is_a_boundary(name: str) -> None:
    """Gates are never cancelled / merged across a mid-circuit measurement."""
    circ = quspect.QuantumCircuit()
    circ.h(0).rz(0.4, 0).measure([0], [0]).rz(-0.4, 0).h(0).measure([0], [1])
    out = _run(name, circ)
    kinds = [
        "m" if i.is_measure else i.gate.name
        for i in out.instructions
        if i.gate or i.is_measure
    ]
    first_m = kinds.index("m")
    # Something remains after the first measurement and before the second.
    assert kinds.count("m") == 2
    assert first_m < len(kinds) - 1
    assert len(kinds) - kinds[::-1].index("m") - 1 > first_m + 1


# =============================================================================
# Parameterized circuits
# =============================================================================

_A, _B = Parameter("a"), Parameter("b")


def _symbolic_circuit() -> quspect.QuantumCircuit:
    """Circuit with symbolic rotations next to numeric ones and inverse pairs."""
    c = quspect.QuantumCircuit()
    c.rx(_A, 0).rx(-1 * _A, 0)  # symbolic opposite pair
    c.rz(_A, 1).rz(0.3, 1).rz(_B, 1)  # symbolic / numeric chain
    c.ry(0.2, 0).ry(_B, 0).ry(0.5, 0)
    c.p(_A, 2).p(-0.7, 2)
    c.crz(_A, 0, 1).crz(-1 * _A, 0, 1)
    c.h(2).u3(_A, _B, 0.1, 2).h(2)
    c.cx(0, 1).rz(2 * _B, 0).cx(0, 1)
    c.rx(_A, 1).rx(_A, 1)
    c.t(2).tdg(2).cp(_B, 2, 0)
    return c


_BINDINGS = [{_A: 0.37, _B: -1.4}, {_A: math.pi, _B: 0.0}, {_A: -2.2, _B: 2.2}]


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("binding", range(len(_BINDINGS)))
def test_symbolic_pass_commutes_with_binding(name: str, binding: int) -> None:
    """pass(circ).bind(v) == bind(v) of circ, for several bindings."""
    circ = _symbolic_circuit()
    out = _run(name, circ)
    values = _BINDINGS[binding]
    _assert_same_unitary(circ.bind_parameters(values), out.bind_parameters(values), 3)
    # Optimizing after binding is also consistent.
    bound = circ.bind_parameters(values)
    _assert_same_unitary(bound, _run(name, bound), 3)


@pytest.mark.parametrize("name", _NAMES)
def test_symbolic_gates_are_not_merged(name: str) -> None:
    """Every symbolic-parameter gate survives (no numeric folding of symbols)."""
    circ = _symbolic_circuit()
    out = _run(name, circ)

    def symbolic(c: quspect.QuantumCircuit) -> list[str]:
        return sorted(
            i.gate.name
            for i in c.instructions
            if i.gate is not None
            and any(not isinstance(p, (int, float)) for p in i.gate.params)
        )

    assert symbolic(out) == symbolic(circ)
    assert out.parameters == circ.parameters
