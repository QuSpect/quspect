"""Tests for :class:`quspect.transpiler.TCountOptimizePass`."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.native import load_core
from quspect.transpiler import TCountOptimizePass, TCountStats
from quspect.transpiler import _tcount_poly as poly
from quspect.transpiler import _tcount_rotations as rot
from tests.synthesis._unitary import circuit_unitary, phase_distance

_HAS_NATIVE = getattr(load_core(), "_tcount_todd_batch", None) is not None
needs_native = pytest.mark.skipif(not _HAS_NATIVE, reason="native extension not built")


def _t_count(circuit: QuantumCircuit) -> int:
    return sum(1 for i in circuit if i.gate is not None and i.gate.name in ("t", "tdg"))


def _num_qubits(circuit: QuantumCircuit) -> int:
    return max(max(i.qubits) for i in circuit) + 1


def _assert_equivalent(a: QuantumCircuit, b: QuantumCircuit) -> None:
    n = max(_num_qubits(a), _num_qubits(b))
    assert phase_distance(circuit_unitary(a, n), circuit_unitary(b, n)) < 1e-8


def _random_circuit(n: int, size: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(size):
        r = rng.random()
        q = rng.randrange(n)
        if r < 0.3:
            getattr(qc, rng.choice(["t", "tdg"]))(q)
        elif r < 0.5:
            getattr(qc, rng.choice(["h", "s", "sdg", "x", "y", "z"]))(q)
        elif r < 0.55:
            getattr(qc, rng.choice(["rz", "p"]))(rng.randrange(-8, 9) * math.pi / 4, q)
        elif n >= 3 and r < 0.65:
            qc.ccx(*rng.sample(range(n), 3))
        else:
            getattr(qc, rng.choice(["cx", "cx", "cz", "swap"]))(
                *rng.sample(range(n), 2)
            )
    return qc


def _gf_multiplier(m: int, modulus: int) -> QuantumCircuit:
    """``c ^= a · b`` in GF(2)[x] / modulus, one Toffoli per partial product."""
    reduced = []
    for d in range(2 * m - 1):
        v = 1 << d
        for s in range(2 * m - 2, m - 1, -1):
            if (v >> s) & 1:
                v ^= modulus << (s - m)
        reduced.append(v)
    qc = QuantumCircuit()
    for i in range(m):
        for j in range(m):
            for k in range(m):
                if (reduced[i + j] >> k) & 1:
                    qc.ccx(i, m + j, 2 * m + k)
    return qc


def _ccz_columns(a: int, b: int, c: int) -> set[int]:
    x, y, z = 1 << a, 1 << b, 1 << c
    return {x, y, z, x ^ y, x ^ z, y ^ z, x ^ y ^ z}


# ---------------------------------------------------------------- the pass
@pytest.mark.parametrize("seed", range(30))
def test_random_clifford_t_circuits_keep_their_unitary(seed: int) -> None:
    rng = random.Random(seed)
    qc = _random_circuit(rng.randint(2, 6), rng.randint(5, 70), seed)
    p = TCountOptimizePass()
    out = p.run(qc)
    _assert_equivalent(qc, out)
    assert p.stats.t_count_out <= p.stats.t_count_in
    if p.stats.regions_improved:
        assert _t_count(out) == p.stats.t_count_out
        assert {i.gate.name for i in out if i.gate is not None} <= {
            "h",
            "s",
            "sdg",
            "t",
            "tdg",
            "cx",
        }


@pytest.mark.parametrize("seed", range(12))
def test_python_reference_gives_the_same_circuit(seed: int) -> None:
    qc = _random_circuit(5, 60, 100 + seed)
    a = TCountOptimizePass(native=True).run(qc)
    b = TCountOptimizePass(native=False).run(qc)
    assert [(i.gate, i.qubits) for i in a] == [(i.gate, i.qubits) for i in b]


def test_toffoli_examples() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).ccx(1, 2, 3).ccx(0, 1, 2)
    p = TCountOptimizePass()
    out = p.run(qc)
    assert (p.stats.t_count_in, p.stats.t_count_out) == (21, 15)
    assert p.stats == TCountStats(21, 15, 1, 1)
    _assert_equivalent(qc, out)

    qc = QuantumCircuit()
    for a, b, c in [(0, 1, 4), (1, 2, 5), (2, 3, 4), (0, 3, 5), (1, 4, 5), (0, 2, 4)]:
        qc.ccx(a, b, c)
    out = p.run(qc)
    assert (p.stats.t_count_in, p.stats.t_count_out) == (42, 21)
    _assert_equivalent(qc, out)


def test_cancelling_toffolis_and_merged_phases() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2).t(0).t(0)
    p = TCountOptimizePass()
    out = p.run(qc)
    assert _t_count(out) == 0 and p.stats.t_count_out == 0
    _assert_equivalent(qc, out)


def test_circuit_without_gain_is_returned_unchanged() -> None:
    qc = QuantumCircuit().h(0).t(0).cx(0, 1).h(1).t(1).cx(1, 2).h(2).t(2)
    p = TCountOptimizePass()
    out = p.run(qc)
    assert [(i.gate, i.qubits) for i in out] == [(i.gate, i.qubits) for i in qc]
    assert p.stats == TCountStats(3, 3, 1, 0)
    assert out is not qc


def test_other_instructions_split_the_circuit_into_regions() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).ccx(0, 1, 2).rz(0.3, 0).ccx(0, 1, 2).ccx(0, 1, 2)
    p = TCountOptimizePass()
    out = p.run(qc)
    # both regions vanish; the rz stays
    assert [i.gate.name for i in out if i.gate is not None] == ["rz"]
    assert p.stats == TCountStats(28, 0, 2, 2)

    # a rotation on another qubit still ends the region: the Toffolis do not cancel
    qc = QuantumCircuit().ccx(0, 1, 2).rx(0.5, 3).ccx(0, 1, 2)
    out = p.run(qc)
    assert p.stats.regions == 2
    _assert_equivalent(qc, out)

    qc = QuantumCircuit().t(0).t(0).measure([0], [0]).t(0).barrier(0, 1).t(0).t(0)
    out = p.run(qc)
    kinds = [
        ("measure" if i.is_measure else "barrier" if i.is_barrier else "gate")
        for i in out
    ]
    assert kinds.count("measure") == 1 and kinds.count("barrier") == 1
    assert kinds.index("measure") < kinds.index("barrier")
    assert p.stats.t_count_in == 5 and p.stats.t_count_out == 1


def test_rotations_by_multiples_of_pi_over_four_are_clifford_t() -> None:
    qc = QuantumCircuit()
    qc.rz(math.pi / 4, 0).cx(0, 1).p(3 * math.pi / 4, 1).cx(0, 1).rz(-math.pi / 4, 0)
    qc.u1(math.pi / 2, 1).rz(2 * math.pi, 0)
    p = TCountOptimizePass()
    out = p.run(qc)
    assert p.stats.t_count_in == 3 and p.stats.t_count_out == 1
    _assert_equivalent(qc, out)


@needs_native
def test_field_multipliers_with_tensor_starts() -> None:
    qc = _gf_multiplier(3, 0b1011)
    plain = TCountOptimizePass()
    plain.run(qc)
    assert (plain.stats.t_count_in, plain.stats.t_count_out) == (84, 31)
    p = TCountOptimizePass(tensor_walks=4, tensor_steps=5000)
    out = p.run(qc)
    assert p.stats.t_count_out == 29
    _assert_equivalent(qc, out)
    again = TCountOptimizePass(tensor_walks=4, tensor_steps=5000).run(qc)
    assert [(i.gate, i.qubits) for i in again] == [(i.gate, i.qubits) for i in out]

    p = TCountOptimizePass(rm_iterations=5, rm_runs=2)
    out = p.run(qc)
    assert p.stats.t_count_out <= 31
    _assert_equivalent(qc, out)


@needs_native
@pytest.mark.slow
def test_gf16_multiplier_reaches_47() -> None:
    qc = _gf_multiplier(4, 0b10011)
    p = TCountOptimizePass(tensor_walks=4, tensor_steps=5000)
    out = p.run(qc)
    assert (p.stats.t_count_in, p.stats.t_count_out) == (154, 47)
    _assert_equivalent(qc, out)


def test_argument_validation() -> None:
    for kwargs in (
        {"plateau": -1},
        {"rm_iterations": -1},
        {"rm_runs": 0},
        {"tensor_walks": -1},
        {"tensor_steps": -1},
        {"lookahead": -1},
        {"seed": -1},
        {"seed": 2**32},
    ):
        with pytest.raises(ValueError):
            TCountOptimizePass(**kwargs)  # type: ignore[arg-type]


def test_empty_circuit() -> None:
    p = TCountOptimizePass()
    assert len(p.run(QuantumCircuit())) == 0
    assert p.stats == TCountStats()


# ---------------------------------------------------------------- with ancillas
def _ancilla_error(
    circuit: QuantumCircuit, out: QuantumCircuit, trials: int = 2
) -> float:
    """How far ``out`` is from implementing ``circuit`` with clean ancillas.

    From random product states (prepared by rotations the pass treats as
    fences), every measurement record of ``out`` must leave the ideal output
    state on the circuit's qubits and ``|0…0⟩`` on the ancillas; the record
    probabilities must add up to 1.  Exact branch simulation
    (:mod:`tests._dynamic`).
    """
    from tests._dynamic import branches

    n = circuit.num_qubits
    total = out.num_qubits
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(trials):
        prep = QuantumCircuit()
        for q in range(n):
            prep.ry(float(rng.uniform(0, 3)), q).rz(float(rng.uniform(0, 6)), q)
        ideal = QuantumCircuit()
        ideal._instructions = [
            i
            for i in list(prep) + list(circuit)
            if not i.is_measure and not i.is_barrier
        ]
        (want,) = branches(ideal, n, 0)[()]
        zeros = np.zeros(2 ** (total - n), dtype=complex)
        zeros[0] = 1.0
        target = np.kron(want.reshape(-1), zeros)
        full = QuantumCircuit()
        full._instructions = list(prep) + list(out)
        probability = 0.0
        for ensemble in branches(full, total, out.num_clbits).values():
            for state in ensemble:
                vec = state.reshape(-1)
                weight = float(np.vdot(vec, vec).real)
                probability += weight
                worst = max(worst, abs(abs(np.vdot(target, vec)) ** 2 - weight))
        worst = max(worst, abs(probability - 1.0))
    return worst


def test_hadamard_gadgets_lower_the_t_count_of_toffoli_circuits() -> None:
    qc = QuantumCircuit().ccx(0, 1, 3).ccx(2, 3, 4).ccx(0, 1, 3)
    plain = TCountOptimizePass()
    plain.run(qc)
    assert plain.stats.t_count_out == 15
    p = TCountOptimizePass(ancillas=True)
    out = p.run(qc)
    assert p.stats == TCountStats(21, 13, 1, 1, 2, 2, 13)
    assert _t_count(out) == 13
    assert sum(i.is_measure for i in out) == 2
    assert sum(i.is_conditional for i in out) == 13
    assert out.num_qubits == qc.num_qubits + 2 and out.num_clbits == 2
    assert _ancilla_error(qc, out) < 1e-9

    qc = QuantumCircuit().ccx(0, 1, 2).h(2).ccx(0, 1, 2).h(1).ccx(0, 2, 1)
    out = p.run(qc)
    assert (p.stats.t_count_out, p.stats.ancillas) == (7, 2)
    assert _ancilla_error(qc, out) < 1e-9


def test_gadgets_are_used_only_when_they_save_t_gates() -> None:
    # no internal Hadamard after optimization: the ancilla-free result is kept
    qc = QuantumCircuit().ccx(0, 1, 2).cx(2, 3).ccx(0, 1, 2).h(3).ccx(1, 3, 0)
    qc.h(3).ccx(1, 3, 0)
    p = TCountOptimizePass(ancillas=True)
    out = p.run(qc)
    assert (p.stats.t_count_out, p.stats.ancillas, p.stats.measurements) == (11, 0, 0)
    assert not any(i.is_measure or i.is_conditional for i in out)
    _assert_equivalent(qc, out)
    # nothing to gain on a single qubit
    qc = QuantumCircuit().h(0).t(0).h(0).t(0).h(0).t(0).h(0).t(0)
    out = p.run(qc)
    assert p.stats.regions_improved == 0 and p.stats.ancillas == 0
    assert [(i.gate, i.qubits) for i in out] == [(i.gate, i.qubits) for i in qc]


def test_max_ancillas_limits_the_gadgets() -> None:
    qc = QuantumCircuit().ccx(0, 1, 2).h(2).ccx(0, 1, 2).h(1).ccx(0, 2, 1)
    limited = TCountOptimizePass(ancillas=True, max_ancillas=1)
    out = limited.run(qc)
    assert (limited.stats.t_count_out, limited.stats.ancillas) == (15, 0)
    _assert_equivalent(qc, out)
    off = TCountOptimizePass(ancillas=True, max_ancillas=0)
    assert [(i.gate, i.qubits) for i in off.run(qc)] == [
        (i.gate, i.qubits) for i in TCountOptimizePass().run(qc)
    ]
    with pytest.raises(ValueError, match="max_ancillas"):
        TCountOptimizePass(max_ancillas=-1)


def test_the_default_pass_adds_no_ancilla_measurement_or_condition() -> None:
    qc = QuantumCircuit().ccx(0, 1, 3).ccx(2, 3, 4).ccx(0, 1, 3)
    p = TCountOptimizePass()
    out = p.run(qc)
    assert out.num_qubits == qc.num_qubits and out.num_clbits == 0
    assert all(i.is_gate for i in out)
    assert (p.stats.ancillas, p.stats.measurements, p.stats.conditional_gates) == (
        0,
        0,
        0,
    )


@pytest.mark.parametrize("seed", range(24))
def test_random_circuits_with_gadgets_implement_the_same_unitary(seed: int) -> None:
    rng = random.Random(seed)
    qc = _random_circuit(rng.randint(2, 4), rng.randint(8, 40), 500 + seed)
    p = TCountOptimizePass(ancillas=True, max_ancillas=5)
    out = p.run(qc)
    assert p.stats.t_count_out <= p.stats.t_count_in
    assert _t_count(out) == p.stats.t_count_out or p.stats.regions_improved == 0
    assert sum(i.is_measure for i in out) == p.stats.measurements
    assert sum(i.is_conditional for i in out) == p.stats.conditional_gates
    assert out.num_qubits <= qc.num_qubits + p.stats.ancillas
    if out.num_qubits <= 10:
        assert _ancilla_error(qc, out) < 1e-8


def test_gadget_bits_and_ancillas_lie_behind_the_circuit_and_are_reused() -> None:
    # two regions separated by a fence; the circuit already has classical bits
    block = [(0, 1, 2), (0, 1, 2)]
    qc = QuantumCircuit().x(5).measure([5], [3])
    for a, b, c in block:
        qc.ccx(a, b, c).h(c)
    qc.rx(0.3, 4)
    for a, b, c in block:
        qc.ccx(a, c, b).h(b)
    p = TCountOptimizePass(ancillas=True)
    out = p.run(qc)
    assert (
        p.stats.regions == 2 and p.stats.regions_improved == 2 and p.stats.ancillas >= 1
    )
    first_ancilla, first_bit = qc.num_qubits, qc.num_clbits
    gadget_bits = {c for i in out if i.is_conditional for c in i.clbits}
    assert gadget_bits and min(gadget_bits) >= first_bit
    measured = [i for i in out if i.is_measure][1:]
    assert all(i.clbits[0] >= first_bit for i in measured)
    # the same ancillas (and bits) serve both regions
    assert out.num_qubits == first_ancilla + p.stats.ancillas
    assert len({i.clbits for i in measured}) < len(measured)
    unitary = QuantumCircuit()
    unitary._instructions = [i for i in qc if not i.is_measure]
    rewritten = QuantumCircuit()
    rewritten._instructions = [
        i for i in out if not (i.is_measure and i.clbits == (3,))
    ]
    assert _ancilla_error(unitary, rewritten, trials=1) < 1e-8


def test_gadget_output_runs_in_the_simulator_and_compiles() -> None:
    from quspect.simulation import Simulator
    from quspect.transpiler import CompilationTarget, transpile
    from tests._dynamic import assert_compiled_behaviour, distribution

    qc = QuantumCircuit().h(0).h(1).ccx(0, 1, 2).h(2).ccx(0, 1, 2).h(1).ccx(0, 2, 1)
    p = TCountOptimizePass(ancillas=True)
    out = p.run(qc)
    assert p.stats.ancillas > 0
    exact = distribution(qc.copy().measure([0, 1, 2], [0, 1, 2]))
    measured = out.copy()
    first = out.num_clbits
    measured.measure([0, 1, 2], [first, first + 1, first + 2])
    counts = Simulator(seed=3).run(measured, shots=3000).counts
    marginal: dict[tuple[int, ...], float] = {}
    for key, n in counts.items():
        record = tuple(int(ch) for ch in key[first : first + 3])
        marginal[record] = marginal.get(record, 0.0) + n / 3000
    for record, probability in exact.items():
        assert abs(marginal.get(record, 0.0) - probability) < 0.05
    target = CompilationTarget.from_coupling_map(
        out.num_qubits,
        [(i, i + 1) for i in range(out.num_qubits - 1)],
        native_gates={"cz", "h", "rz", "x"},
    )
    res = transpile(out, target, optimization_level=1, seed=0)
    assert_compiled_behaviour(
        out, res.circuit, res.final_layout[: out.num_qubits], target.num_qubits
    )


@pytest.mark.parametrize("seed", range(6))
def test_gadgets_python_reference_gives_the_same_circuit(seed: int) -> None:
    qc = _random_circuit(4, 40, 900 + seed)
    a = TCountOptimizePass(ancillas=True, max_ancillas=4, native=True).run(qc)
    b = TCountOptimizePass(ancillas=True, max_ancillas=4, native=False).run(qc)
    assert [repr(i) for i in a] == [repr(i) for i in b]


@pytest.mark.parametrize("seed", range(10))
def test_min_h_emission_keeps_the_unitary(seed: int) -> None:
    from quspect.transpiler import _tcount_ancilla as gadget
    from quspect.transpiler._tcount import _OUT_GATES, _translate

    qc = _random_circuit(4, 40, 700 + seed)
    gates = [g for ins in qc for g in _translate(ins) or []]
    n = _num_qubits(qc)
    clifford, rots = rot.merge_rotations(*rot.from_gates(gates, n), n)
    emitted = gadget.min_h_emit(clifford, rots, n)
    out = QuantumCircuit()
    for g in emitted:
        out.append(_OUT_GATES[g[0]], [g[1]] if g[2] < 0 else [g[1], g[2]])
    if len(out):
        _assert_equivalent(qc, out)
    assert sum(1 for g in emitted if g[0] in (rot.T, rot.TDG)) == len(rots)
    head, mid, tail = gadget.split_head_middle_tail(emitted, n)
    assert len(head) + len(mid) + len(tail) == len(emitted)
    assert not any(g[0] in (rot.T, rot.TDG) for g in head + tail)
    assert gadget.internal_hadamards(emitted, n) == sum(1 for g in mid if g[0] == rot.H)


# ---------------------------------------------------------------- Pauli bookkeeping
@pytest.mark.parametrize("seed", range(10))
def test_incremental_frame_matches_gate_by_gate_conjugation(seed: int) -> None:
    rng = random.Random(seed)
    n = 5
    gates: list[rot.GateT] = []
    for _ in range(60):
        r = rng.random()
        if r < 0.4:
            a, b = rng.sample(range(n), 2)
            gates.append((rot.CX, a, b))
        elif r < 0.8:
            gates.append((rng.choice([rot.H, rot.S, rot.SDG]), rng.randrange(n), -1))
        else:
            gates.append((rng.choice([rot.T, rot.TDG]), rng.randrange(n), -1))
    clifford, rots = rot.from_gates(gates, n)
    seen: list[rot.GateT] = []
    want = []
    for g in gates:
        if g[0] in (rot.T, rot.TDG):
            x, z, s = rot.conj_circ_inv((0, 1 << g[1], 1), seen)
            want.append((x, z, s if g[0] == rot.T else -s))
        else:
            seen.append(g)
    assert rots == want and clifford == seen


@pytest.mark.parametrize("seed", range(10))
def test_clifford_resynthesis_has_the_same_tableau(seed: int) -> None:
    rng = random.Random(seed)
    n = 4
    gates: list[rot.GateT] = []
    for _ in range(40):
        if rng.random() < 0.5:
            a, b = rng.sample(range(n), 2)
            gates.append((rot.CX, a, b))
        else:
            gates.append((rng.choice([rot.H, rot.S, rot.SDG]), rng.randrange(n), -1))
    new = rot.clifford_circuit(gates, n)
    for q in range(n):
        for p in ((1 << q, 0, 1), (0, 1 << q, 1)):
            assert rot.conj_circ(p, gates) == rot.conj_circ(p, new)


# ---------------------------------------------------------------- phase polynomials
def test_clifford_difference_accepts_identities_and_rejects_others() -> None:
    # T² = S on x0
    assert poly.clifford_difference([(1, 1), (1, 1)], []) == ({0: 1}, [])
    # S on x0, x1 and S† on x0⊕x1 is CZ(x0, x1)
    cz = [(1, 1), (1, 1), (2, 1), (2, 1), (3, -1), (3, -1)]
    assert poly.clifford_difference(cz, []) == ({}, [(0, 1)])
    # T on x0, x1 and T† on x0⊕x1 is CS(x0, x1), which is not Clifford
    assert poly.clifford_difference([(1, 1), (2, 1), (3, -1)], []) is None
    # 4-flat identity: the 15 non-zero points of a 4-space make a Clifford
    assert poly.clifford_difference([(p, 1) for p in range(1, 16)], []) is not None
    assert poly.clifford_difference([(p, 1) for p in range(1, 8)], []) is None
    ccz = sorted(_ccz_columns(0, 1, 2))
    assert poly.clifford_difference([(c, 1) for c in ccz], ccz[1:]) is None


@pytest.mark.parametrize("seed", range(20))
def test_todd_keeps_the_polynomial_and_native_matches_python(seed: int) -> None:
    rng = random.Random(seed)
    k = rng.randint(4, 8)
    cols = rng.sample(range(1, 1 << k), min(rng.randint(4, 26), (1 << k) - 1))
    plateau = rng.choice([0, 12])
    ref = poly.todd(cols, k, plateau=plateau, seed=seed, native=False)
    assert len(ref) <= len(cols)
    assert poly.clifford_difference([(c, 1) for c in cols], ref) is not None
    if _HAS_NATIVE:
        assert poly.todd(cols, k, plateau=plateau, seed=seed, native=True) == ref


def test_todd_reduces_six_toffolis() -> None:
    cols: set[int] = set()
    for t in [(0, 1, 2), (2, 3, 4), (1, 4, 5), (0, 3, 5), (1, 2, 6), (4, 5, 6)]:
        cols ^= _ccz_columns(*t)
    assert len(cols) == 24
    assert len(poly.todd(sorted(cols), 7, native=False)) == 17


@pytest.mark.parametrize("seed", range(4))
def test_rm_search_keeps_the_polynomial_and_native_matches_python(seed: int) -> None:
    rng = random.Random(50 + seed)
    k = rng.randint(6, 8)
    cols = rng.sample(range(1, 1 << k), rng.randint(20, 30))
    ref = poly.rm_search([(cols, k, 3, seed)], native=False)[0]
    assert poly.clifford_difference([(c, 1) for c in cols], ref) is not None
    assert len(ref) <= len(poly.todd(cols, k, native=False))
    if _HAS_NATIVE:
        assert poly.rm_search([(cols, k, 3, seed)], native=True)[0] == ref


def test_flat_moves_need_independent_directions() -> None:
    assert poly._independent([1, 2, 4, 8])
    assert not poly._independent([2, 4, 6, 8])
    # a 3-flat (8 points) is not a valid move
    flat3 = sorted(poly._flat(1, [2, 4]) | poly._flat(9, [2, 4]))
    assert len(flat3) == 8
    assert poly.clifford_difference([(p, 1) for p in flat3], []) is None
    flat4 = sorted(poly._flat(1, [2, 4, 8, 16]))
    assert len(flat4) == 16
    assert poly.clifford_difference([(p, 1) for p in flat4], []) is not None


def test_karatsuba_decompositions() -> None:
    for m in range(1, 8):
        tensor = {(i, j, i + j) for i in range(m) for j in range(m)}
        assert poly.tensor_of(poly._karatsuba_poly(m)) == tensor
    assert [len(poly._karatsuba_poly(m)) for m in (2, 3, 4, 6)] == [3, 6, 9, 18]


@pytest.mark.parametrize("seed", range(6))
def test_flip_walk_keeps_the_tensor_and_native_matches_python(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.randint(2, 4)
    start = [
        (1 << rng.randrange(n), 1 << rng.randrange(n), 1 << rng.randrange(n))
        for _ in range(rng.randint(3, 14))
    ]
    ref = poly.flip_walks([(start, 2000, seed)], native=False)[0]
    assert poly.tensor_of(ref) == poly.tensor_of(start)
    assert len(ref) <= len(poly._reduce(list(start)))
    if _HAS_NATIVE:
        assert poly.flip_walks([(start, 2000, seed)], native=True)[0] == ref


def test_tensor_starts_are_equivalent_to_the_polynomial() -> None:
    # GF(2^2) multiplier as a phase polynomial: 4·Σ T_ijk a_i b_j c_k
    tensor = {(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)}
    cols: set[int] = set()
    for i, j, k in tensor:
        cols ^= _ccz_columns(i, 2 + j, 4 + k)
    old = [(c, 1) for c in sorted(cols)]
    parts = poly.tripartition(
        sorted(t for t, v in poly.multilinear(old)[2].items() if v)
    )
    assert parts is not None and sorted(map(len, parts)) == [2, 2, 2]
    starts = poly.tensor_starts(old, walks=3, steps=2000, seed=0, native=False)
    assert starts
    for start in starts:
        assert poly.clifford_difference(old, start) is not None
    # three Toffolis (Karatsuba) instead of five partial products
    assert min(len(s) for s in starts) <= 21
    # a polynomial whose cubic part is not tripartite gives no start
    k4 = [
        (c, 1)
        for t in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        for c in _ccz_columns(*t)
    ]
    assert poly.tensor_starts(k4, walks=2, steps=100, seed=0, native=False) == []


def test_normalise_cancels_pairs_and_keeps_order() -> None:
    assert poly.normalise([5, 3, 0, 5, 7, 3, 3]) == [7, 3]
