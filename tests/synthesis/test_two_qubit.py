"""Tests for minimal-CZ two-qubit KAK synthesis."""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest
from scipy.linalg import expm

from quspect.synthesis import (
    kak_decompose,
    num_cz_required,
    synthesize_two_qubit,
    weyl_coordinates,
)
from quspect.synthesis.two_qubit import (
    canonical_gate,
    euler_zxz_ops,
    synthesize_two_qubit_ops,
)
from tests.synthesis._unitary import circuit_unitary, phase_distance, random_unitary

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
XX, YY, ZZ = np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CX = np.eye(4, dtype=complex)[[0, 1, 3, 2]]
SWAP = np.eye(4, dtype=complex)[[0, 2, 1, 3]]
ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)
ALLOWED = {"cz", "h", "rz", "x"}


def _cp(theta: float) -> np.ndarray:
    return np.diag([1, 1, 1, np.exp(1j * theta)]).astype(complex)


def _check(u: np.ndarray, expected_cz: int | None = None, tol: float = 1e-9) -> int:
    circ = synthesize_two_qubit(u)
    names = {ins.gate.name for ins in circ.instructions if ins.gate is not None}
    assert names <= ALLOWED
    assert phase_distance(u, circuit_unitary(circ, 2)) < tol
    n_cz = sum(1 for ins in circ.instructions if ins.gate and ins.gate.name == "cz")
    if expected_cz is not None:
        assert n_cz == expected_cz
    return n_cz


def _local(rng: np.random.Generator) -> np.ndarray:
    return np.kron(random_unitary(2, rng), random_unitary(2, rng))


# ---------------------------------------------------------------------------
# KAK decomposition
# ---------------------------------------------------------------------------


def test_kak_reconstructs_random_unitaries() -> None:
    rng = np.random.default_rng(11)
    for _ in range(300):
        u = random_unitary(4, rng)
        kak = kak_decompose(u)
        assert np.max(np.abs(kak.unitary() - u)) < 1e-10
        assert math.pi / 4 + 1e-9 >= kak.a >= kak.b - 1e-12 >= abs(kak.c) - 1e-12


def test_canonical_gate_matches_expm() -> None:
    a, b, c = 0.3, -0.2, 0.7
    ref = expm(1j * (a * XX + b * YY + c * ZZ))
    assert np.allclose(canonical_gate(a, b, c), ref)


@pytest.mark.parametrize(
    ("u", "coords"),
    [
        (np.eye(4, dtype=complex), (0.0, 0.0, 0.0)),
        (CZ, (math.pi / 4, 0.0, 0.0)),
        (CX, (math.pi / 4, 0.0, 0.0)),
        (SWAP, (math.pi / 4, math.pi / 4, math.pi / 4)),
        (ISWAP, (math.pi / 4, math.pi / 4, 0.0)),
        (_cp(0.6), (0.15, 0.0, 0.0)),
    ],
)
def test_weyl_coordinates_known_gates(
    u: np.ndarray, coords: tuple[float, float, float]
) -> None:
    assert np.allclose(weyl_coordinates(u), coords, atol=1e-9)


def test_weyl_coordinates_local_invariance() -> None:
    rng = np.random.default_rng(5)
    for _ in range(50):
        u = random_unitary(4, rng)
        v = _local(rng) @ u @ _local(rng)
        assert np.allclose(weyl_coordinates(u), weyl_coordinates(v), atol=1e-8)


def test_rejects_non_unitary() -> None:
    with pytest.raises(ValueError):
        kak_decompose(np.ones((4, 4)))
    with pytest.raises(ValueError):
        num_cz_required(np.eye(2))


# ---------------------------------------------------------------------------
# Synthesis: exactness and minimal CZ counts
# ---------------------------------------------------------------------------


def test_random_su4_three_cz_and_exact() -> None:
    rng = np.random.default_rng(2024)
    worst = 0.0
    for _ in range(2000):
        u = random_unitary(4, rng)
        ops = synthesize_two_qubit_ops(u)
        assert sum(1 for name, _, _ in ops if name == "cz") == 3
        from quspect.circuit.circuit import QuantumCircuit
        from quspect.synthesis.two_qubit import ops_to_instructions

        circ = QuantumCircuit._from_instruction_snapshot(
            ops_to_instructions(ops, (0, 1))
        )
        worst = max(worst, phase_distance(u, circuit_unitary(circ, 2)))
    assert worst < 1e-9


@pytest.mark.parametrize(
    ("name", "u", "expected"),
    [
        ("identity", np.eye(4, dtype=complex), 0),
        ("cz", CZ, 1),
        ("cx", CX, 1),
        ("cx_reversed", SWAP @ CX @ SWAP, 1),
        ("cp", _cp(0.7), 2),
        ("two_cp", _cp(0.3) @ _cp(0.5), 2),
        ("cp_cancel", _cp(0.3) @ _cp(-0.3), 0),
        ("iswap", ISWAP, 2),
        ("swap", SWAP, 3),
        ("swap_cz", SWAP @ CZ, 2),
        ("swap_cp", SWAP @ _cp(0.4), 3),
        ("heisenberg", expm(-0.3j * (XX + YY + ZZ)), 3),
        ("xx_yy", expm(-0.3j * (XX + YY)), 2),
        ("zz", expm(-0.3j * ZZ), 2),
        ("sqrt_swap", expm(1j * math.pi / 8 * (XX + YY + ZZ)), 3),
        ("b_gate", expm(1j * (math.pi / 4 * XX + math.pi / 8 * YY)), 2),
        ("cz_cz", CZ @ CZ, 0),
    ],
)
def test_special_cases_minimal_cz(name: str, u: np.ndarray, expected: int) -> None:
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for variant in (u, _local(rng) @ u, u @ _local(rng), _local(rng) @ u @ _local(rng)):
        _check(variant, expected)
        assert num_cz_required(variant) == expected


def test_degenerate_and_boundary_families() -> None:
    """Chamber faces/edges (a = π/4, a = b, b = |c|) and near-degenerate spectra."""
    rng = np.random.default_rng(99)
    pi4 = math.pi / 4
    for _ in range(200):
        t = rng.uniform(0, pi4)
        s = rng.uniform(0, t)
        for a, b, c in [
            (pi4, t, s),
            (pi4, t, -s),
            (t, t, t),
            (t, t, -t),
            (t, s, s),
            (t, s, -s),
            (t, t, 0.0),
            (pi4, pi4, t),
            (t, 1e-12, 0.0),
            (pi4 - 1e-12, t, -s),
        ]:
            u = _local(rng) @ canonical_gate(a, b, c) @ _local(rng)
            _check(u)


def test_near_identity_and_noise() -> None:
    rng = np.random.default_rng(3)
    for eps in (1e-13, 1e-11):
        h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        u = expm(1j * eps * (h + h.conj().T))
        assert _check(u, tol=1e-9) == 0


def test_euler_zxz_special_angles() -> None:
    h = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    assert [op[0] for op in euler_zxz_ops(h, 0)] == ["h"]
    assert euler_zxz_ops(-np.eye(2), 0) == []
    assert [op[0] for op in euler_zxz_ops(Z, 0)] == ["rz"]
    assert [op[0] for op in euler_zxz_ops(Y, 0)] in (["x", "rz"], ["rz", "x"])
    rng = np.random.default_rng(8)
    for _ in range(200):
        u = random_unitary(2, rng)
        from quspect.circuit.circuit import QuantumCircuit
        from quspect.synthesis.two_qubit import ops_to_instructions

        ops = euler_zxz_ops(u, 0)
        assert len(ops) <= 5
        circ = QuantumCircuit._from_instruction_snapshot(
            ops_to_instructions(ops, (0, 0))
        )
        got = circuit_unitary(circ, 1) if ops else np.eye(2)
        assert phase_distance(u, got) < 1e-12


def test_synthesize_circuit_api_example() -> None:
    circ = synthesize_two_qubit(SWAP)
    assert sum(1 for ins in circ if ins.gate and ins.gate.name == "cz") == 3


# ---------------------------------------------------------------------------
# Tolerance regressions (linear-scale verification, cf. commit 043ae76)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [1e-7, 1e-6, 1e-5, 1e-3, math.pi / 2**15])
def test_small_controlled_phase_is_not_dropped(theta: float) -> None:
    """A weak CP is a 2-CZ gate; treating it as local would lose ~θ/4."""
    rng = np.random.default_rng(zlib.crc32(repr(theta).encode()))
    for u in (_cp(theta), _local(rng) @ _cp(theta) @ _local(rng)):
        assert _check(u, tol=1e-12) == 2


def test_tiny_interactions_dropped_only_within_linear_budget() -> None:
    """Whatever is dropped (Weyl coordinates or Euler rotations ≤ atol)
    must cost ``max|U - e^{iφ}V| ≤ 1e-9`` on the emitted gates.

    Covers the region around the 0-CZ threshold ``a ≤ 1e-9``, where the
    coordinate test alone would drop up to ~1.5e-9 (dressed CP(4e-9)).
    """
    rng = np.random.default_rng(17)
    for _ in range(400):
        theta = 10 ** rng.uniform(-10, -7)
        u = _local(rng) @ _cp(theta) @ _local(rng)
        _check(u, tol=1e-9)
    for _ in range(200):
        eps = 10 ** rng.uniform(-10, -8, size=3)
        u = _local(rng) @ canonical_gate(*eps) @ _local(rng)
        _check(u, tol=1e-9)
    # Undressed CP(θ < 4e-9): Weyl point (θ/4, 0, 0) → no CZ, error ≤ 1e-9.
    for theta in (1e-10, 1e-9, 2e-9, 3.9e-9):
        assert _check(_cp(theta), tol=1e-9) == 0


@pytest.mark.parametrize(
    ("point", "bounds"),
    [
        ((0.125, 0.125, -0.125), (16, 19)),
        ((0.3, 0.3, -0.3), (16, 19)),
        ((0.5, 0.2, -0.2), (17, 22)),
    ],
)
def test_mirrored_degenerate_stabilizer(
    point: tuple[float, float, float], bounds: tuple[int, int]
) -> None:
    """``b = -c`` points have a continuous stabilizer (``V⊗Z·V·Z``, ...).

    Every candidate commutes with ``Can``, and a Clifford-dressed block (a
    Heisenberg ``XX + YY - ZZ`` term) gets short outer layers.  The round-12
    search depends on the KAK factors the platform's LAPACK happens to
    return (it is sometimes already short on Linux), so the new search is
    checked against fixed gate counts and never longer than the old one.
    """
    import quspect.synthesis.two_qubit as tq
    from tests.synthesis._legacy_stabilizer_r12 import legacy_try_synthesize

    can = tq.canonical_gate(*point)
    rng = np.random.default_rng(5)
    pair = tuple(
        tuple(
            np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
            for _ in range(2)
        )
        for _ in range(4)
    )
    for s0, s1 in tq._stabilizer_candidates(3, np.array(point), *pair):
        s = np.kron(s0, s1)
        assert np.allclose(s @ can, can @ s, atol=1e-9)
    u = np.kron(tq._H, tq._S) @ can @ np.kron(tq._SX, tq._H)
    kak = tq._kak4(u)
    for basis, bound in zip(
        (tq.TwoQubitBasis("cx", "zsx"), tq.TwoQubitBasis()), bounds, strict=True
    ):
        new = tq._try_synthesize(u, kak, 3, tq.DEFAULT_ATOL, None, basis, 1e-9)
        old = legacy_try_synthesize(u, kak, 3, tq.DEFAULT_ATOL, None, basis, 1e-9)
        assert new[1] <= 1e-9
        assert len(new[0]) <= min(bound, len(old[0]))


@pytest.mark.parametrize(("theta", "outer"), [(0.0, 0), (0.3, 0), (1.2, 0), (-2.0, 2)])
def test_swap_face_template(theta: float, outer: int) -> None:
    """SWAP·ZZ(θ) blocks (a routed ZZ / CP) need one inner rz (+ outer rz)."""
    import quspect.synthesis.two_qubit as tq

    swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
    zz = np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))
    u = swap @ zz
    with tq.swap_face_templates():
        _check_swap_face(u, outer)


@pytest.mark.parametrize("theta", [0.0, 0.3, 1.2, -2.0])
def test_swap_face_independent_of_kak_gauge(theta: float) -> None:
    """The result does not depend on which gauge the KAK factors land in.

    On the SWAP face ``L·Can·R = Can`` for ``L = Rz(-ψ)⊗Rz(-φ)``,
    ``R = Rz(φ)⊗Rz(ψ)`` (times ``{I, X⊗X}`` on both sides), and the
    eigensolver may return KAK factors anywhere in that family (OpenBLAS
    on arm64 Linux leaves ``Z`` before q0 and after q1 of SWAP·ZZ(0.3)).
    """
    import quspect.synthesis.two_qubit as tq

    swap = np.eye(4)[[0, 2, 1, 3]].astype(complex)
    u = swap @ np.diag(np.exp(-0.5j * theta * np.array([1, -1, -1, 1])))
    phase, k1, v, k2 = tq._kak4(u)
    for basis in (tq.TwoQubitBasis("cx"), tq.TwoQubitBasis("cx", "zsx")):
        counts = set()
        for phi, psi in (
            (0.0, 0.0),
            (0.7, 0.7),
            (-1.9, -1.9),
            (np.pi, 0.0),
            (0.4, -2.2),
        ):
            for coset in (tq._I2, tq._X, tq._Y):
                left = np.kron(tq._rz(-psi) @ coset, tq._rz(-phi) @ coset)
                right = np.kron(coset @ tq._rz(phi), coset @ tq._rz(psi))
                kak = (phase, k1 @ left, v, right @ k2)
                can = tq.canonical_gate(*v)
                assert np.allclose(left @ can @ right, can)
                ops, err = tq._try_synthesize(
                    u, kak, 3, tq.DEFAULT_ATOL, None, basis, 1e-9, swap_face=True
                )
                assert err <= 1e-9
                counts.add(sum(1 for name, _, _ in ops if name != "cx"))
        assert len(counts) == 1, counts


def _check_swap_face(u: np.ndarray, outer: int) -> None:
    import quspect.synthesis.two_qubit as tq

    for basis in (tq.TwoQubitBasis("cx"), tq.TwoQubitBasis("cx", "zsx")):
        ops = tq.synthesize_two_qubit_ops(u, basis=basis)
        assert sum(1 for name, _, _ in ops if name == "cx") == 3
        assert sum(1 for name, _, _ in ops if name != "cx") <= 1 + outer
    ops = tq.synthesize_two_qubit_ops(u)  # CZ: Hadamards + one rz
    assert sum(1 for name, _, _ in ops if name not in ("cz", "h")) <= 1 + outer
    # dressed with random locals: still exact
    rng = np.random.default_rng(3)
    for _ in range(5):
        left = np.kron(*(random_unitary(2, rng) for _ in range(2)))
        right = np.kron(*(random_unitary(2, rng) for _ in range(2)))
        v = left @ u @ right
        for basis in (tq.TwoQubitBasis(), tq.TwoQubitBasis("cx", "zsx")):
            qc = tq.synthesize_two_qubit(v, basis=basis)
            assert phase_distance(circuit_unitary(qc, 2), v) < 1e-8
