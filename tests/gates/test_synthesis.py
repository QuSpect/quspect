"""Tests for basis-aware single-qubit synthesis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quspect.gates import gate_matrix
from quspect.gates.parameterized import RX, RZ, U3
from quspect.gates.standard import H, S, T, X
from quspect.gates.synthesis import (
    _equal_up_to_phase,
    is_identity_up_to_phase,
    synthesize_one_qubit,
)

BASIS = frozenset({"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"})


def _seq_matrix(gates: list) -> np.ndarray:  # type: ignore[type-arg]
    m = np.eye(2, dtype=complex)
    for g in gates:
        m = gate_matrix(g) @ m
    return m


def _phase_aligned_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``max |a - e^{iφ} b|`` with φ aligned on the largest entry of a.

    A trace overlap ``|tr(a†b)|/2`` is quadratic in the angle difference
    (``1 - overlap ≈ θ²/8``) and cannot see a dropped 1e-6 rotation, so the
    comparison is element-wise (linear in θ).
    """
    k = int(np.argmax(np.abs(a)))
    ak, bk = a.flat[k], b.flat[k]
    if abs(bk) < 1e-12:
        return float("inf")
    phase = (ak / bk) / abs(ak / bk)
    return float(np.max(np.abs(a - phase * b)))


def _assert_equiv(a: np.ndarray, b: np.ndarray) -> None:
    assert _phase_aligned_distance(a, b) < 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_random_unitaries_use_at_most_five_basis_gates(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(100):
        m = gate_matrix(U3.with_params(*rng.uniform(-4, 4, 3)))
        seq = synthesize_one_qubit(m, BASIS)
        assert seq is not None
        assert len(seq) <= 5
        assert all(g.name in BASIS for g in seq)
        _assert_equiv(_seq_matrix(seq), m)


@pytest.mark.parametrize(
    ("gates", "expected_len"),
    [
        ([H, H], 0),
        ([RZ.with_params(2 * math.pi)], 0),
        ([H, RZ.with_params(0.3)], 2),
        ([H, S, H], 3),
        ([RX.with_params(math.pi)], 1),
        ([S, S], 1),
        ([H, X, H], 1),
        ([T, H, T], 3),
    ],
)
def test_special_cases_are_short(gates: list, expected_len: int) -> None:  # type: ignore[type-arg]
    m = _seq_matrix(gates)
    seq = synthesize_one_qubit(m, BASIS)
    assert seq is not None
    assert len(seq) == expected_len
    if seq:
        _assert_equiv(_seq_matrix(seq), m)
    else:
        assert is_identity_up_to_phase(m)


def test_clifford_t_basis_without_rz() -> None:
    """Z rotations by multiples of π/4 map to Clifford+T words."""
    basis = {"h", "s", "sdg", "t", "tdg", "z"}
    m = _seq_matrix([H, RZ.with_params(3 * math.pi / 4), H])
    seq = synthesize_one_qubit(m, basis)
    assert seq is not None
    assert all(g.name in basis for g in seq)
    _assert_equiv(_seq_matrix(seq), m)
    assert synthesize_one_qubit(gate_matrix(RZ.with_params(0.3)), basis) is None


def test_u3_basis_and_rx_basis() -> None:
    m = gate_matrix(U3.with_params(0.4, 0.5, 0.6))
    seq = synthesize_one_qubit(m, {"u3", "cx"})
    assert seq is not None and [g.name for g in seq] == ["u3"]
    seq = synthesize_one_qubit(m, {"rx", "rz"})
    assert seq is not None and len(seq) == 3
    _assert_equiv(_seq_matrix(seq), m)


def test_bad_shape_raises() -> None:
    with pytest.raises(ValueError):
        synthesize_one_qubit(np.eye(4), BASIS)


# --- Regression: tiny rotations must never be dropped -----------------------
# QFT-16/20 contains controlled phases down to π/2^19 ≈ 6e-6 rad. The old
# trace-overlap check accepted candidates that silently dropped them.

_TINY_ANGLES = [6e-6, 1e-6, 1e-7, 1e-8]


@pytest.mark.parametrize("eps", _TINY_ANGLES)
def test_equal_up_to_phase_detects_tiny_rotation(eps: float) -> None:
    m = gate_matrix(RZ.with_params(eps))
    assert not _equal_up_to_phase(np.eye(2, dtype=complex), m)
    assert not _equal_up_to_phase(
        gate_matrix(T), gate_matrix(RZ.with_params(math.pi / 4 + eps))
    )
    assert _equal_up_to_phase(m, np.exp(0.7j) * m)


@pytest.mark.parametrize("eps", _TINY_ANGLES)
@pytest.mark.parametrize(
    "gates",
    [
        lambda e: [H, RZ.with_params(e), H],
        lambda e: [T, RZ.with_params(e)],
        lambda e: [H, RZ.with_params(math.pi / 2 + e), H, S],
        lambda e: [RZ.with_params(e), H, T, H, RZ.with_params(-e / 3)],
        lambda e: [X, RZ.with_params(e)],
        lambda e: [H, RZ.with_params(e)],
    ],
)
def test_synthesis_keeps_tiny_rotations(eps: float, gates: object) -> None:
    assert callable(gates)
    m = _seq_matrix(gates(eps))
    seq = synthesize_one_qubit(m, BASIS)
    assert seq is not None
    _assert_equiv(_seq_matrix(seq), m)


def test_synthesis_drops_rotations_below_angle_tolerance() -> None:
    """Rotations far below the 1e-9 angle tolerance are still removed."""
    m = _seq_matrix([H, RZ.with_params(1e-13), H])
    assert synthesize_one_qubit(m, BASIS) == []
