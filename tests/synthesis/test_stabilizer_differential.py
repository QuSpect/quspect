"""Differential tests: the round-13 KAK stabilizer choice equals round 12.

Round 13 scores the stabilizer candidates of
:func:`quspect.synthesis.two_qubit._try_synthesize` with batched products
and a native Euler-length kernel and, for CZ, stops emitting at the first
verified candidate in score order.  The result (gate tuples and error)
must equal the frozen round-12 function
(:mod:`tests.synthesis._legacy_stabilizer_r12`) bit for bit, on generic
and on every kind of degenerate Weyl point — except the mirrored ones
(``b = -c``, round 17), where the stabilizer gained a continuous family:
there the result must verify and be no longer than round 12's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import quspect.synthesis.two_qubit as tq
from tests.synthesis._legacy_stabilizer_r12 import legacy_try_synthesize

Q = math.pi / 4
BASES = [
    tq.TwoQubitBasis(),
    tq.TwoQubitBasis("cx"),
    tq.TwoQubitBasis("cx", "zsx"),
    tq.TwoQubitBasis("cz", "zsx"),
]


def _haar(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return np.asarray(np.linalg.qr(a)[0])


def _unitaries(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pts: list[tuple[float, float, float]] = []
    for _ in range(25):
        a, b, c = sorted(rng.uniform(0, Q, 3), reverse=True)
        pts.append((a, b, c * float(rng.choice([-1, 1]))))
    for x in rng.uniform(0, Q, 6):
        pts += [(x, x, x), (x, 0, 0), (Q, x, x), (x, x, 0), (Q, x, 0), (x, x, -x)]
    pts += [(Q, Q, Q), (Q, 0, 0), (0, 0, 0), (Q, Q, 0)]
    out = []
    for p in pts:
        left = np.kron(_haar(rng, 2), _haar(rng, 2))
        right = np.kron(_haar(rng, 2), _haar(rng, 2))
        out.append(left @ tq.canonical_gate(*p) @ right)
    # Clifford-dressed SWAP / Heisenberg blocks (exact degenerate points)
    h = tq._H
    out.append(np.kron(h, tq._I2) @ tq.canonical_gate(Q, Q, Q) @ np.kron(tq._I2, h))
    out.append(np.kron(tq._S, h) @ tq.canonical_gate(0.3, 0.3, 0.3) @ np.kron(h, h))
    return out


def _mirrored(k: int, coords: np.ndarray) -> bool:
    """Whether the template point of ``_try_synthesize`` has ``b = -c < 0``."""
    vv = tq._kak4(tq._layers_unitary(tq._template(k, coords)))[2]
    return bool(vv[2] < -1e-9 and abs(vv[1] + vv[2]) <= 1e-9)


def _check(
    new: tuple[list[tq.SynthOp], float],
    old: tuple[list[tq.SynthOp], float],
    mirrored: bool,
    verify_atol: float,
) -> None:
    if not mirrored:
        assert repr(new[0]) == repr(old[0])
        assert new[1].hex() == old[1].hex()
        return
    if old[1] <= verify_atol:
        assert new[1] <= verify_atol
        assert len(new[0]) <= len(old[0])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("basis", BASES, ids=lambda b: f"{b.entangler}-{b.euler}")
def test_try_synthesize_matches_round_12(seed: int, basis: tq.TwoQubitBasis) -> None:
    for u in _unitaries(seed):
        kak = tq._kak4(u)
        k_min = tq._count_from_coords(kak[2], tq.DEFAULT_ATOL)
        for k in range(k_min, 4):
            coords: list[np.ndarray | None] = [None]
            if k == 3:
                mirrored = kak[2].copy()
                mirrored[2] = -mirrored[2]
                coords.append(mirrored)
            for c in coords:
                mirrored = _mirrored(k, kak[2] if c is None else c)
                for atol in (tq.DEFAULT_ATOL, tq._EXACT_EULER_ATOL):
                    old = legacy_try_synthesize(u, kak, k, atol, c, basis, 1e-9)
                    new = tq._try_synthesize(u, kak, k, atol, c, basis, 1e-9)
                    _check(new, old, mirrored, 1e-9)


def test_failed_verification_falls_back_like_round_12() -> None:
    """With an impossible tolerance every candidate fails; same choice."""
    for u in _unitaries(7)[-12:]:
        kak = tq._kak4(u)
        for basis in BASES[:2]:
            old = legacy_try_synthesize(u, kak, 3, 1e-9, None, basis, 0.0)
            new = tq._try_synthesize(u, kak, 3, 1e-9, None, basis, 0.0)
            if not _mirrored(3, kak[2]):
                assert repr(new[0]) == repr(old[0])
                assert new[1].hex() == old[1].hex()


@pytest.mark.parametrize("euler", ["zxz", "zsx"])
@pytest.mark.parametrize("atol", [tq.DEFAULT_ATOL, tq._EXACT_EULER_ATOL])
def test_euler_lengths_match_euler_ops(euler: str, atol: float) -> None:
    rng = np.random.default_rng(11)
    mats = [_haar(rng, 2) for _ in range(500)]
    for theta in (0.0, 1e-9, 1.0000001e-9, 1e-12, tq._PI2, math.pi, -tq._PI2):
        for lam in (0.0, 1e-10, 0.4, -tq._PI2):
            mats.append(tq._rz(lam) @ tq._rx(theta) @ tq._rz(0.7))
            mats.append(tq._H @ tq._rz(theta) @ tq._H @ tq._rz(lam))
    mats += [tq._I2, tq._H, tq._X, tq._Y, tq._Z, tq._S, tq._SX, tq._RX90]
    got = tq._euler_lengths(np.array(mats), euler, atol)
    assert got == [len(tq.euler_ops(m, 0, basis=euler, atol=atol)) for m in mats]
