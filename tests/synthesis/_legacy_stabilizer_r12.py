"""Frozen copy of the round-12 KAK stabilizer choice (``a329090``).

Reference implementation for the differential tests in
``test_stabilizer_differential.py``: the batched scoring / early-exit
emission in :func:`quspect.synthesis.two_qubit._try_synthesize` must return
exactly what :func:`legacy_try_synthesize` returns (same gate tuples, same
error).

Copied verbatim from ``quspect/synthesis/two_qubit.py`` (``_try_synthesize``,
``_stabilizer_candidates``, ``_axis_angle``, ``_axis_rot``,
``_emit_fixed_layers``); functions were renamed with a ``legacy_`` prefix
and the shared helpers are imported.  Do not edit.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from quspect.synthesis.two_qubit import (
    _CZ4,
    _DEFAULT_BASIS,
    _I2,
    _X,
    _Y,
    _Z,
    DEFAULT_VERIFY_ATOL,
    SynthOp,
    TwoQubitBasis,
    _emit_layers,
    _factor_local,
    _kak4,
    _layers_unitary,
    _template,
    entangler,
    euler_ops,
)


def legacy_try_synthesize(
    u: np.ndarray,
    kak_u: tuple[float, np.ndarray, np.ndarray, np.ndarray],
    k: int,
    atol: float,
    template_coords: np.ndarray | None = None,
    basis: TwoQubitBasis = _DEFAULT_BASIS,
    verify_atol: float = DEFAULT_VERIFY_ATOL,
) -> tuple[list[SynthOp], float]:
    """Build the ``k``-entangler circuit for ``u``; return ``(ops, error)``.

    ``error`` is :func:`_distance_up_to_phase` between ``u`` and the unitary
    of the *emitted* gate list, so it includes the dropped Weyl coordinates
    of a reduced template as well as every Euler rotation snapped to zero
    (or to ``θ = π/2, π``) by the Euler synthesis. Callers reject results
    whose error exceeds the verification tolerance.

    For a non-CZ entangler each template CZ is dressed into the entangler
    (see :class:`Entangler`); asymmetric entanglers are tried in every
    orientation combination and the one with the fewest gates is returned.
    """
    _, k1u, vu, k2u = kak_u
    layers = _template(k, vu if template_coords is None else template_coords)
    v_mat = _layers_unitary(layers)
    _, k1v, vv, k2v = _kak4(v_mat)
    # u = φ_u k1u Can k2u and V = φ_v k1v Can k2v
    #   ⇒ u ∝ (k1u S k1v^†) V (k2v^† S^† k2u) for every local S that
    #   commutes with Can.  Degenerate chamber points have continuous
    #   stabilizers (e.g. V⊗V for a = b = c); the eigensolver then returns
    #   arbitrary k1u, k2u, so S is chosen to make the outer layers short.
    a = _factor_local(k1u)
    b = _factor_local(k1v)
    d = _factor_local(k2u)
    c = _factor_local(k2v)
    b_dag = (b[0].conj().T, b[1].conj().T)
    c_dag = (c[0].conj().T, c[1].conj().T)

    def fixed(
        stab: tuple[np.ndarray, np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        lefts = [a[i] @ stab[i] @ b_dag[i] for i in (0, 1)]
        rights = [c_dag[i] @ stab[i].conj().T @ d[i] for i in (0, 1)]
        out = [(layers[0][0] @ rights[0], layers[0][1] @ rights[1])]
        out += [(lft, rgt) for lft, rgt in layers[1:]]
        out[-1] = (lefts[0] @ out[-1][0], lefts[1] @ out[-1][1])
        return out

    # Score the candidates by the Euler lengths of their outer layers (the
    # inner ones do not depend on S) and emit the identity plus the best
    # strictly better-scoring ones.
    scored: list[tuple[int, int, list[tuple[np.ndarray, np.ndarray]]]] = []
    for idx, stab in enumerate(legacy_stabilizer_candidates(k, vv, a, b, c, d)):
        factored = fixed(stab)
        outer = (factored[0], factored[-1]) if k else (factored[0],)
        score = sum(
            len(euler_ops(m, 0, basis=basis.euler, atol=atol))
            for pair in outer
            for m in pair
        )
        scored.append((score, idx, factored))
    chosen = [scored[0]]
    chosen += sorted(
        (t for t in scored[1:] if t[0] < scored[0][0]), key=lambda t: (t[0], t[1])
    )[:LEGACY_STABILIZER_EMITS]
    best: tuple[list[SynthOp], float] | None = None
    best_key: tuple[bool, int] | None = None
    for _, _, factored in chosen:
        result = legacy_emit_fixed_layers(u, factored, k, atol, basis, verify_atol)
        key = (result[1] > verify_atol, len(result[0]))
        if best_key is None or key < best_key:
            best, best_key = result, key
    assert best is not None
    return best


#: Stabilizer candidates (besides the identity) fully emitted per template.
LEGACY_STABILIZER_EMITS = 4

_Pair = tuple[np.ndarray, np.ndarray]
_PAULI_LOCAL: tuple[_Pair, ...] = ((_X, _X), (_Y, _Y), (_Z, _Z))


def _axis_angle(w: np.ndarray, pauli: np.ndarray) -> float:
    """Angle ``φ`` of the rotation ``exp(-iφ/2·P)`` closest to ``w`` (up to phase)."""
    x = complex(np.trace(w))
    y = 1j * complex(np.trace(pauli @ w))
    return float(math.atan2(2.0 * (x * y.conjugate()).real, abs(x) ** 2 - abs(y) ** 2))


def _axis_rot(pauli: np.ndarray, phi: float) -> np.ndarray:
    return np.asarray(math.cos(phi / 2) * _I2 - 1j * math.sin(phi / 2) * pauli)


def legacy_stabilizer_candidates(
    k: int,
    v: np.ndarray,
    k1u: _Pair,
    k1v: _Pair,
    k2v: _Pair,
    k2u: _Pair,
) -> list[_Pair]:
    """Local unitaries ``S = s0⊗s1`` commuting with ``Can(v)`` worth trying.

    The factors of the KAK locals are given as ``(q0, q1)`` pairs.  The
    identity and the Pauli pairs ``P⊗P`` (which commute with every
    ``Can``); for degenerate chamber points (``a = b = c``: ``V⊗V``;
    ``b = c = 0``: ``Rx⊗Rx``; ``a = b``: ``Rz(φ)⊗Rz(φ)``; ``b = c``:
    ``Rx(φ)⊗Rx(φ)``) the elements that turn one outer single-qubit factor
    into the identity (or the closest rotation), times the discrete part
    of the stabilizer outside that family.
    """
    eye: _Pair = (_I2, _I2)
    if k == 0:
        return [eye]
    a, b, c = (float(x) for x in v)
    tol = 1e-9
    if abs(a - b) > tol and abs(b - c) > tol and abs(b) > tol:
        return [eye, *_PAULI_LOCAL]
    # s makes the outer factor identity: left_i = a_i s b_i^†, right_i = c_i^† s^† d_i
    targets = [
        (0, k1u[0].conj().T @ k1v[0]),
        (1, k1u[1].conj().T @ k1v[1]),
        (0, k2u[0] @ k2v[0].conj().T),
        (1, k2u[1] @ k2v[1].conj().T),
    ]
    cands: list[_Pair] = [eye]
    # discrete part of the stabilizer not already in the continuous family
    paulis: tuple[_Pair, ...]
    if abs(a - b) <= tol and abs(b - c) <= tol:
        cands += [(w, w) for _, w in targets]
        paulis = _PAULI_LOCAL  # V·P ⊗ V·P: other members of the same family
    elif abs(b) <= tol and abs(c) <= tol:
        paulis = (_PAULI_LOCAL[2],)
        phis: list[list[float]] = [[0.0], [0.0]]
        for slot, w in targets:
            phis[slot].append(_axis_angle(w, _X))
        for p0 in phis[0]:
            for p1 in phis[1]:
                if p0 or p1:
                    cands.append((_axis_rot(_X, p0), _axis_rot(_X, p1)))
    else:
        pauli = _Z if abs(a - b) <= tol else _X
        for _, w in targets:
            r = _axis_rot(pauli, _axis_angle(w, pauli))
            cands.append((r, r))
        paulis = (_PAULI_LOCAL[0] if abs(a - b) <= tol else _PAULI_LOCAL[2],)
    return cands + [(s0 @ p0, s1 @ p1) for s0, s1 in cands for p0, p1 in paulis]


def legacy_emit_fixed_layers(
    u: np.ndarray,
    factored: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    atol: float,
    basis: TwoQubitBasis,
    verify_atol: float,
) -> tuple[list[SynthOp], float]:
    """Emit factored CZ-template layers in ``basis`` (see :func:`_try_synthesize`)."""
    if basis.entangler == "cz":
        return _emit_layers(u, factored, [(_CZ4, None)] * k, atol, basis)
    ent = entangler(basis.entangler)
    choices = (False,) if ent.symmetric else (False, True)
    best: tuple[list[SynthOp], float] | None = None
    best_key: tuple[bool, int] | None = None
    for combo in product(choices, repeat=k):
        dressed = [list(f) for f in factored]
        gates: list[tuple[np.ndarray, tuple[int, int] | None]] = []
        for i, rev in enumerate(combo):
            mat, dress = ent.oriented(rev)
            # CZ = (A0⊗A1) E (B0⊗B1): B joins layer i (after it in time),
            # A joins layer i + 1 (before it in time).
            dressed[i] = [
                dress.before[0] @ dressed[i][0],
                dress.before[1] @ dressed[i][1],
            ]
            dressed[i + 1] = [
                dressed[i + 1][0] @ dress.after[0],
                dressed[i + 1][1] @ dress.after[1],
            ]
            gates.append((mat, (1, 0) if rev else (0, 1)))
        result = _emit_layers(u, [(a, b) for a, b in dressed], gates, atol, basis)
        # 先要求通过校验，再取门数最少的朝向组合
        key = (result[1] > verify_atol, len(result[0]))
        if best_key is None or key < best_key:
            best, best_key = result, key
    assert best is not None
    return best
