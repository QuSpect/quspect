"""Basis-aware synthesis of single-qubit unitaries.

:func:`synthesize_one_qubit` turns an arbitrary 2×2 unitary into a short
gate sequence that uses only the gates of a target basis, up to a global
phase. It is used by the DAG optimizer (single-qubit run resynthesis) and
by the decomposition rules for ``u3``/``u2`` on ``{h, rz}``-style bases.

Supported synthesis forms (the shortest valid one is returned):

* ``[]`` for the identity, a single named basis gate (``h``, ``x``, ``s``…)
  when the unitary equals it, ``rz``/``p``/``u1`` (or a Clifford+T word)
  for diagonal unitaries;
* ``u3`` when available;
* ZXZ with ``rx`` and ZYZ with ``ry`` (three gates at most);
* ``rz · h · rz`` (or ``rz · sx · rz`` / ``rz · sxdg · rz``) when the
  unitary is a "half" rotation (``|u00| = 1/√2``), ``x · rz`` (or
  ``sx · sx · rz``) for anti-diagonal unitaries;
* the general ``rz · h · rz · h · rz`` form (``RX(b) = H·RZ(b)·H``) and,
  for IBM-style ``{rz, sx}`` bases, the ZSX form ``rz · sx · rz · sx · rz``
  (``U3(θ, φ, λ) ∝ RZ(φ+π)·SX·RZ(θ+π)·SX·RZ(λ)``).

Identity rotations are dropped, so every form can come out shorter.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection

import numpy as np

from quspect.gates.gate import Gate
from quspect.synthesis._memo import BoundedCache, matrix_key

#: Absolute tolerance on angles / matrix entries.
_ATOL: float = 1e-9

_TWO_PI = 2.0 * math.pi


def _normalize_angle(theta: float) -> float:
    """Map an angle to ``(-π, π]``."""
    t = math.fmod(theta, _TWO_PI)
    if t <= -math.pi:
        t += _TWO_PI
    elif t > math.pi:
        t -= _TWO_PI
    return t


def is_identity_up_to_phase(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    """Return whether a 2×2 unitary equals the identity up to global phase.

    Args:
        matrix: A 2×2 unitary matrix.
        atol: Absolute tolerance.

    Returns:
        True if ``matrix ≈ e^{iα}·I``.

    Example:
        >>> import numpy as np
        >>> is_identity_up_to_phase(-np.eye(2))
        True
        >>> is_identity_up_to_phase(np.array([[0, 1], [1, 0]]))
        False
    """
    m = np.asarray(matrix, dtype=complex)
    return bool(
        abs(m[0, 1]) < atol
        and abs(m[1, 0]) < atol
        and abs(m[0, 0] - m[1, 1]) < atol
        and abs(abs(m[0, 0]) - 1.0) < atol
    )


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = _ATOL) -> bool:
    """Return whether two 2×2 unitaries agree up to a global phase.

    The global phase is aligned on the largest-magnitude entry of ``a`` and
    the matrices are then compared element-wise: ``max|a - e^{iφ}·b| < atol``.

    A trace overlap ``|tr(a†b)|/2`` must not be used here: for two
    rotations differing by ``θ`` it gives ``1 - overlap ≈ θ²/8``, i.e. a
    quadratic measure. Compared against a linear tolerance it accepted
    candidates that silently dropped rotations as large as ~1e-3 rad
    (QFT controlled phases ``π/2^k`` were lost). The element-wise distance
    is linear in ``θ`` (dropping ``RZ(θ)`` costs ``≈ θ/2``).

    Tolerance budget: each accepted replacement deviates from the run it
    replaces by at most ``~2·atol`` in operator norm, and such errors add
    at most linearly along a circuit. With the default ``atol = 1e-9`` a
    circuit with ``N = 10⁴`` resynthesized runs keeps
    ``‖Δψ‖ ≤ 2·10⁻⁵``, hence ``1 - F ≤ ‖Δψ‖² ≈ 4·10⁻¹⁰`` — far below a
    ``10⁻⁶`` state-fidelity budget — while staying well above the
    ``~10⁻¹⁵`` floating-point noise of the candidate matrices.
    """
    k = int(np.argmax(np.abs(a)))
    ak = complex(a.flat[k])
    bk = complex(b.flat[k])
    if abs(bk) < 0.5 * abs(ak):
        # The largest entry of a unitary has magnitude ≥ 1/√2; a matching
        # entry of b that small means the matrices differ by far more than
        # any tolerance.
        return False
    ratio = ak / bk
    phase = ratio / abs(ratio)
    return bool(np.max(np.abs(a - phase * b)) < atol)


def _sequence_matrix(gates: list[Gate]) -> np.ndarray:
    """Return the unitary of a gate sequence given in circuit order."""
    from quspect.gates._matrix_utils import gate_matrix

    m = np.eye(2, dtype=complex)
    for g in gates:
        m = np.asarray(gate_matrix(g), dtype=complex) @ m
    return m


class _Emitter:
    """Emit basis gates for primitive rotations."""

    def __init__(self, basis: frozenset[str]) -> None:
        from quspect.gates.parameterized import RX, RY, RZ, U1, U3, P
        from quspect.gates.standard import SDG, SX, SXDG, TDG, H, S, T, X, Z

        self.basis = basis
        self._rz_maker: Callable[[float], Gate] | None = None
        if "rz" in basis:
            self._rz_maker = RZ.with_params
        elif "p" in basis:
            self._rz_maker = P.with_params
        elif "u1" in basis:
            self._rz_maker = U1.with_params
        self._clifford_t: dict[int, list[Gate]] = {}
        # Z rotations by k·π/4 as words over {t, tdg, s, sdg, z}.
        words: dict[int, list[list[Gate]]] = {
            1: [[T]],
            2: [[S], [T, T]],
            3: [[S, T], [Z, TDG]],
            4: [[Z], [S, S]],
            -1: [[TDG]],
            -2: [[SDG], [TDG, TDG]],
            -3: [[SDG, TDG], [Z, T]],
        }
        for k, options in words.items():
            for word in options:
                if all(g.name in basis for g in word):
                    self._clifford_t[k] = word
                    break
        self.h = H if "h" in basis else None
        self.x = X if "x" in basis else None
        self.sx = SX if "sx" in basis else None
        self.sxdg = SXDG if "sxdg" in basis else None
        self.rx = RX.with_params if "rx" in basis else None
        self.ry = RY.with_params if "ry" in basis else None
        self.u3 = U3.with_params if "u3" in basis else None
        # Named basis gates in candidate order with their matrices and entry
        # magnitudes (for the exact pre-filter of :func:`_named_match`).
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.standard import Y

        self.named: list[tuple[Gate, np.ndarray, np.ndarray]] = []
        for g in (H, X, Y, Z, S, SDG, T, TDG, SX, SXDG):
            if g.name in basis:
                gm = np.asarray(gate_matrix(g), dtype=complex)
                self.named.append((g, gm, np.abs(gm).ravel()))

    def rz(self, theta: float) -> list[Gate] | None:
        """Return gates for ``RZ(theta)`` up to phase, or None."""
        t = _normalize_angle(theta)
        if abs(t) < _ATOL or abs(abs(t) - _TWO_PI) < _ATOL:
            return []
        if self._rz_maker is not None:
            return [self._rz_maker(t)]
        k = t / (math.pi / 4.0)
        kr = round(k)
        if abs(k - kr) < 1e-9:
            if kr == -4:
                kr = 4
            word = self._clifford_t.get(int(kr))
            if word is not None:
                return list(word)
        return None


def _zyz_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return ``(theta, phi, lam)`` with ``U ∝ RZ(phi)·RY(theta)·RZ(lam)``."""
    from quspect.dag.passes.single_qubit_fusion import _decompose_u3

    return _decompose_u3(np.asarray(matrix, dtype=complex))


#: Slack of the named-gate pre-filter over ``_ATOL``: bounds the rounding of
#: the magnitudes and of the phase normalisation in :func:`_equal_up_to_phase`
#: (all ~1e-16).
_NAMED_SLACK = 1e-12


def _named_match(m: np.ndarray, em: _Emitter) -> Gate | None:
    """First named basis gate equal to ``m`` up to phase (as in the full scan).

    ``_equal_up_to_phase(G, m)`` needs ``|G_i - e^{iφ} m_i| < _ATOL`` for
    every entry, hence ``||G_i| - |m_i|| < _ATOL`` up to rounding; gates whose
    entry magnitudes differ by more than ``_ATOL + _NAMED_SLACK`` are skipped
    without changing the result.
    """
    am = np.abs(m).ravel()
    limit = _ATOL + _NAMED_SLACK
    for g, gm, ag in em.named:
        d = am - ag
        if d.max() > limit or d.min() < -limit:
            continue
        if _equal_up_to_phase(gm, m):
            return g
    return None


def _candidates(m: np.ndarray, em: _Emitter) -> list[list[Gate] | None]:
    """Generate candidate gate sequences (circuit order) for ``m``."""
    cands: list[list[Gate] | None] = []
    if is_identity_up_to_phase(m, 1e-9):
        return [[]]

    # Single named gate.
    g = _named_match(m, em)
    if g is not None:
        cands.append([g])
        return cands

    u00, u01, u10, u11 = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    # Diagonal: RZ(arg(u11/u00)).
    if abs(u01) < _ATOL and abs(u10) < _ATOL:
        cands.append(em.rz(float(np.angle(u11 / u00))))
        return cands

    if em.u3 is not None:
        theta, phi, lam = _zyz_angles(m)
        cands.append([em.u3(theta, phi, lam)])

    # Anti-diagonal: X then RZ(arg(u10/u01)).
    if abs(u00) < _ATOL and abs(u11) < _ATOL:
        a = float(np.angle(u10 / u01))
        tail = em.rz(a)
        if em.x is not None and tail is not None:
            cands.append([em.x, *tail])
        if em.sx is not None and tail is not None:
            cands.append([em.sx, em.sx, *tail])
        if em.h is not None:
            mid = em.rz(math.pi)
            if mid is not None and tail is not None:
                cands.append([em.h, *mid, em.h, *tail])

    # Half rotation: U ∝ RZ(a)·H·RZ(c) (circuit: rz(c), h, rz(a)).
    if em.h is not None and abs(abs(u00) - math.sqrt(0.5)) < 1e-9:
        a = float(np.angle(u10 / u00))
        c = float(np.angle(u01 / u00))
        first, last = em.rz(c), em.rz(a)
        if first is not None and last is not None:
            cands.append([*first, em.h, *last])

    theta, phi, lam = _zyz_angles(m)
    # ZYZ: circuit rz(lam), ry(theta), rz(phi).
    if em.ry is not None:
        first, last = em.rz(lam), em.rz(phi)
        if first is not None and last is not None:
            cands.append([*first, em.ry(theta), *last])
    # ZXZ: U ∝ RZ(phi + π/2)·RX(theta)·RZ(lam - π/2), and the equivalent
    # RZ(phi - π/2)·RX(-theta)·RZ(lam + π/2).
    zxz = [
        (lam - math.pi / 2.0, theta, phi + math.pi / 2.0),
        (lam + math.pi / 2.0, -theta, phi - math.pi / 2.0),
    ]
    for c, b, a in zxz:
        first, last = em.rz(c), em.rz(a)
        if first is None or last is None:
            continue
        if em.rx is not None:
            cands.append([*first, em.rx(b), *last])
        if em.h is not None:
            mid = em.rz(b)
            if mid is not None:
                cands.append([*first, em.h, *mid, em.h, *last])
        # Half rotation with a native √X: RX(±π/2) ∝ SX / SXdg.
        if em.sx is not None and abs(b - math.pi / 2.0) < 1e-9:
            cands.append([*first, em.sx, *last])
        if em.sxdg is not None and abs(b + math.pi / 2.0) < 1e-9:
            cands.append([*first, em.sxdg, *last])
    # ZSX: U ∝ RZ(phi + π)·SX·RZ(theta + π)·SX·RZ(lam) (circuit order
    # rz(lam), sx, rz(theta + π), sx, rz(phi + π)).
    if em.sx is not None:
        first, mid, last = em.rz(lam), em.rz(theta + math.pi), em.rz(phi + math.pi)
        if first is not None and mid is not None and last is not None:
            cands.append([*first, em.sx, *mid, em.sx, *last])
    return cands


def synthesize_one_qubit(
    matrix: np.ndarray, basis_gates: Collection[str]
) -> list[Gate] | None:
    """Synthesize a single-qubit unitary into a short sequence of basis gates.

    The result implements ``matrix`` up to a global phase and only uses
    gate names contained in ``basis_gates``. All candidate forms listed in
    the module docstring are tried and verified numerically; the shortest
    valid sequence is returned.

    Args:
        matrix: A 2×2 unitary matrix.
        basis_gates: Names of the allowed gates (e.g.
            ``{"h", "rz", "cz"}``). Multi-qubit names are ignored.

    Returns:
        The gate sequence in circuit (time) order, ``[]`` for the
        identity, or ``None`` if the unitary cannot be expressed in the
        basis by the supported forms.

    Raises:
        ValueError: If ``matrix`` is not 2×2.

    Example:
        >>> from quspect.gates import gate_matrix
        >>> from quspect.gates.parameterized import U3
        >>> seq = synthesize_one_qubit(
        ...     gate_matrix(U3.with_params(0.3, 0.2, 0.1)), {"h", "rz", "cz"})
        >>> [g.name for g in seq]
        ['rz', 'h', 'rz', 'h', 'rz']
    """
    m = np.asarray(matrix, dtype=complex)
    if m.shape != (2, 2):
        raise ValueError(f"Expected a 2×2 matrix, got shape {m.shape}.")
    basis = frozenset(basis_gates)
    key = (matrix_key(m), basis)
    cached = _SYNTH_1Q_CACHE.get(key)
    if cached is not None:
        return list(cached) if isinstance(cached, tuple) else None
    best = _synthesize_one_qubit(m, basis)
    _SYNTH_1Q_CACHE.put(key, False if best is None else tuple(best))
    return best


#: Cache of :func:`synthesize_one_qubit` keyed on the exact matrix bytes and
#: the basis; ``False`` records "not expressible in this basis".
_SYNTH_1Q_CACHE: BoundedCache[tuple[object, ...], tuple[Gate, ...] | bool] = (
    BoundedCache()
)


#: One (immutable) :class:`_Emitter` per basis.
_EMITTERS: dict[frozenset[str], _Emitter] = {}


def _synthesize_one_qubit(m: np.ndarray, basis: frozenset[str]) -> list[Gate] | None:
    """Uncached core of :func:`synthesize_one_qubit`."""
    em = _EMITTERS.get(basis)
    if em is None:
        em = _EMITTERS[basis] = _Emitter(basis)
    best: list[Gate] | None = None
    for cand in _candidates(m, em):
        if cand is None:
            continue
        if best is not None and len(cand) >= len(best):
            continue
        if _equal_up_to_phase(_sequence_matrix(cand), m):
            best = cand
    return best
