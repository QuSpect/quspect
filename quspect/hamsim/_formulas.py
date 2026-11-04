"""Product formulas, term orders and Trotter-error estimates.

A rotation ``(letters, qubits, theta)`` stands for ``exp(-i theta P)``.

- order 1 (Lie–Trotter): ``Π_j exp(-i c_j dt P_j)`` in term order;
- order 2 (Strang): the half step forward, then the half step backward;
- order 4 (Suzuki): ``S2(p dt)² S2((1 − 4p) dt) S2(p dt)²`` with
  ``p = 1 / (4 − 4^{1/3})``.

Adjacent identical Paulis (the middle of a Strang step, the step
boundaries) are merged.  The error of a product formula depends on the
term order; :func:`order_terms` offers the input order and a
support-grouped order (terms on the same qubits adjacent, groups ordered by
a greedy colouring of their overlap graph), and :func:`state_error` /
:func:`commutator_bound` estimate the error of a concrete rotation
sequence or formula.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from quspect.hamsim._terms import PauliTerm

Rotation: TypeAlias = tuple[str, tuple[int, ...], float]

# 1 / (4 − 4^{1/3}), correctly rounded; a literal rather than ``4.0 ** (1 / 3)`` so that
# the angles do not depend on the platform's ``pow`` (bit-identical output everywhere).
SUZUKI4_P = 0.4144907717943757

TERM_ORDERS = ("input", "support")

#: Largest qubit count for :func:`state_error` (dense state vectors).
STATE_ERROR_MAX_QUBITS = 16


def _strang(terms: Sequence[PauliTerm], dt: float) -> list[Rotation]:
    half = [(p, q, c * dt / 2) for p, q, c in terms]
    return half + half[::-1]


def one_step(terms: Sequence[PauliTerm], dt: float, order: int) -> list[Rotation]:
    """Rotations of one step of length ``dt`` (not merged)."""
    if order == 1:
        return [(p, q, c * dt) for p, q, c in terms]
    if order == 2:
        return _strang(terms, dt)
    if order == 4:
        p = SUZUKI4_P
        out: list[Rotation] = []
        for f in (p, p, 1 - 4 * p, p, p):
            out += _strang(terms, f * dt)
        return out
    raise ValueError(f"unsupported product-formula order {order} (use 1, 2 or 4)")


def merge_adjacent(rotations: Sequence[Rotation]) -> list[Rotation]:
    """Merge neighbouring rotations about the same Pauli; drop zero angles."""
    out: list[Rotation] = []
    for p, q, th in rotations:
        if out and out[-1][0] == p and out[-1][1] == q:
            out[-1] = (p, q, out[-1][2] + th)
        else:
            out.append((p, q, th))
    return [r for r in out if r[2] != 0.0]


def product_formula(
    terms: Sequence[PauliTerm], time: float, steps: int, order: int
) -> list[Rotation]:
    """Rotation sequence of ``steps`` steps of the order-``order`` formula.

    Args:
        terms: Hamiltonian terms in the order the formula uses.
        time: Evolution time ``t`` of ``exp(-i H t)``.
        steps: Number of Trotter steps ``r ≥ 1``.
        order: 1, 2 or 4.

    Returns:
        The rotations in time order (first applied first).
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    step = one_step(terms, time / steps, order)
    return merge_adjacent(step * steps)


def order_terms(terms: Sequence[PauliTerm], strategy: str = "input") -> list[PauliTerm]:
    """Reorder Hamiltonian terms for the product formula.

    ``"input"`` keeps the given order.  ``"support"`` puts terms acting on
    the same qubits next to each other (so that e.g. ``XX + YY + ZZ`` on
    one bond becomes a single two-qubit block) and orders the groups by a
    largest-first greedy colouring of their overlap graph: groups of one
    colour act on disjoint qubits and form one layer.  Unlike reordering
    commuting terms, this changes the Trotter error; compare with
    :func:`state_error` or :func:`commutator_bound`.

    Args:
        terms: Hamiltonian terms.
        strategy: ``"input"`` or ``"support"``.

    Returns:
        The reordered terms.
    """
    if strategy == "input":
        return list(terms)
    if strategy != "support":
        raise ValueError(
            f"unknown term order {strategy!r}; expected one of {TERM_ORDERS}"
        )
    groups: dict[tuple[int, ...], list[PauliTerm]] = {}
    for t in terms:
        groups.setdefault(tuple(sorted(t[1])), []).append(t)
    keys = list(groups)
    by_qubit: dict[int, list[int]] = {}
    for i, k in enumerate(keys):
        for q in k:
            by_qubit.setdefault(q, []).append(i)
    nbrs: list[set[int]] = [set() for _ in keys]
    for members in by_qubit.values():
        for i in members:
            nbrs[i].update(members)
    for i in range(len(keys)):
        nbrs[i].discard(i)
    colour: dict[int, int] = {}
    for i in sorted(range(len(keys)), key=lambda i: -len(nbrs[i])):
        used = {colour[j] for j in nbrs[i] if j in colour}
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    order = sorted(range(len(keys)), key=lambda i: (colour[i], i))
    return [t for i in order for t in groups[keys[i]]]


# ---------------------------------------------------------------- error estimates


def _pauli_action(
    n: int, letters: str, qubits: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    x = z = 0
    ny = 0
    for c, q in zip(letters, qubits, strict=True):
        if c in "XY":
            x |= 1 << q
        if c in "ZY":
            z |= 1 << q
        ny += c == "Y"
    idx = np.arange(1 << n, dtype=np.int64)
    src = idx ^ x
    par = np.bitwise_count(src & z) & 1
    phase = (1j**ny) * (1.0 - 2.0 * par.astype(np.float64))
    return src, phase.astype(np.complex128)


def _test_states(n: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n) for _ in range(count)
    ]
    basis = np.zeros(1 << n, dtype=np.complex128)
    basis[int(("01" * n)[:n], 2)] = 1.0
    psi = np.stack([c / np.linalg.norm(c) for c in cols] + [basis], axis=1)
    return psi


def _apply_rotations(
    n: int, rotations: Sequence[Rotation], psi: np.ndarray
) -> np.ndarray:
    psi = np.array(psi, dtype=np.complex128, copy=True)
    cache: dict[tuple[str, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
    for p, q, th in rotations:
        key = (p, q)
        if key not in cache:
            cache[key] = _pauli_action(n, p, q)
        src, phase = cache[key]
        v = psi[src] * phase[:, None]
        psi *= np.cos(th)
        psi += (-1j * np.sin(th)) * v
    return psi


def _exact_evolve(
    n: int, terms: Sequence[PauliTerm], time: float, psi: np.ndarray
) -> np.ndarray:
    from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
    from scipy.sparse.linalg import expm_multiply  # type: ignore[import-untyped]

    dim = 1 << n
    rows = np.arange(dim)
    mat = csr_matrix((dim, dim), dtype=np.complex128)
    for p, q, c in terms:
        src, phase = _pauli_action(n, p, q)
        # (P ψ)[i] = phase[i] ψ[src[i]]  →  P[i, src[i]] = phase[i]
        mat = mat + csr_matrix((c * phase, (rows, src)), shape=(dim, dim))
    return np.asarray(expm_multiply(-1j * time * mat, psi))


def state_error(
    num_qubits: int,
    terms: Sequence[PauliTerm],
    time: float,
    rotations: Sequence[Rotation],
    *,
    samples: int = 2,
    seed: int = 20261102,
    reference: np.ndarray | None = None,
) -> float:
    """Worst state error of a rotation sequence against ``exp(-i H t)``.

    ``max_ψ min_φ ‖U ψ − e^{iφ} e^{−iHt} ψ‖`` over ``samples`` Haar-random
    states and the basis state ``|0101…⟩`` (dense simulation, at most
    :data:`STATE_ERROR_MAX_QUBITS` qubits).

    Args:
        num_qubits: Number of qubits.
        terms: Hamiltonian terms.
        time: Evolution time.
        rotations: The product formula to assess.
        samples: Number of random states.
        seed: Seed of the random states.
        reference: Precomputed ``exp(-i H t)`` applied to the test states
            (from an earlier call with the same states), to save the exact
            evolution.

    Returns:
        The largest error over the test states.
    """
    if num_qubits > STATE_ERROR_MAX_QUBITS:
        raise ValueError(
            f"state_error simulates at most {STATE_ERROR_MAX_QUBITS} qubits"
        )
    psi = _test_states(num_qubits, samples, seed)
    ref = (
        _exact_evolve(num_qubits, terms, time, psi) if reference is None else reference
    )
    out = _apply_rotations(num_qubits, rotations, psi)
    ov = np.abs(np.sum(np.conj(ref) * out, axis=0))
    return float(np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.minimum(ov, 1.0))).max())


def _bits(num_qubits: int, terms: Sequence[PauliTerm]) -> tuple[np.ndarray, np.ndarray]:
    words = (num_qubits + 63) // 64
    x = np.zeros((len(terms), words), dtype=np.uint64)
    z = np.zeros((len(terms), words), dtype=np.uint64)
    for i, (p, q, _) in enumerate(terms):
        for c, qi in zip(p, q, strict=True):
            w, b = divmod(qi, 64)
            if c in "XY":
                x[i, w] |= np.uint64(1 << b)
            if c in "ZY":
                z[i, w] |= np.uint64(1 << b)
    return x, z


def _anti(x: np.ndarray, z: np.ndarray, xi: np.ndarray, zi: np.ndarray) -> np.ndarray:
    s = np.bitwise_count((x & zi) ^ (z & xi)).sum(axis=-1)
    return np.asarray((s & 1) != 0)


def commutator_bound(
    num_qubits: int, terms: Sequence[PauliTerm], time: float, steps: int, order: int
) -> float:
    """Operator-norm Trotter error bound (commutator scaling) for order 1 or 2.

    Order 1: ``t²/(2r) Σ_{i<j} ‖[H_i, H_j]‖``.  Order 2 (Childs et al.,
    PRX 11, 011020, 2021): ``t³/(12 r²) Σ_i ‖[[H_i, H_{>i}], H_{>i}]‖ +
    t³/(24 r²) Σ_i ‖[[H_i, H_{>i}], H_i]‖`` with each nested commutator
    bounded by the triangle inequality over Pauli products.

    Args:
        num_qubits: Number of qubits.
        terms: Hamiltonian terms (in formula order).
        time: Evolution time.
        steps: Number of steps.
        order: 1 or 2.

    Returns:
        The bound.
    """
    return commutator_prefactor(num_qubits, terms, time, order) / float(steps**order)


def commutator_prefactor(
    num_qubits: int, terms: Sequence[PauliTerm], time: float, order: int
) -> float:
    """The step-independent factor ``C`` of :func:`commutator_bound`
    (``bound = C / steps**order``)."""
    m = len(terms)
    if m == 0:
        return 0.0
    if order not in (1, 2):
        raise ValueError("commutator_bound supports order 1 and 2")
    x, z = _bits(num_qubits, terms)
    c = np.abs(np.array([t[2] for t in terms], dtype=np.float64))
    if order == 1:
        total = 0.0
        for i in range(m - 1):
            a = _anti(x[i + 1 :], z[i + 1 :], x[i], z[i])
            total += 2.0 * c[i] * float(c[i + 1 :][a].sum())
        return time * time / 2.0 * total
    t1 = t2 = 0.0
    for i in range(m - 1):
        later = slice(i + 1, m)
        a = _anti(x[later], z[later], x[i], z[i])
        if not a.any():
            continue
        js = np.nonzero(a)[0] + i + 1
        # [H_i, H_j] = 2 c_i c_j P_i P_j for anticommuting pairs
        rx, rz = x[js] ^ x[i], z[js] ^ z[i]
        w = 2.0 * c[i] * c[js]
        for r in range(len(js)):
            ak = _anti(x[later], z[later], rx[r], rz[r])
            t1 += w[r] * 2.0 * float(c[later][ak].sum())
            if _anti(x[i : i + 1], z[i : i + 1], rx[r], rz[r])[0]:
                t2 += w[r] * 2.0 * c[i]
    return time**3 * (t1 / 12.0 + t2 / 24.0)
