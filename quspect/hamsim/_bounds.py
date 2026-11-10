"""Second-order Trotter error: the commutator bound with exact local norms.

Childs, Su, Tran, Wiebe and Zhu (PRX 11, 011020, 2021) bound one step of the
second-order formula of ``H = Σ_γ H_γ``, ``S₂(t) = Π_γ e^{-i t H_γ / 2} ·
Π_γ^{reversed} e^{-i t H_γ / 2}``, by::

    ‖S₂(t) − e^{−iHt}‖ ≤ |t|³/12 · Σ_γ ‖[H_{>γ}, [H_{>γ}, H_γ]]‖
                        + |t|³/24 · Σ_γ ‖[H_γ, [H_γ, H_{>γ}]]‖

with ``H_{>γ}`` the sum of the later summands.  The summands may be any
Hermitian operators.  This module evaluates the right-hand side more tightly
than a triangle inequality over all products of Pauli terms, without leaving
the theorem:

- the nested commutators are formed as operators (sums of Pauli strings), so
  equal strings merge and cancel before any norm is taken;
- the norm of such an operator is bounded by splitting its strings into
  clusters that act on at most :data:`EXACT_NORM_QUBITS` qubits and adding
  the exact norms of the clusters (triangle inequality between clusters
  only; an operator that fits in one cluster gets its exact norm);
- the same formula is also read with coarser summands.  Consecutive terms
  that commute with one another multiply to the exponential of their sum,
  so the product formula of the Pauli terms *is* the product formula of
  those groups, and the theorem applies to the groups as well.  The smaller
  of the two results is used.

Every step replaces a quantity by an upper bound of it, so the result is
still an upper bound of the Trotter error.  With the Pauli terms as
summands it is never above the plain triangle-inequality value.

The work is capped by counts, so that the result does not depend on the
machine: after :data:`CLUSTER_LIMIT` exact cluster norms the remaining norms
fall back to the sum of the moduli of the merged coefficients, and a
Hamiltonian that needs more than :data:`PRODUCT_LIMIT` Pauli-string products
is not refined at all.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from contextlib import suppress
from typing import TypeAlias

import numpy as np

from quspect.hamsim._terms import PauliTerm

#: ``(x, z)`` bit masks of the Hermitian Pauli string ``i^{|x∧z|} X^x Z^z``.
PauliKey: TypeAlias = tuple[int, int]
#: An operator as ``{Pauli string: coefficient}``.
PauliSum: TypeAlias = dict[PauliKey, complex]

#: Largest support (qubits) of a cluster whose norm is computed exactly
#: (a ``2^k × 2^k`` eigenvalue problem per cluster).
EXACT_NORM_QUBITS = 6

#: Exact cluster norms per call; later norms use the triangle inequality.
CLUSTER_LIMIT = 6_000

#: Pauli-string products per call; beyond it the bound is not refined.
PRODUCT_LIMIT = 3_000_000

_PHASES = (1 + 0j, 1j, -1 + 0j, -1j)

#: Relative size of an imaginary part that is treated as round-off when
#: deciding that a cluster is Hermitian.
_HERMITIAN_TOL = 1e-12


class _OverBudgetError(Exception):
    """The refined evaluation would exceed :data:`PRODUCT_LIMIT`."""


class _Budget:
    """Counters of the work done in one call."""

    def __init__(self) -> None:
        self.products = 0
        self.clusters = 0

    def spend_products(self, count: int) -> None:
        self.products += count
        if self.products > PRODUCT_LIMIT:
            raise _OverBudgetError


def pauli_key(letters: str, qubits: Sequence[int]) -> PauliKey:
    """Bit masks of a Pauli string given as letters on qubits."""
    x = z = 0
    for letter, qubit in zip(letters, qubits, strict=True):
        if letter in "XY":
            x |= 1 << qubit
        if letter in "ZY":
            z |= 1 << qubit
    return x, z


def _anticommute(a: PauliKey, b: PauliKey) -> bool:
    return bool(((a[0] & b[1]).bit_count() + (a[1] & b[0]).bit_count()) & 1)


def _product(a: PauliKey, b: PauliKey) -> tuple[PauliKey, complex]:
    """``P_a · P_b = phase · P_c``: the string ``c`` and the phase."""
    x, z = a[0] ^ b[0], a[1] ^ b[1]
    exponent = (
        (a[0] & a[1]).bit_count()
        + (b[0] & b[1]).bit_count()
        - (x & z).bit_count()
        + 2 * (a[1] & b[0]).bit_count()
    )
    return (x, z), _PHASES[exponent % 4]


def commutator(left: PauliSum, right: PauliSum) -> PauliSum:
    """``[left, right]`` of two sums of Pauli strings."""
    out: PauliSum = {}
    for a, coeff_a in left.items():
        for b, coeff_b in right.items():
            if _anticommute(a, b):
                product, phase = _product(a, b)
                out[product] = out.get(product, 0j) + 2.0 * coeff_a * coeff_b * phase
    return out


def _support_bits(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def pauli_sum_matrix(operator: PauliSum, qubits: Sequence[int]) -> np.ndarray:
    """Dense matrix of ``operator`` on ``qubits`` (``qubits[0]`` is bit 0)."""
    position = {qubit: index for index, qubit in enumerate(qubits)}
    covered = sum(1 << qubit for qubit in position)
    dim = 1 << len(qubits)
    basis = np.arange(dim, dtype=np.int64)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for (x, z), coeff in operator.items():
        if (x | z) & ~covered:
            raise ValueError("operator acts outside the given qubits")
        local_x = local_z = 0
        for qubit, index in position.items():
            local_x |= ((x >> qubit) & 1) << index
            local_z |= ((z >> qubit) & 1) << index
        # P |b> = i^{|x∧z|} (−1)^{|z∧b|} |b ⊕ x>
        sign = 1.0 - 2.0 * (np.bitwise_count(basis & local_z) & 1)
        phase = _PHASES[(local_x & local_z).bit_count() % 4]
        matrix[basis ^ local_x, basis] += coeff * phase * sign
    return matrix


def _cluster_norm(cluster: PauliSum, support: int) -> float:
    """Exact spectral norm of a few Pauli strings on a small support."""
    if len(cluster) == 1:
        return abs(next(iter(cluster.values())))
    matrix = pauli_sum_matrix(cluster, _support_bits(support))
    if all(abs(c.imag) <= _HERMITIAN_TOL * abs(c) for c in cluster.values()):
        # Real coefficients of Hermitian strings: a Hermitian matrix.
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0)
        return float(max(abs(eigenvalues[0]), abs(eigenvalues[-1])))
    return float(np.linalg.norm(matrix, 2))


def norm_upper_bound(
    operator: PauliSum,
    max_qubits: int | None = None,
    budget: _Budget | None = None,
) -> float:
    """Upper bound of the spectral norm of a sum of Pauli strings.

    The strings are ordered by their lowest qubit and packed greedily into
    clusters whose joint support has at most ``max_qubits`` qubits.  The
    result is the sum of the exact norms of the clusters; a string wider
    than ``max_qubits`` counts with the modulus of its coefficient.  When
    everything fits in one cluster the result is the norm itself.

    Args:
        operator: The sum of Pauli strings.
        max_qubits: Largest cluster support (default
            :data:`EXACT_NORM_QUBITS`).
        budget: Work counters of the call; once :data:`CLUSTER_LIMIT`
            cluster norms were computed, the strings left over count with
            the moduli of their coefficients.

    Returns:
        A number that is at least the spectral norm and at most the sum of
        the moduli of the coefficients.
    """
    if max_qubits is None:
        max_qubits = EXACT_NORM_QUBITS
    total = 0.0
    cluster: PauliSum = {}
    cluster_support = 0

    def flush() -> float:
        if not cluster:
            return 0.0
        if budget is not None and len(cluster) > 1:
            if budget.clusters >= CLUSTER_LIMIT:
                return float(sum(abs(c) for c in cluster.values()))
            budget.clusters += 1
        return _cluster_norm(cluster, cluster_support)

    def lowest(item: tuple[PauliKey, complex]) -> tuple[int, int]:
        support = item[0][0] | item[0][1]
        return ((support & -support).bit_length(), support)

    for key, coeff in sorted(operator.items(), key=lowest):
        support = key[0] | key[1]
        if support.bit_count() > max_qubits:
            total += abs(coeff)
            continue
        if (cluster_support | support).bit_count() > max_qubits:
            total += flush()
            cluster = {}
            cluster_support = 0
        cluster[key] = coeff
        cluster_support |= support
    return total + flush()


class _LaterTerms:
    """Hamiltonian terms by qubit, to find those after a position that
    overlap a support (only they can fail to commute with it)."""

    def __init__(
        self, keys: Sequence[PauliKey], coeffs: Sequence[float], budget: _Budget
    ) -> None:
        self.keys = keys
        self.coeffs = coeffs
        self.budget = budget
        self.by_qubit: dict[int, list[int]] = {}
        for index, (x, z) in enumerate(keys):
            for qubit in _support_bits(x | z):
                self.by_qubit.setdefault(qubit, []).append(index)

    def overlapping(self, support: int, after: int) -> set[int]:
        """Indices ``> after`` of the terms that share a qubit with ``support``."""
        found: set[int] = set()
        for qubit in _support_bits(support):
            members = self.by_qubit.get(qubit)
            if members:
                found.update(members[bisect_right(members, after) :])
        self.budget.spend_products(len(found))
        return found

    def commutator_with_later(self, operator: PauliSum, after: int) -> PauliSum:
        """``[H_{>after}, operator]``."""
        out: PauliSum = {}
        for key, coeff in operator.items():
            for index in self.overlapping(key[0] | key[1], after):
                other = self.keys[index]
                if _anticommute(other, key):
                    product, phase = _product(other, key)
                    value = 2.0 * self.coeffs[index] * coeff * phase
                    out[product] = out.get(product, 0j) + value
        return out


def commuting_runs(keys: Sequence[PauliKey]) -> list[tuple[int, int]]:
    """Split the terms into maximal runs ``[start, stop)`` of consecutive,
    mutually commuting terms (greedy, in formula order)."""
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(keys) + 1):
        if index == len(keys) or any(
            _anticommute(keys[index], keys[member]) for member in range(start, index)
        ):
            runs.append((start, index))
            start = index
    return runs


def _coefficient(later: _LaterTerms, runs: Sequence[tuple[int, int]]) -> float:
    """``Σ_γ ‖[H_>γ,[H_>γ,H_γ]]‖/12 + ‖[H_γ,[H_γ,H_>γ]]‖/24`` for the summands
    ``H_γ = Σ_{i in run γ} c_i P_i`` (norms through :func:`norm_upper_bound`)."""
    total = 0.0
    budget = later.budget
    for start, stop in runs:
        summand: PauliSum = {}
        for index in range(start, stop):
            key = later.keys[index]
            summand[key] = summand.get(key, 0j) + later.coeffs[index]
        inner = later.commutator_with_later(summand, stop - 1)
        if not inner:
            continue
        outer = later.commutator_with_later(inner, stop - 1)
        budget.spend_products(len(summand) * len(inner))
        # [H_γ, [H_γ, H_>γ]] = −[H_γ, inner]: the same norm.
        own = commutator(summand, inner)
        total += norm_upper_bound(outer, budget=budget) / 12.0
        total += norm_upper_bound(own, budget=budget) / 24.0
    return total


def second_order_coefficient(terms: Sequence[PauliTerm]) -> float | None:
    """The factor ``C`` of ``‖S₂(t) − e^{−iHt}‖ ≤ C |t|³`` for one step.

    Evaluates the bound of the module docstring for the Pauli terms as
    summands and for the runs of consecutive commuting terms as summands,
    and returns the smaller value.

    Args:
        terms: Hamiltonian terms in formula order.

    Returns:
        The factor, or ``None`` when the evaluation would exceed
        :data:`PRODUCT_LIMIT` Pauli-string products.
    """
    keys = [pauli_key(letters, qubits) for letters, qubits, _ in terms]
    coeffs = [float(coeff) for _, _, coeff in terms]
    later = _LaterTerms(keys, coeffs, _Budget())
    try:
        best = _coefficient(later, [(i, i + 1) for i in range(len(keys))])
    except _OverBudgetError:
        return None
    runs = commuting_runs(keys)
    if len(runs) < len(keys):
        with suppress(_OverBudgetError):  # else the first evaluation stands
            best = min(best, _coefficient(later, runs))
    return best
