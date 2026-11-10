"""The second-order commutator bound with exact local norms.

Every test compares against dense matrices: the Pauli-sum algebra against
matrix products, the norm bound against the spectral norm, and the Trotter
bound against the operator-norm error of the product formula itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import quspect.hamsim._bounds as bounds
from quspect.hamsim import commutator_bound, order_terms, product_formula
from quspect.hamsim._bounds import (
    PauliSum,
    commutator,
    commuting_runs,
    norm_upper_bound,
    pauli_key,
    pauli_sum_matrix,
    second_order_coefficient,
)
from quspect.hamsim._compile import _steps_for_eps
from quspect.hamsim._formulas import commutator_prefactor
from quspect.hamsim._terms import PauliTerm
from tests.hamsim.test_hamsim import (
    hamiltonian_unitary,
    heisenberg,
    hubbard_like,
    rotations_unitary,
)

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _dense(letters: str, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Dense Pauli string with qubit 0 as the lowest bit (rightmost factor)."""
    on = dict(zip(qubits, letters, strict=True))
    out = np.ones((1, 1), dtype=complex)
    for qubit in range(n - 1, -1, -1):
        out = np.kron(out, _PAULI[on.get(qubit, "I")])
    return out


def _random_strings(
    rng: np.random.Generator, n: int, count: int
) -> list[tuple[str, tuple[int, ...]]]:
    out = []
    for _ in range(count):
        width = int(rng.integers(1, n + 1))
        qubits = tuple(int(q) for q in rng.choice(n, size=width, replace=False))
        out.append(("".join(rng.choice(list("XYZ"), size=width)), qubits))
    return out


def _random_sum(
    rng: np.random.Generator, n: int, count: int, *, real: bool = False
) -> tuple[PauliSum, np.ndarray]:
    operator: PauliSum = {}
    matrix = np.zeros((2**n, 2**n), dtype=complex)
    for letters, qubits in _random_strings(rng, n, count):
        coeff = complex(rng.normal(), 0.0 if real else rng.normal())
        key = pauli_key(letters, qubits)
        operator[key] = operator.get(key, 0j) + coeff
        matrix += coeff * _dense(letters, qubits, n)
    return operator, matrix


def _random_terms(rng: np.random.Generator, n: int) -> list[PauliTerm]:
    """A random Hamiltonian; repeated strings and commuting runs do occur."""
    count = int(rng.integers(2, 9))
    terms: list[PauliTerm] = [
        (letters, qubits, float(rng.normal()))
        for letters, qubits in _random_strings(rng, n, count)
    ]
    if rng.random() < 0.3:  # repeat a string later in the formula
        letters, qubits, _ = terms[int(rng.integers(len(terms)))]
        terms.append((letters, qubits, float(rng.normal())))
    if rng.random() < 0.3:  # a run of commuting terms
        terms += [("Z", (q,), float(rng.normal())) for q in range(n)]
    return terms


def _legacy_coefficient(n: int, terms: list[PauliTerm]) -> float:
    """The triangle-inequality value (the refinement switched off)."""
    saved = bounds.PRODUCT_LIMIT
    bounds.PRODUCT_LIMIT = -1
    try:
        return commutator_prefactor(n, terms, 1.0, 2)
    finally:
        bounds.PRODUCT_LIMIT = saved


def _true_error(n: int, terms: list[PauliTerm], time: float, steps: int) -> float:
    formula = rotations_unitary(product_formula(terms, time, steps, 2), n)
    return float(np.linalg.norm(formula - hamiltonian_unitary(terms, time, n), 2))


# ---------------------------------------------------------------- Pauli-sum algebra


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_pauli_sum_matrix_matches_kronecker_products(n: int) -> None:
    rng = np.random.default_rng(900 + n)
    for _ in range(20):
        operator, matrix = _random_sum(rng, n, int(rng.integers(1, 6)))
        np.testing.assert_allclose(
            pauli_sum_matrix(operator, list(range(n))), matrix, atol=1e-12
        )


def test_pauli_sum_matrix_on_a_subset_of_qubits() -> None:
    operator = {pauli_key("XY", (5, 2)): 0.5, pauli_key("Z", (9,)): -1.5j}
    expected = 0.5 * _dense("XY", (1, 0), 3) - 1.5j * _dense("Z", (2,), 3)
    np.testing.assert_allclose(pauli_sum_matrix(operator, [2, 5, 9]), expected)
    with pytest.raises(ValueError, match="outside"):
        pauli_sum_matrix(operator, [2, 5])


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_commutator_matches_dense_commutator(n: int) -> None:
    rng = np.random.default_rng(910 + n)
    for _ in range(40):
        left, left_matrix = _random_sum(rng, n, int(rng.integers(1, 6)))
        right, right_matrix = _random_sum(rng, n, int(rng.integers(1, 6)))
        np.testing.assert_allclose(
            pauli_sum_matrix(commutator(left, right), list(range(n))),
            left_matrix @ right_matrix - right_matrix @ left_matrix,
            atol=1e-11,
        )


def test_commuting_runs_are_maximal_and_in_order() -> None:
    keys = [pauli_key(p, q) for p, q, _ in heisenberg(4)]  # XX.., YY.., ZZ..
    assert commuting_runs(keys) == [(0, 3), (3, 6), (6, 9)]
    by_bond = [pauli_key(p, (i, i + 1)) for i in range(3) for p in ("XX", "YY", "ZZ")]
    assert commuting_runs(by_bond) == [(0, 3), (3, 6), (6, 9)]
    assert commuting_runs([pauli_key("X", (0,)), pauli_key("Z", (0,))]) == [
        (0, 1),
        (1, 2),
    ]
    assert commuting_runs([]) == []


# ---------------------------------------------------------------- norm bound


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_norm_upper_bound_is_exact_when_the_operator_fits(n: int) -> None:
    rng = np.random.default_rng(920 + n)
    for real in (True, False):
        for _ in range(15):
            operator, matrix = _random_sum(rng, n, int(rng.integers(1, 8)), real=real)
            assert norm_upper_bound(operator) == pytest.approx(
                np.linalg.norm(matrix, 2), rel=1e-10, abs=1e-12
            )


@pytest.mark.parametrize("max_qubits", [1, 2, 3])
def test_norm_upper_bound_with_small_clusters_stays_an_upper_bound(
    max_qubits: int,
) -> None:
    rng = np.random.default_rng(930 + max_qubits)
    tighter = 0
    for _ in range(60):
        operator, matrix = _random_sum(rng, 5, int(rng.integers(2, 12)), real=True)
        bound = norm_upper_bound(operator, max_qubits)
        triangle = sum(abs(c) for c in operator.values())
        assert bound >= np.linalg.norm(matrix, 2) - 1e-10
        assert bound <= triangle + 1e-10
        tighter += bound < triangle - 1e-9
    assert tighter > 0 or max_qubits == 1


def test_norm_upper_bound_of_nothing_is_zero() -> None:
    assert norm_upper_bound({}) == 0.0


# ---------------------------------------------------------------- Trotter bound


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_bound_is_never_below_the_dense_error_and_never_above_the_old_bound(
    n: int,
) -> None:
    """Random Hamiltonians up to 5 qubits, both signs of ``t``."""
    rng = np.random.default_rng(940 + n)
    cases = 60 if n < 5 else 25
    ratios = []
    for _ in range(cases):
        terms = _random_terms(rng, n)
        time = float(rng.uniform(0.05, 1.2)) * (-1.0 if rng.random() < 0.4 else 1.0)
        steps = int(rng.integers(1, 5))
        bound = commutator_bound(n, terms, time, steps, 2)
        legacy = _legacy_coefficient(n, terms) * abs(time) ** 3 / steps**2
        assert bound >= _true_error(n, terms, time, steps) - 1e-9
        assert bound <= legacy * (1 + 1e-12)
        assert bound == commutator_bound(n, terms, -time, steps, 2)
        if legacy > 0:
            ratios.append(bound / legacy)
    assert min(ratios) < 0.9  # the refinement does something
    assert np.median(ratios) < 1.0


@pytest.mark.parametrize(
    "terms_and_qubits",
    [
        (heisenberg(4), 4),
        (heisenberg(5, periodic=True), 5),
        (order_terms(heisenberg(5), "support"), 5),
        ([(p, (i, i + 1), 1.0) for i in range(4) for p in ("XX", "YY", "ZZ")], 5),
        (hubbard_like(), 5),
        (
            [("ZZ", (i, i + 1), 1.0) for i in range(4)]
            + [("X", (i,), 0.8) for i in range(5)],
            5,
        ),
    ],
    ids=["heis4", "heis5-ring", "heis5-support", "heis5-bonds", "hubbard", "tfim5"],
)
def test_bound_holds_for_structured_hamiltonians(
    terms_and_qubits: tuple[list[PauliTerm], int],
) -> None:
    terms, n = terms_and_qubits
    for time, steps in ((1.0, 1), (-0.7, 2), (0.3, 5), (2.0, 3)):
        bound = commutator_bound(n, terms, time, steps, 2)
        assert bound >= _true_error(n, terms, time, steps) - 1e-9
        assert bound <= _legacy_coefficient(n, terms) * abs(time) ** 3 / steps**2


@pytest.mark.parametrize(
    ("max_qubits", "cluster_limit"), [(1, 6000), (2, 6000), (3, 6000), (6, 0), (6, 3)]
)
def test_bound_holds_when_norms_fall_back_to_the_triangle_inequality(
    monkeypatch: pytest.MonkeyPatch, max_qubits: int, cluster_limit: int
) -> None:
    """Clusters smaller than the operators, and the cluster budget used up."""
    monkeypatch.setattr(bounds, "CLUSTER_LIMIT", cluster_limit)
    monkeypatch.setattr(bounds, "EXACT_NORM_QUBITS", max_qubits)
    rng = np.random.default_rng(950 + max_qubits + cluster_limit)
    for _ in range(40):
        n = int(rng.integers(3, 6))
        terms = _random_terms(rng, n)
        time, steps = float(rng.uniform(0.1, 1.0)), int(rng.integers(1, 4))
        bound = commutator_bound(n, terms, time, steps, 2)
        assert bound >= _true_error(n, terms, time, steps) - 1e-9
        assert bound <= _legacy_coefficient(n, terms) * time**3 / steps**2 * (1 + 1e-12)


def test_product_budget_keeps_the_triangle_inequality_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terms = heisenberg(6)
    legacy = _legacy_coefficient(6, terms)
    refined = commutator_prefactor(6, terms, 1.0, 2)
    assert refined < 0.7 * legacy
    monkeypatch.setattr(bounds, "PRODUCT_LIMIT", 10)
    assert second_order_coefficient(terms) is None
    assert commutator_prefactor(6, terms, 1.0, 2) == legacy


def test_empty_and_commuting_hamiltonians() -> None:
    assert second_order_coefficient([]) == 0.0
    commuting: list[PauliTerm] = [("ZZ", (0, 1), 1.0), ("Z", (1,), 0.3)]
    assert second_order_coefficient(commuting) == 0.0
    assert commutator_bound(2, commuting, 1.0, 1, 2) == 0.0


def test_heisenberg_chain_of_sixteen_qubits_needs_fewer_steps() -> None:
    """Known issue: 117 steps for eps 1e-2, against 23 with the state metric."""
    terms: list[PauliTerm] = [
        (p, (i, i + 1), 1.0) for i in range(15) for p in ("XX", "YY", "ZZ")
    ]
    legacy = _legacy_coefficient(16, terms)
    refined = commutator_prefactor(16, terms, 1.0, 2)
    assert legacy == pytest.approx(136.0)
    assert refined == pytest.approx(58.863, abs=1e-3)
    steps, estimate = _steps_for_eps(16, terms, 1.0, 2, 1e-2, "bound", 4096)
    assert steps == 77
    assert estimate <= 1e-2 < commutator_bound(16, terms, 1.0, 76, 2)
    # The bound scales as 1/steps² and as |t|³, as before.
    assert commutator_bound(16, terms, 1.0, 10, 2) == pytest.approx(refined / 100)
    assert commutator_bound(16, terms, -2.0, 10, 2) == pytest.approx(8 * refined / 100)


def test_first_order_bound_is_unchanged() -> None:
    terms = heisenberg(4)
    assert commutator_bound(4, terms, 1.0, 4, 1) == pytest.approx(
        _legacy_first_order(terms) / 4
    )


def _legacy_first_order(terms: list[PauliTerm]) -> float:
    """``t²/2 Σ_{i<j} ‖[H_i, H_j]‖`` for ``t = 1``, from dense matrices."""
    n = 1 + max(q for _, qubits, _ in terms for q in qubits)
    dense = [c * _dense(p, q, n) for p, q, c in terms]
    total = 0.0
    for i, a in enumerate(dense):
        for b in dense[i + 1 :]:
            total += float(np.linalg.norm(a @ b - b @ a, 2))
    return total / 2.0
