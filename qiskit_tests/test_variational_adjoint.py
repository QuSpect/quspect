"""Cross-validate ``quspect.variational`` (objective/adjoint/result) with qiskit.

Expectation values are compared against ``qiskit.primitives.StatevectorEstimator``
(1e-9) and adjoint Jacobians against both a central finite difference of the
qiskit estimator and a qiskit-side, occurrence-level parameter-shift rule
(1e-7).

QubitOperator -> SparsePauliOp mapping
--------------------------------------
A quspect term key ``((q0, P0), (q1, P1), ...)`` (sorted by qubit) with
coefficient ``c`` becomes ``SparsePauliOp.from_sparse_list([(P0 P1 ..., [q0,
q1, ...], c)], num_qubits=n)``: qiskit's sparse list attaches the ``i``-th
letter to qubit ``indices[i]``, and the dense qiskit label is then written with
qubit 0 *rightmost*.  The empty key ``()`` is the identity.  The mapping is
self-checked against an explicit little-endian Kronecker product
(``kron(P_{n-1}, ..., P_0)``) and against ``<psi|H|psi>`` on a quspect
statevector.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import qiskit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qiskit_tests.helpers import to_qiskit
from quspect import QuantumCircuit
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.operators.qubit_operator import QubitOperator
from quspect.simulation import Simulator
from quspect.variational import (
    AdjointEngine,
    ObjectiveEvaluation,
    ObjectiveSweep,
    ObservableObjective,
    PreparedObjective,
)

# =============================================================================
# QubitOperator <-> qiskit conversion
# =============================================================================

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def qubit_operator_to_sparse_pauli_op(
    operator: QubitOperator, num_qubits: int
) -> SparsePauliOp:
    """Convert a numeric quspect ``QubitOperator`` to a qiskit ``SparsePauliOp``."""
    items: list[tuple[str, list[int], complex]] = []
    for term, coeff in operator:
        letters = "".join(pauli.upper() for _, pauli in term)
        qubits = [qubit for qubit, _ in term]
        items.append((letters, qubits, complex(coeff)))
    if not items:
        items.append(("", [], 0j))
    return SparsePauliOp.from_sparse_list(items, num_qubits=num_qubits)


def explicit_operator_matrix(operator: QubitOperator, num_qubits: int) -> np.ndarray:
    """Dense little-endian matrix ``sum c * kron(P_{n-1}, ..., P_0)``."""
    dim = 1 << num_qubits
    total = np.zeros((dim, dim), dtype=complex)
    for term, coeff in operator:
        letters = ["I"] * num_qubits
        for qubit, pauli in term:
            letters[qubit] = pauli.upper()
        mat = np.ones((1, 1), dtype=complex)
        for qubit in reversed(range(num_qubits)):
            mat = np.kron(mat, _PAULI[letters[qubit]])
        total += complex(coeff) * mat
    return total


# =============================================================================
# qiskit-side reference evaluation
# =============================================================================

_ESTIMATOR = StatevectorEstimator()


@dataclass
class QiskitObjective:
    """A qiskit translation of an ``ObservableObjective`` used as reference."""

    circuit: qiskit.QuantumCircuit
    table: dict[str, qiskit.circuit.Parameter]
    observables: list[SparsePauliOp]

    @classmethod
    def from_quspect(cls, objective: ObservableObjective) -> QiskitObjective:
        """Translate circuit and observables with the shared helpers."""
        table: dict[str, qiskit.circuit.Parameter] = {}
        qc = to_qiskit(objective.circuit, parameter_table=table)
        n = qc.num_qubits
        observables = [
            qubit_operator_to_sparse_pauli_op(op, n)
            for op in objective.observables.values()
        ]
        return cls(qc, table, observables)

    def _bind(
        self, qc: qiskit.QuantumCircuit, point: Mapping[str, float]
    ) -> qiskit.QuantumCircuit:
        binding = {
            self.table[name]: float(value)
            for name, value in point.items()
            if name in self.table and self.table[name] in qc.parameters
        }
        return qc.assign_parameters(binding)

    def values(
        self, point: Mapping[str, float], qc: qiskit.QuantumCircuit | None = None
    ) -> np.ndarray:
        """Exact expectation values from ``StatevectorEstimator``."""
        bound = self._bind(self.circuit if qc is None else qc, point)
        result = _ESTIMATOR.run([(bound, self.observables)]).result()[0]
        return np.real(np.asarray(result.data.evs, dtype=complex)).reshape(-1)

    def finite_difference(
        self, point: Mapping[str, float], names: Sequence[str], h: float = 1e-5
    ) -> np.ndarray:
        """Central finite-difference Jacobian ``[observable][parameter]``."""
        cols = []
        for name in names:
            plus = dict(point)
            minus = dict(point)
            plus[name] += h
            minus[name] -= h
            cols.append((self.values(plus) - self.values(minus)) / (2 * h))
        return np.stack(cols, axis=1)

    def parameter_shift(
        self, point: Mapping[str, float], names: Sequence[str]
    ) -> np.ndarray:
        """Occurrence-level analytic parameter-shift Jacobian on the qiskit side.

        Every gate parameter that depends on a symbol is shifted on its own;
        the occurrence derivative is multiplied by the affine chain-rule
        coefficient ``d(expr)/d(theta)``.  Two-term rule (generator spectrum
        ``{+-1/2}`` or ``{0, 1}``): ``(f(+pi/2) - f(-pi/2)) / 2``.  Controlled
        rotations (spectrum ``{0, 0, +-1/2}``) use the four-term rule with
        ``c+- = (sqrt2 +- 1) / (4 sqrt2)`` at ``+-pi/2`` and ``+-3pi/2``.
        """
        two = [(math.pi / 2, 0.5), (-math.pi / 2, -0.5)]
        cp_ = (math.sqrt(2) + 1) / (4 * math.sqrt(2))
        cm_ = (math.sqrt(2) - 1) / (4 * math.sqrt(2))
        four = [
            (math.pi / 2, cp_),
            (-math.pi / 2, -cp_),
            (3 * math.pi / 2, -cm_),
            (-3 * math.pi / 2, cm_),
        ]
        rules: dict[str, Callable[[int], list[tuple[float, float]]]] = {
            "rx": lambda _j: two,
            "ry": lambda _j: two,
            "rz": lambda _j: two,
            "p": lambda _j: two,
            "u": lambda _j: two,
            "cp": lambda _j: two,
            "crx": lambda _j: four,
            "cry": lambda _j: four,
            "crz": lambda _j: four,
            "cu": lambda j: four if j == 0 else [],
        }
        jac = np.zeros((len(self.observables), len(names)))
        for index, instr in enumerate(self.circuit.data):
            op = instr.operation
            for j, param in enumerate(op.params):
                if not isinstance(param, qiskit.circuit.ParameterExpression):
                    continue
                if not param.parameters:
                    continue
                rule = rules[op.name](j)
                if not rule:
                    raise NotImplementedError(f"no shift rule for {op.name}[{j}]")
                local = np.zeros(len(self.observables))
                for shift, weight in rule:
                    shifted = self.circuit.copy()
                    new_op = op.to_mutable()
                    new_params = list(op.params)
                    new_params[j] = param + shift
                    new_op.params = new_params
                    shifted.data[index] = instr.replace(operation=new_op)
                    local += weight * self.values(point, shifted)
                for col, name in enumerate(names):
                    sym = self.table.get(name)
                    if sym is None or sym not in param.parameters:
                        continue
                    coeff = float(param.gradient(sym))
                    jac[:, col] += coeff * local
        return jac


# =============================================================================
# Objective builders
# =============================================================================

A, B, C = Parameter("a"), Parameter("b"), Parameter("c")


def _hamiltonians(n: int) -> dict[str, QubitOperator]:
    """A few observables mixing X/Y/Z, identity offsets and all qubits."""
    terms_h = {
        ((0, "Z"),): 0.7,
        ((0, "X"), (1, "Y")): -0.45,
        ((1, "Z"), (n - 1, "X")): 0.3,
        (): 0.25,
    }
    terms_m = {((q, "Y"),): 0.2 + 0.1 * q for q in range(n)}
    terms_m[tuple((q, "Z") for q in range(n))] = -0.6
    return {"H": QubitOperator(terms_h), "M": QubitOperator(terms_m)}


def _rotations() -> tuple[QuantumCircuit, dict[str, float]]:
    circ = QuantumCircuit().h(0).h(1).h(2)
    circ.rx(A, 0).ry(B, 1).rz(C, 2).cx(0, 1).cx(1, 2)
    circ.ry(A, 2).rx(C, 0).cz(0, 2).rz(B, 1).s(0).t(1).sdg(2).tdg(0)
    return circ, {"a": 0.37, "b": -1.21, "c": 2.05}


def _shared() -> tuple[QuantumCircuit, dict[str, float]]:
    circ = QuantumCircuit().h(0)
    circ.rx(A, 0).ry(A, 1).rz(A, 2).cx(0, 1).cx(2, 0)
    circ.ry(A, 0).rx(B, 1).swap(1, 2).rz(A, 1).ry(B, 2).ccx(0, 1, 2)
    return circ, {"a": 0.81, "b": -0.44}


def _affine() -> tuple[QuantumCircuit, dict[str, float]]:
    circ = QuantumCircuit().h(0).h(2)
    circ.rx(2.5 * A - 0.7, 0).ry(-0.5 * A + 1.1 * B + 0.3, 1)
    circ.rz(A - B, 2).cx(0, 1).cx(1, 2).ry(3.0 * C + 1.0, 0)
    circ.rx(B / 4 - C, 2).x(1).y(0).z(2).id(1)
    return circ, {"a": 0.29, "b": 1.33, "c": -0.62}


def _controlled() -> tuple[QuantumCircuit, dict[str, float]]:
    circ = QuantumCircuit().h(0).h(1).ry(0.4, 2)
    circ.crx(A, 0, 1).cry(1.5 * B + 0.2, 1, 2).crz(A - C, 2, 0)
    circ.h(1).crx(-0.8 * C, 1, 0).cry(A, 0, 2)
    return circ, {"a": 1.07, "b": -0.58, "c": 0.91}


def _phase_u() -> tuple[QuantumCircuit, dict[str, float]]:
    circ = QuantumCircuit().h(0).h(1).h(2)
    circ.p(A, 0).cp(B, 0, 1).u3(A, B, C, 2).u1(2.0 * C - 0.1, 1)
    circ.u2(A, B, 0).cx(2, 1).cu3(C, 0.3, -0.2, 1, 2).h(0)
    return circ, {"a": 0.53, "b": -0.97, "c": 1.41}


BUILDERS: dict[str, Callable[[], tuple[QuantumCircuit, dict[str, float]]]] = {
    "rotations": _rotations,
    "shared": _shared,
    "affine": _affine,
    "controlled": _controlled,
    "phase_u": _phase_u,
}


def _objective(name: str) -> tuple[ObservableObjective, dict[str, float]]:
    circ, point = BUILDERS[name]()
    return ObservableObjective(circ, _hamiltonians(circ.num_qubits)), point


def _names(prepared: PreparedObjective) -> list[str]:
    return [p.name for p in prepared.parameter_order]


# =============================================================================
# Conversion self-checks
# =============================================================================


@pytest.mark.qiskit_ref
def test_sparse_pauli_conversion_matches_explicit_matrix() -> None:
    """QubitOperator -> SparsePauliOp agrees with an explicit kron matrix."""
    op = QubitOperator(
        {
            ((0, "X"), (1, "Z")): 0.5,
            ((1, "Y"),): -0.3,
            ((0, "Z"), (2, "Y")): 1.2,
            (): 0.7,
        }
    )
    spo = qubit_operator_to_sparse_pauli_op(op, 3)
    np.testing.assert_allclose(spo.to_matrix(), explicit_operator_matrix(op, 3))
    # X on qubit 0 and Z on qubit 1 => dense qiskit label "IZX" (qubit 0 right).
    labels = dict(zip(spo.paulis.to_labels(), spo.coeffs, strict=True))
    assert labels["IZX"] == pytest.approx(0.5)
    assert labels["YIZ"] == pytest.approx(1.2)


@pytest.mark.qiskit_ref
def test_sparse_pauli_conversion_matches_quspect_statevector() -> None:
    """``<psi|H|psi>`` on a quspect statevector equals the qiskit estimator."""
    circ = QuantumCircuit().h(0).ry(0.3, 1).cx(0, 2).rx(1.1, 2).s(1)
    op = _hamiltonians(3)["H"]
    psi = Simulator(seed=1).statevector(circ)
    direct = np.vdot(psi, explicit_operator_matrix(op, 3) @ psi).real
    ref = QiskitObjective.from_quspect(ObservableObjective(circ, {"H": op}))
    assert ref.values({})[0] == pytest.approx(direct, abs=1e-12)


# =============================================================================
# Values and gradients versus qiskit
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("case", sorted(BUILDERS))
def test_adjoint_values_match_estimator(case: str) -> None:
    """Adjoint expectation values equal StatevectorEstimator within 1e-9."""
    objective, point = _objective(case)
    prepared = AdjointEngine().compile(objective)
    evaluation = prepared.evaluate(point)
    ref = QiskitObjective.from_quspect(objective)
    np.testing.assert_allclose(evaluation.values, ref.values(point), atol=1e-9)
    assert evaluation.observable_order == ("H", "M")


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("case", sorted(BUILDERS))
def test_adjoint_jacobian_matches_finite_difference(case: str) -> None:
    """Adjoint Jacobian equals a central finite difference (h=1e-5) within 1e-7."""
    objective, point = _objective(case)
    prepared = AdjointEngine().compile(objective)
    names = _names(prepared)
    assert sorted(names) == sorted(point)
    jac = prepared.evaluate(point).jacobian
    ref = QiskitObjective.from_quspect(objective)
    assert jac.shape == (2, len(names))
    np.testing.assert_allclose(jac, ref.finite_difference(point, names), atol=1e-7)


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("case", sorted(BUILDERS))
def test_adjoint_jacobian_matches_parameter_shift(case: str) -> None:
    """Adjoint Jacobian equals a qiskit-side occurrence parameter-shift rule."""
    objective, point = _objective(case)
    prepared = AdjointEngine().compile(objective)
    names = _names(prepared)
    ref = QiskitObjective.from_quspect(objective)
    shift = ref.parameter_shift(point, names)
    # The analytic shift rule itself must agree with finite differences.
    np.testing.assert_allclose(shift, ref.finite_difference(point, names), atol=1e-7)
    np.testing.assert_allclose(prepared.evaluate(point).jacobian, shift, atol=1e-7)


@pytest.mark.qiskit_ref
def test_jacobian_layout_is_observable_by_parameter() -> None:
    """Jacobian rows follow observable order and columns follow parameter order."""
    circ = QuantumCircuit().rx(A, 0).ry(B, 1)
    objective = ObservableObjective(
        circ,
        {
            "z1": QubitOperator({((1, "Z"),): 1.0}),
            "z0": QubitOperator({((0, "Z"),): 1.0}),
            "x1z0": QubitOperator({((0, "Z"), (1, "X")): 2.0}),
        },
    )
    prepared = AdjointEngine().compile(objective)
    a, b = 0.4, -1.3
    ev = prepared.evaluate({"a": a, "b": b})
    order = _names(prepared)
    expected = {
        "z1": (math.cos(b), {"a": 0.0, "b": -math.sin(b)}),
        "z0": (math.cos(a), {"a": -math.sin(a), "b": 0.0}),
        "x1z0": (
            2 * math.cos(a) * math.sin(b),
            {"a": -2 * math.sin(a) * math.sin(b), "b": 2 * math.cos(a) * math.cos(b)},
        ),
    }
    assert ev.observable_order == ("z1", "z0", "x1z0")
    for row, name in enumerate(ev.observable_order):
        value, grads = expected[name]
        assert ev.values[row] == pytest.approx(value, abs=1e-12)
        assert ev.value(name) == pytest.approx(value, abs=1e-12)
        np.testing.assert_allclose(
            ev.derivative(row), [grads[p] for p in order], atol=1e-12
        )
    ref = QiskitObjective.from_quspect(objective)
    np.testing.assert_allclose(
        ev.jacobian, ref.finite_difference({"a": a, "b": b}, order), atol=1e-7
    )


@pytest.mark.qiskit_ref
def test_float32_engine_matches_estimator_loosely() -> None:
    """A float32 engine returns float32 arrays close to the double reference."""
    objective, point = _objective("affine")
    engine = AdjointEngine(dtype=np.float32)
    assert engine.dtype == "float32"
    prepared = engine.compile(objective)
    assert prepared.dtype == "float32"
    assert prepared.metadata["dtype"] == "float32"
    ev = prepared.evaluate(point)
    assert ev.values.dtype == np.float32
    assert ev.jacobian.dtype == np.float32
    ref = QiskitObjective.from_quspect(objective)
    np.testing.assert_allclose(ev.values, ref.values(point), atol=1e-5)
    names = _names(prepared)
    np.testing.assert_allclose(
        ev.jacobian, ref.finite_difference(point, names), atol=1e-4
    )


@pytest.mark.qiskit_ref
def test_identity_only_observable_has_zero_gradient() -> None:
    """A pure identity observable evaluates to its coefficient with zero slope."""
    circ = QuantumCircuit().rx(A, 0)
    objective = ObservableObjective(circ, {"I": QubitOperator.identity(1.5)})
    ev = AdjointEngine().compile(objective).evaluate([0.3])
    assert ev.values[0] == pytest.approx(1.5, abs=1e-12)
    np.testing.assert_allclose(ev.jacobian, [[0.0]], atol=1e-12)


# =============================================================================
# Binding styles and sweeps
# =============================================================================


@pytest.mark.qiskit_ref
def test_binding_styles_are_equivalent() -> None:
    """Sequence, str-keyed and Parameter-keyed points give identical results."""
    objective, point = _objective("rotations")
    prepared = AdjointEngine().compile(objective)
    names = _names(prepared)
    seq = [point[n] for n in names]
    by_str = prepared.evaluate(point)
    by_param = prepared.evaluate({Parameter(k): v for k, v in point.items()})
    by_seq = prepared.evaluate(seq)
    by_array = prepared.evaluate(np.asarray(seq))
    for other in (by_param, by_seq, by_array):
        np.testing.assert_array_equal(other.values, by_str.values)
        np.testing.assert_array_equal(other.jacobian, by_str.jacobian)
    assert by_str.parameter_order == prepared.parameter_order
    assert dict(by_str.metadata) == {
        "backend": "native-statevector",
        "differentiation": "adjoint",
        "dtype": "float64",
    }


@pytest.mark.qiskit_ref
def test_evaluate_many_matches_pointwise_and_qiskit() -> None:
    """``evaluate_many`` stacks per-point results for arrays, lists and maps."""
    objective, _ = _objective("controlled")
    prepared = AdjointEngine().compile(objective)
    names = _names(prepared)
    rng = np.random.default_rng(7)
    points = rng.uniform(-math.pi, math.pi, size=(4, len(names)))
    sweep = prepared.evaluate_many(points)
    assert isinstance(sweep, ObjectiveSweep)
    assert len(sweep) == 4
    assert sweep.values.shape == (4, 2)
    assert sweep.jacobian.shape == (4, 2, len(names))
    ref = QiskitObjective.from_quspect(objective)
    for i, row in enumerate(points):
        point = dict(zip(names, row.tolist(), strict=True))
        np.testing.assert_allclose(sweep.values[i], ref.values(point), atol=1e-9)
        at = sweep.at(i)
        assert isinstance(at, ObjectiveEvaluation)
        np.testing.assert_array_equal(at.jacobian, sweep.jacobian[i])
    as_list = prepared.evaluate_many(points.tolist())
    as_maps = prepared.evaluate_many(
        [dict(zip(names, row.tolist(), strict=True)) for row in points]
    )
    np.testing.assert_array_equal(as_list.values, sweep.values)
    np.testing.assert_array_equal(as_maps.jacobian, sweep.jacobian)


def test_evaluate_many_empty() -> None:
    """An empty sweep returns correctly shaped empty arrays."""
    objective, _ = _objective("shared")
    prepared = AdjointEngine().compile(objective)
    sweep = prepared.evaluate_many([])
    assert len(sweep) == 0
    assert sweep.values.shape == (0, 2)
    assert sweep.jacobian.shape == (0, 2, 2)


@pytest.mark.parametrize(
    "points",
    [np.zeros((2, 3)), np.zeros(2), [[0.1, 0.2, 0.3]], [[0.1], [0.2, 0.3]]],
)
def test_evaluate_many_rejects_bad_shapes(points: Any) -> None:
    """Sweeps whose second axis does not match the parameter count fail."""
    objective, _ = _objective("shared")
    prepared = AdjointEngine().compile(objective)
    with pytest.raises(ValueError):
        prepared.evaluate_many(points)


def test_compiled_objective_is_snapshot() -> None:
    """Mutating the source circuit after compile does not change the plan."""
    circ = QuantumCircuit().rx(A, 0)
    obs = {"z": QubitOperator({((0, "Z"),): 1.0})}
    prepared = AdjointEngine().compile(ObservableObjective(circ, obs))
    circ.x(0)
    obs["z"] = QubitOperator({((0, "X"),): 1.0})
    ev = prepared.evaluate([0.6])
    assert ev.values[0] == pytest.approx(math.cos(0.6), abs=1e-12)


def test_memory_warning_for_large_objectives() -> None:
    """Compiling an objective needing >=256 MB of state warns ResourceWarning."""
    circ = QuantumCircuit().rx(A, 0).id(19)
    obs = {f"z{i}": QubitOperator({((i, "Z"),): 1.0}) for i in range(16)}
    with pytest.warns(ResourceWarning, match="costate"):
        AdjointEngine().compile(ObservableObjective(circ, obs))


# =============================================================================
# Error paths
# =============================================================================


@pytest.mark.parametrize(
    ("point", "match"),
    [
        ({"a": 0.1}, "missing"),
        ({"a": 0.1, "b": 0.2, "zz": 0.3}, "unknown"),
        ({"a": 0.1, Parameter("a"): 0.2, "b": 0.3}, "Duplicate"),
        ([0.1], "expected 2"),
        ([[0.1, 0.2]], "expected 2"),
        ([0.1, math.nan], "finite"),
        ({"a": math.inf, "b": 0.0}, "finite"),
    ],
)
def test_evaluate_rejects_bad_points(point: Any, match: str) -> None:
    """Missing/unknown/duplicate/non-finite or mis-sized bindings raise."""
    prepared = AdjointEngine().compile(_objective("shared")[0])
    with pytest.raises(ValueError, match=match):
        prepared.evaluate(point)


def test_evaluate_rejects_bad_key_type() -> None:
    """Binding keys must be Parameter or str."""
    prepared = AdjointEngine().compile(_objective("shared")[0])
    with pytest.raises(TypeError, match="Parameter or str"):
        prepared.evaluate({0: 0.1, "b": 0.2})


@pytest.mark.parametrize(
    ("dtype", "exc"),
    [("complex128", ValueError), ("int32", ValueError), ("no-such-dtype", TypeError)],
)
def test_engine_rejects_bad_dtype(dtype: str, exc: type[Exception]) -> None:
    """Only float32/float64 state vectors are accepted."""
    with pytest.raises(exc):
        AdjointEngine(dtype=dtype)


def test_compile_rejects_non_objective() -> None:
    """``compile`` requires an ObservableObjective."""
    with pytest.raises(TypeError, match="ObservableObjective"):
        AdjointEngine().compile(QuantumCircuit().h(0))  # type: ignore[arg-type]


def _z0() -> dict[str, QubitOperator]:
    return {"z": QubitOperator({((0, "Z"),): 1.0})}


@pytest.mark.parametrize(
    ("circuit", "observables", "exc", "match"),
    [
        (QuantumCircuit().h(0).measure([0], [0]), _z0(), ValueError, "measure"),
        (QuantumCircuit().h(0).depolarize(0.1, 0), _z0(), ValueError, "noise"),
        (QuantumCircuit(), _z0(), ValueError, "at least one qubit"),
        (QuantumCircuit().rx(Parameter(""), 0), _z0(), ValueError, "non-empty"),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((3, "Z"),): 1.0})},
            ValueError,
            "outside",
        ),
        (QuantumCircuit().h(0), {"z": QubitOperator.zero()}, ValueError, "Pauli term"),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): 1.0 + 0.5j})},
            ValueError,
            "real",
        ),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): math.inf})},
            ValueError,
            "finite",
        ),
        (
            QuantumCircuit().h(0),
            {"z": QubitOperator({((0, "Z"),): A})},
            TypeError,
            "numeric",
        ),
    ],
)
def test_compile_rejects_invalid_objectives(
    circuit: QuantumCircuit,
    observables: dict[str, QubitOperator],
    exc: type[Exception],
    match: str,
) -> None:
    """Measurements, noise, empty circuits and bad observables are rejected."""
    with pytest.raises(exc, match=match):
        AdjointEngine().compile(ObservableObjective(circuit, observables))


# =============================================================================
# ObservableObjective
# =============================================================================


def test_observable_objective_copies_and_freezes_mapping() -> None:
    """The observable mapping is copied, ordered and read-only."""
    obs = {"b": QubitOperator({((0, "Z"),): 1.0}), "a": QubitOperator.identity()}
    objective = ObservableObjective(QuantumCircuit().h(0), obs)
    obs["c"] = QubitOperator.identity()
    assert list(objective.observables) == ["b", "a"]
    with pytest.raises(TypeError):
        objective.observables["d"] = QubitOperator.identity()  # type: ignore[index]


@pytest.mark.parametrize(
    ("circuit", "observables", "exc"),
    [
        ("not a circuit", {"z": QubitOperator.identity()}, TypeError),
        (QuantumCircuit().h(0), [("z", QubitOperator.identity())], TypeError),
        (QuantumCircuit().h(0), {"": QubitOperator.identity()}, ValueError),
        (QuantumCircuit().h(0), {"  ": QubitOperator.identity()}, ValueError),
        (QuantumCircuit().h(0), {1: QubitOperator.identity()}, ValueError),
        (QuantumCircuit().h(0), {"z": "Z0"}, TypeError),
        (QuantumCircuit().h(0), {}, ValueError),
    ],
)
def test_observable_objective_validation(
    circuit: Any, observables: Any, exc: type[Exception]
) -> None:
    """Invalid circuits, mappings, names or operators are rejected eagerly."""
    with pytest.raises(exc):
        ObservableObjective(circuit, observables)


# =============================================================================
# Result containers
# =============================================================================


def _evaluation(**overrides: Any) -> ObjectiveEvaluation:
    kwargs: dict[str, Any] = {
        "values": [1.0, 2.0],
        "jacobian": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "parameter_order": (A, B, C),
        "observable_order": ("e", "m"),
        "metadata": {"k": "v"},
    }
    kwargs.update(overrides)
    return ObjectiveEvaluation(**kwargs)


def test_objective_evaluation_accessors() -> None:
    """``value``/``derivative`` select rows by name or index; arrays are frozen."""
    ev = _evaluation()
    assert ev.values.dtype == np.float64
    assert ev.value() == 1.0
    assert ev.value("m") == 2.0
    assert ev.value(1) == 2.0
    np.testing.assert_array_equal(ev.derivative("m"), [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(ev.derivative(), [0.1, 0.2, 0.3])
    assert not ev.values.flags.writeable
    assert not ev.jacobian.flags.writeable
    with pytest.raises(TypeError):
        ev.metadata["k"] = "w"  # type: ignore[index]
    with pytest.raises(KeyError):
        ev.value("missing")
    with pytest.raises(IndexError):
        ev.value(2)
    with pytest.raises(IndexError):
        ev.derivative(-1)
    with pytest.raises(TypeError):
        ev.value(1.0)  # type: ignore[arg-type]


def test_objective_evaluation_preserves_float32_and_copies() -> None:
    """float32 inputs keep their dtype and are copied, not aliased."""
    values = np.array([1.0, 2.0], dtype=np.float32)
    ev = _evaluation(values=values, jacobian=np.zeros((2, 3), dtype=np.float32))
    values[0] = 9.0
    assert ev.values.dtype == np.float32
    assert ev.values[0] == 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"values": [1.0, 2.0, 3.0]},
        {"values": [[1.0, 2.0]]},
        {"jacobian": [[0.1, 0.2], [0.3, 0.4]]},
        {"jacobian": [0.1, 0.2, 0.3]},
    ],
)
def test_objective_evaluation_shape_validation(overrides: dict[str, Any]) -> None:
    """Values/Jacobian shapes must match the declared axes."""
    with pytest.raises(ValueError):
        _evaluation(**overrides)


def test_objective_sweep_accessors_and_validation() -> None:
    """``ObjectiveSweep`` validates its axes and ``at`` extracts one point."""
    sweep = ObjectiveSweep(
        values=np.arange(4.0).reshape(2, 2),
        jacobian=np.arange(12.0).reshape(2, 2, 3),
        parameter_order=(A, B, C),
        observable_order=("e", "m"),
        metadata={"k": "v"},
    )
    assert len(sweep) == 2
    point = sweep.at(1)
    assert point.value("m") == 3.0
    np.testing.assert_array_equal(point.derivative("e"), [6.0, 7.0, 8.0])
    assert dict(point.metadata) == {"k": "v"}
    with pytest.raises(ValueError):
        ObjectiveSweep(
            values=np.zeros((2, 3)),
            jacobian=np.zeros((2, 2, 3)),
            parameter_order=(A, B, C),
            observable_order=("e", "m"),
            metadata={},
        )
    with pytest.raises(ValueError):
        ObjectiveSweep(
            values=np.zeros((2, 2)),
            jacobian=np.zeros((3, 2, 3)),
            parameter_order=(A, B, C),
            observable_order=("e", "m"),
            metadata={},
        )


@pytest.mark.parametrize(
    "angle",
    [A + math.inf, ParameterExpression({}, constant=math.inf), math.inf],
    ids=["affine-offset", "constant-expression", "float"],
)
def test_adjoint_rejects_non_finite_gate_angles(angle: Any) -> None:
    """Non-finite gate angles should raise a clean ValueError at compile."""
    circ = QuantumCircuit().ry(B, 0).rx(angle, 0)
    with pytest.raises(ValueError):
        AdjointEngine().compile(ObservableObjective(circ, _z0()))


def test_prepared_objective_exposes_axes() -> None:
    """``PreparedObjective`` exposes observable/parameter order and metadata."""
    objective, _ = _objective("rotations")
    prepared = AdjointEngine().compile(objective)
    assert prepared.observable_order == ("H", "M")
    assert {p.name for p in prepared.parameter_order} == {"a", "b", "c"}
    assert prepared.metadata["differentiation"] == "adjoint"
