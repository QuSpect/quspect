"""Tests for the native variational-objective execution path."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pytest

from quspect import Parameter, QuantumCircuit, QubitOperator, Simulator
from quspect.variational import AdjointEngine, ObservableObjective, PreparedObjective


def _statevector_expectation(state: np.ndarray, hamiltonian: QubitOperator) -> float:
    """Reference Pauli-sum expectation using a deliberately simple loop."""
    total = 0.0j
    for term, coefficient in hamiltonian:
        transformed = np.zeros_like(state)
        for basis, amplitude in enumerate(state):
            target = basis
            phase = 1.0 + 0.0j
            for qubit, pauli in term:
                bit = (basis >> qubit) & 1
                if pauli == "X":
                    target ^= 1 << qubit
                elif pauli == "Y":
                    target ^= 1 << qubit
                    phase *= -1.0j if bit else 1.0j
                elif pauli == "Z" and bit:
                    phase *= -1.0
            transformed[target] += phase * amplitude
        total += complex(coefficient) * np.vdot(state, transformed)
    return float(total.real)


def _finite_difference(
    simulator: Simulator,
    circuit: QuantumCircuit,
    hamiltonian: QubitOperator,
    values: Mapping[Parameter, float],
    parameter: Parameter,
    epsilon: float = 1e-6,
) -> float:
    plus = dict(values)
    minus = dict(values)
    plus[parameter] += epsilon
    minus[parameter] -= epsilon
    e_plus = _statevector_expectation(
        simulator.statevector(circuit.bind_parameters(plus)), hamiltonian
    )
    e_minus = _statevector_expectation(
        simulator.statevector(circuit.bind_parameters(minus)), hamiltonian
    )
    return (e_plus - e_minus) / (2.0 * epsilon)


def _compile(
    circuit: QuantumCircuit,
    observables: Mapping[str, QubitOperator],
    *,
    dtype: str = "float64",
) -> PreparedObjective:
    objective = ObservableObjective(circuit, observables)
    return AdjointEngine(dtype=dtype).compile(objective)


@pytest.mark.parametrize("gate_name", ["rx", "ry"])
def test_single_rotation_z_gradient(gate_name: str) -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit()
    getattr(circuit, gate_name)(theta, 0)
    plan = _compile(circuit, {"energy": QubitOperator.from_pauli_string("Z")})

    angle = 0.37
    result = plan.evaluate([angle])

    assert result.values.shape == (1,)
    assert result.jacobian.shape == (1, 1)
    assert result.values == pytest.approx([math.cos(angle)], abs=1e-12)
    assert result.jacobian == pytest.approx(np.array([[-math.sin(angle)]]), abs=1e-12)
    assert result.parameter_order == (theta,)
    assert result.observable_order == ("energy",)


def test_rz_gradient_on_plus_state_and_named_accessors() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().h(0).rz(theta, 0)
    plan = _compile(
        circuit, {"transverse_magnetization": QubitOperator.from_pauli_string("X")}
    )

    angle = -0.61
    result = plan.evaluate({theta: angle})

    expected_value = math.cos(angle)
    expected_gradient = -math.sin(angle)
    assert result.value("transverse_magnetization") == pytest.approx(
        expected_value, abs=1e-12
    )
    assert result.value(0) == pytest.approx(expected_value, abs=1e-12)
    assert result.derivative("transverse_magnetization") == pytest.approx(
        [expected_gradient], abs=1e-12
    )
    assert result.derivative(0) == pytest.approx([expected_gradient], abs=1e-12)


def test_affine_expression_chain_rule_and_multiterm_hamiltonian() -> None:
    theta = Parameter("theta")
    angle = 2.0 * theta + 0.1
    circuit = QuantumCircuit().ry(angle, 0)
    hamiltonian = QubitOperator(
        {
            (): 0.7,
            ((0, "Z"),): 0.4,
            ((0, "X"),): 0.2,
        }
    )
    plan = _compile(circuit, {"energy": hamiltonian})

    value = 0.23
    bound_angle = 2.0 * value + 0.1
    result = plan.evaluate({"theta": value})

    expected_energy = 0.7 + 0.4 * math.cos(bound_angle) + 0.2 * math.sin(bound_angle)
    expected_gradient = 2.0 * (
        -0.4 * math.sin(bound_angle) + 0.2 * math.cos(bound_angle)
    )
    assert result.values == pytest.approx([expected_energy], abs=1e-12)
    assert result.jacobian == pytest.approx(np.array([[expected_gradient]]), abs=1e-12)


def test_shared_parameter_accumulates_all_gate_occurrences() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0).ry(theta, 0)
    plan = _compile(circuit, {"z": QubitOperator.from_pauli_string("Z")})

    value = 0.41
    result = plan.evaluate([value])

    assert result.values == pytest.approx([math.cos(2.0 * value)], abs=1e-12)
    assert result.jacobian == pytest.approx(
        np.array([[-2.0 * math.sin(2.0 * value)]]), abs=1e-12
    )


def test_named_observable_mapping_preserves_insertion_order() -> None:
    beta = Parameter("beta")
    alpha = Parameter("alpha")
    circuit = QuantumCircuit().ry(beta, 0).rx(alpha, 1)
    observables = {
        "x_response": QubitOperator.from_pauli_string("X", qubits=(0,)),
        "z_response": QubitOperator.from_pauli_string("Z", qubits=(0,)),
    }
    plan = _compile(circuit, observables)

    # Point mapping order is deliberately unrelated to circuit parameter order.
    result = plan.evaluate({alpha: -0.19, beta: 0.28})

    assert result.observable_order == ("x_response", "z_response")
    assert result.parameter_order == (beta, alpha)
    assert result.values.shape == (2,)
    assert result.jacobian.shape == (2, 2)
    assert result.values == pytest.approx([math.sin(0.28), math.cos(0.28)], abs=1e-12)
    assert result.jacobian == pytest.approx(
        np.array(
            [
                [math.cos(0.28), 0.0],
                [-math.sin(0.28), 0.0],
            ]
        ),
        abs=1e-12,
    )


def test_entangled_h2_like_gradient_matches_finite_difference() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit().ry(theta, 0).rx(phi, 1).cx(0, 1).rz(theta - 0.5 * phi, 1)
    # Representative two-qubit-reduced H2 Pauli structure.
    hamiltonian = QubitOperator(
        {
            (): -1.052373245772859,
            ((0, "Z"),): 0.39793742484318045,
            ((1, "Z"),): -0.39793742484318045,
            ((0, "Z"), (1, "Z")): -0.01128010425623538,
            ((0, "X"), (1, "X")): 0.18093119978423156,
        }
    )
    simulator = Simulator()
    plan = _compile(circuit, {"electronic_energy": hamiltonian})
    values = {theta: 0.31, phi: -0.22}

    result = plan.evaluate(values)
    bound_state = simulator.statevector(circuit.bind_parameters(values))
    expected_energy = _statevector_expectation(bound_state, hamiltonian)
    reference = {
        parameter: _finite_difference(
            simulator, circuit, hamiltonian, values, parameter
        )
        for parameter in (theta, phi)
    }

    assert result.value("electronic_energy") == pytest.approx(
        expected_energy, abs=1e-11
    )
    assert result.derivative("electronic_energy") == pytest.approx(
        [reference[parameter] for parameter in result.parameter_order],
        abs=2e-7,
    )


def test_seeded_random_circuits_match_finite_difference() -> None:
    rng = np.random.default_rng(20260712)
    simulator = Simulator()

    for case in range(5):
        parameters = [Parameter(f"p{index}") for index in range(3)]
        circuit = QuantumCircuit().h(0)
        for layer in range(2):
            for qubit in range(3):
                parameter = parameters[(qubit + layer) % len(parameters)]
                coefficient = float(rng.uniform(-1.4, 1.4))
                offset = float(rng.uniform(-0.4, 0.4))
                angle = coefficient * parameter + offset
                if (qubit + layer) % 2:
                    circuit.rx(angle, qubit)
                else:
                    circuit.ry(angle, qubit)
            circuit.cx(0, 1).cx(1, 2)
        circuit.rz(parameters[0] - 0.3 * parameters[2], 2)

        terms: dict[tuple[tuple[int, str], ...], float] = {
            (): float(rng.uniform(-0.5, 0.5))
        }
        for _ in range(7):
            pauli_word = tuple(
                (qubit, str(rng.choice(["X", "Y", "Z"])))
                for qubit in range(3)
                if rng.random() < 0.65
            )
            terms[pauli_word] = float(rng.uniform(-0.8, 0.8))
        hamiltonian = QubitOperator(terms)
        values = {parameter: float(rng.uniform(-0.9, 0.9)) for parameter in parameters}

        plan = _compile(circuit, {"random_observable": hamiltonian})
        result = plan.evaluate(values)
        reference = {
            parameter: _finite_difference(
                simulator, circuit, hamiltonian, values, parameter
            )
            for parameter in parameters
        }

        assert result.derivative("random_observable") == pytest.approx(
            [reference[parameter] for parameter in result.parameter_order],
            abs=4e-7,
        ), f"random case {case}"


def test_prepared_objective_can_be_reused_but_is_not_callable() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0)
    plan = _compile(circuit, {"energy": QubitOperator.from_pauli_string("Z")})

    first = plan.evaluate([0.1])
    second = plan.evaluate([0.8])

    assert first.value("energy") == pytest.approx(math.cos(0.1))
    assert second.value("energy") == pytest.approx(math.cos(0.8))
    assert first.derivative("energy")[0] == pytest.approx(-math.sin(0.1))
    assert second.derivative("energy")[0] == pytest.approx(-math.sin(0.8))
    assert not callable(plan)
    with pytest.raises(TypeError):
        plan([0.1])  # type: ignore[operator]


def test_evaluate_many_accepts_matrix_and_mapping_batch() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit().ry(theta, 0).rx(phi, 1)
    plan = _compile(
        circuit,
        {
            "z0": QubitOperator.from_pauli_string("Z", qubits=(0,)),
            "z1": QubitOperator.from_pauli_string("Z", qubits=(1,)),
        },
    )
    matrix = np.array([[0.1, -0.2], [0.7, 0.4], [-0.3, 0.8]])
    mappings = [
        {phi: float(phi_value), theta: float(theta_value)}
        for theta_value, phi_value in matrix
    ]

    matrix_sweep = plan.evaluate_many(matrix)
    mapping_sweep = plan.evaluate_many(mappings)

    assert matrix_sweep.values.shape == (3, 2)
    assert matrix_sweep.jacobian.shape == (3, 2, 2)
    assert mapping_sweep.values == pytest.approx(matrix_sweep.values, abs=1e-12)
    assert mapping_sweep.jacobian == pytest.approx(matrix_sweep.jacobian, abs=1e-12)
    expected_values = np.column_stack((np.cos(matrix[:, 0]), np.cos(matrix[:, 1])))
    assert matrix_sweep.values == pytest.approx(expected_values, abs=1e-12)

    middle = matrix_sweep.at(1)
    assert middle.values.shape == (2,)
    assert middle.jacobian.shape == (2, 2)
    assert middle.parameter_order == (theta, phi)
    assert middle.observable_order == ("z0", "z1")
    assert middle.values == pytest.approx(matrix_sweep.values[1], abs=1e-12)
    assert middle.jacobian == pytest.approx(matrix_sweep.jacobian[1], abs=1e-12)


def test_dtype_validation_and_prepared_metadata() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0)
    objective = ObservableObjective(
        circuit, {"energy": QubitOperator.from_pauli_string("Z")}
    )
    engine = AdjointEngine(dtype="float32")
    plan = engine.compile(objective)
    result = plan.evaluate([0.2])

    assert engine.dtype == "float32"
    assert plan.dtype == "float32"
    assert result.metadata["differentiation"] == "adjoint"
    assert result.metadata["dtype"] == "float32"
    assert isinstance(result.metadata["backend"], str)
    assert result.metadata["backend"]
    assert result.parameter_order == (theta,)
    assert result.observable_order == ("energy",)
    assert result.values.shape == (1,)
    assert result.jacobian.shape == (1, 1)
    assert result.values.dtype == np.dtype(np.float32)
    assert result.jacobian.dtype == np.dtype(np.float32)
    assert result.values == pytest.approx([math.cos(0.2)], abs=2e-6)
    assert result.jacobian == pytest.approx(np.array([[-math.sin(0.2)]]), abs=2e-6)

    with pytest.raises(ValueError, match="dtype"):
        AdjointEngine(dtype="float16")


def test_legacy_simulator_gradient_entry_point_is_absent() -> None:
    import quspect

    assert not hasattr(Simulator, "get_expectation_with_grad")
    assert not hasattr(quspect, "ExpectationWithGradient")


def test_constant_parameter_expression_runs_without_binding() -> None:
    theta = Parameter("theta")
    constant_angle = theta - theta + math.pi
    state = Simulator().statevector(QuantumCircuit().rx(constant_angle, 0))

    assert abs(state[0]) == pytest.approx(0.0, abs=1e-12)
    assert state[1] == pytest.approx(-1.0j, abs=1e-12)


def test_evaluation_input_validation() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0)
    plan = _compile(circuit, {"energy": QubitOperator.from_pauli_string("Z")})

    with pytest.raises(ValueError, match="expected 1 parameter"):
        plan.evaluate([])
    with pytest.raises(ValueError, match="missing"):
        plan.evaluate({})
    with pytest.raises(ValueError, match="unknown"):
        plan.evaluate({"theta": 0.2, "extra": 1.0})
    with pytest.raises(ValueError, match="expected shape"):
        plan.evaluate_many(np.zeros((2, 1, 1)))


def test_compile_rejects_measurement_and_accepts_decomposable_gate() -> None:
    theta = Parameter("theta")
    hamiltonian = QubitOperator.from_pauli_string("Z")
    engine = AdjointEngine()

    with pytest.raises(ValueError, match="measurement"):
        engine.compile(
            ObservableObjective(
                QuantumCircuit().ry(theta, 0).measure_all(),
                {"energy": hamiltonian},
            )
        )

    u3_plan = engine.compile(
        ObservableObjective(
            QuantumCircuit().u3(theta, 0.0, 0.0, 0),
            {"energy": hamiltonian},
        )
    )
    u3_result = u3_plan.evaluate({theta: 0.4})
    assert u3_result.values[0] == pytest.approx(math.cos(0.4))
    assert u3_result.jacobian[0, 0] == pytest.approx(-math.sin(0.4))

    non_hermitian = QubitOperator({((0, "Z"),): np.complex64(1.0j)})
    with pytest.raises(ValueError, match="must be real"):
        engine.compile(
            ObservableObjective(
                QuantumCircuit().ry(theta, 0), {"energy": non_hermitian}
            )
        )

    non_finite = QubitOperator({((0, "Z"),): math.nan})
    with pytest.raises(ValueError, match="finite"):
        engine.compile(
            ObservableObjective(QuantumCircuit().ry(theta, 0), {"energy": non_finite})
        )


def test_compile_rejects_empty_parameter_name_and_invalid_observable_qubit() -> None:
    engine = AdjointEngine()
    hamiltonian = QubitOperator.from_pauli_string("Z")

    with pytest.raises(ValueError, match="non-empty"):
        engine.compile(
            ObservableObjective(
                QuantumCircuit().ry(Parameter(""), 0), {"energy": hamiltonian}
            )
        )

    theta = Parameter("theta")
    invalid_hamiltonian = QubitOperator({((-1, "Z"),): 1.0})
    with pytest.raises(ValueError, match="outside the circuit"):
        engine.compile(
            ObservableObjective(
                QuantumCircuit().ry(theta, 0), {"energy": invalid_hamiltonian}
            )
        )
