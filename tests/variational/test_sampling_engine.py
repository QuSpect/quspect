"""Shot-based objective and occurrence-level parameter-shift tests."""

from __future__ import annotations

import math

import pytest

from quspect import (
    AdjointEngine,
    ObservableObjective,
    Parameter,
    QuantumCircuit,
    QubitOperator,
    SamplingEngine,
    Simulator,
)
from quspect.backends import ExecutionBatch
from quspect.backends._exceptions import JobError
from quspect.simulation.result import SimulationResult
from quspect.variational.sampling import (
    _compile_plan,
    _decode_evaluation,
    _decode_group,
    _experiment_labels,
)


class _CapturingSimulator(Simulator):
    def __init__(self, *, seed: int = 7) -> None:
        super().__init__(seed=seed)
        self.requests = []
        self.jobs = []

    def submit_request(self, request):
        self.requests.append(request)
        job = super().submit_request(request)
        self.jobs.append(job)
        return job


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_xyz_eigenstates_use_the_correct_measurement_basis(pauli: str) -> None:
    circuit = QuantumCircuit()
    if pauli == "X":
        circuit.h(0)
    elif pauli == "Y":
        circuit.h(0).s(0)
    else:
        circuit.id(0)
    backend = _CapturingSimulator()
    objective = ObservableObjective(
        circuit,
        {"value": QubitOperator.from_pauli_string(pauli)},
    )

    result = SamplingEngine(backend, shots=64).compile(objective).evaluate([])
    assert result.value() == pytest.approx(1.0)

    measured = backend.requests[0].bound_circuits()[0]
    names = [
        instruction.gate.name
        for instruction in measured.instructions
        if instruction.gate is not None
    ]
    if pauli == "Y":
        assert names[-2:] == ["sdg", "h"]


def test_qwc_grouping_is_stable_but_does_not_group_general_commutation() -> None:
    circuit = QuantumCircuit().id(0).id(1)
    operator = QubitOperator(
        {
            ((0, "X"), (1, "X")): 0.5,
            ((0, "X"),): 0.25,
            ((0, "Y"), (1, "Y")): -0.75,
        }
    )
    objective = ObservableObjective(circuit, {"energy": operator})

    grouped = SamplingEngine(Simulator(), grouping="qwc").compile(objective)
    ungrouped = SamplingEngine(Simulator(), grouping="none").compile(objective)
    assert grouped.num_measurement_groups == 2
    assert ungrouped.num_measurement_groups == 3


def test_identity_only_objective_never_calls_backend() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0)
    backend = _CapturingSimulator()
    objective = ObservableObjective(
        circuit,
        {"constant": QubitOperator.identity(2.5)},
    )
    prepared = SamplingEngine(backend, shots=32).compile(objective)

    job = prepared.submit({theta: 0.3})
    result = job.result()
    assert job.job_id is None
    assert result.value() == 2.5
    assert result.derivative().tolist() == [0.0]
    assert backend.requests == []


def test_shared_parameter_uses_occurrence_level_shift_chain_rule() -> None:
    theta = Parameter("theta")
    circuit = QuantumCircuit().ry(theta, 0).ry(theta, 0)
    objective = ObservableObjective(
        circuit,
        {"z": QubitOperator.from_pauli_string("Z")},
    )
    backend = _CapturingSimulator(seed=11)
    prepared = SamplingEngine(backend, shots=512).compile(objective)

    result = prepared.evaluate({theta: math.pi / 4})
    assert prepared.num_experiments_per_point == 5
    assert abs(result.value()) < 0.15
    assert result.derivative() == pytest.approx([-2.0], abs=1e-12)

    request = backend.requests[0]
    circuits = dict(
        zip(
            request.labels,
            request.bound_circuits(),
            strict=True,
        )
    )
    base_angles = _rotation_angles(circuits["base/group-0"])
    first_plus = _rotation_angles(circuits["occ-0/plus/group-0"])
    second_plus = _rotation_angles(circuits["occ-1/plus/group-0"])
    assert base_angles[:2] == pytest.approx([math.pi / 4, math.pi / 4])
    assert first_plus[:2] == pytest.approx([3 * math.pi / 4, math.pi / 4])
    assert second_plus[:2] == pytest.approx([math.pi / 4, 3 * math.pi / 4])


def test_affine_shared_parameters_apply_the_full_chain_matrix() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    circuit = QuantumCircuit().ry(2 * theta + phi, 0).ry(-theta + 3 * phi, 0)
    objective = ObservableObjective(
        circuit,
        {"z": QubitOperator.from_pauli_string("Z")},
    )
    prepared = SamplingEngine(Simulator(seed=13), shots=512).compile(objective)

    result = prepared.evaluate({theta: math.pi / 2, phi: 0.0})
    assert [parameter.name for parameter in result.parameter_order] == ["phi", "theta"]
    derivatives = {
        parameter.name: value
        for parameter, value in zip(
            result.parameter_order,
            result.derivative(),
            strict=True,
        )
    }
    assert derivatives == pytest.approx({"theta": -1.0, "phi": -4.0}, abs=1e-12)


def test_sparse_qubits_follow_the_declared_clbit_string_order() -> None:
    circuit = QuantumCircuit().id(2).id(5)
    objective = ObservableObjective(
        circuit,
        {
            "z2": QubitOperator({((2, "Z"),): 1.0}),
            "z5": QubitOperator({((5, "Z"),): 1.0}),
        },
    )
    plan = _compile_plan(objective, grouping="qwc")
    assert len(plan.groups) == 1
    result = SimulationResult(
        counts={"10": 10},
        probabilities={"10": 1.0},
        num_shots=10,
        num_qubits=6,
        num_clbits=2,
    )
    decoded = _decode_group(plan, plan.groups[0], result)
    z2_index = plan.terms.index(((2, "Z"),))
    z5_index = plan.terms.index(((5, "Z"),))
    assert decoded[z2_index] == -1.0
    assert decoded[z5_index] == 1.0


def test_decode_rejects_result_count_mismatch_with_job_id() -> None:
    plan = _single_parameter_plan()
    labels = _experiment_labels(plan)
    result = _sampling_result()
    batch = ExecutionBatch(
        results=(result, result),
        labels=labels[:2],
    )

    with pytest.raises(JobError, match="expected 3 results") as raised:
        _decode_evaluation(plan, batch, shots=8, job_id="task-count")
    assert raised.value.job_id == "task-count"


def test_decode_rejects_reordered_labels_without_leaking_key_error() -> None:
    plan = _single_parameter_plan()
    labels = _experiment_labels(plan)
    result = _sampling_result()
    batch = ExecutionBatch(
        results=(result, result, result),
        labels=(labels[1], labels[0], labels[2]),
    )

    with pytest.raises(JobError, match="out of order") as raised:
        _decode_evaluation(plan, batch, shots=8, job_id="task-order")
    assert raised.value.job_id == "task-order"


def test_decode_rejects_wrong_result_shape_with_job_id() -> None:
    plan = _single_parameter_plan()
    labels = _experiment_labels(plan)
    wrong_shape = SimulationResult(
        counts={"00": 8},
        probabilities={"00": 1.0},
        num_shots=8,
        num_qubits=2,
        num_clbits=2,
    )
    batch = ExecutionBatch(
        results=(wrong_shape, wrong_shape, wrong_shape),
        labels=labels,
    )

    with pytest.raises(JobError, match="returned shape") as raised:
        _decode_evaluation(plan, batch, shots=8, job_id="task-shape")
    assert raised.value.job_id == "task-shape"


def test_submit_many_uses_one_recoverable_flattened_job() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator(seed=19)
    objective = ObservableObjective(
        QuantumCircuit().ry(theta, 0),
        {"z": QubitOperator.from_pauli_string("Z")},
    )
    prepared = SamplingEngine(backend, shots=2048).compile(objective)

    sweep_job = prepared.submit_many(({theta: 0.0}, {theta: math.pi / 2}))
    sweep = sweep_job.result()

    assert len(backend.requests) == 1
    assert len(backend.jobs) == 1
    request = backend.requests[0]
    assert len(request) == 2 * prepared.num_experiments_per_point
    assert request.result_format == "auto"
    assert request.metadata["workload_id"] == prepared.workload_id
    assert request.labels == (
        "point-0/base/group-0",
        "point-0/occ-0/plus/group-0",
        "point-0/occ-0/minus/group-0",
        "point-1/base/group-0",
        "point-1/occ-0/plus/group-0",
        "point-1/occ-0/minus/group-0",
    )
    assert sweep.values[:, 0] == pytest.approx([1.0, 0.0], abs=0.06)
    assert sweep.jacobian[:, 0, 0] == pytest.approx([0.0, -1.0], abs=0.06)
    assert sweep_job.num_points == 2
    assert sweep_job.reference is not None
    recovered = prepared.attach_many(backend.jobs[0]).result()
    assert recovered.values.tolist() == sweep.values.tolist()
    assert recovered.jacobian.tolist() == sweep.jacobian.tolist()


def test_evaluate_many_delegates_to_one_flattened_submission() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator(seed=21)
    prepared = SamplingEngine(backend, shots=256).compile(
        ObservableObjective(
            QuantumCircuit().ry(theta, 0),
            {"z": QubitOperator.from_pauli_string("Z")},
        )
    )

    result = prepared.evaluate_many(({theta: 0.0}, {theta: 0.4}))
    assert result.values.shape == (2, 1)
    assert len(backend.requests) == 1


def test_submit_many_identity_and_empty_sweeps_do_not_call_backend() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator()
    prepared = SamplingEngine(backend, shots=32).compile(
        ObservableObjective(
            QuantumCircuit().ry(theta, 0),
            {"constant": QubitOperator.identity(2.5)},
        )
    )

    empty = prepared.submit_many(()).result()
    constants = prepared.submit_many(({theta: 0.1}, {theta: 0.2})).result()
    assert empty.values.shape == (0, 1)
    assert constants.values.tolist() == [[2.5], [2.5]]
    assert backend.requests == []


def test_recovered_execution_job_can_be_attached_to_the_plan() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator(seed=17)
    objective = ObservableObjective(
        QuantumCircuit().ry(theta, 0),
        {"z": QubitOperator.from_pauli_string("Z")},
    )
    prepared = SamplingEngine(backend, shots=128).compile(objective)

    original = prepared.submit({theta: 0.0}).result()
    request = backend.requests[0]
    assert request.result_format == "auto"
    assert request.metadata["workload_id"] == prepared.workload_id
    assert backend.jobs[0].reference.workload_id == prepared.workload_id
    recovered = prepared.attach(backend.jobs[0]).result()
    assert recovered.values.tolist() == original.values.tolist()
    assert recovered.jacobian.tolist() == original.jacobian.tolist()


def test_attach_rejects_same_shape_job_from_another_observable() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator(seed=29)
    circuit = QuantumCircuit().ry(theta, 0)
    prepared_z = SamplingEngine(backend, shots=128).compile(
        ObservableObjective(
            circuit,
            {"value": QubitOperator.from_pauli_string("Z")},
        )
    )
    prepared_x = SamplingEngine(backend, shots=128).compile(
        ObservableObjective(
            circuit,
            {"value": QubitOperator.from_pauli_string("X")},
        )
    )
    wrong_job = prepared_x.submit({theta: 0.2})

    with pytest.raises(JobError, match="workload fingerprint") as raised:
        prepared_z.attach(backend.jobs[-1])
    assert raised.value.job_id == wrong_job.job_id


def test_attach_rejects_matching_plan_with_different_shot_manifest() -> None:
    theta = Parameter("theta")
    backend = _CapturingSimulator(seed=31)
    objective = ObservableObjective(
        QuantumCircuit().ry(theta, 0),
        {"z": QubitOperator.from_pauli_string("Z")},
    )
    expected = SamplingEngine(backend, shots=128).compile(objective)
    other = SamplingEngine(backend, shots=64).compile(objective)
    other.submit({theta: 0.1})

    assert expected.workload_id == other.workload_id
    with pytest.raises(JobError, match="shots"):
        expected.attach(backend.jobs[-1])


def test_symbolic_controlled_rotation_is_decomposed() -> None:
    theta = Parameter("theta")
    objective = ObservableObjective(
        QuantumCircuit().crx(theta, 0, 1),
        {"z": QubitOperator.from_pauli_string("IZ")},
    )
    prepared = SamplingEngine(Simulator()).compile(objective)
    assert prepared.parameter_order == (theta,)
    assert prepared.num_experiments_per_point > prepared.num_measurement_groups


def test_repeated_qubit_reduced_to_imaginary_coefficient_is_rejected() -> None:
    # X0 Z0 = -1j Y0 at QubitOperator construction; the coefficient is not real.
    objective = ObservableObjective(
        QuantumCircuit().h(0),
        {"z": QubitOperator({((0, "X"), (0, "Z")): 1.0})},
    )
    with pytest.raises(ValueError, match="real"):
        SamplingEngine(Simulator()).compile(objective)


def test_repeated_qubit_reduced_to_identity_is_an_exact_offset() -> None:
    # X0 X0 = I at QubitOperator construction; no measurement is needed.
    backend = _CapturingSimulator()
    objective = ObservableObjective(
        QuantumCircuit().h(0),
        {"c": QubitOperator({((0, "X"), (0, "X")): 1.5})},
    )
    prepared = SamplingEngine(backend, shots=32).compile(objective)
    assert prepared.num_measurement_groups == 0
    assert prepared.submit({}).result().value() == 1.5
    assert backend.requests == []


def test_symbolic_observable_coefficient_is_rejected() -> None:
    coefficient = Parameter("weight")
    objective = ObservableObjective(
        QuantumCircuit().id(0),
        {"z": QubitOperator({((0, "Z"),): coefficient})},
    )
    with pytest.raises(ValueError, match="symbolic coefficient"):
        SamplingEngine(Simulator()).compile(objective)


def test_sampled_values_and_gradient_agree_with_adjoint_engine() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    objective = ObservableObjective(
        QuantumCircuit().ry(theta, 0).rz(phi, 0),
        {
            "x": QubitOperator.from_pauli_string("X"),
            "z": QubitOperator.from_pauli_string("Z"),
        },
    )
    point = {theta: 0.71, phi: -0.38}
    exact = AdjointEngine().compile(objective).evaluate(point)
    sampled = SamplingEngine(Simulator(seed=23), shots=8192).compile(objective)
    observed = sampled.evaluate(point)

    assert observed.observable_order == exact.observable_order
    assert [parameter.name for parameter in observed.parameter_order] == [
        parameter.name for parameter in exact.parameter_order
    ]
    assert observed.values == pytest.approx(exact.values, abs=0.035)
    assert observed.jacobian == pytest.approx(exact.jacobian, abs=0.045)


def _rotation_angles(circuit: QuantumCircuit) -> list[float]:
    return [
        float(instruction.gate.params[0])
        for instruction in circuit.instructions
        if instruction.gate is not None and instruction.gate.name in {"rx", "ry", "rz"}
    ]


def _single_parameter_plan():
    theta = Parameter("theta")
    return _compile_plan(
        ObservableObjective(
            QuantumCircuit().ry(theta, 0),
            {"z": QubitOperator.from_pauli_string("Z")},
        ),
        grouping="qwc",
    )


def _sampling_result() -> SimulationResult:
    return SimulationResult(
        counts={"0": 8},
        probabilities={"0": 1.0},
        num_shots=8,
        num_qubits=1,
        num_clbits=1,
    )
