"""Offline tests for LogicalQubit's batch-capable backend adapter."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from quspect import Parameter, QuantumCircuit
from quspect.backends import CompositeJob, ExecutionBatch, ExecutionRequest
from quspect.backends._exceptions import PartialSubmissionError, TranslationError
from quspect.backends.logicalqubit._backend import LogicalQubitBackend


class _RawResult:
    def __init__(self, counts: dict[str, int], job_id: str) -> None:
        self.data = {
            "counts": counts,
            "shots": sum(counts.values()),
            "result_format": "counts",
            "success": True,
        }
        self.status = "completed"
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.metadata = {}

    def get_counts(self):
        return dict(self.data["counts"])


class _RawBatch:
    def __init__(self, results, job_id: str) -> None:
        self.results = list(results)
        self.backend_name = "QZ02"
        self.job_id = job_id
        self.metadata = {}


class _RawJob:
    def __init__(self, job_id: str, result, *, is_batch: bool) -> None:
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.base_url = "https://example.invalid"
        self._use_async = True
        self._result = result
        self._is_batch = is_batch
        self._batch_size = len(result.results) if is_batch else None
        self.cancelled = False

    def result(self, **kwargs):
        return self._result

    def status(self):
        return SimpleNamespace(value="completed")

    def cancel(self):
        self.cancelled = True
        return True


class _FakeSDKBackend:
    def __init__(self) -> None:
        self.name = "QZ02"
        self.base_url = "https://example.invalid"
        self.config = {
            "qubits": 20,
            "supported_modes": ["circuit", "wave"],
            "supported_measurement_types": ["measure2"],
            "topology": {"coupling_map": [[index, index + 1] for index in range(19)]},
        }
        self.calls: list[tuple[object, dict]] = []
        self.jobs: list[_RawJob] = []
        self.fail_on_call: int | None = None

    def run(self, program, **kwargs):
        self.calls.append((program, dict(kwargs)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("submission link failed")
        job_id = f"task_{len(self.calls)}"
        if isinstance(program, list):
            results = [
                _RawResult({str(index % 2): kwargs["shots"]}, f"{job_id}#{index}")
                for index in range(len(program))
            ]
            job = _RawJob(job_id, _RawBatch(results, job_id), is_batch=True)
        else:
            job = _RawJob(
                job_id,
                _RawResult({"0": kwargs["shots"]}, job_id),
                is_batch=False,
            )
        self.jobs.append(job)
        return job


class _IdentityTranslator:
    def __init__(self) -> None:
        self.documents: list[str] = []

    def translate(self, document: str):
        self.documents.append(document)
        return document


@pytest.fixture
def backend() -> LogicalQubitBackend:
    raw = _FakeSDKBackend()
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()
    return backend


def _measured(value: int = 0, *, name: str = "") -> QuantumCircuit:
    circuit = QuantumCircuit(name=name)
    if value:
        circuit.x(0)
    return circuit.measure([0], [0])


def _grover_target_101() -> QuantumCircuit:
    num_search_qubits = 3
    target = 5
    beta = math.asin(1 / math.sqrt(2**num_search_qubits))
    rounds = int((math.pi / 2 - beta) / (2 * beta))
    phase = 2 * math.asin(math.sin(math.pi / (4 * rounds + 6)) / math.sin(beta))

    circuit = QuantumCircuit(name="grover_long_n8_target101")
    for qubit in range(num_search_qubits):
        circuit.h(qubit)

    for _ in range(rounds + 1):
        for qubit in range(num_search_qubits):
            if not (target >> qubit) & 1:
                circuit.x(qubit)
        circuit.ccx(0, 1, 3).cp(phase, 3, 2).ccx(0, 1, 3)
        for qubit in range(num_search_qubits):
            if not (target >> qubit) & 1:
                circuit.x(qubit)

        for qubit in range(num_search_qubits):
            circuit.h(qubit).x(qubit)
        circuit.ccx(0, 1, 3).cp(phase, 3, 2).ccx(0, 1, 3)
        for qubit in range(num_search_qubits):
            circuit.x(qubit).h(qubit)

    return circuit.measure([0, 1, 2], [0, 1, 2])


def test_submit_request_uses_one_native_batch_and_preserves_order(
    backend: LogicalQubitBackend,
) -> None:
    request = ExecutionRequest.batch(
        (_measured(0), _measured(1)),
        labels=("zero", "one"),
        shots=16,
        options={"transpile": False, "readout_correction": True},
    )

    job = backend.submit_request(request)
    assert len(backend._lq_backend.calls) == 1
    program, options = backend._lq_backend.calls[0]
    assert isinstance(program, list) and len(program) == 2
    assert options == {
        "shots": 16,
        "result_format": "counts",
        "readout_correction": True,
    }

    result = job.result()
    assert isinstance(result, ExecutionBatch)
    assert result.labels == ("zero", "one")
    assert result.get_counts() == [{"0": 16}, {"1": 16}]


def test_auto_result_format_is_left_for_sdk_width_selection(
    backend: LogicalQubitBackend,
) -> None:
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.measure(tuple(range(12)), tuple(range(12)))
    request = ExecutionRequest.single(
        circuit,
        shots=16,
        result_format="auto",
        options={"transpile": False},
    )

    job = backend.submit_request(request)
    _, options = backend._lq_backend.calls[0]
    assert options == {"shots": 16}
    assert job.reference.result_format == "auto"


def test_more_than_64_experiments_becomes_recoverable_composite(
    backend: LogicalQubitBackend,
) -> None:
    circuits = tuple(_measured(index % 2) for index in range(65))
    labels = tuple(f"point-{index}" for index in range(65))
    request = ExecutionRequest.batch(
        circuits,
        labels=labels,
        shots=4,
        options={"transpile": False},
    )

    job = backend.submit_request(request)
    assert isinstance(job, CompositeJob)
    assert [len(call[0]) for call in backend._lq_backend.calls] == [64, 1]
    assert job.reference.batch_size == 65
    assert job.reference.child_job_ids == ("task_1", "task_2")

    result = job.result()
    assert result.labels == labels
    assert len(result) == 65


def test_partial_chunk_submission_exposes_created_jobs_and_cancels_them(
    backend: LogicalQubitBackend,
) -> None:
    backend._lq_backend.fail_on_call = 2
    request = ExecutionRequest.batch(
        tuple(_measured() for _ in range(65)),
        shots=4,
        options={"transpile": False},
    )

    with pytest.raises(PartialSubmissionError, match="after 64") as raised:
        backend.submit_request(request)
    error = raised.value
    assert error.submitted_job_ids == ("task_1",)
    assert error.cancellation_failures == ()
    assert backend._lq_backend.jobs[0].cancelled is True


def test_sdk_fallback_reference_infers_async_transport_from_children(
    backend: LogicalQubitBackend,
) -> None:
    raw_job = _RawJob(
        "task_wrapper",
        _RawResult({"0": 4}, "task_wrapper"),
        is_batch=False,
    )
    raw_job._use_async = False  # lqcloud 0.4.1 fallback-wrapper default
    raw_job._children = (
        SimpleNamespace(job_id="task_a", _use_async=True),
        SimpleNamespace(job_id="task_b", _use_async=True),
    )

    reference = backend._make_reference(
        raw_job,
        batch_size=2,
        labels=("a", "b"),
        shapes=((1, 1), (1, 1)),
        shots=4,
        result_format="auto",
        workload_id="sampling-v1:test",
    )

    assert reference.transport == "async"
    assert reference.child_job_ids == ("task_a", "task_b")


def test_individual_provider_batching_avoids_sdk_batch_fallback(
    backend: LogicalQubitBackend,
) -> None:
    request = ExecutionRequest.batch(
        (_measured(0), _measured(1)),
        labels=("zero", "one"),
        shots=4,
        options={"transpile": False, "provider_batching": "individual"},
    )

    job = backend.submit_request(request)
    assert isinstance(job, CompositeJob)
    assert len(backend._lq_backend.calls) == 2
    assert all(
        not isinstance(program, list) for program, _ in backend._lq_backend.calls
    )
    assert job.reference.labels == ("zero", "one")
    assert job.result().get_counts() == [{"0": 4}, {"0": 4}]


def test_individual_provider_batching_reports_every_created_task_on_failure(
    backend: LogicalQubitBackend,
) -> None:
    backend._lq_backend.fail_on_call = 2
    request = ExecutionRequest.batch(
        (_measured(), _measured()),
        shots=4,
        options={"transpile": False, "provider_batching": "individual"},
    )

    with pytest.raises(PartialSubmissionError, match="after 1") as raised:
        backend.submit_request(request)
    assert raised.value.submitted_job_ids == ("task_1",)
    assert backend._lq_backend.jobs[0].cancelled is True


def test_sweep_binds_before_translation(backend: LogicalQubitBackend) -> None:
    theta = Parameter("theta")
    template = QuantumCircuit().ry(theta, 0).measure_all()
    request = ExecutionRequest.sweep(
        template,
        ({theta: 0.25}, {"theta": 0.75}),
        shots=8,
        options={"transpile": False},
    )

    backend.submit_request(request)
    translator = backend._translator
    assert isinstance(translator, _IdentityTranslator)
    angles = [
        json.loads(document)["circuit"]["instructions"][0]["gate"]["params"][0]
        for document in translator.documents
    ]
    assert angles == [0.25, 0.75]


def test_legacy_submit_and_run_still_return_one_result(
    backend: LogicalQubitBackend,
) -> None:
    job = backend.submit(_measured(), shots=7, transpile=False)
    assert job.result().counts == {"0": 7}

    result = backend.run(_measured(), shots=9, transpile=False)
    assert result.counts == {"0": 9}


def test_rejects_unknown_options_before_sdk_submission(
    backend: LogicalQubitBackend,
) -> None:
    request = ExecutionRequest.single(
        _measured(),
        options={"transpile": False, "typo_option": True},
    )
    with pytest.raises(ValueError, match="typo_option"):
        backend.submit_request(request)
    assert backend._lq_backend.calls == []


def test_rejects_unmeasured_hardware_sampling(
    backend: LogicalQubitBackend,
) -> None:
    request = ExecutionRequest.single(
        QuantumCircuit().h(0),
        options={"transpile": False},
    )
    with pytest.raises(TranslationError, match="no measurement"):
        backend.submit_request(request)


def test_qz01_missing_native_gates_lowers_ccx_before_linear_routing() -> None:
    """QZ01's absent native metadata must not leave CCX for SDK lowering."""
    raw = _FakeSDKBackend()
    raw.name = "QZ01-repetition_code"
    raw.config["qubits"] = 17
    raw.config["native_gates"] = None
    raw.config["topology"] = {
        "coupling_map": [[index, index + 1] for index in range(16)]
    }
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()
    source = _grover_target_101()

    backend.submit(
        source,
        shots=8,
        physical_qubits=[7, 8, 9, 10],
    )

    document, options = raw.calls[0]
    assert isinstance(document, str)
    instructions = json.loads(document)["circuit"]["instructions"]
    gates = [item for item in instructions if item["type"] == "GATE"]
    gate_names = [item["gate"]["name"] for item in gates]
    assert "ccx" not in gate_names
    assert "cp" in gate_names  # The fallback is targeted, not a full lowering.
    assert all(item["gate"]["num_qubits"] <= 2 for item in gates)

    layout = options["initial_layout"]
    assert set(layout) == {7, 8, 9, 10}
    coupling = {tuple(sorted(edge)) for edge in raw.config["topology"]["coupling_map"]}
    violations = []
    for item in gates:
        if item["gate"]["num_qubits"] == 2:
            edge = tuple(sorted(layout[qubit] for qubit in item["qubits"]))
            if edge not in coupling:
                violations.append((item["gate"]["name"], item["qubits"], edge))
    assert violations == []

    from quspect import Simulator
    from quspect.ir.qsal import loads

    routed = loads(document).circuit
    assert routed is not None
    expected = Simulator(seed=7).run(source, shots=32).counts
    actual = Simulator(seed=7).run(routed, shots=32).counts
    assert actual == expected == {"101": 32}


def test_advertised_native_ccx_is_still_lowered_before_pairwise_routing() -> None:
    """A native label cannot make a three-qubit gate pairwise routable."""
    raw = _FakeSDKBackend()
    raw.config["qubits"] = 3
    raw.config["native_gates"] = ["ccx", "cz", "h", "t", "tdg"]
    raw.config["topology"] = {"coupling_map": [[0, 1], [1, 2]]}
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()
    circuit = QuantumCircuit().ccx(0, 1, 2).measure_all()

    backend.submit(circuit, shots=8)

    document, _ = raw.calls[0]
    assert isinstance(document, str)
    instructions = json.loads(document)["circuit"]["instructions"]
    gate_names = [
        item["gate"]["name"] for item in instructions if item["type"] == "GATE"
    ]
    assert "ccx" not in gate_names
    assert "cx" not in gate_names
    assert "cz" in gate_names


def test_transpile_preserves_results_after_swap_routing() -> None:
    raw = _FakeSDKBackend()
    raw.config["qubits"] = 5
    raw.config["topology"] = {"coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4]]}
    backend = LogicalQubitBackend(raw)

    circuit = QuantumCircuit(name="routing_regression")
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.measure([0, 1, 2, 3], [3, 1, 0, 2])

    routed, layout = backend._transpile(circuit)

    assert layout is not None
    assert any(
        instruction.is_gate
        and instruction.gate is not None
        and instruction.gate.name == "swap"
        for instruction in routed
    )
    assert all(instruction.is_measure for instruction in routed.instructions[-4:])

    from quspect import Simulator

    expected = Simulator(seed=7).run(circuit, shots=32).counts
    actual = Simulator(seed=7).run(routed, shots=32).counts
    assert actual == expected


def test_submit_compiles_within_selected_physical_qubits() -> None:
    raw = _FakeSDKBackend()
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()

    circuit = QuantumCircuit(name="selected_physical_pool")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.measure_all()

    backend.submit(
        circuit,
        shots=8,
        physical_qubits=[7, 8, 9, 10, 11],
    )

    _, options = raw.calls[0]
    layout = options["initial_layout"]
    assert set(layout) <= {7, 8, 9, 10, 11}
    assert len(layout) >= circuit.num_qubits
    assert "physical_qubits" not in options


def test_submit_routes_from_requested_initial_layout() -> None:
    raw = _FakeSDKBackend()
    raw.config["qubits"] = 5
    raw.config["topology"] = {"coupling_map": [[0, 1], [1, 2], [2, 3], [3, 4]]}
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()

    circuit = QuantumCircuit(name="fixed_layout")
    circuit.h(0).cx(0, 1).measure_all()

    backend.submit(
        circuit,
        shots=8,
        initial_layout=[0, 2],
        physical_qubits=[0, 1, 2],
    )

    translated, options = raw.calls[0]
    layout = options["initial_layout"]
    assert layout[:2] == [0, 2]
    assert set(layout) == {0, 1, 2}
    instructions = json.loads(translated)["circuit"]["instructions"]
    assert any(
        instruction.get("gate", {}).get("name") == "swap"
        for instruction in instructions
    )


def test_batch_compiles_with_shared_selected_physical_qubits(
    backend: LogicalQubitBackend,
) -> None:
    circuit = QuantumCircuit().h(0).cx(0, 1).measure_all()
    request = ExecutionRequest.batch(
        (circuit, circuit),
        shots=8,
        options={
            "transpile": True,
            "initial_layout": [7, 9],
            "physical_qubits": [7, 8, 9],
        },
    )

    backend.submit_request(request)

    assert len(backend._lq_backend.calls) == 1
    _, options = backend._lq_backend.calls[0]
    assert options["initial_layout"][:2] == [7, 9]
    assert set(options["initial_layout"]) == {7, 8, 9}
    assert "physical_qubits" not in options


def test_transpile_false_keeps_direct_initial_layout_passthrough(
    backend: LogicalQubitBackend,
) -> None:
    backend.submit(
        _measured(),
        shots=8,
        transpile=False,
        initial_layout=[7],
    )

    _, options = backend._lq_backend.calls[0]
    assert options["initial_layout"] == [7]


def test_transpile_false_preserves_ccx_for_direct_provider_translation() -> None:
    raw = _FakeSDKBackend()
    backend = LogicalQubitBackend(raw)
    backend._translator = _IdentityTranslator()
    circuit = QuantumCircuit().ccx(0, 1, 2).measure_all()

    backend.submit(circuit, shots=8, transpile=False)

    document, _ = raw.calls[0]
    assert isinstance(document, str)
    instructions = json.loads(document)["circuit"]["instructions"]
    assert any(item.get("gate", {}).get("name") == "ccx" for item in instructions)


def test_physical_qubits_requires_transpilation(
    backend: LogicalQubitBackend,
) -> None:
    with pytest.raises(ValueError, match="requires transpile=True"):
        backend.submit(
            _measured(),
            shots=8,
            transpile=False,
            physical_qubits=[7],
        )


def test_initial_layout_must_be_inside_selected_physical_qubits(
    backend: LogicalQubitBackend,
) -> None:
    with pytest.raises(TranslationError, match="outside physical_qubits"):
        backend.submit(
            QuantumCircuit().h(0).cx(0, 1).measure_all(),
            shots=8,
            initial_layout=[7, 9],
            physical_qubits=[7, 8],
        )


def test_selected_physical_qubits_must_be_connected(
    backend: LogicalQubitBackend,
) -> None:
    with pytest.raises(TranslationError, match="connected coupling subgraph"):
        backend.submit(
            QuantumCircuit().h(0).cx(0, 1).measure_all(),
            shots=8,
            physical_qubits=[1, 2, 7, 8],
        )


def test_capabilities_come_from_backend_metadata(
    backend: LogicalQubitBackend,
) -> None:
    capabilities = backend.capabilities
    assert capabilities.supports_batch
    assert not capabilities.supports_estimator
    assert capabilities.max_circuits_per_job == 64
    assert capabilities.max_shots == 50_000
    assert capabilities.supported_modes == frozenset({"circuit", "wave"})
    assert backend.num_qubits == 20


@pytest.mark.parametrize("workload_id", [" ", "", 7])
def test_invalid_workload_id_is_rejected_before_any_sdk_submission(
    backend: LogicalQubitBackend, workload_id: object
) -> None:
    request = ExecutionRequest.single(
        _measured(), options={"transpile": False}, metadata={"workload_id": workload_id}
    )
    with pytest.raises(ValueError, match="workload_id"):
        backend.submit_request(request)
    assert backend._lq_backend.calls == []
