"""Offline tests for ``quspect.backends.logicalqubit``.

The ``lqcloud`` SDK is **not** installed in this environment, and these tests
must never reach the network.  Every test therefore runs with:

* an autouse fixture that replaces ``socket.socket.connect`` /
  ``connect_ex``, ``socket.create_connection`` and ``socket.getaddrinfo``
  with functions that raise ``RuntimeError`` (any accidental network access
  fails loudly), and
* a fake ``lqcloud`` package injected into ``sys.modules`` (provider, job,
  exceptions and a recording ``QuantumCircuit``).

The translator output (a recording circuit) is converted back to qiskit and
compared with the source circuit's unitary.  The backend is driven end-to-end
by a fake SDK backend that "executes" the translated circuit by sampling from
qiskit's exact distribution, so the full prepare -> transpile -> translate ->
submit -> convert pipeline is cross-checked against qiskit.

Assumption: the fake SDK returns counts in quspect's bitstring convention
(clbit 0 leftmost) because ``_convert_child`` passes provider bitstrings
through unchanged; the real lqcloud wire convention cannot be verified
offline.
"""

from __future__ import annotations

import math
import socket
import sys
import types
import zlib
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Operator

import quspect
from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_counts_match,
    assert_unitary_equal,
    qiskit_probabilities,
    random_circuit_pair,
    to_qiskit,
)
from quspect.backends import CompositeJob, ExecutionBatch, ExecutionRequest, JobStatus
from quspect.backends._exceptions import (
    BackendError,
    JobError,
    JobTimeoutError,
    OptionalDependencyError,
    PartialSubmissionError,
    TranslationError,
)
from quspect.backends._execution import JobReference
from quspect.backends.logicalqubit import (
    LogicalQubitBackend,
    LogicalQubitJob,
    LogicalQubitProvider,
)
from quspect.backends.logicalqubit import _backend as lq_backend_mod
from quspect.backends.logicalqubit import _job as lq_job_mod
from quspect.backends.logicalqubit import _provider as lq_provider_mod
from quspect.backends.logicalqubit._translator import LQCircuitTranslator
from quspect.circuit.parameter import Parameter
from quspect.ir.qsal import dumps

SHOTS = 8000
BASE_URL = "https://lq.fake.invalid"


# =============================================================================
# Network guard + fake lqcloud SDK
# =============================================================================


def _no_network(*args: Any, **kwargs: Any) -> Any:
    """Replacement for socket primitives: any network attempt fails."""
    raise RuntimeError("network access is forbidden in qiskit_tests")


class FakeLQCircuit:
    """Recording stand-in for ``lqcloud.circuit.circuit.QuantumCircuit``."""

    def __init__(self, num_qubits: int, num_clbits: int = 0) -> None:
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.ops: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args: Any) -> FakeLQCircuit:
            self.ops.append((name, args))
            return self

        return record


class FakeLQJobError(Exception):
    """Stand-in for ``lqcloud.exceptions.JobError``."""

    def __init__(
        self, message: str, error_type: str | None = None, job_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.job_id = job_id


class FakeLQJobTimeoutError(Exception):
    """Stand-in for ``lqcloud.exceptions.JobTimeoutError`` (not a JobError)."""


class FakeSDKJob:
    """Stand-in for ``lqcloud.job.Job`` as constructed by the provider."""

    def __init__(
        self,
        job_id: str,
        provider: Any,
        *,
        backend_name: str,
        base_url: str,
        use_async: bool,
    ) -> None:
        self.job_id = job_id
        self.provider = provider
        self.backend_name = backend_name
        self.base_url = base_url
        self._use_async = use_async


class FakeLQCloudProvider:
    """Stand-in for ``lqcloud.LQCloudProvider``."""

    instances: list[FakeLQCloudProvider] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.url = kwargs.get("url", BASE_URL)
        self.backend_calls: list[tuple[str, dict[str, Any]]] = []
        FakeLQCloudProvider.instances.append(self)

    def get_backend(self, name: str, **kwargs: Any) -> Any:
        """Return a fake SDK backend and record the call."""
        self.backend_calls.append((name, kwargs))
        return FakeSDKBackend(name)

    def get_backends(self) -> Iterator[dict[str, Any]]:
        """Yield backend metadata."""
        yield {"name": "QZ02", "status": "online"}
        yield {"name": "QZ01", "status": "offline"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every socket connection attempt raise RuntimeError."""
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)


@pytest.fixture(autouse=True)
def fake_lqcloud(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``lqcloud`` package in sys.modules for each test."""
    root = types.ModuleType("lqcloud")
    circuit_pkg = types.ModuleType("lqcloud.circuit")
    circuit_mod = types.ModuleType("lqcloud.circuit.circuit")
    job_mod = types.ModuleType("lqcloud.job")
    exc_mod = types.ModuleType("lqcloud.exceptions")
    circuit_mod.QuantumCircuit = FakeLQCircuit  # type: ignore[attr-defined]
    job_mod.Job = FakeSDKJob  # type: ignore[attr-defined]
    exc_mod.JobError = FakeLQJobError  # type: ignore[attr-defined]
    exc_mod.JobTimeoutError = FakeLQJobTimeoutError  # type: ignore[attr-defined]
    root.LQCloudProvider = FakeLQCloudProvider  # type: ignore[attr-defined]
    root.circuit = circuit_pkg  # type: ignore[attr-defined]
    root.job = job_mod  # type: ignore[attr-defined]
    root.exceptions = exc_mod  # type: ignore[attr-defined]
    circuit_pkg.circuit = circuit_mod  # type: ignore[attr-defined]
    for name, mod in {
        "lqcloud": root,
        "lqcloud.circuit": circuit_pkg,
        "lqcloud.circuit.circuit": circuit_mod,
        "lqcloud.job": job_mod,
        "lqcloud.exceptions": exc_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(lq_provider_mod, "_HAS_LQCLOUD", True)
    monkeypatch.setattr(lq_provider_mod, "lqcloud", root)
    monkeypatch.delenv("LQCLOUD_API_KEY", raising=False)
    monkeypatch.delenv("LQCLOUD_URL", raising=False)
    FakeLQCloudProvider.instances.clear()
    return root


def test_network_guard_blocks_connections() -> None:
    """The autouse guard turns any connection attempt into RuntimeError."""
    with pytest.raises(RuntimeError, match="forbidden"):
        socket.create_connection(("example.com", 443))
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="forbidden"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


# =============================================================================
# Recorded lqcloud circuit -> qiskit
# =============================================================================


def lq_to_qiskit(lqc: FakeLQCircuit) -> qiskit.QuantumCircuit:
    """Rebuild a qiskit circuit from the operations recorded on FakeLQCircuit."""
    qc = (
        qiskit.QuantumCircuit(lqc.num_qubits, lqc.num_clbits)
        if lqc.num_clbits
        else qiskit.QuantumCircuit(lqc.num_qubits)
    )
    for name, args in lqc.ops:
        if name == "barrier":
            if args:
                qc.barrier(*args)
            else:
                qc.barrier()
        elif name == "measure":
            qc.measure(args[0], args[1])
        else:
            getattr(qc, name)(*args)
    return qc


def _unitary_part(qc: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Return a copy of ``qc`` without measurements and barriers."""
    out = qiskit.QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in {"measure", "barrier"}:
            continue
        out.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    return out


def _translate(circ: quspect.QuantumCircuit) -> FakeLQCircuit:
    """Translate a quspect circuit through QSAL into a recording circuit."""
    out = LQCircuitTranslator().translate(dumps(circ))
    assert isinstance(out, FakeLQCircuit)
    return out


# =============================================================================
# Translator
# =============================================================================

_PHASE_ONLY_GATES = {"p", "u1"}  # lowered to rz: equal up to global phase


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_translator_each_gate_matches_qiskit_unitary(name: str) -> None:
    """Every supported gate translates to an equivalent operation sequence."""
    spec = GATE_SPECS[name]
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for _ in range(4):
        params = [
            float(v) for v in rng.uniform(-2 * math.pi, 2 * math.pi, spec.num_params)
        ]
        n = max(spec.num_qubits, 3)
        qubits = [int(q) for q in rng.permutation(n)[: spec.num_qubits]]
        circ = quspect.QuantumCircuit().id(n - 1)
        spec.apply_quspect(circ, params, qubits)
        lqc = _translate(circ)
        assert (lqc.num_qubits, lqc.num_clbits) == (n, 0)
        got = Operator(lq_to_qiskit(lqc))
        expected = Operator(to_qiskit(circ))
        assert_unitary_equal(got, expected, up_to_phase=True, atol=1e-9)
        if name not in _PHASE_ONLY_GATES:
            assert_unitary_equal(got, expected, atol=1e-9)


def test_translator_gate_name_mapping_and_params() -> None:
    """Gate names, qubit order and parameters are forwarded exactly."""
    circ = (
        quspect.QuantumCircuit()
        .h(2)
        .cx(2, 0)
        .rx(0.25, 1)
        .p(0.5, 0)
        .u2(0.1, 0.2, 1)
        .u3(0.3, 0.4, 0.5, 2)
        .ccx(1, 2, 0)
        .crz(0.7, 0, 1)
    )
    assert _translate(circ).ops == [
        ("h", (2,)),
        ("cx", (2, 0)),
        ("rx", (0.25, 1)),
        ("rz", (0.5, 0)),
        ("u", (math.pi / 2, 0.1, 0.2, 1)),
        ("u", (0.3, 0.4, 0.5, 2)),
        ("ccx", (1, 2, 0)),
        ("crz", (0.7, 0, 1)),
    ]


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(6))
def test_translator_random_circuits_match_qiskit(seed: int) -> None:
    """Random circuits over all gates translate to the same unitary."""
    circ, qc = random_circuit_pair(4, 30, 500 + seed)
    got = Operator(lq_to_qiskit(_translate(circ)))
    assert_unitary_equal(got, Operator(qc), up_to_phase=True, atol=1e-8)


def test_translator_inserts_barrier_before_measure_and_keeps_clbit_map() -> None:
    """Measurements get a covering barrier and keep their qubit->clbit map."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).x(2)
    circ.measure([2, 0], [0, 1])
    lqc = _translate(circ)
    assert (lqc.num_qubits, lqc.num_clbits) == (3, 2)
    assert lqc.ops[-3:] == [
        ("barrier", (2, 0)),
        ("measure", (2, 0)),
        ("measure", (0, 1)),
    ]


def test_translator_barrier_tracking_partial_and_repeated() -> None:
    """Existing barriers are reused; gates and measurements invalidate them."""
    circ = quspect.QuantumCircuit().h(0).h(1).h(2)
    circ.barrier()  # covers all three qubits
    circ.measure([0], [0])  # covered: no extra barrier
    circ.x(1)  # invalidates qubit 1
    circ.measure([1, 2], [1, 2])  # needs barrier on 1 only
    circ.measure([0], [0])  # 0 consumed by first measure: barrier again
    circ.barrier(2)
    ops = _translate(circ).ops
    tail = ops[3:]
    assert tail == [
        ("barrier", (0, 1, 2)),
        ("measure", (0, 0)),
        ("x", (1,)),
        ("barrier", (1,)),
        ("measure", (1, 1)),
        ("measure", (2, 2)),
        ("barrier", (0,)),
        ("measure", (0, 0)),
        ("barrier", (2,)),
    ]


def test_translator_clbit_only_and_qubit_only_circuits() -> None:
    """Circuits with only qubits are built without classical register."""
    lqc = _translate(quspect.QuantumCircuit().h(0))
    assert (lqc.num_qubits, lqc.num_clbits) == (1, 0)


def _qsal(instructions: list[dict[str, Any]], nq: int = 1, nc: int = 1) -> str:
    """Return a minimal QSAL JSON document."""
    import json

    return json.dumps(
        {"circuit": {"num_qubits": nq, "num_clbits": nc, "instructions": instructions}}
    )


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ("{not json", "Failed to parse"),
        ('{"name": "x"}', "no circuit section"),
        (_qsal([], nq=0, nc=0), "zero qubits"),
        (_qsal([{"type": "GATE", "qubits": [0]}]), "missing gate data"),
        (
            _qsal(
                [
                    {
                        "type": "GATE",
                        "qubits": [0],
                        "gate": {"name": "rx", "params": [{"_param": "t"}]},
                    }
                ]
            ),
            "unbound",
        ),
        (
            _qsal([{"type": "GATE", "qubits": [0], "gate": {"name": "iswap"}}]),
            "not supported",
        ),
        (
            _qsal([{"type": "MEASURE", "qubits": [0], "clbits": []}]),
            "classical bits",
        ),
        (_qsal([{"type": "CHANNEL", "qubits": [0]}]), "noise channels"),
        (_qsal([{"type": "RESET", "qubits": [0]}]), "unknown instruction"),
    ],
)
def test_translator_errors(doc: str, match: str) -> None:
    """Malformed or unsupported QSAL raises TranslationError."""
    with pytest.raises(TranslationError, match=match):
        LQCircuitTranslator().translate(doc)


def test_translator_rejects_quspect_noise_channel() -> None:
    """A quspect circuit with a noise channel cannot be translated."""
    circ = quspect.QuantumCircuit().h(0).depolarize(0.1, 0)
    with pytest.raises(TranslationError, match="noise"):
        _translate(circ)


def test_translator_empty_barrier_covers_all_qubits() -> None:
    """A QSAL barrier without qubits becomes a full barrier."""
    doc = _qsal(
        [
            {"type": "GATE", "qubits": [1], "gate": {"name": "x"}},
            {"type": "BARRIER", "qubits": []},
            {"type": "MEASURE", "qubits": [1], "clbits": [0]},
        ],
        nq=2,
    )
    ops = LQCircuitTranslator().translate(doc).ops
    assert ops == [("x", (1,)), ("barrier", ()), ("measure", (1, 0))]


def test_translator_accepts_i_alias_and_uppercase() -> None:
    """Gate names are case-insensitive and 'i' maps to identity."""
    doc = _qsal(
        [
            {"type": "GATE", "qubits": [0], "gate": {"name": "I"}},
            {"type": "GATE", "qubits": [0], "gate": {"name": "H"}},
        ],
        nc=0,
    )
    lqc = LQCircuitTranslator().translate(doc)
    assert lqc.ops == [("id", (0,)), ("h", (0,))]


# =============================================================================
# Fake SDK backend / jobs / results
# =============================================================================


class RawResult:
    """Stand-in for one lqcloud ``Result``."""

    def __init__(
        self,
        counts: dict[str, Any] | None = None,
        *,
        job_id: str = "r",
        status: str = "completed",
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        fallback: Any = None,
    ) -> None:
        if data is None:
            data = {"counts": counts or {}, "shots": sum((counts or {}).values())}
        self.data = data
        self.status = status
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.metadata = metadata if metadata is not None else {}
        self._fallback = fallback

    def get_counts(self) -> Any:
        """Return the fallback counts or raise if configured with an error."""
        if isinstance(self._fallback, Exception):
            raise self._fallback
        return self._fallback


class RawBatch:
    """Stand-in for lqcloud ``BatchResult``."""

    def __init__(self, results: list[RawResult], job_id: str = "b") -> None:
        self.results = results
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.metadata = {"queue": 3, "stdout": "dropped"}


class RawJob:
    """Stand-in for an lqcloud job handle."""

    def __init__(
        self,
        job_id: str,
        result: Any = None,
        *,
        status: Any = "completed",
        cancel: Any = True,
        is_batch: bool = False,
        batch_size: int | None = None,
    ) -> None:
        self.job_id = job_id
        self.backend_name = "QZ02"
        self.base_url = BASE_URL
        self._use_async = True
        self._result = result
        self._status = status
        self._cancel = cancel
        self._is_batch = is_batch
        self._batch_size = batch_size
        self.result_calls: list[dict[str, Any]] = []
        self.cancelled = False

    def result(self, **kwargs: Any) -> Any:
        """Record kwargs and return / raise the scripted result."""
        self.result_calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    def status(self) -> Any:
        """Return / raise the scripted status."""
        if isinstance(self._status, Exception):
            raise self._status
        return self._status

    def cancel(self) -> Any:
        """Return / raise the scripted cancellation response."""
        self.cancelled = True
        if isinstance(self._cancel, Exception):
            raise self._cancel
        return self._cancel


def _sample(lqc: FakeLQCircuit, shots: int, rng: np.random.Generator) -> dict[str, int]:
    """Sample quspect-keyed counts for a recorded circuit using qiskit."""
    probs = qiskit_probabilities(lq_to_qiskit(lqc))
    keys = sorted(probs)
    p = np.array([probs[k] for k in keys])
    draws = rng.multinomial(shots, p / p.sum())
    return {k: int(c) for k, c in zip(keys, draws, strict=True) if c}


class FakeSDKBackend:
    """Fake lqcloud backend that executes recorded circuits with qiskit."""

    def __init__(self, name: str = "QZ02", config: Any = None) -> None:
        self.name = name
        self.base_url = BASE_URL
        self.config = (
            config
            if config is not None
            else {
                "qubits": 6,
                "native_gates": [],
                "supported_modes": ["circuit"],
                "topology": {"coupling_map": [[i, i + 1] for i in range(5)]},
            }
        )
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.jobs: list[RawJob] = []
        self.fail_on_call: int | None = None
        self.cancel_result: Any = True
        self.job_id_override: str | None = None
        self.rng = np.random.default_rng(2024)

    def run(self, program: Any, **kwargs: Any) -> RawJob:
        """Record the submission and return a job with sampled results."""
        self.calls.append((program, dict(kwargs)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("provider rejected submission")
        shots = int(kwargs["shots"])
        job_id = self.job_id_override or f"task_{len(self.calls)}"
        if isinstance(program, list):
            result: Any = RawBatch(
                [
                    RawResult(_sample(c, shots, self.rng), job_id=job_id)
                    for c in program
                ],
                job_id=job_id,
            )
            job = RawJob(
                job_id,
                result,
                cancel=self.cancel_result,
                is_batch=True,
                batch_size=len(program),
            )
        else:
            result = RawResult(_sample(program, shots, self.rng), job_id=job_id)
            job = RawJob(job_id, result, cancel=self.cancel_result)
        self.jobs.append(job)
        return job


@pytest.fixture
def sdk() -> FakeSDKBackend:
    """A fresh fake SDK backend on a 6-qubit line."""
    return FakeSDKBackend()


@pytest.fixture
def backend(sdk: FakeSDKBackend) -> LogicalQubitBackend:
    """A LogicalQubitBackend wrapping the fake SDK backend."""
    return LogicalQubitBackend(sdk)


# =============================================================================
# LogicalQubitBackend: end-to-end against qiskit
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize("transpile", [True, False])
def test_backend_run_matches_qiskit_distribution(
    backend: LogicalQubitBackend, transpile: bool
) -> None:
    """run() through translate (+transpile) reproduces qiskit's distribution."""
    circ = quspect.QuantumCircuit(name="ghz").h(0).cx(0, 1).cx(1, 2).ry(0.7, 3)
    circ.cx(3, 0).measure_all()
    result = backend.run(circ, shots=SHOTS, transpile=transpile, timeout=5.0)
    assert result.num_shots == SHOTS
    assert (result.num_qubits, result.num_clbits) == (4, 4)
    assert_counts_match(result.counts, qiskit_probabilities(circ), SHOTS)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize("seed", range(3))
def test_backend_batch_with_routing_matches_qiskit(
    backend: LogicalQubitBackend, seed: int
) -> None:
    """A routed native batch preserves every experiment's distribution."""
    circuits = []
    for k in range(3):
        circ, _ = random_circuit_pair(
            4,
            20,
            900 + 10 * seed + k,
            gate_set=["h", "rx", "ry", "cx", "cz", "ccx", "swap"],
        )
        circ.measure([3, 1, 0], [0, 1, 2])
        circuits.append(circ)
    request = ExecutionRequest.batch(circuits, shots=SHOTS)
    batch = backend.execute(request)
    assert len(batch) == 3
    for circ, res in zip(circuits, batch, strict=True):
        assert (res.num_qubits, res.num_clbits) == (4, 3)
        assert_counts_match(res.counts, qiskit_probabilities(circ), SHOTS)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_backend_sweep_matches_qiskit(backend: LogicalQubitBackend) -> None:
    """Parameter sweeps are bound before translation."""
    theta = Parameter("theta")
    template = quspect.QuantumCircuit().ry(theta, 0).cx(0, 1).measure_all()
    request = ExecutionRequest.sweep(
        template,
        [{"theta": 0.3}, {"theta": 2.0}],
        shots=SHOTS,
        options={"transpile": False},
    )
    batch = backend.submit_request(request).result()
    assert isinstance(batch, ExecutionBatch)
    assert batch.labels == ("point-0", "point-1")
    for value, res in zip((0.3, 2.0), batch, strict=True):
        bound = template.bind_parameters({theta: value})
        assert_counts_match(res.counts, qiskit_probabilities(bound), SHOTS)


# =============================================================================
# LogicalQubitBackend: submission, options and errors
# =============================================================================


def _measured(name: str = "") -> quspect.QuantumCircuit:
    """Return a 1-qubit measured circuit."""
    circ = quspect.QuantumCircuit(name=name).x(0)
    circ.measure([0], [0])
    return circ


def test_backend_submit_forwards_options_and_reference(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """submit() forwards shots / provider options and builds a reference."""
    job = backend.submit(
        _measured("one"),
        shots=10,
        transpile=False,
        initial_layout=(4,),
        readout_correction=True,
        result_format="memory",
    )
    program, kwargs = sdk.calls[0]
    assert isinstance(program, FakeLQCircuit)
    assert kwargs == {
        "shots": 10,
        "initial_layout": [4],
        "readout_correction": True,
        "result_format": "memory",
    }
    assert isinstance(job, LogicalQubitJob)
    ref = job.reference
    assert ref.platform == "logicalqubit"
    assert ref.backend_name == "QZ02"
    assert ref.base_url == BASE_URL
    assert ref.batch_size is None
    assert ref.labels == ("one",)
    assert ref.result_format == "memory"
    assert job.result().counts == {"1": 10}


def test_backend_submit_auto_format_not_forwarded(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """With result_format='auto' no result_format key is sent to the SDK."""
    backend.submit(_measured(), shots=3, transpile=False)
    assert "result_format" not in sdk.calls[0][1]


def test_backend_transpiled_layout_forwarded(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """With transpilation the routed layout is sent as initial_layout."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure_all()
    backend.submit(circ, shots=4, initial_layout=[2, 3])
    assert sdk.calls[0][1]["initial_layout"] == [2, 3]


def test_backend_physical_qubits_restrict_layout(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """physical_qubits confines the layout to the requested pool."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure_all()
    backend.submit(circ, shots=4, physical_qubits=[3, 4, 5], initial_layout=[5, 4])
    layout = sdk.calls[0][1]["initial_layout"]
    assert layout[:2] == [5, 4]
    assert set(layout) <= {3, 4, 5}


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"physical_qubits": [0, 1], "transpile": False}, ValueError, "requires"),
        ({"physical_qubits": [0, 1, 3, 4]}, TranslationError, "connected"),
        ({"physical_qubits": [0, 2]}, TranslationError, "coupling edge"),
        ({"physical_qubits": [0]}, TranslationError, "coupling edge"),
        (
            {"physical_qubits": [0, 1], "initial_layout": [0, 3]},
            TranslationError,
            "outside",
        ),
        ({"physical_qubits": "01"}, TypeError, "physical_qubits"),
        ({"initial_layout": [0.5, 1]}, TypeError, "initial_layout"),
        ({"initial_layout": [True, 1]}, TypeError, "initial_layout"),
        ({"bogus": 1}, ValueError, "Unsupported"),
        ({"provider_batching": "maybe"}, ValueError, "provider_batching"),
        ({"shots": 0}, ValueError, "shots"),
        ({"shots": 50_001}, ValueError, "shots"),
        ({"shots": 2.0}, TypeError, "shots"),
    ],
)
def test_backend_submit_option_errors(
    backend: LogicalQubitBackend,
    sdk: FakeSDKBackend,
    kwargs: dict[str, Any],
    exc: type[Exception],
    match: str,
) -> None:
    """Invalid submission options are rejected before any SDK call."""
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure_all()
    kwargs = dict(kwargs)
    shots = kwargs.pop("shots", 4)
    with pytest.raises(exc, match=match):
        backend.submit(circ, shots=shots, **kwargs)
    assert sdk.calls == []


def test_backend_transpile_option_must_be_bool(backend: LogicalQubitBackend) -> None:
    """A non-bool 'transpile' option is rejected."""
    request = ExecutionRequest.single(_measured(), options={"transpile": "yes"})
    with pytest.raises(TypeError, match="transpile"):
        backend.submit_request(request)


def test_backend_submit_type_and_parameter_errors(backend: LogicalQubitBackend) -> None:
    """submit() rejects non-circuits, unbound params and unmeasured circuits."""
    with pytest.raises(TypeError, match="QuantumCircuit"):
        backend.submit("circ")  # type: ignore[arg-type]
    unbound = quspect.QuantumCircuit().rx(Parameter("t"), 0).measure_all()
    with pytest.raises(TranslationError, match="unbound"):
        backend.submit(unbound)
    with pytest.raises(TranslationError, match="no measurement"):
        backend.submit(quspect.QuantumCircuit().h(0), transpile=False)
    with pytest.raises(TranslationError, match="noise"):
        backend.submit(
            quspect.QuantumCircuit().h(0).depolarize(0.1, 0).measure_all(),
            transpile=False,
        )


def test_backend_too_many_qubits(backend: LogicalQubitBackend) -> None:
    """A circuit wider than the device fails during transpilation."""
    circ = quspect.QuantumCircuit().h(7).measure_all()
    with pytest.raises(TranslationError, match="physical qubits"):
        backend.submit(circ)


def test_backend_translation_unexpected_error_wrapped(
    backend: LogicalQubitBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-TranslationError failures inside the translator are wrapped."""

    def boom(self: Any, doc: str) -> Any:
        raise KeyError("x")

    monkeypatch.setattr(LQCircuitTranslator, "translate", boom)
    with pytest.raises(TranslationError, match="Failed to translate"):
        backend.submit(_measured(), transpile=False)


def test_backend_submit_request_validation(backend: LogicalQubitBackend) -> None:
    """submit_request rejects non-requests."""
    with pytest.raises(TypeError, match="ExecutionRequest"):
        backend.submit_request(_measured())  # type: ignore[arg-type]


def test_backend_native_batches_of_64(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """More than 64 experiments are split into native batches + CompositeJob."""
    request = ExecutionRequest.batch(
        [_measured() for _ in range(65)], shots=5, options={"transpile": False}
    )
    job = backend.submit_request(request)
    assert isinstance(job, CompositeJob)
    assert [len(p) for p, _ in sdk.calls] == [64, 1]
    assert job.job_ids == ("task_1", "task_2")
    assert job.reference.child_batch_sizes == (64, 1)
    batch = job.result()
    assert len(batch) == 65
    assert batch.get_counts() == [{"1": 5}] * 65


def test_backend_partition_by_layout(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """Experiments with different routed layouts go into separate jobs."""
    groups = LogicalQubitBackend._partition(
        [
            lq_backend_mod._PreparedExperiment(None, "a", (1, 1), (0,)),
            lq_backend_mod._PreparedExperiment(None, "b", (1, 1), (0,)),
            lq_backend_mod._PreparedExperiment(None, "c", (1, 1), (1,)),
        ],
        None,
    )
    assert [([i.label for i in g], lay) for g, lay in groups] == [
        (["a", "b"], (0,)),
        (["c"], (1,)),
    ]
    forced = LogicalQubitBackend._partition(
        [
            lq_backend_mod._PreparedExperiment(None, x, (1, 1), (int(x == "c"),))
            for x in "abc"
        ],
        [7],
    )
    assert len(forced) == 1 and forced[0][1] == (7,)


def test_backend_individual_batching(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """provider_batching='individual' submits one circuit per job."""
    request = ExecutionRequest.batch(
        [_measured("a"), _measured("b")],
        shots=4,
        options={"transpile": False, "provider_batching": "individual"},
    )
    job = backend.submit_request(request)
    assert isinstance(job, CompositeJob)
    assert all(isinstance(p, FakeLQCircuit) for p, _ in sdk.calls)
    assert "provider_batching" not in sdk.calls[0][1]
    assert job.result().labels == ("a", "b")


def test_backend_single_batch_returns_plain_job(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """A request fitting in one native batch returns a LogicalQubitJob."""
    request = ExecutionRequest.batch(
        [_measured("a"), _measured("b")],
        shots=4,
        options={"transpile": False},
        metadata={"workload_id": "wl"},
    )
    job = backend.submit_request(request)
    assert isinstance(job, LogicalQubitJob)
    assert job.reference.workload_id == "wl"
    assert job.reference.batch_size == 2
    assert job.result().get_counts() == [{"1": 4}, {"1": 4}]


def test_backend_unsupported_result_format(
    backend: LogicalQubitBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request format missing from capabilities is rejected."""
    from quspect.backends._execution import BackendCapabilities

    monkeypatch.setattr(
        LogicalQubitBackend,
        "capabilities",
        property(
            lambda self: BackendCapabilities(result_formats=frozenset({"counts"}))
        ),
    )
    request = ExecutionRequest.single(_measured(), result_format="memory")
    with pytest.raises(ValueError, match="result format"):
        backend.submit_request(request)


def test_backend_first_submission_failure_raises_backend_error(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """If no job was created the SDK error is wrapped in BackendError."""
    sdk.fail_on_call = 1
    request = ExecutionRequest.single(_measured(), options={"transpile": False})
    with pytest.raises(BackendError, match="provider rejected"):
        backend.submit_request(request)
    sdk.fail_on_call = len(sdk.calls) + 1
    with pytest.raises(BackendError, match="Failed to submit"):
        backend.submit(_measured(), transpile=False)


def test_backend_unexpected_error_after_sdk_call_wrapped(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """A non-backend failure while building the reference is wrapped."""
    sdk.job_id_override = " "
    request = ExecutionRequest.single(_measured(), options={"transpile": False})
    with pytest.raises(BackendError, match="Failed to submit LogicalQubit batch"):
        backend.submit_request(request)


@pytest.mark.parametrize(
    ("cancel_result", "failures"),
    [(True, ()), (False, ("task_1",)), (RuntimeError("no"), ("task_1",))],
)
def test_backend_partial_submission(
    backend: LogicalQubitBackend,
    sdk: FakeSDKBackend,
    cancel_result: Any,
    failures: tuple[str, ...],
) -> None:
    """A failure after a created job raises PartialSubmissionError and cancels."""
    sdk.fail_on_call = 2
    sdk.cancel_result = cancel_result
    request = ExecutionRequest.batch(
        [_measured() for _ in range(65)], shots=2, options={"transpile": False}
    )
    with pytest.raises(PartialSubmissionError, match="after 64") as info:
        backend.submit_request(request)
    assert info.value.submitted_job_ids == ("task_1",)
    assert info.value.cancellation_failures == failures
    assert sdk.jobs[0].cancelled is True
    assert isinstance(info.value.__cause__, BackendError)


def test_backend_metadata_properties() -> None:
    """num_qubits / native gates / capabilities are parsed from SDK config."""
    sdk = FakeSDKBackend(
        config={
            "qubit_num": "12",
            "native_gate_set": ["CZ", "Rx"],
            "supported_modes": ["Circuit", "script"],
            "supported_measurement_types": ["measure2"],
        }
    )
    be = LogicalQubitBackend(sdk)
    assert be.num_qubits == 12
    assert be.native_gate_set == {"cz", "rx"}
    caps = be.capabilities
    assert caps.supports_sampling and caps.supports_batch and caps.supports_estimator
    assert caps.supports_job_recovery and caps.supports_cancel
    assert caps.max_circuits_per_job == 64 and caps.max_shots == 50_000
    assert caps.result_formats == frozenset({"auto", "counts", "memory"})
    assert caps.measurement_types == frozenset({"measure2"})
    assert be.platform_name == "logicalqubit"
    assert be.name == "QZ02"
    assert repr(be) == "LogicalQubitBackend(name='QZ02', qubits=12)"
    cfg = be.config
    cfg["x"] = 1
    assert "x" not in be.config


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"num_qubits": 5}, 5),
        ({"qubits": [0, 1, 2]}, 3),
        ({"n_qubits": "bad", "topology": {"qubit_coordinates": [[0, 0], [0, 1]]}}, 2),
        ({"topology": {"coupling_map": [[0, 3], [3, 1]]}}, 4),
        ({"topology": "none"}, 0),
        ({}, 0),
    ],
)
def test_backend_num_qubits_sources(config: dict[str, Any], expected: int) -> None:
    """num_qubits falls back through several config keys."""
    assert LogicalQubitBackend(FakeSDKBackend(config=config)).num_qubits == expected


def test_backend_config_errors_and_odd_values() -> None:
    """A failing config yields {} and malformed fields degrade gracefully."""

    class Broken(FakeSDKBackend):
        @property  # type: ignore[override]
        def config(self) -> Any:
            """Raise like an SDK whose metadata request failed."""
            raise RuntimeError("offline")

        @config.setter
        def config(self, value: Any) -> None:
            pass

    be = LogicalQubitBackend(Broken())
    assert be.config == {}
    assert be.num_qubits == 0
    odd = LogicalQubitBackend(
        FakeSDKBackend(
            config={
                "native_gates": "cx",
                "supported_modes": "circuit",
                "supported_measurement_types": "m",
            }
        )
    )
    assert odd.native_gate_set == set()
    assert odd.capabilities.supported_modes == frozenset({"circuit"})
    assert odd.capabilities.measurement_types == frozenset()


def test_backend_no_coupling_map_passes_layout_through() -> None:
    """Without topology, initial_layout is validated and forwarded."""
    sdk = FakeSDKBackend(config={"qubits": 4})
    be = LogicalQubitBackend(sdk)
    circ = quspect.QuantumCircuit().h(0).cx(0, 1).measure_all()
    be.submit(circ, shots=2, initial_layout=[3, 1])
    assert sdk.calls[0][1]["initial_layout"] == [3, 1]
    with pytest.raises(TranslationError, match="length"):
        be.submit(circ, shots=2, initial_layout=[3])
    be.submit(circ, shots=2)
    assert "initial_layout" not in sdk.calls[-1][1]


def test_backend_native_gate_decomposition() -> None:
    """With a CZ-native device CX/CCX are lowered before translation."""
    sdk = FakeSDKBackend(
        config={
            "qubits": 3,
            "native_gates": ["cz", "rx", "ry", "rz", "h", "x", "u", "id"],
            "topology": {"coupling_map": [[0, 1], [1, 2]]},
        }
    )
    be = LogicalQubitBackend(sdk)
    circ = quspect.QuantumCircuit().h(0).h(1).ccx(0, 1, 2).measure_all()
    result = be.run(circ, shots=SHOTS)
    names = {op for op, _ in sdk.calls[0][0].ops}
    assert "ccx" not in names and "cx" not in names
    assert_counts_match(result.counts, qiskit_probabilities(circ), SHOTS)


def test_job_transport_inference() -> None:
    """Transport is inferred from children; mixed transports are rejected."""
    infer = lq_backend_mod._job_transport
    assert infer(SimpleNamespace(_use_async=False), ()) == "sync"
    assert infer(SimpleNamespace(), ()) == "async"
    kids = (SimpleNamespace(_use_async=False), SimpleNamespace(_use_async=None))
    assert infer(SimpleNamespace(_use_async=True), kids) == "sync"
    with pytest.raises(BackendError, match="mixed"):
        infer(
            SimpleNamespace(),
            (SimpleNamespace(_use_async=True), SimpleNamespace(_use_async=False)),
        )


def test_make_reference_with_fallback_children(backend: LogicalQubitBackend) -> None:
    """SDK fallback children are recorded as child job ids."""
    raw = RawJob("wrap")
    raw._children = (  # type: ignore[attr-defined]
        SimpleNamespace(job_id="c1", _use_async=True),
        SimpleNamespace(job_id=None),
        SimpleNamespace(job_id="c2", _use_async=True),
    )
    ref = backend._make_reference(
        raw,
        batch_size=2,
        labels=("a", "b"),
        shapes=((1, 1), (1, 1)),
        shots=3,
        result_format="auto",
        workload_id=None,
    )
    assert ref.child_job_ids == ("c1", "c2")
    assert ref.child_batch_sizes == (1, 1)
    assert ref.transport == "async"


def test_workload_id_validated_before_submission(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """Invalid workload_id must be rejected before any job is submitted."""
    request = ExecutionRequest.single(
        _measured(), options={"transpile": False}, metadata={"workload_id": " "}
    )
    with pytest.raises(ValueError, match="workload_id"):
        backend.submit_request(request)
    assert sdk.calls == []


def test_backend_run_unwraps_batch_result(
    backend: LogicalQubitBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run() returns the single result when the job yields a batch."""
    res = lq_job_mod._convert_child(
        RawResult({"1": 2}), shape=(1, 1), requested_shots=None
    )

    class BatchJob:
        def result(self, timeout: Any = None, poll_interval: float = 2.0) -> Any:
            """Return a one-element batch."""
            return ExecutionBatch(results=(res,))

    monkeypatch.setattr(backend, "submit", lambda *a, **k: BatchJob())
    assert backend.run(_measured()) is res


# =============================================================================
# LogicalQubitJob
# =============================================================================


def _job(raw: RawJob, **kwargs: Any) -> LogicalQubitJob:
    """Wrap a raw job with a simple single-experiment reference."""
    ref = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id=raw.job_id,
        labels=("c",),
        experiment_shapes=((2, 2),),
        shots=10,
        **kwargs,
    )
    return LogicalQubitJob(raw, reference=ref)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("queued", JobStatus.QUEUED),
        ("PENDING", JobStatus.PENDING),
        (" retrying ", JobStatus.RUNNING),
        ("running", JobStatus.RUNNING),
        (SimpleNamespace(value="completed"), JobStatus.COMPLETED),
        ("failed", JobStatus.FAILED),
        ("canceled", JobStatus.CANCELLED),
        ("cancelled", JobStatus.CANCELLED),
        ("weird", JobStatus.UNKNOWN),
    ],
)
def test_job_status_mapping(raw: Any, expected: JobStatus) -> None:
    """Provider status strings map to JobStatus values."""
    assert _job(RawJob("j", status=raw)).status() is expected


def test_job_status_and_cancel_errors_are_mapped() -> None:
    """SDK exceptions from status() / cancel() become JobError."""
    job = _job(RawJob("j", status=RuntimeError("down"), cancel=FakeLQJobError("x")))
    with pytest.raises(JobError, match="down") as info:
        job.status()
    assert info.value.job_id == "j"
    with pytest.raises(JobError, match="x"):
        job.cancel()
    assert _job(RawJob("j", cancel=0)).cancel() is False
    assert _job(RawJob("j", cancel=1)).cancel() is True


def test_job_result_single_success() -> None:
    """A single completed result is converted with reference shape and shots."""
    raw = RawJob("j", RawResult({"01": 6, "10": 4}, job_id="j"))
    res = _job(raw).result(timeout=3.0, poll_interval=0.5)
    assert raw.result_calls == [{"timeout": 3.0, "wait": 0.5, "verbose": False}]
    assert res.counts == {"01": 6, "10": 4}
    assert res.probabilities == {"01": 0.6, "10": 0.4}
    assert (res.num_qubits, res.num_clbits, res.num_shots) == (2, 2, 10)
    assert res.metadata["platform"] == "logicalqubit"
    assert res.metadata["provider_job_id"] == "j"


def test_job_result_single_failure_raises() -> None:
    """A failed single result raises JobError with provider error type."""
    raw = RawJob(
        "j",
        RawResult(
            data={"success": False, "error": "calibration", "error_type": "E42"},
            status="completed",
        ),
    )
    with pytest.raises(JobError, match="calibration") as info:
        _job(raw).result()
    assert info.value.error_type == "E42"
    raw2 = RawJob("j", RawResult(data={}, status="failed"))
    with pytest.raises(JobError, match="j failed"):
        _job(raw2).result()


@pytest.mark.parametrize(
    ("timeout", "poll", "match"),
    [(-1.0, 1.0, "timeout"), (None, 0.0, "poll_interval"), (None, math.nan, "poll")],
)
def test_job_result_argument_validation(
    timeout: float | None, poll: float, match: str
) -> None:
    """Invalid timeout / poll interval values raise ValueError."""
    with pytest.raises(ValueError, match=match):
        _job(RawJob("j")).result(timeout=timeout, poll_interval=poll)


def test_job_result_rejects_nan_timeout() -> None:
    """timeout=nan must be rejected like in CompositeJob.result."""
    raw = RawJob("j", RawResult({"00": 10}))
    with pytest.raises(ValueError, match="timeout"):
        _job(raw).result(timeout=math.nan)


def test_job_result_timeout_zero_semantics() -> None:
    """timeout=0 is non-blocking: raise if not done, else fetch without limit."""
    pending = RawJob("j", RawResult({"00": 10}), status="running")
    with pytest.raises(JobTimeoutError, match="not complete"):
        _job(pending).result(timeout=0)
    assert pending.result_calls == []
    done = RawJob("j", RawResult({"00": 10}))
    _job(done).result(timeout=0)
    assert done.result_calls[0]["timeout"] is None


@pytest.mark.parametrize(
    ("exc", "expected_type", "error_type", "job_id"),
    [
        (FakeLQJobTimeoutError("slow"), JobTimeoutError, None, "j"),
        (FakeLQJobError("bad", error_type="T", job_id="other"), JobError, "T", "other"),
        (FakeLQJobError("bad"), JobError, None, "j"),
        (JobTimeoutError("mine", job_id="k"), JobTimeoutError, None, "k"),
        (RuntimeError("boom"), JobError, None, "j"),
    ],
)
def test_job_result_exception_mapping(
    exc: Exception, expected_type: type[Exception], error_type: str | None, job_id: str
) -> None:
    """SDK exceptions are mapped to quspect JobError / JobTimeoutError."""
    with pytest.raises(JobError) as info:
        _job(RawJob("j", exc)).result()
    assert type(info.value) is expected_type
    assert info.value.error_type == error_type
    assert info.value.job_id == job_id


def test_job_exception_mapping_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without lqcloud.exceptions every error becomes a generic JobError."""
    monkeypatch.setitem(sys.modules, "lqcloud.exceptions", None)
    mapped = lq_job_mod._map_sdk_exception(ValueError("v"), "j")
    assert type(mapped) is JobError
    assert mapped.job_id == "j"


def test_job_batch_result_and_metadata() -> None:
    """Batch jobs return an ExecutionBatch with compacted metadata."""
    raw = RawJob(
        "b",
        RawBatch([RawResult({"0": 3}, job_id="c0"), RawResult({"1": 3}, job_id="c1")]),
    )
    ref = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="b",
        batch_size=2,
        labels=("x", "y"),
        experiment_shapes=((1, 1), (1, 1)),
        shots=3,
    )
    job = LogicalQubitJob(raw, reference=ref)
    batch = job.result()
    assert isinstance(batch, ExecutionBatch)
    assert batch.labels == ("x", "y")
    assert batch.get_counts() == [{"0": 3}, {"1": 3}]
    assert batch.metadata == {"platform": "logicalqubit", "queue": 3}
    assert batch.job_id == "b"
    assert job.job_ids == ("b",)


def test_job_reference_inferred_from_raw_job() -> None:
    """Without an explicit reference one is inferred from the SDK job."""
    raw = RawJob("w", is_batch=True, batch_size=2)
    raw._use_async = False
    raw._children = (  # type: ignore[attr-defined]
        SimpleNamespace(job_id="c1"),
        SimpleNamespace(job_id="c2"),
    )
    job = LogicalQubitJob(raw, requested_shots=7, result_format="counts")
    ref = job.reference
    assert ref.transport == "sync"
    assert ref.batch_size == 2
    assert ref.child_job_ids == ("c1", "c2")
    assert ref.shots == 7
    assert ref.base_url == BASE_URL
    assert job.job_ids == ("c1", "c2")
    labelled = LogicalQubitJob(
        RawJob("v"),
        labels=("a", "b"),
        experiment_shapes=((1, 1), (1, 1)),
        return_batch=True,
    )
    assert labelled.reference.batch_size == 2
    assert labelled.reference.labels == ("a", "b")


def test_job_return_batch_without_labels() -> None:
    """return_batch=True without labels should not crash the constructor."""
    raw = RawJob("w", RawBatch([RawResult({"0": 1})]), is_batch=True, batch_size=1)
    job = LogicalQubitJob(raw, return_batch=True)
    assert isinstance(job.result(), ExecutionBatch)


# ---- _convert_result / _convert_child --------------------------------------


@pytest.mark.parametrize("raw", [None, "error string"])
def test_convert_result_rejects_missing_payload(raw: Any) -> None:
    """None / string payloads raise JobError."""
    with pytest.raises(JobError, match="no circuit result"):
        lq_job_mod._convert_result(raw, job_id="j")


def test_convert_result_shape_and_size_errors() -> None:
    """Empty batches and mismatched labels / shapes raise JobError."""
    with pytest.raises(JobError, match="empty batch"):
        lq_job_mod._convert_result(RawBatch([]))
    batch = RawBatch([RawResult({"0": 1}), RawResult({"1": 1})])
    with pytest.raises(JobError, match="does not match"):
        lq_job_mod._convert_result(batch, labels=("a",))
    with pytest.raises(JobError, match="shapes"):
        lq_job_mod._convert_result(batch, experiment_shapes=((1, 1),))
    out = lq_job_mod._convert_result(batch)
    assert out.labels == ("experiment-0", "experiment-1")
    assert out.backend_name == "QZ02"
    assert out.job_id == "b"


def _child(**kwargs: Any) -> Any:
    """Convert one fake child result with no shape hint."""
    return lq_job_mod._convert_child(
        RawResult(**kwargs), shape=None, requested_shots=None
    )


def test_convert_child_bitstrings_pass_through_unchanged() -> None:
    """Provider bitstrings are kept verbatim (quspect convention assumed)."""
    res = _child(counts={"100": 5, "001": 3})
    assert res.counts == {"001": 3, "100": 5}
    assert (res.num_qubits, res.num_clbits) == (3, 3)
    assert res.num_shots == 8


def test_convert_child_memory_and_corrected_counts() -> None:
    """Memory results aggregate shots; applied readout correction wins."""
    res = _child(data={"memory": ["01", "01", "10"], "result_format": "memory"})
    assert res.counts == {"01": 2, "10": 1}
    assert res.memory == ["01", "01", "10"]
    assert res.metadata["result_format"] == "memory"
    corrected = _child(
        data={
            "memory": ["0", "1"],
            "corrected_counts": {"0": 1.5, "1": 0.5},
            "readout_correction_applied": True,
            "readout_correction_requested": True,
        }
    )
    assert corrected.counts == {"0": 2, "1": 0}
    assert corrected.probabilities == {"0": 0.75, "1": 0.25}
    assert corrected.metadata["quasi_counts"] == {"0": 1.5, "1": 0.5}
    assert corrected.metadata["readout_correction_applied"] is True
    weird_memory = _child(
        data={"result_format": "memory", "memory": "0", "counts": {"1": 2}}
    )
    assert weird_memory.counts == {"1": 2}
    assert weird_memory.memory is None


def test_convert_child_fallback_and_invalid_counts() -> None:
    """get_counts() fallback, decode errors and invalid counts mark failure."""
    assert _child(data={}, fallback={"1": 4}).counts == {"1": 4}
    failed = _child(data={}, fallback=RuntimeError("bad payload"))
    assert failed.success is False
    assert "Unable to decode" in (failed.error or "")
    not_map = _child(data={}, fallback=[1, 2])
    assert not_map.success is False
    assert "counts mapping" in (not_map.error or "")
    invalid = _child(data={"counts": {"0": "many"}})
    assert invalid.success is False and "Invalid count" in (invalid.error or "")
    nonfinite = _child(data={"counts": {"0": math.inf}})
    assert nonfinite.success is False and "Non-finite" in (nonfinite.error or "")
    assert nonfinite.counts == {} and nonfinite.probabilities == {}


def test_convert_child_shots_resolution() -> None:
    """Shots come from data, then requested shots, then the count total."""
    conv = lq_job_mod._convert_child
    r = RawResult(data={"counts": {"0": 2}, "shots": True})
    assert conv(r, shape=(1, 1), requested_shots=None).num_shots == 2
    r = RawResult(data={"counts": {"0": 2}, "shots": "x"})
    assert conv(r, shape=(1, 1), requested_shots=None).num_shots == 2
    r = RawResult(data={"counts": {"0": 2}})
    assert conv(r, shape=(1, 1), requested_shots=9).num_shots == 9
    r = RawResult(data={"counts": {"0": 2}, "shots": -4})
    assert conv(r, shape=(1, 1), requested_shots=None).num_shots == 0
    r = RawResult(data={"counts": {"0": 0}})
    assert conv(r, shape=(1, 1), requested_shots=None).probabilities == {}


def test_convert_child_non_mapping_data_and_metadata() -> None:
    """Non-mapping data / metadata are tolerated; bulky keys are dropped."""
    raw = RawResult(
        data={"counts": {"1": 1}},
        metadata={"stdout": "x", "result": {}, "calib": 1},
    )
    res = lq_job_mod._convert_child(raw, shape=(1, 1), requested_shots=None)
    assert res.metadata["calib"] == 1
    assert "stdout" not in res.metadata and "result" not in res.metadata
    raw.data = "garbage"  # type: ignore[assignment]
    raw._fallback = {"0": 1}
    assert lq_job_mod._convert_child(raw, shape=None, requested_shots=None).counts == {
        "0": 1
    }
    assert lq_job_mod._compact_metadata("x") == {}


# =============================================================================
# LogicalQubitProvider
# =============================================================================


def test_provider_requires_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without lqcloud, constructing a provider raises OptionalDependencyError."""
    monkeypatch.setattr(lq_provider_mod, "_HAS_LQCLOUD", False)
    with pytest.raises(OptionalDependencyError, match="lqcloud"):
        LogicalQubitProvider(api_key="k")


def test_provider_options_from_args_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit args win over env vars; omitted values are not sent."""
    LogicalQubitProvider()
    assert FakeLQCloudProvider.instances[-1].kwargs == {"interactive": False}
    monkeypatch.setenv("LQCLOUD_API_KEY", "env-key")
    monkeypatch.setenv("LQCLOUD_URL", "https://env.invalid")
    LogicalQubitProvider()
    assert FakeLQCloudProvider.instances[-1].kwargs == {
        "interactive": False,
        "api_key": "env-key",
        "url": "https://env.invalid",
    }
    provider = LogicalQubitProvider(api_key="k", url=BASE_URL)
    assert FakeLQCloudProvider.instances[-1].kwargs["api_key"] == "k"
    assert provider.url == BASE_URL
    assert repr(provider) == f"LogicalQubitProvider(url={BASE_URL!r})"


def test_provider_get_backend_and_list() -> None:
    """get_backend wraps the SDK backend; list_backends returns a list."""
    provider = LogicalQubitProvider(url=BASE_URL)
    be = provider.get_backend("QZ02", use_async=False, verify=False)
    assert isinstance(be, LogicalQubitBackend)
    assert be.name == "QZ02"
    assert FakeLQCloudProvider.instances[-1].backend_calls == [
        ("QZ02", {"use_async": False, "verify": False})
    ]
    assert provider.list_backends() == [
        {"name": "QZ02", "status": "online"},
        {"name": "QZ01", "status": "offline"},
    ]


def test_provider_retrieve_by_job_id() -> None:
    """A raw job ID plus context recreates a job with the provider URL."""
    provider = LogicalQubitProvider(url=BASE_URL)
    job = provider.retrieve_job(
        "task_9",
        backend_name="QZ02",
        use_async=False,
        batch_size=2,
        labels=("a", "b"),
        experiment_shapes=((1, 1), (1, 1)),
        shots=5,
        result_format="counts",
    )
    assert isinstance(job, LogicalQubitJob)
    raw = job._lq_job
    assert isinstance(raw, FakeSDKJob)
    assert (raw.job_id, raw.backend_name, raw.base_url) == ("task_9", "QZ02", BASE_URL)
    assert raw._use_async is False
    assert raw._is_batch is True and raw._batch_size == 2  # type: ignore[attr-defined]
    assert job.reference.transport == "sync"
    assert job.reference.labels == ("a", "b")


def test_provider_retrieve_reference_roundtrip(
    backend: LogicalQubitBackend, sdk: FakeSDKBackend
) -> None:
    """A serialized composite reference is recovered into ordered children."""
    request = ExecutionRequest.batch(
        [_measured() for _ in range(65)], shots=2, options={"transpile": False}
    )
    original = backend.submit_request(request)
    data = original.reference.to_dict()
    provider = LogicalQubitProvider(url=BASE_URL + ":443/")
    recovered = provider.retrieve_job(JobReference.from_dict(data))
    assert isinstance(recovered, CompositeJob)
    assert recovered.job_ids == ("task_1", "task_2")
    assert recovered.reference.labels == original.reference.labels
    assert recovered.reference.child_batch_sizes == (64, 1)
    single = provider.retrieve_job(
        JobReference(platform="logicalqubit", backend_name="QZ02", job_id="t")
    )
    assert isinstance(single, LogicalQubitJob)
    assert single._lq_job.base_url == BASE_URL + ":443/"


@pytest.mark.parametrize(
    ("reference", "kwargs", "exc", "match"),
    [
        (
            JobReference(platform="ibm", backend_name="b", job_id="j"),
            {},
            ValueError,
            "platform",
        ),
        ("task", {}, ValueError, "backend_name"),
        ("  ", {}, TypeError, "reference"),
        (42, {}, TypeError, "reference"),
        (
            JobReference(
                platform="logicalqubit",
                backend_name="b",
                job_id="j",
                transport="in-process",
            ),
            {},
            ValueError,
            "transport",
        ),
        (
            JobReference(
                platform="logicalqubit",
                backend_name="b",
                job_id="j",
                base_url="https://evil.invalid",
            ),
            {},
            ValueError,
            "origin",
        ),
        (
            JobReference(
                platform="logicalqubit",
                backend_name="b",
                job_id="j",
                child_job_ids=("a", "b"),
                child_batch_sizes=(1, 2),
            ),
            {},
            ValueError,
            "do not match",
        ),
    ],
)
def test_provider_retrieve_errors(
    reference: Any, kwargs: dict[str, Any], exc: type[Exception], match: str
) -> None:
    """Invalid recovery requests raise before any SDK job is created."""
    provider = LogicalQubitProvider(url=BASE_URL)
    with pytest.raises(exc, match=match):
        provider.retrieve_job(reference, **kwargs)


@pytest.mark.parametrize(
    ("candidate", "configured", "same"),
    [
        ("https://a.invalid", "https://A.invalid:443/x", True),
        ("http://a.invalid", "http://a.invalid:80", True),
        ("http://a.invalid", "https://a.invalid", False),
        ("https://a.invalid:8443", "https://a.invalid", False),
        ("ftp://a.invalid", "ftp://a.invalid", False),
        ("https://", "https://", False),
        ("https://a.invalid:99999", "https://a.invalid", False),
    ],
)
def test_same_origin(candidate: str, configured: str, same: bool) -> None:
    """Origins compare scheme, lowercase host and effective port."""
    assert lq_provider_mod._same_origin(candidate, configured) is same
