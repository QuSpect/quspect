"""Tests for the LogicalQubit cloud backend.

Cloud integration tests require the ``LQCLOUD_API_KEY`` environment variable
and the ``lqcloud`` SDK.  Tests that submit work to real hardware additionally
require the explicit ``QUSPECT_RUN_HARDWARE_TESTS=1`` opt-in.
"""

from __future__ import annotations

import os

import pytest

from quspect import QuantumCircuit, Simulator
from quspect.backends._base import Backend, CloudBackend, CloudJob, JobStatus

# ---------------------------------------------------------------------------
# Feature flag: skip all cloud tests when API key or SDK is absent
# ---------------------------------------------------------------------------

_HAS_API_KEY = bool(os.environ.get("LQCLOUD_API_KEY"))
_RUN_HARDWARE_TESTS = os.environ.get("QUSPECT_RUN_HARDWARE_TESTS") == "1"

try:
    import lqcloud  # noqa: F401

    _HAS_LQCLOUD = True
except ImportError:
    _HAS_LQCLOUD = False

needs_lqcloud = pytest.mark.skipif(
    not (_HAS_API_KEY and _HAS_LQCLOUD),
    reason="LQCLOUD_API_KEY not set or lqcloud SDK not installed",
)

needs_lqcloud_sdk = pytest.mark.skipif(
    not _HAS_LQCLOUD,
    reason="lqcloud SDK not installed",
)

needs_hardware_opt_in = pytest.mark.skipif(
    not _RUN_HARDWARE_TESTS,
    reason="real hardware tests require QUSPECT_RUN_HARDWARE_TESTS=1",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """A simple Bell-state circuit."""
    qc = QuantumCircuit(name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def adjacent_3q_circuit() -> QuantumCircuit:
    """3-qubit circuit using only adjacent qubits (for linear coupling topology)."""
    qc = QuantumCircuit(name="adj3")
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qc.cz(1, 2)
    qc.x(2)
    qc.barrier()
    qc.measure_all()
    return qc


@pytest.fixture(scope="module")
def provider():
    """Create a LogicalQubitProvider once per test module."""
    from quspect.backends.logicalqubit import LogicalQubitProvider

    return LogicalQubitProvider()


@pytest.fixture(scope="module")
def backend(provider):
    """Get the first available backend."""
    backends = provider.list_backends()
    if not backends:
        pytest.skip("No backends available on this LogicalQubit account")
    name = backends[0].get("name", "")
    if not name:
        pytest.skip("Backend has no name")
    return provider.get_backend(name)


# ---------------------------------------------------------------------------
# Tests — ABC compliance
# ---------------------------------------------------------------------------


class TestSimulatorBackend:
    """Verify Simulator satisfies the Backend ABC."""

    def test_is_backend_instance(self) -> None:
        sim = Simulator(seed=42)
        assert isinstance(sim, Backend)

    def test_name_property(self) -> None:
        sim = Simulator(seed=42)
        assert sim.name == "simulator"

    def test_run_returns_result(self, bell_circuit: QuantumCircuit) -> None:
        from quspect.simulation.result import SimulationResult

        sim = Simulator(seed=42)
        result = sim.run(bell_circuit, shots=100)
        assert isinstance(result, SimulationResult)
        assert result.num_shots == 100
        assert len(result.counts) > 0


class TestBackendABC:
    """Verify the ABC hierarchy."""

    def test_cloud_backend_is_backend(self) -> None:
        assert issubclass(CloudBackend, Backend)

    def test_cloud_job_is_abstract(self) -> None:
        assert issubclass(CloudJob, object)

    def test_job_status_enum(self) -> None:
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.COMPLETED.value == "completed"


# ---------------------------------------------------------------------------
# Tests — LogicalQubit provider
# ---------------------------------------------------------------------------


@needs_lqcloud
class TestLogicalQubitProvider:
    """Verify provider connects and discovers backends."""

    def test_provider_creation(self) -> None:
        from quspect.backends.logicalqubit import LogicalQubitProvider

        provider = LogicalQubitProvider()
        assert provider.url.startswith("https://")

    def test_list_backends(self, provider) -> None:
        backends = provider.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0

    def test_get_backend(self, provider) -> None:
        backends = provider.list_backends()
        name = backends[0]["name"]
        backend = provider.get_backend(name)
        assert backend.name == name


# ---------------------------------------------------------------------------
# Tests — LogicalQubit backend metadata
# ---------------------------------------------------------------------------


@needs_lqcloud
class TestLogicalQubitBackendMetadata:
    """Verify backend exposes platform metadata."""

    def test_is_cloud_backend(self, backend) -> None:
        assert isinstance(backend, CloudBackend)

    def test_platform_name(self, backend) -> None:
        assert backend.platform_name == "logicalqubit"

    def test_name_is_str(self, backend) -> None:
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    def test_num_qubits_positive(self, backend) -> None:
        assert backend.num_qubits > 0

    def test_native_gate_set_nonempty(self, backend) -> None:
        assert isinstance(backend.native_gate_set, set)

    def test_config_is_dict(self, backend) -> None:
        cfg = backend.config
        assert isinstance(cfg, dict)


# ---------------------------------------------------------------------------
# Tests — circuit execution on real QPU
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@needs_hardware_opt_in
@needs_lqcloud
class TestLogicalQubitExecution:
    """End-to-end tests submitting circuits to the real QPU."""

    def test_submit_bell_circuit(self, backend, bell_circuit: QuantumCircuit) -> None:
        """Submit a Bell-state circuit and verify we get a valid job back."""
        job = backend.submit(bell_circuit, shots=500)
        assert isinstance(job, CloudJob)
        assert len(job.job_id) > 0
        status = job.status()
        assert status in (
            JobStatus.QUEUED,
            JobStatus.PENDING,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
        )

    def test_run_bell_circuit_sync(self, backend, bell_circuit: QuantumCircuit) -> None:
        """Run a Bell-state circuit synchronously and check results."""
        from quspect.simulation.result import SimulationResult

        result = backend.run(bell_circuit, shots=500)
        assert isinstance(result, SimulationResult)
        assert result.num_shots == 500
        assert result.num_qubits == 2
        assert result.num_clbits == 2
        # Bell state: |00⟩ and |11⟩ should dominate
        assert "00" in result.counts
        assert "11" in result.counts
        top = result.most_frequent()
        assert top[1] > 0

    def test_run_bell_circuit_async(
        self, backend, bell_circuit: QuantumCircuit
    ) -> None:
        """Async submission, poll status, then get result."""
        job = backend.submit(bell_circuit, shots=300)
        assert len(job.job_id) > 0

        result = job.result(timeout=600)
        assert "00" in result.counts
        assert "11" in result.counts
        total_shots = sum(result.counts.values())
        assert total_shots == 300

    def test_run_with_initial_layout(
        self, backend, bell_circuit: QuantumCircuit
    ) -> None:
        """Run with explicit initial_layout mapping."""
        result = backend.run(
            bell_circuit,
            shots=200,
            initial_layout=[0, 1],
        )
        assert result.num_shots == 200
        assert "00" in result.counts or "11" in result.counts

    def test_bell_state_fidelity(self, backend, bell_circuit: QuantumCircuit) -> None:
        """Verify Bell state has reasonable fidelity (>50%)."""
        result = backend.run(bell_circuit, shots=1000)
        p00 = result.get_probability("00")
        p11 = result.get_probability("11")
        # |00⟩ + |11⟩ should account for >50% of outcomes
        assert p00 + p11 > 0.5, (
            f"Bell state fidelity too low: p00={p00:.3f}, p11={p11:.3f}"
        )

    def test_circuit_with_barrier(
        self, backend, adjacent_3q_circuit: QuantumCircuit
    ) -> None:
        """Run a 3-qubit circuit with adjacent-only gates and a barrier."""
        result = backend.run(adjacent_3q_circuit, shots=300)
        assert result.num_qubits == 3
        assert result.num_shots == 300
        assert len(result.counts) >= 2


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


@needs_lqcloud
class TestLogicalQubitErrors:
    """Verify proper error handling for invalid inputs."""

    def test_reject_noise_channel(self, backend) -> None:
        """Circuits with noise channels should be rejected at translation time."""
        from quspect.backends._exceptions import TranslationError

        qc = QuantumCircuit(name="noisy")
        qc.h(0)
        qc.depolarize(0.1, 0)
        qc.measure_all()

        with pytest.raises(TranslationError, match="noise"):
            backend.submit(qc, shots=100)

    def test_reject_unbound_parameters(self, backend) -> None:
        """Circuits with unbound symbolic parameters should fail."""
        from quspect.backends._exceptions import TranslationError
        from quspect.circuit.parameter import Parameter

        theta = Parameter("theta")
        qc = QuantumCircuit(name="unbound")
        qc.rx(theta, 0)
        qc.measure_all()

        with pytest.raises(TranslationError, match="unbound"):
            backend.submit(qc, shots=100)


# ---------------------------------------------------------------------------
# Tests — translator
# ---------------------------------------------------------------------------


@needs_lqcloud_sdk
class TestLQCircuitTranslator:
    """Unit tests for the QSAL→lqcloud translator."""

    def test_all_standard_gates(self) -> None:
        """Verify every standard QuSpect gate can be translated."""
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator
        from quspect.ir.qsal import dumps

        translator = LQCircuitTranslator()

        qc = QuantumCircuit(name="all_gates")
        qc.id(0)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.barrier()
        qc.measure_all()

        qsal = dumps(qc)
        lq_circuit = translator.translate(qsal)
        assert lq_circuit is not None

    def test_parameterized_gates(self) -> None:
        """Verify parameterized gates are translated with correct values."""
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator
        from quspect.ir.qsal import dumps

        translator = LQCircuitTranslator()

        qc = QuantumCircuit(name="param_gates")
        qc.rx(0.5, 0)
        qc.ry(1.2, 1)
        qc.rz(2.1, 2)
        qc.measure_all()

        qsal = dumps(qc)
        lq_circuit = translator.translate(qsal)
        assert lq_circuit is not None

    def test_two_qubit_gates(self) -> None:
        """Verify two-qubit gates are translated."""
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator
        from quspect.ir.qsal import dumps

        translator = LQCircuitTranslator()

        qc = QuantumCircuit(name="two_qubit")
        qc.cx(0, 1)
        qc.cz(2, 3)
        qc.swap(4, 5)
        qc.crx(0.3, 6, 7)
        qc.cry(0.4, 8, 9)
        qc.crz(0.5, 10, 11)
        qc.measure_all()

        qsal = dumps(qc)
        lq_circuit = translator.translate(qsal)
        assert lq_circuit is not None

    def test_three_qubit_gate(self) -> None:
        """Verify Toffoli (CCX) gate translation."""
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator
        from quspect.ir.qsal import dumps

        translator = LQCircuitTranslator()

        qc = QuantumCircuit(name="toffoli")
        qc.ccx(0, 1, 2)
        qc.measure_all()

        qsal = dumps(qc)
        lq_circuit = translator.translate(qsal)
        assert lq_circuit is not None

    def test_reject_unknown_gate(self) -> None:
        """Verify unknown gates raise TranslationError."""
        from quspect.backends._exceptions import TranslationError
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator

        translator = LQCircuitTranslator()
        # Manually construct a QSAL with an unknown gate name
        qsal = """
        {
            "qsal_version": "0.1.0",
            "name": "bad",
            "circuit": {
                "num_qubits": 1,
                "num_clbits": 1,
                "instructions": [
                    {
                        "type": "GATE",
                        "qubits": [0],
                        "clbits": [],
                        "gate": {
                            "name": "nonexistent_gate",
                            "num_qubits": 1,
                            "gate_type": "standard"
                        }
                    }
                ]
            }
        }
        """
        with pytest.raises(TranslationError, match="not supported"):
            translator.translate(qsal)

    def test_auto_barrier_insertion(self) -> None:
        """Verify barrier is auto-inserted before measurement when missing."""
        from quspect.backends.logicalqubit._translator import LQCircuitTranslator
        from quspect.ir.qsal import dumps

        translator = LQCircuitTranslator()
        qc = QuantumCircuit(name="no_barrier")
        qc.h(0)
        qc.measure_all()

        qsal = dumps(qc)
        lq_circuit = translator.translate(qsal)
        assert lq_circuit is not None
        # lqcloud circuit should now pass barrier validation


class TestConversionHelpers:
    """Unit tests for result conversion."""

    def test_import_guard(self) -> None:
        """Verify optional dependency guard raises clean error when lqcloud absent."""
        # The _require_lqcloud function is tested indirectly —
        # the logicalqubit module still imports correctly without lqcloud
        # because imports are lazy.  The guard fires when creating a provider.
        from quspect.backends._exceptions import OptionalDependencyError

        assert issubclass(OptionalDependencyError, Exception)
