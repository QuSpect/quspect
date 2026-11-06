"""Live Origin Quantum checks (opt-in, network).

* Cloud simulator (``full_amplitude``): set ``QPANDA_QCLOUD_API_KEY`` and
  ``QUSPECT_RUN_CLOUD_TESTS=1``.
* Real QPU: additionally ``QUSPECT_RUN_HARDWARE_TESTS=1`` (and optionally
  ``QUSPECT_ORIGINQ_QPU``, default ``WK_C180_2``).  QPU jobs may queue for
  tens of minutes.

The circuits are asymmetric on purpose: a Bell state cannot reveal a
reversed or permuted bit order.
"""

from __future__ import annotations

import os

import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest
from tests.backends._originq_fakes import quspect_exact_distribution

_HAS_KEY = bool(os.environ.get("QPANDA_QCLOUD_API_KEY"))
needs_cloud = pytest.mark.skipif(
    not (_HAS_KEY and os.environ.get("QUSPECT_RUN_CLOUD_TESTS") == "1"),
    reason="set QPANDA_QCLOUD_API_KEY and QUSPECT_RUN_CLOUD_TESTS=1",
)
needs_hardware = pytest.mark.skipif(
    not (_HAS_KEY and os.environ.get("QUSPECT_RUN_HARDWARE_TESTS") == "1"),
    reason="set QPANDA_QCLOUD_API_KEY and QUSPECT_RUN_HARDWARE_TESTS=1",
)


def _bit_order_circuits() -> dict[str, QuantumCircuit]:
    circuits: dict[str, QuantumCircuit] = {}
    c = QuantumCircuit(name="ry_x_identity")
    c.ry(1.0, 0).x(1).measure([0, 1], [0, 1])
    circuits[c.name] = c
    c = QuantumCircuit(name="ry_x_swapped_clbits")
    c.ry(1.0, 0).x(1).measure([0, 1], [1, 0])
    circuits[c.name] = c
    c = QuantumCircuit(name="reverse_measure_order")
    c.x(0).measure([1], [0]).measure([0], [1])
    circuits[c.name] = c
    c = QuantumCircuit(name="sparse_fanout")
    c.ry(0.6, 0).x(2).measure([0, 2, 0], [3, 0, 1])
    circuits[c.name] = c
    c = QuantumCircuit(name="ghz_partial")
    c.h(0).cx(0, 1).cx(1, 2).ry(0.8, 2).measure([2, 0], [0, 1])
    circuits[c.name] = c
    return circuits


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


@pytest.fixture(scope="module")
def provider():
    from quspect.backends.originq import OriginQProvider

    return OriginQProvider()


@needs_cloud
def test_cloud_simulator_bit_order(provider) -> None:
    backend = provider.get_backend("full_amplitude")
    circuits = _bit_order_circuits()
    request = ExecutionRequest.batch(
        list(circuits.values()), labels=list(circuits), shots=4000
    )
    batch = backend.submit_request(request).result(timeout=600, poll_interval=2)
    for name, circuit in circuits.items():
        expected = quspect_exact_distribution(circuit)
        actual = batch.by_label(name).probabilities
        assert _tv_distance(actual, expected) < 0.05, (name, actual, expected)


@pytest.mark.hardware
@needs_hardware
def test_qpu_bit_order_quspect_and_provider_mapping(provider) -> None:
    """One QPU batch per compile mode; hardware noise allows a loose bound."""
    backend = provider.get_backend(os.environ.get("QUSPECT_ORIGINQ_QPU", "WK_C180_2"))
    circuits = _bit_order_circuits()
    for transpile in (True, False):
        request = ExecutionRequest.batch(
            list(circuits.values()),
            labels=list(circuits),
            shots=2000,
            options={"transpile": transpile},
        )
        batch = backend.submit_request(request).result(timeout=7200, poll_interval=15)
        for name, circuit in circuits.items():
            expected = quspect_exact_distribution(circuit)
            actual = batch.by_label(name).probabilities
            assert _tv_distance(actual, expected) < 0.2, (transpile, name, actual)
