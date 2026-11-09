"""Live Tianyan checks (opt-in, network).

* Cloud simulator (``tianyan_sw``): set ``TIANYAN_LOGIN_KEY`` and
  ``QUSPECT_RUN_CLOUD_TESTS=1``.
* Real QPU: additionally ``QUSPECT_RUN_HARDWARE_TESTS=1`` (and optionally
  ``QUSPECT_TIANYAN_QPU``, default ``tianyan176``).  QPU tasks may queue.

The circuits are asymmetric on purpose: a Bell state cannot reveal a
reversed or permuted bit order.
"""

from __future__ import annotations

import os

import pytest

from quspect import QuantumCircuit
from quspect.backends import ExecutionRequest
from tests.backends._originq_fakes import quspect_exact_distribution

_HAS_KEY = bool(os.environ.get("TIANYAN_LOGIN_KEY"))
needs_cloud = pytest.mark.skipif(
    not (_HAS_KEY and os.environ.get("QUSPECT_RUN_CLOUD_TESTS") == "1"),
    reason="set TIANYAN_LOGIN_KEY and QUSPECT_RUN_CLOUD_TESTS=1",
)
needs_hardware = pytest.mark.skipif(
    not (_HAS_KEY and os.environ.get("QUSPECT_RUN_HARDWARE_TESTS") == "1"),
    reason="set TIANYAN_LOGIN_KEY and QUSPECT_RUN_HARDWARE_TESTS=1",
)


def bit_order_circuits() -> dict[str, QuantumCircuit]:
    circuits: dict[str, QuantumCircuit] = {}

    def add(name, build):
        c = QuantumCircuit(name=name)
        build(c)
        circuits[name] = c

    add("ry_x_identity", lambda c: c.ry(1.0, 0).x(1).measure([0, 1], [0, 1]))
    add("ry_x_swapped_clbits", lambda c: c.ry(1.0, 0).x(1).measure([0, 1], [1, 0]))
    add("reverse_measure_order", lambda c: c.x(0).measure([1], [0]).measure([0], [1]))
    add("sparse_fanout", lambda c: c.ry(0.6, 0).x(2).measure([0, 2, 0], [3, 0, 1]))
    add(
        "ghz_partial",
        lambda c: c.h(0).cx(0, 1).cx(1, 2).ry(0.8, 2).measure([2, 0], [0, 1]),
    )
    return circuits


def _tv(a: dict[str, float], b: dict[str, float]) -> float:
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


@pytest.fixture(scope="module")
def provider():
    from quspect.backends.tianyan import TianyanProvider

    return TianyanProvider()


def _check(backend, tolerance: float, shots: int, timeout: float) -> None:
    circuits = bit_order_circuits()
    request = ExecutionRequest.batch(
        list(circuits.values()), labels=list(circuits), shots=shots
    )
    batch = backend.submit_request(request).result(timeout=timeout, poll_interval=10)
    for name, circuit in circuits.items():
        actual = batch.by_label(name).probabilities
        expected = quspect_exact_distribution(circuit)
        assert _tv(actual, expected) < tolerance, (name, actual, expected)


@needs_cloud
def test_cloud_simulator_bit_order(provider) -> None:
    _check(provider.get_backend("tianyan_sw"), 0.05, 4000, 900)


@pytest.mark.hardware
@needs_hardware
def test_qpu_bit_order(provider) -> None:
    backend = provider.get_backend(os.environ.get("QUSPECT_TIANYAN_QPU", "tianyan176"))
    _check(backend, 0.2, 2000, 7200)
