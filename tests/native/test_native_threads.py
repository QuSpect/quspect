"""Tests for the OpenMP thread controls of the native backend.

Results of the parallel C++ kernels must not depend on the thread count:
gate kernels write disjoint amplitudes and reductions use a fixed blocking,
so state vectors, expectations and seeded sampling counts are compared for
1 vs 4 threads on a 16-qubit circuit (above the parallel threshold).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

from quspect._native_loader import BUILD_HINT, load_core
from quspect.native import ExecutionResult

_core = load_core()

pytestmark = pytest.mark.skipif(
    _core is None,
    reason=f"QuSpect native extension not found. {BUILD_HINT}",
)

_NUM_QUBITS = 16


@pytest.fixture(autouse=True)
def _restore_threads() -> Iterator[None]:
    from quspect.native import get_num_threads, set_num_threads

    previous = get_num_threads()
    yield
    set_num_threads(previous)


def _make_circuit_qsal(*, measure_mid_circuit: bool) -> str:
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps
    from quspect.operators.qubit_operator import QubitOperator

    n = _NUM_QUBITS
    circ = QuantumCircuit(name="threads")
    for q in range(n):
        circ.h(q)
        circ.ry(0.1 * (q + 1), q)
    for q in range(n - 1):
        circ.cx(q, q + 1)
        circ.rz(0.07 * (q + 2), q + 1)
    circ.swap(0, n - 1)
    circ.cz(1, n - 2)
    if measure_mid_circuit:
        circ.measure([0], [0])
        circ.h(0)
        circ.measure(list(range(n)), list(range(n)))
    else:
        circ.measure_all()
    observables = [
        ("Z0", QubitOperator({((0, "Z"),): 1.0})),
        ("XY", QubitOperator({((1, "X"), (n - 1, "Y")): 1.0})),
        ("ZZ", QubitOperator({((2, "Z"), (5, "Z")): 0.5})),
    ]
    return dumps(circ, observables=observables)


def _run(
    threads: int, qsal: str, *, precision: str = "double", shots: int = 0
) -> ExecutionResult:
    from quspect.native import NativeExecutor, set_num_threads

    set_num_threads(threads)
    executor = NativeExecutor(precision=precision, seed=1234)
    return executor.run(
        qsal,
        shots=shots,
        return_statevector=shots == 0,
        return_expectations=True,
    )


class TestThreadControls:
    def test_get_num_threads_is_positive(self) -> None:
        from quspect.native import get_num_threads

        assert get_num_threads() >= 1

    def test_set_num_threads_roundtrip(self) -> None:
        from quspect.native import capabilities, get_num_threads, set_num_threads

        set_num_threads(3)
        expected = 3 if capabilities()["openmp"] else 1
        assert get_num_threads() == expected
        set_num_threads(1)
        assert get_num_threads() == 1

    @pytest.mark.parametrize("bad", [0, -1])
    def test_set_num_threads_rejects_non_positive(self, bad: int) -> None:
        from quspect.native import set_num_threads

        with pytest.raises(ValueError):
            set_num_threads(bad)

    def test_set_num_threads_rejects_non_int(self) -> None:
        from quspect.native import set_num_threads

        with pytest.raises(TypeError):
            set_num_threads(2.0)  # type: ignore[arg-type]

    def test_core_set_num_threads_raises_value_error(self) -> None:
        assert _core is not None
        with pytest.raises(ValueError):
            _core.set_num_threads(0)

    def test_capabilities_report_threading(self) -> None:
        from quspect.native import capabilities, get_num_threads, set_num_threads

        set_num_threads(2)
        caps = capabilities()
        assert isinstance(caps["openmp"], bool)
        assert caps["num_threads"] == get_num_threads()


class TestThreadCountInvariance:
    @pytest.mark.parametrize("precision", ["double", "single"])
    def test_statevector_and_expectations_match(self, precision: str) -> None:
        qsal = _make_circuit_qsal(measure_mid_circuit=False)
        serial = _run(1, qsal, precision=precision)
        parallel = _run(4, qsal, precision=precision)

        attr = "statevector_f64" if precision == "double" else "statevector_f32"
        sv1 = np.asarray(getattr(serial, attr))
        sv4 = np.asarray(getattr(parallel, attr))
        assert sv1.shape == (2**_NUM_QUBITS,)
        tol = 1e-12 if precision == "double" else 1e-6
        np.testing.assert_allclose(sv4, sv1, rtol=0, atol=tol)
        assert math.isclose(float(np.vdot(sv1, sv1).real), 1.0, rel_tol=1e-4)

        exp1 = serial.expectations_real
        exp4 = parallel.expectations_real
        assert exp1.keys() == exp4.keys() == {"Z0", "XY", "ZZ"}
        for label, value in exp1.items():
            assert math.isclose(exp4[label], value, rel_tol=1e-9, abs_tol=tol)

    @pytest.mark.parametrize("mid_circuit", [False, True])
    def test_seeded_sampling_counts_identical(self, mid_circuit: bool) -> None:
        qsal = _make_circuit_qsal(measure_mid_circuit=mid_circuit)
        shots = 64 if mid_circuit else 2000
        serial = _run(1, qsal, shots=shots)
        parallel = _run(4, qsal, shots=shots)
        counts1 = serial.counts
        counts4 = parallel.counts
        assert sum(counts1.values()) == shots
        assert counts1 == counts4
