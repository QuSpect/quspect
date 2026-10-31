"""Tests for ``SimulationResult`` and its constructors.

``from_statevector`` is cross-checked statistically against the exact qiskit
``Statevector.probabilities_dict`` (keys reversed to the quspect convention:
``s[i]`` is qubit / clbit ``i``).  ``from_native`` is checked against a fake
native result to pin down the MSB-first -> LSB-first key reversal.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qiskit_tests.helpers import (
    assert_counts_match,
    qiskit_counts_to_quspect,
    random_circuit_pair,
)
from quspect import QuantumCircuit
from quspect.simulation import SimulationResult, Simulator

SHOTS = 4000


def _result() -> SimulationResult:
    return SimulationResult(
        counts={"00": 500, "11": 450, "01": 50},
        probabilities={"00": 0.5, "11": 0.45, "01": 0.05},
        num_shots=1000,
        num_qubits=2,
        num_clbits=2,
    )


def test_defaults_and_detached_mutables() -> None:
    """Defaults; ``metadata`` and ``memory`` are copied on construction."""
    meta = {"backend": "x"}
    mem = ["00", "11"]
    res = SimulationResult(
        counts={"00": 1, "11": 1},
        probabilities={},
        num_shots=2,
        num_qubits=2,
        num_clbits=2,
        metadata=meta,
        memory=mem,
    )
    meta["backend"] = "y"
    mem.append("01")
    assert res.metadata == {"backend": "x"}
    assert res.memory == ["00", "11"]
    assert res.success is True and res.error is None and res.statevector is None
    assert _result().memory is None
    assert _result().metadata == {}


def test_simulator_result_has_no_memory() -> None:
    """The local simulator does not return per-shot memory."""
    res = Simulator(seed=1).run(QuantumCircuit().h(0).measure_all(), shots=10)
    assert res.memory is None
    assert res.num_shots == 10


def test_query_methods() -> None:
    """most_frequent / top_outcomes / get_count / get_probability."""
    res = _result()
    assert res.most_frequent() == ("00", 500)
    assert res.top_outcomes(2) == [("00", 500), ("11", 450)]
    assert res.top_outcomes() == [("00", 500), ("11", 450), ("01", 50)]
    assert res.get_count("11") == 450
    assert res.get_count("10") == 0
    assert res.get_probability("01") == 0.05
    assert res.get_probability("10") == 0.0
    empty = SimulationResult({}, {}, 0, 0, 0)
    assert empty.most_frequent() == ("", 0)
    assert empty.top_outcomes() == []


def test_repr_success_and_failure() -> None:
    """``repr`` summarizes top outcomes or the failure."""
    text = repr(_result())
    assert "shots=1000" in text and "00: 500" in text and "no statevector" in text
    failed = SimulationResult({}, {}, 5, 1, 1, success=False, error="boom")
    assert repr(failed) == (
        "SimulationResult(success=False, shots=5, qubits=1, error='boom')"
    )
    with_sv = SimulationResult({}, {}, 0, 1, 0, statevector=np.array([1, 0]))
    assert "with statevector" in repr(with_sv)


def test_display_smoke() -> None:
    """``display`` builds a SamplingDisplay with text and SVG renderings."""
    from quspect.visualization.sampling_viz import SamplingDisplay

    res = _result()
    disp = res.display()
    assert isinstance(disp, SamplingDisplay)
    assert "00" in str(disp)
    svg = res.display(style="vbar", sort_mode="probability", theme="dark")
    assert svg is not None
    assert "<svg" in res.display(style="svg")._repr_svg_()


# =============================================================================
# from_statevector
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize(("n", "seed"), [(1, 3), (2, 4), (3, 5)])
def test_from_statevector_sampling_matches_qiskit(n: int, seed: int) -> None:
    """Sampled counts follow ``|amp|^2`` with quspect (reversed) keys."""
    _, qc = random_circuit_pair(n, 8 * n, seed)
    sv = Statevector(qc)
    res = SimulationResult.from_statevector(
        sv.data, shots=SHOTS, rng=np.random.default_rng(seed)
    )
    expected = {k[::-1]: v for k, v in sv.probabilities_dict().items()}
    assert res.num_qubits == n and res.num_clbits == n and res.num_shots == SHOTS
    assert len(res.counts) == 1 << n, "all bitstrings present (zero-filled)"
    assert_counts_match(res.counts, expected, SHOTS)
    assert sum(res.probabilities.values()) == pytest.approx(1.0)


def test_from_statevector_bit_order() -> None:
    """|q0=1, q1=0> (index 1) is reported as '10' (quspect) == qiskit '01'."""
    sv = np.zeros(4, dtype=complex)
    sv[1] = 1.0
    res = SimulationResult.from_statevector(sv, shots=7, rng=np.random.default_rng(0))
    assert res.counts == {"00": 0, "01": 0, "10": 7, "11": 0}
    assert qiskit_counts_to_quspect(Statevector(sv).sample_counts(7)) == {"10": 7}
    assert res.get_probability("10") == 1.0


def test_from_statevector_options_and_errors() -> None:
    """save_statevector copies; shots=0; default rng; invalid inputs raise."""
    sv = np.array([1, 1], dtype=complex) / np.sqrt(2)
    res = SimulationResult.from_statevector(sv, shots=0, save_statevector=True)
    assert res.num_shots == 0
    assert res.counts == {"0": 0, "1": 0}
    assert res.probabilities == {"0": 0.0, "1": 0.0}
    assert res.statevector is not None and res.statevector is not sv
    np.testing.assert_array_equal(res.statevector, sv)
    default_rng = SimulationResult.from_statevector(sv, shots=20)
    assert sum(default_rng.counts.values()) == 20
    with pytest.raises(ValueError, match="shots"):
        SimulationResult.from_statevector(sv, shots=-1)
    with pytest.raises(ValueError, match="power of 2"):
        SimulationResult.from_statevector(np.ones(3, dtype=complex) / np.sqrt(3))


# =============================================================================
# from_native
# =============================================================================


def test_from_native_reverses_keys_and_normalizes() -> None:
    """Native MSB-first keys are reversed; probabilities are count / shots."""
    circ = QuantumCircuit().x(0).measure([0], [2])
    native = SimpleNamespace(
        counts={"001": 3, "110": 1},
        statevector_f64=[0j, 1 + 0j],
    )
    res = SimulationResult.from_native(
        native, circuit=circ, shots=4, save_statevector=True
    )
    assert res.counts == {"011": 1, "100": 3}
    assert res.probabilities == {"011": 0.25, "100": 0.75}
    assert res.num_qubits == 1 and res.num_clbits == 3  # sizes from circuit
    assert res.statevector is not None
    np.testing.assert_array_equal(res.statevector, [0, 1])
    no_sv = SimulationResult.from_native(native, circuit=circ, shots=4)
    assert no_sv.statevector is None
    zero = SimulationResult.from_native(
        SimpleNamespace(counts={}, statevector_f64=[]), circuit=circ, shots=0
    )
    assert zero.counts == {} and zero.num_shots == 0


def test_run_result_keys_are_sorted_and_consistent() -> None:
    """Counts / probabilities from ``run`` are sorted and consistent."""
    res = Simulator(seed=2).run(QuantumCircuit().h(0).h(1).measure_all(), shots=SHOTS)
    assert list(res.counts) == sorted(res.counts)
    assert list(res.probabilities) == sorted(res.probabilities)
    for key in res.counts:
        assert res.get_probability(key) == pytest.approx(res.get_count(key) / SHOTS)
    assert_counts_match(
        res.counts, {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}, SHOTS
    )


def test_from_statevector_zero_qubits() -> None:
    """A length-1 state is a 0-qubit register: every shot yields ''."""
    res = SimulationResult.from_statevector(
        np.array([1.0 + 0j]), shots=3, rng=np.random.default_rng(0)
    )
    assert res.num_qubits == 0
    assert res.counts == {"": 3}
