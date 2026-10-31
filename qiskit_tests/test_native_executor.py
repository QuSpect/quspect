"""Cross-validate the ``quspect.native`` wrappers against qiskit.

``NativeExecutor`` runs QSAL JSON (``quspect.ir.qsal.dumps``).  Raw native
conventions verified here:

* ``ExecutionResult.counts`` keys are MSB-first (clbit 0 *rightmost*, i.e.
  the qiskit convention); ``SimulationResult.from_native`` reverses them.
* ``packed_samples`` holds one integer per shot with bit ``c`` = clbit ``c``.
* ``precision="double"`` fills ``statevector_f64``; ``"single"`` fills
  ``statevector_f32`` (agreement with double within ~1e-6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

import quspect.native as native
from qiskit_tests.helpers import (
    assert_counts_match,
    assert_states_equal,
    qiskit_counts_to_quspect,
    qiskit_probabilities,
    random_circuit_pair,
    to_qiskit,
)
from quspect import Parameter, QuantumCircuit, QubitOperator
from quspect.ir.qsal import dumps
from quspect.native import (
    ExecutionResult,
    NativeExecutor,
    NativeImportError,
    capabilities,
    decode_qsal,
    validate,
)

SHOTS = 4000


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(4))
def test_run_statevector_double_and_single_match_qiskit(seed: int) -> None:
    """Double is exact (1e-12); single agrees with double within 2e-6."""
    circ, qc = random_circuit_pair(4, 40, seed)
    ref = Statevector(qc)
    qsal = dumps(circ)
    dbl = NativeExecutor(precision="double", seed=seed).run(
        qsal, return_statevector=True
    )
    sgl = NativeExecutor(precision="single", seed=seed).run(
        qsal, return_statevector=True
    )
    assert dbl.statevector_f32 == [] and sgl.statevector_f64 == []
    assert dbl.metadata["precision"] == "double"
    assert sgl.metadata["precision"] == "single"
    assert_states_equal(
        np.array(dbl.statevector_f64), ref, up_to_phase=False, atol=1e-12
    )
    sv32 = np.array(sgl.statevector_f32)
    assert_states_equal(sv32, ref, up_to_phase=False, atol=2e-6)
    np.testing.assert_allclose(sv32, np.array(dbl.statevector_f64), atol=2e-6)


def test_run_without_statevector_returns_empty_lists() -> None:
    """Outputs that were not requested are empty."""
    res = NativeExecutor(seed=1).run(dumps(QuantumCircuit().h(0)))
    assert res.statevector_f64 == [] and res.statevector_f32 == []
    assert res.counts == {} and res.packed_samples == []
    assert res.metadata["shots"] == "0"


@pytest.mark.qiskit_ref
@pytest.mark.statistical
@pytest.mark.parametrize("precision", ["double", "single"])
def test_native_counts_are_qiskit_ordered(precision: str) -> None:
    """Raw native counts use qiskit (MSB-first) keys and match qiskit stats."""
    circ, _ = random_circuit_pair(3, 18, 77)
    circ.measure([0, 1, 2], [0, 1, 2])
    res = NativeExecutor(precision=precision, seed=5).run(dumps(circ), shots=SHOTS)
    # qiskit_probabilities is quspect-keyed; reverse to the qiskit order.
    expected = {k[::-1]: v for k, v in qiskit_probabilities(circ).items()}
    assert_counts_match(res.counts, expected, SHOTS)
    assert res.metadata["measurement_mode"] == "terminal"


def test_native_count_key_and_packed_sample_bit_order() -> None:
    """``x(0)`` on 3 clbits: native key '001', packed sample 1 (bit0=clbit0)."""
    circ = QuantumCircuit().x(0).id(2).measure([0, 1, 2], [0, 1, 2])
    res = NativeExecutor(seed=0).run(dumps(circ), shots=5, return_samples=True)
    assert res.counts == {"001": 5}
    assert res.packed_samples == [1] * 5
    assert qiskit_counts_to_quspect(res.counts) == {"100": 5}


def test_packed_samples_consistent_with_counts() -> None:
    """Histogram of packed samples equals the counts dictionary."""
    circ = QuantumCircuit().h(0).h(1).ry(0.7, 2).measure([0, 1, 2], [0, 2, 1])
    res = NativeExecutor(seed=3).run(dumps(circ), shots=300, return_samples=True)
    assert len(res.packed_samples) == 300
    hist: dict[str, int] = {}
    for word in res.packed_samples:
        key = format(word, "03b")  # MSB-first: clbit 0 rightmost
        hist[key] = hist.get(key, 0) + 1
    assert hist == res.counts


def test_seed_reproducibility() -> None:
    """Same seed -> same counts; different seed -> different counts."""
    circ = QuantumCircuit()
    for q in range(5):
        circ.h(q)
    qsal = dumps(circ.measure_all())
    a = NativeExecutor(seed=11).run(qsal, shots=500).counts
    b = NativeExecutor(seed=11).run(qsal, shots=500).counts
    c = NativeExecutor(seed=12).run(qsal, shots=500).counts
    assert a == b
    assert a != c


def test_decode_validate_run_program() -> None:
    """decode -> validate (no errors) -> run_program, as in ``run``."""
    circ = QuantumCircuit().h(0).cx(0, 1).measure_all()
    qsal = dumps(circ)
    ex = NativeExecutor(seed=4)
    program = ex.decode(qsal)
    assert validate(program) == []
    assert validate(decode_qsal(qsal)) == []
    res = ex.run_program(
        program, shots=200, return_statevector=True, return_samples=True
    )
    assert set(res.counts) <= {"00", "11"}
    assert sum(res.counts.values()) == 200
    assert len(res.packed_samples) == 200
    ref = Statevector(to_qiskit(QuantumCircuit().h(0).cx(0, 1)))
    assert_states_equal(np.array(res.statevector_f64), ref, up_to_phase=False)


def test_validate_reports_errors() -> None:
    """Validation rejects a gate with the wrong number of qubits."""
    from quspect.gates import Gate

    bad = QuantumCircuit().append(Gate(name="rx", num_qubits=2, params=(0.1,)), [0, 1])
    program = decode_qsal(dumps(bad))
    errors = validate(program)
    assert errors and all(isinstance(e, str) for e in errors)
    with pytest.raises(RuntimeError, match="validation"):
        NativeExecutor().run(dumps(bad))


def test_capabilities_shape() -> None:
    """``capabilities`` returns a dict with version and backend keys."""
    caps = capabilities()
    assert isinstance(caps, dict)
    assert "version" in caps
    assert "backend_f64" in caps and "backend_f32" in caps


@pytest.mark.parametrize(
    ("call", "exc", "match"),
    [
        (lambda: NativeExecutor(precision="float"), ValueError, "precision"),
        (lambda: NativeExecutor().run("not json"), RuntimeError, "decode"),
        (lambda: decode_qsal("{}"), RuntimeError, "version"),
    ],
)
def test_native_errors(call: object, exc: type[Exception], match: str) -> None:
    """Bad precision / malformed QSAL raise informative errors."""
    with pytest.raises(exc, match=match):
        call()  # type: ignore[operator]


def test_missing_core_raises_native_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the compiled core every entry point raises NativeImportError."""
    monkeypatch.setattr(native, "_core", None)
    for call in (
        lambda: NativeExecutor(),
        lambda: decode_qsal("{}"),
        lambda: validate(None),
        lambda: capabilities(),
    ):
        with pytest.raises(NativeImportError, match="cmake"):
            call()
    assert issubclass(NativeImportError, ImportError)


def test_execution_result_defaults() -> None:
    """The Python ``ExecutionResult`` dataclass defaults to empty containers."""
    res = ExecutionResult()
    assert res.statevector_f64 == [] and res.counts == {} and res.metadata == {}


# =============================================================================
# Expectations and the adjoint plan
# =============================================================================


def _hamiltonian() -> tuple[QubitOperator, SparsePauliOp]:
    op = QubitOperator({((0, "Z"), (1, "X")): 0.5, ((1, "Y"),): 0.25, (): 0.1})
    ref = SparsePauliOp.from_sparse_list(
        [("ZX", [0, 1], 0.5), ("Y", [1], 0.25), ("", [], 0.1)], 2
    )
    return op, ref


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("precision", ["double", "single"])
def test_expectations_match_qiskit(precision: str) -> None:
    """``return_expectations`` equals qiskit ``Statevector.expectation_value``."""
    circ = QuantumCircuit().ry(0.4, 0).cx(0, 1).rx(0.3, 1).h(0)
    op, ref = _hamiltonian()
    z0 = QubitOperator({((0, "Z"),): 1.0})
    qsal = dumps(circ, observables=[("e", op), ("z0", z0)])
    res = NativeExecutor(precision=precision, seed=0).run(
        qsal, return_expectations=True
    )
    sv = Statevector(to_qiskit(circ))
    tol = 1e-12 if precision == "double" else 2e-6
    assert res.expectations_real["e"] == pytest.approx(
        float(sv.expectation_value(ref).real), abs=tol
    )
    z_ref = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], 2)
    assert res.expectations_real["z0"] == pytest.approx(
        float(sv.expectation_value(z_ref).real), abs=tol
    )


@pytest.mark.qiskit_ref
def test_adjoint_plan_value_and_gradient_match_qiskit() -> None:
    """Adjoint plan value == qiskit expectation; Jacobian == finite diff."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circ = QuantumCircuit().ry(2 * theta + 0.1, 0).cx(0, 1).rx(phi - theta, 1)
    op, ref = _hamiltonian()
    plan = NativeExecutor()._prepare_adjoint(dumps(circ, observables=[("e", op)]))
    assert set(plan.parameter_names) == {"theta", "phi"}
    assert plan.observable_names == ("e",)

    def energy(t: float, p: float) -> float:
        bound = circ.bind_parameters({theta: t, phi: p})
        return float(Statevector(to_qiskit(bound)).expectation_value(ref).real)

    t0, p0 = 0.37, -0.8
    out = plan.evaluate({"theta": t0, "phi": p0})
    assert out.observable_names == ("e",)
    assert out.values[0] == pytest.approx(energy(t0, p0), abs=1e-12)
    h = 1e-6
    fd = {
        "theta": (energy(t0 + h, p0) - energy(t0 - h, p0)) / (2 * h),
        "phi": (energy(t0, p0 + h) - energy(t0, p0 - h)) / (2 * h),
    }
    for col, name in enumerate(out.parameter_names):
        assert out.jacobian[0][col] == pytest.approx(fd[name], abs=1e-7)
    assert not math.isnan(out.values[0])
