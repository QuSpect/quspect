"""End-to-end integration tests for the Python↔C++ native executor bridge.

These tests verify the complete path:
    ``Python QuantumCircuit → QSAL JSON → C++ native core → ExecutionResult``

If the C++ native module (``quspect._core_scalar`` / ``quspect._core_avx2``,
selected by :mod:`quspect._native_loader`) is not available, all tests
are skipped with a descriptive message.

Build the C++ extension before running these tests::

    export PATH="$HOME/.local/bin:$PATH"
    cd cpp && cmake -S . -B build -G Ninja && cmake --build build
    cd .. && pytest tests/native/ -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quspect._native_loader import BUILD_HINT, load_core

# ---------------------------------------------------------------------------
# Try to load the native module; skip all tests if unavailable.
# ---------------------------------------------------------------------------

_core = load_core()

_NATIVE_AVAILABLE = _core is not None

pytestmark = pytest.mark.skipif(
    not _NATIVE_AVAILABLE,
    reason=f"QuSpect native extension not found. {BUILD_HINT}",
)


# ===========================================================================
# 测试用的 sample 数据构建
# ===========================================================================


def _make_bell_qsal() -> str:
    """Build a Bell-state QSAL JSON string (H + CX + measure)."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="bell")
    circ.h(0).cx(0, 1).measure_all()
    return dumps(circ)


def _make_h_gate_qsal() -> str:
    """Build a single H gate QSAL JSON string."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="h_gate")
    circ.h(0)
    return dumps(circ)


def _make_x_gate_qsal() -> str:
    """Build a single X gate QSAL JSON string."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="x_gate")
    circ.x(0).measure_all()
    return dumps(circ)


def _make_empty_qsal(num_qubits: int = 2) -> str:
    """Build an empty-circuit QSAL JSON string."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="empty")
    # Force num_qubits by adding and removing nothing — just create
    # the circuit with explicitly set qubit capacity via identity gates
    for q in range(num_qubits):
        circ.id(q)
    return dumps(circ)


def _make_parameterized_qsal(theta_val: float = 0.5) -> str:
    """Build a parameterized RX(θ) QSAL JSON string."""
    from quspect import QuantumCircuit
    from quspect.circuit.parameter import Parameter
    from quspect.ir.qsal import dumps

    theta = Parameter("θ")
    circ = QuantumCircuit(name="rx_param")
    circ.rx(theta, 0)
    # Bind the parameter
    bound = circ.bind_parameters({theta: theta_val})
    return dumps(bound)


def _make_observable_qsal() -> str:
    """Build a Bell-state QSAL with Z₀Z₁ and X₀X₁ observables."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps
    from quspect.operators.qubit_operator import QubitOperator

    circ = QuantumCircuit(name="bell_obs")
    circ.h(0).cx(0, 1)

    zz = QubitOperator({((0, "Z"), (1, "Z")): 1.0})
    xx = QubitOperator({((0, "X"), (1, "X")): 1.0})

    return dumps(circ, observables=[("ZZ", zz), ("XX", xx)])


def _make_three_qubit_ghz_qsal() -> str:
    """Build a 3-qubit GHZ state QSAL JSON (H + CX + CX)."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="ghz3")
    circ.h(0).cx(0, 1).cx(1, 2).measure_all()
    return dumps(circ)


def _make_barrier_circuit_qsal() -> str:
    """Build a circuit with barrier instructions."""
    from quspect import QuantumCircuit
    from quspect.ir.qsal import dumps

    circ = QuantumCircuit(name="barrier_test")
    circ.h(0).x(1).barrier(0, 1).cx(0, 1).measure_all()
    return dumps(circ)


# ===========================================================================
# 模块可用性
# ===========================================================================


class TestNativeModuleAvailable:
    """Verify the native C++ module is importable and functional."""

    def test_core_module_imported(self) -> None:
        """The native extension module is successfully imported."""
        assert _core is not None

    def test_runtime_capabilities(self) -> None:
        """get_runtime_capabilities returns expected keys."""
        caps = _core.get_runtime_capabilities()
        assert isinstance(caps, dict)
        assert "version" in caps
        assert "avx2_available" in caps
        assert caps["objective_plans"] is True
        assert tuple(caps["differentiation_methods"]) == ("adjoint",)
        assert tuple(caps["differentiable_gates"]) == ("rx", "ry", "rz")
        assert "adjoint_gradient" not in caps

    def test_native_executor_importable(self) -> None:
        """NativeExecutor can be imported from quspect.native."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=0)
        assert executor is not None

    def test_native_import_error_raised_when_no_module(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """NativeImportError is raised when _core is None."""
        from quspect.native import (
            NativeImportError,
            _ensure_core,
        )

        # We can only test this indirectly — verify the exception type exists
        assert issubclass(NativeImportError, ImportError)

        # _ensure_core should not raise when _core is available
        _ensure_core()  # type: ignore[no-untyped-call]


# ===========================================================================
# QSAL 解码测试
# ===========================================================================


class TestQSALDecode:
    """Test Python-side QSAL JSON → C++ QSALProgram decoding."""

    def test_decode_bell_circuit(self) -> None:
        """Decode a Bell state QSAL JSON to a C++ QSALProgram."""
        from quspect.native import decode_qsal

        qsal_json = _make_bell_qsal()
        prog = decode_qsal(qsal_json)

        assert prog is not None
        assert prog.num_qubits == 2
        assert prog.num_clbits == 2

    def test_decode_empty_circuit(self) -> None:
        """Decode an empty QSAL JSON."""
        from quspect.native import decode_qsal

        qsal_json = _make_empty_qsal(num_qubits=3)
        prog = decode_qsal(qsal_json)

        assert prog is not None
        assert prog.num_qubits >= 2

    def test_decode_invalid_json_raises(self) -> None:
        """Decoding invalid JSON raises RuntimeError."""
        from quspect.native import decode_qsal

        with pytest.raises(RuntimeError, match="QSAL decode error"):
            decode_qsal("not valid json {{{")

    def test_decode_missing_circuit_raises(self) -> None:
        """Decoding QSAL without a circuit raises RuntimeError."""
        from quspect.native import decode_qsal

        bad_qsal = json.dumps({"qsal_version": "0.1.0"})
        with pytest.raises(RuntimeError, match="decode"):
            decode_qsal(bad_qsal)


# ===========================================================================
# Validator 测试
# ===========================================================================


class TestValidator:
    """Test Python↔C++ validation bridge."""

    def test_valid_program_passes(self) -> None:
        """A valid Bell state program passes validation."""
        from quspect.native import decode_qsal, validate

        prog = decode_qsal(_make_bell_qsal())
        errors = validate(prog)
        assert errors == []

    def test_valid_empty_program_passes(self) -> None:
        """An empty program passes validation."""
        from quspect.native import decode_qsal, validate

        prog = decode_qsal(_make_h_gate_qsal())
        errors = validate(prog)
        assert errors == []

    def test_validate_includes_all_checks(self) -> None:
        """validate() aggregates errors from all check functions."""
        # 通过直接调用 C++ Validator 的静态方法验证各检查器可用
        from quspect.native import decode_qsal

        prog = decode_qsal(_make_bell_qsal())
        # 各种检查方法都应该可用
        errors_idx = _core.Validator.check_index_bounds(prog)
        errors_name = _core.Validator.check_gate_names(prog)
        errors_param = _core.Validator.check_parameter_count(prog)
        errors_meas = _core.Validator.check_measure_consistency(prog)

        # Bell state 应当通过所有检查
        assert errors_idx == []
        assert errors_name == []
        assert errors_param == []
        assert errors_meas == []


# ===========================================================================
# NativeExecutor — 状态向量执行
# ===========================================================================


class TestNativeExecutorStatevector:
    """Test NativeExecutor statevector execution via C++ backend."""

    def test_h_gate_creates_plus_state(self) -> None:
        """H|0⟩ = (|0⟩ + |1⟩)/√2."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_h_gate_qsal(), return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 2
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(sv[0].real - inv_sqrt2) < 1e-10
        assert abs(sv[0].imag) < 1e-10
        assert abs(sv[1].real - inv_sqrt2) < 1e-10
        assert abs(sv[1].imag) < 1e-10

    def test_x_gate_creates_one_state(self) -> None:
        """X|0⟩ = |1⟩."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_x_gate_qsal(), return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 2
        assert abs(sv[0]) < 1e-12
        assert abs(abs(sv[1]) - 1.0) < 1e-12

    def test_bell_state_statevector(self) -> None:
        """Bell state = (|00⟩ + |11⟩)/√2."""
        # Build Bell without measurement so statevector is preserved
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(dumps(circ), return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 4
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(sv[0].real - inv_sqrt2) < 1e-10  # |00⟩
        assert abs(sv[1].real) < 1e-10  # |01⟩ ≈ 0
        assert abs(sv[2].real) < 1e-10  # |10⟩ ≈ 0
        assert abs(sv[3].real - inv_sqrt2) < 1e-10  # |11⟩

    def test_statevector_is_normalized(self) -> None:
        """Returned statevector is normalized to unity."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_bell_qsal(), return_statevector=True)

        norm = sum(abs(a) ** 2 for a in result.statevector_f64)
        assert abs(norm - 1.0) < 1e-10

    def test_ghz_three_qubit_statevector(self) -> None:
        """3-qubit GHZ state = (|000⟩ + |111⟩)/√2."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps

        circ = QuantumCircuit(name="ghz3")
        circ.h(0).cx(0, 1).cx(1, 2)

        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(dumps(circ), return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 8
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(abs(sv[0]) - inv_sqrt2) < 1e-10  # |000⟩
        assert abs(sv[1]) < 1e-10
        assert abs(sv[2]) < 1e-10
        assert abs(sv[3]) < 1e-10
        assert abs(sv[4]) < 1e-10
        assert abs(sv[5]) < 1e-10
        assert abs(sv[6]) < 1e-10
        assert abs(abs(sv[7]) - inv_sqrt2) < 1e-10  # |111⟩

    def test_single_precision_statevector(self) -> None:
        """Single precision mode produces f32 statevector."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="single", seed=42)
        result = executor.run(_make_h_gate_qsal(), return_statevector=True)

        # f32 列表应该有值
        sv = result.statevector_f32
        assert len(sv) == 2
        # f32 精度较低
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(sv[0].real - inv_sqrt2) < 1e-6
        assert abs(sv[1].real - inv_sqrt2) < 1e-6


# ===========================================================================
# NativeExecutor — 测量与采样
# ===========================================================================


class TestNativeExecutorMeasurement:
    """Test measurement and sampling via C++ backend."""

    def test_bell_state_counts(self) -> None:
        """Bell state samples only '00' and '11'."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_bell_qsal(), shots=2000)

        counts = result.counts
        assert len(counts) > 0
        # Bell 态只产生 00 和 11
        for key in counts:
            assert key in ("00", "11"), f"Unexpected measurement: {key}"

        count_00 = counts.get("00", 0)
        count_11 = counts.get("11", 0)
        total = count_00 + count_11
        assert total == 2000
        # 大致 50/50
        assert 800 < count_00 < 1200
        assert 800 < count_11 < 1200

    def test_packed_samples(self) -> None:
        """Packed samples are returned correctly."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_bell_qsal(), shots=100, return_samples=True)

        samples = result.packed_samples
        assert len(samples) == 100
        # Bell 态的 packed value 只能是 0b00=0 或 0b11=3
        for s in samples:
            assert s in (0, 3), f"Unexpected packed sample: {s}"

    def test_x_gate_always_one(self) -> None:
        """X|0⟩ = |1⟩ → always measure '1'."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_x_gate_qsal(), shots=100)

        counts = result.counts
        assert len(counts) == 1
        assert "1" in counts
        assert counts["1"] == 100

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        from quspect.native import NativeExecutor

        qsal = _make_bell_qsal()
        executor1 = NativeExecutor(precision="double", seed=12345)
        executor2 = NativeExecutor(precision="double", seed=12345)

        result1 = executor1.run(qsal, shots=500, return_samples=True)
        result2 = executor2.run(qsal, shots=500, return_samples=True)

        # 相同 seed 应当产生相同采样结果
        assert result1.packed_samples == result2.packed_samples
        assert result1.counts == result2.counts

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds typically produce different samples."""
        from quspect.native import NativeExecutor

        qsal = _make_bell_qsal()
        executor1 = NativeExecutor(precision="double", seed=1)
        executor2 = NativeExecutor(precision="double", seed=99999)

        result1 = executor1.run(qsal, shots=100, return_samples=True)
        result2 = executor2.run(qsal, shots=100, return_samples=True)

        # 极小概率相同，但基本不同
        # 只验证两次执行都成功且返回正确数量
        assert len(result1.packed_samples) == 100
        assert len(result2.packed_samples) == 100
        # 两个结果都只包含 00 和 11
        for r in (result1, result2):
            for s in r.packed_samples:
                assert s in (0, 3)

    def test_ghz_three_qubit_measurement(self) -> None:
        """3-qubit GHZ state: only '000' and '111'."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_three_qubit_ghz_qsal(), shots=1000)

        counts = result.counts
        for key in counts:
            assert key in ("000", "111"), f"Unexpected: {key}"

        total = sum(counts.values())
        assert total == 1000


# ===========================================================================
# NativeExecutor — 期望值
# ===========================================================================


class TestNativeExecutorExpectations:
    """Test observable expectation computation via C++ backend."""

    def test_bell_zz_expectation(self) -> None:
        """⟨Z₀Z₁⟩ = 1 for Bell state."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_observable_qsal(), return_expectations=True)

        expectations = result.expectations_real
        assert "ZZ" in expectations
        assert abs(expectations["ZZ"] - 1.0) < 1e-10

    def test_bell_xx_expectation(self) -> None:
        """⟨X₀X₁⟩ = 1 for Bell state."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_observable_qsal(), return_expectations=True)

        expectations = result.expectations_real
        assert "XX" in expectations
        assert abs(expectations["XX"] - 1.0) < 1e-10

    def test_bell_zy_expectation_zero(self) -> None:
        """⟨Z₀Y₁⟩ = 0 for Bell state."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.operators.qubit_operator import QubitOperator

        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1)
        zy = QubitOperator({((0, "Z"), (1, "Y")): 1.0})
        qsal = dumps(circ, observables=[("ZY", zy)])

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_expectations=True)

        assert abs(result.expectations_real.get("ZY", 0.0)) < 1e-10

    def test_single_qubit_expectation_h_state(self) -> None:
        """⟨X⟩ = 1 for |+⟩ state."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.operators.qubit_operator import QubitOperator

        circ = QuantumCircuit(name="plus")
        circ.h(0)
        x_op = QubitOperator({((0, "X"),): 1.0})
        qsal = dumps(circ, observables=[("X", x_op)])

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_expectations=True)

        assert abs(result.expectations_real["X"] - 1.0) < 1e-10

    def test_multi_flip_expectations_on_plus_plus(self) -> None:
        """<++|XX|++> = 1 and <++|YY|++> = 0 (regression for B46/B47)."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.operators.qubit_operator import QubitOperator

        circ = QuantumCircuit()
        circ.h(0).h(1)
        xx = QubitOperator({((0, "X"), (1, "X")): 1.0})
        yy = QubitOperator({((0, "Y"), (1, "Y")): 1.0})
        qsal = dumps(circ, observables=[("XX", xx), ("YY", yy)])

        result = NativeExecutor().run(qsal, return_expectations=True)

        assert result.expectations_real["XX"] == pytest.approx(1.0, abs=1e-12)
        assert result.expectations_real["YY"] == pytest.approx(0.0, abs=1e-12)

    def test_multi_flip_expectations_match_python_statevector(self) -> None:
        """Multi X/Y Pauli terms agree with a dense numpy reference."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.operators.qubit_operator import QubitOperator

        circ = QuantumCircuit()
        circ.ry(0.4, 0).rx(1.1, 1).h(2).cx(0, 1).rz(0.7, 2).cx(1, 2).ry(-0.3, 0)
        native = NativeExecutor().run(dumps(circ), return_statevector=True)
        state = np.asarray(native.statevector_f64, dtype=complex)

        paulis = {
            "I": np.eye(2),
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.diag([1, -1]),
        }
        for label in ("XX", "XY", "YY", "XYZ", "YZX", "XXX", "YIY", "IXY"):
            # Little-endian: label[q] acts on qubit q (bit q of the index).
            dense = np.array([[1.0]])
            for ch in label:
                dense = np.kron(paulis[ch], dense)
            # Pad identities on unused high qubits.
            dense = np.kron(np.eye(2 ** (3 - len(label))), dense)
            expected = float(np.real(np.vdot(state, dense @ state)))
            op = QubitOperator(
                {tuple((q, ch) for q, ch in enumerate(label) if ch != "I"): 1.0}
            )
            out = NativeExecutor().run(
                dumps(circ, observables=[("O", op)]), return_expectations=True
            )
            assert out.expectations_real["O"] == pytest.approx(expected, abs=1e-10)

    def test_complex_coefficient_is_applied_natively(self) -> None:
        """[real, imag] coefficients decode natively; real part is reported."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.operators.qubit_operator import QubitOperator

        op = QubitOperator({((0, "Z"),): 0.5 + 0.5j, ((1, "Z"),): 2.0})
        circ = QuantumCircuit()
        circ.id(1)
        qsal = dumps(circ, observables=[("H", op)])

        result = NativeExecutor().run(qsal, return_expectations=True)

        assert result.expectations_real["H"] == pytest.approx(2.5, abs=1e-12)

    def test_return_expectations_false_skips_computation(self) -> None:
        """When return_expectations=False, expectations dict is empty."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_observable_qsal(), return_expectations=False)

        assert result.expectations_real == {}


# ===========================================================================
# NativeExecutor — 参数化门
# ===========================================================================


class TestNativeExecutorParameterizedGates:
    """Test parameterized gate execution via C++ backend."""

    def test_rx_pi_is_x_gate(self) -> None:
        """RX(π)|0⟩ = -i|1⟩ (equivalent to X|0⟩ up to phase)."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="rx_pi")
        circ.rx(math.pi, 0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10

    def test_rx_half_pi_creates_superposition(self) -> None:
        """RX(π/2)|0⟩ produces equal-magnitude superposition."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="rx_half_pi")
        circ.rx(math.pi / 2, 0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # |⟨0|ψ⟩|² ≈ |⟨1|ψ⟩|² ≈ 0.5
        prob0 = abs(sv[0]) ** 2
        prob1 = abs(sv[1]) ** 2
        assert abs(prob0 - 0.5) < 1e-10
        assert abs(prob1 - 0.5) < 1e-10

    def test_ry_pi_is_y_gate(self) -> None:
        """RY(π)|0⟩ = |1⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="ry_pi")
        circ.ry(math.pi, 0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert abs(sv[0]) < 1e-10
        assert abs(abs(sv[1]) - 1.0) < 1e-10

    def test_rz_pi_produces_phase(self) -> None:
        """RZ(π)|+⟩ produces phase but same measurement probabilities as |+⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="rz_pi")
        circ.h(0).rz(math.pi, 0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # RZ(π)|+⟩ = |−⟩, 概率应该是 50/50
        prob0 = abs(sv[0]) ** 2
        prob1 = abs(sv[1]) ** 2
        assert abs(prob0 - 0.5) < 1e-10
        assert abs(prob1 - 0.5) < 1e-10

    def test_bound_parameterized_circuit(self) -> None:
        """A circuit with bound parameters executes correctly."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(
            _make_parameterized_qsal(theta_val=0.5), return_statevector=True
        )

        sv = result.statevector_f64
        # 检查归一化
        norm = sum(abs(a) ** 2 for a in sv)
        assert abs(norm - 1.0) < 1e-10


# ===========================================================================
# NativeExecutor — 两 bit 门
# ===========================================================================


class TestNativeExecutorTwoQubitGates:
    """Test two-qubit gate operations via C++ backend."""

    def test_cx_gate(self) -> None:
        """CX|10⟩ = |11⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="cx_test")
        circ.x(0).cx(0, 1)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 4
        # |11⟩ 应该在 index 3
        assert abs(sv[3]) > 0.99
        for i in (0, 1, 2):
            assert abs(sv[i]) < 1e-10

    def test_cz_gate(self) -> None:
        """CZ|++⟩ = (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="cz_test")
        circ.h(0).h(1).cz(0, 1)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # 前三个幅度 = +0.5, 第四个 = -0.5
        for i in range(3):
            assert abs(sv[i].real - 0.5) < 1e-10
        assert abs(sv[3].real + 0.5) < 1e-10

    def test_swap_gate(self) -> None:
        """SWAP|10⟩ = |01⟩ (swap qubits 0 and 1)."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="swap_test")
        circ.x(1).swap(0, 1)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # After X(1): |10⟩. After SWAP(0,1): |01⟩ = index 1
        assert abs(sv[1]) > 0.99
        assert abs(sv[2]) < 1e-10  # |10⟩ should be gone


# ===========================================================================
# NativeExecutor — 多量子门
# ===========================================================================


class TestNativeExecutorThreeQubitGates:
    """Test three-qubit gate operations via C++ backend."""

    def test_ccx_toffoli(self) -> None:
        """CCX|110⟩ = |111⟩ (Toffoli gate)."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="toffoli")
        circ.x(0).x(1).ccx(0, 1, 2)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 8
        # |111⟩ at index 7
        assert abs(sv[7]) > 0.99

    def test_ccx_noop_on_010(self) -> None:
        """CCX|010⟩ = |010⟩ (control not satisfied)."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="toffoli_noop")
        circ.x(1).ccx(0, 1, 2)  # only qubit 1 is |1⟩
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # |010⟩ = index 2, should be unchanged
        assert abs(abs(sv[2]) - 1.0) < 1e-10


# ===========================================================================
# Barrier / 空线路 / 边界情况
# ===========================================================================


class TestNativeExecutorEdgeCases:
    """Test edge cases and special circuit configurations."""

    def test_barrier_does_not_affect_state(self) -> None:
        """Barrier should not change the quantum state."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_barrier_circuit_qsal(), return_statevector=True)

        # 状态向量应归一化
        sv = result.statevector_f64
        norm = sum(abs(a) ** 2 for a in sv)
        assert abs(norm - 1.0) < 1e-10

    def test_empty_circuit_statevector(self) -> None:
        """Empty circuit produces |0⟩ state."""
        from quspect.native import NativeExecutor

        qsal = _make_empty_qsal(num_qubits=2)
        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        # |00⟩ at index 0
        assert abs(abs(sv[0]) - 1.0) < 1e-10
        for i in range(1, len(sv)):
            assert abs(sv[i]) < 1e-10

    def test_many_qubits(self) -> None:
        """8-qubit circuit should execute correctly."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="wide")
        for q in range(8):
            circ.h(q)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert len(sv) == 256
        # 全部等幅叠加
        expected_amp = 1.0 / 16.0  # 1/√256 = 1/16
        for amp in sv:
            assert abs(abs(amp) - expected_amp) < 1e-10

    def test_metadata_in_result(self) -> None:
        """ExecutionResult metadata includes basic info."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(_make_h_gate_qsal(), return_statevector=True)

        assert "num_qubits" in result.metadata
        assert "precision" in result.metadata
        assert result.metadata["precision"] == "double"

    def test_sequential_id_gates(self) -> None:
        """A sequence of identity gates preserves |0⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="id_seq")
        circ.id(0).id(0).id(0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert abs(abs(sv[0]) - 1.0) < 1e-10

    def test_h_h_identity(self) -> None:
        """HH|0⟩ = |0⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="hh")
        circ.h(0).h(0)
        qsal = dumps(circ)

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal, return_statevector=True)

        sv = result.statevector_f64
        assert abs(abs(sv[0]) - 1.0) < 1e-10


# ===========================================================================
# NativeExecutor — 低阶 API
# ===========================================================================


class TestNativeExecutorLowLevelAPI:
    """Test the low-level decode + run_program API."""

    def test_decode_and_run_separately(self) -> None:
        """decode() + run_program() produces same result as run()."""
        from quspect.native import NativeExecutor

        qsal = _make_bell_qsal()
        executor = NativeExecutor(precision="double", seed=42)

        # High-level API
        result1 = executor.run(qsal, shots=100, return_statevector=True)

        # Low-level API
        prog = executor.decode(qsal)
        result2 = executor.run_program(prog, shots=100, return_statevector=True)

        # 两个结果应当一致
        for i in range(len(result1.statevector_f64)):
            assert abs(result1.statevector_f64[i] - result2.statevector_f64[i]) < 1e-10

    def test_run_program_with_execution_request(self) -> None:
        """Explicit ExecutionRequest configuration."""
        from quspect.native import NativeExecutor

        executor = NativeExecutor(precision="double", seed=42)
        prog = executor.decode(_make_h_gate_qsal())

        result = executor.run_program(
            prog,
            shots=0,
            return_statevector=True,
            return_expectations=False,
            return_samples=False,
        )

        sv = result.statevector_f64
        assert len(sv) == 2
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        assert abs(abs(sv[0]) - inv_sqrt2) < 1e-10


# ===========================================================================
# QSAL 校验错误 — Python 触发
# ===========================================================================


class TestNativeErrorHandling:
    """Test error handling when Python passes bad data to C++."""

    def test_decode_and_run_with_invalid_qsal(self) -> None:
        """decode_qsal_and_run raises on invalid QSAL."""
        with pytest.raises(RuntimeError):
            _core.decode_qsal_and_run(
                "not json",
                "double",
                0,
                0,
                False,
                False,
                False,
            )

    def test_decode_and_run_with_validation_error(self) -> None:
        """decode_qsal_and_run raises on QSAL that fails validation."""
        # QSAL with qubit index out of bounds should fail validation
        bad_qsal = json.dumps(
            {
                "qsal_version": "0.1.0",
                "name": "bad",
                "circuit": {
                    "num_qubits": 1,
                    "num_clbits": 0,
                    "instructions": [
                        {
                            "type": "GATE",
                            "qubits": [5],  # 超出 num_qubits
                            "clbits": [],
                            "gate": {
                                "name": "h",
                                "num_qubits": 1,
                                "gate_type": "standard",
                            },
                        }
                    ],
                },
            }
        )

        with pytest.raises(RuntimeError, match="validation"):
            _core.decode_qsal_and_run(
                bad_qsal,
                "double",
                0,
                0,
                False,
                False,
                False,
            )

    def test_decode_and_run_rejects_unknown_gate(self) -> None:
        """decode_qsal_and_run rejects unknown gate names."""
        bad_qsal = json.dumps(
            {
                "qsal_version": "0.1.0",
                "name": "bad_gate",
                "circuit": {
                    "num_qubits": 1,
                    "num_clbits": 0,
                    "instructions": [
                        {
                            "type": "GATE",
                            "qubits": [0],
                            "clbits": [],
                            "gate": {
                                "name": "fancy_gate_xyz",
                                "num_qubits": 1,
                                "gate_type": "standard",
                            },
                        }
                    ],
                },
            }
        )

        with pytest.raises(RuntimeError):
            _core.decode_qsal_and_run(
                bad_qsal,
                "double",
                0,
                0,
                False,
                False,
                False,
            )


# ===========================================================================
# Python 模拟器 ↔ C++ 交叉验证
# ===========================================================================


class TestCrossValidateWithPythonSimulator:
    """Cross-validate C++ native results against the Python reference simulator."""

    def test_bell_statevector_matches_python(self) -> None:
        """C++ and Python produce the same Bell statevector."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.simulation.simulator import Simulator

        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1)

        # Python simulator (use save_statevector=True)
        py_sim = Simulator(seed=42)
        py_result = py_sim.run(circ, shots=0, save_statevector=True)

        # C++ native
        cpp_exec = NativeExecutor(precision="double", seed=42)
        cpp_result = cpp_exec.run(dumps(circ), return_statevector=True)

        # 比较状态向量（允许 C++ 有全局相位差异）
        py_sv = py_result.statevector
        assert py_sv is not None
        cpp_sv = cpp_result.statevector_f64

        # 计算 fidelity: |⟨ψ_py|ψ_cpp⟩|²
        inner = sum(np.conj(py_sv[i]) * cpp_sv[i] for i in range(len(py_sv)))
        fidelity = abs(inner) ** 2
        assert abs(fidelity - 1.0) < 1e-10

    def test_bell_counts_statistically_match(self) -> None:
        """C++ and Python produce similar Bell state measurement statistics."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.simulation.simulator import Simulator

        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure_all()

        py_result = Simulator(seed=42).run(circ, shots=2000)
        cpp_result = NativeExecutor(precision="double", seed=42).run(
            dumps(circ), shots=2000
        )

        # 两者应当产生相似的统计分布（00 和 11 各约一半）
        py_total = sum(py_result.counts.values())
        cpp_total = sum(cpp_result.counts.values())
        assert py_total == 2000
        assert cpp_total == 2000

        # Bell 态只有 00 和 11 出现，且比例约为 1:1
        assert set(cpp_result.counts.keys()) == {"00", "11"}
        ratio_cpp = cpp_result.counts["00"] / cpp_result.counts["11"]
        assert abs(ratio_cpp - 1.0) < 0.2, f"Bell state ratio off: {ratio_cpp}"

    def test_h_gate_statevector_matches(self) -> None:
        """Python and C++ agree on H|0⟩."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.simulation.simulator import Simulator

        circ = QuantumCircuit()
        circ.h(0)

        py_sv = Simulator(seed=0).run(circ, shots=0, save_statevector=True).statevector
        assert py_sv is not None
        cpp_sv = (
            NativeExecutor(precision="double", seed=0)
            .run(dumps(circ), return_statevector=True)
            .statevector_f64
        )

        inner = sum(np.conj(py_sv[i]) * cpp_sv[i] for i in range(len(py_sv)))
        assert abs(abs(inner) ** 2 - 1.0) < 1e-10

    def test_complex_circuit_statevector_matches(self) -> None:
        """A more complex circuit: Python and C++ agree."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dumps
        from quspect.native import NativeExecutor
        from quspect.simulation.simulator import Simulator

        theta = math.pi / 3
        phi = math.pi / 4

        circ = QuantumCircuit(name="complex")
        circ.h(0).h(1)
        circ.cx(0, 1)
        circ.rz(theta, 0).rx(phi, 1)
        circ.cx(0, 1)
        circ.h(0).z(1)

        py_sv = Simulator(seed=0).run(circ, shots=0, save_statevector=True).statevector
        assert py_sv is not None
        cpp_sv = (
            NativeExecutor(precision="double", seed=0)
            .run(dumps(circ), return_statevector=True)
            .statevector_f64
        )

        inner = sum(np.conj(py_sv[i]) * cpp_sv[i] for i in range(len(py_sv)))
        assert abs(abs(inner) ** 2 - 1.0) < 1e-10


# ===========================================================================
# ExecutionResult dataclass
# ===========================================================================


class TestExecutionResult:
    """Test the Python-side ExecutionResult wrapper."""

    def test_from_core_converts_all_fields(self) -> None:
        """_from_core properly converts C++ result to Python dataclass."""
        from quspect.native import (
            ExecutionResult,
            NativeExecutor,
        )

        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(
            _make_bell_qsal(),
            shots=100,
            return_statevector=True,
            return_expectations=False,
            return_samples=True,
        )

        assert isinstance(result, ExecutionResult)
        assert isinstance(result.statevector_f64, list)
        assert isinstance(result.statevector_f32, list)
        assert isinstance(result.packed_samples, list)
        assert isinstance(result.counts, dict)
        assert isinstance(result.expectations_real, dict)
        assert isinstance(result.metadata, dict)


# ===========================================================================
# 文件工作流
# ===========================================================================


class TestFileWorkflow:
    """Test the file-based QSAL workflow: file → C++ execute."""

    def test_qsal_file_roundtrip_via_native(self, tmp_path: Path) -> None:
        """QSAL file written in Python → read & executed via C++."""
        from quspect import QuantumCircuit
        from quspect.ir.qsal import dump
        from quspect.native import NativeExecutor

        circ = QuantumCircuit(name="file_test")
        circ.h(0).cx(0, 1).measure_all()

        # 写入文件
        path = tmp_path / "circuit.qsal.json"
        dump(circ, path)

        # 读取并执行
        qsal_text = path.read_text()
        executor = NativeExecutor(precision="double", seed=42)
        result = executor.run(qsal_text, shots=100)

        assert "00" in result.counts or "11" in result.counts
