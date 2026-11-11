"""Tests for SingleQubitGateFusion pass."""

from __future__ import annotations

import random

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.dag.passes import _native
from quspect.dag.passes.single_qubit_fusion import SingleQubitGateFusion, _decompose_u3


class TestDecomposeU3:
    """Tests for the ZYZ / U3 decomposition helper."""

    def test_decompose_identity(self) -> None:
        """Decompose identity matrix → U3(0, 0, 0)."""
        identity_matrix = np.eye(2, dtype=complex)
        theta, phi, lam = _decompose_u3(identity_matrix)
        # Identity: θ ≈ 0
        assert abs(theta) < 1e-10

    def test_decompose_hadamard(self) -> None:
        """Decompose Hadamard matrix."""
        h_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        theta, phi, lam = _decompose_u3(h_mat)
        # H: θ = π/2, φ = 0 (or some valid value), λ ≈ π
        assert abs(theta - np.pi / 2) < 1e-9
        # Verify: the reconstructed matrix matches up to global phase
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))
        # Check unitary equivalence: overlap should be 1
        overlap = abs(np.trace(h_mat @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_pauli_x(self) -> None:
        """Decompose Pauli-X matrix."""
        x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
        theta, phi, lam = _decompose_u3(x_mat)
        # X = U3(π, π/2, -π/2) up to phase
        assert abs(theta - np.pi) < 1e-9

    def test_decompose_roundtrip(self) -> None:
        """Reconstruction should match the original matrix up to global phase."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        theta = 1.3
        phi = 2.1
        lam = 0.7
        gate = U3.with_params(theta, phi, lam)
        original = gate_matrix(gate)

        theta2, phi2, lam2 = _decompose_u3(original)
        gate2 = U3.with_params(theta2, phi2, lam2)
        reconstructed = gate_matrix(gate2)

        # Check unitary equivalence up to global phase
        # The overlap |Tr(U V†)| / 2 should be 1 if U = e^{iα} V
        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_removes_global_phase_before_extracting_angles(self) -> None:
        """A global phase must not change the reconstructed U3 operation."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3

        original = np.exp(0.73j) * gate_matrix(U3.with_params(1.2, -0.4, 0.9))
        theta, phi, lam = _decompose_u3(original)
        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))

        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9

    def test_decompose_x_h_x_regression(self) -> None:
        """X-H-X must fuse to an equivalent U3 despite its π global phase."""
        from quspect.gates._matrix_utils import gate_matrix
        from quspect.gates.parameterized import U3
        from quspect.gates.standard import H, X

        original = gate_matrix(X) @ gate_matrix(H) @ gate_matrix(X)
        theta, phi, lam = _decompose_u3(original)
        reconstructed = gate_matrix(U3.with_params(theta, phi, lam))

        overlap = abs(np.trace(original @ reconstructed.conj().T)) / 2
        assert abs(overlap - 1.0) < 1e-9


class TestSingleQubitGateFusionBasic:
    """Basic fusion tests."""

    def test_h_h_fuses(self) -> None:
        """Two Hadamard gates should fuse (H·H = I → U3(0, 0, 0))."""
        circ = QuantumCircuit()
        circ.h(0).h(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # H·H = I, which is a single U3 gate (or identity is just removed)
        # 2 H gates become 1 fused gate
        assert optimized.num_nodes() == before - 1

    def test_x_h_x_fuses_to_equivalent_operation(self) -> None:
        """The fused X-H-X block must preserve its unitary up to global phase."""
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
        from quspect.gates._matrix_utils import gate_matrix

        circuit = QuantumCircuit().x(0).h(0).x(0)
        optimized = SingleQubitGateFusion().run(circuit.to_dag())
        reconstructed = DAGToCircuitCompiler.compile(optimized)

        original_matrix = (
            gate_matrix(circuit[2].gate)
            @ gate_matrix(circuit[1].gate)
            @ gate_matrix(circuit[0].gate)
        )
        fused_matrix = gate_matrix(reconstructed[0].gate)
        overlap = abs(np.trace(original_matrix @ fused_matrix.conj().T)) / 2

        assert len(reconstructed) == 1
        assert reconstructed[0].gate.name == "u3"
        assert abs(overlap - 1.0) < 1e-9

    def test_h_z_s_fuses(self) -> None:
        """H, Z, S on same qubit should fuse into single U3."""
        circ = QuantumCircuit()
        circ.h(0).z(0).s(0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # 3 single-qubit gates → 1 fused gate
        assert optimized.num_nodes() == before - 2

    def test_rx_rz_fuses(self) -> None:
        """RX(0.3) followed by RZ(0.5) should fuse."""
        circ = QuantumCircuit()
        circ.rx(0.3, 0).rz(0.5, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        assert optimized.num_nodes() == before - 1

    def test_no_fuse_different_qubits(self) -> None:
        """Single-qubit gates on different qubits should NOT fuse."""
        circ = QuantumCircuit()
        circ.h(0).x(1)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        assert optimized.num_nodes() == before


class TestSingleQubitGateFusionMultiQubit:
    """Fusion in multi-qubit context."""

    def test_fuse_with_cx_barrier(self) -> None:
        """CX gate should act as barrier for fusion."""
        circ = QuantumCircuit()
        circ.h(0).rz(0.3, 0).cx(0, 1).x(0)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        # H + RZ on q0 before CX → 1 fused gate + CX + X
        # 2 IN + 3 ops + 2 OUT = 7 (was 8 with H and RZ separate)
        assert optimized.num_nodes() == 7

    def test_independent_wires_fuse(self) -> None:
        """Fusion on different wires should work independently."""
        circ = QuantumCircuit()
        circ.h(0).x(0).h(1).z(1)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        # q0: H+X fused → 1, q1: H+Z fused → 1
        # 2 IN + 2 fused + 2 OUT = 6 (was 8 with 4 single-qubit gates)
        assert optimized.num_nodes() == 6

    def test_parameterized_not_fused(self) -> None:
        """Symbolic parameters should prevent fusion."""
        from quspect import Parameter

        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).rz(0.3, 0)
        dag = circ.to_dag()
        before = dag.num_nodes()
        optimized = SingleQubitGateFusion().run(dag)
        # RX(θ) has symbolic param → don't fuse
        assert optimized.num_nodes() == before


class TestSingleQubitGateFusionDAGReconstruct:
    """Test that the fused DAG can be reconstructed to a circuit."""

    def test_fuse_and_reconstruct(self) -> None:
        """Fuse gates and verify the circuit can be reconstructed."""
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler

        circ = QuantumCircuit()
        circ.h(0).s(0).t(0).z(0)
        dag = circ.to_dag()
        optimized = SingleQubitGateFusion().run(dag)
        reconstructed = DAGToCircuitCompiler.compile(optimized)
        # 4 single-qubit gates → 1 fused U3
        assert len(reconstructed) == 1
        assert reconstructed[0].gate.name == "u3"


# ---------------------------------------------------------------------------
# Basis-aware mode
# ---------------------------------------------------------------------------

_BASES = [
    frozenset({"h", "rz"}),
    frozenset({"rz", "sx", "x"}),
    frozenset({"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}),
    frozenset({"rx", "rz"}),
    frozenset({"u3"}),
]

#: Python reference first, then the native pass when the extension is built.
_BACKENDS = ["python"] + (["native"] if _native.enabled() else [])


def _unitary(circuit: QuantumCircuit, num_qubits: int) -> np.ndarray:
    """Unitary of ``circuit`` with qubit 0 as the most significant bit."""
    from quspect.gates._matrix_utils import gate_matrix

    dim = 2**num_qubits
    u = np.eye(dim, dtype=complex).reshape([2] * (2 * num_qubits))
    for ins in circuit.instructions:
        assert ins.gate is not None
        k = len(ins.qubits)
        g = gate_matrix(ins.gate).reshape([2] * (2 * k))
        u = np.tensordot(g, u, axes=(list(range(k, 2 * k)), list(ins.qubits)))
        u = np.moveaxis(u, list(range(k)), list(ins.qubits))
    return u.reshape(dim, dim)


def _phase_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Operator-norm distance of two unitaries, minimised over the global phase."""
    ov = np.trace(b.conj().T @ a)
    assert abs(ov) > 1e-12
    return float(np.linalg.norm(a - ov / abs(ov) * b, 2))


def _fuse(circuit: QuantumCircuit, basis: frozenset[str], backend: str):
    from quspect.dag.dag_to_circuit import DAGToCircuitCompiler

    fusion = SingleQubitGateFusion(basis_gates=basis)
    if backend == "python":
        with _native.python_backend():
            dag = fusion.run(circuit.to_dag())
    else:
        dag = fusion.run(circuit.to_dag())
    return DAGToCircuitCompiler().compile(dag)


def _names(circuit: QuantumCircuit) -> list[str]:
    return [i.gate.name for i in circuit.instructions if i.gate is not None]


_NON_BASIS_RUNS = {
    "ry": lambda: QuantumCircuit().ry(0.7, 0),
    "rx": lambda: QuantumCircuit().rx(0.7, 0),
    "y": lambda: QuantumCircuit().y(0),
    "u3": lambda: QuantumCircuit().u3(0.4, 1.1, -0.3, 0),
    "ry-rz": lambda: QuantumCircuit().ry(0.7, 0).rz(0.2, 0),
    "h-ry": lambda: QuantumCircuit().h(0).ry(0.7, 0),
    "ry-ry": lambda: QuantumCircuit().ry(0.7, 0).ry(-0.2, 0),
}


class TestSingleQubitGateFusionBasis:
    """Basis-aware resynthesis keeps the unitary of every run."""

    @pytest.mark.parametrize("backend", _BACKENDS)
    @pytest.mark.parametrize("basis", _BASES[:2], ids=["h-rz", "rz-sx-x"])
    @pytest.mark.parametrize("name", sorted(_NON_BASIS_RUNS))
    def test_run_with_gates_outside_the_basis_stays_equivalent(
        self, name: str, basis: frozenset[str], backend: str
    ) -> None:
        """A non-basis gate is costed by its decomposition, so the best
        replacement can have more gates than the run has nodes; it used to be
        written only as far as the nodes went (``ry(0.7)`` became ``sx``)."""
        circuit = _NON_BASIS_RUNS[name]()
        fused = _fuse(circuit, basis, backend)
        assert _phase_distance(_unitary(fused, 1), _unitary(circuit, 1)) < 1e-9
        assert len(fused) <= len(circuit)
        # each gate is either in the basis or an untouched gate of the input
        kept = [i.gate for i in circuit.instructions]
        assert all(i.gate.name in basis or i.gate in kept for i in fused.instructions)

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_identity_gate_next_to_basis_gates(self, backend: str) -> None:
        """An ``id`` outside the basis has no decomposition and makes its run
        arbitrarily expensive: the run is resynthesized whenever that fits."""
        ry_rz = frozenset({"ry", "rz", "cx"})
        zsx = _BASES[1]
        cases = [
            (QuantumCircuit().id(0).ry(-2.2228, 0), ry_rz),  # ZYZ needs 3 gates
            (QuantumCircuit().id(0).h(0), zsx),  # rz·sx·rz needs 3 gates
            (QuantumCircuit().h(0).id(0), zsx),
            (QuantumCircuit().id(0).sx(0).rz(0.3, 0).sx(0), zsx),  # fits: id dropped
        ]
        for circuit, basis in cases:
            fused = _fuse(circuit, basis, backend)
            assert _phase_distance(_unitary(fused, 1), _unitary(circuit, 1)) < 1e-9
            assert len(fused) <= len(circuit)
        assert _names(_fuse(cases[3][0], zsx, backend)) == ["sx", "rz", "sx"]

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_optimize_pass_with_gates_outside_the_basis(self, backend: str) -> None:
        from quspect.transpiler import CompilationTarget
        from quspect.transpiler._passes import OptimizePass

        for basis in _BASES[:2]:
            target = CompilationTarget.from_coupling_map(
                2, [(0, 1)], native_gates=set(basis) | {"cz"}
            )
            passes = [
                OptimizePass(basis_gates=basis | {"cz"}),
                OptimizePass.from_target(target),
            ]
            for make in _NON_BASIS_RUNS.values():
                circuit = make().cz(0, 1).compose(make(), qubits=[0])
                want = _unitary(circuit, 2)
                for optimize in passes:
                    if backend == "python":
                        with _native.python_backend():
                            out = optimize.run(circuit.copy())
                    else:
                        out = optimize.run(circuit.copy())
                    assert _phase_distance(_unitary(out, 2), want) < 1e-9

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_replacement_that_fits_the_run_is_still_written(self, backend: str) -> None:
        h_rz = _BASES[0]
        # one non-basis gate, one basis gate
        fused = _fuse(QuantumCircuit().s(0), h_rz, backend)
        assert _names(fused) == ["rz"]
        assert fused[0].gate.params[0] == pytest.approx(np.pi / 2)
        # a long run with gates outside the basis: at most rz·h·rz·h·rz
        circuit = QuantumCircuit().ry(0.7, 0).rz(0.2, 0).h(0).t(0).rx(0.3, 0).s(0)
        fused = _fuse(circuit, h_rz, backend)
        assert len(fused) <= 5 and set(_names(fused)) <= h_rz
        assert _phase_distance(_unitary(fused, 1), _unitary(circuit, 1)) < 1e-9
        # all gates in the basis: unchanged behaviour
        circuit = QuantumCircuit().t(0).h(0).t(0).h(0).t(0).h(0).t(0)
        fused = _fuse(circuit, h_rz, backend)
        assert len(fused) == 5 and set(_names(fused)) <= h_rz
        assert _phase_distance(_unitary(fused, 1), _unitary(circuit, 1)) < 1e-9
        # identity runs are removed
        assert len(_fuse(QuantumCircuit().ry(0.7, 0).ry(-0.7, 0), h_rz, backend)) == 0

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_random_runs_match_the_numpy_reference(self, backend: str) -> None:
        """Property check: random circuits of single-qubit runs (gates inside and
        outside the basis) between CZ gates, for several bases."""
        rng = random.Random(20261109)
        one_qubit = {
            "h": 0, "x": 0, "y": 0, "z": 0, "s": 0, "sdg": 0, "t": 0, "tdg": 0,
            "rx": 1, "ry": 1, "rz": 1, "p": 1, "u1": 1, "u2": 2, "u3": 3,
        }  # fmt: skip
        angles = [0.0, np.pi, -np.pi / 2, np.pi / 4, 0.7, -2.1, 1e-3]
        for case in range(150):
            basis = _BASES[case % len(_BASES)]
            n = rng.randint(1, 3)
            circuit = QuantumCircuit()
            for q in range(n):
                circuit.id(q)
            for _ in range(rng.randint(1, 14)):
                if n > 1 and rng.random() < 0.2:
                    circuit.cz(*rng.sample(range(n), 2))
                    continue
                name = rng.choice(sorted(one_qubit))
                params = [
                    rng.choice(angles) if rng.random() < 0.5 else rng.uniform(-4, 4)
                    for _ in range(one_qubit[name])
                ]
                getattr(circuit, name)(*params, rng.randrange(n))
            fused = _fuse(circuit, basis, backend)
            assert _phase_distance(_unitary(fused, n), _unitary(circuit, n)) < 1e-8
            assert len(fused) <= len(circuit)
            kept = [i.gate for i in circuit.instructions]
            assert all(
                i.gate.name in basis or i.gate in kept for i in fused.instructions
            )
            if set(_names(circuit)) <= basis | {"id", "cz"}:
                assert set(_names(fused)) <= basis | {"id", "cz"}

    @pytest.mark.skipif(not _native.enabled(), reason="native DAG passes not built")
    def test_native_and_python_agree_on_runs_outside_the_basis(self) -> None:
        for basis in _BASES:
            for make in _NON_BASIS_RUNS.values():
                circuit = make()
                native = _fuse(circuit, basis, "native")
                python = _fuse(circuit, basis, "python")
                assert [repr(i) for i in native] == [repr(i) for i in python]
