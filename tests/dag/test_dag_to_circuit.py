"""Tests for DAGToCircuitCompiler — DAGCircuit → QuantumCircuit reconstruction."""

from __future__ import annotations

from quspect import QuantumCircuit
from quspect.dag.dag_to_circuit import DAGToCircuitCompiler


class TestDAGToCircuitBasic:
    """Basic reconstruction tests."""

    def test_empty_circuit(self) -> None:
        """Reconstruct from an empty circuit's DAG."""
        circ = QuantumCircuit(name="empty")
        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)
        assert len(reconstructed) == 0
        assert reconstructed.num_qubits == 0

    def test_single_gate(self) -> None:
        """Reconstruct a circuit with a single gate."""
        circ = QuantumCircuit()
        circ.h(0)
        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)
        assert len(reconstructed) == 1
        assert reconstructed[0].is_gate
        assert reconstructed[0].gate.name == "h"
        assert reconstructed[0].qubits == (0,)

    def test_bell_state(self) -> None:
        """Reconstruct a Bell state circuit."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag, name="bell_opt")
        assert len(reconstructed) == 2
        assert reconstructed[0].gate.name == "h"
        assert reconstructed[0].qubits == (0,)
        assert reconstructed[1].gate.name == "cx"
        assert reconstructed[1].qubits == (0, 1)

    def test_with_measurements(self) -> None:
        """Reconstruct a circuit with measurements."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).measure([0, 1], [0, 1])
        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)
        assert len(reconstructed) == 3
        assert reconstructed[2].is_measure
        assert reconstructed[2].qubits == (0, 1)
        assert reconstructed[2].clbits == (0, 1)

    def test_with_barriers(self) -> None:
        """Reconstruct a circuit with barriers."""
        circ = QuantumCircuit()
        circ.h(0).barrier(0, 1).cx(0, 1)
        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)
        assert len(reconstructed) == 3
        assert reconstructed[1].is_barrier
        assert reconstructed[1].qubits == (0, 1)


class TestDAGToCircuitRoundtrip:
    """Full roundtrip: Circuit → DAG → Circuit."""

    def test_roundtrip_parameterized(self) -> None:
        """Roundtrip a parameterized circuit."""
        from quspect import Parameter

        theta = Parameter("θ")
        circ = QuantumCircuit(name="vqe")
        circ.rx(theta, 0).ry(0.5, 1).cx(0, 1)

        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)

        assert reconstructed.num_qubits == circ.num_qubits
        assert len(reconstructed) == len(circ)
        assert theta in reconstructed.parameters

    def test_roundtrip_multi_qubit(self) -> None:
        """Roundtrip a multi-qubit circuit with various gates."""
        circ = QuantumCircuit()
        circ.h(0).x(1).z(2).cx(0, 1).cz(1, 2).swap(0, 3)

        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)

        assert reconstructed.num_qubits == 4
        assert len(reconstructed) == 6
        for i in range(len(circ)):
            assert reconstructed[i].gate.name == circ[i].gate.name

    def test_roundtrip_with_measure_all(self) -> None:
        """Roundtrip a circuit with measure_all."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1).measure_all()

        dag = circ.to_dag()
        reconstructed = DAGToCircuitCompiler.compile(dag)

        assert len(reconstructed) == 3
        assert reconstructed[2].is_measure


class TestDAGToCircuitChannels:
    """Noise channels survive the DAG round trip (B19)."""

    def test_channel_roundtrip(self) -> None:
        """A depolarizing channel is kept, in order, with its operands."""
        circ = QuantumCircuit().h(0).depolarize(0.1, 0).cx(0, 1)
        back = DAGToCircuitCompiler.compile(circ.to_dag())
        assert [i.type for i in back.instructions] == [
            i.type for i in circ.instructions
        ]
        channel_instr = back.instructions[1]
        assert channel_instr.is_channel
        assert channel_instr.channel is circ.instructions[1].channel
        assert channel_instr.qubits == (0,)
