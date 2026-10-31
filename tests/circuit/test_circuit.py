"""Tests for QuantumCircuit."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.parameterized import RX, U2
from quspect.gates.standard import CCX, CX, H, X
from quspect.visualization import SvgStyle


class TestQuantumCircuitConstruction:
    """Test circuit creation and basic properties."""

    def test_create_empty_circuit(self) -> None:
        """Test creating an empty circuit."""
        circ = QuantumCircuit()
        assert circ.num_qubits == 0
        assert circ.num_clbits == 0
        assert len(circ) == 0
        assert circ.name == ""

    def test_create_named_circuit(self) -> None:
        """Test creating a named circuit."""
        circ = QuantumCircuit(name="bell")
        assert circ.name == "bell"

    def test_set_name(self) -> None:
        """Test setting circuit name."""
        circ = QuantumCircuit()
        circ.name = "test"
        assert circ.name == "test"


class TestQuantumCircuitGateMethods:
    """Test fluent gate-adding methods."""

    def test_add_single_qubit_gate(self) -> None:
        """Test adding a single-qubit gate."""
        circ = QuantumCircuit()
        result = circ.h(0)
        assert result is circ  # fluent API
        assert len(circ) == 1
        assert circ.num_qubits == 1

    def test_add_two_qubit_gate(self) -> None:
        """Test adding a two-qubit gate."""
        circ = QuantumCircuit()
        circ.cx(0, 1)
        assert len(circ) == 1
        assert circ.num_qubits == 2

    def test_all_standard_gate_methods(self) -> None:
        """Test that all standard gate methods work."""
        circ = QuantumCircuit()
        circ.id(0)
        circ.h(0)
        circ.x(0)
        circ.y(0)
        circ.z(0)
        circ.s(0)
        circ.sdg(0)
        circ.t(0)
        circ.tdg(0)
        assert len(circ) == 9

    def test_cx(self) -> None:
        """Test CNOT gate."""
        circ = QuantumCircuit()
        circ.cx(0, 1)
        assert circ.num_qubits == 2

    def test_cz(self) -> None:
        """Test CZ gate."""
        circ = QuantumCircuit()
        circ.cz(0, 1)
        assert len(circ) == 1

    def test_swap(self) -> None:
        """Test SWAP gate."""
        circ = QuantumCircuit()
        circ.swap(0, 1)
        assert circ.num_qubits == 2

    def test_ccx(self) -> None:
        """Test Toffoli gate."""
        circ = QuantumCircuit()
        circ.ccx(0, 1, 2)
        assert circ.num_qubits == 3

    def test_method_chaining(self) -> None:
        """Test that methods can be chained."""
        circ = QuantumCircuit(name="bell").h(0).cx(0, 1)
        assert circ.name == "bell"
        assert len(circ) == 2
        assert circ.num_qubits == 2


class TestQuantumCircuitParameterizedGates:
    """Test parameterized gate methods."""

    def test_rx_with_float(self) -> None:
        """Test RX gate with concrete float."""
        circ = QuantumCircuit()
        circ.rx(0.5, 0)
        assert len(circ) == 1

    def test_rx_with_parameter(self) -> None:
        """Test RX gate with symbolic Parameter."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0)
        assert len(circ) == 1
        assert theta in circ.parameters

    def test_reversed_parameter_and_qubit_rejected_immediately(self) -> None:
        """A symbolic value cannot be stored as a qubit index."""
        theta = Parameter("theta")

        with pytest.raises(TypeError, match="Qubit index"):
            QuantumCircuit().ry(1, theta)  # type: ignore[arg-type]

    def test_ry(self) -> None:
        """Test RY gate."""
        circ = QuantumCircuit()
        circ.ry(1.0, 0)
        assert len(circ) == 1

    def test_rz(self) -> None:
        """Test RZ gate."""
        circ = QuantumCircuit()
        circ.rz(0.3, 0)
        assert len(circ) == 1

    def test_p(self) -> None:
        """Test Phase gate."""
        circ = QuantumCircuit()
        circ.p(math.pi / 2, 0)
        assert len(circ) == 1

    def test_u3(self) -> None:
        """Test U3 gate."""
        circ = QuantumCircuit()
        circ.u3(0.5, 0.3, 0.1, 0)
        assert len(circ) == 1

    def test_u3_with_parameters(self) -> None:
        """Test U3 gate with symbolic parameters."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        lam = Parameter("λ")
        circ = QuantumCircuit()
        circ.u3(theta, phi, lam, 0)
        assert circ.parameters == {theta, phi, lam}

    def test_crx(self) -> None:
        """Test CRX gate."""
        circ = QuantumCircuit()
        circ.crx(0.5, 0, 1)
        assert len(circ) == 1

    def test_cry(self) -> None:
        """Test CRY gate."""
        circ = QuantumCircuit()
        circ.cry(1.0, 0, 1)
        assert len(circ) == 1

    def test_crz(self) -> None:
        """Test CRZ gate."""
        circ = QuantumCircuit()
        circ.crz(0.3, 0, 1)
        assert len(circ) == 1


class TestQuantumCircuitDynamicQubitCount:
    """Test dynamic qubit/clbit count inference."""

    def test_num_qubits_max_plus_one(self) -> None:
        """Test num_qubits = max(qubit_index) + 1."""
        circ = QuantumCircuit()
        circ.h(2)  # uses qubit 2
        assert circ.num_qubits == 3  # qubits 0,1,2

    def test_num_qubits_with_gaps(self) -> None:
        """Test that gaps in qubit indices are allowed."""
        circ = QuantumCircuit()
        circ.h(0)
        circ.x(5)
        assert circ.num_qubits == 6

    def test_num_clbits(self) -> None:
        """Test num_clbits inference from measurements."""
        circ = QuantumCircuit()
        circ.h(0)
        circ.measure([0], [3])
        assert circ.num_clbits == 4  # clbits 0,1,2,3

    def test_empty_circuit_num_qubits(self) -> None:
        """Test empty circuit returns 0."""
        assert QuantumCircuit().num_qubits == 0

    def test_empty_circuit_num_clbits(self) -> None:
        """Test empty circuit returns 0."""
        assert QuantumCircuit().num_clbits == 0

    def test_no_measurement_num_clbits(self) -> None:
        """Test circuit with no measurement returns 0 clbits."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        assert circ.num_clbits == 0


class TestQuantumCircuitMeasureAndBarrier:
    """Test measurement and barrier instructions."""

    def test_measure(self) -> None:
        """Test adding a measurement."""
        circ = QuantumCircuit()
        circ.measure([0, 1], [0, 1])
        assert len(circ) == 1
        assert circ[0].is_measure

    def test_measure_all(self) -> None:
        """Test measure_all on active qubits."""
        circ = QuantumCircuit()
        circ.h(0).x(1).z(3)
        circ.measure_all()
        # Should measure qubits 0, 1, 3 to clbits 0, 1, 3
        assert len(circ) == 4  # 3 gates + 1 measure
        last_instr = circ[-1]
        assert last_instr.is_measure
        assert last_instr.qubits == (0, 1, 3)
        assert last_instr.clbits == (0, 1, 3)

    def test_measure_all_empty(self) -> None:
        """Test measure_all on empty circuit does nothing."""
        circ = QuantumCircuit()
        circ.measure_all()
        assert len(circ) == 0

    def test_barrier_specific_qubits(self) -> None:
        """Test barrier on specific qubits."""
        circ = QuantumCircuit()
        circ.barrier(0, 2)
        assert len(circ) == 1
        assert circ[0].is_barrier
        assert circ[0].qubits == (0, 2)

    def test_barrier_all_qubits(self) -> None:
        """Test barrier with no args acts on all active qubits."""
        circ = QuantumCircuit()
        circ.h(0).x(2)
        circ.barrier()
        assert circ[-1].is_barrier
        assert set(circ[-1].qubits) == {0, 2}

    def test_measure_qubits_clbits_mismatch(self) -> None:
        """Test measure with mismatched lengths."""
        circ = QuantumCircuit()
        with pytest.raises(ValueError, match="equal number"):
            circ.measure([0, 1], [0])


class TestQuantumCircuitAppend:
    """Test generic append method."""

    def test_append_gate(self) -> None:
        """Test appending a gate."""
        circ = QuantumCircuit()
        circ.append(H, (0,))
        assert len(circ) == 1
        assert circ[0].qubits == (0,)

    def test_append_two_qubit_gate(self) -> None:
        """Test appending a two-qubit gate."""
        circ = QuantumCircuit()
        circ.append(CX, (0, 1))
        assert circ.num_qubits == 2

    def test_append_mismatched_qubits(self) -> None:
        """Test append with wrong qubit count."""
        circ = QuantumCircuit()
        with pytest.raises(ValueError, match="requires"):
            circ.append(CX, (0,))

    @pytest.mark.parametrize("index", [-1, True, 0.5])
    def test_append_rejects_invalid_qubit_index(self, index: object) -> None:
        """Qubit indices must be non-negative integers."""
        circ = QuantumCircuit()

        with pytest.raises((TypeError, ValueError), match="Qubit index"):
            circ.append(H, (index,))  # type: ignore[arg-type]

    @pytest.mark.parametrize("index", [-1, True, 0.5])
    def test_measure_rejects_invalid_clbit_index(self, index: object) -> None:
        """Classical-bit indices use the same validation rules."""
        circ = QuantumCircuit()

        with pytest.raises((TypeError, ValueError), match="Clbit index"):
            circ.measure((0,), (index,))  # type: ignore[arg-type]


class TestQuantumCircuitActiveBits:
    """Test active_qubits and active_clbits."""

    def test_active_qubits(self) -> None:
        """Test active_qubits property."""
        circ = QuantumCircuit()
        circ.h(0).x(3).cx(0, 1)
        assert circ.active_qubits == {0, 1, 3}

    def test_active_qubits_empty(self) -> None:
        """Test active_qubits on empty circuit."""
        assert QuantumCircuit().active_qubits == set()

    def test_active_clbits(self) -> None:
        """Test active_clbits property."""
        circ = QuantumCircuit()
        circ.measure([0, 1], [2, 3])
        assert circ.active_clbits == {2, 3}

    def test_active_clbits_empty(self) -> None:
        """Test active_clbits on circuit without measurements."""
        circ = QuantumCircuit()
        circ.h(0)
        assert circ.active_clbits == set()


class TestQuantumCircuitIteration:
    """Test iteration and indexing."""

    def test_len(self) -> None:
        """Test __len__."""
        circ = QuantumCircuit()
        circ.h(0).x(1).cx(0, 1)
        assert len(circ) == 3

    def test_iter(self) -> None:
        """Test __iter__."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        instructions = list(circ)
        assert len(instructions) == 2
        assert all(isinstance(i, CircuitInstruction) for i in instructions)

    def test_getitem(self) -> None:
        """Test __getitem__."""
        circ = QuantumCircuit()
        circ.h(0).x(1)
        assert circ[0].gate is H
        assert circ[1].gate is X

    def test_getitem_index_error(self) -> None:
        """Test __getitem__ raises IndexError."""
        circ = QuantumCircuit()
        with pytest.raises(IndexError):
            circ[0]

    def test_instructions_tuple(self) -> None:
        """Test instructions property returns tuple."""
        circ = QuantumCircuit()
        circ.h(0)
        assert isinstance(circ.instructions, tuple)
        assert len(circ.instructions) == 1


class TestQuantumCircuitParameters:
    """Test parameter tracking."""

    def test_parameters_empty(self) -> None:
        """Test parameters on circuit with no parameterized gates."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        assert circ.parameters == set()

    def test_parameters_single(self) -> None:
        """Test parameters tracking."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0)
        assert circ.parameters == {theta}

    def test_parameters_multiple(self) -> None:
        """Test multiple parameters."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(phi, 1)
        assert circ.parameters == {theta, phi}

    def test_parameters_deduplicated(self) -> None:
        """Test same parameter used multiple times."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).rx(theta, 1)
        assert circ.parameters == {theta}


class TestQuantumCircuitBindParameters:
    """Test parameter binding."""

    def test_bind_parameters(self) -> None:
        """Test binding parameters to values."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(theta, 1)
        bound = circ.bind_parameters({theta: 0.5})
        assert bound.parameters == set()

    def test_bind_multiple_parameters(self) -> None:
        """Test binding multiple parameters."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(phi, 1)
        bound = circ.bind_parameters({theta: 1.0, phi: 2.0})
        assert bound.parameters == set()
        assert len(bound) == 2

    def test_bind_preserves_original(self) -> None:
        """Test bind_parameters returns new circuit, not modifying original."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0)
        bound = circ.bind_parameters({theta: 0.5})
        assert circ.parameters == {theta}
        assert bound.parameters == set()

    def test_bind_unbound_parameter_raises(self) -> None:
        """Test that unbound parameters raise ValueError."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(phi, 1)
        with pytest.raises(ValueError, match="not bound"):
            circ.bind_parameters({theta: 0.5})

    def test_bind_to_concrete_can_compute_matrix(self) -> None:
        """Test that bound circuit gates can compute matrix."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0)
        bound = circ.bind_parameters({theta: math.pi})
        # bound circuit's gate should be concrete
        assert bound[0].is_gate
        assert bound[0].gate is not None
        m = gate_matrix(bound[0].gate)
        assert m.shape == (2, 2)


class TestQuantumCircuitCompose:
    """Test circuit composition."""

    def test_compose_basic(self) -> None:
        """Test basic composition."""
        sub = QuantumCircuit()
        sub.h(0).cx(0, 1)

        main = QuantumCircuit()
        main = main.compose(sub, qubits=[0, 1])
        assert len(main) == 2
        assert main.num_qubits == 2

    def test_compose_with_qubit_remap(self) -> None:
        """Test composing with qubit remapping."""
        sub = QuantumCircuit()
        sub.h(0).cx(0, 1)  # sub.num_qubits = 2

        main = QuantumCircuit()
        main = main.compose(sub, qubits=[2, 3])
        assert main.num_qubits == 4
        assert main[0].qubits == (2,)  # H remapped to qubit 2
        assert main[1].qubits == (2, 3)  # CX remapped

    def test_compose_wrong_qubit_count(self) -> None:
        """Test compose raises on wrong qubit count."""
        sub = QuantumCircuit()
        sub.h(0).cx(0, 1)  # sub.num_qubits = 2

        main = QuantumCircuit()
        with pytest.raises(ValueError, match="compose"):
            main.compose(sub, qubits=[0])  # need 2, gave 1

    def test_compose_inplace(self) -> None:
        """Test compose with inplace=True."""
        sub = QuantumCircuit()
        sub.h(0)

        main = QuantumCircuit(name="original")
        result = main.compose(sub, qubits=[0], inplace=True)
        assert result is main
        assert len(main) == 1

    def test_compose_not_inplace(self) -> None:
        """Test compose with inplace=False (default)."""
        sub = QuantumCircuit()
        sub.h(0)

        main = QuantumCircuit()
        result = main.compose(sub, qubits=[0], inplace=False)
        assert result is not main
        assert len(main) == 0
        assert len(result) == 1

    def test_compose_multiple(self) -> None:
        """Test composing multiple sub-circuits."""
        bell = QuantumCircuit()
        bell.h(0).cx(0, 1)

        main = QuantumCircuit()
        main = main.compose(bell, qubits=[0, 1])
        main = main.compose(bell, qubits=[2, 3])
        assert main.num_qubits == 4
        assert len(main) == 4

    def test_compose_with_clbit_remap(self) -> None:
        """Test compose with explicit clbit mapping."""
        sub = QuantumCircuit()
        sub.h(0).measure([0], [0])

        main = QuantumCircuit()
        main = main.compose(sub, qubits=[1], clbits=[5])
        assert main[1].clbits == (5,)


class TestQuantumCircuitCopy:
    """Test circuit copying."""

    def test_copy_deep(self) -> None:
        """Test copy creates independent circuit."""
        circ = QuantumCircuit(name="original")
        circ.h(0)
        copy = circ.copy()
        assert copy.name == "original"
        assert len(copy) == 1
        # Modify original, copy unaffected
        circ.x(1)
        assert len(circ) == 2
        assert len(copy) == 1

    def test_copy_with_new_name(self) -> None:
        """Test copy with a new name."""
        circ = QuantumCircuit(name="original")
        copy = circ.copy(name="duplicate")
        assert copy.name == "duplicate"
        assert circ.name == "original"


class TestQuantumCircuitJSON:
    """Test JSON serialization round-trip."""

    def test_roundtrip_empty(self) -> None:
        """Test JSON round-trip of empty circuit."""
        circ = QuantumCircuit(name="empty")
        data = circ.to_json()
        restored = QuantumCircuit.from_json(data)
        assert restored.name == "empty"
        assert len(restored) == 0

    def test_roundtrip_standard_gates(self) -> None:
        """Test JSON round-trip with standard gates."""
        circ = QuantumCircuit(name="test")
        circ.h(0).cx(0, 1).x(1)
        data = circ.to_json()
        restored = QuantumCircuit.from_json(data)
        assert restored.name == "test"
        assert len(restored) == 3
        assert restored[0].gate is not None
        assert restored[0].gate.name == "h"
        assert restored[1].gate is not None
        assert restored[1].gate.name == "cx"

    def test_roundtrip_parameterized_gates(self) -> None:
        """Test JSON round-trip with parameterized gates."""
        theta = Parameter("θ")
        circ = QuantumCircuit()
        circ.rx(theta, 0).ry(1.5, 1)
        data = circ.to_json()
        restored = QuantumCircuit.from_json(data)
        assert len(restored) == 2
        assert restored.parameters == {theta}

    def test_roundtrip_measurement(self) -> None:
        """Test JSON round-trip with measurement."""
        circ = QuantumCircuit()
        circ.h(0).measure([0], [0])
        data = circ.to_json()
        restored = QuantumCircuit.from_json(data)
        assert len(restored) == 2
        assert restored[1].is_measure

    def test_roundtrip_barrier(self) -> None:
        """Test JSON round-trip with barrier."""
        circ = QuantumCircuit()
        circ.h(0).barrier(0, 1)
        data = circ.to_json()
        restored = QuantumCircuit.from_json(data)
        assert restored[1].is_barrier

    def test_from_json_unknown_gate(self) -> None:
        """Test from_json with unknown gate name raises."""
        data = json.dumps(
            {
                "name": "bad",
                "instructions": [
                    {
                        "type": "GATE",
                        "qubits": [0],
                        "clbits": [],
                        "gate": {
                            "name": "nonexistent",
                            "num_qubits": 1,
                            "gate_type": "standard",
                        },
                    }
                ],
            }
        )
        with pytest.raises(ValueError, match="Unknown standard gate"):
            QuantumCircuit.from_json(data)

    def test_to_json_is_valid_json(self) -> None:
        """Test to_json returns valid JSON."""
        circ = QuantumCircuit()
        circ.h(0)
        data = circ.to_json()
        parsed = json.loads(data)
        assert isinstance(parsed, dict)
        assert parsed["name"] == ""


class TestQuantumCircuitToDAG:
    """Test DAG compilation stub."""

    def test_to_dag_returns_dag(self) -> None:
        """Test that to_dag returns a DAGCircuit."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        dag = circ.to_dag()
        assert dag.num_qubits == 2
        assert dag.num_nodes() == 6  # 2 input + 2 ops + 2 output


class TestQuantumCircuitRepr:
    """Test string representation."""

    def test_repr_empty(self) -> None:
        """Test repr of empty circuit."""
        r = repr(QuantumCircuit())
        assert "QuantumCircuit" in r

    def test_repr_named(self) -> None:
        """Test repr includes name."""
        r = repr(QuantumCircuit(name="test"))
        assert "test" in r

    def test_repr_shows_counts(self) -> None:
        """Test repr shows qubit/clbit counts."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        r = repr(circ)
        assert "2" in r  # num_qubits


class TestQuantumCircuitDraw:
    """Test text-based circuit drawing."""

    def test_draw_empty(self) -> None:
        """Test draw of empty circuit."""
        assert "(empty circuit)" in QuantumCircuit().draw()

    def test_draw_non_empty(self) -> None:
        """Test draw produces multi-line string."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        result = circ.draw()
        assert "q0:" in result
        assert "q1:" in result
        assert "[H]" in result
        assert "o" in result
        assert "[X]" in result

    def test_draw_shows_vertical_wire_for_non_adjacent_gate(self) -> None:
        """Test draw shows a vertical wire between non-adjacent qubits."""
        circ = QuantumCircuit()
        circ.cx(0, 2)
        result = circ.draw().splitlines()
        assert "|" in result[1]

    def test_draw_includes_classical_lanes_for_measurement(self) -> None:
        """Test draw renders classical bit lanes for measurements."""
        circ = QuantumCircuit()
        circ.h(0).measure([0, 1], [0, 1])
        result = circ.draw()
        assert "c0:" in result
        assert "c1:" in result
        assert "[M]" in result

    def test_draw_renders_barrier(self) -> None:
        """Test draw renders barrier markers on covered qubits."""
        circ = QuantumCircuit()
        circ.h(0).barrier(0, 1).x(1)
        result = circ.draw()
        assert "|" in result

    def test_draw_parameterized_gate_label(self) -> None:
        """Test draw includes formatted parameterized gate labels."""
        theta = Parameter("theta")
        result = QuantumCircuit().rx(theta, 0).draw()
        assert "[RX(theta)]" in result

    def test_draw_swap_renders_cross_marks(self) -> None:
        """Test draw uses cross marks for swap gates."""
        lines = QuantumCircuit().swap(0, 1).draw().splitlines()
        assert "x" in lines[0]
        assert "x" in lines[1]

    def test_draw_multi_control_gate_shows_all_controls(self) -> None:
        """Test draw renders multiple control markers for CCX."""
        result = QuantumCircuit().ccx(0, 1, 2).draw()
        assert result.count("o") == 2
        assert "[X]" in result


class TestQuantumCircuitSvg:
    """Test SVG-based circuit drawing."""

    def test_svg_non_empty(self) -> None:
        """Test SVG output contains H and the standard CNOT target icon."""
        circ = QuantumCircuit()
        circ.h(0).cx(0, 1)
        svg = circ.svg()
        svg_text = str(svg)
        root = ET.fromstring(svg_text)
        circles = [item for item in root.iter() if item.tag.endswith("circle")]

        assert svg_text.startswith("<svg")
        assert "</svg>" in svg_text
        assert ">H<" in svg_text
        assert sum(item.attrib.get("r") == "5.5" for item in circles) == 1
        assert sum(item.attrib.get("r") == "15.0" for item in circles) == 1

    def test_svg_measurement_includes_classical_lines(self) -> None:
        """Test measurement rendering includes its icon and classical connector."""
        circ = QuantumCircuit()
        circ.h(0).measure([0], [1])
        svg_text = str(circ.svg())
        root = ET.fromstring(svg_text)
        paths = [item for item in root.iter() if item.tag.endswith("path")]
        lines = [item for item in root.iter() if item.tag.endswith("line")]
        circles = [item for item in root.iter() if item.tag.endswith("circle")]

        assert "c1:" in svg_text
        assert any(item.attrib.get("stroke-dasharray") == "5 4" for item in lines)
        assert any("A 10 10 0 0 1" in item.attrib.get("d", "") for item in paths)
        assert sum(item.attrib.get("r") == "2.2" for item in circles) == 1

    def test_svg_empty(self) -> None:
        """Test empty circuit SVG rendering."""
        svg_text = str(QuantumCircuit().svg())
        assert "(empty circuit)" in svg_text

    def test_svg_repr_method(self) -> None:
        """Test SVG wrapper exposes notebook representation."""
        svg = QuantumCircuit().h(0).svg()
        assert svg._repr_svg_() == str(svg)

    def test_svg_parameterized_gate_label(self) -> None:
        """Test SVG output keeps formatted parameter labels."""
        theta = Parameter("theta")
        svg_text = str(QuantumCircuit().rx(theta, 0).svg())
        assert "RX(theta)" in svg_text

    def test_svg_swap_renders_cross_lines(self) -> None:
        """Test SVG output uses line elements for swap marks."""
        svg_text = str(QuantumCircuit().swap(0, 1).svg())
        assert svg_text.count("<line") >= 3

    def test_svg_multi_control_gate_renders_multiple_controls(self) -> None:
        """Test SVG output contains one control circle per control qubit."""
        root = ET.fromstring(str(QuantumCircuit().ccx(0, 1, 2).svg()))
        circles = [item for item in root.iter() if item.tag.endswith("circle")]

        assert sum(item.attrib.get("r") == "5.5" for item in circles) == 2
        assert sum(item.attrib.get("r") == "15.0" for item in circles) == 1

    def test_svg_accepts_style_object(self) -> None:
        """Test svg accepts a strongly typed style object."""
        style = SvgStyle(
            background="#101010",
            wire="#aaaaaa",
            classical_wire="#bbbbbb",
            gate_fill="#222222",
            gate_stroke="#cccccc",
            text="#dddddd",
            control="#eeeeee",
            barrier="#444444",
            measure_fill="#333333",
            channel_fill="#3a1111",
            channel_stroke="#cc4444",
        )
        svg_text = str(QuantumCircuit().h(0).svg(style=style))
        assert "#101010" in svg_text
        assert "#222222" in svg_text

    def test_svg_invalid_style_mapping_raises(self) -> None:
        """Test svg rejects incomplete custom style mappings."""
        with pytest.raises(ValueError, match="Missing SVG style keys"):
            QuantumCircuit().h(0).svg(style={"background": "#fff"})


class TestQuantumCircuitDagger:
    """Tests for the dagger() method."""

    # -- basic properties --

    def test_dagger_empty_circuit(self) -> None:
        """Dagger of empty circuit is empty."""
        circ = QuantumCircuit()
        dag = circ.dagger()
        assert len(dag) == 0

    def test_dagger_preserves_name(self) -> None:
        """Dagger appends '†' to the circuit name."""
        circ = QuantumCircuit(name="bell")
        dag = circ.dagger()
        assert dag.name == "bell†"

    def test_dagger_unnamed(self) -> None:
        """Dagger of unnamed circuit gets name 'dagger'."""
        dag = QuantumCircuit().dagger()
        assert dag.name == "dagger"

    # -- gate order reversal --

    def test_dagger_reverses_gate_order(self) -> None:
        """Dagger reverses the order of gates."""
        circ = QuantumCircuit().h(0).z(0).x(0)
        dag = circ.dagger()
        assert dag[0].gate.name == "x"
        assert dag[1].gate.name == "z"
        assert dag[2].gate.name == "h"

    def test_dagger_self_inverse(self) -> None:
        """Dagger of dagger returns the original circuit."""
        circ = QuantumCircuit().h(0).cx(0, 1).t(0).s(1)
        dag2 = circ.dagger().dagger()
        assert len(dag2) == len(circ)
        for a, b in zip(circ.instructions, dag2.instructions, strict=True):
            assert a.gate.name == b.gate.name
            assert a.gate.params == b.gate.params
            assert a.qubits == b.qubits

    # -- self-adjoint gates --

    def test_dagger_h_unchanged(self) -> None:
        """H† = H."""
        dag = QuantumCircuit().h(0).dagger()
        assert dag[0].gate.name == "h"

    def test_dagger_x_unchanged(self) -> None:
        """X† = X."""
        dag = QuantumCircuit().x(0).dagger()
        assert dag[0].gate.name == "x"

    def test_dagger_y_unchanged(self) -> None:
        """Y† = Y."""
        dag = QuantumCircuit().y(0).dagger()
        assert dag[0].gate.name == "y"

    def test_dagger_z_unchanged(self) -> None:
        """Z† = Z."""
        dag = QuantumCircuit().z(0).dagger()
        assert dag[0].gate.name == "z"

    def test_dagger_cx_unchanged(self) -> None:
        """CX† = CX."""
        dag = QuantumCircuit().cx(0, 1).dagger()
        assert dag[0].gate.name == "cx"

    def test_dagger_cz_unchanged(self) -> None:
        """CZ† = CZ."""
        dag = QuantumCircuit().cz(0, 1).dagger()
        assert dag[0].gate.name == "cz"

    def test_dagger_swap_unchanged(self) -> None:
        """SWAP† = SWAP."""
        dag = QuantumCircuit().swap(0, 1).dagger()
        assert dag[0].gate.name == "swap"

    def test_dagger_ccx_unchanged(self) -> None:
        """CCX† = CCX."""
        dag = QuantumCircuit().ccx(0, 1, 2).dagger()
        assert dag[0].gate.name == "ccx"

    # -- S / T interchange --

    def test_dagger_s_to_sdg(self) -> None:
        """S† = SDG."""
        dag = QuantumCircuit().s(0).dagger()
        assert dag[0].gate.name == "sdg"

    def test_dagger_sdg_to_s(self) -> None:
        """SDG† = S."""
        dag = QuantumCircuit().sdg(0).dagger()
        assert dag[0].gate.name == "s"

    def test_dagger_t_to_tdg(self) -> None:
        """T† = TDG."""
        dag = QuantumCircuit().t(0).dagger()
        assert dag[0].gate.name == "tdg"

    def test_dagger_tdg_to_t(self) -> None:
        """TDG† = T."""
        dag = QuantumCircuit().tdg(0).dagger()
        assert dag[0].gate.name == "t"

    # -- parameterized gates: sign flip --

    def test_dagger_rx_flips_theta(self) -> None:
        """RX(θ)† = RX(-θ)."""
        dag = QuantumCircuit().rx(0.7, 0).dagger()
        assert dag[0].gate.name == "rx"
        assert dag[0].gate.params == (-0.7,)

    def test_dagger_ry_flips_theta(self) -> None:
        """RY(θ)† = RY(-θ)."""
        dag = QuantumCircuit().ry(1.2, 0).dagger()
        assert dag[0].gate.name == "ry"
        assert dag[0].gate.params == (-1.2,)

    def test_dagger_rz_flips_theta(self) -> None:
        """RZ(θ)† = RZ(-θ)."""
        dag = QuantumCircuit().rz(0.3, 0).dagger()
        assert dag[0].gate.name == "rz"
        assert dag[0].gate.params == (-0.3,)

    def test_dagger_p_flips_theta(self) -> None:
        """P(θ)† = P(-θ)."""
        dag = QuantumCircuit().p(math.pi / 2, 0).dagger()
        assert dag[0].gate.name == "p"
        assert dag[0].gate.params == (-math.pi / 2,)

    def test_dagger_u3_flips_all_params(self) -> None:
        """U3(θ, φ, λ)† = U3(-θ, -λ, -φ)."""
        dag = QuantumCircuit().u3(0.5, 0.3, 0.7, 0).dagger()
        assert dag[0].gate.name == "u3"
        assert dag[0].gate.params == (-0.5, -0.7, -0.3)  # (-θ, -λ, -φ)

    def test_dagger_u2_to_u3(self) -> None:
        """U2(φ, λ)† = U3(-π/2, -λ, -φ)."""
        circ = QuantumCircuit()
        circ.append(U2.with_params(0.5, 1.2), (0,))
        dag = circ.dagger()
        assert dag[0].gate.name == "u3"
        assert dag[0].gate.params == (-math.pi / 2, -1.2, -0.5)

    def test_dagger_crx_flips_theta(self) -> None:
        """CRX(θ)† = CRX(-θ)."""
        dag = QuantumCircuit().crx(0.5, 0, 1).dagger()
        assert dag[0].gate.name == "crx"
        assert dag[0].gate.params == (-0.5,)

    def test_dagger_cry_flips_theta(self) -> None:
        """CRY(θ)† = CRY(-θ)."""
        dag = QuantumCircuit().cry(0.5, 0, 1).dagger()
        assert dag[0].gate.name == "cry"
        assert dag[0].gate.params == (-0.5,)

    def test_dagger_crz_flips_theta(self) -> None:
        """CRZ(θ)† = CRZ(-θ)."""
        dag = QuantumCircuit().crz(0.5, 0, 1).dagger()
        assert dag[0].gate.name == "crz"
        assert dag[0].gate.params == (-0.5,)

    # -- barriers, measurements, channels --

    def test_dagger_barrier_preserved(self) -> None:
        """Barriers are preserved in dagger."""
        dag = QuantumCircuit().h(0).barrier(0, 1).x(0).dagger()
        assert dag[1].is_barrier
        assert dag[1].qubits == (0, 1)

    def test_dagger_measurement_preserved(self) -> None:
        """Measurements are preserved in dagger."""
        dag = QuantumCircuit().h(0).measure([0], [0]).dagger()
        assert dag[0].is_measure
        assert dag[0].qubits == (0,)
        assert dag[0].clbits == (0,)

    def test_dagger_channel_preserved(self) -> None:
        """Noise channels are preserved in dagger."""
        circ = QuantumCircuit()
        circ.h(0).depolarize(0.1, 0).x(0)
        dag = circ.dagger()
        assert dag[1].is_channel

    # -- matrix verification via simulator --

    def test_dagger_matrix_is_conjugate_transpose(self) -> None:
        """The dagger circuit's unitary equals U† for each gate."""
        from quspect.simulation import Simulator

        circ = QuantumCircuit().z(0).ry(math.pi / 2, 0)
        dag = circ.dagger()

        sim = Simulator(seed=42)
        sv_orig = sim.statevector(circ)  # U|0⟩
        sv_dag_ket = sim.statevector(dag)  # U†|0⟩

        # ⟨0|U is the bra of sv_orig: we verify ⟨0|U†U|0⟩ = 1
        inner = np.dot(sv_orig.conj(), sv_orig)
        assert abs(inner - 1.0) < 1e-10

        # U = RY(π/2) · Z, so U†|0⟩ is the first column of U†.
        from quspect.gates import gate_matrix
        from quspect.gates.parameterized import RY
        from quspect.gates.standard import Z

        unitary = gate_matrix(RY.with_params(math.pi / 2)) @ gate_matrix(Z)
        np.testing.assert_allclose(sv_dag_ket, unitary.conj().T[:, 0], atol=1e-10)

    def test_dagger_unitary_property(self) -> None:
        """U† U |0⟩ = |0⟩ for a nontrivial circuit."""
        from quspect.simulation import Simulator

        circ = QuantumCircuit()
        circ.ry(0.8, 0).rz(1.2, 0).x(0)
        dag = circ.dagger()

        sim = Simulator(seed=42)
        u0 = sim.statevector(circ)  # U|0⟩

        # Apply U† via the dagger circuit
        udag = sim.statevector(dag)  # U†|0⟩
        assert abs(np.vdot(udag, udag) - 1.0) < 1e-10

        # Overlap: ⟨0|U†U|0⟩ = ⟨U|U⟩ = 1
        inner = np.vdot(u0, u0)
        assert abs(inner - 1.0) < 1e-10

        # Running U followed by U† returns |0⟩ (up to global phase).
        roundtrip = sim.statevector(circ.compose(dag, [0]))
        assert abs(abs(roundtrip[0]) - 1.0) < 1e-10


class TestQuantumCircuitParameterExpressions:
    """QuantumCircuit should preserve and bind affine gate parameters."""

    def test_parameters_include_every_expression_term(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        circ = QuantumCircuit().rx(2 * theta - 0.5 * phi + 1, 0)
        assert circ.parameters == {theta, phi}

    def test_bind_expression(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        circ = QuantumCircuit().rx(2 * theta - phi + 0.25, 0)
        bound = circ.bind_parameters({theta: 0.5, phi: 0.75})
        assert bound.parameters == set()
        assert bound[0].gate.params == (pytest.approx(0.5),)

    def test_bind_expression_requires_complete_binding(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        circ = QuantumCircuit().rx(theta + phi, 0)
        with pytest.raises(ValueError, match="phi"):
            circ.bind_parameters({theta: 0.5})

    def test_symbolic_dagger_negates_affine_expression(self) -> None:
        theta = Parameter("theta")
        expr = 2 * theta + 0.25
        dag_param = QuantumCircuit().rx(expr, 0).dagger()[0].gate.params[0]
        assert isinstance(dag_param, ParameterExpression)
        assert dag_param == -expr

    def test_symbolic_u3_dagger_negates_and_swaps(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        lam = Parameter("lambda")
        dag = QuantumCircuit().u3(2 * theta, phi + 1, lam - 2, 0).dagger()
        assert dag[0].gate.params == (-2 * theta, -(lam - 2), -(phi + 1))

    def test_symbolic_double_dagger_is_identity(self) -> None:
        theta = Parameter("theta")
        original = QuantumCircuit().rx(3 * theta - 0.5, 0)
        restored = original.dagger().dagger()
        assert restored[0].gate.params == original[0].gate.params

    def test_circuit_json_expression_roundtrip(self) -> None:
        theta = Parameter("theta")
        phi = Parameter("phi")
        original = QuantumCircuit().rx(2 * theta - phi + 0.125, 0)

        raw = json.loads(original.to_json())
        encoded = raw["instructions"][0]["gate"]["params"][0]
        assert encoded == {
            "_expr": {
                "constant": 0.125,
                "terms": {"phi": -1.0, "theta": 2.0},
            }
        }

        restored = QuantumCircuit.from_json(json.dumps(raw))
        assert restored.parameters == {theta, phi}
        assert restored[0].gate.params == original[0].gate.params


class TestQuantumCircuitEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_qubit_index(self) -> None:
        """Test circuit with large qubit index."""
        circ = QuantumCircuit()
        circ.h(100)
        assert circ.num_qubits == 101

    def test_bell_state_circuit(self) -> None:
        """Test building a Bell state circuit."""
        circ = QuantumCircuit(name="bell")
        circ.h(0).cx(0, 1).measure([0, 1], [0, 1])
        assert circ.num_qubits == 2
        assert circ.num_clbits == 2
        assert len(circ) == 3

    def test_ghz_state_circuit(self) -> None:
        """Test building a 3-qubit GHZ state circuit."""
        circ = QuantumCircuit(name="ghz")
        circ.h(0).cx(0, 1).cx(0, 2).measure_all()
        assert circ.num_qubits == 3
        assert len(circ) == 4

    def test_using_predefined_gate_objects(self) -> None:
        """Test append with predefined gate singleton."""
        circ = QuantumCircuit()
        circ.append(H, (0,))
        circ.append(CX, (0, 1))
        circ.append(CCX, (0, 1, 2))
        assert len(circ) == 3

    def test_append_with_generic_gate_function(self) -> None:
        """Test append with gate created from factory function."""
        gate = RX.with_params(0.5)
        circ = QuantumCircuit()
        circ.append(gate, (0,))
        assert len(circ) == 1


class TestCircuitRegressionWP1:
    """Regression tests for compose / JSON / binding / numpy-angle bugs."""

    def test_compose_preserves_and_remaps_noise_channels(self) -> None:
        """B07: compose() keeps channel instructions and remaps their qubits."""
        other = QuantumCircuit().h(0).depolarize(0.1, 0).bit_flip(0.2, 1)
        out = QuantumCircuit().compose(other, [2, 0])
        assert out[1].is_channel and out[1].qubits == (2,)
        assert out[1].channel is other[1].channel
        assert out[2].is_channel and out[2].qubits == (0,)
        assert out[2].channel is other[2].channel

    def _noisy(self) -> QuantumCircuit:
        circ = QuantumCircuit().h(0)
        circ.bit_flip(0.11, 0).phase_flip(0.12, 1).bit_phase_flip(0.13, 0)
        circ.depolarize(0.14, 1).pauli_error(0.05, 0.06, 0.07, 0)
        circ.amplitude_damp(0.21, 1).phase_damp(0.22, 0)
        circ.generalized_amplitude_damp(0.3, 0.2, 1)
        return circ

    def test_json_roundtrip_preserves_channel_strengths(self) -> None:
        """B08: every channel's p / gamma / p_thermal survives to/from_json."""
        circ = self._noisy()
        back = QuantumCircuit.from_json(circ.to_json())
        channels = [(a.channel, b.channel) for a, b in zip(circ, back, strict=True)]
        for orig, new in channels:
            if orig is not None:
                assert new == orig

    def test_json_contains_strength_fields(self) -> None:
        """B08: strength fields are written explicitly to the JSON payload."""
        data = json.loads(QuantumCircuit().depolarize(0.25, 0).to_json())
        assert data["instructions"][0]["channel"]["p"] == 0.25

    def test_generalized_amplitude_damping_uses_canonical_name(self) -> None:
        """B10: the fluent helper uses the name understood by from_json."""
        circ = QuantumCircuit().generalized_amplitude_damp(0.3, 0.2, 0)
        assert circ[0].channel is not None
        assert circ[0].channel.name == "generalized_amplitude_damping"
        assert QuantumCircuit.from_json(circ.to_json())[0].channel == circ[0].channel

    def test_from_json_accepts_legacy_gen_amp_damp_name(self) -> None:
        """B10: old documents with ``gen_amp_damp`` are still readable."""
        payload = json.dumps(
            {
                "instructions": [
                    {
                        "type": "CHANNEL",
                        "qubits": [0],
                        "channel": {
                            "channel_type": "gen_amp_damp",
                            "num_qubits": 1,
                            "p_thermal": 0.2,
                        },
                    }
                ]
            }
        )
        channel = QuantumCircuit.from_json(payload)[0].channel
        assert channel is not None
        assert channel.name == "generalized_amplitude_damping"
        assert channel.p_thermal == 0.2  # type: ignore[attr-defined]
        assert channel.gamma == 0.0  # type: ignore[attr-defined]

    def test_from_json_legacy_channel_without_strength(self) -> None:
        """B08: old documents without ``p`` still load (strength 0.0)."""
        payload = json.dumps(
            {
                "instructions": [
                    {
                        "type": "CHANNEL",
                        "qubits": [0],
                        "channel": {"channel_type": "bit_flip", "num_qubits": 1},
                    }
                ]
            }
        )
        channel = QuantumCircuit.from_json(payload)[0].channel
        assert channel is not None and channel.p == 0.0  # type: ignore[attr-defined]

    @pytest.mark.parametrize("value", ["oops", 1j, None, [0.5]])
    def test_bind_parameters_rejects_non_real_values(self, value: object) -> None:
        """B11: bound values must be real numbers."""
        theta = Parameter("theta")
        with pytest.raises(TypeError, match="real"):
            QuantumCircuit().rx(theta, 0).bind_parameters({theta: value})  # type: ignore[dict-item]

    def test_bind_parameters_converts_numpy_scalar_to_float(self) -> None:
        """B11/B12: numpy real scalars are stored as Python floats."""
        theta = Parameter("theta")
        bound = QuantumCircuit().rx(theta, 0).bind_parameters({theta: np.float32(0.5)})
        params = bound[0].gate.params  # type: ignore[union-attr]
        assert params == (0.5,) and type(params[0]) is float

    @pytest.mark.parametrize("value", [np.int64(1), np.float32(0.5), np.float64(0.3)])
    def test_numpy_scalar_angles_become_float(self, value: object) -> None:
        """B12: numpy scalar angles are normalized and serializable."""
        circ = QuantumCircuit().rx(value, 0).u3(value, value, value, 1)  # type: ignore[arg-type]
        circ.cu3(value, value, value, 0, 1)  # type: ignore[arg-type]
        for instr in circ:
            assert instr.gate is not None
            assert all(type(p) is float for p in instr.gate.params)
        assert QuantumCircuit.from_json(circ.to_json())[0].gate == circ[0].gate

    def test_fluent_angle_rejects_non_numeric(self) -> None:
        """B12: non-numeric angles are rejected instead of stored."""
        with pytest.raises(TypeError, match="real number"):
            QuantumCircuit().rx("0.5", 0)  # type: ignore[arg-type]
