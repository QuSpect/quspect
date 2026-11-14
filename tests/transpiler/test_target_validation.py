"""CompilationTarget.from_coupling_map validates its arguments."""

from __future__ import annotations

import numpy as np
import pytest

from quspect import QuantumCircuit
from quspect.gates import registry
from quspect.synthesis.two_qubit import _ENTANGLERS, register_entangler
from quspect.transpiler import CompilationTarget, transpile


def test_valid_targets_are_built_as_before() -> None:
    target = CompilationTarget.from_coupling_map(
        3, [(0, 1), (2, 1), (1, 0), (2, 2)], native_gates={"cz", "h", "rz", "x"}
    )
    assert target.num_qubits == 3
    assert target.coupling_graph == {(0, 1), (1, 2)}
    assert target.adjacency == [[1], [0, 2], [1]]
    assert target.native_gates == {"cz", "h", "rz", "x"}
    assert CompilationTarget.from_coupling_map(1, []).native_gates == set()
    assert CompilationTarget.from_coupling_map(0, []).adjacency == []


def test_edges_may_be_lists_and_numpy_integers() -> None:
    edges = np.array([[0, 1], [1, 2]])
    target = CompilationTarget.from_coupling_map(np.int64(3), [list(e) for e in edges])
    assert target.coupling_graph == {(0, 1), (1, 2)}
    assert all(type(q) is int for edge in target.coupling_graph for q in edge)
    assert type(target.num_qubits) is int
    assert CompilationTarget.from_coupling_map(3, edges).adjacency == [[1], [0, 2], [1]]


@pytest.mark.parametrize("edge", [(0, 5), (2, 0), (7, 8)])
def test_edge_outside_the_device_is_a_value_error(edge: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="outside the 2 qubits") as excinfo:
        CompilationTarget.from_coupling_map(2, [(0, 1), edge])
    assert str(edge) in str(excinfo.value)


def test_negative_qubit_in_an_edge_is_refused() -> None:
    """``adjacency[-1]`` would silently address the last qubit."""
    with pytest.raises(ValueError, match="outside the 3 qubits"):
        CompilationTarget.from_coupling_map(3, [(0, -1)])


@pytest.mark.parametrize("edge", [(0,), (0, 1, 2), 5, "ab", (0.0, 1), (True, 1)])
def test_malformed_edge_is_a_type_error(edge: object) -> None:
    with pytest.raises(TypeError, match="pair of qubit indices"):
        CompilationTarget.from_coupling_map(3, [edge])  # type: ignore[list-item]


@pytest.mark.parametrize("num_qubits", [-1, 1.5, True, "3", None])
def test_invalid_num_qubits(num_qubits: object) -> None:
    with pytest.raises((TypeError, ValueError), match="num_qubits"):
        CompilationTarget.from_coupling_map(num_qubits, [])  # type: ignore[arg-type]


def test_upper_case_gate_names_get_a_hint() -> None:
    with pytest.raises(ValueError, match="Unknown native gate") as excinfo:
        CompilationTarget.from_coupling_map(1, [], native_gates={"RZ", "SX"})
    message = str(excinfo.value)
    assert "'RZ' (did you mean 'rz'?)" in message
    assert "'SX' (did you mean 'sx'?)" in message
    assert "lower-case" in message


def test_unknown_gate_name_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown native gate.*'foo'") as excinfo:
        CompilationTarget.from_coupling_map(
            2, [(0, 1)], native_gates={"cz", "rz", "sx", "foo"}
        )
    assert "did you mean" not in str(excinfo.value)
    assert "register_entangler" in str(excinfo.value)


@pytest.mark.parametrize("native_gates", ["cz", {"cz", 3}, [None]])
def test_native_gates_must_be_names(native_gates: object) -> None:
    with pytest.raises(TypeError, match="native_gates"):
        CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates=native_gates)  # type: ignore[arg-type]


def test_every_builtin_gate_and_instruction_name_is_accepted() -> None:
    from quspect.gates.parameterized import ALL_PARAMETERIZED
    from quspect.gates.standard import ALL_STANDARD

    names = {gate.name for gate in (*ALL_STANDARD, *ALL_PARAMETERIZED)}
    names |= {"measure", "barrier"}
    target = CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates=names)
    assert target.native_gates == names
    assert target.native_gates is not names


def test_registered_entangler_is_a_known_name() -> None:
    assert "test_ecr" not in _ENTANGLERS
    with pytest.raises(ValueError, match="Unknown native gate"):
        CompilationTarget.from_coupling_map(2, [(0, 1)], native_gates={"test_ecr"})
    ecr = np.array(
        [[0, 1, 0, 1j], [1, 0, -1j, 0], [0, 1j, 0, 1], [-1j, 0, 1, 0]]
    ) / np.sqrt(2)
    register_entangler("test_ecr", ecr)
    try:
        target = CompilationTarget.from_coupling_map(
            2, [(0, 1)], native_gates={"test_ecr", "rz", "sx", "x"}
        )
        result = transpile(QuantumCircuit().h(0).cx(0, 1), target)
        assert "test_ecr" in {i.gate.name for i in result.circuit.instructions}
    finally:
        del _ENTANGLERS["test_ecr"]
        registry._REGISTRY.pop("test_ecr", None)


def test_other_constructors_stay_permissive() -> None:
    """Vendor metadata may list names QuSpect does not know."""
    target = CompilationTarget(num_qubits=2, native_gates={"cz", "rz", "x2p", "reset"})
    assert target.native_gates == {"cz", "rz", "x2p", "reset"}


def test_induced_subgraph_keeps_vendor_gate_names() -> None:
    """A device target read from vendor metadata is not re-validated."""
    device = CompilationTarget(
        num_qubits=4,
        coupling_graph={(0, 1), (1, 2), (2, 3)},
        adjacency=[[1], [0, 2], [1, 3], [2]],
        native_gates={"cz", "rx", "ry", "delay", "xy2p"},
    )
    dense, dense_to_physical = device.induced_subgraph([2, 1, 3])
    assert dense_to_physical == [2, 1, 3]
    assert dense.num_qubits == 3
    assert dense.coupling_graph == {(0, 1), (0, 2)}
    assert dense.native_gates == device.native_gates
    assert dense.native_gates is not device.native_gates
