"""Gates registered with :func:`quspect.gates.register_gate`."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from quspect import QuantumCircuit, Simulator
from quspect.circuit import Parameter
from quspect.gates import Gate, gate_definition, gate_matrix, register_gate, registry
from quspect.gates.parameterized import RX
from quspect.gates.standard import CX, CZ, H, X
from quspect.ir import QASM2Exporter, decompose_deep, get_rules_for
from quspect.ir.compilation import _lower_registered_gates
from quspect.simulation import DensityMatrixSimulator

#: CX written as H(1) · CZ · H(1), registered under a name of its own.
MY_CX = [(H, (1,)), (CZ, (0, 1)), (H, (1,))]


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    saved = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def _statevector(circuit: QuantumCircuit) -> np.ndarray:
    return np.asarray(Simulator().statevector(circuit))


def test_register_gate_returns_the_gate_and_its_definition() -> None:
    gate = register_gate("my_cx", gate_matrix(CX), MY_CX)

    assert gate == Gate(name="my_cx", num_qubits=2)
    definition = gate_definition("my_cx")
    assert definition is not None
    assert definition.gate == gate
    assert definition.decomposition == tuple(MY_CX)
    assert definition.lower((3, 5)) == [(H, (5,)), (CZ, (3, 5)), (H, (5,))]
    with pytest.raises(ValueError, match="acts on 2 qubit"):
        definition.lower((3,))
    assert gate_definition("h") is None
    assert gate_definition("not_registered") is None


def test_gate_matrix_knows_a_registered_gate() -> None:
    gate = register_gate("my_cx", gate_matrix(CX), MY_CX)

    matrix = gate_matrix(gate)
    assert np.array_equal(matrix, gate_matrix(CX))
    matrix[0, 0] = 7  # a copy: the registered matrix stays as it was
    assert np.array_equal(gate_matrix(gate), gate_matrix(CX))
    with pytest.raises(ValueError, match="takes no parameters"):
        gate_matrix(Gate(name="my_cx", num_qubits=2, params=(0.1,)))
    with pytest.raises(ValueError, match="acts on 2 qubit"):
        gate_matrix(Gate(name="my_cx", num_qubits=1))


def test_registering_a_name_again_replaces_its_definition() -> None:
    register_gate("flip", gate_matrix(X), [(X, (0,))])
    register_gate("flip", gate_matrix(H), [(H, (0,))])

    definition = gate_definition("flip")
    assert definition is not None
    assert definition.decomposition == ((H, (0,)),)


def test_a_decomposition_may_differ_by_a_global_phase() -> None:
    # RX(π) = -i X.
    register_gate("minus_i_x", -1j * gate_matrix(X), [(RX.with_params(np.pi), (0,))])
    assert gate_definition("minus_i_x") is not None


@pytest.mark.parametrize(
    ("name", "matrix", "decomposition", "message"),
    [
        ("cx", gate_matrix(CX), MY_CX, "built-in"),
        ("rx", gate_matrix(X), [(X, (0,))], "built-in"),
        ("measure", gate_matrix(X), [(X, (0,))], "built-in"),
        ("", gate_matrix(X), [(X, (0,))], "empty"),
        ("g", np.ones((2, 2)), [(X, (0,))], "not unitary"),
        ("g", np.eye(3), [], r"2\^n × 2\^n"),
        ("g", np.eye(2)[:1], [], r"2\^n × 2\^n"),
        ("g", [[np.nan, 0], [0, 1]], [], "finite"),
        ("g", "not a matrix", [], "not numeric|2\\^n"),
        ("g", gate_matrix(X), [(H, (0,))], "does not implement"),
        ("g", gate_matrix(X), [], "does not implement"),
        ("g", gate_matrix(X), [(Gate(name="mystery", num_qubits=1), (0,))], "built-in"),
        ("g", gate_matrix(X), [("x", (0,))], "built-in"),
        ("g", gate_matrix(X), [(X, (1,))], "range"),
        ("g", gate_matrix(CX), [(CZ, (0, 0))], "distinct"),
        ("g", gate_matrix(X), [(X, (True,))], "range"),
        ("g", gate_matrix(X), [(X,)], "pairs"),
        ("g", gate_matrix(X), [(RX.with_params(Parameter("t")), (0,))], "numeric"),
    ],
)
def test_register_gate_refuses_what_it_cannot_define(
    name: str, matrix: object, decomposition: list, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        register_gate(name, matrix, decomposition)
    assert gate_definition(name) is None or name == "cx"


def test_register_gate_refuses_a_name_that_is_not_a_string() -> None:
    with pytest.raises(TypeError, match="string"):
        register_gate(3, gate_matrix(X), [(X, (0,))])  # type: ignore[arg-type]


def test_a_registered_gate_cannot_decompose_into_another_registered_gate() -> None:
    flip = register_gate("flip", gate_matrix(X), [(X, (0,))])
    with pytest.raises(ValueError, match="built-in"):
        register_gate("flip_again", gate_matrix(X), [(flip, (0,))])


def test_decomposition_rule_and_decompose_deep_lower_a_registered_gate() -> None:
    gate = register_gate("my_cx", gate_matrix(CX), MY_CX)
    circuit = QuantumCircuit().x(2).append(gate, [2, 0]).measure([0], [0])

    (rule,) = get_rules_for("my_cx", 2)
    assert rule.decompose(gate, (2, 0)) == [(H, (0,)), (CZ, (2, 0)), (H, (0,))]
    assert get_rules_for("my_cx", 1) == []
    out = decompose_deep(circuit, supported_gates={"x", "h", "cz"})
    assert [(i.gate.name, i.qubits) for i in out if i.is_gate] == [
        ("x", (2,)),
        ("h", (0,)),
        ("cz", (2, 0)),
        ("h", (0,)),
    ]


def test_simulators_and_exporter_run_a_registered_gate_as_its_decomposition() -> None:
    gate = register_gate("my_cx", gate_matrix(CX), MY_CX)
    circuit = QuantumCircuit().h(0).append(gate, [0, 1])
    reference = QuantumCircuit().h(0).cx(0, 1)

    assert np.allclose(_statevector(circuit), _statevector(reference), atol=1e-12)
    measured = circuit.copy().measure_all()
    expected = Simulator(seed=4).run(reference.copy().measure_all(), shots=64).counts
    assert Simulator(seed=4).run(measured, shots=64).counts == expected
    density = DensityMatrixSimulator()
    fidelity = density.density_matrix(circuit).fidelity(
        density.density_matrix(reference)
    )
    assert fidelity == pytest.approx(1.0, abs=1e-12)
    assert QASM2Exporter().export(circuit).splitlines()[-3:] == [
        "h q[1];",
        "cz q[0],q[1];",
        "h q[1];",
    ]


def test_lowering_keeps_labels_and_returns_other_circuits_themselves() -> None:
    plain = QuantumCircuit().h(0).cx(0, 1).measure_all()
    assert _lower_registered_gates(plain) is plain
    gate = register_gate("my_cx", gate_matrix(CX), MY_CX)
    assert _lower_registered_gates(plain) is plain

    labelled = QuantumCircuit(name="c")
    labelled.append(gate, [0, 1]).barrier().measure_all()
    labelled._instructions[0] = labelled._instructions[0].__class__(
        type=labelled._instructions[0].type,
        gate=gate,
        qubits=(0, 1),
        label="entangle",
    )
    lowered = _lower_registered_gates(labelled)
    assert lowered.name == "c"
    assert [i.label for i in lowered.instructions[:3]] == ["entangle"] * 3
    assert lowered.instructions[3:] == labelled.instructions[1:]
    with pytest.raises(ValueError, match="takes no parameters"):
        _lower_registered_gates(
            QuantumCircuit().append(
                Gate(name="my_cx", num_qubits=2, params=(0.5,)), [0, 1]
            )
        )
