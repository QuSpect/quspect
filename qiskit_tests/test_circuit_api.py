"""Cross-validate the ``quspect.circuit`` construction API against qiskit.

Covers every fluent gate method, measure / measure_all / barrier, the
sequence protocol, active bits, compose (with qubit remapping), copy, dagger
(vs qiskit ``inverse()``), to_dag, draw / svg smoke tests,
``CircuitInstruction`` validation, and input errors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Operator

from qiskit_tests.helpers import (
    GATE_SPECS,
    assert_unitary_equal,
    qiskit_circuit_unitary,
    quspect_circuit_unitary,
    random_circuit_pair,
    to_qiskit,
)
from quspect import QuantumCircuit
from quspect.circuit import CircuitInstruction, Parameter
from quspect.circuit.instruction import InstructionType
from quspect.gates import Gate
from quspect.gates.standard import CX, H
from quspect.simulation import Simulator


def _params_for(num: int, rng: np.random.Generator) -> list[float]:
    return [float(x) for x in rng.uniform(-2 * math.pi, 2 * math.pi, size=num)]


# =============================================================================
# Fluent gate API
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_fluent_gate_method_on_gapped_qubits_matches_qiskit(
    name: str, rng: np.random.Generator
) -> None:
    """Each fluent method applied to non-contiguous qubits equals qiskit."""
    spec = GATE_SPECS[name]
    qubits = [3, 0, 2][: spec.num_qubits]
    params = _params_for(spec.num_params, rng)
    circ = QuantumCircuit()
    circ.h(1)
    result = getattr(circ, name)(*params, *qubits)
    assert result is circ, "fluent methods must return self"
    qc = qiskit.QuantumCircuit(circ.num_qubits)
    qc.h(1)
    spec.apply_qiskit(qc, params, qubits)
    assert circ.num_qubits == max(*qubits, 1) + 1
    assert circ[-1].gate is not None and circ[-1].gate.name == name
    assert circ[-1].qubits == tuple(qubits)
    assert_unitary_equal(quspect_circuit_unitary(circ), Operator(qc).data)


@pytest.mark.parametrize(
    "name", ["rx", "ry", "rz", "p", "u1", "crx", "cry", "crz", "cp"]
)
def test_fluent_int_angle_is_stored_as_float(name: str) -> None:
    """Integer angles are converted to float parameters by the fluent API."""
    circ = QuantumCircuit()
    nq = GATE_SPECS[name].num_qubits
    getattr(circ, name)(1, *range(nq))
    gate = circ[0].gate
    assert gate is not None
    assert gate.params == (1.0,)
    assert isinstance(gate.params[0], float)


def test_fluent_chaining_builds_expected_sequence() -> None:
    """Chained calls append instructions in order and infer sizes."""
    circ = QuantumCircuit(name="chain")
    circ.h(0).cx(0, 2).rz(0.1, 2).measure([2, 0], [1, 0]).barrier()
    assert len(circ) == 5
    assert circ.num_qubits == 3
    assert circ.num_clbits == 2
    assert [i.type for i in circ] == [
        InstructionType.GATE,
        InstructionType.GATE,
        InstructionType.GATE,
        InstructionType.MEASURE,
        InstructionType.BARRIER,
    ]
    assert circ.name == "chain"
    circ.name = "renamed"
    assert circ.name == "renamed"
    assert "renamed" in repr(circ)
    assert "num_qubits=3" in repr(circ)


def test_generic_append_matches_fluent() -> None:
    """``append(gate, qubits)`` is equivalent to the named method."""
    a = QuantumCircuit().append(H, [0]).append(CX, (0, 1))
    b = QuantumCircuit().h(0).cx(0, 1)
    assert a.instructions == b.instructions


# =============================================================================
# Sequence protocol, sizes and active bits
# =============================================================================


def test_len_iter_getitem() -> None:
    """``len``, iteration, indexing (incl. negative) and IndexError."""
    circ = QuantumCircuit().h(0).x(1).cz(0, 1)
    assert len(circ) == 3
    names = [i.gate.name for i in circ if i.gate is not None]
    assert names == ["h", "x", "cz"]
    assert circ[0].qubits == (0,)
    assert circ[-1].gate is not None and circ[-1].gate.name == "cz"
    assert circ.instructions == tuple(circ)
    with pytest.raises(IndexError):
        circ[3]


def test_empty_circuit_properties() -> None:
    """An empty circuit has zero size and empty active sets."""
    circ = QuantumCircuit()
    assert len(circ) == 0
    assert circ.num_qubits == 0
    assert circ.num_clbits == 0
    assert circ.active_qubits == set()
    assert circ.active_clbits == set()
    assert circ.parameters == set()
    assert circ.measure_all() is circ
    assert len(circ) == 0


def test_active_qubits_and_clbits_with_gaps() -> None:
    """Active sets contain only referenced indices; sizes use max + 1."""
    circ = QuantumCircuit().h(4).cx(4, 1).measure([1], [5])
    assert circ.active_qubits == {1, 4}
    assert circ.active_clbits == {5}
    assert circ.num_qubits == 5
    assert circ.num_clbits == 6


def test_measure_all_uses_active_qubits_only() -> None:
    """``measure_all`` maps each active qubit q to clbit q (gaps preserved)."""
    circ = QuantumCircuit().h(0).x(3)
    circ.measure_all()
    last = circ[-1]
    assert last.is_measure
    assert last.qubits == (0, 3)
    assert last.clbits == (0, 3)


def test_barrier_default_and_explicit() -> None:
    """An argument-less barrier spans all active qubits; explicit is kept."""
    circ = QuantumCircuit().h(2).x(0)
    circ.barrier()
    circ.barrier(1)
    assert circ[2].is_barrier and circ[2].qubits == (0, 2)
    assert circ[3].qubits == (1,)
    assert circ.num_qubits == 3
    empty = QuantumCircuit().barrier()
    assert empty[0].qubits == ()
    assert empty.num_qubits == 0


@pytest.mark.qiskit_ref
def test_barrier_does_not_change_unitary() -> None:
    """Barriers are no-ops for the unitary, as in qiskit."""
    circ = QuantumCircuit().h(0).barrier().cx(0, 1).barrier(1).t(1)
    qc = to_qiskit(circ)
    assert_unitary_equal(quspect_circuit_unitary(circ), qiskit_circuit_unitary(qc))


# =============================================================================
# Error handling
# =============================================================================


@pytest.mark.parametrize(
    ("build", "exc"),
    [
        (lambda: QuantumCircuit().h(-1), ValueError),
        (lambda: QuantumCircuit().h(1.0), TypeError),  # type: ignore[arg-type]
        (lambda: QuantumCircuit().h(True), TypeError),
        (lambda: QuantumCircuit().cx(0, -2), ValueError),
        (lambda: QuantumCircuit().measure([0], [-1]), ValueError),
        (lambda: QuantumCircuit().measure([0], [0, 1]), ValueError),
        (lambda: QuantumCircuit().append(H, [0, 1]), ValueError),
        (lambda: QuantumCircuit().append(CX, [0]), ValueError),
        (lambda: QuantumCircuit().barrier(-1), ValueError),
        (lambda: Gate(name="h", num_qubits=0), ValueError),
    ],
)
def test_invalid_construction_raises(build: object, exc: type[Exception]) -> None:
    """Out-of-range / wrong-type / wrong-arity inputs raise immediately."""
    with pytest.raises(exc):
        build()  # type: ignore[operator]


def test_duplicate_qubits_in_gate_rejected() -> None:
    """``cx(0, 0)`` must be rejected at construction or simulation time."""
    with pytest.raises((ValueError, RuntimeError)):
        circ = QuantumCircuit().x(0).cx(0, 0)
        Simulator(seed=0).statevector(circ)


def test_duplicate_qubit_measure_records_both_clbits() -> None:
    """Measuring one qubit into two clbits (allowed in qiskit) copies it."""
    circ = QuantumCircuit().h(0).measure([0, 0], [0, 1])
    counts = Simulator(seed=5).run(circ, shots=200).counts
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": InstructionType.GATE, "qubits": (0,)},
        {"type": InstructionType.GATE, "qubits": (0,), "gate": H, "channel": object()},
        {"type": InstructionType.MEASURE, "qubits": (0,), "clbits": (0,), "gate": H},
        {
            "type": InstructionType.MEASURE,
            "qubits": (0,),
            "clbits": (0,),
            "channel": object(),
        },
        {"type": InstructionType.BARRIER, "qubits": (0,), "gate": H},
        {"type": InstructionType.BARRIER, "qubits": (0,), "channel": object()},
        {"type": InstructionType.CHANNEL, "qubits": (0,)},
        {"type": InstructionType.CHANNEL, "qubits": (0,), "gate": H, "channel": 1},
        {"type": InstructionType.GATE, "qubits": (0,), "clbits": ("a",), "gate": H},
    ],
)
def test_circuit_instruction_validation(kwargs: dict[str, object]) -> None:
    """Inconsistent ``CircuitInstruction`` field combinations are rejected."""
    with pytest.raises((ValueError, TypeError)):
        CircuitInstruction(**kwargs)  # type: ignore[arg-type]


def test_circuit_instruction_predicates_and_matrix() -> None:
    """Type predicates and ``matrix()`` (gate only, big-endian)."""
    g = CircuitInstruction(type=InstructionType.GATE, gate=CX, qubits=(0, 1))
    m = CircuitInstruction(type=InstructionType.MEASURE, qubits=(0,), clbits=(0,))
    b = CircuitInstruction(type=InstructionType.BARRIER, qubits=())
    assert (g.is_gate, g.is_measure, g.is_barrier, g.is_channel) == (
        True,
        False,
        False,
        False,
    )
    assert m.is_measure and b.is_barrier
    mat = g.matrix()
    assert mat is not None
    expected = Operator(qiskit.circuit.library.CXGate()).reverse_qargs().data
    assert_unitary_equal(mat, expected)
    assert m.matrix() is None
    assert b.matrix() is None


def test_instruction_matrix_symbolic_raises_value_error() -> None:
    """A symbolic gate's ``matrix()`` should raise the documented ValueError."""
    circ = QuantumCircuit().rx(Parameter("a"), 0)
    with pytest.raises(ValueError):
        circ[0].matrix()


# =============================================================================
# compose
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_compose_with_qubit_mapping_matches_qiskit(seed: int) -> None:
    """``compose(other, qubits)`` equals qiskit ``compose(other, qubits=...)``."""
    base, qbase = random_circuit_pair(3, 8, seed)
    other, qother = random_circuit_pair(2, 6, seed + 100)
    mapping = [[2, 0], [0, 1], [1, 2], [2, 1]][seed]
    composed = base.compose(other, mapping)
    qcomposed = qbase.compose(qother, qubits=mapping)
    assert len(composed) == len(base) + len(other)
    assert_unitary_equal(
        quspect_circuit_unitary(composed, num_qubits=3),
        qiskit_circuit_unitary(qcomposed),
    )


@pytest.mark.qiskit_ref
def test_compose_onto_wider_range_and_clbit_map() -> None:
    """Compose may target new qubits; clbits are remapped when given."""
    base = QuantumCircuit().h(0)
    other = QuantumCircuit().x(0).cx(0, 1).measure([0, 1], [0, 1])
    out = base.compose(other, [3, 1], [2, 0])
    assert out.num_qubits == 4
    meas = out[-1]
    assert meas.qubits == (3, 1)
    assert meas.clbits == (2, 0)
    qc = qiskit.QuantumCircuit(4, 3)
    qc.h(0)
    qo = qiskit.QuantumCircuit(2, 2)
    qo.x(0)
    qo.cx(0, 1)
    qo.measure([0, 1], [0, 1])
    qref = qc.compose(qo, qubits=[3, 1], clbits=[2, 0])
    assert to_qiskit(out, num_clbits=3) == qref


def test_compose_default_clbits_identity_and_inplace() -> None:
    """Without ``clbits`` other's clbit indices are kept; inplace mutates."""
    base = QuantumCircuit().h(0)
    other = QuantumCircuit().measure([0], [1])
    copy_out = base.compose(other, [2])
    assert copy_out is not base
    assert len(base) == 1
    assert copy_out[-1].qubits == (2,) and copy_out[-1].clbits == (1,)
    same = base.compose(other, [2], inplace=True)
    assert same is base
    assert len(base) == 2


def test_compose_wrong_mapping_length_raises() -> None:
    """``len(qubits) != other.num_qubits`` raises ValueError."""
    other = QuantumCircuit().cx(0, 1)
    with pytest.raises(ValueError, match="compose"):
        QuantumCircuit().compose(other, [0])


def test_compose_preserves_noise_channels() -> None:
    """Noise channels in ``other`` are carried over (remapped) by compose."""
    other = QuantumCircuit().h(0).depolarize(0.1, 0)
    out = QuantumCircuit().compose(other, [1])
    assert out[-1].is_channel
    assert out[-1].qubits == (1,)
    assert out[-1].channel is other[-1].channel


# =============================================================================
# copy
# =============================================================================


def test_copy_is_independent() -> None:
    """``copy`` yields an equal but independent instruction list."""
    circ = QuantumCircuit(name="orig").h(0).cx(0, 1)
    cp = circ.copy()
    assert cp.name == "orig"
    assert cp.instructions == circ.instructions
    cp.x(2)
    assert len(circ) == 2
    assert len(cp) == 3
    assert circ.copy(name="other").name == "other"


# =============================================================================
# dagger
# =============================================================================


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("name", sorted(GATE_SPECS))
def test_dagger_single_gate_matches_qiskit_inverse(
    name: str, rng: np.random.Generator
) -> None:
    """The adjoint of each gate equals qiskit ``Operator(gate).adjoint()``."""
    spec = GATE_SPECS[name]
    qubits = [1, 0, 2][: spec.num_qubits]
    for _ in range(3):
        params = _params_for(spec.num_params, rng)
        circ = QuantumCircuit()
        circ.id(2)
        spec.apply_quspect(circ, params, qubits)
        qc = to_qiskit(circ)
        dag = circ.dagger()
        assert_unitary_equal(
            quspect_circuit_unitary(dag, num_qubits=3),
            qiskit_circuit_unitary(qc.inverse()),
        )


@pytest.mark.qiskit_ref
@pytest.mark.parametrize("seed", range(6))
def test_dagger_random_circuit_matches_qiskit_inverse(seed: int) -> None:
    """``circ.dagger()`` equals qiskit ``qc.inverse()`` for random circuits."""
    circ, qc = random_circuit_pair(3, 20, seed)
    dag = circ.dagger()
    u_dag = quspect_circuit_unitary(dag, num_qubits=3)
    assert_unitary_equal(u_dag, qiskit_circuit_unitary(qc.inverse()))
    u = quspect_circuit_unitary(circ)
    assert_unitary_equal(u_dag @ u, np.eye(8))


def test_dagger_structure_and_names() -> None:
    """Dagger reverses order, keeps measure/barrier/channel, renames."""
    circ = QuantumCircuit(name="c").h(0).barrier().s(0).measure([0], [0])
    circ.bit_flip(0.1, 0)
    dag = circ.dagger()
    assert dag.name == "c†"
    assert QuantumCircuit().h(0).dagger().name == "dagger"
    assert dag[0].is_channel and dag[0].channel is circ[-1].channel
    assert dag[1].is_measure and dag[1].clbits == (0,)
    assert dag[2].gate is not None and dag[2].gate.name == "sdg"
    assert dag[3].is_barrier and dag[3].qubits == (0,)
    assert dag[4].gate is not None and dag[4].gate.name == "h"
    empty_barrier = QuantumCircuit().barrier().dagger()
    assert empty_barrier[0].is_barrier and empty_barrier[0].qubits == ()


@pytest.mark.qiskit_ref
def test_dagger_symbolic_parameters_then_bind() -> None:
    """Dagger of a parameterized circuit, bound afterwards, equals qiskit."""
    a, b = Parameter("a"), Parameter("b")
    circ = QuantumCircuit().rx(a, 0).u3(a, b, 0.4, 1).u2(b, a, 0).cu3(a, 0.2, b, 0, 1)
    circ.cp(2 * a - b, 1, 0)
    binding = {a: 0.37, b: -1.21}
    bound_dag = circ.dagger().bind_parameters(binding)
    ref = to_qiskit(circ.bind_parameters(binding)).inverse()
    assert_unitary_equal(
        quspect_circuit_unitary(bound_dag), qiskit_circuit_unitary(ref)
    )


def test_dagger_unknown_gate_returned_unchanged() -> None:
    """A gate without parameters and unknown adjoint is kept as is."""
    custom = Gate(name="mystery", num_qubits=1)
    dag = QuantumCircuit().append(custom, [0]).dagger()
    assert dag[0].gate == custom


# =============================================================================
# to_dag / draw / svg
# =============================================================================


def test_to_dag_node_counts() -> None:
    """``to_dag`` exposes the gate and measurement nodes with wire counts."""
    circ = QuantumCircuit().h(0).cx(0, 1).measure_all()
    dag = circ.to_dag()
    assert dag.num_qubits == 2
    assert dag.num_clbits == 2
    assert len(dag.operation_nodes()) == 3
    assert dag.num_nodes() > 3


def test_draw_and_svg_smoke() -> None:
    """``draw`` returns text mentioning every qubit; ``svg`` returns SVG."""
    circ = QuantumCircuit().h(0).cx(0, 1).rz(0.5, 1).measure_all()
    text = circ.draw()
    assert isinstance(text, str)
    assert "q0" in text and "q1" in text
    svg = circ.svg()
    markup = svg._repr_svg_() if hasattr(svg, "_repr_svg_") else str(svg)
    assert "<svg" in markup
    themed = circ.svg(style="dark", width=500, scale=1.5)
    assert themed is not None
