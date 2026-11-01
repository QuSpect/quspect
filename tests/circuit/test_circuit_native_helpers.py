"""Bulk native helpers of the circuit layer against their Python loops.

``QuantumCircuit.num_qubits`` / ``num_clbits``, the instruction remapping of
``QuantumCircuit.compose``, the transpiler's gate / depth statistics and
``two_qubit.ops_to_instructions`` have native bulk forms; each must return
exactly what the Python loop returns, and hand anything unusual (subclassed
instructions, non-int bits, cases in which the loop raises) back to it.
"""

from __future__ import annotations

import random

import pytest

from quspect import QuantumCircuit
from quspect.circuit import _native
from quspect.circuit.instruction import CircuitInstruction, InstructionType
from quspect.gates.gate import Gate
from quspect.gates.standard import CX, H, X
from quspect.synthesis import two_qubit
from quspect.transpiler import _transpile

pytestmark = pytest.mark.skipif(
    _native._circuit_helpers() is None, reason="native extension not built"
)


def _random_circuit(rng: random.Random, n: int, m: int) -> QuantumCircuit:
    qc = QuantumCircuit()
    for _ in range(m):
        r = rng.random()
        a = rng.randrange(n)
        b = (a + 1 + rng.randrange(n - 1)) % n
        if r < 0.35:
            qc.cx(a, b)
        elif r < 0.6:
            qc.rz(0.3, a)
        elif r < 0.75:
            qc.h(a)
        elif r < 0.85:
            qc.barrier(a, b)
        elif r < 0.9:
            qc.barrier()
        elif r < 0.95:
            qc.measure([a], [rng.randrange(4)])
        elif n >= 3:
            qc.ccx(a, b, next(q for q in range(n) if q not in (a, b)))
    return qc


def _py_counts(qc: QuantumCircuit) -> tuple[int, int]:
    mq = max((max(i.qubits) for i in qc if i.qubits), default=-1)
    mc = max((max(i.clbits) for i in qc if i.clbits), default=-1)
    return mq + 1, mc + 1


def _py_depth(qc: QuantumCircuit, two_only: bool) -> int:
    level: dict[int, int] = {}
    deepest = 0
    for ins in qc:
        if ins.is_barrier or not ins.qubits:
            continue
        if two_only and not (ins.is_gate and len(ins.qubits) == 2):
            continue
        lv = 1 + max(level.get(q, 0) for q in ins.qubits)
        for q in ins.qubits:
            level[q] = lv
        deepest = max(deepest, lv)
    return deepest


@pytest.mark.parametrize("seed", range(20))
def test_counts_and_stats_match_python(seed: int) -> None:
    rng = random.Random(seed)
    qc = _random_circuit(rng, rng.randrange(2, 9), rng.randrange(0, 60))
    assert (qc.num_qubits, qc.num_clbits) == _py_counts(qc)
    expected = (
        sum(1 for i in qc if i.is_gate and len(i.qubits) == 2),
        _py_depth(qc, False),
        _py_depth(qc, True),
        sum(1 for i in qc if i.is_gate),
    )
    assert _native.instruction_stats(qc._instructions) == expected
    assert _transpile._finish_key(qc) == expected
    assert _transpile._score(qc) == (expected[0], expected[2], expected[3])


def test_unusual_instructions_fall_back() -> None:
    class Sub(CircuitInstruction):
        pass

    qc = QuantumCircuit().h(3)
    qc._instructions.append(Sub(InstructionType.GATE, (5,), gate=X))
    assert _native.bit_counts(qc._instructions) is None
    assert _native.instruction_stats(qc._instructions) is None
    assert qc.num_qubits == 6
    assert _transpile._finish_key(qc) == (0, 1, 0, 2)


@pytest.mark.parametrize("seed", range(10))
def test_compose_remap_matches_loop(seed: int) -> None:
    rng = random.Random(seed)
    qc = _random_circuit(rng, 5, 40)
    mapping = rng.sample(range(12), qc.num_qubits)
    fast = QuantumCircuit().x(0)
    fast.compose(qc, mapping, inplace=True)
    expected = [
        CircuitInstruction(
            type=i.type,
            gate=i.gate,
            channel=i.channel,
            qubits=tuple(mapping[q] for q in i.qubits),
            clbits=i.clbits,
            label=i.label,
        )
        for i in qc
    ]
    assert list(fast.instructions[1:]) == expected
    assert [vars(a) for a in fast.instructions[1:]] == [vars(b) for b in expected]


def test_compose_errors_come_from_the_loop() -> None:
    qc = QuantumCircuit().cx(0, 1)
    with pytest.raises(ValueError, match="distinct"):
        QuantumCircuit().compose(qc, [3, 3])
    with pytest.raises(ValueError, match="compose"):
        QuantumCircuit().compose(qc, [3])


def test_instruction_snapshot_type_check() -> None:
    ins = CircuitInstruction(InstructionType.GATE, (0,), gate=H)
    assert QuantumCircuit._from_instruction_snapshot([ins]).instructions == (ins,)
    with pytest.raises(TypeError):
        QuantumCircuit._from_instruction_snapshot([ins, "x"])  # type: ignore[list-item]


def test_ops_to_instructions_native_matches_python() -> None:
    ops = [
        ("rz", (0,), (0.25,)),
        ("cz", (0, 1), ()),
        ("sx", (1,), ()),
        ("cx", (1, 0), ()),
        ("rz", (1,), (-1.5,)),
    ]
    native = two_qubit.ops_to_instructions(ops, (7, 2))
    from quspect.circuit.instruction import _gate_instructions
    from quspect.gates.parameterized import RZ

    fixed = two_qubit._standard_gates()
    expected = _gate_instructions(
        [
            (
                RZ.with_params(*p) if name == "rz" else fixed[name],
                tuple((7, 2)[q] for q in local),
            )
            for name, local, p in ops
        ]
    )
    assert native == expected
    assert [vars(a) for a in native] == [vars(b) for b in expected]
    assert [vars(a.gate) for a in native] == [vars(b.gate) for b in expected]
    assert native[1].gate is fixed["cz"]
    assert isinstance(native[0].gate, Gate) and native[0].gate.label == RZ.label


def test_ops_to_instructions_unusual_inputs() -> None:
    # int parameter: normalized to float by Gate (Python path).
    out = two_qubit.ops_to_instructions([("rz", (0,), (1,))], (4, 5))
    assert out[0].gate.params == (1.0,) and type(out[0].gate.params[0]) is float
    with pytest.raises(KeyError):
        two_qubit.ops_to_instructions([("nope", (0,), ())], (0, 1))
    with pytest.raises(ValueError):
        two_qubit.ops_to_instructions([("rz", (0,), ())], (0, 1))
    with pytest.raises(ValueError):
        two_qubit.ops_to_instructions([("cx", (0, 1), ())], (3, 3))
    neg = two_qubit.ops_to_instructions([("cx", (-1, 0), ())], (3, 4))
    assert neg[0].qubits == (4, 3) and neg[0].gate is CX
