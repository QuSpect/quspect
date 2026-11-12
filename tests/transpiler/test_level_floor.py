"""Higher optimization levels never use more two-qubit gates than level 1.

Level 2 routes on two-qubit blocks with mirror gates and a SWAP-fusion
bonus, which trades more SWAPs for merges that sparse circuits do not always
give back; level 1's plain routing is then better.  ``transpile()`` therefore
compiles at level 1 as well and keeps that result when it has fewer two-qubit
gates.
"""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.ir import qasm2_loads
from quspect.transpiler import CompilationTarget, generate_random_circuit, transpile

IBM_CZ = {"cz", "rz", "sx", "x"}


def _grid(rows: int, cols: int) -> CompilationTarget:
    edges = [
        (r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)
    ]
    edges += [
        (r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)
    ]
    return CompilationTarget.from_coupling_map(
        rows * cols, edges, native_gates=set(IBM_CZ)
    )


def _counts(circuit: QuantumCircuit, target: CompilationTarget, seed: int) -> list[int]:
    return [
        transpile(
            circuit, target, optimization_level=level, seed=seed
        ).stats.two_qubit_gates
        for level in (1, 2, 3)
    ]


@pytest.mark.parametrize("circuit_seed", [2, 3, 11])
def test_level2_is_not_worse_than_level1_on_a_grid(circuit_seed: int) -> None:
    """Seeds on which level 2 used to need more CZ than level 1 (201 / 205,
    239 / 262 and 209 / 214 on a 5x5 grid)."""
    circuit = generate_random_circuit(20, 20, seed=circuit_seed)
    one, two, three = _counts(circuit, _grid(5, 5), 0)
    assert three <= two <= one


@pytest.mark.parametrize("circuit_seed", range(12))
def test_levels_are_monotone_in_two_qubit_gates(circuit_seed: int) -> None:
    circuit = generate_random_circuit(9, 10, seed=circuit_seed)
    for target in (_grid(3, 3), _grid(2, 5)):
        for seed in (0, 1):
            one, two, three = _counts(circuit, target, seed)
            assert three <= two <= one


@pytest.mark.parametrize("name", ["adder_6", "adder_10"])
@pytest.mark.parametrize("kind", ["line", "ring", "grid"])
def test_golden_adders_are_not_worse_at_level2(name: str, kind: str) -> None:
    """The Toffoli-heavy golden circuits needed 46 / 44, 41 / 40, 86 / 84 and
    74 / 71 CZ at level 2 / level 1 on these couplings."""
    from tests.transpiler.test_golden_transpile import CIRCUITS, _edges

    circuit = qasm2_loads(CIRCUITS[name])
    n, edges = _edges(kind, circuit.num_qubits)
    target = CompilationTarget.from_coupling_map(n, edges, native_gates=set(IBM_CZ))
    one, two, three = _counts(circuit, target, 0)
    assert three <= two <= one


def test_level1_result_is_returned_as_it_is_when_it_wins() -> None:
    circuit = generate_random_circuit(20, 20, seed=3)
    target = _grid(5, 5)
    one = transpile(circuit, target, optimization_level=1, seed=0)
    two = transpile(circuit, target, optimization_level=2, seed=0)
    assert two.stats.selected == f"level1/{one.stats.selected}"
    assert two.stats.optimization_level == 2
    # level 2's own candidates stay listed, the level-1 result comes last
    assert [label for label, *_ in two.stats.candidate_scores][-1] == "level1/default"
    assert two.stats.candidate_scores[0][0] == "default"
    assert two.stats.candidate_scores[-1][1] == one.stats.two_qubit_gates
    assert two.stats.candidate_scores[-1][1] < two.stats.candidate_scores[0][1]
    assert two.stats.num_evaluated == len(two.stats.candidate_scores)
    assert two.circuit.instructions == one.circuit.instructions
    assert two.initial_layout == one.initial_layout
    assert two.final_layout == one.final_layout
    assert two.stats.two_qubit_gates == one.stats.two_qubit_gates


def test_level2_result_is_kept_when_it_is_not_worse() -> None:
    circuit = generate_random_circuit(20, 20, seed=4)  # 175 CZ at level 1, 161 at 2
    two = transpile(circuit, _grid(5, 5), optimization_level=2, seed=0)
    assert not two.stats.selected.startswith("level1/")
