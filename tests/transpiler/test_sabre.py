"""Tests for the SABRE qubit mapping and routing algorithm."""

from __future__ import annotations

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def line3_target() -> CompilationTarget:
    """3-qubit linear coupling: 0-1-2."""
    return CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])


@pytest.fixture
def line5_target() -> CompilationTarget:
    """5-qubit linear coupling: 0-1-2-3-4."""
    return CompilationTarget.from_coupling_map(5, [(0, 1), (1, 2), (2, 3), (3, 4)])


@pytest.fixture
def ring5_target() -> CompilationTarget:
    """5-qubit ring coupling."""
    return CompilationTarget.from_coupling_map(
        5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    )


# ---------------------------------------------------------------------------
# CompilationTarget tests
# ---------------------------------------------------------------------------


class TestCompilationTarget:
    def test_from_coupling_map_basic(self) -> None:
        target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        assert target.num_qubits == 3
        assert len(target.coupling_graph) == 2
        assert (0, 1) in target.coupling_graph
        assert (1, 2) in target.coupling_graph

    def test_adjacency_built(self, line3_target: CompilationTarget) -> None:
        assert line3_target.adjacency[0] == [1]
        assert line3_target.adjacency[1] == [0, 2]
        assert line3_target.adjacency[2] == [1]

    def test_duplicate_edges_deduplicated(self) -> None:
        target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 0)])
        assert len(target.coupling_graph) == 1

    def test_self_loop_ignored(self) -> None:
        target = CompilationTarget.from_coupling_map(3, [(0, 0), (0, 1)])
        assert (0, 0) not in target.coupling_graph

    def test_native_gates_stored(self) -> None:
        target = CompilationTarget.from_coupling_map(
            2, [(0, 1)], native_gates={"cz", "rx", "ry", "rz"}
        )
        assert target.native_gates == {"cz", "rx", "ry", "rz"}

    def test_induced_subgraph_preserves_selected_physical_order(self) -> None:
        target = CompilationTarget.from_coupling_map(
            8,
            [(1, 3), (3, 6), (6, 7), (0, 2)],
            native_gates={"cx", "rz"},
        )

        induced, dense_to_physical = target.induced_subgraph([6, 3, 1])

        assert dense_to_physical == [6, 3, 1]
        assert induced.num_qubits == 3
        assert induced.coupling_graph == {(0, 1), (1, 2)}
        assert induced.native_gates == {"cx", "rz"}

    @pytest.mark.parametrize(
        ("physical_qubits", "error"),
        [
            ([], ValueError),
            ([1, 1], ValueError),
            ([1, 8], ValueError),
            ([True], TypeError),
        ],
    )
    def test_induced_subgraph_rejects_invalid_selection(
        self,
        physical_qubits: list[int],
        error: type[Exception],
    ) -> None:
        target = CompilationTarget.from_coupling_map(8, [(1, 2)])

        with pytest.raises(error):
            target.induced_subgraph(physical_qubits)


# ---------------------------------------------------------------------------
# SABRE — basic correctness
# ---------------------------------------------------------------------------


class TestSABREBasic:
    """Tests on circuits that should need no SWAPs."""

    def test_empty_circuit(self, line5_target: CompilationTarget) -> None:
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="empty")
        result, pi0, pi1 = sabre.solve(qc, iter_num=1, seed=0)
        assert len(result.instructions) == 0
        assert pi0 == []  # no logical qubits → empty mapping
        assert pi1 == []

    def test_single_qubit_only(self, line5_target: CompilationTarget) -> None:
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="single")
        qc.h(0)
        qc.x(1)
        qc.z(2)
        result, pi0, _ = sabre.solve(qc, iter_num=1, seed=0)
        assert len(result.instructions) == 3
        # No SWAPs should be inserted for single-qubit gates
        for instr in result:
            assert not (instr.is_gate and instr.gate and instr.gate.name == "swap")

    def test_adjacent_cnot_no_routing(self, line5_target: CompilationTarget) -> None:
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="adj")
        qc.cx(0, 1)
        qc.cx(2, 3)
        result, pi0, pi1 = sabre.solve(qc, iter_num=3, seed=0)
        assert _all_two_qubit_adjacent(result, pi0, line5_target.coupling_graph)

    def test_measurements_preserved(self, line5_target: CompilationTarget) -> None:
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="meas")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        result, _, _ = sabre.solve(qc, iter_num=1, seed=0)
        n_meas = sum(1 for i in result if i.is_measure)
        assert n_meas >= 2

    def test_measurement_classical_targets_preserved(
        self, line5_target: CompilationTarget
    ) -> None:
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="permuted_measurement")
        qc.x(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [1, 0])

        result, _, _ = sabre.solve(qc, iter_num=1, seed=0)

        measured_pairs = {
            (instr.qubits[0], instr.clbits[0]) for instr in result if instr.is_measure
        }
        assert {clbit for _, clbit in measured_pairs} == {0, 1}

    def test_higher_arity_gate_requires_decomposition(
        self, line3_target: CompilationTarget
    ) -> None:
        from quspect.gates.gate import Gate

        circuit = QuantumCircuit().append(Gate(name="cswap", num_qubits=3), (0, 1, 2))

        with pytest.raises(
            ValueError,
            match=r"cannot route 3-qubit gate 'cswap'.*decompose",
        ):
            SABRE(line3_target).solve(circuit, iter_num=1, seed=0)

    def test_ccx_routed_as_unit(self, line3_target: CompilationTarget) -> None:
        circuit = QuantumCircuit().ccx(0, 1, 2)
        routed, _, _ = SABRE(line3_target).solve(circuit, iter_num=1, seed=0)
        assert all(len(i.qubits) <= 2 for i in routed)
        assert sum(1 for i in routed if i.gate is not None and i.gate.name == "cx") == 8


# ---------------------------------------------------------------------------
# SABRE — routing
# ---------------------------------------------------------------------------


class TestSABRERouting:
    """Tests where SWAP insertion is unavoidable."""

    def test_star_graph_needs_swap(self, line5_target: CompilationTarget) -> None:
        """Qubit 0 talks to 1,2,3 → needs SWAP on a linear chain."""
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="star")
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        result, pi0, pi1 = sabre.solve(qc, iter_num=5, seed=0)
        assert _all_two_qubit_adjacent(result, pi0, line5_target.coupling_graph)

    def test_routed_circuit_uses_static_initial_mapping(
        self, line5_target: CompilationTarget
    ) -> None:
        """Provider operands stay adjacent under the static initial layout."""
        qc = QuantumCircuit(name="static_layout")
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.measure([0, 1, 2, 3], [3, 1, 0, 2])

        result, pi0, _ = SABRE(line5_target).solve(qc, iter_num=3, seed=0)

        assert _count_swaps(result) > 0
        assert _all_two_qubit_adjacent(result, pi0, line5_target.coupling_graph)
        measurement_positions = [
            index for index, instr in enumerate(result) if instr.is_measure
        ]
        assert measurement_positions == list(range(len(result) - 4, len(result)))
        assert len({instr.qubits[0] for instr in result if instr.is_measure}) == 4

        from quspect import Simulator

        expected = Simulator(seed=7).run(qc, shots=32).counts
        actual = Simulator(seed=7).run(result, shots=32).counts
        assert actual == expected

    def test_non_adjacent_pair_routed(self, line3_target: CompilationTarget) -> None:
        """cx(0,2) on 3-qubit line — routing may find a good initial layout."""
        sabre = SABRE(line3_target)
        qc = QuantumCircuit(name="nonadj")
        qc.h(0)
        qc.cx(0, 2)
        qc.measure_all()
        result, pi0, _ = sabre.solve(qc, iter_num=5, seed=0)
        assert _all_two_qubit_adjacent(result, pi0, line3_target.coupling_graph)

    def test_ring_topology(self, ring5_target: CompilationTarget) -> None:
        """Ring topology should need fewer SWAPs than linear."""
        sabre = SABRE(ring5_target)
        qc = QuantumCircuit(name="ring_test")
        qc.cx(0, 2)
        qc.cx(1, 4)
        result, pi0, _ = sabre.solve(qc, iter_num=3, seed=0)
        assert _all_two_qubit_adjacent(result, pi0, ring5_target.coupling_graph)

    def test_deterministic_with_seed(self, line5_target: CompilationTarget) -> None:
        """Same seed should produce same result."""
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="det")
        qc.cx(0, 1)
        qc.cx(2, 3)

        r1, _, _ = sabre.solve(qc, iter_num=3, seed=42)
        r2, _, _ = sabre.solve(qc, iter_num=3, seed=42)

        assert len(r1.instructions) == len(r2.instructions)
        for i1, i2 in zip(r1, r2, strict=False):
            if i1.is_gate and i2.is_gate:
                assert i1.gate == i2.gate
                assert i1.qubits == i2.qubits

    def test_routes_from_fixed_initial_mapping(
        self, line3_target: CompilationTarget
    ) -> None:
        """A non-adjacent fixed layout is retained and routed with a SWAP."""
        qc = QuantumCircuit(name="fixed_initial_mapping")
        qc.h(0).cx(0, 1).measure_all()

        result, pi0, _ = SABRE(line3_target).solve(
            qc,
            initial_mapping=[0, 2],
        )

        assert pi0[:2] == [0, 2]
        assert set(pi0) == {0, 1, 2}
        assert _count_swaps(result) > 0
        assert _all_two_qubit_adjacent(
            result,
            pi0,
            line3_target.coupling_graph,
        )

        from quspect import Simulator

        expected = Simulator(seed=9).run(qc, shots=64).counts
        actual = Simulator(seed=9).run(result, shots=64).counts
        assert actual == expected


# ---------------------------------------------------------------------------
# SABRE — edge cases
# ---------------------------------------------------------------------------


class TestSABREEdgeCases:
    def test_more_logical_than_physical(self, line3_target: CompilationTarget) -> None:
        """Should raise when logical qubits > physical qubits."""
        sabre = SABRE(line3_target)
        qc = QuantumCircuit(name="too_many")
        qc.cx(0, 1)
        qc.cx(2, 3)  # uses 4 logical qubits, target has 3
        with pytest.raises(ValueError, match="logical"):
            sabre.solve(qc, iter_num=1, seed=0)

    def test_single_qubit_circuit(self, line5_target: CompilationTarget) -> None:
        """Single-qubit circuit needs no SWAPs."""
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="single_q")
        qc.h(0)
        qc.rx(0.5, 0)
        qc.measure_all()
        result, _, _ = sabre.solve(qc, iter_num=1, seed=0)
        assert _count_swaps(result) == 0

    def test_all_qubits_used(self, line5_target: CompilationTarget) -> None:
        """Circuit using all 5 qubits with adjacent gates only."""
        sabre = SABRE(line5_target)
        qc = QuantumCircuit(name="all")
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        result, pi0, _ = sabre.solve(qc, iter_num=3, seed=0)
        assert _all_two_qubit_adjacent(result, pi0, line5_target.coupling_graph)

    @pytest.mark.parametrize(
        ("mapping", "error"),
        [
            ([0], ValueError),
            ([0, 0], ValueError),
            ([0, 3], ValueError),
            ([0, True], TypeError),
        ],
    )
    def test_invalid_initial_mapping_rejected(
        self,
        line3_target: CompilationTarget,
        mapping: list[int],
        error: type[Exception],
    ) -> None:
        qc = QuantumCircuit().cx(0, 1)

        with pytest.raises(error):
            SABRE(line3_target).solve(qc, initial_mapping=mapping)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_swaps(circuit: QuantumCircuit) -> int:
    return sum(
        1 for i in circuit if i.is_gate and i.gate is not None and i.gate.name == "swap"
    )


def _all_two_qubit_adjacent(
    circuit: QuantumCircuit,
    final_mapping: list[int],
    edges: set[tuple[int, int]],
) -> bool:
    """Check all two-qubit gates are adjacent under the final mapping.

    This is only valid when no SWAPs were inserted (static mapping).
    """
    for instr in circuit:
        if (
            instr.is_gate
            and instr.gate is not None
            and instr.gate.num_qubits >= 2
            and instr.gate.name != "swap"
        ):
            pq = [final_mapping[q] for q in instr.qubits]
            if (pq[0], pq[1]) not in edges and (pq[1], pq[0]) not in edges:
                return False
    return True


# ---------------------------------------------------------------------------
# Disconnected coupling graphs (B76)
# ---------------------------------------------------------------------------


class TestSABREDisconnected:
    """SABRE fails fast when interacting qubits cannot meet."""

    def test_mapping_across_components_raises(self) -> None:
        """A cx between two components raises ValueError immediately."""
        target = CompilationTarget.from_coupling_map(4, [(0, 1), (2, 3)])
        circ = QuantumCircuit().cx(0, 1)
        with pytest.raises(ValueError, match="connected component"):
            SABRE(target).solve(circ, initial_mapping=[0, 2])

    def test_transitive_interaction_across_components_raises(self) -> None:
        """q0-q1 and q1-q2 interact, so q0 and q2 must share a component."""
        target = CompilationTarget.from_coupling_map(4, [(0, 1), (2, 3)])
        circ = QuantumCircuit().cx(0, 1).cx(1, 2)
        with pytest.raises(ValueError, match="connected component"):
            SABRE(target).solve(circ, initial_mapping=[0, 1, 2])

    def test_group_too_large_for_any_component_raises(self) -> None:
        """Automatic layout raises when no component can host a group."""
        target = CompilationTarget.from_coupling_map(5, [(0, 1), (1, 2), (3, 4)])
        circ = QuantumCircuit().cx(0, 1).cx(1, 2).cx(2, 3)
        with pytest.raises(ValueError, match="connected component"):
            SABRE(target).solve(circ, seed=0)

    @pytest.mark.parametrize("seed", range(5))
    def test_automatic_layout_respects_components(self, seed: int) -> None:
        """Groups that fit are placed inside one component and routed."""
        target = CompilationTarget.from_coupling_map(5, [(0, 1), (1, 2), (3, 4)])
        circ = QuantumCircuit().cx(0, 1).cx(1, 2).cx(0, 2).cx(3, 4)
        routed, initial, _ = SABRE(target).solve(circ, seed=seed)
        comp = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        assert len({comp[initial[q]] for q in (0, 1, 2)}) == 1
        assert comp[initial[3]] == comp[initial[4]]
        for instr in routed.instructions:
            if instr.gate is not None and len(instr.qubits) == 2:
                a, b = (initial[q] for q in instr.qubits)
                assert (min(a, b), max(a, b)) in {(0, 1), (1, 2), (3, 4)}


# ---------------------------------------------------------------------------
# SABRE v2 — layout trials, release valve, determinism
# ---------------------------------------------------------------------------


def _line_target(n: int) -> CompilationTarget:
    return CompilationTarget.from_coupling_map(n, [(i, i + 1) for i in range(n - 1)])


def _cuccaro_adder(n: int) -> QuantumCircuit:
    """Cuccaro ripple-carry adder on ``n = 2k + 2`` qubits, lowered to 1q/2q."""
    from quspect.ir.compilation import decompose_deep

    k = (n - 2) // 2
    a = [2 * i + 1 for i in range(k)]
    b = [2 * i + 2 for i in range(k)]
    cin, cout = 0, 2 * k + 1
    qc = QuantumCircuit(name=f"adder_{n}")
    for q in a[::2] + b[1::2]:
        qc.x(q)

    def maj(x: int, y: int, z: int) -> None:
        qc.cx(z, y).cx(z, x).ccx(x, y, z)

    def uma(x: int, y: int, z: int) -> None:
        qc.ccx(x, y, z).cx(z, x).cx(x, y)

    maj(cin, b[0], a[0])
    for i in range(1, k):
        maj(a[i - 1], b[i], a[i])
    qc.cx(a[k - 1], cout)
    for i in range(k - 1, 0, -1):
        uma(a[i - 1], b[i], a[i])
    uma(cin, b[0], a[0])
    return decompose_deep(
        qc, supported_gates={"cx", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"}
    )


def _routed_gates_on_edges(
    routed: QuantumCircuit, pi0: list[int], target: CompilationTarget
) -> bool:
    """Every routed 2q gate (SWAPs included) acts on a coupling edge."""
    edges = target.coupling_graph
    for instr in routed:
        if instr.is_gate and len(instr.qubits) == 2:
            a, b = (pi0[q] for q in instr.qubits)
            if (a, b) not in edges and (b, a) not in edges:
                return False
    return True


class TestSABREv2:
    """Regression tests for the LightSABRE-style improvements."""

    @pytest.mark.parametrize(("n", "seed"), [(20, 0), (32, 1), (50, 0), (50, 1)])
    def test_adder_on_line_terminates(self, n: int, seed: int) -> None:
        """The release valve prevents the old livelock on line topologies."""
        target = _line_target(n)
        circuit = _cuccaro_adder(n)
        routed, pi0, _ = SABRE(target).solve(circuit, seed=seed)
        n_two_qubit = sum(1 for i in circuit if i.is_gate and len(i.qubits) == 2)
        assert _count_swaps(routed) <= n_two_qubit
        assert _routed_gates_on_edges(routed, pi0, target)

    def test_release_valve_forces_progress(self) -> None:
        """A threshold of one SWAP still routes everything correctly."""
        target = _line_target(12)
        circuit = _cuccaro_adder(12)
        circuit.measure_all()
        sabre = SABRE(target, release_valve_threshold=1)
        routed, pi0, _ = sabre.solve(circuit, seed=3)
        assert _routed_gates_on_edges(routed, pi0, target)

        from quspect import Simulator

        expected = Simulator(seed=1).run(circuit, shots=8).counts
        actual = Simulator(seed=1).run(routed, shots=8).counts
        assert actual == expected and len(expected) == 1

    @pytest.mark.parametrize("n", [8, 32])
    def test_ghz_chain_on_line_needs_no_swaps(self, n: int) -> None:
        qc = QuantumCircuit().h(0)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for seed in range(3):
            routed, _, _ = SABRE(_line_target(n)).solve(qc, seed=seed)
            assert _count_swaps(routed) == 0

    def test_deterministic_with_seed(self) -> None:
        target = CompilationTarget.from_coupling_map(
            9,
            [(r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)]
            + [(r * 3 + c, r * 3 + c + 3) for r in range(2) for c in range(3)],
        )
        from quspect.transpiler import generate_random_circuit

        circuit = generate_random_circuit(9, 20, seed=5)
        r1, a1, f1 = SABRE(target).solve(circuit, seed=11)
        r2, a2, f2 = SABRE(target).solve(circuit, seed=11)
        assert a1 == a2 and f1 == f2
        assert [(i.gate and i.gate.name, i.qubits) for i in r1] == [
            (i.gate and i.gate.name, i.qubits) for i in r2
        ]

    @pytest.mark.parametrize("trials", [0, -3])
    def test_layout_trials_must_be_positive(self, trials: int) -> None:
        qc = QuantumCircuit().cx(0, 1)
        with pytest.raises(ValueError, match="layout_trials"):
            SABRE(_line_target(3)).solve(qc, layout_trials=trials)

    @pytest.mark.parametrize("trials", [1.5, True, "4"])
    def test_layout_trials_must_be_int(self, trials: object) -> None:
        qc = QuantumCircuit().cx(0, 1)
        with pytest.raises(TypeError, match="layout_trials"):
            SABRE(_line_target(3)).solve(qc, layout_trials=trials)  # type: ignore[arg-type]

    @pytest.mark.parametrize("trials", [1, 3])
    def test_routed_gates_on_coupling_edges(self, trials: int) -> None:
        from quspect.transpiler import generate_random_circuit

        target = CompilationTarget.from_coupling_map(
            7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 5)]
        )
        circuit = generate_random_circuit(6, 25, seed=2)
        routed, pi0, final = SABRE(target).solve(circuit, seed=4, layout_trials=trials)
        assert _routed_gates_on_edges(routed, pi0, target)
        assert len(final) == len(pi0)
        assert len(set(pi0)) == len(pi0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"extended_set_size": -1},
            {"decay_reset_interval": 0},
            {"release_valve_threshold": 0},
        ],
    )
    def test_invalid_constructor_parameters(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            SABRE(_line_target(3), **kwargs)  # type: ignore[arg-type]
