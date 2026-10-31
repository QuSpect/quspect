"""Differential tests: the unified routing core equals the round-7 loops.

:mod:`quspect.transpiler._routing` replaced three routing loops
(``SABRE._route`` for two-qubit DAGs, ``route_with_toffoli`` for DAGs with
``ccx`` nodes and the commutation-aware variant with the SWAP-fusion
bonus).  ``_legacy_routing_r7`` keeps a frozen copy of the old code;
:class:`LegacySABRE` plugs it into the current :class:`SABRE`, so every
comparison below runs the same layout search, elision and replay and only
the routing pass (and the routing-DAG builder) differ.  Outputs must be
identical: routing events, final permutation, SWAP count and depth for
single passes; instruction lists and mappings for ``solve`` and
``route_candidates``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget
from quspect.transpiler._mapping import (
    _build_routing_dag,
    _extract_gates,
    _Prepared,
)
from tests.transpiler._legacy_routing_r7 import (
    legacy_build_routing_dag,
    legacy_route,
)

BENCHMARK_ROOT = Path("/Users/quspect/xuxs/quspect_benchmark")


class LegacySABRE(SABRE):
    """SABRE with the frozen round-7 routing pass and routing-DAG builder."""

    def _route(self, layout: list[int], dag: Any, **kwargs: Any) -> Any:
        return legacy_route(self, layout, dag, **kwargs)

    def _prepare(self, circuit: QuantumCircuit, elide: bool) -> _Prepared:
        prep = super()._prepare(circuit, elide)
        legacy = legacy_build_routing_dag(
            prep.gates,
            prep.num_logical,
            circuit,
            commutation=self.commutation_aware
            and not any(g.q3 >= 0 for g in prep.gates),
        )
        _assert_same_dag(prep.dag, legacy)
        prep.dag = legacy
        return prep


def _assert_same_dag(new: Any, old: Any) -> None:
    assert new.nq1 == old.nq1
    assert new.nq2 == old.nq2
    assert new.nq3 == old.nq3
    assert new.has_three_qubit == old.has_three_qubit
    assert new.succ == old.succ
    assert new.pred == old.pred
    assert [[g.index for g in p] for p in new.prefix] == [
        [g.index for g in p] for p in old.prefix
    ]
    assert [g.index for g in new.trailing] == [g.index for g in old.trailing]


# =============================================================================
# Corpus
# =============================================================================


def _line(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _ring(n: int) -> list[tuple[int, int]]:
    return [*_line(n), (0, n - 1)]


def _grid(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def _heavy_square(cols: int) -> tuple[int, list[tuple[int, int]]]:
    """Two rows joined by bridge qubits every other column (heavy-hex-like)."""
    top = list(range(cols))
    bottom = list(range(cols, 2 * cols))
    edges = _line(cols) + [(a + cols, b + cols) for a, b in _line(cols)]
    n = 2 * cols
    for c in range(0, cols, 2):
        edges += [(top[c], n), (n, bottom[c])]
        n += 1
    return n, edges


def _random_connected(n: int, rng: random.Random) -> list[tuple[int, int]]:
    edges = [(rng.randrange(i), i) for i in range(1, n)]
    for _ in range(n // 3):
        a, b = rng.sample(range(n), 2)
        edges.append((min(a, b), max(a, b)))
    return sorted(set(edges))


TARGETS: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "line7": (7, _line(7)),
    "ring8": (8, _ring(8)),
    "grid3x4": (12, _grid(3, 4)),
    "heavy": _heavy_square(5),
    "random10": (10, _random_connected(10, random.Random(5))),
    "full5": (5, [(i, j) for i in range(5) for j in range(i + 1, 5)]),
    "tri_tail": (5, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)]),
}

# Two components: 0-1-2-3 and 4-5-6 (circuits keep interactions inside halves).
DISCONNECTED = (7, [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)])


def _random_circuit(
    n: int,
    rng: random.Random,
    *,
    depth: int,
    ccx: bool,
    measure: bool,
    groups: list[list[int]] | None = None,
) -> QuantumCircuit:
    qc = QuantumCircuit()
    pools = groups or [list(range(n))]
    kinds = ["1q", "1q", "cx", "cx", "cz", "cp", "crz", "swap"]
    if ccx:
        kinds += ["ccx", "ccx"]
    if measure:
        kinds.append("measure")
    for _ in range(depth):
        pool = rng.choice(pools)
        kind = rng.choice(kinds)
        if kind == "ccx" and len(pool) < 3:
            kind = "cx"
        if kind in ("cx", "cz", "cp", "crz", "swap") and len(pool) < 2:
            kind = "1q"
        if kind == "1q":
            q = rng.choice(pool)
            g = rng.choice(["h", "x", "t", "rz", "rx"])
            if g == "rz":
                qc.rz(rng.uniform(-3, 3), q)
            elif g == "rx":
                qc.rx(rng.uniform(-3, 3), q)
            else:
                getattr(qc, g)(q)
        elif kind == "ccx":
            a, b, c = rng.sample(pool, 3)
            qc.ccx(a, b, c)
        elif kind == "measure":
            q = rng.choice(pool)
            qc.measure([q], [q])
        else:
            a, b = rng.sample(pool, 2)
            if kind == "cp":
                qc.cp(rng.uniform(-3, 3), a, b)
            elif kind == "crz":
                qc.crz(rng.uniform(-3, 3), a, b)
            else:
                getattr(qc, kind)(a, b)
    return qc


def _signature(circuit: QuantumCircuit) -> list[tuple[Any, ...]]:
    return [
        (
            str(i.type),
            None if i.gate is None else i.gate.name,
            tuple(i.qubits),
            () if i.gate is None else tuple(repr(float(p)) for p in i.gate.params),
            tuple(i.clbits),
        )
        for i in circuit
    ]


SABRE_OPTIONS: list[dict[str, Any]] = [
    {},
    {"commutation_aware": True},
    {"commutation_aware": True, "swap_fusion_bonus": 0.3},
    {"swap_fusion_bonus": 0.3},
    {"release_valve_threshold": 2},
    {"release_valve_threshold": 1, "commutation_aware": True},
    {"extended_set_size": 0},
    {"lookahead_weight": 1.3, "decay_delta": 0.05, "decay_reset_interval": 2},
    {"permutation_synthesis": True},
]


def _cases(
    count: int, seed: int
) -> list[tuple[str, int, QuantumCircuit, dict[str, Any]]]:
    rng = random.Random(seed)
    out = []
    names = sorted(TARGETS)
    for i in range(count):
        tname = names[i % len(names)]
        n_phys, _ = TARGETS[tname]
        n = rng.randint(3, n_phys)
        qc = _random_circuit(
            n,
            rng,
            depth=rng.randint(5, 60),
            ccx=i % 3 != 0,
            measure=i % 5 == 0,
        )
        out.append((tname, i, qc, SABRE_OPTIONS[i % len(SABRE_OPTIONS)]))
    return out


def _target(name: str) -> CompilationTarget:
    n, edges = TARGETS[name]
    return CompilationTarget.from_coupling_map(
        n, edges, native_gates={"cz", "h", "rz", "x", "s", "sdg", "t", "tdg"}
    )


def _assert_same_solve(
    target: CompilationTarget,
    circuit: QuantumCircuit,
    options: dict[str, Any],
    **kwargs: Any,
) -> None:
    new = SABRE(target, **options).solve(circuit, **kwargs)
    old = LegacySABRE(target, **options).solve(circuit, **kwargs)
    assert _signature(new[0]) == _signature(old[0])
    assert new[1] == old[1]
    assert new[2] == old[2]


def _assert_same_candidates(
    target: CompilationTarget,
    circuit: QuantumCircuit,
    options: dict[str, Any],
    **kwargs: Any,
) -> None:
    new = SABRE(target, **options).route_candidates(circuit, **kwargs)
    old = LegacySABRE(target, **options).route_candidates(circuit, **kwargs)
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert _signature(a.circuit) == _signature(b.circuit)
        assert (a.initial_mapping, a.final_mapping) == (
            b.initial_mapping,
            b.final_mapping,
        )
        assert (a.num_swaps, a.depth, a.estimated_two_qubit_gates) == (
            b.num_swaps,
            b.depth,
            b.estimated_two_qubit_gates,
        )
        assert (a.label, a.is_default) == (b.label, b.is_default)


# =============================================================================
# Single routing passes
# =============================================================================


@pytest.mark.parametrize("seed", range(4))
def test_single_passes_identical(seed: int) -> None:
    """Random layouts, both directions: events, permutation, swaps, depth."""
    for tname, i, qc, options in _cases(40, seed):
        target = _target(tname)
        gates, n_log = _extract_gates(qc)
        # SABRE never builds a commutation-aware DAG with ``ccx`` nodes (the
        # round-7 Toffoli loop assumed one front node per qubit), so that
        # combination is not part of the contract.
        has_ccx = any(g.q3 >= 0 for g in gates)
        for commutation in (False,) if has_ccx else (False, True):
            dag = _build_routing_dag(gates, n_log, qc, commutation=commutation)
            _assert_same_dag(
                dag, legacy_build_routing_dag(gates, n_log, qc, commutation=commutation)
            )
            sabre = SABRE(target, **options)
            rng = random.Random(seed * 1000 + i)
            for _ in range(3):
                layout = list(range(target.num_qubits))
                rng.shuffle(layout)
                for forward in (True, False):
                    s = rng.getrandbits(32)
                    new = sabre._route(
                        layout, dag, forward=forward, rng=random.Random(s), record=True
                    )
                    old = legacy_route(
                        sabre,
                        layout,
                        dag,
                        forward=forward,
                        rng=random.Random(s),
                        record=True,
                    )
                    assert new.events == old.events, (tname, i, options)
                    assert new.final_pi == old.final_pi
                    assert (new.num_swaps, new.depth) == (old.num_swaps, old.depth)


# =============================================================================
# Full layout search + routing
# =============================================================================


@pytest.mark.parametrize("seed", range(3))
def test_solve_identical_random_corpus(seed: int) -> None:
    for tname, _, qc, options in _cases(45, 100 + seed):
        _assert_same_solve(_target(tname), qc, options, seed=seed, layout_trials=3)


@pytest.mark.parametrize("seed", range(2))
def test_route_candidates_identical_random_corpus(seed: int) -> None:
    for tname, _, qc, options in _cases(27, 200 + seed):
        _assert_same_candidates(
            _target(tname),
            qc,
            options,
            seed=seed,
            layout_trials=4,
            swap_trials=2,
            reference_trials=2,
            exact_layout=seed == 0,
        )


def test_initial_mapping_identical() -> None:
    rng = random.Random(7)
    for tname, i, qc, options in _cases(36, 300):
        target = _target(tname)
        mapping = rng.sample(range(target.num_qubits), qc.num_qubits)
        _assert_same_solve(target, qc, options, initial_mapping=mapping)
        if i % 3 == 0:
            _assert_same_candidates(
                target, qc, options, initial_mapping=mapping, seed=i, swap_trials=3
            )


def test_release_valve_triggers_identical() -> None:
    """Tiny valve thresholds force the valve on 2q and ccx front nodes."""
    rng = random.Random(11)
    target = _target("line7")
    for k in range(30):
        qc = _random_circuit(7, rng, depth=25, ccx=k % 2 == 0, measure=k % 4 == 0)
        for threshold in (1, 2, 3):
            _assert_same_solve(
                target,
                qc,
                {"release_valve_threshold": threshold},
                seed=k,
                layout_trials=2,
            )


def test_disconnected_target_identical() -> None:
    n, edges = DISCONNECTED
    target = CompilationTarget.from_coupling_map(n, edges)
    rng = random.Random(3)
    for k in range(30):
        qc = _random_circuit(
            6,
            rng,
            depth=30,
            ccx=k % 2 == 0,
            measure=k % 3 == 0,
            groups=[[0, 1, 2], [3, 4, 5]],
        )
        options = SABRE_OPTIONS[k % len(SABRE_OPTIONS)]
        _assert_same_solve(target, qc, options, seed=k, layout_trials=3)


def test_disconnected_error_identical() -> None:
    """Both versions fail the same way on an unroutable fixed mapping."""
    n, edges = DISCONNECTED
    target = CompilationTarget.from_coupling_map(n, edges)
    qc = QuantumCircuit().cx(0, 1)
    for cls in (SABRE, LegacySABRE):
        with pytest.raises(ValueError, match="Unroutable"):
            cls(target).solve(qc, initial_mapping=[0, 5])


# =============================================================================
# Benchmark circuits
# =============================================================================


def _benchmark_inputs() -> list[tuple[str, str]]:
    if not (BENCHMARK_ROOT / "compilation" / "circuits.py").exists():
        pytest.skip("quspect_benchmark checkout not available")
    pytest.importorskip("qiskit")
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    from compilation import circuits

    out = [(bc.name, bc.qasm) for bc in circuits.load_qasm_dir()]
    for fam in circuits.GENERATORS:
        bc = circuits.generate(fam, 12)
        out.append((bc.name, bc.qasm))
    return out


def _benchmark_target(kind: str, n: int) -> CompilationTarget:
    from compilation import topologies

    topo = topologies.build(kind, n)
    return CompilationTarget.from_coupling_map(
        topo.num_qubits,
        list(topo.edges),
        native_gates={"cz", "h", "rz", "s", "sdg", "t", "tdg", "x", "y", "z"},
    )


@pytest.mark.parametrize("kind", ["line", "grid", "heavy_hex"])
def test_benchmark_circuits_identical(kind: str) -> None:
    """Benchmark circuits (QASM set + generators) at the routing stage.

    Routes the pre-routing (``ccx`` kept) circuit as level 2 does and the
    CX-level circuit on the commutation-aware DAG as level 3 does.
    """
    from quspect.ir import qasm2_loads
    from quspect.transpiler._passes import DecomposePass
    from quspect.transpiler._transpile import _COMM_FUSION, _COMMUTING_KEEP

    for _, qasm in _benchmark_inputs():
        circuit = qasm2_loads(qasm)
        target = _benchmark_target(kind, circuit.num_qubits)
        logical = DecomposePass(target, keep={"ccx"}).run(circuit)
        _assert_same_solve(
            target, logical, {"permutation_synthesis": True}, seed=0, layout_trials=2
        )
        comm = DecomposePass(target, keep=_COMMUTING_KEEP).run(circuit)
        _assert_same_solve(
            target,
            comm,
            {"commutation_aware": True, "swap_fusion_bonus": _COMM_FUSION},
            seed=1,
            layout_trials=1,
            exact_layout=False,
        )
