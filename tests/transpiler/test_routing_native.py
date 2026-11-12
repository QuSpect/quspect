"""Differential tests: the native (C++) routing core equals the Python one.

:func:`quspect.transpiler._routing.route_dag` dispatches to the C++ core
when the extension module is built.  These tests compare it bit for bit
with the pure-Python reference implementation (forced through
``QUSPECT_ROUTING_BACKEND=python``):

* the CPython-compatible Mersenne Twister (seeding, ``getrandbits``,
  ``_randbelow``) on many seeds;
* single routing passes on the randomized corpus of
  ``test_routing_differential`` (two-qubit, ``ccx``, commutation-aware,
  SWAP fusion, release valve, measurements), including the caller's random
  state after the pass;
* full ``solve`` / ``route_candidates`` runs and end-to-end
  ``transpile(level=2/3)`` on random and benchmark circuits.
"""

from __future__ import annotations

import random
import sys
from typing import Any

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget, _routing, transpile
from quspect.transpiler._mapping import _build_routing_dag, _extract_gates
from quspect.transpiler._routing import ROUTING_BACKEND_ENV, route_dag
from tests.transpiler.test_routing_differential import (
    BENCHMARK_ROOT,
    DISCONNECTED,
    _benchmark_inputs,
    _benchmark_target,
    _cases,
    _random_circuit,
    _signature,
    _target,
)

CORE = _routing._native_core()
pytestmark = pytest.mark.skipif(CORE is None, reason="native core not built")


def _with_backend(backend: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(ROUTING_BACKEND_ENV, backend)
        return fn(*args, **kwargs)


# =============================================================================
# CPython Mersenne Twister
# =============================================================================


def test_seed_state_matches_cpython() -> None:
    rng = random.Random(2024)
    seeds = [0, 1, 2**32 - 1, 2**32, 2**64 - 1]
    seeds += [rng.getrandbits(rng.choice([8, 31, 33, 64])) for _ in range(200)]
    for seed in seeds:
        assert (
            CORE._cpython_random_seed_state(seed) == random.Random(seed).getstate()[1]
        )


@pytest.mark.parametrize("k", [1, 2, 3, 7, 31, 32, 33, 53, 63, 64])
def test_getrandbits_matches_cpython(k: int) -> None:
    for seed in range(60):
        ref = random.Random(seed * 7919 + k)
        for _ in range(seed % 5):  # start from different state indices
            ref.getrandbits(32)
        state = ref.getstate()[1]
        draws, new_state = CORE._cpython_random_draws(state, k, 0, 700)
        assert draws == [ref.getrandbits(k) for _ in range(700)]
        assert new_state == ref.getstate()[1]


def test_randbelow_and_choice_match_cpython() -> None:
    gen = random.Random(99)
    for seed in range(300):
        n = gen.choice([2, 3, 5, 6, 7, 8, 9, 31, 64, 100, 1000, 2**31 + 5, 2**40])
        ref = random.Random(seed)
        state = ref.getstate()[1]
        draws, new_state = CORE._cpython_random_draws(state, 0, n, 50)
        assert draws == [ref._randbelow(n) for _ in range(50)]
        assert new_state == ref.getstate()[1]
        # choice(seq) is seq[_randbelow(len(seq))]
        ref2 = random.Random(seed)
        seq = list(range(n)) if n < 2000 else None
        if seq is not None:
            assert draws[:10] == [ref2.choice(seq) for _ in range(10)]


# =============================================================================
# Single routing passes
# =============================================================================


def _compare_pass(
    sabre: SABRE, layout: list[int], dag: Any, forward: bool, seed: int
) -> None:
    rng_native = random.Random(seed)
    rng_python = random.Random(seed)
    native = _with_backend(
        "native",
        route_dag,
        sabre,
        layout,
        dag,
        forward=forward,
        rng=rng_native,
        record=True,
    )
    python = _with_backend(
        "python",
        route_dag,
        sabre,
        layout,
        dag,
        forward=forward,
        rng=rng_python,
        record=True,
    )
    assert native.events == python.events
    assert native.final_pi == python.final_pi
    assert (native.num_swaps, native.depth) == (python.num_swaps, python.depth)
    assert rng_native.getstate() == rng_python.getstate()


@pytest.mark.parametrize("seed", range(4))
def test_single_passes_identical(seed: int) -> None:
    for tname, i, qc, options in _cases(40, 500 + seed):
        target = _target(tname)
        gates, n_log = _extract_gates(qc)
        has_ccx = any(g.q3 >= 0 for g in gates)
        for commutation in (False,) if has_ccx else (False, True):
            dag = _build_routing_dag(gates, n_log, qc, commutation=commutation)
            sabre = SABRE(target, **options)
            rng = random.Random(seed * 1000 + i)
            for _ in range(3):
                layout = list(range(target.num_qubits))
                rng.shuffle(layout)
                for forward in (True, False):
                    _compare_pass(sabre, layout, dag, forward, rng.getrandbits(64))


def test_release_valve_passes_identical() -> None:
    rng = random.Random(17)
    target = _target("grid3x4")
    for k in range(40):
        qc = _random_circuit(12, rng, depth=40, ccx=k % 2 == 0, measure=k % 4 == 0)
        gates, n_log = _extract_gates(qc)
        dag = _build_routing_dag(gates, n_log, qc)
        for threshold in (1, 2, 3, 5):
            sabre = SABRE(target, release_valve_threshold=threshold)
            layout = list(range(12))
            rng.shuffle(layout)
            _compare_pass(sabre, layout, dag, k % 3 != 0, k)


def test_record_false_identical() -> None:
    for tname, i, qc, options in _cases(20, 77):
        target = _target(tname)
        gates, n_log = _extract_gates(qc)
        dag = _build_routing_dag(gates, n_log, qc)
        sabre = SABRE(target, **options)
        layout = list(range(target.num_qubits))
        random.Random(i).shuffle(layout)
        results = []
        for backend in ("native", "python"):
            rng = random.Random(i)
            res = _with_backend(
                backend, route_dag, sabre, layout, dag, forward=True, rng=rng
            )
            results.append((res.events, res.final_pi, res.num_swaps, res.depth))
            results.append(rng.getstate())
        assert results[0] == results[2]
        assert results[1] == results[3]


def test_disconnected_error_identical() -> None:
    n, edges = DISCONNECTED
    target = CompilationTarget.from_coupling_map(n, edges)
    qc = QuantumCircuit().cx(0, 1)
    gates, n_log = _extract_gates(qc)
    dag = _build_routing_dag(gates, n_log, qc)
    layout = [0, 5, 1, 2, 3, 4, 6]
    for backend in ("native", "python"):
        with pytest.raises(ValueError, match="No routing SWAP is available"):
            _with_backend(
                backend,
                route_dag,
                SABRE(target),
                layout,
                dag,
                forward=True,
                rng=random.Random(0),
            )


# =============================================================================
# Backend selection
# =============================================================================


def test_backend_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _target("line7")
    qc = QuantumCircuit().cx(0, 6).cx(1, 5)
    gates, n_log = _extract_gates(qc)
    dag = _build_routing_dag(gates, n_log, qc)
    layout = list(range(7))
    monkeypatch.setenv(ROUTING_BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match=ROUTING_BACKEND_ENV):
        route_dag(SABRE(target), layout, dag, forward=True, rng=random.Random(0))
    monkeypatch.setenv(ROUTING_BACKEND_ENV, "python")
    assert _routing.routing_backend() == "python"
    monkeypatch.setenv(ROUTING_BACKEND_ENV, "native")
    assert _routing.routing_backend() == "native"
    monkeypatch.setattr(_routing, "_native_core", lambda: None)
    with pytest.raises(RuntimeError, match="native routing core"):
        route_dag(SABRE(target), layout, dag, forward=True, rng=random.Random(0))
    monkeypatch.setenv(ROUTING_BACKEND_ENV, "auto")
    assert _routing.routing_backend() == "python"
    res = route_dag(SABRE(target), layout, dag, forward=True, rng=random.Random(0))
    assert res.num_swaps > 0


def test_random_subclass_uses_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Random subclass may override choice(): the Python loop is used."""

    class Last(random.Random):
        def choice(self, seq: Any) -> Any:
            return seq[-1]

    target = _target("ring8")
    qc = QuantumCircuit().cx(0, 4).cx(2, 6).cx(1, 5)
    gates, n_log = _extract_gates(qc)
    dag = _build_routing_dag(gates, n_log, qc)
    called = []
    orig = _routing._route_dag_python

    def spy(*args: Any, **kwargs: Any) -> Any:
        called.append(True)
        return orig(*args, **kwargs)

    monkeypatch.setattr(_routing, "_route_dag_python", spy)
    route_dag(SABRE(target), list(range(8)), dag, forward=True, rng=Last(1))
    assert called


# =============================================================================
# Layout search, candidates and transpile
# =============================================================================


@pytest.mark.parametrize("seed", range(2))
def test_solve_and_candidates_identical(seed: int) -> None:
    for tname, _, qc, options in _cases(30, 600 + seed):
        target = _target(tname)
        native = _with_backend(
            "native", SABRE(target, **options).solve, qc, seed=seed, layout_trials=3
        )
        python = _with_backend(
            "python", SABRE(target, **options).solve, qc, seed=seed, layout_trials=3
        )
        assert _signature(native[0]) == _signature(python[0])
        assert native[1:] == python[1:]
        kwargs = {"seed": seed, "layout_trials": 3, "swap_trials": 2}
        cn = _with_backend(
            "native", SABRE(target, **options).route_candidates, qc, **kwargs
        )
        cp = _with_backend(
            "python", SABRE(target, **options).route_candidates, qc, **kwargs
        )
        assert [(_signature(c.circuit), c.label, c.initial_mapping) for c in cn] == [
            (_signature(c.circuit), c.label, c.initial_mapping) for c in cp
        ]


def _tsig(result: Any) -> tuple[Any, ...]:
    return (
        _signature(result.circuit),
        list(result.initial_layout),
        list(result.final_layout),
    )


# 第 3 级的两个端到端比较很慢（两种后端各编译一遍）：完整语料 / 完整基准集标记为
# slow（每晚全量时运行），PR 上跑下面下标给出的小子集（不同拓扑、不同规模）。
_QUICK_CORPUS = (1, 5, 11)  # grid3x4、ring8、random10
_QUICK_BENCHMARKS = (3, 4, 5)  # grid、heavy_hex、line 各一个

_LEVEL3_SPLIT = pytest.mark.parametrize(
    ("level", "quick"),
    [
        pytest.param(2, False, id="2"),
        pytest.param(3, True, id="3-quick"),
        pytest.param(3, False, id="3", marks=pytest.mark.slow),
    ],
)


@_LEVEL3_SPLIT
def test_transpile_random_corpus_identical(level: int, quick: bool) -> None:
    cases = _cases(18, 700 + level)
    if quick:
        cases = [cases[k] for k in _QUICK_CORPUS]
    for tname, i, qc, _ in cases:
        target = _target(tname)
        kwargs: dict[str, Any] = {"optimization_level": level, "seed": i}
        if level == 3:
            kwargs["work_limit"] = 20_000
        native = _with_backend("native", transpile, qc, target, **kwargs)
        python = _with_backend("python", transpile, qc, target, **kwargs)
        assert _tsig(native) == _tsig(python), (tname, i)


@pytest.mark.parametrize("kind", ["line", "grid", "heavy_hex"])
def test_benchmark_circuits_identical(kind: str) -> None:
    """Benchmark QASM set + generators: SABRE solve as level 2/3 route them."""
    from quspect.ir import qasm2_loads
    from quspect.transpiler._passes import DecomposePass
    from quspect.transpiler._transpile import _COMM_FUSION, _COMMUTING_KEEP

    for _, qasm in _benchmark_inputs():
        circuit = qasm2_loads(qasm)
        target = _benchmark_target(kind, circuit.num_qubits)
        logical = DecomposePass(target, keep={"ccx"}).run(circuit)
        comm = DecomposePass(target, keep=_COMMUTING_KEEP).run(circuit)
        for inp, options, kw in (
            (logical, {"permutation_synthesis": True}, {"layout_trials": 3}),
            (
                comm,
                {"commutation_aware": True, "swap_fusion_bonus": _COMM_FUSION},
                {"layout_trials": 2, "exact_layout": False},
            ),
        ):
            native = _with_backend(
                "native", SABRE(target, **options).solve, inp, seed=3, **kw
            )
            python = _with_backend(
                "python", SABRE(target, **options).solve, inp, seed=3, **kw
            )
            assert _signature(native[0]) == _signature(python[0])
            assert native[1:] == python[1:]


def _benchmark_small_qasm(max_qubits: int, limit: int) -> list[tuple[str, str]]:
    from quspect.ir import qasm2_loads

    out = []
    for name, qasm in _benchmark_inputs():
        if qasm2_loads(qasm).num_qubits <= max_qubits:
            out.append((name, qasm))
    return out[:: max(1, len(out) // limit)][:limit]


@_LEVEL3_SPLIT
def test_transpile_benchmark_identical(level: int, quick: bool) -> None:
    """End-to-end transpile gate lists on benchmark circuits (both backends)."""
    from quspect.ir import qasm2_loads

    for idx, (name, qasm) in enumerate(_benchmark_small_qasm(16, 10)):
        if quick and idx not in _QUICK_BENCHMARKS:
            continue
        circuit = qasm2_loads(qasm)
        kind = ("grid", "heavy_hex", "line")[idx % 3]
        target = _benchmark_target(kind, circuit.num_qubits)
        kwargs: dict[str, Any] = {"optimization_level": level, "seed": 0}
        if level == 3:
            kwargs["work_limit"] = 30_000
        native = _with_backend("native", transpile, circuit, target, **kwargs)
        python = _with_backend("python", transpile, circuit, target, **kwargs)
        assert _tsig(native) == _tsig(python), (name, kind)


LARGE = [
    "qiskit_quantum_volume_48",
    "qiskit_heisenberg_2d_trotter_64",
    "qiskit_qaoa_3reg_64",
    "qiskit_hrs_multiplier_41",
]


@pytest.mark.parametrize("name", LARGE)
def test_large_benchmark_circuits_identical(name: str) -> None:
    """A few 40–64 qubit circuits from ``compilation/qasm_large``."""
    path = BENCHMARK_ROOT / "compilation" / "qasm_large" / f"{name}.qasm"
    if not path.exists():
        pytest.skip("quspect_benchmark large set not available")
    pytest.importorskip("qiskit")
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    from compilation import circuits

    from quspect.ir import qasm2_loads
    from quspect.transpiler._passes import DecomposePass
    from quspect.transpiler._transpile import _COMM_FUSION, _COMMUTING_KEEP

    circuit = qasm2_loads(circuits.normalize_qasm(path.read_text()))
    for kind in ("grid", "heavy_hex"):
        target = _benchmark_target(kind, circuit.num_qubits)
        logical = DecomposePass(target, keep={"ccx"}).run(circuit)
        comm = DecomposePass(target, keep=_COMMUTING_KEEP).run(circuit)
        for inp, options in (
            (logical, {"permutation_synthesis": True}),
            (comm, {"commutation_aware": True, "swap_fusion_bonus": _COMM_FUSION}),
        ):
            kw = {"seed": 0, "layout_trials": 1, "iter_num": 1, "exact_layout": False}
            native = _with_backend("native", SABRE(target, **options).solve, inp, **kw)
            python = _with_backend("python", SABRE(target, **options).solve, inp, **kw)
            assert _signature(native[0]) == _signature(python[0]), (name, kind)
            assert native[1:] == python[1:]


# =============================================================================
# Mirror-gate-aware routing and block nodes
# =============================================================================

MIRROR_OPTIONS: list[dict[str, Any]] = [
    {"mirror_gates": True},
    {"mirror_gates": True, "block_nodes": True},
    {
        "mirror_gates": True,
        "block_nodes": True,
        "mirror_threshold": 0.25,
        "swap_fusion_bonus": 0.2,
    },
    {"mirror_gates": True, "mirror_cz_penalty": 2.0, "swap_fusion_bonus": 0.3},
    {"mirror_gates": True, "block_nodes": True, "release_valve_threshold": 2},
    {"block_nodes": True},
]


@pytest.mark.parametrize("seed", range(3))
def test_mirror_passes_identical(seed: int) -> None:
    from quspect.transpiler._mapping import _annotate_mirror_costs

    mirrors = 0
    for tname, i, qc, _ in _cases(40, 800 + seed):
        target = _target(tname)
        gates, n_log = _extract_gates(qc)
        options = MIRROR_OPTIONS[i % len(MIRROR_OPTIONS)]
        dag = _build_routing_dag(
            gates, n_log, qc, merge_pairs=options.get("block_nodes", False)
        )
        _annotate_mirror_costs(dag, gates, qc)
        sabre = SABRE(target, **options)
        rng = random.Random(seed * 1000 + i)
        for _ in range(2):
            layout = list(range(target.num_qubits))
            rng.shuffle(layout)
            for forward in (True, False):
                s = rng.getrandbits(64)
                rn, rp = random.Random(s), random.Random(s)
                native = _with_backend(
                    "native",
                    route_dag,
                    sabre,
                    layout,
                    dag,
                    forward=forward,
                    rng=rn,
                    record=True,
                )
                python = _with_backend(
                    "python",
                    route_dag,
                    sabre,
                    layout,
                    dag,
                    forward=forward,
                    rng=rp,
                    record=True,
                )
                assert native == python, (tname, i, options)
                assert rn.getstate() == rp.getstate()
                mirrors += native.num_mirrors
    assert mirrors > 0


@pytest.mark.parametrize("seed", range(2))
def test_mirror_solve_and_candidates_identical(seed: int) -> None:
    for tname, i, qc, _ in _cases(24, 850 + seed):
        target = _target(tname)
        options = {
            **MIRROR_OPTIONS[i % len(MIRROR_OPTIONS)],
            "permutation_synthesis": True,
        }
        native = _with_backend(
            "native", SABRE(target, **options).solve, qc, seed=seed, layout_trials=3
        )
        python = _with_backend(
            "python", SABRE(target, **options).solve, qc, seed=seed, layout_trials=3
        )
        assert _signature(native[0]) == _signature(python[0])
        assert native[1:] == python[1:]
        kwargs = {"seed": seed, "layout_trials": 2, "swap_trials": 2}
        cn = _with_backend(
            "native", SABRE(target, **options).route_candidates, qc, **kwargs
        )
        cp = _with_backend(
            "python", SABRE(target, **options).route_candidates, qc, **kwargs
        )
        assert [(_signature(c.circuit), c.label, c.num_swaps) for c in cn] == [
            (_signature(c.circuit), c.label, c.num_swaps) for c in cp
        ]


@pytest.mark.parametrize("name", LARGE[:3])
def test_large_benchmark_mirror_identical(name: str) -> None:
    """Large QV / Heisenberg / QAOA circuits routed as level 2/3 do (mirrors on)."""
    path = BENCHMARK_ROOT / "compilation" / "qasm_large" / f"{name}.qasm"
    if not path.exists():
        pytest.skip("quspect_benchmark large set not available")
    pytest.importorskip("qiskit")
    if str(BENCHMARK_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCHMARK_ROOT))
    from compilation import circuits

    from quspect.ir import qasm2_loads
    from quspect.transpiler._consolidate import ConsolidateBlocksPass
    from quspect.transpiler._passes import DecomposePass
    from quspect.transpiler._transpile import _mirror_options

    circuit = qasm2_loads(circuits.normalize_qasm(path.read_text()))
    for kind in ("line", "heavy_hex"):
        target = _benchmark_target(kind, circuit.num_qubits)
        pre = DecomposePass(target, keep={"ccx"})
        logical = pre.run(ConsolidateBlocksPass(target).run(pre.run(circuit)))
        options = {"permutation_synthesis": True, **_mirror_options(2)}
        kw = {"seed": 0, "layout_trials": 1, "iter_num": 1, "exact_layout": False}
        native = _with_backend("native", SABRE(target, **options).solve, logical, **kw)
        python = _with_backend("python", SABRE(target, **options).solve, logical, **kw)
        assert _signature(native[0]) == _signature(python[0]), (name, kind)
        assert native[1:] == python[1:]


# =============================================================================
# Batched routing chains and native event replay
# =============================================================================


@pytest.mark.parametrize("threads", ["1", "3", "8"])
def test_route_chains_match_serial_routing(
    threads: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quspect.transpiler import _mapping

    monkeypatch.setenv("QUSPECT_COMPILE_THREADS", threads)
    for tname, _, qc, options in _cases(6, 900):
        target = _target(tname)
        router = SABRE(target, **options)
        prep = router._prepare(qc, True)
        n = target.num_qubits
        rng = random.Random(5)
        chains = []
        for k in range(5):
            layout = list(range(n))
            rng.shuffle(layout)
            chains.append((layout, k % 3, rng.getrandbits(70)))
        got = _routing.route_chains(router, prep.dag, chains)
        assert got is not None
        for (layout, iters, seed), item in zip(chains, got, strict=True):
            r = random.Random(seed)
            swaps = []
            for _ in range(iters):
                fwd = route_dag(router, layout, prep.dag, forward=True, rng=r)
                swaps.append(fwd.num_swaps)
                bwd = route_dag(router, fwd.final_pi, prep.dag, forward=False, rng=r)
                swaps.append(bwd.num_swaps)
                layout = bwd.final_pi
            res = route_dag(router, layout, prep.dag, forward=True, rng=r, record=True)
            swaps.append(res.num_swaps)
            assert item == (layout, swaps, res)
        # The candidates are the same with and without the batched chains.
        kwargs = {"seed": 3, "layout_trials": 4, "swap_trials": 3, "work_limit": 400}
        batched = router._candidate_plans(qc, **kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(_mapping, "route_chains", lambda *a: None)
        serial = router._candidate_plans(qc, **kwargs)  # type: ignore[arg-type]
        monkeypatch.undo()
        monkeypatch.setenv("QUSPECT_COMPILE_THREADS", threads)
        assert [(p.label, p.gates, p.route_events) for p in batched] == [
            (p.label, p.gates, p.route_events) for p in serial
        ]


def test_route_chains_defer_to_python_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    target = _target("ring8")
    qc = QuantumCircuit().cx(0, 4).cx(2, 6)
    router = SABRE(target)
    prep = router._prepare(qc, False)
    monkeypatch.setenv(ROUTING_BACKEND_ENV, "python")
    assert _routing.route_chains(router, prep.dag, [(list(range(8)), 1, 3)]) is None


def _python_finish(
    router: SABRE, prep: Any, events: list[int], layout: list[int]
) -> Any:
    from quspect.transpiler import _mapping

    gates, initial, final = _mapping._replay(
        events, layout, prep.dag, prep.num_logical, router._num_physical
    )
    gates = _mapping.expand_toffolis(gates, router._adj_sets, prep.circuit)
    if prep.elide:
        gates, initial, final = _mapping._elide_trailing_swaps(
            gates, initial, prep.num_logical
        )
        final = [
            final[prep.sigma[w]] if w < prep.num_logical else final[w]
            for w in range(len(final))
        ]
    return (
        gates,
        initial,
        final,
        _mapping.estimate_two_qubit_cost(gates, router._num_physical),
    )


@pytest.mark.parametrize("seed", range(4))
def test_native_replay_matches_python(seed: int) -> None:
    from quspect.transpiler import _mapping

    rng = random.Random(seed)
    for tname, _, qc, options in _cases(8, 1000 + seed):
        qc = qc.copy()
        if seed % 2 == 0 and qc.num_qubits >= 4:
            qc.swap(0, 3)
            qc.swap(1, 2)
        if seed >= 2:
            qc.measure_all()
        target = _target(tname)
        router = SABRE(target, **options)
        for elide in (False, True):
            prep = router._prepare(qc, elide)
            layout = list(range(target.num_qubits))
            rng.shuffle(layout)
            res = router._route(
                layout, prep.dag, forward=True, rng=random.Random(seed), record=True
            )
            fast = _mapping._native_replay(
                prep, res.events, layout, router._num_physical
            )
            assert fast is not None
            want = _python_finish(router, prep, res.events, layout)
            assert router._finish_scored(prep, res.events, layout) == want
            assert router._finish(prep, res.events, layout) == want[:3]
            # Field-by-field identical objects, attributes in the same order
            # (a Toffoli DAG is only replayed natively).
            ref = (
                want[0]
                if fast[4]
                else _mapping._replay(
                    res.events, layout, prep.dag, prep.num_logical, target.num_qubits
                )[0]
            )
            for a, b in zip(fast[0], ref, strict=True):
                assert list(vars(a).items()) == list(vars(b).items())


def test_native_replay_falls_back_on_bad_input() -> None:
    from quspect.transpiler import _mapping

    target = _target("ring8")
    qc = QuantumCircuit().cx(0, 4).cx(2, 6)
    router = SABRE(target)
    prep = router._prepare(qc, False)
    # Events out of range: the Python replay raises its own error.
    assert _mapping._native_replay(prep, [99], list(range(8)), 8) is None
    with pytest.raises(IndexError):
        router._finish(prep, [99], list(range(8)))
    # Not a permutation of the physical qubits.
    assert _mapping._native_replay(prep, [], [0] * 8, 8) is None
