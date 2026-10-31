"""``SABRE.route_candidates(swap_keep=...)``: cheap swap trials, few scored."""

from __future__ import annotations

import random
import re

import pytest

from quspect import QuantumCircuit
from quspect.transpiler import SABRE, CompilationTarget


def _grid_target(rows: int, cols: int) -> CompilationTarget:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return CompilationTarget.from_coupling_map(rows * cols, edges)


def _random_cx(n: int, depth: int, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit()
    for _ in range(depth):
        a, b = rng.sample(range(n), 2)
        qc.cx(a, b)
        if rng.random() < 0.3:
            qc.h(a)
    return qc


def _labels(cands: list) -> dict[str, tuple[int, int]]:
    return {c.label: (c.num_swaps, c.depth) for c in cands}


def _swap_index(label: str) -> tuple[int, int]:
    m = re.fullmatch(r"trial(\d+)/swap(\d+)", label)
    assert m is not None
    return int(m.group(1)), int(m.group(2))


@pytest.mark.parametrize("seed", range(4))
def test_swap_keep_selects_fewest_swaps(seed: int) -> None:
    target = _grid_target(3, 4)
    qc = _random_cx(12, 60, seed)
    kwargs = {"seed": seed, "layout_trials": 3, "swap_trials": 6, "exact_layout": False}
    full = _labels(SABRE(target).route_candidates(qc, **kwargs))
    kept = SABRE(target).route_candidates(qc, swap_keep=1, **kwargs)
    kept_labels = _labels(kept)
    # 第一轮路由（swap0）不受影响；每个布局至多保留一个额外路由
    assert {k for k in full if k.endswith("/swap0")} == {
        k for k in kept_labels if k.endswith("/swap0")
    }
    per_trial: dict[int, list[int]] = {}
    for label in kept_labels:
        trial, s_idx = _swap_index(label)
        if s_idx:
            per_trial.setdefault(trial, []).append(s_idx)
    for trial, kept_idx in per_trial.items():
        assert len(kept_idx) == 1
        # 保留的是该布局所有额外路由中 (SWAP 数, 深度, 序号) 最小的一个
        others = {
            s: v
            for label, v in full.items()
            for t, s in [_swap_index(label)]
            if t == trial and s > 0
        }
        if others:
            best = min(others, key=lambda s: (*others[s], s))
            assert (*kept_labels[f"trial{trial}/swap{kept_idx[0]}"], kept_idx[0]) <= (
                *others[best],
                best,
            )
    # 默认候选（solve 的结果）不变
    default_full = [
        c for c in SABRE(target).route_candidates(qc, **kwargs) if c.is_default
    ]
    default_kept = [c for c in kept if c.is_default]
    assert [c.label for c in default_full] == [c.label for c in default_kept]


def test_swap_keep_all_equals_unfiltered_set() -> None:
    target = _grid_target(3, 3)
    qc = _random_cx(9, 40, 7)
    kwargs = {"seed": 2, "layout_trials": 2, "swap_trials": 4, "exact_layout": False}
    full = SABRE(target).route_candidates(qc, **kwargs)
    kept = SABRE(target).route_candidates(qc, swap_keep=3, **kwargs)
    assert sorted((c.label, c.num_swaps) for c in full) == sorted(
        (c.label, c.num_swaps) for c in kept
    )


def test_swap_keep_deterministic_and_validated() -> None:
    target = _grid_target(2, 4)
    qc = _random_cx(8, 30, 3)
    kwargs = {"seed": 5, "layout_trials": 2, "swap_trials": 5, "swap_keep": 2}
    a = SABRE(target).route_candidates(qc, **kwargs)
    b = SABRE(target).route_candidates(qc, **kwargs)
    assert [(c.label, c.initial_mapping) for c in a] == [
        (c.label, c.initial_mapping) for c in b
    ]
    with pytest.raises(ValueError, match="swap_keep"):
        SABRE(target).route_candidates(qc, swap_keep=0)
