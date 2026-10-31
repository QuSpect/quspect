"""Tests for :mod:`quspect.visualization.sampling_viz`.

``SamplingStats`` is checked against independent numpy formulas, sort modes
against explicit orderings, text charts by parsing their rows, and SVG charts
by parsing them with ``xml.etree``.  One statistical test compares the
sampled entropy with the exact qiskit distribution.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import qiskit
from qiskit.quantum_info import Statevector

import quspect
from quspect.simulation import Simulator
from quspect.simulation.result import SimulationResult
from quspect.visualization import (
    ChartStyle,
    ComparisonDisplay,
    SamplingDisplay,
    SamplingStats,
    SortMode,
    compare_results,
    display_sampling,
)
from quspect.visualization.sampling_viz import _SVG_THEME_DATA

NS = "{http://www.w3.org/2000/svg}"
_ROW = re.compile(r"│  (\S+)\s+(\d+)\s+([\d.]+)%  (█*)([▏▎▍▌▋▊▉]?) ")


def _result(
    counts: dict[str, int],
    *,
    probabilities: dict[str, float] | None = None,
    num_qubits: int | None = None,
) -> SimulationResult:
    """Build a SimulationResult from counts (probabilities = frequencies)."""
    shots = sum(counts.values())
    probs = probabilities or {
        k: (v / shots if shots else 0.0) for k, v in counts.items()
    }
    width = num_qubits if num_qubits is not None else len(next(iter(counts), ""))
    return SimulationResult(
        counts=dict(counts),
        probabilities=probs,
        num_shots=shots,
        num_qubits=width,
        num_clbits=width,
    )


def _numpy_stats(counts: dict[str, int]) -> dict[str, float]:
    """Independent reference for the documented statistics."""
    c = np.array(list(counts.values()), dtype=float)
    n = c.sum()
    p = c / n
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    k = len(c)
    srt = np.sort(c)[::-1]
    return {
        "entropy": entropy,
        "normalized_entropy": entropy / math.log2(k) if k > 1 else 0.0,
        "variance": float(np.var(c)),
        "std": float(np.std(c)),
        "top_gap": float((srt[0] - srt[1]) / n) if k > 1 else 1.0,
        "effective_rank": int(np.sum(p >= 1 / np.sqrt(n))),
        "num_outcomes": k,
    }


def _parse(svg: str) -> ET.Element:
    """Parse SVG text (asserts well-formed XML)."""
    root = ET.fromstring(svg)
    assert root.tag == f"{NS}svg"
    return root


SAMPLE = {"000": 410, "101": 260, "011": 260, "110": 70}


# =============================================================================
# SamplingStats
# =============================================================================


@pytest.mark.parametrize("seed", range(8))
def test_stats_match_numpy_formulas(seed: int) -> None:
    """Entropy, variance, std, top gap, effective rank match numpy."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 12))
    keys = [format(i, "04b") for i in rng.choice(16, size=k, replace=False)]
    counts = {key: int(rng.integers(1, 500)) for key in keys}
    stats = display_sampling(_result(counts)).stats
    ref = _numpy_stats(counts)
    assert isinstance(stats, SamplingStats)
    for field_name, value in ref.items():
        assert getattr(stats, field_name) == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_stats_special_cases() -> None:
    """Single outcome -> gap 1, entropy 0; uniform -> H_norm 1; empty -> zeros."""
    single = display_sampling(_result({"01": 100})).stats
    assert (single.entropy, single.normalized_entropy, single.top_gap) == (0, 0, 1.0)
    assert (single.variance, single.effective_rank, single.num_outcomes) == (0, 1, 1)

    uniform = display_sampling(_result({"00": 25, "01": 25, "10": 25, "11": 25})).stats
    assert uniform.entropy == pytest.approx(2.0)
    assert uniform.normalized_entropy == pytest.approx(1.0)
    assert uniform.top_gap == 0.0

    empty = display_sampling(_result({}, num_qubits=1)).stats
    assert empty == SamplingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    zero_shot = display_sampling(_result({"0": 0, "1": 0})).stats
    # Zero shots means no outcome was observed (B79 semantics).
    assert zero_shot.num_outcomes == 0
    assert zero_shot.entropy == 0.0


def test_effective_rank_threshold_is_sqrt_shots() -> None:
    """Outcomes with count >= sqrt(shots) are counted (boundary inclusive)."""
    counts = {"00": 9919, "01": 100, "10": 99}  # shots=10118, sqrt~100.6
    stats = display_sampling(_result(counts)).stats
    assert stats.effective_rank == int(
        sum(c >= math.sqrt(10118) for c in counts.values())
    )
    counts2 = {"0": 90, "1": 10}  # shots=100, sqrt=10 -> both count
    assert display_sampling(_result(counts2)).stats.effective_rank == 2


def test_stats_ignore_unobserved_outcomes() -> None:
    """``num_outcomes`` is the number of outcomes actually observed."""
    stats = display_sampling(_result({"00": 50, "01": 0, "10": 0, "11": 50})).stats
    assert stats.num_outcomes == 2
    assert stats.normalized_entropy == pytest.approx(1.0)


@pytest.mark.qiskit_ref
@pytest.mark.statistical
def test_sampled_entropy_matches_qiskit_distribution() -> None:
    """Entropy of sampled GHZ-like counts is close to qiskit's exact entropy."""
    qc = qiskit.QuantumCircuit(3)
    qc.ry(1.1, 0)
    qc.cx(0, 1)
    qc.h(2)
    probs = Statevector(qc).probabilities()
    exact = float(-sum(p * math.log2(p) for p in probs if p > 1e-15))
    circ = quspect.QuantumCircuit()
    circ.ry(1.1, 0).cx(0, 1).h(2).measure([0, 1, 2], [0, 1, 2])
    shots = 20000
    result = Simulator(seed=7).run(circ, shots=shots)
    stats = display_sampling(result).stats
    # Plug-in entropy bias is O(K / shots); 5-sigma of its fluctuation is
    # well below 0.05 bit for K = 4 outcomes and 20000 shots.
    assert stats.entropy == pytest.approx(exact, abs=0.05)
    assert stats.effective_rank == int(np.sum(probs > 0.01))


# =============================================================================
# Sorting
# =============================================================================


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (SortMode.COUNT_DESC, ["000", "101", "011", "110"]),
        (SortMode.COUNT_ASC, ["110", "101", "011", "000"]),
        (SortMode.BITSTRING, ["000", "011", "101", "110"]),
    ],
)
def test_sort_modes_in_hbar(mode: str, expected: list[str]) -> None:
    """Rows follow the requested order (ties keep insertion order)."""
    text = str(display_sampling(_result(SAMPLE), sort_mode=mode))
    assert [m.group(1) for m in _ROW.finditer(text)] == expected


def test_probability_sort_uses_probabilities_not_counts() -> None:
    """``PROBABILITY`` orders by ``result.probabilities``."""
    res = _result(
        {"00": 10, "01": 20, "10": 30},
        probabilities={"00": 0.5, "01": 0.2, "10": 0.3},
    )
    text = str(display_sampling(res, sort_mode=SortMode.PROBABILITY))
    assert [m.group(1) for m in _ROW.finditer(text)] == ["00", "10", "01"]


def test_constants() -> None:
    """ChartStyle / SortMode expose the documented string values."""
    assert (ChartStyle.HBAR, ChartStyle.VBAR, ChartStyle.DOT, ChartStyle.SVG) == (
        "hbar",
        "vbar",
        "dot",
        "svg",
    )
    assert (
        SortMode.COUNT_DESC,
        SortMode.COUNT_ASC,
        SortMode.BITSTRING,
        SortMode.PROBABILITY,
    ) == ("count_desc", "count_asc", "bitstring", "probability")


# =============================================================================
# Text charts
# =============================================================================


def test_hbar_rows_bars_and_stats_line() -> None:
    """Counts, percentages, bar lengths (eighth blocks) and the stats panel."""
    text = str(display_sampling(_result(SAMPLE)))
    assert "shots: 1000  qubits: 3  clbits: 3" in text
    rows = {m.group(1): m for m in _ROW.finditer(text)}
    assert set(rows) == set(SAMPLE)
    for bs, m in rows.items():
        assert int(m.group(2)) == SAMPLE[bs]
        assert float(m.group(3)) == pytest.approx(SAMPLE[bs] / 10, abs=0.05)
        eighths = 8 * len(m.group(4)) + " ▏▎▍▌▋▊▉".index(m.group(5) or " ")
        assert eighths == round(SAMPLE[bs] / 410 * 40 * 8)
    stats = _numpy_stats(SAMPLE)
    assert f"H={stats['entropy']:.3f} bit" in text
    assert f"H_norm={stats['normalized_entropy']:.3f}" in text
    assert f"σ={stats['std']:.1f}" in text
    assert f"gap={stats['top_gap']:.1%}" in text
    assert f"eff_rank={stats['effective_rank']}/4" in text


def test_hbar_truncation_note() -> None:
    """``max_outcomes`` truncates and reports the number of hidden outcomes."""
    text = str(display_sampling(_result(SAMPLE), max_outcomes=2))
    assert [m.group(1) for m in _ROW.finditer(text)] == ["000", "101"]
    assert "+2 more outcomes" in text
    assert "more outcomes" not in str(display_sampling(_result(SAMPLE)))


def test_vbar_chart() -> None:
    """VBAR: 10 bar rows, axis ticks, one label per shown outcome, stats."""
    disp = display_sampling(_result(SAMPLE), style=ChartStyle.VBAR, max_outcomes=3)
    lines = str(disp).splitlines()
    assert "1000 shots" in lines[1] and "3 qubits" in lines[1]
    bar_rows = lines[2:12]
    assert bar_rows[0].lstrip().startswith("410 ┤")
    assert bar_rows[-1].lstrip().startswith("0 ─┤")
    # Heights: 410 -> 10 rows, 260 -> round(6.34) = 6 rows (twice).
    assert [row.count("██") for row in bar_rows] == [1, 1, 1, 1, 3, 3, 3, 3, 3, 3]
    assert lines[13].split() == ["000", "101", "011"]
    assert "Stats: H=" in lines[15]


def test_vbar_bars_align_with_labels() -> None:
    """Bars are spaced exactly like their x-axis labels."""
    disp = display_sampling(_result(SAMPLE), style=ChartStyle.VBAR)
    lines = str(disp).splitlines()
    bottom_row = lines[11]  # every outcome has a bar on the lowest row
    label_line = lines[13]
    bar_cols = [i for i in range(len(bottom_row)) if bottom_row.startswith("██", i)]
    label_cols = [label_line.index(bs) for bs in ("000", "101", "011", "110")]
    bar_steps = np.diff(bar_cols).tolist()
    label_steps = np.diff(label_cols).tolist()
    assert bar_steps == label_steps


def test_dot_chart() -> None:
    """DOT: one line per outcome with mini-bar, count, percent, and a star line."""
    text = str(display_sampling(_result(SAMPLE), style=ChartStyle.DOT))
    rows = re.findall(r"● (\S+)\s+(─+)\s+(\d+)\s+\(\s*([\d.]+)%\)", text)
    assert [r[0] for r in rows] == ["000", "101", "011", "110"]
    assert [len(r[1]) for r in rows] == [
        min(20, max(1, round(c / 410 * 20))) for c in (410, 260, 260, 70)
    ]
    assert "most frequent: 000  (410 shots)" in text
    assert "eff_rank=4/4" in text


def test_text_charts_with_empty_counts() -> None:
    """All text styles render a result with no outcomes at all."""
    res = _result({}, num_qubits=2)
    for style in (ChartStyle.HBAR, ChartStyle.VBAR, ChartStyle.DOT):
        text = str(display_sampling(res, style=style))
        assert "Sampling Result" in text


def test_zero_shot_hbar_vbar_svg() -> None:
    """Zero-shot results (all counts 0) render without errors in hbar/vbar/svg."""
    res = SimulationResult.from_statevector(np.array([1, 0, 0, 0], complex), shots=0)
    assert "shots: 0" in str(display_sampling(res))
    assert "0 shots" in str(display_sampling(res, style=ChartStyle.VBAR))
    root = _parse(display_sampling(res)._repr_svg_())
    assert not [r for r in root.iter(f"{NS}rect") if r.get("rx") == "5"]


def test_zero_shot_dot_chart() -> None:
    """DOT style handles a result whose counts are all zero."""
    res = SimulationResult.from_statevector(np.array([1, 0, 0, 0], complex), shots=0)
    assert "0 shots" in str(display_sampling(res, style=ChartStyle.DOT))


def test_repr_str_and_svg_style_text_fallback() -> None:
    """``repr == str``; the SVG style falls back to the hbar text chart."""
    res = _result(SAMPLE)
    disp = SamplingDisplay(res, style=ChartStyle.SVG)
    assert repr(disp) == str(disp) == str(display_sampling(res))


# =============================================================================
# SVG charts
# =============================================================================


@pytest.mark.parametrize("theme", ["official", "light", "dark"])
def test_svg_chart_structure(theme: str) -> None:
    """One gradient + bar per shown outcome, labels in order, stats panel."""
    disp = display_sampling(
        _result(SAMPLE),
        style=ChartStyle.SVG,
        theme=theme,
        sort_mode=SortMode.BITSTRING,
        svg_width=500,
        max_outcomes=3,
    )
    root = _parse(disp._repr_svg_())
    assert root.get("width") == "500"
    rects = list(root.iter(f"{NS}rect"))
    assert rects[0].get("fill") == _SVG_THEME_DATA[theme]["background"]
    bars = [r for r in rects if r.get("rx") == "5"]
    assert len(bars) == 3
    assert len(list(root.iter(f"{NS}linearGradient"))) == 3
    widths = [float(b.get("width", "0")) for b in bars]
    assert widths[0] == pytest.approx(500 - 80 - 40, abs=0.06)
    assert widths[1] / widths[0] == pytest.approx(260 / 410, abs=1e-3)
    texts = [t.text or "" for t in root.iter(f"{NS}text")]
    labels = [t for t in texts if t in SAMPLE]
    assert labels == ["000", "011", "101"]
    assert "410 (41.0%)" in texts
    assert any(t.startswith("Entropy: ") for t in texts)
    assert any("Effective rank: 4/4" in t for t in texts)


def test_svg_to_file(tmp_path: Path) -> None:
    """``to_file`` writes the same SVG as ``_repr_svg_``."""
    disp = display_sampling(_result(SAMPLE), theme="dark")
    out = tmp_path / "s.svg"
    disp.to_file(out)
    assert out.read_text(encoding="utf-8") == disp._repr_svg_()


def test_display_validation() -> None:
    """Unknown style / sort mode / theme raise ValueError."""
    res = _result(SAMPLE)
    for kwargs in ({"style": "pie"}, {"sort_mode": "random"}, {"theme": "neon"}):
        with pytest.raises(ValueError, match="Unknown"):
            SamplingDisplay(res, **kwargs)  # type: ignore[arg-type]


def test_display_sampling_and_result_display_are_equivalent() -> None:
    """``display_sampling`` and ``SimulationResult.display`` build the same object."""
    res = _result(SAMPLE)
    kwargs = {
        "style": "dot",
        "sort_mode": "bitstring",
        "max_outcomes": 2,
        "theme": "light",
        "svg_width": 600,
    }
    a = display_sampling(res, **kwargs)  # type: ignore[arg-type]
    b = res.display(**kwargs)  # type: ignore[arg-type]
    c = SamplingDisplay(res, **kwargs)  # type: ignore[arg-type]
    assert a == b == c


# =============================================================================
# Comparison
# =============================================================================


A = {"00": 500, "01": 100, "11": 400}
B = {"00": 300, "10": 200, "11": 500}


def _comparison_rows(svg: str) -> list[str]:
    """Outcome labels drawn in the centre column, top to bottom."""
    root = _parse(svg)
    return [
        t.text or ""
        for t in root.iter(f"{NS}text")
        if t.get("text-anchor") == "middle" and t.get("font-size") == "13"
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (SortMode.COUNT_DESC, ["00", "11", "01", "10"]),
        (SortMode.COUNT_ASC, ["10", "01", "11", "00"]),
        (SortMode.BITSTRING, ["00", "01", "10", "11"]),
        (SortMode.PROBABILITY, ["00", "11", "01", "10"]),  # falls back to A's counts
        ("unknown", ["00", "11", "01", "10"]),
    ],
)
def test_compare_results_sorting(mode: str, expected: list[str]) -> None:
    """Union of outcomes, ordered by result A's count / bitstring."""
    cmp = compare_results(_result(A), _result(B), sort_mode=mode)
    assert _comparison_rows(cmp._repr_svg_()) == expected


def test_compare_results_bars_and_labels() -> None:
    """Bars only for non-zero counts, scaled to the joint maximum; escaped labels."""
    cmp = compare_results(
        _result(A),
        _result(B),
        label_a="ideal<1>",
        label_b="noisy & co",
        theme="dark",
        svg_width=700,
        max_outcomes=3,
    )
    assert isinstance(cmp, ComparisonDisplay)
    svg = cmp._repr_svg_()
    root = _parse(svg)
    assert "ideal&lt;1&gt;" in svg and "noisy &amp; co" in svg
    bars = [r for r in root.iter(f"{NS}rect") if r.get("rx") == "4"]
    left = [b for b in bars if b.get("fill") == "#3b82f6"]
    right = [b for b in bars if b.get("fill") == "#f59e0b"]
    # Rows shown: 00, 11, 01 (A desc, top 3); B has no "01".
    assert len(left) == 3 and len(right) == 2
    half = (700 - 160 - 60) // 2
    widths = sorted(float(b.get("width", "0")) for b in left + right)
    assert widths[-1] == pytest.approx(half, abs=0.06)
    assert widths[0] == pytest.approx(100 / 500 * half, abs=0.06)
    stat_lines = [
        t.text or "" for t in root.iter(f"{NS}text") if t.get("font-size") == "10"
    ]
    assert len(stat_lines) == 2 and all(s.startswith("H=") for s in stat_lines)
    assert (
        root.iter(f"{NS}rect").__next__().get("fill")
        == _SVG_THEME_DATA["dark"]["background"]
    )


def test_comparison_repr_file_and_constructor(tmp_path: Path) -> None:
    """``repr``, ``to_file`` and the dataclass constructor."""
    cmp = compare_results(_result(A), _result(B), label_a="x", label_b="y")
    assert repr(cmp) == "ComparisonDisplay(x vs y, shots: 1000/1000)"
    out = tmp_path / "cmp.svg"
    cmp.to_file(out)
    assert out.read_text(encoding="utf-8") == cmp._repr_svg_()
    assert cmp == ComparisonDisplay(_result(A), _result(B), label_a="x", label_b="y")
