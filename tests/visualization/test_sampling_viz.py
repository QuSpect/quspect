"""Tests for the sampling visualization module."""

from __future__ import annotations

from pathlib import Path

import pytest

from quspect.simulation.result import SimulationResult
from quspect.visualization.sampling_viz import (
    ChartStyle,
    ComparisonDisplay,
    SamplingDisplay,
    SamplingStats,
    SortMode,
    _bar_width,
    _compute_stats,
    _render_dot,
    _render_hbar,
    _render_svg,
    _render_vbar,
    _sorted_outcomes,
    compare_results,
    display_sampling,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_result() -> SimulationResult:
    """A Bell-state-like result: roughly 50/50 '00' and '11'."""
    return SimulationResult(
        counts={"00": 512, "11": 488},
        probabilities={"00": 0.512, "11": 0.488},
        num_shots=1000,
        num_qubits=2,
        num_clbits=2,
    )


@pytest.fixture
def uniform_result() -> SimulationResult:
    """A uniform distribution over 4 basis states."""
    return SimulationResult(
        counts={"00": 250, "01": 250, "10": 250, "11": 250},
        probabilities={"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25},
        num_shots=1000,
        num_qubits=2,
        num_clbits=2,
    )


@pytest.fixture
def deterministic_result() -> SimulationResult:
    """A deterministic result: |00> with 100% probability."""
    return SimulationResult(
        counts={"00": 500},
        probabilities={"00": 1.0},
        num_shots=500,
        num_qubits=2,
        num_clbits=2,
    )


@pytest.fixture
def empty_result() -> SimulationResult:
    """A result with zero shots."""
    return SimulationResult(
        counts={},
        probabilities={},
        num_shots=0,
        num_qubits=2,
        num_clbits=2,
    )


# ---------------------------------------------------------------------------
#  _bar_width
# ---------------------------------------------------------------------------


class TestBarWidth:
    """Tests for the _bar_width helper."""

    def test_full_bar(self) -> None:
        """Count equals max → full bar of blocks."""
        bar = _bar_width(100, 100, bar_columns=10)
        assert bar == "█" * 10

    def test_empty_bar(self) -> None:
        """Zero count → empty bar."""
        bar = _bar_width(0, 100, bar_columns=10)
        assert bar == ""

    def test_half_bar(self) -> None:
        """Half count → roughly half blocks."""
        bar = _bar_width(50, 100, bar_columns=10)
        # 50/100 * 10 * 8 = 40 eighths = 5 full blocks
        assert bar == "█" * 5

    def test_zero_max_returns_empty(self) -> None:
        """Zero max_count → empty string (avoid division by zero)."""
        bar = _bar_width(0, 0, bar_columns=10)
        assert bar == ""


# ---------------------------------------------------------------------------
#  _compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    """Tests for the _compute_stats helper."""

    def test_bell_stats(self, bell_result: SimulationResult) -> None:
        """Stats for a roughly 50/50 distribution."""
        stats = _compute_stats(bell_result)
        # Entropy should be close to 1 bit (2 equally likely outcomes)
        assert 0.99 <= stats.entropy <= 1.01
        assert 0.99 <= stats.normalized_entropy <= 1.01
        assert stats.num_outcomes == 2
        assert stats.effective_rank == 2

    def test_uniform_stats(self, uniform_result: SimulationResult) -> None:
        """Stats for a perfectly uniform distribution."""
        stats = _compute_stats(uniform_result)
        # Entropy should be exactly 2 bits (4 equally likely outcomes)
        assert stats.entropy == pytest.approx(2.0)
        assert stats.normalized_entropy == pytest.approx(1.0)
        assert stats.variance == pytest.approx(0.0)
        assert stats.top_gap == pytest.approx(0.0)
        assert stats.num_outcomes == 4
        assert stats.effective_rank == 4  # all >= sqrt(1000) ≈ 31.6

    def test_deterministic_stats(self, deterministic_result: SimulationResult) -> None:
        """Stats for a deterministic result."""
        stats = _compute_stats(deterministic_result)
        assert stats.entropy == pytest.approx(0.0)
        assert stats.normalized_entropy == pytest.approx(0.0)
        assert stats.top_gap == pytest.approx(1.0)
        assert stats.effective_rank == 1

    def test_empty_stats(self, empty_result: SimulationResult) -> None:
        """Stats for a zero-shot (empty) result."""
        stats = _compute_stats(empty_result)
        assert stats.entropy == 0.0
        assert stats.normalized_entropy == 0.0
        assert stats.variance == 0.0
        assert stats.top_gap == 0.0
        assert stats.effective_rank == 0
        assert stats.num_outcomes == 0


# ---------------------------------------------------------------------------
#  _sorted_outcomes
# ---------------------------------------------------------------------------


class TestStatsIgnoreZeroCounts:
    """Regression tests for B79: zero-count entries are not 'observed'."""

    def test_zero_count_entries_are_ignored(self) -> None:
        """Stats match the result with the zero-count keys removed."""
        with_zeros = SimulationResult(
            counts={"00": 50, "01": 0, "10": 0, "11": 50},
            probabilities={"00": 0.5, "01": 0.0, "10": 0.0, "11": 0.5},
            num_shots=100,
            num_qubits=2,
            num_clbits=2,
        )
        without_zeros = SimulationResult(
            counts={"00": 50, "11": 50},
            probabilities={"00": 0.5, "11": 0.5},
            num_shots=100,
            num_qubits=2,
            num_clbits=2,
        )
        stats = _compute_stats(with_zeros)
        assert stats == _compute_stats(without_zeros)
        assert stats.num_outcomes == 2
        assert stats.normalized_entropy == pytest.approx(1.0)
        assert stats.variance == pytest.approx(0.0)


class TestZeroShotStats:
    """Zero-shot results follow the 'observed outcomes only' semantics."""

    def test_zero_shot_with_zero_count_keys(self) -> None:
        """All-zero counts give num_outcomes == 0 and finite zero metrics."""
        result = SimulationResult(
            counts={"00": 0, "01": 0},
            probabilities={"00": 1.0, "01": 0.0},
            num_shots=0,
            num_qubits=2,
            num_clbits=0,
        )
        stats = _compute_stats(result)
        assert stats == SamplingStats(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
        for style in (ChartStyle.HBAR, ChartStyle.VBAR, ChartStyle.DOT):
            text = str(display_sampling(result, style=style))
            assert "eff_rank=0/0" in text
            assert "nan" not in text.lower()
        assert "nan" not in display_sampling(result)._repr_svg_().lower()


class TestSortedOutcomes:
    """Tests for the _sorted_outcomes helper."""

    def test_count_desc(self, bell_result: SimulationResult) -> None:
        """Sort by count descending."""
        items = _sorted_outcomes(bell_result, SortMode.COUNT_DESC)
        assert items[0][1] >= items[1][1]

    def test_count_asc(self, bell_result: SimulationResult) -> None:
        """Sort by count ascending."""
        items = _sorted_outcomes(bell_result, SortMode.COUNT_ASC)
        assert items[0][1] <= items[1][1]

    def test_bitstring(self, bell_result: SimulationResult) -> None:
        """Sort by bitstring lexicographically."""
        items = _sorted_outcomes(bell_result, SortMode.BITSTRING)
        assert items[0][0] == "00"
        assert items[1][0] == "11"

    def test_probability(self, bell_result: SimulationResult) -> None:
        """Sort by probability descending."""
        items = _sorted_outcomes(bell_result, SortMode.PROBABILITY)
        assert items[0][1] >= items[1][1]


# ---------------------------------------------------------------------------
#  Text renderers
# ---------------------------------------------------------------------------


class TestRenderHbar:
    """Tests for the horizontal bar text renderer."""

    def test_includes_counts(self, bell_result: SimulationResult) -> None:
        """Output includes count values."""
        out = _render_hbar(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "512" in out
        assert "488" in out
        assert "00" in out
        assert "11" in out

    def test_includes_header_info(self, bell_result: SimulationResult) -> None:
        """Output includes shots/qubit/clbit info."""
        out = _render_hbar(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "shots: 1000" in out
        assert "qubits: 2" in out

    def test_includes_stats(self, bell_result: SimulationResult) -> None:
        """Output includes the statistical panel."""
        out = _render_hbar(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "H=" in out
        assert "entropy" in out or "H_norm" in out

    def test_respects_max_outcomes(self, uniform_result: SimulationResult) -> None:
        """Only displays up to max_outcomes entries."""
        out = _render_hbar(
            uniform_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=2
        )
        # Should show exactly 2 outcomes and a "... more" line
        assert "+2 more" in out or "...more" in out.replace(" ", "")

    def test_empty_result(self, empty_result: SimulationResult) -> None:
        """Does not crash on empty result."""
        out = _render_hbar(empty_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "shots: 0" in out


class TestRenderVbar:
    """Tests for the vertical bar text renderer."""

    def test_includes_counts(self, bell_result: SimulationResult) -> None:
        """Output includes bitstring labels."""
        out = _render_vbar(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "00" in out
        assert "11" in out

    def test_includes_stats(self, bell_result: SimulationResult) -> None:
        """Output includes stats line."""
        out = _render_vbar(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "H=" in out

    def test_empty_result(self, empty_result: SimulationResult) -> None:
        """Does not crash on empty result."""
        out = _render_vbar(empty_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "0 shots" in out

    def test_bars_axis_and_labels_share_column_width(self) -> None:
        """Bars stay centred over their labels for every column (B80)."""
        result = SimulationResult(
            counts={"000": 410, "101": 260, "011": 260, "110": 70},
            probabilities={"000": 0.41, "101": 0.26, "011": 0.26, "110": 0.07},
            num_shots=1000,
            num_qubits=3,
            num_clbits=3,
        )
        out = _render_vbar(result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        lines = out.splitlines()
        bottom_row = next(line for line in lines if "─┤" in line)
        label_line = lines[lines.index(bottom_row) + 2]
        bar_cols = [i for i in range(len(bottom_row)) if bottom_row.startswith("██", i)]
        label_cols = [label_line.index(bs) for bs in ("000", "101", "011", "110")]
        steps = {b - a for a, b in zip(bar_cols, bar_cols[1:], strict=False)}
        assert steps == {
            b - a for a, b in zip(label_cols, label_cols[1:], strict=False)
        }
        assert len(steps) == 1
        # Each bar lies inside the span of its label cell.
        for bar, label in zip(bar_cols, label_cols, strict=True):
            assert abs(bar - label) <= 1
        # The y-axis glyph is in the same column on every chart row.
        chart_rows = [line for line in lines if "┤" in line or "│" in line]
        axis_cols = {max(row.rfind("┤"), row.rfind("│")) for row in chart_rows}
        assert len(axis_cols) == 1


class TestRenderDot:
    """Tests for the dot chart text renderer."""

    def test_includes_counts(self, bell_result: SimulationResult) -> None:
        """Output includes count values."""
        out = _render_dot(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "512" in out
        assert "488" in out

    def test_most_frequent_highlighted(self, bell_result: SimulationResult) -> None:
        """The most frequent outcome is highlighted with a star."""
        out = _render_dot(bell_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "★" in out

    def test_empty_result(self, empty_result: SimulationResult) -> None:
        """Does not crash on empty result."""
        out = _render_dot(empty_result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "0 shots" in out

    def test_zero_shot_result_with_zero_counts(self) -> None:
        """All-zero counts do not divide by zero (B81)."""
        result = SimulationResult(
            counts={"00": 0, "01": 0},
            probabilities={"00": 1.0, "01": 0.0},
            num_shots=0,
            num_qubits=2,
            num_clbits=0,
        )
        out = _render_dot(result, sort_mode=SortMode.COUNT_DESC, max_outcomes=16)
        assert "0 shots" in out
        assert "★" not in out


# ---------------------------------------------------------------------------
#  SVG renderer
# ---------------------------------------------------------------------------


class TestRenderSvg:
    """Tests for the SVG renderer."""

    def test_svg_includes_elements(self, bell_result: SimulationResult) -> None:
        """SVG output contains standard SVG elements."""
        svg = _render_svg(
            bell_result,
            sort_mode=SortMode.COUNT_DESC,
            max_outcomes=16,
            theme_name="official",
            width=680,
        )
        assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        assert "</svg>" in svg
        assert "Sampling Result" in svg
        assert "Entropy:" in svg

    def test_svg_different_themes(self, bell_result: SimulationResult) -> None:
        """Different theme names produce different output."""
        svg_light = _render_svg(
            bell_result,
            sort_mode=SortMode.COUNT_DESC,
            max_outcomes=16,
            theme_name="light",
            width=680,
        )
        svg_dark = _render_svg(
            bell_result,
            sort_mode=SortMode.COUNT_DESC,
            max_outcomes=16,
            theme_name="dark",
            width=680,
        )
        assert svg_light != svg_dark

    def test_svg_empty_result(self, empty_result: SimulationResult) -> None:
        """Does not crash on empty result."""
        svg = _render_svg(
            empty_result,
            sort_mode=SortMode.COUNT_DESC,
            max_outcomes=16,
            theme_name="official",
            width=680,
        )
        assert "<svg" in svg


# ---------------------------------------------------------------------------
#  SamplingDisplay
# ---------------------------------------------------------------------------


class TestSamplingDisplay:
    """Tests for the SamplingDisplay class."""

    def test_construction(self, bell_result: SimulationResult) -> None:
        """Basic construction with defaults."""
        d = SamplingDisplay(bell_result)
        assert d.result is bell_result
        assert d.style == ChartStyle.HBAR
        assert d.sort_mode == SortMode.COUNT_DESC
        assert d.max_outcomes == 16
        assert d.theme == "official"

    def test_construction_custom(self, bell_result: SimulationResult) -> None:
        """Construction with custom arguments."""
        d = SamplingDisplay(
            bell_result,
            style=ChartStyle.VBAR,
            sort_mode=SortMode.BITSTRING,
            max_outcomes=8,
            theme="dark",
            svg_width=800,
        )
        assert d.style == ChartStyle.VBAR
        assert d.sort_mode == SortMode.BITSTRING
        assert d.max_outcomes == 8
        assert d.theme == "dark"
        assert d.svg_width == 800

    def test_invalid_style_raises(self, bell_result: SimulationResult) -> None:
        """Unknown style raises ValueError."""
        with pytest.raises(ValueError, match="Unknown style"):
            SamplingDisplay(bell_result, style="invalid")

    def test_invalid_sort_mode_raises(self, bell_result: SimulationResult) -> None:
        """Unknown sort_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sort_mode"):
            SamplingDisplay(bell_result, sort_mode="invalid")

    def test_invalid_theme_raises(self, bell_result: SimulationResult) -> None:
        """Unknown theme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown theme"):
            SamplingDisplay(bell_result, theme="invalid")

    def test_repr_hbar(self, bell_result: SimulationResult) -> None:
        """__repr__ returns text-based bar chart."""
        d = SamplingDisplay(bell_result, style=ChartStyle.HBAR)
        r = repr(d)
        assert "512" in r
        assert "00" in r

    def test_repr_vbar(self, bell_result: SimulationResult) -> None:
        """__repr__ with vbar style."""
        d = SamplingDisplay(bell_result, style=ChartStyle.VBAR)
        r = repr(d)
        assert "00" in r

    def test_repr_dot(self, bell_result: SimulationResult) -> None:
        """__repr__ with dot style."""
        d = SamplingDisplay(bell_result, style=ChartStyle.DOT)
        r = repr(d)
        assert "★" in r

    def test_svg_repr(self, bell_result: SimulationResult) -> None:
        """_repr_svg_ returns valid SVG."""
        d = SamplingDisplay(bell_result)
        svg = d._repr_svg_()
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_stats_property(self, bell_result: SimulationResult) -> None:
        """stats property returns SamplingStats."""
        d = SamplingDisplay(bell_result)
        s = d.stats
        assert isinstance(s, SamplingStats)
        assert s.entropy > 0

    def test_to_file(self, bell_result: SimulationResult, tmp_path: Path) -> None:
        """to_file writes SVG to disk."""
        path = tmp_path / "test_output.svg"
        d = SamplingDisplay(bell_result)
        d.to_file(path)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert "</svg>" in content


# ---------------------------------------------------------------------------
#  ComparisonDisplay
# ---------------------------------------------------------------------------


class TestComparisonDisplay:
    """Tests for the ComparisonDisplay class."""

    def test_construction(
        self, bell_result: SimulationResult, deterministic_result: SimulationResult
    ) -> None:
        """Basic construction with two results."""
        c = ComparisonDisplay(
            bell_result, deterministic_result, label_a="bell", label_b="det"
        )
        assert c.result_a is bell_result
        assert c.result_b is deterministic_result
        assert c.label_a == "bell"
        assert c.label_b == "det"

    def test_svg_repr(
        self, bell_result: SimulationResult, deterministic_result: SimulationResult
    ) -> None:
        """_repr_svg_ returns valid SVG comparison."""
        c = ComparisonDisplay(bell_result, deterministic_result)
        svg = c._repr_svg_()
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "bell" not in svg.lower()  # default labels are "A" and "B"

    def test_custom_labels_in_svg(
        self, bell_result: SimulationResult, deterministic_result: SimulationResult
    ) -> None:
        """Custom labels appear in the SVG."""
        c = ComparisonDisplay(
            bell_result, deterministic_result, label_a="ideal", label_b="noisy"
        )
        svg = c._repr_svg_()
        assert "ideal" in svg
        assert "noisy" in svg

    def test_repr(
        self, bell_result: SimulationResult, deterministic_result: SimulationResult
    ) -> None:
        """__repr__ includes basic info."""
        c = ComparisonDisplay(bell_result, deterministic_result)
        r = repr(c)
        assert "ComparisonDisplay" in r
        assert "1000" in r  # num_shots

    def test_to_file(
        self,
        bell_result: SimulationResult,
        deterministic_result: SimulationResult,
        tmp_path: Path,
    ) -> None:
        """to_file writes comparison SVG to disk."""
        path = tmp_path / "comparison.svg"
        c = ComparisonDisplay(bell_result, deterministic_result)
        c.to_file(path)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("<svg")


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------


class TestDisplaySampling:
    """Tests for the display_sampling convenience function."""

    def test_returns_sampling_display(self, bell_result: SimulationResult) -> None:
        """Returns a SamplingDisplay instance."""
        d = display_sampling(
            bell_result, style=ChartStyle.DOT, sort_mode=SortMode.COUNT_ASC
        )
        assert isinstance(d, SamplingDisplay)
        assert d.style == ChartStyle.DOT
        assert d.sort_mode == SortMode.COUNT_ASC


class TestCompareResults:
    """Tests for the compare_results convenience function."""

    def test_returns_comparison_display(
        self, bell_result: SimulationResult, deterministic_result: SimulationResult
    ) -> None:
        """Returns a ComparisonDisplay instance."""
        c = compare_results(bell_result, deterministic_result, label_a="X", label_b="Y")
        assert isinstance(c, ComparisonDisplay)
        assert c.label_a == "X"
        assert c.label_b == "Y"


# ---------------------------------------------------------------------------
#  Integration: SimulationResult.display()
# ---------------------------------------------------------------------------


class TestSimulationResultDisplay:
    """Tests for the SimulationResult.display() method."""

    def test_returns_sampling_display(self, bell_result: SimulationResult) -> None:
        """display() returns a SamplingDisplay."""
        d = bell_result.display()
        assert isinstance(d, SamplingDisplay)

    def test_accepts_keyword_args(self, bell_result: SimulationResult) -> None:
        """display() passes keyword args through."""
        d = bell_result.display(style="vbar", sort_mode="bitstring", max_outcomes=4)
        assert d.style == "vbar"
        assert d.sort_mode == "bitstring"
        assert d.max_outcomes == 4
