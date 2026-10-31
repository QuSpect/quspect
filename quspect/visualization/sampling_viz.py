"""Sampling result visualization.

Provides rich terminal-based and SVG visualizations of
:class:`~quspect.simulation.result.SimulationResult` objects, with
multiple chart styles, statistical annotations, and color themes.

Differentiation from MindQuantum:
    - Multiple chart styles (horizontal bars, vertical bars, dot chart)
    - 8-level Unicode block gradients for finer-grained terminal bars
    - Statistical panel (entropy, variance, top-gap, effective rank)
    - Three SVG color themes matching the circuit drawer
    - Side-by-side comparison of two results
    - Flexible sorting (by count, bitstring, or probability)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspect.simulation.result import SimulationResult

# ---------------------------------------------------------------------------
#  Unicode block elements for 8-level horizontal bar rendering
# ---------------------------------------------------------------------------

_BLOCKS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

# ---------------------------------------------------------------------------
#  SVG themes (reuse the same palette as the circuit SVG drawer)
# ---------------------------------------------------------------------------

_SVG_THEME_DATA: dict[str, dict[str, str]] = {
    "official": {
        "background": "#ffffff",
        "text": "#0f172a",
        "bar_fill": "#3b82f6",
        "bar_stroke": "#2563eb",
        "grid": "#e2e8f0",
        "border": "#cbd5e1",
        "accent": "#0f172a",
        "stat_bg": "#f8fafc",
    },
    "light": {
        "background": "#f8fafc",
        "text": "#0f172a",
        "bar_fill": "#6366f1",
        "bar_stroke": "#4f46e5",
        "grid": "#e2e8f0",
        "border": "#cbd5e1",
        "accent": "#1e293b",
        "stat_bg": "#f1f5f9",
    },
    "dark": {
        "background": "#0f172a",
        "text": "#e2e8f0",
        "bar_fill": "#38bdf8",
        "bar_stroke": "#0ea5e9",
        "grid": "#334155",
        "border": "#475569",
        "accent": "#e2e8f0",
        "stat_bg": "#1e293b",
    },
}

_BAR_COLORS: list[str] = [
    "#3b82f6",  # blue
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f97316",  # orange
]


# ---------------------------------------------------------------------------
#  Chart style enum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChartStyle:
    """Constants for chart rendering styles."""

    HBAR: str = "hbar"  #: Horizontal bar chart (terminal Unicode).
    VBAR: str = "vbar"  #: Vertical bar chart (terminal ASCII / Unicode).
    DOT: str = "dot"  #: Dot / stem chart (minimal, terminal).
    SVG: str = "svg"  #: SVG chart (Jupyter / file export).


# ---------------------------------------------------------------------------
#  Sort mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortMode:
    """Constants for outcome sort ordering."""

    COUNT_DESC: str = "count_desc"  #: Sort by count, highest first (default).
    COUNT_ASC: str = "count_asc"  #: Sort by count, lowest first.
    BITSTRING: str = "bitstring"  #: Sort lexicographically by bitstring.
    PROBABILITY: str = "probability"  #: Sort by probability, highest first.


# ---------------------------------------------------------------------------
#  Result stats
# ---------------------------------------------------------------------------


@dataclass
class SamplingStats:
    """Statistical summary of a sampling result.

    Attributes:
        entropy: Shannon entropy of the empirical distribution (bits).
        normalized_entropy: Entropy divided by log2(N), where N is
            ``num_outcomes`` — 0→1 range.
        variance: Variance of the counts of the observed outcomes.
        std: Standard deviation of the counts of the observed outcomes.
        top_gap: Difference in proportion between the most- and
            second-most frequent outcomes (0..1).
        effective_rank: Number of outcomes whose frequency ≥1/sqrt(shots),
            a heuristic for distribution "flatness."
        num_outcomes: Number of distinct outcomes observed (count > 0);
            zero-count entries in ``result.counts`` are ignored.
    """

    entropy: float
    normalized_entropy: float
    variance: float
    std: float
    top_gap: float
    effective_rank: int
    num_outcomes: int


def _compute_stats(result: SimulationResult) -> SamplingStats:
    """Compute statistical summary from a :class:`SimulationResult`.

    Args:
        result: The simulation result to summarize.

    All metrics are computed over the *observed* outcomes only, i.e. entries
    of ``result.counts`` with a count greater than zero.  For a zero-shot
    result nothing has been observed, so every metric (including
    ``num_outcomes``) is zero.

    Returns:
        A :class:`SamplingStats` instance with derived metrics.
    """
    n = result.num_shots
    if n == 0:
        return SamplingStats(
            entropy=0.0,
            normalized_entropy=0.0,
            variance=0.0,
            std=0.0,
            top_gap=0.0,
            effective_rank=0,
            num_outcomes=0,
        )

    # Only outcomes that were actually observed (count > 0) contribute.
    # ``SimulationResult.counts`` may list every basis state with a zero
    # count, which would otherwise inflate ``num_outcomes`` and ``variance``
    # and deflate ``normalized_entropy``.
    counts = [c for c in result.counts.values() if c > 0]
    probs = [c / n for c in counts]
    num_outcomes = len(counts)

    # Shannon entropy (bits)
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalized entropy
    max_entropy = math.log2(max(num_outcomes, 1))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Variance / std of counts
    mean_c = sum(counts) / num_outcomes if num_outcomes else 0.0
    variance = (
        sum((c - mean_c) ** 2 for c in counts) / num_outcomes if num_outcomes else 0.0
    )
    std = math.sqrt(variance)

    # Top gap
    sorted_counts = sorted(counts, reverse=True)
    if len(sorted_counts) >= 2:
        top_gap = (sorted_counts[0] - sorted_counts[1]) / n
    elif len(sorted_counts) == 1:
        top_gap = 1.0
    else:
        top_gap = 0.0

    # Effective rank: outcomes with count >= sqrt(shots)
    threshold = math.sqrt(n) if n > 0 else 0
    effective_rank = sum(1 for c in counts if c >= threshold)

    return SamplingStats(
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        variance=variance,
        std=std,
        top_gap=top_gap,
        effective_rank=effective_rank,
        num_outcomes=num_outcomes,
    )


# ---------------------------------------------------------------------------
#  Unicode text renderers
# ---------------------------------------------------------------------------


def _bar_width(count: int, max_count: int, bar_columns: int = 40) -> str:
    """Render a single bar as Unicode block characters.

    Uses full block ``█`` plus a fractional eighth-block character
    for finer granularity than MindQuantum's single-character bars.

    Args:
        count: The count to represent.
        max_count: The maximum count (fills the full bar width).
        bar_columns: Total width in block-character units.

    Returns:
        A string of Unicode block characters representing the bar.
    """
    if max_count == 0:
        return ""
    ratio = count / max_count
    total_eighths = round(ratio * bar_columns * 8)
    full_blocks = total_eighths // 8
    remainder = total_eighths % 8
    return "█" * full_blocks + _BLOCKS[remainder]


def _render_hbar(result: SimulationResult, *, sort_mode: str, max_outcomes: int) -> str:
    """Render a horizontal bar chart using Unicode block characters.

    Args:
        result: The simulation result to visualize.
        sort_mode: One of :class:`SortMode` values.
        max_outcomes: Maximum number of outcomes to display.

    Returns:
        A multi-line string suitable for terminal display.
    """
    stats = _compute_stats(result)

    # Sort outcomes
    items = _sorted_outcomes(result, sort_mode)
    items = items[:max_outcomes]

    max_count = max((c for _, c in items), default=1)
    bar_w = 40
    # Determine column widths for alignment
    # Bitstring width: max bitstring length, at least 2
    bs_width = max((len(bs) for bs, _ in items), default=2)
    count_width = max(len(str(max_count)), 5)
    pct_width = 6  # "100.0%"

    lines: list[str] = []
    lines.append("")
    lines.append(
        f"  ╭── Sampling Result ── shots: {result.num_shots}  "
        f"qubits: {result.num_qubits}  clbits: {result.num_clbits} ──╮"
    )

    header = (
        f"  │  {'Outcome':<{bs_width}}  {'Count':>{count_width}}  "
        f"{'%':>{pct_width}}  Distribution{' ' * (bar_w - 12)}│"
    )
    lines.append(header)
    lines.append(f"  │{'─' * (len(header) - 4)}│")

    for bs, cnt in items:
        pct = cnt / result.num_shots * 100 if result.num_shots else 0.0
        bar = _bar_width(cnt, max_count, bar_w)
        lines.append(
            f"  │  {bs:<{bs_width}}  {cnt:>{count_width}d}  "
            f"{pct:>{pct_width - 1}.1f}%  {bar} {pct:.1f}%│"
        )

    if len(_sorted_outcomes(result, sort_mode)) > max_outcomes:
        omitted = len(_sorted_outcomes(result, sort_mode)) - max_outcomes
        lines.append(
            f"  │  {'...':<{bs_width}}  +{omitted} more outcomes"
            + " " * (bar_w + 5)
            + "│"
        )

    lines.append(
        "  ├── Stats ──────────────────────────────────────────────────────────────┤"
    )
    lines.append(
        f"  │  H={stats.entropy:.3f} bit  "
        f"H_norm={stats.normalized_entropy:.3f}  "
        f"σ={stats.std:.1f}  "
        f"gap={stats.top_gap:.1%}  "
        f"eff_rank={stats.effective_rank}/{stats.num_outcomes}"
        f"{' ' * 6}│"
    )
    lines.append(
        "  ╰──────────────────────────────────────────────────────────────────────╯"
    )
    lines.append("")
    return "\n".join(lines)


def _render_vbar(result: SimulationResult, *, sort_mode: str, max_outcomes: int) -> str:
    """Render a vertical bar chart using Unicode box-drawing characters.

    Args:
        result: The simulation result to visualize.
        sort_mode: One of :class:`SortMode` values.
        max_outcomes: Maximum number of outcomes to display.

    Returns:
        A multi-line string suitable for terminal display.
    """
    items = _sorted_outcomes(result, sort_mode)
    items = items[:max_outcomes]
    max_count = max((c for _, c in items), default=1)
    stats = _compute_stats(result)

    bar_height = 10  # rows for the bars

    # Build the chart top-down
    lines: list[str] = []
    lines.append("")
    lines.append(
        f"  Sampling Result  ·  {result.num_shots} shots  ·  {result.num_qubits} qubits"
    )

    # Compute bar heights (in rows of full blocks)
    heights = [
        max(1, round(c / max_count * bar_height)) if max_count > 0 and c > 0 else 0
        for _, c in items
    ]

    # Y-axis labels.  Every row prefix is ``y_width + 3`` characters wide so
    # that the axis glyph sits in the same column on every line.
    y_width = len(str(max_count)) + 1

    # One uniform column width for bars, empty cells, axis and labels so
    # that each bar stays centred above its x-axis label.
    bs_width = max((len(bs) for bs, _ in items), default=0)
    col_w = max(6, bs_width + 2)
    bar_cell = f"{'██':^{col_w}}"
    empty_cell = " " * col_w

    for row in range(bar_height, 0, -1):
        tick = int(max_count * row / bar_height)
        if row == bar_height:
            y_label = f"{tick:>{y_width + 1}d} ┤"
        elif row == 1:
            y_label = f"{0:>{y_width}d} ─┤"
        else:
            y_label = f"{'':>{y_width}s}  │"

        row_bars = "".join(bar_cell if h >= row else empty_cell for h in heights)
        lines.append(y_label + row_bars)

    # X-axis line
    lines.append(f"{'':>{y_width}s} ─┴" + "─" * (col_w * len(items)))

    # X-axis labels
    label_line = f"{'':>{y_width}s}   "
    for bs, _ in items:
        label_line += f"{bs:^{col_w}s}"
    lines.append(label_line)

    lines.append("")
    lines.append(
        f"  Stats: H={stats.entropy:.3f} bit  "
        f"H_norm={stats.normalized_entropy:.3f}  "
        f"σ={stats.std:.1f}  gap={stats.top_gap:.1%}  "
        f"eff_rank={stats.effective_rank}/{stats.num_outcomes}"
    )
    lines.append("")
    return "\n".join(lines)


def _render_dot(result: SimulationResult, *, sort_mode: str, max_outcomes: int) -> str:
    """Render a minimal dot / stem chart.

    Each outcome is a dot followed by a label and a count + percentage.

    Args:
        result: The simulation result to visualize.
        sort_mode: One of :class:`SortMode` values.
        max_outcomes: Maximum number of outcomes to display.

    Returns:
        A multi-line string suitable for terminal display.
    """
    items = _sorted_outcomes(result, sort_mode)
    items = items[:max_outcomes]
    stats = _compute_stats(result)

    lines: list[str] = []
    lines.append("")
    lines.append(
        f"  ● Sampling Result  ·  {result.num_shots} shots  ·  {result.num_qubits}q"
    )
    lines.append("")

    bs_width = max((len(bs) for bs, _ in items), default=2)
    count_width = max(len(str(c)) for _, c in items) if items else 1

    max_item = max((c for _, c in items), default=0)
    for bs, cnt in items:
        pct = cnt / result.num_shots * 100 if result.num_shots else 0.0
        # Mini bar: up to 20 chars wide.  With no shots at all (every count
        # is zero) there is nothing to scale against, so no bar is drawn.
        mini_bar_w = min(20, max(1, round(cnt / max_item * 20))) if max_item > 0 else 0
        mini = "─" * mini_bar_w
        lines.append(
            f"  ● {bs:<{bs_width}}  {mini}  {cnt:>{count_width}d}  ({pct:5.1f}%)"
        )

    if len(items) > 0 and max_item > 0:
        best_bs, best_cnt = items[0]
        lines.append("")
        lines.append(f"  ★  most frequent: {best_bs}  ({best_cnt} shots)")

    lines.append("")
    lines.append(
        f"  H={stats.entropy:.3f} | "
        f"H_norm={stats.normalized_entropy:.3f} | "
        f"σ={stats.std:.1f} | "
        f"gap={stats.top_gap:.1%} | "
        f"eff_rank={stats.effective_rank}/{stats.num_outcomes}"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  SVG renderer
# ---------------------------------------------------------------------------


def _render_svg(
    result: SimulationResult,
    *,
    sort_mode: str,
    max_outcomes: int,
    theme_name: str,
    width: int,
) -> str:
    """Render a sampling result as an SVG bar chart.

    Args:
        result: The simulation result to visualize.
        sort_mode: One of :class:`SortMode` values.
        max_outcomes: Maximum number of outcomes to display.
        theme_name: One of ``"official"``, ``"light"``, ``"dark"``.
        width: SVG width in pixels.

    Returns:
        An SVG string.
    """
    theme = _SVG_THEME_DATA.get(theme_name, _SVG_THEME_DATA["official"])
    stats = _compute_stats(result)
    items = _sorted_outcomes(result, sort_mode)
    items = items[:max_outcomes]
    max_count = max((c for _, c in items), default=1)

    # Layout constants
    pad_left = 80
    pad_right = 40
    pad_top = 30
    pad_bottom = 20
    bar_h = 28
    bar_gap = 8
    stat_h = 60

    chart_left = pad_left
    chart_bottom = pad_top + len(items) * (bar_h + bar_gap) + 8
    svg_height = chart_bottom + pad_bottom + stat_h
    chart_width = width - pad_left - pad_right

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{svg_height}" '
        f'viewBox="0 0 {width} {svg_height}">',
        f'<rect x="0" y="0" width="{width}" height="{svg_height}" '
        f'fill="{theme["background"]}" rx="8" />',
    ]

    # Title
    parts.append(
        f'<text x="{pad_left}" y="{pad_top - 8}" '
        f'fill="{theme["text"]}" font-family="sans-serif" font-size="15" '
        f'font-weight="bold">'
        f"Sampling Result · {result.num_shots} shots · "
        f"{result.num_qubits} qubits · {result.num_clbits} clbits"
        f"</text>"
    )

    # Grid lines
    grid_steps = 5
    for i in range(grid_steps + 1):
        x = chart_left + chart_width * i / grid_steps
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top:.1f}" '
            f'x2="{x:.1f}" y2="{chart_bottom:.1f}" '
            f'stroke="{theme["grid"]}" stroke-width="1" />'
        )

    # Bars
    for idx, (bs, cnt) in enumerate(items):
        y = pad_top + idx * (bar_h + bar_gap)
        bar_w = (cnt / max_count * chart_width) if max_count > 0 else 0
        color = _BAR_COLORS[idx % len(_BAR_COLORS)]

        # Gradient definition
        grad_id = f"bar_grad_{idx}"
        parts.append("<defs>")
        parts.append(
            f'<linearGradient id="{grad_id}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.25" />'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0.85" />'
            f"</linearGradient>"
        )
        parts.append("</defs>")

        # Bar rect
        if bar_w > 0:
            parts.append(
                f'<rect x="{chart_left:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'rx="5" fill="url(#{grad_id})" '
                f'stroke="{color}" stroke-width="1.2" />'
            )

        # Bitstring label
        parts.append(
            f'<text x="{chart_left - 10:.1f}" y="{y + bar_h / 2 + 5:.1f}" '
            f'text-anchor="end" fill="{theme["text"]}" '
            f'font-family="monospace" font-size="13" font-weight="bold">'
            f"{escape(bs)}</text>"
        )

        # Count label on the right of the bar
        pct = cnt / result.num_shots * 100 if result.num_shots else 0.0
        label_x = chart_left + bar_w + 8
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + bar_h / 2 + 5:.1f}" '
            f'fill="{theme["text"]}" font-family="monospace" font-size="12">'
            f"{cnt} ({pct:.1f}%)</text>"
        )

    # Statistics panel
    stat_y = chart_bottom + 12
    stat_bg_y = stat_y
    parts.append(
        f'<rect x="{pad_left:.1f}" y="{stat_bg_y:.1f}" '
        f'width="{chart_width:.1f}" height="{stat_h - 10:.1f}" '
        f'rx="6" fill="{theme["stat_bg"]}" '
        f'stroke="{theme["border"]}" stroke-width="1" />'
    )

    stat_text_lines = [
        f"Entropy: {stats.entropy:.3f} bits  |  "
        f"Normalized: {stats.normalized_entropy:.3f}  |  "
        f"σ: {stats.std:.1f}",
        f"Top gap: {stats.top_gap:.1%}  |  "
        f"Effective rank: {stats.effective_rank}/{stats.num_outcomes}  |  "
        f"Shots: {result.num_shots}",
    ]
    for li, line in enumerate(stat_text_lines):
        parts.append(
            f'<text x="{pad_left + 12:.1f}" y="{stat_y + 22 + li * 20:.1f}" '
            f'fill="{theme["text"]}" font-family="monospace" font-size="11">'
            f"{escape(line)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Comparison SVG renderer
# ---------------------------------------------------------------------------


def _render_comparison_svg(
    result_a: SimulationResult,
    result_b: SimulationResult,
    *,
    label_a: str,
    label_b: str,
    sort_mode: str,
    max_outcomes: int,
    theme_name: str,
    width: int,
) -> str:
    """Render a side-by-side comparison of two results as SVG.

    Args:
        result_a: First (left) simulation result.
        result_b: Second (right) simulation result.
        label_a: Label for the first result.
        label_b: Label for the second result.
        sort_mode: One of :class:`SortMode` values.
        max_outcomes: Maximum number of outcomes to display.
        theme_name: One of ``"official"``, ``"light"``, ``"dark"``.
        width: SVG width in pixels.

    Returns:
        An SVG string.
    """
    theme = _SVG_THEME_DATA.get(theme_name, _SVG_THEME_DATA["official"])
    stats_a = _compute_stats(result_a)
    stats_b = _compute_stats(result_b)

    # Union of all bitstrings from both results, sorted
    all_bs = sorted(set(result_a.counts) | set(result_b.counts))
    items: list[tuple[str, int, int]] = [
        (bs, result_a.get_count(bs), result_b.get_count(bs)) for bs in all_bs
    ]

    # Sort by count in result_a descending
    if sort_mode == SortMode.COUNT_DESC:
        items.sort(key=lambda x: x[1], reverse=True)
    elif sort_mode == SortMode.COUNT_ASC:
        items.sort(key=lambda x: x[1])
    elif sort_mode == SortMode.BITSTRING:
        items.sort(key=lambda x: x[0])
    else:
        items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_outcomes]

    max_count = max(
        max((c1 for _, c1, _ in items), default=1),
        max((c2 for _, _, c2 in items), default=1),
    )

    # Layout
    center_gap = 60
    half_chart_w = (width - 160 - center_gap) // 2
    pad_left = 80
    pad_top = 50
    pad_bottom = 20
    bar_h = 24
    bar_gap = 6
    stat_h = 55
    col_a_left = pad_left
    col_b_left = pad_left + half_chart_w + center_gap

    chart_bottom = pad_top + len(items) * (bar_h + bar_gap) + 8
    svg_height = chart_bottom + pad_bottom + stat_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{svg_height}" '
        f'viewBox="0 0 {width} {svg_height}">',
        f'<rect x="0" y="0" width="{width}" height="{svg_height}" '
        f'fill="{theme["background"]}" rx="8" />',
    ]

    # Column labels
    parts.append(
        f'<text x="{col_a_left + half_chart_w / 2:.1f}" y="30" '
        f'text-anchor="middle" fill="{theme["text"]}" '
        f'font-family="sans-serif" font-size="14" font-weight="bold">'
        f"{escape(label_a)} · {result_a.num_shots} shots</text>"
    )
    parts.append(
        f'<text x="{col_b_left + half_chart_w / 2:.1f}" y="30" '
        f'text-anchor="middle" fill="{theme["text"]}" '
        f'font-family="sans-serif" font-size="14" font-weight="bold">'
        f"{escape(label_b)} · {result_b.num_shots} shots</text>"
    )

    # Grid lines for both columns
    for col_left in [col_a_left, col_b_left]:
        for i in range(5 + 1):
            x = col_left + half_chart_w * i / 5
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad_top:.1f}" '
                f'x2="{x:.1f}" y2="{chart_bottom:.1f}" '
                f'stroke="{theme["grid"]}" stroke-width="1" />'
            )

    # Bars — left column
    for idx, (_bs, cnt, _) in enumerate(items):
        y = pad_top + idx * (bar_h + bar_gap)
        bar_w = (cnt / max_count * half_chart_w) if max_count > 0 else 0
        color = _BAR_COLORS[0]

        if bar_w > 0:
            parts.append(
                f'<rect x="{col_a_left:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" '
                f'fill="{color}" fill-opacity="0.75" '
                f'stroke="{color}" stroke-width="1" />'
            )
            label_x = col_a_left + bar_w + 6
            parts.append(
                f'<text x="{label_x:.1f}" y="{y + bar_h / 2 + 5:.1f}" '
                f'fill="{theme["text"]}" font-family="monospace" font-size="11">'
                f"{cnt}</text>"
            )

    # Bars — right column
    for idx, (_bs, _, cnt) in enumerate(items):
        y = pad_top + idx * (bar_h + bar_gap)
        bar_w = (cnt / max_count * half_chart_w) if max_count > 0 else 0
        color = _BAR_COLORS[1]

        if bar_w > 0:
            parts.append(
                f'<rect x="{col_b_left:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" '
                f'fill="{color}" fill-opacity="0.75" '
                f'stroke="{color}" stroke-width="1" />'
            )
            label_x = col_b_left + bar_w + 6
            parts.append(
                f'<text x="{label_x:.1f}" y="{y + bar_h / 2 + 5:.1f}" '
                f'fill="{theme["text"]}" font-family="monospace" font-size="11">'
                f"{cnt}</text>"
            )

    # Outcome labels in the center gap
    for idx, (bs, _, _) in enumerate(items):
        y = pad_top + idx * (bar_h + bar_gap)
        parts.append(
            f'<text x="{width / 2:.1f}" y="{y + bar_h / 2 + 5:.1f}" '
            f'text-anchor="middle" fill="{theme["text"]}" '
            f'font-family="monospace" font-size="13" font-weight="bold">'
            f"{escape(bs)}</text>"
        )

    # Stats below each column
    stat_y = chart_bottom + 14
    for col_left, stats, _label, _result_obj in [
        (col_a_left, stats_a, label_a, result_a),
        (col_b_left, stats_b, label_b, result_b),
    ]:
        parts.append(
            f'<rect x="{col_left:.1f}" y="{stat_y:.1f}" '
            f'width="{half_chart_w:.1f}" height="{stat_h - 8:.1f}" '
            f'rx="6" fill="{theme["stat_bg"]}" '
            f'stroke="{theme["border"]}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{col_left + 10:.1f}" y="{stat_y + 16:.1f}" '
            f'fill="{theme["text"]}" font-family="monospace" font-size="10">'
            f"H={stats.entropy:.3f}  H_norm={stats.normalized_entropy:.3f}  "
            f"σ={stats.std:.1f}  gap={stats.top_gap:.1%}  "
            f"rank={stats.effective_rank}/{stats.num_outcomes}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _sorted_outcomes(result: SimulationResult, sort_mode: str) -> list[tuple[str, int]]:
    """Return outcomes sorted by the given mode.

    Args:
        result: The simulation result.
        sort_mode: One of :class:`SortMode` values.

    Returns:
        List of ``(bitstring, count)`` tuples in the requested order.
    """
    if sort_mode == SortMode.COUNT_DESC:
        return sorted(result.counts.items(), key=lambda kv: kv[1], reverse=True)
    elif sort_mode == SortMode.COUNT_ASC:
        return sorted(result.counts.items(), key=lambda kv: kv[1])
    elif sort_mode == SortMode.BITSTRING:
        return sorted(result.counts.items(), key=lambda kv: kv[0])
    elif sort_mode == SortMode.PROBABILITY:
        # Sort counts by probability (descending), but return (str, int) tuples
        return sorted(
            result.counts.items(),
            key=lambda kv: result.probabilities.get(kv[0], 0.0),
            reverse=True,
        )
    else:
        return sorted(result.counts.items(), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
#  Main display class
# ---------------------------------------------------------------------------


@dataclass
class SamplingDisplay:
    """Visual representation of a :class:`SimulationResult`.

    Provides terminal-based bar charts and SVG export for sampling
    results. Supports multiple chart styles, sorting modes, and
    color themes.

    Differentiated from MindQuantum's display:
        - Three chart styles: horizontal bars (``"hbar"``), vertical
          bars (``"vbar"``), and dot/stem chart (``"dot"``).
        - 8-level Unicode block gradient for precise bar rendering.
        - Built-in statistical panel with entropy, variance, top-gap,
          and effective rank.
        - SVG export with three themes matching the circuit drawer.
        - Flexible sorting by count, bitstring, or probability.

    Args:
        result: The simulation result to display.
        style: Chart style — ``"hbar"``, ``"vbar"``, ``"dot"``, or
            ``"svg"``.
        sort_mode: Outcome sort order — ``"count_desc"`` (default),
            ``"count_asc"``, ``"bitstring"``, or ``"probability"``.
        max_outcomes: Maximum number of outcomes to show (default 16).
        theme: SVG theme name — ``"official"`` (default), ``"light"``,
            or ``"dark"``. Ignored for text styles.
        svg_width: Width of the generated SVG in pixels (default 680).

    Example:
        >>> result = SimulationResult.from_statevector(sv, shots=1024)
        >>> display = SamplingDisplay(result, style="hbar")
        >>> print(display)
        >>> # Or in Jupyter: display  (renders SVG automatically)
        >>> display.to_file("result.svg")
    """

    result: SimulationResult
    style: str = ChartStyle.HBAR
    sort_mode: str = SortMode.COUNT_DESC
    max_outcomes: int = 16
    theme: str = "official"
    svg_width: int = 680

    def __post_init__(self) -> None:
        valid_styles = (
            ChartStyle.HBAR,
            ChartStyle.VBAR,
            ChartStyle.DOT,
            ChartStyle.SVG,
        )
        if self.style not in valid_styles:
            raise ValueError(
                f"Unknown style {self.style!r}. Choose from: {valid_styles}"
            )
        valid_sorts = (
            SortMode.COUNT_DESC,
            SortMode.COUNT_ASC,
            SortMode.BITSTRING,
            SortMode.PROBABILITY,
        )
        if self.sort_mode not in valid_sorts:
            raise ValueError(
                f"Unknown sort_mode {self.sort_mode!r}. Choose from: {valid_sorts}"
            )
        valid_themes = ("official", "light", "dark")
        if self.theme not in valid_themes:
            raise ValueError(
                f"Unknown theme {self.theme!r}. Choose from: {valid_themes}"
            )

    # -- Text rendering -------------------------------------------------

    def __repr__(self) -> str:
        """Return a terminal-friendly bar chart (horizontal by default)."""
        return self._text_render()

    def __str__(self) -> str:
        """Return a terminal-friendly bar chart (horizontal)."""
        return self._text_render()

    def _text_render(self) -> str:
        """Render as text based on the selected style."""
        if self.style == ChartStyle.VBAR:
            return _render_vbar(
                self.result, sort_mode=self.sort_mode, max_outcomes=self.max_outcomes
            )
        elif self.style == ChartStyle.DOT:
            return _render_dot(
                self.result, sort_mode=self.sort_mode, max_outcomes=self.max_outcomes
            )
        else:
            # Default: horizontal bar
            return _render_hbar(
                self.result, sort_mode=self.sort_mode, max_outcomes=self.max_outcomes
            )

    # -- SVG / Jupyter ----------------------------------------------------

    def _repr_svg_(self) -> str:
        """Render as SVG, automatically used by Jupyter."""
        return _render_svg(
            self.result,
            sort_mode=self.sort_mode,
            max_outcomes=self.max_outcomes,
            theme_name=self.theme,
            width=self.svg_width,
        )

    # -- File export -------------------------------------------------------

    def to_file(self, filename: str | Path) -> None:
        """Write the chart as an SVG file.

        Args:
            filename: Destination file path (``.svg`` recommended).
        """
        svg = _render_svg(
            self.result,
            sort_mode=self.sort_mode,
            max_outcomes=self.max_outcomes,
            theme_name=self.theme,
            width=self.svg_width,
        )
        Path(filename).write_text(svg, encoding="utf-8")

    # -- Statistics --------------------------------------------------------

    @property
    def stats(self) -> SamplingStats:
        """The computed statistical summary for this result."""
        return _compute_stats(self.result)


# ---------------------------------------------------------------------------
#  Comparison helper
# ---------------------------------------------------------------------------


@dataclass
class ComparisonDisplay:
    """Side-by-side comparison of two sampling results.

    Renders two :class:`SimulationResult` objects as a unified SVG
    chart with bars for both results aligned by outcome.

    Args:
        result_a: First (left) sampling result.
        result_b: Second (right) sampling result.
        label_a: Label for the first result (default ``"A"``).
        label_b: Label for the second result (default ``"B"``).
        sort_mode: Outcome sort order.
        max_outcomes: Maximum number of outcomes to display.
        theme: SVG theme name.
        svg_width: SVG width in pixels.

    Example:
        >>> ideal = SimulationResult.from_statevector(sv, shots=1024)
        >>> noisy = sim.run(circuit.with_noise(), shots=1024)
        >>> cmp = ComparisonDisplay(ideal, noisy, label_a="ideal", label_b="noisy")
        >>> cmp  # renders SVG in Jupyter
        >>> cmp.to_file("comparison.svg")
    """

    result_a: SimulationResult
    result_b: SimulationResult
    label_a: str = "A"
    label_b: str = "B"
    sort_mode: str = SortMode.COUNT_DESC
    max_outcomes: int = 16
    theme: str = "official"
    svg_width: int = 680

    def _repr_svg_(self) -> str:
        return _render_comparison_svg(
            self.result_a,
            self.result_b,
            label_a=self.label_a,
            label_b=self.label_b,
            sort_mode=self.sort_mode,
            max_outcomes=self.max_outcomes,
            theme_name=self.theme,
            width=self.svg_width,
        )

    def to_file(self, filename: str | Path) -> None:
        """Write the comparison chart as an SVG file.

        Args:
            filename: Destination file path (``.svg`` recommended).
        """
        svg = _render_comparison_svg(
            self.result_a,
            self.result_b,
            label_a=self.label_a,
            label_b=self.label_b,
            sort_mode=self.sort_mode,
            max_outcomes=self.max_outcomes,
            theme_name=self.theme,
            width=self.svg_width,
        )
        Path(filename).write_text(svg, encoding="utf-8")

    def __repr__(self) -> str:
        return (
            f"ComparisonDisplay({self.label_a} vs {self.label_b}, "
            f"shots: {self.result_a.num_shots}/{self.result_b.num_shots})"
        )


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------


def display_sampling(
    result: SimulationResult,
    *,
    style: str = ChartStyle.HBAR,
    sort_mode: str = SortMode.COUNT_DESC,
    max_outcomes: int = 16,
    theme: str = "official",
    svg_width: int = 680,
) -> SamplingDisplay:
    """Create a :class:`SamplingDisplay` for a simulation result.

    Convenience function that mirrors the constructor signature but
    uses keyword-only arguments for clarity.

    Args:
        result: The simulation result to display.
        style: Chart style — ``"hbar"`` (default), ``"vbar"``, ``"dot"``,
            or ``"svg"``.
        sort_mode: Outcome sort order — ``"count_desc"`` (default),
            ``"count_asc"``, ``"bitstring"``, or ``"probability"``.
        max_outcomes: Maximum outcomes to display (default 16).
        theme: SVG theme — ``"official"`` (default), ``"light"``, or ``"dark"``.
        svg_width: SVG width in pixels (default 680).

    Returns:
        A configured :class:`SamplingDisplay` instance.
    """
    return SamplingDisplay(
        result=result,
        style=style,
        sort_mode=sort_mode,
        max_outcomes=max_outcomes,
        theme=theme,
        svg_width=svg_width,
    )


def compare_results(
    result_a: SimulationResult,
    result_b: SimulationResult,
    *,
    label_a: str = "A",
    label_b: str = "B",
    sort_mode: str = SortMode.COUNT_DESC,
    max_outcomes: int = 16,
    theme: str = "official",
    svg_width: int = 680,
) -> ComparisonDisplay:
    """Create a side-by-side :class:`ComparisonDisplay` of two results.

    Args:
        result_a: First (left) sampling result.
        result_b: Second (right) sampling result.
        label_a: Label for the first result.
        label_b: Label for the second result.
        sort_mode: Outcome sort order.
        max_outcomes: Maximum outcomes to display.
        theme: SVG theme name.
        svg_width: SVG width in pixels.

    Returns:
        A configured :class:`ComparisonDisplay` instance.
    """
    return ComparisonDisplay(
        result_a=result_a,
        result_b=result_b,
        label_a=label_a,
        label_b=label_b,
        sort_mode=sort_mode,
        max_outcomes=max_outcomes,
        theme=theme,
        svg_width=svg_width,
    )
