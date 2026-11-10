"""Tests for the calibration-aware placement helpers shared by cloud backends."""

from __future__ import annotations

import math
import random

import pytest

from quspect import QuantumCircuit
from quspect.backends._compile import (
    estimated_log_fidelity,
    fill_missing_fidelities,
    keep_best,
    placement_regions,
    select_region,
)


def _grid(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if col + 1 < cols:
                edges.append((index, index + 1))
            if row + 1 < rows:
                edges.append((index, index + cols))
    return edges


# ---------------------------------------------------------------------------
# select_region(compact_ties=True)
# ---------------------------------------------------------------------------


def test_default_tie_break_is_unchanged() -> None:
    """Origin Quantum and Tianyan rely on the lowest-index tie-break."""
    edges = _grid(4, 4)
    assert select_region(4, range(16), edges, {}, {}) == [0, 1, 2, 3]
    assert select_region(6, range(16), edges, {}, {}) == [0, 1, 2, 3, 4, 5]


def test_compact_ties_grow_a_block_instead_of_a_line() -> None:
    edges = _grid(4, 4)

    def region(size: int) -> list[int] | None:
        return select_region(size, range(16), edges, {}, {}, compact_ties=True)

    assert region(1) == [0]
    assert region(2) == [0, 1]
    assert region(4) == [0, 1, 4, 5]
    assert region(6) == [0, 1, 2, 4, 5, 6]
    assert region(16) == list(range(16))
    assert region(17) is None


def test_compact_ties_never_override_calibration() -> None:
    """With distinct fidelities nothing ties: both modes pick the same region."""
    rng = random.Random(5)
    edges = _grid(5, 5)
    for _ in range(20):
        qubit = {q: rng.uniform(0.9, 0.99) for q in range(25)}
        edge = {e: rng.uniform(0.9, 0.995) for e in edges}
        for size in (2, 3, 5, 8):
            assert select_region(
                size, range(25), edges, qubit, edge, compact_ties=True
            ) == select_region(size, range(25), edges, qubit, edge)


def test_compact_ties_only_decide_among_equally_good_neighbours() -> None:
    # A 3x3 grid whose top row is joined by better couplers than the rest:
    # the three best-joined qubits win although they form a line.
    edges = _grid(3, 3)
    edge = dict.fromkeys(edges, 0.9)
    edge[(0, 1)] = edge[(1, 2)] = 0.99
    assert select_region(3, range(9), edges, {}, edge, compact_ties=True) == [0, 1, 2]
    # The fourth qubit is a tie between 3, 4 and 5; 3 is nearest the seed (0).
    assert select_region(4, range(9), edges, {}, edge, compact_ties=True) == [
        0,
        1,
        2,
        3,
    ]


# ---------------------------------------------------------------------------
# placement_regions
# ---------------------------------------------------------------------------


def test_placement_regions_offer_the_circuit_size_and_a_workspace() -> None:
    edges = _grid(4, 4)
    readout = {q: (0.6 if q < 8 else 0.99) for q in range(16)}
    regions = placement_regions(3, range(16), edges, readout, {})
    assert [len(region) for region in regions] == [3, 5]
    assert all(min(region) >= 8 for region in regions)


def test_placement_regions_skip_sizes_that_do_not_fit() -> None:
    line = [(0, 1), (1, 2), (2, 3)]
    assert placement_regions(3, range(4), line, {}, {}) == [[0, 1, 2]]
    assert placement_regions(4, range(4), line, {}, {}) == [[0, 1, 2, 3]]
    assert placement_regions(5, range(4), line, {}, {}) == []
    # Two regions of the same qubits are offered once.
    assert placement_regions(2, range(4), line, {}, {}, workspace=0) == [[0, 1]]


# ---------------------------------------------------------------------------
# fill_missing_fidelities
# ---------------------------------------------------------------------------


def test_missing_values_count_as_the_worst_published_one() -> None:
    assert fill_missing_fidelities({0: 0.99, 1: 0.9}, range(4)) == {
        0: 0.99,
        1: 0.9,
        2: 0.9,
        3: 0.9,
    }
    # A published zero stays zero, and is not what a missing value becomes.
    assert fill_missing_fidelities({0: 0.0, 1: 0.95}, range(3)) == {
        0: 0.0,
        1: 0.95,
        2: 0.95,
    }


def test_nothing_published_stays_empty() -> None:
    assert fill_missing_fidelities({}, range(4)) == {}


# ---------------------------------------------------------------------------
# estimated_log_fidelity
# ---------------------------------------------------------------------------


def test_estimate_multiplies_gates_and_readouts() -> None:
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.swap(1, 2)  # three two-qubit gates on the chip
    circuit.measure([0, 2], [0, 1])
    estimate = estimated_log_fidelity(
        circuit,
        edge_fidelity={(0, 1): 0.9, (1, 2): 0.8},
        readout_fidelity={0: 0.95, 1: 0.5, 2: 0.7},
    )
    assert math.exp(estimate) == pytest.approx(0.9 * 0.8**3 * 0.95 * 0.7)

    with_gates = estimated_log_fidelity(
        circuit,
        edge_fidelity={(0, 1): 0.9, (1, 2): 0.8},
        readout_fidelity={0: 0.95, 1: 0.5, 2: 0.7},
        gate_fidelity={0: 0.99},
    )
    assert math.exp(with_gates) == pytest.approx(0.9 * 0.8**3 * 0.95 * 0.7 * 0.99)


def test_estimate_reads_wires_through_the_layout() -> None:
    circuit = QuantumCircuit()
    circuit.cx(0, 1)
    circuit.measure([1], [0])
    estimate = estimated_log_fidelity(
        circuit,
        edge_fidelity={(4, 7): 0.5},
        readout_fidelity={7: 0.25},
        layout=[7, 4],
    )
    assert math.exp(estimate) == pytest.approx(0.5 * 1.0)  # wire 1 is qubit 4
    # Uncalibrated gates and readouts count as perfect.
    assert estimated_log_fidelity(circuit, edge_fidelity={}, readout_fidelity={}) == 0.0


# ---------------------------------------------------------------------------
# keep_best
# ---------------------------------------------------------------------------


def test_keep_best_prefers_the_first_candidate_on_ties() -> None:
    scores = {"today": -1.0, "same": -1.0, "better": -0.5, "worse": -2.0}
    assert keep_best(["today", "same", "worse"], scores.__getitem__) == "today"
    assert keep_best(["today", "better", "same"], scores.__getitem__) == "better"
    # Differences below the tolerance do not move a placement.
    close = {"today": -1.0, "noise": -1.0 + 1e-12}
    assert keep_best(["today", "noise"], close.__getitem__) == "today"


def test_keep_best_compiles_lazily_and_needs_a_candidate() -> None:
    compiled: list[str] = []

    def candidates():
        for name in ("a", "b"):
            compiled.append(name)
            yield name

    assert keep_best(candidates(), {"a": 0.0, "b": -1.0}.__getitem__) == "a"
    assert compiled == ["a", "b"]
    with pytest.raises(ValueError, match="at least one"):
        keep_best([], lambda item: 0.0)
