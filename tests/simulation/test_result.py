"""Tests for the SimulationResult class."""

from __future__ import annotations

import numpy as np
import pytest

from quspect.simulation.result import SimulationResult


class TestSimulationResultConstruction:
    """Tests for SimulationResult construction."""

    def test_construction_basic(self) -> None:
        """Basic construction with counts and probs."""
        result = SimulationResult(
            counts={"00": 512, "11": 488},
            probabilities={"00": 0.512, "11": 0.488},
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        assert result.num_shots == 1000
        assert result.num_qubits == 2
        assert result.num_clbits == 2
        assert result.statevector is None

    def test_construction_with_statevector(self) -> None:
        """Construction with an optional statevector as np.ndarray."""
        sv = np.zeros(4, dtype=complex)
        sv[0] = 1.0 + 0j
        result = SimulationResult(
            counts={},
            probabilities={},
            num_shots=0,
            num_qubits=2,
            num_clbits=2,
            statevector=sv,
        )
        assert result.statevector is not None
        assert result.statevector.shape == (4,)

    def test_construction_zero_shots(self) -> None:
        """Construction with 0 shots is valid."""
        result = SimulationResult(
            counts={},
            probabilities={},
            num_shots=0,
            num_qubits=3,
            num_clbits=0,
        )
        assert result.num_shots == 0


class TestMostFrequent:
    """Tests for most_frequent method."""

    def test_most_frequent_basic(self) -> None:
        """Returns the outcome with highest count."""
        result = SimulationResult(
            counts={"00": 512, "11": 488},
            probabilities={"00": 0.512, "11": 0.488},
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        assert result.most_frequent() == ("00", 512)

    def test_most_frequent_empty(self) -> None:
        """Returns ('', 0) when counts is empty."""
        result = SimulationResult(
            counts={},
            probabilities={},
            num_shots=0,
            num_qubits=2,
            num_clbits=2,
        )
        assert result.most_frequent() == ("", 0)

    def test_most_frequent_multiple_equal(self) -> None:
        """When counts are equal, returns any (first from max)."""
        result = SimulationResult(
            counts={"00": 500, "11": 500},
            probabilities={"00": 0.5, "11": 0.5},
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        bitstring, count = result.most_frequent()
        assert count == 500


class TestTopOutcomes:
    """Tests for top_outcomes method."""

    def test_top_outcomes_sorted(self) -> None:
        """Returns outcomes sorted by count descending."""
        result = SimulationResult(
            counts={"10": 100, "00": 300, "01": 200, "11": 400},
            probabilities={
                "10": 0.1,
                "00": 0.3,
                "01": 0.2,
                "11": 0.4,
            },
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        top = result.top_outcomes(4)
        assert top[0] == ("11", 400)
        assert top[1] == ("00", 300)
        assert top[2] == ("01", 200)
        assert top[3] == ("10", 100)

    def test_top_outcomes_limit(self) -> None:
        """Returns at most n outcomes."""
        result = SimulationResult(
            counts={"10": 100, "00": 300, "01": 200, "11": 400},
            probabilities={
                "10": 0.1,
                "00": 0.3,
                "01": 0.2,
                "11": 0.4,
            },
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        top = result.top_outcomes(2)
        assert len(top) == 2

    def test_top_outcomes_empty(self) -> None:
        """Returns empty list when counts is empty."""
        result = SimulationResult(
            counts={},
            probabilities={},
            num_shots=0,
            num_qubits=2,
            num_clbits=2,
        )
        top = result.top_outcomes(5)
        assert top == []


class TestGetCount:
    """Tests for get_count method."""

    def test_get_count_existing(self) -> None:
        """Returns count for existing bitstring."""
        result = SimulationResult(
            counts={"00": 500},
            probabilities={"00": 1.0},
            num_shots=500,
            num_qubits=1,
            num_clbits=1,
        )
        assert result.get_count("00") == 500

    def test_get_count_missing(self) -> None:
        """Returns 0 for missing bitstring."""
        result = SimulationResult(
            counts={"00": 500},
            probabilities={"00": 1.0},
            num_shots=500,
            num_qubits=1,
            num_clbits=1,
        )
        assert result.get_count("11") == 0


class TestGetProbability:
    """Tests for get_probability method."""

    def test_get_probability_existing(self) -> None:
        """Returns probability for existing bitstring."""
        result = SimulationResult(
            counts={"00": 500},
            probabilities={"00": 1.0},
            num_shots=500,
            num_qubits=1,
            num_clbits=1,
        )
        assert result.get_probability("00") == 1.0

    def test_get_probability_missing(self) -> None:
        """Returns 0.0 for missing bitstring."""
        result = SimulationResult(
            counts={"00": 500},
            probabilities={"00": 1.0},
            num_shots=500,
            num_qubits=1,
            num_clbits=1,
        )
        assert result.get_probability("11") == 0.0


class TestFromStatevector:
    """Tests for from_statevector constructor."""

    @staticmethod
    def _make_zero_state(n: int) -> np.ndarray:
        """Create |0…0⟩ state vector."""
        sv = np.zeros(1 << n, dtype=complex)
        sv[0] = 1.0 + 0j
        return sv

    def test_from_statevector_deterministic(self) -> None:
        """Sampling from |00> always gives '00'."""
        sv = self._make_zero_state(2)
        result = SimulationResult.from_statevector(sv, shots=100)
        assert result.counts["00"] == 100
        assert result.probabilities["00"] == 1.0

    def test_from_statevector_with_statevector_saved(self) -> None:
        """save_statevector=True includes the statevector."""
        sv = self._make_zero_state(2)
        result = SimulationResult.from_statevector(sv, shots=10, save_statevector=True)
        assert result.statevector is not None
        assert result.statevector.shape == (4,)

    def test_from_statevector_without_statevector(self) -> None:
        """save_statevector=False leaves statevector=None."""
        sv = self._make_zero_state(2)
        result = SimulationResult.from_statevector(sv, shots=10, save_statevector=False)
        assert result.statevector is None

    def test_from_statevector_negative_shots(self) -> None:
        """Negative shots raises ValueError."""
        sv = self._make_zero_state(2)
        with pytest.raises(ValueError, match="shots"):
            SimulationResult.from_statevector(sv, shots=-1)

    def test_from_statevector_zero_shots(self) -> None:
        """Zero shots gives result with all bitstrings and zero counts."""
        sv = self._make_zero_state(2)
        result = SimulationResult.from_statevector(sv, shots=0)
        assert result.num_shots == 0
        assert all(v == 0 for v in result.counts.values())
        assert all(v == 0.0 for v in result.probabilities.values())

    def test_from_statevector_reproducible(self) -> None:
        """Same seed gives same result."""
        sv = self._make_zero_state(2)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result1 = SimulationResult.from_statevector(sv, shots=100, rng=rng1)
        result2 = SimulationResult.from_statevector(sv, shots=100, rng=rng2)
        assert result1.counts == result2.counts


class TestRepr:
    """Tests for __repr__."""

    def test_repr_includes_info(self) -> None:
        """__repr__ includes key information."""
        result = SimulationResult(
            counts={"00": 500, "11": 500},
            probabilities={"00": 0.5, "11": 0.5},
            num_shots=1000,
            num_qubits=2,
            num_clbits=2,
        )
        r = repr(result)
        assert "SimulationResult" in r
        assert "shots=1000" in r
        assert "qubits=2" in r
