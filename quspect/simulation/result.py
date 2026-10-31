"""Simulation result data class.

The :class:`SimulationResult` collects measurement outcomes from
quantum circuit execution, providing counts, probabilities, and
an optional final state vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.visualization.sampling_viz import SamplingDisplay


@dataclass
class SimulationResult:
    """Result of executing a quantum circuit via the simulator.

    Accumulates measurement outcomes across multiple shots and provides
    counts, probabilities, and optionally the final state vector.

    Attributes:
        counts: Dict mapping bitstring → count. The i-th character of
            a bitstring corresponds to classical bit ``i``.
        probabilities: Dict mapping bitstring → probability.
        num_shots: Total number of simulation shots executed.
        num_qubits: Number of qubits in the simulated circuit.
        num_clbits: Number of classical bits measured.
        statevector: Optional final :class:`~quspect.simulation.StateVector`
            (included only when ``save_statevector=True``).
        success: Whether this circuit completed successfully. This is mainly
            useful for partial failures in a cloud batch.
        error: Provider error message for a failed batch item, if any.
        metadata: Backend and provider metadata associated with this result.
        memory: Optional per-shot measurement outcomes. Cloud backends use
            this when a provider returns ``result_format="memory"``.

    Example:
        >>> result = SimulationResult(
        ...     counts={"00": 512, "11": 488},
        ...     probabilities={"00": 0.512, "11": 0.488},
        ...     num_shots=1000,
        ...     num_qubits=2,
        ...     num_clbits=2,
        ... )
        >>> result.most_frequent()
        ('00', 512)
    """

    counts: dict[str, int]
    probabilities: dict[str, float]
    num_shots: int
    num_qubits: int
    num_clbits: int
    statevector: np.ndarray | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    memory: list[str] | None = None

    def __post_init__(self) -> None:
        """Detach mutable provider metadata from its source payload."""
        self.metadata = dict(self.metadata)
        self.memory = None if self.memory is None else list(self.memory)

    def most_frequent(self) -> tuple[str, int]:
        """Return the most frequent measurement outcome.

        Returns:
            A tuple ``(bitstring, count)`` for the outcome with the
            highest count. If there are no counts, returns ``("", 0)``.

        Example:
            >>> result.most_frequent()
            ('00', 512)
        """
        if not self.counts:
            return ("", 0)
        best = max(self.counts.items(), key=lambda kv: kv[1])
        return (best[0], best[1])

    def top_outcomes(self, n: int = 5) -> list[tuple[str, int]]:
        """Return the top ``n`` most frequent outcomes, sorted by count.

        Args:
            n: Maximum number of outcomes to return.

        Returns:
            A list of ``(bitstring, count)`` tuples, sorted descending
            by count.

        Example:
            >>> result.top_outcomes(3)
            [('00', 512), ('11', 488), ('01', 0)]
        """
        sorted_items = sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)
        return [(k, v) for k, v in sorted_items[:n]]

    def get_count(self, bitstring: str) -> int:
        """Return the count for a specific bitstring.

        Args:
            bitstring: The measurement outcome to look up.

        Returns:
            The count for ``bitstring``, or 0 if not present.
        """
        return self.counts.get(bitstring, 0)

    def get_probability(self, bitstring: str) -> float:
        """Return the probability for a specific bitstring.

        Args:
            bitstring: The measurement outcome to look up.

        Returns:
            The probability for ``bitstring``, or 0.0 if not present.
        """
        return self.probabilities.get(bitstring, 0.0)

    def display(
        self,
        *,
        style: str = "hbar",
        sort_mode: str = "count_desc",
        max_outcomes: int = 16,
        theme: str = "official",
        svg_width: int = 680,
    ) -> SamplingDisplay:
        """Create a :class:`SamplingDisplay` for this result.

        Convenience method that constructs a
        :class:`~quspect.visualization.sampling_viz.SamplingDisplay`
        directly from the result instance.  In Jupyter the returned
        object automatically renders as an SVG bar chart.

        Args:
            style: Chart style — ``"hbar"``, ``"vbar"``, ``"dot"``,
                or ``"svg"``.
            sort_mode: Outcome sort order — ``"count_desc"`` (default),
                ``"count_asc"``, ``"bitstring"``, or ``"probability"``.
            max_outcomes: Maximum outcomes to display (default 16).
            theme: SVG theme — ``"official"`` (default), ``"light"``,
                or ``"dark"``.
            svg_width: SVG width in pixels (default 680).

        Returns:
            A configured :class:`SamplingDisplay` instance.

        Example:
            >>> result.display()
            >>> result.display(style="vbar", sort_mode="probability")
            >>> result.display(theme="dark").to_file("result.svg")
        """
        from quspect.visualization.sampling_viz import SamplingDisplay

        return SamplingDisplay(
            result=self,
            style=style,
            sort_mode=sort_mode,
            max_outcomes=max_outcomes,
            theme=theme,
            svg_width=svg_width,
        )

    def __repr__(self) -> str:
        if not self.success:
            return (
                f"SimulationResult(success=False, shots={self.num_shots}, "
                f"qubits={self.num_qubits}, error={self.error!r})"
            )
        top = self.top_outcomes(5)
        summary = ", ".join(f"{k}: {v}" for k, v in top[:3])
        sv_info = (
            "with statevector" if self.statevector is not None else "no statevector"
        )
        return (
            f"SimulationResult(shots={self.num_shots}, "
            f"qubits={self.num_qubits}, "
            f"clbits={self.num_clbits}, "
            f"top=[{summary}], "
            f"{sv_info})"
        )

    @classmethod
    def from_native(
        cls,
        native_result: Any,
        *,
        circuit: Any,
        shots: int,
        save_statevector: bool = False,
    ) -> SimulationResult:
        """Create a SimulationResult from a C++ native ExecutionResult.

        Args:
            native_result: A :class:`~quspect.native.ExecutionResult` from
                the C++ backend.
            circuit: The :class:`~quspect.circuit.QuantumCircuit` that was
                executed.
            shots: Number of measurement shots.
            save_statevector: Whether the native result includes a state
                vector.

        Returns:
            A :class:`SimulationResult` with counts, probabilities, and
            optional state vector.
        """
        counts: dict[str, int] = {}
        for bs, cnt in native_result.counts.items():
            # C++ returns bitstrings MSB-first; Python convention is LSB-first
            # (clbit 0 as position 0). Reverse to match.
            counts[bs[::-1]] = cnt

        total = max(shots, 1)
        probabilities = {bs: cnt / total for bs, cnt in counts.items()}

        statevector = None
        if save_statevector and native_result.statevector_f64:
            statevector = np.array(native_result.statevector_f64, dtype=complex)

        return cls(
            counts=dict(sorted(counts.items())),
            probabilities=dict(sorted(probabilities.items())),
            num_shots=shots,
            num_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            statevector=statevector,
        )

    @classmethod
    def from_statevector(
        cls,
        sv: np.ndarray,
        shots: int = 1024,
        *,
        save_statevector: bool = False,
        rng: np.random.Generator | None = None,
    ) -> SimulationResult:
        """Sample measurements from a state vector directly.

        This is a convenience constructor for testing and for sampling
        from a final state vector. All qubits are measured with qubit
        ``i`` mapped to classical bit ``i``.

        Args:
            sv: The state vector as a numpy complex array of shape ``(2^n,)``.
            shots: Number of measurement shots.
            save_statevector: If True, include the state vector in the
                result.
            rng: A numpy random :class:`~numpy.random.Generator`. If None,
                a default RNG is used.

        Returns:
            A :class:`SimulationResult` with sampled counts.

        Raises:
            ValueError: If ``shots`` is negative.
        """
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")

        dim = len(sv)
        n = int(np.log2(dim))
        if 1 << n != dim:
            raise ValueError(f"State vector length {dim} is not a power of 2")

        if rng is None:
            rng = np.random.default_rng()

        probs = np.abs(sv) ** 2
        indices = (
            rng.choice(dim, size=shots, p=probs)
            if shots > 0
            else np.array([], dtype=int)
        )

        counts: dict[str, int] = {}
        for idx in indices:
            bs = "".join(str((idx >> q) & 1) for q in range(n))
            counts[bs] = counts.get(bs, 0) + 1

        all_bitstrings = _all_bitstrings(n)
        for bs in all_bitstrings:
            counts.setdefault(bs, 0)

        probabilities = {
            bs: cnt / shots if shots > 0 else 0.0 for bs, cnt in counts.items()
        }

        return cls(
            counts=dict(sorted(counts.items())),
            probabilities=dict(sorted(probabilities.items())),
            num_shots=shots,
            num_qubits=n,
            num_clbits=n,
            statevector=sv.copy() if save_statevector else None,
        )


def _all_bitstrings(num_qubits: int) -> list[str]:
    """Return all bitstrings for a given number of qubits, sorted."""
    if num_qubits == 0:
        return [""]
    result = []
    for i in range(1 << num_qubits):
        bs = "".join(str((i >> q) & 1) for q in range(num_qubits))
        result.append(bs)
    return sorted(result)
