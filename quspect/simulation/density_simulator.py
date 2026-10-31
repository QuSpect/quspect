"""Density matrix simulator backed by the C++ native backend.

The :class:`DensityMatrixSimulator` executes quantum circuits using the
C++ ``DensityMatrix64`` for density matrix evolution, with gate matrices
applied via the optimized C++ ``apply_unitary`` kernel.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from quspect.backends._base import Backend
from quspect.gates._matrix_utils import gate_matrix
from quspect.simulation.density_matrix import DensityMatrix, _outcome_mask
from quspect.simulation.result import SimulationResult

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)

_PROB_EPS = 1e-15
_WARN_BRANCHES = 256
_MAX_BRANCHES = 4096


class DensityMatrixSimulator(Backend):
    """Execute quantum circuits using density matrix evolution via C++ backend.

    Uses the C++ ``DensityMatrix64`` for state storage and gate/channel
    application.  Suitable for circuits with noise channels and for
    systems up to ~12 qubits.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.simulation.density_simulator import (
        ...     DensityMatrixSimulator,
        ... )
        >>> circ = QuantumCircuit().h(0).depolarize(0.1, 0)
        >>> sim = DensityMatrixSimulator(seed=42)
        >>> result = sim.run(circ, shots=1000)
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self.seed = seed

    @property
    def name(self) -> str:
        return "density_matrix_simulator"

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Execute the circuit using density matrix evolution.

        Counts are keyed exactly like :meth:`Simulator.run`: each key is a
        classical-register bitstring of length ``circuit.num_clbits`` with
        clbit 0 as the **leftmost** character, filled according to the
        circuit's ``measure`` qubit -> clbit mapping.  Clbits that are never
        written read ``0``.  All ``2^num_clbits`` keys are present in the
        result (zero counts included).

        Mid-circuit measurements are handled exactly: the evolution is
        branched on every measurement outcome into ``(projected ρ,
        classical record)`` pairs, branches with identical classical
        records are merged, and the shots are drawn by multinomial sampling
        from the resulting exact joint distribution of the classical
        register.  Measurements followed by no further gate or channel are
        evaluated directly from the final branch states without branching.

        A circuit **without any measurement** is sampled as if every qubit
        ``q`` were measured into clbit ``q`` (keys have ``num_qubits`` bits).

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots. Must be >= 0.
            save_statevector: Unused (density matrix has no statevector).
            **options: Ignored extra options.

        Returns:
            A :class:`SimulationResult` with counts and empirical
            probabilities over the classical register.

        Raises:
            ValueError: If ``shots`` is negative, or if mid-circuit
                measurements create more than ``_MAX_BRANCHES`` distinct
                classical branches.

        Example:
            >>> from quspect import QuantumCircuit
            >>> circ = QuantumCircuit().x(0).id(2)
            >>> circ = circ.measure([0, 2], [1, 0])  # c0 <- q2, c1 <- q0
            >>> DensityMatrixSimulator(seed=1).run(circ, shots=10).counts
            {'00': 0, '01': 10, '10': 0, '11': 0}
        """
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")

        n = circuit.num_qubits
        instructions = circuit.instructions
        has_measurements = any(instr.is_measure for instr in instructions)
        num_clbits = circuit.num_clbits if has_measurements else n

        # Index of the last gate/channel: measurements after it are terminal and
        # are read off the final branch states instead of branching.
        last_op = max(
            (i for i, ins in enumerate(instructions) if ins.is_gate or ins.is_channel),
            default=-1,
        )

        # Unnormalized branch states (trace = branch probability) keyed by the
        # classical record (tuple of clbit values, clbit 0 first).
        branches: dict[tuple[int, ...], DensityMatrix] = {
            (0,) * num_clbits: DensityMatrix(n)
        }
        terminal: dict[int, int] = {}  # clbit -> qubit for terminal measurements

        for idx, instr in enumerate(instructions):
            if instr.is_gate and instr.gate is not None:
                mat = gate_matrix(instr.gate)
                # C++ ApplyUnitary 的比特序(gate bit i -> qubits[i])与 gate_matrix
                # 的大端约定相反,多比特门需反转 qubit 顺序才与 statevector 后端一致。
                for rho in branches.values():
                    rho.apply_gate(mat, tuple(instr.qubits)[::-1])
            elif instr.is_channel and instr.channel is not None:
                kraus = instr.channel.kraus_operators()
                for rho in branches.values():
                    rho.apply_channel(kraus, tuple(instr.qubits)[::-1])
            elif instr.is_measure:
                pairs = list(zip(instr.qubits, instr.clbits, strict=True))
                if idx > last_op:
                    for q, c in pairs:
                        terminal[c] = q
                else:
                    branches = _branch_on_measurement(branches, pairs)

        if not has_measurements:
            terminal = {q: q for q in range(n)}

        distribution = _clbit_distribution(branches, terminal, num_clbits)

        counts: dict[str, int] = {}
        if shots > 0:
            keys = list(distribution)
            pvals = np.clip(np.array([distribution[k] for k in keys]), 0.0, None)
            pvals = pvals / pvals.sum()
            rng = np.random.default_rng(self.seed)
            sampled = rng.multinomial(shots, pvals)
            for key, cnt in zip(keys, sampled, strict=True):
                if cnt:
                    counts[key] = int(cnt)

        total = max(shots, 1)
        probabilities = {bs: cnt / total for bs, cnt in counts.items()}

        # Fill in zero-count bitstrings
        for bs in _all_bitstrings(num_clbits):
            counts.setdefault(bs, 0)
            probabilities.setdefault(bs, 0.0)

        return SimulationResult(
            counts=dict(sorted(counts.items())),
            probabilities=dict(sorted(probabilities.items())),
            num_shots=shots,
            num_qubits=n,
            num_clbits=num_clbits,
            statevector=None,
        )

    def density_matrix(self, circuit: QuantumCircuit) -> DensityMatrix:
        """Evolve the circuit and return the final density matrix (no sampling).

        Args:
            circuit: The circuit to execute.

        Returns:
            The final :class:`DensityMatrix` after evolution.
        """
        n = circuit.num_qubits
        rho = DensityMatrix(n)

        for instr in circuit.instructions:
            if instr.is_gate and instr.gate is not None:
                mat = gate_matrix(instr.gate)
                # 见 run():多比特门需反转 qubit 顺序以匹配 gate_matrix 大端约定。
                rho.apply_gate(mat, tuple(instr.qubits)[::-1])
            elif instr.is_channel and instr.channel is not None:
                kraus = instr.channel.kraus_operators()
                rho.apply_channel(kraus, tuple(instr.qubits)[::-1])
            # Skip measurements when computing density matrix

        return rho.copy()


def _branch_on_measurement(
    branches: dict[tuple[int, ...], DensityMatrix],
    pairs: list[tuple[int, int]],
) -> dict[tuple[int, ...], DensityMatrix]:
    """Split every branch on the outcomes of a mid-circuit measurement.

    Args:
        branches: Unnormalized branch states keyed by classical record.
        pairs: ``(qubit, clbit)`` pairs of the measurement instruction.

    Returns:
        The new branches; branches with equal records are merged by adding
        their (unnormalized) density matrices, and zero-probability branches
        are dropped.

    Raises:
        ValueError: If the number of branches exceeds ``_MAX_BRANCHES``.
    """
    qubits = tuple(q for q, _ in pairs)
    k = len(qubits)
    new: dict[tuple[int, ...], DensityMatrix] = {}
    for record, rho in branches.items():
        data = rho._data
        for outcome in range(1 << k):
            mask = _outcome_mask(rho.dim, qubits, outcome)
            projected = data * np.outer(mask, mask)
            if float(np.real(np.trace(projected))) <= _PROB_EPS:
                continue
            rec = list(record)
            for j, (_, c) in enumerate(pairs):
                rec[c] = (outcome >> j) & 1
            key = tuple(rec)
            if key in new:
                new[key]._data[...] += projected
            else:
                new[key] = DensityMatrix.from_array(projected)
    if len(new) > _MAX_BRANCHES:
        raise ValueError(
            f"Mid-circuit measurements produced {len(new)} distinct classical "
            f"branches (limit {_MAX_BRANCHES}); the exact density-matrix "
            "simulation is too expensive for this circuit"
        )
    if len(new) > _WARN_BRANCHES:
        warnings.warn(
            f"Mid-circuit measurements produced {len(new)} classical branches; "
            "density-matrix simulation cost grows linearly with this number",
            ResourceWarning,
            stacklevel=3,
        )
    return new


def _clbit_distribution(
    branches: dict[tuple[int, ...], DensityMatrix],
    terminal: dict[int, int],
    num_clbits: int,
) -> dict[str, float]:
    """Exact joint distribution of the classical register.

    Args:
        branches: Unnormalized branch states keyed by classical record.
        terminal: ``clbit -> qubit`` map of terminal measurements.
        num_clbits: Width of the classical register.

    Returns:
        Mapping from bitstring (clbit 0 leftmost) to probability.
    """
    clbits = sorted(terminal)
    qubits = tuple(sorted(set(terminal.values())))
    pos = {q: j for j, q in enumerate(qubits)}
    dist: dict[str, float] = {}
    for record, rho in branches.items():
        # Branch states are unnormalized, so these are joint probabilities.
        probs = rho.probabilities(qubits)
        for outcome, p in enumerate(probs):
            if p <= _PROB_EPS:
                continue
            rec = list(record)
            for c in clbits:
                rec[c] = (outcome >> pos[terminal[c]]) & 1
            key = "".join(str(b) for b in rec[:num_clbits])
            dist[key] = dist.get(key, 0.0) + float(p)
    return dist


def _all_bitstrings(num_qubits: int) -> list[str]:
    if num_qubits == 0:
        return [""]
    result = []
    for i in range(1 << num_qubits):
        bs = "".join(str((i >> q) & 1) for q in range(num_qubits))
        result.append(bs)
    return sorted(result)
