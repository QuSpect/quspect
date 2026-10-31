"""Base classes for the DAG optimization pass framework.

All passes follow a common interface and can be composed via
:class:`PassManager` into optimization pipelines.

The recommended pipeline order is **cancel → commute → rotate**,
supported by 2024 research on NISQ circuit optimization:

    1. GateCancellation — remove inverse gate pairs
    2. CommutationAnalysis — reorder to enable further cancellations
    3. RotationMerging — merge same-axis rotations
    4. SingleQubitGateFusion — fuse arbitrary single-qubit gates
    5. EliminateDiagonalBeforeMeasurement — remove redundant diagonal gates
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspect.dag.dag_circuit import DAGCircuit

logger = logging.getLogger(__name__)


class BasePass(ABC):
    """Abstract base class for DAG optimization passes.

    Each pass receives a :class:`DAGCircuit`, performs a specific
    transformation, and returns a (possibly modified) DAGCircuit.

    Passes should be **deterministic** — given the same input DAG,
    they must produce the same output DAG (architecture requirement
    from the Qiskit 2024 design principles).

    Attributes:
        name: Human-readable pass name (defaults to class name).
        enabled: Whether this pass is active. Set to ``False`` to skip.

    Example:
        >>> class MyPass(BasePass):
        ...     def run(self, dag):
        ...         # modify dag...
        ...         return dag
    """

    #: True for passes that change a DAG only through :class:`DAGCircuit`
    #: methods (so :attr:`DAGCircuit.modification_count` records every
    #: change).  :class:`PassManager` then skips the structural signature
    #: when such a pass returns the same, unmodified DAG.
    reports_modifications: bool = False

    def __init__(self) -> None:
        self.name: str = self.__class__.__name__
        self.enabled: bool = True

    @abstractmethod
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Execute this pass on a DAGCircuit.

        Args:
            dag: The input DAG to transform.

        Returns:
            The transformed DAG (may be the same object if
            modified in-place, or a new DAGCircuit).
        """
        ...

    def __repr__(self) -> str:
        status = "" if self.enabled else " (disabled)"
        return f"{self.name}(){status}"


# =========================================================================
# Pass statistics
# =========================================================================


@dataclass
class PassStats:
    """Statistics collected during a single pass execution.

    Attributes:
        pass_name: Name of the pass that was executed.
        nodes_before: Node count before the pass.
        nodes_after: Node count after the pass.
        elapsed_ms: Execution time in milliseconds.
        structure_changed: Whether the pass modified the circuit structure
            (gate sequence on any wire), even if the node count is
            unchanged, e.g. a commutation-based reordering.
        changed: Whether the DAG was modified: the node count changed or
            ``structure_changed`` is True.
    """

    pass_name: str
    nodes_before: int
    nodes_after: int
    elapsed_ms: float
    structure_changed: bool = False
    changed: bool = field(init=False)

    def __post_init__(self) -> None:
        self.changed = self.structure_changed or self.nodes_before != self.nodes_after

    @property
    def nodes_removed(self) -> int:
        """Return the number of nodes removed by this pass."""
        return max(0, self.nodes_before - self.nodes_after)

    @property
    def nodes_added(self) -> int:
        """Return the number of nodes added by this pass."""
        return max(0, self.nodes_after - self.nodes_before)

    @property
    def reduction_pct(self) -> float:
        """Return the percentage reduction in node count."""
        if self.nodes_before == 0:
            return 0.0
        return (self.nodes_before - self.nodes_after) / self.nodes_before * 100.0


# =========================================================================
# Structural signature
# =========================================================================


def dag_signature(dag: DAGCircuit) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return a canonical structural signature of a DAG.

    The signature lists, for every qubit and clbit wire, the ordered
    sequence of operation instructions on that wire (instruction type,
    gate / channel name, parameters and ordered operands, via the
    instruction ``repr``). Two DAGs with equal signatures describe the
    same circuit regardless of node ids or topological tie-breaking,
    so the signature detects both gate removal and reordering.

    Args:
        dag: The DAG to fingerprint.

    Returns:
        A hashable tuple ``((wire, (instr_repr, ...)), ...)`` sorted by
        wire name.

    Example:
        >>> from quspect import QuantumCircuit
        >>> a = QuantumCircuit().h(0).h(1).to_dag()
        >>> b = QuantumCircuit().h(1).h(0).to_dag()
        >>> dag_signature(a) == dag_signature(b)
        True
    """
    return _dag_signature(dag, None)


#: Signature type: ``((wire, (instr_repr, ...)), ...)``.
_Signature = tuple[tuple[str, tuple[str, ...]], ...]


def _dag_signature(
    dag: DAGCircuit, cache: dict[int, tuple[object, str]] | None
) -> _Signature:
    """Compute :func:`dag_signature`, memoising instruction reprs.

    ``cache`` maps ``id(instruction)`` to ``(instruction, repr)``; keeping
    a strong reference to the instruction prevents ``id`` reuse.
    Instructions are immutable, so identity implies an identical repr.
    """
    wires = _wire_sequences(dag, cache)
    named = (
        (f"q{w}" if w >= 0 else f"c{-1 - w}", tuple(seq)) for w, seq in wires.items()
    )
    return tuple(sorted(named))


def _wire_sequences(
    dag: DAGCircuit, cache: dict[int, tuple[object, str]] | None
) -> dict[int, list[str]]:
    """Per-wire instruction reprs (qubit ``q`` → key ``q``, clbit ``c`` → ``-1 - c``).

    Two DAGs have equal :func:`dag_signature` iff these dicts are equal;
    the pass manager compares them directly (no wire names, no sorting).
    """
    wires: dict[int, list[str]] = {}
    for node in dag.operation_nodes():
        instr = node.instruction
        if cache is None:
            key = repr(instr)
        else:
            hit = cache.get(id(instr))
            if hit is not None and hit[0] is instr:
                key = hit[1]
            else:
                key = repr(instr)
                cache[id(instr)] = (instr, key)
        for q in node.qubits:
            seq = wires.get(q)
            if seq is None:
                wires[q] = seq = []
            seq.append(key)
        for c in node.clbits:
            w = -1 - c
            seq = wires.get(w)
            if seq is None:
                wires[w] = seq = []
            seq.append(key)
    return wires


# =========================================================================
# PassManager
# =========================================================================


class PassManager:
    """Manages a sequence of optimization passes with statistics tracking.

    PassManager runs passes sequentially, optionally repeating the
    entire pipeline until convergence (no pass reports changes).

    This implements the **cancel → commute → rotate** pipeline
    recommended by 2024 research on NISQ circuit optimization.

    Attributes:
        passes: The registered pass list.
        stats_history: Per-iteration statistics, appended on each
            call to :meth:`run`.

    Example:
        >>> pm = PassManager([
        ...     GateCancellation(),
        ...     CommutationAnalysis(),
        ...     RotationMerging(),
        ...     SingleQubitGateFusion(),
        ...     EliminateDiagonalBeforeMeasurement(),
        ... ])
        >>> optimized_dag = pm.run(dag, max_iterations=3)
        >>> print(pm.summary())
    """

    def __init__(self, passes: list[BasePass]) -> None:
        """Initialize the pass manager.

        Args:
            passes: Ordered list of passes to execute.
        """
        self._passes = passes
        self.stats_history: list[PassStats] = []

    def run(self, dag: DAGCircuit, max_iterations: int = 10) -> DAGCircuit:
        """Run all passes sequentially, repeating until convergence.

        Convergence is detected when a full iteration of all enabled
        passes leaves the circuit structure unchanged. The structure is
        compared via :func:`dag_signature` (the sequence of instructions
        -- name, parameters, operands -- on every wire), so a pass that
        only reorders commuting gates also counts as a change. This
        handles the ``cancel → commute → rotate → (re-cancel)`` pattern
        where commutation enables further cancellations.

        Args:
            dag: The input DAG to optimize.
            max_iterations: Maximum number of full pipeline iterations.

        Returns:
            The optimized DAGCircuit.

        Raises:
            RuntimeError: If the pipeline does not converge within
                ``max_iterations``.
        """
        self.stats_history.clear()
        current = dag
        repr_cache: dict[int, tuple[object, str]] = {}
        signature = _wire_sequences(current, repr_cache)

        for iteration in range(max_iterations):
            initial_nodes = current.num_nodes()
            logger.debug(
                "PassManager iteration %d: %d nodes",
                iteration + 1,
                initial_nodes,
            )

            iteration_changed = False
            for p in self._passes:
                if not p.enabled:
                    logger.debug("  %s (skipped — disabled)", p.name)
                    continue

                nodes_before = current.num_nodes()
                signature_before = signature
                dag_before, version_before = current, current.modification_count
                t0 = time.perf_counter()
                current = p.run(current)
                t1 = time.perf_counter()
                nodes_after = current.num_nodes()
                if not (
                    p.reports_modifications
                    and current is dag_before
                    and current.modification_count == version_before
                ):
                    signature = _wire_sequences(current, repr_cache)

                stats = PassStats(
                    pass_name=p.name,
                    nodes_before=nodes_before,
                    nodes_after=nodes_after,
                    elapsed_ms=(t1 - t0) * 1000,
                    structure_changed=signature != signature_before,
                )
                self.stats_history.append(stats)

                if stats.changed:
                    iteration_changed = True
                    logger.debug(
                        "  %s → %d nodes (%.1f ms, %+d nodes)",
                        p.name,
                        nodes_after,
                        stats.elapsed_ms,
                        nodes_after - nodes_before,
                    )
                else:
                    logger.debug(
                        "  %s → unchanged (%.1f ms)",
                        p.name,
                        stats.elapsed_ms,
                    )

            if not iteration_changed:
                logger.debug(
                    "PassManager converged after %d iteration(s)",
                    iteration + 1,
                )
                return current

        # Max iterations reached without convergence
        logger.warning(
            "PassManager: did not converge within %d iterations (final node count: %d)",
            max_iterations,
            current.num_nodes(),
        )
        raise RuntimeError(
            f"PassManager did not converge within {max_iterations} iteration(s) "
            f"(final node count: {current.num_nodes()})"
        )

    @property
    def passes(self) -> list[BasePass]:
        """Return the list of registered passes."""
        return list(self._passes)

    @property
    def total_nodes_removed(self) -> int:
        """Return total nodes removed across all passes."""
        return sum(s.nodes_removed for s in self.stats_history)

    @property
    def total_elapsed_ms(self) -> float:
        """Return total execution time in milliseconds."""
        return sum(s.elapsed_ms for s in self.stats_history)

    def summary(self) -> str:
        """Return a human-readable optimization summary.

        Returns:
            Multi-line string with per-pass statistics and totals.

        Example:
            >>> pm = PassManager([...])
            >>> pm.run(dag)
            >>> print(pm.summary())
            ============================================================
            PassManager Optimization Summary
            ============================================================
              GateCancellation              : 12 → 10 nodes (2 removed, 3.2 ms)
              CommutationAnalysis           : 10 → 10 nodes (unchanged, 1.5 ms)
              RotationMerging               : 10 →  9 nodes (1 removed, 2.1 ms)
            ------------------------------------------------------------
              Total                         :  3 nodes removed, 6.8 ms
            ============================================================
        """
        lines = ["=" * 60, "PassManager Optimization Summary", "=" * 60]

        for stats in self.stats_history:
            delta = ""
            if stats.nodes_removed > 0:
                delta = f" ({stats.nodes_removed} removed)"
            elif stats.nodes_added > 0:
                delta = f" ({stats.nodes_added} added)"
            else:
                delta = " (unchanged)"

            lines.append(
                f"  {stats.pass_name:<30}: "
                f"{stats.nodes_before:>3} → {stats.nodes_after:>3} nodes"
                f"{delta}, {stats.elapsed_ms:.1f} ms"
            )

        lines.append("-" * 60)
        lines.append(
            f"  {'Total':<30}: "
            f"  {self.total_nodes_removed} nodes removed, "
            f"{self.total_elapsed_ms:.1f} ms"
        )
        lines.append("=" * 60)
        return "\n".join(lines)
