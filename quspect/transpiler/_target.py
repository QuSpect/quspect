"""Compilation target — describes the capabilities of a target quantum device."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quspect.backends._base import CloudBackend


@dataclass
class CompilationTarget:
    """Describes the compilation-relevant capabilities of a target device.

    Attributes:
        num_qubits: Number of physical qubits on the device.
        coupling_graph: Undirected edges as a set of ``(q_i, q_j)`` tuples
            with ``q_i < q_j``.
        adjacency: Adjacency list — ``adjacency[i]`` is a list of qubit
            indices adjacent to qubit ``i``.
        native_gates: Set of gate names directly executable on the device.
    """

    num_qubits: int
    coupling_graph: set[tuple[int, int]] = field(default_factory=set)
    adjacency: list[list[int]] = field(default_factory=list)
    native_gates: set[str] = field(default_factory=set)

    @classmethod
    def from_backend(cls, backend: CloudBackend) -> CompilationTarget:
        """Build a target from a :class:`CloudBackend` by reading its config.

        Extracts ``coupling_map`` from ``backend.config["topology"]`` and
        ``native_gate_set`` from ``backend.native_gate_set``.

        Args:
            backend: A cloud backend instance (e.g. :class:`LogicalQubitBackend`).

        Returns:
            A populated :class:`CompilationTarget`.
        """
        coupling_graph: set[tuple[int, int]] = set()
        adjacency: list[list[int]] = []

        cfg: Any = backend.config if hasattr(backend, "config") else {}
        topo: Any = cfg.get("topology", {}) if isinstance(cfg, dict) else {}
        coupling_map = topo.get("coupling_map", []) if isinstance(topo, dict) else []

        num_qubits = backend.num_qubits

        adjacency = [[] for _ in range(num_qubits)]
        if isinstance(coupling_map, list):
            for pair in coupling_map:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    q0, q1 = int(pair[0]), int(pair[1])
                    if q0 != q1:
                        a, b = (q0, q1) if q0 < q1 else (q1, q0)
                        coupling_graph.add((a, b))  # noqa: ASYNC102
                        if b not in adjacency[a]:
                            adjacency[a].append(b)
                        if a not in adjacency[b]:
                            adjacency[b].append(a)

        native_gates: set[str] = (
            backend.native_gate_set if hasattr(backend, "native_gate_set") else set()
        )

        return cls(
            num_qubits=num_qubits,
            coupling_graph=coupling_graph,
            adjacency=adjacency,
            native_gates=native_gates,
        )

    @classmethod
    def from_coupling_map(
        cls,
        num_qubits: int,
        coupling_map: list[tuple[int, int]],
        *,
        native_gates: set[str] | None = None,
    ) -> CompilationTarget:
        """Build a target from explicit coupling map data.

        Useful for testing without a live backend.

        Args:
            num_qubits: Total number of physical qubits.
            coupling_map: List of ``(q_i, q_j)`` edges.
            native_gates: Optional set of native gate names.

        Returns:
            A configured :class:`CompilationTarget`.
        """
        coupling_graph: set[tuple[int, int]] = set()
        adjacency: list[list[int]] = [[] for _ in range(num_qubits)]

        for q0, q1 in coupling_map:
            if q0 == q1:
                continue
            a, b = (q0, q1) if q0 < q1 else (q1, q0)
            coupling_graph.add((a, b))  # noqa: ASYNC102
            if b not in adjacency[a]:
                adjacency[a].append(b)
            if a not in adjacency[b]:
                adjacency[b].append(a)

        return cls(
            num_qubits=num_qubits,
            coupling_graph=coupling_graph,
            adjacency=adjacency,
            native_gates=native_gates or set(),
        )

    def induced_subgraph(
        self,
        physical_qubits: Sequence[int],
    ) -> tuple[CompilationTarget, list[int]]:
        """Return the dense target induced by selected physical qubits.

        The returned target uses dense local indices ``0..n-1``.  The second
        return value maps each dense index back to its physical device index.

        Args:
            physical_qubits: Ordered, unique physical qubit indices.

        Returns:
            ``(dense_target, dense_to_physical)``.

        Raises:
            TypeError: If the value is not an integer sequence.
            ValueError: If it is empty, duplicated, or out of range.
        """
        if not isinstance(physical_qubits, (list, tuple)):
            raise TypeError("physical_qubits must be a list or tuple of integers.")
        selected = list(physical_qubits)
        if not selected:
            raise ValueError("physical_qubits must not be empty.")
        if not all(
            isinstance(qubit, int) and not isinstance(qubit, bool) for qubit in selected
        ):
            raise TypeError("physical_qubits must contain only integer qubits.")
        if len(set(selected)) != len(selected):
            raise ValueError("physical_qubits must not contain duplicates.")
        invalid = [qubit for qubit in selected if qubit < 0 or qubit >= self.num_qubits]
        if invalid:
            raise ValueError(
                "physical_qubits contains indices outside the backend range "
                f"[0, {self.num_qubits - 1}]: {invalid}."
            )

        physical_to_dense = {physical: dense for dense, physical in enumerate(selected)}
        dense_edges = [
            (physical_to_dense[left], physical_to_dense[right])
            for left, right in self.coupling_graph
            if left in physical_to_dense and right in physical_to_dense
        ]
        dense_target = self.from_coupling_map(
            len(selected),
            dense_edges,
            native_gates=set(self.native_gates),
        )
        return dense_target, selected
