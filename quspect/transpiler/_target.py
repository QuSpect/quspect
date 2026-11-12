"""Compilation target — describes the capabilities of a target quantum device."""

from __future__ import annotations

import functools
import operator
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quspect.backends._base import CloudBackend

#: Instruction names a device may list among its native operations although
#: they are not gates.
_NON_GATE_NAMES = frozenset({"measure", "barrier"})


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
        coupling_map: Iterable[Sequence[int]],
        *,
        native_gates: Collection[str] | None = None,
    ) -> CompilationTarget:
        """Build a target from explicit coupling map data.

        Useful for testing without a live backend.  Unlike
        :meth:`from_backend`, which takes a vendor's metadata as it is, this
        constructor checks what it is given.

        Args:
            num_qubits: Total number of physical qubits.
            coupling_map: ``(q_i, q_j)`` edges between qubits
                ``0 .. num_qubits - 1``.  Direction and duplicates do not
                matter; an edge from a qubit to itself is ignored.
            native_gates: Optional native gate names (lower-case, as in
                ``Gate.name``): built-in gates, ``"measure"``,
                ``"barrier"`` and entanglers registered with
                :func:`~quspect.synthesis.two_qubit.register_entangler`.
                Empty or ``None`` means no restriction.

        Returns:
            A configured :class:`CompilationTarget`.

        Raises:
            TypeError: If ``num_qubits`` is not an integer, an edge is not a
                pair of qubit indices, or ``native_gates`` is not a
                collection of names.
            ValueError: If ``num_qubits`` is negative, an edge names a qubit
                outside the device, or a native gate name is unknown.

        Example:
            >>> target = CompilationTarget.from_coupling_map(
            ...     3, [(0, 1), (1, 2)], native_gates={"cz", "rz", "sx", "x"}
            ... )
            >>> target.adjacency
            [[1], [0, 2], [1]]
        """
        num_qubits = _checked_num_qubits(num_qubits)
        native = _checked_native_gates(native_gates)
        edges = [_checked_edge(edge, num_qubits) for edge in coupling_map]
        return cls._from_edges(num_qubits, edges, native)

    @classmethod
    def _from_edges(
        cls,
        num_qubits: int,
        edges: Iterable[tuple[int, int]],
        native_gates: set[str],
    ) -> CompilationTarget:
        """Build a target from edges and gate names that are already checked.

        Also used for targets derived from a device target, whose native
        gate names come from vendor metadata and are kept as they are.
        """
        coupling_graph: set[tuple[int, int]] = set()
        adjacency: list[list[int]] = [[] for _ in range(num_qubits)]

        for q0, q1 in edges:
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
            native_gates=native_gates,
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
        dense_target = self._from_edges(
            len(selected), dense_edges, set(self.native_gates)
        )
        return dense_target, selected


def _checked_num_qubits(num_qubits: object) -> int:
    """Return ``num_qubits`` as a plain non-negative ``int``."""
    if isinstance(num_qubits, (bool, np.bool_)) or not hasattr(num_qubits, "__index__"):
        raise TypeError(
            f"num_qubits must be a non-negative integer, got {num_qubits!r}."
        )
    value = operator.index(num_qubits)
    if value < 0:
        raise ValueError(f"num_qubits must be a non-negative integer, got {value}.")
    return value


def _checked_edge(edge: object, num_qubits: int) -> tuple[int, int]:
    """Return a coupling-map entry as two qubit indices of the device."""
    malformed = TypeError(
        f"Every coupling_map entry must be a pair of qubit indices, got {edge!r}."
    )
    try:
        pair: tuple[Any, ...] = tuple(edge)  # type: ignore[arg-type]
    except TypeError:
        raise malformed from None
    if len(pair) != 2:
        raise malformed
    qubits: list[int] = []
    for qubit in pair:
        if isinstance(qubit, (bool, np.bool_)) or not hasattr(qubit, "__index__"):
            raise malformed
        qubits.append(operator.index(qubit))
    q0, q1 = qubits
    if not (0 <= q0 < num_qubits and 0 <= q1 < num_qubits):
        raise ValueError(
            f"coupling_map edge ({q0}, {q1}) is outside the {num_qubits} qubits "
            f"of the target (valid indices are 0 to {num_qubits - 1})."
        )
    return q0, q1


@functools.cache
def _builtin_gate_names() -> frozenset[str]:
    from quspect.gates.parameterized import ALL_PARAMETERIZED
    from quspect.gates.standard import ALL_STANDARD

    return frozenset(gate.name for gate in (*ALL_STANDARD, *ALL_PARAMETERIZED))


def _checked_native_gates(native_gates: object) -> set[str]:
    """Return the native gate names as a new set, refusing unknown names.

    A name the compiler does not know would silently drop out of the basis
    (``{"RZ", "SX"}`` used to end in "native single-qubit gates []").
    """
    if native_gates is None:
        return set()
    if isinstance(native_gates, (str, bytes)) or not isinstance(
        native_gates, Collection
    ):
        raise TypeError(
            "native_gates must be a collection of gate names such as "
            f"{{'cz', 'rz', 'sx', 'x'}}, got {native_gates!r}."
        )
    names: set[str] = set()
    for name in native_gates:
        if not isinstance(name, str):
            raise TypeError(
                f"native_gates must contain gate names (str), got {name!r}."
            )
        names.add(name)

    from quspect.synthesis.two_qubit import _ENTANGLERS

    known = _builtin_gate_names() | _NON_GATE_NAMES | frozenset(_ENTANGLERS)
    unknown = sorted(names - known)
    if unknown:
        described = ", ".join(
            f"{name!r} (did you mean {name.lower()!r}?)"
            if name.lower() in known
            else repr(name)
            for name in unknown
        )
        raise ValueError(
            f"Unknown native gate name(s): {described}. Gate names are "
            f"lower-case; known names are {sorted(known)}. A custom two-qubit "
            "gate must be registered with "
            "quspect.synthesis.two_qubit.register_entangler() first."
        )
    return names
