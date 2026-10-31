"""DAG representation of quantum circuits.

The :class:`DAGCircuit` is the core intermediate representation (IR)
for quantum circuit compilation. It makes qubit dependencies explicit
through a directed acyclic graph structure.

Internally every node keeps its incoming and outgoing edges in small
lists ordered by an edge *sequence number* (the global insertion order),
so that local queries (wire neighbours, adjacency tests, node removal)
cost O(degree) instead of O(edges). Wire traversal walks the per-wire
chain from the wire's INPUT node, and the topological order is cached
until the next structural modification.
"""

from __future__ import annotations

from collections import deque

from quspect.dag.dag_node import DAGEdge, DAGNode, DAGNodeType

#: An adjacency entry: ``(edge sequence number, neighbour node id, wire)``.
_AdjEntry = tuple[int, int, str]

#: A journal entry recorded while a transaction is open (see
#: :meth:`DAGCircuit.begin_transaction`).
_JournalEntry = tuple[str, int, object]


def _wire_index(wire: str) -> tuple[str, int] | None:
    """Parse a wire label such as ``"q3"`` into ``("q", 3)``.

    Args:
        wire: Wire identifier.

    Returns:
        The ``(kind, index)`` pair, or ``None`` if the label is malformed.
    """
    if len(wire) < 2 or wire[0] not in "qc" or not wire[1:].isdigit():
        return None
    return wire[0], int(wire[1:])


def _node_wires(node: DAGNode) -> list[str]:
    """Return the wire labels a node lives on (qubits first, then clbits)."""
    wires = [f"q{q}" for q in node.qubits]
    wires.extend(f"c{c}" for c in node.clbits)
    return wires


class DAGCircuit:
    """DAG representation of a quantum circuit.

    A DAGCircuit is a directed acyclic graph that makes qubit dependencies
    explicit. It is the primary intermediate representation for circuit
    compilation, optimization, and analysis.

    The graph has explicit INPUT and OUTPUT nodes marking the boundaries
    of each qubit and classical bit line. Between them, OPERATION nodes
    are connected by edges that follow the qubit/bit flow.

    .. note::

        Adjacency is stored per node, ordered by edge insertion, so that
        local updates are O(degree). The topological order is cached and
        recomputed lazily after a structural change. Node IDs are
        assigned sequentially starting from 0 for INPUT nodes, N for
        OUTPUT nodes, and continuing for OPERATION nodes.

    Example:
        >>> from quspect import QuantumCircuit
        >>> circ = QuantumCircuit()
        >>> circ.h(0).cx(0, 1)
        >>> dag = circ.to_dag()
        >>> dag.num_nodes()
        6  # 2 inputs (q0,q1) + 2 gates + 2 outputs (q0,q1)
    """

    def __init__(self) -> None:
        self._nodes: dict[int, DAGNode] = {}
        # Edge sequence number -> edge.  Sequence numbers define the
        # global edge order (``edges`` / adjacency order).
        self._edges: dict[int, DAGEdge] = {}
        self._edges_in_order: bool = True
        self._next_seq: int = 0
        # Per-node adjacency, each list sorted by edge sequence number.
        self._out: dict[int, list[_AdjEntry]] = {}
        self._in: dict[int, list[_AdjEntry]] = {}
        # Wire bookkeeping used by the fast ``nodes_on_wire`` walk.
        self._wire_input: dict[str, int] = {}
        self._ambiguous_wires: set[str] = set()
        self._wire_count: dict[str, int] = {}
        self._num_qubits: int = 0
        self._num_clbits: int = 0
        # Caches.
        self._topo_cache: list[int] | None = None
        self._version: int = 0
        # nodes_on_wire results, valid while _version == _wire_cache_version.
        self._wire_cache: dict[str, list[int]] = {}
        self._wire_cache_version: int = -1
        # Topological numbering maintained for incremental cycle checks.
        self._pos: dict[int, int] | None = None
        self._next_pos: int = 0
        # Undo journal (None when no transaction is open).
        self._journal: list[_JournalEntry] | None = None
        self._txn_depth: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Return the number of qubit lines in this DAG."""
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        """Return the number of classical bit lines in this DAG."""
        return self._num_clbits

    @property
    def nodes(self) -> list[DAGNode]:
        """Return all nodes, sorted by node_id."""
        return [self._nodes[i] for i in sorted(self._nodes)]

    @property
    def edges(self) -> list[DAGEdge]:
        """Return all edges in insertion order."""
        if not self._edges_in_order:
            self._edges = dict(sorted(self._edges.items()))
            self._edges_in_order = True
        return list(self._edges.values())

    @property
    def modification_count(self) -> int:
        """Return a counter that increases on every structural modification.

        Adding or removing nodes or edges, and replacing a node's
        instruction through :meth:`substitute_node_instruction`, increase
        the counter. Comparing two readings is a cheap way to detect
        whether a pass touched the DAG.

        Returns:
            A monotonically increasing integer.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).h(0).to_dag()
            >>> before = dag.modification_count
            >>> dag.remove_node(dag.operation_nodes()[0].node_id)
            >>> dag.modification_count > before
            True
        """
        return self._version

    def num_nodes(self) -> int:
        """Return the total number of nodes."""
        return len(self._nodes)

    def num_edges(self) -> int:
        """Return the total number of edges."""
        return len(self._edges)

    def get_node(self, node_id: int) -> DAGNode:
        """Get a node by its ID.

        Args:
            node_id: The node identifier.

        Returns:
            The DAGNode with the given ID.

        Raises:
            KeyError: If the node_id is not found.
        """
        return self._nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        """Return whether a node with the given ID exists.

        Args:
            node_id: The node identifier.

        Returns:
            True if the node is part of the DAG.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> dag.has_node(0), dag.has_node(99)
            (True, False)
        """
        return node_id in self._nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        """Record a structural modification (invalidates caches)."""
        self._version += 1
        self._topo_cache = None

    def _link(self, seq: int, edge: DAGEdge) -> None:
        """Insert an edge with a given sequence number into all indexes."""
        self._edges[seq] = edge
        out = self._out.setdefault(edge.source_id, [])
        inc = self._in.setdefault(edge.target_id, [])
        if seq >= self._next_seq:
            self._next_seq = seq + 1
            out.append((seq, edge.target_id, edge.wire))
            inc.append((seq, edge.source_id, edge.wire))
        else:
            # Restoring an older edge: keep lists sorted by sequence.
            self._edges_in_order = False
            out.append((seq, edge.target_id, edge.wire))
            out.sort()
            inc.append((seq, edge.source_id, edge.wire))
            inc.sort()
        self._touch()

    def _unlink(self, seq: int) -> DAGEdge:
        """Remove the edge with the given sequence number from all indexes."""
        edge = self._edges.pop(seq)
        out = self._out[edge.source_id]
        for k, entry in enumerate(out):
            if entry[0] == seq:
                del out[k]
                break
        inc = self._in[edge.target_id]
        for k, entry in enumerate(inc):
            if entry[0] == seq:
                del inc[k]
                break
        self._touch()
        return edge

    def _find_edge_seq(self, source_id: int, target_id: int, wire: str) -> int | None:
        """Return the sequence number of the first matching edge, if any."""
        for seq, tgt, w in self._out.get(source_id, ()):
            if tgt == target_id and w == wire:
                return seq
        return None

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG.

        Args:
            node: The DAGNode to add.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node with id={node.node_id} already exists in DAG.")
        self._nodes[node.node_id] = node
        for wire in _node_wires(node):
            self._wire_count[wire] = self._wire_count.get(wire, 0) + 1
            if node.node_type == DAGNodeType.INPUT:
                if wire in self._wire_input:
                    self._ambiguous_wires.add(wire)
                self._wire_input[wire] = node.node_id
        if self._pos is not None:
            self._pos[node.node_id] = self._next_pos
            self._next_pos += 1
        self._touch()

    def add_edge(self, edge: DAGEdge, *, defer_order_check: bool = False) -> None:
        """Add a directed edge to the DAG.

        Args:
            edge: The DAGEdge to add.
            defer_order_check: Internal flag for reordering passes. When
                True the incremental topological numbering is not checked;
                the caller must call :meth:`reverse_adjacent_edge_order`
                afterwards.
        """
        seq = self._next_seq
        self._link(seq, edge)
        if self._journal is not None:
            self._journal.append(("add", seq, None))
        if defer_order_check:
            return
        pos = self._pos
        if pos is not None:
            ps = pos.get(edge.source_id)
            pt = pos.get(edge.target_id)
            if ps is None or pt is None or ps >= pt:
                self._invalidate_positions()

    def remove_edge(self, source_id: int, target_id: int, wire: str) -> None:
        """Remove one directed edge from the DAG.

        Unlike :meth:`remove_node`, the endpoints are kept.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            wire: Wire label of the edge (e.g. ``"q0"``).

        Raises:
            KeyError: If no such edge exists.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> h = dag.operation_nodes()[0].node_id
            >>> dag.remove_edge(h, dag.wire_successor(h, "q0"), "q0")
            >>> dag.wire_successor(h, "q0") is None
            True
        """
        seq = self._find_edge_seq(source_id, target_id, wire)
        if seq is None:
            raise KeyError(f"Edge {source_id}->{target_id} on {wire!r} not found.")
        edge = self._unlink(seq)
        if self._journal is not None:
            self._journal.append(("remove", seq, edge))

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its incident edges from the DAG.

        The neighbours are **not** reconnected; callers that want to keep
        the wires connected should use :meth:`remove_node_and_reconnect`.
        Runs in O(degree) time.

        Args:
            node_id: The ID of the node to remove.

        Raises:
            KeyError: If the node_id is not found.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node id={node_id} not found in DAG.")

        node = self._nodes.pop(node_id)
        for wire in _node_wires(node):
            self._wire_count[wire] = self._wire_count.get(wire, 0) - 1
            if node.node_type == DAGNodeType.INPUT:
                self._ambiguous_wires.add(wire)

        incident = {seq for seq, _, _ in self._out.get(node_id, ())}
        incident.update(seq for seq, _, _ in self._in.get(node_id, ()))
        for seq in sorted(incident):
            edge = self._unlink(seq)
            if self._journal is not None:
                self._journal.append(("remove", seq, edge))
        self._out.pop(node_id, None)
        self._in.pop(node_id, None)
        if self._pos is not None:
            self._pos.pop(node_id, None)
        self._touch()

    def remove_node_and_reconnect(self, node_id: int) -> None:
        """Remove an operation node and splice each of its wires back together.

        For every wire of the node (qubits in instruction order, then
        clbits) the wire predecessor is connected to the wire successor,
        so the DAG stays a valid circuit DAG. Runs in O(degree) time.

        Args:
            node_id: The ID of the node to remove.

        Raises:
            KeyError: If the node_id is not found.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).x(0).to_dag()
            >>> h = dag.operation_nodes()[0].node_id
            >>> dag.remove_node_and_reconnect(h)
            >>> [n.instruction.gate.name for n in dag.operation_nodes()]
            ['x']
        """
        node = self.get_node(node_id)
        if node.instruction is not None:
            wires = [f"q{q}" for q in node.instruction.qubits]
            wires.extend(f"c{c}" for c in node.instruction.clbits)
        else:
            wires = _node_wires(node)
        reconnect: list[tuple[int, int, str]] = []
        for wire in dict.fromkeys(wires):
            pred = self.wire_predecessor(node_id, wire)
            succ = self.wire_successor(node_id, wire)
            if pred is not None and succ is not None:
                reconnect.append((pred, succ, wire))
        self.remove_node(node_id)
        for pred, succ, wire in reconnect:
            self.add_edge(DAGEdge(source_id=pred, target_id=succ, wire=wire))

    def substitute_node_instruction(self, node_id: int, instruction: object) -> None:
        """Replace the instruction of an operation node in place.

        The node keeps its ID, wires and edges; only its payload changes.
        Using this method (rather than assigning ``node.instruction``)
        also records the change in :attr:`modification_count`.

        Args:
            node_id: The node whose instruction is replaced.
            instruction: The new
                :class:`~quspect.circuit.instruction.CircuitInstruction`.
                It must act on the same qubits/clbits as the old one.

        Raises:
            KeyError: If the node_id is not found.

        Example:
            >>> from quspect import QuantumCircuit
            >>> from quspect.circuit.instruction import (
            ...     CircuitInstruction, InstructionType)
            >>> from quspect.gates.standard import X
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> h = dag.operation_nodes()[0].node_id
            >>> dag.substitute_node_instruction(
            ...     h, CircuitInstruction(InstructionType.GATE, gate=X, qubits=(0,)))
            >>> dag.get_node(h).instruction.gate.name
            'x'
        """
        from quspect.circuit.instruction import CircuitInstruction

        node = self._nodes[node_id]
        if not isinstance(instruction, CircuitInstruction):
            raise TypeError(
                "instruction must be a CircuitInstruction, got "
                f"{type(instruction).__name__}."
            )
        node.instruction = instruction
        self._version += 1

    def set_qubit_count(self, n: int) -> None:
        """Set the number of qubit lines.

        Args:
            n: Number of qubits.
        """
        self._num_qubits = n

    def set_clbit_count(self, n: int) -> None:
        """Set the number of classical bit lines.

        Args:
            n: Number of classical bits.
        """
        self._num_clbits = n

    # ------------------------------------------------------------------
    # Transactions (used by reordering passes)
    # ------------------------------------------------------------------

    def begin_transaction(self) -> int:
        """Start (or nest into) an undo journal for edge modifications.

        While a transaction is open, edge insertions and removals (also
        those performed by :meth:`remove_node`, whose node removal itself
        is *not* undone) are journaled so that :meth:`rollback` can
        restore the exact previous edge order.

        Returns:
            A savepoint to pass to :meth:`rollback`.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> sp = dag.begin_transaction()
            >>> h = dag.operation_nodes()[0].node_id
            >>> dag.remove_edge(h, dag.wire_successor(h, "q0"), "q0")
            >>> dag.rollback(sp)
            >>> dag.end_transaction()
            >>> dag.wire_successor(h, "q0") is not None
            True
        """
        if self._journal is None:
            self._journal = []
        self._txn_depth += 1
        return len(self._journal)

    def rollback(self, savepoint: int) -> None:
        """Undo all journaled edge modifications made after ``savepoint``.

        Args:
            savepoint: A value returned by :meth:`begin_transaction`.

        Raises:
            RuntimeError: If no transaction is open.
        """
        journal = self._journal
        if journal is None:
            raise RuntimeError("rollback() called without an open transaction.")
        while len(journal) > savepoint:
            kind, key, payload = journal.pop()
            if kind == "add":
                if key in self._edges:
                    self._unlink(key)
            elif kind == "remove":
                assert isinstance(payload, DAGEdge)
                self._link(key, payload)
            elif kind == "pos":
                if self._pos is not None:
                    assert isinstance(payload, int)
                    self._pos[key] = payload
            elif kind == "posinv":
                self._pos = payload if isinstance(payload, dict) else None

    def end_transaction(self) -> None:
        """Close the innermost transaction, keeping all modifications.

        The journal is discarded when the outermost transaction ends.
        """
        if self._txn_depth > 0:
            self._txn_depth -= 1
        if self._txn_depth == 0:
            self._journal = None

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    def predecessors(self, node_id: int) -> list[int]:
        """Return node_ids of all direct predecessors.

        Args:
            node_id: The target node ID.

        Returns:
            List of predecessor node IDs (one entry per edge).
        """
        return [src for _, src, _ in self._in.get(node_id, ())]

    def successors(self, node_id: int) -> list[int]:
        """Return node_ids of all direct successors.

        Args:
            node_id: The source node ID.

        Returns:
            List of successor node IDs (one entry per edge).
        """
        return [tgt for _, tgt, _ in self._out.get(node_id, ())]

    def out_edges_with_wires(self, node_id: int) -> list[tuple[int, str]]:
        """Return ``(target_id, wire)`` for every outgoing edge, in edge order.

        Args:
            node_id: The source node ID.

        Returns:
            List of ``(target node ID, wire label)`` pairs.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> dag.out_edges_with_wires(0)  # INPUT q0 -> h
            [(1, 'q0')]
        """
        return [(tgt, w) for _, tgt, w in self._out.get(node_id, ())]

    def in_edges_with_wires(self, node_id: int) -> list[tuple[int, str]]:
        """Return ``(source_id, wire)`` for every incoming edge, in edge order.

        Args:
            node_id: The target node ID.

        Returns:
            List of ``(source node ID, wire label)`` pairs.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> dag.in_edges_with_wires(1)  # h <- INPUT q0
            [(0, 'q0')]
        """
        return [(src, w) for _, src, w in self._in.get(node_id, ())]

    def has_edge(self, source_id: int, target_id: int, wire: str | None = None) -> bool:
        """Return whether a direct edge ``source -> target`` exists.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            wire: If given, the edge must carry this wire label.

        Returns:
            True if a matching edge exists. Runs in O(out-degree).

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).x(0).to_dag()
            >>> h, x = (n.node_id for n in dag.operation_nodes())
            >>> dag.has_edge(h, x, "q0"), dag.has_edge(x, h)
            (True, False)
        """
        for _, tgt, w in self._out.get(source_id, ()):
            if tgt == target_id and (wire is None or w == wire):
                return True
        return False

    def wire_predecessor(self, node_id: int, wire: str) -> int | None:
        """Return the node feeding ``node_id`` on ``wire``.

        Args:
            node_id: The node ID.
            wire: Wire label such as ``"q0"``.

        Returns:
            The predecessor node ID on that wire, or ``None`` if there is
            none. If several edges match, the most recently added wins.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).x(0).to_dag()
            >>> h, x = (n.node_id for n in dag.operation_nodes())
            >>> dag.wire_predecessor(x, "q0") == h
            True
        """
        result: int | None = None
        for _, src, w in self._in.get(node_id, ()):
            if w == wire:
                result = src
        return result

    def wire_successor(self, node_id: int, wire: str) -> int | None:
        """Return the node following ``node_id`` on ``wire``.

        Args:
            node_id: The node ID.
            wire: Wire label such as ``"q0"``.

        Returns:
            The successor node ID on that wire, or ``None`` if there is
            none. If several edges match, the most recently added wins.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).x(0).to_dag()
            >>> h, x = (n.node_id for n in dag.operation_nodes())
            >>> dag.wire_successor(h, "q0") == x
            True
        """
        result: int | None = None
        for _, tgt, w in self._out.get(node_id, ()):
            if w == wire:
                result = tgt
        return result

    def input_nodes(self) -> list[DAGNode]:
        """Return all INPUT boundary nodes."""
        return [n for n in self._nodes.values() if n.node_type == DAGNodeType.INPUT]

    def output_nodes(self) -> list[DAGNode]:
        """Return all OUTPUT boundary nodes."""
        return [n for n in self._nodes.values() if n.node_type == DAGNodeType.OUTPUT]

    def operation_nodes(self) -> list[DAGNode]:
        """Return all OPERATION nodes in topological order."""
        nodes = self._nodes
        return [
            nodes[nid]
            for nid in self._topological_order_cached()
            if nodes[nid].node_type == DAGNodeType.OPERATION
        ]

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_order_cached(self) -> list[int]:
        """Return the cached topological order (do not mutate the result)."""
        if self._topo_cache is not None:
            return self._topo_cache

        # Compute in-degree for each node
        in_degree: dict[int, int] = {
            nid: len(self._in.get(nid, ())) for nid in self._nodes
        }

        # Start with nodes that have no predecessors (INPUT nodes)
        queue: deque[int] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        result: list[int] = []
        out = self._out

        while queue:
            nid = queue.popleft()
            result.append(nid)
            for _, succ, _ in out.get(nid, ()):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(result) != len(self._nodes):
            raise RuntimeError(
                "Cycle detected in DAG — this should not happen for "
                "circuits built from sequential instructions."
            )

        self._topo_cache = result
        return result

    def topological_order(self) -> list[int]:
        """Return node_ids in topological order.

        Uses Kahn's algorithm. The DAG must be acyclic (guaranteed by
        construction from a sequential circuit). The result is cached
        until the next structural modification.

        Returns:
            List of node IDs in topological order (INPUTs first,
            OUTPUTs last).

        Raises:
            RuntimeError: If a cycle is detected (should not happen
                for DAGs constructed from valid circuits).
        """
        return list(self._topological_order_cached())

    # ------------------------------------------------------------------
    # Incremental topological numbering (cycle checks for reordering)
    # ------------------------------------------------------------------

    def _invalidate_positions(self) -> None:
        """Drop the topological numbering (journaled when in a transaction)."""
        if self._pos is None:
            return
        if self._journal is not None:
            self._journal.append(("posinv", 0, self._pos))
        self._pos = None

    def ensure_topological_positions(self) -> None:
        """Compute the incremental topological numbering if it is missing.

        Reordering passes call this before rewiring the DAG so that the
        subsequent :meth:`reverse_adjacent_edge_order` checks are local.

        Raises:
            RuntimeError: If the DAG contains a cycle.

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).to_dag()
            >>> dag.ensure_topological_positions()
        """
        self._ensure_positions()

    def _ensure_positions(self) -> dict[int, int]:
        """Return a valid topological numbering, computing it if needed."""
        if self._pos is None:
            order = self._topological_order_cached()
            self._pos = {nid: k for k, nid in enumerate(order)}
            self._next_pos = len(order)
        return self._pos

    def reverse_adjacent_edge_order(self, first_id: int, second_id: int) -> bool:
        """Repair the topological numbering after ``second`` was moved before ``first``.

        This is the incremental (Pearce-Kelly) step used by commutation
        reordering: the caller has rewired the DAG so that ``second_id``
        now precedes ``first_id`` (an edge ``second -> first`` exists and
        the old ``first -> second`` edges are gone), where previously
        ``first_id`` preceded ``second_id``. The method checks in time
        proportional to the affected region whether the rewiring created
        a cycle and, if not, updates the internal numbering.

        Args:
            first_id: The node that used to come first.
            second_id: The node that now comes first.

        Returns:
            True if the DAG is still acyclic, False if a cycle was
            created (the numbering is left unchanged in that case).

        Example:
            >>> from quspect import QuantumCircuit
            >>> dag = QuantumCircuit().h(0).x(0).to_dag()
            >>> h, x = (n.node_id for n in dag.operation_nodes())
            >>> sp = dag.begin_transaction()
            >>> pred, succ = dag.wire_predecessor(h, "q0"), dag.wire_successor(x, "q0")
            >>> for s, t in ((pred, h), (h, x), (x, succ)):
            ...     dag.remove_edge(s, t, "q0")
            >>> for s, t in ((pred, x), (x, h), (h, succ)):
            ...     dag.add_edge(DAGEdge(s, t, "q0"), defer_order_check=True)
            >>> dag.reverse_adjacent_edge_order(h, x)
            True
            >>> dag.end_transaction()
        """
        if self._pos is None or first_id not in self._pos or second_id not in self._pos:
            # No usable numbering: fall back to a full (cycle-checking) sort
            # of the rewired graph and adopt it as the new numbering.
            try:
                order = self._topological_order_cached()
            except RuntimeError:
                return False
            if self._journal is not None:
                self._journal.append(("posinv", 0, self._pos))
            self._pos = {nid: k for k, nid in enumerate(order)}
            self._next_pos = len(order)
            return True
        pos = self._pos
        lb = pos[first_id]
        ub = pos[second_id]
        if lb > ub:
            # Already consistent with the new order.
            return True

        out = self._out
        inc = self._in
        # Forward search from ``first`` bounded by ``second``'s position.
        forward: list[int] = [first_id]
        seen_f = {first_id}
        stack = [first_id]
        while stack:
            nid = stack.pop()
            for _, succ, _ in out.get(nid, ()):
                if succ == second_id:
                    return False
                if succ not in seen_f:
                    p = pos.get(succ)
                    if p is not None and p < ub:
                        seen_f.add(succ)
                        forward.append(succ)
                        stack.append(succ)
        # Backward search from ``second`` bounded by ``first``'s position.
        backward: list[int] = [second_id]
        seen_b = {second_id}
        stack = [second_id]
        while stack:
            nid = stack.pop()
            for _, pred, _ in inc.get(nid, ()):
                if pred not in seen_b:
                    p = pos.get(pred)
                    if p is not None and p > lb:
                        seen_b.add(pred)
                        backward.append(pred)
                        stack.append(pred)

        backward.sort(key=pos.__getitem__)
        forward.sort(key=pos.__getitem__)
        slots = sorted(pos[n] for n in backward + forward)
        journal = self._journal
        for nid, slot in zip(backward + forward, slots, strict=True):
            if journal is not None:
                journal.append(("pos", nid, pos[nid]))
            pos[nid] = slot
        return True

    # ------------------------------------------------------------------
    # Layering (depth-based partition)
    # ------------------------------------------------------------------

    def layers(self) -> list[list[int]]:
        """Partition nodes into layers (depth levels) for scheduling.

        Each layer contains nodes whose longest path from any INPUT
        node is the same length. Within a layer, nodes have no
        dependencies on each other and can be executed in parallel.

        Returns:
            A list of layers, each layer is a list of node_ids.

        Example:
            >>> dag = circuit.to_dag()
            >>> for i, layer in enumerate(dag.layers()):
            ...     print(f"Depth {i}: {layer}")
        """
        topo = self._topological_order_cached()

        # Compute longest path from INPUT to each node
        depth: dict[int, int] = {}

        for nid in topo:
            preds = self._in.get(nid, ())
            if not preds:
                depth[nid] = 0
            else:
                depth[nid] = 1 + max(depth[p] for _, p, _ in preds)

        # Group nodes by depth
        max_depth = max(depth.values()) if depth else 0
        layers_result: list[list[int]] = [[] for _ in range(max_depth + 1)]
        for nid in topo:
            layers_result[depth[nid]].append(nid)

        return layers_result

    # ------------------------------------------------------------------
    # Critical path
    # ------------------------------------------------------------------

    def critical_path_length(self) -> int:
        """Return the length of the critical path through the DAG.

        The critical path is the longest chain of dependent operations,
        which determines the minimum execution time. INPUT and OUTPUT
        nodes are excluded from the count.

        Returns:
            Number of operation nodes on the longest dependency chain.
        """
        layers = self.layers()
        if len(layers) <= 2:
            # Only INPUT and OUTPUT layers, no operations
            return 0
        # Exclude first layer (INPUTs) and last layer (OUTPUTs)
        op_layers = layers[1:-1]
        return len(op_layers)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _walk_wire(self, wire: str) -> list[int] | None:
        """Walk a wire from its INPUT node along wire-labelled edges.

        Returns:
            The node IDs on the wire, or ``None`` when the wire structure
            is not a simple chain (the caller then falls back to a
            topological scan).
        """
        if wire in self._ambiguous_wires:
            return None
        start = self._wire_input.get(wire)
        if start is None or start not in self._nodes:
            return None
        parsed = _wire_index(wire)
        if parsed is None:
            return None
        kind, idx = parsed
        expected = self._wire_count.get(wire, 0)
        out = self._out
        nodes = self._nodes
        result = [start]
        nid = start
        while True:
            nxt: int | None = None
            for _, tgt, w in out.get(nid, ()):
                if w == wire:
                    if nxt is not None:
                        return None
                    nxt = tgt
            if nxt is None:
                break
            node = nodes.get(nxt)
            if node is None:
                return None
            if idx not in (node.qubits if kind == "q" else node.clbits):
                return None
            result.append(nxt)
            if len(result) > expected:
                return None
            nid = nxt
        if len(result) != expected:
            return None
        return result

    def nodes_on_wire(self, wire: str) -> list[int]:
        """Return node_ids on a given qubit/bit wire in topological order.

        For a well-formed circuit DAG this walks the wire from its INPUT
        node in O(length of the wire); otherwise it falls back to scanning
        the (cached) topological order.

        Args:
            wire: Wire identifier, e.g. ``"q0"`` or ``"c1"``.

        Returns:
            List of node IDs that involve the given wire.
        """
        if self._wire_cache_version != self._version:
            self._wire_cache.clear()
            self._wire_cache_version = self._version
        cached = self._wire_cache.get(wire)
        if cached is None:
            cached = self._wire_cache[wire] = self._nodes_on_wire(wire)
        return list(cached)

    def _nodes_on_wire(self, wire: str) -> list[int]:
        """Uncached :meth:`nodes_on_wire`."""
        walked = self._walk_wire(wire)
        if walked is not None:
            return walked
        topo = self._topological_order_cached()
        result = []
        for nid in topo:
            node = self._nodes[nid]
            # Check if wire is in the qubits or clbits of this node
            if wire.startswith("q"):
                q_idx = int(wire[1:])
                if q_idx in node.qubits:
                    result.append(nid)
            elif wire.startswith("c"):
                c_idx = int(wire[1:])
                if c_idx in node.clbits:
                    result.append(nid)
        return result

    def qubit_wires(self) -> list[int]:
        """Return the list of qubit indices present in this DAG."""
        return list(range(self._num_qubits))

    def clbit_wires(self) -> list[int]:
        """Return the list of classical bit indices present in this DAG."""
        return list(range(self._num_clbits))

    # ------------------------------------------------------------------
    # Statistics / Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the DAG.

        Returns:
            A multi-line string with DAG statistics.
        """
        lines = [
            f"DAGCircuit: {self.num_nodes()} nodes, {self.num_edges()} edges",
            f"  Qubits: {self._num_qubits}",
            f"  Clbits: {self._num_clbits}",
            f"  INPUT nodes: {len(self.input_nodes())}",
            f"  OUTPUT nodes: {len(self.output_nodes())}",
            f"  OPERATION nodes: {len(self.operation_nodes())}",
            f"  Critical path length: {self.critical_path_length()}",
            f"  Depth (layers): {len(self.layers())}",
        ]
        return "\n".join(lines)
