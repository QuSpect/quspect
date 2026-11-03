"""Pauli-network synthesis of a rotation sequence on a coupling graph.

The native core (``cpp/src/synthesis/pauli_network.cpp``) keeps a Clifford
*frame* ``F`` such that the circuit emitted so far equals ``F`` times the
rotations implemented so far.  Each next rotation ``exp(-iθP)`` is reduced
in the frame, ``Q = F P F†``, to a single-qubit rotation by two-qubit
Cliffords on coupling edges (one CX each); the frame is not undone, so
consecutive rotations share their Clifford prefixes.  A reducing gate
removes one end of an edge inside the support of ``Q``; when the support
is not connected on the device, a pair of such gates moves one Pauli to a
neighbour closer to the rest (two CX instead of a three-CX SWAP, and the
move is absorbed by the frame instead of being undone).  Gates are chosen
by the estimated cost (a Steiner-tree estimate on the device) of ``Q`` and
of the next rotations; the next rotation is the cheapest one of the
*commuting front* (rotations commuting with every earlier pending one),
so the product is unchanged.  At the end the frame itself is synthesized
on the same graph, qubit by qubit from non-cut vertices, and the circuit
equals the rotation product exactly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from quspect.circuit.circuit import QuantumCircuit
from quspect.hamsim._formulas import Rotation

# NetGateKind of the native core
_H, _S, _SDG, _CX, _X, _Y, _Z, _ROT, _SWAP = range(9)


def _core() -> Any:
    from quspect._native_loader import load_core

    core = load_core()
    if core is None or not hasattr(core, "_pauli_network"):
        raise RuntimeError(
            "quspect.hamsim needs the QuSpect native core (build the C++ extension)"
        )
    return core


def _xz(
    rotations: Sequence[Rotation], layout: Sequence[int]
) -> tuple[list[list[int]], list[list[int]]]:
    xs: list[list[int]] = []
    zs: list[list[int]] = []
    for p, q, _ in rotations:
        x: list[int] = []
        z: list[int] = []
        for c, qi in zip(p, q, strict=True):
            phys = layout[qi]
            if c in "XY":
                x.append(phys)
            if c in "ZY":
                z.append(phys)
        xs.append(x)
        zs.append(z)
    return xs, zs


def merge_commuting(
    rotations: Sequence[Rotation], num_qubits: int, *, span: int = 256
) -> list[Rotation]:
    """Merge rotations about the same Pauli separated only by commuting ones.

    ``exp(-iaP) · C · exp(-ibP) = exp(-i(a+b)P) · C`` when every rotation in
    ``C`` commutes with ``P``, so this is exact.  Looks back at most
    ``span`` merged rotations.

    Args:
        rotations: Rotation sequence (time order).
        num_qubits: Number of qubits.
        span: Look-back limit.

    Returns:
        The merged sequence (time order of the first member of each group).
    """
    xs, zs = _xz(rotations, list(range(num_qubits)))
    slots = _core()._merge_commuting_rotations(num_qubits, xs, zs, span)
    angles: list[float] = []
    reps: list[int] = []
    for k, s in enumerate(slots):
        if s == len(angles):
            angles.append(0.0)
            reps.append(k)
        angles[s] += rotations[k][2]
    return [
        (rotations[r][0], rotations[r][1], a)
        for r, a in zip(reps, angles, strict=True)
        if a != 0.0
    ]


@dataclass
class NetworkCircuit:
    """Output of :func:`synthesize_network`.

    Attributes:
        circuit: Circuit on the physical qubits (index = physical qubit);
            every CX is on a coupling edge.
        num_cx: Number of CX gates.
        finalize_cx: CX gates spent on synthesizing the final frame.
        order: Implementation order (indices into the merged rotations).
        blocks: Two-qubit blocks emitted in place (counted in ``num_cx`` at
            their estimated synthesis cost, at most 3 CX each).
        swap_blocks: Blocks of those followed by a mirror SWAP.
        final_permutation: ``final_permutation[p]`` is the physical qubit
            that holds, at the end, the state that started on physical
            qubit ``p`` (identity unless ``finalize_permutation``).
        aborted: The synthesis stopped at ``cx_limit``; ``circuit`` is
            incomplete and must not be used.
    """

    circuit: QuantumCircuit
    num_cx: int
    finalize_cx: int
    order: list[int]
    blocks: int = 0
    swap_blocks: int = 0
    final_permutation: list[int] = field(default_factory=list)
    aborted: bool = False


def connected_active_set(
    num_qubits: int, edges: Sequence[tuple[int, int]], qubits: Sequence[int]
) -> list[int]:
    """``qubits`` plus connector qubits so that they induce a connected subgraph.

    Components are joined greedily along shortest paths (BFS, smallest index
    first), so the result is deterministic.
    """
    adj: list[list[int]] = [[] for _ in range(num_qubits)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    for row in adj:
        row.sort()
    if not qubits:
        return []
    active = {int(q) for q in qubits}

    def component(start: int) -> set[int]:
        seen = {start}
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v in active and v not in seen:
                    seen.add(v)
                    dq.append(v)
        return seen

    comp = component(min(active))
    while len(comp) < len(active):
        # BFS from the component to the nearest active qubit outside it
        prev = dict.fromkeys(sorted(comp), -1)
        dq = deque(sorted(comp))
        hit = -1
        while dq and hit < 0:
            u = dq.popleft()
            for v in adj[u]:
                if v in prev:
                    continue
                prev[v] = u
                if v in active:
                    hit = v
                    break
                dq.append(v)
        if hit < 0:
            raise ValueError("the qubits are not connected on the coupling graph")
        v = prev[hit]
        while v >= 0 and v not in comp:
            active.add(v)
            v = prev[v]
        comp = component(min(active))
    return sorted(active)


def long_path(
    num_qubits: int, edges: Sequence[tuple[int, int]], *, starts: int = 8
) -> list[int]:
    """A long simple path on a coupling graph (a Hamiltonian path when found).

    Greedy depth-first walks by Warnsdorff's rule (always step to the
    unvisited neighbour with the fewest unvisited neighbours, smallest
    index on ties) from the ``starts`` lowest-degree qubits; the longest
    walk is returned (ties: the first).  On a square grid this is the
    serpentine path; on heavy-hex lattices it follows the long rows.
    Deterministic.

    Args:
        num_qubits: Number of physical qubits.
        edges: Coupling edges (undirected).
        starts: Number of start qubits tried.

    Returns:
        The physical qubits along the path.
    """
    adj: list[list[int]] = [[] for _ in range(num_qubits)]
    for a, b in edges:
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    for row in adj:
        row[:] = sorted(set(row))
    nodes = [v for v in range(num_qubits) if adj[v]]
    if not nodes:
        return [0] if num_qubits else []
    nodes.sort(key=lambda v: (len(adj[v]), v))
    best: list[int] = []
    for s in nodes[: max(1, starts)]:
        path = [s]
        seen = [False] * num_qubits
        seen[s] = True
        free = [len(adj[v]) for v in range(num_qubits)]  # unvisited neighbours
        for w in adj[s]:
            free[w] -= 1
        while True:
            u = path[-1]
            nxt = -1
            for v in adj[u]:
                if not seen[v] and (nxt < 0 or free[v] < free[nxt]):
                    nxt = v
            if nxt < 0:
                break
            path.append(nxt)
            seen[nxt] = True
            for w in adj[nxt]:
                free[w] -= 1
        if len(path) > len(best):
            best = path
    return best


def synthesize_network(
    rotations: Sequence[Rotation],
    num_physical: int,
    edges: Sequence[tuple[int, int]],
    layout: Sequence[int],
    *,
    active: Sequence[int] | None = None,
    window: int = 12,
    front: int = 64,
    next_weight: int = 512,
    decay: int = 614,
    finalize: bool = True,
    block_mode: int = 0,
    block_bias: int = 1024,
    finalize_permutation: bool = False,
    cx_limit: int = 0,
) -> NetworkCircuit:
    """Synthesize ``Π exp(-iθ_k P_k)`` on a coupling graph.

    Args:
        rotations: Rotations in time order, on logical qubits.
        num_physical: Number of physical qubits.
        edges: Coupling edges (undirected).
        layout: ``layout[logical] = physical``.
        active: Physical qubits the synthesis may use (connected); default:
            the layout image plus connectors (:func:`connected_active_set`).
        window: Number of upcoming rotations in the gate score.
        front: Search range of the commuting front.
        next_weight: Weight of the first upcoming rotation (1/1024 units).
        decay: Weight decay per upcoming rotation (1/1024 units).
        finalize: Synthesize the final Clifford frame (``False`` leaves the
            circuit equal to ``F · U`` for the final frame ``F``).
        block_mode: ``1``: when the frame image of the next rotation is a
            two-qubit Pauli on a coupling edge, also consider emitting it,
            together with the front rotations whose images lie on the same
            edge, as one two-qubit block in place (frame unchanged; the
            backend synthesizes it with at most 3 CX), or the block followed
            by a SWAP that is absorbed into the frame (a generic SU(4) block
            times SWAP still needs 3 CX, so the move is free); the cheapest of
            frame reduction, block and block + SWAP by the look-ahead score
            is taken.  ``0``: frame reduction only.
        block_bias: Weight of the block cost in the comparison (1/1024
            units; above 1024 favours frame reduction).
        finalize_permutation: Synthesize the final frame only up to a qubit
            permutation (each pair of frame rows is reduced onto the cheapest
            free qubit instead of its own); the permutation is returned in
            ``final_permutation`` and is absorbed by the output layout.
        cx_limit: Stop early (``aborted``) once more than ``cx_limit`` CX
            have been emitted; ``0``: no limit.

    Returns:
        The circuit and gate counts.
    """
    used = sorted({layout[q] for _, qs, _ in rotations for q in qs})
    if active is None:
        active = connected_active_set(num_physical, edges, used or [layout[0]])
    xs, zs = _xz(rotations, layout)
    gates, order, num_cx, fin_cx, blocks, swap_blocks, perm, aborted = (
        _core()._pauli_network(
            num_physical,
            [(int(a), int(b)) for a, b in edges],
            [int(q) for q in active],
            xs,
            zs,
            window,
            front,
            next_weight,
            decay,
            finalize,
            block_mode,
            block_bias,
            finalize_permutation,
            cx_limit,
        )
    )
    if aborted:
        return NetworkCircuit(
            QuantumCircuit(),
            num_cx,
            fin_cx,
            list(order),
            blocks,
            swap_blocks,
            list(perm),
            True,
        )
    qc = QuantumCircuit()
    for kind, q0, q1, rot, axis, neg in gates:
        if kind == _CX:
            qc.cx(q0, q1)
        elif kind == _SWAP:
            qc.swap(q0, q1)
        elif kind == _ROT:
            theta = 2.0 * rotations[rot][2]
            if neg:
                theta = -theta
            if axis == "Z":
                qc.rz(theta, q0)
            elif axis == "X":
                qc.rx(theta, q0)
            else:
                qc.ry(theta, q0)
        elif kind == _H:
            qc.h(q0)
        elif kind == _S:
            qc.s(q0)
        elif kind == _SDG:
            qc.sdg(q0)
        elif kind == _X:
            qc.x(q0)
        elif kind == _Y:
            qc.y(q0)
        else:
            qc.z(q0)
    return NetworkCircuit(
        qc, num_cx, fin_cx, list(order), blocks, swap_blocks, list(perm)
    )


def ladder_circuit(rotations: Sequence[Rotation], num_qubits: int) -> QuantumCircuit:
    """Textbook circuit: basis change, CX ladder, ``rz``, inverse per rotation."""
    qc = QuantumCircuit()
    for p, q, th in rotations:
        for c, qi in zip(p, q, strict=True):
            if c == "X":
                qc.h(qi)
            elif c == "Y":
                qc.sdg(qi)
                qc.h(qi)
        for a, b in zip(q, q[1:], strict=False):
            qc.cx(a, b)
        qc.rz(2.0 * th, q[-1])
        for a, b in reversed(list(zip(q, q[1:], strict=False))):
            qc.cx(a, b)
        for c, qi in zip(p, q, strict=True):
            if c == "X":
                qc.h(qi)
            elif c == "Y":
                qc.h(qi)
                qc.s(qi)
    return qc
