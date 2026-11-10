"""Shared device compilation for cloud QPU backends.

Cloud backends that let QuSpect compile for the device (rather than leaving
qubit mapping to the provider) all need the same pipeline: lower the circuit
to the device basis, optimize, then route it on the device coupling graph.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._target import CompilationTarget

# SABRE models pairwise coupling edges, so higher-arity gates must be lowered
# before topology routing even when provider metadata claims they are native
# or omits a native gate set altogether.  Keep this list deliberately narrow:
# a provider translator can still accept CCX when transpilation is disabled.
PAIRWISE_ROUTING_DECOMPOSITION_GATES = frozenset({"ccx"})

# Native two-qubit gates a two-qubit instruction costs on a CZ/CX chip, for
# :func:`estimated_log_fidelity`: a SWAP is three, a controlled rotation two.
_TWO_QUBIT_GATE_COST = {"cx": 1, "cz": 1, "swap": 3}
_OTHER_TWO_QUBIT_GATE_COST = 2

# Estimates closer than this (in log-fidelity) count as equal.
_ESTIMATE_TOLERANCE = 1e-9

_K = TypeVar("_K", bound=Hashable)
_T = TypeVar("_T")


def decompose_for_pairwise_routing(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower gates that cannot be represented by pairwise topology routing.

    This is intentionally a single, targeted decomposition pass.  Provider
    metadata may omit ``native_gates``, but CCX must still be expanded before
    SABRE sees the circuit.  Other gates remain available for the
    provider-specific decomposition pass or translator.
    """
    from quspect.ir.compilation import decompose_circuit

    gate_names = {
        instruction.gate.name
        for instruction in circuit
        if instruction.is_gate and instruction.gate is not None
    }
    selected = gate_names & PAIRWISE_ROUTING_DECOMPOSITION_GATES
    if not selected:
        return circuit
    return decompose_circuit(circuit, supported_gates=gate_names - selected)


def compile_for_device(
    device_target: CompilationTarget,
    circuit: QuantumCircuit,
    *,
    initial_layout: Sequence[int] | None = None,
    physical_qubits: Sequence[int] | None = None,
) -> tuple[QuantumCircuit, list[int] | None]:
    """Decompose, optimize, then deterministically route a bound circuit.

    Args:
        device_target: The full device target (all physical qubits).
        circuit: Bound logical circuit.
        initial_layout: Optional starting mapping where
            ``initial_layout[logical] = physical``.
        physical_qubits: Optional ordered pool of physical qubits; routing and
            inserted SWAPs are restricted to the induced coupling subgraph,
            which must be connected.

    Returns:
        ``(routed_circuit, layout)``.  When the target has a coupling graph,
        ``routed_circuit`` is in SABRE *wire space*: wire ``w`` is pinned to
        physical qubit ``layout[w]`` for the whole circuit (SWAPs move state
        between wires).  Measurements keep their original classical bits, so
        results are already in logical clbit order and need no qubit-level
        un-permutation.  ``layout`` is ``None`` when no routing happened.
    """
    from quspect.transpiler._passes import (
        DecomposePass,
        LayoutRoutingPass,
        OptimizePass,
    )

    # SABRE supports only one- and two-qubit operations.  This fallback is
    # required even when native gate metadata is absent or includes CCX:
    # pairwise coupling data cannot describe a three-qubit interaction.
    circuit = decompose_for_pairwise_routing(circuit)

    # If the provider advertises a basis, recursively lower the fallback
    # output as well (for example, CCX-generated CX gates on a CZ-native
    # device).  With missing metadata, preserve translator-supported gates.
    if device_target.native_gates:
        circuit = DecomposePass(device_target).run(circuit)

    # Optimize after decomposition so redundant primitive gates are
    # removed before routing. Running optimization after SABRE would be
    # unsafe because the generic DAG does not carry its evolving mapping.
    # With a known basis, single-qubit runs are resynthesized directly
    # into native gates instead of being fused into ``u3``.
    circuit = OptimizePass.from_target(device_target).run(circuit)

    target = device_target
    dense_to_physical = list(range(device_target.num_qubits))
    dense_initial_layout: list[int] | None = None

    if physical_qubits is not None:
        target, dense_to_physical = device_target.induced_subgraph(physical_qubits)
        if not target.coupling_graph and circuit.num_qubits > 1:
            raise ValueError(
                "The selected physical_qubits do not contain a usable "
                "two-qubit coupling edge."
            )
        if target.num_qubits > 1 and circuit.num_qubits > 1:
            reachable = {0}
            frontier = [0]
            while frontier:
                current = frontier.pop()
                for neighbor in target.adjacency[current]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        frontier.append(neighbor)
            if len(reachable) != target.num_qubits:
                raise ValueError(
                    "physical_qubits must induce a connected coupling "
                    "subgraph on the backend."
                )

    if initial_layout is not None:
        if not isinstance(initial_layout, (list, tuple)):
            raise TypeError("initial_layout must be a list or tuple of integers.")
        requested_layout = list(initial_layout)
        if physical_qubits is None:
            dense_initial_layout = requested_layout
        else:
            physical_to_dense = {
                physical: dense for dense, physical in enumerate(dense_to_physical)
            }
            outside = [
                physical
                for physical in requested_layout
                if physical not in physical_to_dense
            ]
            if outside:
                raise ValueError(
                    "initial_layout contains qubits outside physical_qubits: "
                    f"{outside}."
                )
            dense_initial_layout = [
                physical_to_dense[physical] for physical in requested_layout
            ]

    if circuit.num_qubits > target.num_qubits:
        raise ValueError(
            f"Circuit uses {circuit.num_qubits} qubits but the selected "
            f"target provides only {target.num_qubits} physical qubits."
        )

    layout: list[int] | None = None
    if target.coupling_graph:
        routed, candidate_layout, _ = LayoutRoutingPass(
            target,
            iter_num=3,
            seed=0,
            initial_mapping=dense_initial_layout,
        ).run(circuit)
        # SABRE may add workspace wires when routing through otherwise
        # unused physical qubits.  Keep every wire that occurs in the
        # routed circuit and trim only unused trailing search workspace.
        if len(candidate_layout) >= routed.num_qubits:
            circuit = routed
            layout = [
                dense_to_physical[dense]
                for dense in candidate_layout[: routed.num_qubits]
            ]
    elif initial_layout is not None:
        if len(initial_layout) != circuit.num_qubits:
            raise ValueError(
                f"initial_layout length ({len(initial_layout)}) must equal "
                f"the circuit qubit count ({circuit.num_qubits})."
            )
        layout = list(initial_layout)

    return circuit, layout


def select_region(
    size: int,
    available: Sequence[int],
    coupling_map: Sequence[Sequence[int]],
    qubit_fidelity: Mapping[int, float],
    edge_fidelity: Mapping[tuple[int, int], float],
    *,
    compact_ties: bool = False,
) -> list[int] | None:
    """Pick a connected set of ``size`` qubits with the best calibration.

    Every usable qubit seeds a greedy growth that repeatedly adds the
    neighbour maximizing ``edge fidelity × qubit fidelity``; the region with
    the largest product of its qubit and joining-edge fidelities wins (ties:
    lowest qubit indices).  Qubits without calibration data count as
    perfect; qubits whose calibration is ``0`` are never used.  Returns
    ``None`` when no connected region of that size exists.

    With ``compact_ties`` equally good neighbours are told apart by shape
    before index: the one with the most couplers into the region, then the
    one nearest the seed.  Where calibration is flat the lowest-index rule
    grows a line, which leaves routing no shortcut; this grows a block.
    Calibration still decides whenever it differs.
    """
    usable = {q for q in available if qubit_fidelity.get(q, 1.0) > 0.0}
    adjacency: dict[int, set[int]] = {q: set() for q in usable}
    for a, b in coupling_map:
        if (
            a in usable
            and b in usable
            and edge_fidelity.get((min(a, b), max(a, b)), 1.0) > 0
        ):
            adjacency[a].add(b)
            adjacency[b].add(a)

    def log(value: float) -> float:
        return math.log(max(value, 1e-12))

    best: tuple[float, list[int]] | None = None
    for start in sorted(usable):
        region = [start]
        members = {start}
        distance = {start: 0}
        score = log(qubit_fidelity.get(start, 1.0))
        while len(region) < size:
            candidates = [
                (
                    log(edge_fidelity.get((min(u, v), max(u, v)), 1.0))
                    + log(qubit_fidelity.get(v, 1.0)),
                    -v,
                    v,
                )
                for u in region
                for v in adjacency[u]
                if v not in members
            ]
            if not candidates:
                break
            gain, _, chosen = max(candidates)
            if compact_ties:
                tied = {v for other, _, v in candidates if other >= gain - 1e-12}
                chosen = min(
                    tied,
                    key=lambda v: (
                        -len(adjacency[v] & members),
                        min(distance[u] for u in adjacency[v] & members),
                        v,
                    ),
                )
                gain = max(other for other, _, v in candidates if v == chosen)
                distance[chosen] = 1 + min(
                    distance[u] for u in adjacency[chosen] & members
                )
            region.append(chosen)
            members.add(chosen)
            score += gain
        if len(region) < size:
            continue
        ordered = sorted(region)
        if best is None or score > best[0] + 1e-12:
            best = (score, ordered)
    return None if best is None else best[1]


def placement_regions(
    num_qubits: int,
    available: Sequence[int],
    coupling_map: Sequence[Sequence[int]],
    qubit_fidelity: Mapping[int, float],
    edge_fidelity: Mapping[tuple[int, int], float],
    *,
    workspace: int = 2,
) -> list[list[int]]:
    """Calibration-chosen pools worth compiling a ``num_qubits`` circuit on.

    The best-calibrated connected region of exactly ``num_qubits`` qubits,
    and the best one with ``workspace`` more: the first uses nothing but the
    best qubits, the second leaves routing room to move.  Which serves a
    circuit better depends on the circuit, so callers compile on each and
    keep the best program (:func:`keep_best`).  Sizes the chip cannot hold
    are left out; the list is empty when not even ``num_qubits`` fit.
    """
    regions: list[list[int]] = []
    for size in (num_qubits, num_qubits + workspace):
        region = select_region(
            max(1, size),
            available,
            coupling_map,
            qubit_fidelity,
            edge_fidelity,
            compact_ties=True,
        )
        if region is not None and region not in regions:
            regions.append(region)
    return regions


def fill_missing_fidelities(
    known: Mapping[_K, float], keys: Iterable[_K]
) -> dict[_K, float]:
    """Give every key a fidelity: its own, else the worst published one.

    A qubit or coupler the calibration skips must not look better than the
    ones it measured, so it counts as the lowest positive value given for
    its kind.  When nothing is published the result is empty and lookups
    fall back to "perfect", which keeps an uncalibrated kind neutral.
    """
    if not known:
        return {}
    worst = min((value for value in known.values() if value > 0.0), default=1.0)
    filled = dict(known)
    for key in keys:
        filled.setdefault(key, worst)
    return filled


def estimated_log_fidelity(
    circuit: QuantumCircuit,
    *,
    edge_fidelity: Mapping[tuple[int, int], float],
    readout_fidelity: Mapping[int, float],
    gate_fidelity: Mapping[int, float] | None = None,
    layout: Sequence[int] | None = None,
) -> float:
    """Log of the product of the calibrated fidelities a program uses.

    One factor per two-qubit gate (the coupler's fidelity; a SWAP counts as
    three gates, a two-qubit gate other than CX/CZ as two), per measured
    qubit (its readout fidelity) and, with ``gate_fidelity``, per
    single-qubit gate.  Anything without a calibration value counts as
    perfect.  ``layout[wire]`` names the physical qubit of each circuit
    wire; without it the circuit's indices are physical.

    This ranks placements of one circuit against each other; it is not a
    prediction of the measured success probability.
    """

    def physical(wire: int) -> int:
        return wire if layout is None else layout[wire]

    def log(value: float) -> float:
        return math.log(max(value, 1e-12))

    total = 0.0
    for instruction in circuit.instructions:
        qubits = [physical(wire) for wire in instruction.qubits]
        if instruction.is_measure:
            total += sum(log(readout_fidelity.get(q, 1.0)) for q in qubits)
        elif not instruction.is_gate or instruction.gate is None:
            continue
        elif len(qubits) == 2:
            a, b = sorted(qubits)
            cost = _TWO_QUBIT_GATE_COST.get(
                instruction.gate.name, _OTHER_TWO_QUBIT_GATE_COST
            )
            total += cost * log(edge_fidelity.get((a, b), 1.0))
        elif len(qubits) == 1 and gate_fidelity is not None:
            total += log(gate_fidelity.get(qubits[0], 1.0))
    return total


def keep_best(candidates: Iterable[_T], score: Callable[[_T], float]) -> _T:
    """Return the candidate with the highest ``score``; the first wins ties.

    Callers list today's placement first, so a calibration-chosen region
    replaces it only when its program is estimated to be strictly better.
    ``candidates`` may be a generator that compiles one program per item.
    """
    best: tuple[float, _T] | None = None
    for candidate in candidates:
        value = score(candidate)
        if best is None or value > best[0] + _ESTIMATE_TOLERANCE:
            best = (value, candidate)
    if best is None:
        raise ValueError("keep_best needs at least one candidate.")
    return best[1]
