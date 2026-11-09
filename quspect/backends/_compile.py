"""Shared device compilation for cloud QPU backends.

Cloud backends that let QuSpect compile for the device (rather than leaving
qubit mapping to the provider) all need the same pipeline: lower the circuit
to the device basis, optimize, then route it on the device coupling graph.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._target import CompilationTarget

# SABRE models pairwise coupling edges, so higher-arity gates must be lowered
# before topology routing even when provider metadata claims they are native
# or omits a native gate set altogether.  Keep this list deliberately narrow:
# a provider translator can still accept CCX when transpilation is disabled.
PAIRWISE_ROUTING_DECOMPOSITION_GATES = frozenset({"ccx"})


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
) -> list[int] | None:
    """Pick a connected set of ``size`` qubits with the best calibration.

    Every usable qubit seeds a greedy growth that repeatedly adds the
    neighbour maximizing ``edge fidelity × qubit fidelity``; the region with
    the largest product of its qubit and joining-edge fidelities wins (ties:
    lowest qubit indices).  Qubits without calibration data count as
    perfect; qubits whose calibration is ``0`` are never used.  Returns
    ``None`` when no connected region of that size exists.
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
            region.append(chosen)
            members.add(chosen)
            score += gain
        if len(region) < size:
            continue
        ordered = sorted(region)
        if best is None or score > best[0] + 1e-12:
            best = (score, ordered)
    return None if best is None else best[1]
