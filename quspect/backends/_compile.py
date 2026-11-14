"""Shared device compilation for cloud QPU backends.

Cloud backends that let QuSpect compile for the device (rather than leaving
qubit mapping to the provider) all go through :func:`compile_for_device`.  It
is the one seam between the backends and the compiler: a circuit that
:func:`~quspect.transpiler.transpile` can take is compiled by it, so hardware
runs get the same compiler as ``transpile()`` calls in user code; everything
else keeps the earlier pipeline (lower to the device basis, optimize, route
once with SABRE).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction
    from quspect.transpiler._target import CompilationTarget

logger = logging.getLogger(__name__)

#: ``optimization_level`` and ``seed`` of the :func:`~quspect.transpiler.transpile`
#: call behind :func:`compile_for_device`.  Level 2 is ``transpile()``'s own
#: default; the fixed seed makes a submission reproducible (the same circuit,
#: target and options always give the same program).
DEVICE_OPTIMIZATION_LEVEL = 2
DEVICE_SEED = 0
#: Largest circuit :func:`compile_for_device` hands to ``transpile()``, in
#: two-qubit gates of the input (a three-qubit gate counts as six).  Level 2
#: has no time limit: it takes up to about a millisecond per two-qubit gate of
#: its output, ten to sixty times the earlier pipeline, and a submission
#: compiles every circuit of a batch.  Longer circuits keep the earlier
#: pipeline and its compile time.
DEVICE_TRANSPILE_MAX_TWO_QUBIT_GATES = 200

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


def _terminal_measurements(instructions: Sequence[CircuitInstruction]) -> set[int]:
    """Indices of the terminal measurements of an instruction list.

    A measurement is *terminal* when no later instruction acts on its qubits
    or writes its classical bits, other than a barrier or another terminal
    measurement.
    """
    blocked_qubits: set[int] = set()
    blocked_clbits: set[int] = set()
    terminal: set[int] = set()
    for index in range(len(instructions) - 1, -1, -1):
        instruction = instructions[index]
        if instruction.is_barrier:
            continue
        if (
            instruction.is_measure
            and blocked_qubits.isdisjoint(instruction.qubits)
            and blocked_clbits.isdisjoint(instruction.clbits)
        ):
            terminal.add(index)
            continue
        blocked_qubits.update(instruction.qubits)
        blocked_clbits.update(instruction.clbits)
    return terminal


def defer_terminal_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    """Move every terminal measurement behind all other instructions.

    A measurement is *terminal* when no later instruction acts on its qubits
    or writes its classical bits, other than a barrier or another terminal
    measurement.  It commutes with everything that follows it, so issuing it
    last leaves the circuit's results unchanged.  Terminal measurements keep
    their relative order; a mid-circuit measurement (one that is followed by
    a gate on its qubit, or whose classical bit is written again by a
    mid-circuit measurement) stays where it is.

    Device compilation needs this before routing.  The optimizer may emit a
    qubit's measurement as soon as that qubit's gates are done, ahead of
    gates on other qubits; a SWAP inserted for one of those gates could then
    cross the measured wire and turn a terminal measurement into a
    mid-circuit one.  With the measurements issued last, the router (which
    treats a measurement as a scheduling fence) places every SWAP before
    them and each classical bit reads the physical qubit that holds its
    logical qubit after the final SWAP.

    Returns:
        ``circuit`` itself when its terminal measurements are already last,
        otherwise a reordered copy.
    """
    from quspect.circuit.circuit import QuantumCircuit

    instructions = circuit.instructions
    terminal = _terminal_measurements(instructions)
    first = min(terminal, default=len(instructions))
    if all(
        index in terminal or instructions[index].is_barrier
        for index in range(first, len(instructions))
    ):
        return circuit
    body = [ins for index, ins in enumerate(instructions) if index not in terminal]
    tail = [instructions[index] for index in sorted(terminal)]
    return QuantumCircuit._from_instruction_snapshot(body + tail, name=circuit.name)


def compile_for_device(
    device_target: CompilationTarget,
    circuit: QuantumCircuit,
    *,
    initial_layout: Sequence[int] | None = None,
    physical_qubits: Sequence[int] | None = None,
) -> tuple[QuantumCircuit, list[int] | None]:
    """Compile a bound circuit for a device, deterministically.

    The circuit is compiled by :func:`~quspect.transpiler.transpile`
    (``optimization_level=DEVICE_OPTIMIZATION_LEVEL``, ``seed=DEVICE_SEED``)
    whenever ``transpile()`` can take it: the routed qubits have couplings,
    the native gates are either undeclared or a basis ``transpile()``
    supports, and the circuit is gates followed by terminal measurements.
    The measurements are taken off first and put back behind the compiled
    gates, each on the wire that holds its logical qubit at the end
    (``TranspileResult.final_layout``).

    Every other circuit keeps the earlier pipeline (decompose, optimize, one
    SABRE routing; see :func:`_compile_with_passes`): mid-circuit
    measurements, barriers between gates (``transpile()`` optimizes across
    them), noise channels, a native gate set that is not a universal basis
    (``{"cz"}``, ``{"cz", "rx"}``), qubits without couplings, more than
    ``DEVICE_TRANSPILE_MAX_TWO_QUBIT_GATES`` two-qubit gates, arguments the
    earlier pipeline rejects, and anything ``transpile()`` refuses with a
    ``ValueError``.

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
        ``routed_circuit`` is in *wire space*: wire ``w`` is pinned to
        physical qubit ``layout[w]`` for the whole circuit (SWAPs move state
        between wires).  Measurements keep their original classical bits, so
        results are already in logical clbit order and need no qubit-level
        un-permutation.  ``layout`` is ``None`` when no routing happened.

        When the target declares native gates, the returned circuit holds
        only those: ``transpile()`` compiles to them, and the earlier
        pipeline lowers its routing SWAPs to them after routing.

        Terminal measurements stay terminal: they come after every gate of
        the returned circuit, each on the wire that holds its logical qubit
        after the final SWAP.  A gate follows a measurement only when the
        input circuit itself used a qubit after measuring it (see
        :func:`defer_terminal_measurements`).

    Raises:
        ValueError: If a gate, or a SWAP that routing needs, cannot be
            expressed in the target's native gates.
    """
    compiled = _compile_with_transpile(
        device_target,
        circuit,
        initial_layout=initial_layout,
        physical_qubits=physical_qubits,
    )
    if compiled is not None:
        return compiled
    return _compile_with_passes(
        device_target,
        circuit,
        initial_layout=initial_layout,
        physical_qubits=physical_qubits,
    )


def _gates_and_terminal_measurements(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, list[CircuitInstruction]] | None:
    """Split a circuit into its gates and its terminal measurements.

    Returns ``None`` unless the circuit is exactly that: at least one gate,
    then terminal measurements.  A barrier behind the last gate orders
    nothing and is dropped; a barrier between gates is a fence the caller
    asked for, so such a circuit is not split (nor is one with a mid-circuit
    measurement or a noise channel).
    """
    from quspect.circuit.circuit import QuantumCircuit

    instructions = circuit.instructions
    terminal = _terminal_measurements(instructions)
    last_gate = max(
        (index for index, ins in enumerate(instructions) if ins.is_gate), default=-1
    )
    if last_gate < 0:
        return None
    gates: list[CircuitInstruction] = []
    for index, instruction in enumerate(instructions):
        if index in terminal:
            continue
        if instruction.is_gate and instruction.gate is not None:
            gates.append(instruction)
        elif not (instruction.is_barrier and index > last_gate):
            return None
    body = QuantumCircuit._from_instruction_snapshot(gates, name=circuit.name)
    return body, [instructions[index] for index in sorted(terminal)]


def _transpile_target(
    device_target: CompilationTarget,
    num_logical: int,
    initial_layout: Sequence[int] | None,
    physical_qubits: Sequence[int] | None,
) -> tuple[CompilationTarget, list[int], list[int] | None] | None:
    """The target ``transpile()`` routes on, or ``None`` if there is none.

    Returns:
        ``(target, dense_to_physical, dense_initial_layout)``: the device, or
        the connected subgraph induced by ``physical_qubits`` in dense
        indices, and the requested layout in those indices.  ``None`` when
        the qubits have no couplings or are too few, or when an argument is
        invalid (:func:`_compile_with_passes` then raises its own error).
    """
    from quspect.transpiler._mapping import validate_initial_mapping

    target = device_target
    dense_to_physical = list(range(device_target.num_qubits))
    try:
        if physical_qubits is not None:
            target, dense_to_physical = device_target.induced_subgraph(physical_qubits)
        if not target.coupling_graph or num_logical > target.num_qubits:
            return None
        if physical_qubits is not None and not _is_connected(target):
            return None
        if initial_layout is None:
            return target, dense_to_physical, None
        if not isinstance(initial_layout, (list, tuple)):
            return None
        dense_of = {physical: dense for dense, physical in enumerate(dense_to_physical)}
        if any(physical not in dense_of for physical in initial_layout):
            return None
        requested = validate_initial_mapping(
            [dense_of[physical] for physical in initial_layout],
            num_logical,
            target.num_qubits,
        )
    except (TypeError, ValueError):
        return None
    return target, dense_to_physical, requested


def _is_connected(target: CompilationTarget) -> bool:
    reachable = {0}
    frontier = [0]
    while frontier:
        for neighbor in target.adjacency[frontier.pop()]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    return len(reachable) == target.num_qubits


def _compile_with_transpile(
    device_target: CompilationTarget,
    circuit: QuantumCircuit,
    *,
    initial_layout: Sequence[int] | None,
    physical_qubits: Sequence[int] | None,
) -> tuple[QuantumCircuit, list[int]] | None:
    """Compile with :func:`~quspect.transpiler.transpile`, or return ``None``.

    Arguments and result as in :func:`compile_for_device`.  ``None`` means
    that ``transpile()`` cannot take this circuit or target, or that an
    argument is invalid; :func:`_compile_with_passes` then compiles the
    circuit or raises the error.
    """
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._transpile import transpile

    split = _gates_and_terminal_measurements(circuit)
    if split is None:
        return None
    body, measurements = split
    load = sum(
        1 if len(instruction.qubits) == 2 else 6
        for instruction in body.instructions
        if len(instruction.qubits) >= 2
    )
    if load > DEVICE_TRANSPILE_MAX_TWO_QUBIT_GATES:
        return None
    routed_on = _transpile_target(
        device_target, circuit.num_qubits, initial_layout, physical_qubits
    )
    if routed_on is None:
        return None
    target, dense_to_physical, requested = routed_on
    native = set(target.native_gates)
    width = body.num_qubits
    try:
        result = transpile(
            body,
            target,
            optimization_level=DEVICE_OPTIMIZATION_LEVEL,
            seed=DEVICE_SEED,
            initial_layout=None if requested is None else requested[:width],
        )
    except ValueError as exc:
        # A gate set that is not a basis, a gate that cannot be lowered to
        # it, ...: the earlier pipeline compiles the circuit or reports the
        # problem the way the backends expect.
        logger.debug("transpile() cannot take the circuit: %s", exc)
        return None
    except Exception:
        # A submission that the earlier pipeline can compile must not fail on
        # a defect of the larger compiler.
        logger.warning(
            "transpile() failed; compiling with the earlier pipeline instead.",
            exc_info=True,
        )
        return None

    compiled = result.wire_circuit
    layout = list(result.wire_layout)
    final = list(result.final_layout)
    problem = _placement_problem(compiled, layout, final, width, target, native)
    if problem is not None:
        logger.warning(
            "transpile() returned a circuit the device cannot run (%s); "
            "compiling with the earlier pipeline instead.",
            problem,
        )
        return None

    # A logical qubit above the gates' width is only measured.  It is in
    # |0>, like every physical qubit that holds no logical qubit at the end:
    # read it from its requested qubit when that one is free, else from a
    # qubit the compiled circuit never touches, else from routing workspace.
    wire_of = {dense: wire for wire, dense in enumerate(layout)}
    taken = set(final)
    untouched = [dense for dense in range(target.num_qubits) if dense not in wire_of]
    for logical in range(width, circuit.num_qubits):
        wanted = [] if requested is None else [requested[logical]]
        final.append(
            next(
                dense for dense in (*wanted, *untouched, *layout) if dense not in taken
            )
        )
        taken.add(final[-1])

    tail: list[CircuitInstruction] = []
    for measurement in measurements:
        wires = []
        for logical in measurement.qubits:
            dense = final[logical]
            if dense not in wire_of:
                wire_of[dense] = len(layout)
                layout.append(dense)
            wires.append(wire_of[dense])
        tail.append(replace(measurement, qubits=tuple(wires)))
    gates = _without_diagonals_before_measurement(
        compiled.instructions,
        {wire for measurement in tail for wire in measurement.qubits},
    )
    routed = QuantumCircuit._from_instruction_snapshot(
        [*gates, *tail], name=circuit.name
    )
    return routed, [dense_to_physical[dense] for dense in layout[: routed.num_qubits]]


def _without_diagonals_before_measurement(
    gates: Sequence[CircuitInstruction], measured: set[int]
) -> list[CircuitInstruction]:
    """Drop the diagonal single-qubit gates that end a measured wire.

    The wires in ``measured`` are measured in the Z basis right after
    ``gates``; a diagonal gate that is the last one on such a wire only adds
    a phase the measurement does not see.  This is the rule of
    :class:`~quspect.dag.passes.EliminateDiagonalBeforeMeasurement`, which
    the optimizer applies when it sees the measurements; here they were taken
    off before compiling.
    """
    from quspect.dag.passes.diagonal_before_measure import DIAGONAL_ELIMINABLE

    open_wires = set(measured)
    kept: list[CircuitInstruction] = []
    for instruction in reversed(gates):
        gate = instruction.gate
        if (
            gate is not None
            and len(instruction.qubits) == 1
            and instruction.qubits[0] in open_wires
            and gate.name in DIAGONAL_ELIMINABLE
        ):
            continue
        open_wires.difference_update(instruction.qubits)
        kept.append(instruction)
    kept.reverse()
    return kept


def _placement_problem(
    compiled: QuantumCircuit,
    layout: Sequence[int],
    final: Sequence[int],
    width: int,
    target: CompilationTarget,
    native: set[str],
) -> str | None:
    """Why a ``transpile()`` result cannot be sent to the device, if anything.

    A structural check of the result against the target (linear in the
    circuit): the submit path does not send a program that names a missing
    coupling or a foreign gate, whatever produced it.
    """
    if len(final) != width or len(set(final)) != len(final):
        return f"final layout {list(final)} does not place {width} logical qubits"
    if len(set(layout)) != len(layout) or compiled.num_qubits > len(layout):
        return f"wire layout {list(layout)} does not cover the circuit"
    if any(not 0 <= dense < target.num_qubits for dense in (*layout, *final)):
        return f"layout {list(layout)} -> {list(final)} leaves the qubit pool"
    for instruction in compiled.instructions:
        gate = instruction.gate
        if not instruction.is_gate or gate is None:
            return "the compiled gates contain a non-gate instruction"
        if native and gate.name not in native:
            return f"gate {gate.name!r} is not native"
        if len(instruction.qubits) > 2:
            return f"gate {gate.name!r} acts on {len(instruction.qubits)} qubits"
        if len(instruction.qubits) == 2:
            a, b = (layout[wire] for wire in instruction.qubits)
            if (min(a, b), max(a, b)) not in target.coupling_graph:
                return f"gate {gate.name!r} on uncoupled qubits {(a, b)}"
    return None


def _compile_with_passes(
    device_target: CompilationTarget,
    circuit: QuantumCircuit,
    *,
    initial_layout: Sequence[int] | None = None,
    physical_qubits: Sequence[int] | None = None,
) -> tuple[QuantumCircuit, list[int] | None]:
    """Decompose, optimize, then deterministically route a bound circuit.

    The pipeline behind :func:`compile_for_device` before it called
    ``transpile()``, kept for what ``transpile()`` cannot take.  Arguments
    and result as in :func:`compile_for_device`.  When the target declares
    native gates, the routing SWAPs are lowered to them; without a declared
    set they stay ``swap`` for the provider to lower.
    """
    from quspect.transpiler._mapping import validate_initial_mapping
    from quspect.transpiler._passes import (
        DecomposePass,
        LayoutRoutingPass,
        OptimizePass,
    )

    # A QuantumCircuit infers its width from the highest qubit in use, so the
    # optimized circuit below is narrower than the input when the top qubits
    # lose their gates (gates that cancel).  ``initial_layout`` has one entry
    # per logical qubit of the input.
    num_logical = circuit.num_qubits

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

    # The optimizer orders instructions per wire, so it may emit a qubit's
    # measurement ahead of gates on other qubits.  Issue terminal
    # measurements last: no routing SWAP may cross a measured wire, and
    # providers read out all measured qubits together.
    circuit = defer_terminal_measurements(circuit)

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
        if dense_initial_layout is not None:
            # Route the wires that still carry instructions from their
            # requested qubits.  A logical qubit without any instruction left
            # needs no wire: its physical qubit is free for routing.
            dense_initial_layout = validate_initial_mapping(
                dense_initial_layout, num_logical, target.num_qubits
            )[: circuit.num_qubits]
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
        if len(initial_layout) != num_logical:
            raise ValueError(
                f"initial_layout length ({len(initial_layout)}) must equal "
                f"the circuit qubit count ({num_logical})."
            )
        layout = list(initial_layout)[: circuit.num_qubits]

    # Everything else is native already (decomposed above); routing adds
    # SWAPs, which a declared set need not hold.  A decomposition acts on
    # the gate's own wires, so the placement stays valid.
    if device_target.native_gates:
        try:
            circuit = DecomposePass(device_target).run(circuit)
        except ValueError as exc:
            raise ValueError(
                "The routing SWAPs cannot be expressed in the native gates "
                f"{sorted(device_target.native_gates)}: {exc}"
            ) from exc

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
