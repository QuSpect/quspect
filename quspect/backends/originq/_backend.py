"""Origin Quantum (本源量子) cloud backend over the cloud's HTTP API.

Who compiles and who maps qubits
--------------------------------
Real QPUs (e.g. ``WK_C180_2``) natively run ``RPhi`` and ``CZ`` on a subset
of "available" physical qubits.  Two compilation modes are supported:

``transpile=True`` (default)
    QuSpect lowers the circuit to ``{cz, u3, rz}``, optimizes it and routes
    it with SABRE on the coupling graph of the available qubits (the largest
    connected component unless ``physical_qubits`` is given).  The routed
    wire-space circuit is placed on physical qubits by QuSpect and submitted
    with provider mapping **disabled**; the provider still lowers
    ``u3``/``rz`` to its ``RPhi`` pulses.
``transpile=False``
    The circuit is submitted as written.  With ``provider_mapping=True``
    (the default in this mode) Origin Quantum chooses the layout and
    inserts SWAPs; with ``provider_mapping=False`` the circuit's qubit
    indices are used as physical qubits.

Either way results come back keyed by measured qubit and are decoded into
QuSpect classical-bit order through the measurement map recorded at
submission (see :mod:`quspect.backends.originq._translator`).  Routing never
requires un-permuting result bits: measurements keep their classical bits.

The cloud simulator ``full_amplitude`` takes no compilation options and no
native batches; requests are submitted one task per circuit and combined
into a :class:`~quspect.backends.CompositeJob`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quspect.backends._base import (
    CloudBackend,
    CloudJob,
    CompositeJob,
    _request_workload_id,
)
from quspect.backends._exceptions import (
    BackendError,
    PartialSubmissionError,
    TranslationError,
)
from quspect.backends._execution import (
    BackendCapabilities,
    ExecutionBatch,
    ExecutionRequest,
    JobReference,
)
from quspect.backends.originq._client import OriginQClient
from quspect.backends.originq._job import PLATFORM, OriginQJob
from quspect.backends.originq._translator import (
    SUPPORTED_GATES,
    OriginIRProgram,
    translate,
)

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.simulation.result import SimulationResult

#: ``machineType`` of the full-amplitude cloud simulator.
SIMULATOR_MACHINE_TYPE = 0
#: ``machineType`` of real QPUs.
QPU_MACHINE_TYPE = 5
#: Cloud simulators that return amplitudes instead of samples.
NON_SAMPLING_BACKENDS = frozenset({"partial_amplitude", "single_amplitude"})
#: Basis QuSpect compiles to for QPUs; the provider lowers it to RPhi + CZ.
QPU_COMPILE_BASIS = frozenset({"cz", "u3", "rz"})

# The provider does not document a per-batch circuit limit.  Chunk QPU
# batches conservatively; larger requests become a CompositeJob.
_MAX_BATCH_CIRCUITS = 64

_QPU_OPTIONS = frozenset(
    {
        "transpile",
        "initial_layout",
        "physical_qubits",
        "readout_correction",
        "provider_mapping",
        "provider_optimization",
    }
)
_SIMULATOR_OPTIONS = frozenset({"transpile"})


@dataclass(frozen=True)
class _Prepared:
    program: OriginIRProgram
    label: str
    shape: tuple[int, int]


def _int_sequence(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must be a sequence of integer qubits.")
    return list(value)


def _optional_bool(options: Mapping[str, Any], name: str) -> bool | None:
    value = options.get(name)
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"Origin Quantum option {name!r} must be bool.")
    return value


def edges_from_adjacency(adjacency: Any) -> list[list[int]]:
    """Undirected, de-duplicated edges from a chipInfo ``adjMatrix``."""
    edges: set[tuple[int, int]] = set()
    if isinstance(adjacency, Mapping):
        for source, neighbors in adjacency.items():
            for neighbor in neighbors or ():
                target = neighbor.get("v") if isinstance(neighbor, Mapping) else None
                if target is None:
                    continue
                a, b = int(source), int(target)
                if a != b:
                    edges.add((min(a, b), max(a, b)))
    return [list(edge) for edge in sorted(edges)]


def qubit_fidelities(params: Any) -> dict[int, float]:
    """``single-gate fidelity × readout fidelity`` per qubit from ``QubitParams``.

    A calibration value of ``0`` (failed calibration) yields ``0`` and keeps
    the qubit out of automatic placement.
    """
    scores: dict[int, float] = {}
    if isinstance(params, list):
        for entry in params:
            if not isinstance(entry, Mapping) or "qubitId" not in entry:
                continue
            try:
                single = float(entry.get("SingleGateFidelity", 1.0))
                readout = float(entry.get("ReadoutFidelity", 1.0))
                scores[int(entry["qubitId"])] = max(0.0, single) * max(0.0, readout)
            except (TypeError, ValueError):
                continue
    return scores


def edge_fidelities(adjacency: Any) -> dict[tuple[int, int], float]:
    """CZ fidelity per undirected edge from a chipInfo ``adjMatrix``."""
    weights: dict[tuple[int, int], float] = {}
    if isinstance(adjacency, Mapping):
        for source, neighbors in adjacency.items():
            for neighbor in neighbors or ():
                if not isinstance(neighbor, Mapping) or neighbor.get("v") is None:
                    continue
                a, b = int(source), int(neighbor["v"])
                if a == b:
                    continue
                try:
                    weight = float(neighbor.get("w", 1.0))
                except (TypeError, ValueError):
                    weight = 1.0
                key = (min(a, b), max(a, b))
                weights[key] = min(weights.get(key, weight), weight)
    return weights


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


class OriginQBackend(CloudBackend):
    """An Origin Quantum QPU or the ``full_amplitude`` cloud simulator."""

    def __init__(
        self,
        client: OriginQClient,
        resource: Mapping[str, Any],
        *,
        provider: Any = None,
    ) -> None:
        name = str(resource.get("pyqpandaChipId") or "")
        machine_type = resource.get("machineType")
        if name in NON_SAMPLING_BACKENDS:
            raise ValueError(
                f"Origin Quantum backend {name!r} returns amplitudes, not "
                "samples, and cannot be used as a QuSpect sampling backend."
            )
        if machine_type not in (SIMULATOR_MACHINE_TYPE, QPU_MACHINE_TYPE):
            raise ValueError(
                f"Origin Quantum backend {name!r} (machineType={machine_type!r}) "
                "is not a supported sampling backend."
            )
        self._client = client
        self._resource = dict(resource)
        self._name = name
        self._machine_type = int(machine_type)
        self._provider = provider
        self._chip_cache: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def platform_name(self) -> str:
        return PLATFORM

    @property
    def is_simulator(self) -> bool:
        return self._machine_type == SIMULATOR_MACHINE_TYPE

    @property
    def online(self) -> bool:
        """Whether the provider currently reports this backend online."""
        return bool(self._resource.get("onlineFlag")) and bool(
            self._resource.get("available", True)
        )

    def _chip(self) -> dict[str, Any]:
        if self.is_simulator:
            return {}
        if self._chip_cache is None:
            info = self._client.chip_info(self._name)
            try:
                self._chip_cache = {
                    "num_qubits": int(info["QubitCount"]),
                    "available_qubits": sorted(
                        int(q) for q in info.get("AvailableQubits", [])
                    ),
                    "coupling_map": edges_from_adjacency(info.get("adjMatrix")),
                    "basic_gates": [str(g) for g in info.get("BasicQGate", [])],
                    "max_shots": int(info["MaxShots"])
                    if info.get("MaxShots")
                    else None,
                    "qubit_fidelity": qubit_fidelities(info.get("QubitParams")),
                    "edge_fidelity": edge_fidelities(info.get("adjMatrix")),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"Unexpected chip information for {self._name!r}: {exc}"
                ) from exc
        return self._chip_cache

    @property
    def config(self) -> dict[str, Any]:
        if self.is_simulator:
            return {
                "name": self._name,
                "kind": "simulator",
                "num_qubits": self.num_qubits,
                "topology": {},
            }
        chip = self._chip()
        return {
            "name": self._name,
            "kind": "qpu",
            "num_qubits": chip["num_qubits"],
            "available_qubits": list(chip["available_qubits"]),
            "basic_gates": list(chip["basic_gates"]),
            "max_shots": chip["max_shots"],
            "topology": {"coupling_map": [list(e) for e in chip["coupling_map"]]},
        }

    @property
    def num_qubits(self) -> int:
        """Physical qubits on the chip, or the simulator's qubit limit."""
        if self.is_simulator:
            return int(self._resource.get("qubitNum") or 0)
        return int(self._chip()["num_qubits"])

    @property
    def available_qubits(self) -> list[int]:
        """Physical qubits currently usable on the chip."""
        return list(self._chip().get("available_qubits", []))

    @property
    def native_gate_set(self) -> set[str]:
        """QuSpect gate names this backend compiles to.

        For QPUs this is :data:`QPU_COMPILE_BASIS`; the provider's own native
        names (``RPhi``, ``CZ``) are in ``config["basic_gates"]``.
        """
        return set(SUPPORTED_GATES if self.is_simulator else QPU_COMPILE_BASIS)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_sampling=True,
            supports_batch=True,
            supports_statevector=False,
            supports_job_recovery=True,
            supports_cancel=False,
            max_circuits_per_job=1 if self.is_simulator else _MAX_BATCH_CIRCUITS,
            max_shots=None if self.is_simulator else self._chip().get("max_shots"),
            supported_modes=frozenset({"circuit"}),
            result_formats=frozenset({"auto", "counts"}),
        )

    def placement_qubits(self, num_qubits: int) -> list[int]:
        """Physical qubits QuSpect routes a ``num_qubits`` circuit on.

        The best-calibrated connected region of exactly that size (see
        :func:`select_region`).  On WK_C180 this matters: low-index qubits
        such as 0-2 have a failed single-gate calibration or poor readout and
        returned wrong results, while the provider's own mapping avoids them.
        Falls back to :meth:`default_physical_qubits` when no region fits.
        """
        chip = self._chip()
        region = select_region(
            max(1, int(num_qubits)),
            chip.get("available_qubits", []),
            chip.get("coupling_map", []),
            chip.get("qubit_fidelity", {}),
            chip.get("edge_fidelity", {}),
        )
        return region if region is not None else self.default_physical_qubits()

    def default_physical_qubits(self) -> list[int]:
        """Largest connected component of the available qubits."""
        chip = self._chip()
        available = set(chip.get("available_qubits", []))
        adjacency: dict[int, set[int]] = {q: set() for q in available}
        for a, b in chip.get("coupling_map", []):
            if a in available and b in available:
                adjacency[a].add(b)
                adjacency[b].add(a)
        best: list[int] = []
        seen: set[int] = set()
        for start in sorted(available):
            if start in seen:
                continue
            component, frontier = [], [start]
            seen.add(start)
            while frontier:
                node = frontier.pop()
                component.append(node)
                for neighbor in adjacency[node] - seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
            if len(component) > len(best):
                best = component
        return sorted(best)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        timeout: float | None = None,
        poll_interval: float = 2.0,
        **options: Any,
    ) -> SimulationResult:
        """Submit one circuit and block until its result is ready.

        Options: ``transpile`` (bool, default True), ``initial_layout``,
        ``physical_qubits``, ``readout_correction`` (provider "amend",
        default True), ``provider_mapping`` and ``provider_optimization``.
        All but ``transpile`` are QPU-only.
        """
        options.pop("save_statevector", None)
        job = self.submit(circuit, shots=shots, **options)
        result = job.result(timeout=timeout, poll_interval=poll_interval)
        if isinstance(result, ExecutionBatch):
            return result.single()
        return result

    def submit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        **options: Any,
    ) -> CloudJob:
        """Submit one circuit and return a job that yields one result."""
        from quspect.circuit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                "OriginQBackend.submit expects QuantumCircuit; use "
                "submit_request() for labelled batches."
            )
        result_format = options.pop("result_format", "auto")
        request = ExecutionRequest.single(
            circuit,
            label=circuit.name or "circuit",
            shots=shots,
            result_format=result_format,
            options=options,
        )
        return self._submit(request, single=True)

    def submit_request(self, request: ExecutionRequest) -> CloudJob:
        """Submit an ordered request; results keep the request order."""
        if not isinstance(request, ExecutionRequest):
            raise TypeError(
                "OriginQBackend.submit_request expects ExecutionRequest, "
                f"got {type(request).__name__}."
            )
        return self._submit(request, single=False)

    def _submit(self, request: ExecutionRequest, *, single: bool) -> CloudJob:
        if request.result_format not in self.capabilities.result_formats:
            raise ValueError(
                "Origin Quantum does not support result format "
                f"{request.result_format!r}."
            )
        workload_id = _request_workload_id(request)
        prepared, configuration = self._prepare(request)

        if self.is_simulator:
            groups = [[item] for item in prepared]
        else:
            groups = [
                prepared[i : i + _MAX_BATCH_CIRCUITS]
                for i in range(0, len(prepared), _MAX_BATCH_CIRCUITS)
            ]

        jobs: list[CloudJob] = []
        submitted = 0
        try:
            for group in groups:
                batch = not self.is_simulator and not single
                task_id = self._submit_group(group, configuration, batch=batch)
                reference = JobReference(
                    platform=PLATFORM,
                    backend_name=self._name,
                    job_id=task_id,
                    base_url=self._client.base_url,
                    transport="async",
                    batch_size=None if single else len(group),
                    labels=tuple(item.label for item in group),
                    experiment_shapes=tuple(item.shape for item in group),
                    shots=request.shots,
                    result_format=request.result_format,
                    workload_id=workload_id,
                    measurement_maps=tuple(
                        item.program.measurement.clbits for item in group
                    ),
                )
                jobs.append(
                    OriginQJob(
                        self._client,
                        task_id,
                        reference=reference,
                        return_batch=not single,
                    )
                )
                submitted += len(group)
        except Exception as exc:
            if jobs:
                raise PartialSubmissionError(
                    "Origin Quantum submission failed after "
                    f"{submitted} experiment(s); Origin Quantum tasks cannot be "
                    "cancelled, the submitted task IDs are on the exception: "
                    f"{exc}",
                    submitted_references=tuple(job.reference for job in jobs),
                    cancellation_failures=[job.job_id for job in jobs],
                ) from exc
            if isinstance(exc, (BackendError, TranslationError)):
                raise
            raise BackendError(f"Origin Quantum submission failed: {exc}") from exc

        if len(jobs) == 1:
            return jobs[0]
        return CompositeJob(jobs)

    def _submit_group(
        self,
        group: Sequence[_Prepared],
        configuration: dict[str, Any],
        *,
        batch: bool,
    ) -> str:
        first = group[0].program
        try:
            return self._client.submit(
                [item.program.text for item in group],
                machine_type=str(self._machine_type),
                configuration=configuration,
                chip_id=None if self.is_simulator else self._name,
                qubit_num=None if batch else first.num_qubits,
                clbit_num=None if batch else first.num_clbits,
                batch=batch,
            )
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(
                f"Failed to submit to Origin Quantum backend {self._name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _prepare(
        self, request: ExecutionRequest
    ) -> tuple[list[_Prepared], dict[str, Any]]:
        shots = request.shots
        if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
            raise ValueError(f"shots must be a positive integer, got {shots!r}.")
        options = dict(request.options)
        allowed = _SIMULATOR_OPTIONS if self.is_simulator else _QPU_OPTIONS
        unknown = sorted(set(options) - allowed)
        if unknown:
            kind = "the cloud simulator" if self.is_simulator else "Origin Quantum"
            raise ValueError(f"Unsupported option(s) for {kind}: {unknown}.")
        transpile = options.get("transpile", True)
        if not isinstance(transpile, bool):
            raise TypeError("Origin Quantum option 'transpile' must be bool.")

        if self.is_simulator:
            prepared = []
            for circuit, label, shape in self._experiments(request):
                item = self._translate(circuit, label, shape, qubit_map=None)
                if self.num_qubits and item.program.num_qubits > self.num_qubits:
                    raise TranslationError(
                        f"Experiment {label!r} uses {item.program.num_qubits} "
                        f"qubits; {self._name} supports {self.num_qubits}."
                    )
                prepared.append(item)
            return prepared, {"shot": shots, "measure_type": 1}

        max_shots = self._chip().get("max_shots")
        if max_shots and shots > max_shots:
            raise ValueError(f"shots must be at most {max_shots} on {self._name}.")
        initial_layout = options.get("initial_layout")
        if initial_layout is not None:
            initial_layout = _int_sequence(initial_layout, "initial_layout")
        physical_qubits = options.get("physical_qubits")
        if physical_qubits is not None:
            physical_qubits = _int_sequence(physical_qubits, "physical_qubits")
        provider_mapping = _optional_bool(options, "provider_mapping")
        provider_optimization = _optional_bool(options, "provider_optimization")
        readout_correction = _optional_bool(options, "readout_correction")

        if transpile:
            if provider_mapping:
                raise ValueError(
                    "provider_mapping=True conflicts with transpile=True: "
                    "QuSpect already placed the circuit on physical qubits."
                )
            provider_mapping = False
            if physical_qubits is None and initial_layout is not None:
                # An explicit layout names physical qubits itself; route on the
                # whole usable chip around it.
                physical_qubits = self.default_physical_qubits()
            elif physical_qubits is not None:
                unavailable = sorted(set(physical_qubits) - set(self.available_qubits))
                if unavailable:
                    raise ValueError(
                        f"physical_qubits contains unavailable qubit(s) {unavailable}."
                    )
        else:
            if initial_layout is not None or physical_qubits is not None:
                raise ValueError(
                    "initial_layout/physical_qubits require transpile=True. With "
                    "transpile=False write physical qubit indices directly into "
                    "the circuit and pass provider_mapping=False."
                )
            if provider_mapping is None:
                provider_mapping = True

        prepared = []
        for circuit, label, shape in self._experiments(request):
            qubit_map: list[int] | None = None
            to_translate = circuit
            if transpile:
                pool = physical_qubits or self.placement_qubits(circuit.num_qubits)
                to_translate, qubit_map = self._compile(
                    circuit,
                    label,
                    initial_layout=initial_layout,
                    physical_qubits=pool,
                )
            prepared.append(
                self._translate(to_translate, label, shape, qubit_map=qubit_map)
            )

        # Field names and defaults mirror pyqpanda3's QCloudOptions.
        configuration: dict[str, Any] = {
            "shot": shots,
            "isProbcount": True,
            "amendFlag": 0 if readout_correction is False else 1,
            "use_cz_pattern": False,
            "mappingFlag": bool(provider_mapping),
            "PointLabel": 1,
            "circuitOptimization": provider_optimization is not False,
            "specified_block": [],
        }
        return prepared, configuration

    @staticmethod
    def _experiments(
        request: ExecutionRequest,
    ) -> list[tuple[QuantumCircuit, str, tuple[int, int]]]:
        return list(
            zip(
                request.bound_circuits(),
                request.labels,
                request.experiment_shapes,
                strict=True,
            )
        )

    @staticmethod
    def _translate(
        circuit: QuantumCircuit,
        label: str,
        shape: tuple[int, int],
        *,
        qubit_map: Sequence[int] | None,
    ) -> _Prepared:
        try:
            program = translate(circuit, qubit_map=qubit_map)
        except TranslationError as exc:
            raise TranslationError(f"Experiment {label!r}: {exc}") from exc
        except Exception as exc:
            raise TranslationError(
                f"Failed to translate experiment {label!r}: {exc}"
            ) from exc
        return _Prepared(program, label, shape)

    def _compile(
        self,
        circuit: QuantumCircuit,
        label: str,
        *,
        initial_layout: Sequence[int] | None,
        physical_qubits: Sequence[int],
    ) -> tuple[QuantumCircuit, list[int]]:
        """Route on the chip and return ``(wire circuit, wire → physical)``."""
        from quspect.backends._compile import compile_for_device
        from quspect.transpiler._passes import DecomposePass
        from quspect.transpiler._target import CompilationTarget

        try:
            target = CompilationTarget.from_backend(self)
            routed, layout = compile_for_device(
                target,
                circuit,
                initial_layout=initial_layout,
                physical_qubits=physical_qubits,
            )
            # Lower routing SWAPs into the chip basis.  Decomposition is
            # local to each gate's qubits, so the placement stays valid.
            routed = DecomposePass(target).run(routed)
        except Exception as exc:
            raise TranslationError(
                f"Failed to transpile experiment {label!r}: {exc}"
            ) from exc
        if layout is None and routed.num_qubits <= len(physical_qubits):
            # A pool without coupling edges (e.g. one qubit) is not routed;
            # place wire i on pool qubit i.  _check_routed rejects any
            # two-qubit gate this would leave off the coupling graph.
            layout = list(physical_qubits)[: routed.num_qubits]
        if layout is None or len(layout) < routed.num_qubits:
            raise TranslationError(
                f"Experiment {label!r}: routing produced no physical layout."
            )
        self._check_routed(routed, layout, label)
        return routed, layout

    def _check_routed(
        self, routed: QuantumCircuit, layout: Sequence[int], label: str
    ) -> None:
        """Reject a placement that the chip cannot execute without remapping."""
        edges = {
            (min(a, b), max(a, b)) for a, b in self._chip().get("coupling_map", [])
        }
        available = set(self.available_qubits)
        for instruction in routed.instructions:
            physical = [layout[q] for q in instruction.qubits]
            outside = [q for q in physical if q not in available]
            if outside:
                raise TranslationError(
                    f"Experiment {label!r}: routed onto unavailable qubit(s) {outside}."
                )
            if instruction.is_gate and len(physical) > 2:
                raise TranslationError(
                    f"Experiment {label!r}: {len(physical)}-qubit gate survived "
                    "routing."
                )
            if instruction.is_gate and len(physical) == 2:
                a, b = physical
                if (min(a, b), max(a, b)) not in edges:
                    gate = instruction.gate.name if instruction.gate else "?"
                    raise TranslationError(
                        f"Experiment {label!r}: routed gate {gate} on {physical} "
                        "is not a coupling edge."
                    )

    def __repr__(self) -> str:
        kind = "simulator" if self.is_simulator else f"qubits={self.num_qubits}"
        return f"OriginQBackend(name={self._name!r}, {kind})"
