"""Tianyan (天衍) cloud backend over the cloud's HTTP API.

Who compiles and who maps qubits
--------------------------------
The Tianyan cloud does **not** place or route circuits: a ``CZ`` between
uncoupled qubits, or any use of a disabled qubit, is rejected at submission.
QuSpect therefore does everything:

``transpile=True`` (default)
    Lower to the QCIS basis, optimize, and route with SABRE on the
    best-calibrated connected region of the chip (enabled qubits and
    couplers only, ranked by the chip's calibration data), or on
    ``physical_qubits``.  The routed wire-space circuit is placed on
    physical qubits by QuSpect.
``transpile=False``
    Only lower gates to the QCIS basis (a per-gate rewrite that keeps qubit
    indices) and submit with the circuit's own indices as physical qubits.

Simulators (``tianyan_sw`` full amplitude, ``tianyan_tn``, ``tianyan_swn``,
``tianyan_s``) have no topology; circuits are only lowered to QCIS.

Results come back in measurement order and are decoded into QuSpect
classical-bit order (see :mod:`quspect.backends.tianyan._translator`).
Routing never requires un-permuting result bits: measurements keep their
classical bits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quspect.backends._base import (
    CloudBackend,
    CloudJob,
    CompositeJob,
    _request_workload_id,
)
from quspect.backends._compile import select_region
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
from quspect.backends.tianyan._client import TianyanClient
from quspect.backends.tianyan._job import PLATFORM, QUERY_LIMIT, TianyanJob
from quspect.backends.tianyan._translator import QCIS_BASIS, QCISProgram, translate

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.simulation.result import SimulationResult

#: Sampling simulators (``tianyan_sa`` is single-amplitude, not sampling).
SIMULATORS = frozenset({"tianyan_sw", "tianyan_swn", "tianyan_tn", "tianyan_s"})
NON_SAMPLING = frozenset({"tianyan_sa"})
#: Machine ``status`` codes (cqlib): 0 running, 1 calibrating, 2 maintenance,
#: 3 offline.
STATUS_NAMES = {0: "running", 1: "calibrating", 2: "maintenance", 3: "offline"}


def status_name(code: Any) -> str:
    """Human-readable machine status for a provider ``status`` code."""
    try:
        return STATUS_NAMES.get(int(code), "unknown")
    except (TypeError, ValueError):
        return "unknown"


#: Documented batch limit (cqlib ``submit_job``): fewer than 50 circuits with
#: fewer than 15 measured qubits each, at most 100 000 shots.
_MAX_BATCH = min(49, QUERY_LIMIT)
_BATCH_MAX_MEASURED = 14
_MAX_SHOTS = 100_000
_OPTIONS = frozenset({"transpile", "initial_layout", "physical_qubits"})


@dataclass(frozen=True)
class _Prepared:
    program: QCISProgram
    label: str
    shape: tuple[int, int]


def _qubit(name: Any) -> int:
    text = str(name).strip()
    if not text.startswith("Q") or not text[1:].isdigit():
        raise ValueError(f"unexpected qubit name {name!r}")
    return int(text[1:])


def _names(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _table(block: Any) -> dict[str, float]:
    if not isinstance(block, Mapping):
        return {}
    names, values = block.get("qubit_used"), block.get("param_list")
    if not isinstance(names, list) or not isinstance(values, list):
        return {}
    table: dict[str, float] = {}
    for name, value in zip(names, values, strict=False):
        try:
            table[str(name)] = float(value)
        except (TypeError, ValueError):
            continue
    return table


def parse_chip(config: Mapping[str, Any]) -> dict[str, Any]:
    """Topology and calibration from a Tianyan machine configuration.

    Disabled qubits and couplers are excluded.  Qubit fidelity is
    ``(1 - single-gate error) × mean(|0>, |1> readout fidelity)``; edge
    fidelity is ``1 - CZ error`` (errors are given in percent).  Values are
    looked up by the ``qubit_used`` names, never by list position: the
    parameter lists skip disabled qubits and couplers.
    """
    overview = config.get("overview") or {}
    qubits = [_qubit(q) for q in overview.get("qubits", [])]
    disabled_qubits = {_qubit(q) for q in _names(config.get("disabledQubits"))}
    disabled_couplers = _names(config.get("disabledCouplers"))
    available = sorted(set(qubits) - disabled_qubits)
    usable = set(available)

    single = _table(
        ((config.get("qubit") or {}).get("singleQubit") or {}).get("gate error")
    )
    readout = (config.get("readout") or {}).get("readoutArray") or {}
    f0 = _table(readout.get("|0> readout fidelity"))
    f1 = _table(readout.get("|1> readout fidelity"))
    cz = _table(
        ((config.get("twoQubitGate") or {}).get("czGate") or {}).get("gate error")
    )

    qubit_fidelity: dict[int, float] = {}
    for qubit in available:
        name = f"Q{qubit}"
        if name in single or name in f0 or name in f1:
            gate = 1.0 - single.get(name, 0.0) / 100.0
            read = (f0.get(name, 1.0) + f1.get(name, 1.0)) / 2.0
            qubit_fidelity[qubit] = max(0.0, gate) * max(0.0, read)

    coupling: list[list[int]] = []
    edge_fidelity: dict[tuple[int, int], float] = {}
    couplers = overview.get("coupler_map") or {}
    for coupler, pair in couplers.items():
        if coupler in disabled_couplers or not isinstance(pair, list) or len(pair) != 2:
            continue
        a, b = _qubit(pair[0]), _qubit(pair[1])
        if a in usable and b in usable and a != b:
            edge = (min(a, b), max(a, b))
            coupling.append(list(edge))
            if coupler in cz:
                edge_fidelity[edge] = max(0.0, 1.0 - cz[coupler] / 100.0)
    return {
        "num_qubits": (max(qubits) + 1) if qubits else 0,
        "available_qubits": available,
        "coupling_map": sorted(coupling),
        "qubit_fidelity": qubit_fidelity,
        "edge_fidelity": edge_fidelity,
        "calibration_time": config.get("calibrationTime"),
    }


class TianyanBackend(CloudBackend):
    """A Tianyan QPU (e.g. ``tianyan176``) or sampling simulator."""

    def __init__(
        self,
        client: TianyanClient,
        machine: Mapping[str, Any],
        *,
        provider: Any = None,
    ) -> None:
        name = str(machine.get("code") or machine.get("machineName") or "")
        if not name:
            raise ValueError("Tianyan machine entry has no name.")
        if name in NON_SAMPLING:
            raise ValueError(
                f"Tianyan backend {name!r} computes single amplitudes, not samples."
            )
        self._client = client
        self._machine = dict(machine)
        self._name = name
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
        return self._name in SIMULATORS

    @property
    def status(self) -> str:
        """Provider machine status (``running``, ``calibrating``, ...)."""
        return status_name(self._machine.get("status"))

    def _chip(self) -> dict[str, Any]:
        if self.is_simulator:
            return {}
        if self._chip_cache is None:
            try:
                self._chip_cache = parse_chip(self._client.machine_config(self._name))
            except BackendError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"Unexpected Tianyan configuration for {self._name!r}: {exc}"
                ) from exc
        return self._chip_cache

    @property
    def config(self) -> dict[str, Any]:
        if self.is_simulator:
            return {"name": self._name, "kind": "simulator", "topology": {}}
        chip = self._chip()
        return {
            "name": self._name,
            "kind": "qpu",
            "num_qubits": chip["num_qubits"],
            "available_qubits": list(chip["available_qubits"]),
            "calibration_time": chip["calibration_time"],
            "topology": {"coupling_map": [list(e) for e in chip["coupling_map"]]},
        }

    @property
    def num_qubits(self) -> int:
        """Physical qubits on the chip (``0`` for simulators)."""
        return int(self._chip().get("num_qubits", 0))

    @property
    def available_qubits(self) -> list[int]:
        """Enabled physical qubits."""
        return list(self._chip().get("available_qubits", []))

    @property
    def native_gate_set(self) -> set[str]:
        """QuSpect gate names this backend compiles to (the QCIS basis)."""
        return set(QCIS_BASIS)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_sampling=True,
            supports_batch=True,
            supports_statevector=False,
            supports_job_recovery=True,
            supports_cancel=False,
            max_circuits_per_job=_MAX_BATCH,
            max_shots=_MAX_SHOTS,
            supported_modes=frozenset({"circuit"}),
            result_formats=frozenset({"auto", "counts"}),
        )

    def placement_qubits(self, num_qubits: int) -> list[int]:
        """Best-calibrated connected region of ``num_qubits`` enabled qubits."""
        chip = self._chip()
        region = select_region(
            max(1, int(num_qubits)),
            chip["available_qubits"],
            chip["coupling_map"],
            chip["qubit_fidelity"],
            chip["edge_fidelity"],
        )
        if region is None:
            raise TranslationError(
                f"{self._name} has no connected region of {num_qubits} enabled qubits."
            )
        return region

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        timeout: float | None = None,
        poll_interval: float = 5.0,
        **options: Any,
    ) -> SimulationResult:
        """Submit one circuit and block until its result is ready.

        Options: ``transpile`` (bool, default True), ``initial_layout`` and
        ``physical_qubits`` (QPU only).
        """
        options.pop("save_statevector", None)
        job = self.submit(circuit, shots=shots, **options)
        result = job.result(timeout=timeout, poll_interval=poll_interval)
        if isinstance(result, ExecutionBatch):
            return result.single()
        return result

    def submit(
        self, circuit: QuantumCircuit, *, shots: int = 1024, **options: Any
    ) -> CloudJob:
        """Submit one circuit and return a job that yields one result."""
        from quspect.circuit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                "TianyanBackend.submit expects QuantumCircuit; use "
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
                "TianyanBackend.submit_request expects ExecutionRequest, "
                f"got {type(request).__name__}."
            )
        return self._submit(request, single=False)

    def _groups(self, prepared: Sequence[_Prepared]) -> list[list[_Prepared]]:
        """Consecutive runs that one native batch may hold.

        A batch must measure the same qubits in every program, hold fewer
        than 50 programs, and measure fewer than 15 qubits per program.
        """
        groups: list[list[_Prepared]] = []
        for item in prepared:
            measured = item.program.measurement.measured_qubits
            current = groups[-1] if groups else None
            if (
                current
                and len(current) < _MAX_BATCH
                and len(measured) <= _BATCH_MAX_MEASURED
                and current[0].program.measurement.measured_qubits == measured
            ):
                current.append(item)
            else:
                groups.append([item])
        return groups

    def _submit(self, request: ExecutionRequest, *, single: bool) -> CloudJob:
        if request.result_format not in self.capabilities.result_formats:
            raise ValueError(
                f"Tianyan does not support result format {request.result_format!r}."
            )
        workload_id = _request_workload_id(request)
        prepared = self._prepare(request)

        jobs: list[CloudJob] = []
        submitted = 0
        try:
            for group in self._groups(prepared):
                try:
                    ids = self._client.submit(
                        [item.program.text for item in group],
                        machine=self._name,
                        shots=request.shots,
                    )
                except BackendError:
                    raise
                except Exception as exc:
                    raise BackendError(
                        f"Failed to submit to Tianyan backend {self._name!r}: {exc}"
                    ) from exc
                reference = JobReference(
                    platform=PLATFORM,
                    backend_name=self._name,
                    job_id=ids[0],
                    base_url=self._client.base_url,
                    transport="async",
                    batch_size=None if single else len(group),
                    labels=tuple(item.label for item in group),
                    experiment_shapes=tuple(item.shape for item in group),
                    shots=request.shots,
                    result_format=request.result_format,
                    workload_id=workload_id,
                    child_job_ids=tuple(ids) if len(ids) > 1 else (),
                    child_batch_sizes=(1,) * len(ids) if len(ids) > 1 else (),
                    measurement_maps=tuple(
                        item.program.measurement.clbits for item in group
                    ),
                )
                jobs.append(
                    TianyanJob(
                        self._client, ids, reference=reference, return_batch=not single
                    )
                )
                submitted += len(group)
        except Exception as exc:
            if jobs:
                raise PartialSubmissionError(
                    f"Tianyan submission failed after {submitted} experiment(s); "
                    "Tianyan tasks cannot be cancelled, the submitted task IDs are "
                    f"on the exception: {exc}",
                    submitted_references=tuple(job.reference for job in jobs),
                    cancellation_failures=[i for job in jobs for i in job.job_ids],
                ) from exc
            if isinstance(exc, (BackendError, TranslationError)):
                raise
            raise BackendError(f"Tianyan submission failed: {exc}") from exc

        if len(jobs) == 1:
            return jobs[0]
        return CompositeJob(jobs)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _prepare(self, request: ExecutionRequest) -> list[_Prepared]:
        shots = request.shots
        if (
            isinstance(shots, bool)
            or not isinstance(shots, int)
            or not (1 <= shots <= _MAX_SHOTS)
        ):
            raise ValueError(f"shots must be an integer in [1, {_MAX_SHOTS}].")
        options = dict(request.options)
        unknown = sorted(set(options) - _OPTIONS)
        if unknown:
            raise ValueError(f"Unsupported option(s) for Tianyan: {unknown}.")
        transpile = options.get("transpile", True)
        if not isinstance(transpile, bool):
            raise TypeError("Tianyan option 'transpile' must be bool.")
        initial_layout = _int_list(options.get("initial_layout"), "initial_layout")
        physical_qubits = _int_list(options.get("physical_qubits"), "physical_qubits")
        if self.is_simulator and (initial_layout or physical_qubits):
            raise ValueError("initial_layout/physical_qubits apply to QPUs only.")
        if not transpile and (initial_layout or physical_qubits):
            raise ValueError(
                "initial_layout/physical_qubits require transpile=True. With "
                "transpile=False write physical qubit indices into the circuit."
            )
        if physical_qubits:
            unavailable = sorted(set(physical_qubits) - set(self.available_qubits))
            if unavailable:
                raise ValueError(
                    f"physical_qubits contains disabled or unknown qubit(s) "
                    f"{unavailable}."
                )

        prepared: list[_Prepared] = []
        for circuit, label, shape in zip(
            request.bound_circuits(),
            request.labels,
            request.experiment_shapes,
            strict=True,
        ):
            qubit_map: list[int] | None = None
            to_translate = circuit
            if self.is_simulator or not transpile:
                to_translate = self._lower(circuit, label)
                if not self.is_simulator:
                    self._check_placement(to_translate, None, label)
            else:
                pool = physical_qubits or (
                    None
                    if initial_layout
                    else self.placement_qubits(circuit.num_qubits)
                )
                to_translate, qubit_map = self._route(
                    circuit, label, initial_layout=initial_layout, pool=pool
                )
            try:
                program = translate(to_translate, qubit_map=qubit_map)
            except TranslationError as exc:
                raise TranslationError(f"Experiment {label!r}: {exc}") from exc
            prepared.append(_Prepared(program, label, shape))
        return prepared

    def _target(self) -> Any:
        from quspect.transpiler._target import CompilationTarget

        if self.is_simulator:
            return CompilationTarget(num_qubits=0, native_gates=set(QCIS_BASIS))
        return CompilationTarget.from_backend(self)

    def _lower(self, circuit: QuantumCircuit, label: str) -> QuantumCircuit:
        """Per-gate lowering into the QCIS basis (qubit indices unchanged)."""
        from quspect.backends._compile import decompose_for_pairwise_routing
        from quspect.transpiler._passes import DecomposePass

        try:
            lowered = decompose_for_pairwise_routing(circuit)
            return DecomposePass(self._target()).run(lowered)
        except Exception as exc:
            raise TranslationError(
                f"Failed to lower experiment {label!r} to QCIS: {exc}"
            ) from exc

    def _route(
        self,
        circuit: QuantumCircuit,
        label: str,
        *,
        initial_layout: Sequence[int] | None,
        pool: Sequence[int] | None,
    ) -> tuple[QuantumCircuit, list[int]]:
        from quspect.backends._compile import compile_for_device
        from quspect.transpiler._passes import DecomposePass

        target = self._target()
        try:
            routed, layout = compile_for_device(
                target,
                circuit,
                initial_layout=initial_layout,
                physical_qubits=pool if pool is not None else self.available_qubits,
            )
            # Lower the routing SWAPs the earlier pipeline leaves in (a
            # circuit compiled by transpile() is already in the QCIS basis).
            routed = DecomposePass(target).run(routed)
        except Exception as exc:
            raise TranslationError(
                f"Failed to transpile experiment {label!r}: {exc}"
            ) from exc
        if layout is None and pool is not None and routed.num_qubits <= len(pool):
            layout = list(pool)[: routed.num_qubits]
        if layout is None or len(layout) < routed.num_qubits:
            raise TranslationError(
                f"Experiment {label!r}: routing produced no physical layout."
            )
        self._check_placement(routed, layout, label)
        return routed, layout

    def _check_placement(
        self,
        circuit: QuantumCircuit,
        layout: Sequence[int] | None,
        label: str,
    ) -> None:
        """Reject what the chip would refuse: disabled qubits, uncoupled CZs."""
        chip = self._chip()
        edges = {tuple(edge) for edge in chip["coupling_map"]}
        available = set(chip["available_qubits"])
        for instruction in circuit.instructions:
            physical = [
                layout[q] if layout is not None else q for q in instruction.qubits
            ]
            outside = [q for q in physical if q not in available]
            if outside:
                raise TranslationError(
                    f"Experiment {label!r} uses disabled or unknown qubit(s) "
                    f"{outside} on {self._name}."
                )
            if instruction.is_gate and len(physical) >= 2:
                if len(physical) > 2:
                    raise TranslationError(
                        f"Experiment {label!r}: {len(physical)}-qubit gate on a QPU."
                    )
                a, b = physical
                if (min(a, b), max(a, b)) not in edges:
                    raise TranslationError(
                        f"Experiment {label!r}: two-qubit gate on uncoupled qubits "
                        f"{physical}; use transpile=True or pick coupled qubits."
                    )

    def __repr__(self) -> str:
        return f"TianyanBackend(name={self._name!r}, status={self.status!r})"


def _int_list(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must be a sequence of integer qubits.")
    return list(value)
