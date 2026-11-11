"""Quafu-SQC superconducting QPU exposed as a QuSpect cloud backend.

Compilation happens in QuSpect: circuits are decomposed into the chip basis
(``h, rx, ry, rz, cz``), optimised, laid out and routed on the calibrated
coupling graph, then submitted with physical qubit indices and the Quafu
server compiler disabled (``options.compiler = None``). The server then runs
the circuit exactly as submitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quspect.backends._base import CloudBackend, CloudJob, CompositeJob
from quspect.backends._compile import (
    estimated_log_fidelity,
    fill_missing_fidelities,
    keep_best,
    placement_regions,
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
from quspect.backends.quafu._client import QuafuClient
from quspect.backends.quafu._job import QuafuJob

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.simulation.result import SimulationResult
    from quspect.transpiler._target import CompilationTarget

SHOTS_QUANTUM = 1024
_SERVER_COMPILERS = frozenset({"quarkcircuit", "qiskit", "qsteed"})
_DD_SEQUENCES = frozenset({"XY4", "CPMG"})
_SUPPORTED_OPTIONS = frozenset(
    {
        "transpile",
        "initial_layout",
        "physical_qubits",
        "readout_correction",
        "dynamic_decoupling",
        "server_compiler",
    }
)


def round_shots(shots: int) -> int:
    """Validate *shots* and round it up to Quafu's 1024-shot granularity."""
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise TypeError("shots must be an integer.")
    if shots < 1:
        raise ValueError(f"shots must be positive, got {shots}.")
    return math.ceil(shots / SHOTS_QUANTUM) * SHOTS_QUANTUM


@dataclass(frozen=True)
class _Prepared:
    qasm: str
    label: str
    shape: tuple[int, int]
    layout: tuple[int, ...] | None


@dataclass(frozen=True)
class _Plan:
    transpile: bool
    initial_layout: list[int] | None
    physical_qubits: list[int] | None
    server_compiler: str | None
    provider_options: dict[str, Any]


class QuafuBackend(CloudBackend):
    """One Quafu chip (e.g. ``"Haituo"``, ``"Baihua"``).

    Args:
        name: Chip name as listed by :meth:`QuafuProvider.list_backends`.
        client: Authenticated :class:`QuafuClient`.
        min_coupler_fidelity: CZ couplers with a calibrated fidelity at or
            below this value are excluded from routing. ``0`` (default)
            drops only uncalibrated/disabled couplers, which the server
            rejects as well.
        chip_info: Pre-fetched ``/task/backendtest`` payload (tests, offline).
    """

    def __init__(
        self,
        name: str,
        client: QuafuClient,
        *,
        provider: Any = None,
        min_coupler_fidelity: float = 0.0,
        chip_info: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Quafu chip name must be a non-empty string.")
        self._name = name.strip()
        self._client = client
        self._provider = provider
        self._min_coupler_fidelity = float(min_coupler_fidelity)
        self._chip_info: dict[str, Any] | None = dict(chip_info) if chip_info else None
        self._config_cache: dict[str, Any] | None = None
        self._region_cache: dict[int, list[list[int]]] = {}

    # ------------------------------------------------------------------
    # Device description
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def platform_name(self) -> str:
        return "quafu"

    @property
    def chip_info(self) -> dict[str, Any]:
        """Raw topology/calibration payload published by Quafu."""
        if self._chip_info is None:
            payload = self._client.get(f"/task/backendtest/{self._name}1", auth=False)
            if not isinstance(payload, Mapping) or "qubits_info" not in payload:
                raise BackendError(
                    f"Quafu returned no chip information for {self._name!r}: "
                    f"{str(payload)[:200]}"
                )
            self._chip_info = dict(payload)
        return self._chip_info

    def refresh_calibration(self) -> None:
        """Drop cached calibration so the next access refetches it."""
        self._chip_info = None
        self._config_cache = None
        self._region_cache = {}

    @property
    def calibration_time(self) -> str | None:
        value = self.chip_info.get("calibration_time")
        return str(value) if value else None

    @property
    def config(self) -> dict[str, Any]:
        """Normalised device configuration (consumed by ``CompilationTarget``)."""
        if self._config_cache is None:
            self._config_cache = self._build_config()
        return dict(self._config_cache)

    def _build_config(self) -> dict[str, Any]:
        info = self.chip_info
        qubits: dict[int, dict[str, Any]] = {}
        for entry in dict(info.get("qubits_info") or {}).values():
            if isinstance(entry, Mapping) and "index" in entry:
                qubits[int(entry["index"])] = dict(entry)
        usable_qubits = {
            index for index, entry in qubits.items() if _positive(entry.get("fidelity"))
        }
        couplers: list[tuple[int, int, float]] = []
        for entry in dict(info.get("couplers_info") or {}).values():
            if not isinstance(entry, Mapping):
                continue
            pair = entry.get("qubits_index")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = int(pair[0]), int(pair[1])
            fidelity = _as_float(entry.get("fidelity"))
            couplers.append((min(a, b), max(a, b), fidelity))
        coupling_map = sorted(
            (a, b)
            for a, b, fidelity in couplers
            if fidelity > self._min_coupler_fidelity
            and a in usable_qubits
            and b in usable_qubits
        )
        num_qubits = max(qubits, default=-1) + 1
        return {
            "name": self._name,
            "num_qubits": num_qubits,
            "native_gate_set": [str(g).lower() for g in info.get("basis_gates", [])],
            "topology": {"coupling_map": [list(edge) for edge in coupling_map]},
            "usable_qubits": sorted(usable_qubits),
            "coupler_fidelity": {(a, b): f for a, b, f in couplers},
            "qubit_fidelity": {
                index: _as_float(entry.get("fidelity"))
                for index, entry in qubits.items()
            },
            "readout_fidelity": {
                index: readout
                for index, entry in qubits.items()
                if (readout := _readout_fidelity(entry)) is not None
            },
            "priority_qubits": [
                [int(q) for q in subset]
                for subset in info.get("priority_qubits", [])
                if isinstance(subset, (list, tuple))
            ],
            "calibration_time": info.get("calibration_time"),
        }

    @property
    def num_qubits(self) -> int:
        return int(self.config["num_qubits"])

    @property
    def native_gate_set(self) -> set[str]:
        return set(self.config["native_gate_set"])

    @property
    def coupling_map(self) -> list[tuple[int, int]]:
        """Usable (calibrated) CZ couplings as sorted physical pairs."""
        edges = self.config["topology"]["coupling_map"]
        return [(int(a), int(b)) for a, b in edges]

    def queue_status(self) -> int | str:
        """Queue length, or a state such as ``"Calibrating"``/``"Offline"``."""
        payload = self._client.get("/task/status/0")
        if not isinstance(payload, Mapping) or self._name not in payload:
            raise BackendError(f"Quafu did not report a status for {self._name!r}.")
        value = payload[self._name]
        return value if isinstance(value, (int, str)) else str(value)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_sampling=True,
            supports_batch=True,
            supports_estimator=False,
            supports_statevector=False,
            supports_job_recovery=True,
            supports_cancel=True,
            max_circuits_per_job=None,
            max_shots=None,
            supported_modes=frozenset({"circuit"}),
            result_formats=frozenset({"auto", "counts"}),
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        transpile: bool = True,
        initial_layout: Sequence[int] | None = None,
        physical_qubits: Sequence[int] | None = None,
        timeout: float | None = None,
        poll_interval: float = 2.0,
        **options: Any,
    ) -> SimulationResult:
        """Submit one circuit and block until its result is ready.

        See :meth:`submit` for the options.
        """
        job = self.submit(
            circuit,
            shots=shots,
            transpile=transpile,
            initial_layout=initial_layout,
            physical_qubits=physical_qubits,
            **options,
        )
        result = job.result(timeout=timeout, poll_interval=poll_interval)
        if isinstance(result, ExecutionBatch):
            return result.single()
        return result

    def submit(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        transpile: bool = True,
        initial_layout: Sequence[int] | None = None,
        physical_qubits: Sequence[int] | None = None,
        **options: Any,
    ) -> CloudJob:
        """Submit one circuit and return a :class:`QuafuJob`.

        Args:
            circuit: Bound circuit with at least one measurement.
            shots: Requested samples; rounded up to a multiple of 1024
                because Quafu only executes whole 1024-shot blocks.
            transpile: ``True`` (default) compiles in QuSpect for this chip.
                ``False`` submits the circuit as is: its qubit indices are
                physical unless ``initial_layout`` maps them, or a
                ``server_compiler`` is chosen.
            initial_layout: ``initial_layout[logical] = physical``. Seeds
                routing when transpiling; applied directly otherwise.
            physical_qubits: Physical qubit pool for layout and routing.
                Defaults to the smallest connected subset in the chip's
                ``priority_qubits`` (Quafu's fidelity-ranked suggestions)
                that holds the circuit; when none does, the circuit is
                placed by the chip's calibration.
            readout_correction: Ask Quafu for readout-error-mitigated
                quasi-probabilities.
            dynamic_decoupling: ``"XY4"`` or ``"CPMG"``.
            server_compiler: ``"quarkcircuit"``, ``"qiskit"`` or ``"qsteed"``
                to let Quafu compile instead of QuSpect (requires
                ``transpile=False``).
        """
        from quspect.circuit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(
                "QuafuBackend.submit expects QuantumCircuit; use "
                "submit_request() for labelled batches."
            )
        if circuit.parameters:
            names = sorted(parameter.name for parameter in circuit.parameters)
            raise TranslationError(
                f"Circuit contains unbound parameter(s): {names}. "
                "Bind them or use ExecutionRequest.sweep()."
            )
        result_format = options.pop("result_format", "auto")
        run_options: dict[str, Any] = {"transpile": transpile, **options}
        if initial_layout is not None:
            run_options["initial_layout"] = list(initial_layout)
        if physical_qubits is not None:
            run_options["physical_qubits"] = list(physical_qubits)
        request = ExecutionRequest.single(
            circuit,
            label=circuit.name or "circuit",
            shots=shots,
            result_format=result_format,
            options=run_options,
        )
        job = self.submit_request(request)
        if isinstance(job, QuafuJob):
            return QuafuJob(self._client, job.reference, return_batch=False)
        return job

    def submit_request(self, request: ExecutionRequest) -> CloudJob:
        """Compile every experiment first, then submit one task per circuit."""
        if not isinstance(request, ExecutionRequest):
            raise TypeError(
                "QuafuBackend.submit_request expects ExecutionRequest, "
                f"got {type(request).__name__}."
            )
        if request.result_format not in self.capabilities.result_formats:
            raise ValueError(
                f"Quafu does not support result format {request.result_format!r}."
            )
        workload_id = _workload_id(request)
        shots = round_shots(request.shots)
        plan = self._plan(request.options)
        prepared = [
            self._prepare(circuit, label, shape, plan)
            for circuit, label, shape in zip(
                request.bound_circuits(),
                request.labels,
                request.experiment_shapes,
                strict=True,
            )
        ]

        jobs: list[QuafuJob] = []
        try:
            for item in prepared:
                tid = self._client.post(
                    "/task/run/",
                    params={
                        "name": _task_name(item.label),
                        "chip": self._name,
                        "shots": shots,
                    },
                    body={
                        "circuit": item.qasm,
                        "compile": True,
                        "options": dict(plan.provider_options),
                    },
                )
                if isinstance(tid, bool) or not isinstance(tid, int):
                    raise BackendError(
                        f"Quafu rejected experiment {item.label!r}: {str(tid)[:300]}"
                    )
                if tid <= 0:
                    # Task ids are positive; /task/status/0 is the chip list.
                    raise BackendError(
                        f"Quafu answered experiment {item.label!r} with an "
                        f"invalid task id: {tid}."
                    )
                reference = JobReference(
                    platform="quafu",
                    backend_name=self._name,
                    job_id=str(tid),
                    base_url=self._client.url,
                    transport="async",
                    batch_size=1,
                    labels=(item.label,),
                    experiment_shapes=(item.shape,),
                    shots=shots,
                    result_format=request.result_format,
                    workload_id=workload_id,
                )
                jobs.append(QuafuJob(self._client, reference, return_batch=True))
        except Exception as exc:
            if jobs:
                failures: list[str] = []
                for submitted in jobs:
                    try:
                        if not submitted.cancel():
                            failures.append(submitted.job_id)
                    except Exception:
                        failures.append(submitted.job_id)
                raise PartialSubmissionError(
                    f"Quafu submission failed after {len(jobs)} experiment(s): {exc}",
                    submitted_references=tuple(job.reference for job in jobs),
                    cancellation_failures=failures,
                ) from exc
            if isinstance(exc, BackendError):
                raise
            raise BackendError(
                f"Failed to submit to Quafu {self._name!r}: {exc}"
            ) from exc

        if len(jobs) == 1:
            return jobs[0]
        return CompositeJob(jobs)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _plan(self, raw: Mapping[str, Any]) -> _Plan:
        options = dict(raw)
        unknown = sorted(set(options) - _SUPPORTED_OPTIONS)
        if unknown:
            raise ValueError(f"Unsupported Quafu option(s): {unknown}.")
        transpile = options.get("transpile", True)
        if not isinstance(transpile, bool):
            raise TypeError("Quafu option 'transpile' must be bool.")
        initial_layout = _int_list(options.get("initial_layout"), "initial_layout")
        physical_qubits = _int_list(options.get("physical_qubits"), "physical_qubits")

        server_compiler = options.get("server_compiler")
        if server_compiler is not None:
            if server_compiler not in _SERVER_COMPILERS:
                raise ValueError(
                    f"server_compiler must be one of {sorted(_SERVER_COMPILERS)}."
                )
            if transpile:
                raise ValueError("server_compiler requires transpile=False.")
            if initial_layout is not None:
                raise ValueError(
                    "initial_layout cannot be combined with server_compiler; "
                    "use physical_qubits to restrict Quafu's layout."
                )
        elif physical_qubits is not None and not transpile:
            raise ValueError(
                "physical_qubits requires transpile=True or a server_compiler."
            )

        provider_options: dict[str, Any] = {"compiler": server_compiler}
        if server_compiler is not None and physical_qubits is not None:
            provider_options["target_qubits"] = list(physical_qubits)
        correction = options.get("readout_correction", False)
        if not isinstance(correction, bool):
            raise TypeError("readout_correction must be bool.")
        if correction:
            provider_options["correct"] = True
        dd = options.get("dynamic_decoupling")
        if dd is not None:
            if dd not in _DD_SEQUENCES:
                raise ValueError(
                    f"dynamic_decoupling must be one of {sorted(_DD_SEQUENCES)}."
                )
            provider_options["open_dd"] = dd
        return _Plan(
            transpile=transpile,
            initial_layout=initial_layout,
            physical_qubits=physical_qubits,
            server_compiler=server_compiler,
            provider_options=provider_options,
        )

    def _prepare(
        self,
        circuit: QuantumCircuit,
        label: str,
        shape: tuple[int, int],
        plan: _Plan,
    ) -> _Prepared:
        from quspect.ir.qasm2 import QASM2ExportConfig, QASM2Exporter, QASM2ExportMode

        if not any(instruction.is_measure for instruction in circuit.instructions):
            raise TranslationError(
                f"Experiment {label!r} has no measurement. "
                "A sampling request sent to quantum hardware must measure."
            )
        _check_measurements(circuit, label)
        try:
            layout: list[int] | None = None
            if plan.transpile:
                physical, layout = self._transpile(
                    circuit,
                    initial_layout=plan.initial_layout,
                    physical_qubits=plan.physical_qubits,
                )
                self._check_physical(physical)
            elif plan.server_compiler is not None:
                physical = circuit
            else:
                physical = circuit
                if plan.initial_layout is not None:
                    layout = list(plan.initial_layout)
                    physical = _relabel(circuit, layout)
                self._check_physical(physical)
            mode = (
                QASM2ExportMode.STRICT if plan.transpile else QASM2ExportMode.DECOMPOSE
            )
            qasm = QASM2Exporter(QASM2ExportConfig(mode=mode)).export(physical)
        except BackendError:
            raise
        except Exception as exc:
            raise TranslationError(
                f"Failed to compile experiment {label!r}: {exc}"
            ) from exc
        return _Prepared(
            qasm=qasm,
            label=label,
            shape=shape,
            layout=tuple(layout) if layout is not None else None,
        )

    def _check_physical(self, circuit: QuantumCircuit) -> None:
        """Reject circuits the server would refuse without its compiler."""
        n = self.num_qubits
        edges = set(self.coupling_map)
        for instruction in circuit:
            if any(q >= n for q in instruction.qubits):
                raise TranslationError(
                    f"Qubit index out of range for {self._name} ({n} qubits): "
                    f"{instruction.qubits}."
                )
            if not instruction.is_gate:
                continue
            if len(instruction.qubits) > 2:
                raise TranslationError(
                    f"Gate {_gate_name(instruction)!r} acts on more than two qubits; "
                    "use transpile=True."
                )
            if len(instruction.qubits) == 2:
                a, b = sorted(instruction.qubits)
                if (a, b) not in edges:
                    raise TranslationError(
                        f"Qubits {a} and {b} are not coupled (or the coupler is "
                        f"uncalibrated) on {self._name}; use transpile=True."
                    )

    def compilation_target(self) -> CompilationTarget:
        from quspect.transpiler._target import CompilationTarget

        return CompilationTarget.from_backend(self)

    def default_physical_qubits(self, num_qubits: int) -> list[int]:
        """Pick a connected, well-calibrated physical pool for *num_qubits*.

        Prefers the smallest of Quafu's fidelity-ranked ``priority_qubits``
        subsets that is large enough and connected on the usable coupling
        graph; falls back to the largest connected usable component.
        Automatic placement does not stop at that fallback: see
        :meth:`placement_pools`.
        """
        suggested = self._suggested_pool(num_qubits)
        if suggested is not None:
            return suggested
        usable = set(self.config["usable_qubits"])
        adjacency = _adjacency(self.coupling_map)
        component = max(_components(usable, adjacency), key=len, default=[])
        if len(component) < num_qubits:
            raise TranslationError(
                f"{self._name} has no connected group of {num_qubits} usable "
                f"qubits (largest: {len(component)})."
            )
        return sorted(component)

    def _suggested_pool(self, num_qubits: int) -> list[int] | None:
        """Smallest usable, connected ``priority_qubits`` group that fits."""
        cfg = self.config
        usable = set(cfg["usable_qubits"])
        adjacency = _adjacency(self.coupling_map)
        candidates = sorted(
            (subset for subset in cfg["priority_qubits"] if len(subset) >= num_qubits),
            key=len,
        )
        for subset in candidates:
            if set(subset) <= usable and _connected(subset, adjacency):
                return list(subset)
        return None

    def placement_pools(self, num_qubits: int) -> list[list[int]]:
        """Physical pools automatic placement compiles a circuit on.

        A fitting ``priority_qubits`` group is the only pool.  When Quafu
        suggests none, routing on :meth:`default_physical_qubits` alone
        would settle on the chip's lowest indices whatever their fidelity,
        so the best-calibrated connected regions of the circuit's size and
        of two qubits more follow it (see
        :func:`~quspect.backends._compile.placement_regions`).  The circuit
        is compiled on each pool and the program with the highest product
        of calibrated gate and readout fidelities is sent; the first pool
        wins ties, so a chip with flat calibration keeps its placement
        unless a region saves gates.
        """
        suggested = self._suggested_pool(num_qubits)
        if suggested is not None:
            return [suggested]
        pools = [self.default_physical_qubits(num_qubits)]
        if num_qubits not in self._region_cache:
            cfg = self.config
            usable = cfg["usable_qubits"]
            readout = fill_missing_fidelities(cfg["readout_fidelity"], usable)
            self._region_cache[num_qubits] = placement_regions(
                num_qubits,
                usable,
                self.coupling_map,
                {
                    q: cfg["qubit_fidelity"].get(q, 1.0) * readout.get(q, 1.0)
                    for q in usable
                },
                cfg["coupler_fidelity"],
            )
        pools += [
            list(region)
            for region in self._region_cache[num_qubits]
            if region != pools[0]
        ]
        return pools

    def _estimated_log_fidelity(self, physical: QuantumCircuit) -> float:
        """Calibration estimate of a physically indexed program."""
        cfg = self.config
        return estimated_log_fidelity(
            physical,
            edge_fidelity=cfg["coupler_fidelity"],
            readout_fidelity=fill_missing_fidelities(
                cfg["readout_fidelity"], cfg["usable_qubits"]
            ),
            gate_fidelity=cfg["qubit_fidelity"],
        )

    def _transpile(
        self,
        circuit: QuantumCircuit,
        *,
        initial_layout: Sequence[int] | None,
        physical_qubits: Sequence[int] | None,
    ) -> tuple[QuantumCircuit, list[int]]:
        """Decompose, optimise and route; return a physically indexed circuit."""
        from quspect.backends._compile import defer_terminal_measurements
        from quspect.ir.compilation import decompose_circuit
        from quspect.transpiler._passes import DecomposePass, OptimizePass

        device = self.compilation_target()
        if not device.native_gates:
            raise TranslationError(f"{self._name} reports no basis gates.")

        # SABRE handles one- and two-qubit gates only.
        wide = {
            _gate_name(instruction)
            for instruction in circuit
            if instruction.is_gate and len(instruction.qubits) > 2
        }
        if wide:
            names = {_gate_name(i) for i in circuit if i.is_gate}
            circuit = decompose_circuit(circuit, supported_gates=names - wide)
        circuit = DecomposePass(device).run(circuit)
        circuit = OptimizePass.from_target(device).run(circuit)
        # Keep terminal measurements terminal: routed after every gate, so no
        # SWAP crosses a measured qubit.
        circuit = defer_terminal_measurements(circuit)

        if physical_qubits is not None:
            pools = [list(physical_qubits)]
        elif initial_layout is not None:
            pools = [self._component_containing(initial_layout)]
        else:
            pools = self.placement_pools(circuit.num_qubits)
        if len(pools) == 1:
            return self._route(device, circuit, pools[0], initial_layout)
        return keep_best(
            (self._route(device, circuit, pool, None) for pool in pools),
            lambda placed: self._estimated_log_fidelity(placed[0]),
        )

    def _route(
        self,
        device: CompilationTarget,
        circuit: QuantumCircuit,
        pool: list[int],
        initial_layout: Sequence[int] | None,
    ) -> tuple[QuantumCircuit, list[int]]:
        """Lay out and route a basis circuit on *pool*; index it physically."""
        from quspect.transpiler._passes import DecomposePass, LayoutRoutingPass

        if circuit.num_qubits > len(pool):
            raise TranslationError(
                f"Circuit uses {circuit.num_qubits} qubits but only {len(pool)} "
                "physical qubits are available for it."
            )
        unusable = sorted(set(pool) - set(self.config["usable_qubits"]))
        if unusable:
            raise TranslationError(
                f"physical_qubits contains uncalibrated qubits: {unusable}."
            )
        target, dense_to_physical = device.induced_subgraph(pool)
        if circuit.num_qubits > 1 and not _connected(
            pool, _adjacency(self.coupling_map)
        ):
            raise TranslationError(
                "physical_qubits must induce a connected coupling subgraph."
            )

        dense_initial = None
        if initial_layout is not None:
            index = {
                physical: dense for dense, physical in enumerate(dense_to_physical)
            }
            outside = [q for q in initial_layout if q not in index]
            if outside:
                raise TranslationError(
                    "initial_layout contains qubits outside the routing pool: "
                    f"{outside}."
                )
            dense_initial = [index[q] for q in initial_layout]

        routed, mapping, _ = LayoutRoutingPass(
            target, iter_num=3, seed=0, initial_mapping=dense_initial
        ).run(circuit)
        # Routing inserts SWAPs; lower them into the chip basis.
        routed = DecomposePass(target).run(routed)
        layout = [dense_to_physical[dense] for dense in mapping[: routed.num_qubits]]
        return _relabel(routed, layout), layout

    def _component_containing(self, qubits: Sequence[int]) -> list[int]:
        adjacency = _adjacency(self.coupling_map)
        usable = set(self.config["usable_qubits"])
        for component in _components(usable, adjacency):
            if set(qubits) <= set(component):
                return sorted(component)
        raise TranslationError(
            f"initial_layout {list(qubits)} is not contained in one connected "
            f"group of usable qubits on {self._name}."
        )

    def __repr__(self) -> str:
        return f"QuafuBackend(name={self._name!r})"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _relabel(circuit: QuantumCircuit, layout: Sequence[int]) -> QuantumCircuit:
    from quspect.circuit.circuit import QuantumCircuit

    if len(layout) < circuit.num_qubits:
        raise TranslationError(
            f"Layout of length {len(layout)} cannot place {circuit.num_qubits} qubits."
        )
    if len(set(layout)) != len(layout):
        raise TranslationError("Layout must not repeat physical qubits.")
    out = QuantumCircuit(name=circuit.name)
    return out.compose(
        circuit,
        qubits=list(layout[: circuit.num_qubits]),
        clbits=list(range(circuit.num_clbits)),
    )


def _gate_name(instruction: Any) -> str:
    gate = instruction.gate
    return str(gate.name) if gate is not None else ""


def _check_measurements(circuit: QuantumCircuit, label: str) -> None:
    seen: set[int] = set()
    for instruction in circuit:
        if instruction.is_measure:
            for clbit in instruction.clbits:
                if clbit in seen:
                    raise TranslationError(
                        f"Experiment {label!r} writes classical bit {clbit} more than "
                        "once; Quafu reports only one value per bit."
                    )
                seen.add(clbit)


def _adjacency(edges: Sequence[tuple[int, int]]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return adjacency


def _connected(qubits: Sequence[int], adjacency: Mapping[int, set[int]]) -> bool:
    nodes = set(qubits)
    if len(nodes) <= 1:
        return True
    start = next(iter(nodes))
    seen, stack = {start}, [start]
    while stack:
        for neighbor in adjacency.get(stack.pop(), ()):
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == nodes


def _components(nodes: set[int], adjacency: Mapping[int, set[int]]) -> list[list[int]]:
    remaining, components = set(nodes), []
    while remaining:
        start = remaining.pop()
        component, stack = [start], [start]
        while stack:
            for neighbor in adjacency.get(stack.pop(), ()):
                if neighbor in remaining:
                    remaining.discard(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _int_list(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must be a sequence of integer qubits.")
    return list(value)


def _workload_id(request: ExecutionRequest) -> str | None:
    value = request.metadata.get("workload_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Execution metadata workload_id must be a non-empty string.")
    return value.strip()


def _task_name(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label)
    return (cleaned or "quspect")[:64]


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _positive(value: Any) -> bool:
    return _as_float(value) > 0.0


def _readout_fidelity(entry: Mapping[str, Any]) -> float | None:
    """Mean of a qubit's two readout fidelities, when Quafu measured both.

    Quafu writes ``0`` for a readout it did not measure.
    """
    ground = _as_float(entry.get("readout g_fidelity"))
    excited = _as_float(entry.get("readout e_fidelity"))
    if 0.0 < ground <= 1.0 and 0.0 < excited <= 1.0:
        return (ground + excited) / 2.0
    return None
