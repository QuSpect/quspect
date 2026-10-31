"""Provider-neutral execution requests, batches, and job references."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, overload
from urllib.parse import parse_qsl, urlsplit

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.parameter import Parameter
from quspect.simulation.result import SimulationResult


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata and option keys must be non-empty strings.")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True)
class CircuitExperiment:
    """A circuit template, one complete binding, and a stable result label.

    The circuit is snapshotted at construction time. Bindings accept either
    :class:`Parameter` objects or parameter names and are normalized to the
    template's own ``Parameter`` instances.
    """

    circuit: QuantumCircuit
    bindings: Mapping[Parameter | str, float] = field(default_factory=dict)
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError(
                "CircuitExperiment.circuit must be QuantumCircuit, "
                f"got {type(self.circuit).__name__}."
            )

        snapshot = self.circuit.copy()
        parameters_by_name = {
            parameter.name: parameter for parameter in snapshot.parameters
        }
        normalized_by_name: dict[str, float] = {}
        for key, raw_value in self.bindings.items():
            if isinstance(key, Parameter):
                name = key.name
            elif isinstance(key, str) and key:
                name = key
            else:
                raise TypeError(
                    "Experiment binding keys must be Parameter or non-empty str."
                )
            if name in normalized_by_name:
                raise ValueError(f"Duplicate binding for parameter {name!r}.")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"Binding for parameter {name!r} must be a real scalar."
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"Binding for parameter {name!r} must be finite.")
            normalized_by_name[name] = value

        expected = set(parameters_by_name)
        provided = set(normalized_by_name)
        missing = sorted(expected - provided)
        unknown = sorted(provided - expected)
        if missing:
            raise ValueError(f"Experiment binding is missing parameter(s): {missing}.")
        if unknown:
            raise ValueError(
                f"Experiment binding contains unknown parameter(s): {unknown}."
            )

        normalized = {
            parameters_by_name[name]: normalized_by_name[name]
            for name in sorted(normalized_by_name)
        }
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label.strip()
        ):
            raise ValueError("Experiment label must be a non-empty string or None.")

        object.__setattr__(self, "circuit", snapshot)
        object.__setattr__(self, "bindings", MappingProxyType(normalized))
        object.__setattr__(self, "label", self.label.strip() if self.label else None)
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    def bound_circuit(self) -> QuantumCircuit:
        """Return a fresh, fully bound circuit for this experiment."""
        if self.bindings:
            normalized: dict[Parameter, float] = {}
            for parameter, value in self.bindings.items():
                if not isinstance(parameter, Parameter):
                    raise AssertionError("Experiment bindings were not normalized.")
                normalized[parameter] = value
            return self.circuit.bind_parameters(normalized)
        return self.circuit.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(num_qubits, num_clbits)`` for result reconstruction."""
        return (self.circuit.num_qubits, self.circuit.num_clbits)


@dataclass(frozen=True)
class ExecutionRequest:
    """An ordered collection of sampling experiments submitted as one task."""

    experiments: Sequence[CircuitExperiment]
    shots: int = 1024
    result_format: str = "counts"
    options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _labels: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        experiments = tuple(self.experiments)
        if not experiments:
            raise ValueError("ExecutionRequest requires at least one experiment.")
        if not all(isinstance(item, CircuitExperiment) for item in experiments):
            raise TypeError(
                "ExecutionRequest.experiments must contain CircuitExperiment values."
            )
        if isinstance(self.shots, bool) or not isinstance(self.shots, int):
            raise TypeError("ExecutionRequest.shots must be an integer.")
        if self.shots <= 0:
            raise ValueError(
                f"ExecutionRequest.shots must be positive, got {self.shots}."
            )

        result_format = str(self.result_format).strip().lower()
        if result_format not in {"auto", "counts", "memory"}:
            raise ValueError("result_format must be 'auto', 'counts', or 'memory'.")

        labels: list[str] = []
        used: set[str] = set()
        for index, experiment in enumerate(experiments):
            base = experiment.label or experiment.circuit.name or f"experiment-{index}"
            label = base
            suffix = 1
            while label in used:
                label = f"{base}-{suffix}"
                suffix += 1
            used.add(label)
            labels.append(label)

        object.__setattr__(self, "experiments", experiments)
        object.__setattr__(self, "result_format", result_format)
        object.__setattr__(self, "options", _readonly_mapping(self.options))
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))
        object.__setattr__(self, "_labels", tuple(labels))

    @classmethod
    def single(
        cls,
        circuit: QuantumCircuit,
        *,
        bindings: Mapping[Parameter | str, float] | None = None,
        label: str | None = None,
        shots: int = 1024,
        result_format: str = "counts",
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionRequest:
        """Build a request containing one circuit experiment."""
        return cls(
            experiments=(
                CircuitExperiment(
                    circuit,
                    bindings=bindings or {},
                    label=label,
                ),
            ),
            shots=shots,
            result_format=result_format,
            options=options or {},
            metadata=metadata or {},
        )

    @classmethod
    def batch(
        cls,
        circuits: Sequence[QuantumCircuit],
        *,
        labels: Sequence[str] | None = None,
        shots: int = 1024,
        result_format: str = "counts",
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionRequest:
        """Build a request from already-bound circuits."""
        circuit_items = tuple(circuits)
        if labels is not None and len(labels) != len(circuit_items):
            raise ValueError("labels must have the same length as circuits.")
        experiments = tuple(
            CircuitExperiment(
                circuit,
                label=labels[index] if labels is not None else None,
            )
            for index, circuit in enumerate(circuit_items)
        )
        return cls(
            experiments=experiments,
            shots=shots,
            result_format=result_format,
            options=options or {},
            metadata=metadata or {},
        )

    @classmethod
    def sweep(
        cls,
        circuit: QuantumCircuit,
        bindings: Sequence[Mapping[Parameter | str, float]],
        *,
        labels: Sequence[str] | None = None,
        shots: int = 1024,
        result_format: str = "counts",
        options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionRequest:
        """Expand parameter bindings while preserving their input order."""
        binding_items = tuple(bindings)
        if labels is not None and len(labels) != len(binding_items):
            raise ValueError("labels must have the same length as bindings.")
        experiments = tuple(
            CircuitExperiment(
                circuit,
                bindings=binding,
                label=labels[index] if labels is not None else f"point-{index}",
            )
            for index, binding in enumerate(binding_items)
        )
        return cls(
            experiments=experiments,
            shots=shots,
            result_format=result_format,
            options=options or {},
            metadata=metadata or {},
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def experiment_shapes(self) -> tuple[tuple[int, int], ...]:
        return tuple(experiment.shape for experiment in self.experiments)

    def bound_circuits(self) -> tuple[QuantumCircuit, ...]:
        """Materialize bound circuit snapshots in experiment order."""
        return tuple(experiment.bound_circuit() for experiment in self.experiments)

    def __len__(self) -> int:
        return len(self.experiments)


@dataclass(frozen=True)
class ExecutionBatch(Sequence[SimulationResult]):
    """Ordered per-circuit results from one logical execution request."""

    results: Sequence[SimulationResult]
    labels: Sequence[str] = field(default_factory=tuple)
    backend_name: str = ""
    job_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if not all(isinstance(item, SimulationResult) for item in results):
            raise TypeError(
                "ExecutionBatch.results must contain SimulationResult values."
            )
        labels = (
            tuple(self.labels)
            if self.labels
            else tuple(f"experiment-{index}" for index in range(len(results)))
        )
        if len(labels) != len(results):
            raise ValueError(
                "ExecutionBatch labels and results must have equal length."
            )
        if len(set(labels)) != len(labels):
            raise ValueError("ExecutionBatch labels must be unique.")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))

    def __len__(self) -> int:
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> SimulationResult: ...

    @overload
    def __getitem__(self, index: slice) -> ExecutionBatch: ...

    def __getitem__(self, index: int | slice) -> SimulationResult | ExecutionBatch:
        if isinstance(index, slice):
            return ExecutionBatch(
                results=self.results[index],
                labels=self.labels[index],
                backend_name=self.backend_name,
                job_id=self.job_id,
                metadata=self.metadata,
            )
        return self.results[index]

    def __iter__(self) -> Iterator[SimulationResult]:
        return iter(self.results)

    def single(self) -> SimulationResult:
        """Return the only result, rejecting accidental batch truncation."""
        if len(self.results) != 1:
            raise ValueError(
                f"ExecutionBatch.single() requires one result, got {len(self.results)}."
            )
        return self.results[0]

    def by_label(self, label: str) -> SimulationResult:
        """Return the result associated with a stable experiment label."""
        try:
            return self.results[self.labels.index(label)]
        except ValueError as exc:
            raise KeyError(f"Unknown execution label {label!r}.") from exc

    @property
    def success(self) -> bool:
        return bool(self.results) and all(result.success for result in self.results)

    @property
    def failed_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, result in enumerate(self.results) if not result.success
        )

    def get_counts(self) -> list[dict[str, int]]:
        return [dict(result.counts) for result in self.results]


@dataclass(frozen=True)
class JobReference:
    """Credential-free data required to recover a submitted job."""

    platform: str
    backend_name: str
    job_id: str
    base_url: str | None = None
    transport: str = "async"
    batch_size: int | None = None
    labels: Sequence[str] = field(default_factory=tuple)
    experiment_shapes: Sequence[tuple[int, int]] = field(default_factory=tuple)
    shots: int | None = None
    result_format: str | None = None
    workload_id: str | None = None
    child_job_ids: Sequence[str] = field(default_factory=tuple)
    child_batch_sizes: Sequence[int] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("platform", "backend_name", "job_id", "transport"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"JobReference.{field_name} must be non-empty.")
            object.__setattr__(self, field_name, value.strip())
        if self.transport not in {"async", "sync", "in-process"}:
            raise ValueError(
                "JobReference.transport must be 'async', 'sync', or 'in-process'."
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("JobReference.batch_size must be positive or None.")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("JobReference.shots must be positive or None.")
        if self.result_format is not None:
            result_format = str(self.result_format).strip().lower()
            if result_format not in {"auto", "counts", "memory"}:
                raise ValueError(
                    "JobReference.result_format must be 'auto', 'counts', "
                    "'memory', or None."
                )
            object.__setattr__(self, "result_format", result_format)
        if self.workload_id is not None:
            if not isinstance(self.workload_id, str) or not self.workload_id.strip():
                raise ValueError(
                    "JobReference.workload_id must be a non-empty string or None."
                )
            object.__setattr__(self, "workload_id", self.workload_id.strip())
        if self.base_url:
            parsed_url = urlsplit(self.base_url)
            if parsed_url.username or parsed_url.password:
                raise ValueError("JobReference.base_url must not contain credentials.")
            sensitive_query_keys = {
                "api_key",
                "apikey",
                "x_api_key",
                "token",
                "access_token",
                "authorization",
                "password",
                "secret",
            }
            query_keys = {
                key.lower().replace("-", "_") for key, _ in parse_qsl(parsed_url.query)
            }
            if sensitive_query_keys & query_keys:
                raise ValueError("JobReference.base_url must not contain credentials.")

        raw_labels = tuple(self.labels)
        if any(not isinstance(label, str) or not label.strip() for label in raw_labels):
            raise ValueError("JobReference labels must be non-empty strings.")
        labels = tuple(label.strip() for label in raw_labels)
        if len(set(labels)) != len(labels):
            raise ValueError("JobReference labels must be unique.")
        shapes = tuple((int(nq), int(nc)) for nq, nc in self.experiment_shapes)
        child_job_ids = tuple(str(job_id) for job_id in self.child_job_ids)
        child_batch_sizes = tuple(int(size) for size in self.child_batch_sizes)
        if self.batch_size is not None:
            if labels and len(labels) != self.batch_size:
                raise ValueError("JobReference labels do not match batch_size.")
            if shapes and len(shapes) != self.batch_size:
                raise ValueError(
                    "JobReference experiment_shapes do not match batch_size."
                )
        if any(nq < 0 or nc < 0 for nq, nc in shapes):
            raise ValueError("JobReference experiment shapes cannot be negative.")
        if any(not job_id for job_id in child_job_ids):
            raise ValueError("JobReference child job IDs must be non-empty.")
        if child_batch_sizes and len(child_batch_sizes) != len(child_job_ids):
            raise ValueError("child_batch_sizes must align with child_job_ids.")
        if any(size <= 0 for size in child_batch_sizes):
            raise ValueError("JobReference child batch sizes must be positive.")
        if (
            child_batch_sizes
            and self.batch_size is not None
            and sum(child_batch_sizes) != self.batch_size
        ):
            raise ValueError("Child batch sizes do not add up to batch_size.")

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "experiment_shapes", shapes)
        object.__setattr__(self, "child_job_ids", child_job_ids)
        object.__setattr__(self, "child_batch_sizes", child_batch_sizes)

    @property
    def is_batch(self) -> bool:
        """Whether result retrieval must preserve a batch container."""
        return self.batch_size is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation containing no credentials."""
        return {
            "version": 1,
            "platform": self.platform,
            "backend_name": self.backend_name,
            "job_id": self.job_id,
            "base_url": self.base_url,
            "transport": self.transport,
            "batch_size": self.batch_size,
            "labels": list(self.labels),
            "experiment_shapes": [list(shape) for shape in self.experiment_shapes],
            "shots": self.shots,
            "result_format": self.result_format,
            "workload_id": self.workload_id,
            "child_job_ids": list(self.child_job_ids),
            "child_batch_sizes": list(self.child_batch_sizes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JobReference:
        """Restore a reference created by :meth:`to_dict`."""
        if data.get("version", 1) != 1:
            raise ValueError(
                f"Unsupported JobReference version {data.get('version')!r}."
            )
        return cls(
            platform=data["platform"],
            backend_name=data["backend_name"],
            job_id=data["job_id"],
            base_url=data.get("base_url"),
            transport=data.get("transport", "async"),
            batch_size=data.get("batch_size"),
            labels=data.get("labels", ()),
            experiment_shapes=data.get("experiment_shapes", ()),
            shots=data.get("shots"),
            result_format=data.get("result_format"),
            workload_id=data.get("workload_id"),
            child_job_ids=data.get("child_job_ids", ()),
            child_batch_sizes=data.get("child_batch_sizes", ()),
        )


@dataclass(frozen=True)
class BackendCapabilities:
    """Execution features advertised by a backend."""

    supports_sampling: bool = True
    supports_batch: bool = False
    supports_estimator: bool = False
    supports_statevector: bool = False
    supports_job_recovery: bool = False
    supports_cancel: bool = False
    max_circuits_per_job: int | None = 1
    max_shots: int | None = None
    supported_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"circuit"})
    )
    result_formats: frozenset[str] = field(
        default_factory=lambda: frozenset({"counts"})
    )
    measurement_types: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.max_circuits_per_job is not None and self.max_circuits_per_job <= 0:
            raise ValueError("max_circuits_per_job must be positive or None.")
        if self.max_shots is not None and self.max_shots <= 0:
            raise ValueError("max_shots must be positive or None.")
        object.__setattr__(self, "supported_modes", frozenset(self.supported_modes))
        object.__setattr__(self, "result_formats", frozenset(self.result_formats))
        object.__setattr__(self, "measurement_types", frozenset(self.measurement_types))
