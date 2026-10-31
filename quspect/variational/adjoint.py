"""Native adjoint engine for compiled observable objectives."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import DTypeLike

from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.ir.compilation import decompose_deep
from quspect.ir.qsal import dumps
from quspect.native import NativeExecutor, _NativeAdjointPlan
from quspect.variational.objective import ObservableObjective
from quspect.variational.result import FloatArray, ObjectiveEvaluation, ObjectiveSweep

ParameterPoint: TypeAlias = Sequence[float] | Mapping[str | Parameter, float]


class AdjointEngine:
    """Compile differentiable Pauli objectives for the native state vector.

    ``AdjointEngine`` is a differentiation strategy, not a mutable simulator.
    Compilation snapshots an :class:`ObservableObjective`; subsequent
    evaluations transfer only parameter values across the Python/C++ boundary.
    """

    def __init__(self, *, dtype: DTypeLike = "float64") -> None:
        self._numpy_dtype, precision = _normalize_dtype(dtype)
        self._dtype = self._numpy_dtype.name
        self._native = NativeExecutor(precision=precision)

    @property
    def dtype(self) -> str:
        """State-vector scalar type: ``"float32"`` or ``"float64"``."""
        return self._dtype

    def compile(self, objective: ObservableObjective) -> PreparedObjective:
        """Validate and snapshot an objective into a reusable native plan.

        Args:
            objective: The observable objective to compile.

        Returns:
            A :class:`PreparedObjective` reusable across evaluations.

        Raises:
            TypeError: If ``objective`` is not an ``ObservableObjective`` or an
                observable coefficient is not numeric.
            ValueError: If the objective is unsupported (measurements, noise,
                unsupported gates, invalid observables) or a gate angle, an
                affine offset or an affine coefficient is not finite.
        """
        if not isinstance(objective, ObservableObjective):
            raise TypeError(
                "AdjointEngine.compile expects ObservableObjective, "
                f"got {type(objective).__name__}."
            )
        circuit = decompose_deep(
            objective.circuit,
            supported_gates={
                "id",
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdg",
                "t",
                "tdg",
                "cx",
                "cz",
                "swap",
                "ccx",
                "rx",
                "ry",
                "rz",
            },
        )
        objective = ObservableObjective(circuit, objective.observables)
        _validate_objective(objective)

        working_memory_mb = (len(objective.observables) + 1) * _estimate_memory(
            circuit.num_qubits, self._numpy_dtype
        )
        if working_memory_mb >= 256.0:
            warnings.warn(
                "Adjoint differentiation requires one primal state vector plus "
                f"one costate per observable (approximately {working_memory_mb:.0f} "
                "MB for this objective).",
                ResourceWarning,
                stacklevel=2,
            )

        qsal_json = dumps(circuit, observables=list(objective.observables.items()))
        native_plan = self._native._prepare_adjoint(qsal_json)
        return PreparedObjective(
            native_plan,
            dtype=self._dtype,
            numpy_dtype=self._numpy_dtype,
        )


class PreparedObjective:
    """An immutable compiled objective with explicit evaluation methods."""

    def __init__(
        self,
        native_plan: _NativeAdjointPlan,
        *,
        dtype: str,
        numpy_dtype: np.dtype[Any],
    ) -> None:
        self._native_plan = native_plan
        self._dtype = dtype
        self._numpy_dtype = numpy_dtype
        self._parameter_order = tuple(
            Parameter(name) for name in native_plan.parameter_names
        )
        self._observable_order = native_plan.observable_names
        self._metadata: dict[str, str] = {
            "backend": "native-statevector",
            "differentiation": "adjoint",
            "dtype": dtype,
        }

    @property
    def dtype(self) -> str:
        """State-vector scalar type used by the compiled plan."""
        return self._dtype

    @property
    def parameter_order(self) -> tuple[Parameter, ...]:
        """Parameter axis order used for sequences and Jacobian columns."""
        return self._parameter_order

    @property
    def observable_order(self) -> tuple[str, ...]:
        """Observable axis order used by values and Jacobian rows."""
        return self._observable_order

    @property
    def metadata(self) -> Mapping[str, str]:
        """Execution and differentiation metadata."""
        return dict(self._metadata)

    def evaluate(self, point: ParameterPoint) -> ObjectiveEvaluation:
        """Evaluate all observables and their Jacobian at one point."""
        bindings = _coerce_point(point, self._parameter_order)
        native = self._native_plan.evaluate(bindings)
        return ObjectiveEvaluation(
            values=np.asarray(native.values, dtype=self._numpy_dtype),
            jacobian=np.asarray(native.jacobian, dtype=self._numpy_dtype),
            parameter_order=self._parameter_order,
            observable_order=self._observable_order,
            metadata=self._metadata,
        )

    def evaluate_many(
        self,
        points: Sequence[ParameterPoint] | Sequence[Sequence[float]] | np.ndarray,
    ) -> ObjectiveSweep:
        """Evaluate a parameter sweep with leading shape ``(num_points,)``."""
        normalized = _coerce_sweep(points, len(self._parameter_order))
        evaluations = [self.evaluate(point) for point in normalized]
        if evaluations:
            values = np.stack([result.values for result in evaluations])
            jacobian = np.stack([result.jacobian for result in evaluations])
        else:
            values = np.empty((0, len(self._observable_order)), dtype=self._numpy_dtype)
            jacobian = np.empty(
                (0, len(self._observable_order), len(self._parameter_order)),
                dtype=self._numpy_dtype,
            )
        return ObjectiveSweep(
            values=cast(FloatArray, values),
            jacobian=cast(FloatArray, jacobian),
            parameter_order=self._parameter_order,
            observable_order=self._observable_order,
            metadata=self._metadata,
        )


def _normalize_dtype(dtype: DTypeLike) -> tuple[np.dtype[Any], str]:
    try:
        normalized = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"Unsupported state-vector dtype {dtype!r}.") from exc
    if normalized == np.dtype(np.float32):
        return normalized, "single"
    if normalized == np.dtype(np.float64):
        return normalized, "double"
    raise ValueError("AdjointEngine dtype must be float32 or float64.")


def _coerce_point(
    point: ParameterPoint,
    parameter_order: tuple[Parameter, ...],
) -> dict[str, float]:
    expected_names = tuple(parameter.name for parameter in parameter_order)
    if isinstance(point, Mapping):
        bindings: dict[str, float] = {}
        for key, raw_value in point.items():
            if isinstance(key, Parameter):
                name = key.name
            elif isinstance(key, str):
                name = key
            else:
                raise TypeError(
                    "Parameter binding keys must be Parameter or str, "
                    f"got {type(key).__name__}."
                )
            if name in bindings:
                raise ValueError(f"Duplicate binding for parameter {name!r}.")
            bindings[name] = float(raw_value)

        missing = [name for name in expected_names if name not in bindings]
        unknown = [name for name in bindings if name not in expected_names]
        if missing:
            raise ValueError(f"Parameter binding is missing: {missing}.")
        if unknown:
            raise ValueError(f"Parameter binding contains unknown names: {unknown}.")
    else:
        array = np.asarray(point, dtype=float)
        if array.ndim != 1 or array.size != len(expected_names):
            raise ValueError(
                f"Objective expected {len(expected_names)} parameter value(s), "
                f"got shape {array.shape}."
            )
        bindings = dict(zip(expected_names, array.tolist(), strict=True))

    if not all(math.isfinite(value) for value in bindings.values()):
        raise ValueError("Parameter values must be finite.")
    return bindings


def _coerce_sweep(
    points: Sequence[ParameterPoint] | Sequence[Sequence[float]] | np.ndarray,
    num_parameters: int,
) -> list[ParameterPoint]:
    if isinstance(points, np.ndarray):
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != num_parameters:
            raise ValueError(
                f"Parameter sweep expected shape (B, {num_parameters}), "
                f"got {array.shape}."
            )
        return list(array)

    items = list(points)
    if not items:
        return []
    if all(isinstance(item, Mapping) for item in items):
        return items

    array = np.asarray(items, dtype=float)
    if array.ndim != 2 or array.shape[1] != num_parameters:
        raise ValueError(
            f"Parameter sweep expected shape (B, {num_parameters}), got {array.shape}."
        )
    return list(array)


def _validate_objective(objective: ObservableObjective) -> None:
    circuit = objective.circuit
    if circuit.num_qubits == 0:
        raise ValueError("Objective circuit must contain at least one qubit.")
    if any(not parameter.name for parameter in circuit.parameters):
        raise ValueError("Objective parameter names must be non-empty.")

    supported_gates = {
        "id",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "cnot",
        "cz",
        "swap",
        "ccx",
        "ccnot",
        "rx",
        "ry",
        "rz",
    }
    differentiable_gates = {"rx", "ry", "rz"}
    for instruction in circuit.instructions:
        if instruction.is_measure:
            raise ValueError("Adjoint objectives cannot contain measurements.")
        if instruction.is_channel:
            raise ValueError("Adjoint objectives cannot contain noise channels.")
        if instruction.is_gate and instruction.gate is not None:
            gate = instruction.gate
            name = gate.name.lower()
            if name not in supported_gates:
                raise ValueError(
                    "The adjoint engine supports fixed native unitary gates plus "
                    f"parameterized RX, RY, and RZ; got {gate.name!r}."
                )
            if name not in differentiable_gates and any(
                isinstance(parameter, (Parameter, ParameterExpression))
                for parameter in gate.params
            ):
                raise ValueError(
                    "Only RX, RY, and RZ may contain symbolic parameters in an "
                    "adjoint objective."
                )
            for value in gate.params:
                _validate_finite_gate_parameter(gate.name, value)

    for observable_name, operator in objective.observables.items():
        if operator.is_zero:
            raise ValueError(
                f"Observable {observable_name!r} must contain a Pauli term."
            )
        for term, coefficient in operator:
            invalid_qubits = [
                qubit for qubit, _ in term if qubit < 0 or qubit >= circuit.num_qubits
            ]
            if invalid_qubits:
                raise ValueError(
                    f"Observable {observable_name!r} contains qubits outside the "
                    f"circuit: {invalid_qubits}."
                )
            if not isinstance(coefficient, (int, float, complex, np.number)):
                raise TypeError(
                    f"Observable {observable_name!r} coefficients must be numeric "
                    "and real."
                )
            numeric = complex(coefficient)
            if not (math.isfinite(numeric.real) and math.isfinite(numeric.imag)):
                raise ValueError(
                    f"Observable {observable_name!r} coefficients must be finite."
                )
            if abs(numeric.imag) > 1e-14:
                raise ValueError(
                    f"Observable {observable_name!r} coefficients must be real."
                )


def _validate_finite_gate_parameter(gate_name: str, value: object) -> None:
    """Reject non-finite numeric angles and affine constants/coefficients.

    Mirrors the checks performed by ``SamplingEngine`` so both engines raise
    a ``ValueError`` (rather than a native parser error) for ``inf``/``nan``.
    """
    if isinstance(value, Parameter):
        return
    if isinstance(value, ParameterExpression):
        if not math.isfinite(value.constant):
            label = "constant" if value.is_constant else "offset"
            raise ValueError(f"Gate {gate_name!r} has a non-finite {label}.")
        for raw_coefficient in value.terms.values():
            if not math.isfinite(float(raw_coefficient)):
                raise ValueError(
                    f"Gate {gate_name!r} has a non-finite affine coefficient."
                )
        return
    is_number = isinstance(value, (int, float, np.number)) and not isinstance(
        value, bool
    )
    if is_number and not math.isfinite(float(value)):  # type: ignore[arg-type]
        raise ValueError(f"Gate {gate_name!r} has a non-finite angle.")


def _estimate_memory(num_qubits: int, dtype: np.dtype[Any]) -> float:
    complex_bytes = 8 if dtype == np.dtype(np.float32) else 16
    return (1 << num_qubits) * complex_bytes / (1024 * 1024)
