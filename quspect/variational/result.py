"""Structured results returned by prepared observable objectives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from quspect.circuit.parameter import Parameter

FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]


def _readonly_float_array(value: object, *, ndim: int) -> FloatArray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        array = np.asarray(value, dtype=np.float64)
    else:
        array = array.copy()
    if array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional result, got {array.shape}.")
    array.setflags(write=False)
    return cast(FloatArray, array)


def _readonly_metadata(metadata: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(metadata))


def _observable_index(name_or_index: str | int, names: tuple[str, ...]) -> int:
    if isinstance(name_or_index, str):
        try:
            return names.index(name_or_index)
        except ValueError as exc:
            raise KeyError(f"Unknown observable {name_or_index!r}.") from exc
    if isinstance(name_or_index, int):
        if name_or_index < 0 or name_or_index >= len(names):
            raise IndexError(
                f"Observable index {name_or_index} is outside [0, {len(names)})."
            )
        return name_or_index
    raise TypeError("Observable selector must be a name or integer index.")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Expectation values and their parameter Jacobian at one point."""

    values: FloatArray
    jacobian: FloatArray
    parameter_order: tuple[Parameter, ...]
    observable_order: tuple[str, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=1)
        jacobian = _readonly_float_array(self.jacobian, ndim=2)
        expected_shape = (len(self.observable_order), len(self.parameter_order))
        if values.shape != (len(self.observable_order),):
            raise ValueError(
                "Objective values do not match observable_order: "
                f"{values.shape} != {(len(self.observable_order),)}."
            )
        if jacobian.shape != expected_shape:
            raise ValueError(
                "Objective Jacobian does not match its axes: "
                f"{jacobian.shape} != {expected_shape}."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "metadata", _readonly_metadata(self.metadata))

    def value(self, observable: str | int = 0) -> float:
        """Return one named or indexed expectation value."""
        return float(self.values[_observable_index(observable, self.observable_order)])

    def derivative(self, observable: str | int = 0) -> FloatArray:
        """Return one observable's derivative in ``parameter_order``."""
        return cast(
            FloatArray,
            self.jacobian[_observable_index(observable, self.observable_order)],
        )


@dataclass(frozen=True)
class ObjectiveSweep:
    """Array-shaped results for multiple parameter points."""

    values: FloatArray
    jacobian: FloatArray
    parameter_order: tuple[Parameter, ...]
    observable_order: tuple[str, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=2)
        jacobian = _readonly_float_array(self.jacobian, ndim=3)
        if values.shape[1:] != (len(self.observable_order),):
            raise ValueError("Sweep values do not match observable_order.")
        if jacobian.shape != (
            values.shape[0],
            len(self.observable_order),
            len(self.parameter_order),
        ):
            raise ValueError("Sweep Jacobian does not match its declared axes.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "jacobian", jacobian)
        object.__setattr__(self, "metadata", _readonly_metadata(self.metadata))

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def at(self, index: int) -> ObjectiveEvaluation:
        """Extract one point while preserving names and metadata."""
        return ObjectiveEvaluation(
            values=self.values[index],
            jacobian=self.jacobian[index],
            parameter_order=self.parameter_order,
            observable_order=self.observable_order,
            metadata=self.metadata,
        )
