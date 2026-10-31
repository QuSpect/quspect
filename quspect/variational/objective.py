"""Declarative observable objectives for variational algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from quspect.circuit.circuit import QuantumCircuit
from quspect.operators.qubit_operator import QubitOperator


@dataclass(frozen=True)
class ObservableObjective:
    """A circuit together with the named observables to evaluate.

    The mapping order defines the row order of every compiled result.  The
    mapping itself is copied and made read-only so later dictionary mutations
    cannot silently change an objective.  The compiled plan also snapshots the
    circuit and operators in the native runtime.
    """

    circuit: QuantumCircuit
    observables: Mapping[str, QubitOperator]

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError(
                "ObservableObjective.circuit must be QuantumCircuit, "
                f"got {type(self.circuit).__name__}."
            )
        if not isinstance(self.observables, Mapping):
            raise TypeError("ObservableObjective.observables must be a mapping.")

        copied: dict[str, QubitOperator] = {}
        for name, observable in self.observables.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Observable names must be non-empty strings.")
            if not isinstance(observable, QubitOperator):
                raise TypeError(
                    f"Observable {name!r} must be QubitOperator, "
                    f"got {type(observable).__name__}."
                )
            copied[name] = observable

        if not copied:
            raise ValueError("ObservableObjective requires at least one observable.")
        object.__setattr__(self, "observables", MappingProxyType(copied))
