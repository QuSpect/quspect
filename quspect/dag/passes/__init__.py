"""Optimization and transformation passes for the DAG intermediate representation.

Passes operate on :class:`DAGCircuit` instances and implement the
``cancel → commute → fuse → diagonal-remove`` pipeline:

    1. :class:`GateCancellation` — remove inverse gate pairs
    2. :class:`CommutationAnalysis` — reorder to enable more cancellations
    3. :class:`RotationMerging` — merge same-axis rotations
    4. :class:`SingleQubitGateFusion` — fuse arbitrary single-qubit gates
    5. :class:`EliminateDiagonalBeforeMeasurement` — remove redundant diagonals
    6. :class:`RemoveIdentityGates` — drop identity rotations such as ``rz(0)``
"""

from quspect.dag.passes.base import BasePass, PassManager, PassStats
from quspect.dag.passes.commutation import CommutationAnalysis
from quspect.dag.passes.diagonal_before_measure import (
    EliminateDiagonalBeforeMeasurement,
)
from quspect.dag.passes.gate_cancellation import GateCancellation
from quspect.dag.passes.identity_removal import RemoveIdentityGates
from quspect.dag.passes.rotation_merging import RotationMerging
from quspect.dag.passes.single_qubit_fusion import SingleQubitGateFusion

__all__ = [
    "BasePass",
    "PassManager",
    "PassStats",
    "GateCancellation",
    "CommutationAnalysis",
    "RotationMerging",
    "SingleQubitGateFusion",
    "EliminateDiagonalBeforeMeasurement",
    "RemoveIdentityGates",
]
