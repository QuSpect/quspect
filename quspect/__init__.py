"""QuSpect — A quantum computing SDK for research and education."""

from quspect.backends import (
    Backend,
    BackendCapabilities,
    CircuitExperiment,
    CloudBackend,
    CloudJob,
    CompositeJob,
    ExecutionBatch,
    ExecutionJob,
    ExecutionRequest,
    JobReference,
    JobStatus,
    PartialSubmissionError,
)
from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction
from quspect.circuit.parameter import Parameter, ParameterExpression, ParamValue
from quspect.dag.passes import (
    CommutationAnalysis,
    GateCancellation,
    PassManager,
    RotationMerging,
)
from quspect.gates.gate import Gate
from quspect.operators.fermion_operator import FermionOperator
from quspect.operators.qubit_operator import QubitOperator
from quspect.simulation.result import SimulationResult
from quspect.simulation.simulator import Simulator
from quspect.variational import (
    AdjointEngine,
    ObjectiveEvaluation,
    ObjectiveJob,
    ObjectiveSweep,
    ObjectiveSweepJob,
    ObservableObjective,
    PreparedObjective,
    PreparedSamplingObjective,
    SamplingEngine,
)

__all__ = [
    # Circuit
    "QuantumCircuit",
    "CircuitInstruction",
    "Parameter",
    "ParameterExpression",
    "ParamValue",
    # Gates
    "Gate",
    # Operators
    "QubitOperator",
    "FermionOperator",
    # DAG / optimization
    "PassManager",
    "GateCancellation",
    "CommutationAnalysis",
    "RotationMerging",
    # Simulation
    "Simulator",
    "SimulationResult",
    # Variational objectives
    "ObservableObjective",
    "AdjointEngine",
    "PreparedObjective",
    "ObjectiveEvaluation",
    "ObjectiveSweep",
    "SamplingEngine",
    "PreparedSamplingObjective",
    "ObjectiveJob",
    "ObjectiveSweepJob",
    # Backends
    "Backend",
    "CloudBackend",
    "CloudJob",
    "CompositeJob",
    "ExecutionJob",
    "JobStatus",
    "CircuitExperiment",
    "ExecutionRequest",
    "ExecutionBatch",
    "JobReference",
    "BackendCapabilities",
    "PartialSubmissionError",
]
