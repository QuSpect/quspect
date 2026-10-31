"""QuSpect transpiler — circuit compilation for target devices."""

from quspect.transpiler._consolidate import ConsolidateBlocksPass, ConsolidationStats
from quspect.transpiler._highlevel import HighLevelOptimizePass, HighLevelStats
from quspect.transpiler._mapping import SABRE, RoutingCandidate
from quspect.transpiler._phase_fold import PhaseFoldingPass, PhaseFoldingStats
from quspect.transpiler._target import CompilationTarget
from quspect.transpiler._three_qubit import ThreeQubitBlockPass, ThreeQubitBlockStats
from quspect.transpiler._transpile import (
    TranspileResult,
    TranspileStats,
    approximation_tolerance,
    transpile,
)
from quspect.transpiler.validation import (
    CompilationConsistencyResult,
    check_compilation_consistency,
    generate_random_circuit,
)

__all__ = [
    "ConsolidateBlocksPass",
    "ConsolidationStats",
    "CompilationTarget",
    "HighLevelOptimizePass",
    "HighLevelStats",
    "PhaseFoldingPass",
    "PhaseFoldingStats",
    "SABRE",
    "ThreeQubitBlockPass",
    "ThreeQubitBlockStats",
    "RoutingCandidate",
    "TranspileResult",
    "TranspileStats",
    "approximation_tolerance",
    "transpile",
    "CompilationConsistencyResult",
    "check_compilation_consistency",
    "generate_random_circuit",
]
