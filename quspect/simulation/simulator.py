"""Quantum circuit simulator.

The :class:`Simulator` class orchestrates quantum circuit execution
using the C++ native simulation backend for high-performance state
vector evolution, projective measurement with state collapse, and
shot-based sampling.
"""

from __future__ import annotations

import logging
import secrets
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from quspect.backends._base import Backend
from quspect.ir.qsal import dumps
from quspect.native import NativeExecutor
from quspect.simulation.result import SimulationResult

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)

# Warn when the state vector exceeds ~256 MB
_LARGE_N_WARN = 25


class Simulator(Backend):
    """Orchestrates quantum circuit execution using the C++ native backend.

    The simulator takes a :class:`~quspect.circuit.QuantumCircuit` with
    all parameters bound to concrete float values, serializes it to QSAL
    JSON, and delegates execution to the optimized C++ state-vector engine.

    Key behaviors:
        - **Empty circuit**: Returns a result with 0 shots.
        - **No measurements**: Evolves gates once and returns the final
          state vector (no sampling).
        - **Terminal measurements**: Evolves gates once, then samples
          from the final state vector.
        - **Mid-circuit measurements**: Evolves the full circuit
          independently for each shot, collapsing the state at each
          measurement.
        - **Noise channels**: Evolves one complete Monte Carlo trajectory
          per shot with a persistent random stream for that trajectory.

    Attributes:
        seed: The random seed used, or None.

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.simulation import Simulator
        >>> circ = QuantumCircuit(name="bell")
        >>> circ.h(0).cx(0, 1).measure_all()
        >>> sim = Simulator(seed=42)
        >>> result = sim.run(circ, shots=1000)
        >>> result.most_frequent()
        ('00', ...)
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Initialize the simulator.

        Args:
            seed: Optional seed for the random number generator.
                Using the same seed guarantees reproducible results
                across runs. If None, uses system entropy.

        Example:
            >>> sim = Simulator(seed=42)
            >>> sim2 = Simulator(seed=42)
            >>> # Both produce identical results for the same circuit
        """
        self.seed = seed
        native_seed = seed if seed is not None else secrets.randbits(32)
        self._native = NativeExecutor(precision="double", seed=native_seed)

    @property
    def name(self) -> str:
        return "simulator"

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        shots: int = 1024,
        save_statevector: bool = False,
        **options: Any,
    ) -> SimulationResult:
        """Execute the circuit and return measurement results.

        Delegates to the C++ native simulation backend for all execution
        paths (no measurements, terminal measurements, mid-circuit
        measurements).

        Args:
            circuit: The circuit to execute. All parameters must be
                bound to concrete float values.
            shots: Number of repetitions for measurement sampling.
                Must be >= 0. If 0 and there are no measurements,
                returns the state vector without sampling.
            save_statevector: If True, include the final
                numpy complex array in the result.
                Note: for circuits with noise or mid-circuit measurement,
                this is the final state vector from the last trajectory,
                not an ensemble density matrix.

        Returns:
            A :class:`~quspect.simulation.SimulationResult` with
            counts and probabilities.

        Raises:
            ValueError: If the circuit contains unbound symbolic
                parameters.
            ValueError: If ``shots`` is negative.

        Example:
            >>> from quspect import QuantumCircuit
            >>> from quspect.simulation import Simulator
            >>> circ = QuantumCircuit().h(0).cx(0, 1).measure_all()
            >>> sim = Simulator(seed=42)
            >>> result = sim.run(circ, shots=1024)
            >>> "00" in result.counts
            True
        """
        if shots < 0:
            raise ValueError(f"shots must be >= 0, got {shots}")

        self._validate_circuit(circuit)

        n = circuit.num_qubits
        if n >= _LARGE_N_WARN:
            warnings.warn(
                f"Simulating {n} qubits requires a state vector of "
                f"size 2^{n} = {1 << n} ({_estimate_memory(n):.0f} MB). "
                f"Simulation may be slow or run out of memory.",
                ResourceWarning,
                stacklevel=2,
            )

        instructions = circuit.instructions
        if len(instructions) == 0:
            sv = None
            if n > 0 and save_statevector:
                sv = np.zeros(1 << n, dtype=complex)
                sv[0] = 1.0 + 0j
            return SimulationResult(
                counts={},
                probabilities={},
                num_shots=0,
                num_qubits=n,
                num_clbits=0,
                statevector=sv,
            )

        # 无测量线路：不采样，num_shots 始终为 0
        has_measurements = any(instr.is_measure for instr in instructions)
        cpp_shots = shots if has_measurements else 0
        result_shots = shots if has_measurements else 0

        qsal_json = dumps(circuit)
        cpp_result = self._native.run(
            qsal_json,
            shots=cpp_shots,
            return_statevector=save_statevector,
        )

        return SimulationResult.from_native(
            cpp_result,
            circuit=circuit,
            shots=result_shots,
            save_statevector=save_statevector,
        )

    def statevector(self, circuit: QuantumCircuit) -> NDArray[np.complex128]:
        """Evolve the circuit and return the final state vector.

        Delegates to the C++ native backend, which processes gates
        and skips measurement sampling. Terminal measurements are ignored
        (the pre-measurement state is returned). An empty (0-qubit)
        circuit returns the 0-qubit state ``[1+0j]``.

        Args:
            circuit: The circuit to execute. All parameters must be
                bound to concrete float values.

        Returns:
            A complex numpy array of shape ``(2^n,)`` representing
            the final quantum state.

        Raises:
            ValueError: If the circuit contains unbound symbolic
                parameters.
            ValueError: If the circuit contains a mid-circuit measurement
                (a gate or noise channel after a measurement), because the
                final state then depends on random measurement outcomes.
                Use :meth:`run` with ``save_statevector=True`` instead.

        Example:
            >>> from quspect import QuantumCircuit
            >>> from quspect.simulation import Simulator
            >>> circ = QuantumCircuit().h(0).cx(0, 1)
            >>> sim = Simulator()
            >>> sv = sim.statevector(circ)
            >>> sv.shape
            (4,)
            >>> abs(sv[0])  # Bell state |00> amplitude ≈ 1/√2
            0.707...
        """
        self._validate_circuit(circuit)
        if circuit.num_qubits == 0:
            return np.ones(1, dtype=np.complex128)
        if _has_mid_circuit_measurement(circuit):
            raise ValueError(
                "statevector() does not support mid-circuit measurements: the "
                "final state depends on random measurement outcomes. Use "
                "run(..., save_statevector=True) to obtain one sampled "
                "post-measurement state."
            )
        qsal_json = dumps(circuit)
        cpp_result = self._native.run(qsal_json, shots=0, return_statevector=True)
        return np.array(cpp_result.statevector_f64, dtype=np.complex128)

    # -- private helpers --

    def _validate_circuit(self, circuit: QuantumCircuit) -> None:
        """Check that all gate parameters are concrete floats.

        Args:
            circuit: The circuit to validate.

        Raises:
            ValueError: If any gate has symbolic parameters.
        """
        from quspect.circuit.parameter import Parameter, ParameterExpression

        for instr in circuit.instructions:
            if (
                instr.is_gate
                and instr.gate is not None
                and instr.gate.params
                and any(
                    isinstance(p, Parameter)
                    or (isinstance(p, ParameterExpression) and not p.is_constant)
                    for p in instr.gate.params
                )
            ):
                raise ValueError(
                    f"Gate '{instr.gate.name}' has unbound symbolic "
                    f"parameters. Bind all parameters before simulation "
                    f"using circuit.bind_parameters()."
                )


def _has_mid_circuit_measurement(circuit: QuantumCircuit) -> bool:
    """Return True if a gate or channel follows any measurement.

    Barriers and further measurements after a measurement do not make it a
    mid-circuit measurement (they leave the pre-measurement state intact).
    """
    seen_measure = False
    for instr in circuit.instructions:
        if instr.is_measure:
            seen_measure = True
        elif seen_measure and (instr.is_gate or instr.is_channel):
            return True
    return False


def _estimate_memory(num_qubits: int) -> float:
    """Estimate memory usage in MB for a state vector with n qubits.

    Uses complex128 = 16 bytes per amplitude.
    """
    return (1 << num_qubits) * 16 / (1024 * 1024)
