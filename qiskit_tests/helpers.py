"""Shared helpers for cross-validating quspect against qiskit.

Every agent writing tests under ``qiskit_tests/`` should import from this
module instead of re-implementing conversions.  The conventions these helpers
rely on are documented (and enforced by tests) in ``qiskit_tests/conftest.py``
and ``qiskit_tests/test_conventions.py``.  Summary:

* Statevectors: quspect and qiskit are both little-endian (qubit 0 is the
  least-significant bit of the amplitude index).  No permutation needed.
* ``gate_matrix`` (quspect) is big-endian: the gate's first qubit argument is
  the most-significant bit.  ``gate_matrix(g) ==
  Operator(qiskit_gate).reverse_qargs().data``.  The same big-endian
  convention applies to ``NoiseChannel.kraus_operators()``.
* Count bitstrings: quspect ``s[i]`` is clbit ``i`` (clbit 0 leftmost);
  qiskit puts clbit 0 rightmost.  ``quspect_bitstring == qiskit_bitstring[::-1]``.

Statistical tolerance design
----------------------------
All sampled quantities are compared with a Gaussian ``n_sigma`` rule on top of
an exact reference value (usually computed by qiskit ``Statevector`` or
``DensityMatrix``):

* A frequency ``p_hat = k / shots`` with ``k ~ Binomial(shots, p)`` has
  standard deviation ``sqrt(p (1 - p) / shots)``.  With ``N_SIGMA = 5`` the
  two-sided false-alarm rate of a single check is ``2 (1 - Phi(5)) ~= 5.7e-7``;
  even with 32 outcomes and hundreds of tests the union bound keeps the suite
  false-failure probability below ~1e-3 per full run, while a genuine
  convention bug (e.g. a swapped bit order) shifts probabilities by O(1) and
  is caught with certainty for any ``shots >= ~100``.
* When ``p`` is close to 0 or 1 the binomial variance collapses and the
  normal approximation breaks down; the variance is therefore floored at
  ``1 / shots`` (i.e. "at least one count of noise"), which keeps the
  tolerance at ``>= n_sigma / shots``.
* On top of the per-outcome check, :func:`assert_counts_match` runs a Pearson
  chi-square goodness-of-fit test (cells with expected count < 5 pooled, the
  classical validity condition for the chi-square approximation) and requires
  ``p-value > alpha`` (default ``1e-6``).  The per-outcome check catches a
  single badly wrong probability; the chi-square test catches many small
  coherent deviations that individually stay inside ``n_sigma``.
* Mean estimators (e.g. Pauli expectation values from shots) use
  ``|mean - expected| <= n_sigma * std / sqrt(n)`` (central limit theorem).
  For a +/-1 valued observable with expectation ``e``, the single-shot std is
  ``sqrt(1 - e**2)`` (see :func:`pm1_std`).

Statistical tests must always use a fixed seed (``Simulator(seed=...)`` /
``np.random.default_rng(seed)``) and should use 4000-20000 shots: with 4000
shots the 5-sigma resolution on a p=0.5 outcome is ~0.04, with 20000 shots
~0.018.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.circuit.library as qlib
from numpy.typing import NDArray
from qiskit.quantum_info import DensityMatrix as QiskitDensityMatrix
from qiskit.quantum_info import Kraus, Operator, Statevector
from scipy import stats

import quspect
from quspect.circuit.instruction import CircuitInstruction
from quspect.circuit.parameter import Parameter, ParameterExpression
from quspect.gates._matrix_utils import gate_matrix
from quspect.gates.gate import Gate
from quspect.simulation.simulator import Simulator

__all__ = [
    "GATE_SPECS",
    "N_SIGMA",
    "GateSpec",
    "assert_counts_match",
    "assert_mean_close",
    "assert_probability_close",
    "assert_states_equal",
    "assert_unitary_equal",
    "binomial_tolerance",
    "circuit_unitary",
    "global_phase",
    "pm1_std",
    "qiskit_circuit_unitary",
    "qiskit_counts_to_quspect",
    "qiskit_gate_matrix_be",
    "qiskit_probabilities",
    "qiskit_sample_counts",
    "quspect_circuit_unitary",
    "quspect_counts_to_qiskit",
    "quspect_matrix",
    "random_circuit_pair",
    "to_qiskit",
]

N_SIGMA: float = 5.0
"""Default Gaussian tolerance multiplier (two-sided false alarm ~5.7e-7)."""

ComplexArray = NDArray[np.complex128]
QiskitParamValue = Any  # float | qiskit Parameter | qiskit ParameterExpression


# =============================================================================
# Gate correspondence table
# =============================================================================


@dataclass(frozen=True)
class GateSpec:
    """Correspondence between one quspect gate and its qiskit counterpart.

    Attributes:
        name: quspect gate name (``Gate.name`` and fluent method name).
        num_qubits: Number of qubits the gate acts on.
        num_params: Number of real parameters.
        qiskit_factory: Callable taking ``num_params`` parameters and
            returning the qiskit gate with identical matrix (no global phase
            difference) when applied to the same qubit list.
    """

    name: str
    num_qubits: int
    num_params: int
    qiskit_factory: Callable[..., qiskit.circuit.Gate]

    def quspect_gate(self, *params: Any) -> Gate:
        """Build the quspect ``Gate`` value object with ``params``."""
        return Gate(name=self.name, num_qubits=self.num_qubits, params=tuple(params))

    def qiskit_gate(self, *params: QiskitParamValue) -> qiskit.circuit.Gate:
        """Build the corresponding qiskit gate with ``params``."""
        return self.qiskit_factory(*params)

    def apply_quspect(
        self,
        circ: quspect.QuantumCircuit,
        params: Sequence[Any],
        qubits: Sequence[int],
    ) -> None:
        """Apply the gate via the quspect fluent API (``circ.<name>(...)``)."""
        getattr(circ, self.name)(*params, *qubits)

    def apply_qiskit(
        self,
        qc: qiskit.QuantumCircuit,
        params: Sequence[QiskitParamValue],
        qubits: Sequence[int],
    ) -> None:
        """Append the qiskit counterpart to ``qc`` on the same qubit list."""
        qc.append(self.qiskit_gate(*params), list(qubits))


def _spec(
    name: str, nq: int, npar: int, factory: Callable[..., qiskit.circuit.Gate]
) -> GateSpec:
    return GateSpec(name=name, num_qubits=nq, num_params=npar, qiskit_factory=factory)


GATE_SPECS: dict[str, GateSpec] = {
    s.name: s
    for s in [
        _spec("id", 1, 0, qlib.IGate),
        _spec("h", 1, 0, qlib.HGate),
        _spec("x", 1, 0, qlib.XGate),
        _spec("y", 1, 0, qlib.YGate),
        _spec("z", 1, 0, qlib.ZGate),
        _spec("s", 1, 0, qlib.SGate),
        _spec("sdg", 1, 0, qlib.SdgGate),
        _spec("t", 1, 0, qlib.TGate),
        _spec("tdg", 1, 0, qlib.TdgGate),
        _spec("sx", 1, 0, qlib.SXGate),
        _spec("sxdg", 1, 0, qlib.SXdgGate),
        _spec("cx", 2, 0, qlib.CXGate),
        _spec("cz", 2, 0, qlib.CZGate),
        _spec("swap", 2, 0, qlib.SwapGate),
        _spec("ccx", 3, 0, qlib.CCXGate),
        _spec("rx", 1, 1, qlib.RXGate),
        _spec("ry", 1, 1, qlib.RYGate),
        _spec("rz", 1, 1, qlib.RZGate),
        _spec("p", 1, 1, qlib.PhaseGate),
        _spec("u1", 1, 1, qlib.PhaseGate),
        _spec("u2", 1, 2, lambda phi, lam: qlib.UGate(math.pi / 2, phi, lam)),
        _spec("u3", 1, 3, qlib.UGate),
        _spec("crx", 2, 1, qlib.CRXGate),
        _spec("cry", 2, 1, qlib.CRYGate),
        _spec("crz", 2, 1, qlib.CRZGate),
        _spec("cp", 2, 1, qlib.CPhaseGate),
        _spec("cu3", 2, 3, lambda th, phi, lam: qlib.CUGate(th, phi, lam, 0.0)),
    ]
}
"""All gates quspect supports, keyed by quspect gate name.

Every entry has been verified (``test_conventions.py``) to satisfy
``gate_matrix(spec.quspect_gate(*p)) == qiskit_gate_matrix_be(spec.qiskit_gate(*p))``
exactly, i.e. without a global phase difference.
"""


# =============================================================================
# Circuit conversion
# =============================================================================


def _convert_param(
    value: Any, table: dict[str, qiskit.circuit.Parameter]
) -> QiskitParamValue:
    """Convert a quspect parameter value to a qiskit parameter value."""
    if isinstance(value, Parameter):
        if value.name not in table:
            table[value.name] = qiskit.circuit.Parameter(value.name)
        return table[value.name]
    if isinstance(value, ParameterExpression):
        if value.is_constant:
            return float(value)
        expr: QiskitParamValue = float(value.constant)
        for param, coef in value.terms.items():
            expr = expr + float(coef) * _convert_param(param, table)
        return expr
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    raise TypeError(f"Unsupported gate parameter type: {type(value).__name__}")


def _append_instruction(
    qc: qiskit.QuantumCircuit,
    instr: CircuitInstruction,
    noise: str,
    table: dict[str, qiskit.circuit.Parameter],
) -> None:
    """Append one quspect instruction to a qiskit circuit."""
    if instr.is_gate:
        assert instr.gate is not None
        gate = instr.gate
        params = [_convert_param(p, table) for p in gate.params]
        spec = GATE_SPECS.get(gate.name)
        if spec is not None:
            spec.apply_qiskit(qc, params, instr.qubits)
        else:
            # Unknown name: fall back to the big-endian matrix.
            mat = gate_matrix(gate)
            qc.append(qlib.UnitaryGate(mat, label=gate.name), list(instr.qubits)[::-1])
    elif instr.is_measure:
        qc.measure(list(instr.qubits), list(instr.clbits))
    elif instr.is_barrier:
        if instr.qubits:
            qc.barrier(*instr.qubits)
        else:
            qc.barrier()
    elif instr.is_channel:
        assert instr.channel is not None
        if noise != "kraus":
            raise NotImplementedError(
                f"Noise channel {instr.channel.name!r} cannot be converted with "
                f"noise={noise!r}; use to_qiskit(..., noise='kraus')."
            )
        kraus = Kraus(
            [np.asarray(k, dtype=complex) for k in instr.channel.kraus_operators()]
        )
        # Kraus ops are big-endian like gate_matrix; qiskit qargs[0] is the LSB.
        qc.append(kraus.to_instruction(), list(instr.qubits)[::-1])
    else:  # pragma: no cover - defensive
        raise NotImplementedError(f"Unsupported instruction type {instr.type}")


def to_qiskit(
    circ: quspect.QuantumCircuit,
    *,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
    noise: str = "error",
    parameter_table: dict[str, qiskit.circuit.Parameter] | None = None,
) -> qiskit.QuantumCircuit:
    """Convert a quspect circuit to an equivalent qiskit circuit.

    Qubit ``i`` / clbit ``i`` of quspect map to qubit ``i`` / clbit ``i`` of
    qiskit, so statevectors agree element-wise (both little-endian) and only
    count bitstrings need reversing (see :func:`qiskit_counts_to_quspect`).

    Mapping: standard gates map 1:1; ``u1 -> PhaseGate``,
    ``u2(phi, lam) -> UGate(pi/2, phi, lam)``, ``u3 -> UGate``,
    ``cu3(t, p, l) -> CUGate(t, p, l, 0)``; unknown gate names fall back to a
    ``UnitaryGate`` built from ``gate_matrix``.  ``measure -> measure``,
    ``barrier -> barrier`` (an empty quspect barrier becomes a full barrier).

    Symbolic parameters: each quspect ``Parameter`` becomes a qiskit
    ``Parameter`` with the same name (one object per name, reused across the
    circuit and stored in ``parameter_table``); a quspect
    ``ParameterExpression`` (affine: ``constant + sum(coef * param)``) becomes
    the equivalent qiskit expression.  Bind with
    ``qc.assign_parameters({parameter_table["theta"]: 0.3})``.

    Args:
        circ: The quspect circuit.
        num_qubits: Width of the qiskit circuit; defaults to
            ``circ.num_qubits``.  Use it to pad idle high qubits.
        num_clbits: Number of clbits; defaults to ``circ.num_clbits``.
        noise: ``"error"`` (default) raises ``NotImplementedError`` on noise
            channels; ``"kraus"`` appends each channel as a
            ``qiskit.quantum_info.Kraus`` instruction (evolve the result with
            ``qiskit.quantum_info.DensityMatrix``; ``Statevector`` and
            ``Operator`` cannot handle it).
        parameter_table: Optional dict (name -> qiskit Parameter) that is
            reused and filled in place.

    Returns:
        The qiskit circuit.

    Raises:
        NotImplementedError: On a noise channel when ``noise != "kraus"``.
        TypeError: On an unsupported parameter type.
        ValueError: If ``noise`` is not ``"error"`` or ``"kraus"``.
    """
    if noise not in ("error", "kraus"):
        raise ValueError(f"noise must be 'error' or 'kraus', got {noise!r}")
    nq = circ.num_qubits if num_qubits is None else num_qubits
    nc = circ.num_clbits if num_clbits is None else num_clbits
    qc = qiskit.QuantumCircuit(nq, nc) if nc > 0 else qiskit.QuantumCircuit(nq)
    table = parameter_table if parameter_table is not None else {}
    for instr in circ.instructions:
        _append_instruction(qc, instr, noise, table)
    return qc


# =============================================================================
# Matrices and states
# =============================================================================


def quspect_matrix(gate: Gate) -> ComplexArray:
    """Return quspect's big-endian matrix of ``gate`` (``gate_matrix``)."""
    return np.asarray(gate_matrix(gate), dtype=complex)


def qiskit_gate_matrix_be(qiskit_gate: Any) -> ComplexArray:
    """Return the big-endian matrix of a qiskit gate/operator.

    Big-endian means the gate's first qarg is the most-significant bit,
    matching :func:`quspect_matrix`.  Equivalent to
    ``Operator(qiskit_gate).reverse_qargs().data``.
    """
    return np.asarray(Operator(qiskit_gate).reverse_qargs().data, dtype=complex)


def _as_array(x: Any) -> ComplexArray:
    """Convert numpy arrays / qiskit Statevector / Operator to ndarray."""
    data = getattr(x, "data", x)
    return np.asarray(data, dtype=complex)


def global_phase(a: Any, b: Any) -> complex:
    """Return the unit complex ``phase`` minimizing ``||a - phase * b||``.

    Works for vectors and matrices (``phase = <b, a> / |<b, a>|``).  Returns
    ``1`` if the overlap vanishes.
    """
    av, bv = _as_array(a).ravel(), _as_array(b).ravel()
    inner = complex(np.vdot(bv, av))
    if abs(inner) < 1e-15:
        return 1.0 + 0j
    return inner / abs(inner)


def assert_unitary_equal(
    a: Any, b: Any, *, up_to_phase: bool = False, atol: float = 1e-9
) -> complex:
    """Assert two matrices are equal, optionally up to a global phase.

    Args:
        a: Matrix (ndarray or qiskit ``Operator``).
        b: Matrix (ndarray or qiskit ``Operator``).
        up_to_phase: Allow ``a == phase * b`` for a unit complex ``phase``.
        atol: Absolute element-wise tolerance.

    Returns:
        The phase used (``1`` when ``up_to_phase`` is False).

    Raises:
        AssertionError: If shapes differ or entries differ by more than atol.
    """
    am, bm = _as_array(a), _as_array(b)
    assert am.shape == bm.shape, f"shape mismatch: {am.shape} vs {bm.shape}"
    phase = global_phase(am, bm) if up_to_phase else 1.0 + 0j
    diff = float(np.max(np.abs(am - phase * bm))) if am.size else 0.0
    assert diff <= atol, (
        f"matrices differ (max |diff| = {diff:.3e} > atol {atol:.1e}, "
        f"phase={phase:.6f})\na=\n{np.round(am, 6)}\nb=\n{np.round(bm, 6)}"
    )
    return phase


def assert_states_equal(
    a: Any, b: Any, *, up_to_phase: bool = True, atol: float = 1e-9
) -> complex:
    """Assert two statevectors are equal, by default up to a global phase.

    Args:
        a: Statevector (ndarray or qiskit ``Statevector``).
        b: Statevector (ndarray or qiskit ``Statevector``).
        up_to_phase: Allow ``a == phase * b`` for a unit complex ``phase``.
        atol: Absolute element-wise tolerance (also applied to ``1 - F``).

    Returns:
        The phase used (``1`` when ``up_to_phase`` is False).

    Raises:
        AssertionError: If the states differ.
    """
    av, bv = _as_array(a).ravel(), _as_array(b).ravel()
    assert av.shape == bv.shape, f"dimension mismatch: {av.shape} vs {bv.shape}"
    fidelity = abs(np.vdot(av, bv)) ** 2
    phase = global_phase(av, bv) if up_to_phase else 1.0 + 0j
    diff = float(np.max(np.abs(av - phase * bv))) if av.size else 0.0
    assert diff <= atol, (
        f"states differ (max |diff| = {diff:.3e}, fidelity={fidelity:.12f}, "
        f"phase={phase:.6f})\na={np.round(av, 6)}\nb={np.round(bv, 6)}"
    )
    return phase


def _basis_prep_circuit(index: int, num_qubits: int) -> quspect.QuantumCircuit:
    """Return a quspect circuit preparing |index> (little-endian) on n qubits."""
    prep = quspect.QuantumCircuit()
    prep.id(num_qubits - 1)  # pin the width
    for q in range(num_qubits):
        if (index >> q) & 1:
            prep.x(q)
    return prep


def quspect_circuit_unitary(
    circ: quspect.QuantumCircuit,
    *,
    num_qubits: int | None = None,
    simulator: Simulator | None = None,
) -> ComplexArray:
    """Return the little-endian unitary of ``circ`` using only quspect.

    Column ``k`` is ``Simulator().statevector(prep_k + circ)`` where
    ``prep_k`` applies X gates preparing basis state ``|k>``.  Since X has no
    phase, the columns carry consistent global phase.  Barriers are kept,
    measurements and channels are rejected.

    Args:
        circ: Unitary quspect circuit (all parameters bound).
        num_qubits: Width; defaults to ``circ.num_qubits``.
        simulator: Simulator to use (defaults to ``Simulator(seed=0)``).

    Returns:
        ``(2**n, 2**n)`` complex array, little-endian (qiskit ``Operator``
        convention).

    Raises:
        ValueError: If the circuit contains measurements or channels.
    """
    n = circ.num_qubits if num_qubits is None else num_qubits
    if any(i.is_measure or i.is_channel for i in circ.instructions):
        raise ValueError("quspect_circuit_unitary requires a unitary circuit")
    sim = simulator if simulator is not None else Simulator(seed=0)
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        full = _basis_prep_circuit(k, n)
        for instr in circ.instructions:
            if instr.is_gate:
                assert instr.gate is not None
                full.append(instr.gate, instr.qubits)
            elif instr.is_barrier:
                full.barrier(*instr.qubits)
        out[:, k] = sim.statevector(full)
    return out


def qiskit_circuit_unitary(
    circ: quspect.QuantumCircuit | qiskit.QuantumCircuit,
    *,
    num_qubits: int | None = None,
) -> ComplexArray:
    """Return the little-endian unitary of a circuit computed by qiskit.

    Accepts a quspect circuit (converted with :func:`to_qiskit`) or a qiskit
    circuit.  Barriers are removed; measurements raise.

    Raises:
        ValueError: If the circuit contains measurements.
    """
    qc = (
        to_qiskit(circ, num_qubits=num_qubits)
        if isinstance(circ, quspect.QuantumCircuit)
        else circ
    )
    stripped = qc.copy_empty_like()
    for ci in qc.data:
        if ci.operation.name == "measure":
            raise ValueError("qiskit_circuit_unitary requires a unitary circuit")
        if ci.operation.name != "barrier":
            stripped.append(ci)
    return np.asarray(Operator(stripped).data, dtype=complex)


def circuit_unitary(
    circ: quspect.QuantumCircuit, *, num_qubits: int | None = None
) -> ComplexArray:
    """Little-endian unitary of a quspect circuit (alias of the quspect path).

    Uses :func:`quspect_circuit_unitary`; use :func:`qiskit_circuit_unitary`
    for the reference value.
    """
    return quspect_circuit_unitary(circ, num_qubits=num_qubits)


# =============================================================================
# Counts and exact distributions
# =============================================================================


def qiskit_counts_to_quspect(counts: Mapping[str, int]) -> dict[str, int]:
    """Convert qiskit count keys to quspect convention.

    Spaces (multi-register output) are stripped first, then the string is
    reversed so that ``s[i]`` is clbit ``i``.  Colliding keys are summed.
    """
    out: dict[str, int] = {}
    for key, val in counts.items():
        k = key.replace(" ", "")[::-1]
        out[k] = out.get(k, 0) + int(val)
    return out


def quspect_counts_to_qiskit(counts: Mapping[str, int]) -> dict[str, int]:
    """Convert quspect count keys to qiskit convention (clbit 0 rightmost)."""
    out: dict[str, int] = {}
    for key, val in counts.items():
        k = key[::-1]
        out[k] = out.get(k, 0) + int(val)
    return out


def _split_terminal_measurements(
    qc: qiskit.QuantumCircuit,
) -> tuple[qiskit.QuantumCircuit, dict[int, int]]:
    """Split ``qc`` into its unitary/Kraus part and a clbit -> qubit map."""
    body = qc.copy_empty_like()
    clbit_to_qubit: dict[int, int] = {}
    measured: set[int] = set()
    for ci in qc.data:
        name = ci.operation.name
        qidx = [qc.find_bit(q).index for q in ci.qubits]
        if name == "measure":
            c = qc.find_bit(ci.clbits[0]).index
            clbit_to_qubit[c] = qidx[0]
            measured.add(qidx[0])
            continue
        if name == "barrier":
            continue
        if measured.intersection(qidx):
            raise ValueError("mid-circuit measurement is not supported")
        body.append(ci)
    return body, clbit_to_qubit


def qiskit_probabilities(
    circ: qiskit.QuantumCircuit | quspect.QuantumCircuit,
) -> dict[str, float]:
    """Exact measurement distribution computed by qiskit, quspect-keyed.

    The circuit is evolved with ``Statevector`` (or ``DensityMatrix`` when it
    contains Kraus instructions).  Measurements must be terminal.  Keys are
    in **quspect convention**: ``key[c]`` is the value of clbit ``c``;
    unmeasured clbits read ``0``.  If the circuit has no measurement at all,
    every qubit ``q`` is treated as measured into clbit ``q``.  Outcomes with
    probability below 1e-14 are omitted.

    Args:
        circ: A qiskit circuit, or a quspect circuit (converted with
            ``to_qiskit(circ, noise="kraus")``).

    Returns:
        Dict ``bitstring -> probability``.

    Raises:
        ValueError: On mid-circuit measurement.
    """
    qc = (
        to_qiskit(circ, noise="kraus")
        if isinstance(circ, quspect.QuantumCircuit)
        else circ
    )
    body, c2q = _split_terminal_measurements(qc)
    n = qc.num_qubits
    if not c2q:
        c2q = {q: q for q in range(n)}
        num_c = n
    else:
        num_c = max(qc.num_clbits, max(c2q) + 1)
    has_kraus = any(
        not isinstance(ci.operation, qiskit.circuit.Gate) for ci in body.data
    )
    state: Any = QiskitDensityMatrix(body) if has_kraus else Statevector(body)
    probs = np.asarray(state.probabilities(), dtype=float)
    out: dict[str, float] = {}
    for idx, p in enumerate(probs):
        if p < 1e-14:
            continue
        bits = ["0"] * num_c
        for c, q in c2q.items():
            bits[c] = str((idx >> q) & 1)
        key = "".join(bits)
        out[key] = out.get(key, 0.0) + float(p)
    return out


def qiskit_sample_counts(
    circ: qiskit.QuantumCircuit | quspect.QuantumCircuit, shots: int, seed: int
) -> dict[str, int]:
    """Sample counts from the exact qiskit distribution (quspect-keyed).

    Useful as a reference sampler (qiskit-aer is not available).
    """
    probs = qiskit_probabilities(circ)
    keys = sorted(probs)
    p = np.array([probs[k] for k in keys])
    draws = np.random.default_rng(seed).multinomial(shots, p / p.sum())
    return {k: int(c) for k, c in zip(keys, draws, strict=True) if c > 0}


# =============================================================================
# Random circuits
# =============================================================================

_SPECIAL_ANGLES = (0.0, math.pi, -math.pi, math.pi / 2, -math.pi / 2)


def _random_angle(rng: np.random.Generator) -> float:
    """Random angle: 20% special values, otherwise uniform in [-2pi, 2pi]."""
    if rng.random() < 0.2:
        return float(_SPECIAL_ANGLES[int(rng.integers(len(_SPECIAL_ANGLES)))])
    return float(rng.uniform(-2 * math.pi, 2 * math.pi))


def random_circuit_pair(
    num_qubits: int,
    depth: int,
    seed: int,
    *,
    gate_set: Iterable[str] | None = None,
    parametric: bool = True,
) -> tuple[quspect.QuantumCircuit, qiskit.QuantumCircuit]:
    """Build the same random unitary circuit in quspect and in qiskit.

    Both circuits are built **independently** from :data:`GATE_SPECS` (quspect
    via the fluent API ``circ.<name>(*params, *qubits)``, qiskit via
    ``qc.append``), so comparing them tests the fluent API argument order as
    well as the simulator.  The circuit starts with an ``id`` on every qubit so
    that ``circ.num_qubits == num_qubits``.  Angles are uniform in
    ``[-2pi, 2pi]`` with 20% drawn from ``{0, +-pi, +-pi/2}``.

    Args:
        num_qubits: Number of qubits (>= 1).
        depth: Number of random gates (excluding the initial ``id`` layer).
        seed: Seed for ``np.random.default_rng``.
        gate_set: quspect gate names to draw from (default: all of
            :data:`GATE_SPECS`).  Gates wider than ``num_qubits`` are dropped.
        parametric: If False, parameterized gates are excluded.

    Returns:
        ``(quspect_circuit, qiskit_circuit)``.

    Raises:
        ValueError: If no gate in ``gate_set`` fits.
    """
    rng = np.random.default_rng(seed)
    names = list(GATE_SPECS) if gate_set is None else list(gate_set)
    specs = [
        GATE_SPECS[n]
        for n in names
        if GATE_SPECS[n].num_qubits <= num_qubits
        and (parametric or GATE_SPECS[n].num_params == 0)
    ]
    if not specs:
        raise ValueError("no gate in gate_set fits the requested circuit")
    circ = quspect.QuantumCircuit()
    qc = qiskit.QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        circ.id(q)
        qc.id(q)
    for _ in range(depth):
        spec = specs[int(rng.integers(len(specs)))]
        qubits = [int(q) for q in rng.permutation(num_qubits)[: spec.num_qubits]]
        params = [_random_angle(rng) for _ in range(spec.num_params)]
        spec.apply_quspect(circ, params, qubits)
        spec.apply_qiskit(qc, params, qubits)
    return circ, qc


# =============================================================================
# Statistical tolerance tools
# =============================================================================


def binomial_tolerance(p: float, shots: int, n_sigma: float = N_SIGMA) -> float:
    """Tolerance for a frequency estimate of probability ``p``.

    ``n_sigma * sqrt(max(p (1 - p), 1 / shots) / shots)``, never below
    ``1 / shots``.  The variance floor ``1/shots`` handles ``p ~ 0`` or
    ``p ~ 1`` where the binomial variance (and the normal approximation)
    degenerates.

    Raises:
        ValueError: If ``shots <= 0``.
    """
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}")
    var = max(p * (1.0 - p), 1.0 / shots)
    return max(n_sigma * math.sqrt(var / shots), 1.0 / shots)


def assert_probability_close(
    p_hat: float, p: float, shots: int, n_sigma: float = N_SIGMA
) -> None:
    """Assert ``|p_hat - p| <= binomial_tolerance(p, shots, n_sigma)``."""
    tol = binomial_tolerance(p, shots, n_sigma)
    assert abs(p_hat - p) <= tol, (
        f"estimated probability {p_hat:.6f} differs from expected {p:.6f} by "
        f"{abs(p_hat - p):.3e} > tol {tol:.3e} ({n_sigma} sigma, {shots} shots)"
    )


def _comparison_table(
    keys: Sequence[str],
    counts: Mapping[str, int],
    probs: Mapping[str, float],
    shots: int,
) -> str:
    """Format an observed vs expected table."""
    lines = [f"{'outcome':>12} {'observed':>9} {'expected':>11} {'p_hat':>9} {'p':>9}"]
    for k in keys:
        c = counts.get(k, 0)
        p = probs.get(k, 0.0)
        lines.append(f"{k:>12} {c:>9d} {p * shots:>11.2f} {c / shots:>9.5f} {p:>9.5f}")
    return "\n".join(lines)


def assert_counts_match(
    counts: Mapping[str, int],
    expected_probs: Mapping[str, float],
    shots: int,
    *,
    n_sigma: float = N_SIGMA,
    alpha: float = 1e-6,
    allow_unexpected: bool = False,
) -> None:
    """Two-level statistical check of sampled counts against exact probabilities.

    (a) For every outcome, ``|count/shots - p| <= binomial_tolerance(p)``; an
    outcome with expected probability 0 (< 1e-12) must have zero counts unless
    ``allow_unexpected``.  (b) A Pearson chi-square goodness-of-fit test over
    all outcomes (cells with expected count < 5 pooled together; the pooled
    cell is merged into the smallest remaining cell if still < 5) must have
    p-value > ``alpha``.  Both counts and probabilities use the same bitstring
    convention (normally quspect's).

    Args:
        counts: Observed counts (zero entries allowed).
        expected_probs: Exact probabilities (must sum to 1 within 1e-6).
        shots: Total number of shots; must equal ``sum(counts.values())``.
        n_sigma: Per-outcome tolerance multiplier.
        alpha: Minimum accepted chi-square p-value.
        allow_unexpected: Allow counts on zero-probability outcomes (only the
            per-outcome tolerance applies to them).

    Raises:
        AssertionError: With an observed/expected table on failure.
    """
    total = sum(int(v) for v in counts.values())
    keys = sorted(set(counts) | set(expected_probs))
    table = _comparison_table(keys, counts, expected_probs, shots)
    assert total == shots, f"sum(counts)={total} != shots={shots}\n{table}"
    psum = sum(expected_probs.values())
    assert abs(psum - 1.0) < 1e-6, f"expected probabilities sum to {psum}"

    failures = []
    for k in keys:
        c = int(counts.get(k, 0))
        p = float(expected_probs.get(k, 0.0))
        if p < 1e-12 and c > 0 and not allow_unexpected:
            failures.append(f"{k}: {c} counts on a zero-probability outcome")
            continue
        tol = binomial_tolerance(p, shots, n_sigma)
        if abs(c / shots - p) > tol:
            failures.append(f"{k}: |{c / shots:.5f} - {p:.5f}| > {tol:.5f}")
    assert not failures, (
        "per-outcome check failed:\n" + "\n".join(failures) + "\n" + table
    )

    obs = np.array([counts.get(k, 0) for k in keys], dtype=float)
    exp = (
        np.array([expected_probs.get(k, 0.0) for k in keys], dtype=float) / psum * shots
    )
    big = exp >= 5.0
    cells_obs = list(obs[big])
    cells_exp = list(exp[big])
    small_obs, small_exp = float(obs[~big].sum()), float(exp[~big].sum())
    if small_exp > 0 or small_obs > 0:
        if small_exp >= 5.0 or not cells_exp:
            cells_obs.append(small_obs)
            cells_exp.append(small_exp)
        else:
            j = int(np.argmin(cells_exp))
            cells_obs[j] += small_obs
            cells_exp[j] += small_exp
    if len(cells_exp) < 2:
        return
    f_exp = np.array(cells_exp)
    f_obs = np.array(cells_obs)
    f_exp = f_exp * (f_obs.sum() / f_exp.sum())
    res = stats.chisquare(f_obs, f_exp)
    pvalue = float(res.pvalue)
    assert pvalue > alpha, (
        f"chi-square goodness-of-fit failed: stat={float(res.statistic):.3f}, "
        f"dof={len(f_exp) - 1}, p-value={pvalue:.3e} <= alpha={alpha:.1e}\n{table}"
    )


def pm1_std(expectation: float) -> float:
    """Single-shot std of a +/-1 valued observable with mean ``expectation``."""
    return math.sqrt(max(0.0, 1.0 - expectation * expectation))


def assert_mean_close(
    samples_or_mean: float | Sequence[float] | NDArray[np.float64],
    expected: float,
    std: float | None = None,
    n: int | None = None,
    *,
    n_sigma: float = N_SIGMA,
    atol: float = 1e-12,
) -> None:
    """Assert a sample mean agrees with ``expected`` within ``n_sigma * std / sqrt(n)``.

    Args:
        samples_or_mean: Either raw samples (array-like) or a precomputed mean.
        expected: Exact expected value.
        std: Single-sample standard deviation.  For raw samples it defaults
            to the sample std (``ddof=1``); for a scalar mean it is required.
            Prefer the *exact* std (e.g. :func:`pm1_std`) when known.
        n: Number of samples; defaults to ``len(samples)``; required for a
            scalar mean.
        n_sigma: Tolerance multiplier.
        atol: Absolute floor added to the tolerance (handles ``std == 0``).

    Raises:
        ValueError: If ``std``/``n`` are missing for a scalar mean.
        AssertionError: If the mean is outside the tolerance.
    """
    arr = np.asarray(samples_or_mean, dtype=float)
    if arr.ndim == 0:
        if std is None or n is None:
            raise ValueError("std and n are required when passing a scalar mean")
        mean = float(arr)
    else:
        mean = float(arr.mean())
        n = arr.size if n is None else n
        if std is None:
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    tol = n_sigma * std / math.sqrt(n) + atol
    assert abs(mean - expected) <= tol, (
        f"mean {mean:.6f} differs from expected {expected:.6f} by "
        f"{abs(mean - expected):.3e} > tol {tol:.3e} "
        f"({n_sigma} sigma, std={std:.4f}, n={n})"
    )
