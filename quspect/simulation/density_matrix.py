"""Density matrix backed by the C++ native backend.

The :class:`DensityMatrix` class wraps the C++ ``DensityMatrix64`` and
provides a Python-friendly API for mixed-state quantum simulation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quspect.native import _core, _ensure_core


class DensityMatrix:
    """N-qubit density matrix backed by the C++ native backend.

    Represents a mixed or pure quantum state as a ``2^n × 2^n`` complex
    matrix ρ.  Gate application, Kraus channels, and basic operations
    delegate to optimized C++ kernels.

    Example:
        >>> rho = DensityMatrix(2)  # |00⟩⟨00|
        >>> rho.purity()
        1.0
        >>> rho = DensityMatrix.maximally_mixed(1)
        >>> abs(rho.trace() - 1.0) < 1e-12
        True
    """

    def __init__(self, num_qubits: int) -> None:
        """Create a density matrix initialized to |0…0⟩⟨0…0|.

        Args:
            num_qubits: Number of qubits. Must be >= 0.

        Raises:
            ValueError: If ``num_qubits < 0``.
        """
        _ensure_core()
        if num_qubits < 0:
            raise ValueError(f"num_qubits must be >= 0, got {num_qubits}")
        self._cpp = _core.DensityMatrix64(num_qubits)

    # -- properties --------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Number of qubits."""
        return int(self._cpp.num_qubits)

    @property
    def dim(self) -> int:
        """Linear dimension of the matrix (2^n)."""
        return int(self._cpp.dim)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the density matrix ``(2^n, 2^n)``."""
        d = self.dim
        return (d, d)

    @property
    def _data(self) -> NDArray[np.complex128]:
        """The underlying complex array."""
        data: NDArray[np.complex128] = self._cpp.data()
        return data

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_array(cls, data: NDArray[np.complex128]) -> DensityMatrix:
        """Create a DensityMatrix from an existing array.

        Args:
            data: A ``(d, d)`` complex matrix where ``d`` is a power of 2.

        Returns:
            A new DensityMatrix initialized from ``data``.

        Raises:
            ValueError: If ``data`` is not square or its dimension is not
                a power of 2.
        """
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {data.shape}")
        d = data.shape[0]
        if d & (d - 1) != 0:
            raise ValueError(f"Dimension {d} is not a power of 2")
        n = int(np.log2(d))
        dm = cls(n)
        dm._data[...] = data
        return dm

    @classmethod
    def from_pure_state(cls, statevector: NDArray[np.complex128]) -> DensityMatrix:
        """Create a DensityMatrix from a pure state |ψ⟩⟨ψ|.

        Args:
            statevector: A complex state vector of length ``2^n``.

        Returns:
            A new DensityMatrix representing the pure state.
        """
        sv = np.asarray(statevector, dtype=complex).ravel()
        d = len(sv)
        if d & (d - 1) != 0:
            raise ValueError(f"State vector length {d} is not a power of 2")
        n = int(np.log2(d))
        dm = cls(n)
        data = dm._data
        data[...] = np.outer(sv, sv.conj())
        return dm

    @classmethod
    def from_labels(cls, label_probs: list[tuple[str, float]]) -> DensityMatrix:
        """Create a mixed state from computational-basis labels and probabilities.

        Args:
            label_probs: List of ``(bitstring, probability)`` pairs.  All
                bitstrings must have the same length.

        Returns:
            A new DensityMatrix.
        """
        if not label_probs:
            raise ValueError("label_probs must be non-empty")
        n = len(label_probs[0][0])
        dm = cls(n)
        dm._cpp.set_zero()
        data = dm._data
        for bs, prob in label_probs:
            idx = int(bs, 2)
            data[idx, idx] = complex(prob, 0)
        return dm

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> DensityMatrix:
        """Create the maximally mixed state I / 2^n.

        Args:
            num_qubits: Number of qubits.

        Returns:
            A new DensityMatrix.
        """
        dm = cls(num_qubits)
        dm._cpp.set_zero()
        dm._data[...] = np.eye(dm.dim, dtype=complex) / dm.dim
        return dm

    # -- operations --------------------------------------------------------

    def trace(self) -> complex:
        """Return the trace of the density matrix."""
        return complex(self._cpp.trace())

    def normalize(self) -> None:
        """Normalize the density matrix to have trace 1."""
        self._cpp.normalize()

    def purity(self) -> float:
        """Return Tr(ρ²), a measure of mixedness (1 = pure, <1 = mixed)."""
        return float(np.real(np.trace(self._data @ self._data)))

    def subsystem_purity(self, qubits: tuple[int, ...]) -> float:
        """Purity of the reduced density matrix on the given qubits.

        The complement of ``qubits`` is traced out and ``Tr(ρ_A²)`` of the
        remaining subsystem ``A = qubits`` is returned.

        Args:
            qubits: Indices of the qubits forming the subsystem of interest.

        Returns:
            The purity of the reduced state on ``qubits`` (1 for a pure
            reduced state, ``1 / 2^k`` for a maximally mixed one).

        Raises:
            ValueError: If ``qubits`` contains duplicates or out-of-range
                indices.

        Example:
            >>> import numpy as np
            >>> rho = np.kron(np.eye(2) / 2, np.diag([1.0, 0.0])).astype(complex)
            >>> dm = DensityMatrix.from_array(rho)  # q0 = |0>, q1 = I/2
            >>> round(dm.subsystem_purity((0,)), 6)
            1.0
            >>> round(dm.subsystem_purity((1,)), 6)
            0.5
        """
        self._validate_qubits(qubits)
        complement = tuple(q for q in range(self.num_qubits) if q not in qubits)
        reduced = self.partial_trace(complement)
        return reduced.purity()

    def partial_trace(self, qubits: tuple[int, ...]) -> DensityMatrix:
        """Trace out the specified qubits, returning the reduced density matrix.

        Args:
            qubits: Indices of qubits to trace out.

        Returns:
            A new DensityMatrix on the remaining qubits.
        """
        n = self.num_qubits
        keep = sorted(set(range(n)) - set(qubits))
        if not keep:
            return DensityMatrix(0)
        return _partial_trace_numpy(self._data, n, tuple(keep))

    def fidelity(self, other: DensityMatrix) -> float:
        """Compute the fidelity F(ρ, σ) = Tr(√(√ρ σ √ρ))².

        Args:
            other: Another density matrix with the same number of qubits.

        Returns:
            Fidelity in [0, 1].
        """
        evals = np.linalg.eigvalsh(self._data)
        if np.any(evals < -1e-12):
            raise ValueError("Density matrix is not positive semi-definite")
        sqrt_rho = _sqrtm_psd(self._data)
        inner = sqrt_rho @ other._data @ sqrt_rho
        inner_evals = np.linalg.eigvalsh(inner)
        inner_evals = np.clip(inner_evals, 0, None)
        sqrt_inner = np.sum(np.sqrt(inner_evals))
        return float(np.real(sqrt_inner**2))

    # -- gate / channel application ---------------------------------------

    def apply_gate(
        self, matrix: NDArray[np.complex128], qubits: tuple[int, ...]
    ) -> None:
        """Apply a unitary gate in-place: ρ → U ρ U† via C++ backend.

        Args:
            matrix: The unitary matrix of shape ``(2^k, 2^k)``.
            qubits: Qubit indices the gate acts on.
        """
        self._cpp.apply_unitary(
            np.asarray(matrix, dtype=complex), np.array(qubits, dtype=np.uint32)
        )

    def apply_channel(
        self, kraus_ops: list[NDArray[np.complex128]], qubits: tuple[int, ...]
    ) -> None:
        """Apply a Kraus channel in-place via C++ backend.

        Args:
            kraus_ops: List of Kraus operator matrices.
            qubits: Qubit indices the channel acts on.
        """
        num_ops = len(kraus_ops)
        k = len(qubits)
        expected_shape = (1 << k, 1 << k)
        flat = np.zeros((num_ops, 1 << k, 1 << k), dtype=complex)
        for i, op in enumerate(kraus_ops):
            op_arr = np.asarray(op, dtype=complex)
            if op_arr.shape != expected_shape:
                raise ValueError(
                    f"Kraus operator {i} has shape {op_arr.shape}, "
                    f"expected {expected_shape}"
                )
            flat[i] = op_arr
        self._cpp.apply_kraus_channel(flat, num_ops, np.array(qubits, dtype=np.uint32))

    # -- measurement ------------------------------------------------------

    def probabilities(self, qubits: tuple[int, ...]) -> NDArray[np.float64]:
        """Compute marginal measurement probabilities for the specified qubits.

        All qubits not listed in ``qubits`` are traced out.  The result is
        ordered like qiskit's ``probabilities(qargs)``: bit ``j`` of the
        returned index is the outcome of ``qubits[j]`` (``qubits[0]`` is the
        least-significant bit).

        Args:
            qubits: Qubit indices to measure, in the order that defines the
                bit significance of the result index.

        Returns:
            A 1-D array of length ``2^len(qubits)`` with the marginal
            probabilities.

        Raises:
            ValueError: If ``qubits`` contains duplicates or out-of-range
                indices.

        Example:
            >>> dm = DensityMatrix.from_labels([("01", 1.0)])  # q0=1, q1=0
            >>> dm.probabilities((0,)).tolist()
            [0.0, 1.0]
            >>> dm.probabilities((1, 0)).tolist()  # index bit 1 is q0
            [0.0, 0.0, 1.0, 0.0]
        """
        self._validate_qubits(qubits)
        diag = np.real(np.diag(self._data)).astype(np.float64)
        out_index = _reduced_indices(self.dim, qubits)
        probs: NDArray[np.float64] = np.bincount(
            out_index, weights=diag, minlength=1 << len(qubits)
        ).astype(np.float64)
        return probs

    def measure(
        self, qubits: tuple[int, ...], rng: np.random.Generator | None = None
    ) -> tuple[str, DensityMatrix]:
        """Perform a projective measurement and return outcome + post-measurement state.

        Only the listed qubits are measured; the other qubits are left
        untouched apart from the correlations implied by the projection.

        Args:
            qubits: Qubit indices to measure.
            rng: Optional numpy random Generator.

        Returns:
            A tuple ``(bitstring, collapsed_state)`` where ``bitstring[j]``
            is the outcome of ``qubits[j]`` and ``collapsed_state`` is the
            normalized post-measurement density matrix.

        Raises:
            ValueError: If ``qubits`` contains duplicates or out-of-range
                indices.

        Example:
            >>> import numpy as np
            >>> dm = DensityMatrix.from_labels([("01", 1.0)])  # q0=1, q1=0
            >>> bits, post = dm.measure((1, 0), np.random.default_rng(0))
            >>> bits
            '01'
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = np.clip(self.probabilities(qubits), 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            raise ValueError("Cannot measure a density matrix with zero trace")
        k = len(qubits)
        outcome = int(rng.choice(1 << k, p=probs / total))
        bitstring = "".join(str((outcome >> j) & 1) for j in range(k))
        mask = _outcome_mask(self.dim, qubits, outcome)
        new_data = self._data * np.outer(mask, mask)
        tr_val = float(np.real(np.trace(new_data)))
        if tr_val > 1e-15:
            new_data /= tr_val
        result = DensityMatrix(self.num_qubits)
        result._data[...] = new_data
        return bitstring, result

    def _validate_qubits(self, qubits: tuple[int, ...]) -> None:
        """Raise ValueError for duplicate or out-of-range qubit indices."""
        n = self.num_qubits
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Duplicate qubit indices in {qubits}")
        for q in qubits:
            if not 0 <= q < n:
                raise ValueError(f"Qubit index {q} out of range for {n} qubits")

    # -- utility ----------------------------------------------------------

    def copy(self) -> DensityMatrix:
        """Return a deep copy of this density matrix."""
        dm = DensityMatrix(self.num_qubits)
        dm._data[...] = self._data
        return dm

    def __repr__(self) -> str:
        p = float(self.purity())
        return (
            f"DensityMatrix(num_qubits={self.num_qubits}, "
            f"dim={self.dim}, purity={p:.6f}, "
            f"trace={float(np.real(self.trace())):.6f})"
        )


# -- helpers ----------------------------------------------------------------


def _reduced_indices(dim: int, qubits: tuple[int, ...]) -> NDArray[np.int64]:
    """Map every full basis index to its index over ``qubits`` (qubits[0]=LSB)."""
    idx = np.arange(dim, dtype=np.int64)
    out = np.zeros(dim, dtype=np.int64)
    for j, q in enumerate(qubits):
        out |= ((idx >> q) & 1) << j
    return out


def _outcome_mask(dim: int, qubits: tuple[int, ...], outcome: int) -> NDArray[np.bool_]:
    """Boolean mask of basis states whose ``qubits`` bits equal ``outcome``.

    Bit ``j`` of ``outcome`` is the required value of ``qubits[j]``.
    """
    mask: NDArray[np.bool_] = _reduced_indices(dim, qubits) == outcome
    return mask


def _partial_trace_numpy(
    rho: NDArray[np.complex128],
    n_qubits: int,
    keep: tuple[int, ...],
) -> DensityMatrix:
    """Compute the reduced density matrix by tracing out the complement of *keep*."""
    d_total = 1 << n_qubits
    d_keep = 1 << len(keep)
    result = np.zeros((d_keep, d_keep), dtype=complex)
    trace_out = sorted(set(range(n_qubits)) - set(keep))
    for i in range(d_total):
        for j in range(d_total):
            val = rho[i, j]
            if abs(val) < 1e-15:
                continue
            match = True
            for q in trace_out:
                if ((i >> q) & 1) != ((j >> q) & 1):
                    match = False
                    break
            if not match:
                continue
            i_red = _compress_index(i, keep)
            j_red = _compress_index(j, keep)
            result[i_red, j_red] += val
    dm = DensityMatrix(len(keep))
    dm._data[...] = result
    return dm


def _compress_index(idx: int, qubits: tuple[int, ...]) -> int:
    """Compress a full-index to a reduced index by keeping only the listed qubits."""
    result = 0
    for new_pos, q in enumerate(qubits):
        if (idx >> q) & 1:
            result |= 1 << new_pos
    return result


def _sqrtm_psd(a: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Matrix square root for a positive semi-definite matrix."""
    evals, evecs = np.linalg.eigh(a)
    evals = np.clip(evals, 0, None)
    sqrt_evals = np.sqrt(evals)
    return (evecs * sqrt_evals) @ evecs.conj().T
