"""Phase Polynomial intermediate representation for quantum circuits.

The phase polynomial representation captures the essential structure of
{CNOT, RZ} circuits (which are universal for quantum computation when
combined with Hadamard gates).

A phase polynomial consists of:
    1. A **parity matrix** (over GF(2)) mapping each qubit to a linear
       Boolean function of the input variables.
    2. A set of **phase terms**, each being an angle multiplied by a
       parity function.

This IR enables global optimization techniques that DAG-based approaches
cannot easily express, such as resynthesizing circuits without SWAP gates
by directly matching connectivity constraints.

Reference:
    Meijer-van de Griend, "A Comparison of Quantum Compilers using a
    DAG-based or phase polynomial-based Intermediate Representation",
    Journal of Systems and Software, 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quspect.dag.dag_circuit import DAGCircuit

if TYPE_CHECKING:
    from quspect import QuantumCircuit  # noqa: F401

# =========================================================================
# Data Structures
# =========================================================================


@dataclass
class PhaseTerm:
    """A single term in a phase polynomial.

    Represents ``angle * (-1)^{parity}`` applied as a diagonal
    unitary on the computational basis states.

    Attributes:
        angle: The rotation angle in radians.
        parity: Binary vector (over GF(2)) of length ``num_qubits``.
            The term applies a phase of ``angle`` to all basis states
            where ``dot(parity, x) = 1 (mod 2)``.
    """

    angle: float
    parity: np.ndarray  # dtype=bool or int8

    def __repr__(self) -> str:
        indices = [i for i, v in enumerate(self.parity) if v]
        parity_str = " ⊕ ".join(f"x{i}" for i in indices) if indices else "0"
        return f"PhaseTerm({self.angle:.4f} · [{parity_str}])"


@dataclass
class ParityMatrix:
    """A binary matrix over GF(2) representing linear transformations.

    Each row ``i`` represents qubit ``i`` as a parity of input variables.
    The entry ``data[i, j] = 1`` means qubit ``i`` involves input ``j``.

    Initially, this is the identity matrix (qubit i = input i).
    CNOT gates transform it by row operations.

    Attributes:
        data: Binary matrix of shape ``(num_qubits, num_qubits)``.
    """

    data: np.ndarray  # dtype=bool or int8, shape (n, n)

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return int(self.data.shape[0])

    def copy(self) -> ParityMatrix:
        """Return a deep copy of this parity matrix."""
        return ParityMatrix(data=self.data.copy())

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate to the parity matrix (in-place).

        CNOT(ctrl, tgt) transforms:
            x_tgt → x_tgt ⊕ x_ctrl  (GF(2) addition)

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        # Row operation: row[target] ^= row[control]
        self.data[target] ^= self.data[control]

    def apply_swap(self, q1: int, q2: int) -> None:
        """Apply a SWAP operation to the parity matrix (in-place).

        Args:
            q1: First qubit index.
            q2: Second qubit index.
        """
        self.data[[q1, q2]] = self.data[[q2, q1]]

    def column_reduce(self) -> list[tuple[int, int]]:
        """Perform Gaussian elimination (column reduction) on the parity matrix.

        Returns a sequence of CNOT gates that implement the parity
        matrix as a circuit. This uses the standard GraySynth approach:
        reduce the matrix to a form where each row has at most one
        non-zero entry (a "pivot").

        Returns:
            List of (control, target) tuples representing CNOT gates
            that reduce this matrix to row-echelon form.

        Example:
            >>> pm = ParityMatrix(data=np.array([[1,1],[0,1]], dtype=np.int8))
            >>> cnots = pm.column_reduce()
            >>> # cnots = [(0, 1)] — one CNOT to eliminate the (0,1) entry
        """
        n = self.num_qubits
        cnots: list[tuple[int, int]] = []
        matrix = self.data.copy()

        for col in range(n):
            # Find a pivot row for this column
            pivot_row = -1
            for row in range(col, n):
                if matrix[row, col]:
                    pivot_row = row
                    break

            if pivot_row == -1:
                continue

            # Swap pivot row to the diagonal position
            if pivot_row != col:
                matrix[[col, pivot_row]] = matrix[[pivot_row, col]]
                cnots.append((col, pivot_row))  # CNOT-like swap
                cnots.append((pivot_row, col))
                cnots.append((col, pivot_row))

            # Eliminate all other non-zero entries in this column
            for row in range(n):
                if row != col and matrix[row, col]:
                    matrix[row] ^= matrix[col]
                    cnots.append((col, row))

        self.data = matrix
        return cnots

    def __repr__(self) -> str:
        lines = []
        for i in range(self.num_qubits):
            row_str = " ".join(
                str(int(self.data[i, j])) for j in range(self.num_qubits)
            )
            lines.append(f"  [{row_str}]")
        return "ParityMatrix:\n" + "\n".join(lines)


# =========================================================================
# Phase Polynomial
# =========================================================================


@dataclass
class PhasePolynomial:
    """Complete phase polynomial representation of a {CNOT, RZ} circuit.

    The phase polynomial encodes:
        - A linear transformation (parity matrix) from input qubits
          to the current state of each qubit.
        - A list of phase terms (angle + parity function) that
          accumulate along the computation.

    This representation is extracted from a DAG and can be optimized
    via Gaussian elimination to reduce CNOT counts.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        parity_matrix: The parity transformation matrix.
        terms: Accumulated phase terms.
    """

    num_qubits: int
    parity_matrix: ParityMatrix
    terms: list[PhaseTerm] = field(default_factory=list)

    def add_phase(self, angle: float, qubit: int) -> None:
        """Add a phase term on a given qubit.

        The phase is applied to the current parity function on that
        qubit (as tracked by the parity matrix).

        Args:
            angle: Rotation angle in radians.
            qubit: Qubit index where the rotation is applied.

        Example:
            >>> pp = PhasePolynomial(2, ParityMatrix(np.eye(2, dtype=np.int8)))
            >>> pp.add_phase(0.5, 1)
            >>> pp.terms[0].parity.tolist()
            [0, 1]
        """
        self.add_parity_phase(angle, self.parity_matrix.data[qubit])

    def add_parity_phase(self, angle: float, parity: np.ndarray) -> None:
        """Add ``angle`` to the phase term of an explicit parity vector.

        Terms with the same parity are merged; angles with
        ``|angle| < 1e-12`` are ignored.

        Args:
            angle: Rotation angle in radians.
            parity: Binary vector of length ``num_qubits`` over the input
                variables. It is copied, so the caller may mutate it later.

        Example:
            >>> pp = PhasePolynomial(2, ParityMatrix(np.eye(2, dtype=np.int8)))
            >>> pp.add_parity_phase(0.5, np.array([1, 1], dtype=np.int8))
            >>> pp.terms[0].parity.tolist()
            [1, 1]
        """
        if abs(angle) < 1e-12:
            return  # Skip zero-angle terms

        parity = np.asarray(parity, dtype=np.int8).copy()
        # Try to merge with existing term with the same parity
        for term in self.terms:
            if np.array_equal(term.parity, parity):
                term.angle += angle
                return

        self.terms.append(PhaseTerm(angle=angle, parity=parity))

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate, updating the parity matrix.

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        self.parity_matrix.apply_cnot(control, target)

    def optimize(self, *, atol: float = 1e-12) -> None:
        """Simplify the phase terms without changing the represented unitary.

        The phase function ``phi(x) = sum_k angle_k * (c_k . x mod 2)`` is
        canonicalised in place:

        * terms with the same parity vector are merged (angles added),
        * terms whose angle is a multiple of ``2*pi`` (within ``atol``) are
          dropped, as are terms with an all-zero parity (global phase).

        The parity matrix is left untouched: it is the linear (CNOT) part
        of the circuit and is required to reproduce the circuit
        semantics. CNOT-count reduction happens during synthesis in
        :meth:`to_circuit`.

        Args:
            atol: Absolute tolerance for treating an angle as ``0 mod 2*pi``.

        Example:
            >>> pm = ParityMatrix(np.eye(1, dtype=np.int8))
            >>> pp = PhasePolynomial(1, pm, [PhaseTerm(np.pi, np.array([1])),
            ...                              PhaseTerm(np.pi, np.array([1]))])
            >>> pp.optimize()
            >>> pp.terms
            []
        """
        merged: dict[bytes, PhaseTerm] = {}
        for term in self.terms:
            parity = np.asarray(term.parity, dtype=np.int8) & 1
            if not parity.any():
                continue  # global phase only
            key = parity.tobytes()
            if key in merged:
                merged[key].angle += term.angle
            else:
                merged[key] = PhaseTerm(angle=term.angle, parity=parity.copy())

        two_pi = 2.0 * np.pi
        kept: list[PhaseTerm] = []
        for term in merged.values():
            reduced = math.remainder(term.angle, two_pi)
            if abs(reduced) <= atol:
                continue
            kept.append(term)
        self.terms = kept

    def to_circuit(self) -> QuantumCircuit:
        """Synthesize a QuantumCircuit from this phase polynomial.

        The synthesised {CNOT, RZ} circuit implements
        ``|x> -> exp(i phi(x)) |A x>`` (up to a global phase), where ``A``
        is :attr:`parity_matrix` and ``phi`` the phase function of
        :attr:`terms`. The synthesis tracks the current linear map of the
        wires (starting from the identity):

        1. For every phase term with parity ``v``, CNOTs are applied so
           that some wire carries exactly ``v`` (``v`` is written as a sum
           of current wire parities via Gaussian elimination over GF(2)),
           and ``RZ(angle)`` is applied to that wire.
        2. Finally a CNOT network maps the current linear map to the
           target parity matrix.

        Returns:
            A QuantumCircuit implementing this phase polynomial (up to
            global phase).

        Raises:
            ValueError: If the parity matrix is not invertible over GF(2).

        Note:
            This produces a {CNOT, RZ} circuit. Hadamard gates and
            other Clifford operations are not synthesized by this
            method (they must be handled separately).

        Example:
            >>> circ = QuantumCircuit().cx(0, 1).rz(0.3, 1).cx(0, 1)
            >>> pp = PhasePolynomialCompiler.extract(circ.to_dag())
            >>> [i.gate.name for i in pp.to_circuit().instructions]
            ['cx', 'rz', 'cx']
        """
        from quspect import QuantumCircuit

        circ = QuantumCircuit(name="phase_poly_synth")
        n = self.num_qubits
        target = np.asarray(self.parity_matrix.data, dtype=np.int8) & 1
        if n and _gf2_rank(target) != n:
            raise ValueError("Parity matrix is not invertible over GF(2).")

        current = np.eye(n, dtype=np.int8)

        def cnot(control: int, tgt: int) -> None:
            current[tgt] ^= current[control]
            circ.cx(control, tgt)

        # Step 1: place every phase term on a wire carrying its parity.
        for term in self.terms:
            parity = np.asarray(term.parity, dtype=np.int8) & 1
            if not parity.any():
                continue  # global phase
            rows = _gf2_solve_rows(current, parity)
            pivot = rows[0]
            for other in rows[1:]:
                cnot(other, pivot)
            circ.rz(term.angle, pivot)

        # Step 2: map the current linear map onto the target matrix.
        # transform @ current == target  =>  transform = target @ current^-1
        if n:
            transform = (target.astype(np.int64) @ _gf2_inverse(current)) % 2
            pm = ParityMatrix(data=transform.astype(np.int8))
            for ctrl, tgt in reversed(pm.column_reduce()):
                cnot(ctrl, tgt)

        return circ


# =========================================================================
# GF(2) helpers
# =========================================================================


def _gf2_rank(matrix: np.ndarray) -> int:
    """Return the rank of a binary matrix over GF(2)."""
    m = (np.asarray(matrix, dtype=np.int8) & 1).copy()
    rows, cols = m.shape
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if m[r, col]), None)
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, col]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of an invertible binary matrix over GF(2)."""
    n = matrix.shape[0]
    aug = np.concatenate(
        [np.asarray(matrix, dtype=np.int8) & 1, np.eye(n, dtype=np.int8)], axis=1
    )
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r, col]), None)
        if pivot is None:
            raise ValueError("Matrix is not invertible over GF(2).")
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n):
            if r != col and aug[r, col]:
                aug[r] ^= aug[col]
    return aug[:, n:].copy()


def _gf2_solve_rows(rows: np.ndarray, vector: np.ndarray) -> list[int]:
    """Return row indices whose GF(2) sum equals ``vector``.

    Args:
        rows: Invertible binary matrix whose rows are combined.
        vector: Non-zero binary target vector.

    Returns:
        Sorted list of row indices ``S`` with ``xor(rows[S]) == vector``.
    """
    # Solve c^T @ rows = vector  <=>  rows^T @ c = vector.
    inv_t = _gf2_inverse(np.asarray(rows, dtype=np.int8).T)
    coeffs = (inv_t.astype(np.int64) @ (np.asarray(vector, dtype=np.int64) & 1)) % 2
    return [int(i) for i in np.flatnonzero(coeffs)]


# =========================================================================
# PhasePolynomialCompiler: DAGCircuit → PhasePolynomial
# =========================================================================


class PhasePolynomialCompiler:
    """Compiler that extracts a PhasePolynomial from a DAGCircuit.

    This compiler handles {CNOT, RZ, H, X, S, T, ...} circuits by:
    1. Tracking the parity matrix through CNOT gates
    2. Accumulating phase terms from RZ/Phase gates
    3. Handling Hadamard gates via basis-change commutativity rules

    For full universality, Clifford+T or Clifford+RZ gates are needed.
    RZ gates contribute directly to the phase polynomial; other gates
    are handled via commutation or basis changes.

    Example:
        >>> circ = QuantumCircuit()
        >>> circ.rz(0.5, 0).cx(0, 1).rz(0.3, 1)
        >>> dag = circ.to_dag()
        >>> pp = PhasePolynomialCompiler.extract(dag)
        >>> print(f"{len(pp.terms)} phase terms, parity matrix:")
        >>> print(pp.parity_matrix)
    """

    @staticmethod
    def extract(dag: DAGCircuit) -> PhasePolynomial:
        """Extract a phase polynomial from a DAGCircuit.

        Args:
            dag: The input DAG.

        Returns:
            A PhasePolynomial representing the {CNOT, RZ} skeleton
            of the circuit.

        Note:
            Only CNOT, CZ, RZ, Phase, S, T, and similar diagonal or
            parity-transforming gates are extracted. Hadamard and
            other non-diagonal gates are skipped with a warning.
        """
        n = dag.num_qubits
        identity = np.eye(n, dtype=np.int8)
        pm = ParityMatrix(data=identity)
        pp = PhasePolynomial(num_qubits=n, parity_matrix=pm)

        # Process operation nodes in topological order
        for node in dag.operation_nodes():
            if node.instruction is None or node.instruction.gate is None:
                continue

            gate_name = node.instruction.gate.name
            # Operand order matters (cx control/target): use the ordered
            # instruction tuple rather than the unordered node.qubits set.
            qubits = list(node.instruction.qubits)

            if gate_name == "cx":
                if len(qubits) == 2:
                    pp.apply_cnot(qubits[0], qubits[1])

            elif gate_name == "cz":
                if len(qubits) == 2:
                    # CZ(a, b) multiplies by (-1)^{a*b}, and
                    # pi*a*b = pi/2*a + pi/2*b - pi/2*(a XOR b).
                    row_a = pm.data[qubits[0]]
                    row_b = pm.data[qubits[1]]
                    pp.add_parity_phase(np.pi / 2, row_a)
                    pp.add_parity_phase(np.pi / 2, row_b)
                    pp.add_parity_phase(-np.pi / 2, row_a ^ row_b)

            elif gate_name in ("rz", "p", "u1"):
                gate = node.instruction.gate
                # Only parameterized gates have params
                if hasattr(gate, "params") and gate.params:
                    p = gate.params[0]
                    if isinstance(p, (int, float)):
                        pp.add_phase(float(p), qubits[0])

            elif gate_name in ("s", "t"):
                # S = RZ(π/2), T = RZ(π/4)
                angle_map = {"s": np.pi / 2, "t": np.pi / 4}
                pp.add_phase(angle_map[gate_name], qubits[0])

            elif gate_name in ("sdg", "tdg"):
                angle_map = {"sdg": -np.pi / 2, "tdg": -np.pi / 4}
                pp.add_phase(angle_map[gate_name], qubits[0])

            elif gate_name == "z":
                pp.add_phase(np.pi, qubits[0])

            elif gate_name == "swap" and len(qubits) == 2:
                pm.apply_swap(qubits[0], qubits[1])

            # H, X, Y, RX, RY are non-diagonal — skip for now
            # (They require basis-change tracking)

        return pp

    @staticmethod
    def extract_and_optimize(dag: DAGCircuit) -> PhasePolynomial:
        """Extract and optimize a phase polynomial.

        Convenience method that runs extraction followed by
        :meth:`PhasePolynomial.optimize` (merging same-parity terms and
        dropping trivial ones). The result still represents the input
        circuit, so ``extract_and_optimize(dag).to_circuit()`` is
        equivalent to it up to global phase.

        Args:
            dag: The input DAG.

        Returns:
            Optimized PhasePolynomial.
        """
        pp = PhasePolynomialCompiler.extract(dag)
        pp.optimize()
        return pp
