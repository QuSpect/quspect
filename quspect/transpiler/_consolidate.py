"""Two-qubit block consolidation and KAK resynthesis.

:class:`ConsolidateBlocksPass` collects maximal runs of gates acting on the
same pair of qubits (including interleaved single-qubit gates on those
qubits), multiplies them into a ``4×4`` unitary and resynthesizes it with
:func:`quspect.synthesis.two_qubit.synthesize_two_qubit_ops`, which uses the
minimal number (0–3) of CZ gates — or of the target's native entangler
(e.g. CX), see :class:`~quspect.synthesis.two_qubit.TwoQubitBasis`. A block
is replaced only when the new circuit has fewer two-qubit gates (ties broken
by total gate count).

The pass works directly on the flat instruction list of a
:class:`~quspect.circuit.circuit.QuantumCircuit` in a single linear sweep, so
it does not depend on :class:`~quspect.dag.DAGCircuit` internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quspect.circuit.circuit import QuantumCircuit
from quspect.circuit.instruction import CircuitInstruction
from quspect.synthesis._memo import BoundedCache, matrix_key
from quspect.synthesis.two_qubit import (
    DEFAULT_ATOL,
    SynthOp,
    TwoQubitBasis,
    _approximate_two_qubit_ops,
    _ops_layer_matrix,
    euler_ops,
    num_cz_required,
    ops_to_instructions,
    synthesize_two_qubit_approx,
    synthesize_two_qubit_ops,
)

if TYPE_CHECKING:
    from quspect.gates.gate import Gate
    from quspect.synthesis.two_qubit import ApproximationError
    from quspect.transpiler._target import CompilationTarget

__all__ = ["ConsolidateBlocksPass", "ConsolidationStats"]

logger = logging.getLogger(__name__)

#: Gates the resynthesized blocks are written in by default (CZ targets).
SYNTHESIS_BASIS = frozenset({"cz", "h", "rz", "x"})

#: Known CZ costs of common two-qubit gates (others are computed via KAK).
_KNOWN_2Q_COST = {"cz": 1, "cx": 1, "swap": 3}

_I2 = np.eye(2, dtype=complex)
_REV = [0, 2, 1, 3]  # basis permutation swapping the two qubits


#: Acceptance tolerance of a re-synthesized single-qubit run (phase-aligned,
#: element-wise; see :mod:`quspect.synthesis.two_qubit` for the budget).
_VERIFY_1Q_ATOL = 1e-9


#: Exact-input cache of :func:`_verified_euler_ops` (``False`` = no result).
_EULER_CACHE: BoundedCache[tuple[object, ...], tuple[SynthOp, ...] | bool] = (
    BoundedCache()
)


def _verified_euler_ops(
    u: np.ndarray, atol: float, euler: str = "zxz"
) -> list[SynthOp] | None:
    """Euler-synthesize ``u`` (ZXZ or ZSX) and verify the emitted gates linearly.

    Rotations within ``atol`` of zero are dropped first; if the dropped
    angles cost more than :data:`_VERIFY_1Q_ATOL` (``max|u - e^{iφ}v|``),
    the run is re-synthesized with exact angles. Returns ``None`` if neither
    attempt verifies (the caller then keeps the original run).  Memoized on
    the exact bytes of ``u``.
    """
    key = (matrix_key(u), atol, euler)
    cached = _EULER_CACHE.get(key)
    if cached is not None:
        return list(cached) if isinstance(cached, tuple) else None
    ops = _verified_euler_ops_uncached(u, atol, euler)
    _EULER_CACHE.put(key, False if ops is None else tuple(ops))
    return ops


def _verified_euler_ops_uncached(
    u: np.ndarray, atol: float, euler: str = "zxz"
) -> list[SynthOp] | None:
    """Uncached core of :func:`_verified_euler_ops`."""
    for tol in (atol, min(atol, 1e-12)):
        ops = euler_ops(u, 0, basis=euler, atol=tol)
        v = _ops_layer_matrix(ops)
        overlap = complex(np.vdot(v, u))
        phase = overlap / abs(overlap) if abs(overlap) > 1e-300 else 1.0
        if float(np.max(np.abs(u - phase * v))) <= _VERIFY_1Q_ATOL:
            return ops
    return None


@dataclass
class ConsolidationStats:
    """Counters describing what the last :meth:`ConsolidateBlocksPass.run` did.

    Attributes:
        blocks_examined: Blocks with at least two CZ-equivalent gates.
        blocks_replaced: Blocks that were resynthesized.
        two_qubit_before: CZ-equivalent cost of the examined blocks.
        two_qubit_after: CZ cost of the examined blocks after the pass.
        blocks_approximated: Blocks replaced by an approximation (only with
            ``max_infidelity > 0``).
        approximation_error: Sum of the operator-norm distance bounds
            (:attr:`~quspect.synthesis.two_qubit.ApproximationError.distance`)
            of the approximated blocks: the output differs from the input by
            at most this much in operator norm, up to global phase (plus the
            ``1e-9`` element-wise verification of the exact resyntheses).
        max_block_infidelity: Largest average gate infidelity of an
            approximated block.

    Example:
        >>> ConsolidationStats().blocks_replaced
        0
    """

    blocks_examined: int = 0
    blocks_replaced: int = 0
    two_qubit_before: int = 0
    two_qubit_after: int = 0
    blocks_approximated: int = 0
    approximation_error: float = 0.0
    max_block_infidelity: float = 0.0


@dataclass
class _Block:
    qubits: tuple[int, int]
    members: list[int]
    first_2q: int
    cost: int = 0
    num_2q: int = 0


class ConsolidateBlocksPass:
    """Resynthesize two-qubit blocks with the minimal number of CZ gates.

    The pass sweeps the instruction list once. Each qubit is either *free*
    or owned by one open block on a qubit pair. A two-qubit gate on the
    block's pair (or a single-qubit gate on one of its qubits) joins the
    block; any other operation touching one of the qubits closes it.
    Single-qubit gates seen on a free qubit are held back and absorbed into
    the next block that starts on that qubit. Measurements, barriers, noise
    channels, gates with symbolic parameters, gates of three or more qubits
    and gates without a known matrix act as block boundaries.

    A closed block whose CZ-equivalent cost (``cz``/``cx`` = 1, ``swap`` = 3,
    other two-qubit gates = their KAK CZ count) is at least 2 is multiplied
    out and resynthesized; the replacement is kept only if it has fewer CZ
    gates, or the same number and fewer gates overall. Replacements use the
    ``{cz, h, rz, x}`` gates by default; for a target the synthesis basis is
    picked from its native gates by
    :meth:`~quspect.synthesis.two_qubit.TwoQubitBasis.from_gates` (``cz``
    or ``cx`` as entangler, ``{rz, h, x}`` or ``{rz, sx, x}`` for the
    single-qubit layers), with the same entangler counts. Routed ``SWAP``
    gates (or their three-CNOT expansion) adjacent to other gates on the
    same pair are therefore absorbed into a single block of at most three
    entanglers.

    The sweep is linear in the circuit size (each block costs a constant
    amount of ``4×4`` linear algebra).

    Args:
        target: Optional compilation target. When it declares native gates,
            they must contain a synthesis basis: ``cz`` or ``cx`` plus
            ``{rz, h, x}`` or ``{rz, sx, x}``.
        atol: Tolerance on Weyl coordinates used to detect when fewer CZ
            gates suffice.
        fuse_single_qubit: If True (default), finish with a linear sweep that
            re-synthesizes every run of two or more adjacent single-qubit
            gates on a wire as a ZXZ Euler sequence when that is shorter
            (this merges the outer layers of neighbouring blocks).
        basis: Explicit synthesis basis; overrides the one derived from
            ``target``.
        max_infidelity: Per-block approximation budget (default ``0``:
            exact).  When positive, a block may be replaced by a circuit with
            fewer entanglers whose average gate infidelity to the block is at
            most ``max_infidelity`` (see
            :func:`~quspect.synthesis.two_qubit.synthesize_two_qubit_approx`);
            :attr:`ConsolidationStats.approximation_error` then bounds the
            total change of the circuit.
        max_entanglers: Approximations keep at most this many entanglers
            per block (default ``2``: any reduction; ``0``: only blocks whose
            interaction is dropped entirely, which removes a qubit-pair
            interaction from the circuit).

    Raises:
        ValueError: If the target's native gate set contains no synthesis
            basis, or ``max_infidelity`` is negative.

    Example:
        >>> from quspect.circuit.circuit import QuantumCircuit
        >>> qc = QuantumCircuit()
        >>> _ = qc.cx(0, 1).cx(1, 0).cx(0, 1).cz(0, 1)  # SWAP · CZ
        >>> out = ConsolidateBlocksPass().run(qc)
        >>> sum(1 for ins in out if ins.gate.name == "cz")  # iSWAP-class
        2
    """

    def __init__(
        self,
        target: CompilationTarget | None = None,
        *,
        atol: float = DEFAULT_ATOL,
        fuse_single_qubit: bool = True,
        basis: TwoQubitBasis | None = None,
        max_infidelity: float = 0.0,
        max_entanglers: int = 2,
    ) -> None:
        if not max_infidelity >= 0:
            raise ValueError(
                f"max_infidelity must be non-negative, got {max_infidelity!r}."
            )
        self._max_infidelity = float(max_infidelity)
        self._max_entanglers = int(max_entanglers)
        if basis is None and target is not None and target.native_gates:
            basis = TwoQubitBasis.from_gates(target.native_gates)
            if basis is None:
                raise ValueError(
                    "ConsolidateBlocksPass requires a native entangler (cz or cx) "
                    "and a single-qubit basis {rz, h, x} or {rz, sx, x}; target has "
                    f"{sorted(target.native_gates)}."
                )
        self._basis = TwoQubitBasis() if basis is None else basis
        self._atol = atol
        self._fuse_1q = fuse_single_qubit
        self._matrix_cache: dict[Gate, np.ndarray | None] = {}
        self._cost_cache: dict[Gate, int] = {}
        # (id(matrix), slot) -> (matrix, 4×4 lifted matrix); the matrix is
        # kept alive so its id cannot be reused.
        self._lifted_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.stats = ConsolidationStats()

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _matrix(self, gate: Gate) -> np.ndarray | None:
        try:
            return self._matrix_cache[gate]
        except KeyError:
            pass
        except TypeError:  # unhashable parameter value
            return self._compute_matrix(gate)
        mat = self._compute_matrix(gate)
        self._matrix_cache[gate] = mat
        return mat

    @staticmethod
    def _compute_matrix(gate: Gate) -> np.ndarray | None:
        from quspect.gates._matrix_utils import gate_matrix

        try:
            return np.asarray(gate_matrix(gate), dtype=complex)
        except (KeyError, ValueError, TypeError):
            return None

    def _cost(self, gate: Gate, mat: np.ndarray) -> int:
        known = _KNOWN_2Q_COST.get(gate.name)
        if known is not None and not gate.params:
            return known
        cached = self._cost_cache.get(gate)
        if cached is None:
            cached = num_cz_required(mat, atol=self._atol)
            self._cost_cache[gate] = cached
        return cached

    # ------------------------------------------------------------------
    # Gate-level approximation
    # ------------------------------------------------------------------

    def approximate_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Replace individual weak two-qubit gates by cheaper approximations.

        Every parametrized two-qubit gate with a known matrix (``cp``,
        ``crz``, ``rzz``, ...) is resynthesized with fewer entanglers when
        that changes it by at most the pass's ``max_infidelity`` (average
        gate infidelity); e.g. ``CP(θ)`` with ``θ²/20 ≤ max_infidelity``
        becomes ``rz(θ/2) ⊗ rz(θ/2)``.  Unlike :meth:`run`, no blocks are
        formed, so the rest of the circuit (and its gate names) is left
        untouched.  :attr:`stats` is reset and records the replaced gates
        as approximated blocks.  Without an approximation budget the
        circuit is returned unchanged.

        Args:
            circuit: The circuit to approximate. It is not modified.

        Returns:
            The approximated circuit (``circuit`` itself if nothing changed).

        Example:
            >>> from quspect.circuit.circuit import QuantumCircuit
            >>> qc = QuantumCircuit().cp(1e-6, 0, 1).cp(0.5, 0, 1)
            >>> p = ConsolidateBlocksPass(max_infidelity=1e-12)
            >>> [i.gate.name for i in p.approximate_gates(qc)]
            ['rz', 'rz', 'cp']
            >>> p.stats.blocks_approximated, p.stats.approximation_error < 3e-7
            (1, True)
        """
        self.stats = ConsolidationStats()
        if self._max_infidelity <= 0:
            return circuit
        memo: dict[object, tuple[list[SynthOp], ApproximationError] | None] = {}
        out: list[CircuitInstruction] = []
        changed = False
        for ins in circuit:
            gate = ins.gate
            if (
                not ins.is_gate
                or gate is None
                or len(ins.qubits) != 2
                or not gate.params
                or gate.name in _KNOWN_2Q_COST
            ):
                out.append(ins)
                continue
            mat = self._matrix(gate)
            if mat is None:
                out.append(ins)
                continue
            key = matrix_key(mat)
            if key in memo:
                found = memo[key]
            else:
                found = _approximate_two_qubit_ops(
                    mat,
                    self._max_infidelity,
                    self._atol,
                    self._basis,
                    self._max_entanglers,
                )
                memo[key] = found
            if found is None:
                out.append(ins)
                continue
            ops, err = found
            out.extend(ops_to_instructions(ops, (ins.qubits[0], ins.qubits[1])))
            changed = True
            self.stats.blocks_approximated += 1
            self.stats.approximation_error += err.distance
            self.stats.max_block_infidelity = max(
                self.stats.max_block_infidelity, err.infidelity
            )
        if not changed:
            return circuit
        return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Consolidate and resynthesize two-qubit blocks of *circuit*.

        Args:
            circuit: The circuit to optimize. It is not modified.

        Returns:
            A new circuit, equivalent to *circuit* up to global phase, whose
            two-qubit gate count is never larger than the input's.

        Example:
            >>> from quspect.circuit.circuit import QuantumCircuit
            >>> qc = QuantumCircuit()
            >>> _ = qc.cp(0.3, 0, 1).cp(0.5, 0, 1)
            >>> from quspect.ir.compilation import decompose_deep
            >>> native = decompose_deep(qc, supported_gates={"cz", "h", "rz"})
            >>> out = ConsolidateBlocksPass().run(native)
            >>> sum(1 for ins in out if ins.gate.name == "cz")
            2
        """
        self.stats = ConsolidationStats()
        instrs = circuit.instructions
        slots: list[list[CircuitInstruction] | CircuitInstruction | None] = list(instrs)
        mats: list[np.ndarray | None] = [None] * len(instrs)
        open_block: dict[int, _Block] = {}
        pending: dict[int, list[int]] = {}

        def close(block: _Block) -> None:
            for q in block.qubits:
                if open_block.get(q) is block:
                    del open_block[q]
            self._finalize(block, instrs, mats, slots)

        for idx, ins in enumerate(instrs):
            gate = ins.gate
            mat = None
            if ins.is_gate and gate is not None and len(ins.qubits) <= 2:
                mat = self._matrix(gate)
            if mat is None:
                # Opaque operation: boundary on all its qubits.
                qubits = ins.qubits
                if not qubits and ins.is_barrier:
                    for blk in list({id(b): b for b in open_block.values()}.values()):
                        close(blk)
                    pending.clear()
                    continue
                for q in qubits:
                    hit = open_block.get(q)
                    if hit is not None:
                        close(hit)
                    pending.pop(q, None)
                continue

            mats[idx] = mat
            if len(ins.qubits) == 1:
                q = ins.qubits[0]
                owner = open_block.get(q)
                if owner is not None:
                    owner.members.append(idx)
                else:
                    pending.setdefault(q, []).append(idx)
                continue

            a, b = ins.qubits
            cost = self._cost(gate, mat) if gate is not None else 1
            cur = open_block.get(a)
            if cur is not None and open_block.get(b) is cur:
                cur.members.append(idx)
                cur.cost += cost
                cur.num_2q += 1
                continue
            for q in (a, b):
                old = open_block.get(q)
                if old is not None:
                    close(old)
            members = sorted(pending.pop(a, []) + pending.pop(b, []))
            members.append(idx)
            new = _Block(
                qubits=(a, b), members=members, first_2q=idx, cost=cost, num_2q=1
            )
            open_block[a] = new
            open_block[b] = new

        for blk in list({id(b): b for b in open_block.values()}.values()):
            close(blk)

        flat: list[CircuitInstruction] = []
        for slot in slots:
            if slot is None:
                continue
            if isinstance(slot, list):
                flat.extend(slot)
            else:
                flat.append(slot)
        if self._fuse_1q:
            flat = self._fuse_single_qubit_runs(flat)
        logger.debug(
            "ConsolidateBlocksPass: %d/%d blocks replaced, 2q cost %d -> %d",
            self.stats.blocks_replaced,
            self.stats.blocks_examined,
            self.stats.two_qubit_before,
            self.stats.two_qubit_after,
        )
        return QuantumCircuit._from_instruction_snapshot(flat, name=circuit.name)

    def _finalize(
        self,
        block: _Block,
        instrs: tuple[CircuitInstruction, ...],
        mats: list[np.ndarray | None],
        slots: list[list[CircuitInstruction] | CircuitInstruction | None],
    ) -> None:
        if block.cost < 2:
            return
        self.stats.blocks_examined += 1
        self.stats.two_qubit_before += block.cost
        qa, _ = block.qubits
        u = np.eye(4, dtype=complex)
        lifted = self._lifted_cache
        for m in block.members:
            mat = mats[m]
            assert mat is not None
            qs = instrs[m].qubits
            if len(qs) == 2 and qs[0] == qa:
                full = mat
            else:
                # 4×4 形式只取决于矩阵本身与它作用的位置：按矩阵对象缓存
                # （矩阵来自按门缓存的 _matrix，同一个门总是同一个对象）。
                slot = 0 if len(qs) == 2 else (1 if qs[0] == qa else 2)
                hit = lifted.get((id(mat), slot))
                if hit is not None and hit[0] is mat:
                    full = hit[1]
                else:
                    if slot == 0:
                        full = mat[np.ix_(_REV, _REV)]
                    elif slot == 1:
                        full = np.kron(mat, _I2)
                    else:
                        full = np.kron(_I2, mat)
                    lifted[(id(mat), slot)] = (mat, full)
            u = full @ u
        err = None
        try:
            if self._max_infidelity > 0:
                ops, err = synthesize_two_qubit_approx(
                    u,
                    self._max_infidelity,
                    atol=self._atol,
                    basis=self._basis,
                    max_entanglers=self._max_entanglers,
                )
            else:
                ops = synthesize_two_qubit_ops(u, atol=self._atol, basis=self._basis)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            logger.warning("Two-qubit resynthesis failed, keeping block: %s", exc)
            self.stats.two_qubit_after += block.cost
            return
        new_2q = sum(1 for _, local, _ in ops if len(local) == 2)
        # Two-qubit count first, total gate count as tie-breaker: a block
        # with fewer CZ is accepted even if it needs more single-qubit gates
        # (on a 16-qubit benchmark sample, also requiring "no more gates in total"
        # costs ~7% more 2q gates, ~36% on QFT-16/line, for ~1.4% fewer
        # gates overall). With an equal CZ count the total must shrink.
        if new_2q < block.cost or (
            new_2q == block.cost
            and block.num_2q == block.cost
            and len(ops) < len(block.members)
        ):
            for m in block.members:
                slots[m] = None
            slots[block.first_2q] = ops_to_instructions(ops, block.qubits)
            self.stats.blocks_replaced += 1
            self.stats.two_qubit_after += new_2q
            if err is not None:
                self.stats.blocks_approximated += 1
                self.stats.approximation_error += err.distance
                self.stats.max_block_infidelity = max(
                    self.stats.max_block_infidelity, err.infidelity
                )
        else:
            self.stats.two_qubit_after += block.cost

    def _fuse_single_qubit_runs(
        self, instrs: list[CircuitInstruction]
    ) -> list[CircuitInstruction]:
        """Re-synthesize maximal single-qubit runs when it shortens them."""
        slots: list[list[CircuitInstruction] | CircuitInstruction | None] = list(instrs)
        runs: dict[int, list[int]] = {}

        def flush(q: int) -> None:
            run = runs.pop(q, None)
            if run is None or len(run) < 2:
                return
            u = np.eye(2, dtype=complex)
            for i in run:
                gate = instrs[i].gate
                assert gate is not None
                mat = self._matrix(gate)
                assert mat is not None
                u = mat @ u
            ops = _verified_euler_ops(u, self._atol, self._basis.euler)
            if ops is None or len(ops) >= len(run):
                return
            for i in run:
                slots[i] = None
            slots[run[0]] = ops_to_instructions(ops, (q, q))

        for idx, ins in enumerate(instrs):
            gate = ins.gate
            if (
                ins.is_gate
                and gate is not None
                and len(ins.qubits) == 1
                and self._matrix(gate) is not None
            ):
                runs.setdefault(ins.qubits[0], []).append(idx)
                continue
            if not ins.qubits and ins.is_barrier:
                for q in list(runs):
                    flush(q)
                continue
            for q in ins.qubits:
                flush(q)
        for q in list(runs):
            flush(q)

        flat: list[CircuitInstruction] = []
        for slot in slots:
            if slot is None:
                continue
            if isinstance(slot, list):
                flat.extend(slot)
            else:
                flat.append(slot)
        return flat
