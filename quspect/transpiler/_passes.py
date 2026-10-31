"""Transpiler passes — decomposition, layout, routing, optimization."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quspect.circuit.circuit import QuantumCircuit
    from quspect.transpiler._target import CompilationTarget


class DecomposePass:
    """Decompose non-native gates down to the target's native gate set.

    Uses :func:`quspect.ir.compilation.decompose_deep` to recursively apply
    the decomposition rules (``cx → h·cz·h``, ``cz → h·cx·h``, ``swap → 3
    cx``, ``ccx → 6 cx``, ``rx → h·rz·h``, ``sx → h·s·h``, ...).  Gates that
    the rules cannot bring into the native set are synthesized from their
    matrix: single-qubit gates with
    :func:`~quspect.gates.synthesis.synthesize_one_qubit` in the native
    single-qubit gates (e.g. ``h → rz·sx·rz`` for ``{rz, sx, x}``),
    two-qubit gates by KAK synthesis
    (:func:`~quspect.synthesis.two_qubit.synthesize_two_qubit_ops`) followed
    by the rules.  Every substitution is exact up to global phase (KAK and
    single-qubit synthesis are verified to ``1e-9``, phase-aligned).

    Args:
        target: Compilation target whose ``native_gates`` are the goal set.
            An empty set disables decomposition.
        keep: Extra gate names left untouched although not native.  Pass
            ``{"ccx"}`` before :class:`~quspect.transpiler.SABRE` routing so
            that Toffolis are routed as three-qubit units and synthesized
            for the coupling they end up on (see :meth:`SABRE.solve`).

    Raises:
        ValueError: From :meth:`run`, if a gate cannot be expressed in the
            native gates (symbolic parameters without a rule, a non-universal
            single-qubit set, or no ``cz``/``cx`` entangler).

    Example:
        >>> from quspect import QuantumCircuit
        >>> from quspect.transpiler import CompilationTarget
        >>> target = CompilationTarget.from_coupling_map(2, [(0, 1)],
        ...     native_gates={"cx", "rz", "sx", "x"})
        >>> out = DecomposePass(target).run(QuantumCircuit().h(0).cz(0, 1))
        >>> sorted({i.gate.name for i in out}) <= ["cx", "rz", "sx", "x"]
        True
    """

    def __init__(
        self, target: CompilationTarget, *, keep: Collection[str] = ()
    ) -> None:
        self._native_gates = (
            set(target.native_gates) | set(keep) if target.native_gates else set()
        )

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Decompose *circuit* to the target's native gate set.

        Args:
            circuit: The circuit to decompose.

        Returns:
            A new circuit containing only native gates (and the ``keep``
            gates).

        Raises:
            ValueError: If some gate cannot be expressed in the native gates.
        """
        from quspect.ir.compilation import decompose_deep

        supported = self._native_gates
        if not supported:
            return circuit
        unsupported = _unsupported(circuit, supported)
        if not unsupported:
            return circuit
        if any(n >= 2 for _, n in unsupported) and not supported & {"cx", "cz"}:
            raise ValueError(
                "Cannot decompose multi-qubit gates "
                f"{sorted({g for g, n in unsupported if n >= 2})}: the target basis "
                f"{sorted(supported)} has neither cx nor cz."
            )
        out = decompose_deep(circuit, supported_gates=supported)
        if _unsupported(out, supported):
            out = decompose_deep(
                _synthesize_unsupported(out, supported), supported_gates=supported
            )
            out = _synthesize_unsupported(out, supported, one_qubit_only=True)
        left = _unsupported(out, supported)
        if left:
            raise ValueError(
                "Cannot express gate(s) "
                f"{sorted({g for g, _ in left})} in the target basis "
                f"{sorted(supported)} (no decomposition rule, symbolic parameters, "
                "or a single-qubit set that is not universal)."
            )
        return out


def _unsupported(circuit: QuantumCircuit, supported: set[str]) -> set[tuple[str, int]]:
    """``(name, num_qubits)`` of the gates of *circuit* not in *supported*."""
    return {
        (i.gate.name, len(i.qubits))
        for i in circuit
        if i.is_gate and i.gate is not None and i.gate.name not in supported
    }


def _synthesize_unsupported(
    circuit: QuantumCircuit, supported: set[str], *, one_qubit_only: bool = False
) -> QuantumCircuit:
    """Replace unsupported 1q/2q gates with a numeric matrix by synthesis.

    Single-qubit gates go straight into the supported single-qubit gates;
    two-qubit gates are KAK-synthesized into ``{cz, h, rz, x}`` (lowered to
    the target by the rules afterwards).  Gates that cannot be synthesized
    are kept.
    """
    import numpy as np

    from quspect.circuit.circuit import QuantumCircuit
    from quspect.circuit.instruction import CircuitInstruction, InstructionType
    from quspect.gates._matrix_utils import gate_matrix
    from quspect.gates.synthesis import synthesize_one_qubit
    from quspect.synthesis.two_qubit import (
        ops_to_instructions,
        synthesize_two_qubit_ops,
    )

    out: list[CircuitInstruction] = []
    for ins in circuit:
        gate = ins.gate
        if (
            not ins.is_gate
            or gate is None
            or gate.name in supported
            or len(ins.qubits) > (1 if one_qubit_only else 2)
        ):
            out.append(ins)
            continue
        try:
            mat = np.asarray(gate_matrix(gate), dtype=complex)
        except (KeyError, ValueError, TypeError):
            out.append(ins)
            continue
        if len(ins.qubits) == 1:
            seq = synthesize_one_qubit(mat, supported)
            if seq is None:
                out.append(ins)
                continue
            out.extend(
                CircuitInstruction(type=InstructionType.GATE, gate=g, qubits=ins.qubits)
                for g in seq
            )
            continue
        try:
            ops = synthesize_two_qubit_ops(mat)
        except (RuntimeError, ValueError):
            out.append(ins)
            continue
        a, b = ins.qubits
        out.extend(ops_to_instructions(ops, (a, b)))
    return QuantumCircuit._from_instruction_snapshot(out, name=circuit.name)


class LayoutRoutingPass:
    """Qubit layout and SWAP routing via SABRE.

    Finds an initial logical→physical mapping and inserts SWAP gates so
    that all two-qubit gates in the circuit act on adjacent physical qubits.

    Args:
        target: Compilation target describing the coupling graph.
        iter_num: Forward/backward refinement iterations per layout trial.
        seed: Optional random seed for reproducible results.
        initial_mapping: Optional fixed ``logical -> physical`` layout;
            disables the layout search.
        layout_trials: Number of random SABRE layout trials (``None`` uses
            the :class:`SABRE` default).
        exact_layout: Try a SWAP-free VF2++ exact layout before the SABRE
            layout trials (see :meth:`SABRE.solve`).
        elide_permutations: Absorb SWAP-equivalent blocks and trailing
            routing SWAPs into the returned final mapping (see
            :meth:`SABRE.solve`).

    Example:
        >>> target = CompilationTarget.from_coupling_map(3, [(0, 1), (1, 2)])
        >>> pass_ = LayoutRoutingPass(target, iter_num=3, seed=42)
        >>> routed, initial_map, final_map = pass_.run(circuit)
    """

    def __init__(
        self,
        target: CompilationTarget,
        *,
        iter_num: int = 3,
        seed: int | None = None,
        initial_mapping: Sequence[int] | None = None,
        layout_trials: int | None = None,
        exact_layout: bool = True,
        elide_permutations: bool = True,
    ) -> None:
        self._target = target
        self._exact_layout = exact_layout
        self._elide_permutations = elide_permutations
        self._iter_num = iter_num
        self._seed = seed
        self._initial_mapping = initial_mapping
        self._layout_trials = layout_trials

    @property
    def target(self) -> CompilationTarget:
        return self._target

    def run(
        self, circuit: QuantumCircuit
    ) -> tuple[QuantumCircuit, list[int], list[int]]:
        """Run SABRE layout and routing.

        Returns:
            ``(routed_circuit, initial_mapping, final_mapping)``.
        """
        from quspect.transpiler._mapping import DEFAULT_LAYOUT_TRIALS, SABRE

        sabre = SABRE(self._target)
        return sabre.solve(
            circuit,
            iter_num=self._iter_num,
            seed=self._seed,
            initial_mapping=self._initial_mapping,
            layout_trials=(
                DEFAULT_LAYOUT_TRIALS
                if self._layout_trials is None
                else self._layout_trials
            ),
            exact_layout=self._exact_layout,
            elide_permutations=self._elide_permutations,
        )


class OptimizePass:
    """Post-transpile optimization: gate cancellation, rotation merging, fusion.

    Runs the standard DAG optimization pipeline on the circuit to eliminate
    redundant gates introduced during decomposition and routing. The
    pipeline is iterated until the circuit structure stops changing, so
    commutation-enabled cancellations (e.g. ``cx · rz(control) · cx`` →
    ``rz``) are found. Identity-equivalent gates (``rz(0)``, ``rz(2π)``,
    ``u3(0, 0, 0)``…) are removed as part of the pipeline.

    Without ``basis_gates`` every run of single-qubit gates is fused into
    one ``u3``. With ``basis_gates`` the fusion is basis-aware: a run is
    resynthesized directly into the basis (e.g. ``rz·h·rz·h·rz`` for an
    ``{h, rz, cz, …}`` basis) and replaced only when that is strictly
    cheaper, so the output stays in the basis and never grows.

    Args:
        max_iterations: Maximum number of pipeline iterations passed to
            :meth:`PassManager.run`.
        basis_gates: Optional target gate set used by the single-qubit
            resynthesis. ``None`` keeps the ``u3`` fusion behaviour.

    Example:
        >>> OptimizePass().run(circuit)
        >>> OptimizePass(basis_gates={"cz", "h", "rz", "s", "sdg", "t",
        ...                           "tdg", "x", "y", "z"}).run(circuit)
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        basis_gates: Collection[str] | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.basis_gates: frozenset[str] | None = (
            None if basis_gates is None else frozenset(basis_gates)
        )

    @classmethod
    def from_target(
        cls, target: CompilationTarget, *, max_iterations: int = 50
    ) -> OptimizePass:
        """Create an OptimizePass that resynthesizes into a target's gate set.

        Args:
            target: The compilation target. Its ``native_gates`` become the
                basis; an empty set means no basis restriction.
            max_iterations: Maximum number of pipeline iterations.

        Returns:
            A configured :class:`OptimizePass`.

        Example:
            >>> target = CompilationTarget.from_coupling_map(
            ...     2, [(0, 1)], native_gates={"cz", "h", "rz"})
            >>> OptimizePass.from_target(target).basis_gates == {"cz", "h", "rz"}
            True
        """
        native = target.native_gates
        return cls(max_iterations=max_iterations, basis_gates=native or None)

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize *circuit* via DAG passes.

        Args:
            circuit: The circuit to optimize.

        Returns:
            An optimized circuit.

        Raises:
            RuntimeError: If the pass pipeline does not converge within
                ``max_iterations`` iterations.
        """
        from quspect.dag.compiler import CircuitCompiler
        from quspect.dag.dag_to_circuit import DAGToCircuitCompiler
        from quspect.dag.passes import (
            CommutationAnalysis,
            EliminateDiagonalBeforeMeasurement,
            GateCancellation,
            PassManager,
            RemoveIdentityGates,
            RotationMerging,
            SingleQubitGateFusion,
        )

        dag = CircuitCompiler().compile(circuit)

        pm = PassManager(
            [
                GateCancellation(),
                CommutationAnalysis(),
                RotationMerging(),
                SingleQubitGateFusion(basis_gates=self.basis_gates),
                EliminateDiagonalBeforeMeasurement(),
                RemoveIdentityGates(),
            ]
        )
        optimized_dag = pm.run(dag, max_iterations=self.max_iterations)

        return DAGToCircuitCompiler().compile(optimized_dag)


class TranspilerPipeline:
    """Orchestrate a sequence of transpiler passes.

    Example:
        >>> pipeline = TranspilerPipeline([
        ...     DecomposePass(target),
        ...     LayoutRoutingPass(target),
        ... ])
        >>> result = pipeline.run(circuit)
    """

    def __init__(self, passes: list[Any]) -> None:
        self._passes = passes

    def run(self, circuit: QuantumCircuit) -> Any:
        """Run all passes in sequence.

        Returns:
            The result of the last pass.
        """
        current: Any = circuit
        for p in self._passes:
            current = p.run(current)
        return current
