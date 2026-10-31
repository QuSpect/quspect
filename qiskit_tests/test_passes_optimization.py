"""Optimization effectiveness of ``quspect.dag.passes`` and the PassManager.

Each pass is checked on circuits with a known reducible structure: the gate
count must drop to the expected value and the unitary must be unchanged.
The effect is then compared with the closest qiskit pass:

* ``GateCancellation`` vs ``InverseCancellation()`` (default gate list);
* ``CommutationAnalysis + GateCancellation`` (``+ RotationMerging`` in the
  aggregate) vs ``CommutativeCancellation()``;
* ``SingleQubitGateFusion`` vs ``Optimize1qGatesDecomposition(basis=["u"])``;
* ``EliminateDiagonalBeforeMeasurement`` vs
  ``RemoveDiagonalGatesBeforeMeasure()``.

quspect is not required to match qiskit's gate counts exactly; the
comparison tests only require it not to be markedly worse (``<= 1.5x + 2``)
and print the aggregated numbers (run with ``-s`` to see them).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import pytest
import qiskit
from qiskit.transpiler import PassManager as QiskitPassManager
from qiskit.transpiler.basepasses import BasePass as QiskitBasePass
from qiskit.transpiler.passes import (
    CommutativeCancellation,
    InverseCancellation,
    Optimize1qGatesDecomposition,
    RemoveDiagonalGatesBeforeMeasure,
)

import quspect
from qiskit_tests._dag_helpers import (
    apply_pass,
    apply_passes,
    qiskit_gate_counts,
    quspect_gate_counts,
)
from qiskit_tests.helpers import (
    assert_unitary_equal,
    qiskit_circuit_unitary,
    qiskit_probabilities,
    to_qiskit,
)
from qiskit_tests.test_passes_semantics import _motif_circuit
from quspect.dag import DAGCircuit, DAGEdge, DAGNode, DAGNodeType
from quspect.dag.passes import (
    BasePass,
    CommutationAnalysis,
    EliminateDiagonalBeforeMeasurement,
    GateCancellation,
    PassManager,
    PassStats,
    RotationMerging,
    SingleQubitGateFusion,
)

pytestmark = pytest.mark.qiskit_ref

Circ = quspect.QuantumCircuit


def _n_gates(circ: quspect.QuantumCircuit) -> int:
    """Number of gates (measure / barrier excluded)."""
    return sum(quspect_gate_counts(circ).values())


def _qiskit_n_gates(qc: qiskit.QuantumCircuit, passes: list[QiskitBasePass]) -> int:
    """Gate count after running qiskit passes on ``qc``."""
    return sum(qiskit_gate_counts(QiskitPassManager(passes).run(qc)).values())


def _check_reduction(
    passes: list[Callable[[], BasePass]],
    circ: quspect.QuantumCircuit,
    expected: int,
) -> quspect.QuantumCircuit:
    """Run passes; assert the gate count and the unitary (up to phase)."""
    out = (
        apply_passes(passes, circ)[0]
        if len(passes) > 1
        else apply_pass(passes[0], circ)
    )
    assert _n_gates(out) == expected, [str(i) for i in out.instructions]
    n = circ.num_qubits
    assert_unitary_equal(
        qiskit_circuit_unitary(out, num_qubits=n),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )
    return out


# =============================================================================
# GateCancellation
# =============================================================================

_CANCEL_CASES: dict[str, tuple[quspect.QuantumCircuit, int]] = {
    "hh": (Circ().h(0).h(0), 0),
    "xyz_pairs": (Circ().x(0).x(0).y(1).y(1).z(0).z(0), 0),
    "id_id": (Circ().id(0).id(0).h(1), 1),
    "s_sdg_t_tdg": (Circ().s(0).sdg(0).tdg(0).t(0), 0),
    "nested": (Circ().x(0).h(0).s(0).sdg(0).h(0).x(0), 0),
    "odd_chain": (Circ().h(0).h(0).h(0), 1),
    "rotations": (
        Circ().rx(0.3, 0).rx(-0.3, 0).ry(1.1, 0).ry(-1.1, 0).rz(2, 1).rz(-2, 1),
        0,
    ),
    "p_pair": (Circ().p(0.4, 0).p(-0.4, 0), 0),
    "two_qubit": (Circ().cx(0, 1).cx(0, 1).cz(1, 2).cz(1, 2).swap(0, 2).swap(0, 2), 0),
    "controlled_rot": (
        Circ().crx(0.5, 0, 1).crx(-0.5, 0, 1).cry(1, 1, 0).cry(-1, 1, 0),
        0,
    ),
    "crz_cp": (Circ().crz(0.5, 0, 1).crz(-0.5, 0, 1).cp(2, 1, 0).cp(-2, 1, 0), 0),
    "reversed_cx_kept": (Circ().cx(0, 1).cx(1, 0), 2),
    "blocked_on_other_wire": (Circ().cx(0, 1).rz(0.3, 1).cx(0, 1), 3),
    "barrier_blocks": (Circ().h(0).barrier(0).h(0), 2),
    "not_inverse_angles": (Circ().rx(0.3, 0).rx(0.3, 0), 2),
    "s_s_kept": (Circ().s(0).s(0), 2),
    "different_operands": (Circ().crz(0.5, 0, 1).crz(-0.5, 1, 0), 2),
}


@pytest.mark.parametrize("case", list(_CANCEL_CASES))
def test_gate_cancellation_expected_counts(case: str) -> None:
    """GateCancellation removes exactly the adjacent inverse pairs."""
    circ, expected = _CANCEL_CASES[case]
    _check_reduction([GateCancellation], circ, expected)


@pytest.mark.parametrize("case", list(_CANCEL_CASES))
def test_gate_cancellation_not_worse_than_inverse_cancellation(case: str) -> None:
    """On the hand-made cases quspect cancels at least as much as qiskit."""
    circ, _ = _CANCEL_CASES[case]
    mine = _n_gates(apply_pass(GateCancellation, circ))
    ref = _qiskit_n_gates(to_qiskit(circ), [InverseCancellation()])
    assert mine <= ref


# =============================================================================
# CommutationAnalysis (+ GateCancellation)
# =============================================================================

_COMMUTE_CASES: dict[str, tuple[quspect.QuantumCircuit, int]] = {
    "rz_on_control": (Circ().cx(0, 1).rz(0.3, 0).cx(0, 1), 1),
    "t_on_control": (Circ().cx(0, 1).t(0).s(0).cx(0, 1), 2),
    "x_on_target": (Circ().cx(0, 1).x(1).cx(0, 1), 1),
    "rx_on_target": (Circ().cx(0, 1).rx(0.2, 1).cx(0, 1), 1),
    "shared_control": (Circ().cx(0, 1).cx(0, 2).cx(0, 1), 1),
    "shared_target": (Circ().cx(0, 2).cx(1, 2).cx(0, 2), 1),
    "diag_family": (Circ().rz(0.3, 0).t(0).rz(-0.3, 0), 1),
    "cz_diag": (Circ().cz(0, 1).rz(0.4, 1).cz(0, 1), 1),
    "h_t_t_h": (Circ().h(0).z(0).s(0).z(0).h(0), 3),
    "rz_on_target_blocks": (Circ().cx(0, 1).rz(0.3, 1).cx(0, 1), 3),
    "x_on_control_blocks": (Circ().cx(0, 1).x(0).cx(0, 1), 3),
}


@pytest.mark.parametrize("case", list(_COMMUTE_CASES))
def test_commutation_enables_cancellation(case: str) -> None:
    """CommutationAnalysis + GateCancellation reach the expected count."""
    circ, expected = _COMMUTE_CASES[case]
    out = _check_reduction([CommutationAnalysis, GateCancellation], circ, expected)
    ref = _qiskit_n_gates(to_qiskit(circ), [CommutativeCancellation()])
    assert _n_gates(out) <= 1.5 * ref + 2


@pytest.mark.parametrize(
    "case", ["rz_on_control", "x_on_target", "shared_control", "cz_diag"]
)
def test_cancel_then_commute_pipeline_recancels_bug(case: str) -> None:
    """[GateCancellation, CommutationAnalysis] should iterate to the optimum."""
    circ, expected = _COMMUTE_CASES[case]
    _check_reduction([GateCancellation, CommutationAnalysis], circ, expected)


def test_commutation_alone_only_reorders() -> None:
    """CommutationAnalysis keeps every gate and makes the pair adjacent."""
    circ = Circ().cx(0, 1).rz(0.3, 0).cx(0, 1)
    dag = CommutationAnalysis().run(circ.to_dag())
    names = [
        dag.get_node(nid).instruction.gate.name  # type: ignore[union-attr]
        for nid in dag.nodes_on_wire("q0")
        if dag.get_node(nid).is_gate
    ]
    assert names in (["cx", "cx", "rz"], ["rz", "cx", "cx"])


def test_commutation_lookahead_limit() -> None:
    """Pairs farther apart than MAX_LOOKAHEAD commuting gates are not joined."""
    far = Circ().cx(0, 1)
    for _ in range(CommutationAnalysis.MAX_LOOKAHEAD + 1):
        far.t(0)
    far.cx(0, 1)
    out = apply_passes([CommutationAnalysis, GateCancellation], far)[0]
    assert quspect_gate_counts(out)["cx"] == 2
    near = Circ().cx(0, 1).t(0).t(0).cx(0, 1)
    out = apply_passes([CommutationAnalysis, GateCancellation], near)[0]
    assert quspect_gate_counts(out)["cx"] == 0


def test_commutation_blocked_by_cross_wire_dependency() -> None:
    """A reorder that would create a cycle via another wire is rolled back."""
    circ = Circ().cx(0, 1).cx(1, 2).rz(0.1, 0).cx(2, 0).cx(0, 1)
    out = apply_passes([CommutationAnalysis, GateCancellation], circ)[0]
    assert_unitary_equal(
        qiskit_circuit_unitary(out, num_qubits=3),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


# =============================================================================
# RotationMerging
# =============================================================================

_MERGE_CASES: dict[str, tuple[quspect.QuantumCircuit, int, str, float]] = {
    "rx_chain": (Circ().rx(0.3, 0).rx(0.5, 0).rx(-0.1, 0), 1, "rx", 0.7),
    "ry_pair": (Circ().ry(1.0, 0).ry(2.0, 0), 1, "ry", 3.0),
    "rz_pair": (Circ().rz(math.pi / 2, 0).rz(math.pi / 4, 0), 1, "rz", 3 * math.pi / 4),
    "p_u1_mixed": (Circ().p(0.2, 0).u1(0.3, 0).p(0.4, 0), 1, "p", 0.9),
    "rz_then_p": (Circ().rz(0.2, 0).p(0.3, 0), 1, "rz", 0.5),
    "p_then_rz": (Circ().p(0.2, 0).rz(0.3, 0), 1, "p", 0.5),
}


@pytest.mark.parametrize("case", list(_MERGE_CASES))
def test_rotation_merging_expected(case: str) -> None:
    """Same-axis rotations fold into one gate with the summed angle."""
    circ, expected, name, angle = _MERGE_CASES[case]
    out = _check_reduction([RotationMerging], circ, expected)
    (instr,) = out.instructions
    assert instr.gate is not None and instr.gate.name == name
    assert float(instr.gate.params[0]) == pytest.approx(angle)


def test_rotation_merging_respects_axes_and_boundaries() -> None:
    """Different axes, other qubits' gates and barriers block merging."""
    circ = Circ().rx(0.1, 0).ry(0.2, 0).rz(0.3, 0).rz(0.4, 1)
    _check_reduction([RotationMerging], circ, 4)
    circ = Circ().rz(0.1, 0).cx(0, 1).rz(0.2, 0)
    _check_reduction([RotationMerging], circ, 3)
    circ = Circ().rz(0.1, 0).barrier(0).rz(0.2, 0)
    _check_reduction([RotationMerging], circ, 2)
    circ = Circ().rz(0.1, 0).rz(0.2, 0).h(0).rz(0.3, 0).rz(0.4, 0).rz(0.5, 0)
    _check_reduction([RotationMerging], circ, 3)


# =============================================================================
# SingleQubitGateFusion
# =============================================================================


@pytest.mark.parametrize(
    "names",
    [
        ("h", "z", "s", "x"),
        ("x", "h", "x"),
        ("h", "t", "h", "tdg", "y"),
        ("s", "sdg"),
        ("u2", "u3", "rx", "ry", "rz"),
        ("p", "u1", "id", "h"),
    ],
)
def test_single_qubit_fusion_to_one_u3(names: tuple[str, ...]) -> None:
    """Any run of fixed 1q gates becomes exactly one u3 with the same unitary."""
    circ = Circ()
    params = {"u2": (0.3, -1.2), "u3": (2.0, 0.4, -0.9)}
    for k, name in enumerate(names):
        if name in params:
            getattr(circ, name)(*params[name], 0)
        elif name in ("rx", "ry", "rz", "p", "u1"):
            getattr(circ, name)(0.37 * (k + 1), 0)
        else:
            getattr(circ, name)(0)
    out = _check_reduction([SingleQubitGateFusion], circ, 1)
    assert out.instructions[0].gate is not None
    assert out.instructions[0].gate.name == "u3"


def test_single_qubit_fusion_identity_run_is_kept_as_u3() -> None:
    """quspect keeps U3(0,0,0) for an identity run; qiskit removes it."""
    circ = Circ().h(0).h(0)
    out = _check_reduction([SingleQubitGateFusion], circ, 1)
    assert out.instructions[0].gate is not None
    params = [float(p) for p in out.instructions[0].gate.params]
    assert params == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    ref = _qiskit_n_gates(to_qiskit(circ), [Optimize1qGatesDecomposition(basis=["u"])])
    assert ref == 0


def test_single_qubit_fusion_stops_at_two_qubit_gates() -> None:
    """Runs are split by multi-qubit gates and barriers."""
    circ = Circ().h(0).t(0).cx(0, 1).s(0).h(0).barrier(0).x(0).y(0).h(1).h(1)
    _check_reduction([SingleQubitGateFusion], circ, 5)


@pytest.mark.parametrize(
    "u",
    [
        (math.pi, 0.0, 0.0),  # theta = pi singular branch
        (math.pi, 1.3, -0.4),
        (0.0, 0.0, 1.1),  # theta = 0 branch
        (1e-12, 0.5, 0.5),
        (2.1, -2.5, 3.0),
    ],
)
def test_single_qubit_fusion_decomposition_branches(
    u: tuple[float, float, float],
) -> None:
    """Edge branches of the ZYZ decomposition (theta ~ 0, theta ~ pi)."""
    circ = Circ().u3(*u, 0).id(0)
    _check_reduction([SingleQubitGateFusion], circ, 1)
    circ = Circ().x(0).u3(*u, 0).h(0)
    _check_reduction([SingleQubitGateFusion], circ, 1)


# =============================================================================
# EliminateDiagonalBeforeMeasurement
# =============================================================================


def _diag_tail_circuit() -> quspect.QuantumCircuit:
    """H layer, entangler, then diagonal chains before terminal measurements."""
    circ = Circ().h(0).h(1).h(2).cx(0, 1).cry(0.7, 1, 2)
    circ.rz(0.3, 0).t(0).z(0).s(1).sdg(1).p(0.9, 1).u1(0.2, 2).tdg(2).id(2)
    circ.measure([0, 1, 2], [2, 0, 1])
    return circ


def test_diagonal_elimination_removes_all_tail_diagonals() -> None:
    """Every diagonal 1q gate right before measurement is removed."""
    circ = _diag_tail_circuit()
    out = apply_pass(EliminateDiagonalBeforeMeasurement, circ)
    assert quspect_gate_counts(out) == {"h": 3, "cx": 1, "cry": 1}
    assert qiskit_probabilities(out) == pytest.approx(qiskit_probabilities(circ))
    ref = QiskitPassManager([RemoveDiagonalGatesBeforeMeasure()]).run(to_qiskit(circ))
    assert _n_gates(out) <= sum(qiskit_gate_counts(ref).values())


def test_diagonal_elimination_keeps_non_terminal_and_unmeasured() -> None:
    """Diagonals followed by gates or on unmeasured qubits stay."""
    circ = Circ().h(0).h(1).t(0).h(0).rz(0.3, 1).measure([0], [0])
    out = apply_pass(EliminateDiagonalBeforeMeasurement, circ)
    assert quspect_gate_counts(out) == {"h": 3, "t": 1, "rz": 1}
    assert qiskit_probabilities(out) == pytest.approx(qiskit_probabilities(circ))


def test_diagonal_elimination_mid_circuit_measure() -> None:
    """A diagonal right before a mid-circuit measurement is also removable."""
    circ = Circ().h(0).s(0).measure([0], [0]).h(0).measure([0], [1])
    out = apply_pass(EliminateDiagonalBeforeMeasurement, circ)
    assert quspect_gate_counts(out) == {"h": 2}


def test_diagonal_elimination_keeps_cz_unlike_qiskit() -> None:
    """quspect does not remove 2q diagonals (CZ); qiskit does (recorded)."""
    circ = Circ().h(0).h(1).cz(0, 1).measure([0, 1], [0, 1])
    out = apply_pass(EliminateDiagonalBeforeMeasurement, circ)
    ref = QiskitPassManager([RemoveDiagonalGatesBeforeMeasure()]).run(to_qiskit(circ))
    assert _n_gates(out) == 3
    assert sum(qiskit_gate_counts(ref).values()) == 2
    assert qiskit_probabilities(out) == pytest.approx(qiskit_probabilities(circ))


def test_diagonal_elimination_symbolic() -> None:
    """Symbolic diagonal gates before measurement are removed too (valid ∀θ)."""
    from quspect.circuit.parameter import Parameter

    th = Parameter("th")
    circ = Circ().h(0).rz(th, 0).measure([0], [0])
    out = apply_pass(EliminateDiagonalBeforeMeasurement, circ)
    assert quspect_gate_counts(out) == {"h": 1}


# =============================================================================
# Aggregate comparison with qiskit on motif circuits
# =============================================================================

_AGG_SEEDS = [(n, s) for n in range(2, 6) for s in range(5)]


def _comparison_row(n: int, seed: int) -> dict[str, tuple[int, int]]:
    """(quspect, qiskit) gate counts per pass family for one motif circuit."""
    circ = _motif_circuit(n, 25, 700 * n + seed)
    qc = to_qiskit(circ)
    return {
        "cancel": (
            _n_gates(apply_pass(GateCancellation, circ)),
            _qiskit_n_gates(qc, [InverseCancellation()]),
        ),
        "commute+cancel+merge": (
            _n_gates(
                apply_passes(
                    [CommutationAnalysis, GateCancellation, RotationMerging], circ
                )[0]
            ),
            _qiskit_n_gates(qc, [CommutativeCancellation()]),
        ),
        "fuse1q": (
            _n_gates(apply_pass(SingleQubitGateFusion, circ)),
            _qiskit_n_gates(qc, [Optimize1qGatesDecomposition(basis=["u"])]),
        ),
    }


@pytest.mark.parametrize(("n", "seed"), _AGG_SEEDS)
def test_motif_counts_not_markedly_worse_than_qiskit(n: int, seed: int) -> None:
    """Per circuit: quspect <= 1.5 * qiskit + 2 for every pass family."""
    for family, (mine, ref) in _comparison_row(n, seed).items():
        assert mine <= 1.5 * ref + 2, f"{family}: quspect {mine} vs qiskit {ref}"


def test_motif_counts_aggregate_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the aggregated quspect-vs-qiskit gate counts (informational)."""
    totals: dict[str, list[int]] = {}
    original = 0
    for n, seed in _AGG_SEEDS:
        original += _n_gates(_motif_circuit(n, 25, 700 * n + seed))
        for family, (mine, ref) in _comparison_row(n, seed).items():
            acc = totals.setdefault(family, [0, 0])
            acc[0] += mine
            acc[1] += ref
    with capsys.disabled():
        print(f"\n[pass comparison] original gates: {original}")
        for family, (mine, ref) in totals.items():
            print(f"  {family:<16} quspect={mine:>4}  qiskit={ref:>4}")
    assert totals["cancel"][0] <= totals["cancel"][1]
    for mine, ref in totals.values():
        assert mine <= 1.25 * ref


# =============================================================================
# PassManager / PassStats / BasePass
# =============================================================================


class _GrowPass(BasePass):
    """Test pass that adds one isolated OPERATION node per run."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Add a dummy node so the node count always changes."""
        nid = max((n.node_id for n in dag.nodes), default=-1) + 1
        dag.add_node(DAGNode(nid, DAGNodeType.OPERATION))
        return dag


class _NoOpPass(BasePass):
    """Test pass that does nothing."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Return the DAG untouched."""
        return dag


def _full_pipeline() -> list[Callable[[], BasePass]]:
    return [
        GateCancellation,
        CommutationAnalysis,
        RotationMerging,
        SingleQubitGateFusion,
        EliminateDiagonalBeforeMeasurement,
    ]


def test_base_pass_is_abstract_and_repr() -> None:
    """BasePass cannot be instantiated; name / enabled / repr work."""
    with pytest.raises(TypeError):
        BasePass()  # type: ignore[abstract]
    p = _NoOpPass()
    assert p.name == "_NoOpPass" and p.enabled
    assert repr(p) == "_NoOpPass()"
    p.enabled = False
    assert repr(p) == "_NoOpPass() (disabled)"


def test_pass_stats_properties() -> None:
    """PassStats derived fields: changed, removed, added, reduction_pct."""
    s = PassStats("x", nodes_before=10, nodes_after=7, elapsed_ms=1.0)
    assert s.changed and s.nodes_removed == 3 and s.nodes_added == 0
    assert s.reduction_pct == pytest.approx(30.0)
    s = PassStats("x", nodes_before=4, nodes_after=6, elapsed_ms=0.0)
    assert s.nodes_added == 2 and s.nodes_removed == 0
    s = PassStats("x", nodes_before=0, nodes_after=0, elapsed_ms=0.0)
    assert not s.changed and s.reduction_pct == 0.0


def test_pass_manager_composite_run_and_stats() -> None:
    """Pipeline on a reducible circuit: stats per pass, totals and summary."""
    circ = Circ().h(0).h(0).cx(0, 1).rz(0.3, 0).cx(0, 1).rz(0.2, 0).rx(0.1, 1)
    circ.rx(0.2, 1).t(1).measure([0, 1], [0, 1])
    out, pm = apply_passes(_full_pipeline(), circ)
    assert qiskit_probabilities(out) == pytest.approx(qiskit_probabilities(circ))
    assert _n_gates(out) == 1  # only the fused/merged rx on q1 survives
    stats = pm.stats_history
    assert len(stats) % 5 == 0 and len(stats) >= 10  # >= 2 iterations
    assert [s.pass_name for s in stats[:5]] == [p.__name__ for p in _full_pipeline()]
    assert all(not s.changed for s in stats[-5:])  # converged
    first_before = stats[0].nodes_before
    assert first_before == circ.to_dag().num_nodes()
    assert pm.total_nodes_removed == first_before - stats[-1].nodes_after
    assert pm.total_elapsed_ms >= 0.0
    text = pm.summary()
    assert "PassManager Optimization Summary" in text
    assert "removed" in text and "unchanged" in text
    assert f"{pm.total_nodes_removed} nodes removed" in text
    assert [type(p) for p in pm.passes] == _full_pipeline()
    pm.passes.clear()  # returns a copy
    assert len(pm.passes) == 5


@pytest.mark.parametrize("seed", range(4))
def test_pass_manager_idempotent_and_deterministic(seed: int) -> None:
    """Re-running on the output changes nothing; two runs give equal output."""
    circ = _motif_circuit(3, 25, 4242 + seed)
    out1, _ = apply_passes(_full_pipeline(), circ)
    out1b, _ = apply_passes(_full_pipeline(), circ)
    assert [repr(i) for i in out1.instructions] == [repr(i) for i in out1b.instructions]
    out2, pm2 = apply_passes(_full_pipeline(), out1)
    assert len(pm2.stats_history) == 5 and not any(s.changed for s in pm2.stats_history)
    assert [repr(i) for i in out2.instructions] == [repr(i) for i in out1.instructions]
    assert_unitary_equal(
        qiskit_circuit_unitary(out2, num_qubits=3),
        qiskit_circuit_unitary(circ),
        up_to_phase=True,
    )


def test_pass_manager_skips_disabled_and_reports_added() -> None:
    """Disabled passes produce no stats; growth is reported as 'added'."""
    grow, noop = _GrowPass(), _NoOpPass()
    noop.enabled = False
    pm = PassManager([grow, noop])
    dag = Circ().h(0).to_dag()
    with pytest.raises(RuntimeError):  # _GrowPass never converges
        pm.run(dag, max_iterations=2)
    assert [s.pass_name for s in pm.stats_history] == ["_GrowPass", "_GrowPass"]
    assert all(s.nodes_added == 1 for s in pm.stats_history)
    assert "added" in pm.summary()


def test_pass_manager_nonconvergence_stops_after_max_iterations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-converging pipelines stop after max_iterations, warn and raise."""
    pm = PassManager([_GrowPass()])
    dag = Circ().h(0).to_dag()
    n0 = dag.num_nodes()
    with (
        caplog.at_level(logging.WARNING, logger="quspect.dag.passes.base"),
        pytest.raises(RuntimeError, match="did not converge"),
    ):
        pm.run(dag, max_iterations=3)
    assert len(pm.stats_history) == 3
    assert pm.stats_history[-1].nodes_after == n0 + 3
    assert "did not converge" in caplog.text


def test_pass_manager_nonconvergence_raises_as_documented() -> None:
    """Documented contract: RuntimeError on non-convergence."""
    with pytest.raises(RuntimeError):
        PassManager([_GrowPass()]).run(Circ().h(0).to_dag(), max_iterations=2)


def test_pass_manager_stats_reset_between_runs() -> None:
    """stats_history is cleared at the start of each run."""
    pm = PassManager([GateCancellation()])
    pm.run(Circ().h(0).h(0).to_dag())
    pm.run(Circ().x(0).to_dag())
    assert len(pm.stats_history) == 1 and not pm.stats_history[0].changed


def test_gate_cancellation_reconnects_all_wires() -> None:
    """After cancelling a 2q pair every wire is reconnected (valid DAG)."""
    dag = Circ().h(0).h(1).cx(0, 1).cx(0, 1).h(0).h(1).to_dag()
    GateCancellation().run(dag)
    # h h on each wire are now adjacent and cancelled by a second run.
    GateCancellation().run(dag)
    assert dag.operation_nodes() == []
    assert all(isinstance(e, DAGEdge) for e in dag.edges)
    assert dag.num_edges() == 2
