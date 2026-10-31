#!/usr/bin/env python3
"""Generate SVG sample files for the documentation homepage.

Produces:

1. ``circuit_sample.svg`` — A feature-rich quantum circuit showcasing
   single-qubit gates, multi-controlled gates, SWAP, barrier, and
   measurements.

2. ``sampling_sample.svg`` — A Bell state / GHZ sampling result with
   gradient bar charts and a statistical summary panel.
"""

from __future__ import annotations

from pathlib import Path

from quspect.circuit.circuit import QuantumCircuit
from quspect.simulation.result import SimulationResult
from quspect.visualization.sampling_viz import SamplingDisplay
from quspect.visualization.svg_drawer import draw_svg_circuit


def build_showcase_circuit() -> QuantumCircuit:
    """Build a rich demo circuit that exercises all the SVG rendering features.

    The circuit spans 4 qubits + 2 classical bits and includes:
        - Hadamard (H) on q0, q1
        - CNOT with q0→q2 (control + target)
        - Toffoli (CCX) with q0,q1→q3 (two controls + target)
        - Barrier across all qubits
        - X on q3
        - SWAP between q2 and q3
        - T gate on q0
        - CNOT with q1→q3
        - Measurement on q0→c0 and q1→c1
    """
    qc = QuantumCircuit()

    # Layer 1 — superposition
    qc.h(0)
    qc.h(1)

    # Layer 2 — CNOT (control=0, target=2)
    qc.cx(0, 2)

    # Layer 3 — Toffoli (controls=0,1, target=3)
    qc.ccx(0, 1, 3)

    # Layer 4 — barrier across all qubits
    qc.barrier(0, 1, 2, 3)

    # Layer 5 — X gate on q3
    qc.x(3)

    # Layer 6 — another X on q3 (dense demo)
    qc.x(3)

    # Layer 7 — SWAP between q2 and q3
    qc.swap(2, 3)

    # Layer 8 — T gate on q0
    qc.t(0)

    # Layer 9 — CNOT (control=1, target=3)
    qc.cx(1, 3)

    # Layer 10 — Measurements
    qc.measure([0, 1], [0, 1])

    return qc


def generate_circuit_svg(output_dir: Path) -> None:
    """Generate the circuit showcase SVG."""
    qc = build_showcase_circuit()
    svg = draw_svg_circuit(qc, style="official", scale=1.5)
    dest = output_dir / "circuit_sample.svg"
    svg.to_file(dest)
    print(f"✓ Circuit SVG written to {dest} ({len(svg.content)} bytes)")


def generate_sampling_svg(output_dir: Path) -> None:
    """Generate a sampling visualization SVG using a Bell-state-like distribution.

    The distribution mimics a Bell state (|00⟩ + |11⟩)/√2 measured 1024 times,
    with realistic statistical noise:
        - |00⟩ ≈ 500 counts (~49%)
        - |11⟩ ≈ 490 counts (~48%)
        - |01⟩ ≈  18 counts (~1.8%)
        - |10⟩ ≈  16 counts (~1.6%)
    """
    counts: dict[str, int] = {
        "00": 502,
        "11": 490,
        "01": 18,
        "10": 14,
    }
    total = sum(counts.values())
    probabilities = {k: v / total for k, v in counts.items()}
    # 2 qubits, 2 classical bits, 1024 shots
    result = SimulationResult(
        counts=counts,
        probabilities=probabilities,
        num_qubits=2,
        num_clbits=2,
        num_shots=1024,
    )

    display = SamplingDisplay(
        result=result,
        style="svg",
        theme="official",
        svg_width=720,
        max_outcomes=16,
    )
    dest = output_dir / "sampling_sample.svg"
    display.to_file(dest)
    print(f"✓ Sampling SVG written to {dest}")


def main() -> None:
    """Entry point — generate both samples into docs/source/_static/."""
    repo_root = Path(__file__).resolve().parent.parent
    static_dir = repo_root / "docs" / "source" / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    generate_circuit_svg(static_dir)
    generate_sampling_svg(static_dir)

    print("\nDone. Both samples are ready for the documentation homepage.")


if __name__ == "__main__":
    main()
