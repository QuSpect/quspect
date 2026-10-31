"""Generate the SVG images used by the QuSpect documentation and READMEs.

Run from the repository root (circuit drawing is pure Python; no native
extension is required)::

    python tools/generate_doc_images.py

This script creates:

1. ``docs/source/_static/circuit_sample.svg`` -- a representative quantum
   circuit with various gate types, barriers, and measurements.
2. ``docs/source/_static/sampling_sample.svg`` -- a measurement outcome
   distribution with statistical annotations.
3. ``docs/source/_static/readme/bell_circuit_{light,dark}.svg`` -- the Bell
   circuit shown in the "Visualize" section of ``README.md`` and
   ``README_CN.md``. The READMEs embed them via absolute
   ``raw.githubusercontent.com`` URLs so they also render on PyPI.
"""

from __future__ import annotations

from pathlib import Path

from quspect.circuit.circuit import QuantumCircuit
from quspect.simulation.result import SimulationResult
from quspect.visualization import CircuitSVG, SamplingDisplay, draw_svg_circuit

DOC_STATIC = Path(__file__).resolve().parent.parent / "docs" / "source" / "_static"
README_IMAGES = DOC_STATIC / "readme"


def generate_circuit_svg() -> None:
    """Generate a sample quantum circuit diagram as SVG.

    Builds a 4-qubit circuit that demonstrates:
    - single-qubit gates (H, X, T)
    - two-qubit gates (CX/CNOT)
    - three-qubit gate (CCX/Toffoli)
    - SWAP
    - barrier
    - measurement
    """
    circ = QuantumCircuit(name="sample")

    # Layer 0: superposition on qubits 0,1 via Hadamard
    circ.h(0)
    circ.h(1)

    # Layer 1: entangle qubits 0-2 via CNOT
    circ.cx(0, 2)

    # Layer 2: barrier to separate logical blocks
    circ.barrier(0, 1, 2, 3)

    # Layer 3: Toffoli gate (CCX) on qubits 0,1 → 3
    circ.ccx(0, 1, 3)

    # Layer 4: more operations
    circ.x(3)

    # Layer 5: SWAP qubits 2 and 3
    circ.swap(2, 3)

    # Layer 6: T gate as an example of a phase gate
    circ.t(0)

    # Layer 7: another CNOT
    circ.cx(1, 3)

    # Layer 8: measure all active qubits
    circ.measure([0, 1], [0, 1])

    # Draw as SVG with "official" theme
    svg: CircuitSVG = draw_svg_circuit(circ, style="official", scale=1.5)

    out_path = DOC_STATIC / "circuit_sample.svg"
    svg.to_file(out_path)
    print(f"  ✓ Circuit SVG saved to: {out_path}")


def generate_sampling_svg() -> None:
    """Generate a sample sampling result visualization as SVG.

    Creates an artificial sampling result that mimics a Bell-state
    measurement on 2 qubits with 1024 shots, resulting in roughly
    balanced |00⟩ and |11⟩ outcomes.
    """
    # Bell-state sampling: |00⟩ and |11⟩ should dominate with ~50% each
    # We add some minor noise outcomes for realism
    counts = {
        "00": 502,
        "01": 12,
        "10": 8,
        "11": 510,
    }
    n_shots = sum(counts.values())

    probs = {bs: cnt / n_shots for bs, cnt in counts.items()}

    result = SimulationResult(
        counts=counts,
        probabilities=probs,
        num_shots=n_shots,
        num_qubits=2,
        num_clbits=2,
    )

    display = SamplingDisplay(
        result=result,
        style="svg",
        max_outcomes=8,
        theme="official",
        svg_width=720,
    )

    out_path = DOC_STATIC / "sampling_sample.svg"
    display.to_file(out_path)
    print(f"  ✓ Sampling SVG saved to: {out_path}")


def generate_readme_bell_svgs() -> None:
    """Generate light and dark Bell-circuit SVGs for the READMEs.

    The circuit matches the "Build a Circuit" example in ``README.md``.
    Both variants carry an opaque background so they read well on either
    GitHub theme; the READMEs pick one with ``<picture>`` and
    ``prefers-color-scheme``.
    """
    circ = QuantumCircuit(name="bell")
    circ.h(0)
    circ.cx(0, 1)
    circ.measure_all()

    README_IMAGES.mkdir(parents=True, exist_ok=True)
    for variant in ("light", "dark"):
        out_path = README_IMAGES / f"bell_circuit_{variant}.svg"
        circ.svg(style=variant).to_file(out_path)
        print(f"  ✓ README Bell circuit ({variant}) saved to: {out_path}")


def main() -> None:
    """Generate all documentation images."""
    DOC_STATIC.mkdir(parents=True, exist_ok=True)
    print("Generating documentation images...")
    generate_circuit_svg()
    generate_sampling_svg()
    generate_readme_bell_svgs()
    print("Done.")


if __name__ == "__main__":
    main()
