"""Regenerate the tests/pulse goldens from the envelopes-qc 0.1.5 reference wheel.

/path/to/venv-with-envelopes-qc/bin/python tools/pulse_compat/make_golden.py

Writes ``golden.json`` (SHA-256 of every part of every case) and
``golden_waveforms.npz`` (the decoded sample arrays of the ``ERF_DEPENDENT``
cases, compared with a tolerance on platforms whose libm ``erf`` differs from
the macOS arm64 one used to generate the goldens).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "pulse"))

from envelopes import envelopes_cpp  # noqa: E402
from golden_cases import ERF_DEPENDENT, cases  # noqa: E402


def main() -> None:
    golden = {}
    waveforms = {}
    for name, case in cases().items():
        parts = case(envelopes_cpp)
        golden[name] = [hashlib.sha256(part).hexdigest() for part in parts]
        if name in ERF_DEPENDENT:
            # _full() cases end with (dtype name, sample bytes).
            waveforms[name] = np.frombuffer(parts[-1], dtype=parts[-2].decode())
    out = ROOT / "tests" / "pulse" / "golden.json"
    out.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    npz = ROOT / "tests" / "pulse" / "golden_waveforms.npz"
    np.savez_compressed(npz, **waveforms)  # type: ignore[arg-type]
    print(f"{len(golden)} cases -> {out}, {len(waveforms)} waveforms -> {npz}")


if __name__ == "__main__":
    main()
