"""Decoded waveforms and serialization bytes match envelopes-qc 0.1.5.

``golden.json`` holds SHA-256 digests of every part (serialization bytes, start
time, dtype, samples) produced by the reference wheel on macOS arm64, and
``golden_waveforms.npz`` the reference sample arrays of the ``ERF_DEPENDENT``
cases (regenerate both with ``tools/pulse_compat/make_golden.py``).

On macOS arm64 every case must be bit-identical.  Elsewhere the ``ERF_DEPENDENT``
cases (Flattop, RippleRect) go through the platform libm ``erf``, whose last bit
differs from Apple's, so their samples are compared within one float32 ulp:
these envelopes are O(1) (|value| <= ~1), where a float32 ulp is <= 1.2e-7, and
a last-bit difference in a double ``erf`` changes the float32 result by at most
one ulp.  Their serialization bytes, start time, dtype and shape stay exact, as
does every other case.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pytest

from tests.pulse.golden_cases import ERF_DEPENDENT, cases

HERE = Path(__file__).parent
GOLDEN: dict[str, list[str]] = json.loads((HERE / "golden.json").read_text())
with np.load(HERE / "golden_waveforms.npz") as _npz:
    WAVEFORMS: dict[str, np.ndarray] = {name: _npz[name] for name in _npz.files}
CASES = cases()

EXACT_LIBM = platform.system() == "Darwin" and platform.machine() == "arm64"
"""The goldens' platform: libm ``erf`` matches the reference build bit for bit."""

RTOL = 1e-6
ATOL = 1e-7


def assert_matches_reference(name: str, parts: list[bytes]) -> None:
    """Assert that the raw ``parts`` of case ``name`` match the reference.

    Args:
        name: Golden case name.
        parts: Bytes returned by the case; ``ERF_DEPENDENT`` cases end with the
            dtype name and the sample buffer.

    Raises:
        AssertionError: If the parts differ from the reference.
    """
    digests = [hashlib.sha256(part).hexdigest() for part in parts]
    if EXACT_LIBM or name not in ERF_DEPENDENT:
        assert digests == GOLDEN[name]
        return
    assert digests[:-1] == GOLDEN[name][:-1]
    wave = np.frombuffer(parts[-1], dtype=parts[-2].decode())
    reference = WAVEFORMS[name]
    assert wave.shape == reference.shape
    np.testing.assert_allclose(wave, reference, rtol=RTOL, atol=ATOL)


def test_golden_covers_all_cases() -> None:
    assert sorted(GOLDEN) == sorted(CASES)
    assert sorted(WAVEFORMS) == sorted(ERF_DEPENDENT)


@pytest.mark.parametrize("name", sorted(ERF_DEPENDENT))
def test_stored_waveforms_are_the_reference(name: str) -> None:
    assert hashlib.sha256(WAVEFORMS[name].tobytes()).hexdigest() == GOLDEN[name][-1]


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_reference(name: str) -> None:
    from quspect import _pulse

    assert_matches_reference(name, CASES[name](_pulse))
