"""Tests for :mod:`quspect.pulse.envelopes_compat` (the ``envelopes`` stand-in).

A tiny consumer package written against envelopes-qc (``from envelopes import
evc``, like ``lqcloud.sequence``) is generated into ``tmp_path`` and imported
under the stand-in.  Its outputs are pinned by SHA-256 digests produced by
running the same source against the envelopes-qc 0.1.5 wheel.
"""

from __future__ import annotations

import hashlib
import importlib
import struct
import sys
import textwrap
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import quspect._pulse as evc
from quspect.pulse import (
    ENVELOPES_MODULE_NAMES,
    install_as_envelopes,
    make_envelopes_module,
    use_as_envelopes,
)
from tests.pulse.test_golden import EXACT_LIBM, assert_matches_reference

CONSUMER = "fakeconsumer"

CONSUMER_SOURCES = {
    "__init__.py": '''
        """Package written against envelopes-qc, lqcloud style."""
    ''',
    "sequence.py": """
        import numpy as np
        from envelopes import EVC_COMPLEX_DTYPE, EVC_FLOAT_DTYPE, evc


        def build():
            xy = evc.GaussianDRAG(t0=0, w=10, amp=1.0, coef=0.1, df=0.0, phase=0.0)
            amp = np.array([1.0, 0.5, 0.25], dtype=EVC_FLOAT_DTYPE)
            seq = evc.align(xy, np.array([0.0, 100.0, 200.0]), amp)
            camp = np.exp(1j * np.array([0.3, -1.1])).astype(EVC_COMPLEX_DTYPE)
            seq += evc.align(xy, np.array([300.0, 400.0]), camp)
            return seq


        def roundtrip(env):
            return evc.deserialization(evc.serialization(env))
    """,
    "gate_seqs.py": """
        from envelopes import evc


        def readout():
            ring = evc.Gaussian(10.0, w=4.0, amp=0.3)
            flat = evc.Flattop(t0=20.0, tlen=100.0, w_left=2.0, w_right=2.0, amp=0.2)
            return evc.mix(ring + flat, 0.05, 0.0, True)
    """,
    "cpp.py": """
        import envelopes.envelopes_cpp as module

        MODULE = module
    """,
}

# (serialization sha256, decoded start, decoded samples sha256, dtype, shape),
# produced by the same consumer source running on envelopes-qc 0.1.5.
GOLDEN = {
    "sequence": (
        "b32a6a4fff8af8bd99189a6d27a51e40a0869f024fd935db22229850c640f6af",
        -30.0,
        "b67ae0134e904dad2c14f9b55593d5ffabd6b0223fbba902b7611f62d06d5265",
        "complex64",
        (922,),
    ),
    "readout": (
        "04b465204bf236ba4167749e2439ad80e72b5494274fc2ab3c36ccf6566988f3",
        -2.0,
        "31a55164c35094dab40ff8397ccba3ca759b577752240bd967432b15a1d6e311",
        "complex64",
        (258,),
    ),
}


@pytest.fixture
def consumer_path(tmp_path: Path) -> Iterator[Path]:
    """Write the consumer package to ``tmp_path`` and put it on ``sys.path``."""
    package = tmp_path / CONSUMER
    package.mkdir()
    for name, source in CONSUMER_SOURCES.items():
        (package / name).write_text(textwrap.dedent(source).lstrip())
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        yield tmp_path
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in sys.modules if n.split(".")[0] == CONSUMER]:
            del sys.modules[name]


def snapshot() -> dict[str, object]:
    """Entries of ``sys.modules`` that the stand-in may touch."""
    return {
        name: module
        for name, module in sys.modules.items()
        if name.split(".")[0] in ("envelopes", CONSUMER)
    }


def digest(env: evc.AbstractEnvelope) -> tuple[str, float, str, str, tuple[int, ...]]:
    t, wave = evc.decode_envelope(env, evc.WaveCache(0.5))
    return (
        hashlib.sha256(evc.serialization(env)).hexdigest(),
        t,
        hashlib.sha256(wave.tobytes()).hexdigest(),
        str(wave.dtype),
        wave.shape,
    )


def test_make_envelopes_module_exports() -> None:
    shim = make_envelopes_module()
    assert shim.__name__ == "envelopes"
    assert shim.evc is evc
    assert shim.envelopes_cpp is evc
    assert shim.EVC_COMPLEX_DTYPE is np.complex64
    assert shim.EVC_FLOAT_DTYPE is np.float32
    assert shim.__path__ == []
    with pytest.raises(AttributeError, match="evp"):
        _ = shim.evp
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = shim.nope


def test_consumer_runs_on_quspect_pulse(consumer_path: Path) -> None:
    before = snapshot()
    with use_as_envelopes(purge=(CONSUMER,)) as shim:
        assert sys.modules["envelopes"] is shim
        for name in ENVELOPES_MODULE_NAMES[1:]:
            assert sys.modules[name] is evc
        sequence = importlib.import_module(f"{CONSUMER}.sequence")
        gate_seqs = importlib.import_module(f"{CONSUMER}.gate_seqs")
        cpp = importlib.import_module(f"{CONSUMER}.cpp")
        assert cpp.MODULE is evc
        seq = sequence.build()
        readout = gate_seqs.readout()
        assert isinstance(seq, evc.EnvSum)
        assert isinstance(readout, evc.EnvSum)
        assert digest(seq) == GOLDEN["sequence"]
        # The readout contains a Flattop (libm erf): same check as the golden
        # case "compat/readout", exact on macOS arm64, within 1 ulp elsewhere.
        t, wave = evc.decode_envelope(readout, evc.WaveCache(0.5))
        assert_matches_reference(
            "compat/readout",
            [
                evc.serialization(readout),
                struct.pack("<d", t),
                str(wave.dtype).encode(),
                wave.tobytes(),
            ],
        )
        assert digest(readout) == GOLDEN["readout"] or not EXACT_LIBM
        again = sequence.roundtrip(seq)
        assert evc.serialization(again) == evc.serialization(seq)
        from envelopes import evc as imported

        assert imported is evc
        with pytest.raises(ImportError, match="evp"):
            from envelopes import evp  # noqa: F401
    assert snapshot() == before


def test_restores_real_envelopes_and_cached_consumers(consumer_path: Path) -> None:
    real = types.ModuleType("envelopes")
    real_cpp = types.ModuleType("envelopes.envelopes_cpp")
    real_py = types.ModuleType("envelopes.envelopes_py")
    stale = types.ModuleType(f"{CONSUMER}.sequence")
    parent = types.ModuleType(CONSUMER)
    parent.__path__ = []
    parent.sequence = stale  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "envelopes": real,
            "envelopes.envelopes_cpp": real_cpp,
            "envelopes.envelopes_py": real_py,
            CONSUMER: parent,
            f"{CONSUMER}.sequence": stale,
        }
    )
    try:
        before = snapshot()
        with use_as_envelopes(purge=(CONSUMER,)):
            assert "envelopes.envelopes_py" not in sys.modules
            assert CONSUMER not in sys.modules
            fresh = importlib.import_module(f"{CONSUMER}.sequence")
            assert fresh is not stale
            assert isinstance(fresh.build(), evc.EnvSum)
        assert snapshot() == before
        assert parent.sequence is stale  # type: ignore[attr-defined]
    finally:
        for name in list(snapshot()):
            del sys.modules[name]


def test_purged_submodule_attribute_is_restored(consumer_path: Path) -> None:
    package = importlib.import_module(CONSUMER)
    assert not hasattr(package, "gate_seqs")
    with use_as_envelopes(purge=(f"{CONSUMER}.gate_seqs",)):
        importlib.import_module(f"{CONSUMER}.gate_seqs")
        assert hasattr(package, "gate_seqs")
    assert not hasattr(package, "gate_seqs")
    assert f"{CONSUMER}.gate_seqs" not in sys.modules
    assert sys.modules[CONSUMER] is package


def test_restores_after_exception(consumer_path: Path) -> None:
    before = snapshot()
    with pytest.raises(RuntimeError, match="boom"), use_as_envelopes(purge=(CONSUMER,)):
        importlib.import_module(f"{CONSUMER}.sequence")
        raise RuntimeError("boom")
    assert snapshot() == before


def test_nested_blocks(consumer_path: Path) -> None:
    before = snapshot()
    with use_as_envelopes(purge=(CONSUMER,)) as outer:
        first = importlib.import_module(f"{CONSUMER}.sequence")
        with use_as_envelopes(purge=(CONSUMER,)) as inner:
            assert inner is not outer
            assert sys.modules["envelopes"] is inner
            second = importlib.import_module(f"{CONSUMER}.sequence")
            assert second is not first
        assert sys.modules["envelopes"] is outer
        assert sys.modules[f"{CONSUMER}.sequence"] is first
    assert snapshot() == before


def test_install_is_permanent_and_idempotent(consumer_path: Path) -> None:
    before = snapshot()
    try:
        with use_as_envelopes(purge=(CONSUMER,)):
            # Inside a block so that the permanent install is undone afterwards.
            del sys.modules["envelopes"]
            shim = install_as_envelopes(purge=(CONSUMER,))
            assert sys.modules["envelopes"] is shim
            assert install_as_envelopes(purge=(CONSUMER,)) is shim
            sequence = importlib.import_module(f"{CONSUMER}.sequence")
            assert digest(sequence.build()) == GOLDEN["sequence"]
    finally:
        assert snapshot() == before


def test_conftest_fixture(envelopes_shim: types.ModuleType) -> None:
    assert sys.modules["envelopes"] is envelopes_shim
    from envelopes import evc as imported

    assert imported is evc


def test_real_lqcloud_sequence_uses_quspect_pulse() -> None:
    with use_as_envelopes():
        lqcloud = pytest.importorskip("lqcloud")
        assert lqcloud.HAS_SEQUENCE
        sequence = importlib.import_module("lqcloud.sequence.sequence")
        template = evc.GaussianDRAG(t0=0, w=10, amp=1.0, coef=0.1, df=0.0, phase=0.0)
        aligner = sequence.EnvelopeAligner(template)
        assert aligner.env is template
        assert sequence.evc is evc
        assert importlib.import_module("lqcloud.sequence.gate_seqs").evc is evc
