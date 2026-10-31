"""Type stubs for the ``quspect._pulse`` native extension (see :mod:`quspect.pulse`)."""

from typing import SupportsComplex, SupportsFloat, SupportsIndex, overload

import numpy as np
import numpy.typing as npt

_Float = SupportsFloat | SupportsIndex
_Wave = npt.NDArray[np.float32] | npt.NDArray[np.complex64]

class WaveCache:
    def __init__(self, resolution: _Float) -> None: ...
    def resolution(self) -> float: ...

class AbstractEnvelope:
    def start(self) -> float: ...
    def end(self) -> float: ...
    def is_complex(self) -> bool: ...
    def copy(self) -> AbstractEnvelope: ...
    def __copy__(self) -> AbstractEnvelope: ...
    def __add__(self, other: _Float | AbstractEnvelope) -> AbstractEnvelope: ...
    def __radd__(self, other: _Float | AbstractEnvelope) -> AbstractEnvelope: ...
    # In-place add only takes envelopes at runtime (adding a number raises TypeError).
    def __iadd__(self, other: AbstractEnvelope) -> AbstractEnvelope: ...  # type: ignore[misc]
    def __mul__(
        self, other: _Float | SupportsComplex | complex
    ) -> AbstractEnvelope: ...
    def __rmul__(
        self, other: _Float | SupportsComplex | complex
    ) -> AbstractEnvelope: ...
    def __rshift__(self, other: _Float) -> AbstractEnvelope: ...

class _PicklableEnvelope(AbstractEnvelope):
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...

class Envelope(AbstractEnvelope): ...

class Gaussian(Envelope, _PicklableEnvelope):
    def __init__(self, t0: _Float, w: _Float, amp: _Float) -> None: ...

class GaussianDRAG(Envelope, _PicklableEnvelope):
    def __init__(
        self,
        t0: _Float,
        w: _Float,
        amp: _Float,
        coef: _Float,
        df: _Float,
        phase: _Float,
    ) -> None: ...

class CosineDRAG(Envelope, _PicklableEnvelope):
    def __init__(
        self,
        t0: _Float,
        w: _Float,
        amp: _Float,
        coef: _Float,
        df: _Float,
        phase: _Float,
    ) -> None: ...

class Triangle(Envelope, _PicklableEnvelope):
    def __init__(self, t0: _Float, tlen: _Float, amp: _Float, fall: bool) -> None: ...

class Rect(Envelope, _PicklableEnvelope):
    def __init__(self, t0: _Float, tlen: _Float, amp: _Float) -> None: ...

class Flattop(Envelope, _PicklableEnvelope):
    def __init__(
        self, t0: _Float, tlen: _Float, w_left: _Float, w_right: _Float, amp: _Float
    ) -> None: ...

class RippleRect(Envelope, _PicklableEnvelope):
    def __init__(
        self,
        t0: _Float,
        tlen: _Float,
        amp: _Float,
        w: _Float,
        ripple0: _Float,
        ripple1: _Float,
        ripple2: _Float,
        ripple3: _Float,
    ) -> None: ...

class PowerSineCosine(Envelope, _PicklableEnvelope):
    def __init__(
        self,
        t0: _Float,
        tlen: _Float,
        amp: _Float,
        m: SupportsIndex,
        n: SupportsIndex,
        k: _Float,
        phase: _Float,
    ) -> None: ...

class GaussianXPower(Envelope, _PicklableEnvelope):
    def __init__(
        self, t0: _Float, w: _Float, amp: _Float, m: SupportsIndex
    ) -> None: ...

class Custom(Envelope, _PicklableEnvelope):
    def __init__(
        self, start: _Float, dt: _Float, data: npt.NDArray[np.generic]
    ) -> None: ...

class MixExp(_PicklableEnvelope): ...
class EnvMix(_PicklableEnvelope): ...
class EnvShift(_PicklableEnvelope): ...
class RealEnvProd(_PicklableEnvelope): ...
class COMPLEXEnvProd(_PicklableEnvelope): ...
class EnvSum(_PicklableEnvelope): ...

@overload
def decode_envelope(env: AbstractEnvelope, wc: WaveCache) -> tuple[float, _Wave]: ...
@overload
def decode_envelope(
    env: AbstractEnvelope, wc: WaveCache, start: _Float, end: _Float
) -> tuple[float, _Wave]: ...
def serialization(env: AbstractEnvelope) -> bytes: ...
def deserialization(serialized: bytes) -> AbstractEnvelope: ...
@overload
def align(env: AbstractEnvelope, dt: npt.ArrayLike) -> AbstractEnvelope: ...
@overload
def align(
    env: AbstractEnvelope, dt: npt.ArrayLike, amp: npt.ArrayLike
) -> AbstractEnvelope: ...
def split(
    env: AbstractEnvelope, starts: npt.ArrayLike, ends: npt.ArrayLike
) -> list[AbstractEnvelope]: ...
def mix(
    env: AbstractEnvelope, df: _Float, phase: _Float, dynamical: bool
) -> AbstractEnvelope: ...
