"""Origin Quantum (本源量子) cloud backends.

Provides :class:`OriginQProvider` for authenticating with the Origin Quantum
cloud and :class:`OriginQBackend` for its QPUs (e.g. ``WK_C180_2``) and the
``full_amplitude`` cloud simulator.  QuSpect talks to the cloud's HTTP API
directly with the standard library; pyqpanda3 is not required.
"""

from quspect.backends.originq._backend import OriginQBackend
from quspect.backends.originq._client import CancelResponse, OriginQClient
from quspect.backends.originq._job import OriginQJob
from quspect.backends.originq._provider import OriginQProvider

__all__ = [
    "OriginQProvider",
    "OriginQBackend",
    "OriginQJob",
    "OriginQClient",
    "CancelResponse",
]
