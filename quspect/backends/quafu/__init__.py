"""Quafu-SQC (夸父) superconducting quantum cloud backend.

Talks to https://quafu-sqc.baqis.ac.cn directly over HTTP using only the
standard library — no Quafu SDK is required. Circuits are compiled by
QuSpect for the selected chip and executed without server-side compilation.

Example:
    >>> from quspect.backends.quafu import QuafuProvider
    >>> backend = QuafuProvider().get_backend("Haituo")
    >>> result = backend.run(circuit, shots=1024)
"""

from quspect.backends.quafu._backend import QuafuBackend
from quspect.backends.quafu._client import QuafuClient
from quspect.backends.quafu._job import QuafuJob
from quspect.backends.quafu._provider import QuafuProvider

__all__ = [
    "QuafuProvider",
    "QuafuBackend",
    "QuafuJob",
    "QuafuClient",
]
