"""Tianyan (中电信天衍) quantum cloud backends.

Provides :class:`TianyanProvider` for authenticating with the Tianyan cloud
(https://qc.zdxlz.com) and :class:`TianyanBackend` for its QPUs (e.g.
``tianyan176``) and sampling simulators (``tianyan_sw``, ...).  QuSpect
generates QCIS and talks to the cloud's HTTPS API directly with the standard
library; cqlib is not required.
"""

from quspect.backends.tianyan._backend import TianyanBackend
from quspect.backends.tianyan._client import TianyanClient
from quspect.backends.tianyan._job import TianyanJob
from quspect.backends.tianyan._provider import TianyanProvider

__all__ = ["TianyanProvider", "TianyanBackend", "TianyanJob", "TianyanClient"]
