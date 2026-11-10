"""Every Python example of the ``quspect-api`` skill runs on this checkout.

The skill is read by coding agents as the description of the SDK, so an
example that no longer runs teaches a wrong API.  Each Markdown file is
executed top to bottom in one namespace, exactly as written.

The only substitution is the cloud entry point: ``LogicalQubitProvider`` is
replaced by an offline stand-in whose backend is the real
:class:`LogicalQubitBackend` on a fake 20-qubit line chip that simulates what
it is sent.  No SDK, network or credential is involved.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from quspect._native_loader import load_core

_SKILL = Path(__file__).resolve().parents[2] / "skills" / "quspect-api"

# skills/ is not shipped in the sdist; skip when running from an unpacked one.
if not _SKILL.is_dir():
    pytest.skip(f"{_SKILL} not found", allow_module_level=True)
if load_core() is None:
    pytest.skip("QuSpect native extension not built", allow_module_level=True)
pytest.importorskip("scipy")

from quspect import Simulator  # noqa: E402
from quspect.backends.logicalqubit._backend import LogicalQubitBackend  # noqa: E402
from quspect.ir.qsal import loads  # noqa: E402
from tests.backends.test_logicalqubit_backend import (  # noqa: E402
    _RawBatch,
    _RawJob,
    _RawResult,
)

_FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.S | re.M)
_FILES = [_SKILL / "SKILL.md", *sorted((_SKILL / "references").glob("*.md"))]


class _PassThroughTranslator:
    def translate(self, document: str) -> str:
        return document


class _OfflineChip:
    """Stands in for the provider SDK's backend object: a 20-qubit line."""

    def __init__(self) -> None:
        self.name = "OFFLINE-20"
        self.base_url = "https://example.invalid"
        self.config = {
            "qubits": 20,
            "supported_modes": ["circuit"],
            "supported_measurement_types": ["measure2"],
            "topology": {"coupling_map": [[i, i + 1] for i in range(19)]},
        }
        self.calls = 0

    def run(self, program: Any, **kwargs: Any) -> _RawJob:
        self.calls += 1
        job_id = f"task_{self.calls}"

        def simulate(document: str, index: int) -> _RawResult:
            circuit = loads(document).circuit
            assert circuit is not None
            seed = 1000 + 97 * self.calls + index
            counts = Simulator(seed=seed).run(circuit, shots=kwargs["shots"]).counts
            return _RawResult(dict(counts), f"{job_id}#{index}")

        if isinstance(program, list):
            results = [simulate(document, i) for i, document in enumerate(program)]
            return _RawJob(job_id, _RawBatch(results, job_id), is_batch=True)
        return _RawJob(job_id, simulate(program, 0), is_batch=False)


class _OfflineProvider:
    """Stands in for ``LogicalQubitProvider``."""

    url = "https://example.invalid"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._jobs: dict[str, Any] = {}
        self._backend = LogicalQubitBackend(_OfflineChip(), provider=self)
        self._backend._translator = _PassThroughTranslator()  # type: ignore[assignment]
        submit_request = self._backend.submit_request

        def remember(request: Any) -> Any:
            job = submit_request(request)
            for job_id in job.job_ids:
                self._jobs[job_id] = job
            return job

        self._backend.submit_request = remember  # type: ignore[method-assign]

    def list_backends(self) -> list[dict[str, Any]]:
        return [{"name": "OFFLINE-20", "status": "online"}]

    def get_backend(self, name: str, **kwargs: Any) -> LogicalQubitBackend:
        return self._backend

    def retrieve_job(self, reference: Any, **kwargs: Any) -> Any:
        return self._jobs[reference.job_id]


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [
        (text[: match.start()].count("\n") + 1, match.group(2))
        for match in _FENCE.finditer(text)
        if match.group(1) == "python"
    ]


def test_the_skill_has_examples_to_run() -> None:
    assert sum(len(_python_blocks(path)) for path in _FILES) >= 40
    assert (_SKILL / "references" / "compilation.md") in _FILES


@pytest.mark.parametrize("path", _FILES, ids=[path.name for path in _FILES])
def test_python_examples_run(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import quspect.backends.logicalqubit as logicalqubit

    monkeypatch.setattr(logicalqubit, "LogicalQubitProvider", _OfflineProvider)
    monkeypatch.setenv("LQCLOUD_BACKEND", "OFFLINE-20")
    namespace: dict[str, Any] = {"__name__": "__skill_example__"}
    for line, code in _python_blocks(path):
        try:
            exec(compile(code, f"{path.name}:{line}", "exec"), namespace)
        except Exception as exc:
            pytest.fail(
                f"{path.name}, example at line {line}: {type(exc).__name__}: {exc}"
            )
