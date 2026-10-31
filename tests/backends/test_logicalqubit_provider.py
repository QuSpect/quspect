"""Offline tests for LogicalQubit provider configuration and job recovery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("lqcloud")

from quspect.backends import CompositeJob, JobReference
from quspect.backends.logicalqubit import _provider as provider_module


class _SDKProvider:
    instances: list[_SDKProvider] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.url = kwargs.get("url", "https://config.example")
        self.backends: list[dict] = [{"name": "QZ02"}]
        self.get_backend_calls: list[tuple[str, bool, bool]] = []
        _SDKProvider.instances.append(self)

    def get_backends(self):
        return list(self.backends)

    def get_backend(self, name, *, use_async=True, verify=True):
        self.get_backend_calls.append((name, use_async, verify))
        return SimpleNamespace(name=name, config={"qubits": 20})


class _SDKJob:
    instances: list[_SDKJob] = []

    def __init__(
        self,
        job_id,
        provider,
        backend_name=None,
        base_url=None,
        use_async=False,
    ) -> None:
        self.job_id = job_id
        self.provider = provider
        self.backend_name = backend_name
        self.base_url = base_url
        self._use_async = use_async
        self._is_batch = False
        self._batch_size = None
        _SDKJob.instances.append(self)

    def status(self):
        return SimpleNamespace(value="completed")

    def cancel(self):
        return True


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    _SDKProvider.instances.clear()
    _SDKJob.instances.clear()
    monkeypatch.delenv("LQCLOUD_API_KEY", raising=False)
    monkeypatch.delenv("LQCLOUD_URL", raising=False)
    monkeypatch.setattr(provider_module, "_HAS_LQCLOUD", True)
    monkeypatch.setattr(
        provider_module,
        "lqcloud",
        SimpleNamespace(LQCloudProvider=_SDKProvider),
    )
    import lqcloud.job

    monkeypatch.setattr(lqcloud.job, "Job", _SDKJob)


def test_constructor_preserves_sdk_config_fallback() -> None:
    provider = provider_module.LogicalQubitProvider()
    assert _SDKProvider.instances[0].kwargs == {"interactive": False}
    assert provider.url == "https://config.example"


def test_explicit_credentials_and_transport_are_forwarded(monkeypatch) -> None:
    monkeypatch.setenv("LQCLOUD_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LQCLOUD_URL", "https://env.example")
    provider = provider_module.LogicalQubitProvider()
    backend = provider.get_backend("QZ02", use_async=False, verify=False)

    assert _SDKProvider.instances[0].kwargs == {
        "interactive": False,
        "api_key": "not-a-real-key",
        "url": "https://env.example",
    }
    assert backend.name == "QZ02"
    assert _SDKProvider.instances[0].get_backend_calls == [("QZ02", False, False)]


def test_retrieve_native_batch_restores_route_and_batch_hints() -> None:
    provider = provider_module.LogicalQubitProvider()
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_batch",
        base_url="https://config.example/jobs",
        transport="async",
        batch_size=2,
        labels=("a", "b"),
        experiment_shapes=((1, 1), (2, 2)),
        shots=32,
        result_format="counts",
    )

    job = provider.retrieve_job(reference)
    raw = _SDKJob.instances[0]
    assert raw.job_id == "task_batch"
    assert raw.base_url == "https://config.example/jobs"
    assert raw._use_async is True
    assert raw._is_batch is True
    assert raw._batch_size == 2
    assert job.reference == reference


def test_retrieve_composite_uses_persisted_child_sizes() -> None:
    provider = provider_module.LogicalQubitProvider()
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_first",
        transport="async",
        batch_size=3,
        labels=("a", "b", "c"),
        experiment_shapes=((1, 1), (1, 1), (1, 1)),
        shots=16,
        result_format="counts",
        child_job_ids=("task_first", "task_second"),
        child_batch_sizes=(2, 1),
    )

    job = provider.retrieve_job(reference)
    assert isinstance(job, CompositeJob)
    assert [raw.job_id for raw in _SDKJob.instances] == ["task_first", "task_second"]
    assert job.reference.labels == ("a", "b", "c")
    assert job.reference.child_batch_sizes == (2, 1)


def test_retrieve_rejects_another_platform() -> None:
    provider = provider_module.LogicalQubitProvider()
    reference = JobReference(
        platform="other",
        backend_name="backend",
        job_id="job",
    )
    with pytest.raises(ValueError, match="other"):
        provider.retrieve_job(reference)


def test_retrieve_rejects_foreign_endpoint_before_reusing_credentials() -> None:
    provider = provider_module.LogicalQubitProvider()
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task",
        base_url="https://evil.example/jobs",
    )

    with pytest.raises(ValueError, match="configured origin"):
        provider.retrieve_job(reference)
    assert _SDKJob.instances == []
