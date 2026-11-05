"""Offline tests for LogicalQubit provider configuration and job recovery."""

from __future__ import annotations

import warnings
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

    # Mirrors lqcloud 0.6 minus the deprecated ``use_async`` parameter, so
    # forwarding it again fails loudly.
    def get_backend(self, name, verify=True):
        self.get_backend_calls.append((name, verify))
        return SimpleNamespace(name=name, config={"qubits": 20})


class _SDKJob:
    instances: list[_SDKJob] = []

    # Same signature as lqcloud 0.6 ``Job`` (no ``use_async``).
    def __init__(
        self,
        job_id,
        provider,
        backend_name=None,
        base_url=None,
    ) -> None:
        self.job_id = job_id
        self.provider = provider
        self.backend_name = backend_name
        self.base_url = base_url
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
    for name in ("LQCLOUD_API_KEY", "LQCLOUD_TOKEN", "LQCLOUD_URL"):
        monkeypatch.delenv(name, raising=False)
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


def test_environment_is_left_to_the_sdk(monkeypatch) -> None:
    monkeypatch.setenv("LQCLOUD_API_KEY", "not-a-real-key")
    monkeypatch.setenv("LQCLOUD_URL", "https://env.example")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provider_module.LogicalQubitProvider()

    # lqcloud reads (and since 0.6 prioritizes) the environment itself.
    assert _SDKProvider.instances[0].kwargs == {"interactive": False}


def test_explicit_arguments_and_store_are_forwarded() -> None:
    provider_module.LogicalQubitProvider(
        api_key="not-a-real-key",
        url="https://explicit.example",
        store=False,
    )
    assert _SDKProvider.instances[0].kwargs == {
        "interactive": False,
        "api_key": "not-a-real-key",
        "url": "https://explicit.example",
        "store": False,
    }


def test_explicit_arguments_overridden_by_environment_warn(monkeypatch) -> None:
    monkeypatch.setenv("LQCLOUD_TOKEN", "env-token")
    monkeypatch.setenv("LQCLOUD_URL", "https://env.example")
    with pytest.warns(UserWarning) as caught:
        provider_module.LogicalQubitProvider(
            api_key="explicit-key", url="https://explicit.example"
        )
    messages = [str(item.message) for item in caught]
    assert any("LQCLOUD_API_KEY / LQCLOUD_TOKEN" in text for text in messages)
    assert any("LQCLOUD_URL" in text for text in messages)


def test_matching_environment_does_not_warn(monkeypatch) -> None:
    monkeypatch.setenv("LQCLOUD_API_KEY", "same-key")
    monkeypatch.setenv("LQCLOUD_URL", "https://same.example/")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        provider_module.LogicalQubitProvider(
            api_key="same-key", url="https://same.example"
        )


def test_get_backend_does_not_forward_use_async() -> None:
    provider = provider_module.LogicalQubitProvider()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = provider.get_backend("QZ02", verify=False)

    assert backend.name == "QZ02"
    assert _SDKProvider.instances[0].get_backend_calls == [("QZ02", False)]


def test_use_async_is_deprecated_and_sync_is_rejected() -> None:
    provider = provider_module.LogicalQubitProvider()
    with pytest.warns(DeprecationWarning, match="use_async"):
        provider.get_backend("QZ02", use_async=True, verify=False)
    with pytest.raises(ValueError, match="use_async=False"):
        provider.get_backend("QZ02", use_async=False, verify=False)
    with pytest.raises(ValueError, match="use_async=False"):
        provider.retrieve_job("task_x", backend_name="QZ02", use_async=False)
    assert _SDKProvider.instances[0].get_backend_calls == [("QZ02", False)]
    assert _SDKJob.instances == []


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


def test_retrieve_accepts_reference_saved_with_sync_transport() -> None:
    # References persisted before lqcloud 0.6 may say "sync"; the SDK now
    # polls every job through the asynchronous task endpoint.
    provider = provider_module.LogicalQubitProvider()
    reference = JobReference(
        platform="logicalqubit",
        backend_name="QZ02",
        job_id="task_old",
        transport="sync",
    )

    job = provider.retrieve_job(reference)
    assert [raw.job_id for raw in _SDKJob.instances] == ["task_old"]
    assert job.reference == reference


def test_retrieve_by_job_id_builds_async_reference() -> None:
    provider = provider_module.LogicalQubitProvider()
    job = provider.retrieve_job("task_new", backend_name="QZ02", shots=8)
    assert job.reference.transport == "async"
    assert job.reference.base_url == "https://config.example"
