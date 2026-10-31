"""Exhaustive objective identities and mocked cloud protocol, without hardware."""

import io
import itertools
import json

import httpx
import numpy as np
import pytest

from quspect.optimization import (
    BinaryQuadraticModel,
    IsingQClient,
    IsingQError,
    OptimizationSense,
    VariableType,
    bqm_from_qubo_matrix,
    compile_isingq,
)


@pytest.mark.parametrize("domain", list(VariableType))
@pytest.mark.parametrize("sense", list(OptimizationSense))
def test_every_assignment_preserves_offset_direction_and_names(domain, sense):
    model = BinaryQuadraticModel(
        ("a", "b", "c"),
        [1.5, -3, 0.25],
        [[0, 2, -4], [2, 0, 0.5], [-4, 0.5, 0]],
        offset=7.25,
        variable_type=domain,
        sense=sense,
    )
    program = compile_isingq(model)
    matrix = np.loadtxt(io.BytesIO(program.csv), delimiter=",")
    assert np.allclose(matrix, np.triu(matrix))
    for bits in itertools.product((0, 1), repeat=3):
        decoded = program.decode(bits, provider_energy=123)
        sample = bits if domain is VariableType.BINARY else 2 * np.asarray(bits) - 1
        assert decoded["objective_value"] == pytest.approx(model.evaluate(sample))
        assert decoded["submitted_objective_value"] == pytest.approx(
            np.array(bits) @ matrix @ bits
        )
        assert decoded["variables"] == ["a", "b", "c"]
        assert decoded["provider_energy"] == 123  # not substituted for the objective


def program():
    return compile_isingq(bqm_from_qubo_matrix([[-1, 2], [0, -3]], offset=5))


def signature():
    return {
        "data": {
            "host": "https://upload.example.test",
            "policy": "test-policy",
            "signature": "test-signature",
            "x_oss_credential": "temporary",
            "x_oss_date": "date",
            "security_token": "temporary-token",
        }
    }


def test_upload_key_isolation_submission_and_recovery():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "upload.example.test":
            assert "authorization" not in request.headers
            assert b"account-secret" not in request.content
            return httpx.Response(200)
        assert request.headers["Authorization"] == "account-secret"
        if request.url.path.endswith("getPostSignatureForOssUpload"):
            return httpx.Response(200, json=signature())
        if request.url.path == "/tasks/create-general":
            payload = json.loads(request.content)
            assert payload["caculateCount"] == 2
            assert payload["questionType"] == 1
            assert "gear" not in payload
            return httpx.Response(200, json={"data": {"id": "task-1"}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "status": 1,
                    "result": {"spin_config": [-1, 1], "energy": -3},
                }
            },
        )

    transport = httpx.MockTransport(respond)
    with IsingQClient("account-secret", transport=transport) as client:
        handle = client.submit(program(), calculate_count=2)
    # A new client can recover using only the saved ID and compiled model.
    with IsingQClient("account-secret", transport=transport) as client:
        result = client.poll(handle, program())
    assert len(requests) == 4
    assert result["status"] == "succeeded"
    assert result["result"]["objective_value"] == 2


@pytest.mark.parametrize(
    "failure", ["signature", "upload", "create", "parse", "handle"]
)
def test_failures_classify_uncertainty_and_never_retry_or_leak(failure):
    creates = []

    def respond(request):
        if request.url.path.endswith("getPostSignatureForOssUpload"):
            if failure == "signature":
                return httpx.Response(500, text="account-secret")
            return httpx.Response(200, json=signature())
        if request.url.host == "upload.example.test":
            return httpx.Response(500 if failure == "upload" else 200)
        creates.append(request)
        if failure == "create":
            raise httpx.ReadTimeout("account-secret", request=request)
        if failure == "parse":
            return httpx.Response(200, text="account-secret")
        return httpx.Response(200, json={"id": "../../invalid"})

    with (
        IsingQClient(
            "account-secret", transport=httpx.MockTransport(respond)
        ) as client,
        pytest.raises(IsingQError) as caught,
    ):
        client.submit(program())
    assert "account-secret" not in str(caught.value)
    assert caught.value.submission_outcome == (
        "not_created" if failure in ("signature", "upload") else "unknown"
    )
    assert len(creates) == (0 if failure in ("signature", "upload") else 1)


@pytest.mark.parametrize(
    "raw", [None, {"bits": [1]}, {"bits": [True, 1]}, {"bits": [0, 2]}]
)
def test_terminal_invalid_result_is_failed(raw):
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"status": 2, "result": raw})
    )
    with IsingQClient("test-key", transport=transport) as client:
        assert client.poll("saved-task", program())["status"] == "failed"


def test_redirect_does_not_forward_credentials():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://other.example.test"})

    with (
        IsingQClient("test-key", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(IsingQError),
    ):
        client.submit(program())
    assert len(requests) == 1


def test_poll_failure_does_not_claim_the_task_was_not_created():
    transport = httpx.MockTransport(lambda _: httpx.Response(503))
    with (
        IsingQClient("test-key", transport=transport) as client,
        pytest.raises(IsingQError) as caught,
    ):
        client.poll("existing-task", program())
    assert caught.value.phase == "poll"
    assert caught.value.submission_outcome == "not_applicable"


def test_integer_task_id_is_accepted_and_pollable():
    polled = []

    def respond(request):
        if request.url.path.endswith("getPostSignatureForOssUpload"):
            return httpx.Response(200, json=signature())
        if request.url.host == "upload.example.test":
            return httpx.Response(200)
        if request.url.path == "/tasks/create-general":
            return httpx.Response(200, json={"data": {"id": 12345}})
        polled.append(request.url.path)
        return httpx.Response(200, json={"data": {"status": 0}})

    transport = httpx.MockTransport(respond)
    with IsingQClient("test-key", transport=transport) as client:
        handle = client.submit(program())
        assert handle == "12345"
        assert client.poll(handle, program())["status"] == "running"
        assert client.poll(12345, program())["status"] == "running"
    assert polled == ["/tasks/12345", "/tasks/12345"]


def test_boolean_task_id_is_rejected():
    def respond(request):
        if request.url.path.endswith("getPostSignatureForOssUpload"):
            return httpx.Response(200, json=signature())
        if request.url.host == "upload.example.test":
            return httpx.Response(200)
        return httpx.Response(200, json={"data": {"id": True}})

    with (
        IsingQClient("test-key", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(IsingQError) as caught,
    ):
        client.submit(program())
    assert caught.value.submission_outcome == "unknown"


def test_null_bits_fall_back_to_spin_config():
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200,
            json={"status": 1, "result": {"bits": None, "spin_config": [-1, 1]}},
        )
    )
    with IsingQClient("test-key", transport=transport) as client:
        snapshot = client.poll("saved-task", program())
    assert snapshot["status"] == "succeeded"
    assert snapshot["result"]["bits"] == [0, 1]


def test_missing_httpx_raises_optional_dependency_error(monkeypatch):
    import builtins

    from quspect.backends._exceptions import OptionalDependencyError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("No module named 'httpx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(OptionalDependencyError, match=r"quspect\[isingq\]"):
        IsingQClient("test-key")
