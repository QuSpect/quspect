"""Exhaustive objective identities and mocked cloud protocol, without hardware."""

import io
import itertools
import json
import re

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


def signature_with(**extra):
    body = signature()
    body["data"].update(extra)
    return body


def form_key(request):
    """Object key of a multipart upload request."""
    found = re.search(rb'name="key"\r\n\r\n([^\r]*)\r\n', request.content)
    assert found is not None
    return found.group(1).decode()


class FakeIsing:
    """Mocked iSing API and upload host.

    Like the provider's storage policy (``["starts-with", "$key", dir]``), the
    upload host answers 403 to an object key outside the signed directory.
    ``upload``, ``create`` and ``poll`` replace the default answers.
    """

    def __init__(self, *, sign=None, upload=None, create=None, poll=None):
        self.sign = signature() if sign is None else sign
        self.upload = upload
        self.create = create
        self.poll = poll
        self.keys = []
        self.created = []
        self.polled = 0

    def __call__(self, request):
        if request.url.host == "upload.example.test":
            assert "authorization" not in request.headers
            if self.upload is not None:
                return self.upload(request)
            key = form_key(request)
            required = self.sign["data"].get("dir")
            if isinstance(required, str) and not key.startswith(required):
                return httpx.Response(
                    403, text="<Error><Code>AccessDenied</Code></Error>"
                )
            self.keys.append(key)
            return httpx.Response(200)
        if request.url.path.endswith("getPostSignatureForOssUpload"):
            return httpx.Response(200, json=self.sign)
        if request.url.path == "/tasks/create-general":
            self.created.append(json.loads(request.content))
            if self.create is not None:
                return self.create(request)
            return httpx.Response(
                200, json={"data": {"id": f"task-{len(self.created)}"}}
            )
        self.polled += 1
        return self.poll(request)

    def client(self, key="account-secret"):
        return IsingQClient(key, transport=httpx.MockTransport(self))


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


SIGNED_DIR = "open-api/774/20261109/bf0ff1ff56fe6f4a514f3fce57efb89c/"
LEGACY_KEY = r"[0-9a-f]{32}/quspect-qubo\.csv"


def test_upload_key_is_under_the_signed_dir_and_unique_per_submission():
    """Since November 2026 the storage policy admits keys under ``dir`` only."""
    fake = FakeIsing(sign=signature_with(dir=SIGNED_DIR))
    other = compile_isingq(bqm_from_qubo_matrix([[5, -2], [0, 1]]))
    with fake.client() as client:
        assert client.submit(program()) == "task-1"
        assert client.submit(other) == "task-2"
        assert client.submit(program()) == "task-3"
    assert len(fake.keys) == 3
    for key, created in zip(fake.keys, fake.created, strict=True):
        assert re.fullmatch(re.escape(SIGNED_DIR) + LEGACY_KEY, key)
        assert created["inputJFile"] == "https://upload.example.test/" + key
    # The provider may sign the same dir twice: the key must not repeat.
    assert len(set(fake.keys)) == 3


@pytest.mark.parametrize(
    "sign", [signature(), signature_with(dir=None), signature_with(dir="")]
)
def test_upload_key_without_a_signed_dir_keeps_the_previous_layout(sign):
    fake = FakeIsing(sign=sign)
    with fake.client() as client:
        client.submit(program())
        client.submit(program())
    assert all(re.fullmatch(LEGACY_KEY, key) for key in fake.keys)
    assert len(set(fake.keys)) == 2


@pytest.mark.parametrize(
    ("signed", "prefix"),
    [
        ("open-api/774/dt=20261109/", "open-api/774/dt=20261109/"),
        ("open-api/774/user@example.com/", "open-api/774/user@example.com/"),
        ("open-api/774/no-trailing-slash", "open-api/774/no-trailing-slash/"),
        ("开放平台/a b+c~(1)/", "开放平台/a b+c~(1)/"),
        ("a..b/.hidden/...", "a..b/.hidden/.../"),
    ],
)
def test_signed_dir_is_not_limited_to_a_character_class(signed, prefix):
    fake = FakeIsing(sign=signature_with(dir=signed))
    with fake.client() as client:
        client.submit(program())
    assert fake.keys[0].startswith(prefix)
    assert re.fullmatch(LEGACY_KEY, fake.keys[0][len(prefix) :])


@pytest.mark.parametrize(
    "signed",
    [
        "/absolute/path/",
        "../escape/",
        "open-api/../other-account/",
        "open-api/./774/",
        "open-api\\..\\other/",
        "open-api/774\n/",
        "open-api/\x00/",
        "open-api/\x7f/",
        "open-api/\x85/",
        "x" * 2000 + "/",
        774,
        ["open-api"],
        {"path": "open-api"},
    ],
)
def test_dangerous_signed_dir_fails_before_anything_is_uploaded(signed):
    fake = FakeIsing(sign=signature_with(dir=signed))
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.phase == "upload_signature"
    assert caught.value.submission_outcome == "not_created"
    assert fake.keys == []
    assert fake.created == []


def test_refused_upload_reports_the_http_status():
    fake = FakeIsing(
        upload=lambda _: httpx.Response(
            403, text="<Error><Code>AccessDenied</Code></Error>"
        )
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.phase == "upload"
    assert caught.value.submission_outcome == "not_created"
    assert caught.value.status_code == 403
    assert caught.value.provider_message is None  # not the provider's JSON message
    assert "HTTP 403" in str(caught.value)
    assert fake.created == []


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_refused_task_creation_is_rejected_with_status_and_message(status):
    """A client error is a definite refusal, not an unknown outcome."""
    fake = FakeIsing(
        create=lambda _: httpx.Response(
            status, json={"code": status, "message": "余额不足"}
        )
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.phase == "create_task"
    assert caught.value.submission_outcome == "rejected"
    assert caught.value.status_code == status
    assert caught.value.provider_message == "余额不足"
    assert f"HTTP {status}" in str(caught.value)
    assert "rejected" in str(caught.value)
    assert len(fake.created) == 1  # never retried


@pytest.mark.parametrize("status", [302, 408, 409, 425, 429, 500, 502, 503, 504])
def test_ambiguous_task_creation_status_stays_unknown(status):
    """Timing, conflict and server errors may hide a created task."""
    fake = FakeIsing(
        create=lambda _: httpx.Response(status, json={"message": "try again later"})
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.submission_outcome == "unknown"
    assert caught.value.status_code == status
    assert caught.value.provider_message == "try again later"
    assert len(fake.created) == 1


def test_task_creation_without_a_response_has_no_status():
    def create(request):
        raise httpx.ReadTimeout("account-secret", request=request)

    fake = FakeIsing(create=create)
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.submission_outcome == "unknown"
    assert caught.value.status_code is None
    assert caught.value.provider_message is None
    assert "account-secret" not in str(caught.value)


def test_accepted_response_without_task_id_stays_unknown_and_keeps_the_message():
    """HTTP 200 does not say the task was refused; the reason is kept."""
    fake = FakeIsing(
        create=lambda _: httpx.Response(
            200,
            json={"code": 500, "message": "余额不足", "data": None},
        )
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.submission_outcome == "unknown"
    assert caught.value.status_code == 200
    assert caught.value.provider_message == "余额不足"


def test_created_status_without_task_id_reports_the_actual_status():
    fake = FakeIsing(create=lambda _: httpx.Response(201, json={"data": {}}))
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.submission_outcome == "unknown"
    assert caught.value.status_code == 201
    assert caught.value.provider_message is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "key account-secret is disabled, token temporary-token expired",
            "key [REDACTED] is disabled, token [REDACTED] expired",
        ),
        (
            "denied for Authorization: abc123; see https://oss.example.test/x?sig=1",
            "denied for Authorization=[REDACTED]; see [REDACTED_URL]",
        ),
        ("line one\r\n\tline\x00two \u202e end", "line one line two end"),
        ("<html><body>502 Bad Gateway</body></html>", None),
        ('Traceback (most recent call last):\n  File "app.py"', None),
        ("java.lang.IllegalStateException: boom\n\tat com.example.Task", None),
        ("", None),
        ("   ", None),
        (["not", "text"], None),
        (12345, None),
    ],
)
def test_provider_message_is_sanitized(message, expected):
    fake = FakeIsing(create=lambda _: httpx.Response(400, json={"message": message}))
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert caught.value.submission_outcome == "rejected"
    assert caught.value.provider_message == expected
    assert "account-secret" not in str(caught.value)
    assert "account-secret" not in repr(caught.value)


def test_provider_message_is_bounded():
    fake = FakeIsing(
        create=lambda _: httpx.Response(400, json={"message": "reason " * 400})
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    assert len(caught.value.provider_message) == 512

    fake = FakeIsing(
        create=lambda _: httpx.Response(400, json={"message": "x" * (64 * 1024)})
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.submit(program())
    # An oversized error body is not read; the status still classifies it.
    assert caught.value.provider_message is None
    assert caught.value.submission_outcome == "rejected"
    assert caught.value.status_code == 400


def test_error_defaults_stay_compatible():
    error = IsingQError("create_task", "unknown")
    assert str(error) == "iSing request failed during create_task; submission unknown"
    assert error.status_code is None
    assert error.provider_message is None
    assert IsingQError("upload").submission_outcome == "not_created"
    assert IsingQError("poll").submission_outcome == "not_applicable"


def poll_once(payload, *, status_code=200):
    fake = FakeIsing(poll=lambda _: httpx.Response(status_code, json=payload))
    with fake.client() as client:
        return client.poll("saved-task", program())


@pytest.mark.parametrize("status", ["2", " 2 ", 2.0, "1", "3"])
def test_poll_accepts_a_numeric_status_sent_as_text_or_float(status):
    snapshot = poll_once(
        {"data": {"status": status, "result": {"spin_config": [0, 1], "energy": -3}}}
    )
    assert snapshot["status"] == "succeeded"
    assert snapshot["provider_status"] == int(float(status))
    assert type(snapshot["provider_status"]) is int
    assert snapshot["result"]["objective_value"] == 2


@pytest.mark.parametrize("status", ["0", "4", 0.0, 7])
def test_poll_keeps_other_numeric_statuses_running(status):
    snapshot = poll_once({"data": {"status": status}})
    assert snapshot == {"status": "running", "provider_status": int(float(status))}


@pytest.mark.parametrize(
    "status", [None, True, "done", "", "2.5", 2.5, float("nan"), [2], {"code": 2}]
)
def test_poll_refuses_a_status_that_is_not_an_integer(status):
    body = json.dumps({"data": {"status": status, "result": {"bits": [0, 1]}}})
    fake = FakeIsing(poll=lambda _: httpx.Response(200, content=body))
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.poll("saved-task", program())
    assert caught.value.phase == "poll"
    assert caught.value.submission_outcome == "not_applicable"
    assert caught.value.status_code == 200


@pytest.mark.parametrize(
    ("energy", "expected"),
    [("-3.0", -3.0), (" -3 ", -3.0), ("1e-3", 0.001), (-3, -3), (-3.5, -3.5)],
)
def test_poll_accepts_energy_sent_as_text(energy, expected):
    snapshot = poll_once(
        {"data": {"status": 2, "result": {"spin_config": [0, 1], "energy": energy}}}
    )
    assert snapshot["status"] == "succeeded"
    assert snapshot["result"]["provider_energy"] == expected
    assert type(snapshot["result"]["provider_energy"]) is type(expected)
    assert snapshot["result"]["objective_value"] == 2


@pytest.mark.parametrize(
    "energy", ["n/a", "", "nan", "inf", True, [1.0], {"value": 1.0}, 1e400]
)
def test_unusable_energy_does_not_lose_a_finished_task(energy):
    """The provider energy is diagnostic only; the solution vector is the result."""
    body = json.dumps(
        {"data": {"status": 2, "result": {"bits": [0, 1], "energy": energy}}}
    )
    fake = FakeIsing(poll=lambda _: httpx.Response(200, content=body))
    with fake.client() as client:
        snapshot = client.poll("saved-task", program())
    assert snapshot["status"] == "succeeded"
    assert snapshot["result"]["bits"] == [0, 1]
    assert snapshot["result"]["provider_energy"] is None
    assert snapshot["result"]["objective_value"] == 2


def test_failed_task_keeps_the_provider_message():
    snapshot = poll_once(
        {
            "data": {
                "status": 3,
                "result": None,
                "message": "matrix rejected: value out of range",
            }
        }
    )
    assert snapshot == {
        "status": "failed",
        "provider_status": 3,
        "provider_message": "matrix rejected: value out of range",
    }
    # The envelope's message is used when the task object has none.
    snapshot = poll_once(
        {"message": "task failed", "data": {"status": 1, "result": {"bits": [1]}}}
    )
    assert snapshot == {
        "status": "failed",
        "provider_status": 1,
        "error": "Invalid iSing solution",
        "provider_message": "task failed",
    }
    # No message, no key; a message never contains the account key.
    assert poll_once({"status": 2, "result": None}) == {
        "status": "failed",
        "provider_status": 2,
    }
    fake = FakeIsing(
        poll=lambda _: httpx.Response(
            200, json={"status": 2, "message": "bad key account-secret"}
        )
    )
    with fake.client() as client:
        snapshot = client.poll("saved-task", program())
    assert snapshot["provider_message"] == "bad key [REDACTED]"


def test_message_is_not_attached_to_running_or_succeeded_snapshots():
    assert poll_once({"data": {"status": 0, "message": "queued"}}) == {
        "status": "running",
        "provider_status": 0,
    }
    snapshot = poll_once(
        {"data": {"status": 2, "message": "ok", "result": {"bits": [0, 1]}}}
    )
    assert set(snapshot) == {"status", "provider_status", "result"}


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_poll_http_failure_reports_status_and_message(status_code):
    fake = FakeIsing(
        poll=lambda _: httpx.Response(status_code, json={"message": "no such task"})
    )
    with fake.client() as client, pytest.raises(IsingQError) as caught:
        client.poll("saved-task", program())
    assert caught.value.phase == "poll"
    assert caught.value.submission_outcome == "not_applicable"
    assert caught.value.status_code == status_code
    assert caught.value.provider_message == "no such task"
