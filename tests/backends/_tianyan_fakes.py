"""Offline emulation of the Tianyan cloud HTTP API.

:class:`FakeTianyan` is a transport for :class:`TianyanClient`.  It enforces
the server behaviour recorded on the real cloud (November 2026):

* login is a form POST (``grant_type=openId``); other calls need
  ``Authorization: Bearer`` and an expired token answers HTTP 200 with
  ``code: 401``;
* a qubit may be measured once and nothing may follow its measurement;
  ``I`` needs a positive duration;
* on a QPU, disabled qubits and CZs between uncoupled qubits are rejected,
  and a native batch must measure the same qubits in the same order;
* results report ``resultStatus[0]`` = measured qubits in ``M`` order, one
  row per shot in that order, and ``probability`` keys with the first
  measured qubit **rightmost**.

With ``evaluate=True`` (needs cqlib) programs are parsed by cqlib's QCIS
parser and evaluated exactly on its statevector simulator; shots are a
deterministic largest-remainder allocation of the exact probabilities.
"""

from __future__ import annotations

import itertools
import json
import pathlib
import urllib.parse
from typing import Any

KEY = "TESTKEY+/=="
TOKEN = "token-1"


def real_config() -> dict:
    """tianyan176 configuration recorded from the real cloud (trimmed)."""
    path = pathlib.Path(__file__).parent / "data" / "tianyan_config_tianyan176.json"
    return json.loads(path.read_text())


def line_config(n: int) -> dict:
    """A tiny chip: qubits Q0..Q{n-1} on a line, perfect calibration."""
    qubits = [f"Q{i}" for i in range(n)]
    couplers = {f"G{i}": [f"Q{i}", f"Q{i + 1}"] for i in range(n - 1)}
    perfect = {"param_list": [0.0] * n, "qubit_used": qubits}
    ones = {"param_list": [1.0] * n, "qubit_used": qubits}
    return {
        "disabledQubits": "",
        "disabledCouplers": "",
        "overview": {
            "qubits": qubits,
            "couplers": list(couplers),
            "coupler_map": couplers,
        },
        "qubit": {"singleQubit": {"gate error": perfect}},
        "readout": {
            "readoutArray": {"|0> readout fidelity": ones, "|1> readout fidelity": ones}
        },
        "twoQubitGate": {
            "czGate": {
                "gate error": {
                    "param_list": [0.0] * (n - 1),
                    "qubit_used": list(couplers),
                }
            }
        },
    }


def _qubit(token: str) -> int:
    return int(token[1:])


def check_program(text: str, chip: dict | None) -> list[int]:
    """Apply the server's submission checks; return measured qubits in M order."""
    measured: list[int] = []
    disabled: set[int] = set()
    edges: set[tuple[int, int]] = set()
    if chip is not None:
        disabled = {_qubit(q) for q in chip["disabledQubits"].split(",") if q}
        bad_couplers = {g for g in chip["disabledCouplers"].split(",") if g}
        for g, (a, b) in chip["overview"]["coupler_map"].items():
            if g not in bad_couplers:
                edges.add(tuple(sorted((_qubit(a), _qubit(b)))))
        known = {_qubit(q) for q in chip["overview"]["qubits"]}
    for row, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        op = parts[0]
        qubits = [_qubit(p) for p in parts[1:] if p.startswith("Q")]
        params = [p for p in parts[1:] if not p.startswith("Q")]
        if op == "I" and (not params or float(params[0]) <= 0):
            raise ValueError(
                f"The I gate in QCIS command does not pass the test(row:{row})"
            )
        if chip is not None:
            for q in qubits:
                if q in disabled or q not in known:
                    raise ValueError(f"QCIS command Q{q} is not available(row:{row})")
            if op == "CZ" and tuple(sorted(qubits)) not in edges:
                raise ValueError(
                    "The selected qubits cannot execute the current circuit"
                )
        if op == "M":
            for q in qubits:
                if q in measured:
                    raise ValueError(f"no other gate operation after M (row:{row})")
                measured.append(q)
        elif op != "B" and any(q in measured for q in qubits):
            raise ValueError(f"no other gate operation after M (row:{row})")
    if not measured:
        raise ValueError("no measurement")
    return measured


def exact_probabilities(text: str, measured: list[int]) -> dict[tuple[int, ...], float]:
    """Exact distribution of the measured qubits (tuple in M order)."""
    from cqlib.circuits import Circuit
    from cqlib.simulator import StatevectorSimulator

    used = sorted(
        {
            _qubit(t)
            for line in text.splitlines()
            for t in line.split()[1:]
            if t.startswith("Q")
        }
    )
    dense = {q: i for i, q in enumerate(used)}
    body = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] in {"M", "B"}:
            continue
        body.append(
            " ".join(f"Q{dense[_qubit(p)]}" if p.startswith("Q") else p for p in parts)
        )
    # cqlib's simulator orders qubits by first appearance, not by index:
    # declare every wire first, in ascending order (I is skipped by the
    # simulator), so state index bit i is dense qubit i.
    body = [f"I Q{i} 0" for i in range(len(used))] + body
    state = StatevectorSimulator(Circuit.load("\n".join(body))).statevector()
    out: dict[tuple[int, ...], float] = {}
    for key, amplitude in state.items():
        weight = abs(complex(amplitude)) ** 2
        if weight < 1e-15:
            continue
        index = int(key, 2)  # cqlib: Q0 is the rightmost character
        bits = tuple((index >> dense[q]) & 1 for q in measured)
        out[bits] = out.get(bits, 0.0) + weight
    return out


def allocate(probabilities: dict, shots: int) -> dict:
    """Deterministic largest-remainder allocation of ``shots``."""
    raw = {k: v * shots for k, v in probabilities.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rest = shots - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:rest]:
        counts[k] += 1
    return counts


class FakeTianyan:
    """An in-memory Tianyan cloud usable as a ``TianyanClient`` transport."""

    _ids = itertools.count(1)

    def __init__(self, *, chips: dict[str, dict] | None = None, evaluate: bool = True):
        self.chips = {"tianyan176": real_config()} if chips is None else chips
        self.evaluate = evaluate
        self.submissions: list[dict[str, Any]] = []
        self.tasks: dict[str, dict[str, Any]] = {}
        self.logins = 0
        self.token = TOKEN
        self.expire_token_once = False
        self.fail_next: dict[str, int] = {}  # path suffix -> number of 504s
        self.pending_polls = 1

    def machines(self) -> list[dict]:
        out = [
            {"code": name, "id": str(i), "label": "1", "isToll": 1, "status": 0}
            for i, name in enumerate(self.chips)
        ]
        out += [
            {"code": "tianyan_sw", "id": "3", "label": "3", "isToll": 1, "status": 0},
            {"code": "tianyan_sa", "id": "9", "label": "6", "isToll": 1, "status": 0},
            {"code": "tianyan24", "id": "99", "label": "1", "isToll": 1, "status": 1},
        ]
        return out

    # transport ---------------------------------------------------------

    def __call__(self, method, url, headers, body):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        for suffix, remaining in list(self.fail_next.items()):
            if path.endswith(suffix) and remaining:
                self.fail_next[suffix] = remaining - 1
                return 504, b"<html>504 Gateway Time-out</html>"
        if path.endswith("/oauth2/sdk/opnId"):
            form = dict(urllib.parse.parse_qsl(body.decode()))
            assert headers["Content-Type"] == "application/x-www-form-urlencoded"
            if form != {
                "grant_type": "openId",
                "openId": KEY,
                "account_type": "member",
            }:
                return 200, self._json({"code": 1, "msg": "login failed"})
            self.logins += 1
            self.token = f"token-{self.logins}"
            return 200, self._json({"code": 0, "data": {"access_token": self.token}})
        if headers.get("Authorization") != f"Bearer {self.token}" or (
            self.expire_token_once
        ):
            self.expire_token_once = False
            return 200, self._json(
                {"code": 401, "msg": '{"CN":"令牌失效","EN":"The token is invalid"}'}
            )
        assert headers.get("basicToken") == self.token
        payload = json.loads(body) if body else None
        if path.endswith("/quantumComputer/list"):
            return 200, self._ok(self.machines())
        if "/experiment/download/config/" in path:
            chip = self.chips.get(path.rsplit("/", 1)[-1])
            if chip is None:
                return 200, self._json(
                    {"code": 1, "msg": "Only quantum physics machines"}
                )
            return 200, self._ok(chip)
        if path.endswith("/experiment/temporary/save"):
            return 200, self._submit(payload)
        if path.endswith("/experiment/result/find"):
            return 200, self._results(payload["query_ids"])
        return 404, b"{}"

    @staticmethod
    def _json(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode()

    def _ok(self, data) -> bytes:
        return self._json({"code": 0, "msg": "success", "data": data})

    def _submit(self, payload) -> bytes:
        programs = payload["inputCode"]
        machine = payload["computerCode"]
        assert payload["languageCode"] == "qcis" and payload["source"] == "SDK"
        chip = self.chips.get(machine)
        if machine not in self.chips and machine != "tianyan_sw":
            return self._json({"code": 1, "msg": f"unknown machine {machine}"})
        try:
            orders = [check_program(text, chip) for text in programs]
        except ValueError as exc:
            return self._json({"code": 1, "msg": json.dumps({"EN": str(exc)})})
        if (
            chip is not None
            and len(programs) > 1
            and len({tuple(o) for o in orders}) > 1
        ):
            return self._json({"code": 1, "msg": "提交任务失败：参数非法"})
        self.submissions.append(payload)
        ids = []
        for text, order in zip(programs, orders, strict=True):
            task_id = str(next(self._ids) + 2118000000000000000)
            probs = (
                exact_probabilities(text, order)
                if self.evaluate
                else {tuple(0 for _ in order): 1.0}
            )
            self.tasks[task_id] = {
                "order": order,
                "probs": probs,
                "shots": payload["shots"],
                "polls": self.pending_polls,
            }
            ids.append(task_id)
        return self._ok({"query_ids": ids})

    def _results(self, ids) -> bytes:
        assert len(ids) <= 50
        tasks = [self.tasks[i] for i in ids]
        for task in tasks:
            if task["polls"] > 0:
                task["polls"] -= 1
                return self._ok({"experimentResultModelList": []})
        out = []
        for task_id, task in zip(ids, tasks, strict=True):
            counts = allocate(task["probs"], task["shots"])
            rows = [list(task["order"])]
            for bits, n in sorted(counts.items()):
                rows += [list(bits)] * n
            probability = {
                "".join(str(b) for b in reversed(bits)): round(p, 12)
                for bits, p in task["probs"].items()
            }
            out.append(
                {
                    "experimentTaskId": task_id,
                    "resultStatus": rows,
                    "probability": probability,
                }
            )
        return self._ok({"experimentResultModelList": out})
