"""Offline emulation of the Origin Quantum cloud HTTP API.

:class:`FakeCloud` is a transport for :class:`OriginQClient`.  It enforces the
server behaviour recorded on the real cloud (November 2026):

* ``Authorization: oqcs_auth=<key>`` on every call, a ``task_mode`` header on
  submit and taskDetail (``2`` exactly for multi-program submissions);
* result bit ``k`` is the qubit of the ``k``-th ``MEASURE``; the
  classical-bit operand is ignored and non-ascending measurement order is
  corrupted on the server, so the fake refuses it;
* the simulator takes one program per task; QPUs with ``mappingFlag=false``
  must only use coupling edges between available qubits;
* results are probabilities with hex keys.

Cancellation (``/oqcs/task/cancelTask.json``): the fake expects the request
pyqpanda3 0.5.1 sends (``task_mode: 0``, ``{"taskId": id}``) and answers as
the real cloud did on 12 November 2026 (WK_C180):

* accepted: HTTP 200, ``success: true``, code 10000; the task then reports
  ``taskState`` :data:`LIVE_CANCELLED_STATE` (``FakeCloud.cancelled_state``);
* refused: HTTP 200, ``success: false``, code 10096 and the message
  :data:`LIVE_CANCEL_REFUSAL`, seen for a task that was already cancelled and
  (by another client) for one submitted 3.4 s earlier, which then completed
  normally; ``FakeCloud.refuse_cancel`` reproduces the latter.

Assumptions, not observations: that a finished task is refused the same way,
and the answer for an unknown task.

Programs are parsed by pyqpanda3's own OriginIR parser and evaluated exactly
on its CPUQVM, so QuSpect's emitter is checked by an independent
implementation.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

import numpy as np

_QUBIT = re.compile(r"q\[(\d+)\]")
KEY = "00" + "11" * 40

#: The real cloud's message when it declines a cancel (with code 10096).
LIVE_CANCEL_REFUSAL = "Unable to cancel this task. Please refresh and try again."
LIVE_CANCEL_REFUSAL_CODE = 10096
#: ``taskState`` the real cloud reported after it accepted a cancel.
LIVE_CANCELLED_STATE = "35"

#: WK_C180_2 available qubits and coupling map as reported by chipInfo.
WK_C180_2_AVAILABLE = [
    39, 40, 48, 49, 50, 57, 58, 59, 66, 67, 68, 75, 76, 77, 84, 85, 86,
]  # fmt: skip
WK_C180_2_EDGES = [
    (39, 48), (39, 49), (40, 49), (40, 50), (48, 57), (49, 58), (50, 59),
    (57, 66), (57, 67), (58, 67), (58, 68), (59, 68), (66, 75), (67, 76),
    (68, 77), (75, 84), (75, 85), (76, 85), (76, 86), (77, 86),
]  # fmt: skip


def real_chip_info() -> dict:
    """chipInfo ``obj`` recorded from WK_C180_2 on the real cloud."""
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "originq_chip_WK_C180_2.json"
    return json.loads(path.read_text())


def chip_info_payload(num_qubits, available, edges, max_shots=49600) -> dict:
    adjacency: dict[str, list[dict[str, float]]] = {}
    for a, b in edges:
        adjacency.setdefault(str(a), []).append({"v": b, "w": 0.99})
        adjacency.setdefault(str(b), []).append({"v": a, "w": 0.99})
    return {
        "ChipID": "fake",
        "QubitCount": num_qubits,
        "AvailableQubits": list(available),
        "adjMatrix": adjacency,
        "BasicQGate": ["RPhi", "CZ"],
        "MaxShots": max_shots,
        "errCode": 0,
        "errInfo": "",
    }


def exact_cloud_probabilities(ir: str, *, edges=None, available=None) -> dict:
    """Emulate the cloud on OriginIR text: probabilities by MEASURE order."""
    from pyqpanda3 import core
    from pyqpanda3.intermediate_compiler import convert_originir_string_to_qprog

    # Parse the full program first: anything pyqpanda3 rejects would sit in
    # "computing" forever on the real cloud.
    convert_originir_string_to_qprog(ir)

    measured: list[int] = []
    body: list[str] = []
    controls: list[list[int]] = []
    used: set[int] = set()
    for raw_line in ir.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("QINIT", "CREG")):
            continue
        qubits = [int(q) for q in _QUBIT.findall(line)]
        used.update(qubits)
        if line.startswith("MEASURE"):
            measured.append(qubits[0])
            continue
        if line.startswith("CONTROL"):
            controls.append(qubits)
        elif line.startswith("ENDCONTROL"):
            controls.pop()
        elif edges is not None and not line.startswith(
            ("BARRIER", "DAGGER", "ENDDAGGER")
        ):
            acting = [q for group in controls for q in group] + qubits
            if available is not None and any(q not in available for q in acting):
                raise RuntimeError(f"gate on unavailable qubit: {line}")
            if len(acting) > 2:
                raise RuntimeError(f"{len(acting)}-qubit gate on a QPU: {line}")
            if len(acting) == 2 and tuple(sorted(acting)) not in edges:
                raise RuntimeError(f"gate {line} is not on a coupling edge")
        body.append(line)

    if measured != sorted(measured) or len(set(measured)) != len(measured):
        # The real cloud silently corrupts these results (two outcomes
        # collapse onto one key); refuse instead of returning garbage.
        raise RuntimeError(f"cloud cannot decode MEASURE order {measured}")

    # Monotone compaction keeps the qubit order (and so the key semantics)
    # while letting a 180-qubit chip program run on a small state vector.
    dense = {q: i for i, q in enumerate(sorted(used))}
    rewritten = [
        _QUBIT.sub(lambda m: f"q[{dense[int(m.group(1))]}]", line) for line in body
    ]
    n = max(1, len(dense))
    program = convert_originir_string_to_qprog(
        "\n".join([f"QINIT {n}", "CREG 1", *rewritten]) + "\n"
    )
    for q in range(n):  # make sure every wire exists in the state vector
        program << core.I(q)
    vm = core.CPUQVM()
    vm.run(program, 1)
    state = np.asarray(vm.result().get_state_vector())
    probs: dict[str, float] = {}
    order = [dense[q] for q in measured]
    for index, amplitude in enumerate(state):
        weight = float(abs(amplitude) ** 2)
        if weight < 1e-15:
            continue
        outcome = sum(((index >> q) & 1) << k for k, q in enumerate(order))
        probs[hex(outcome)] = probs.get(hex(outcome), 0.0) + weight
    return probs


class FakeCloud:
    """An in-memory Origin Quantum cloud usable as an ``OriginQClient`` transport."""

    _ids = itertools.count(1)

    def __init__(self, *, chips: dict[str, dict] | None = None, key: str = KEY):
        self.key = key
        self.chips = {"WK_C180_2": real_chip_info()} if chips is None else chips
        self.tasks: dict[str, dict[str, Any]] = {}
        self.submissions: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str, dict]] = []
        #: taskState sequence handed out by taskDetail before "3".
        self.pending_states: list[str] = ["1", "2"]
        self.fail_next_detail = 0
        self.fail_next_cancel = 0
        #: When set, every cancel is refused with this message and the task
        #: runs on, as the real cloud did (see :data:`LIVE_CANCEL_REFUSAL`).
        self.refuse_cancel: str | None = None
        #: ``taskState`` reported once a cancel was accepted: the value the
        #: real cloud reported.  ``None`` leaves the task's own state
        #: sequence untouched.
        self.cancelled_state: str | None = LIVE_CANCELLED_STATE

    def resources(self) -> list[dict]:
        out = [
            {"pyqpandaChipId": name, "machineType": 5, "onlineFlag": True,
             "available": True, "qubitNum": chip["QubitCount"]}
            for name, chip in self.chips.items()
        ]  # fmt: skip
        out += [
            {"pyqpandaChipId": "full_amplitude", "machineType": 0,
             "onlineFlag": True, "available": True, "qubitNum": 35},
            {"pyqpandaChipId": "full_amplitude", "machineType": 1,
             "onlineFlag": True, "available": True, "qubitNum": 34},
            {"pyqpandaChipId": "partial_amplitude", "machineType": 2,
             "onlineFlag": True, "available": True, "qubitNum": 68},
            {"pyqpandaChipId": "iontrap_sim", "machineType": 400,
             "onlineFlag": True, "available": True, "qubitNum": 30},
        ]  # fmt: skip
        return out

    # transport ---------------------------------------------------------

    def __call__(self, method, url, headers, body):
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path
        payload = json.loads(body) if body else None
        self.requests.append((method, path, dict(headers)))
        if headers.get("Authorization") != f"oqcs_auth={self.key}":
            return 200, self._fail("auth")
        if path.endswith("getResourceList.json"):
            return 200, json.dumps(self.resources()).encode()
        if path.endswith("chipInfo.json"):
            chip = self.chips.get(payload["ChipID"])
            return 200, self._ok(chip) if chip else self._fail("no chip")
        if path.endswith("submit.json"):
            return 200, self._submit(headers, payload)
        if path.endswith("taskDetail.json"):
            if "task_mode" not in headers:
                return 200, self._fail("Required request header 'task_mode'")
            if self.fail_next_detail:
                self.fail_next_detail -= 1
                raise ConnectionResetError("Remote end closed connection")
            return 200, self._detail(payload["taskId"])
        if path.endswith("cancelTask.json"):
            assert headers.get("task_mode") == "0", headers
            if self.fail_next_cancel:
                self.fail_next_cancel -= 1
                raise ConnectionResetError("Remote end closed connection")
            return 200, self._cancel(payload["taskId"])
        return 404, b"{}"

    @staticmethod
    def _ok(obj) -> bytes:
        return json.dumps(
            {"success": True, "code": 10000, "message": "success", "obj": obj}
        ).encode()

    @staticmethod
    def _fail(message) -> bytes:
        return json.dumps(
            {"success": False, "code": 10001, "message": message, "obj": None}
        ).encode()

    def _submit(self, headers, payload) -> bytes:
        programs = payload["QProg"]
        expected_mode = "2" if len(programs) > 1 else "1"
        assert headers.get("task_mode") == expected_mode, headers
        machine = payload["QMachineType"]
        config = payload["Configuration"]
        self.submissions.append(payload)
        if machine == "0":
            assert len(programs) == 1, "the simulator takes one program per task"
            assert set(config) == {"shot", "measure_type"}
            edges = available = None
        else:
            assert machine == "5"
            chip = self.chips[payload["ChipID"]]
            assert config["shot"] <= chip["MaxShots"]
            if config["mappingFlag"]:
                edges = available = None
            else:
                edges = {
                    (min(a, int(n["v"])), max(a, int(n["v"])))
                    for a_str, ns in chip["adjMatrix"].items()
                    for a in [int(a_str)]
                    for n in ns
                }
                available = set(chip["AvailableQubits"])
        results = [
            exact_cloud_probabilities(ir, edges=edges, available=available)
            for ir in programs
        ]
        task_id = f"FAKE{next(self._ids):028X}"
        self.tasks[task_id] = {"results": results, "states": list(self.pending_states)}
        return self._ok({"taskId": task_id})

    def _cancel(self, task_id) -> bytes:
        task = self.tasks.get(task_id)
        if task is None:
            return self._fail("unknown task")
        if self.refuse_cancel is not None:
            return self._refuse(self.refuse_cancel)
        # Already cancelled (observed), or the next taskDetail is final (assumed).
        if task.get("cancelled") or not task["states"]:
            return self._refuse(LIVE_CANCEL_REFUSAL)
        task["cancelled"] = True
        return self._ok(None)

    @staticmethod
    def _refuse(message) -> bytes:
        return json.dumps(
            {
                "success": False,
                "code": LIVE_CANCEL_REFUSAL_CODE,
                "message": message,
                "obj": None,
            }
        ).encode()

    def _detail(self, task_id) -> bytes:
        task = self.tasks.get(task_id)
        if task is None:
            return self._fail("unknown task")
        if task.get("cancelled") and self.cancelled_state is not None:
            return self._ok({"taskId": task_id, "taskState": self.cancelled_state})
        if task["states"]:
            return self._ok({"taskId": task_id, "taskState": task["states"].pop(0)})
        if task.get("error"):
            return self._ok(
                {"taskId": task_id, "taskState": "4", "errCode": 1,
                 "errInfo": task["error"]}
            )  # fmt: skip
        entries = [
            json.dumps({"key": list(r), "value": list(r.values())})
            for r in task["results"]
        ]
        return self._ok({"taskId": task_id, "taskState": "3", "taskResult": entries})


def quspect_exact_distribution(circuit) -> dict[str, float]:
    """Exact QuSpect-convention distribution of a terminal-measured circuit."""
    from quspect import QuantumCircuit
    from quspect.simulation import Simulator

    gates = QuantumCircuit()
    feeds: dict[int, list[int]] = {}
    for instruction in circuit.instructions:
        if instruction.is_measure:
            for q, c in zip(instruction.qubits, instruction.clbits, strict=True):
                feeds.setdefault(q, []).append(c)
        elif instruction.is_gate:
            gates.append(instruction.gate, instruction.qubits)
    for q in range(circuit.num_qubits):
        gates.id(q)
    sv = np.asarray(Simulator().run(gates, shots=1, save_statevector=True).statevector)
    out: dict[str, float] = {}
    for index, amplitude in enumerate(sv):
        weight = float(abs(amplitude) ** 2)
        if weight < 1e-15:
            continue
        chars = ["0"] * circuit.num_clbits
        for q, clbits in feeds.items():
            if (index >> q) & 1:
                for c in clbits:
                    chars[c] = "1"
        key = "".join(chars)
        out[key] = out.get(key, 0.0) + weight
    return out


def assert_same_distribution(actual: dict, expected: dict, tol: float = 1e-9) -> None:
    keys = set(actual) | set(expected)
    diff = {
        k: (actual.get(k, 0.0), expected.get(k, 0.0))
        for k in keys
        if abs(actual.get(k, 0.0) - expected.get(k, 0.0)) > tol
    }
    assert not diff, f"distribution mismatch (actual, expected): {diff}"


def start_capture_server(workdir, chip, resources, *, timeout: float = 60.0):
    """Launch ``_originq_capture_server.py`` and wait until it listens.

    Returns ``(url, recorded, process)``; ``recorded()`` reads the requests
    logged so far.  If the child exits early or never reports its port, the
    error includes its stderr.
    """
    import pathlib
    import subprocess
    import sys
    import time

    workdir = pathlib.Path(workdir)
    log_path, port_path = workdir / "requests.jsonl", workdir / "port"
    config = workdir / "config.json"
    config.write_text(
        json.dumps(
            {
                "log_path": str(log_path),
                "port_path": str(port_path),
                "chip": chip,
                "resources": resources,
            }
        )
    )
    script = pathlib.Path(__file__).with_name("_originq_capture_server.py")
    process = subprocess.Popen(
        [sys.executable, "-I", str(script), str(config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    while not port_path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            _, stderr = process.communicate(timeout=10)
            raise RuntimeError(
                f"capture server did not start (exit {process.returncode}): {stderr}"
            )
        time.sleep(0.05)

    def recorded() -> list[dict]:
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text().splitlines()]

    return f"http://127.0.0.1:{int(port_path.read_text())}", recorded, process
