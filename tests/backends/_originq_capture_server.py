"""Recording stand-in for the Origin Quantum cloud, run as a separate process.

Standard library only and imported by nothing: the contract tests launch it
with ``python -I <this file> <config.json>``.  pyqpanda3's blocking calls hold
the GIL, so an in-process server thread would never get to answer.

The config JSON holds ``log_path``, ``port_path``, ``chip`` and
``resources``.  Every POST is appended to ``log_path`` as one JSON line; the
bound port is written to ``port_path`` once the server is listening.

The task ID steers the two answers a test needs to vary: ``cancelTask``
refuses IDs starting with ``REFUSE``, and ``taskDetail`` reports the text
after ``STATE-`` as ``taskState`` (``1`` otherwise).
"""

import http.server
import json
import os
import sys

with open(sys.argv[1]) as handle:
    CONFIG = json.load(handle)
OK = {"success": True, "code": 10000, "message": "success"}


class Handler(http.server.BaseHTTPRequestHandler):
    def _reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._reply(CONFIG["resources"])

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        entry = {
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        }
        with open(CONFIG["log_path"], "a") as handle:
            handle.write(json.dumps(entry) + "\n")
        task_id = str(body.get("taskId") or "CONTRACT")
        if self.path.endswith("chipInfo.json"):
            self._reply({**OK, "obj": CONFIG["chip"]})
        elif self.path.endswith("submit.json"):
            self._reply({**OK, "obj": {"taskId": "CONTRACT"}})
        elif self.path.endswith("cancelTask.json"):
            if task_id.startswith("REFUSE"):
                refusal = {"success": False, "code": 10001, "message": "refused"}
                self._reply({**refusal, "obj": None})
            else:
                self._reply({**OK, "obj": None})
        else:
            state = task_id.partition("STATE-")[2] or "1"
            self._reply({**OK, "obj": {"taskId": task_id, "taskState": state}})

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
temporary = CONFIG["port_path"] + ".tmp"
with open(temporary, "w") as handle:
    handle.write(str(server.server_address[1]))
os.replace(temporary, CONFIG["port_path"])  # atomic: the reader never sees half
server.serve_forever()
