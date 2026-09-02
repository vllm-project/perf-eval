#!/usr/bin/env python3
"""Stdlib-only regression tests for the shell server lifecycle helpers."""

import hashlib
import http.server
import json
import os
import socket
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_SH = os.path.join(ROOT, "lib", "server.sh")


def run_bash(command, *, env=None, timeout=10):
    return subprocess.run(
        ["bash", "-c", f"source {SERVER_SH!r}; {command}"],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def first_port(seed):
    digest = hashlib.sha256(seed.encode()).digest()
    return 20000 + int.from_bytes(digest[:4], "big") % 40000


def test_job_ids_get_distinct_stable_ports():
    first = run_bash("pick_server_port", env={"BUILDKITE_JOB_ID": "job-a"})
    again = run_bash("pick_server_port", env={"BUILDKITE_JOB_ID": "job-a"})
    second = run_bash("pick_server_port", env={"BUILDKITE_JOB_ID": "job-b"})
    assert first.returncode == again.returncode == second.returncode == 0
    assert first.stdout == again.stdout
    assert first.stdout != second.stdout


def test_port_picker_skips_a_port_already_in_use():
    seed = "occupied-port"
    occupied = first_port(seed)
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", occupied))
        result = run_bash("pick_server_port", env={"BUILDKITE_JOB_ID": seed})
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) != occupied


def test_health_check_verifies_the_served_model():
    expected_model = "org/expected-model"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = b""
            elif self.path == "/v1/models":
                body = json.dumps({"data": [{"id": expected_model}]}).encode()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        match = run_bash(f"server_is_healthy {port} org/expected-model")
        mismatch = run_bash(f"server_is_healthy {port} org/other-model")
    finally:
        server.shutdown()
        thread.join()
    assert match.returncode == 0, match.stderr
    assert mismatch.returncode != 0


def test_stop_process_is_bounded_when_term_is_ignored():
    started = time.monotonic()
    result = run_bash(
        "bash -c 'trap \"\" TERM; while :; do sleep 1; done' & "
        "pid=$!; sleep 0.1; stop_process \"$pid\" 1",
        timeout=4,
    )
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 3


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
