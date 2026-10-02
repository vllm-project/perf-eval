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
NSYS_SH = os.path.join(ROOT, "lib", "nsys.sh")
NSYS_SERVE_SH = os.path.join(ROOT, "lib", "nsys_serve.sh")


def run_bash(command, *, env=None, timeout=10):
    return subprocess.run(
        ["bash", "-c", f"source {SERVER_SH!r}; source {NSYS_SH!r}; {command}"],
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


def test_readiness_fails_promptly_for_stopped_or_removed_docker_container():
    for inspect_result in ("echo false", "return 1"):
        result = run_bash(
            "set -euo pipefail; "
            "VLLM_SERVER_CONTAINER=failed-server; "
            "server_is_healthy() { return 1; }; "
            "docker() { case $1 in "
            f"inspect) {inspect_result} ;; "
            "logs) echo 'engine initialization failed'; return 1 ;; "
            "esac; }; "
            "wait_healthy 8000 3600 expected-model",
            timeout=3,
        )
        assert result.returncode == 1, result.stderr
        assert "stopped or became unavailable" in result.stderr
        assert "engine initialization failed" in result.stderr
        assert "server never came up" not in result.stderr


def test_readiness_waits_for_running_docker_container_to_become_healthy():
    result = run_bash(
        "set -euo pipefail; "
        "VLLM_SERVER_CONTAINER=starting-server; attempts=0; "
        "server_is_healthy() { attempts=$((attempts + 1)); (( attempts >= 2 )); }; "
        "docker() { [[ $1 == inspect ]] && echo true; }; "
        "sleep() { :; }; "
        "wait_healthy 8000 3600 expected-model; "
        '[[ $attempts == 2 ]]',
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "server healthy" in result.stdout


def test_native_readiness_does_not_require_docker():
    result = run_bash(
        "set -euo pipefail; "
        "VLLM_SERVER_PID=$$; VLLM_SERVER_CONTAINER=; attempts=0; "
        "server_is_healthy() { attempts=$((attempts + 1)); (( attempts >= 2 )); }; "
        "docker() { echo 'unexpected Docker call' >&2; return 1; }; "
        "sleep() { :; }; "
        "wait_healthy 8000 3600 expected-model",
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "unexpected Docker call" not in result.stderr


def test_nsys_configure_hands_capture_range_to_nsys():
    result = run_bash(
        "set -euo pipefail; cd \"$(mktemp -d)\"; "
        "WORKLOAD_SERVE_ARGS='-tp 8'; WORKLOAD_ENV='A=1'; "
        "nsys_configure docker results/wl wl-smoke; "
        'printf "%s\\n" "$NSYS_REPORT" "$WORKLOAD_SERVE_ARGS" "$WORKLOAD_ENV"; '
        "test -d results/wl/nsys",
    )
    assert result.returncode == 0, result.stderr
    report, serve_args, *env = result.stdout.splitlines()[1:]
    assert report == "/tmp/perf-eval-nsys/wl-smoke"
    assert serve_args.startswith("-tp 8 --profiler-config.profiler cuda")
    assert "--profiler-config.max_iterations 100" in serve_args
    assert env == ["A=1", "VLLM_WORKER_MULTIPROC_METHOD=spawn"]


def test_nsys_docker_server_keeps_container_for_report_copy():
    result = run_bash(
        "set -euo pipefail; NSYS_REPORT=/tmp/perf-eval-nsys/wl; "
        "docker() { [[ $1 == run ]] && printf '%s\\0' \"$@\"; return 0; }; "
        "start_server c 8000 vllm/vllm-openai:nightly org/model '-tp 8' '' docker; "
        "kill $VLLM_LOGS_PID 2>/dev/null || true",
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.split("\0")
    assert "--rm" not in args
    assert args[args.index("--stop-signal") + 1] == "SIGINT"
    assert args[args.index("--entrypoint") + 1] == "bash"
    script = args.index("-c") + 1
    assert args[script].startswith("#!/usr/bin/env bash")
    assert args[script + 1:script + 7] == [
        "nsys_serve.sh", "/tmp/perf-eval-nsys/wl", "org/model", "--port", "8000", "-tp",
    ]


def test_nsys_serve_wraps_vllm_serve_in_capture_range():
    with __import__("tempfile").TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "nsys")
        with open(fake, "w") as f:
            f.write('#!/usr/bin/env bash\n[[ $1 == --version ]] && exit 0\n'
                    'printf "%s\\n" "$@"\n')
        os.chmod(fake, 0o755)
        result = subprocess.run(
            ["bash", NSYS_SERVE_SH, os.path.join(tmp, "out", "rep"),
             "org/model", "--port", "1"],
            env={**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}"},
            capture_output=True, text=True, timeout=10,
        )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == "profile"
    assert "--capture-range=cudaProfilerApi" in args
    assert "--capture-range-end=stop" in args
    assert f"--output={os.path.join(tmp, 'out', 'rep')}" in args
    assert args[-5:] == ["vllm", "serve", "org/model", "--port", "1"]


def test_nsys_profile_runs_one_wave_with_profile_flag():
    row = "\t".join(["8k-conc-64", "openai", "random", "8192", "1024", "256", "64",
                     "3", "-", "-", "eyJudW0td2FybXVwcyI6NjR9"])
    result = run_bash(
        "set -euo pipefail; NSYS_REPORT=/r; "
        "run_vllm_bench() { printf '%s\\n' \"$@\"; }; "
        "wait_for_nsys_report() { echo waited; }; "
        "collect_nsys_report() { echo collected; }; "
        f"run_nsys_profile c 8000 org/model $'{row}' false results/wl",
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout.splitlines()
    assert out[1:15] == [
        "c", "8000", "org/model", "nsys-8k-conc-64", "openai", "random", "8192",
        "1024", "64", "64", "-", "-", "1",
        "eyJudW0td2FybXVwcyI6NjQsInByb2ZpbGUiOnRydWV9",
    ]
    assert out[-2:] == ["waited", "collected"]


def test_nsys_profile_failure_does_not_fail_workload():
    row = "\t".join(["n", "-", "random", "1", "1", "1", "1", "1", "-", "-", "e30="])
    result = run_bash(
        "set -euo pipefail; NSYS_REPORT=/r; "
        "run_vllm_bench() { false; echo 'continued after failure'; }; "
        "wait_for_nsys_report() { echo waited; }; "
        f"run_nsys_profile c 8000 org/model $'{row}' false results/wl; "
        "echo survived",
    )
    assert result.returncode == 0, result.stderr
    assert "continued after failure" not in result.stdout
    assert "waited" not in result.stdout
    assert "survived" in result.stdout
    assert "profiled bench run failed" in result.stderr


def test_nsys_report_wait_stops_when_nsys_was_unavailable():
    result = run_bash(
        "set -euo pipefail; d=$(mktemp -d); WORKLOAD_SERVER_RUNTIME=native; "
        "NSYS_REPORT=$d/rep; touch $d/rep.unavailable; "
        "! wait_for_nsys_report c",
        timeout=3,
    )
    assert result.returncode == 0, result.stderr
    assert "not available" in result.stderr


def test_nsys_finalize_interrupts_native_server_and_collects():
    result = run_bash(
        "set -euo pipefail; d=$(mktemp -d); WORKLOAD_SERVER_RUNTIME=native; "
        "NSYS_REPORT=$d/rep; NSYS_HOST_DIR=$d; "
        # Background jobs start with SIGINT ignored, which bash can't trap;
        # like nsys, the stand-in installs its own handler.
        "python3 -c \"import signal, sys, time; "
        "signal.signal(signal.SIGINT, lambda *_: (open('$d/rep.nsys-rep', 'w')"
        ".write('x'), sys.exit(0))); time.sleep(30)\" & "
        "VLLM_SERVER_PID=$!; sleep 0.5; "
        "nsys_finalize c; [[ $NSYS_COLLECTED == true ]]",
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "nsys: saved" in result.stdout


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
