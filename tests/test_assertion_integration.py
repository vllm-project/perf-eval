"""Exercise the real parser, orchestrator and benchmark wrapper with fake I/O."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    return {"name": "case", "input_len": 1024, "output_len": 256, "num_prompts": 10,
            "max_concurrency": 8, "assertions": {"mean_tpot_ms": {"max": 2.5}}, **changes}


FAKE_BENCH = r'''
import json, os, re, sys
from pathlib import Path
if os.environ.get("TEST_COMMAND_FAIL"):
    sys.exit(7)
if os.environ.get("TEST_NO_RESULT"):
    sys.exit(0)
def arg(flag):
    return sys.argv[sys.argv.index(flag) + 1]
path = Path(arg("--result-filename"))
match = re.search(r"-run-(\d+)\.json$", path.name)
index = int(match.group(1)) - 1 if match else 0
runs = json.loads(os.environ["TEST_RESULTS"]).get(arg("--max-concurrency"), [{}])
result = {"completed": int(arg("--num-prompts")), "failed": 0, "mean_tpot_ms": 2.5}
result.update(runs[index])
path.write_text(json.dumps(result))
'''


@pytest.fixture
def harness(tmp_path):
    lib = tmp_path / "lib"
    shutil.copytree(ROOT / "lib", lib, ignore=shutil.ignore_patterns("__pycache__"))
    (lib / "server.sh").write_text(
        'start_server() { echo start >> "$TEST_EVENTS"; }\n'
        'wait_healthy() { :; }\n'
        'stop_server() { echo stop >> "$TEST_EVENTS"; }\n'
    )
    (lib / "run_aiperf.sh").write_text(
        'run_aiperf() { echo aiperf >> "$TEST_EVENTS"; return "${TEST_AIPERF_EXIT:-0}"; }\n'
    )
    (lib / "run_lm_eval.sh").write_text(
        'run_lm_eval() { echo lm_eval >> "$TEST_EVENTS"; }\n'
    )
    (lib / "run_bfcl.py").write_text(
        'import os\nwith open(os.environ["TEST_EVENTS"], "a") as f: f.write("bfcl\\n")\n'
    )
    upload = (
        'import os, sys\nfrom pathlib import Path\n'
        'event = ("upload:" + Path(sys.argv[sys.argv.index("--raw-result") + 1]).name\n'
        '         if "--raw-result" in sys.argv else "quality-upload")\n'
        'with open(os.environ["TEST_EVENTS"], "a") as f: f.write(event + "\\n")\n'
        'sys.exit(int(os.environ.get("TEST_UPLOAD_EXIT", "0")))\n'
    )
    for name in ("ingest_perf.py", "ingest.py"):
        (lib / name).write_text(upload)
    (lib / "gpu_profiles.yaml").write_text("CPU_TEST:\n  server_runtime: native\n")
    registry = tmp_path / "lm_eval"
    registry.mkdir()
    (registry / "__init__.py").write_text("")
    (registry / "tasks.py").write_text('class TaskManager:\n    all_tasks = ["quality"]\n')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3").symlink_to(sys.executable)
    (fake_bin / "vllm").write_text(f"#!{sys.executable}\n" + FAKE_BENCH)
    (fake_bin / "vllm").chmod(0o755)

    (tmp_path / "workloads").mkdir()

    def run(configs, rows=None, bench_only="0", **overrides):
        workload = {
            "name": "cpu-integration", "gpu": "CPU_TEST",
            "vllm": {"model": "synthetic-model"},
            "vllm_bench": {"configs": configs},
            "aiperf": {"configs": [{"name": "profile", "args": {}}]},
            "lm_eval": {"tasks": [{"name": "quality", "num_fewshot": 0}]},
            "bfcl": {"test_categories": ["simple_python"], "max_test_cases": 1},
        }
        path = tmp_path / "workloads/case.yaml"
        path.write_text(yaml.safe_dump(workload))
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("VLLM_", "WORKLOAD_", "PERF_EVAL_", "INGEST_", "TEST_"))}
        env.update({
            "PATH": f"{fake_bin}:{os.environ['PATH']}", "PYTHONPATH": str(tmp_path),
            "PERF_EVAL_SERVER_PORT": "1", "BENCH_ONLY": bench_only,
            "TEST_EVENTS": str(tmp_path / "events"), "TEST_RESULTS": json.dumps(rows or {}),
            **overrides,
        })
        proc = subprocess.run(["bash", str(lib / "run.sh"), str(path)], cwd=tmp_path,
                              env=env, capture_output=True, text=True, timeout=20)
        events = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
        return proc, tmp_path / "results/cpu-integration", events

    return run, tmp_path


@pytest.mark.parametrize("bench_only", ["0", "1"])
@pytest.mark.parametrize("value, enabled, expected", [
    (2.5, True, 0), (4, True, 1), (float("nan"), True, 2), (99, False, 0),
])
def test_optional_gate_finishes_later_evaluations(harness, bench_only, value, enabled, expected):
    run, _ = harness
    cfg = config()
    if not enabled:
        del cfg["assertions"]
    proc, results, events = run([cfg], {"8": [{"mean_tpot_ms": value}]}, bench_only)
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert [e for e in events if "upload" not in e] == (
        ["start", "aiperf", "stop"] if bench_only == "1"
        else ["start", "aiperf", "lm_eval", "bfcl", "stop"]
    )
    assert events.count("quality-upload") == (0 if bench_only == "1" else 2)
    assert ("upload:bench-case-conc-8.json" in events) == (expected != 2)
    report = results / "assertions-case-conc-8.json"
    assert report.exists() == enabled
    if enabled:
        assert json.loads(report.read_text())["status"] == {0: "pass", 1: "fail", 2: "error"}[expected]
    assert (results / "bench-case-conc-8.json").exists()


def test_sweep_preserves_failures_and_later_unchecked_upload(harness):
    run, _ = harness
    unchecked = config(name="unchecked", max_concurrency=32)
    del unchecked["assertions"]
    proc, results, events = run(
        [config(max_concurrency=[1, 8, 16]), unchecked],
        {"1": [{"mean_tpot_ms": 4}], "8": [{"mean_tpot_ms": float("nan")}]},
        TEST_UPLOAD_EXIT="9",
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert [json.loads((results / f"assertions-case-conc-{c}.json").read_text())["status"]
            for c in (1, 8, 16)] == ["fail", "error", "pass"]
    assert [e for e in events if e.startswith("upload:")] == [
        "upload:bench-case-conc-1.json", "upload:bench-case-conc-16.json",
        "upload:bench-unchecked-conc-32.json",
    ]
    assert not (results / "assertions-unchecked-conc-32.json").exists()
    assert events[-1] == "stop"


@pytest.mark.parametrize("middle, expected", [(9, 0), (float("nan"), 2)])
def test_raw_metrics_checked_but_existing_median_used(harness, middle, expected):
    run, _ = harness
    proc, results, events = run(
        [config(repetitions=3)], {"8": [{"mean_tpot_ms": v} for v in (2, middle, 2.5)]},
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert len(list(results.glob("bench-*.json"))) == 4
    report = json.loads((results / "assertions-case-conc-8.json").read_text())
    if expected == 0:
        summary = json.loads((results / "bench-case-conc-8.json").read_text())
        assert report["checks"][0]["actual"] == summary["mean_tpot_ms"] == 2.5
    else:
        assert report["status"] == "error"
        assert not any(e.startswith("upload:") for e in events)


@pytest.mark.parametrize("overrides, rows, expected", [
    ({"TEST_COMMAND_FAIL": "1"}, {}, 7),
    ({}, {"8": [{"completed": 9, "failed": 1}]}, 1),
    ({"TEST_NO_RESULT": "1"}, {}, 1),
])
def test_existing_benchmark_validation_remains_fatal(harness, overrides, rows, expected):
    run, _ = harness
    proc, results, events = run([config()], rows, **overrides)
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert events == ["start", "stop"]
    assert not list(results.glob("assertions-*.json"))


def test_invalid_bounds_rejected_before_start(harness):
    run, _ = harness
    proc, _, events = run([config(assertions=None)])
    assert proc.returncode != 0
    assert "assertions must be" in proc.stderr
    assert events == []


@pytest.mark.parametrize("failure, expected", [
    ("raise SystemExit(3)", 3),
    ("raise RuntimeError('simulated checker crash')", 2),
    ("raise SystemExit(1)", 2),
    ("print('{broken'); raise SystemExit(1)", 2),
    ('print(\'{"status":"fail","checks":[]}\'); raise SystemExit(1)', 2),
])
def test_checker_error_skips_upload_without_suppressing_quality(harness, failure, expected):
    run, root = harness
    checker = root / "lib/check_perf_assertions.py"
    checker.write_text(f'if __name__ == "__main__": {failure}\n' + checker.read_text())
    proc, results, events = run([config()])
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert not any(e.startswith("upload:") for e in events)
    assert events.count("quality-upload") == 2 and events[-1] == "stop"
    assert (results / "bench-case-conc-8.json").exists()


def test_later_command_failure_is_not_swallowed(harness):
    run, _ = harness
    proc, _, events = run([config()], {"8": [{"mean_tpot_ms": 4}]}, TEST_AIPERF_EXIT="9")
    assert proc.returncode == 9, proc.stdout + proc.stderr
    assert events[-2:] == ["aiperf", "stop"]
    assert "lm_eval" not in events
