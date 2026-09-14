"""CPU integration: real parser/orchestrator/bench wrapper, fake model and I/O.

All generated files and subprocesses stay in pytest temporary directories.
No model server, Docker, GPU allocation, or external ingestion is used.
"""

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
import parse_workload


def config(**changes):
    value = {
        "name": "1k-in-256-out", "backend": "openai", "dataset": "random",
        "input_len": 1024, "output_len": 256, "num_prompts": 10,
        "max_concurrency": 8,
        "assertions": {"mean_tpot_ms": {"max": 2.5}, "output_throughput": {"min": 100}},
    }
    value.update(changes)
    return value


def test_encoded_specs_are_bound_to_expanded_runs():
    rows = parse_workload.bench_tsv(
        [config(max_concurrency=[1, 8], num_prompts=[5, 10], repetitions=3)],
        "test.yaml", "synthetic-model",
    ).splitlines()
    decoded = [json.loads(base64.b64decode(row.split("\t")[-1])) for row in rows]
    assert [value["name"] for value in decoded] == [
        "1k-in-256-out-conc-1", "1k-in-256-out-conc-8",
    ]
    assert [value["max_concurrency"] for value in decoded] == [1, 8]
    assert [value["num_prompts"] for value in decoded] == [5, 10]
    assert all(value["model_id"] == "synthetic-model" for value in decoded)
    assert all(value["repetitions"] == 3 for value in decoded)


def test_unconfigured_row_only_appends_disabled_sentinel():
    value = config()
    del value["assertions"]
    fields = parse_workload.bench_tsv([value], "test.yaml").split("\t")
    assert len(fields) == 12
    assert fields[-1] == "-"
    assert fields[:8] == [
        "1k-in-256-out-conc-8", "openai", "random", "1024", "256", "10", "8", "1",
    ]
    assert json.loads(base64.b64decode(fields[10])) == {}


FAKE_BENCH = r'''
import json
import os
from pathlib import Path
import re
import sys

args = sys.argv[1:]
assert args[:2] == ["bench", "serve"], args
with open(os.environ["TEST_INVOCATIONS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")
mode = os.environ.get("TEST_BENCH_MODE", "")
if mode == "command_failure":
    sys.exit(7)

def arg(name):
    return args[args.index(name) + 1]

path = Path(arg("--result-filename"))
if mode == "no_output":
    sys.exit(0)
conc = int(arg("--max-concurrency"))
prompts = int(arg("--num-prompts"))
match = re.search(r"-run-(\d+)\.json$", path.name)
index = int(match.group(1)) - 1 if match else 0
timings = json.loads(os.environ.get("TEST_TIMINGS", "{}")).get(str(conc), [2.5])
value = {
    "backend": arg("--backend"), "model_id": arg("--model"),
    "max_concurrency": conc, "num_prompts": prompts,
    "completed": prompts, "failed": 0,
    "mean_tpot_ms": timings[index % len(timings)], "output_throughput": 100,
}
if mode == "mixed_status" and conc == 8:
    value["mean_tpot_ms"] = float("nan")
elif mode == "nan":
    value["mean_tpot_ms"] = float("nan")
elif mode == "missing_metric":
    del value["mean_tpot_ms"]
elif mode == "wrong_concurrency":
    value["max_concurrency"] = conc + 1
elif mode == "failed_request":
    value["completed"] -= 1
    value["failed"] = 1
elif mode == "nan_one_run" and index == 1:
    value["mean_tpot_ms"] = float("nan")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(value), encoding="utf-8")
'''


@pytest.fixture
def harness(tmp_path):
    lib = tmp_path / "lib"
    shutil.copytree(ROOT / "lib", lib, ignore=shutil.ignore_patterns("__pycache__"))
    (lib / "server.sh").write_text(
        'start_server() { printf "start\\n" >> "$TEST_LIFECYCLE"; }\n'
        'wait_healthy() { :; }\n'
        'stop_server() { printf "stop\\n" >> "$TEST_LIFECYCLE"; }\n'
        'pick_server_port() { printf "1\\n"; }\n'
    )
    (lib / "ingest_perf.py").write_text(
        'import json, os, sys\n'
        'with open(os.environ["TEST_UPLOADS"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'sys.exit(int(os.environ.get("TEST_UPLOAD_EXIT", "0")))\n'
    )
    # Model/evaluation work and all upload clients are local fakes.
    (lib / "run_aiperf.sh").write_text(
        'run_aiperf() { printf "aiperf\\n" >> "$TEST_TASK_ORDER"; '
        'return "$TEST_AIPERF_EXIT"; }\n'
    )
    (lib / "run_lm_eval.sh").write_text(
        'run_lm_eval() { printf "lm_eval\\n" >> "$TEST_TASK_ORDER"; '
        'return "$TEST_LM_EVAL_EXIT"; }\n'
    )
    (lib / "run_bfcl.py").write_text(
        'import os, sys\n'
        'with open(os.environ["TEST_TASK_ORDER"], "a") as stream:\n'
        '    stream.write("bfcl\\n")\n'
        'sys.exit(int(os.environ["TEST_BFCL_EXIT"]))\n'
    )
    (lib / "ingest.py").write_text(
        'import json, os, sys\n'
        'with open(os.environ["TEST_QUALITY_UPLOADS"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
    )
    module_root = tmp_path / "modules"
    registry = module_root / "lm_eval"
    registry.mkdir(parents=True)
    (registry / "__init__.py").write_text("")
    (registry / "tasks.py").write_text(
        'class TaskManager:\n'
        '    all_tasks = ["synthetic-quality"]\n'
    )
    (lib / "gpu_profiles.yaml").write_text(yaml.safe_dump({
        "CPU_TEST": {"queue": "not-used", "server_runtime": "native"},
    }))
    (tmp_path / "workloads").mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable = fake_bin / "vllm"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_BENCH)
    executable.chmod(0o755)
    # Ensure shell helpers use this test environment's interpreter.
    (fake_bin / "python3").symlink_to(sys.executable)

    def run(configs, *, extra_sections=None, **overrides):
        workload = {
            "name": "cpu-integration", "gpu": "CPU_TEST",
            "vllm": {"model": "synthetic-model", "serve_args": ""},
            "vllm_bench": {"configs": configs},
        }
        workload.update(extra_sections or {})
        path = tmp_path / "workloads" / "case.yaml"
        path.write_text(yaml.safe_dump(workload))
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("VLLM_", "WORKLOAD_", "PERF_EVAL_", "INGEST_", "TEST_"))}
        env.update({
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "BENCH_ONLY": "1", "PERF_EVAL_SERVER_PORT": "1",
            "TEST_LIFECYCLE": str(tmp_path / "lifecycle"),
            "TEST_INVOCATIONS": str(tmp_path / "invocations"),
            "TEST_UPLOADS": str(tmp_path / "uploads"),
            "TEST_TASK_ORDER": str(tmp_path / "task-order"),
            "TEST_QUALITY_UPLOADS": str(tmp_path / "quality-uploads"),
            "TEST_AIPERF_EXIT": "0", "TEST_LM_EVAL_EXIT": "0", "TEST_BFCL_EXIT": "0",
            "PYTHONPATH": str(module_root),
            **overrides,
        })
        proc = subprocess.run(
            ["bash", str(lib / "run.sh"), str(path)], cwd=tmp_path,
            env=env, capture_output=True, text=True, timeout=20,
        )
        return proc, tmp_path / "results" / "cpu-integration"

    return run, tmp_path


def test_yaml_to_success_and_report(harness):
    run, root = harness
    proc, results = run([config()])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((results / "assertions-1k-in-256-out-conc-8.json").read_text())
    assert report["status"] == "pass"
    assert report["checks"][0]["actual"] == 2.5
    assert (results / "bench-1k-in-256-out-conc-8.json").exists()
    assert (root / "lifecycle").read_text().splitlines() == ["start", "stop"]
    assert len((root / "uploads").read_text().splitlines()) == 1


def test_unconfigured_run_keeps_previous_behavior(harness):
    run, root = harness
    value = config()
    del value["assertions"]
    proc, results = run([value], TEST_TIMINGS='{"8": [99]}')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not list(results.glob("assertions-*.json"))
    assert len((root / "uploads").read_text().splitlines()) == 1


def test_bound_failure_does_not_stop_sweep_or_drop_results(harness):
    run, root = harness
    proc, results = run(
        [config(max_concurrency=[1, 8, 16], num_prompts=[5, 10, 20])],
        TEST_TIMINGS='{"1": [2], "8": [4], "16": [2.5]}',
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    statuses = {}
    for conc in (1, 8, 16):
        report = json.loads((results / f"assertions-1k-in-256-out-conc-{conc}.json").read_text())
        statuses[conc] = report["status"]
        assert report["run"].endswith(f"-conc-{conc}")
    assert statuses == {1: "pass", 8: "fail", 16: "pass"}
    assert len(list(results.glob("bench-*.json"))) == 3
    assert len((root / "uploads").read_text().splitlines()) == 3
    assert (root / "lifecycle").read_text().endswith("stop\n")


def test_separate_configs_have_independent_thresholds(harness):
    run, _ = harness
    proc, results = run([
        config(name="strict", assertions={"mean_tpot_ms": {"max": 2}}),
        config(name="relaxed", assertions={"mean_tpot_ms": {"max": 3}}),
    ])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads((results / "assertions-strict-conc-8.json").read_text())["status"] == "fail"
    assert json.loads((results / "assertions-relaxed-conc-8.json").read_text())["status"] == "pass"


def test_repetitions_keep_raw_runs_and_check_median(harness):
    run, _ = harness
    proc, results = run([config(repetitions=3)], TEST_TIMINGS='{"8": [2, 9, 2.5]}')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(list(results.glob("bench-*-run-*.json"))) == 3
    aggregate = json.loads((results / "bench-1k-in-256-out-conc-8.json").read_text())
    assert aggregate["mean_tpot_ms"] == 2.5
    report = json.loads((results / "assertions-1k-in-256-out-conc-8.json").read_text())
    assert report["aggregation"] == "median"
    assert report["checks"][0]["actual"] == 2.5


@pytest.mark.parametrize("mode, repetitions", [
    ("nan", 1), ("nan", 3), ("missing_metric", 1), ("missing_metric", 3),
    ("wrong_concurrency", 1), ("wrong_concurrency", 3), ("nan_one_run", 3),
])
def test_invalid_raw_data_cannot_pass(harness, mode, repetitions):
    run, root = harness
    proc, results = run([config(repetitions=repetitions)], TEST_BENCH_MODE=mode)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    report = json.loads((results / "assertions-1k-in-256-out-conc-8.json").read_text())
    assert report["status"] == "error"
    expected_files = repetitions + (1 if repetitions > 1 else 0)
    assert len(list(results.glob("bench-*.json"))) == expected_files
    assert not (root / "uploads").exists()
    assert "Skipping dashboard upload" in proc.stderr
    assert (root / "lifecycle").read_text().endswith("stop\n")


@pytest.mark.parametrize("mode", ["command_failure", "failed_request", "no_output"])
def test_benchmark_failure_is_not_swallowed(harness, mode):
    run, root = harness
    proc, results = run([config()], TEST_BENCH_MODE=mode)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    if mode == "command_failure":
        assert proc.returncode == 7
    assert not list(results.glob("assertions-*.json"))
    assert not (root / "uploads").exists()
    assert (root / "lifecycle").read_text().endswith("stop\n")


@pytest.mark.parametrize("value", [
    None, {}, [], {"mean_tpot_ms": {}}, {"mean_tpot_ms": {"max": True}},
    {"mean_tpot_ms": {"max": float("nan")}},
    {"mean_tpot_ms": {"max": float("inf")}},
    {"mean_tpot_ms": {"min": 4, "max": 3}},
    {"mean_tpot_ms": {"maximum": 3}},
])
def test_bad_assertions_fail_before_server_start(harness, value):
    run, root = harness
    proc, _ = run([config(assertions=value)])
    assert proc.returncode != 0
    assert not (root / "lifecycle").exists()
    assert not (root / "invocations").exists()


def test_assertions_require_explicit_backend(harness):
    run, root = harness
    value = config()
    del value["backend"]
    proc, _ = run([value])
    assert proc.returncode != 0
    assert not (root / "lifecycle").exists()


def test_upload_failure_cannot_hide_gate_failure(harness):
    run, _ = harness
    proc, results = run([config()], TEST_TIMINGS='{"8": [4]}', TEST_UPLOAD_EXIT="9")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert json.loads((results / "assertions-1k-in-256-out-conc-8.json").read_text())["status"] == "fail"


def test_malformed_encoded_spec_is_error(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "lib/check_perf_assertions.py"),
         "--spec-base64", "not-base64!", "--results-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["status"] == "error"


def test_invalid_input_dominates_bound_failure_and_later_success(harness):
    run, root = harness
    proc, results = run(
        [config(max_concurrency=[1, 8, 16])],
        TEST_TIMINGS='{"1": [4], "8": [2.5], "16": [2]}',
        TEST_BENCH_MODE="mixed_status",
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    statuses = [json.loads((results / f"assertions-1k-in-256-out-conc-{conc}.json").read_text())["status"]
                for conc in (1, 8, 16)]
    assert statuses == ["fail", "error", "pass"]
    uploads = [json.loads(line) for line in (root / "uploads").read_text().splitlines()]
    uploaded_files = [Path(args[args.index("--raw-result") + 1]).name for args in uploads]
    assert uploaded_files == [
        "bench-1k-in-256-out-conc-1.json", "bench-1k-in-256-out-conc-16.json",
    ]
    assert len(list(results.glob("bench-*.json"))) == 3


def test_upload_failure_remains_nonfatal_without_assertions(harness):
    run, _ = harness
    value = config()
    del value["assertions"]
    proc, _ = run([value], TEST_UPLOAD_EXIT="9")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_all_shipped_benchmark_configs_parse_without_assertions():
    checked = 0
    for path in sorted((ROOT / "workloads").glob("*.yaml")):
        workload = yaml.safe_load(path.read_text())
        configs = (workload.get("vllm_bench") or {}).get("configs") or []
        assert all("assertions" not in value for value in configs), path
        rows = parse_workload.bench_tsv(
            configs, str(path), (workload.get("vllm") or {}).get("model"),
        ).splitlines()
        for row in rows:
            fields = row.split("\t")
            assert len(fields) == 12 and fields[-1] == "-", (path, fields)
            assert isinstance(json.loads(base64.b64decode(fields[10])), dict)
            checked += 1
    assert checked > 0


def test_invalid_result_does_not_suppress_later_unconfigured_upload(harness):
    run, root = harness
    unchecked = config(name="unchecked", max_concurrency=16)
    del unchecked["assertions"]
    proc, results = run(
        [config(name="invalid", max_concurrency=8), unchecked],
        TEST_BENCH_MODE="mixed_status",
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    report = json.loads((results / "assertions-invalid-conc-8.json").read_text())
    assert report["status"] == "error"
    assert not (results / "assertions-unchecked-conc-16.json").exists()
    uploads = [json.loads(line) for line in (root / "uploads").read_text().splitlines()]
    assert len(uploads) == 1
    assert Path(uploads[0][uploads[0].index("--raw-result") + 1]).name == (
        "bench-unchecked-conc-16.json"
    )
    assert len(list(results.glob("bench-*.json"))) == 2


@pytest.mark.parametrize("exit_code", [3, 7, 137])
def test_checker_failure_skips_upload_and_preserves_results(harness, exit_code):
    run, root = harness
    checker = root / "lib" / "check_perf_assertions.py"
    # Keep imports/parse-time validation intact; fail only the CLI invocation.
    checker.write_text(
        'if __name__ == "__main__":\n'
        '    import sys\n'
        '    print("simulated checker failure", file=sys.stderr)\n'
        f'    sys.exit({exit_code})\n'
        + checker.read_text()
    )
    proc, results = run([config()])
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    assert not (root / "uploads").exists()
    assert (results / "bench-1k-in-256-out-conc-8.json").exists()
    assert (results / "assertions-1k-in-256-out-conc-8.json").exists()
    assert "simulated checker failure" in proc.stderr
    assert "Skipping dashboard upload" in proc.stderr
    assert (root / "lifecycle").read_text().endswith("stop\n")


def later_evaluations():
    return {
        "aiperf": {"configs": [{"name": "synthetic-profile", "args": {}}]},
        "lm_eval": {"tasks": [{"name": "synthetic-quality", "num_fewshot": 0}]},
        "bfcl": {"test_categories": ["simple_python"], "max_test_cases": 1},
    }


@pytest.mark.parametrize("mode, timing, expected", [
    ("", 2.5, 0), ("", 4, 1), ("nan", 2.5, 2),
])
def test_assertion_status_is_reported_after_all_evaluations(harness, mode, timing, expected):
    run, root = harness
    proc, results = run(
        [config()], extra_sections=later_evaluations(), BENCH_ONLY="0",
        TEST_BENCH_MODE=mode, TEST_TIMINGS=json.dumps({"8": [timing]}),
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert (root / "task-order").read_text().splitlines() == ["aiperf", "lm_eval", "bfcl"]
    assert len((root / "quality-uploads").read_text().splitlines()) == 2
    assert (root / "uploads").exists() == (expected != 2)
    report = json.loads((results / "assertions-1k-in-256-out-conc-8.json").read_text())
    assert report["status"] == {0: "pass", 1: "fail", 2: "error"}[expected]
    assert (root / "lifecycle").read_text().splitlines() == ["start", "stop"]


@pytest.mark.parametrize("mode, expected", [("", 1), ("nan", 2)])
def test_bench_only_runs_aiperf_without_losing_assertion_failure(harness, mode, expected):
    run, root = harness
    proc, _ = run(
        [config()], extra_sections=later_evaluations(), BENCH_ONLY="1",
        TEST_BENCH_MODE=mode, TEST_TIMINGS='{"8": [4]}',
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert (root / "task-order").read_text().splitlines() == ["aiperf"]
    assert not (root / "quality-uploads").exists()
    assert (root / "lifecycle").read_text().endswith("stop\n")


@pytest.mark.parametrize("variable, exit_code, expected_order", [
    ("TEST_AIPERF_EXIT", 9, ["aiperf"]),
    ("TEST_LM_EVAL_EXIT", 10, ["aiperf", "lm_eval"]),
    ("TEST_BFCL_EXIT", 11, ["aiperf", "lm_eval", "bfcl"]),
])
def test_later_command_failure_remains_fatal(harness, variable, exit_code, expected_order):
    run, root = harness
    proc, _ = run(
        [config()], extra_sections=later_evaluations(), BENCH_ONLY="0",
        TEST_TIMINGS='{"8": [4]}', **{variable: str(exit_code)},
    )
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    assert (root / "task-order").read_text().splitlines() == expected_order
    assert (root / "lifecycle").read_text().endswith("stop\n")


@pytest.mark.parametrize("mode, expected", [("command_failure", 7), ("failed_request", 1)])
def test_benchmark_execution_failure_still_stops_later_evaluations(harness, mode, expected):
    run, root = harness
    proc, _ = run(
        [config()], extra_sections=later_evaluations(), BENCH_ONLY="0",
        TEST_BENCH_MODE=mode,
    )
    assert proc.returncode == expected, proc.stdout + proc.stderr
    assert not (root / "task-order").exists()
    assert not (root / "quality-uploads").exists()
    assert (root / "lifecycle").read_text().endswith("stop\n")
