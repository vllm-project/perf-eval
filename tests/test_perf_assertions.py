"""Unit coverage for metric assertions; no model or external I/O."""

import base64
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from check_perf_assertions import check_metrics, is_failure_report, main, validate_assertions
from parse_workload import bench_tsv

BOUNDS = {"mean_tpot_ms": {"min": 2, "max": 3}, "output_throughput": {"min": 100}}


@pytest.mark.parametrize("tpot, throughput, expected", [
    (2, 100, "pass"), (3, 101, "pass"), (1.99, 100, "fail"),
    (3.01, 100, "fail"), (2.5, 99, "fail"),
])
def test_inclusive_bounds(tpot, throughput, expected):
    report = check_metrics(BOUNDS, {"mean_tpot_ms": tpot, "output_throughput": throughput})
    assert report["status"] == expected
    assert report["checks"][0]["actual"] == tpot


@pytest.mark.parametrize("assertions", [
    None, {}, [], {"": {"max": 1}}, {1: {"max": 1}}, {"tpot": {}},
    {"tpot": {"maximum": 2}}, {"tpot": {"min": 3, "max": 2}},
    *({"tpot": {"max": value}} for value in [True, "2", None, float("nan"), float("inf")]),
])
def test_invalid_bounds(assertions):
    with pytest.raises(ValueError):
        validate_assertions(assertions)


@pytest.mark.parametrize("result", [
    [], {}, {"tpot": True}, {"tpot": "2"}, {"tpot": None},
    {"tpot": float("nan")}, {"tpot": float("inf")}, {"tpot": -float("inf")},
])
def test_invalid_metrics_cannot_hide_in_a_passing_summary(result):
    with pytest.raises(ValueError):
        check_metrics({"tpot": {"max": 3}}, {"tpot": 2}, [result])
    with pytest.raises(ValueError):
        check_metrics({"tpot": {"max": 3}}, result)


def test_parser_only_appends_bounds_to_expanded_rows():
    config = {"name": "case", "input_len": 10, "output_len": 10,
              "num_prompts": 10, "max_concurrency": [1, 8]}
    disabled = bench_tsv([config], "case.yaml").splitlines()
    enabled = bench_tsv([{**config, "assertions": BOUNDS}], "case.yaml").splitlines()
    assert len(enabled) == 2
    for before, after in zip(disabled, enabled):
        original, fields = before.split("\t"), after.split("\t")
        assert fields[:-1] == original[:-1]
        assert original[-1] == "-"
        assert json.loads(base64.b64decode(fields[-1])) == BOUNDS
    with pytest.raises(SystemExit):
        bench_tsv([{**config, "assertions": None}], "case.yaml")


@pytest.mark.parametrize("raw", [None, "{broken", "[]", '{"tpot":NaN}'])
def test_cli_reports_invalid_data(tmp_path, monkeypatch, capsys, raw):
    path = tmp_path / "bench-case.json"
    if raw is not None:
        path.write_text(raw)
    encoded = base64.b64encode(b'{"tpot":{"max":3}}').decode()
    monkeypatch.setattr(sys, "argv", ["checker", "--assertions-base64", encoded,
                                    "--result", str(path)])
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


@pytest.mark.parametrize("change", ["valid", "path", "status", "checks", "flag", "bound", "actual"])
def test_failure_report_must_match_current_bounds_and_result(tmp_path, change):
    assertions = {"tpot": {"max": 3}}
    encoded = base64.b64encode(json.dumps(assertions).encode()).decode()
    report = {"result": "bench.json", **check_metrics(assertions, {"tpot": 4})}
    if change == "path":
        report["result"] = "other.json"
    elif change == "status":
        report["status"] = "pass"
    elif change == "checks":
        report["checks"] = []
    elif change == "flag":
        report["checks"][0]["passed"] = True
    elif change == "bound":
        report["checks"][0]["max"] = 5
    elif change == "actual":
        report["checks"][0]["actual"] = 2
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert is_failure_report(path, "bench.json", encoded) == (change == "valid")
