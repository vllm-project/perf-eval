#!/usr/bin/env python3
"""Stdlib + PyYAML regression tests for repeated serving benchmarks."""

import base64
import copy
import importlib.util
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, relative_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate_perf = load_module("aggregate_perf", "lib/aggregate_perf.py")
parse_workload = load_module("parse_workload", "lib/parse_workload.py")


def benchmark_config(**overrides):
    config = {
        "name": "test",
        "backend": "openai",
        "dataset": "random",
        "input_len": 8192,
        "output_len": 1024,
        "num_prompts": 512,
        "max_concurrency": 128,
    }
    config.update(overrides)
    return config


def raw_result(**overrides):
    result = {
        "backend": "openai",
        "model_id": "model",
        "tokenizer_id": "tokenizer",
        "num_prompts": 512,
        "max_concurrency": 128,
        "completed": 512,
        "failed": 0,
        "mean_ttft_ms": 100.0,
        "p99_ttft_ms": 150.0,
        "request_rate": "inf",
    }
    result.update(overrides)
    return result


def test_parser_defaults_to_one_repetition():
    fields = parse_workload.bench_tsv(
        [benchmark_config()], "workload.yaml"
    ).split("\t")
    assert fields[7] == "1", fields


def test_parser_emits_repetitions_and_warmups():
    fields = parse_workload.bench_tsv(
        [
            benchmark_config(
                repetitions=3,
                args={"num_warmups": 128},
            )
        ],
        "workload.yaml",
    ).split("\t")
    assert fields[7] == "3", fields
    args = json.loads(base64.b64decode(fields[10]))
    assert args == {"num-warmups": 128}, args


def test_parser_rejects_non_positive_or_even_repetitions():
    for repetitions in (0, 2, -1, True):
        try:
            parse_workload.bench_tsv(
                [benchmark_config(repetitions=repetitions)], "workload.yaml"
            )
        except SystemExit as exc:
            assert "positive odd integer" in str(exc), exc
        else:
            raise AssertionError(f"accepted repetitions={repetitions!r}")


def test_aggregate_uses_run_level_median_and_retains_metadata():
    aggregate = aggregate_perf.aggregate_results(
        [
            raw_result(mean_ttft_ms=130.0, p99_ttft_ms=210.0),
            raw_result(mean_ttft_ms=100.0, p99_ttft_ms=150.0),
            raw_result(mean_ttft_ms=110.0, p99_ttft_ms=180.0),
        ],
        ["run-1.json", "run-2.json", "run-3.json"],
    )
    assert aggregate["mean_ttft_ms"] == 110.0, aggregate
    assert aggregate["p99_ttft_ms"] == 180.0, aggregate
    assert aggregate["request_rate"] == "inf", aggregate
    assert aggregate["aggregation_method"] == "median", aggregate
    assert aggregate["aggregated_repetitions"] == 3, aggregate
    assert aggregate["individual_result_files"] == [
        "run-1.json",
        "run-2.json",
        "run-3.json",
    ], aggregate


def test_aggregate_rejects_mismatched_identity():
    mismatched = copy.deepcopy(raw_result())
    mismatched["model_id"] = "different-model"
    try:
        aggregate_perf.aggregate_results(
            [raw_result(), mismatched, raw_result()],
            ["run-1.json", "run-2.json", "run-3.json"],
        )
    except ValueError as exc:
        assert "model_id" in str(exc), exc
    else:
        raise AssertionError("accepted mismatched model identities")


def test_all_h200_benchmarks_use_saturated_warmups_and_three_runs():
    import yaml

    workload_dir = os.path.join(ROOT, "workloads")
    paths = sorted(
        os.path.join(workload_dir, name)
        for name in os.listdir(workload_dir)
        if name.endswith("_h200.yaml")
    )
    assert paths, "no H200 workloads found"
    for path in paths:
        with open(path) as f:
            workload = yaml.safe_load(f)
        for config in (workload.get("vllm_bench") or {}).get("configs") or []:
            assert config.get("repetitions") == 3, path
            assert (config.get("args") or {}).get("num_warmups") == config.get(
                "max_concurrency"
            ), path


def test_long_h200_workloads_have_explicit_timeouts():
    import yaml

    workload_dir = os.path.join(ROOT, "workloads")
    expected_timeouts = {
        "deepseek_v4_pro_5_h200.yaml": 180,
        "gemma_4_31b_it_h200.yaml": 360,
        "glm_5_1_h200.yaml": 180,
    }
    for name, expected_timeout in expected_timeouts.items():
        path = os.path.join(workload_dir, name)
        with open(path) as f:
            workload = yaml.safe_load(f)
        assert workload.get("timeout_in_minutes") == expected_timeout, path


def main():
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (AssertionError, ValueError) as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
