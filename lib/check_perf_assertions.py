#!/usr/bin/env python3
"""Check absolute bounds on raw vllm bench result files.

Accept an offline JSON spec or the encoded spec emitted by the workload parser.
Each invocation covers exactly one expanded concurrency configuration. Validate
every raw repetition before using the existing run-level median aggregation;
an invalid run must not disappear inside a passing median.
"""

import argparse
import base64
import json
import math
import re
import sys
from pathlib import Path

from aggregate_perf import aggregate_results


SPEC_FIELDS = {
    "name", "model_id", "backend", "max_concurrency", "num_prompts",
    "repetitions", "assertions",
}
COMPLETED_KEYS = ("completed", "successful", "successful_requests")
FAILED_KEYS = ("failed", "errored", "failed_requests", "num_failed_requests")


def finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def positive_int(value):
    return type(value) is int and value > 0


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != SPEC_FIELDS:
        raise ValueError(f"spec must have exactly these fields: {sorted(SPEC_FIELDS)}")
    for key in ("model_id", "backend"):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("max_concurrency", "num_prompts", "repetitions"):
        if not positive_int(spec[key]):
            raise ValueError(f"{key} must be a positive integer")
    if spec["repetitions"] % 2 == 0:
        raise ValueError("repetitions must be odd, matching the workload parser")
    name = spec["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("name must be a safe expanded benchmark name")
    if not name.endswith(f"-conc-{spec['max_concurrency']}"):
        raise ValueError("name must end with the configured -conc-<max_concurrency>")
    assertions = spec["assertions"]
    if not isinstance(assertions, dict) or not assertions:
        raise ValueError("assertions must be a nonempty metric-to-bounds object")
    reserved = SPEC_FIELDS | set(COMPLETED_KEYS + FAILED_KEYS) | {
        "tokenizer_id", "aggregation_method", "aggregated_repetitions",
        "individual_result_files",
    }
    for metric, bounds in assertions.items():
        if not isinstance(metric, str) or not metric.strip() or metric in reserved:
            raise ValueError(f"invalid metric name: {metric!r}")
        if not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max"}:
            raise ValueError(f"{metric}: supply min and/or max only")
        if not all(finite_number(value) for value in bounds.values()):
            raise ValueError(f"{metric}: bounds must be finite numbers, not strings or booleans")
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError(f"{metric}: min exceeds max")


def validate_result(result, spec, label):
    if not isinstance(result, dict):
        raise ValueError(f"{label}: result must be a JSON object")
    if any(key in result for key in (
        "aggregation_method", "aggregated_repetitions", "individual_result_files",
    )):
        raise ValueError(f"{label}: provide individual raw runs, not an aggregate")
    for key in ("model_id", "backend", "max_concurrency"):
        if type(result.get(key)) is not type(spec[key]) or result[key] != spec[key]:
            raise ValueError(f"{label}: {key} does not match spec")
    if "num_prompts" in result and (
        type(result["num_prompts"]) is not int or result["num_prompts"] != spec["num_prompts"]
    ):
        raise ValueError(f"{label}: num_prompts does not match spec")
    completed = [result[key] for key in COMPLETED_KEYS if key in result]
    if not completed or any(
        type(value) is not int or value != spec["num_prompts"] for value in completed
    ):
        raise ValueError(f"{label}: completion count missing, invalid or incomplete")
    # Some vllm bench versions omit failed counts. Exact successful count is
    # still required; every supplied failure alias must agree with zero.
    for key in FAILED_KEYS:
        if key in result and (type(result[key]) is not int or result[key] != 0):
            raise ValueError(f"{label}: {key} must be zero")
    for metric in spec["assertions"]:
        if metric not in result or not finite_number(result[metric]):
            raise ValueError(f"{label}: {metric} missing or not a finite numeric scalar")


def check_results(spec, results, source_paths):
    validate_spec(spec)
    if len(results) != spec["repetitions"] or len(source_paths) != len(results):
        raise ValueError("raw result count must match repetitions")
    paths = [Path(path) for path in source_paths]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate raw result paths")
    stem = f"bench-{spec['name']}"
    expected_names = (
        {f"{stem}.json"} if len(results) == 1 else
        {f"{stem}-run-{index}.json" for index in range(1, len(results) + 1)}
    )
    if {path.name for path in paths} != expected_names:
        raise ValueError("result filenames do not match this expanded run and repetitions")
    for result, path in zip(results, paths):
        validate_result(result, spec, str(path))
    aggregate = aggregate_results(results, [str(path) for path in paths])
    checks = []
    for metric, bounds in spec["assertions"].items():
        actual = aggregate[metric]
        passed = (
            ("min" not in bounds or actual >= bounds["min"])
            and ("max" not in bounds or actual <= bounds["max"])
        )
        checks.append({"metric": metric, "actual": actual, **bounds, "passed": passed})
    return {
        "run": spec["name"],
        "aggregation": "median" if len(results) > 1 else "single",
        "repetitions": len(results),
        "source_files": [str(path) for path in paths],
        "checks": checks,
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="offline JSON spec")
    source.add_argument("--spec-base64", help="encoded spec from the workload parser")
    parser.add_argument("--results-dir", help="resolve expected raw run filenames here")
    parser.add_argument("results", nargs="*", help="individual raw benchmark JSON files")
    args = parser.parse_args(argv)
    if args.results_dir and args.results:
        parser.error("use --results-dir or positional results, not both")
    try:
        if args.spec:
            spec = read_json(args.spec)
        else:
            spec = json.loads(
                base64.b64decode(args.spec_base64, validate=True),
                object_pairs_hook=unique_object,
            )
        validate_spec(spec)
        sources = args.results
        if args.results_dir:
            stem = f"bench-{spec['name']}"
            filenames = (
                [f"{stem}.json"] if spec["repetitions"] == 1 else
                [f"{stem}-run-{i}.json" for i in range(1, spec["repetitions"] + 1)]
            )
            sources = [str(Path(args.results_dir) / name) for name in filenames]
        report = check_results(spec, [read_json(path) for path in sources], sources)
    except (ValueError, OSError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
