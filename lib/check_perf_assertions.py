#!/usr/bin/env python3
"""Check metric bounds on results produced by run_vllm_bench."""

import argparse
import base64
import json
import math
from pathlib import Path


def finite_number(value):
    return type(value) is int or (type(value) is float and math.isfinite(value))


def validate_assertions(assertions):
    if not isinstance(assertions, dict) or not assertions:
        raise ValueError("assertions must be a nonempty metric-to-bounds map")
    for metric, bounds in assertions.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric names must be nonempty strings")
        if not isinstance(bounds, dict) or not bounds or set(bounds) - {"min", "max"}:
            raise ValueError(f"{metric}: supply min and/or max only")
        if not all(finite_number(value) for value in bounds.values()):
            raise ValueError(f"{metric}: bounds must be finite numbers")
        if "min" in bounds and "max" in bounds and bounds["min"] > bounds["max"]:
            raise ValueError(f"{metric}: min exceeds max")


def check_metrics(assertions, summary, raw_results=()):
    # The benchmark helper owns request validation and median aggregation.
    # Inspect raw metrics too: a NaN must not hide in an otherwise valid median.
    for result in [*raw_results, summary]:
        if not isinstance(result, dict):
            raise ValueError("benchmark result must be a JSON object")
        for metric in assertions:
            if not finite_number(result.get(metric)):
                raise ValueError(f"{metric}: missing or non-finite numeric metric")
    checks = []
    for metric, bounds in assertions.items():
        actual = summary[metric]
        passed = (
            ("min" not in bounds or actual >= bounds["min"])
            and ("max" not in bounds or actual <= bounds["max"])
        )
        checks.append({"metric": metric, "actual": actual, **bounds, "passed": passed})
    return {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "checks": checks,
    }


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def is_failure_report(path, result, encoded_assertions):
    """Accept exit 1 only with a complete, consistent bound-failure report."""
    try:
        report = read_json(path)
        assertions = json.loads(base64.b64decode(encoded_assertions, validate=True))
        validate_assertions(assertions)
        actuals = {check["metric"]: check["actual"] for check in report["checks"]}
        expected = {"result": result, **check_metrics(assertions, actuals)}
        return expected["status"] == "fail" and report == expected
    except (KeyError, TypeError, ValueError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assertions-base64", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    try:
        assertions = json.loads(base64.b64decode(args.assertions_base64, validate=True))
        validate_assertions(assertions)
        raw_paths = [
            args.result.with_name(f"{args.result.stem}-run-{i}.json")
            for i in range(1, args.repetitions + 1)
        ] if args.repetitions > 1 else []
        report = check_metrics(
            assertions, read_json(args.result), [read_json(path) for path in raw_paths],
        )
    except (ValueError, OSError, OverflowError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps({"result": str(args.result), **report}, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
