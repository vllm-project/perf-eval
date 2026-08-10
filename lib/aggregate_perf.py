#!/usr/bin/env python3
"""Median-aggregate repeated ``vllm bench serve`` result files."""

import argparse
import json
import numbers
import os
import statistics


IDENTITY_KEYS = (
    "backend",
    "model_id",
    "tokenizer_id",
    "num_prompts",
    "max_concurrency",
)


def aggregate_results(results: list[dict], source_paths: list[str]) -> dict:
    if not results:
        raise ValueError("at least one benchmark result is required")

    first = results[0]
    if not isinstance(first, dict):
        raise ValueError("benchmark results must be JSON objects")

    first_keys = set(first)
    for index, result in enumerate(results[1:], start=2):
        if not isinstance(result, dict):
            raise ValueError(f"benchmark result {index} is not a JSON object")
        if set(result) != first_keys:
            raise ValueError(f"benchmark result {index} has a different schema")

    for key in IDENTITY_KEYS:
        values = [result.get(key) for result in results]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"benchmark results disagree on {key}: {values}")

    aggregate = dict(first)
    for key in first:
        values = [result[key] for result in results]
        if all(
            isinstance(value, numbers.Real) and not isinstance(value, bool)
            for value in values
        ):
            aggregate[key] = statistics.median(values)

    aggregate["aggregation_method"] = "median"
    aggregate["aggregated_repetitions"] = len(results)
    aggregate["individual_result_files"] = [
        os.path.basename(path) for path in source_paths
    ]
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    loaded = []
    for path in args.results:
        with open(path) as f:
            loaded.append(json.load(f))

    aggregate = aggregate_results(loaded, args.results)
    with open(args.output, "w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"  median aggregate: {len(loaded)} repetitions -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
