#!/usr/bin/env python3
"""POST lm_eval JSON artifacts to the eval data ingestion endpoint.

Walks a per-task results dir produced by lm_eval and uploads:
  - results_*.json   -> one event per file
  - samples_*.jsonl  -> one event per line (per sample)

Each event is wrapped with workload/task/Buildkite metadata so rows in the
backing Databricks table can be filtered by run.

Failures are logged but never fatal: ingestion is best-effort and must not
abort the lm_eval pipeline.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://vllm-eval-data-ingest-224810116257.us-central1.run.app/"
TIMEOUT = 30
# Databricks Zerobus rejects records larger than 10 MiB and closes the stream.
# Pack samples into batches that stay safely under that ceiling.
SAMPLES_BATCH_BYTES = 4 * 1024 * 1024
BK_ENV_VARS = (
    "BUILDKITE_BUILD_ID",
    "BUILDKITE_BUILD_NUMBER",
    "BUILDKITE_BUILD_URL",
    "BUILDKITE_BRANCH",
    "BUILDKITE_COMMIT",
    "BUILDKITE_PIPELINE_SLUG",
)
# Top-level fields the dashboard reads to show "image" and the vLLM commit.
# WORKLOAD_IMAGE is the resolved docker URI (set by parse_workload.py via the
# VLLM_IMAGE / VLLM_COMMIT override env vars or the workload yaml's vllm.image).
# WORKLOAD_VLLM_COMMIT is the commit used by that resolved image, when it can
# be determined from VLLM_COMMIT or a commit-bearing image tag.
VLLM_ENV_VARS = (
    ("WORKLOAD_IMAGE", "image"),
    ("WORKLOAD_VLLM_COMMIT", "vllm_commit"),
)


def post(endpoint: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status}")


def metadata(workload: str, task: str) -> dict:
    md = {"workload": workload, "task": task}
    for k in BK_ENV_VARS:
        v = os.environ.get(k)
        if v:
            md[k.lower()] = v
    for env_key, field in VLLM_ENV_VARS:
        v = (os.environ.get(env_key) or "").strip()
        if v:
            md[field] = v
    return md


def ingest_results(path: Path, md: dict, endpoint: str) -> None:
    with path.open() as f:
        data = json.load(f)
    payload = {"kind": "results", "source_file": str(path), **md, "data": data}
    post(endpoint, payload)


def ingest_samples(path: Path, md: dict, endpoint: str) -> int:
    sent = 0
    batch: list = []
    batch_bytes = 0
    overhead = len(
        json.dumps({"kind": "samples", "source_file": str(path), **md, "samples": []})
    )

    def flush() -> None:
        nonlocal batch, batch_bytes, sent
        if not batch:
            return
        payload = {"kind": "samples", "source_file": str(path), **md, "samples": batch}
        post(endpoint, payload)
        sent += len(batch)
        batch = []
        batch_bytes = 0

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    skip malformed sample line: {e}", file=sys.stderr)
                continue
            sample_bytes = len(json.dumps(sample))
            # +1 for the array-element comma
            if batch and overhead + batch_bytes + sample_bytes + 1 > SAMPLES_BATCH_BYTES:
                flush()
            batch.append(sample)
            batch_bytes += sample_bytes + 1
    flush()
    return sent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, help="Per-task results dir from lm_eval")
    p.add_argument("--workload", required=True, help="Workload (recipe) name")
    p.add_argument("--task", required=True, help="lm_eval task name")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("INGEST_URL", DEFAULT_ENDPOINT),
        help="Ingestion endpoint (env: INGEST_URL)",
    )
    p.add_argument("--no-samples", action="store_true", help="Skip samples_*.jsonl uploads")
    args = p.parse_args()

    root = Path(args.results_dir)
    if not root.is_dir():
        print(f"  ingest: results dir not found: {root}", file=sys.stderr)
        return 0

    results_files = sorted(root.glob("**/results_*.json"))
    samples_files = [] if args.no_samples else sorted(root.glob("**/samples_*.jsonl"))
    print(
        f"  ingest -> {args.endpoint}  ({len(results_files)} results, "
        f"{len(samples_files)} sample file(s))"
    )

    md = metadata(args.workload, args.task)

    for f in results_files:
        try:
            ingest_results(f, md, args.endpoint)
            print(f"    posted {f.relative_to(root)}")
        except (urllib.error.URLError, RuntimeError, OSError) as e:
            print(f"    failed {f.relative_to(root)}: {e}", file=sys.stderr)

    for f in samples_files:
        try:
            n = ingest_samples(f, md, args.endpoint)
            print(f"    posted {f.relative_to(root)} ({n} samples)")
        except OSError as e:
            print(f"    failed {f.relative_to(root)}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
