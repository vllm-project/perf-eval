#!/usr/bin/env python3
"""Transform a `vllm bench serve` raw JSON result and POST it to the perf
dashboard's ingestion endpoint.

The dashboard at perf.vllm.ai reads from the `vllm_perf_data_ingest`
Databricks table; that table is populated by the Cloud Run endpoint pinged
here. Schema modeled after vllm-project/perf-dashboard/benchmark/process_result.py.

Latencies in the raw result are in milliseconds (e.g., `mean_ttft_ms`).
The dashboard expects seconds, so the transform drops the `_ms` suffix and
divides by 1000. Aggregate throughput is divided by `tp` to match the
dashboard's per-GPU columns (`tput_per_gpu`, `output_tput_per_gpu`,
`input_tput_per_gpu`).

Failures are logged but never fatal: ingestion is best-effort and must not
abort the bench step.
"""

import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "https://vllm-perf-data-ingest-224810116257.us-central1.run.app/"
TIMEOUT = 30


def post(endpoint: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "X-Source": "perf-eval"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"HTTP {resp.status}")


def transform(raw: dict, args: argparse.Namespace) -> dict:
    """Map the raw `vllm bench serve` JSON to the dashboard's row shape."""
    tp = max(args.tp, 1)
    total_token_throughput = float(raw.get("total_token_throughput", 0) or 0)
    output_throughput = float(raw.get("output_throughput", 0) or 0)
    input_throughput = total_token_throughput - output_throughput

    data = {
        "date": args.date or datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "conc": int(raw.get("max_concurrency") or args.conc),
        "image": args.image,
        "model": raw.get("model_id") or args.model,
        "framework": "vllm",
        "precision": args.precision,
        "spec_decoding": "false",
        "disagg": "false",
        "isl": int(args.isl),
        "osl": int(args.osl),
        "is_multinode": "false",
        "tp": tp,
        "ep": 1,
        "dp_attention": "false",
        "tput_per_gpu": total_token_throughput / tp,
        "output_tput_per_gpu": output_throughput / tp,
        "input_tput_per_gpu": input_throughput / tp,
    }

    # Convert *_ms fields to seconds and emit interactivity (1000/tpot_ms).
    for key, value in raw.items():
        if not key.endswith("_ms"):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        data[key.removesuffix("_ms")] = v / 1000.0
        if "tpot" in key:
            data[key.removesuffix("_ms").replace("tpot", "intvty")] = (
                1000.0 / v if v else 0.0
            )

    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-result", required=True, help="Raw JSON from `vllm bench serve --save-result`")
    p.add_argument("--device", required=True, help="Device tag (e.g. h200)")
    p.add_argument("--tp", type=int, required=True, help="Effective parallel-degree (TP * DP)")
    p.add_argument("--precision", required=True, help="Precision tag (e.g. fp8, bf16)")
    p.add_argument("--model", required=True, help="HF model identifier (fallback if raw json lacks model_id)")
    p.add_argument("--image", required=True, help="Docker image used for the run")
    p.add_argument("--isl", type=int, required=True, help="Input sequence length used in the bench config")
    p.add_argument("--osl", type=int, required=True, help="Output sequence length used in the bench config")
    p.add_argument("--conc", type=int, required=True, help="Concurrency used (fallback if raw json lacks max_concurrency)")
    p.add_argument("--date", default=None, help="Override timestamp (YYYY-MM-DD HH:MM:SS); defaults to UTC now")
    p.add_argument(
        "--endpoint",
        default=os.environ.get("PERF_INGEST_URL", DEFAULT_ENDPOINT),
        help="Ingestion endpoint (env: PERF_INGEST_URL)",
    )
    args = p.parse_args()

    if not os.path.isfile(args.raw_result):
        print(f"  perf-ingest: raw result file not found: {args.raw_result}", file=sys.stderr)
        return 0

    with open(args.raw_result) as f:
        raw = json.load(f)

    data = transform(raw, args)
    print(f"  perf-ingest -> {args.endpoint}")
    print(f"    tput_per_gpu={data['tput_per_gpu']:.2f}  "
          f"mean_ttft={data.get('mean_ttft', 0):.4f}s  "
          f"mean_tpot={data.get('mean_tpot', 0):.4f}s")
    try:
        post(args.endpoint, data)
        print("    posted ok")
    except (urllib.error.URLError, RuntimeError, OSError) as e:
        print(f"    failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
