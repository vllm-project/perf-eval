#!/usr/bin/env python3
"""Summarize AC GLM-5.2 ablation artifacts into JSON and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


TTFT_P50_SLA_S = 1.6
TTFT_P95_SLA_S = 3.5
THROUGHPUT_FLOOR = 0.98


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def gpu_stats(path: Path) -> dict:
    values = []
    if path.exists():
        with path.open(errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[1].strip().isdigit():
                    continue
                match = re.search(r"[0-9.]+", row[2])
                if match:
                    values.append(float(match.group()))
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def value(metrics: dict, key: str, quantile: str) -> float | None:
    return (metrics.get(key) or {}).get(quantile)


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("load-*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        arm_dir = path.parent
        config, run = data["config"], data["run"]
        offered = float(config["target_prompt_tpm"])
        achieved = float(run["steady_achieved_prompt_tpm"])
        ttft_p50 = value(run, "ttft_s", "p50")
        ttft_p95 = value(run, "ttft_s", "p95")
        success = float(run["success_fraction"])
        row = {
            "arm": arm_dir.name,
            "artifact": str(path.relative_to(root)),
            "offered_prompt_tpm": offered,
            "planned_prompt_tpm": config.get("actual_planned_prompt_tpm"),
            "achieved_prompt_tpm": achieved,
            "achieved_fraction": achieved / offered,
            "success_fraction": success,
            "ttft_s": run.get("ttft_s"),
            "mean_itl_s": run.get("mean_itl_s"),
            "stream_chunk_itl_s": run.get("stream_chunk_itl_s"),
            "latency_s": run.get("latency_s"),
            "cache_hit_fraction": run.get("cache_hit_fraction"),
            "drain_s": run.get("drain_s"),
            "gpu_utilization_pct": gpu_stats(
                arm_dir / f"gpu-{int(offered)}.csv"
            ),
            "throughput_pass": achieved >= offered * THROUGHPUT_FLOOR,
            "ttft_pass": (
                ttft_p50 is not None and ttft_p95 is not None
                and ttft_p50 <= TTFT_P50_SLA_S
                and ttft_p95 <= TTFT_P95_SLA_S
            ),
            "acceptance_pass": (
                success == 1.0
                and achieved >= offered * THROUGHPUT_FLOOR
                and ttft_p50 is not None and ttft_p95 is not None
                and ttft_p50 <= TTFT_P50_SLA_S
                and ttft_p95 <= TTFT_P95_SLA_S
            ),
        }
        rows.append(row)
    return rows


def fnum(value_: float | None, scale: float = 1.0, digits: int = 3) -> str:
    return "—" if value_ is None else f"{value_ / scale:.{digits}f}"


def markdown(rows: list[dict], root: Path) -> str:
    lines = [
        "# Applied Compute GLM-5.2 8×B300 ablation",
        "",
        "Acceptance boundary: exactly one node with 8×NVIDIA B300 GPUs. "
        "A cell passes only with 100% request success, ≥98% of offered prompt "
        "TPM achieved in the steady completion window, TTFT p50 ≤1.6 s, and "
        "TTFT p95 ≤3.5 s. No ITL threshold was supplied, so ITL is reported "
        "but is not silently treated as a pass/fail gate.",
        "",
        "| Arm | Offered M TPM | Achieved M TPM | Success | TTFT p50/p95/p99 (s) | mean ITL p50/p95/p99 (ms) | Cache hit | GPU util mean/p95 | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        ttft, itl, gpu = row["ttft_s"], row["mean_itl_s"], row["gpu_utilization_pct"]
        lines.append(
            f"| {row['arm']} | {fnum(row['offered_prompt_tpm'], 1e6)} | "
            f"{fnum(row['achieved_prompt_tpm'], 1e6)} | "
            f"{row['success_fraction'] * 100:.1f}% | "
            f"{fnum(ttft.get('p50'))}/{fnum(ttft.get('p95'))}/{fnum(ttft.get('p99'))} | "
            f"{fnum(itl.get('p50'), 0.001, 1)}/{fnum(itl.get('p95'), 0.001, 1)}/{fnum(itl.get('p99'), 0.001, 1)} | "
            f"{fnum(row['cache_hit_fraction'] * 100 if row['cache_hit_fraction'] is not None else None, digits=1)}% | "
            f"{fnum(gpu['mean'], digits=1)}/{fnum(gpu['p95'], digits=1)}% | "
            f"{'PASS' if row['acceptance_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "Raw evidence root: `" + str(root) + "`.",
        "",
        "`mean ITL` is per-request TPOT: time from first to last non-empty "
        "stream chunk divided by completion tokens minus one. Raw SSE chunk "
        "inter-arrivals are also retained because a server may place multiple "
        "tokens in one chunk.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.root)
    args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(markdown(rows, args.root))
    print(f"summarized {len(rows)} load cells")


if __name__ == "__main__":
    main()
