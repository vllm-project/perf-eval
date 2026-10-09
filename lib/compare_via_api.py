#!/usr/bin/env python3
"""
Thin client that calls the ci.vllm.ai compare API after eval ingestion
and prints a formatted regression table to the Buildkite log.

Usage:
    python3 lib/compare_via_api.py --candidate <IMAGE> [OPTIONS]

Exit codes:
    0  All checks passed (or comparison was skipped).
    1  At least one regression detected.
    2  API or argument error.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
import urllib.parse


def fetch_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def resolve_baseline(dashboard_url: str) -> str | None:
    """Ask the dashboard for the latest release baseline image."""
    url = f"{dashboard_url}/api/eval/baseline"
    try:
        data = fetch_json(url)
        return data.get("baselineImage")
    except Exception as exc:
        print(f"  Warning: could not resolve baseline: {exc}", file=sys.stderr)
        return None


def run_comparison(
    dashboard_url: str,
    baseline: str,
    candidate: str,
    eval_sigma: float,
    perf_threshold: float,
) -> dict:
    """Call /api/compare and return the parsed JSON."""
    params = urllib.parse.urlencode({
        "baseline": baseline,
        "candidate": candidate,
        "eval_sigma": str(eval_sigma),
        "perf_threshold": str(perf_threshold),
    })
    url = f"{dashboard_url}/api/compare?{params}"
    return fetch_json(url)


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}pp"


def fmt_sigma(s: float | None) -> str:
    if s is None:
        return ""
    return f"{s:.1f}σ"


def status_icon(status: str) -> str:
    icons = {
        "regression": "REGRESSION",
        "improvement": "IMPROVED",
        "unchanged": "PASS",
        "noisy": "NOISY",
    }
    return icons.get(status, status.upper())


def print_report(result: dict, baseline: str, candidate: str) -> int:
    """Print a formatted table and return 0 (pass) or 1 (regression)."""
    summary = result.get("summary", {})
    regressions = summary.get("regressions", 0)

    print()
    print("=" * 72)
    print("  Eval Regression Check")
    print(f"  Baseline:  {baseline}")
    print(f"  Candidate: {candidate}")
    print("=" * 72)
    print()

    # Collect all deltas (eval + perf).
    all_deltas = []
    for section in ("eval", "perf"):
        for delta in result.get(section, {}).get("deltas", []):
            all_deltas.append(delta)

    if not all_deltas:
        print("  No comparable metrics found between baseline and candidate.")
        print()
        return 0

    # Column widths.
    header = f"  {'TASK':<22} {'METRIC':<18} {'BASELINE':>10} {'CURRENT':>10} {'DELTA':>10} {'STATUS'}"
    sep = "  " + "-" * 68
    print(header)
    print(sep)

    for d in all_deltas:
        # For eval deltas, dimension is "task - n_shot-shot - filter".
        dim = d.get("dimension", "")
        task = dim.split(" - ")[0] if " - " in dim else dim
        metric = d.get("metric", "")
        bv = d.get("baselineValue", 0)
        cv = d.get("candidateValue", 0)
        delta = d.get("delta", 0)
        status = d.get("status", "unchanged")
        sig = d.get("significance")

        status_str = status_icon(status)
        if status == "regression" and sig is not None:
            status_str += f" ({fmt_sigma(sig)})"

        print(
            f"  {task:<22} {metric:<18} {fmt_pct(bv):>10} {fmt_pct(cv):>10} "
            f"{fmt_delta(delta):>10}   {status_str}"
        )

    print()
    print(
        f"  Summary: {regressions} regression(s), "
        f"{summary.get('improvements', 0)} improvement(s), "
        f"{summary.get('unchanged', 0) + summary.get('noisy', 0)} unchanged"
    )

    compare_url = (
        f"https://ci.vllm.ai/compare?"
        f"{urllib.parse.urlencode({'baseline': baseline, 'candidate': candidate})}"
    )
    print(f"  Dashboard: {compare_url}")
    print()

    if regressions > 0:
        print("  REGRESSION DETECTED")
        print()
        return 1
    else:
        print("  ALL CHECKS PASSED")
        print()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eval results against baseline via dashboard API",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate vLLM image to check (e.g. vllm/vllm-openai:nightly-abc123f)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline image (default: auto-resolve latest release from dashboard)",
    )
    parser.add_argument(
        "--dashboard-url",
        default="https://ci.vllm.ai",
        help="Dashboard base URL (default: https://ci.vllm.ai)",
    )
    parser.add_argument(
        "--eval-sigma",
        type=float,
        default=2.0,
        help="Sigma threshold for eval regression (default: 2.0)",
    )
    parser.add_argument(
        "--perf-threshold",
        type=float,
        default=0.02,
        help="Relative threshold for perf regression (default: 0.02)",
    )
    args = parser.parse_args()

    # Resolve baseline.
    baseline = args.baseline
    if not baseline:
        print("--- Resolving baseline from dashboard...")
        baseline = resolve_baseline(args.dashboard_url)
        if not baseline:
            print("  Could not resolve baseline. Skipping comparison.")
            return 0
        print(f"  Resolved baseline: {baseline}")

    if baseline == args.candidate:
        print(f"  Candidate is the same as baseline ({baseline}). Skipping.")
        return 0

    # Run comparison.
    print(f"--- Comparing {args.candidate} against {baseline}...")
    try:
        result = run_comparison(
            args.dashboard_url,
            baseline,
            args.candidate,
            args.eval_sigma,
            args.perf_threshold,
        )
    except urllib.error.HTTPError as exc:
        print(f"  API error: {exc.code} {exc.reason}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"  Error calling compare API: {exc}", file=sys.stderr)
        return 2

    return print_report(result, baseline, args.candidate)


if __name__ == "__main__":
    sys.exit(main())
