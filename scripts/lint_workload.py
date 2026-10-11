#!/usr/bin/env python3
"""Lint workload recipes before merge -- no GPU, no model download required.

Runs the same structural checks the nightly pipeline relies on, so a broken
recipe fails in review instead of at 3am on a B200:

  1. ``lib/parse_workload.py`` accepts the recipe -- required fields, known GPU
     profile, well-formed lm_eval / vllm_bench / bfcl blocks, and (when
     ``lm_eval`` is importable) task names against the registry.
  2. ``.buildkite/generate_pipeline.py`` emits a buildkite step for it.

This is the automatic gate; it does NOT prove accuracy. The real proof is a
scoped hardware run -- see CONTRIBUTING.md ("Validate on hardware").

Usage:
  scripts/lint_workload.py workloads/foo_b200.yaml [more.yaml ...]
  scripts/lint_workload.py --changed   # recipes added/changed vs origin/main
  scripts/lint_workload.py --all       # every workloads/*.yaml
"""

import argparse
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        cmd, cwd=ROOT, env=env, capture_output=True, text=True
    )


def _lm_eval_available():
    return _run([sys.executable, "-c", "import lm_eval"]).returncode == 0


def lint_one(path, skip_registry):
    parse_env = {"SKIP_TASK_REGISTRY": "1"} if skip_registry else {}
    parsed = _run([sys.executable, "lib/parse_workload.py", path], parse_env)
    if parsed.returncode != 0:
        return False, (parsed.stderr or parsed.stdout).strip()

    gen = _run(
        [sys.executable, ".buildkite/generate_pipeline.py"],
        {"WORKLOADS": path, "VLLM_IMAGE": "vllm/vllm-openai:lint"},
    )
    if gen.returncode != 0:
        return False, "generate_pipeline: " + gen.stderr.strip()
    if "label:" not in gen.stdout:
        return False, "generate_pipeline emitted no step for this recipe"
    return True, "ok"


def changed_workloads():
    base = os.environ.get("LINT_BASE_REF", "origin/main")
    diff = _run(
        ["git", "diff", "--name-only", "--diff-filter=d", base, "--", "workloads/"]
    )
    return [ln for ln in diff.stdout.splitlines() if ln.endswith(".yaml")]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="*", help="workload YAML paths to lint")
    ap.add_argument(
        "--changed", action="store_true", help="lint recipes changed vs origin/main"
    )
    ap.add_argument("--all", action="store_true", help="lint every workloads/*.yaml")
    args = ap.parse_args()

    os.chdir(ROOT)
    if args.all:
        paths = sorted(glob.glob("workloads/*.yaml"))
    elif args.changed:
        paths = changed_workloads()
    else:
        paths = args.paths
    if not paths:
        print("lint_workload: no workloads to lint")
        return 0

    skip_registry = not _lm_eval_available()
    if skip_registry:
        print("lint_workload: lm_eval not importable; skipping task-name validation")

    failed = 0
    for path in paths:
        ok, msg = lint_one(path, skip_registry)
        print(f"{'PASS' if ok else 'FAIL'}  {path}")
        if not ok:
            print(f"      {msg}")
            failed += 1
    print(f"lint_workload: {len(paths) - failed}/{len(paths)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
