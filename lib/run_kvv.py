"""Run one Kimi Vendor Verifier (KVV) job against the running vLLM server.

Usage: python3 lib/run_kvv.py <model> <base_url> <job> <ref> <out_dir> <task>

KVV (https://github.com/MoonshotAI/Kimi-Vendor-Verifier) is cloned at <ref>
into $KVV_CACHE_DIR (default .kvv/), with its git-lfs files fetched over
HTTPS so the image needs no git-lfs, and the job runs through KVV's own
`run_eval.sh <job>`. The local server is also the BEAM judge: run_eval.sh
asks for `kimi-k2.6`, so BEAM workloads serve that name as an alias.

<out_dir> gets the job's log and reports, a summary.json and, for scored
jobs, an lm_eval-format results_<ts>.json for ingest.py. A row is appended to
<out_dir>/../kvv-summary.md for the build annotation. Exits non-zero if the
job fails or its result is incomplete (errored inspect samples, unanswered
BEAM questions, judge calls that gave up).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

KVV_REPO = "MoonshotAI/Kimi-Vendor-Verifier"
LFS_URL = f"https://media.githubusercontent.com/media/{KVV_REPO}/{{ref}}/{{path}}"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
DONE = ".perf-eval-complete"
INSPECT_JOBS = ("ocrbench_", "MMMU_pro_", "AIME_", "toolcall_")
# beam_judge.py's default --max-retries; a call failing that often scores 0.
JUDGE_GAVE_UP = re.compile(r"judge attempt 8 failed")


def log(msg: str) -> None:
    print(f"[kvv] {msg}", flush=True)


def fetch_kvv(ref: str, cache: Path) -> Path:
    root = cache / ref
    if (root / DONE).is_file():
        return root
    shutil.rmtree(root, ignore_errors=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "clone", "-q", f"https://github.com/{KVV_REPO}.git", str(root)],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(root), "checkout", "-q", ref], check=True, env=env)
    for p in sorted(root.rglob("*")):
        if ".git" in p.parts or not p.is_file() or p.stat().st_size > 1024:
            continue
        data = p.read_bytes()
        if not data.startswith(LFS_POINTER):
            continue
        oid = dict(line.split(" ", 1) for line in data.decode().splitlines())["oid"]
        rel = p.relative_to(root).as_posix()
        with urllib.request.urlopen(LFS_URL.format(ref=ref, path=rel), timeout=300) as r:
            content = r.read()
        if f"sha256:{hashlib.sha256(content).hexdigest()}" != oid.strip():
            sys.exit(f"[kvv] LFS file {rel} does not match its pointer")
        p.write_bytes(content)
        log(f"fetched LFS file {rel}")
    (root / DONE).write_text(ref + "\n")
    return root


def run_job(root: Path, job: str, env: dict, log_path: Path) -> tuple[int, list[str]]:
    """run_eval.sh <job>, teeing its output to stdout and log_path."""
    lines = []
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(["bash", "run_eval.sh", job], cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line.rstrip("\n"))
    return proc.wait(), lines


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def junit_result(path: Path) -> tuple[dict, list[str]]:
    """Per-test outcomes of a pytest JUnit report; score = passed / ran."""
    passed, failed, skipped = 0, [], 0
    for case in ET.parse(path).getroot().iter("testcase"):
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            failed.append(f"{case.get('classname')}::{case.get('name')}")
        elif "skipped" in tags:
            skipped += 1
        else:
            passed += 1
    ran = passed + len(failed)
    result = {"score": passed / ran if ran else None, "passed": passed,
              "failed": len(failed), "skipped": skipped, "failures": failed}
    return result, [f"{len(failed)} test(s) failed"] if failed else []


def inspect_result(log_dir: Path) -> tuple[dict, list[str]]:
    logs = sorted(log_dir.glob("*.eval"))
    if len(logs) != 1:
        return {}, [f"expected one inspect log, found {len(logs)}"]
    with zipfile.ZipFile(logs[0]) as zf:
        if "header.json" not in zf.namelist():
            return {}, ["inspect log has no results"]
        header = json.loads(zf.read("header.json"))
    results = header.get("results") or {}
    metrics = (results.get("scores") or [{}])[0].get("metrics", {})
    result = {"score": metrics.get("accuracy", {}).get("value"),
              "stderr": metrics.get("stderr", {}).get("value"),
              "total_samples": results.get("total_samples"),
              "completed_samples": results.get("completed_samples")}
    problems = []
    if header.get("status") != "success":
        problems.append(f"inspect status {header.get('status')!r}")
    if result["total_samples"] is None or result["completed_samples"] != result["total_samples"]:
        problems.append(f"{result['completed_samples']}/{result['total_samples']} samples completed")
    if result["score"] is None:
        problems.append("no accuracy in inspect log")
    return result, problems


def beam_generate_result(answers: list[dict], lines: list[str]) -> tuple[dict, list[str]]:
    loaded = [int(m[1]) for line in lines if (m := re.match(r"loaded (\d+) questions", line))]
    result = {"questions": loaded[-1] if loaded else None, "answers": len(answers),
              "errors": sum(1 for a in answers if a.get("error"))}
    problems = []
    if result["errors"]:
        problems.append(f"{result['errors']} question(s) failed and scored as empty answers")
    if result["answers"] != result["questions"]:
        problems.append(f"{result['answers']}/{result['questions']} questions answered")
    return result, problems


def beam_judge_result(answers: list[dict], scores: list[dict],
                      lines: list[str]) -> tuple[dict, list[str]]:
    """The BEAM score as beam_judge.py's print_summary computes it: the mean
    over judged answers, overall and per question type."""
    def key(r):
        return r["chat_id"], r["question_type"], r["question_index"]

    by_key = {key(r): float(r["score"]) for r in scores}
    judged = [(a["question_type"], by_key[key(a)]) for a in answers if key(a) in by_key]
    values = [v for _, v in judged]
    n = len(values)
    mean = sum(values) / n if n else None
    per_type = {}
    for qt, v in judged:
        per_type.setdefault(qt, []).append(v)
    result = {
        "score": mean,
        "stderr": math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else 0.0,
        "judged": n, "answers": len(answers),
        "per_type": {qt: sum(v) / len(v) for qt, v in sorted(per_type.items())},
        "judge_calls_gave_up": sum(1 for line in lines if JUDGE_GAVE_UP.search(line)),
    }
    problems = []
    if n != len(answers):
        problems.append(f"{n}/{len(answers)} answers judged")
    if result["judge_calls_gave_up"]:
        problems.append(f"{result['judge_calls_gave_up']} judge call(s) gave up and scored 0")
    return result, problems


def collect(job: str, root: Path, out_dir: Path, lines: list[str]) -> tuple[dict, list[str]]:
    """Copy the job's outputs to out_dir; return its result and why it is incomplete."""
    for p in [*root.glob("*-report.*"), root / "logs" / "kimi_retry.log",
              *([root / "beam" / "answers.jsonl", root / "beam" / "scores.jsonl"]
                if job.startswith("beam_") else [])]:
        if p.is_file():
            shutil.copy2(p, out_dir)
    if job.startswith("verify_"):
        reports = list(root.glob("*-report.xml"))
        return junit_result(reports[0]) if len(reports) == 1 else ({}, ["no JUnit report"])
    if job.startswith(INSPECT_JOBS):
        return inspect_result(out_dir / "inspect")
    answers = read_jsonl(root / "beam" / "answers.jsonl")
    if job == "beam_generate":
        return beam_generate_result(answers, lines)
    if job == "beam_judge":
        return beam_judge_result(answers, read_jsonl(root / "beam" / "scores.jsonl"), lines)
    return {}, []


def main() -> None:
    if len(sys.argv) != 7:
        sys.exit("usage: run_kvv.py <model> <base_url> <job> <ref> <out_dir> <task>")
    model, base_url, job, ref = sys.argv[1:5]
    out_dir, task = Path(sys.argv[5]).resolve(), sys.argv[6]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not shutil.which("uv"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    root = fetch_kvv(ref, Path(os.environ.get("KVV_CACHE_DIR") or ".kvv").resolve())

    # Remove what an earlier job left in the checkout. BEAM resumes from its
    # output files; beam_judge keeps the answers beam_generate just wrote.
    stale = [*root.glob("*-report.*"), root / "logs" / "kimi_retry.log", root / "beam" / "scores.jsonl"]
    if job != "beam_judge":
        stale.append(root / "beam" / "answers.jsonl")
    for p in stale:
        p.unlink(missing_ok=True)
    shutil.rmtree(out_dir / "inspect", ignore_errors=True)

    api = base_url.rstrip("/") + "/v1"
    # perf-eval's venv is active in CI; run_eval.sh must use KVV's own uv venv.
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    env.update({
        # vLLM is served without --api-key, but run_eval.sh requires one.
        "KIMI_BASE_URL": api, "KIMI_API_KEY": "EMPTY", "MODEL_NAME": model,
        "JUDGE_BASE_URL": api, "JUDGE_API_KEY": "EMPTY",
        # KVV's settings for an open-source server and a K3-series model.
        "THINK_MODE": "opensource", "ENABLE_THINKING_EFFORT": "true",
        "INSPECT_LOG_DIR": str(out_dir / "inspect"),
    })
    log(f"{job}: model={model} kvv={ref}")
    start = time.monotonic()
    rc, lines = run_job(root, job, env, out_dir / "run_eval.log")
    minutes = (time.monotonic() - start) / 60

    result, problems = collect(job, root, out_dir, lines)
    summary = {"job": job, "task": task, "model": model, "kvv_ref": ref, "exit_code": rc,
               "minutes": round(minutes, 1), "problems": problems, **result}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    score = result.get("score")
    status = "; ".join(([f"exit {rc}"] if rc else []) + problems) or "ok"
    table = out_dir.parent / "kvv-summary.md"
    if not table.exists():
        table.write_text("| job | score | time | status |\n|---|---|---|---|\n")
    with table.open("a") as f:
        f.write(f"| {job} | {'-' if score is None else f'{score:.4f}'} | {minutes:.1f} min | {status} |\n")
    if score is not None:
        (out_dir / f"results_{time.strftime('%Y-%m-%dT%H-%M-%S')}.json").write_text(json.dumps({
            "results": {task: {"acc,none": score, "acc_stderr,none": result.get("stderr") or 0.0,
                               "alias": task}},
            "configs": {task: {"task": task, "num_fewshot": 0}},
            "versions": {task: 1}, "n-shot": {task: 0},
            "config": {"model": "local-chat-completions", "model_args": f"model={model}"},
        }, indent=2))
    log(f"{job}: score={score} ({status}) in {minutes:.1f} min")
    sys.exit(1 if rc or problems else 0)


if __name__ == "__main__":
    main()
