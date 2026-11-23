"""Run one Kimi Vendor Verifier (KVV) job against a running vLLM server.

Usage:
    python3 lib/run_kvv.py <model> <base_url> <job> <ref> <thinking_effort> \
        <out_dir> <task>

<job> is a KVV run_eval.sh job (e.g. verify_parameters, beam_generate,
ocrbench_thinking_effort:max); <ref> is the KVV commit to run. <out_dir> and
<task> come from parse_workload.py (kvv_result_names).

KVV (https://github.com/MoonshotAI/Kimi-Vendor-Verifier) is fetched as a
tarball at <ref> into $KVV_CACHE_DIR (default .kvv/), with its git-lfs files
resolved from GitHub's media endpoint, so the image needs neither git nor
git-lfs. run_eval.sh then builds KVV's own uv environment (Python >= 3.12)
and runs the job.

Each job's outputs land in <out_dir>: the run_eval.sh log, inspect-ai logs,
JUnit reports, BEAM answers/scores, a summary.json, and (for jobs with a
score) an lm_eval-compatible results_<ts>.json for ingest.py. A row is also
appended to <out_dir>/../kvv-summary.md for the build annotation.

beam_judge judges another step's beam_generate answers. They come from
$KVV_ANSWERS_FILE, or else from the Buildkite artifact $KVV_ANSWERS_ARTIFACT
of step $KVV_ANSWERS_STEP (set by generate_pipeline.py). The judge model is
the local server, which must serve the name kimi-k2.6.

Exits non-zero when the job fails or its result is incomplete (inspect
samples that errored, BEAM questions with no answer, judge calls that gave
up), after writing whatever results it has.
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
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

KVV_REPO = "MoonshotAI/Kimi-Vendor-Verifier"
TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/{ref}"
LFS_MEDIA_URL = "https://media.githubusercontent.com/media/{repo}/{ref}/{path}"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
# Pointer files are ~130 bytes; anything this large is real content.
LFS_POINTER_MAX_BYTES = 1024
COMPLETE_MARKER = ".perf-eval-complete"
HTTP_TIMEOUT = 300

# JUnit report each pytest job writes into the KVV root (see run_eval.sh).
JUNIT_REPORTS = {
    "verify_parameters": "params-report.xml",
    "verify_tool_call_json_schema": "tool-call-schema-report.xml",
    "verify_toolcall_result_reject": "toolcall-reject-report.xml",
    "verify_prompt_tokens": "prompt-tokens-report.xml",
    "verify_k3_features": "k3-features-report.xml",
}
# Extra per-job files worth keeping alongside the JUnit report.
EXTRA_REPORTS = {
    "verify_tool_call_json_schema": ["tool-call-schema-report.json"],
}
INSPECT_JOB_PREFIXES = ("ocrbench_", "MMMU_pro_", "AIME_", "toolcall_")
# run_eval.sh runs beam_judge.py with its default --max-retries; a judge call
# that fails this many times scores 0 without being recorded anywhere else.
BEAM_JUDGE_MAX_RETRIES = 8
BEAM_JUDGE_GAVE_UP = re.compile(rf"judge attempt {BEAM_JUDGE_MAX_RETRIES} failed")
BEAM_GENERATE_LOADED = re.compile(r"^loaded (\d+) questions")


def log(msg: str) -> None:
    print(f"[kvv] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Fetching KVV
# ---------------------------------------------------------------------------

def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "vllm-perf-eval"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)


def parse_lfs_pointer(data: bytes) -> tuple[str, int] | None:
    """(sha256 oid, size) if `data` is a git-lfs pointer file, else None."""
    if not data.startswith(LFS_POINTER_PREFIX):
        return None
    fields = dict(
        line.split(" ", 1) for line in data.decode("utf-8", "replace").splitlines() if " " in line
    )
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:") or not fields.get("size", "").isdigit():
        return None
    return oid.removeprefix("sha256:"), int(fields["size"])


def find_lfs_pointers(root: Path) -> list[tuple[Path, str, int]]:
    pointers = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink() or p.stat().st_size > LFS_POINTER_MAX_BYTES:
            continue
        parsed = parse_lfs_pointer(p.read_bytes())
        if parsed:
            pointers.append((p, *parsed))
    return pointers


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_lfs(root: Path, ref: str) -> None:
    for path, oid, size in find_lfs_pointers(root):
        rel = path.relative_to(root).as_posix()
        url = LFS_MEDIA_URL.format(repo=KVV_REPO, ref=ref, path=rel)
        tmp = path.with_name(path.name + ".lfs-download")
        download(url, tmp)
        got_size, got_oid = tmp.stat().st_size, sha256_of(tmp)
        if got_size != size or got_oid != oid:
            tmp.unlink()
            sys.exit(
                f"[kvv] LFS object for {rel} does not match its pointer "
                f"(want {oid}/{size} bytes, got {got_oid}/{got_size} bytes)"
            )
        tmp.replace(path)
        log(f"resolved LFS file {rel} ({size} bytes)")


def fetch_kvv(ref: str, cache_dir: Path) -> Path:
    """A complete KVV checkout at `ref`, fetched once per cache dir."""
    root = cache_dir / ref
    if (root / COMPLETE_MARKER).is_file():
        return root
    if root.exists():
        shutil.rmtree(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        tarball = Path(tmp) / "kvv.tar.gz"
        log(f"fetching {KVV_REPO}@{ref}")
        download(TARBALL_URL.format(repo=KVV_REPO, ref=ref), tarball)
        extract = Path(tmp) / "src"
        with tarfile.open(tarball) as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(extract, filter="data")
            else:
                tf.extractall(extract)  # noqa: S202 - pinned upstream tarball
        (top,) = extract.iterdir()
        resolve_lfs(top, ref)
        top.rename(root)
    (root / COMPLETE_MARKER).write_text(ref + "\n")
    return root


# ---------------------------------------------------------------------------
# Running a job
# ---------------------------------------------------------------------------

def job_env(model: str, base_url: str, thinking_effort: str, inspect_dir: Path,
            judge: bool) -> dict:
    api = base_url.rstrip("/") + "/v1"
    env = dict(os.environ)
    # perf-eval's own venv is active in CI; `uv pip install -e .` in run_eval.sh
    # would install KVV into it instead of KVV's project venv.
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(var, None)
    env.update({
        "KIMI_BASE_URL": api,
        # vLLM is served without --api-key, but run_eval.sh requires one.
        "KIMI_API_KEY": "EMPTY",
        "MODEL_NAME": model,
        "THINK_MODE": "opensource",
        "ENABLE_THINKING_EFFORT": thinking_effort,
        "INSPECT_LOG_DIR": str(inspect_dir),
    })
    if judge:
        env["JUDGE_BASE_URL"] = api
        env["JUDGE_API_KEY"] = "EMPTY"
    return env


def run_job(kvv_root: Path, job: str, env: dict, log_path: Path) -> tuple[int, list[str]]:
    """Run run_eval.sh <job>, teeing output to stdout and log_path."""
    lines = []
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            ["bash", "run_eval.sh", job], cwd=kvv_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            lines.append(line.rstrip("\n"))
        rc = proc.wait()
    return rc, lines


def fetch_answers(dest: Path) -> None:
    """Put the beam_generate answers to judge at dest."""
    local = (os.environ.get("KVV_ANSWERS_FILE") or "").strip()
    if local:
        shutil.copyfile(local, dest)
        return
    artifact = (os.environ.get("KVV_ANSWERS_ARTIFACT") or "").strip()
    step = (os.environ.get("KVV_ANSWERS_STEP") or "").strip()
    if not artifact or not step:
        sys.exit(
            "[kvv] beam_judge needs answers: set KVV_ANSWERS_FILE, or "
            "KVV_ANSWERS_ARTIFACT and KVV_ANSWERS_STEP"
        )
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["buildkite-agent", "artifact", "download", artifact, tmp, "--step", step],
            check=True,
        )
        found = sorted(Path(tmp).rglob(Path(artifact).name))
        if len(found) != 1:
            sys.exit(f"[kvv] expected one {artifact} from step {step}, found {len(found)}")
        shutil.copyfile(found[0], dest)


# ---------------------------------------------------------------------------
# Collecting results
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_junit(path: Path) -> dict:
    """Final per-test outcomes from a pytest JUnit XML report."""
    root = ET.parse(path).getroot()
    passed, failed, skipped = 0, [], 0
    for case in root.iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}".lstrip(":")
        tags = {child.tag for child in case}
        if tags & {"failure", "error"}:
            failed.append(name)
        elif "skipped" in tags:
            skipped += 1
        else:
            passed += 1
    ran = passed + len(failed)
    return {
        "score": passed / ran if ran else None,
        "stderr": 0.0,
        "passed": passed,
        "failed": len(failed),
        "skipped": skipped,
        "failures": failed,
    }


def read_inspect_header(path: Path) -> dict:
    """The log header (everything but samples) of an inspect-ai .eval/.json log."""
    if path.suffix == ".eval":
        with zipfile.ZipFile(path) as zf:
            if "header.json" not in zf.namelist():
                return {"status": "incomplete"}
            return json.loads(zf.read("header.json"))
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_inspect_log(path: Path) -> dict:
    header = read_inspect_header(path)
    results = header.get("results") or {}
    scores = results.get("scores") or []
    metrics = scores[0].get("metrics", {}) if scores else {}

    def metric(name):
        m = metrics.get(name)
        return m.get("value") if isinstance(m, dict) else None

    return {
        "score": metric("accuracy"),
        "stderr": metric("stderr"),
        "status": header.get("status"),
        "total_samples": results.get("total_samples"),
        "completed_samples": results.get("completed_samples"),
        "log": path.name,
    }


def beam_answers_summary(rows: list[dict]) -> dict:
    errors = [r for r in rows if r.get("error")]
    return {"answers": len(rows), "errors": len(errors)}


def beam_judge_summary(answers: list[dict], scores: list[dict]) -> dict:
    """Overall and per-question-type BEAM means over the judged answers."""
    def key(r):
        return (r["chat_id"], r["question_type"], r["question_index"])

    by_key = {key(r): float(r["score"]) for r in scores}
    per_type: dict[str, list[float]] = {}
    values = []
    for row in answers:
        s = by_key.get(key(row))
        if s is None:
            continue
        values.append(s)
        per_type.setdefault(row["question_type"], []).append(s)
    n = len(values)
    mean = sum(values) / n if n else None
    stderr = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else 0.0
    )
    return {
        "score": mean,
        "stderr": stderr,
        "judged": n,
        "answers": len(answers),
        "per_type": {qt: sum(v) / len(v) for qt, v in sorted(per_type.items())},
    }


def to_lm_eval_format(model: str, task: str, score: float, stderr: float | None) -> dict:
    return {
        "results": {
            task: {
                "acc,none": score,
                "acc_stderr,none": stderr if stderr is not None else 0.0,
                "alias": task,
            }
        },
        "configs": {task: {"task": task, "num_fewshot": 0}},
        "versions": {task: 1},
        "n-shot": {task: 0},
        "config": {
            "model": "local-chat-completions",
            "model_args": f"model={model}",
        },
    }


def is_inspect_job(job: str) -> bool:
    return job.startswith(INSPECT_JOB_PREFIXES)


def collect(job: str, kvv_root: Path, out_dir: Path, lines: list[str]) -> tuple[dict, list[str]]:
    """Result metrics for a finished job, and why it is incomplete (if it is)."""
    problems: list[str] = []
    if job in JUNIT_REPORTS:
        report = kvv_root / JUNIT_REPORTS[job]
        if not report.is_file():
            return {}, [f"no JUnit report {report.name}"]
        for name in [JUNIT_REPORTS[job], *EXTRA_REPORTS.get(job, [])]:
            if (kvv_root / name).is_file():
                shutil.copy2(kvv_root / name, out_dir / name)
        result = parse_junit(report)
        if result["failed"]:
            problems.append(f"{result['failed']} test(s) failed")
        return result, problems

    if is_inspect_job(job):
        logs = sorted((out_dir / "inspect").glob("*.eval")) + sorted(
            (out_dir / "inspect").glob("*.json")
        )
        if len(logs) != 1:
            return {}, [f"expected one inspect log in {out_dir / 'inspect'}, found {len(logs)}"]
        result = parse_inspect_log(logs[0])
        if result["status"] != "success":
            problems.append(f"inspect status {result['status']!r}")
        total, done = result["total_samples"], result["completed_samples"]
        if total is None or done != total:
            problems.append(f"{done}/{total} samples completed")
        if result["score"] is None:
            problems.append("no accuracy in inspect log")
        return result, problems

    if job == "beam_generate":
        answers = kvv_root / "beam" / "answers.jsonl"
        if not answers.is_file():
            return {}, ["no beam/answers.jsonl"]
        shutil.copy2(answers, out_dir / "answers.jsonl")
        result = beam_answers_summary(read_jsonl(answers))
        loaded = [int(m.group(1)) for line in lines if (m := BEAM_GENERATE_LOADED.match(line))]
        result["questions"] = loaded[-1] if loaded else None
        if result["errors"]:
            problems.append(f"{result['errors']} question(s) failed and scored as empty answers")
        if result["questions"] is None or result["answers"] != result["questions"]:
            problems.append(f"{result['answers']}/{result['questions']} questions answered")
        return result, problems

    if job == "beam_judge":
        scores = kvv_root / "beam" / "scores.jsonl"
        if not scores.is_file():
            return {}, ["no beam/scores.jsonl"]
        shutil.copy2(scores, out_dir / "scores.jsonl")
        answers = read_jsonl(kvv_root / "beam" / "answers.jsonl")
        result = beam_judge_summary(answers, read_jsonl(scores))
        result["judge_calls_gave_up"] = sum(1 for line in lines if BEAM_JUDGE_GAVE_UP.search(line))
        if result["judged"] != result["answers"]:
            problems.append(f"{result['judged']}/{result['answers']} answers judged")
        if result["judge_calls_gave_up"]:
            problems.append(
                f"{result['judge_calls_gave_up']} judge call(s) gave up and scored 0"
            )
        return result, problems

    return {}, problems


def clear_job_outputs(job: str, kvv_root: Path) -> None:
    """Remove outputs a previous job in this checkout left behind.

    BEAM generate/judge resume from their output files, so stale ones would
    be silently merged into this run.
    """
    stale = [kvv_root / "logs" / "kimi_retry.log"]
    if job in JUNIT_REPORTS:
        stale += [kvv_root / n for n in [JUNIT_REPORTS[job], *EXTRA_REPORTS.get(job, [])]]
    if job in ("beam_generate", "beam_judge"):
        stale += [kvv_root / "beam" / "answers.jsonl", kvv_root / "beam" / "scores.jsonl"]
    for p in stale:
        p.unlink(missing_ok=True)


def append_summary_row(path: Path, job: str, result: dict, seconds: float,
                       problems: list[str], rc: int) -> None:
    if not path.exists():
        path.write_text("| job | score | time | status |\n|---|---|---|---|\n")
    score = result.get("score")
    shown = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
    status = "ok" if rc == 0 and not problems else "; ".join(
        ([f"exit {rc}"] if rc else []) + problems
    )
    with path.open("a") as f:
        f.write(f"| {job} | {shown} | {seconds / 60:.1f} min | {status} |\n")


def main() -> None:
    if len(sys.argv) != 8:
        sys.exit(
            "usage: run_kvv.py <model> <base_url> <job> <ref> <thinking_effort> "
            "<out_dir> <task>"
        )
    model, base_url, job, ref, thinking_effort = sys.argv[1:6]
    out_dir, task = Path(sys.argv[6]).resolve(), sys.argv[7]
    out_dir.mkdir(parents=True, exist_ok=True)

    kvv_root = fetch_kvv(ref, Path(os.environ.get("KVV_CACHE_DIR") or ".kvv").resolve())
    clear_job_outputs(job, kvv_root)
    if job == "beam_judge":
        fetch_answers(kvv_root / "beam" / "answers.jsonl")

    inspect_dir = out_dir / "inspect"
    if inspect_dir.exists():
        shutil.rmtree(inspect_dir)
    inspect_dir.mkdir()
    env = job_env(model, base_url, thinking_effort, inspect_dir, judge=job == "beam_judge")

    log(f"{job}: model={model} kvv={ref}")
    start = time.monotonic()
    rc, lines = run_job(kvv_root, job, env, out_dir / "run_eval.log")
    seconds = time.monotonic() - start

    result, problems = collect(job, kvv_root, out_dir, lines)
    retry_log = kvv_root / "logs" / "kimi_retry.log"
    if retry_log.is_file():
        shutil.copy2(retry_log, out_dir / "kimi_retry.log")
    if not any(inspect_dir.iterdir()):
        inspect_dir.rmdir()

    summary = {
        "job": job, "task": task, "model": model, "kvv_ref": ref,
        "exit_code": rc, "seconds": round(seconds, 1), "problems": problems, **result,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    append_summary_row(out_dir.parent / "kvv-summary.md", job, result, seconds, problems, rc)

    if isinstance(result.get("score"), (int, float)):
        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        out_path = out_dir / f"results_{ts}.json"
        out_path.write_text(json.dumps(
            to_lm_eval_format(model, task, result["score"], result.get("stderr")), indent=2
        ))
        log(f"{job}: score={result['score']:.4f} -> {out_path}")

    log(f"{job}: exit {rc} in {seconds / 60:.1f} min")
    for p in problems:
        log(f"{job}: {p}")
    if rc != 0 or problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
