#!/usr/bin/env python3
"""Stdlib-only regression tests for lib/run_kvv.py.

Run with ``python3 .buildkite/test_run_kvv.py``. No KVV checkout, GPU or
network required: the KVV outputs each job is scored from are built here in
the shapes KVV writes them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_kvv", ROOT / "lib" / "run_kvv.py")
run_kvv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_kvv)

REF = "7b938c38bbd89c2404a9ed5e7cb40da8079206b4"


def lfs_pointer(content: bytes) -> bytes:
    oid = hashlib.sha256(content).hexdigest()
    return (
        b"version https://git-lfs.github.com/spec/v1\n"
        + f"oid sha256:{oid}\nsize {len(content)}\n".encode()
    )


def write_eval_log(path: Path, header: dict | None) -> None:
    """An inspect-ai .eval log: a zip whose header.json holds the results."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("_journal/start.json", "{}")
        if header is not None:
            zf.writestr("header.json", json.dumps(header))


def inspect_header(status="success", total=10, completed=10, accuracy=0.8):
    metrics = {"stderr": {"name": "stderr", "value": 0.05}}
    if accuracy is not None:
        metrics["accuracy"] = {"name": "accuracy", "value": accuracy}
    return {
        "status": status,
        "results": {
            "total_samples": total,
            "completed_samples": completed,
            "scores": [{"name": "match", "metrics": metrics}],
        },
    }


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="5">
  <testcase classname="tests.test_params" name="test_ok"/>
  <testcase classname="tests.test_params" name="test_ok2"/>
  <testcase classname="tests.test_params" name="test_bad"><failure message="x"/></testcase>
  <testcase classname="tests.test_params" name="test_err"><error message="y"/></testcase>
  <testcase classname="tests.test_params" name="test_skip"><skipped message="z"/></testcase>
</testsuite></testsuites>
"""


def answer(chat, qtype, idx, error=None):
    return {"chat_id": chat, "question_type": qtype, "question_index": idx,
            "response": "r", "error": error}


def score(chat, qtype, idx, value):
    return {"chat_id": chat, "question_type": qtype, "question_index": idx, "score": value}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TmpDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.kvv = self.tmp / "kvv"
        self.out = self.tmp / "out"
        self.kvv.mkdir()
        self.out.mkdir()

    def tearDown(self):
        self._tmp.cleanup()


class LfsTests(TmpDirTest):
    def test_parses_a_pointer(self):
        content = b"real data"
        oid, size = run_kvv.parse_lfs_pointer(lfs_pointer(content))
        self.assertEqual(oid, hashlib.sha256(content).hexdigest())
        self.assertEqual(size, len(content))

    def test_ordinary_files_are_not_pointers(self):
        self.assertIsNone(run_kvv.parse_lfs_pointer(b"import os\n"))
        self.assertIsNone(run_kvv.parse_lfs_pointer(
            b"version https://git-lfs.github.com/spec/v1\noid md5:abc\nsize 3\n"
        ))

    def test_finds_only_pointer_files(self):
        (self.kvv / "data").mkdir()
        (self.kvv / "data" / "q.parquet").write_bytes(lfs_pointer(b"parquet"))
        (self.kvv / "run_eval.sh").write_text("#!/bin/bash\n")
        big = lfs_pointer(b"x") + b"#" * run_kvv.LFS_POINTER_MAX_BYTES
        (self.kvv / "big.txt").write_bytes(big)
        found = [p.relative_to(self.kvv).as_posix()
                 for p, _, _ in run_kvv.find_lfs_pointers(self.kvv)]
        self.assertEqual(found, ["data/q.parquet"])


def fake_download(files: dict):
    def download(url, dest):
        dest.write_bytes(files[url])
    return download


def kvv_tarball(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(f"Kimi-Vendor-Verifier-{REF}/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FetchTests(TmpDirTest):
    def urls(self, lfs_content: bytes, served: bytes) -> dict:
        tarball = run_kvv.TARBALL_URL.format(repo=run_kvv.KVV_REPO, ref=REF)
        media = run_kvv.LFS_MEDIA_URL.format(
            repo=run_kvv.KVV_REPO, ref=REF, path="data/q.parquet"
        )
        members = {"run_eval.sh": b"#!/bin/bash\n", "data/q.parquet": lfs_pointer(lfs_content)}
        return {tarball: kvv_tarball(members), media: served}

    def test_fetches_tarball_and_resolves_lfs_once(self):
        files = self.urls(b"parquet bytes", b"parquet bytes")
        cache = self.tmp / "cache"
        with mock.patch.object(run_kvv, "download", side_effect=fake_download(files)) as dl:
            root = run_kvv.fetch_kvv(REF, cache)
            self.assertEqual(root, cache / REF)
            self.assertEqual((root / "data" / "q.parquet").read_bytes(), b"parquet bytes")
            self.assertTrue((root / run_kvv.COMPLETE_MARKER).is_file())
            self.assertEqual(dl.call_count, 2)
            run_kvv.fetch_kvv(REF, cache)
            self.assertEqual(dl.call_count, 2, "a complete checkout is reused")

    def test_lfs_content_must_match_its_pointer(self):
        files = self.urls(b"parquet bytes", b"something else")
        cache = self.tmp / "cache"
        with mock.patch.object(run_kvv, "download", side_effect=fake_download(files)):
            with self.assertRaises(SystemExit) as cm:
                run_kvv.fetch_kvv(REF, cache)
        self.assertIn("does not match its pointer", str(cm.exception))
        self.assertFalse((cache / REF / run_kvv.COMPLETE_MARKER).exists())


class JunitTests(TmpDirTest):
    def test_counts_failures_errors_and_skips(self):
        (self.kvv / "params-report.xml").write_text(JUNIT)
        result, problems = run_kvv.collect("verify_parameters", self.kvv, self.out, [])
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["failures"], [
            "tests.test_params::test_bad", "tests.test_params::test_err",
        ])
        self.assertEqual(problems, ["2 test(s) failed"])
        self.assertTrue((self.out / "params-report.xml").is_file())

    def test_missing_report_is_a_problem(self):
        result, problems = run_kvv.collect("verify_parameters", self.kvv, self.out, [])
        self.assertEqual(result, {})
        self.assertEqual(problems, ["no JUnit report params-report.xml"])


class InspectTests(TmpDirTest):
    def write(self, header, name="2026-11-18T00-00-00_ocrbench.eval"):
        (self.out / "inspect").mkdir(exist_ok=True)
        write_eval_log(self.out / "inspect" / name, header)

    def collect(self):
        return run_kvv.collect("ocrbench_thinking_effort:max", self.kvv, self.out, [])

    def test_complete_log(self):
        self.write(inspect_header())
        result, problems = self.collect()
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["stderr"], 0.05)
        self.assertEqual(problems, [])

    def test_errored_samples_are_a_problem(self):
        self.write(inspect_header(status="error", completed=7))
        result, problems = self.collect()
        self.assertEqual(result["score"], 0.8)
        self.assertEqual(problems, ["inspect status 'error'", "7/10 samples completed"])

    def test_log_without_header_is_incomplete(self):
        self.write(None)
        result, problems = self.collect()
        self.assertIsNone(result["score"])
        self.assertIn("inspect status 'incomplete'", problems)
        self.assertIn("no accuracy in inspect log", problems)

    def test_exactly_one_log_is_required(self):
        self.write(inspect_header(), "a.eval")
        self.write(inspect_header(), "b.eval")
        result, problems = self.collect()
        self.assertEqual(result, {})
        self.assertIn("found 2", problems[0])


class BeamTests(TmpDirTest):
    ANSWERS = [answer("c1", "summarization", 0), answer("c1", "abstention", 0),
               answer("c2", "summarization", 1)]

    def test_generate_counts_answers_against_loaded_questions(self):
        rows = self.ANSWERS[:2] + [answer("c2", "summarization", 1, error="Timeout()")]
        write_jsonl(self.kvv / "beam" / "answers.jsonl", rows)
        lines = ["Installed 80 packages", "loaded 4 questions", "answered 3/4"]
        result, problems = run_kvv.collect("beam_generate", self.kvv, self.out, lines)
        self.assertEqual((result["answers"], result["errors"], result["questions"]), (3, 1, 4))
        self.assertEqual(problems, [
            "1 question(s) failed and scored as empty answers", "3/4 questions answered",
        ])
        self.assertTrue((self.out / "answers.jsonl").is_file())

    def test_generate_without_loaded_line_is_a_problem(self):
        write_jsonl(self.kvv / "beam" / "answers.jsonl", self.ANSWERS)
        _, problems = run_kvv.collect("beam_generate", self.kvv, self.out, [])
        self.assertEqual(problems, ["3/None questions answered"])

    def test_judge_scores_and_per_type_means(self):
        write_jsonl(self.kvv / "beam" / "answers.jsonl", self.ANSWERS)
        write_jsonl(self.kvv / "beam" / "scores.jsonl", [
            score("c1", "summarization", 0, 1.0), score("c1", "abstention", 0, 0.0),
            score("c2", "summarization", 1, 0.5),
        ])
        result, problems = run_kvv.collect("beam_judge", self.kvv, self.out, [])
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertAlmostEqual(result["stderr"], 0.5 / 3 ** 0.5)
        self.assertEqual(result["per_type"], {"abstention": 0.0, "summarization": 0.75})
        self.assertEqual(problems, [])

    def test_judge_flags_unjudged_answers_and_gave_up_calls(self):
        write_jsonl(self.kvv / "beam" / "answers.jsonl", self.ANSWERS)
        write_jsonl(self.kvv / "beam" / "scores.jsonl", [
            score("c1", "summarization", 0, 1.0), score("c1", "abstention", 0, 0.0),
        ])
        lines = [
            "  judge attempt 1 failed: APITimeoutError(); retry in 1s",
            "  judge attempt 8 failed: APITimeoutError(); retry in 60s",
            "  judge attempt 18 failed: not a give-up at the default retries",
        ]
        result, problems = run_kvv.collect("beam_judge", self.kvv, self.out, lines)
        self.assertEqual((result["judged"], result["answers"]), (2, 3))
        self.assertEqual(result["judge_calls_gave_up"], 1)
        self.assertEqual(problems, [
            "2/3 answers judged", "1 judge call(s) gave up and scored 0",
        ])


class OutputTests(TmpDirTest):
    def test_lm_eval_format_is_what_ingest_reads(self):
        doc = run_kvv.to_lm_eval_format("org/model", "kvv_beam_judge", 0.39, None)
        self.assertEqual(doc["results"]["kvv_beam_judge"]["acc,none"], 0.39)
        self.assertEqual(doc["results"]["kvv_beam_judge"]["acc_stderr,none"], 0.0)
        self.assertEqual(doc["config"]["model_args"], "model=org/model")

    def test_clears_resumable_beam_outputs(self):
        for rel in ["beam/answers.jsonl", "beam/scores.jsonl", "logs/kimi_retry.log",
                    "params-report.xml"]:
            (self.kvv / rel).parent.mkdir(exist_ok=True)
            (self.kvv / rel).write_text("stale")
        run_kvv.clear_job_outputs("beam_judge", self.kvv)
        self.assertFalse((self.kvv / "beam" / "answers.jsonl").exists())
        self.assertFalse((self.kvv / "beam" / "scores.jsonl").exists())
        self.assertFalse((self.kvv / "logs" / "kimi_retry.log").exists())
        self.assertTrue((self.kvv / "params-report.xml").exists())

    def test_job_env_points_kvv_and_judge_at_the_local_server(self):
        with mock.patch.dict("os.environ", {"VIRTUAL_ENV": "/x/.venv"}):
            env = run_kvv.job_env("org/model", "http://localhost:8000/", "true",
                                  self.out / "inspect", judge=True)
        self.assertNotIn("VIRTUAL_ENV", env)
        self.assertEqual(env["KIMI_BASE_URL"], "http://localhost:8000/v1")
        self.assertEqual(env["JUDGE_BASE_URL"], "http://localhost:8000/v1")
        self.assertEqual(env["MODEL_NAME"], "org/model")
        self.assertEqual(env["INSPECT_LOG_DIR"], str(self.out / "inspect"))

    def test_summary_rows(self):
        path = self.out / "kvv-summary.md"
        run_kvv.append_summary_row(path, "AIME_thinking", {"score": 0.5}, 120, [], 0)
        run_kvv.append_summary_row(path, "beam_judge", {}, 60, ["no beam/scores.jsonl"], 1)
        rows = path.read_text().splitlines()
        self.assertEqual(rows[2], "| AIME_thinking | 0.5000 | 2.0 min | ok |")
        self.assertEqual(rows[3], "| beam_judge | - | 1.0 min | exit 1; no beam/scores.jsonl |")


if __name__ == "__main__":
    unittest.main()
