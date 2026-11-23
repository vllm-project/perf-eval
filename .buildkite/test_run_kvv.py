#!/usr/bin/env python3
"""Stdlib-only tests for lib/run_kvv.py scoring and the kvv workload block.

Run with ``python3 .buildkite/test_run_kvv.py``. KVV's outputs are built here
in the shapes KVV writes them; no checkout, GPU or network is needed.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lib" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_kvv, parse_workload = load("run_kvv"), load("parse_workload")


class CollectTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root, self.out = Path(tmp.name) / "kvv", Path(tmp.name) / "out"
        (self.root / "beam").mkdir(parents=True)
        (self.out / "inspect").mkdir(parents=True)

    def test_verify_scores_passed_over_ran(self):
        (self.root / "params-report.xml").write_text(
            '<testsuites><testsuite><testcase classname="t" name="a"/>'
            '<testcase classname="t" name="b"><failure/></testcase>'
            '<testcase classname="t" name="c"><skipped/></testcase></testsuite></testsuites>')
        result, problems = run_kvv.collect("verify_parameters", self.root, self.out, [])
        self.assertEqual((result["score"], result["skipped"], result["failures"]), (0.5, 1, ["t::b"]))
        self.assertEqual(problems, ["1 test(s) failed"])
        self.assertTrue((self.out / "params-report.xml").is_file())

    def write_eval(self, completed):
        with zipfile.ZipFile(self.out / "inspect" / "x.eval", "w") as zf:
            zf.writestr("header.json", json.dumps({"status": "success", "results": {
                "total_samples": 10, "completed_samples": completed,
                "scores": [{"metrics": {"accuracy": {"value": 0.8}, "stderr": {"value": 0.05}}}]}}))

    def test_inspect_score_and_incomplete_samples(self):
        self.write_eval(10)
        result, problems = run_kvv.collect("ocrbench_thinking_effort:max", self.root, self.out, [])
        self.assertEqual((result["score"], result["stderr"], problems), (0.8, 0.05, []))
        self.write_eval(9)
        _, problems = run_kvv.collect("ocrbench_thinking_effort:max", self.root, self.out, [])
        self.assertEqual(problems, ["9/10 samples completed"])

    def test_beam(self):
        rows = [{"chat_id": c, "question_type": t, "question_index": 0, "error": e}
                for c, t, e in [("1", "abstention", None), ("2", "abstention", None),
                                ("3", "summarization", "timeout")]]
        (self.root / "beam" / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        result, problems = run_kvv.collect("beam_generate", self.root, self.out, ["loaded 4 questions"])
        self.assertEqual((result["answers"], result["errors"]), (3, 1))
        self.assertEqual(len(problems), 2)

        (self.root / "beam" / "scores.jsonl").write_text("".join(
            json.dumps({**r, "score": s}) + "\n" for r, s in zip(rows, [1.0, 0.5, 0.0])))
        lines = ["  judge attempt 8 failed: timeout"]
        result, problems = run_kvv.collect("beam_judge", self.root, self.out, lines)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["per_type"], {"abstention": 0.75, "summarization": 0.0})
        self.assertEqual(problems, ["1 judge call(s) gave up and scored 0"])
        self.assertTrue((self.out / "scores.jsonl").is_file())


class WorkloadTest(unittest.TestCase):
    def test_kvv_jobs(self):
        ref = "7b938c38bbd89c2404a9ed5e7cb40da8079206b4"
        parse_workload.validate_kvv(
            {"ref": ref, "jobs": ["verify_k3_features", "beam_judge", "MMMU_pro_thinking_effort:max"]}, "w")
        for bad in ({"ref": ref, "jobs": ["all"]}, {"ref": ref, "jobs": ["MMMU_pro_thinking_effort"]},
                    {"ref": "main", "jobs": ["beam_generate"]}, {"ref": ref, "jobs": []}):
            with self.assertRaises(SystemExit, msg=bad):
                parse_workload.validate_kvv(bad, "w")


if __name__ == "__main__":
    unittest.main()
