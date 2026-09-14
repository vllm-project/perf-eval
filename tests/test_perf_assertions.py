"""CPU-only synthetic fixtures; these are NOT GPU performance measurements."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from check_perf_assertions import check_results, validate_spec
from aggregate_perf import aggregate_results


def spec(**changes):
    value = {
        "name": "1k-in-256-out-conc-8", "model_id": "synthetic-model",
        "backend": "openai", "num_prompts": 10, "max_concurrency": 8,
        "repetitions": 1,
        "assertions": {"mean_tpot_ms": {"max": 2.5}, "output_throughput": {"min": 100}},
    }
    value.update(changes)
    return value


def result(**changes):
    value = {
        "model_id": "synthetic-model", "backend": "openai", "num_prompts": 10,
        "max_concurrency": 8, "completed": 10, "failed": 0,
        "mean_tpot_ms": 2.5, "output_throughput": 100, "request_rate": "inf",
    }
    value.update(changes)
    return value


def paths(config):
    stem = f"bench-{config['name']}"
    if config["repetitions"] == 1:
        return [f"{stem}.json"]
    return [f"{stem}-run-{i}.json" for i in range(1, config["repetitions"] + 1)]


class AssertionsTest(unittest.TestCase):
    def check(self, values=None, config=None):
        config = config or spec()
        return check_results(config, values or [result()], paths(config))

    def test_inclusive_bounds(self):
        report = self.check()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checks"][0]["actual"], 2.5)

    def test_upper_and_lower_failure(self):
        for values in ({"mean_tpot_ms": 2.50001}, {"output_throughput": 99.99}):
            with self.subTest(values=values):
                self.assertEqual(self.check([result(**values)])["status"], "fail")

    def test_two_sided_bound(self):
        config = spec(assertions={"mean_tpot_ms": {"min": 2, "max": 3}})
        self.assertEqual(self.check(config=config)["status"], "pass")
        for value in (1, 4):
            self.assertEqual(self.check([result(mean_tpot_ms=value)], config)["status"], "fail")

    def test_invalid_specs(self):
        cases = [
            None, {}, spec(repetitions=2), spec(repetitions=True), spec(repetitions=0),
            spec(max_concurrency=True), spec(num_prompts=0), spec(model_id=""),
            spec(name="wrong-conc-16"), spec(name="../escape-conc-8"),
            spec(assertions={}), spec(assertions={"mean_tpot_ms": {}}),
            spec(assertions={"mean_tpot_ms": {"maximum": 3}}),
            spec(assertions={"mean_tpot_ms": {"min": 4, "max": 3}}),
            spec(assertions={"completed": {"min": 1}}),
        ]
        cases.append({**spec(), "unknown": 1})
        for value in (float("nan"), float("inf"), -float("inf"), True, "2.5", None):
            cases.append(spec(assertions={"mean_tpot_ms": {"max": value}}))
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_spec(config)

    def test_invalid_or_missing_metrics(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "2.5", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check([result(mean_tpot_ms=value)])
        value = result()
        del value["mean_tpot_ms"]
        with self.assertRaises(ValueError):
            self.check([value])

    def test_all_identity_fields_checked(self):
        for changes in (
            {"max_concurrency": 16}, {"max_concurrency": 8.0},
            {"model_id": "other"}, {"backend": "openai-chat"}, {"num_prompts": 11},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check([result(**changes)])

    def test_request_health(self):
        for changes in (
            {"completed": 9}, {"completed": True}, {"completed": 10.0},
            {"failed": 1}, {"failed": None}, {"failed": True},
            {"errored": 1}, {"successful_requests": 9},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check([result(**changes)])
        value = result()
        del value["completed"]
        with self.assertRaises(ValueError):
            self.check([value])

    def test_completion_alias_and_optional_failed(self):
        value = result()
        value["successful_requests"] = value.pop("completed")
        del value["failed"]
        del value["num_prompts"]
        self.assertEqual(self.check([value])["status"], "pass")

    def test_median_not_each_run(self):
        config = spec(repetitions=3)
        values = [result(mean_tpot_ms=v) for v in (2, 9, 2.5)]
        report = self.check(values, config)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["aggregation"], "median")
        self.assertEqual(report["checks"][0]["actual"], 2.5)

    def test_bad_run_cannot_hide_in_passing_median(self):
        for changes in ({"failed": 1}, {"mean_tpot_ms": float("nan")}, {"completed": 9}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check([result(), result(**changes), result()], spec(repetitions=3))

    def test_aggregate_input_rejected(self):
        value = aggregate_results([result(), result(), result()], ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "individual raw"):
            self.check([value])

    def test_repetition_count_and_filenames(self):
        config = spec(repetitions=3)
        values = [result(), result(), result()]
        for filenames in (paths(config)[:2], [paths(config)[0]] * 3, ["other.json"] * 3):
            with self.subTest(paths=filenames), self.assertRaises(ValueError):
                check_results(config, values, filenames)
        with self.assertRaises(ValueError):
            check_results(config, values[:2], paths(config)[:2])

    def test_schema_mismatch_rejected(self):
        values = [result(), result(), result(extra_metric=1)]
        with self.assertRaisesRegex(ValueError, "different schema"):
            self.check(values, spec(repetitions=3))

    def test_other_concurrency_uses_own_spec(self):
        config = spec(name="1k-in-256-out-conc-16", max_concurrency=16,
                      assertions={"mean_tpot_ms": {"max": 5}})
        self.assertEqual(self.check([result(max_concurrency=16, mean_tpot_ms=4)], config)["status"], "pass")
        with self.assertRaises(ValueError):
            self.check([result(max_concurrency=16, mean_tpot_ms=4)])

    def test_inputs_not_mutated(self):
        config, values = spec(), [result()]
        before = copy.deepcopy((config, values))
        self.check(values, config)
        self.assertEqual((config, values), before)

    def test_cli_exit_codes_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "spec.json"
            config_path.write_text(json.dumps(spec()), encoding="utf-8")
            result_path = root / paths(spec())[0]
            for value, exit_code, status in (
                (result(), 0, "pass"), (result(mean_tpot_ms=4), 1, "fail"),
                (result(failed=1), 2, "error"),
                (result(mean_tpot_ms=float("nan")), 2, "error"),
            ):
                result_path.write_text(json.dumps(value), encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "lib/check_perf_assertions.py"),
                     "--spec", str(config_path), str(result_path)], capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, exit_code, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], status)
            for text in ('{"completed":10,"completed":0}', '{broken', '[]'):
                result_path.write_text(text, encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(ROOT / "lib/check_perf_assertions.py"),
                     "--spec", str(config_path), str(result_path)], capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["status"], "error")


if __name__ == "__main__":
    unittest.main()
