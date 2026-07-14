import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / ".buildkite" / "generate_pipeline.py"
SPEC = importlib.util.spec_from_file_location("generate_pipeline", GENERATOR_PATH)
generate_pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generate_pipeline)


class SlurmPipelineTests(unittest.TestCase):
    def setUp(self):
        self.path = "workloads/kimi_k2_5_gb300_pd.yaml"
        with (REPO_ROOT / self.path).open() as f:
            self.workload = yaml.safe_load(f)
        self.profiles = generate_pipeline.load_profiles()

    def test_pd_workload_uses_slurm_queue_without_kubernetes_plugin(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            step = generate_pipeline.make_step(
                self.path, self.workload, self.profiles,
            )

        self.assertEqual(step["agents"]["queue"], "gb300-slurm")
        self.assertEqual(step["timeout_in_minutes"], 240)
        self.assertNotIn("plugins", step)
        self.assertEqual(step["concurrency"], 1)
        self.assertEqual(
            step["concurrency_group"], "perf-eval/gb300-slurm",
        )
        self.assertEqual(step["concurrency_method"], "eager")
        self.assertEqual(step["env"]["BENCH_ONLY"], "1")

    def test_queue_can_be_overridden_for_agent_bringup(self):
        with mock.patch.dict(
            os.environ, {"GB300_QUEUE": "gb300-slurm-canary"}, clear=True,
        ):
            step = generate_pipeline.make_step(
                self.path, self.workload, self.profiles,
            )
        self.assertEqual(step["agents"]["queue"], "gb300-slurm-canary")
        self.assertEqual(
            step["concurrency_group"], "perf-eval/gb300-slurm-canary",
        )

    def test_non_slurm_workload_has_no_serialization_guard(self):
        workload = {
            "name": "standalone-h200",
            "gpu": "H200",
            "bench_only": True,
            "vllm": {"model": "example/model"},
        }
        with mock.patch.dict(os.environ, {}, clear=True):
            step = generate_pipeline.make_step(
                "workloads/standalone_h200.yaml", workload, self.profiles,
            )

        self.assertNotIn("concurrency", step)
        self.assertNotIn("concurrency_group", step)
        self.assertNotIn("concurrency_method", step)

    def test_pd_workload_is_not_selected_by_default_nightly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            selected = generate_pipeline.select_workloads(
                [{"path": self.path, "data": self.workload}],
            )
        self.assertEqual(selected, [])

    def test_setup_fallback_supports_externally_managed_python(self):
        command = generate_pipeline.setup_command("pyyaml")

        self.assertIn(
            "if ! python3 -m pip --version >/dev/null 2>&1; then",
            command,
        )
        self.assertIn(
            "python3 - --user --break-system-packages",
            command,
        )
        self.assertIn(
            "PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user",
            command,
        )


if __name__ == "__main__":
    unittest.main()
