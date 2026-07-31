import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest import mock

import yaml

from lib import parse_workload


REPO_ROOT = Path(__file__).resolve().parents[1]


def pd_workload() -> dict:
    return {
        "name": "kimi-k2.5-pd",
        "gpu": "H200",
        "num_gpus": 12,
        "vllm": {
            "model": "/model",
            "image": "vllm/vllm-openai:v0.18.1-cu130",
            "env": {
                "SHARED": "common",
                "VLLM_NIXL_SIDE_CHANNEL_PORT": 5600,
            },
        },
        "serving": {
            "version": 1,
            "mode": "pd_disagg",
            "launcher": "slurm",
            "common_serve_args": (
                "--trust-remote-code "
                "--attention-config "
                "'{\"use_trtllm_ragged_deepseek_prefill\":true}'"
            ),
            "slurm": {
                "partition": "batch",
                "time_limit": "03:00:00",
                "grace_period_s": 120,
                "container": {
                    "runtime": "pyxis",
                    "mounts": [
                        {
                            "source": "/raid/shared/nvidia/Kimi-K2.5-NVFP4",
                            "source_env": "KIMI_K2_5_MODEL_PATH",
                            "target": "/model",
                            "read_only": True,
                        },
                        {
                            "source": "/dev/infiniband",
                            "target": "/dev/infiniband",
                        },
                    ],
                },
            },
            "kv_transfer": {
                "connector": "NixlConnector",
                "load_failure_policy": "fail",
                "extra_config": {"num_threads": 4},
            },
            "roles": [
                {
                    "role": "prefill",
                    "count": 1,
                    "nodes_per_instance": 1,
                    "gpus_per_node": 4,
                    "base_port": 8000,
                    "kv_role": "kv_producer",
                    "serve_args": "--enforce-eager --max-num-seqs 7",
                    "env": {"SHARED": "prefill", "ROLE_ONLY": True},
                    "health_check": {
                        "path": "/health",
                        "timeout_s": 1200,
                        "poll_interval_s": 10,
                    },
                },
                {
                    "role": "decode",
                    "count": 2,
                    "nodes_per_instance": 1,
                    "gpus_per_node": 4,
                    "tensor_parallel_size": 4,
                    "base_port": 8000,
                    "kv_role": "kv_consumer",
                    "serve_args": (
                        "--max-num-seqs 256"
                    ),
                },
            ],
            "router": {
                "repo_path": "${HOME}/Kimi-PD/vllm-router",
                "revision": "v0.1.12",
                "command": ["target/release/vllm-router"],
                "policy": "round_robin",
                "listen_host": "0.0.0.0",
                "client_host": "127.0.0.1",
                "port": 31000,
                "metrics_port": 31001,
                "nofile_limit": 65536,
                "intra_node_data_parallel_size": 1,
                "prefill_endpoints": "all_instances",
                "decode_endpoints": "all_nodes",
                "env": {"PROTOC": "${HOME}/.local/protoc/bin/protoc"},
                "health_check": {
                    "path": "/readiness",
                    "timeout_s": 600,
                    "poll_interval_s": 5,
                },
            },
        },
        "vllm_bench": {
            "configs": [
                {
                    "name": "smoke",
                    "input_len": 32,
                    "output_len": 8,
                    "num_prompts": 1,
                    "max_concurrency": 1,
                },
            ],
        },
    }


def emitted_value(stdout: str, name: str) -> str:
    prefix = f"WORKLOAD_{name}="
    for line in stdout.splitlines():
        if line.startswith(prefix):
            values = shlex.split(line[len(prefix):])
            return values[0] if values else ""
    raise AssertionError(f"missing {prefix} in parser output")


class ServingSchemaTests(unittest.TestCase):
    def normalize(self, data: dict, image: str = "resolved:image") -> dict:
        profile = {
            "env": {"PROFILE_ENV": "profile", "SHARED": "profile"},
            "hf_home": "/profile/hf",
        }
        return parse_workload.normalize_serving(
            data["serving"], data, profile, image, "fixture.yaml",
        )

    def run_parser(self, data: dict, **environment: str) -> str:
        tests_dir = REPO_ROOT / "tests"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            dir=tests_dir,
            delete=False,
        ) as f:
            yaml.safe_dump(data, f, sort_keys=False)
            path = Path(f.name)
        try:
            output = io.StringIO()
            test_env = {"VLLM_IMAGE": "", "VLLM_COMMIT": "", **environment}
            with mock.patch.dict(os.environ, test_env, clear=False):
                with contextlib.redirect_stdout(output):
                    parse_workload.main(str(path))
            return output.getvalue()
        finally:
            path.unlink()

    def test_normalizes_mixed_per_role_tp_and_dp(self):
        normalized = self.normalize(pd_workload())

        self.assertEqual(normalized["total_nodes"], 3)
        self.assertEqual(normalized["total_gpus"], 12)
        self.assertEqual(normalized["gpus_per_node"], 4)
        self.assertEqual(normalized["common_env"]["SHARED"], "common")
        self.assertEqual(normalized["common_env"]["HF_HOME"], "/profile/hf")
        self.assertEqual(
            normalized["common_argv"],
            [
                "--trust-remote-code",
                "--attention-config",
                '{"use_trtllm_ragged_deepseek_prefill":true}',
            ],
        )

        prefill, decode = normalized["roles"]
        self.assertEqual(
            (
                prefill["role"],
                prefill["tensor_parallel_size"],
                prefill["local_dp_size"],
                prefill["dp_size"],
            ),
            ("prefill", 1, 4, 4),
        )
        self.assertEqual(
            (
                decode["role"],
                decode["tensor_parallel_size"],
                decode["local_dp_size"],
                decode["dp_size"],
            ),
            ("decode", 4, 1, 1),
        )
        self.assertEqual(prefill["env"]["SHARED"], "prefill")
        self.assertEqual(prefill["env"]["ROLE_ONLY"], "True")
        self.assertEqual(decode["env"]["SHARED"], "common")
        self.assertEqual(
            normalized["slurm"]["container"]["mounts"][0]["source_env"],
            "KIMI_K2_5_MODEL_PATH",
        )
        self.assertEqual(
            normalized["router"]["command_argv"],
            ["target/release/vllm-router"],
        )
        self.assertEqual(
            normalized["router"]["intra_node_data_parallel_size"], 1,
        )
        self.assertEqual(normalized["router"]["nofile_limit"], 65536)

    def test_pd_exports_compact_json_and_resolved_image(self):
        stdout = self.run_parser(
            pd_workload(), VLLM_IMAGE="registry.example/vllm:override",
        )

        self.assertEqual(emitted_value(stdout, "SERVING_MODE"), "pd_disagg")
        serialized = emitted_value(stdout, "SERVING_JSON")
        self.assertNotIn(": ", serialized)
        self.assertNotIn(", ", serialized)
        normalized = json.loads(serialized)
        self.assertEqual(normalized["image"], "registry.example/vllm:override")
        self.assertEqual(normalized["total_nodes"], 3)
        self.assertEqual(emitted_value(stdout, "BENCH_GPU_COUNT"), "12")
        self.assertEqual(emitted_value(stdout, "BENCH_DISAGG"), "true")
        self.assertEqual(emitted_value(stdout, "BENCH_IS_MULTINODE"), "true")

    def test_standalone_workload_emits_no_serving_exports(self):
        data = pd_workload()
        del data["serving"]
        data["num_gpus"] = 1
        data["vllm"]["serve_args"] = "--tensor-parallel-size 1"

        stdout = self.run_parser(data)

        self.assertNotIn("WORKLOAD_SERVING_", stdout)
        self.assertEqual(
            emitted_value(stdout, "SERVE_ARGS"),
            "--tensor-parallel-size 1",
        )
        self.assertEqual(emitted_value(stdout, "BENCH_TP"), "1")
        self.assertEqual(emitted_value(stdout, "BENCH_GPU_COUNT"), "1")
        self.assertEqual(emitted_value(stdout, "BENCH_DISAGG"), "false")
        self.assertEqual(emitted_value(stdout, "BENCH_IS_MULTINODE"), "false")

    def test_rejects_invalid_pd_topologies_and_fields(self):
        cases = []

        wrong_total = pd_workload()
        wrong_total["num_gpus"] = 11
        cases.append((wrong_total, "serving.roles require 12"))

        missing_decode = pd_workload()
        missing_decode["serving"]["roles"].pop()
        cases.append((missing_decode, "exactly one 'prefill' and one 'decode'"))

        duplicate_prefill = pd_workload()
        duplicate_prefill["serving"]["roles"][1]["role"] = "prefill"
        duplicate_prefill["serving"]["roles"][1]["kv_role"] = "kv_producer"
        cases.append((duplicate_prefill, "exactly one 'prefill' role"))

        zero_count = pd_workload()
        zero_count["serving"]["roles"][0]["count"] = 0
        cases.append((zero_count, "count must be a positive integer"))

        bad_launcher = pd_workload()
        bad_launcher["serving"]["launcher"] = "local"
        cases.append((bad_launcher, "launcher must be 'slurm'"))

        bad_runtime = pd_workload()
        bad_runtime["serving"]["slurm"]["container"]["runtime"] = "docker"
        cases.append((bad_runtime, "must be 'pyxis'"))

        unknown_role_field = pd_workload()
        unknown_role_field["serving"]["roles"][0]["mystery"] = True
        cases.append((unknown_role_field, "unsupported fields ['mystery']"))

        legacy_args = pd_workload()
        legacy_args["vllm"]["serve_args"] = "--max-num-seqs 1"
        cases.append((legacy_args, "vllm.serve_args must be empty"))

        launcher_arg = pd_workload()
        launcher_arg["serving"]["roles"][1]["serve_args"] += " --port 9000"
        cases.append((launcher_arg, "launcher-owned flag '--port'"))

        hybrid_arg = pd_workload()
        hybrid_arg["serving"]["roles"][1]["serve_args"] += (
            " --data-parallel-hybrid-lb"
        )
        cases.append(
            (hybrid_arg, "launcher-owned flag '--data-parallel-hybrid-lb'")
        )

        legacy_kv_role = pd_workload()
        legacy_kv_role["serving"]["roles"][0]["kv_role"] = "kv_both"
        cases.append((legacy_kv_role, "must be 'kv_producer'"))

        zero_tp = pd_workload()
        zero_tp["serving"]["roles"][0]["tensor_parallel_size"] = 0
        cases.append((zero_tp, "tensor_parallel_size must be a positive integer"))

        cross_node_tp = pd_workload()
        cross_node_tp["serving"]["roles"][0]["tensor_parallel_size"] = 8
        cases.append((cross_node_tp, "must stay within one node (4 GPUs)"))

        uneven_tp = pd_workload()
        uneven_tp["serving"]["roles"][0]["tensor_parallel_size"] = 3
        cases.append((uneven_tp, "must divide gpus_per_node (4)"))

        launcher_tp = pd_workload()
        launcher_tp["serving"]["roles"][0]["serve_args"] += " --tp 2"
        cases.append((launcher_tp, "launcher-owned flag '--tp'"))

        incompatible_router_dp = pd_workload()
        incompatible_router_dp["serving"]["router"][
            "intra_node_data_parallel_size"
        ] = 4
        cases.append(
            (
                incompatible_router_dp,
                "values greater than 1 must equal local_dp_size for every role",
            )
        )

        invalid_nofile_limit = pd_workload()
        invalid_nofile_limit["serving"]["router"]["nofile_limit"] = 0
        cases.append(
            (invalid_nofile_limit, "nofile_limit must be a positive integer")
        )

        for data, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(SystemExit) as raised:
                    self.normalize(copy.deepcopy(data))
                self.assertIn(expected, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
