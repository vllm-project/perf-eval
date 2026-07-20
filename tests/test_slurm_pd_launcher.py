import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lib import slurm_pd_launcher as launcher


def serving_config() -> dict:
    common_env = {
        "NCCL_SOCKET_IFNAME": "enP22p3s0f0np0",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
    }
    return {
        "version": 1,
        "mode": "pd_disagg",
        "launcher": "slurm",
        "model": "/model",
        "image": "vllm/vllm-openai:v0.18.1-cu130",
        "total_nodes": 3,
        "total_gpus": 12,
        "gpus_per_node": 4,
        "common_env": common_env,
        "common_argv": ["--trust-remote-code", "--enforce-eager"],
        "slurm": {
            "partition": "batch",
            "time_limit": "03:00:00",
            "grace_period_s": 0,
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
                        "source_env": "",
                        "target": "/dev/infiniband",
                        "read_only": False,
                    },
                    {
                        "source": "/home/${USER}/.cache/flashinfer",
                        "source_env": "FLASHINFER_CACHE_PATH",
                        "target": "/root/.cache/flashinfer",
                        "read_only": False,
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
                "tensor_parallel_size": 1,
                "local_dp_size": 4,
                "dp_size": 4,
                "base_port": 8000,
                "kv_role": "kv_producer",
                "serve_argv": ["--max-num-seqs", "7"],
                "env": common_env,
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
                "local_dp_size": 1,
                "dp_size": 1,
                "base_port": 8000,
                "kv_role": "kv_consumer",
                "serve_argv": ["--max-num-seqs", "256"],
                "env": common_env,
                "health_check": {
                    "path": "/health",
                    "timeout_s": 1200,
                    "poll_interval_s": 10,
                },
            },
        ],
        "router": {
            "repo_path": "${HOME}/Kimi-PD/vllm-router",
            "revision": "v0.1.12",
            "command_argv": ["target/release/vllm-router"],
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
    }


def option_value(argv: list[str], option: str) -> str:
    index = argv.index(option)
    return argv[index + 1]


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = serving_config()
        self.nodes = [f"c{index:02d}" for index in range(2, 5)]
        self.env = {
            "HOME": "/home/inf-kevin",
            "USER": "inf-kevin",
            "KIMI_K2_5_MODEL_PATH": "/raid/models/kimi",
        }
        self.plan = launcher.build_launch_plan(
            self.config,
            self.nodes,
            environ=self.env,
            controller_host="slogin-01",
        )

    def test_assigns_one_dep4_prefill_then_two_tep4_decoders(self):
        instances = self.plan["role_instances"]
        self.assertEqual(len(instances), 3)
        self.assertEqual(
            [instance["nodes"] for instance in instances],
            [["c02"], ["c03"], ["c04"]],
        )
        prefill, decode0, decode1 = instances
        self.assertEqual(
            (prefill["tensor_parallel_size"], prefill["dp_size"]),
            (1, 4),
        )
        for decode in (decode0, decode1):
            self.assertEqual(decode["role"], "decode")
            self.assertEqual(
                (decode["tensor_parallel_size"], decode["dp_size"]),
                (4, 1),
            )
            self.assertEqual(decode["node_commands"][0]["global_ranks"], [0])

    def test_dep_prefill_gets_dp_flags_while_tep_decoders_do_not(self):
        prefill, *decoders = self.plan["role_instances"]
        prefill_argv = prefill["node_commands"][0]["argv"]
        self.assertEqual(option_value(prefill_argv, "--tensor-parallel-size"), "1")
        self.assertEqual(option_value(prefill_argv, "--data-parallel-size"), "4")
        self.assertEqual(
            option_value(prefill_argv, "--data-parallel-size-local"), "4"
        )
        self.assertEqual(
            option_value(prefill_argv, "--data-parallel-rpc-port"), "29500"
        )
        self.assertEqual(prefill_argv.count("--data-parallel-hybrid-lb"), 1)

        for decode in decoders:
            argv = decode["node_commands"][0]["argv"]
            self.assertEqual(option_value(argv, "--tensor-parallel-size"), "4")
            self.assertNotIn("--data-parallel-size", argv)
            self.assertNotIn("--data-parallel-size-local", argv)
            self.assertNotIn("--data-parallel-rpc-port", argv)
            self.assertNotIn("--data-parallel-hybrid-lb", argv)

    def test_role_specific_nixl_configs_and_per_node_hosts(self):
        for instance in self.plan["role_instances"]:
            expected_role = (
                "kv_producer" if instance["role"] == "prefill" else "kv_consumer"
            )
            for command in instance["node_commands"]:
                transfer = json.loads(
                    option_value(command["argv"], "--kv-transfer-config")
                )
                self.assertEqual(transfer["kv_connector"], "NixlConnector")
                self.assertEqual(transfer["kv_role"], expected_role)
                self.assertEqual(
                    command["env"]["VLLM_NIXL_SIDE_CHANNEL_HOST"],
                    command["node"],
                )

    def test_router_registers_endpoint_level_prefill_and_decode_replicas(self):
        router = self.plan["router"]
        self.assertEqual(router["prefill_urls"], ["http://c02:8000"])
        self.assertEqual(
            router["decode_urls"], ["http://c03:8000", "http://c04:8000"]
        )
        self.assertEqual(len(router["workers"]), 3)
        self.assertTrue(
            all(worker["node_local_ranks"] == [0] for worker in router["workers"])
        )
        self.assertEqual(router["argv"].count("--prefill"), 1)
        self.assertEqual(router["argv"].count("--decode"), 2)
        self.assertEqual(
            option_value(router["argv"], "--intra-node-data-parallel-size"),
            "1",
        )
        self.assertEqual(option_value(router["argv"], "--prometheus-port"), "31001")
        self.assertEqual(router["cwd"], "/home/inf-kevin/Kimi-PD/vllm-router")
        self.assertEqual(router["local_url"], "http://127.0.0.1:31000")
        self.assertEqual(router["allocation_url"], "http://slogin-01:31000")
        self.assertEqual(router["nofile_limit"], 65536)

    def test_pyxis_is_writable_and_preserves_required_mounts(self):
        self.assertEqual(
            self.plan["container"]["image"],
            "docker://vllm/vllm-openai:v0.18.1-cu130",
        )
        mounts = self.plan["container"]["mounts"]
        self.assertIn(
            {"source": "/dev/infiniband", "target": "/dev/infiniband", "read_only": False},
            mounts,
        )
        self.assertIn(
            {
                "source": "/home/inf-kevin/.cache/flashinfer",
                "target": "/root/.cache/flashinfer",
                "read_only": False,
            },
            mounts,
        )
        self.assertEqual(mounts[0]["source"], "/raid/models/kimi")
        for instance in self.plan["role_instances"]:
            srun = instance["srun_argv"]
            self.assertIn("--container-writable", srun)
            self.assertIn("--label", srun)
            self.assertIn(
                "--container-image=docker://vllm/vllm-openai:v0.18.1-cu130",
                srun,
            )
            mount_arg = next(arg for arg in srun if arg.startswith("--container-mounts="))
            self.assertIn("/dev/infiniband:/dev/infiniband", mount_arg)
            self.assertIn(
                "/home/inf-kevin/.cache/flashinfer:/root/.cache/flashinfer",
                mount_arg,
            )

    def test_salloc_requests_one_exact_topology(self):
        argv = launcher.build_salloc_argv(
            self.config, "results/pd.json", script_path="/repo/lib/slurm_pd_launcher.py"
        )
        self.assertEqual(argv[0], "salloc")
        self.assertIn("--nodes=3", argv)
        self.assertIn("--ntasks=3", argv)
        self.assertIn("--ntasks-per-node=1", argv)
        self.assertIn("--gpus-per-node=4", argv)
        self.assertIn("--partition=batch", argv)
        self.assertIn("--time=03:00:00", argv)
        self.assertEqual(argv[-2:], ["--inside-allocation", "supervise"])

    def test_client_runs_in_same_allocation_and_pyxis_image(self):
        state = {
            "job_id": "1234",
            "nodes": self.nodes,
            "allocation_router_url": "http://slogin-01:31000",
        }
        argv = launcher.build_client_srun_argv(
            self.config,
            state,
            [
                "vllm", "bench", "serve",
                "--base-url", "{router_url}",
                "--host", "{router_host}",
                "--port", "{router_port}",
                "--model", "{model}",
            ],
            self.env,
        )
        self.assertIn("--jobid=1234", argv)
        self.assertIn("--nodelist=c02", argv)
        self.assertIn("--container-writable", argv)
        self.assertIn(f"--container-workdir={Path.cwd().resolve()}", argv)
        mount_arg = next(arg for arg in argv if arg.startswith("--container-mounts="))
        self.assertIn(
            f"{Path.cwd().resolve()}:{Path.cwd().resolve()}", mount_arg
        )
        self.assertIn("OPENAI_BASE_URL=http://slogin-01:31000", argv)
        self.assertIn("http://slogin-01:31000", argv)
        self.assertIn("slogin-01", argv)
        self.assertIn("31000", argv)
        self.assertIn("/model", argv)


class ValidationAndLifecycleTests(unittest.TestCase):
    def test_router_raises_nofile_soft_limit_before_start(self):
        supervisor = object.__new__(launcher.Supervisor)
        supervisor.plan = {
            "router": {
                "argv": ["vllm-router"],
                "cwd": "/tmp",
                "env": {},
                "nofile_limit": 65536,
                "prefill_urls": ["http://c02:8000"],
                "decode_urls": ["http://c03:8000", "http://c04:8000"],
            }
        }
        supervisor.children = []
        supervisor._write_state = mock.Mock()
        supervisor.popen = mock.Mock(return_value=mock.Mock())

        with mock.patch.object(
            launcher.resource, "getrlimit", return_value=(1024, 1048576)
        ), mock.patch.object(launcher.resource, "setrlimit") as setrlimit:
            supervisor._start_router()

        setrlimit.assert_called_once_with(
            launcher.resource.RLIMIT_NOFILE, (65536, 1048576)
        )
        supervisor.popen.assert_called_once()

    def test_wait_ready_fails_when_supervisor_dies_before_state(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "missing.json")
            with self.assertRaisesRegex(
                launcher.LaunchError, "exited before creating readiness state"
            ):
                launcher.wait_ready(
                    state_file,
                    60,
                    supervisor_pid=1234,
                    pid_alive=lambda _pid: False,
                )

    def test_supervise_refuses_ambient_allocation(self):
        config = serving_config()
        with tempfile.TemporaryDirectory() as directory:
            state_file = str(Path(directory) / "state.json")
            stderr = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {
                    "HOME": "/home/tester",
                    "USER": "tester",
                    "SLURM_JOB_ID": "9999",
                },
                clear=False,
            ):
                with contextlib.redirect_stderr(stderr):
                    status = launcher.main(
                        [
                            "--config", json.dumps(config),
                            "--state-file", state_file,
                            "supervise",
                        ]
                    )
            self.assertEqual(status, 2)
            self.assertIn("refusing to adopt", stderr.getvalue())
            state = json.loads(Path(state_file).read_text())
            self.assertIn("refusing to adopt", state["error"])
            self.assertEqual(state["job_id"], "")
            self.assertFalse(state["owns_allocation"])
            self.assertEqual(state["grace_period_s"], 0)

    def test_stop_refuses_unowned_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "job_id": "9999",
                        "owns_allocation": False,
                        "controller_host": os.uname().nodename,
                        "children": [],
                    }
                )
            )
            with self.assertRaisesRegex(
                launcher.LaunchError, "not owned by this launcher"
            ):
                launcher.stop_from_state(str(state_file))

    def test_stop_signals_owned_local_controller_before_scancel(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "job_id": "9999",
                        "owns_allocation": True,
                        "controller_host": "login-01",
                        "controller_pid": 4242,
                        "grace_period_s": 5,
                    }
                )
            )
            pid_alive = mock.Mock(side_effect=[True, True, False])
            send_signal = mock.Mock()
            run_command = mock.Mock()

            with mock.patch.object(
                launcher.socket, "gethostname", return_value="login-01"
            ):
                launcher.stop_from_state(
                    str(state_file),
                    pid_alive=pid_alive,
                    send_signal=send_signal,
                    run_command=run_command,
                    sleep=lambda _seconds: None,
                )

            send_signal.assert_called_once_with(4242, launcher.signal.SIGTERM)
            run_command.assert_not_called()

    def test_stop_falls_back_to_scancel_after_controller_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "job_id": "9999",
                        "owns_allocation": True,
                        "controller_host": "login-01",
                        "controller_pid": 4242,
                        "grace_period_s": 0,
                    }
                )
            )
            current_time = [0.0]

            def advance(seconds):
                current_time[0] += seconds

            send_signal = mock.Mock()
            run_command = mock.Mock(
                return_value=launcher.subprocess.CompletedProcess(
                    ["scancel", "9999"], 0
                )
            )
            with mock.patch.object(
                launcher.socket, "gethostname", return_value="login-01"
            ):
                launcher.stop_from_state(
                    str(state_file),
                    pid_alive=lambda _pid: True,
                    send_signal=send_signal,
                    run_command=run_command,
                    now=lambda: current_time[0],
                    sleep=advance,
                )

            send_signal.assert_called_once_with(4242, launcher.signal.SIGTERM)
            run_command.assert_called_once_with(["scancel", "9999"], check=False)
            self.assertEqual(current_time[0], launcher.STOP_EXIT_SLACK_S)

    def test_stop_uses_scancel_for_remote_controller_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            state_file = Path(directory) / "state.json"
            state_file.write_text(
                json.dumps(
                    {
                        "job_id": "9999",
                        "owns_allocation": True,
                        "controller_host": "another-login",
                        "controller_pid": 4242,
                        "grace_period_s": 120,
                    }
                )
            )
            run_command = mock.Mock(
                return_value=launcher.subprocess.CompletedProcess(
                    ["scancel", "9999"], 1
                )
            )

            with self.assertRaisesRegex(launcher.LaunchError, "scancel failed"):
                launcher.stop_from_state(
                    str(state_file),
                    send_signal=mock.Mock(),
                    run_command=run_command,
                )

    def test_stop_wait_is_bounded_even_for_large_configured_grace(self):
        self.assertEqual(
            launcher._stop_wait_s({"grace_period_s": 10_000}),
            launcher.MAX_STOP_WAIT_S,
        )

    def test_requires_canonical_role_specific_kv_roles(self):
        config = serving_config()
        config["roles"][0]["kv_role"] = "kv_both"
        with self.assertRaisesRegex(launcher.ConfigError, "kv_producer"):
            launcher.validate_config(config)

    def test_requires_infiniband_and_persistent_flashinfer_mounts(self):
        for target, message in (
            ("/dev/infiniband", "NIXL"),
            ("/root/.cache/flashinfer", "FlashInfer"),
        ):
            with self.subTest(target=target):
                config = serving_config()
                mounts = config["slurm"]["container"]["mounts"]
                config["slurm"]["container"]["mounts"] = [
                    mount for mount in mounts if mount["target"] != target
                ]
                with self.assertRaisesRegex(launcher.ConfigError, message):
                    launcher.validate_config(config)

    def test_controller_only_validates_shared_flashinfer_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "flashinfer"
            mounts = [
                {
                    "source": "/raid/model-only-on-compute-nodes",
                    "target": "/model",
                    "read_only": True,
                },
                {
                    "source": "/dev/infiniband-only-on-compute-nodes",
                    "target": "/dev/infiniband",
                    "read_only": False,
                },
                {
                    "source": str(cache),
                    "target": "/root/.cache/flashinfer",
                    "read_only": False,
                },
            ]
            launcher.ensure_runtime_mounts(mounts)
            self.assertTrue(cache.is_dir())

    def test_wait_for_checks_fails_as_soon_as_a_child_exits(self):
        calls = []

        def failed_guard():
            calls.append("guard")
            raise launcher.LaunchError("decode[0] exited unexpectedly")

        with self.assertRaisesRegex(launcher.LaunchError, r"decode\[0\]"):
            launcher.wait_for_checks(
                [{"url": "http://c08:8000/health", "timeout_s": 10, "poll_interval_s": 1}],
                process_guard=failed_guard,
                healthy=lambda _url: False,
                now=lambda: 0,
                sleep=lambda _seconds: None,
            )
        self.assertEqual(calls, ["guard"])

    def test_supervisor_guard_detects_any_dead_srun_during_runtime(self):
        alive = mock.Mock(pid=100, poll=mock.Mock(return_value=None))
        dead = mock.Mock(pid=101, poll=mock.Mock(return_value=9))
        supervisor = object.__new__(launcher.Supervisor)
        supervisor.children = [
            launcher.ChildProcess("prefill[0]", alive),
            launcher.ChildProcess("decode[0]", dead),
        ]
        with self.assertRaisesRegex(launcher.LaunchError, r"decode\[0\].*status 9"):
            supervisor._guard_children()

    def test_dry_run_needs_no_slurm_and_redacts_secret_env(self):
        config = serving_config()
        config["roles"][0]["env"]["HF_TOKEN"] = "very-secret"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        output = io.StringIO()
        try:
            with mock.patch.dict(
                os.environ,
                {"HOME": "/home/tester", "USER": "tester"},
                clear=False,
            ):
                with contextlib.redirect_stdout(output):
                    status = launcher.main(
                        [
                            "--config",
                            config_path,
                            "--nodes",
                            ",".join(f"c{i:02d}" for i in range(2, 5)),
                            "dry-run",
                        ]
                    )
            self.assertEqual(status, 0)
            self.assertNotIn("very-secret", output.getvalue())
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["allocation"]["total_nodes"], 3)
            self.assertEqual(plan["role_instances"][-1]["nodes"], ["c04"])
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    unittest.main()
