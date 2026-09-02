#!/usr/bin/env python3
"""Launch schema-v1 vLLM prefill/decode serving in one Slurm allocation.

The launcher accepts the normalized JSON emitted as ``WORKLOAD_SERVING_JSON``
by ``parse_workload.py``.  It can also read JSON from ``--config`` (an inline
object, a file path, or ``-`` for stdin).  The normal login-node flow is::

    python3 lib/slurm_pd_launcher.py --config "$WORKLOAD_SERVING_JSON" \
      --state-file results/pd-serving.json supervise

When not already in an allocation, ``supervise`` re-execs itself under one
``salloc``.  It then starts one overlapping ``srun`` per role instance, waits
for every node-local API server, starts vllm-router, and remains in the
foreground.  SIGINT/SIGTERM and child failure terminate every child step.

``dry-run`` is deliberately Slurm-free and prints the complete deterministic
plan.  It is also the primary local test surface for this module.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import socket
import string
import subprocess
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


DP_RPC_PORT_BASE = 29500
DEFAULT_STATE_FILE = "/tmp/perf-eval-pd-serving.json"
FLASHINFER_CACHE_TARGET = "/root/.cache/flashinfer"
INFINIBAND_TARGET = "/dev/infiniband"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_STOP_GRACE_S = 120
MAX_STOP_WAIT_S = 300
STOP_EXIT_SLACK_S = 10
STOP_POLL_INTERVAL_S = 0.2

MANAGED_SERVE_FLAGS = {
    "--host",
    "--port",
    "--tensor-parallel-size",
    "-tp",
    "--tp",
    "--data-parallel-size",
    "-dp",
    "--dp",
    "--data-parallel-size-local",
    "--data-parallel-address",
    "--data-parallel-rpc-port",
    "--data-parallel-start-rank",
    "--data-parallel-rank",
    "--kv-transfer-config",
}
HYBRID_LB_FLAG = "--data-parallel-hybrid-lb"
SENSITIVE_ENV_MARKERS = ("TOKEN", "PASSWORD", "SECRET", "API_KEY")


class ConfigError(ValueError):
    """The normalized serving configuration is not launchable."""


class LaunchError(RuntimeError):
    """A Slurm step, health check, or router process failed."""


class StopRequested(Exception):
    """Internal control flow used to leave readiness waits on SIGINT/SIGTERM."""


@dataclass
class ChildProcess:
    label: str
    process: subprocess.Popen


def _positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{location} must be a positive integer")
    return value


def _mapping(value: object, location: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{location} must be an object")
    return value


def _argv(value: object, location: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ConfigError(f"{location} must be an array of strings")
    return list(value)


def load_config(source: str | None, environ: Mapping[str, str] | None = None) -> dict:
    """Load normalized serving JSON from inline text, a file, stdin, or env."""

    env = os.environ if environ is None else environ
    source = source or env.get("WORKLOAD_SERVING_JSON")
    if not source:
        raise ConfigError(
            "missing serving config: pass --config or set WORKLOAD_SERVING_JSON"
        )
    if source == "-":
        raw = sys.stdin.read()
    elif source.lstrip().startswith("{"):
        raw = source
    else:
        path = Path(source)
        if not path.is_file():
            raise ConfigError(f"serving config file does not exist: {source}")
        raw = path.read_text()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid serving JSON: {exc}") from exc
    return validate_config(config)


def validate_config(value: object) -> dict:
    """Validate the launcher-facing subset of the normalized schema."""

    config = _mapping(value, "serving config")
    if config.get("version") != 1 or isinstance(config.get("version"), bool):
        raise ConfigError("serving config version must be integer 1")
    if config.get("mode") != "pd_disagg":
        raise ConfigError("serving config mode must be 'pd_disagg'")
    if config.get("launcher") != "slurm":
        raise ConfigError("serving config launcher must be 'slurm'")
    for key in ("model", "image"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ConfigError(f"serving config {key} must be a non-empty string")

    total_nodes = _positive_int(config.get("total_nodes"), "total_nodes")
    total_gpus = _positive_int(config.get("total_gpus"), "total_gpus")
    gpus_per_node = _positive_int(
        config.get("gpus_per_node"), "gpus_per_node"
    )
    _argv(config.get("common_argv", []), "common_argv")
    _mapping(config.get("common_env", {}), "common_env")

    slurm = _mapping(config.get("slurm"), "slurm")
    for key in ("partition", "time_limit"):
        if not isinstance(slurm.get(key), str) or not slurm[key]:
            raise ConfigError(f"slurm.{key} must be a non-empty string")
    grace = slurm.get("grace_period_s", 120)
    if isinstance(grace, bool) or not isinstance(grace, int) or grace < 0:
        raise ConfigError("slurm.grace_period_s must be a non-negative integer")
    container = _mapping(slurm.get("container"), "slurm.container")
    if container.get("runtime") != "pyxis":
        raise ConfigError("slurm.container.runtime must be 'pyxis'")
    mounts = container.get("mounts")
    if not isinstance(mounts, list):
        raise ConfigError("slurm.container.mounts must be an array")
    mount_targets = set()
    for index, raw_mount in enumerate(mounts):
        mount = _mapping(raw_mount, f"slurm.container.mounts[{index}]")
        target = mount.get("target")
        if not isinstance(target, str) or not target.startswith("/"):
            raise ConfigError(
                f"slurm.container.mounts[{index}].target must be absolute"
            )
        if not mount.get("source") and not mount.get("source_env"):
            raise ConfigError(
                f"slurm.container.mounts[{index}] needs source or source_env"
            )
        mount_targets.add(target)
    if INFINIBAND_TARGET not in mount_targets:
        raise ConfigError(
            f"slurm.container.mounts must expose {INFINIBAND_TARGET} for NIXL"
        )
    if FLASHINFER_CACHE_TARGET not in mount_targets:
        raise ConfigError(
            "slurm.container.mounts must include a persistent FlashInfer cache "
            f"at {FLASHINFER_CACHE_TARGET}"
        )

    transfer = _mapping(config.get("kv_transfer"), "kv_transfer")
    if transfer.get("connector") != "NixlConnector":
        raise ConfigError("kv_transfer.connector must be 'NixlConnector'")
    _mapping(transfer.get("extra_config", {}), "kv_transfer.extra_config")

    roles = config.get("roles")
    if not isinstance(roles, list) or len(roles) != 2:
        raise ConfigError("roles must contain one prefill and one decode role")
    role_names = {role.get("role") for role in roles if isinstance(role, dict)}
    if role_names != {"prefill", "decode"}:
        raise ConfigError("roles must contain one prefill and one decode role")
    computed_nodes = 0
    computed_gpus = 0
    for index, raw_role in enumerate(roles):
        role = _mapping(raw_role, f"roles[{index}]")
        name = role["role"]
        count = _positive_int(role.get("count"), f"roles[{index}].count")
        nodes = _positive_int(
            role.get("nodes_per_instance"),
            f"roles[{index}].nodes_per_instance",
        )
        gpus = _positive_int(
            role.get("gpus_per_node"), f"roles[{index}].gpus_per_node"
        )
        if gpus != gpus_per_node:
            raise ConfigError(
                f"roles[{index}].gpus_per_node must equal {gpus_per_node}"
            )
        tensor_parallel_size = _positive_int(
            role.get("tensor_parallel_size"),
            f"roles[{index}].tensor_parallel_size",
        )
        if tensor_parallel_size > gpus or gpus % tensor_parallel_size:
            raise ConfigError(
                f"roles[{index}].tensor_parallel_size must divide "
                f"gpus_per_node ({gpus}) and stay within one node"
            )
        local_dp_size = gpus // tensor_parallel_size
        if role.get("local_dp_size") != local_dp_size:
            raise ConfigError(
                f"roles[{index}].local_dp_size must equal gpus_per_node / "
                f"tensor_parallel_size ({local_dp_size})"
            )
        expected_dp = nodes * local_dp_size
        if role.get("dp_size") != expected_dp:
            raise ConfigError(
                f"roles[{index}].dp_size must equal nodes_per_instance * "
                f"local_dp_size ({expected_dp})"
            )
        expected_kv_role = "kv_producer" if name == "prefill" else "kv_consumer"
        if role.get("kv_role") != expected_kv_role:
            raise ConfigError(
                f"{name} kv_role must be {expected_kv_role!r}, not "
                f"{role.get('kv_role')!r}"
            )
        port = _positive_int(role.get("base_port"), f"roles[{index}].base_port")
        if port > 65535:
            raise ConfigError(f"roles[{index}].base_port must be at most 65535")
        _argv(role.get("serve_argv", []), f"roles[{index}].serve_argv")
        _mapping(role.get("env", {}), f"roles[{index}].env")
        health = _mapping(role.get("health_check"), f"roles[{index}].health_check")
        if not isinstance(health.get("path"), str) or not health["path"].startswith("/"):
            raise ConfigError(f"roles[{index}].health_check.path must start with /")
        computed_nodes += count * nodes
        computed_gpus += count * nodes * gpus
    if computed_nodes != total_nodes or computed_gpus != total_gpus:
        raise ConfigError(
            "role topology does not match total_nodes/total_gpus "
            f"({computed_nodes}/{computed_gpus} != {total_nodes}/{total_gpus})"
        )

    router = _mapping(config.get("router"), "router")
    _argv(router.get("command_argv"), "router.command_argv")
    if not router["command_argv"]:
        raise ConfigError("router.command_argv must not be empty")
    if not isinstance(router.get("repo_path"), str) or not router["repo_path"]:
        raise ConfigError("router.repo_path must be a non-empty string")
    _positive_int(router.get("port"), "router.port")
    if router.get("nofile_limit") is not None:
        _positive_int(router["nofile_limit"], "router.nofile_limit")
    router_dp_size = _positive_int(
        router.get("intra_node_data_parallel_size"),
        "router.intra_node_data_parallel_size",
    )
    local_dp_sizes = {int(role["local_dp_size"]) for role in roles}
    if router_dp_size != 1 and local_dp_sizes != {router_dp_size}:
        raise ConfigError(
            "router.intra_node_data_parallel_size must be 1 for endpoint-level "
            "routing or equal every role's local_dp_size"
        )
    _mapping(router.get("health_check"), "router.health_check")
    return config


def _expand(value: str, environ: Mapping[str, str]) -> str:
    expanded = string.Template(value).safe_substitute(environ)
    if expanded.startswith("~"):
        home = environ.get("HOME")
        if home:
            expanded = home + expanded[1:]
        else:
            expanded = os.path.expanduser(expanded)
    return expanded


def resolve_mounts(
    config: Mapping[str, object], environ: Mapping[str, str] | None = None
) -> list[dict]:
    """Resolve source_env/fallback sources and ${USER}/${HOME} substitutions."""

    env = os.environ if environ is None else environ
    mounts = config["slurm"]["container"]["mounts"]
    resolved = []
    for mount in mounts:
        source_env = mount.get("source_env", "")
        source = env.get(source_env, "") if source_env else ""
        source = source or mount.get("source", "")
        source = _expand(str(source), env)
        if not source or "$" in source:
            raise ConfigError(
                f"cannot resolve mount source for target {mount['target']!r}"
            )
        resolved.append(
            {
                "source": source,
                "target": mount["target"],
                "read_only": bool(mount.get("read_only", False)),
            }
        )
    return resolved


def normalize_image(image: str) -> str:
    """Pyxis needs an explicit transport for ordinary Docker image names."""

    return image if "://" in image else f"docker://{image}"


def pyxis_mount_arg(mounts: Sequence[Mapping[str, object]]) -> str:
    rendered = []
    for mount in mounts:
        item = f"{mount['source']}:{mount['target']}"
        if mount.get("read_only"):
            item += ":ro"
        rendered.append(item)
    return ",".join(rendered)


def _clean_user_argv(argv: Sequence[str], location: str) -> list[str]:
    """Reject launcher-owned flags; tolerate and deduplicate hybrid LB."""

    cleaned = []
    for token in argv:
        flag = token.partition("=")[0]
        if flag == HYBRID_LB_FLAG:
            continue
        if flag in MANAGED_SERVE_FLAGS:
            raise ConfigError(f"{location} contains launcher-owned flag {flag!r}")
        cleaned.append(token)
    return cleaned


def kv_transfer_config(config: Mapping[str, object], kv_role: str) -> str:
    transfer = config["kv_transfer"]
    payload = {
        "kv_connector": transfer["connector"],
        "kv_role": kv_role,
        "kv_load_failure_policy": transfer.get("load_failure_policy", "fail"),
        "kv_connector_extra_config": transfer.get("extra_config", {}),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def vllm_argv(
    config: Mapping[str, object],
    role: Mapping[str, object],
    *,
    master_addr: str,
    rpc_port: int,
    start_rank: int | str,
) -> list[str]:
    """Build one node's vLLM command with every orchestration flag explicit."""

    common = _clean_user_argv(config.get("common_argv", []), "common_argv")
    role_args = _clean_user_argv(
        role.get("serve_argv", []), f"{role['role']}.serve_argv"
    )
    argv = [
        "vllm",
        "serve",
        config["model"],
        *common,
        *role_args,
        "--host",
        "0.0.0.0",
        "--port",
        str(role["base_port"]),
        "--tensor-parallel-size",
        str(role["tensor_parallel_size"]),
    ]
    if int(role["dp_size"]) > 1:
        argv.extend(
            [
                "--data-parallel-size",
                str(role["dp_size"]),
                "--data-parallel-size-local",
                str(role["local_dp_size"]),
                "--data-parallel-start-rank",
                str(start_rank),
                "--data-parallel-address",
                master_addr,
                "--data-parallel-rpc-port",
                str(rpc_port),
                HYBRID_LB_FLAG,
            ]
        )
    argv.extend(
        [
            "--kv-transfer-config",
            kv_transfer_config(config, str(role["kv_role"])),
        ]
    )
    return argv


def _expanded_env(
    values: Mapping[str, object], environ: Mapping[str, str]
) -> dict[str, str]:
    return {key: _expand(str(value), environ) for key, value in values.items()}


def _node_shell_script(
    config: Mapping[str, object],
    role: Mapping[str, object],
    nodes: Sequence[str],
    master_addr: str,
    rpc_port: int,
    mounts: Sequence[Mapping[str, object]],
    environ: Mapping[str, str],
) -> str:
    cases = "\n".join(
        f"  {shlex.quote(node)}) start_rank={index * role['local_dp_size']} ;;"
        for index, node in enumerate(nodes)
    )
    command = vllm_argv(
        config,
        role,
        master_addr=master_addr,
        rpc_port=rpc_port,
        start_rank="__START_RANK__",
    )
    marker = (
        command.index("__START_RANK__")
        if "__START_RANK__" in command
        else -1
    )
    command_text = " ".join(
        '"$start_rank"' if index == marker else shlex.quote(token)
        for index, token in enumerate(command)
    )
    env = _expanded_env(role.get("env", {}), environ)
    env_text = " ".join(
        f"{shlex.quote(key)}={shlex.quote(value)}" for key, value in sorted(env.items())
    )
    if env_text:
        command_text = f"env {env_text} {command_text}"
    mount_checks = [
        (
            f"test -e {shlex.quote(str(mount['target']))} || "
            "{ echo "
            + shlex.quote(
                f"required container path is missing: {mount['target']} "
                f"(host source {mount['source']})"
            )
            + " >&2; exit 2; }"
        )
        for mount in mounts
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            'node="${SLURMD_NODENAME:-$(hostname -s)}"',
            'case "$node" in',
            cases,
            '  *) echo "unexpected node: $node" >&2; exit 2 ;;',
            "esac",
            'export VLLM_NIXL_SIDE_CHANNEL_HOST="$node"',
            *mount_checks,
            f"exec {command_text}",
        ]
    )


def _srun_argv(
    config: Mapping[str, object],
    role: Mapping[str, object],
    nodes: Sequence[str],
    mounts: Sequence[Mapping[str, object]],
    shell_script: str,
) -> list[str]:
    return [
        "srun",
        "--overlap",
        "--label",
        "--kill-on-bad-exit=1",
        f"--nodes={len(nodes)}",
        f"--ntasks={len(nodes)}",
        "--ntasks-per-node=1",
        f"--nodelist={','.join(nodes)}",
        f"--gpus-per-task={role['gpus_per_node']}",
        f"--container-image={normalize_image(str(config['image']))}",
        f"--container-mounts={pyxis_mount_arg(mounts)}",
        "--container-writable",
        "bash",
        "-lc",
        shell_script,
    ]


def _router_argv(
    config: Mapping[str, object],
    prefill_urls: Sequence[str],
    decode_urls: Sequence[str],
    environ: Mapping[str, str],
) -> tuple[list[str], str, dict[str, str]]:
    router = config["router"]
    argv = [_expand(arg, environ) for arg in router["command_argv"]]
    argv.extend(
        [
            "--policy",
            router["policy"],
            "--host",
            router["listen_host"],
            "--port",
            str(router["port"]),
            "--intra-node-data-parallel-size",
            str(router["intra_node_data_parallel_size"]),
            "--vllm-pd-disaggregation",
        ]
    )
    if router.get("metrics_port") is not None:
        argv.extend(["--prometheus-port", str(router["metrics_port"])])
    for url in prefill_urls:
        argv.extend(["--prefill", url])
    for url in decode_urls:
        argv.extend(["--decode", url])
    cwd = _expand(router["repo_path"], environ)
    env = _expanded_env(router.get("env", {}), environ)
    return argv, cwd, env


def build_launch_plan(
    config: Mapping[str, object],
    nodes: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    controller_host: str | None = None,
) -> dict:
    """Assign nodes and return a deterministic, directly executable plan."""

    config = validate_config(dict(config))
    env = dict(os.environ if environ is None else environ)
    if len(nodes) < config["total_nodes"]:
        raise ConfigError(
            f"allocation has {len(nodes)} nodes, need {config['total_nodes']}"
        )
    selected_nodes = list(nodes[: config["total_nodes"]])
    if len(set(selected_nodes)) != len(selected_nodes):
        raise ConfigError("allocation node list contains duplicates")
    mounts = resolve_mounts(config, env)

    role_instances = []
    cursor = 0
    rpc_offset = 0
    for role in config["roles"]:
        for instance_index in range(role["count"]):
            stop = cursor + role["nodes_per_instance"]
            instance_nodes = selected_nodes[cursor:stop]
            cursor = stop
            master_addr = instance_nodes[0]
            rpc_port = DP_RPC_PORT_BASE + rpc_offset
            rpc_offset += 1
            node_commands = []
            endpoints = []
            for node_index, node in enumerate(instance_nodes):
                start_rank = node_index * role["local_dp_size"]
                argv = vllm_argv(
                    config,
                    role,
                    master_addr=master_addr,
                    rpc_port=rpc_port,
                    start_rank=start_rank,
                )
                node_env = _expanded_env(role.get("env", {}), env)
                node_env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = node
                node_commands.append(
                    {
                        "node": node,
                        "start_rank": start_rank,
                        "local_ranks": list(range(role["local_dp_size"])),
                        "global_ranks": list(
                            range(start_rank, start_rank + role["local_dp_size"])
                        ),
                        "argv": argv,
                        "env": node_env,
                    }
                )
                endpoints.append(f"http://{node}:{role['base_port']}")
            script = _node_shell_script(
                config, role, instance_nodes, master_addr, rpc_port, mounts, env
            )
            role_instances.append(
                {
                    "role": role["role"],
                    "instance": instance_index,
                    "nodes": instance_nodes,
                    "master_addr": master_addr,
                    "tensor_parallel_size": role["tensor_parallel_size"],
                    "dp_size": role["dp_size"],
                    "local_dp_size": role["local_dp_size"],
                    "rpc_port": rpc_port,
                    "port": role["base_port"],
                    "kv_role": role["kv_role"],
                    "node_commands": node_commands,
                    "endpoints": endpoints,
                    "health_check": dict(role["health_check"]),
                    "srun_argv": _srun_argv(
                        config, role, instance_nodes, mounts, script
                    ),
                }
            )

    prefill_instances = [
        instance for instance in role_instances if instance["role"] == "prefill"
    ]
    decode_instances = [
        instance for instance in role_instances if instance["role"] == "decode"
    ]
    # Schema v1 routes one URL per independent prefill instance and every
    # node-local HTTP frontend in the multi-node decoder.
    prefill_urls = [instance["endpoints"][0] for instance in prefill_instances]
    decode_urls = [
        url for instance in decode_instances for url in instance["endpoints"]
    ]
    host = controller_host or socket.gethostname()
    router_argv, router_cwd, router_env = _router_argv(
        config, prefill_urls, decode_urls, env
    )
    router = config["router"]
    local_router_url = f"http://{router['client_host']}:{router['port']}"
    routable_host = host if router["client_host"] in LOOPBACK_HOSTS else router["client_host"]
    allocation_router_url = f"http://{routable_host}:{router['port']}"

    workers = []
    for instance in role_instances:
        urls = (
            [instance["endpoints"][0]]
            if instance["role"] == "prefill"
            else instance["endpoints"]
        )
        for url in urls:
            node_index = instance["endpoints"].index(url)
            workers.append(
                {
                    "role": instance["role"],
                    "url": url,
                    "node": instance["nodes"][node_index],
                    "node_local_ranks": list(
                        range(config["router"]["intra_node_data_parallel_size"])
                    ),
                    "global_start_rank": node_index * instance["local_dp_size"],
                }
            )

    return {
        "version": 1,
        "allocation": {
            "nodes": selected_nodes,
            "total_nodes": config["total_nodes"],
            "total_gpus": config["total_gpus"],
            "gpus_per_node": config["gpus_per_node"],
        },
        "container": {
            "image": normalize_image(config["image"]),
            "writable": True,
            "mounts": mounts,
        },
        "role_instances": role_instances,
        "router": {
            "argv": router_argv,
            "cwd": router_cwd,
            "env": router_env,
            "nofile_limit": router.get("nofile_limit"),
            "prefill_urls": prefill_urls,
            "decode_urls": decode_urls,
            "workers": workers,
            "local_url": local_router_url,
            "allocation_url": allocation_router_url,
            "health_check": {
                **router["health_check"],
                "url": local_router_url + router["health_check"]["path"],
            },
        },
    }


def build_salloc_argv(
    config: Mapping[str, object], state_file: str, script_path: str | None = None
) -> list[str]:
    """Build the single login-node allocation wrapper command."""

    config = validate_config(dict(config))
    script = script_path or str(Path(__file__).resolve())
    return [
        "salloc",
        f"--nodes={config['total_nodes']}",
        f"--ntasks={config['total_nodes']}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={config['gpus_per_node']}",
        f"--partition={config['slurm']['partition']}",
        f"--time={config['slurm']['time_limit']}",
        "--job-name=perf-eval-pd",
        sys.executable,
        script,
        "--state-file",
        str(Path(state_file).resolve()),
        "--inside-allocation",
        "supervise",
    ]


def discover_allocation_nodes(
    environ: Mapping[str, str] | None = None,
    check_output: Callable[..., str] = subprocess.check_output,
) -> list[str]:
    env = os.environ if environ is None else environ
    node_expr = env.get("SLURM_JOB_NODELIST") or env.get("SLURM_NODELIST")
    if not node_expr:
        raise LaunchError("SLURM_JOB_NODELIST is not set inside the allocation")
    try:
        output = check_output(
            ["scontrol", "show", "hostnames", node_expr], text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchError(f"cannot expand Slurm node list {node_expr!r}: {exc}") from exc
    nodes = [line.strip() for line in output.splitlines() if line.strip()]
    if not nodes:
        raise LaunchError(f"Slurm node list {node_expr!r} expanded to no nodes")
    return nodes


def _is_sensitive_key(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SENSITIVE_ENV_MARKERS)


def redacted_plan(plan: Mapping[str, object]) -> dict:
    """Return a dry-run-safe plan with secret environment values removed."""

    result = copy.deepcopy(plan)
    replacements = set()
    for instance in result["role_instances"]:
        for command in instance["node_commands"]:
            for key, value in command["env"].items():
                if _is_sensitive_key(key) and value:
                    replacements.add(value)
                    command["env"][key] = "<redacted>"
    for key, value in result["router"]["env"].items():
        if _is_sensitive_key(key) and value:
            replacements.add(value)
            result["router"]["env"][key] = "<redacted>"

    def scrub(value: object) -> object:
        if isinstance(value, str):
            for secret in replacements:
                value = value.replace(secret, "<redacted>")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = scrub(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                value[key] = scrub(item)
        return value

    return scrub(result)


def _url_healthy(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urlrequest.urlopen(url, timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (OSError, urlerror.URLError):
        return False


def wait_for_checks(
    checks: Sequence[Mapping[str, object]],
    *,
    process_guard: Callable[[], None] | None = None,
    healthy: Callable[[str], bool] = _url_healthy,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until every URL is healthy, retaining per-check deadlines."""

    started = now()
    pending = {check["url"]: dict(check) for check in checks}
    deadlines = {
        url: started + float(check.get("timeout_s", 1200))
        for url, check in pending.items()
    }
    while pending:
        if process_guard:
            process_guard()
        current = now()
        for url, check in list(pending.items()):
            if healthy(url):
                del pending[url]
                continue
            if current >= deadlines[url]:
                raise LaunchError(f"health check timed out: {url}")
        if pending:
            interval = min(
                float(check.get("poll_interval_s", 5))
                for check in pending.values()
            )
            sleep(interval)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _state_summary(
    plan: Mapping[str, object],
    children: Sequence[ChildProcess],
    *,
    ready: bool,
    grace_period_s: int,
) -> dict:
    role_checks = []
    for instance in plan["role_instances"]:
        for endpoint in instance["endpoints"]:
            role_checks.append(
                {
                    **instance["health_check"],
                    "role": instance["role"],
                    "instance": instance["instance"],
                    "url": endpoint + instance["health_check"]["path"],
                }
            )
    return {
        "version": 1,
        "ready": ready,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "owns_allocation": True,
        "controller_host": socket.gethostname(),
        "controller_pid": os.getpid(),
        "grace_period_s": grace_period_s,
        "nodes": plan["allocation"]["nodes"],
        "role_checks": role_checks,
        "router_check": plan["router"]["health_check"],
        "router_url": plan["router"]["local_url"],
        "allocation_router_url": plan["router"]["allocation_url"],
        "router_port": int(plan["router"]["local_url"].rsplit(":", 1)[1]),
        "children": [
            {"label": child.label, "pid": child.process.pid} for child in children
        ],
    }


class Supervisor:
    """Own all role srun steps and the login-node router process."""

    def __init__(
        self,
        config: Mapping[str, object],
        plan: Mapping[str, object],
        state_file: str,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.config = config
        self.plan = plan
        self.state_file = Path(state_file)
        self.popen = popen
        self.children: list[ChildProcess] = []
        self.stop_requested = False
        self.failure: str | None = None

    def _write_state(self, ready: bool) -> None:
        state = _state_summary(
            self.plan,
            self.children,
            ready=ready,
            grace_period_s=int(self.config["slurm"]["grace_period_s"]),
        )
        if self.failure:
            state["error"] = self.failure
        _write_json_atomic(self.state_file, state)

    def _guard_children(self) -> None:
        if getattr(self, "stop_requested", False):
            raise StopRequested
        for child in self.children:
            status = child.process.poll()
            if status is not None:
                raise LaunchError(
                    f"{child.label} exited unexpectedly with status {status}"
                )

    def request_stop(self, signum: int, _frame: object) -> None:
        self.stop_requested = True

    def _start_role_steps(self) -> None:
        for instance in self.plan["role_instances"]:
            label = f"{instance['role']}[{instance['instance']}]"
            print(f"starting {label} on {','.join(instance['nodes'])}", flush=True)
            process = self.popen(instance["srun_argv"], start_new_session=True)
            self.children.append(ChildProcess(label, process))
            self._write_state(ready=False)

    def _start_router(self) -> None:
        router = self.plan["router"]
        router_env = os.environ.copy()
        router_env.update(router["env"])
        nofile_limit = router.get("nofile_limit")
        if nofile_limit is not None:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            if (
                hard_limit != resource.RLIM_INFINITY
                and nofile_limit > hard_limit
            ):
                raise LaunchError(
                    f"router nofile_limit {nofile_limit} exceeds the process "
                    f"hard limit {hard_limit}"
                )
            if soft_limit < nofile_limit:
                try:
                    resource.setrlimit(
                        resource.RLIMIT_NOFILE,
                        (nofile_limit, hard_limit),
                    )
                except (OSError, ValueError) as exc:
                    raise LaunchError(
                        f"cannot raise router nofile limit to {nofile_limit}: {exc}"
                    ) from exc
                print(
                    f"raised router nofile soft limit from {soft_limit} "
                    f"to {nofile_limit}",
                    flush=True,
                )
        print(
            f"starting router with {len(router['prefill_urls'])} prefill and "
            f"{len(router['decode_urls'])} decode endpoints",
            flush=True,
        )
        process = self.popen(
            router["argv"],
            cwd=router["cwd"],
            env=router_env,
            start_new_session=True,
        )
        self.children.append(ChildProcess("router", process))
        self._write_state(ready=False)

    def _role_checks(self) -> list[dict]:
        return _state_summary(
            self.plan,
            self.children,
            ready=False,
            grace_period_s=int(self.config["slurm"]["grace_period_s"]),
        )["role_checks"]

    def run(self) -> int:
        old_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, self.request_stop)
        self._write_state(ready=False)
        try:
            self._start_role_steps()
            wait_for_checks(self._role_checks(), process_guard=self._guard_children)
            self._start_router()
            wait_for_checks(
                [self.plan["router"]["health_check"]],
                process_guard=self._guard_children,
            )
            self._write_state(ready=True)
            print(f"PD_ROUTER_URL={shlex.quote(self.plan['router']['local_url'])}")
            print(f"PD_ROUTER_PORT={self.config['router']['port']}", flush=True)
            while not self.stop_requested:
                self._guard_children()
                time.sleep(1)
            return 0
        except StopRequested:
            self._write_state(ready=False)
            return 0
        except (ConfigError, LaunchError, OSError) as exc:
            self.failure = str(exc)
            self._write_state(ready=False)
            print(f"PD serving failed: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            self.terminate_children(int(self.config["slurm"]["grace_period_s"]))
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)

    def terminate_children(self, grace_period_s: int) -> None:
        live = [child for child in reversed(self.children) if child.process.poll() is None]
        for child in live:
            try:
                os.killpg(os.getpgid(child.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + grace_period_s
        while live and time.monotonic() < deadline:
            live = [child for child in live if child.process.poll() is None]
            if live:
                time.sleep(0.2)
        for child in live:
            try:
                os.killpg(os.getpgid(child.process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        for child in reversed(self.children):
            try:
                child.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass


def ensure_runtime_mounts(mounts: Sequence[Mapping[str, object]]) -> None:
    """Create/validate only the login-visible persistent cache.

    Model shards and RDMA devices may intentionally exist only on compute
    nodes.  Every container task checks its resolved mount targets immediately
    before vLLM starts instead of rejecting those valid node-local sources on
    the controller.
    """

    for mount in mounts:
        if mount["target"] == FLASHINFER_CACHE_TARGET:
            source = Path(str(mount["source"]))
            source.mkdir(parents=True, exist_ok=True)
            if not source.is_dir():
                raise LaunchError(
                    f"FlashInfer cache is not a directory: {source}"
                )


def _parse_nodes(value: str | None, count: int) -> list[str]:
    if value:
        nodes = [item.strip() for item in value.split(",") if item.strip()]
        if not nodes:
            raise ConfigError("--nodes must contain at least one node")
        return nodes
    return [f"node{index:03d}" for index in range(count)]


def _read_state(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise LaunchError(f"state file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchError(f"state file is invalid JSON: {path}: {exc}") from exc


def _pid_alive(pid: int) -> bool:
    """Return false for a missing process (and Linux zombies)."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    # A dead background process can remain visible to kill(0) until its parent
    # reaps it. The production launcher runs on Linux, where /proc exposes that
    # state directly; other platforms conservatively treat it as alive.
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        close_paren = stat.rfind(")")
        if close_paren >= 0 and stat[close_paren + 2 :].split(None, 1)[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True


def wait_ready(
    state_file: str,
    timeout_s: float,
    supervisor_pid: int | None = None,
    *,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> dict:
    deadline = time.monotonic() + timeout_s
    state = None
    while time.monotonic() < deadline:
        try:
            state = _read_state(state_file)
        except LaunchError:
            if supervisor_pid is not None and not pid_alive(supervisor_pid):
                raise LaunchError(
                    "PD supervisor exited before creating readiness state"
                )
            time.sleep(1)
            continue
        if state.get("error"):
            raise LaunchError(str(state["error"]))
        checks = [*state.get("role_checks", [])]
        if state.get("router_check"):
            checks.append(state["router_check"])
        if state.get("ready") and checks and all(
            _url_healthy(check["url"]) for check in checks
        ):
            return state
        if supervisor_pid is not None and not pid_alive(supervisor_pid):
            raise LaunchError("PD supervisor exited before readiness")
        time.sleep(1)
    raise LaunchError(f"PD serving was not ready within {timeout_s:g}s")


def _stop_wait_s(state: Mapping[str, object]) -> float:
    raw_grace = state.get("grace_period_s", DEFAULT_STOP_GRACE_S)
    if isinstance(raw_grace, bool) or not isinstance(raw_grace, (int, float)):
        raw_grace = DEFAULT_STOP_GRACE_S
    grace = max(0.0, float(raw_grace))
    return min(grace + STOP_EXIT_SLACK_S, MAX_STOP_WAIT_S)


def _wait_for_pid_exit(
    pid: int,
    timeout_s: float,
    *,
    pid_alive: Callable[[int], bool],
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> bool:
    deadline = now() + timeout_s
    while pid_alive(pid):
        remaining = deadline - now()
        if remaining <= 0:
            return False
        sleep(min(STOP_POLL_INTERVAL_S, remaining))
    return True


def stop_from_state(
    state_file: str,
    *,
    pid_alive: Callable[[int], bool] = _pid_alive,
    send_signal: Callable[[int, int], None] = os.kill,
    run_command: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    state = _read_state(state_file)
    if state.get("job_id"):
        if not state.get("owns_allocation"):
            raise LaunchError(
                "refusing to cancel an allocation not owned by this launcher"
            )

        controller_pid = state.get("controller_pid")
        controller_is_local = state.get("controller_host") == socket.gethostname()
        if (
            controller_is_local
            and isinstance(controller_pid, int)
            and not isinstance(controller_pid, bool)
            and controller_pid > 0
            and pid_alive(controller_pid)
        ):
            try:
                send_signal(controller_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            else:
                if _wait_for_pid_exit(
                    controller_pid,
                    _stop_wait_s(state),
                    pid_alive=pid_alive,
                    now=now,
                    sleep=sleep,
                ):
                    return

        result = run_command(["scancel", str(state["job_id"])], check=False)
        if result.returncode:
            raise LaunchError(
                f"scancel failed for Slurm job {state['job_id']} "
                f"with status {result.returncode}"
            )
        return
    if state.get("controller_host") != socket.gethostname():
        raise LaunchError("cannot stop local child PIDs from another controller host")
    for child in reversed(state.get("children", [])):
        try:
            os.killpg(os.getpgid(int(child["pid"])), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def build_client_srun_argv(
    config: Mapping[str, object],
    state: Mapping[str, object],
    command: Sequence[str],
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    if not command:
        raise ConfigError("exec-client requires a command after --")
    if not state.get("job_id"):
        raise LaunchError("state has no live Slurm job_id")
    env = os.environ if environ is None else environ
    mounts = resolve_mounts(config, env)
    # Benchmark clients write JSON/result files relative to the checkout.
    # Mounting the shared cwd at the identical path keeps those artifacts on
    # the host and makes relative paths behave exactly as they do in run.sh.
    checkout = str(Path.cwd().resolve())
    if not any(mount["target"] == checkout for mount in mounts):
        mounts.append(
            {"source": checkout, "target": checkout, "read_only": False}
        )
    router_url = str(state["allocation_router_url"])
    parsed_router_url = urlparse.urlsplit(router_url)
    router_host = parsed_router_url.hostname
    router_port = parsed_router_url.port
    if not router_host or router_port is None:
        raise LaunchError(
            f"state has invalid allocation_router_url: {router_url!r}"
        )
    expanded_command = [
        token.replace("{router_url}", router_url)
        .replace("{router_host}", router_host)
        .replace("{router_port}", str(router_port))
        .replace("{model}", str(config["model"]))
        for token in command
    ]
    return [
        "srun",
        "--overlap",
        f"--jobid={state['job_id']}",
        "--nodes=1",
        "--ntasks=1",
        f"--nodelist={state['nodes'][0]}",
        f"--container-image={normalize_image(str(config['image']))}",
        f"--container-mounts={pyxis_mount_arg(mounts)}",
        "--container-writable",
        f"--container-workdir={checkout}",
        "env",
        f"OPENAI_BASE_URL={router_url}",
        f"PD_ROUTER_URL={router_url}",
        "CUDA_VISIBLE_DEVICES=",
        *expanded_command,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="normalized JSON object, JSON file path, or - (defaults to WORKLOAD_SERVING_JSON)",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("PD_SERVING_STATE_FILE", DEFAULT_STATE_FILE),
    )
    parser.add_argument(
        "--nodes", help="comma-separated nodes for dry-run (defaults to synthetic names)"
    )
    parser.add_argument("--inside-allocation", action="store_true", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run", help="print the deterministic launch plan")
    subparsers.add_parser("supervise", help="allocate, launch, wait, and supervise")
    ready = subparsers.add_parser("wait-ready", help="wait on an existing state file")
    ready.add_argument("--timeout", type=float, default=1800)
    ready.add_argument("--supervisor-pid", type=int)
    subparsers.add_parser("stop", help="cancel the allocation recorded in state")
    client = subparsers.add_parser(
        "exec-client", help="run a client command in the serving allocation/image"
    )
    client.add_argument("client_command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = None
    try:
        if args.command == "wait-ready":
            state = wait_ready(
                args.state_file,
                args.timeout,
                supervisor_pid=args.supervisor_pid,
            )
            print(f"PD_ROUTER_URL={shlex.quote(state['router_url'])}")
            print(f"PD_ROUTER_PORT={state['router_port']}")
            return 0
        if args.command == "stop":
            stop_from_state(args.state_file)
            return 0

        config = load_config(args.config)
        if args.command == "dry-run":
            nodes = _parse_nodes(args.nodes, config["total_nodes"])
            plan = build_launch_plan(config, nodes)
            plan["allocation"]["salloc_argv"] = build_salloc_argv(
                config, args.state_file
            )
            print(json.dumps(redacted_plan(plan), indent=2, sort_keys=True))
            return 0

        if args.command == "exec-client":
            state = _read_state(args.state_file)
            command = list(args.client_command)
            if command and command[0] == "--":
                command.pop(0)
            client_argv = build_client_srun_argv(config, state, command)
            return subprocess.run(client_argv).returncode

        if args.command == "supervise" and not args.inside_allocation:
            if os.environ.get("SLURM_JOB_ID"):
                raise LaunchError(
                    "refusing to adopt an ambient Slurm allocation; run the "
                    "launcher from the login shell so it can own its allocation"
                )
            allocation_argv = build_salloc_argv(config, args.state_file)
            allocation_env = os.environ.copy()
            allocation_env["WORKLOAD_SERVING_JSON"] = json.dumps(
                config, separators=(",", ":")
            )
            os.execvpe(allocation_argv[0], allocation_argv, allocation_env)

        nodes = discover_allocation_nodes()
        plan = build_launch_plan(config, nodes)
        ensure_runtime_mounts(plan["container"]["mounts"])
        return Supervisor(config, plan, args.state_file).run()
    except (ConfigError, LaunchError, OSError) as exc:
        if args.command == "supervise":
            try:
                _write_json_atomic(
                    Path(args.state_file),
                    {
                        "version": 1,
                        "ready": False,
                        "job_id": (
                            os.environ.get("SLURM_JOB_ID", "")
                            if args.inside_allocation
                            else ""
                        ),
                        "owns_allocation": bool(args.inside_allocation),
                        "controller_host": socket.gethostname(),
                        "controller_pid": os.getpid(),
                        "grace_period_s": (
                            int(config["slurm"]["grace_period_s"])
                            if config is not None
                            else DEFAULT_STOP_GRACE_S
                        ),
                        "error": str(exc),
                    },
                )
            except OSError:
                pass
        print(f"slurm_pd_launcher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
