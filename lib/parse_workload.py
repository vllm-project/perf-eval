"""Read a workload YAML and emit `WORKLOAD_*` shell exports for run.sh.

Usage: eval "$(python3 lib/parse_workload.py workloads/foo.yaml)"

The README documents the recipe schema. This script validates it and
projects it into shell variables: top-level metadata, server config
(image, model, serve_args, env, runtime), the lm_eval task list, the
vllm_bench config list, and bench ingest metadata (device/tp/precision).

Image precedence: VLLM_IMAGE > VLLM_COMMIT > workload `vllm.image` >
`vllm/vllm-openai:latest`. When BENCH_ONLY is truthy, lm_eval task names
are not validated against the registry (because they will not run).
"""

import base64
import json
import os
import re
import shlex
import sys

import yaml

TASK_FIELDS = {"name", "num_fewshot", "model_args"}
BENCH_FIELDS = {
    "name", "backend", "dataset", "input_len", "output_len",
    "num_prompts", "max_concurrency", "args",
    "speed_bench_dataset_subset", "speed_bench_category",
}
BENCH_REQUIRED = ("name", "input_len", "output_len", "num_prompts", "max_concurrency")
BENCH_RESERVED_ARGS = {
    "backend", "base-url", "host", "port", "model", "dataset-name",
    "num-prompts", "max-concurrency", "trust-remote-code",
    "random-input-len", "random-output-len", "ignore-eos", "dataset-path",
    "speed-bench-output-len", "speed-bench-dataset-subset",
    "speed-bench-category", "skip-tokenizer-init", "save-result",
    "result-filename",
}
BFCL_FIELDS = {
    "test_categories", "num_threads", "temperature",
    "maximum_step_limit", "max_test_cases",
}
BFCL_DEFAULT_MAXIMUM_STEP_LIMIT = 10
BFCL_KNOWN_CATEGORIES = {
    "simple_python", "simple_java", "simple_javascript",
    "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel",
    "live_parallel_multiple", "live_irrelevance", "live_relevance",
    "multi_turn_base", "multi_turn_miss_func",
    "multi_turn_miss_param", "multi_turn_long_context",
    "memory_kv", "memory_vector", "memory_rec_sum",
    "all", "all_scoring", "single_turn", "multi_turn",
    "live", "non_live", "non_python", "python", "memory", "agentic",
}

SERVING_FIELDS = {
    "version", "mode", "launcher", "slurm", "kv_transfer",
    "common_serve_args", "roles", "router",
}
SLURM_FIELDS = {"partition", "time_limit", "grace_period_s", "container"}
CONTAINER_FIELDS = {"runtime", "mounts"}
MOUNT_FIELDS = {"source", "source_env", "target", "read_only"}
KV_TRANSFER_FIELDS = {"connector", "load_failure_policy", "extra_config"}
ROLE_FIELDS = {
    "role", "count", "nodes_per_instance", "gpus_per_node", "base_port",
    "tensor_parallel_size", "kv_role", "serve_args", "env", "health_check",
}
ROUTER_FIELDS = {
    "repo_path", "revision", "command", "policy", "listen_host",
    "client_host", "port", "metrics_port", "nofile_limit",
    "intra_node_data_parallel_size", "prefill_endpoints",
    "decode_endpoints", "env", "health_check",
}
HEALTH_CHECK_FIELDS = {"path", "timeout_s", "poll_interval_s"}
ORCHESTRATION_SERVE_FLAGS = {
    "--port", "--data-parallel-size", "-dp", "--dp",
    "--tensor-parallel-size", "-tp", "--tp",
    "--data-parallel-size-local", "--data-parallel-address",
    "--data-parallel-rpc-port", "--data-parallel-start-rank",
    "--data-parallel-rank", "--data-parallel-hybrid-lb",
    "--kv-transfer-config",
}


def emit(name: str, value: object) -> None:
    print(f"WORKLOAD_{name}={shlex.quote(str(value))}")


def fmt(v: object) -> str:
    """Render a Python value in lm-eval's expected literal format."""
    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return "None"
    return str(v)


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def commit_from_image(image: str) -> str:
    """Extract a commit SHA from an image tag, if one is embedded."""
    _, sep, tag = image.rpartition(":")
    if not sep:
        return ""
    tag = tag.split("@", 1)[0]
    m = (re.match(r"nightly-([0-9a-f]{7,40})(?:[-_.].*)?$", tag, re.IGNORECASE)
         or re.search(r"(?:^|[-_.])([0-9a-f]{12,40})(?:$|[-_.])", tag, re.IGNORECASE))
    return m.group(1) if m else ""


def known_task_names() -> set:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        sys.exit(f"cannot validate task names: lm_eval not importable ({e})")
    return set(TaskManager().all_tasks)


def load_profile(gpu: str, workload_path: str) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(workload_path)))
    profiles_path = os.path.join(repo_root, "lib", "gpu_profiles.yaml")
    with open(profiles_path) as f:
        profiles = yaml.safe_load(f)
    if gpu not in profiles:
        sys.exit(f"unknown gpu {gpu!r} in {profiles_path} (have {', '.join(profiles)})")
    return profiles[gpu]


def resolve_image(vllm: dict, profile: dict) -> tuple[str, str]:
    """Pick the image and commit using VLLM_IMAGE / VLLM_COMMIT / workload."""
    override_image = (os.environ.get("VLLM_IMAGE") or "").strip()
    override_commit = (os.environ.get("VLLM_COMMIT") or "").strip()
    # ROCm images are located at vllm/vllm-openai-rocm. The default
    # images (CUDA) are stored at vllm/vllm-openai
    custom_repo = (profile.get("image_repo") or "").strip()
    repo = custom_repo or "vllm/vllm-openai"
    # Don't use VLLM_IMAGE for AMD workloads unless it is a ROCm image
    if override_image and (not custom_repo or "rocm" in override_image.lower()):
        return override_image, override_commit or commit_from_image(override_image)

    commit = override_commit or commit_from_image(override_image)
    if commit:
        return f"{repo}:nightly-{commit}", commit

    image = vllm.get("image", f"{repo}:nightly")
    return image, commit_from_image(str(image))


def parse_tp(serve_args: str) -> int:
    """Effective parallel degree (TP * DP) from serve_args; defaults to 1.

    `vllm bench serve` reports aggregate throughput; we divide by this to get
    per-GPU metrics for the dashboard.
    """
    toks = serve_args.split()

    def find(*names):
        for i, t in enumerate(toks):
            if "=" in t:
                key, _, val = t.partition("=")
                if key in names:
                    try:
                        return int(val)
                    except ValueError:
                        return None
            elif t in names and i + 1 < len(toks):
                try:
                    return int(toks[i + 1])
                except ValueError:
                    return None
        return None

    tp = find("--tensor-parallel-size", "-tp", "--tp") or 1
    dp = find("--data-parallel-size", "-dp", "--dp") or 1
    return tp * dp


def precision_from_model(model: str) -> str:
    name = model.lower()
    for marker in ("fp4", "fp8", "int4", "int8", "bf16", "fp16"):
        if marker in name:
            return marker
    return "bf16"


def validate_tasks(tasks: list, path: str) -> None:
    if not tasks:
        sys.exit(f"{path}: missing or empty `lm_eval.tasks`")
    skip_registry = env_truthy("BENCH_ONLY")
    known = set() if skip_registry else known_task_names()
    for t in tasks:
        extra = set(t) - TASK_FIELDS
        if extra:
            sys.exit(
                f"{path}: task {t['name']!r} has unsupported top-level fields "
                f"{sorted(extra)}; move them under `model_args:`"
            )
        if not skip_registry and t["name"] not in known:
            sys.exit(f"{path}: unknown lm_eval task {t['name']!r}")


def task_tsv(tasks: list, base_args: dict) -> str:
    lines = []
    for t in tasks:
        merged = {**base_args, **(t.get("model_args") or {})}
        args = ",".join(f"{k}={fmt(v)}" for k, v in merged.items())
        lines.append(f"{t['name']}\t{t.get('num_fewshot', 0)}\t{args}")
    return "\n".join(lines)


def normalize_bench_arg_name(name: str) -> str:
    return name.lstrip("-").replace("_", "-")


def encode_bench_args(args: object, config_name: str, path: str) -> str:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        sys.exit(f"{path}: vllm_bench config {config_name!r} args must be a map")
    normalized = {}
    for name, value in args.items():
        if not isinstance(name, str) or not normalize_bench_arg_name(name):
            sys.exit(
                f"{path}: vllm_bench config {config_name!r} args keys must be non-empty strings"
            )
        normalized_name = normalize_bench_arg_name(name)
        if normalized_name in BENCH_RESERVED_ARGS:
            sys.exit(
                f"{path}: vllm_bench config {config_name!r} args cannot override "
                f"wrapper-owned option --{normalized_name}"
            )
        if normalized_name in normalized:
            sys.exit(
                f"{path}: vllm_bench config {config_name!r} args contains duplicate "
                f"option --{normalized_name} after normalization"
            )
        normalized[normalized_name] = value
    payload = json.dumps(normalized, separators=(",", ":")).encode()
    return base64.b64encode(payload).decode()


def bench_tsv(configs: list, path: str) -> str:
    seen = set()
    lines = []
    for c in configs:
        extra = set(c) - BENCH_FIELDS
        if extra:
            sys.exit(
                f"{path}: vllm_bench config {c.get('name')!r} has unsupported "
                f"fields {sorted(extra)}; allowed: {sorted(BENCH_FIELDS)}"
            )
        for k in BENCH_REQUIRED:
            if c.get(k) is None:
                sys.exit(f"{path}: vllm_bench config {c.get('name')!r} missing required field {k!r}")
        if c["name"] in seen:
            sys.exit(f"{path}: duplicate vllm_bench config name {c['name']!r}")
        seen.add(c["name"])

        def opt(key):
            v = c.get(key)
            return str(v) if v not in (None, "") else "-"

        lines.append(
            "\t".join(
                [
                    c["name"],
                    opt("backend"),
                    str(c.get("dataset", "random")),
                    str(c["input_len"]),
                    str(c["output_len"]),
                    str(c["num_prompts"]),
                    str(c["max_concurrency"]),
                    opt("speed_bench_dataset_subset"),
                    opt("speed_bench_category"),
                    encode_bench_args(c.get("args"), c["name"], path),
                ]
            )
        )
    return "\n".join(lines)


def _validate_bfcl_limits(bfcl: dict, path: str) -> None:
    limit = bfcl.get("maximum_step_limit")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        sys.exit(f"{path}: bfcl.maximum_step_limit must be a positive integer")

    cases = bfcl.get("max_test_cases")
    if cases is None:
        return
    if isinstance(cases, int):
        if cases < 1:
            sys.exit(f"{path}: bfcl.max_test_cases must be a positive integer")
        return
    if not isinstance(cases, dict):
        sys.exit(
            f"{path}: bfcl.max_test_cases must be a positive integer or category map"
        )
    for cat, count in cases.items():
        if cat not in BFCL_KNOWN_CATEGORIES:
            sys.exit(f"{path}: unknown bfcl max_test_cases category {cat!r}")
        if not isinstance(count, int) or count < 1:
            sys.exit(f"{path}: bfcl.max_test_cases[{cat!r}] must be a positive integer")


def validate_bfcl(bfcl: dict, serve_args: str, path: str) -> None:
    extra = set(bfcl) - BFCL_FIELDS
    if extra:
        sys.exit(f"{path}: bfcl block has unsupported fields {sorted(extra)}")
    cats = bfcl.get("test_categories") or []
    if not cats:
        sys.exit(f"{path}: bfcl block requires at least one test_categories entry")
    for cat in cats:
        if cat not in BFCL_KNOWN_CATEGORIES:
            sys.exit(f"{path}: unknown bfcl test category {cat!r}")
    if "--tool-call-parser" not in serve_args:
        print(
            f"WARNING: {path}: bfcl without --tool-call-parser in serve_args; "
            "some models may need it for function-calling",
            file=sys.stderr,
        )
    _validate_bfcl_limits(bfcl, path)


def max_test_cases_for_category(bfcl: dict, category: str) -> int | None:
    cases = bfcl.get("max_test_cases")
    if isinstance(cases, int):
        return cases
    if isinstance(cases, dict):
        return cases.get(category)
    return None


def bfcl_tsv(bfcl: dict) -> str:
    """Emit per-category rows; use '-' for unset optional columns (bash read drops empties)."""
    cats = bfcl.get("test_categories") or []
    num_threads = bfcl.get("num_threads", 8)
    temperature = bfcl.get("temperature", 0.001)
    limit = bfcl.get("maximum_step_limit")

    def opt(value: object) -> str:
        return "-" if value in (None, "") else str(value)

    return "\n".join(
        "\t".join(
            [
                cat,
                str(num_threads),
                str(temperature),
                opt(limit),
                opt(max_test_cases_for_category(bfcl, cat)),
            ]
        )
        for cat in cats
    )


def schema_error(path: str, location: str, message: str) -> None:
    sys.exit(f"{path}: {location} {message}")


def require_mapping(value: object, path: str, location: str) -> dict:
    if not isinstance(value, dict):
        schema_error(path, location, "must be a mapping")
    return value


def reject_unknown_fields(
    value: dict, allowed: set, path: str, location: str,
) -> None:
    extra = set(value) - allowed
    if extra:
        schema_error(
            path,
            location,
            f"has unsupported fields {sorted(str(k) for k in extra)}",
        )


def positive_int(value: object, path: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        schema_error(path, location, "must be a positive integer")
    return value


def boolean(value: object, path: str, location: str) -> bool:
    if not isinstance(value, bool):
        schema_error(path, location, "must be a boolean")
    return value


def port_number(value: object, path: str, location: str) -> int:
    port = positive_int(value, path, location)
    if port > 65535:
        schema_error(path, location, "must be at most 65535")
    return port


def nonempty_string(value: object, path: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        schema_error(path, location, "must be a non-empty string")
    return value


def normalize_env(value: object, path: str, location: str) -> dict:
    env = require_mapping(value, path, location)
    normalized = {}
    for key, raw_value in env.items():
        if not isinstance(key, str) or not key:
            schema_error(path, location, "keys must be non-empty strings")
        if isinstance(raw_value, (dict, list)):
            schema_error(path, f"{location}.{key}", "must be a scalar value")
        normalized[key] = fmt(raw_value)
    return normalized


def split_argv(value: object, path: str, location: str) -> list:
    if value in (None, ""):
        return []
    if not isinstance(value, str):
        schema_error(path, location, "must be a shell-style argument string")
    try:
        return shlex.split(value)
    except ValueError as e:
        schema_error(path, location, f"is not valid shell-style syntax ({e})")


def reject_orchestration_flags(argv: list, path: str, location: str) -> None:
    for token in argv:
        flag = token.partition("=")[0]
        if flag in ORCHESTRATION_SERVE_FLAGS:
            schema_error(
                path,
                location,
                f"must not set launcher-owned flag {flag!r}",
            )


def normalize_health_check(
    value: object,
    path: str,
    location: str,
    *,
    default_path: str = "/health",
) -> dict:
    health = require_mapping({} if value is None else value, path, location)
    reject_unknown_fields(health, HEALTH_CHECK_FIELDS, path, location)

    check_path = health.get("path", default_path)
    if not isinstance(check_path, str) or not check_path.startswith("/"):
        schema_error(path, f"{location}.path", "must start with '/'")

    timeout_s = health.get("timeout_s", 1200)
    timeout_s = positive_int(timeout_s, path, f"{location}.timeout_s")

    poll_interval_s = health.get("poll_interval_s", 5)
    if (isinstance(poll_interval_s, bool)
            or not isinstance(poll_interval_s, (int, float))
            or poll_interval_s <= 0):
        schema_error(
            path,
            f"{location}.poll_interval_s",
            "must be a positive number",
        )

    return {
        "path": check_path,
        "timeout_s": timeout_s,
        "poll_interval_s": poll_interval_s,
    }


def normalize_mounts(value: object, path: str) -> list:
    if value is None:
        value = []
    if not isinstance(value, list):
        schema_error(path, "serving.slurm.container.mounts", "must be a list")

    normalized = []
    targets = set()
    for i, raw_mount in enumerate(value):
        location = f"serving.slurm.container.mounts[{i}]"
        mount = require_mapping(raw_mount, path, location)
        reject_unknown_fields(mount, MOUNT_FIELDS, path, location)

        source = mount.get("source", "")
        source_env = mount.get("source_env", "")
        if source:
            source = nonempty_string(source, path, f"{location}.source")
        if source_env:
            source_env = nonempty_string(
                source_env, path, f"{location}.source_env",
            )
        if not source and not source_env:
            schema_error(
                path,
                location,
                "requires at least one of 'source' or 'source_env'",
            )

        target = nonempty_string(
            mount.get("target"), path, f"{location}.target",
        )
        if not target.startswith("/"):
            schema_error(path, f"{location}.target", "must be an absolute path")
        if target in targets:
            schema_error(path, location, f"duplicates mount target {target!r}")
        targets.add(target)

        read_only = mount.get("read_only", False)
        if not isinstance(read_only, bool):
            schema_error(path, f"{location}.read_only", "must be a boolean")
        normalized.append({
            "source": source,
            "source_env": source_env,
            "target": target,
            "read_only": read_only,
        })
    return normalized


def normalize_slurm(value: object, path: str) -> dict:
    slurm = require_mapping(value, path, "serving.slurm")
    reject_unknown_fields(slurm, SLURM_FIELDS, path, "serving.slurm")

    partition = nonempty_string(
        slurm.get("partition"), path, "serving.slurm.partition",
    )
    time_limit = nonempty_string(
        slurm.get("time_limit"), path, "serving.slurm.time_limit",
    )
    grace_period_s = slurm.get("grace_period_s", 120)
    if (isinstance(grace_period_s, bool)
            or not isinstance(grace_period_s, int)
            or grace_period_s < 0):
        schema_error(
            path,
            "serving.slurm.grace_period_s",
            "must be a non-negative integer",
        )

    container = require_mapping(
        slurm.get("container"), path, "serving.slurm.container",
    )
    reject_unknown_fields(
        container, CONTAINER_FIELDS, path, "serving.slurm.container",
    )
    runtime = container.get("runtime")
    if runtime != "pyxis":
        schema_error(
            path,
            "serving.slurm.container.runtime",
            "must be 'pyxis' for schema version 1",
        )

    return {
        "partition": partition,
        "time_limit": time_limit,
        "grace_period_s": grace_period_s,
        "container": {
            "runtime": runtime,
            "mounts": normalize_mounts(container.get("mounts"), path),
        },
    }


def normalize_kv_transfer(value: object, path: str) -> dict:
    transfer = require_mapping(
        {} if value is None else value, path, "serving.kv_transfer",
    )
    reject_unknown_fields(
        transfer, KV_TRANSFER_FIELDS, path, "serving.kv_transfer",
    )

    connector = transfer.get("connector", "NixlConnector")
    if connector != "NixlConnector":
        schema_error(
            path,
            "serving.kv_transfer.connector",
            "must be 'NixlConnector' for schema version 1",
        )
    policy = nonempty_string(
        transfer.get("load_failure_policy", "fail"),
        path,
        "serving.kv_transfer.load_failure_policy",
    )
    raw_extra_config = transfer.get("extra_config")
    extra_config = require_mapping(
        {} if raw_extra_config is None else raw_extra_config,
        path,
        "serving.kv_transfer.extra_config",
    )
    try:
        json.dumps(extra_config)
    except (TypeError, ValueError) as e:
        schema_error(
            path,
            "serving.kv_transfer.extra_config",
            f"must be JSON-serializable ({e})",
        )
    return {
        "connector": connector,
        "load_failure_policy": policy,
        "extra_config": extra_config,
    }


def normalize_role(
    value: object, path: str, index: int, common_env: dict,
) -> dict:
    location = f"serving.roles[{index}]"
    role = require_mapping(value, path, location)
    reject_unknown_fields(role, ROLE_FIELDS, path, location)

    role_name = role.get("role")
    if role_name not in {"prefill", "decode"}:
        schema_error(path, f"{location}.role", "must be 'prefill' or 'decode'")
    count = positive_int(role.get("count"), path, f"{location}.count")
    nodes = positive_int(
        role.get("nodes_per_instance"),
        path,
        f"{location}.nodes_per_instance",
    )
    gpus = positive_int(
        role.get("gpus_per_node"), path, f"{location}.gpus_per_node",
    )
    tensor_parallel_size = positive_int(
        role.get("tensor_parallel_size", 1),
        path,
        f"{location}.tensor_parallel_size",
    )
    if tensor_parallel_size > gpus:
        schema_error(
            path,
            f"{location}.tensor_parallel_size",
            f"must stay within one node ({gpus} GPUs)",
        )
    if gpus % tensor_parallel_size:
        schema_error(
            path,
            f"{location}.tensor_parallel_size",
            f"must divide gpus_per_node ({gpus})",
        )
    local_dp_size = gpus // tensor_parallel_size
    base_port = port_number(
        role.get("base_port", 8000), path, f"{location}.base_port",
    )

    expected_kv_role = (
        "kv_producer" if role_name == "prefill" else "kv_consumer"
    )
    kv_role = role.get("kv_role", expected_kv_role)
    if kv_role != expected_kv_role:
        schema_error(
            path,
            f"{location}.kv_role",
            f"must be {expected_kv_role!r} for the {role_name} role",
        )

    serve_argv = split_argv(
        role.get("serve_args"), path, f"{location}.serve_args",
    )
    reject_orchestration_flags(serve_argv, path, f"{location}.serve_args")

    raw_role_env = role.get("env")
    role_env = normalize_env(
        {} if raw_role_env is None else raw_role_env,
        path,
        f"{location}.env",
    )
    return {
        "role": role_name,
        "count": count,
        "nodes_per_instance": nodes,
        "gpus_per_node": gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "local_dp_size": local_dp_size,
        "dp_size": nodes * local_dp_size,
        "base_port": base_port,
        "kv_role": kv_role,
        "serve_argv": serve_argv,
        "env": {**common_env, **role_env},
        "health_check": normalize_health_check(
            role.get("health_check"), path, f"{location}.health_check",
        ),
    }


def normalize_router(
    value: object, path: str, roles: list,
) -> dict:
    router = require_mapping(value, path, "serving.router")
    reject_unknown_fields(router, ROUTER_FIELDS, path, "serving.router")

    command = router.get("command", ["vllm-router"])
    if isinstance(command, str):
        command_argv = split_argv(command, path, "serving.router.command")
    elif isinstance(command, list):
        if not command or any(not isinstance(arg, str) or not arg for arg in command):
            schema_error(
                path,
                "serving.router.command",
                "must contain non-empty string arguments",
            )
        command_argv = list(command)
    else:
        schema_error(
            path,
            "serving.router.command",
            "must be a string or list of strings",
        )
    if not command_argv:
        schema_error(path, "serving.router.command", "must not be empty")

    repo_path = nonempty_string(
        router.get("repo_path"), path, "serving.router.repo_path",
    )
    revision = router.get("revision", "")
    if not isinstance(revision, str):
        schema_error(path, "serving.router.revision", "must be a string")

    policy = nonempty_string(
        router.get("policy", "round_robin"), path, "serving.router.policy",
    )
    listen_host = nonempty_string(
        router.get("listen_host", "0.0.0.0"),
        path,
        "serving.router.listen_host",
    )
    client_host = nonempty_string(
        router.get("client_host", "127.0.0.1"),
        path,
        "serving.router.client_host",
    )
    port = port_number(router.get("port", 31000), path, "serving.router.port")
    nofile_limit = router.get("nofile_limit")
    if nofile_limit is not None:
        nofile_limit = positive_int(
            nofile_limit, path, "serving.router.nofile_limit",
        )

    metrics_port = router.get("metrics_port")
    if metrics_port is not None:
        metrics_port = port_number(
            metrics_port, path, "serving.router.metrics_port",
        )
        if metrics_port == port:
            schema_error(
                path,
                "serving.router.metrics_port",
                "must differ from serving.router.port",
            )

    intra_node_dp = positive_int(
        router.get("intra_node_data_parallel_size", 1),
        path,
        "serving.router.intra_node_data_parallel_size",
    )
    local_dp_sizes = {role["local_dp_size"] for role in roles}
    if intra_node_dp > 1 and local_dp_sizes != {intra_node_dp}:
        schema_error(
            path,
            "serving.router.intra_node_data_parallel_size",
            "values greater than 1 must equal local_dp_size for every role "
            f"(have {sorted(local_dp_sizes)})",
        )

    prefill_endpoints = router.get("prefill_endpoints", "all_instances")
    if prefill_endpoints != "all_instances":
        schema_error(
            path,
            "serving.router.prefill_endpoints",
            "must be 'all_instances' for schema version 1",
        )
    decode_endpoints = router.get("decode_endpoints", "all_nodes")
    if decode_endpoints != "all_nodes":
        schema_error(
            path,
            "serving.router.decode_endpoints",
            "must be 'all_nodes' for schema version 1",
        )

    normalized = {
        "repo_path": repo_path,
        "revision": revision,
        "command_argv": command_argv,
        "policy": policy,
        "listen_host": listen_host,
        "client_host": client_host,
        "port": port,
        "intra_node_data_parallel_size": intra_node_dp,
        "prefill_endpoints": prefill_endpoints,
        "decode_endpoints": decode_endpoints,
        "env": normalize_env(
            {} if router.get("env") is None else router.get("env"),
            path,
            "serving.router.env",
        ),
        "health_check": normalize_health_check(
            router.get("health_check"),
            path,
            "serving.router.health_check",
            default_path="/readiness",
        ),
    }
    if metrics_port is not None:
        normalized["metrics_port"] = metrics_port
    if nofile_limit is not None:
        normalized["nofile_limit"] = nofile_limit
    return normalized


def normalize_serving(
    serving_value: object,
    data: dict,
    profile: dict,
    image: str,
    path: str,
) -> dict:
    """Validate schema-v1 Slurm PD serving and return launcher-ready JSON data."""
    serving = require_mapping(serving_value, path, "serving")
    reject_unknown_fields(serving, SERVING_FIELDS, path, "serving")

    version = serving.get("version")
    if version != 1 or isinstance(version, bool):
        schema_error(path, "serving.version", "must be integer 1")
    if serving.get("mode") != "pd_disagg":
        schema_error(path, "serving.mode", "must be 'pd_disagg'")
    if serving.get("launcher") != "slurm":
        schema_error(path, "serving.launcher", "must be 'slurm'")

    vllm = require_mapping(data.get("vllm") or {}, path, "vllm")
    model = nonempty_string(vllm.get("model"), path, "vllm.model")
    legacy_serve_args = vllm.get("serve_args") or ""
    if not isinstance(legacy_serve_args, str):
        schema_error(path, "vllm.serve_args", "must be a string")
    if legacy_serve_args.strip():
        schema_error(
            path,
            "vllm.serve_args",
            "must be empty when serving.mode is 'pd_disagg'; use "
            "serving.common_serve_args and serving.roles[].serve_args",
        )

    profile_env = normalize_env(
        {} if profile.get("env") is None else profile.get("env"),
        path,
        "gpu profile env",
    )
    workload_env = normalize_env(
        {} if vllm.get("env") is None else vllm.get("env"),
        path,
        "vllm.env",
    )
    common_env = {**profile_env, **workload_env}
    if "HF_HOME" not in common_env and profile.get("hf_home"):
        common_env["HF_HOME"] = str(profile["hf_home"])

    common_argv = split_argv(
        serving.get("common_serve_args"),
        path,
        "serving.common_serve_args",
    )
    reject_orchestration_flags(
        common_argv, path, "serving.common_serve_args",
    )

    raw_roles = serving.get("roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        schema_error(path, "serving.roles", "must be a non-empty list")
    roles_by_name = {}
    for i, raw_role in enumerate(raw_roles):
        role = normalize_role(raw_role, path, i, common_env)
        role_name = role["role"]
        if role_name in roles_by_name:
            schema_error(
                path,
                "serving.roles",
                f"must contain exactly one {role_name!r} role",
            )
        roles_by_name[role_name] = role
    missing_roles = {"prefill", "decode"} - set(roles_by_name)
    if missing_roles:
        schema_error(
            path,
            "serving.roles",
            "must contain exactly one 'prefill' and one 'decode' role",
        )
    roles = [roles_by_name["prefill"], roles_by_name["decode"]]

    per_node_gpu_counts = {role["gpus_per_node"] for role in roles}
    if len(per_node_gpu_counts) != 1:
        schema_error(
            path,
            "serving.roles",
            "must use the same gpus_per_node in one Slurm allocation",
        )
    gpus_per_node = next(iter(per_node_gpu_counts))
    total_nodes = sum(
        role["count"] * role["nodes_per_instance"] for role in roles
    )
    total_gpus = sum(
        role["count"] * role["nodes_per_instance"] * role["gpus_per_node"]
        for role in roles
    )
    declared_gpus = positive_int(
        data.get("num_gpus"), path, "num_gpus",
    )
    if declared_gpus != total_gpus:
        schema_error(
            path,
            "num_gpus",
            f"is {declared_gpus}, but serving.roles require {total_gpus}",
        )

    return {
        "version": version,
        "mode": "pd_disagg",
        "launcher": "slurm",
        "model": model,
        "image": image,
        "total_nodes": total_nodes,
        "total_gpus": total_gpus,
        "gpus_per_node": gpus_per_node,
        "common_env": common_env,
        "common_argv": common_argv,
        "slurm": normalize_slurm(serving.get("slurm"), path),
        "kv_transfer": normalize_kv_transfer(
            serving.get("kv_transfer"), path,
        ),
        "roles": roles,
        "router": normalize_router(
            serving.get("router"), path, roles,
        ),
    }


def main(path: str) -> None:
    with open(path) as f:
        data = yaml.safe_load(f)

    gpu = data.get("gpu")
    if not gpu:
        sys.exit(f"{path}: missing required 'gpu' field")
    profile = load_profile(gpu, path)
    vllm = data.get("vllm") or {}
    lm_eval = data.get("lm_eval") or {}
    bench = data.get("vllm_bench") or {}

    tasks = lm_eval.get("tasks") or []
    bfcl = data.get("bfcl") or {}
    bench_configs = bench.get("configs") or []

    if not tasks and not bench_configs and not bfcl:
        sys.exit(
            f"{path}: workload must define at least one of lm_eval, vllm_bench, or bfcl"
        )

    if tasks:
        validate_tasks(tasks, path)

    serve_args = vllm.get("serve_args") or ""
    if bfcl:
        validate_bfcl(bfcl, serve_args, path)

    image, vllm_commit = resolve_image(vllm, profile)
    env = {**(profile.get("env") or {}), **(vllm.get("env") or {})}
    if "HF_HOME" not in env and profile.get("hf_home"):
        env["HF_HOME"] = profile["hf_home"]

    serving = None
    if "serving" in data:
        serving = normalize_serving(
            data.get("serving"), data, profile, image, path,
        )

    metadata = bench.get("metadata") or {}
    tp = metadata.get("tp")
    if tp is None:
        tp = parse_tp(serve_args)
    bench_gpu_count = metadata.get("total_gpus")
    if bench_gpu_count is None:
        bench_gpu_count = serving["total_gpus"] if serving is not None else tp
    bench_gpu_count = positive_int(
        bench_gpu_count, path, "vllm_bench.metadata.total_gpus",
    )
    bench_disagg = boolean(
        metadata.get("disagg", serving is not None),
        path,
        "vllm_bench.metadata.disagg",
    )
    bench_is_multinode = boolean(
        metadata.get(
            "is_multinode",
            serving is not None and serving["total_nodes"] > 1,
        ),
        path,
        "vllm_bench.metadata.is_multinode",
    )

    emit("NAME", data.get("name", ""))
    emit("IMAGE", image)
    emit("VLLM_COMMIT", vllm_commit)
    emit("MODEL", vllm.get("model", ""))
    emit("SERVE_ARGS", serve_args)
    emit("SERVER_RUNTIME", profile.get("server_runtime", "docker"))
    emit("ENV", "\n".join(f"{k}={fmt(v)}" for k, v in env.items()))
    if serving is not None:
        emit("SERVING_MODE", serving["mode"])
        emit(
            "SERVING_JSON",
            json.dumps(
                serving,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    emit("LM_EVAL_TASKS_TSV", task_tsv(tasks, lm_eval.get("model_args") or {}))
    emit("VLLM_BENCH_TSV", bench_tsv(bench_configs, path))
    emit("BFCL_TSV", bfcl_tsv(bfcl) if bfcl else "")
    emit("BENCH_DEVICE", metadata.get("device") or gpu.lower())
    emit("BENCH_TP", tp)
    emit("BENCH_GPU_COUNT", bench_gpu_count)
    emit("BENCH_DISAGG", str(bench_disagg).lower())
    emit("BENCH_IS_MULTINODE", str(bench_is_multinode).lower())
    emit(
        "BENCH_PRECISION",
        metadata.get("precision") or precision_from_model(vllm.get("model") or ""),
    )


if __name__ == "__main__":
    main(sys.argv[1])
