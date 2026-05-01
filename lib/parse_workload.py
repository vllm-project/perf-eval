"""Read a workload YAML and emit shell exports.

Usage:
    eval "$(python3 lib/parse_workload.py workloads/foo.yaml)"

Sets WORKLOAD_NAME (top-level), WORKLOAD_MODEL/IMAGE/VLLM_COMMIT/SERVE_ARGS
(from the `vllm:` block, override env vars, and GPU profile),
WORKLOAD_SERVER_RUNTIME, WORKLOAD_ENV (newline-separated KEY=VALUE pairs),
WORKLOAD_LM_EVAL_TASKS_TSV, and WORKLOAD_VLLM_BENCH_TSV (bench configs to run
after lm_eval). Each lm_eval TSV line is "name\\tnum_fewshot\\tmodel_args";
each bench TSV line is
"name\\tbackend\\tdataset\\tinput_len\\toutput_len\\tnum_prompts\\tmax_concurrency\\tspeed_bench_dataset_subset\\tspeed_bench_category".
`lm_eval.model_args` (workload-level) is merged under each task's `model_args` block.

Also sets WORKLOAD_BENCH_DEVICE/TP/PRECISION — metadata used to compute
per-GPU metrics and tag rows in the perf dashboard. Auto-derived from the
GPU profile and serve_args; override via `vllm_bench.metadata`.

Machine-specific defaults (image, HF_HOME, env) come from gpu_profiles.yaml,
keyed by the workload's `gpu` field. The profile's `env:` block is merged
under the workload's `vllm.env` (workload values win on conflict). The
workload can override `vllm.image` or `vllm.env.HF_HOME` if needed.

Image precedence (highest first):
  1. `VLLM_IMAGE` env var (full image URI)
  2. `VLLM_COMMIT` env var (commit SHA → public ECR vllm-openai image)
  3. `vllm.image` from the workload YAML
  4. `vllm/vllm-openai:latest` (default)

Per-task top-level fields are limited to `name`, `num_fewshot`, and
`model_args`; any other top-level field is rejected with a hint to move
it under `model_args:`.

Each name in `lm_eval.tasks` is validated against lm_eval's task registry
(tasks + groups + tags); unknown names exit non-zero before the server
is started. When BENCH_ONLY is truthy, task registry validation is skipped
because lm_eval tasks will not run.
"""

import os
import re
import shlex
import sys

import yaml

TOP_FIELDS = ("name",)
VLLM_FIELDS = ("model", "image", "serve_args")
TASK_FIELDS = {"name", "num_fewshot", "model_args"}
BENCH_FIELDS = {
    "name",
    "backend",
    "dataset",
    "input_len",
    "output_len",
    "num_prompts",
    "max_concurrency",
    "speed_bench_dataset_subset",
    "speed_bench_category",
}
BENCH_REQUIRED = ("name", "input_len", "output_len", "num_prompts", "max_concurrency")

# When VLLM_COMMIT is set without VLLM_IMAGE, build the image URI from this
# template. vLLM publishes per-commit nightly images to Docker Hub as
# vllm/vllm-openai:nightly-<sha>.
COMMIT_IMAGE_TEMPLATE = "vllm/vllm-openai:nightly-{commit}"


def commit_from_image(image: str) -> str:
    slash = image.rfind("/")
    colon = image.rfind(":")
    if colon <= slash:
        return ""
    tag = image[colon + 1 :].split("@", 1)[0]
    nightly = re.match(r"nightly-([0-9a-f]{7,40})(?:[-_.].*)?$", tag, re.IGNORECASE)
    if nightly:
        return nightly.group(1)
    sha = re.search(r"(?:^|[-_.])([0-9a-f]{12,40})(?:$|[-_.])", tag, re.IGNORECASE)
    return sha.group(1) if sha else ""


def fmt(v: object) -> str:
    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return "None"
    return str(v)


def serialize(args: dict) -> str:
    return ",".join(f"{k}={fmt(v)}" for k, v in args.items())


def optional_bench_field(config: dict, key: str) -> str:
    value = config.get(key)
    if value is None or value == "":
        return "-"
    return str(value)


def known_task_names() -> set:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        sys.exit(f"cannot validate task names: lm_eval not importable ({e})")
    return set(TaskManager().all_tasks)


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def load_profile(gpu: str, workload_path: str) -> dict:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(workload_path)))
    profiles_path = os.path.join(repo_root, "lib", "gpu_profiles.yaml")
    with open(profiles_path) as f:
        profiles = yaml.safe_load(f)
    if gpu not in profiles:
        sys.exit(f"unknown gpu {gpu!r} in {profiles_path} (have {', '.join(profiles)})")
    return profiles[gpu]


def main(path: str) -> None:
    with open(path) as f:
        data = yaml.safe_load(f)

    gpu = data.get("gpu")
    if not gpu:
        sys.exit(f"{path}: missing required 'gpu' field")
    profile = load_profile(gpu, path)

    lm_eval = data.get("lm_eval") or {}
    tasks = lm_eval.get("tasks") or []
    if not tasks:
        sys.exit(f"{path}: missing or empty `lm_eval.tasks`")
    validate_tasks = not env_truthy("BENCH_ONLY")
    known = known_task_names() if validate_tasks else set()
    for t in tasks:
        extra = set(t) - TASK_FIELDS
        if extra:
            sys.exit(
                f"{path}: task {t['name']!r} has unsupported top-level fields "
                f"{sorted(extra)}; move them under `model_args:`"
            )
        if validate_tasks and t["name"] not in known:
            sys.exit(f"{path}: unknown lm_eval task {t['name']!r}")

    vllm = data.get("vllm") or {}
    for key in TOP_FIELDS:
        print(f"WORKLOAD_{key.upper()}={shlex.quote(str(data.get(key, '')))}")
    override_image = (os.environ.get("VLLM_IMAGE") or "").strip()
    override_commit = (os.environ.get("VLLM_COMMIT") or "").strip()
    if override_image:
        image = override_image
        vllm_commit = commit_from_image(image)
    elif override_commit:
        image = COMMIT_IMAGE_TEMPLATE.format(commit=override_commit)
        vllm_commit = override_commit
    else:
        image = vllm.get("image", "vllm/vllm-openai:latest")
        vllm_commit = commit_from_image(str(image))
    print(f"WORKLOAD_IMAGE={shlex.quote(image)}")
    print(f"WORKLOAD_VLLM_COMMIT={shlex.quote(vllm_commit)}")
    for key in ("model", "serve_args"):
        print(f"WORKLOAD_{key.upper()}={shlex.quote(str(vllm.get(key, '')))}")
    env = {**(profile.get("env") or {}), **(vllm.get("env") or {})}
    if "HF_HOME" not in env and profile.get("hf_home"):
        env["HF_HOME"] = profile["hf_home"]
    env_lines = "\n".join(f"{k}={fmt(v)}" for k, v in env.items())
    print(f"WORKLOAD_SERVER_RUNTIME={shlex.quote(str(profile.get('server_runtime', 'docker')))}")
    print(f"WORKLOAD_ENV={shlex.quote(env_lines)}")
    base_args = lm_eval.get("model_args") or {}
    lines = []
    for t in tasks:
        merged = {**base_args, **(t.get("model_args") or {})}
        lines.append(f"{t['name']}\t{t.get('num_fewshot', 0)}\t{serialize(merged)}")
    print(f"WORKLOAD_LM_EVAL_TASKS_TSV={shlex.quote(chr(10).join(lines))}")

    # vllm_bench is optional; emit empty TSV when absent so run.sh can iterate.
    bench = data.get("vllm_bench") or {}
    bench_configs = bench.get("configs") or []
    bench_names = set()
    bench_lines = []
    for c in bench_configs:
        extra = set(c) - BENCH_FIELDS
        if extra:
            sys.exit(
                f"{path}: vllm_bench config {c.get('name')!r} has unsupported fields "
                f"{sorted(extra)}; allowed: {sorted(BENCH_FIELDS)}"
            )
        for k in BENCH_REQUIRED:
            if c.get(k) is None:
                sys.exit(f"{path}: vllm_bench config {c.get('name')!r} missing required field {k!r}")
        if c["name"] in bench_names:
            sys.exit(f"{path}: duplicate vllm_bench config name {c['name']!r}")
        bench_names.add(c["name"])
        bench_lines.append(
            f"{c['name']}\t{optional_bench_field(c, 'backend')}\t"
            f"{c.get('dataset', 'random')}\t{c['input_len']}\t{c['output_len']}\t"
            f"{c['num_prompts']}\t{c['max_concurrency']}\t"
            f"{optional_bench_field(c, 'speed_bench_dataset_subset')}\t"
            f"{optional_bench_field(c, 'speed_bench_category')}"
        )
    print(f"WORKLOAD_VLLM_BENCH_TSV={shlex.quote(chr(10).join(bench_lines))}")

    # Bench ingest metadata (device, tp, precision). Auto-derive defaults from
    # the GPU profile and serve_args; allow `vllm_bench.metadata` to override.
    metadata = bench.get("metadata") or {}
    device = metadata.get("device") or gpu.lower()
    tp = metadata.get("tp")
    if tp is None:
        tp = parse_tp(vllm.get("serve_args") or "")
    precision = metadata.get("precision") or precision_from_model(vllm.get("model") or "")
    print(f"WORKLOAD_BENCH_DEVICE={shlex.quote(str(device))}")
    print(f"WORKLOAD_BENCH_TP={shlex.quote(str(tp))}")
    print(f"WORKLOAD_BENCH_PRECISION={shlex.quote(str(precision))}")


def parse_tp(serve_args: str) -> int:
    """Best-effort parse of the effective parallel-degree (TP * DP) from serve_args.

    `vllm bench serve` reports aggregate throughput; we divide by this to get
    per-GPU metrics for the dashboard. Falls back to 1 if nothing matches.
    """
    toks = serve_args.split()
    def find(*names):
        for i, t in enumerate(toks):
            if t in names and i + 1 < len(toks):
                try:
                    return int(toks[i + 1])
                except ValueError:
                    return None
            if "=" in t:
                key, _, val = t.partition("=")
                if key in names:
                    try:
                        return int(val)
                    except ValueError:
                        return None
        return None
    tp = find("--tensor-parallel-size", "-tp", "--tp") or 1
    dp = find("--data-parallel-size", "-dp", "--dp") or 1
    return tp * dp


def precision_from_model(model: str) -> str:
    """Best-effort derivation: look at the HF repo name for a precision suffix."""
    name = model.lower()
    for marker in ("fp4", "fp8", "int4", "int8", "bf16", "fp16"):
        if marker in name:
            return marker
    return "bf16"


if __name__ == "__main__":
    main(sys.argv[1])
