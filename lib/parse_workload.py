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
VLLM_FIELDS = {
    "model", "image", "serve_args", "env", "attention_backends", "moe_backends",
}
KNOWN_ATTENTION_BACKENDS = {
    "default",
    "FLASH_ATTN", "FLASHINFER", "XFORMERS", "TRITON_ATTN",
    "TRITON_MLA", "ROCM_FLASH", "PAGED_ATTENTION", "ROCM_AITER_FA", "ROCM_AITER_MHA",
    "ROCM_AITER_UNIFIED_ATTN", "ROCM_ATTN", "ROCM_AITER_MLA",
    "ROCM_AITER_MLA_SPARSE", "ROCM_AITER_TRITON_MLA"
}
KNOWN_MOE_BACKENDS = {
    "default",
    "AITER", "TRITON", "FUSED_MOE",
    "AITER_MXFP4_BF16", "AITER_MXFP4_FP8", "TRITON_UNFUSED",
    "AITER_MXFP4_MXFP4"
}

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
        print(f"WARNING: {path}: bfcl without --tool-call-parser in serve_args; "
              "some models may need it for function-calling", file=sys.stderr)


#def bfcl_tsv(bfcl: dict) -> str:
#    cats = bfcl.get("test_categories") or []
#    num_threads = bfcl.get("num_threads", 8)
#    temperature = bfcl.get("temperature", 0.001)
#    return "\n".join(f"{cat}\t{num_threads}\t{temperature}" for cat in cats)


def validate_attention_backends(backends: list, path: str) -> None:
    if not backends:
        sys.exit(f"{path}: vllm.attention_backends must not be empty if specified")
    for b in backends:
        if b not in KNOWN_ATTENTION_BACKENDS:
            sys.exit(
                f"{path}: unknown attention backend {b!r}; "
                f"known: {', '.join(sorted(KNOWN_ATTENTION_BACKENDS))}"
            )
    if len(backends) != len(set(backends)):
        sys.exit(f"{path}: duplicate entries in vllm.attention_backends")
        print(
            f"WARNING: {path}: bfcl without --tool-call-parser in serve_args; "
            "some models may need it for function-calling",
            file=sys.stderr,
        )
    


def validate_moe_backends(backends: list, path: str) -> None:
    if not backends:
        sys.exit(f"{path}: vllm.moe_backends must not be empty if specified")
    for b in backends:
        if b not in KNOWN_MOE_BACKENDS:
            sys.exit(
                f"{path}: unknown moe backend {b!r}; "
                f"known: {', '.join(sorted(KNOWN_MOE_BACKENDS))}"
            )
    if len(backends) != len(set(backends)):
        sys.exit(f"{path}: duplicate entries in vllm.moe_backends")


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

    attention_backends = vllm.get("attention_backends") or []
    if attention_backends:
        validate_attention_backends(attention_backends, path)

    moe_backends = vllm.get("moe_backends") or []
    if moe_backends:
        validate_moe_backends(moe_backends, path)
        if re.search(r"(^|[\s])--moe-backend(\s|=)", serve_args):
            sys.exit(
                f"{path}: vllm.moe_backends and --moe-backend in serve_args are mutually exclusive; "
                "remove --moe-backend from serve_args when using the moe_backends sweep"
            )

    if attention_backends and moe_backends:
        sys.exit(
            f"{path}: vllm.attention_backends and vllm.moe_backends are mutually exclusive; "
            "run attention and moe sweeps in separate workloads"
        )

    image, vllm_commit = resolve_image(vllm, profile)
    env = {**(profile.get("env") or {}), **(vllm.get("env") or {})}
    if "HF_HOME" not in env and profile.get("hf_home"):
        env["HF_HOME"] = profile["hf_home"]

    metadata = bench.get("metadata") or {}
    tp = metadata.get("tp")
    if tp is None:
        tp = parse_tp(serve_args)

    emit("NAME", data.get("name", ""))
    emit("IMAGE", image)
    emit("VLLM_COMMIT", vllm_commit)
    emit("MODEL", vllm.get("model", ""))
    emit("SERVE_ARGS", serve_args)
    emit("SERVER_RUNTIME", profile.get("server_runtime", "docker"))
    emit("ENV", "\n".join(f"{k}={fmt(v)}" for k, v in env.items()))
    emit("LM_EVAL_TASKS_TSV", task_tsv(tasks, lm_eval.get("model_args") or {}))
    emit("VLLM_BENCH_TSV", bench_tsv(bench_configs, path))
    emit("BFCL_TSV", bfcl_tsv(bfcl) if bfcl else "")
    emit("BENCH_DEVICE", metadata.get("device") or gpu.lower())
    emit("BENCH_TP", tp)
    #emit("BENCH_PRECISION", metadata.get("precision") or precision_from_model(vllm.get("model") or ""))
    # One backend per line; empty string when not specified (run.sh treats this
    # as a single "default" pass with no override).
    emit("ATTENTION_BACKENDS", "\n".join(attention_backends))
    emit("MOE_BACKENDS", "\n".join(moe_backends))
    emit(
        "BENCH_PRECISION",
        metadata.get("precision") or precision_from_model(vllm.get("model") or ""),
    )


if __name__ == "__main__":
    main(sys.argv[1])
