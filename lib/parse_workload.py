"""Read a workload YAML and emit `WORKLOAD_*` shell exports for run.sh.

Usage: eval "$(python3 lib/parse_workload.py workloads/foo.yaml)"

The README documents the recipe schema. This script validates it and
projects it into shell variables: top-level metadata, server config
(image, model, serve_args, env, runtime), the lm_eval task list, the
vllm_bench config list, and bench ingest metadata (device/tp/precision).

Image precedence: VLLM_IMAGE_CUDA / VLLM_IMAGE_ROCM (whichever matches the
workload's GPU) > VLLM_IMAGE > VLLM_COMMIT > workload `vllm.image` >
`vllm/vllm-openai:latest`. A build that pins images per platform but not this
workload's, and sets no VLLM_IMAGE, has nothing to run here and is an error.
When BENCH_ONLY is truthy, lm_eval task names are not validated against the
registry (because they will not run).
"""

from __future__ import annotations

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
    "num_prompts", "max_concurrency", "repetitions", "args",
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
AIPERF_FIELDS = {"name", "args"}
AIPERF_REQUIRED = ("name",)
AIPERF_RESERVED_ARGS = {
    "model", "tokenizer", "url", "api-key", "output-artifact-dir",
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


def platform_of(profile: dict) -> str:
    """The platform a profile runs, from the repo its images come from."""
    repo = (profile.get("image_repo") or "").strip() or "vllm/vllm-openai"
    return "ROCM" if "rocm" in repo.lower() else "CUDA"


def platform_image(profile: dict) -> str:
    """VLLM_IMAGE_CUDA / VLLM_IMAGE_ROCM — this platform's image, if pinned.

    For a build whose platforms are separate artifacts with unrelated tags,
    which nothing else here can name.
    """
    return (os.environ.get(f"VLLM_IMAGE_{platform_of(profile)}") or "").strip()


def pins_only_other_platforms(profile: dict) -> bool:
    """True when the build pins per-platform images, but not this platform's."""
    mine = f"VLLM_IMAGE_{platform_of(profile)}"
    return any(
        (os.environ.get(k) or "").strip()
        for k in ("VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM")
        if k != mine
    )


def resolve_image(vllm: dict, profile: dict) -> tuple[str, str]:
    """Pick the image and commit using VLLM_IMAGE / VLLM_COMMIT / workload.

    A workload that sets ``pin_image: true`` keeps its own ``vllm.image`` even
    when VLLM_IMAGE / VLLM_COMMIT or a platform pin are set. Use it for models
    that only exist in a dedicated image (e.g. kimi-k3, minimax-m3) where the
    nightly override would pull an image that cannot serve the model. Failing
    that, VLLM_IMAGE_CUDA / VLLM_IMAGE_ROCM decide their own platform.
    """
    override_image = (os.environ.get("VLLM_IMAGE") or "").strip()
    override_commit = (os.environ.get("VLLM_COMMIT") or "").strip()
    # ROCm images are located at vllm/vllm-openai-rocm. The default
    # images (CUDA) are stored at vllm/vllm-openai
    custom_repo = (profile.get("image_repo") or "").strip()
    repo = custom_repo or "vllm/vllm-openai"
    if vllm.get("pin_image") is True and vllm.get("image"):
        image = vllm["image"]
        return image, commit_from_image(str(image))
    platform_pin = platform_image(profile)
    if platform_pin:
        return platform_pin, override_commit or commit_from_image(platform_pin)
    # This build pins images per platform and didn't pin ours. The generator
    # skips these workloads, so only a direct run.sh gets here.
    if pins_only_other_platforms(profile) and not override_image:
        platform = platform_of(profile)
        sys.exit(f"no {platform} image: set VLLM_IMAGE_{platform} or VLLM_IMAGE")

    # Don't use VLLM_IMAGE for AMD workloads unless it is a ROCm image.
    # A CUDA release image may embed a commit in its tag, but that must not
    # implicitly select an unrelated ROCm nightly for AMD jobs.
    if override_image:
        if not custom_repo or "rocm" in override_image.lower():
            return override_image, override_commit or commit_from_image(override_image)
        if not override_commit:
            image = vllm.get("image", f"{repo}:nightly")
            return image, commit_from_image(str(image))

    commit = override_commit
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


def encode_arg_map(
    args: object, config_name: str, path: str, reserved: set, kind: str
) -> str:
    """Normalize a config `args` map to `--kebab-case` keys and base64-encode it.

    Shared by vllm_bench and aiperf; each passes its own set of wrapper-owned
    reserved options that a workload must not override.
    """
    if args is None:
        args = {}
    if not isinstance(args, dict):
        sys.exit(f"{path}: {kind} config {config_name!r} args must be a map")
    normalized = {}
    for name, value in args.items():
        if not isinstance(name, str) or not normalize_bench_arg_name(name):
            sys.exit(
                f"{path}: {kind} config {config_name!r} args keys must be non-empty strings"
            )
        normalized_name = normalize_bench_arg_name(name)
        if normalized_name in reserved:
            sys.exit(
                f"{path}: {kind} config {config_name!r} args cannot override "
                f"wrapper-owned option --{normalized_name}"
            )
        if normalized_name in normalized:
            sys.exit(
                f"{path}: {kind} config {config_name!r} args contains duplicate "
                f"option --{normalized_name} after normalization"
            )
        normalized[normalized_name] = value
    payload = json.dumps(normalized, separators=(",", ":")).encode()
    return base64.b64encode(payload).decode()


def encode_bench_args(args: object, config_name: str, path: str) -> str:
    return encode_arg_map(args, config_name, path, BENCH_RESERVED_ARGS, "vllm_bench")


def expand_bench_config(c: dict, path: str) -> list:
    """Expand a bench config's concurrency sweep into concrete runs.

    max_concurrency may be a single int or a list of
    them (one run per value). Either way each run's name is suffixed with
    `-conc-<value>`, so the config name is the shape description without the
    concurrency. `num_prompts` is either a single value applied to every run,
    or — only when `max_concurrency` is a list — a list of the same length
    giving a per-concurrency request count. Returns a list of
    (name, num_prompts, max_concurrency) tuples.
    """
    name = c["name"]
    mc = c["max_concurrency"]
    npr = c["num_prompts"]
    is_sweep = isinstance(mc, list)

    concs = mc if is_sweep else [mc]
    if is_sweep and not concs:
        sys.exit(f"{path}: vllm_bench config {name!r} has an empty max_concurrency list")

    if isinstance(npr, list):
        if not is_sweep:
            sys.exit(
                f"{path}: vllm_bench config {name!r} num_prompts may only be a list when "
                f"max_concurrency is a list"
            )
        if len(npr) != len(concs):
            sys.exit(
                f"{path}: vllm_bench config {name!r} num_prompts list has {len(npr)} entries "
                f"but max_concurrency has {len(concs)}"
            )
        nprompts = npr
    else:
        nprompts = [npr] * len(concs)
    for v in nprompts:
        if not (isinstance(v, int) and v > 0):
            sys.exit(
                f"{path}: vllm_bench config {name!r} num_prompts must be a positive integer "
                f"(or a list of them matching max_concurrency)"
            )

    return [
        (f"{name}-conc-{conc}", n, conc)
        for conc, n in zip(concs, nprompts)
    ]


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

        repetitions = c.get("repetitions", 1)
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions < 1
            or repetitions % 2 == 0
        ):
            sys.exit(
                f"{path}: vllm_bench config {c['name']!r} repetitions must be "
                "a positive odd integer"
            )

        def opt(key):
            v = c.get(key)  # noqa: B023
            return str(v) if v not in (None, "") else "-"

        encoded_args = encode_bench_args(c.get("args"), c["name"], path)
        for run_name, nprompts, conc in expand_bench_config(c, path):
            if run_name in seen:
                sys.exit(f"{path}: duplicate vllm_bench config name {run_name!r}")
            seen.add(run_name)
            lines.append(
                "\t".join(
                    [
                        run_name,
                        opt("backend"),
                        str(c.get("dataset", "random")),
                        str(c["input_len"]),
                        str(c["output_len"]),
                        str(nprompts),
                        str(conc),
                        str(repetitions),
                        opt("speed_bench_dataset_subset"),
                        opt("speed_bench_category"),
                        encoded_args,
                    ]
                )
            )
    return "\n".join(lines)


def aiperf_tsv(configs: list, path: str) -> str:
    """Emit one row per aiperf config: name plus base64-encoded arg map.

    The wrapper owns --model, --tokenizer, --url, --api-key, and
    --output-artifact-dir; everything else the profile needs goes under `args`.
    """
    seen = set()
    lines = []
    for c in configs:
        extra = set(c) - AIPERF_FIELDS
        if extra:
            sys.exit(
                f"{path}: aiperf config {c.get('name')!r} has unsupported "
                f"fields {sorted(extra)}; allowed: {sorted(AIPERF_FIELDS)}"
            )
        for k in AIPERF_REQUIRED:
            if c.get(k) is None:
                sys.exit(f"{path}: aiperf config {c.get('name')!r} missing required field {k!r}")
        if c["name"] in seen:
            sys.exit(f"{path}: duplicate aiperf config name {c['name']!r}")
        seen.add(c["name"])
        lines.append(
            "\t".join(
                [
                    c["name"],
                    encode_arg_map(
                        c.get("args"), c["name"], path, AIPERF_RESERVED_ARGS, "aiperf"
                    ),
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

    aiperf = data.get("aiperf") or {}

    tasks = lm_eval.get("tasks") or []
    bfcl = data.get("bfcl") or {}
    bench_configs = bench.get("configs") or []
    aiperf_configs = aiperf.get("configs") or []

    if not tasks and not bench_configs and not bfcl and not aiperf_configs:
        sys.exit(
            f"{path}: workload must define at least one of lm_eval, vllm_bench, "
            f"aiperf, or bfcl"
        )

    if tasks:
        validate_tasks(tasks, path)

    serve_args = vllm.get("serve_args") or ""
    startup_timeout_s = vllm.get("startup_timeout_s", 3600)
    if (
        isinstance(startup_timeout_s, bool)
        or not isinstance(startup_timeout_s, int)
        or startup_timeout_s < 1
    ):
        sys.exit(f"{path}: vllm.startup_timeout_s must be a positive integer")
    if bfcl:
        validate_bfcl(bfcl, serve_args, path)

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
    emit("SERVER_STARTUP_TIMEOUT", startup_timeout_s)
    emit("SERVER_RUNTIME", profile.get("server_runtime", "docker"))
    emit("ENV", "\n".join(f"{k}={fmt(v)}" for k, v in env.items()))
    emit("LM_EVAL_TASKS_TSV", task_tsv(tasks, lm_eval.get("model_args") or {}))
    emit("VLLM_BENCH_TSV", bench_tsv(bench_configs, path))
    emit("AIPERF_TSV", aiperf_tsv(aiperf_configs, path))
    emit("BFCL_TSV", bfcl_tsv(bfcl) if bfcl else "")
    emit("BENCH_DEVICE", metadata.get("device") or gpu.lower())
    emit("BENCH_TP", tp)
    emit(
        "BENCH_PRECISION",
        metadata.get("precision") or precision_from_model(vllm.get("model") or ""),
    )


if __name__ == "__main__":
    main(sys.argv[1])
