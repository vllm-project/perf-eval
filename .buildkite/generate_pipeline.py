#!/usr/bin/env python3
"""Generate Buildkite pipeline steps from workload YAML files.

Always emits one GPU-profiled step per selected workload. Selection rules:

  WORKLOADS env var set?  → run exactly those workloads (comma- or newline-
                             separated paths, file stems, or group names
                             from a workload's ``groups`` list)
  Otherwise               → run every workload with ``nightly: true``

Override env vars are propagated to each step:
  VLLM_IMAGE   full docker image URI; overrides workload's vllm.image
  VLLM_IMAGE_CUDA / VLLM_IMAGE_ROCM
               that platform's image URI; overrides everything below, for a
               build whose CUDA and ROCm images are unrelated artifacts. A
               platform with no pin of its own and no VLLM_IMAGE to fall back
               on has its workloads emitted as skipped steps.
  VLLM_COMMIT  commit SHA → vllm/vllm-openai:nightly-<sha> (Docker Hub)
  BENCH_ONLY   when truthy, run vllm bench configs and skip lm_eval tasks
  NSYS_PROFILE truthy/falsy forces nsys tracing on/off for NVIDIA workloads
               (default: the GPU profile's `nsys`); NSYS_MAX_ITERATIONS and
               NSYS_DELAY_ITERATIONS tune the traced window

Workloads can also set ``bench_only: true`` to apply BENCH_ONLY to that step
without forcing the whole build to skip lm_eval.

A workload whose ``kvv`` block judges another workload's BEAM answers
(``answers_from``) runs after that workload's step. Selecting either one of
the pair selects both.

Writes pipeline YAML to stdout for ``buildkite-agent pipeline upload``.
"""

import glob
import os
import re
import sys

import yaml

def setup_command(packages):
    return (
        "if python3 -m venv .venv; then\n"
        "  . .venv/bin/activate\n"
        "  (python -m ensurepip --upgrade --default-pip 2>/dev/null"
        " || curl -fsSL https://bootstrap.pypa.io/get-pip.py | python)\n"
        f"  python -m pip install --upgrade {packages}\n"
        "else\n"
        "  rm -rf .venv\n"
        "  (python3 -m ensurepip --user --upgrade --default-pip 2>/dev/null"
        " || curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 - --user)\n"
        f"  PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --user --upgrade {packages}\n"
        "fi"
    )


FULL_SETUP_COMMANDS = [setup_command("'lm-eval[api]' pyyaml")]

BENCH_ONLY_SETUP_COMMANDS = [setup_command("pyyaml")]

RUN_TEMPLATE = (
    'INGEST_BEARER_TOKEN="$(buildkite-agent secret get INGEST_BEARER_TOKEN)"'
    " && export INGEST_BEARER_TOKEN"
    ' && export HF_HOME="$(pwd)/.hf-cache" PATH="$(pwd)/.venv/bin:$HOME/.local/bin:$PATH"'
    " && ./lib/run.sh {path}"
)

DEFAULT_TIMEOUT = 120
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "gpu_profiles.yaml")
DEFAULT_IMAGE_REPO = "vllm/vllm-openai"

GPU_EMOJI = {
    "H200": ":h200:",
    "B200": ":b200:",
    "A100": ":a100:",
    "MI355X": ":amd:",
    "MI300X": ":amd:",
}

ECR_PUBLIC_PREFIX = "public.ecr.aws/"
ECR_PULL_THROUGH_CACHE = (
    "936637512419.dkr.ecr.us-west-2.amazonaws.com/vllm-ci-pull-through-cache/"
)


def ecr_pull_through(image):
    """Rewrite public ECR URLs to the private pull-through cache."""
    if image.startswith(ECR_PUBLIC_PREFIX):
        return ECR_PULL_THROUGH_CACHE + image[len(ECR_PUBLIC_PREFIX):]
    return image


def route_ecr_image(image, profile):
    """Use the pull-through cache only on clusters configured for it."""
    if profile.get("ecr_pull_through_cache", True):
        return ecr_pull_through(image)
    return image


def is_truthy(value):
    return str(value or "").lower() in {"1", "true", "yes"}


def commit_from_image(image):
    """Extract a commit SHA from an image tag, if one is embedded."""
    _, sep, tag = image.rpartition(":")
    if not sep:
        return ""
    tag = tag.split("@", 1)[0]
    m = (re.match(r"nightly-([0-9a-f]{7,40})(?:[-_.].*)?$", tag, re.IGNORECASE)
         or re.search(r"(?:^|[-_.])([0-9a-f]{12,40})(?:$|[-_.])", tag, re.IGNORECASE))
    return m.group(1) if m else ""


def platform_of(profile):
    """The platform a profile runs, from the repo its images come from."""
    repo = (profile.get("image_repo") or "").strip() or DEFAULT_IMAGE_REPO
    return "ROCM" if "rocm" in repo.lower() else "CUDA"


def platform_image(profile):
    return (os.environ.get(f"VLLM_IMAGE_{platform_of(profile)}") or "").strip()


def pins_only_other_platforms(profile):
    """True when the build pins per-platform images, but not this platform's."""
    mine = f"VLLM_IMAGE_{platform_of(profile)}"
    return any(
        (os.environ.get(k) or "").strip()
        for k in ("VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM")
        if k != mine
    )


def resolved_image(data, profile):
    """This workload's image, or "" if its platform has none in this build."""
    vllm = data.get("vllm") or {}
    override_image = (os.environ.get("VLLM_IMAGE") or "").strip()
    override_commit = (os.environ.get("VLLM_COMMIT") or "").strip()
    custom_repo = (profile.get("image_repo") or "").strip()
    repo = custom_repo or DEFAULT_IMAGE_REPO
    # A workload that sets pin_image: true keeps its own vllm.image even when
    # VLLM_IMAGE / VLLM_COMMIT or a platform pin are set: the model exists only
    # in that image, so the build's own image could not serve it at all.
    if vllm.get("pin_image") is True and vllm.get("image"):
        return vllm["image"]
    platform_pin = platform_image(profile)
    if platform_pin:
        return platform_pin
    # A build pinning images per platform names every platform it wants run, so
    # with no pin of ours and no VLLM_IMAGE to fall back on there is nothing
    # representative to benchmark: the caller skips this workload.
    if pins_only_other_platforms(profile) and not override_image:
        return ""
    # Don't use VLLM_IMAGE for AMD workloads unless it is a ROCm image.
    # A CUDA release image may embed a commit in its tag, but that must not
    # implicitly select an unrelated ROCm nightly for AMD jobs.
    if override_image:
        if not custom_repo or "rocm" in override_image.lower():
            return override_image
        if not override_commit:
            return vllm.get("image", f"{repo}:nightly")
    commit = override_commit
    if commit:
        return f"{repo}:nightly-{commit}"
    return vllm.get("image", f"{repo}:nightly")


def b200_k8s_plugin(image, num_gpus, profile=None, gpu=None):
    hf_home = profile["hf_home"]  # set per profile in lib/gpu_profiles.yaml
    return {
        "kubernetes": {
            "podSpec": {
                "runtimeClassName": "nvidia",
                "hostNetwork": True,
                "dnsPolicy": "ClusterFirstWithHostNet",
                "imagePullSecrets": [
                    {"name": "k8s-ecr-login-renew-docker-secret"},
                ],
                "containers": [
                    {
                        "image": image,
                        "resources": {"limits": {"nvidia.com/gpu": num_gpus}},
                        "securityContext": {
                            "capabilities": {
                                "add": ["IPC_LOCK", "SYS_RESOURCE"],
                            },
                        },
                        "volumeMounts": [
                            {"name": "devshm", "mountPath": "/dev/shm"},
                            {"name": "raid", "mountPath": "/raid"},
                        ],
                        "env": [
                            {"name": "VLLM_USAGE_SOURCE", "value": "ci-test"},
                            {"name": "NCCL_CUMEM_HOST_ENABLE", "value": "0"},
                            {"name": "HF_HOME", "value": hf_home},
                            {
                                "name": "HF_TOKEN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "hf-token-secret",
                                        "key": "token",
                                    },
                                },
                            },
                        ],
                    },
                ],
                "volumes": [
                    {"name": "devshm", "emptyDir": {"medium": "Memory"}},
                    {
                        "name": "raid",
                        "hostPath": {"path": "/raid", "type": "DirectoryOrCreate"},
                    },
                ],
            },
        },
    }


def hf_cache_volume(gpu, profile):
    """Resolve the Kubernetes volume backing the HF cache for a k8s plugin.

    The source is per-cluster, so it can be overridden by a ``{GPU}_HF_CACHE_VOLUME``
    env var (JSON, minus the ``name`` key) — the same override idiom as
    ``{GPU}_QUEUE`` — or set in the profile's ``hf_cache_volume``. It defaults to
    an ``emptyDir``: the cache is scoped to the pod, so it is reclaimed when the
    benchmark pod exits and can never accumulate on the node's disk. Clusters that
    want a warm, cross-run cache point this at their own PVC (or a real hostPath
    mount) via the override — the pod's ``HF_HOME`` mount path is unchanged either
    way, so only cross-run persistence differs.
    """
    override = (os.environ.get(f"{gpu.upper()}_HF_CACHE_VOLUME") or "").strip()
    if override:
        source = yaml.safe_load(override)
    else:
        source = profile.get("hf_cache_volume") or {"emptyDir": {}}
    return {"name": "hf-cache", **source}


def amd_k8s_plugin(image, num_gpus, profile=None, gpu=None):
    profile = profile or {}
    hf_home = profile.get("hf_home") or "/root/.cache/huggingface"
    return {
        "kubernetes": {
            "podSpecPatch": {
                "imagePullSecrets": [
                    {"name": "docker-config"},
                ],
                "containers": [
                    {
                        "name": "container-0",
                        "image": image,
                        "resources": {"limits": {"amd.com/gpu": num_gpus}},
                        "securityContext": {
                            "seccompProfile": {"type": "Unconfined"},
                            "capabilities": {"add": ["IPC_LOCK", "SYS_PTRACE"]},
                        },
                        "volumeMounts": [
                            {"name": "devshm", "mountPath": "/dev/shm"},
                            {"name": "hf-cache", "mountPath": hf_home},
                        ],
                        "env": [
                            {"name": "VLLM_USAGE_SOURCE", "value": "ci-test"},
                            {"name": "HF_HOME", "value": hf_home},
                            {
                                "name": "HF_TOKEN",
                                "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "hf-token",
                                        "key": "TOKEN",
                                    },
                                },
                            },
                        ],
                    },
                ],
                "volumes": [
                    {"name": "devshm", "emptyDir": {"medium": "Memory"}},
                    hf_cache_volume(gpu, profile),
                ],
            },
        },
    }


K8S_PLUGINS = {
    "nvidia": b200_k8s_plugin,
    "amd": amd_k8s_plugin,
}


def load_profiles():
    with open(PROFILES_PATH) as f:
        return yaml.safe_load(f)


def load_workloads():
    workloads = []
    for path in sorted(glob.glob("workloads/*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        workloads.append({"path": path, "data": data})
    return workloads


def queue_for_gpu(gpu, profile):
    override = (os.environ.get(f"{gpu.upper()}_QUEUE") or "").strip()
    return override or profile["queue"]


def make_step(path, data, profiles):
    name = data.get("name", os.path.basename(path).removesuffix(".yaml"))
    gpu = data.get("gpu")
    if not gpu:
        sys.exit(f"{path}: missing required 'gpu' field")
    profile = profiles.get(gpu)
    if not profile:
        sys.exit(f"{path}: unknown gpu {gpu!r} (expected one of {', '.join(profiles)})")
    queue = queue_for_gpu(gpu, profile)
    timeout = data.get("timeout_in_minutes", DEFAULT_TIMEOUT)
    emoji = GPU_EMOJI.get(gpu, ":buildkite:")
    bench_only = is_truthy(os.environ.get("BENCH_ONLY")) or is_truthy(
        data.get("bench_only")
    )
    has_bfcl = bool(data.get("bfcl"))
    has_kvv = bool(data.get("kvv"))
    has_lm_eval = bool((data.get("lm_eval") or {}).get("tasks"))
    if bench_only:
        setup_commands = BENCH_ONLY_SETUP_COMMANDS
    elif has_bfcl:
        # bfcl-eval pins numpy==1.26.4, and pyarrow 26 needs numpy>=2, so
        # lm_eval cannot be imported. datasets needs pyarrow>=24.
        setup_commands = [setup_command(
            "'lm-eval[api]' pyyaml bfcl-eval soundfile 'pyarrow>=24,<26'"
            + (" uv" if has_kvv else "")
        )]
    elif has_kvv:
        # KVV builds its own uv environment, so it only needs uv here.
        setup_commands = [setup_command(
            "'lm-eval[api]' pyyaml uv" if has_lm_eval else "pyyaml uv"
        )]
    else:
        setup_commands = FULL_SETUP_COMMANDS
    step = {
        "label": f"{emoji} {name}",
        "agents": {"queue": queue},
        "timeout_in_minutes": timeout,
        "commands": setup_commands + [RUN_TEMPLATE.format(path=path)],
        "artifact_paths": ["results/**/*"],
        "retry": data.get("retry", {"automatic": {"limit": 2}}),
    }
    image = resolved_image(data, profile)
    if not image:
        # Skipped, not dropped: a platform the build didn't pin an image for is
        # then visible in the build view instead of quietly missing from it.
        platform = platform_of(profile)
        step["skip"] = f"no {platform} image: set VLLM_IMAGE_{platform}"
    elif profile.get("server_runtime") == "native":
        kind = profile.get("k8s_plugin")
        if not kind:
            sys.exit(
                f"{path}: profile {gpu!r} sets server_runtime: native but no"
                f" k8s_plugin; set one explicitly (have {', '.join(K8S_PLUGINS)})"
            )
        builder = K8S_PLUGINS.get(kind)
        if builder is None:
            sys.exit(f"{path}: unknown k8s_plugin {kind!r} (have {', '.join(K8S_PLUGINS)})")
        step["plugins"] = [
            builder(
                route_ecr_image(image, profile), data.get("num_gpus", 1), profile, gpu
            )
        ]
    step_env = {
        k: os.environ[k]
        for k in ("VLLM_IMAGE", "VLLM_IMAGE_CUDA", "VLLM_IMAGE_ROCM",
                  "VLLM_COMMIT", "BENCH_ONLY", "NSYS_PROFILE",
                  "NSYS_MAX_ITERATIONS", "NSYS_DELAY_ITERATIONS")
        if os.environ.get(k)
    }
    if bench_only and "BENCH_ONLY" not in step_env:
        step_env["BENCH_ONLY"] = "1"
    if step_env:
        step["env"] = step_env
    return step


def workload_stem(path):
    return os.path.basename(path).removesuffix(".yaml")


def workload_groups(w):
    groups = w["data"].get("groups") or []
    if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
        sys.exit(f"{w['path']}: groups must be a list of group names")
    return groups


def workloads_by_name(workloads):
    """Every name a WORKLOADS entry can use, mapped to the workloads it selects:
    a workload's path or file stem selects it; a group selects its members."""
    names = {}
    for w in workloads:
        names[w["path"]] = [w]
        names[workload_stem(w["path"])] = [w]
    files = set(names)
    for w in workloads:
        for group in workload_groups(w):
            if group in files:
                sys.exit(f"{w['path']}: group {group!r} is also a workload name")
            names.setdefault(group, []).append(w)
    return names


def select_workloads(workloads):
    raw = (os.environ.get("WORKLOADS") or "").strip()
    if not raw:
        return [w for w in workloads if w["data"].get("nightly") is True]
    # Comma- or newline-separated workload paths (workloads/qwen3_5_h200.yaml),
    # stems (qwen3_5_h200), or group names.
    names = workloads_by_name(workloads)
    selected = {}
    for entry in filter(None, map(str.strip, raw.replace(",", "\n").split("\n"))):
        if entry not in names:
            sys.exit(
                f"WORKLOADS entry {entry!r} did not match any file in workloads/"
                " or any workload's groups"
            )
        for w in names[entry]:
            # Keyed by path: overlapping groups still run a workload once.
            selected.setdefault(w["path"], w)
    return list(selected.values())


def kvv_answers_from(w):
    return ((w["data"].get("kvv") or {}).get("answers_from") or "").strip()


def kvv_beam_partners(w, by_stem):
    """The workloads w's KVV BEAM job can't run without: for a judge
    (``kvv.answers_from: <workload>``), the workload generating its answers;
    for a generator, the judges reading them. [] for any other workload."""
    source = kvv_answers_from(w)
    if not source:
        stem = workload_stem(w["path"])
        return [j for j in by_stem.values() if kvv_answers_from(j) == stem]
    gen = by_stem.get(source)
    if gen is None:
        sys.exit(f"{w['path']}: kvv.answers_from {source!r} is not a workload in workloads/")
    if "beam_generate" not in ((gen["data"].get("kvv") or {}).get("jobs") or []):
        sys.exit(f"{w['path']}: kvv.answers_from {source!r} has no beam_generate job")
    return [gen]


def with_kvv_beam_partners(selected, workloads):
    """``selected`` plus the other half of every KVV BEAM pair in it, so a
    build never generates answers it doesn't judge, or judges answers it
    didn't generate. Builds without BEAM workloads come back unchanged."""
    by_stem = {workload_stem(w["path"]): w for w in workloads}
    result = {w["path"]: w for w in selected}
    for w in selected:
        for partner in kvv_beam_partners(w, by_stem):
            result.setdefault(partner["path"], partner)
    return list(result.values())


def kvv_step_key(path):
    """Key for the step that runs a workload's beam_generate job. Workload file
    names only use characters Buildkite allows in a key."""
    return f"kvv-{workload_stem(path)}"


def link_kvv_answers(selected, steps):
    """Make each kvv beam_judge step wait for, and read, its answers.

    The judge step depends on the step that generates the answers (even if
    it failed: partial answers are still worth a score, and the generate
    step's own failure stays visible) and is told which artifact to
    download. with_kvv_beam_partners has put both steps in the build.
    """
    by_stem = {workload_stem(w["path"]): (w, s) for w, s in zip(selected, steps)}
    for w, step in zip(selected, steps):
        source = kvv_answers_from(w)
        if not source:
            continue
        gen, gen_step = by_stem[source]
        key = kvv_step_key(gen["path"])
        gen_step["key"] = key
        step["depends_on"] = [{"step": key, "allow_failure": True}]
        gen_name = gen["data"].get("name", source)
        # run_kvv.py copies beam_generate's answers to this path (see
        # kvv_result_names in lib/parse_workload.py).
        step.setdefault("env", {}).update({
            "KVV_ANSWERS_STEP": key,
            "KVV_ANSWERS_ARTIFACT": f"results/{gen_name}/kvv-beam_generate/answers.jsonl",
        })


def images_by_platform(selected, profiles):
   # The image each platform resolved to, and how many workloads it covers.
    counts = {}
    for w in selected:
        profile = profiles[w["data"]["gpu"]]
        key = (platform_of(profile), resolved_image(w["data"], profile))
        counts[key] = counts.get(key, 0) + 1
    return counts


def report_images(selected, profiles):
    # Log the image per platform
    for (platform, image), n in sorted(images_by_platform(selected, profiles).items()):
        what = image or f"skipped, set VLLM_IMAGE_{platform}"
        print(f"{platform}: {what} ({n} workload{'s' if n != 1 else ''})",
              file=sys.stderr)


def main():
    profiles = load_profiles()
    workloads = load_workloads()
    if not workloads:
        sys.exit("no workload files found in workloads/")
    selected = select_workloads(workloads)
    if not selected:
        sys.exit(
            "no workloads to run: WORKLOADS env var not set and no workload"
            " has `nightly: true`"
        )
    selected = with_kvv_beam_partners(selected, workloads)
    steps = [make_step(w["path"], w["data"], profiles) for w in selected]
    link_kvv_answers(selected, steps)
    report_images(selected, profiles)
    print(yaml.dump({"steps": steps}, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
