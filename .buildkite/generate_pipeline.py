#!/usr/bin/env python3
"""Generate Buildkite pipeline steps from workload YAML files.

Always emits one GPU-profiled step per selected workload. Selection rules:

  WORKLOADS env var set?  → run exactly those paths (comma- or newline-
                             separated; resolved against workloads/*.yaml)
  Otherwise               → run every workload with ``nightly: true``

Override env vars are propagated to each step:
  VLLM_IMAGE   full docker image URI; overrides workload's vllm.image
  VLLM_COMMIT  commit SHA → vllm/vllm-openai:nightly-<sha> (Docker Hub)
  BENCH_ONLY   when truthy, run vllm bench configs and skip lm_eval tasks

Workloads can also set ``bench_only: true`` to apply BENCH_ONLY to that step
without forcing the whole build to skip lm_eval.

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
    'export HF_HOME="$(pwd)/.hf-cache" PATH="$(pwd)/.venv/bin:$HOME/.local/bin:$PATH"'
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


def resolved_image(data, profile):
    vllm = data.get("vllm") or {}
    override_image = (os.environ.get("VLLM_IMAGE") or "").strip()
    override_commit = (os.environ.get("VLLM_COMMIT") or "").strip()
    custom_repo = (profile.get("image_repo") or "").strip()
    repo = custom_repo or DEFAULT_IMAGE_REPO
    # Don't use VLLM_IMAGE for AMD workloads unless it is a ROCm image
    if override_image and (not custom_repo or "rocm" in override_image.lower()):
        return override_image
    commit = override_commit or commit_from_image(override_image)
    if commit:
        return f"{repo}:nightly-{commit}"
    return vllm.get("image", f"{repo}:nightly")


def b200_k8s_plugin(image, num_gpus, profile=None):
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
                            {"name": "shared", "mountPath": "/mnt/shared"},
                        ],
                        "env": [
                            {"name": "VLLM_USAGE_SOURCE", "value": "ci-test"},
                            {"name": "NCCL_CUMEM_HOST_ENABLE", "value": "0"},
                            {"name": "HF_HOME", "value": "/mnt/shared/hf_cache"},
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
                    {
                        "name": "shared",
                        "hostPath": {"path": "/mnt/shared", "type": "DirectoryOrCreate"},
                    },
                ],
            },
        },
    }


def amd_k8s_plugin(image, num_gpus, profile=None):
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
                    {
                        "name": "hf-cache",
                        "hostPath": {"path": hf_home, "type": "DirectoryOrCreate"},
                    },
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
    if bench_only:
        setup_commands = BENCH_ONLY_SETUP_COMMANDS
    elif has_bfcl:
        setup_commands = [setup_command("'lm-eval[api]' pyyaml bfcl-eval soundfile")]
    else:
        setup_commands = FULL_SETUP_COMMANDS
    agents = {"queue": queue, **(profile.get("agent_tags") or {})}
    step = {
        "label": f"{emoji} {name}",
        "agents": agents,
        "timeout_in_minutes": timeout,
        "commands": setup_commands + [RUN_TEMPLATE.format(path=path)],
        "artifact_paths": ["results/**/*"],
    }
    if profile.get("server_runtime") == "native":
        kind = profile.get("k8s_plugin")
        if not kind:
            sys.exit(
                f"{path}: profile {gpu!r} sets server_runtime: native but no"
                f" k8s_plugin; set one explicitly (have {', '.join(K8S_PLUGINS)})"
            )
        builder = K8S_PLUGINS.get(kind)
        if builder is None:
            sys.exit(f"{path}: unknown k8s_plugin {kind!r} (have {', '.join(K8S_PLUGINS)})")
        image = ecr_pull_through(resolved_image(data, profile))
        step["plugins"] = [builder(image, data.get("num_gpus", 1), profile)]
    step_env = {
        k: os.environ[k]
        for k in ("VLLM_IMAGE", "VLLM_COMMIT", "BENCH_ONLY")
        if os.environ.get(k)
    }
    if bench_only and "BENCH_ONLY" not in step_env:
        step_env["BENCH_ONLY"] = "1"
    if step_env:
        step["env"] = step_env
    return step


def select_workloads(workloads):
    raw = (os.environ.get("WORKLOADS") or "").strip()
    if raw:
        # Accept comma- or newline-separated. Each entry is a workload path
        # (e.g. workloads/qwen3_5_h200.yaml) or a bare name (qwen3_5_h200).
        entries = [e.strip() for e in raw.replace(",", "\n").split("\n") if e.strip()]
        by_path = {w["path"]: w for w in workloads}
        by_stem = {os.path.basename(w["path"]).removesuffix(".yaml"): w for w in workloads}
        selected = []
        for e in entries:
            if e in by_path:
                selected.append(by_path[e])
            elif e in by_stem:
                selected.append(by_stem[e])
            elif (f"workloads/{e}.yaml") in by_path:
                selected.append(by_path[f"workloads/{e}.yaml"])
            else:
                sys.exit(f"WORKLOADS entry {e!r} did not match any file in workloads/")
        return selected
    return [w for w in workloads if w["data"].get("nightly") is True]


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
    steps = [make_step(w["path"], w["data"], profiles) for w in selected]
    print(yaml.dump({"steps": steps}, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
