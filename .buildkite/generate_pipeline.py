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

Writes pipeline YAML to stdout for ``buildkite-agent pipeline upload``.
"""

import glob
import os
import sys

import yaml

FULL_SETUP_COMMANDS = [
    "python3 -m venv .venv",
    ". .venv/bin/activate"
    " && (python -m ensurepip --upgrade --default-pip 2>/dev/null"
    " || curl -fsSL https://bootstrap.pypa.io/get-pip.py | python)"
    " && python -m pip install --upgrade 'lm-eval[api]' pyyaml",
]

BENCH_ONLY_SETUP_COMMANDS = [
    "python3 -m venv .venv",
    ". .venv/bin/activate"
    " && (python -m ensurepip --upgrade --default-pip 2>/dev/null"
    " || curl -fsSL https://bootstrap.pypa.io/get-pip.py | python)"
    " && python -m pip install --upgrade pyyaml",
]

RUN_TEMPLATE = (
    'export HF_HOME="$(pwd)/.hf-cache" PATH="$(pwd)/.venv/bin:$HOME/.local/bin:$PATH"'
    " && ./lib/run.sh {path}"
)

DEFAULT_TIMEOUT = 120
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "gpu_profiles.yaml")

GPU_EMOJI = {
    "H200": ":h200:",
    "B200": ":b200:",
    "A100": ":a100:",
}


def is_truthy(value):
    return str(value or "").lower() in {"1", "true", "yes"}


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


def make_step(path, data, profiles):
    name = data.get("name", os.path.basename(path).removesuffix(".yaml"))
    gpu = data.get("gpu")
    if not gpu:
        sys.exit(f"{path}: missing required 'gpu' field")
    profile = profiles.get(gpu)
    if not profile:
        sys.exit(f"{path}: unknown gpu {gpu!r} (expected one of {', '.join(profiles)})")
    queue = profile["queue"]
    timeout = data.get("timeout_in_minutes", DEFAULT_TIMEOUT)
    emoji = GPU_EMOJI.get(gpu, ":buildkite:")
    setup_commands = (
        BENCH_ONLY_SETUP_COMMANDS
        if is_truthy(os.environ.get("BENCH_ONLY"))
        else FULL_SETUP_COMMANDS
    )
    step = {
        "label": f"{emoji} {name}",
        "agents": {"queue": queue},
        "timeout_in_minutes": timeout,
        "commands": setup_commands + [RUN_TEMPLATE.format(path=path)],
        "artifact_paths": ["results/**/*"],
    }
    step_env = {
        k: os.environ[k]
        for k in ("VLLM_IMAGE", "VLLM_COMMIT", "BENCH_ONLY")
        if os.environ.get(k)
    }
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
