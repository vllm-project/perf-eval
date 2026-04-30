#!/usr/bin/env python3
"""Generate Buildkite pipeline steps from workload YAML files.

Supports three trigger modes (via TRIGGER_MODE env var):

  nightly      (default) Run every workload with ``nightly: true``.
  manual       Present a Buildkite input step so the user can pick a workload,
               then a follow-up step uploads the real workload step.
  run-selected Run a single workload specified by WORKLOAD env var.
               Used internally by the manual-mode follow-up step.

Writes pipeline YAML to stdout for ``buildkite-agent pipeline upload``.
"""

import glob
import os
import sys

import yaml

SETUP_COMMANDS = [
    "python3 -m ensurepip --upgrade --default-pip 2>/dev/null"
    " || curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3 - --user",
    "python3 -m pip install --user --upgrade 'lm-eval[api]' pyyaml",
]

RUN_TEMPLATE = (
    'export HF_HOME="$PWD/.hf-cache" PATH="$HOME/.local/bin:$PATH"'
    " && ./lib/run.sh {path}"
)


def load_workloads():
    workloads = []
    for path in sorted(glob.glob("workloads/*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        workloads.append({"path": path, "data": data})
    return workloads


DEFAULT_TIMEOUT = 120
PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "lib", "gpu_profiles.yaml")

GPU_EMOJI = {
    "H200": ":h200:",
    "A100": ":a100:",
}


def load_profiles():
    with open(PROFILES_PATH) as f:
        return yaml.safe_load(f)


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
    step = {
        "label": f"{emoji} {name}",
        "agents": {"queue": queue},
        "timeout_in_minutes": timeout,
        "commands": SETUP_COMMANDS + [RUN_TEMPLATE.format(path=path)],
        "artifact_paths": ["results/**/*"],
    }
    # Propagate VLLM_IMAGE / VLLM_COMMIT to the H200 step so parse_workload.py
    # picks them up. Set on the step (not just at build level) so the manual
    # mode follow-up step can pass them through after reading meta-data.
    step_env = {
        k: os.environ[k]
        for k in ("VLLM_IMAGE", "VLLM_COMMIT")
        if os.environ.get(k)
    }
    if step_env:
        step["env"] = step_env
    return step


def nightly(workloads, profiles):
    steps = [
        make_step(w["path"], w["data"], profiles)
        for w in workloads
        if w["data"].get("nightly") is True
    ]
    if not steps:
        sys.exit("TRIGGER_MODE=nightly but no workloads have nightly: true")
    return steps


def manual(workloads):
    if not workloads:
        sys.exit("TRIGGER_MODE=manual but no workload files found in workloads/")
    options = [
        {
            "label": w["data"].get("name", os.path.basename(w["path"]).removesuffix(".yaml")),
            "value": w["path"],
        }
        for w in workloads
    ]
    input_step = {
        "input": "Select a workload to run",
        "fields": [
            {
                "select": "Workload",
                "key": "workload",
                "required": True,
                "options": options,
            },
            {
                "text": "Image override (optional)",
                "key": "image",
                "required": False,
                "hint": "Full docker image URI; overrides workload's vllm.image",
            },
            {
                "text": "vLLM commit (optional)",
                "key": "vllm_commit",
                "required": False,
                "hint": (
                    "Commit SHA → public.ecr.aws/q9t5s3a7/vllm/vllm-openai:<sha>."
                    " Ignored if Image override is set."
                ),
            },
        ],
    }
    followup_step = {
        "label": ":pipeline: upload selected workload",
        "agents": {"queue": "small_cpu_queue_premerge"},
        "commands": [
            "python3 -m pip install --user pyyaml 2>/dev/null || true",
            'WORKLOAD="$(buildkite-agent meta-data get workload)"'
            ' VLLM_IMAGE="$(buildkite-agent meta-data get image --default \'\')"'
            ' VLLM_COMMIT="$(buildkite-agent meta-data get vllm_commit --default \'\')"'
            " TRIGGER_MODE=run-selected python3 .buildkite/generate_pipeline.py"
            " | buildkite-agent pipeline upload",
        ],
    }
    return [input_step, followup_step]


def run_selected(profiles):
    path = os.environ.get("WORKLOAD", "")
    if not path:
        sys.exit("TRIGGER_MODE=run-selected but WORKLOAD env var is not set")
    if not os.path.isfile(path):
        sys.exit(f"workload not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return [make_step(path, data, profiles)]


def main():
    mode = os.environ.get("TRIGGER_MODE", "nightly").lower()

    profiles = load_profiles()

    if mode == "nightly":
        steps = nightly(load_workloads(), profiles)
    elif mode == "manual":
        steps = manual(load_workloads())
    elif mode == "run-selected":
        steps = run_selected(profiles)
    else:
        sys.exit(f"unknown TRIGGER_MODE={mode!r} (expected nightly, manual, or run-selected)")

    print(yaml.dump({"steps": steps}, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
