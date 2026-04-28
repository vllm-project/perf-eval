#!/usr/bin/env python3
"""Generate Buildkite pipeline steps from workload YAML files.

Supports three trigger modes (via TRIGGER_MODE env var):

  nightly      (default) Run every workload with ``nightly: true``.
  manual       Present a Buildkite input step so the user can pick a workload,
               then a follow-up step uploads the real H200 step.
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
    " && ./run.sh {path}"
)


def load_workloads():
    workloads = []
    for path in sorted(glob.glob("workloads/*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        name = data.get("name", os.path.basename(path).removesuffix(".yaml"))
        workloads.append({"path": path, "name": name, "data": data})
    return workloads


def make_h200_step(path, name):
    return {
        "label": f":h200: {name}",
        "agents": {"queue": "H200"},
        "timeout_in_minutes": 120,
        "commands": SETUP_COMMANDS + [RUN_TEMPLATE.format(path=path)],
        "artifact_paths": ["results/**/*"],
    }


def nightly(workloads):
    steps = [
        make_h200_step(w["path"], w["name"])
        for w in workloads
        if w["data"].get("nightly") is True
    ]
    if not steps:
        sys.exit("TRIGGER_MODE=nightly but no workloads have nightly: true")
    return steps


def manual(workloads):
    if not workloads:
        sys.exit("TRIGGER_MODE=manual but no workload files found in workloads/")
    options = [{"label": w["name"], "value": w["path"]} for w in workloads]
    input_step = {
        "input": "Select a workload to run",
        "fields": [
            {
                "select": "Workload",
                "key": "workload",
                "required": True,
                "options": options,
            }
        ],
    }
    followup_step = {
        "label": ":pipeline: upload selected workload",
        "agents": {"queue": "H200"},
        "commands": [
            "python3 -m pip install --user pyyaml 2>/dev/null || true",
            'WORKLOAD="$(buildkite-agent meta-data get workload)"'
            " TRIGGER_MODE=run-selected python3 .buildkite/generate_pipeline.py"
            " | buildkite-agent pipeline upload",
        ],
    }
    return [input_step, followup_step]


def run_selected():
    path = os.environ.get("WORKLOAD", "")
    if not path:
        sys.exit("TRIGGER_MODE=run-selected but WORKLOAD env var is not set")
    if not os.path.isfile(path):
        sys.exit(f"workload not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    name = data.get("name", os.path.basename(path).removesuffix(".yaml"))
    return [make_h200_step(path, name)]


def main():
    mode = os.environ.get("TRIGGER_MODE", "nightly").lower()

    if mode == "nightly":
        steps = nightly(load_workloads())
    elif mode == "manual":
        steps = manual(load_workloads())
    elif mode == "run-selected":
        steps = run_selected()
    else:
        sys.exit(f"unknown TRIGGER_MODE={mode!r} (expected nightly, manual, or run-selected)")

    print(yaml.dump({"steps": steps}, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    main()
