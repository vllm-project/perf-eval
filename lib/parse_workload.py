"""Read a workload YAML and emit shell exports.

Usage:
    eval "$(python3 lib/parse_workload.py workloads/foo.yaml)"

Sets WORKLOAD_NAME, WORKLOAD_MODEL, WORKLOAD_IMAGE, WORKLOAD_SERVE_ARGS,
WORKLOAD_HF_HOME, and WORKLOAD_TASKS_TSV (one "name\\tnum_fewshot" line per task).
"""

import shlex
import sys

import yaml

SCALAR_FIELDS = ("name", "model", "image", "serve_args", "hf_home")


def main(path: str) -> None:
    with open(path) as f:
        data = yaml.safe_load(f)
    for key in SCALAR_FIELDS:
        print(f"WORKLOAD_{key.upper()}={shlex.quote(str(data.get(key, '')))}")
    tasks_tsv = "\n".join(
        f"{t['name']}\t{t.get('num_fewshot', 0)}" for t in data.get("tasks", [])
    )
    print(f"WORKLOAD_TASKS_TSV={shlex.quote(tasks_tsv)}")


if __name__ == "__main__":
    main(sys.argv[1])
