"""Read a workload YAML and emit shell exports.

Usage:
    eval "$(python3 lib/parse_workload.py workloads/foo.yaml)"

Sets WORKLOAD_NAME (top-level), WORKLOAD_MODEL/IMAGE/SERVE_ARGS (from the
`vllm:` block), WORKLOAD_ENV (newline-separated KEY=VALUE pairs from
`vllm.env`), and WORKLOAD_LM_EVAL_TASKS_TSV. Each TSV line is
"name\\tnum_fewshot\\tmodel_args", where model_args is the comma-separated
key=value string lm_eval expects. `lm_eval.model_args` (workload-level)
is merged under each task's `model_args` block.

Per-task top-level fields are limited to `name`, `num_fewshot`, and
`model_args`; any other top-level field is rejected with a hint to move
it under `model_args:`.

Each name in `lm_eval.tasks` is validated against lm_eval's task registry
(tasks + groups + tags); unknown names exit non-zero before the server
is started.
"""

import shlex
import sys

import yaml

TOP_FIELDS = ("name",)
VLLM_FIELDS = ("model", "image", "serve_args")
TASK_FIELDS = {"name", "num_fewshot", "model_args"}


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


def known_task_names() -> set:
    try:
        from lm_eval.tasks import TaskManager
    except ImportError as e:
        sys.exit(f"cannot validate task names: lm_eval not importable ({e})")
    return set(TaskManager().all_tasks)


def main(path: str) -> None:
    with open(path) as f:
        data = yaml.safe_load(f)
    lm_eval = data.get("lm_eval") or {}
    tasks = lm_eval.get("tasks") or []
    if not tasks:
        sys.exit(f"{path}: missing or empty `lm_eval.tasks`")
    known = known_task_names()
    for t in tasks:
        if t["name"] not in known:
            sys.exit(f"{path}: unknown lm_eval task {t['name']!r}")
        extra = set(t) - TASK_FIELDS
        if extra:
            sys.exit(
                f"{path}: task {t['name']!r} has unsupported top-level fields "
                f"{sorted(extra)}; move them under `model_args:`"
            )

    vllm = data.get("vllm") or {}
    for key in TOP_FIELDS:
        print(f"WORKLOAD_{key.upper()}={shlex.quote(str(data.get(key, '')))}")
    for key in VLLM_FIELDS:
        print(f"WORKLOAD_{key.upper()}={shlex.quote(str(vllm.get(key, '')))}")
    env = vllm.get("env") or {}
    env_lines = "\n".join(f"{k}={fmt(v)}" for k, v in env.items())
    print(f"WORKLOAD_ENV={shlex.quote(env_lines)}")
    base_args = lm_eval.get("model_args") or {}
    lines = []
    for t in tasks:
        merged = {**base_args, **(t.get("model_args") or {})}
        lines.append(f"{t['name']}\t{t.get('num_fewshot', 0)}\t{serialize(merged)}")
    print(f"WORKLOAD_LM_EVAL_TASKS_TSV={shlex.quote(chr(10).join(lines))}")


if __name__ == "__main__":
    main(sys.argv[1])
