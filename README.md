# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes.

Today: gsm8k + aime25 on Qwen3.5 (H200). Will grow.

## Layout

```
workloads/
  qwen3_5_h200.yaml      # one recipe = one (model, hardware, set of tasks)
lib/
  run.sh                 # orchestrator: parses recipe, brings up vLLM, dispatches tasks
  parse_workload.py      # YAML → shell exports + lm_eval task validation
  server.sh              # start/health/stop functions for the vLLM container
  run_lm_eval.sh         # per-task runner for lm-evaluation-harness tasks
  gpu_profiles.yaml      # machine-specific defaults (queue, image, HF_HOME) per GPU type
.buildkite/
  pipeline.yaml          # bootstrap step: runs generate_pipeline.py
  generate_pipeline.py   # generates per-workload steps (nightly or manual)
CLAUDE.md                # agent instructions (testing, build triggers, conventions)
```

## Run locally

```bash
./lib/run.sh workloads/qwen3_5_h200.yaml
```

Needs Docker, `lm-eval[api]`, and `pyyaml` on the host. The parser validates each task name against `lm_eval`'s registry, so `lm-eval` must be importable; without it the parser exits with `cannot validate task names: lm_eval not importable` (intentional, never silently skip validation).

## Workload schema

A recipe is a single YAML file with a few top-level fields and two config groups:

```yaml
name: qwen3_5-h200       # used in container name + results/<name>/
gpu: H200                # required — selects queue, image, HF_HOME from gpu_profiles.yaml
num_gpus: 8              # number of GPUs available on the target machine
nightly: true            # include in nightly scheduled builds (default: false)

vllm:                    # everything about the served model
  model: Qwen/Qwen3.5-397B-A17B-FP8
  serve_args: >-         # appended to `vllm serve <model>`; word-split
    -dp 8 --enable-expert-parallel
    --reasoning-parser qwen3
    --enable-prefix-caching
    --language-model-only
    --trust-remote-code

lm_eval:                 # everything about the eval client
  model_args:            # workload-level defaults, applied to every task
    tokenized_requests: false
    tokenizer_backend: null
    timeout: 6000
  tasks:
    - name: gsm8k        # must match a name in lm_eval's task registry
      num_fewshot: 5
      model_args:        # per-task overrides; merged over workload defaults
        num_concurrent: 1024
        max_length: 40960
        max_gen_toks: 32768
    - name: aime25
      num_fewshot: 0
      model_args:
        num_concurrent: 128
        max_length: 40960
```

### `vllm:` block

| field        | type   | description                                                                                                                                          |
| ------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`      | string | HF repo id or local path; passed as the first positional arg to `vllm/vllm-openai`'s entrypoint.                                                     |
| `image`      | string | (optional) Docker image override. Defaults to `vllm/vllm-openai:nightly`.                                                                            |
| `env`        | dict   | (optional) Extra env vars passed to the container with `-e`. The GPU profile's `env:` block is merged underneath this (workload values win on conflict), and `HF_HOME` falls back to the profile's `hf_home` field. |
| `serve_args` | string | Appended to `vllm serve <model>`. Word-split, so don't put fancy quoting in here.                                                                    |

### `lm_eval:` block

| field        | type | description                                                                                              |
| ------------ | ---- | -------------------------------------------------------------------------------------------------------- |
| `model_args` | dict | Defaults merged into every task's `model_args`. Values are coerced to lm-eval's expected literal format (`true`→`True`, `null`→`None`). |
| `tasks`      | list | One entry per `lm_eval` invocation.                                                                      |

Each task object:

| field         | type   | description                                                                                                                       |
| ------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | string | lm-eval task name (e.g. `gsm8k`, `aime25`). Validated against `lm_eval`'s registry; unknown names abort before the server starts. |
| `num_fewshot` | int    | Passed to `lm_eval --num_fewshot`. Use `0` for zero-shot.                                                                         |
| `model_args`  | dict   | Per-task model-arg overrides. Merged on top of `lm_eval.model_args`.                                                              |

Per-task top-level fields are limited to `name`, `num_fewshot`, `model_args`. Anything else is rejected with a hint to move it under `model_args:`.

`num_fewshot` lives on the task (not the workload) because `lm_eval --num_fewshot` is a single global value — different tasks need different shot counts, so each runs as a separate `lm_eval` invocation. Results land in `results/<recipe-name>/<task-name>/`.

## Add a recipe

Copy `workloads/qwen3_5_h200.yaml`, edit the fields above, and set `nightly: true` if the workload should run in nightly scheduled builds. The pipeline dynamically discovers workloads — no need to edit `.buildkite/pipeline.yaml`.

## Buildkite pipeline

The pipeline always emits one H200 step per selected workload. Selection is controlled by env vars set on the Buildkite build (via `environment` when triggering through the API, or the "Environment Variables" field in the UI's New Build dialog):

- `WORKLOADS` (optional) — comma- or newline-separated list of workload paths or stems (`workloads/qwen3_5_h200.yaml`, `qwen3_5_h200`, both work). When set, runs exactly those workloads; when unset, runs every workload with `nightly: true`.
- `VLLM_IMAGE` (optional) — full Docker image URI. Overrides every workload's `vllm.image`.
- `VLLM_COMMIT` (optional) — commit SHA; resolved as `vllm/vllm-openai:nightly-<sha>` on Docker Hub. Ignored if `VLLM_IMAGE` is set.

With no env vars set, the build runs the nightly schedule. Image precedence is `VLLM_IMAGE` > `VLLM_COMMIT` > workload's `vllm.image` > `vllm/vllm-openai:latest`.

## Agents

`CLAUDE.md` has the workflow for AI agents working in this repo: how to smoke-test changes locally, how to launch a Buildkite build for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.
