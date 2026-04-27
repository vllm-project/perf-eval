# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes.

Today: gsm8k on Qwen3.5 (H200). Will grow.

## Layout

```
workloads/
  qwen3_5_h200.yaml      # one recipe = one (model, hardware, set of tasks)
run.sh                   # entry: parses recipe, manages server, loops tasks
lib/
  server.sh              # start/health/stop functions for the vLLM container
  parse_workload.py      # YAML → shell exports
.buildkite/pipeline.yaml # one step per recipe
```

## Run locally

```bash
./run.sh workloads/qwen3_5_h200.yaml
```

Needs Docker, `lm-eval`, and `pyyaml` on the host.

## Workload schema

A recipe is a single YAML file with these fields:

| field         | type                | description                                                                 |
| ------------- | ------------------- | --------------------------------------------------------------------------- |
| `name`        | string              | Used in container name and result path (`results/<name>/`). Keep it slug-y. |
| `model`       | string              | HF repo id or local path; passed to `vllm serve`.                           |
| `image`       | string              | Docker image with `vllm` installed.                                         |
| `serve_args`  | string              | Appended to `vllm serve {model}`. Word-split, so quote nothing fancy.       |
| `hf_home`     | string              | Host path mounted into the container as `HF_HOME` (model cache).            |
| `tasks`       | list of task objects | One `lm_eval` invocation per entry.                                         |

Each task object:

| field         | type    | description                                                |
| ------------- | ------- | ---------------------------------------------------------- |
| `name`        | string  | lm-eval task name (e.g. `gsm8k`, `gpqa_diamond_cot_zeroshot`). |
| `num_fewshot` | int     | Passed to `lm_eval --num_fewshot`. Use `0` for zero-shot.  |

Example:

```yaml
name: qwen3_5-h200-gsm8k
model: Qwen/Qwen3.5-397B-A17B-FP8
image: vllm/vllm-openai:latest
serve_args: --tensor-parallel-size 8 --trust-remote-code
hf_home: /mnt/shared/hf-models
tasks:
  - name: gsm8k
    num_fewshot: 5
```

`num_fewshot` lives on the task (not the recipe) because `lm_eval --num_fewshot`
is a single global value — different tasks need different shot counts, so each
runs as a separate `lm_eval` invocation. Results land in
`results/<recipe-name>/<task-name>/`.

## Add a recipe

Copy `workloads/qwen3_5_h200.yaml`, edit the fields above, and add a step to
`.buildkite/pipeline.yaml` pointing at the new file.
