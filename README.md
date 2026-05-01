# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes.

Workloads are split by model and hardware. H200 recipes are currently included
in the nightly schedule; B200 recipes are separate opt-in configs.

## Layout

```
workloads/
  qwen3_5_h200.yaml      # one recipe = one (model, hardware, set of tasks)
  qwen3_5_b200.yaml      # hardware variants live in separate recipe files
lib/
  run.sh                 # orchestrator: parses recipe, brings up vLLM, dispatches tasks
  parse_workload.py      # YAML → shell exports + lm_eval task validation
  server.sh              # start/health/stop functions for the vLLM container
  run_lm_eval.sh         # per-task runner for lm-evaluation-harness tasks
  gpu_profiles.yaml      # machine-specific defaults (queue, image, HF_HOME) per GPU type
.buildkite/
  pipeline.yaml          # bootstrap step: runs generate_pipeline.py
  generate_pipeline.py   # generates per-workload steps (nightly or selected)
CLAUDE.md                # agent instructions (testing, build triggers, conventions)
```

## Run locally

```bash
./lib/run.sh workloads/qwen3_5_h200.yaml
```

Needs Docker, `lm-eval[api]`, and `pyyaml` on the host. The parser validates each task name against `lm_eval`'s registry, so `lm-eval` must be importable; without it the parser exits with `cannot validate task names: lm_eval not importable` (intentional, never silently skip validation). When `BENCH_ONLY` is truthy, the parser skips lm-eval task registry validation because eval tasks will not run.

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

### `vllm_bench:` block (optional)

Before `lm_eval` runs, the orchestrator can run one or more `vllm bench serve` configs against the same server to capture throughput / latency metrics for the perf dashboard. Bench runs first so a single failing recipe surfaces perf-pipeline bugs without waiting on a full lm_eval pass. Skip the block entirely if you don't want any bench runs.

```yaml
vllm_bench:
  metadata:                # optional; auto-derived if omitted
    device: h200           # default: lowercased `gpu` field
    tp: 8                  # default: parsed from vllm.serve_args (TP * DP)
    precision: fp8         # default: matched against the model name (fp4/fp8/int4/int8/bf16/fp16)
  configs:
    - name: 1k-in-1k-out-conc-256   # used as the result filename
      backend: null                  # optional; passed to `--backend` when set
      dataset: random               # passed to `vllm bench serve --dataset-name`
      input_len: 1024                # passed as `--random-input-len` for random; used as ingest metadata for speed_bench
      output_len: 1024               # passed as `--random-output-len` for random, `--speed-bench-output-len` for speed_bench
      num_prompts: 500
      max_concurrency: 256
      speed_bench_dataset_subset: null       # optional; passed to `--speed-bench-dataset-subset`
      speed_bench_category: null     # optional; passed to `--speed-bench-category`
```

Each config is invoked as `docker exec <container> vllm bench serve …`, the raw JSON is copied to `results/<recipe-name>/bench-<config-name>.json`, and `lib/ingest_perf.py` transforms it (latencies ms → seconds, throughputs ÷ tp) before POSTing to the perf-dashboard ingest endpoint at `vllm-perf-data-ingest-…run.app`.

For `dataset: random`, the runner uses vLLM's random dataset length flags. For `dataset: speed_bench`, the runner uses `--speed-bench-output-len`; `input_len` is still required so dashboard ingestion can tag the row. When `backend` is set, the runner passes `--base-url http://127.0.0.1:<port>` automatically.
When `backend: openai-chat` is set, the runner also passes `--endpoint /v1/chat/completions`.
When `vllm.serve_args` includes `--trust-remote-code`, the runner also passes `--trust-remote-code` to `vllm bench serve` so the bench-side tokenizer can load models that require custom code.
`speed_bench` requires a vLLM image whose `vllm bench serve` CLI includes that dataset; older images such as `vllm/vllm-openai:v0.19.0` only support the random/spec/custom/HF dataset families.

## Add a recipe

Copy an existing workload, edit the fields above, and set `nightly: true` if the workload should run in nightly scheduled builds. Keep hardware variants as separate recipe files, for example `*_h200.yaml` and `*_b200.yaml`. The pipeline dynamically discovers workloads — no need to edit `.buildkite/pipeline.yaml`.

## Buildkite pipeline

The pipeline emits one step per selected workload, using the workload's `gpu` field to choose the Buildkite queue and GPU defaults from `lib/gpu_profiles.yaml`. Selection is controlled by env vars set on the Buildkite build (via `environment` when triggering through the API, or the "Environment Variables" field in the UI's New Build dialog):

- `WORKLOADS` (optional) — comma- or newline-separated list of workload paths or stems (`workloads/qwen3_5_h200.yaml`, `qwen3_5_h200`, `qwen3_5_b200` all work). When set, runs exactly those workloads; when unset, runs every workload with `nightly: true`.
- `VLLM_IMAGE` (optional) — full Docker image URI. Overrides every workload's `vllm.image`.
- `VLLM_COMMIT` (optional) — commit SHA; resolved as `vllm/vllm-openai:nightly-<sha>` on Docker Hub. Ignored if `VLLM_IMAGE` is set.
- `BENCH_ONLY` (optional) — set to `true`, `1`, or `yes` to run `vllm_bench` configs and skip `lm_eval` tasks. Buildkite bench-only jobs install only `pyyaml` and skip lm-eval task registry validation.

With no env vars set, the build runs the nightly schedule. Image precedence is `VLLM_IMAGE` > `VLLM_COMMIT` > workload's `vllm.image` > `vllm/vllm-openai:latest`.

B200 configs use the `B200` GPU profile, which routes to the `b200-k8s` queue and uses `/mnt/shared/hf_cache` for Hugging Face cache. They are currently `nightly: false`, so run them explicitly with `WORKLOADS`.

Eval result ingestion includes the resolved Docker image as `image`. It also includes `vllm_commit` when the run used `VLLM_COMMIT` or when the resolved image tag carries a commit, such as `vllm/vllm-openai:nightly-<sha>`.

## Agents

`CLAUDE.md` has the workflow for AI agents working in this repo: how to smoke-test changes locally, how to launch a Buildkite build for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.
