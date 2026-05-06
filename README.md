# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator (run.sh), helpers, GPU profiles
.buildkite/       pipeline bootstrap and step generator
CLAUDE.md         agent conventions and detailed Buildkite workflow
```

## How to use this repo

### Add a new recipe

1. Copy an existing workload that targets the same GPU — e.g. `workloads/qwen3_5_h200.yaml` is a small, complete example.
2. Name the file `<model>_<hardware>.yaml`. Keep hardware variants in separate files.
3. Edit the fields to match your model and tasks. Set `nightly: true` if it should run in the nightly schedule; leave it off for opt-in recipes.
4. Open a PR. The pipeline auto-discovers `workloads/*.yaml` — no Buildkite YAML edits needed.

### Recipe schema

A recipe has top-level metadata plus three blocks:

- **`vllm:`** — *how the server runs.* Defines what model to serve and how (`model`, `serve_args`, optional image/env overrides). Required.
- **`lm_eval:`** — *what accuracy to measure.* Lists lm-evaluation-harness tasks to run against the live server (e.g. `gsm8k`, `aime25`). Each task's score is saved under `results/<name>/<task-name>/`. Optional.
- **`vllm_bench:`** — *what perf to measure.* Lists `vllm bench serve` configs (input/output lengths, concurrency, dataset). Raw JSON is saved and ingested into the perf dashboard. Optional.

Include either or both of `lm_eval:` / `vllm_bench:` depending on what you want out of this recipe.

```yaml
name: qwen3_5-h200       # used in container name and results/<name>/
gpu: H200                # picks queue/image/HF cache from lib/gpu_profiles.yaml
num_gpus: 8
nightly: true            # include in the nightly schedule (default: false)

vllm:                    # how the server is brought up
  model: Qwen/Qwen3.5-397B-A17B-FP8
  image: vllm/vllm-openai:nightly      # optional; falls back to VLLM_IMAGE / VLLM_COMMIT / latest
  env:                                  # optional; merged over the GPU profile's env
    SOME_VAR: value
  serve_args: >-                        # appended to `vllm serve <model>`; word-split
    -dp 8 --enable-expert-parallel
    --trust-remote-code

lm_eval:                 # accuracy tasks (optional)
  model_args:            # workload-level defaults, merged into every task
    tokenized_requests: false
    timeout: 6000
  tasks:
    - name: gsm8k                     # must match an lm-eval task name
      num_fewshot: 5
      model_args:                     # per-task overrides (merged on top of workload defaults)
        num_concurrent: 1024
        max_length: 40960
    - name: aime25
      num_fewshot: 0

vllm_bench:              # perf runs (optional) — fed to the perf dashboard
  configs:
    - name: 1k-in-1k-out-conc-256
      dataset: random                 # or speed_bench
      input_len: 1024
      output_len: 1024
      num_prompts: 500
      max_concurrency: 256
```

A few things worth knowing:

- **`gpu`** must match a key in `lib/gpu_profiles.yaml`. The profile sets the Buildkite queue, default image, HF cache path, and baseline env vars.
- **`nightly`** controls only the nightly schedule. Recipes with `nightly: false` (or omitted) are still triggerable explicitly via the `WORKLOADS` env var.
- **`lm_eval.tasks` is a list** because each entry runs as a separate `lm_eval` invocation — `--num_fewshot` is a single global flag, so different shot counts need separate runs. Each task's results land in `results/<name>/<task-name>/`.
- **`vllm_bench` runs first** if both blocks are present — that way perf-pipeline bugs surface quickly instead of waiting on a full lm-eval pass.

For everything else (the full set of supported fields, defaults, validation rules), the existing files in `workloads/` are the working reference and `lib/parse_workload.py` is the source of truth.

### Trigger a Buildkite build

The pipeline is **`vllm/perf-eval`**. With no extra config, a build runs every workload that has `nightly: true`.

**From the UI:** open the pipeline → New Build → pick branch and commit (must be pushed to GitHub) → optionally fill Environment Variables to scope the run → Create Build.

**Common env vars** to set on a build:

- `WORKLOADS` — comma- or newline-separated list of workload paths or stems. Runs exactly those.
- `VLLM_IMAGE` — full Docker image URI; overrides every workload's image.
- `VLLM_COMMIT` — vLLM commit SHA; resolved as `vllm/vllm-openai:nightly-<sha>` unless `VLLM_IMAGE` is also set.
- `BENCH_ONLY` — `true` to run only `vllm_bench` configs and skip `lm_eval` tasks.

**From an agent:** see `CLAUDE.md` for the Buildkite MCP workflow (don't shell out to `curl` or `bk`).

### Run a recipe end-to-end

A real run needs a GPU host with Docker, vLLM, and lm-eval available:

```bash
./lib/run.sh workloads/qwen3_5_h200.yaml
```

Locally, you can smoke-test recipe changes without a GPU — see `CLAUDE.md` for the parser stub and shell-syntax checks.

## Agents

`CLAUDE.md` has conventions for AI agents working in this repo: smoke-testing changes, launching Buildkite builds for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.
