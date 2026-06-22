# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator (run.sh), helpers, GPU profiles
.buildkite/       pipeline bootstrap and step generator
gen_report.py     generate HTML benchmark reports from results/
CLAUDE.md         agent conventions and detailed Buildkite workflow
```

## How to use this repo

### Add a new recipe

1. Copy an existing workload that targets the same GPU — e.g. `workloads/qwen3_5_h200.yaml` is a small, complete example.
2. Name the file `<model>_<hardware>.yaml`. Keep hardware variants in separate files.
3. Edit the fields to match your model and tasks. Set `nightly: true` if it should run in the nightly schedule; leave it off for opt-in recipes.
4. Open a PR. The pipeline auto-discovers `workloads/*.yaml` — no Buildkite YAML edits needed.

### Recipe schema

A recipe has top-level metadata plus up to three eval blocks:

- **`vllm:`** — *how the server runs.* Defines what model to serve and how (`model`, `serve_args`, optional image/env overrides, optional `attention_backends` list). Required.
- **`lm_eval:`** — *what accuracy to measure.* Lists lm-evaluation-harness tasks to run against the live server (e.g. `gsm8k`, `aime25`). Each task's score is saved under `results/<name>/<task-name>/`. Optional.
- **`vllm_bench:`** — *what perf to measure.* Lists `vllm bench serve` configs (input/output lengths, concurrency, dataset). Raw JSON is saved and ingested into the perf dashboard. Optional.
- **`bfcl:`** — *function-calling eval.* Runs [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) test categories against the live server. Some models need `--enable-auto-tool-choice` and `--tool-call-parser` in `serve_args`. Results are transformed to lm_eval format and ingested as `bfcl_<category>` tasks. Optional.

Include one or more of `lm_eval:` / `vllm_bench:` / `bfcl:` depending on what you want out of this recipe.

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
  attention_backends:                   # optional; list of VLLM_ATTENTION_BACKEND values
    - FLASH_ATTN                        # when set, the full eval suite runs once per
    - FLASHINFER                        # backend; results land in attn-<BACKEND>/ subdirs

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

bfcl:                    # function-calling eval (optional)
  test_categories:       # BFCL test categories to run
    - simple_python
    - multiple
    - parallel
  num_threads: 8         # optional, default 8
  temperature: 0.001     # optional, default 0.001

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

- **`vllm.attention_backends`** is an optional list of vLLM attention backend names (`FLASH_ATTN`, `FLASHINFER`, `XFORMERS`, `TRITON_ATTN`, `TRITON_MLA`, `ROCM_FLASH`, `PAGED_ATTENTION`,`ROCM_AITER_FA`,`ROCM_AITER_UNIFIED_ATTN`, `ROCM_ATTN`,`ROCM_AITER_MLA`,`ROCM_AITER_MLA_SPARSE`, `ROCM_AITER_TRITON_MLA`). When set, the orchestrator starts the server once per backend — adding `--attention-backend $ATTN_BACKEND` — and runs the complete eval suite (bench, lm_eval, bfcl) for each. Results are stored under `results/<name>/attn-<BACKEND>/` so every backend gets its own isolated output directory. Without this field, the server starts once with whatever attention backend vLLM selects by default and results go to `results/<name>/` as usual. See `workloads/attn-sweep-gpt-oss-120b-mi355x.yaml` for an example.
- **`gpu`** must match a key in `lib/gpu_profiles.yaml`. The profile sets the Buildkite queue, default image, HF cache path, and baseline env vars.
- **`nightly`** controls only the nightly schedule. Recipes with `nightly: false` (or omitted) are still triggerable explicitly via the `WORKLOADS` env var.
- **`lm_eval.tasks` is a list** because each entry runs as a separate `lm_eval` invocation — `--num_fewshot` is a single global flag, so different shot counts need separate runs. Each task's results land in `results/<name>/<task-name>/`.
- **`vllm_bench` runs first** if both blocks are present — that way perf-pipeline bugs surface quickly instead of waiting on a full lm-eval pass.
- **`bfcl` may need tool-call serve args.** Some models require `--enable-auto-tool-choice` and `--tool-call-parser` for function-calling; the parser warns if `--tool-call-parser` is absent. Each category runs as a separate generate + evaluate pass; scores appear on the eval dashboard as `bfcl_<category>` tasks.

For everything else (the full set of supported fields, defaults, validation rules), the existing files in `workloads/` are the working reference and `lib/parse_workload.py` is the source of truth.

### Trigger a Buildkite build

The pipeline is [**`vllm/perf-eval`**](https://buildkite.com/vllm/perf-eval). With no extra config, a build runs every workload that has `nightly: true`.

**From the UI:** open the pipeline → New Build → pick branch and commit (must be pushed to GitHub) → optionally fill Environment Variables to scope the run → Create Build.

**Required env vars** — both must be set on every build:

- `VLLM_COMMIT` — vLLM commit SHA being tested. Used to tag results and track which vLLM version produced them.
- `VLLM_IMAGE` — full Docker image URI (e.g. `vllm/vllm-openai:nightly-abc1234`). This is the image that gets pulled and run.

**Optional env vars:**

- `WORKLOADS` — comma- or newline-separated list of workload paths or stems. Runs exactly those instead of the default `nightly: true` set.
- `NIGHTLY` — set to `1` to tag every ingested row with `nightly: true`. The dashboard's `/nightly` view filters on this to pair adjacent nightly builds; only the scheduled nightly cron should set it.

**Example — trigger a build from the Buildkite UI:**

1. Open the `vllm/perf-eval` pipeline → **New Build**.
2. Pick the branch and commit (must already be pushed to GitHub).
3. Set the environment variables:
   ```
   VLLM_COMMIT=abc1234def5678
   VLLM_IMAGE=vllm/vllm-openai:nightly-abc1234def5678
   WORKLOADS=qwen3_5_h200
   ```
4. Click **Create Build**.

This runs the `qwen3_5_h200` workload against the specified vLLM nightly image. Omit `WORKLOADS` to run all `nightly: true` workloads.

**From an agent:** see `CLAUDE.md` for the Buildkite MCP workflow (don't shell out to `curl` or `bk`).

### Run a recipe end-to-end

A real run needs a GPU host with Docker, vLLM, and lm-eval available:

```bash
./lib/run.sh workloads/qwen3_5_h200.yaml
```

Locally, you can smoke-test recipe changes without a GPU — see `CLAUDE.md` for the parser stub and shell-syntax checks.

## Benchmark reports

After a run completes, generate interactive HTML reports from the `results/` directory:

```bash
python3 gen_report.py
```

This writes one `benchmark-<model>.html` per model directory found under `results/`, plus a `benchmark-index.html` wrapper. Open `benchmark-index.html` in a browser to tab between all models in one page — each model's report loads on demand when its tab is clicked.

If reports from previous rounds are already present in the directory, `benchmark-index.html` will include them alongside any newly generated ones, so the index always covers every available model regardless of which models were in the current run.

Each per-model report shows attention backend results side by side, with tabs for each input sequence length, color-coded best/worst values per metric, and percentage deltas relative to the default backend.

## Agents

`CLAUDE.md` has conventions for AI agents working in this repo: smoke-testing changes, launching Buildkite builds for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.
