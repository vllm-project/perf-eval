# perf-eval

Run accuracy + perf workloads against vLLM, defined by small YAML recipes in `workloads/`. Most recipes launch one vLLM server; opt-in schema-v1 recipes can instead launch a multi-node prefill/decode-disaggregated deployment through Slurm and Pyxis.

Each recipe is one `(model, hardware, set of tasks)` combination. The Buildkite pipeline picks recipes up automatically — to ship a new run, you write a YAML file, push it, and trigger a build.

## Repo layout

```
workloads/        one YAML per (model, hardware) recipe
lib/              orchestrator, server/Slurm launchers, helpers, GPU profiles
.buildkite/       pipeline bootstrap and step generator
.agents/skills/   repo-scoped agent workflows, including PD model onboarding
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

- **`vllm:`** — *how the server runs.* Defines what model to serve and how (`model`, `serve_args`, optional image/env overrides). Required.
- **`serving:`** — *optional multi-node orchestration.* Schema v1 supports Slurm/Pyxis prefill/decode disaggregation. Omit it for the existing standalone Docker/native path.
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

- **`gpu`** must match a key in `lib/gpu_profiles.yaml`. The profile sets the Buildkite queue, default image, HF cache path, and baseline env vars.
- **`nightly`** controls only the nightly schedule. Recipes with `nightly: false` (or omitted) are still triggerable explicitly via the `WORKLOADS` env var.
- **`lm_eval.tasks` is a list** because each entry runs as a separate `lm_eval` invocation — `--num_fewshot` is a single global flag, so different shot counts need separate runs. Each task's results land in `results/<name>/<task-name>/`.
- **`vllm_bench` runs first** if both blocks are present — that way perf-pipeline bugs surface quickly instead of waiting on a full lm-eval pass.
- **`bfcl` may need tool-call serve args.** Some models require `--enable-auto-tool-choice` and `--tool-call-parser` for function-calling; the parser warns if `--tool-call-parser` is absent. Each category runs as a separate generate + evaluate pass; scores appear on the eval dashboard as `bfcl_<category>` tasks.

For everything else (the full set of supported fields, defaults, validation rules), the existing files in `workloads/` are the working reference and `lib/parse_workload.py` is the source of truth.

### Slurm prefill/decode disaggregation

`workloads/kimi_k2_5_gb300_pd.yaml` is the first schema-v1 example. It allocates three four-GPU GB300 nodes: one TP1 × DP4/EP prefill instance (DEP4) and two independent TP4 × DP1/EP decode replicas (TEP4×2). `num_gpus` must equal the total implied by the roles.

The parser turns this block into a validated launch plan for `lib/slurm_pd_launcher.py`:

```yaml
serving:
  version: 1
  mode: pd_disagg
  launcher: slurm
  common_serve_args: >-
    --enable-expert-parallel --enable-ep-weight-filter
  slurm:
    partition: batch
    time_limit: "03:00:00"
    grace_period_s: 120
    container:
      runtime: pyxis
      mounts:
        - source: /raid/shared/nvidia/Kimi-K2.5-NVFP4
          source_env: KIMI_K2_5_MODEL_PATH
          target: /model
          read_only: true
        - source: /dev/infiniband
          target: /dev/infiniband
        - source: /home/${USER}/.cache/flashinfer
          source_env: FLASHINFER_CACHE_PATH
          target: /root/.cache/flashinfer
  kv_transfer:
    connector: NixlConnector
    load_failure_policy: fail
    extra_config: {num_threads: 4}
  roles:
    - role: prefill
      count: 1
      nodes_per_instance: 1
      gpus_per_node: 4
      tensor_parallel_size: 1
      kv_role: kv_producer
    - role: decode
      count: 2
      nodes_per_instance: 1
      gpus_per_node: 4
      tensor_parallel_size: 4
      kv_role: kv_consumer
  router:
    repo_path: ${HOME}/Kimi-PD/vllm-router
    revision: v0.1.12
    command: [target/release/vllm-router]
    port: 31000
    intra_node_data_parallel_size: 1
```

`tensor_parallel_size` defaults to 1 and must divide `gpus_per_node`; the launcher derives local DP as `gpus_per_node / tensor_parallel_size` and global DP as `nodes_per_instance × local DP`. It owns `--tensor-parallel-size`, `--port`, all global/local DP ranks and addresses, shared per-instance RPC ports, `--data-parallel-hybrid-lb`, and the NIXL KV-transfer JSON. Do not repeat those flags in `common_serve_args` or a role's `serve_args`.

The serving state file records the allocation-owning controller PID and `grace_period_s`. Normal cleanup signals that controller first so it can stop the router and every `srun` cleanly; `scancel` is used only if the controller is unavailable or does not exit within the bounded shutdown wait.

Router v0.1.12 has one global `intra_node_data_parallel_size`. Set it to `1` for mixed local DP sizes: the router treats each URL as one endpoint, vLLM internally balances the prefill endpoint across its four local DP ranks, and each TP4 decode replica remains one logical worker. Values above 1 are valid only when every role has the same local DP size, because the router expands both prefill and decode URLs uniformly.

Mount `source_env` values make site-specific paths overridable without editing YAML. The Kimi workload accepts `KIMI_K2_5_MODEL_PATH` and `FLASHINFER_CACHE_PATH`. NIXL also requires `/dev/infiniband` inside the container. The checked-in GB300 settings match the `nvidia-b300-login` cluster (`mlx5_4:1` and `enP22p3s0f0np0`); other clusters should override the workload environment rather than copying the GB200 NVL72 settings.

The Kimi workload uses BF16 KV cache because vLLM v0.25.1 does not load the checkpoint's calibrated FP8 KV scales for MLA models. Revisit FP8 only after the loader remap is fixed and correctness is revalidated.

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

For the GB300 Slurm recipe, run from a login-node shell outside any existing Slurm allocation, with `salloc`, `srun`, `scontrol`, Pyxis, and the v0.1.12 router binary already built at the configured path. The launcher owns and cancels one allocation, starts all role steps, then runs `vllm bench serve` inside the same image and allocation:

```bash
export KIMI_K2_5_MODEL_PATH=/raid/shared/nvidia/Kimi-K2.5-NVFP4
export FLASHINFER_CACHE_PATH="$HOME/.cache/flashinfer"
export ENROOT_CACHE_PATH="/raid/users/$USER/enroot-cache"
export ENROOT_DATA_PATH="/raid/users/$USER/enroot-data"
export ENROOT_RUNTIME_PATH="/tmp/enroot-$USER"
./lib/run.sh workloads/kimi_k2_5_gb300_pd.yaml
```

The checkout and results directory must be on storage visible at the same path from the compute nodes; the benchmark client bind-mounts that path to persist its JSON artifact. For this first bring-up it runs as an overlapping CPU-only step on the first prefill node, so reported performance can include client-side contention; a dedicated client node is a follow-up for authoritative peak numbers. The `GB300` profile targets the `gb300-slurm` Buildkite login-node agent queue. This workload remains manual/opt-in (`nightly: false`); `GB300_QUEUE` can target an alternate cluster or canary queue. Generated Slurm steps are serialized with `concurrency: 1`, `concurrency_group: perf-eval/<resolved-queue>`, and `concurrency_method: eager`, preventing fixed router and DP RPC ports from colliding even if another agent is later attached to the queue. Keep the login-node agent itself at one worker as an additional guard. Its agent environment hook must put the cluster's Slurm binaries on `PATH` and set any site-required `SLURM_CONF` and `LD_LIBRARY_PATH`; the dynamic command defers `HOME` and `PATH` expansion until this GPU job runs. Before triggering the workload, ensure the queue has an online agent and the router/model prerequisites above are present.

## Agents

`CLAUDE.md` has conventions for AI agents working in this repo: smoke-testing changes, launching Buildkite builds for a chosen branch/commit, and the AI-assistance disclosure rule for PRs and commits.

For onboarding another model to the schema-v1 Slurm prefill/decode path, invoke the repository skill with `$vllm-pd-disagg-model-onboarding`. It captures the reusable model-contract, topology, NIXL/Pyxis, router, staged cluster bring-up, correctness, and Buildkite workflow behind `workloads/kimi_k2_5_gb300_pd.yaml`.
