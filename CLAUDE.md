# Agent instructions for perf-eval

This repo orchestrates lm-evaluation-harness runs against vLLM via YAML workloads in `workloads/`. `lib/run.sh` parses a workload, brings up vLLM in Docker, and dispatches each task to a helper in `lib/`. Real runs need GPUs and are exercised on Buildkite.

## Keep the README in sync

Whenever a change touches the workload schema, repo layout, run command, or anything else a new user reads on day one, **update `README.md` in the same change**. The README is the entry point — if it lies, people waste time. This applies to schema renames (e.g. moving fields under a new top-level key), new files in `lib/` or `workloads/`, changed local-run prerequisites, and additions to the Buildkite pipeline. If you're not sure whether a change deserves a README update, default to updating it.

## AI assistance disclosure

If an AI agent (Claude Code, Cursor, Copilot, etc.) wrote, edited, or substantially shaped any code or config in a change, the change **must say so**. Concretely:

- **PR description** — include a line like `This PR was authored with assistance from <tool>` (or "AI-assisted") in the body. Don't bury it; reviewers should see it before they read the diff.
- **Commit messages** — include a `Co-Authored-By:` trailer naming the model, e.g. `Co-Authored-By: Claude <noreply@anthropic.com>`. Claude Code adds this by default; don't strip it.
- **Code comments** — not required. Don't sprinkle "// AI-generated" through the source; the PR + commit metadata is the durable record.

This applies to non-trivial changes (new code, refactors, design decisions). Pure mechanical edits the agent ran on the user's behalf — formatter runs, find-and-replace, bumping a version — don't need a callout, but err on the side of disclosing if unsure.

## Local testing

You cannot run a real eval locally — it needs a GPU host with Docker, vLLM, and lm-eval installed. What you *can* run locally:

**Parser smoke test** — exec the parser with a stubbed `lm_eval` registry to verify TSV output and validation behavior. The stub avoids needing `lm-eval` installed; populate `all_tasks` with the names referenced by the YAML under test:

```bash
python3 -c "
import sys, types
m = types.ModuleType('lm_eval'); t = types.ModuleType('lm_eval.tasks')
class TM: all_tasks = ['gsm8k', 'aime25']
t.TaskManager = TM
sys.modules['lm_eval'] = m; sys.modules['lm_eval.tasks'] = t
sys.argv = ['parse_workload.py', 'workloads/qwen3_5_h200.yaml']
exec(open('lib/parse_workload.py').read())
"
```

**Shell syntax** — catches typos in the orchestrator and helpers without executing them:

```bash
bash -n lib/run.sh && bash -n lib/server.sh && bash -n lib/run_lm_eval.sh
```

If you actually need real validation (parser hitting lm-eval's task registry rather than a stub), `pip install 'lm-eval[api]' pyyaml` first. Without it the parser exits with `cannot validate task names: lm_eval not importable` — that's intentional, never silently skip validation.

## Launching a Buildkite build

Use the Buildkite MCP tools — never shell out to `curl` or `bk`. Pipeline metadata:

- **org**: `vllm`
- **pipeline**: `perf-eval`
- **repo**: `github.com/vllm-project/perf-eval`
- **default branch**: `main`
- **what it runs**: a dynamic pipeline. A bootstrap step runs `.buildkite/generate_pipeline.py` and generates per-workload steps using each workload's GPU profile. When `WORKLOADS` is set, it runs exactly those workload paths or stems. Otherwise it discovers all `workloads/*.yaml` with `nightly: true`.

### Workflow

1. **Make sure the commit is on the remote.** Buildkite clones from GitHub; unpushed commits will fail to resolve. If the user has local changes you've been working on, ask before pushing; once pushed, capture the SHA from `git rev-parse <branch>`.
2. **Trigger the build** via `mcp__claude_ai_Buildkite__create_build`:
   - `org_slug: "vllm"`
   - `pipeline_slug: "perf-eval"`
   - `commit: "<full SHA>"`
   - `branch: "<branch name>"` (use the actual branch, not `main`, when testing a feature branch)
   - `message: "<short description of what this tests>"` — match the existing convention: short, action-oriented (e.g. "Add gpqa diamond", "Writable HF_HOME for lm_eval datasets cache"). No emoji unless the user asks.
   - `environment`: pass `WORKLOADS` for an explicit workload list, `VLLM_IMAGE` or `VLLM_COMMIT` for image selection, and `BENCH_ONLY=true` when only the `vllm_bench` configs should run. Omit `WORKLOADS` to run all `nightly: true` workloads.
3. **Report the build URL** back to the user immediately so they can follow along; the response includes `web_url`.

### Watching a running build

- `mcp__claude_ai_Buildkite__get_build` with `job_state: "failed,broken,canceled"` to check for failures without pulling full logs.
- `mcp__claude_ai_Buildkite__tail_logs` with the `job_id` for the most recent log lines — start here for failure diagnosis, it's far cheaper than `read_logs`.
- `mcp__claude_ai_Buildkite__search_logs` with patterns like `"error|failed|exception|Traceback"` if `tail_logs` doesn't show the failure.

A typical build takes ~30–90 minutes (the step has a 120-min hard timeout) — it downloads model weights into the workload GPU profile's HF cache, then runs every task in the workload. GPU queues are shared with other vLLM pipelines, so don't trigger duplicate builds for the same commit unless asked.

### Don't

- Don't push to `main` or trigger builds without the user asking. Triggering a build is visible to the team and consumes GPU minutes.
- Don't `--no-verify` past failing pre-commit hooks just to get a build out. Fix the hook failure first.
- Don't hard-code GPU queues in `.buildkite/pipeline.yaml` or `generate_pipeline.py`; add or update entries in `lib/gpu_profiles.yaml` instead.
