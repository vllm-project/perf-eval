# ruler_chat

lm-eval's 8 RULER needle-in-a-haystack (NIAH) tasks, set up to run against a thinking chat model served by vLLM at 32K, 128K and 256K. They use RULER's data, prompts and scoring. What's different is how the model is asked and how many samples are built, so it runs the way vLLM PRs run RULER ([vllm-project/vllm#60397](https://github.com/vllm-project/vllm/pull/60397)): greedy, thinking on, 32K output tokens.

**Scores are not comparable to stock lm-eval `niah_*` or published RULER numbers.** Compare them across perf-eval runs and against vLLM PR runs set up the same way.

## What's the same as lm-eval

The source is [`lm_eval/tasks/ruler`](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler), with `niah_single_1.yaml` as the base config.

- **Sample generation:** each sample is built by `prepare_niah.generate_samples` and fitted to the target length with the served model's tokenizer.
- **Prompt:** `niah_utils.TEMPLATE`.
- **Haystack and needles:** each task uses the same haystack, needle types and needle counts as `niah_utils` (see `utils.py`).
- **Decoding:** greedy (`temperature: 0`), with `until: []`.
- **Scoring:** `postprocess_pred` then `string_match_all`, so a sample's score is the fraction of its expected answers found in the response.

## What's changed and why

| | lm-eval | ruler_chat | Why |
|---|---|---|---|
| Answer prefix | `gen_prefix: "{{gen_prefix}}"` | none | perf-eval calls `/v1/chat/completions` with `--apply_chat_template`. lm-eval adds `gen_prefix` as a trailing `assistant` message without `continue_final_message`. vLLM therefore renders it as a finished assistant turn, then opens a new turn where the model starts thinking. The prefix never starts the answer, so it's dropped. |
| Output budget | `max_gen_toks: 128` | `32768` | A thinking model is cut off mid-reasoning at 128 tokens and never answers. |
| Lengths | 4K, 8K, 16K, 32K, 64K, 128K | 32K, 128K, 256K | lm-eval's `metric_list` stops at 128K, so it can't report 256K. |
| Per-doc scoring | `common_utils.process_results` fills every other length with -1 and `aggregate_metrics` drops them | `utils.process_results` reports only the doc's own length, aggregated with a plain `mean` | Same result without the placeholder values. |
| Samples | 500 per length (hard-coded in `niah_utils`) | 40 per length | Workaround, see below. |
| Tasks | 13 (8 NIAH plus vt, cwe, fwe, qa_squad, qa_hotpot) | the 8 NIAH tasks | The other five each needed workarounds for lm-eval bugs: a `ruler_vt` IndexError, QA partial-match scoring, and `ruler_cwe` never sending its worked example. They were dropped together with those workarounds. |

`tag: longcxt` and `repeats: 1` are also left out; dropping them changes nothing.

## Workaround: 40 samples

lm-eval's NIAH generator hard-codes `num_samples=500`, and `--limit` only trims after all 500 are built. At 256K with 32K-token thinking, that's far too long for a nightly job. The `custom_dataset` wrappers in `utils.py` call lm-eval's own generator with `NUM_SAMPLES = 40` and reseed for each length (seed 42, RULER's default).

**Removed by:** [EleutherAI/lm-evaluation-harness#4640](https://github.com/EleutherAI/lm-evaluation-harness/pull/4640), which reads `num_samples` from task metadata. Once the pinned lm-eval has it, delete the wrappers, point `custom_dataset` back at `niah_utils`, and set `num_samples: 40` in `_template_yaml` metadata.

## Reading the scores

- **What's reported:** `ruler_chat` reports one score per length, the unweighted mean over the 8 tasks. That's 320 samples per length; per task, it's 40.
- **Noise:** at a true score of 0.9, one standard error is about 4.7 points per task and about 1.7 points for the group mean. A few-point move on one task in one run isn't a signal.
- **Request count:** 8 tasks × 3 lengths × 40 = 960 requests.
- **Context length:** the server needs `--max-model-len` of at least 262144 + 32768 = 294912.
- **Reasoning effort:** set per model in the workload's `gen_kwargs`.

## Running it as-is instead

The closest to stock RULER would be to turn thinking off, send the prefix with `continue_final_message=true`, and set `max_gen_toks: 128`. That measures retrieval without reasoning, which is a different test. It still can't go past 128K without changing the metric list, and it still needs the sample-count workaround until #4640 lands.
