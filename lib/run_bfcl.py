"""Run a single BFCL test group/category against a running vLLM server.

Usage:
    python3 lib/run_bfcl.py <model> <base_url> <category> \
        <num_threads> <temperature> <results_dir> [maximum_step_limit] [max_test_cases]

maximum_step_limit comes from the workload YAML when set; BFCL_MAXIMUM_STEP_LIMIT
env overrides YAML; otherwise the perf-eval default (10) applies.

max_test_cases subsamples each category to the first N cases (by BFCL id order).
Set per category in YAML or globally via BFCL_MAX_TEST_CASES env (env wins).
Uses BFCL's test_case_ids_to_generate.json + --partial-eval.

Registers the model in BFCL's config, runs generate + evaluate, then
writes an lm_eval-compatible results JSON so the existing ingest.py
and dashboard auto-discover the scores without any adapter.

BFCL test groups (e.g. multi_turn, live, agentic) expand to multiple
sub-categories at runtime. For those groups we write the overall score
plus one result per sub-category. A manifest at
  <results_dir>/.bfcl_ingest/<category>.txt
lists every task run.sh should ingest.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

BFCL_DEFAULT_MAXIMUM_STEP_LIMIT = 10
BFCL_MAXIMUM_STEP_LIMIT_ENV = "BFCL_MAXIMUM_STEP_LIMIT"
BFCL_MAX_TEST_CASES_ENV = "BFCL_MAX_TEST_CASES"
BFCL_TSV_UNSET = "-"


def unset_tsv_field(raw: str) -> str:
    value = raw.strip()
    return "" if value in ("", BFCL_TSV_UNSET) else value


def resolve_maximum_step_limit(workload_limit: str) -> tuple[int, str]:
    """Return (limit, source) with env overriding workload YAML."""
    env_value = os.environ.get(BFCL_MAXIMUM_STEP_LIMIT_ENV, "").strip()
    if env_value:
        try:
            limit = int(env_value)
        except ValueError:
            sys.exit(
                f"[bfcl] {BFCL_MAXIMUM_STEP_LIMIT_ENV} must be a positive integer, "
                f"got {env_value!r}"
            )
        if limit < 1:
            sys.exit(
                f"[bfcl] {BFCL_MAXIMUM_STEP_LIMIT_ENV} must be a positive integer, "
                f"got {limit}"
            )
        return limit, f"env:{BFCL_MAXIMUM_STEP_LIMIT_ENV}"

    workload_value = unset_tsv_field(workload_limit)
    if workload_value:
        try:
            limit = int(workload_value)
        except ValueError:
            sys.exit(
                f"[bfcl] workload maximum_step_limit must be a positive integer, "
                f"got {workload_value!r}"
            )
        if limit < 1:
            sys.exit(
                f"[bfcl] workload maximum_step_limit must be a positive integer, "
                f"got {limit}"
            )
        return limit, "workload"

    return BFCL_DEFAULT_MAXIMUM_STEP_LIMIT, "default"


def resolve_max_test_cases(workload_limit: str) -> tuple[int | None, str | None]:
    """Return (limit, source). None limit means run the full category."""
    env_value = os.environ.get(BFCL_MAX_TEST_CASES_ENV, "").strip()
    if env_value:
        try:
            limit = int(env_value)
        except ValueError:
            sys.exit(
                f"[bfcl] {BFCL_MAX_TEST_CASES_ENV} must be a positive integer, "
                f"got {env_value!r}"
            )
        if limit < 1:
            sys.exit(
                f"[bfcl] {BFCL_MAX_TEST_CASES_ENV} must be a positive integer, "
                f"got {limit}"
            )
        return limit, f"env:{BFCL_MAX_TEST_CASES_ENV}"

    workload_value = unset_tsv_field(workload_limit)
    if not workload_value:
        return None, None

    try:
        limit = int(workload_value)
    except ValueError:
        sys.exit(
            f"[bfcl] workload max_test_cases must be a positive integer, "
            f"got {workload_value!r}"
        )
    if limit < 1:
        sys.exit(
            f"[bfcl] workload max_test_cases must be a positive integer, "
            f"got {limit}"
        )
    return limit, "workload"


def write_test_case_ids(work_dir: Path, category: str, max_cases: int) -> int:
    """Write BFCL's id file and return how many cases were selected."""
    from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument, sort_key

    leaf_categories = parse_test_category_argument([category])
    entries: list[tuple[str, dict]] = []
    for leaf in leaf_categories:
        for entry in load_dataset_entry(leaf):
            entries.append((leaf, entry))

    entries.sort(key=lambda item: sort_key(item[1]))
    selected = entries[:max_cases]
    if not selected:
        sys.exit(f"[bfcl] max_test_cases={max_cases} but no cases found for {category}")

    ids_by_category: dict[str, list[str]] = {}
    for leaf, entry in selected:
        ids_by_category.setdefault(leaf, []).append(entry["id"])

    path = work_dir / "test_case_ids_to_generate.json"
    with open(path, "w") as f:
        json.dump(ids_by_category, f, indent=2)
    return len(selected)


def apply_maximum_step_limit(limit: int) -> None:
    """Patch BFCL before base_handler is imported."""
    import bfcl_eval.constants.default_prompts as bfcl_prompts

    bfcl_prompts.MAXIMUM_STEP_LIMIT = limit


def get_typer_defaults(func):
    """Extract default kwargs from a Typer-decorated function."""
    import typer

    defaults = {}
    for name, default in zip(
        func.__annotations__.keys(),
        func.__defaults__,
        strict=True,
    ):
        if isinstance(default, typer.models.OptionInfo):
            defaults[name] = default.default
    return defaults


def register_model(model: str, base_url: str):
    """Inject the model into BFCL's MODEL_CONFIG_MAPPING."""
    import bfcl_eval.constants.model_config as bfcl_model_config
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    bfcl_model_config.MODEL_CONFIG_MAPPING = {
        model: bfcl_model_config.ModelConfig(
            model_name=model,
            display_name=f"{model} (FC) (vLLM)",
            url=f"https://huggingface.co/{model}",
            org="",
            license="apache-2.0",
            model_handler=OpenAICompletionsHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )
    }


def run_generate(
    model, category, num_threads, temperature, *, use_test_subset: bool
):
    from bfcl_eval.__main__ import generate

    kwargs = get_typer_defaults(generate)
    kwargs["model"] = [model]
    kwargs["test_category"] = category
    kwargs["skip_server_setup"] = True
    kwargs["num_threads"] = num_threads
    kwargs["temperature"] = temperature
    if use_test_subset:
        kwargs["run_ids"] = True
        kwargs["allow_overwrite"] = True
    generate(**kwargs)


def run_evaluate(model, category, *, partial: bool):
    from bfcl_eval.__main__ import evaluate

    kwargs = get_typer_defaults(evaluate)
    kwargs["model"] = [model]
    kwargs["test_category"] = category
    if partial:
        kwargs["partial_eval"] = True
    evaluate(**kwargs)


CATEGORY_TO_CSV = {
    "simple_python": "data_non_live.csv",
    "simple_java": "data_non_live.csv",
    "simple_javascript": "data_non_live.csv",
    "multiple": "data_non_live.csv",
    "parallel": "data_non_live.csv",
    "parallel_multiple": "data_non_live.csv",
    "irrelevance": "data_non_live.csv",
    "live_simple": "data_live.csv",
    "live_multiple": "data_live.csv",
    "live_parallel": "data_live.csv",
    "live_parallel_multiple": "data_live.csv",
    "live_irrelevance": "data_live.csv",
    "live_relevance": "data_live.csv",
    "multi_turn_base": "data_multi_turn.csv",
    "multi_turn_miss_func": "data_multi_turn.csv",
    "multi_turn_miss_param": "data_multi_turn.csv",
    "multi_turn_long_context": "data_multi_turn.csv",
}

# BFCL aggregate categories expand to multiple sub-categories at runtime.
# Their scores live in summary columns of the category CSV, not as
# BFCL_v4_{category} columns (and there is no per-aggregate *_score.json).
AGGREGATE_CATEGORY_SCORES = {
    "multi_turn": ("data_multi_turn.csv", "Multi Turn Overall Acc"),
    "live": ("data_live.csv", "Live Overall Acc"),
    "non_live": ("data_non_live.csv", "Non-Live Overall Acc"),
    "agentic": ("data_agentic.csv", "Agentic Overall Acc"),
    "web_search": ("data_agentic.csv", "Web Search Summary"),
    "memory": ("data_agentic.csv", "Memory Summary"),
    "all": ("data_overall.csv", "Overall Acc"),
    "all_scoring": ("data_overall.csv", "Overall Acc"),
}


def _score_dir(work_dir: Path) -> Path:
    return work_dir / "score"


def _model_row_match(model: str, row: dict) -> bool:
    model_name = row.get("Model", row.get("model", ""))
    return model in model_name or model.replace("/", "_") in model_name


def _accuracy_from_percentage(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value) / 100.0
    except (ValueError, TypeError):
        return None


def _csv_values_for_lookup(
    row: dict, *, column: str | None, column_contains: str | None
) -> list[object]:
    if column is not None:
        return [row.get(column)]
    return [value for key, value in row.items() if column_contains in key]


def _parse_csv_accuracy(
    work_dir: Path,
    model: str,
    csv_name: str,
    *,
    column: str | None = None,
    column_contains: str | None = None,
) -> dict | None:
    """Read a percentage accuracy from a BFCL score CSV for the given model."""
    csv_path = _score_dir(work_dir) / csv_name
    if not csv_path.exists():
        return None

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not _model_row_match(model, row):
                continue
            for value in _csv_values_for_lookup(
                row, column=column, column_contains=column_contains
            ):
                if accuracy := _accuracy_from_percentage(value):
                    return {"accuracy": accuracy}
    return None


def _read_score_json(path: Path) -> dict:
    with open(path) as f:
        return json.loads(f.readline())


def _find_score_json(work_dir: Path, category: str) -> dict | None:
    for path in _score_dir(work_dir).rglob(f"*{category}_score.json"):
        return _read_score_json(path)
    return None


def _parse_subcategory_json_average(
    work_dir: Path, subcategories: list[str]
) -> dict | None:
    accuracies = []
    for subcat in subcategories:
        score = _find_score_json(work_dir, subcat)
        if score and (acc := score.get("accuracy")) is not None:
            accuracies.append(float(acc))
    if not accuracies:
        return None
    return {"accuracy": sum(accuracies) / len(accuracies)}


def _parse_aggregate_score(
    work_dir: Path, model: str, category: str
) -> dict | None:
    csv_name, column = AGGREGATE_CATEGORY_SCORES[category]
    if score := _parse_csv_accuracy(work_dir, model, csv_name, column=column):
        return score

    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING

    subcategories = TEST_COLLECTION_MAPPING.get(category, [])
    if subcategories:
        return _parse_subcategory_json_average(work_dir, subcategories)
    return None


def _parse_leaf_csv_score(
    work_dir: Path, model: str, category: str
) -> dict | None:
    csv_candidates = []
    if csv_name := CATEGORY_TO_CSV.get(category):
        csv_candidates.append(csv_name)
    csv_candidates.append("data_overall.csv")

    marker = f"BFCL_v4_{category}"
    for csv_name in csv_candidates:
        if score := _parse_csv_accuracy(
            work_dir, model, csv_name, column_contains=marker
        ):
            return score
    return None


def parse_score_from_csv(work_dir: Path, model: str, category: str) -> dict | None:
    """Extract per-category accuracy from BFCL V4 score CSVs and JSON files."""
    if category in AGGREGATE_CATEGORY_SCORES:
        return _parse_aggregate_score(work_dir, model, category)

    if score := _parse_leaf_csv_score(work_dir, model, category):
        return score

    return _find_score_json(work_dir, category)


def _test_collection_subcategories(category: str) -> list[str]:
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING

    return list(TEST_COLLECTION_MAPPING.get(category, []))


def _parse_overall_score(
    work_dir: Path,
    model: str,
    category: str,
    subcategories: list[str],
) -> dict | None:
    if category in AGGREGATE_CATEGORY_SCORES:
        if score := _parse_aggregate_score(work_dir, model, category):
            return score
    return _parse_subcategory_json_average(work_dir, subcategories)


def collect_scores(
    work_dir: Path, model: str, category: str
) -> dict[str, dict] | None:
    """Collect overall and sub-category scores for dashboard ingest."""
    subcategories = _test_collection_subcategories(category)
    if not subcategories:
        if score := parse_score_from_csv(work_dir, model, category):
            return {category: score}
        return None

    sub_scores: dict[str, dict] = {}
    for subcat in subcategories:
        if sub_score := parse_score_from_csv(work_dir, model, subcat):
            sub_scores[subcat] = sub_score

    missing = [subcat for subcat in subcategories if subcat not in sub_scores]
    if missing:
        print(
            f"[bfcl] warning: missing sub-category scores: {', '.join(missing)}",
            flush=True,
        )

    overall = _parse_overall_score(work_dir, model, category, subcategories)
    if not overall and sub_scores:
        accuracies = [s["accuracy"] for s in sub_scores.values()]
        overall = {"accuracy": sum(accuracies) / len(accuracies)}

    if not overall and not sub_scores:
        return None

    scores = {category: overall} if overall else {}
    scores.update(sub_scores)
    return scores


def to_lm_eval_format(
    model: str, category: str, score: dict, *, group_average: bool = False
) -> dict:
    """Transform BFCL aggregate score into lm_eval-compatible results JSON."""
    task_name = f"bfcl_{category}"
    alias = f"{task_name} (overall)" if group_average else task_name
    accuracy = score.get("accuracy", 0.0)

    return {
        "results": {
            task_name: {
                "acc,none": accuracy,
                "acc_stderr,none": 0.0,
                "alias": alias,
            }
        },
        "configs": {
            task_name: {"task": task_name, "num_fewshot": 0}
        },
        "versions": {task_name: 1},
        "n-shot": {task_name: 0},
        "config": {
            "model": "local-completions",
            "model_args": f"model={model}",
        },
    }


def write_results(
    results_dir: Path,
    model: str,
    scores: dict[str, dict],
    ts: str,
    run_category: str,
) -> list[str]:
    is_group = bool(_test_collection_subcategories(run_category))
    written = []
    for cat, score in scores.items():
        out_dir = results_dir / f"bfcl-{cat}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"results_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(
                to_lm_eval_format(
                    model,
                    cat,
                    score,
                    group_average=is_group and cat == run_category,
                ),
                f,
                indent=2,
            )
        print(f"[bfcl] {cat}: accuracy={score.get('accuracy', '?')}", flush=True)
        print(f"[bfcl] results written to {out_path}", flush=True)
        written.append(cat)
    return written


def write_ingest_manifest(
    results_dir: Path, run_category: str, categories: list[str]
) -> Path:
    manifest_dir = results_dir / ".bfcl_ingest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{run_category}.txt"
    manifest_path.write_text("\n".join(categories) + "\n")
    return manifest_path


def main():
    if len(sys.argv) not in (7, 8, 9):
        sys.exit(
            "usage: run_bfcl.py <model> <base_url> <category> "
            "<num_threads> <temperature> <results_dir> "
            "[maximum_step_limit] [max_test_cases]"
        )

    model, base_url, category = sys.argv[1], sys.argv[2], sys.argv[3]
    num_threads, temperature = int(sys.argv[4]), float(sys.argv[5])
    results_dir = Path(sys.argv[6])
    workload_step_limit = sys.argv[7] if len(sys.argv) >= 8 else ""
    workload_max_test_cases = sys.argv[8] if len(sys.argv) == 9 else ""

    step_limit, step_limit_source = resolve_maximum_step_limit(workload_step_limit)
    apply_maximum_step_limit(step_limit)
    print(
        f"[bfcl] maximum_step_limit={step_limit} (from {step_limit_source})",
        flush=True,
    )

    max_test_cases, max_test_cases_source = resolve_max_test_cases(workload_max_test_cases)

    work_dir = (results_dir / ".bfcl_work").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000

    os.environ["OPENAI_BASE_URL"] = base_url.rstrip("/") + "/v1"
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
    os.environ["BFCL_PROJECT_ROOT"] = str(work_dir)
    os.environ["LOCAL_SERVER_ENDPOINT"] = f"http://{host}"
    os.environ["LOCAL_SERVER_PORT"] = str(port)

    use_test_subset = max_test_cases is not None
    if use_test_subset:
        selected = write_test_case_ids(work_dir, category, max_test_cases)
        print(
            f"[bfcl] max_test_cases={selected} for {category} "
            f"(from {max_test_cases_source})",
            flush=True,
        )

    register_model(model, base_url)

    print(f"[bfcl] generate: model={model} category={category}", flush=True)
    run_generate(
        model, category, num_threads, temperature, use_test_subset=use_test_subset
    )

    print(f"[bfcl] evaluate: model={model} category={category}", flush=True)
    run_evaluate(model, category, partial=use_test_subset)

    scores = collect_scores(work_dir, model, category)
    if not scores:
        sys.exit(f"[bfcl] no score found for {category}")

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    written = write_results(results_dir, model, scores, ts, category)
    manifest = write_ingest_manifest(results_dir, category, written)
    print(f"[bfcl] ingest manifest written to {manifest}", flush=True)


if __name__ == "__main__":
    main()
