"""Run a single BFCL test category against a running vLLM server.

Usage:
    python3 lib/run_bfcl.py <model> <base_url> <category> \
        <num_threads> <temperature> <results_dir>

Registers the model in BFCL's config, runs generate + evaluate, then
writes an lm_eval-compatible results JSON so the existing ingest.py
and dashboard auto-discover the scores without any adapter.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


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


def run_generate(model, category, num_threads, temperature):
    from bfcl_eval.__main__ import generate

    kwargs = get_typer_defaults(generate)
    kwargs["model"] = [model]
    kwargs["test_category"] = category
    kwargs["skip_server_setup"] = True
    kwargs["num_threads"] = num_threads
    kwargs["temperature"] = temperature
    generate(**kwargs)


def run_evaluate(model, category):
    from bfcl_eval.__main__ import evaluate

    kwargs = get_typer_defaults(evaluate)
    kwargs["model"] = [model]
    kwargs["test_category"] = category
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


def parse_score_from_csv(work_dir: Path, model: str, category: str) -> dict | None:
    """Extract per-category accuracy from BFCL V4 aggregate CSV files."""
    csv_name = CATEGORY_TO_CSV.get(category)
    csv_candidates = [work_dir / "score" / csv_name] if csv_name else []
    csv_candidates.append(work_dir / "score" / "data_overall.csv")

    bfcl_category = f"BFCL_v4_{category}"
    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row.get("Model", row.get("model", ""))
                if model not in model_name and model.replace("/", "_") not in model_name:
                    continue
                for key, value in row.items():
                    if bfcl_category in key:
                        try:
                            return {"accuracy": float(value) / 100.0}
                        except (ValueError, TypeError):
                            continue

    # Fallback: look for per-category JSONL score files (older BFCL versions)
    model_slug = model.replace("/", "_")
    for p in (work_dir / "score").rglob(f"*{category}_score.json"):
        with open(p) as f:
            return json.loads(f.readline())

    return None


def to_lm_eval_format(model: str, category: str, score: dict) -> dict:
    """Transform BFCL aggregate score into lm_eval-compatible results JSON."""
    task_name = f"bfcl_{category}"
    accuracy = score.get("accuracy", 0.0)

    return {
        "results": {
            task_name: {
                "acc,none": accuracy,
                "acc_stderr,none": 0.0,
                "alias": task_name,
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


def main():
    if len(sys.argv) != 7:
        sys.exit(
            "usage: run_bfcl.py <model> <base_url> <category> "
            "<num_threads> <temperature> <results_dir>"
        )

    model, base_url, category = sys.argv[1], sys.argv[2], sys.argv[3]
    num_threads, temperature = int(sys.argv[4]), float(sys.argv[5])
    results_dir = Path(sys.argv[6])

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

    register_model(model, base_url)

    print(f"[bfcl] generate: model={model} category={category}", flush=True)
    run_generate(model, category, num_threads, temperature)

    print(f"[bfcl] evaluate: model={model} category={category}", flush=True)
    run_evaluate(model, category)

    score = parse_score_from_csv(work_dir, model, category)
    if not score:
        sys.exit(f"[bfcl] no score found for {category}")

    print(
        f"[bfcl] {category}: accuracy={score.get('accuracy', '?')}",
        flush=True,
    )

    out_dir = results_dir / f"bfcl-{category}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    out_path = out_dir / f"results_{ts}.json"

    lm_eval_results = to_lm_eval_format(model, category, score)
    with open(out_path, "w") as f:
        json.dump(lm_eval_results, f, indent=2)

    print(f"[bfcl] results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
