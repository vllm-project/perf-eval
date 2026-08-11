#!/usr/bin/env python3
"""Deterministic open-loop GLM-5.2 acceptance load with ITL evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass(frozen=True)
class Shape:
    name: str
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int

    @property
    def new_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens


@dataclass
class Result:
    request_id: int
    endpoint_index: int
    shape: str
    scheduled_s: float
    started_s: float
    ttft_s: float | None
    last_token_s: float | None
    completed_s: float
    latency_s: float
    mean_itl_s: float | None
    stream_chunk_itl_s: list[float]
    nonempty_stream_chunks: int
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    error: str | None


P50 = Shape("p50", 12_500, 10_700, 95)
P95 = Shape("p95", 26_500, 24_000, 450)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lower, upper = math.floor(pos), math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def cached_from_usage(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    value = getattr(details, "cached_tokens", None) if details is not None else None
    return int(value) if value is not None else None


class PromptFactory:
    def __init__(self, tokenizer_path: str, seed: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True, local_files_only=True
        )
        self.seed = seed
        candidates = (
            "alpha ", "beta ", "cat ", "dog ", "red ",
            "blue ", "one ", "two ", "north ", "south ",
        )
        self.words = [
            word for word in candidates
            if self.count("x " + word) - self.count("x ") == 1
        ]
        if len(self.words) < 4:
            raise RuntimeError(f"too few one-token words: {self.words}")
        self._bases: dict[int, str] = {}

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def base(self, token_count: int) -> str:
        if token_count not in self._bases:
            value = "x " * (token_count - 1)
            actual = self.count(value)
            if actual != token_count:
                raise RuntimeError(
                    f"base token mismatch: target={token_count} actual={actual}"
                )
            self._bases[token_count] = value
        return self._bases[token_count]

    def prompt(self, shape: Shape, request_id: int) -> str:
        rng = random.Random(self.seed + request_id * 1_000_003)
        suffix = "".join(rng.choice(self.words) for _ in range(shape.new_tokens))
        value = self.base(shape.cached_tokens) + suffix
        actual = self.count(value)
        if actual != shape.prompt_tokens:
            raise RuntimeError(
                f"prompt token mismatch id={request_id}: "
                f"target={shape.prompt_tokens} actual={actual}"
            )
        return value


async def one_completion(
    client: AsyncOpenAI,
    *,
    endpoint_index: int,
    model: str,
    prompt: str,
    output_tokens: int,
    request_id: int,
    shape: Shape,
    epoch: float,
    scheduled_s: float,
) -> Result:
    started = time.perf_counter() - epoch
    token_times: list[float] = []
    usage = None
    error = None
    try:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=output_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if chunk.choices and (chunk.choices[0].text or ""):
                token_times.append(time.perf_counter() - epoch)
            if chunk.usage is not None:
                usage = chunk.usage
        if not token_times:
            raise RuntimeError("stream returned no non-empty token chunk")
        if usage is None:
            raise RuntimeError("stream returned no usage")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = time.perf_counter() - epoch
    first_token = token_times[0] if token_times else None
    last_token = token_times[-1] if token_times else None
    completion_tokens = int(usage.completion_tokens) if usage is not None else None
    mean_itl = None
    if (
        first_token is not None
        and last_token is not None
        and completion_tokens is not None
        and completion_tokens > 1
    ):
        mean_itl = (last_token - first_token) / (completion_tokens - 1)
    return Result(
        request_id=request_id,
        endpoint_index=endpoint_index,
        shape=shape.name,
        scheduled_s=scheduled_s,
        started_s=started,
        ttft_s=(first_token - started) if first_token is not None else None,
        last_token_s=last_token,
        completed_s=completed,
        latency_s=completed - started,
        mean_itl_s=mean_itl,
        stream_chunk_itl_s=[
            later - earlier for earlier, later in zip(token_times, token_times[1:])
        ],
        nonempty_stream_chunks=len(token_times),
        prompt_tokens=int(usage.prompt_tokens) if usage is not None else None,
        cached_tokens=cached_from_usage(usage) if usage is not None else None,
        completion_tokens=completion_tokens,
        error=error,
    )


async def warm_prefixes(
    clients: list[AsyncOpenAI],
    *,
    model: str,
    factory: PromptFactory,
    shapes: tuple[Shape, ...],
) -> list[dict]:
    evidence = []
    for endpoint_index, client in enumerate(clients):
        for shape in shapes:
            started = time.perf_counter()
            response = await client.completions.create(
                model=model,
                prompt=factory.base(shape.cached_tokens),
                max_tokens=1,
                temperature=0.0,
                extra_body={"ignore_eos": True},
            )
            evidence.append({
                "endpoint_index": endpoint_index,
                "shape": shape.name,
                "requested_base_tokens": shape.cached_tokens,
                "server_prompt_tokens": (
                    response.usage.prompt_tokens if response.usage else None
                ),
                "elapsed_s": time.perf_counter() - started,
            })
    return evidence


def sum_known(rows: list[Result], field: str) -> int:
    return sum(
        int(value) for row in rows
        if (value := getattr(row, field)) is not None
    )


def summarize_rows(rows: list[Result]) -> dict:
    chunk_itls = [value for row in rows for value in row.stream_chunk_itl_s]
    prompt_tokens = sum_known(rows, "prompt_tokens")
    cached_tokens = sum_known(rows, "cached_tokens")
    return {
        "requests": len(rows),
        "ttft_s": stats([row.ttft_s for row in rows if row.ttft_s is not None]),
        "mean_itl_s": stats([
            row.mean_itl_s for row in rows if row.mean_itl_s is not None
        ]),
        "stream_chunk_itl_s": stats(chunk_itls),
        "latency_s": stats([row.latency_s for row in rows]),
        "prompt_tokens": stats([
            float(row.prompt_tokens) for row in rows if row.prompt_tokens is not None
        ]),
        "cached_tokens": stats([
            float(row.cached_tokens) for row in rows if row.cached_tokens is not None
        ]),
        "completion_tokens": stats([
            float(row.completion_tokens)
            for row in rows if row.completion_tokens is not None
        ]),
        "cache_hit_fraction": (
            cached_tokens / prompt_tokens if prompt_tokens else None
        ),
    }


async def main_async(args: argparse.Namespace) -> dict:
    shapes = (P50, P95)
    endpoints = [value.rstrip("/") for value in args.endpoint]
    factory = PromptFactory(args.tokenizer, args.seed)
    samples = {
        shape.name: factory.prompt(shape, 10_000 + index)
        for index, shape in enumerate(shapes)
    }
    clients = [
        AsyncOpenAI(
            base_url=endpoint,
            api_key="unused",
            timeout=args.timeout,
            max_retries=0,
        )
        for endpoint in endpoints
    ]
    try:
        model_lists = [await client.models.list() for client in clients]
        warmups = await warm_prefixes(
            clients, model=args.model, factory=factory, shapes=shapes
        )
        smoke_epoch = time.perf_counter()
        smoke = []
        for endpoint_index, client in enumerate(clients):
            for shape_index, shape in enumerate(shapes):
                smoke.append(await one_completion(
                    client,
                    endpoint_index=endpoint_index,
                    model=args.model,
                    prompt=samples[shape.name],
                    output_tokens=shape.output_tokens,
                    request_id=20_000 + endpoint_index * 10 + shape_index,
                    shape=shape,
                    epoch=smoke_epoch,
                    scheduled_s=time.perf_counter() - smoke_epoch,
                ))

        mean_prompt_tokens = (
            P50.prompt_tokens * (args.mix_period - args.long_per_period)
            + P95.prompt_tokens * args.long_per_period
        ) / args.mix_period
        arrival_rate = args.target_prompt_tpm / 60.0 / mean_prompt_tokens
        total_requests = math.floor(args.duration * arrival_rate)
        interval = 1.0 / arrival_rate
        epoch = time.perf_counter()
        tasks = []
        for request_id in range(total_requests):
            scheduled_s = request_id * interval
            delay = epoch + scheduled_s - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            is_long = request_id % args.mix_period < args.long_per_period
            shape = P95 if is_long else P50
            endpoint_index = request_id % len(clients)
            tasks.append(asyncio.create_task(one_completion(
                clients[endpoint_index],
                endpoint_index=endpoint_index,
                model=args.model,
                prompt=factory.prompt(shape, request_id),
                output_tokens=shape.output_tokens,
                request_id=request_id,
                shape=shape,
                epoch=epoch,
                scheduled_s=scheduled_s,
            )))
        admission_end_s = time.perf_counter() - epoch
        results = await asyncio.gather(*tasks)
        run_end_s = time.perf_counter() - epoch
    finally:
        await asyncio.gather(*(client.close() for client in clients))

    successes = [row for row in results if row.error is None]
    failures = [row for row in results if row.error is not None]
    steady_start = args.duration * args.warmup_fraction
    steady_end = args.duration
    steady_started = [
        row for row in successes if steady_start <= row.started_s <= steady_end
    ]
    steady_completed = [
        row for row in successes if steady_start <= row.completed_s <= steady_end
    ]
    steady_seconds = steady_end - steady_start
    planned_prompt_tokens = sum(
        P95.prompt_tokens if request_id % args.mix_period < args.long_per_period
        else P50.prompt_tokens
        for request_id in range(total_requests)
    )
    by_shape = {
        shape.name: summarize_rows([
            row for row in steady_started if row.shape == shape.name
        ])
        for shape in shapes
    }
    by_endpoint = {
        str(index): summarize_rows([
            row for row in steady_started if row.endpoint_index == index
        ])
        for index in range(len(endpoints))
    }
    overall = summarize_rows(steady_started)
    summary = {
        "schema_version": 2,
        "metric_notes": {
            "mean_itl_s": (
                "Per-request (last non-empty stream chunk - first non-empty "
                "stream chunk) / (completion_tokens - 1), then percentiles "
                "across requests."
            ),
            "stream_chunk_itl_s": (
                "Inter-arrival time between non-empty SSE chunks. A server may "
                "place multiple tokens in one chunk; raw per-request values are retained."
            ),
            "steady_achieved_prompt_tpm": (
                "Prompt tokens of requests completing inside the post-warmup "
                "admission window divided by that fixed window."
            ),
        },
        "config": {
            "endpoints": endpoints,
            "model": args.model,
            "tokenizer": args.tokenizer,
            "duration_s": args.duration,
            "warmup_fraction": args.warmup_fraction,
            "target_prompt_tpm": args.target_prompt_tpm,
            "actual_planned_prompt_tpm": planned_prompt_tokens / args.duration * 60,
            "mean_offered_prompt_tokens": mean_prompt_tokens,
            "arrival_rate_rps": arrival_rate,
            "mix_period": args.mix_period,
            "long_per_period": args.long_per_period,
            "seed": args.seed,
            "shapes": [asdict(shape) | {"new_tokens": shape.new_tokens} for shape in shapes],
        },
        "server_models": [[item.id for item in listing.data] for listing in model_lists],
        "warmups": warmups,
        "smoke": [asdict(row) for row in smoke],
        "run": {
            "offered_requests": total_requests,
            "successful_requests": len(successes),
            "failed_requests": len(failures),
            "success_fraction": len(successes) / total_requests if total_requests else None,
            "admission_end_s": admission_end_s,
            "run_end_s": run_end_s,
            "drain_s": run_end_s - admission_end_s,
            "steady_window_s": [steady_start, steady_end],
            "steady_completed_requests": len(steady_completed),
            "steady_achieved_prompt_tpm": (
                sum_known(steady_completed, "prompt_tokens") / steady_seconds * 60
            ),
            "steady_achieved_completion_tok_s": (
                sum_known(steady_completed, "completion_tokens") / steady_seconds
            ),
            "full_run_prompt_tpm_including_drain": (
                sum_known(successes, "prompt_tokens") / run_end_s * 60
            ),
            "schedule_lag_s": stats([
                row.started_s - row.scheduled_s for row in steady_started
            ]),
            **overall,
            "by_shape": by_shape,
            "by_endpoint": by_endpoint,
            "errors": [asdict(row) for row in failures[:100]],
        },
        "results": [asdict(row) for row in results],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", action="append", default=[],
        help="OpenAI /v1 endpoint; repeat for round-robin multi-engine load",
    )
    parser.add_argument("--model", default="nvidia/GLM-5.2-NVFP4")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--target-prompt-tpm", type=float, required=True)
    parser.add_argument("--mix-period", type=int, default=20)
    parser.add_argument("--long-per-period", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.endpoint:
        args.endpoint = ["http://127.0.0.1:8000/v1"]
    if not 0 < args.long_per_period < args.mix_period:
        parser.error("long-per-period must be between 0 and mix-period")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup-fraction must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    summary = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), **summary["run"]}, indent=2))


if __name__ == "__main__":
    main()
