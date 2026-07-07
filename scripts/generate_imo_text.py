#!/usr/bin/env python3
"""Generate IMO-AnswerBench responses and save project-local text bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_vibethinker import (
    DEFAULT_MODEL,
    build_imo_answerbench_prompt,
    extract_final_answer_text,
    infer_text,
    infer_text_batch,
    load_imo_answerbench_rows,
    load_model_and_tokenizer,
    normalize_short_answer,
    resolve_model_id,
    save_imo_generation_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IMO-AnswerBench text for later activation capture.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the selected rows into this many process-level shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to run. Use with --shard-count.",
    )
    parser.add_argument("--problem-id", action="append")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of problems to generate in one batched call. Default keeps legacy one-at-a-time behavior.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--generated-text-dir", type=Path, default=Path("outputs/imo-answerbench-text"))
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--stream-mode", choices=["token", "sentence"], default="token")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-interval", type=float, default=2.0)
    parser.add_argument("--stream-with-progress", action="store_true")
    return parser.parse_args(argv)


def should_stream(args: argparse.Namespace) -> bool:
    return not args.no_stream and (args.stream_with_progress or not args.progress)


def shard_rows(rows: list[dict], shard_count: int, shard_index: int) -> list[dict]:
    if shard_count < 1:
        raise SystemExit("--shard-count must be at least 1.")
    if not 0 <= shard_index < shard_count:
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < shard-count.")
    if shard_count == 1:
        return rows
    return [row for index, row in enumerate(rows) if index % shard_count == shard_index]


def run_rows(
    args: argparse.Namespace,
    rows: list[dict],
    selected_count: int | None = None,
    worker_label: str = "",
    model=None,
    tokenizer=None,
) -> int:
    model_id = resolve_model_id(args.model)
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(args)
    correct = 0

    if args.shard_count > 1:
        print(
            f"Running shard {args.shard_index}/{args.shard_count}: "
            f"{len(rows)} of {selected_count} selected row(s).",
            flush=True,
        )

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    if len(rows) > 1 and args.batch_size > 1 and not should_stream(args):
        label = f"{worker_label} " if worker_label else ""
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            prompts = [build_imo_answerbench_prompt(row["Problem"], args.answer_only) for row in batch_rows]
            print("=" * 80, flush=True)
            print(f"{label}Batched generation: {len(batch_rows)} problem(s)", flush=True)
            results = infer_text_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                auto_max_new_tokens=True,
                progress=args.progress,
                progress_desc=f"{label}IMO batch x{len(batch_rows)}".strip(),
                progress_interval=args.progress_interval,
            )

            for batch_offset, (row, prompt, result) in enumerate(zip(batch_rows, prompts, results), start=1):
                row_number = batch_start + batch_offset
                dataset_index = row.get("_dataset_index")
                print("=" * 80, flush=True)
                print(
                    f"{label}Sample {row_number}/{len(rows)} | dataset row {dataset_index} | "
                    f"IMO-AnswerBench {row['Problem ID']}",
                    flush=True,
                )
                text_path = save_imo_generation_bundle(
                    output_dir=args.generated_text_dir,
                    row=row,
                    row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
                    model_id=model_id,
                    prompt=prompt,
                    result=result,
                )

                prediction = extract_final_answer_text(result.text)
                gold = str(row["Short Answer"]).strip()
                is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
                correct += int(is_correct)

                print(result.text)
                print("-" * 80)
                print("Result:")
                print(f"Generated tokens: {result.generated_tokens}")
                print(f"Hit token limit: {result.hit_token_limit}")
                print(f"Gold short answer: {gold}")
                print(f"Parsed prediction: {prediction}")
                print(f"Normalized exact match: {is_correct}")
                print(f"Saved text: {text_path}")

        print("=" * 80)
        print(f"Exact-match score: {correct}/{len(rows)}")
        return correct

    for row_number, row in enumerate(rows, start=1):
        label = f"{worker_label} " if worker_label else ""
        dataset_index = row.get("_dataset_index")
        print("=" * 80, flush=True)
        print(
            f"{label}Sample {row_number}/{len(rows)} | dataset row {dataset_index} | "
            f"IMO-AnswerBench {row['Problem ID']}",
            flush=True,
        )
        print(f"Category: {row['Category']} | Subcategory: {row['Subcategory']}", flush=True)
        print(f"Source: {row['Source']}", flush=True)
        print("-" * 80, flush=True)
        print(row["Problem"], flush=True)
        print("-" * 80, flush=True)
        print("Model response:", flush=True)

        prompt = build_imo_answerbench_prompt(row["Problem"], args.answer_only)
        result = infer_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=should_stream(args),
            stream_mode=args.stream_mode,
            auto_max_new_tokens=True,
            progress=args.progress,
            progress_desc=f"{label}IMO text {row_number}/{len(rows)}".strip(),
            progress_interval=args.progress_interval,
        )
        text_path = save_imo_generation_bundle(
            output_dir=args.generated_text_dir,
            row=row,
            row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
            model_id=model_id,
            prompt=prompt,
            result=result,
        )

        prediction = extract_final_answer_text(result.text)
        gold = str(row["Short Answer"]).strip()
        is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
        correct += int(is_correct)

        if not should_stream(args):
            print(result.text)
        print("-" * 80)
        print("Result:")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"Hit token limit: {result.hit_token_limit}")
        print(f"Gold short answer: {gold}")
        print(f"Parsed prediction: {prediction}")
        print(f"Normalized exact match: {is_correct}")
        print(f"Saved text: {text_path}")

    print("=" * 80)
    print(f"Exact-match score: {correct}/{len(rows)}")
    return correct


def main() -> None:
    args = parse_args()
    rows = load_imo_answerbench_rows(
        args.sample_size,
        args.seed,
        args.problem_id,
        args.start_index,
        args.shuffle,
    )
    if not rows:
        raise SystemExit("No IMO-AnswerBench rows matched the request.")
    selected_count = len(rows)
    rows = shard_rows(rows, args.shard_count, args.shard_index)
    if not rows:
        raise SystemExit(
            f"Shard {args.shard_index}/{args.shard_count} has no rows "
            f"from the {selected_count} selected row(s)."
        )
    run_rows(args, rows, selected_count=selected_count)


if __name__ == "__main__":
    main()
