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
    load_imo_answerbench_rows,
    load_model_and_tokenizer,
    normalize_short_answer,
    resolve_model_id,
    save_imo_generation_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IMO-AnswerBench text for later activation capture.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-id", action="append")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
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
    return parser.parse_args()


def should_stream(args: argparse.Namespace) -> bool:
    return not args.no_stream and (args.stream_with_progress or not args.progress)


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

    model_id = resolve_model_id(args.model)
    model, tokenizer = load_model_and_tokenizer(args)
    correct = 0

    for row_number, row in enumerate(rows, start=1):
        dataset_index = row.get("_dataset_index")
        print("=" * 80, flush=True)
        print(
            f"Sample {row_number}/{len(rows)} | dataset row {dataset_index} | IMO-AnswerBench {row['Problem ID']}",
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
            progress_desc=f"IMO text {row_number}/{len(rows)}",
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


if __name__ == "__main__":
    main()
