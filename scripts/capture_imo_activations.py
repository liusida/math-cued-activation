#!/usr/bin/env python3
"""Replay saved IMO-AnswerBench generations and save layer activations."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_vibethinker import (
    DEFAULT_MODEL,
    build_imo_answerbench_prompt,
    extract_final_answer_text,
    load_imo_answerbench_rows,
    load_imo_generation_bundle,
    load_model_and_tokenizer,
    normalize_short_answer,
    resolve_model_id,
    result_from_saved_generation,
    save_imo_activation_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture activations from saved IMO-AnswerBench text bundles.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-id", action="append")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--generated-text-dir", type=Path, default=Path("outputs/imo-answerbench-text"))
    parser.add_argument("--activation-dir", type=Path, default=Path("~/data/ICA-data/math-cued-activation"))
    parser.add_argument("--capture-layer", type=int, default=32)
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--capture-prompt-activations", action="store_true")
    return parser.parse_args()


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
            f"Capture {row_number}/{len(rows)} | dataset row {dataset_index} | IMO-AnswerBench {row['Problem ID']}",
            flush=True,
        )

        metadata, metadata_path = load_imo_generation_bundle(args.generated_text_dir, row, model_id)
        prompt = build_imo_answerbench_prompt(row["Problem"], args.answer_only)
        if metadata["text"]["prompt"] != prompt:
            raise SystemExit(
                f"Saved prompt mismatch for {row['Problem ID']}. "
                "Use the same --answer-only setting as the generation pass."
            )
        if metadata["generation"]["model"] != model_id:
            raise SystemExit(
                f"Saved model mismatch for {row['Problem ID']}: "
                f"{metadata['generation']['model']} != {model_id}"
            )

        print(f"Replay metadata: {metadata_path}", flush=True)
        print(f"Category: {row['Category']} | Subcategory: {row['Subcategory']}", flush=True)
        print(f"Layer: {args.capture_layer}", flush=True)

        result = result_from_saved_generation(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            capture_layer=args.capture_layer,
            activation_dtype=args.activation_dtype,
            capture_prompt_activations=args.capture_prompt_activations,
        )
        activation_path = save_imo_activation_bundle(
            output_dir=args.activation_dir,
            row=row,
            row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
            model_id=model_id,
            prompt=prompt,
            result=result,
            write_sidecars=False,
        )

        prediction = extract_final_answer_text(result.text)
        gold = str(row["Short Answer"]).strip()
        is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
        correct += int(is_correct)

        print("-" * 80)
        print("Result:")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"Captured tokens: {result.activation_capture.captured_tokens}")
        print(f"Activation shape: {tuple(result.activation_capture.activations.shape)}")
        print(f"Gold short answer: {gold}")
        print(f"Parsed prediction: {prediction}")
        print(f"Normalized exact match: {is_correct}")
        print(f"Saved activations: {activation_path}")

    print("=" * 80)
    print(f"Exact-match score: {correct}/{len(rows)}")


if __name__ == "__main__":
    main()
