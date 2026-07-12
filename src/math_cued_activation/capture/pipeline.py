#!/usr/bin/env python3
"""Replay saved IMO-AnswerBench generations and save layer activations."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm

from ..config import PipelineConfig
from ..datasets import build_prompt, load_rows
from .._compat_scripts.run_vibethinker import (
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
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--generated-text-dir",
        type=Path,
        default=Path("outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B"),
    )
    parser.add_argument("--activation-dir", type=Path, default=Path("~/data/ICA-data/math-cued-activation"))
    parser.add_argument("--capture-layer", type=int, default=32)
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--sanity-check-next-token",
        action="store_true",
        help=(
            "On the first captured example, verify that final-layer activations predict "
            "the saved generated next tokens after final norm + lm_head."
        ),
    )
    parser.add_argument(
        "--sanity-check-max-positions",
        type=int,
        default=256,
        help="Generated next-token positions to check; use 0 to check the full continuation.",
    )
    parser.add_argument(
        "--sanity-check-chunk-size",
        type=int,
        default=64,
        help="Number of hidden states per lm_head chunk during the sanity check.",
    )
    parser.add_argument(
        "--sanity-check-strict-top1",
        action="store_true",
        help="Fail if any checked saved next token is not the model's top-1 prediction.",
    )
    parser.add_argument(
        "--capture-prompt-activations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture prompt/chat-template tokens as well as generated tokens. "
            "Enabled by default so saved bundles contain prompt_activations and "
            "generated_activations."
        ),
    )
    return parser.parse_args()


def module_device(module, fallback: torch.device) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def module_dtype(module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.dtype
    return fallback


def sanity_check_next_token_predictions(
    model,
    result,
    max_positions: int,
    chunk_size: int,
    strict_top1: bool,
) -> None:
    capture = result.activation_capture
    if capture is None:
        return
    if not capture.capture_prompt:
        raise SystemExit("Next-token sanity check requires prompt activations; pass --capture-prompt-activations.")

    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is not None and capture.layer != len(layers) - 1:
        raise SystemExit(
            f"Next-token sanity check expects the final decoder layer. "
            f"Got layer {capture.layer}, but final layer is {len(layers) - 1}."
        )
    if decoder is None or not hasattr(decoder, "norm") or not hasattr(model, "lm_head"):
        raise SystemExit("Next-token sanity check requires model.model.norm and model.lm_head.")

    sequence_token_ids = result.sequence_token_ids
    if len(sequence_token_ids) < result.prompt_tokens + 1:
        raise SystemExit("No generated tokens are available for next-token sanity check.")

    start_pos = result.prompt_tokens - 1
    total_positions = min(result.generated_tokens, capture.activations.shape[0] - start_pos - 1)
    if max_positions > 0:
        total_positions = min(total_positions, max_positions)
    if total_positions <= 0:
        raise SystemExit("No activation rows are available for next-token sanity check.")

    chunk_size = max(1, chunk_size)
    norm = decoder.norm
    lm_head = model.lm_head
    norm_device = module_device(norm, getattr(model, "device", torch.device("cpu")))
    norm_dtype = module_dtype(norm, module_dtype(lm_head, torch.float32))
    lm_head_device = module_device(lm_head, norm_device)

    mismatches: list[tuple[int, int, int]] = []
    checked = 0
    same = 0
    different = 0
    with torch.inference_mode():
        for offset in tqdm(
            range(0, total_positions, chunk_size),
            total=(total_positions + chunk_size - 1) // chunk_size,
            desc="Next-token sanity check",
            unit="chunk",
            dynamic_ncols=True,
        ):
            absolute_start = start_pos + offset
            absolute_stop = start_pos + min(offset + chunk_size, total_positions)
            hidden = capture.activations[absolute_start:absolute_stop].to(device=norm_device, dtype=norm_dtype)
            hidden = norm(hidden.unsqueeze(0)).squeeze(0)
            if hidden.device != lm_head_device:
                hidden = hidden.to(lm_head_device)
            predictions = lm_head(hidden).argmax(dim=-1).detach().cpu().tolist()
            targets = sequence_token_ids[absolute_start + 1 : absolute_stop + 1]
            for position, predicted, target in zip(range(absolute_start, absolute_stop), predictions, targets):
                checked += 1
                if int(predicted) != int(target):
                    different += 1
                    mismatches.append((position, int(predicted), int(target)))
                else:
                    same += 1

    accuracy = same / checked if checked else 0.0
    print(
        "Next-token sanity check summary: "
        f"checked={checked}, same={same}, different={different}, top1_match={accuracy:.2%}",
        flush=True,
    )
    if mismatches:
        examples = ", ".join(
            f"pos {position}: predicted {predicted}, target {target}"
            for position, predicted, target in mismatches[:5]
        )
        print(
            f"First next-token mismatches: {examples}",
            flush=True,
        )
        if strict_top1:
            raise SystemExit(
                f"Strict next-token sanity check failed: {different}/{checked} checked position(s) differed."
            )
    print(
        f"Next-token sanity check completed for {checked} generated position(s) "
        f"starting at prompt boundary.",
        flush=True,
    )


def capture_from_config(config: PipelineConfig, *, layer: int) -> None:
    args = SimpleNamespace(
        model=config.model.id,
        sample_size=config.dataset.sample_size,
        start_index=config.dataset.start_index,
        shuffle=False,
        seed=config.ica.seed,
        problem_id=None,
        answer_only=config.prompt.answer_only,
        device="cuda",
        dtype=config.model.dtype,
        generated_text_dir=config.storage.responses,
        activation_dir=config.storage.activations,
        capture_layer=layer,
        activation_dtype=config.capture.activation_dtype,
        sanity_check_next_token=config.capture.sanity_check_next_token,
        sanity_check_max_positions=256,
        sanity_check_chunk_size=64,
        sanity_check_strict_top1=False,
        capture_prompt_activations=config.capture.capture_prompt_activations,
        pipeline_config=config,
    )
    run_capture(args)


def run_capture(args: argparse.Namespace | SimpleNamespace) -> None:
    if getattr(args, "pipeline_config", None) is not None:
        rows = load_rows(args.pipeline_config)
    else:
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
    decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    final_layer = len(decoder_layers) - 1 if decoder_layers is not None else args.capture_layer
    correct = 0
    sanity_check_done = False

    for row_number, row in enumerate(tqdm(rows, desc="IMO activation capture", unit="problem"), start=1):
        dataset_index = row.get("_dataset_index")
        print("=" * 80, flush=True)
        print(
            f"Capture {row_number}/{len(rows)} | dataset row {dataset_index} | IMO-AnswerBench {row['Problem ID']}",
            flush=True,
        )

        metadata, metadata_path = load_imo_generation_bundle(args.generated_text_dir, row, model_id)
        if getattr(args, "pipeline_config", None) is not None:
            prompt = build_prompt(args.pipeline_config, row)
        else:
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
        print(
            "Replay tokens: "
            f"prompt={metadata['generation']['prompt_tokens']}, "
            f"generated={metadata['generation']['generated_tokens']}, "
            f"sequence={len(metadata['tokens']['sequence_token_ids'])}",
            flush=True,
        )

        if args.sanity_check_next_token and not sanity_check_done and args.capture_layer != final_layer:
            print(
                f"Next-token sanity check will replay final layer {final_layer}; "
                f"saved activations will still use layer {args.capture_layer}.",
                flush=True,
            )
            sanity_result = result_from_saved_generation(
                model=model,
                tokenizer=tokenizer,
                metadata=metadata,
                capture_layer=final_layer,
                activation_dtype=args.activation_dtype,
                capture_prompt_activations=True,
            )
            sanity_check_next_token_predictions(
                model=model,
                result=sanity_result,
                max_positions=args.sanity_check_max_positions,
                chunk_size=args.sanity_check_chunk_size,
                strict_top1=args.sanity_check_strict_top1,
            )
            sanity_check_done = True
            del sanity_result

        result = result_from_saved_generation(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            capture_layer=args.capture_layer,
            activation_dtype=args.activation_dtype,
            capture_prompt_activations=args.capture_prompt_activations,
        )
        if args.sanity_check_next_token and not sanity_check_done:
            sanity_check_next_token_predictions(
                model=model,
                result=result,
                max_positions=args.sanity_check_max_positions,
                chunk_size=args.sanity_check_chunk_size,
                strict_top1=args.sanity_check_strict_top1,
            )
            sanity_check_done = True
        activation_path = save_imo_activation_bundle(
            output_dir=args.activation_dir,
            row=row,
            row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
            model_id=model_id,
            prompt=prompt,
            result=result,
            write_sidecars=False,
            dataset_id=(args.pipeline_config.dataset.id if getattr(args, "pipeline_config", None) is not None else "OpenEvals/IMO-AnswerBench"),
        )

        prediction = extract_final_answer_text(result.text)
        gold = str(row["Short Answer"]).strip()
        is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
        correct += int(is_correct)

        print("-" * 80)
        print("Result:")
        print(f"Generated tokens: {result.generated_tokens}")
        capture = result.activation_capture
        print(f"Captured tokens: {capture.captured_tokens}")
        print(f"Activation shape: {tuple(capture.activations.shape)}")
        if capture.capture_prompt:
            prompt_shape = tuple(capture.activations[: capture.prompt_tokens].shape)
            generated_shape = tuple(capture.activations[capture.prompt_tokens :].shape)
        else:
            hidden_size = capture.activations.shape[-1]
            prompt_shape = (0, hidden_size)
            generated_shape = tuple(capture.activations.shape)
        print(f"Prompt activation shape: {prompt_shape}")
        print(f"Generated activation shape: {generated_shape}")
        print(f"Gold short answer: {gold}")
        print(f"Parsed prediction: {prediction}")
        print(f"Normalized exact match: {is_correct}")
        print(f"Saved activations: {activation_path}")

    print("=" * 80)
    print(f"Exact-match score: {correct}/{len(rows)}")


def main() -> None:
    run_capture(parse_args())


if __name__ == "__main__":
    main()
