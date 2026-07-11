#!/usr/bin/env python3
"""Inspect top ICA components for the same prompt in two models."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from run_vibethinker import (
    capture_sequence_activations,
    load_model_and_tokenizer,
)


QWEN_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
VIBETHINKER_MODEL = "WeiboAI/VibeThinker-3B"
DEFAULT_ICA = Path("results/ica/qwen_vibethinker_mixed_layer32_c2048_iter100.pt")
DEFAULT_PROMPT = (
    "For a given positive integer N, Henry writes the quotient of ab divided by "
    "N+1 on the board for each integer pair (a,b) where 1 <= a,b <= N. "
    "Find all N such that the sum of the N^2 numbers Henry wrote on the board "
    "is (N^3-N^2+2)/4."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one prompt through two models, capture a layer, project "
            "through a fitted mixed ICA model, and compare top components per token."
        )
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--ica-artifact", type=Path, default=DEFAULT_ICA)
    parser.add_argument("--qwen-model", default=QWEN_MODEL)
    parser.add_argument("--vibethinker-model", default=VIBETHINKER_MODEL)
    parser.add_argument("--left-model", help="Alias for --qwen-model.")
    parser.add_argument("--right-model", help="Alias for --vibethinker-model.")
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Encode prompt via each tokenizer's chat template. Default encodes raw prompt text.",
    )
    parser.add_argument(
        "--show-token-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show decoded token text for each compared position.",
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.expanduser().read_text()
    return args.prompt


def model_device(model) -> torch.device:
    try:
        return torch.device(model.device)
    except Exception:
        for parameter in model.parameters():
            return parameter.device
    return torch.device("cpu")


def encode_prompt(tokenizer, prompt: str, chat_template: bool) -> tuple[str, list[int]]:
    if chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, [int(token_id) for token_id in ids]


def load_prompt_scores(
    *,
    model_id: str,
    prompt: str,
    args: argparse.Namespace,
    components: torch.Tensor,
    mean: torch.Tensor | None,
) -> dict:
    load_args = argparse.Namespace(model=model_id, device=args.device, dtype=args.dtype)
    model, tokenizer = load_model_and_tokenizer(load_args)
    encoded_text, token_ids = encode_prompt(tokenizer, prompt, args.chat_template)
    if args.max_tokens > 0:
        token_ids = token_ids[: args.max_tokens]
    sequence_ids = torch.tensor(token_ids, dtype=torch.long)

    capture = capture_sequence_activations(
        model=model,
        sequence_ids=sequence_ids,
        prompt_tokens=len(token_ids),
        layer=args.layer,
        activation_dtype=args.activation_dtype,
        capture_prompt_activations=True,
        progress=True,
        progress_desc=f"{model_id} layer {args.layer}",
    )
    if capture is None:
        raise RuntimeError("Activation capture returned None.")

    activations = capture.activations.to(dtype=torch.float32)
    normalized_activations = F.normalize(activations, p=2, dim=1, eps=1e-12)
    activations_for_ica = normalized_activations
    if mean is not None:
        activations_for_ica = activations_for_ica - mean.to(dtype=activations_for_ica.dtype)
    scores = activations_for_ica @ components.to(dtype=activations_for_ica.dtype).T
    abs_scores = scores.abs()
    top_scores, top_indices = torch.topk(abs_scores, k=args.top_k, dim=1)

    token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
        for token_id in token_ids
    ]

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "encoded_text": encoded_text,
        "token_ids": token_ids,
        "token_texts": token_texts,
        "normalized_activations": normalized_activations.cpu(),
        "scores": scores.cpu(),
        "top_abs_scores": top_scores.cpu(),
        "top_indices": top_indices.cpu(),
    }


def format_top(indices: torch.Tensor, abs_scores: torch.Tensor, signed_scores: torch.Tensor) -> str:
    parts = []
    for component, abs_score in zip(indices.tolist(), abs_scores.tolist(), strict=True):
        score = signed_scores[int(component)].item()
        parts.append(f"{int(component)}:{score:+.2f}")
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    prompt = prompt_text(args)
    if args.left_model:
        args.qwen_model = args.left_model
    if args.right_model:
        args.vibethinker_model = args.right_model
    artifact = torch.load(args.ica_artifact.expanduser(), map_location="cpu", weights_only=False)
    components = artifact["components"].to(dtype=torch.float32)
    mean = artifact.get("mean")
    if mean is not None:
        mean = mean.to(dtype=torch.float32)

    qwen = load_prompt_scores(
        model_id=args.qwen_model,
        prompt=prompt,
        args=args,
        components=components,
        mean=mean,
    )
    vibethinker = load_prompt_scores(
        model_id=args.vibethinker_model,
        prompt=prompt,
        args=args,
        components=components,
        mean=mean,
    )

    n = min(len(qwen["token_ids"]), len(vibethinker["token_ids"]))
    same_token_ids = qwen["token_ids"][:n] == vibethinker["token_ids"][:n]
    print("=" * 120)
    print(f"Compared positions: {n}")
    print(f"{args.left_label} model: {qwen['model_id']}")
    print(f"{args.right_label} model: {vibethinker['model_id']}")
    print(f"{args.left_label} tokens: {len(qwen['token_ids'])}")
    print(f"{args.right_label} tokens: {len(vibethinker['token_ids'])}")
    print(f"Token ids identical over compared prefix: {same_token_ids}")
    print(f"Top-k components per position: k={args.top_k}")
    print("=" * 120)

    overlaps = []
    jaccards = []
    activation_cosines = []
    score_cosines = []
    for pos in range(n):
        activation_cos = torch.dot(
            qwen["normalized_activations"][pos],
            vibethinker["normalized_activations"][pos],
        ).item()
        score_cos = F.cosine_similarity(
            qwen["scores"][pos].reshape(1, -1),
            vibethinker["scores"][pos].reshape(1, -1),
            dim=1,
            eps=1e-12,
        ).item()
        activation_cosines.append(activation_cos)
        score_cosines.append(score_cos)

        q_set = set(int(x) for x in qwen["top_indices"][pos].tolist())
        v_set = set(int(x) for x in vibethinker["top_indices"][pos].tolist())
        overlap = len(q_set & v_set)
        union = len(q_set | v_set)
        jaccard = overlap / union if union else 0.0
        overlaps.append(overlap)
        jaccards.append(jaccard)

        if not args.summary_only:
            q_tok = qwen["token_texts"][pos]
            v_tok = vibethinker["token_texts"][pos]
            token_note = f"{args.left_label}={q_tok!r} {args.right_label}={v_tok!r}" if args.show_token_text else ""
            print(
                f"[{pos:03d}] act_cos={activation_cos:+.4f} score_cos={score_cos:+.4f} "
                f"overlap={overlap:02d}/{args.top_k} jaccard={jaccard:.3f} {token_note}"
            )
            print(
                f"  {args.left_label:<10} "
                + format_top(
                    qwen["top_indices"][pos],
                    qwen["top_abs_scores"][pos],
                    qwen["scores"][pos],
                )
            )
            print(
                f"  {args.right_label:<10} "
                + format_top(
                    vibethinker["top_indices"][pos],
                    vibethinker["top_abs_scores"][pos],
                    vibethinker["scores"][pos],
                )
            )

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0
    zero_overlap = sum(1 for value in overlaps if value == 0)
    mean_activation_cos = sum(activation_cosines) / len(activation_cosines) if activation_cosines else 0.0
    mean_score_cos = sum(score_cosines) / len(score_cosines) if score_cosines else 0.0
    print("=" * 120)
    if activation_cosines:
        print(
            f"Activation cosine mean/min/max: "
            f"{mean_activation_cos:+.4f} / {min(activation_cosines):+.4f} / {max(activation_cosines):+.4f}"
        )
        print(
            f"ICA score cosine mean/min/max: "
            f"{mean_score_cos:+.4f} / {min(score_cosines):+.4f} / {max(score_cosines):+.4f}"
        )
    print(f"Mean overlap@{args.top_k}: {mean_overlap:.3f}")
    print(f"Mean Jaccard@{args.top_k}: {mean_jaccard:.4f}")
    print(f"Zero-overlap positions: {zero_overlap}/{len(overlaps)}")


if __name__ == "__main__":
    main()
