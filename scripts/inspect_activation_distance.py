#!/usr/bin/env python3
"""Compare corresponding activation vectors for the same prompt in two models."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from run_vibethinker import capture_sequence_activations, load_model_and_tokenizer


DEFAULT_LEFT_MODEL = "Qwen/Qwen2.5-Coder-3B"
DEFAULT_RIGHT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_PROMPT = (
    "For a given positive integer N, Henry writes the quotient of ab divided by "
    "N+1 on the board for each integer pair (a,b) where 1 <= a,b <= N. "
    "Find all N such that the sum of the N^2 numbers Henry wrote on the board "
    "is (N^3-N^2+2)/4."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-model", default=DEFAULT_LEFT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_RIGHT_MODEL)
    parser.add_argument("--left-label", default="base")
    parser.add_argument("--right-label", default="instruct")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--show-examples", type=int, default=16)
    return parser.parse_args()


def get_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.expanduser().read_text()
    return args.prompt


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


def load_activations(model_id: str, prompt: str, args: argparse.Namespace) -> dict:
    load_args = argparse.Namespace(model=model_id, device=args.device, dtype=args.dtype)
    model, tokenizer = load_model_and_tokenizer(load_args)
    _text, token_ids = encode_prompt(tokenizer, prompt, args.chat_template)
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
    normalized = F.normalize(activations, p=2, dim=1, eps=1e-12)
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
        "token_ids": token_ids,
        "token_texts": token_texts,
        "activations": activations.cpu(),
        "normalized": normalized.cpu(),
        "raw_norms": activations.norm(dim=1).cpu(),
    }


def stat_line(name: str, values: torch.Tensor) -> str:
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return (
        f"{name}: mean={values.mean().item():+.4f} std={values.std(unbiased=False).item():.4f} "
        f"min={values.min().item():+.4f} p01={quantiles[0].item():+.4f} "
        f"p05={quantiles[1].item():+.4f} median={quantiles[2].item():+.4f} "
        f"p95={quantiles[3].item():+.4f} p99={quantiles[4].item():+.4f} "
        f"max={values.max().item():+.4f}"
    )


def main() -> None:
    args = parse_args()
    prompt = get_prompt(args)

    left = load_activations(args.left_model, prompt, args)
    right = load_activations(args.right_model, prompt, args)

    n = min(len(left["token_ids"]), len(right["token_ids"]))
    if n == 0:
        raise SystemExit("No overlapping token positions to compare.")

    same_token_ids = left["token_ids"][:n] == right["token_ids"][:n]
    left_norm = left["normalized"][:n]
    right_norm = right["normalized"][:n]
    cosine = (left_norm * right_norm).sum(dim=1)
    normalized_l2 = (left_norm - right_norm).norm(dim=1)
    raw_l2 = (left["activations"][:n] - right["activations"][:n]).norm(dim=1)

    print("=" * 100)
    print(f"{args.left_label} model: {left['model_id']}")
    print(f"{args.right_label} model: {right['model_id']}")
    print(f"Layer: {args.layer}")
    print(f"Compared positions: {n}")
    print(f"{args.left_label} tokens: {len(left['token_ids'])}")
    print(f"{args.right_label} tokens: {len(right['token_ids'])}")
    print(f"Token ids identical over compared prefix: {same_token_ids}")
    print("=" * 100)
    print(stat_line("cosine(normalized activations)", cosine))
    print(stat_line("L2(normalized activations)", normalized_l2))
    print(stat_line("L2(raw activations)", raw_l2))
    print(stat_line(f"{args.left_label} raw norm", left["raw_norms"][:n]))
    print(stat_line(f"{args.right_label} raw norm", right["raw_norms"][:n]))

    if args.show_examples > 0:
        print("=" * 100)
        print("Examples:")
        for pos in range(min(args.show_examples, n)):
            print(
                f"[{pos:04d}] cos={cosine[pos].item():+.4f} "
                f"norm_l2={normalized_l2[pos].item():.4f} "
                f"{args.left_label}={left['token_texts'][pos]!r} "
                f"{args.right_label}={right['token_texts'][pos]!r}"
            )


if __name__ == "__main__":
    main()
