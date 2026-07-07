#!/usr/bin/env python3
"""Write .txt and .json sidecars for existing activation .pt bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_ROOT = Path("~/data/ICA-data/math-cued-activation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Activation root or .pt file. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing .txt/.json sidecars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    paths = [root] if root.is_file() else sorted(root.rglob("*.pt"))
    if not paths:
        raise SystemExit(f"No .pt files found under {root}")

    written = 0
    skipped = 0
    for path in paths:
        text_path = path.with_suffix(".txt")
        json_path = path.with_suffix(".json")
        if not args.overwrite and text_path.exists() and json_path.exists():
            skipped += 1
            continue

        bundle = torch.load(path, map_location="cpu")
        metadata = {key: value for key, value in bundle.items() if key != "activations"}
        metadata.setdefault("storage", {})
        metadata["storage"].update(
            {
                "relative_path": path.name,
                "text_relative_path": text_path.name,
                "json_relative_path": json_path.name,
            }
        )

        json_path.write_text(json.dumps(to_jsonable(metadata), indent=2, ensure_ascii=False) + "\n")
        text_path.write_text(format_text_sidecar(metadata))
        print(f"wrote {text_path}")
        print(f"wrote {json_path}")
        written += 1

    print(f"Done. Wrote sidecars for {written} file(s); skipped {skipped}.")


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def format_text_sidecar(metadata: dict) -> str:
    problem = metadata.get("problem", {})
    text = metadata.get("text", {})
    capture = metadata.get("capture", {})
    generation = metadata.get("generation", {})
    return (
        f"Dataset: {problem.get('dataset', 'unknown')}\n"
        f"Problem ID: {problem.get('problem_id', 'unknown')}\n"
        f"Category: {problem.get('category', 'unknown')} / {problem.get('subcategory', 'unknown')}\n"
        f"Source: {problem.get('source', 'unknown')}\n"
        f"Model: {generation.get('model', 'unknown')}\n"
        f"Activation: {metadata.get('activation_name', 'unknown')} "
        f"shape={metadata.get('activation_shape', 'unknown')}\n"
        f"Generated tokens: {capture.get('generated_tokens', 'unknown')}\n"
        f"Captured tokens: {capture.get('captured_tokens', 'unknown')}\n"
        f"Gold short answer: {problem.get('short_answer', 'unknown')}\n"
        "\n"
        "=== Problem ===\n"
        f"{problem.get('problem', '')}\n"
        "\n"
        "=== Prompt ===\n"
        f"{text.get('prompt', '')}\n"
        "\n"
        "=== Generated ===\n"
        f"{text.get('generated', '')}\n"
    )


if __name__ == "__main__":
    main()
