#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .config import get_nested, load_config, optional_str, slug_model_id
from .pipeline import capture_post_block_activations
from .runtime import iter_dataset_texts, load_model_and_tokenizer, transformer_layers
from .sampling import resolve_layers
from .storage import store_input_embedding_layer


DEFAULT_OUTPUT_ROOT = Path("/home/liusida/research/ICA-paper/data/activations_v9")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    capture_activations_from_config(
        config=args.config,
        model_id=args.model_id,
        model_short_name=args.model_short_name,
        model_dtype=args.model_dtype,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        streaming=bool(args.streaming),
        context_length=args.context_length,
        token_budget=args.token_budget,
        run_name=args.run_name,
        seed=args.seed,
        activation_dtype=args.activation_dtype,
        shard_token_budget=args.shard_token_budget,
        layers=args.layers,
        output_root=args.output_root,
        device=args.device,
        store_embedding=bool(args.store_embedding),
        force=bool(args.force),
    )


def capture_activations_from_config(
    *,
    config: Path | None = None,
    model_id: str | None = None,
    model_short_name: str | None = None,
    model_dtype: str | None = None,
    dataset_path: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    text_column: str | None = None,
    streaming: bool = False,
    context_length: int | None = None,
    token_budget: int | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    activation_dtype: str | None = None,
    shard_token_budget: int | None = None,
    layers: list[str] | None = None,
    output_root: Path | None = None,
    device: str = "auto",
    store_embedding: bool = False,
    force: bool = False,
) -> Path:
    cfg = load_config(config) if config else {}

    model_id = str(model_id or get_nested(cfg, "model", "id"))
    model_short_name = str(model_short_name or get_nested(cfg, "model", "short_name", default=slug_model_id(model_id)))
    model_dtype = str(model_dtype or get_nested(cfg, "model", "dtype", default="bfloat16"))

    dataset_path = str(dataset_path or get_nested(cfg, "dataset", "path", default="NeelNanda/pile-10k"))
    dataset_name = optional_str(dataset_name if dataset_name is not None else get_nested(cfg, "dataset", "name", default=None))
    dataset_split = str(dataset_split or get_nested(cfg, "dataset", "split", default="train"))
    text_column = str(text_column or get_nested(cfg, "dataset", "text_column", default="text"))
    streaming = bool(streaming or get_nested(cfg, "dataset", "streaming", default=False))
    context_length = int(context_length or get_nested(cfg, "dataset", "context_length", default=1024))
    token_budget = int(token_budget or get_nested(cfg, "dataset", "token_budget", default=1_000_000))

    run_name = str(run_name or get_nested(cfg, "capture", "run_name", default=f"{model_short_name}_post_block"))
    seed = int(seed if seed is not None else get_nested(cfg, "capture", "seed", default=0))
    activation_dtype = str(activation_dtype or get_nested(cfg, "capture", "activation_dtype", default="bfloat16"))
    shard_token_budget = int(shard_token_budget or get_nested(cfg, "capture", "shard_token_budget", default=250_000))

    output_root = Path(output_root or get_nested(cfg, "output", "root", default=DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    capture_dir = output_root / f"{run_name}_tok{token_budget}"
    if capture_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {capture_dir}. Pass --force intentionally.")
    capture_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model_and_tokenizer(model_id, device=device, dtype=model_dtype)
    layer_modules = transformer_layers(model)
    layer_names = [f"layer_{idx:02d}" for idx in range(len(layer_modules))]
    selected_layer_names = resolve_layers(layers, layer_names)
    selected_layer_indices = [layer_names.index(name) for name in selected_layer_names]

    texts = list(iter_dataset_texts(
        path=dataset_path,
        name=dataset_name,
        split=dataset_split,
        text_column=text_column,
        streaming=streaming,
    ))

    manifest_path = capture_post_block_activations(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        layer_modules=layer_modules,
        selected_layer_indices=selected_layer_indices,
        output_dir=capture_dir,
        run_name=run_name,
        model_id=model_id,
        model_short_name=model_short_name,
        dataset_manifest={
            "path": dataset_path,
            "name": dataset_name,
            "split": dataset_split,
            "text_column": text_column,
            "streaming": streaming,
        },
        context_length=context_length,
        token_budget=token_budget,
        activation_dtype=activation_dtype,
        shard_token_budget=shard_token_budget,
        seed=seed,
    )

    if store_embedding:
        store_input_embedding_layer(manifest_path=manifest_path, model=model, storage_dtype=activation_dtype, shard_token_budget=shard_token_budget)

    print(f"wrote activation manifest: {manifest_path}")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture v9 post-block residual activations with explicit layer hooks.")
    parser.add_argument("--config", type=Path, default=None, help="Optional v5-style TOML config to use as defaults.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-short-name", default=None)
    parser.add_argument("--model-dtype", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--activation-dtype", default=None)
    parser.add_argument("--shard-token-budget", type=int, default=None)
    parser.add_argument("--layers", nargs="*", default=None, help="Layer names or indices. Default: all transformer layers.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--store-embedding", action="store_true", help="Also store input embedding rows as an 'embedding' layer.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
