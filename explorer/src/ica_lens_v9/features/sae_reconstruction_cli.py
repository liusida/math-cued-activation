#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import load_json
from .sae_reconstruction import (
    DEFAULT_ICA_ROOT,
    DEFAULT_MODEL_SPECS,
    DEFAULT_OUTPUT_ROOT,
    measure_sae_reconstruction_error,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure SAE counterpart reconstruction error.")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_SPECS), choices=list(DEFAULT_MODEL_SPECS))
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Override the default layer list for every selected model.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--norm-eps", type=float, default=1e-12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64", "bfloat16", "float16"), default="float32")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for model_name in args.models:
        spec = DEFAULT_MODEL_SPECS[model_name]
        run_name = str(spec["run"])
        ica_run_dir = args.ica_root / run_name
        layers = [str(layer) for layer in args.layers] if args.layers is not None else _layers_from_manifest(ica_run_dir)
        measure_sae_reconstruction_error(
            model_name=model_name,
            ica_run_dir=ica_run_dir,
            layers=layers,
            output_root=args.output_root,
            batch_size=int(args.batch_size),
            norm_eps=float(args.norm_eps),
            device_name=str(args.device),
            dtype_name=str(args.dtype),
            force=bool(args.force),
        )


def _layers_from_manifest(ica_run_dir: Path) -> list[str]:
    manifest = load_json(ica_run_dir / "manifest.json")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"No layers found in ICA manifest: {ica_run_dir / 'manifest.json'}")
    return [str(layer) for layer in layers]


if __name__ == "__main__":
    main()
