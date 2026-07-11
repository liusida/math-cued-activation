#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ..io_utils import load_json
from .interface import DEFAULT_METHOD
from .reconstruction import (
    DEFAULT_ACTIVATION_THRESHOLDS,
    DEFAULT_FEATURE_INTERFACE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOP_KS,
    measure_reconstruction_error,
)


DEFAULT_MODEL_SPECS = {
    "gpt2": {
        "run": "gpt2_tok1000000_c768_iter200",
    },
    "gemma2_2b": {
        "run": "gemma2_2b_tok1000000_c2304_iter200",
    },
    "qwen3_5_2b_base": {
        "run": "qwen3_5_2b_base_tok1000000_c2048_iter200",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure ICA Lens feature reconstruction error in row-normalized activation space."
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODEL_SPECS), choices=list(DEFAULT_MODEL_SPECS))
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Override the default layer list for every selected model.",
    )
    parser.add_argument(
        "--include-top-k",
        action="store_true",
        help="Also compute diagnostic top-k rows. By default only threshold and full rows are measured.",
    )
    parser.add_argument("--top-ks", nargs="*", type=int, default=DEFAULT_TOP_KS)
    parser.add_argument("--activation-thresholds", nargs="*", type=float, default=DEFAULT_ACTIVATION_THRESHOLDS)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--norm-eps", type=float, default=1e-12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for model_name in args.models:
        spec = DEFAULT_MODEL_SPECS[model_name]
        run_name = str(spec["run"])
        feature_interface_dir = args.feature_interface_root / run_name / str(args.method)
        layers = [str(layer) for layer in args.layers] if args.layers is not None else _layers_from_manifest(feature_interface_dir)
        measure_reconstruction_error(
            model_name=model_name,
            feature_interface_dir=feature_interface_dir,
            layers=layers,
            output_root=args.output_root,
            top_ks=[int(k) for k in args.top_ks] if args.include_top_k else [],
            activation_thresholds=[float(t) for t in args.activation_thresholds],
            batch_size=int(args.batch_size),
            norm_eps=float(args.norm_eps),
            device_name=str(args.device),
            dtype_name=str(args.dtype),
            force=bool(args.force),
        )


def _layers_from_manifest(feature_interface_dir: Path) -> list[str]:
    manifest_path = feature_interface_dir / "manifest.json"
    manifest = load_json(manifest_path)
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"No layers found in feature interface manifest: {manifest_path}")
    return [str(layer) for layer in layers]


if __name__ == "__main__":
    main()
