#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..io_utils import load_json
from ..paths import V9_ROOT
from ..torch_utils import resolve_device
from .interface import DEFAULT_METHOD, build_layer_feature_interface


DEFAULT_ICA_RUN_DIR = Path("v9/artifacts/ica/gpt2_tok1000000_c768_iter200")
DEFAULT_FEATURE_INTERFACE_ROOT = V9_ROOT / "artifacts" / "feature_interfaces"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a feature interface from fitted FastICA artifacts."
    )
    parser.add_argument("--ica-run-dir", type=Path, default=DEFAULT_ICA_RUN_DIR)
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--layers", nargs="*", default=None, help="Layers to build. Default: all layers in the ICA run manifest.")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--norm-eps", type=float, default=1e-12)
    parser.add_argument("--dead-kurtosis-threshold", type=float, default=6.0)
    parser.add_argument("--histogram-bin-width-log1p", type=float, default=0.25)
    parser.add_argument("--histogram-max-feature-value", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_feature_interface_run(
        ica_run_dir=args.ica_run_dir,
        feature_interface_root=args.feature_interface_root,
        layers=[str(layer) for layer in args.layers] if args.layers is not None else None,
        method=str(args.method),
        device_name=str(args.device),
        dtype_name=str(args.dtype),
        batch_size=int(args.batch_size),
        norm_eps=float(args.norm_eps),
        dead_kurtosis_threshold=float(args.dead_kurtosis_threshold),
        histogram_bin_width_log1p=float(args.histogram_bin_width_log1p),
        histogram_max_feature_value=float(args.histogram_max_feature_value),
        force=bool(args.force),
    )


def build_feature_interface_run(
    *,
    ica_run_dir: Path = DEFAULT_ICA_RUN_DIR,
    feature_interface_root: Path = DEFAULT_FEATURE_INTERFACE_ROOT,
    layers: list[str] | None = None,
    method: str = DEFAULT_METHOD,
    device_name: str = "cuda",
    dtype_name: str = "float32",
    batch_size: int = 8192,
    norm_eps: float = 1e-12,
    dead_kurtosis_threshold: float = 6.0,
    histogram_bin_width_log1p: float = 0.25,
    histogram_max_feature_value: float = 100.0,
    force: bool = False,
) -> Path:
    started_at = time.time()
    ica_run_dir = ica_run_dir.resolve()
    if str(method) != DEFAULT_METHOD:
        raise ValueError(f"Only {DEFAULT_METHOD!r} is implemented for now.")
    run_manifest_path = ica_run_dir / "manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing ICA run manifest: {run_manifest_path}")

    run_manifest = load_json(run_manifest_path)
    layers = [str(layer) for layer in (layers if layers is not None else run_manifest["layers"])]
    activation_manifest_path = Path(str(run_manifest["activation_manifest"])).resolve()
    activation_manifest = load_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent
    output_dir = feature_interface_root.resolve() / ica_run_dir.name / method
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]
    layer_summaries: dict[str, Any] = {}
    for layer in layers:
        summary = build_layer_feature_interface(
            ica_run_dir=ica_run_dir,
            activation_dir=activation_dir,
            activation_manifest=activation_manifest,
            output_dir=output_dir,
            layer=layer,
            device=device,
            dtype=dtype,
            batch_size=int(batch_size),
            norm_eps=float(norm_eps),
            dead_kurtosis_threshold=float(dead_kurtosis_threshold),
            histogram_bin_width_log1p=float(histogram_bin_width_log1p),
            histogram_max_feature_value=float(histogram_max_feature_value),
            force=bool(force),
        )
        layer_summaries[layer] = summary
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "artifact": f"feature_interface_{method}_{ica_run_dir.name}",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": method,
        "description": "Expose each signed ICA coordinate as two nonnegative ReLU features split at the origin, then sort exposed features by descending active-mirrored raw kurtosis.",
        "feature_rule": {
            "score": "(row_normalize(x) - mean) @ components.T",
            "positive_feature": "relu(score)",
            "negative_feature": "relu(-score)",
            "feature_id_convention": "feature_id is the sorted feature index, ordered by descending active-mirrored raw kurtosis",
            "source_feature_id_convention": "source_feature_id = 2 * source_component_index for positive side, 2 * source_component_index + 1 for negative side",
        },
        "dead_policy": {
            "dead_if": "kurtosis < dead_kurtosis_threshold",
            "kurtosis": "active-mirrored raw fourth standardized moment of the active side; not excess kurtosis",
            "dead_kurtosis_threshold": float(dead_kurtosis_threshold),
        },
        "source_ica_run_dir": str(ica_run_dir),
        "source_ica_manifest": str(run_manifest_path),
        "source_activation_manifest": str(activation_manifest_path),
        "output_dir": str(output_dir),
        "layers": layers,
        "settings": {
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "batch_size": int(batch_size),
            "norm_eps": float(norm_eps),
            "histogram_bin_width_log1p": float(histogram_bin_width_log1p),
            "histogram_max_feature_value": float(histogram_max_feature_value),
        },
        "layer_summaries": layer_summaries,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote feature interface: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
