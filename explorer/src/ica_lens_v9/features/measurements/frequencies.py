from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ...layers import activation_layers, layer_shard_records
from ...paths import DEFAULT_FEATURE_INDEX, V9_ROOT


DEFAULT_FEATURE_INTERFACE_DIR = (
    V9_ROOT / "artifacts" / "feature_interfaces" / "gpt2_tok1000000_c768_iter200" / "split_origin_relu"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-feature thresholded activation frequencies and write SQLite.")
    parser.add_argument("--feature-interface-dir", type=Path, default=DEFAULT_FEATURE_INTERFACE_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--layers", nargs="*", default=None, help="Layers to update. Default: all layers in the feature manifest.")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = compute_threshold_frequency_run(
        feature_interface_dir=args.feature_interface_dir,
        db_path=args.db_path,
        layers=args.layers,
        threshold=float(args.threshold),
        batch_size=int(args.batch_size),
        device=torch.device(str(args.device)),
        dtype=_torch_dtype(str(args.dtype)),
    )
    print(f"updated {len(output)} layer(s) in {args.db_path}")


def compute_threshold_frequency_run(
    *,
    feature_interface_dir: Path,
    db_path: Path = DEFAULT_FEATURE_INDEX,
    layers: list[str] | None = None,
    threshold: float = 1.0,
    batch_size: int = 8192,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> list[dict[str, Any]]:
    if float(threshold) != 1.0:
        raise ValueError("Only threshold=1.0 is currently wired to SQLite column activation_frequency_gt_1.")
    feature_interface_dir = feature_interface_dir.resolve()
    manifest = _read_json(feature_interface_dir / "manifest.json")
    activation_manifest_path = Path(str(manifest["source_activation_manifest"])).resolve()
    activation_manifest = _read_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent
    run_id = feature_interface_dir.parent.name
    selected_layers = list(layers) if layers else _manifest_layers(manifest, activation_manifest)

    summaries = []
    with _connect_for_update(db_path.resolve()) as conn:
        _ensure_threshold_column(conn)
        for layer in selected_layers:
            frequencies = compute_layer_threshold_frequency(
                feature_interface_dir=feature_interface_dir,
                activation_dir=activation_dir,
                activation_manifest=activation_manifest,
                layer=layer,
                threshold=float(threshold),
                batch_size=int(batch_size),
                device=device,
                dtype=dtype,
            )
            _write_layer_threshold_frequency(
                conn,
                run_id=run_id,
                layer=layer,
                frequencies=frequencies,
            )
            summaries.append(
                {
                    "run_id": run_id,
                    "layer": layer,
                    "threshold": float(threshold),
                    "n_features": int(frequencies.numel()),
                    "mean_frequency": float(frequencies.mean().item()),
                    "max_frequency": float(frequencies.max().item()),
                }
            )
        conn.commit()
    return summaries


def compute_layer_threshold_frequency(
    *,
    feature_interface_dir: Path,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    threshold: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    feature_artifact = torch.load(feature_interface_dir / f"{layer}_features.pt", map_location="cpu", weights_only=False)
    feature_tensors = feature_artifact["tensors"]
    layer_metadata = feature_artifact["metadata"]
    ica_artifact = torch.load(Path(str(layer_metadata["source_ica_artifact"])), map_location="cpu", weights_only=False)

    feature_directions = feature_tensors["feature_directions"].to(device=device, dtype=dtype)
    mean = ica_artifact["tensors"]["mean"].to(device=device, dtype=dtype)
    norm_eps = float(ica_artifact["metadata"].get("norm_eps", 1e-12))
    counts = torch.zeros(int(feature_directions.shape[0]), dtype=torch.float64)
    total_rows = 0
    shard_records = layer_shard_records(activation_manifest, layer)
    total_expected = sum(int(shard.get("tokens", 0)) for shard in shard_records)
    with torch.no_grad(), tqdm(total=total_expected, desc=f"p(f>{threshold:g}) {layer}", unit="tok", dynamic_ncols=True) as pbar:
        for shard in shard_records:
            layer_path = shard["layers"].get(layer)
            if not isinstance(layer_path, str):
                raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
            shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
            if not isinstance(shard_tensor, torch.Tensor):
                raise TypeError(f"Expected tensor in {layer_path}, got {type(shard_tensor).__name__}.")
            for start in range(0, int(shard_tensor.shape[0]), batch_size):
                batch = shard_tensor[start : start + batch_size].to(device=device, dtype=dtype, non_blocking=True)
                normalized = batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)
                activations = torch.relu((normalized - mean) @ feature_directions.T)
                counts += (activations > float(threshold)).sum(dim=0).detach().cpu().to(torch.float64)
                rows = int(batch.shape[0])
                total_rows += rows
                pbar.update(rows)
                del batch, normalized, activations
    if total_rows <= 0:
        raise ValueError(f"No activation rows found for {layer}.")
    return counts / float(total_rows)


def _write_layer_threshold_frequency(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    layer: str,
    frequencies: torch.Tensor,
) -> None:
    rows = [(float(value), run_id, layer, feature_id) for feature_id, value in enumerate(frequencies.tolist())]
    conn.executemany(
        """
        UPDATE features
        SET activation_frequency_gt_1 = ?
        WHERE run_id = ? AND layer = ? AND feature_id = ?
        """,
        rows,
    )


def _connect_for_update(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_threshold_column(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    if "activation_frequency_gt_1" not in existing:
        conn.execute("ALTER TABLE features ADD COLUMN activation_frequency_gt_1 REAL")


def _manifest_layers(feature_manifest: dict[str, Any], activation_manifest: dict[str, Any]) -> list[str]:
    layers = feature_manifest.get("layers")
    if isinstance(layers, list) and all(isinstance(layer, str) for layer in layers):
        return layers
    return activation_layers(activation_manifest)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


if __name__ == "__main__":
    main()
