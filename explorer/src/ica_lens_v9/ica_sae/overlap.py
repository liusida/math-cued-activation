from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm.auto import tqdm

from ..paths import V9_ROOT
from ..saebench.config import DEFAULT_FEATURE_INTERFACE_ROOT, DEFAULT_METHOD, RUN_NAMES, canonical_layer
from ..saes.counterparts import SAE_COUNTERPARTS
from ..saes.loaders import load_counterpart_decoder


Basis = Literal["ica_components", "ica_lens_features"]
NearestMetric = Literal["maximum_absolute_cosine", "maximum_cosine"]

DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "ica_sae_comparison" / "overlap"
DEFAULT_ICA_ROOT = V9_ROOT / "artifacts" / "ica"
DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")


def main() -> None:
    args = parse_args()
    run_overlap(
        models=list(args.models),
        layers=[canonical_layer(layer) for layer in args.layers] if args.layers else None,
        basis=args.basis,
        ica_root=args.ica_root,
        feature_interface_root=args.feature_interface_root,
        method=args.method,
        output_root=args.output_root,
        top_k=int(args.top_k),
        sae_chunk_size=int(args.sae_chunk_size),
        device_name=str(args.device),
        dtype_name=str(args.dtype),
        norm_eps=float(args.norm_eps),
        force=bool(args.force),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v9 ICA directions to fixed SAE counterpart decoder directions.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--layers", nargs="*", default=None, help="Layer names or indices. Default: all layers per model.")
    parser.add_argument("--basis", choices=("ica_components", "ica_lens_features"), default="ica_components")
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--sae-chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--norm-eps", type=float, default=1e-12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_overlap(
    *,
    models: list[str],
    layers: list[str] | None,
    basis: Basis,
    ica_root: Path,
    feature_interface_root: Path,
    method: str,
    output_root: Path,
    top_k: int,
    sae_chunk_size: int,
    device_name: str,
    dtype_name: str,
    norm_eps: float,
    force: bool,
) -> None:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    device = _resolve_device(device_name)
    dtype = _torch_dtype(dtype_name)
    for model in models:
        run_id = RUN_NAMES[model]
        model_ica_dir = ica_root / run_id
        model_feature_dir = feature_interface_root / run_id / method
        selected_layers = layers or _layers_for_model(model_ica_dir, model_feature_dir, basis)
        result = compare_model(
            model=model,
            run_id=run_id,
            layers=selected_layers,
            basis=basis,
            ica_dir=model_ica_dir,
            feature_interface_dir=model_feature_dir,
            top_k=top_k,
            sae_chunk_size=sae_chunk_size,
            device=device,
            dtype=dtype,
            norm_eps=norm_eps,
        )
        out_dir = output_root / basis / model
        metrics_path = out_dir / f"{model}.json"
        summary_path = out_dir / f"{model}_summary.csv"
        rows_path = out_dir / f"{model}_{_row_kind(basis)}.csv"
        if not force and any(path.exists() for path in (metrics_path, summary_path, rows_path)):
            raise FileExistsError(f"Overlap outputs already exist under {out_dir}; pass --force.")
        _write_json(metrics_path, result["metrics"])
        _write_csv(summary_path, result["summary_rows"])
        _write_csv(rows_path, result["direction_rows"])
        print(f"wrote {metrics_path}")
        print(f"wrote {summary_path}")
        print(f"wrote {rows_path}")


def compare_model(
    *,
    model: str,
    run_id: str,
    layers: list[str],
    basis: Basis,
    ica_dir: Path,
    feature_interface_dir: Path,
    top_k: int,
    sae_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    norm_eps: float,
) -> dict[str, Any]:
    started_at = time.time()
    counterpart = SAE_COUNTERPARTS[model]
    metrics: dict[str, Any] = {
        "analysis": "ica_sae_direction_overlap",
        "model": model,
        "run_id": run_id,
        "basis": basis,
        "ica_dir": str(ica_dir),
        "feature_interface_dir": str(feature_interface_dir) if basis == "ica_lens_features" else None,
        "settings": {
            "nearest_metric": _nearest_metric_for_basis(basis),
            "signed_cosine_saved": True,
            "layers": layers,
            "top_k": top_k,
            "sae_chunk_size": sae_chunk_size,
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "norm_eps": norm_eps,
        },
        "sae_counterpart": _counterpart_json(counterpart),
        "layers": {},
    }
    summary_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    all_abs: list[torch.Tensor] = []
    all_signed: list[torch.Tensor] = []

    for layer in layers:
        layer_index = int(layer.removeprefix("layer_"))
        layer_started = time.time()
        dirs, direction_meta = _load_basis_directions(
            basis=basis,
            ica_dir=ica_dir,
            feature_interface_dir=feature_interface_dir,
            layer=layer,
            device=device,
            norm_eps=norm_eps,
        )
        sae_name, decoder = load_counterpart_decoder(
            counterpart=counterpart,
            layer_index=layer_index,
            hidden_size=int(dirs.shape[1]),
        )
        decoder = _normalize_decoder(decoder, hidden_size=int(dirs.shape[1]), norm_eps=norm_eps)
        matches = _nearest_sae_features(
            directions=dirs,
            decoder=decoder,
            top_k=top_k,
            chunk_size=sae_chunk_size,
            device=device,
            norm_eps=norm_eps,
            nearest_metric=_nearest_metric_for_basis(basis),
            desc=f"{model} {layer} {basis} -> SAE",
        )
        nearest_abs = matches["abs_cosine"][:, 0].detach().cpu()
        nearest_signed = matches["cosine"][:, 0].detach().cpu()
        all_abs.append(nearest_abs)
        all_signed.append(nearest_signed)
        layer_result = {
            "layer": layer,
            "elapsed_seconds": round(time.time() - layer_started, 3),
            "basis": basis,
            "basis_artifact": direction_meta["artifact"],
            "n_directions": int(dirs.shape[0]),
            "hidden_size": int(dirs.shape[1]),
            "sae_name": sae_name,
            "sae_width": int(decoder.shape[0]),
            "nearest_abs_cosine_summary": _tensor_summary(nearest_abs),
            "nearest_signed_cosine_summary": _tensor_summary(nearest_signed),
        }
        metrics["layers"][layer] = layer_result
        summary_rows.append(_summary_row(model, run_id, layer_result))
        direction_rows.extend(_direction_rows(model, run_id, layer, basis, direction_meta, matches, top_k=top_k))
        del dirs, decoder, matches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_abs:
        metrics["nearest_abs_cosine_summary"] = _tensor_summary(torch.cat(all_abs))
        metrics["nearest_signed_cosine_summary"] = _tensor_summary(torch.cat(all_signed))
    metrics["elapsed_seconds"] = round(time.time() - started_at, 3)
    return {"metrics": metrics, "summary_rows": summary_rows, "direction_rows": direction_rows}


def _load_basis_directions(
    *,
    basis: Basis,
    ica_dir: Path,
    feature_interface_dir: Path,
    layer: str,
    device: torch.device,
    norm_eps: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if basis == "ica_components":
        artifact_path = ica_dir / f"{layer}_fastica.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        dirs = artifact["tensors"]["components"].to(dtype=torch.float32)
        ids = torch.arange(int(dirs.shape[0]), dtype=torch.long)
        meta = {
            "artifact": str(artifact_path),
            "source_id": ids,
            "source_component_index": ids,
            "source_sign": torch.ones(int(dirs.shape[0]), dtype=torch.int8),
        }
    elif basis == "ica_lens_features":
        artifact_path = feature_interface_dir / f"{layer}_features.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        tensors = artifact["tensors"]
        dirs = tensors["feature_directions"].to(dtype=torch.float32)
        meta = {
            "artifact": str(artifact_path),
            "source_id": tensors["feature_id"].detach().cpu().to(torch.long),
            "source_component_index": tensors["source_component_index"].detach().cpu().to(torch.long),
            "source_sign": tensors["source_sign"].detach().cpu().to(torch.int8),
        }
    else:
        raise ValueError(f"Unsupported basis: {basis}")
    dirs = dirs / torch.linalg.vector_norm(dirs, dim=1, keepdim=True).clamp_min(norm_eps)
    return dirs.to(device=device).contiguous(), meta


def _normalize_decoder(decoder: torch.Tensor, *, hidden_size: int, norm_eps: float) -> torch.Tensor:
    decoder = _orient_decoder(decoder.detach().to(dtype=torch.float32), hidden_size=hidden_size)
    return decoder / torch.linalg.vector_norm(decoder, dim=1, keepdim=True).clamp_min(norm_eps)


def _orient_decoder(tensor: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    if int(tensor.shape[1]) == hidden_size:
        return tensor
    if int(tensor.shape[0]) == hidden_size:
        return tensor.T
    raise ValueError(f"Decoder shape {tuple(tensor.shape)} does not contain hidden size {hidden_size}.")


def _nearest_sae_features(
    *,
    directions: torch.Tensor,
    decoder: torch.Tensor,
    top_k: int,
    chunk_size: int,
    device: torch.device,
    norm_eps: float,
    nearest_metric: NearestMetric,
    desc: str,
) -> dict[str, torch.Tensor]:
    n_directions = int(directions.shape[0])
    width = int(decoder.shape[0])
    best_abs = torch.empty((n_directions, 0), device=device, dtype=torch.float32)
    best_signed = torch.empty((n_directions, 0), device=device, dtype=torch.float32)
    best_feature = torch.empty((n_directions, 0), device=device, dtype=torch.long)
    pbar = tqdm(range(0, width, chunk_size), unit="chunk", dynamic_ncols=True, desc=desc)
    try:
        for start in pbar:
            end = min(start + chunk_size, width)
            decoder_chunk = decoder[start:end]
            decoder_chunk = decoder_chunk.to(device=device, dtype=torch.float32, non_blocking=True)
            decoder_chunk = decoder_chunk / torch.linalg.vector_norm(decoder_chunk, dim=1, keepdim=True).clamp_min(norm_eps)
            cosine = directions @ decoder_chunk.T
            local_k = min(top_k, int(cosine.shape[1]))
            score = cosine.abs() if nearest_metric == "maximum_absolute_cosine" else cosine
            _local_score, local_pos = torch.topk(score, k=local_k, dim=1)
            local_abs = cosine.abs().gather(1, local_pos)
            local_signed = cosine.gather(1, local_pos)
            local_feature = local_pos + start
            candidate_abs = torch.cat([best_abs, local_abs], dim=1)
            candidate_signed = torch.cat([best_signed, local_signed], dim=1)
            candidate_feature = torch.cat([best_feature, local_feature], dim=1)
            candidate_score = candidate_abs if nearest_metric == "maximum_absolute_cosine" else candidate_signed
            _keep_score, keep_pos = torch.topk(candidate_score, k=min(top_k, int(candidate_score.shape[1])), dim=1)
            best_abs = candidate_abs.gather(1, keep_pos)
            best_signed = candidate_signed.gather(1, keep_pos)
            best_feature = candidate_feature.gather(1, keep_pos)
    finally:
        pbar.close()
    return {"feature": best_feature.cpu(), "cosine": best_signed.cpu(), "abs_cosine": best_abs.cpu()}


def _direction_rows(
    model: str,
    run_id: str,
    layer: str,
    basis: Basis,
    direction_meta: dict[str, Any],
    matches: dict[str, torch.Tensor],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    feature_ids = matches["feature"]
    signed = matches["cosine"]
    absolute = matches["abs_cosine"]
    source_id = direction_meta["source_id"]
    source_component_index = direction_meta["source_component_index"]
    source_sign = direction_meta["source_sign"]
    for direction_index in range(int(feature_ids.shape[0])):
        for rank in range(top_k):
            rows.append(
                {
                    "model": model,
                    "run_id": run_id,
                    "layer": layer,
                    "basis": basis,
                    "direction_index": direction_index,
                    "source_id": int(source_id[direction_index].item()),
                    "source_component_index": int(source_component_index[direction_index].item()),
                    "source_sign": int(source_sign[direction_index].item()),
                    "rank": rank + 1,
                    "nearest_sae_feature": int(feature_ids[direction_index, rank].item()),
                    "cosine": float(signed[direction_index, rank].item()),
                    "abs_cosine": float(absolute[direction_index, rank].item()),
                }
            )
    return rows


def _summary_row(model: str, run_id: str, layer_result: dict[str, Any]) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model,
        "run_id": run_id,
        "layer": layer_result["layer"],
        "basis": layer_result["basis"],
        "n_directions": layer_result["n_directions"],
        "hidden_size": layer_result["hidden_size"],
        "sae_width": layer_result["sae_width"],
        "sae_name": layer_result["sae_name"],
        "elapsed_seconds": layer_result["elapsed_seconds"],
    }
    for metric, key in (
        ("nearest_abs_cosine", "nearest_abs_cosine_summary"),
        ("nearest_signed_cosine", "nearest_signed_cosine_summary"),
    ):
        for stat_name, value in layer_result[key].items():
            row[f"{metric}_{stat_name}"] = value
    return row


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    values64 = values.detach().cpu().to(torch.float64)
    quantiles = torch.quantile(values64, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99], dtype=torch.float64))
    return {
        "min": float(values64.min().item()),
        "p01": float(quantiles[0].item()),
        "p05": float(quantiles[1].item()),
        "median": float(quantiles[2].item()),
        "p95": float(quantiles[3].item()),
        "p99": float(quantiles[4].item()),
        "max": float(values64.max().item()),
        "mean": float(values64.mean().item()),
        "std": float(values64.std(unbiased=False).item()),
    }


def _counterpart_json(counterpart: Any) -> dict[str, Any]:
    return {
        "source": counterpart.source,
        "repo_id": counterpart.repo_id,
        "sae_model_name": counterpart.sae_model_name,
        "hidden_size": counterpart.hidden_size,
        "hook_name_template": counterpart.hook_name_template,
        "release_pattern": counterpart.release_pattern,
        "id_pattern_template": counterpart.id_pattern_template,
        "release_name_template": counterpart.release_name_template,
        "checkpoint_template": counterpart.checkpoint_template,
        "checkpoint_format": counterpart.checkpoint_format,
        "decoder_key": counterpart.decoder_key,
        "activation": counterpart.activation,
        "top_k": counterpart.top_k,
        "layer_checkpoints": {str(k): v for k, v in counterpart.layer_checkpoints.items()},
    }


def _layers_for_model(ica_dir: Path, feature_interface_dir: Path, basis: Basis) -> list[str]:
    root = feature_interface_dir if basis == "ica_lens_features" else ica_dir
    suffix = "_features.pt" if basis == "ica_lens_features" else "_fastica.pt"
    layers = sorted(path.name.removesuffix(suffix) for path in root.glob(f"layer_*{suffix}"))
    if not layers:
        raise FileNotFoundError(f"No {basis} layer artifacts found in {root}")
    return layers


def _row_kind(basis: Basis) -> str:
    return "features" if basis == "ica_lens_features" else "components"


def _nearest_metric_for_basis(basis: Basis) -> NearestMetric:
    if basis == "ica_lens_features":
        return "maximum_cosine"
    return "maximum_absolute_cosine"


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _torch_dtype(name: str) -> torch.dtype:
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    leading = [
        "model",
        "run_id",
        "layer",
        "basis",
        "direction_index",
        "source_id",
        "source_component_index",
        "source_sign",
        "rank",
        "nearest_sae_feature",
    ]
    fieldnames = [key for key in leading if key in fieldnames] + [key for key in fieldnames if key not in leading]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
