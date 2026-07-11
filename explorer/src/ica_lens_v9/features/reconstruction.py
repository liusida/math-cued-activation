from __future__ import annotations

"""Measure how well ICA split-origin features reconstruct activation directions.

The main metric intentionally lives in row-L2-normalized activation space:
ICA is fitted after row normalization, so the representation is evaluated on
whether it reconstructs the normalized activation direction. Following SAE
evaluation convention, MSE is normalized by the baseline error from always
predicting the mean preprocessed activation. Original activation norms are only
restored for explicitly labeled oracle-norm diagnostics.
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..io_utils import load_json
from ..layers import layer_shard_records
from ..paths import V9_ROOT
from ..torch_utils import resolve_device
from .decoder import build_feature_decoder_from_tensors


DEFAULT_FEATURE_INTERFACE_ROOT = V9_ROOT / "artifacts" / "feature_interfaces"
DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "reconstruction_error"
DEFAULT_TOP_KS = [1, 2, 5, 10, 20, 50, 100, 200]
DEFAULT_ACTIVATION_THRESHOLDS = [1.0, 0.1, 0.01]


def measure_reconstruction_error(
    *,
    model_name: str,
    feature_interface_dir: Path,
    layers: list[str],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    top_ks: list[int] = DEFAULT_TOP_KS,
    activation_thresholds: list[float] = DEFAULT_ACTIVATION_THRESHOLDS,
    batch_size: int = 4096,
    norm_eps: float = 1e-12,
    device_name: str = "cuda",
    dtype_name: str = "float32",
    force: bool = False,
) -> Path:
    started_at = time.time()
    feature_interface_dir = feature_interface_dir.resolve()
    output_dir = output_root.resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = {"float32": torch.float32, "float64": torch.float64}[dtype_name]

    feature_manifest_path = feature_interface_dir / "manifest.json"
    feature_manifest = load_json(feature_manifest_path)
    activation_manifest_path = Path(str(feature_manifest["source_activation_manifest"])).resolve()
    activation_manifest = load_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent

    layer_outputs: dict[str, Any] = {}
    for layer in layers:
        output_csv = output_dir / f"{layer}.csv"
        if output_csv.exists() and not force:
            raise FileExistsError(f"Reconstruction result already exists: {output_csv}; pass --force.")
        rows = measure_layer_reconstruction_error(
            feature_interface_dir=feature_interface_dir,
            activation_dir=activation_dir,
            activation_manifest=activation_manifest,
            layer=layer,
            top_ks=top_ks,
            activation_thresholds=activation_thresholds,
            batch_size=batch_size,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        _write_rows(output_csv, rows)
        layer_outputs[layer] = {
            "csv": str(output_csv),
            "rows": len(rows),
            "top_ks": [row["k"] for row in rows],
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "description": "ICA split-origin feature reconstruction error. Main metric asks how well features reconstruct the normalized activation direction after row L2 normalization.",
        "model_name": model_name,
        "source_feature_interface_dir": str(feature_interface_dir),
        "source_feature_interface_manifest": str(feature_manifest_path),
        "source_activation_manifest": str(activation_manifest_path),
        "output_dir": str(output_dir),
        "layers": layers,
        "top_ks": top_ks,
        "activation_thresholds": activation_thresholds,
        "metrics": {
            "mse": "mean elementwise squared error between normalized activation direction z and reconstruction z_hat",
            "baseline_mse": "mean elementwise squared error from always predicting mean normalized activation direction mu",
            "normalized_mse": "sum ||z - z_hat||^2 / sum ||z - mu||^2",
            "explained_variance": "1 - normalized_mse",
            "mean_cosine": "mean cosine(z, z_hat)",
            "mean_active_features": "mean split-origin ICA features with positive activation per token; for k rows this is the measured number of selected positive top-k values",
            "std_active_features": "standard deviation of split-origin active-feature counts per token",
            "direction_relative_mse": "sum ||z - z_hat||^2 / sum ||z||^2; diagnostic because ||z|| is approximately 1 per token",
            "oracle_norm_relative_mse": "same reconstruction multiplied by each token's original activation norm before comparison; diagnostic only",
        },
        "settings": {
            "batch_size": batch_size,
            "norm_eps": norm_eps,
            "device": str(device),
            "dtype": dtype_name,
            "decoder": "feature artifact tensor: decoder",
            "preprocess_mean": "feature artifact tensor: preprocess_mean",
            "top_k_policy": "top k active split-origin features per token by feature activation magnitude",
            "threshold_policy": "keep split-origin ICA Lens features whose activation magnitude is greater than the threshold",
        },
        "layer_outputs": layer_outputs,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote reconstruction error results: {output_dir}")
    return output_dir


def measure_layer_reconstruction_error(
    *,
    feature_interface_dir: Path,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    top_ks: list[int],
    activation_thresholds: list[float],
    batch_size: int,
    norm_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    feature_path = feature_interface_dir / f"{layer}_features.pt"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing feature artifact: {feature_path}")

    feature_artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    feature_tensors = feature_artifact["tensors"]
    feature_decoder = build_feature_decoder_from_tensors(
        feature_tensors,
        feature_path=feature_path,
        device=device,
        dtype=dtype,
        norm_eps=float(norm_eps),
    )
    mean = feature_decoder.preprocess_mean
    n_components = feature_decoder.n_components
    hidden_size = feature_decoder.hidden_size
    n_features = feature_decoder.n_features
    effective_top_ks = sorted({int(k) for k in top_ks if int(k) > 0 and int(k) <= n_features})
    effective_thresholds = sorted({float(t) for t in activation_thresholds if float(t) > 0.0})
    threshold_keys = [_threshold_key(threshold) for threshold in effective_thresholds]
    metric_keys = ["all", *[str(k) for k in effective_top_ks], *threshold_keys]
    accum = {key: _empty_accumulator() for key in metric_keys}

    shard_records = layer_shard_records(activation_manifest, layer)
    total_expected = sum(int(shard.get("tokens", 0)) for shard in shard_records)
    pbar = tqdm(total=total_expected, unit="tok", dynamic_ncols=True, desc=f"reconstruct {layer}")
    with torch.no_grad():
        for shard in shard_records:
            layer_path = shard["layers"].get(layer)
            if not isinstance(layer_path, str):
                raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
            shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
            if not isinstance(shard_tensor, torch.Tensor):
                raise TypeError(f"Expected tensor in {layer_path}, got {type(shard_tensor).__name__}.")
            for start in range(0, int(shard_tensor.shape[0]), batch_size):
                raw = shard_tensor[start : start + batch_size].to(device=device, dtype=dtype, non_blocking=True)
                raw_norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(norm_eps)
                # Main target: reconstruct the normalized activation direction.
                # The raw token norm was removed before fitting ICA, so using it
                # for the primary metric would introduce side information.
                target = raw / raw_norm
                feature_values = feature_decoder.feature_values(target)
                component_scores = feature_decoder.component_scores(feature_values)

                full_recon = feature_decoder.reconstruct_normalized_from_component_scores(component_scores)
                full_active_counts = (feature_values > 0).sum(dim=1)
                _accumulate(
                    accum["all"],
                    target=target,
                    recon=full_recon,
                    baseline=mean,
                    raw=raw,
                    raw_norm=raw_norm,
                    active_feature_counts=full_active_counts,
                )

                if effective_top_ks:
                    max_k = max(effective_top_ks)
                    top_values, top_indices = torch.topk(feature_values, k=max_k, dim=1)
                    for k in effective_top_ks:
                        signed_scores = torch.zeros((int(raw.shape[0]), n_components), device=device, dtype=dtype)
                        idx = top_indices[:, :k]
                        values = top_values[:, :k]
                        comp_idx = feature_decoder.source_component_index[idx]
                        signs = feature_decoder.source_sign[idx]
                        signed_scores.scatter_add_(1, comp_idx, values * signs)
                        recon = feature_decoder.reconstruct_normalized_from_component_scores(signed_scores)
                        active_counts = (values > 0).sum(dim=1)
                        _accumulate(
                            accum[str(k)],
                            target=target,
                            recon=recon,
                            baseline=mean,
                            raw=raw,
                            raw_norm=raw_norm,
                            active_feature_counts=active_counts,
                        )

                for threshold, key in zip(effective_thresholds, threshold_keys):
                    mask = feature_values > float(threshold)
                    recon = feature_decoder.reconstruct_normalized_from_feature_values(feature_values * mask.to(dtype=dtype))
                    _accumulate(
                        accum[key],
                        target=target,
                        recon=recon,
                        baseline=mean,
                        raw=raw,
                        raw_norm=raw_norm,
                        active_feature_counts=mask.sum(dim=1),
                    )

                pbar.update(int(raw.shape[0]))
                del raw, raw_norm, target, feature_values, component_scores, full_recon, full_active_counts
        pbar.close()

    rows: list[dict[str, float | int | str]] = []
    for key in metric_keys:
        rows.append(_finish_row(key=key, layer=layer, hidden_size=hidden_size, acc=accum[key]))
    return rows


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "tokens": 0,
        "elements": 0,
        "active_feature_count_sum": 0.0,
        "active_feature_count_sq_sum": 0.0,
        "normalized_sse": 0.0,
        "normalized_target_ss": 0.0,
        "baseline_sse": 0.0,
        "normalized_cosine_sum": 0.0,
        "oracle_norm_sse": 0.0,
        "oracle_norm_target_ss": 0.0,
    }


def _accumulate(
    acc: dict[str, float | int],
    *,
    target: torch.Tensor,
    recon: torch.Tensor,
    baseline: torch.Tensor,
    raw: torch.Tensor,
    raw_norm: torch.Tensor,
    active_feature_counts: torch.Tensor,
) -> None:
    err = target - recon
    baseline_err = target - baseline
    token_count = int(target.shape[0])
    acc["tokens"] = int(acc["tokens"]) + token_count
    acc["elements"] = int(acc["elements"]) + int(target.numel())
    counts = active_feature_counts.to(torch.float64)
    acc["active_feature_count_sum"] = float(acc["active_feature_count_sum"]) + float(
        counts.sum().detach().cpu().item()
    )
    acc["active_feature_count_sq_sum"] = float(acc["active_feature_count_sq_sum"]) + float(
        (counts * counts).sum().detach().cpu().item()
    )
    acc["normalized_sse"] = float(acc["normalized_sse"]) + float((err * err).sum().detach().cpu().item())
    acc["normalized_target_ss"] = float(acc["normalized_target_ss"]) + float((target * target).sum().detach().cpu().item())
    acc["baseline_sse"] = float(acc["baseline_sse"]) + float((baseline_err * baseline_err).sum().detach().cpu().item())
    cosine = torch.nn.functional.cosine_similarity(target, recon, dim=1, eps=1e-12)
    acc["normalized_cosine_sum"] = float(acc["normalized_cosine_sum"]) + float(cosine.sum().detach().cpu().item())
    oracle_recon = recon * raw_norm
    oracle_err = raw - oracle_recon
    acc["oracle_norm_sse"] = float(acc["oracle_norm_sse"]) + float((oracle_err * oracle_err).sum().detach().cpu().item())
    acc["oracle_norm_target_ss"] = float(acc["oracle_norm_target_ss"]) + float((raw * raw).sum().detach().cpu().item())


def _finish_row(*, key: str, layer: str, hidden_size: int, acc: dict[str, float | int]) -> dict[str, float | int | str]:
    tokens = int(acc["tokens"])
    elements = int(acc["elements"])
    normalized_sse = float(acc["normalized_sse"])
    normalized_target_ss = float(acc["normalized_target_ss"])
    baseline_sse = float(acc["baseline_sse"])
    oracle_norm_sse = float(acc["oracle_norm_sse"])
    oracle_norm_target_ss = float(acc["oracle_norm_target_ss"])
    mse = normalized_sse / max(elements, 1)
    baseline_mse = baseline_sse / max(elements, 1)
    normalized_mse = normalized_sse / max(baseline_sse, 1e-30)
    mean_active_features = float(acc["active_feature_count_sum"]) / max(tokens, 1)
    mean_active_features_sq = float(acc["active_feature_count_sq_sum"]) / max(tokens, 1)
    active_feature_variance = max(mean_active_features_sq - mean_active_features * mean_active_features, 0.0)
    return {
        "layer": layer,
        "k": key,
        "tokens": tokens,
        "hidden_size": hidden_size,
        "mean_active_features": mean_active_features,
        "std_active_features": active_feature_variance**0.5,
        "mse": mse,
        "baseline_mse": baseline_mse,
        "normalized_mse": normalized_mse,
        "explained_variance": 1.0 - normalized_mse,
        "rmse": mse ** 0.5,
        "mean_cosine": float(acc["normalized_cosine_sum"]) / max(tokens, 1),
        "direction_relative_mse": normalized_sse / max(normalized_target_ss, 1e-30),
        "oracle_norm_mse": oracle_norm_sse / max(elements, 1),
        "oracle_norm_relative_mse": oracle_norm_sse / max(oracle_norm_target_ss, 1e-30),
        "oracle_norm_rmse": (oracle_norm_sse / max(elements, 1)) ** 0.5,
    }


def _threshold_key(threshold: float) -> str:
    text = f"{threshold:.8g}".replace(".", "p")
    return f"threshold_{text}"


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
