from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..layers import layer_shard_records
from ..torch_utils import summary as tensor_summary
from .exports import write_feature_exports_from_artifact
from .histograms import compute_split_feature_histograms
from .stats import (
    accumulate_feature_side_stats,
    empty_signed_feature_stats,
    finish_feature_side_stats,
    interleave_signed_feature_stats,
)


DEFAULT_METHOD = "split_origin_relu"


def build_layer_feature_interface(
    *,
    ica_run_dir: Path,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    output_dir: Path,
    layer: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    norm_eps: float,
    dead_kurtosis_threshold: float,
    histogram_bin_width_log1p: float,
    histogram_max_feature_value: float,
    force: bool,
) -> dict[str, Any]:
    artifact_path = ica_run_dir / f"{layer}_fastica.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing ICA layer artifact: {artifact_path}")
    pt_path = output_dir / f"{layer}_features.pt"
    json_path = output_dir / f"{layer}_features.json"
    ranking_path = output_dir / f"{layer}_ranking.csv"
    histogram_csv_path = output_dir / f"{layer}_histograms.csv"
    guarded_paths = [pt_path, json_path, ranking_path, histogram_csv_path]
    if not force and any(path.exists() for path in guarded_paths):
        raise FileExistsError(f"Feature outputs already exist for {layer}; pass --force.")

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    mean = tensors["mean"].to(device=device, dtype=dtype)
    components = tensors["components"].to(device=device, dtype=dtype)
    n_components, hidden_size = int(components.shape[0]), int(components.shape[1])
    n_features = 2 * n_components

    stats = empty_signed_feature_stats(n_components)
    total_rows = 0
    shard_records = layer_shard_records(activation_manifest, layer)
    total_expected = sum(int(shard.get("tokens", 0)) for shard in shard_records)
    pbar = tqdm(total=total_expected, unit="tok", dynamic_ncols=True, desc=f"features {layer}")
    with torch.no_grad():
        for shard in shard_records:
            layer_path = shard["layers"].get(layer)
            if not isinstance(layer_path, str):
                raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
            shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
            if not isinstance(shard_tensor, torch.Tensor):
                raise TypeError(f"Expected tensor in {layer_path}, got {type(shard_tensor).__name__}.")
            for start in range(0, int(shard_tensor.shape[0]), batch_size):
                batch = shard_tensor[start : start + batch_size].to(device=device, dtype=dtype, non_blocking=True)
                batch = batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)
                scores = (batch - mean) @ components.T
                pos = torch.relu(scores)
                neg = torch.relu(-scores)
                accumulate_feature_side_stats(stats["pos"], pos)
                accumulate_feature_side_stats(stats["neg"], neg)
                rows = int(batch.shape[0])
                total_rows += rows
                pbar.update(rows)
                del batch, scores, pos, neg
    pbar.close()

    pos = finish_feature_side_stats(stats["pos"], total_rows)
    neg = finish_feature_side_stats(stats["neg"], total_rows)
    feature_stats = interleave_signed_feature_stats(pos, neg)
    feature_component_index = torch.arange(n_components, dtype=torch.long).repeat_interleave(2)
    feature_sign = torch.empty(n_features, dtype=torch.int8)
    feature_sign[0::2] = 1
    feature_sign[1::2] = -1
    feature_directions = torch.empty((n_features, hidden_size), dtype=torch.float32)
    components_cpu = components.detach().cpu().to(torch.float32)
    feature_directions[0::2] = components_cpu
    feature_directions[1::2] = -components_cpu
    decoder = torch.linalg.pinv(components_cpu.T).to(torch.float32)

    dead = feature_stats["kurtosis"] < float(dead_kurtosis_threshold)
    order = torch.argsort(feature_stats["kurtosis"], descending=True)
    histogram_counts, histogram_edges_log1p, histogram_edges = compute_split_feature_histograms(
        shard_records=shard_records,
        activation_dir=activation_dir,
        layer=layer,
        mean=mean,
        components=components,
        feature_max=feature_stats["max"],
        histogram_bin_width_log1p=histogram_bin_width_log1p,
        histogram_max_feature_value=histogram_max_feature_value,
        batch_size=batch_size,
        norm_eps=norm_eps,
        device=device,
        dtype=dtype,
    )

    feature_id = torch.arange(n_features, dtype=torch.long)
    source_feature_id = order.to(torch.long)
    output_tensors = {
        "feature_id": feature_id,
        "feature_directions": feature_directions[order],
        "preprocess_mean": mean.detach().cpu().to(torch.float32),
        "decoder": decoder,
        "source_feature_id": source_feature_id,
        "source_component_index": feature_component_index[order],
        "source_sign": feature_sign[order],
        "kurtosis": feature_stats["kurtosis"][order].to(torch.float32),
        "excess_kurtosis": feature_stats["excess_kurtosis"][order].to(torch.float32),
        "dead": dead[order],
        "activation_frequency": feature_stats["activation_frequency"][order].to(torch.float32),
        "mean": feature_stats["mean"][order].to(torch.float32),
        "variance": feature_stats["variance"][order].to(torch.float32),
        "max": feature_stats["max"][order].to(torch.float32),
        "histogram_counts": histogram_counts[order],
        "histogram_bin_edges_log1p": histogram_edges_log1p.to(torch.float32),
        "histogram_bin_edges": histogram_edges.to(torch.float32),
    }
    metadata = {
        "layer": layer,
        "method": DEFAULT_METHOD,
        "source_ica_artifact": str(artifact_path),
        "rows": int(total_rows),
        "hidden_size": hidden_size,
        "n_components": n_components,
        "n_features": n_features,
        "dead_kurtosis_threshold": float(dead_kurtosis_threshold),
        "dead_count": int(dead.sum().item()),
        "alive_count": int((~dead).sum().item()),
        "kurtosis_summary": tensor_summary(output_tensors["kurtosis"]),
        "excess_kurtosis_summary": tensor_summary(output_tensors["excess_kurtosis"]),
        "activation_frequency_summary": tensor_summary(output_tensors["activation_frequency"]),
        "feature_id_convention": "feature_id is the sorted exposed-feature index, ordered by descending active-mirrored raw kurtosis",
        "source_feature_id_convention": "source_feature_id = 2 * source_component_index for positive side, 2 * source_component_index + 1 for negative side",
        "reconstruction_convention": (
            "This feature artifact is self-contained for ICA Lens reconstruction: "
            "feature activations are relu((row_normalize(x) - preprocess_mean) @ feature_directions.T); "
            "signed component scores are recovered by summing feature activation * source_sign into "
            "source_component_index; reconstruction is preprocess_mean + signed_component_scores @ decoder."
        ),
        "kurtosis_convention": (
            "Feature kurtosis is active-mirrored raw kurtosis. For the positive side it is "
            "E[s^4 | s > 0] / E[s^2 | s > 0]^2; for the negative side it is "
            "E[s^4 | s < 0] / E[s^2 | s < 0]^2 using magnitudes. "
            "Activation frequency is stored separately."
        ),
        "ranking_csv": str(ranking_path),
        "ranking_plot": None,
        "histogram_csv": str(histogram_csv_path),
        "histogram_png_dir": None,
        "mini_histogram_svg_dir": None,
        "histogram": {
            "bins": int(histogram_counts.shape[1]),
            "bin_width_log1p": float(histogram_bin_width_log1p),
            "max_feature_value_floor": float(histogram_max_feature_value),
            "binning": "shared per layer, fixed-width bins in log1p(feature activation), extending to at least max_feature_value_floor",
            "csv_format": "one long CSV row per feature/bin",
        },
    }
    torch.save({"tensors": output_tensors, "metadata": metadata}, pt_path)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_feature_exports_from_artifact(
        pt_path,
        ranking_csv_path=ranking_path,
        histogram_csv_path=histogram_csv_path,
    )
    return metadata
