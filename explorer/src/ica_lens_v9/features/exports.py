from __future__ import annotations

import csv
from pathlib import Path

import torch

from ..torch_utils import float_item


def write_feature_exports_from_artifact(
    feature_artifact_path: Path,
    *,
    ranking_csv_path: Path,
    histogram_csv_path: Path,
) -> None:
    """Export human-readable tables from the assembled ICA Lens feature artifact."""
    artifact = torch.load(feature_artifact_path, map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    write_feature_ranking_csv(ranking_csv_path, tensors=tensors)
    write_feature_histogram_csv(histogram_csv_path, tensors=tensors)


def write_feature_ranking_csv(path: Path, *, tensors: dict[str, torch.Tensor]) -> None:
    fieldnames = [
        "feature_id",
        "source_feature_id",
        "source_component_index",
        "source_sign",
        "source_side",
        "dead",
        "kurtosis",
        "excess_kurtosis",
        "activation_frequency",
        "mean",
        "variance",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id in range(int(tensors["kurtosis"].numel())):
            source_sign = int(tensors["source_sign"][feature_id].item())
            writer.writerow(
                {
                    "feature_id": feature_id,
                    "source_feature_id": int(tensors["source_feature_id"][feature_id].item()),
                    "source_component_index": int(tensors["source_component_index"][feature_id].item()),
                    "source_sign": source_sign,
                    "source_side": "positive" if source_sign > 0 else "negative",
                    "dead": bool(tensors["dead"][feature_id].item()),
                    "kurtosis": float_item(tensors["kurtosis"][feature_id]),
                    "excess_kurtosis": float_item(tensors["excess_kurtosis"][feature_id]),
                    "activation_frequency": float_item(tensors["activation_frequency"][feature_id]),
                    "mean": float_item(tensors["mean"][feature_id]),
                    "variance": float_item(tensors["variance"][feature_id]),
                    "max": float_item(tensors["max"][feature_id]),
                }
            )


def write_feature_histogram_csv(path: Path, *, tensors: dict[str, torch.Tensor]) -> None:
    counts = tensors["histogram_counts"].detach().cpu()
    edges = tensors["histogram_bin_edges"].detach().cpu().to(torch.float64)
    edges_log1p = tensors["histogram_bin_edges_log1p"].detach().cpu().to(torch.float64)
    fieldnames = [
        "feature_id",
        "source_feature_id",
        "source_component_index",
        "source_sign",
        "source_side",
        "bin_index",
        "bin_left",
        "bin_right",
        "bin_left_log1p",
        "bin_right_log1p",
        "count",
        "probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id in range(int(counts.shape[0])):
            total = max(1, int(counts[feature_id].sum().item()))
            source_sign = int(tensors["source_sign"][feature_id].item())
            for bin_index in range(int(counts.shape[1])):
                count = int(counts[feature_id, bin_index].item())
                writer.writerow(
                    {
                        "feature_id": feature_id,
                        "source_feature_id": int(tensors["source_feature_id"][feature_id].item()),
                        "source_component_index": int(tensors["source_component_index"][feature_id].item()),
                        "source_sign": source_sign,
                        "source_side": "positive" if source_sign > 0 else "negative",
                        "bin_index": bin_index,
                        "bin_left": float(edges[bin_index].item()),
                        "bin_right": float(edges[bin_index + 1].item()),
                        "bin_left_log1p": float(edges_log1p[bin_index].item()),
                        "bin_right_log1p": float(edges_log1p[bin_index + 1].item()),
                        "count": count,
                        "probability": count / total,
                    }
                )
