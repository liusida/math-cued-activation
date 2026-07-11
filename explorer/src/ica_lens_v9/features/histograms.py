from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm


def compute_split_feature_histograms(
    *,
    shard_records: list[dict[str, Any]],
    activation_dir: Path,
    layer: str,
    mean: torch.Tensor,
    components: torch.Tensor,
    feature_max: torch.Tensor,
    histogram_bin_width_log1p: float,
    histogram_max_feature_value: float,
    batch_size: int,
    norm_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if histogram_bin_width_log1p <= 0:
        raise ValueError("--histogram-bin-width-log1p must be positive.")
    if histogram_max_feature_value <= 0:
        raise ValueError("--histogram-max-feature-value must be positive.")
    n_components = int(components.shape[0])
    n_features = 2 * n_components
    max_value = float(feature_max.detach().cpu().max().item())
    display_max = max(float(histogram_max_feature_value), max_value)
    max_log1p = max(1e-12, torch.log1p(torch.tensor(display_max, dtype=torch.float64)).item())
    width = float(histogram_bin_width_log1p)
    edges_log1p = torch.arange(0.0, max_log1p, width, dtype=torch.float64)
    if edges_log1p.numel() == 0 or edges_log1p[0] != 0:
        edges_log1p = torch.cat([torch.tensor([0.0], dtype=torch.float64), edges_log1p])
    if edges_log1p[-1] < max_log1p:
        edges_log1p = torch.cat([edges_log1p, torch.tensor([max_log1p], dtype=torch.float64)])
    else:
        edges_log1p[-1] = torch.tensor(max_log1p, dtype=torch.float64)
    edges = torch.expm1(edges_log1p)
    n_bins = int(edges_log1p.numel() - 1)
    pos_counts = torch.zeros((n_components, n_bins), dtype=torch.long)
    neg_counts = torch.zeros((n_components, n_bins), dtype=torch.long)
    total_expected = sum(int(shard.get("tokens", 0)) for shard in shard_records)
    pbar = tqdm(total=total_expected, unit="tok", dynamic_ncols=True, desc=f"histograms {layer}")
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
                pos_counts += _histogram_side_counts(torch.relu(scores), edges_log1p)
                neg_counts += _histogram_side_counts(torch.relu(-scores), edges_log1p)
                rows = int(batch.shape[0])
                pbar.update(rows)
                del batch, scores
    pbar.close()
    counts = torch.empty((n_features, n_bins), dtype=torch.long)
    counts[0::2] = pos_counts
    counts[1::2] = neg_counts
    return counts, edges_log1p, edges


def _histogram_side_counts(values: torch.Tensor, edges_log1p: torch.Tensor) -> torch.Tensor:
    n_components = int(values.shape[1])
    n_bins = int(edges_log1p.numel() - 1)
    edges_device = edges_log1p.to(device=values.device, dtype=torch.float64)
    log_values = torch.log1p(values.to(torch.float64))
    bin_idx = torch.bucketize(log_values, edges_device, right=False) - 1
    bin_idx = bin_idx.clamp_(0, n_bins - 1).to(torch.long)
    offsets = (torch.arange(n_components, device=values.device, dtype=torch.long) * n_bins).view(1, -1)
    flat = (bin_idx + offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=n_components * n_bins).reshape(n_components, n_bins)
    return counts.detach().cpu()
