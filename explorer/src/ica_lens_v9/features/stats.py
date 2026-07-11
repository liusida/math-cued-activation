from __future__ import annotations

import torch


def empty_signed_feature_stats(n_components: int) -> dict[str, dict[str, torch.Tensor]]:
    def side() -> dict[str, torch.Tensor]:
        return {
            "active": torch.zeros(n_components, dtype=torch.float64),
            "sum1": torch.zeros(n_components, dtype=torch.float64),
            "sum2": torch.zeros(n_components, dtype=torch.float64),
            "sum4": torch.zeros(n_components, dtype=torch.float64),
            "max": torch.zeros(n_components, dtype=torch.float64),
        }

    return {"pos": side(), "neg": side()}


def accumulate_feature_side_stats(stats: dict[str, torch.Tensor], values: torch.Tensor) -> None:
    values_cpu = values.detach().cpu().to(torch.float64)
    stats["active"] += (values_cpu > 0).sum(dim=0).to(torch.float64)
    stats["sum1"] += values_cpu.sum(dim=0)
    stats["sum2"] += (values_cpu**2).sum(dim=0)
    stats["sum4"] += (values_cpu**4).sum(dim=0)
    stats["max"] = torch.maximum(stats["max"], values_cpu.max(dim=0).values)


def finish_feature_side_stats(stats: dict[str, torch.Tensor], total_rows: int) -> dict[str, torch.Tensor]:
    n = float(total_rows)
    active = stats["active"]
    mean = stats["sum1"] / n
    raw2 = stats["sum2"] / n
    variance = (raw2 - mean * mean).clamp_min(torch.finfo(torch.float64).tiny)

    # Kurtosis is active-mirrored: compute shape from nonzero magnitudes only.
    active_n = active.clamp_min(1.0)
    active_second = stats["sum2"] / active_n
    active_fourth = stats["sum4"] / active_n
    active_kurtosis = active_fourth / (active_second * active_second).clamp_min(torch.finfo(torch.float64).tiny)
    valid_active = (active > 0) & (active_second > torch.finfo(torch.float64).tiny)
    active_kurtosis = torch.where(valid_active, active_kurtosis, torch.zeros_like(active_kurtosis))
    return {
        "activation_frequency": active / n,
        "mean": mean,
        "variance": variance,
        "kurtosis": active_kurtosis,
        "excess_kurtosis": active_kurtosis - 3.0,
        "max": stats["max"],
    }


def interleave_signed_feature_stats(
    pos: dict[str, torch.Tensor],
    neg: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in pos:
        values = torch.empty(pos[key].numel() * 2, dtype=torch.float64)
        values[0::2] = pos[key]
        values[1::2] = neg[key]
        result[key] = values
    return result
