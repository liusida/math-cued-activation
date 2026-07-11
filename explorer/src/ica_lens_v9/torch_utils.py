from __future__ import annotations

import torch


DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "fp64": torch.float64,
}


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def torch_dtype(name: str) -> torch.dtype:
    try:
        return DTYPE_ALIASES[name.lower()]
    except KeyError as exc:
        valid = ", ".join(sorted(DTYPE_ALIASES))
        raise ValueError(f"Unsupported dtype {name!r}; expected one of: {valid}") from exc


def summary(values: torch.Tensor) -> dict[str, float | int]:
    x = values.detach().cpu().to(torch.float64).flatten()
    return {
        "n": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p05": float(torch.quantile(x, 0.05).item()),
        "median": float(torch.quantile(x, 0.50).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "max": float(x.max().item()),
    }


def float_item(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
