from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..io_utils import load_toml
from ..layers import activation_layers, layer_shard_records


def load_activation_config(config_path: Path) -> dict[str, Any]:
    cfg = load_toml(config_path)
    rel = str(dict(cfg.get("activation", {})).get("config", ""))
    if rel:
        activation_path = Path(rel)
        if not activation_path.is_absolute():
            activation_path = config_path.resolve().parent / activation_path
        return load_toml(activation_path)
    if "capture" in cfg and "model" in cfg and "dataset" in cfg:
        return cfg
    raise ValueError(f"{config_path} is neither an activation config nor a fit config with [activation].config.")


def activation_manifest_path(
    *,
    explicit: Path | None,
    activation_root: Path,
    activation_cfg: dict[str, Any],
    token_budget: int,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    run_name = str(activation_cfg["capture"]["run_name"])
    return activation_root.resolve() / f"{run_name}_tok{token_budget}" / "manifest.json"


def resolve_fit_layers(cli_layers: list[str] | None, manifest: dict[str, Any]) -> list[str]:
    all_layers = activation_layers(manifest)
    layers = cli_layers if cli_layers is not None else [layer for layer in all_layers if layer != "embedding"]
    missing = [layer for layer in layers if layer not in all_layers]
    if missing:
        raise ValueError(f"Layer(s) not found in activation manifest: {', '.join(missing)}")
    return layers


def load_layer_activations(
    *,
    capture_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    fit_rows: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    rows_seen = 0
    total = fit_rows or sum(int(shard.get("tokens", 0)) for shard in layer_shard_records(manifest, layer))
    pbar = tqdm(total=total, unit="tok", dynamic_ncols=True, desc=f"load {layer}")
    try:
        for shard in layer_shard_records(manifest, layer):
            remaining = None if fit_rows is None else fit_rows - rows_seen
            if remaining is not None and remaining <= 0:
                break
            layer_path = shard["layers"].get(layer)
            if not isinstance(layer_path, str):
                raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
            tensor = torch.load(capture_dir / layer_path, map_location="cpu")
            if remaining is not None and int(tensor.shape[0]) > remaining:
                tensor = tensor[:remaining]
            tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
            chunks.append(tensor)
            rows_seen += int(tensor.shape[0])
            pbar.update(int(tensor.shape[0]))
    finally:
        pbar.close()
    if not chunks:
        raise ValueError(f"No activation rows loaded for {layer}.")
    return torch.cat(chunks, dim=0)
