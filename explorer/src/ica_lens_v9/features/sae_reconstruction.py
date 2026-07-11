from __future__ import annotations

"""Measure reconstruction error for the SAE counterparts used in v5.

The SAE-native metric evaluates reconstruction in the original activation
space. The direction metric normalizes both the activation and SAE
reconstruction before measuring error, so it is directly comparable to the v9
ICA reconstruction metric that operates after row L2 normalization.
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
from ..saes.counterparts import SAE_COUNTERPARTS
from ..saes.loaders import load_counterpart_lightweight_sae
from ..torch_utils import resolve_device


DEFAULT_ICA_ROOT = V9_ROOT / "artifacts" / "ica"
DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "sae_reconstruction_error"
DEFAULT_MODEL_SPECS = {
    "gpt2": {
        "run": "gpt2_tok1000000_c768_iter200",
    },
    "gemma2_2b": {
        "run": "gemma2_2b_tok1000000_c2304_iter200",
    },
    "qwen3_5_2b_base": {
        "run": "qwen3_5_2b_base_tok1000000_c2048_iter200",
    },
}


def measure_sae_reconstruction_error(
    *,
    model_name: str,
    ica_run_dir: Path,
    layers: list[str],
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 4096,
    norm_eps: float = 1e-12,
    device_name: str = "cuda",
    dtype_name: str = "float32",
    force: bool = False,
) -> Path:
    started_at = time.time()
    ica_run_dir = ica_run_dir.resolve()
    output_dir = output_root.resolve() / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    dtype = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    counterpart = SAE_COUNTERPARTS[model_name]

    ica_manifest_path = ica_run_dir / "manifest.json"
    ica_manifest = load_json(ica_manifest_path)
    activation_manifest_path = Path(str(ica_manifest["activation_manifest"])).resolve()
    activation_manifest = load_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent

    layer_outputs: dict[str, Any] = {}
    for layer in layers:
        output_csv = output_dir / f"{layer}.csv"
        if output_csv.exists() and not force:
            raise FileExistsError(f"SAE reconstruction result already exists: {output_csv}; pass --force.")
        layer_index = _layer_index(layer)
        sae_name, sae = load_counterpart_lightweight_sae(
            counterpart=counterpart,
            layer_index=layer_index,
            device=str(device),
            dtype=dtype,
        )
        rows = measure_layer_sae_reconstruction_error(
            sae=sae,
            activation_dir=activation_dir,
            activation_manifest=activation_manifest,
            layer=layer,
            batch_size=batch_size,
            norm_eps=norm_eps,
            device=device,
            dtype=dtype,
        )
        _write_rows(output_csv, rows)
        layer_outputs[layer] = {
            "csv": str(output_csv),
            "sae_name": sae_name,
            "rows": len(rows),
        }
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "description": "Reconstruction error for the fixed SAE counterpart of each v9 ICA model.",
        "model_name": model_name,
        "sae_counterpart": {
            "source": counterpart.source,
            "repo_id": counterpart.repo_id,
            "sae_model_name": counterpart.sae_model_name,
            "hidden_size": counterpart.hidden_size,
            "release_pattern": counterpart.release_pattern,
            "id_pattern_template": counterpart.id_pattern_template,
            "release_name_template": counterpart.release_name_template,
            "checkpoint_template": counterpart.checkpoint_template,
            "checkpoint_format": counterpart.checkpoint_format,
            "activation": counterpart.activation,
            "top_k": counterpart.top_k,
        },
        "source_ica_run_dir": str(ica_run_dir),
        "source_activation_manifest": str(activation_manifest_path),
        "output_dir": str(output_dir),
        "layers": layers,
        "metrics": {
            "native_normalized_mse": "sum ||x_sae - x_hat_sae||^2 / sum ||x_sae - mean(x_sae)||^2, where x_sae is the SAE counterpart's configured input space",
            "direction_normalized_mse": "sum ||z - z_hat_sae||^2 / sum ||z - mean(z)||^2, where z = x_sae / ||x_sae|| and z_hat_sae = x_hat_sae / ||x_hat_sae||",
            "mean_l0": "mean number of nonzero SAE feature activations per token",
            "native_mean_cosine": "mean cosine(x_sae, x_hat_sae)",
            "direction_mean_cosine": "mean cosine(z, z_hat_sae)",
        },
        "settings": {
            "batch_size": batch_size,
            "norm_eps": norm_eps,
            "device": str(device),
            "dtype": dtype_name,
            "sae_source": "v9 fixed counterpart constants",
            "sae_input_space": "The raw activation is transformed by sae.preprocess_input before encoding and before reconstruction metrics.",
        },
        "layer_outputs": layer_outputs,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote SAE reconstruction error results: {output_dir}")
    return output_dir


def measure_layer_sae_reconstruction_error(
    *,
    sae: Any,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    batch_size: int,
    norm_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    acc = _empty_accumulator()
    hidden_size = int(getattr(sae.cfg, "d_in"))
    d_sae = int(getattr(sae.cfg, "d_sae"))
    shard_records = layer_shard_records(activation_manifest, layer)
    total_expected = sum(int(shard.get("tokens", 0)) for shard in shard_records)
    pbar = tqdm(total=total_expected, unit="tok", dynamic_ncols=True, desc=f"sae reconstruct {layer}")

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
                target = sae.preprocess_input(raw) if hasattr(sae, "preprocess_input") else raw
                acts = sae.encode(raw)
                recon = sae.decode(acts)
                target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True).clamp_min(norm_eps)
                recon_norm = torch.linalg.vector_norm(recon, dim=1, keepdim=True).clamp_min(norm_eps)
                target_direction = target / target_norm
                recon_direction = recon / recon_norm
                l0 = (acts != 0).sum(dim=1).to(torch.float64)
                _accumulate(
                    acc,
                    raw=target,
                    recon=recon,
                    target_direction=target_direction,
                    recon_direction=recon_direction,
                    l0=l0,
                )
                pbar.update(int(raw.shape[0]))
                del raw, acts, recon, target_norm, recon_norm, target_direction, recon_direction, l0
        pbar.close()

    return [_finish_row(layer=layer, hidden_size=hidden_size, d_sae=d_sae, acc=acc)]


def _empty_accumulator() -> dict[str, torch.Tensor | float | int]:
    return {
        "tokens": 0,
        "elements": 0,
        "native_sse": 0.0,
        "native_target_ss": 0.0,
        "native_target_sum": None,
        "direction_sse": 0.0,
        "direction_target_ss": 0.0,
        "direction_target_sum": None,
        "native_cosine_sum": 0.0,
        "direction_cosine_sum": 0.0,
        "l0_sum": 0.0,
        "l0_sq_sum": 0.0,
    }


def _accumulate(
    acc: dict[str, torch.Tensor | float | int],
    *,
    raw: torch.Tensor,
    recon: torch.Tensor,
    target_direction: torch.Tensor,
    recon_direction: torch.Tensor,
    l0: torch.Tensor,
) -> None:
    token_count = int(raw.shape[0])
    acc["tokens"] = int(acc["tokens"]) + token_count
    acc["elements"] = int(acc["elements"]) + int(raw.numel())

    native_err = raw - recon
    direction_err = target_direction - recon_direction
    acc["native_sse"] = float(acc["native_sse"]) + _scalar((native_err * native_err).sum())
    acc["native_target_ss"] = float(acc["native_target_ss"]) + _scalar((raw * raw).sum())
    acc["direction_sse"] = float(acc["direction_sse"]) + _scalar((direction_err * direction_err).sum())
    acc["direction_target_ss"] = float(acc["direction_target_ss"]) + _scalar((target_direction * target_direction).sum())

    native_sum = raw.sum(dim=0, dtype=torch.float64).detach().cpu()
    direction_sum = target_direction.sum(dim=0, dtype=torch.float64).detach().cpu()
    acc["native_target_sum"] = native_sum if acc["native_target_sum"] is None else acc["native_target_sum"] + native_sum
    acc["direction_target_sum"] = (
        direction_sum if acc["direction_target_sum"] is None else acc["direction_target_sum"] + direction_sum
    )

    native_cosine = torch.nn.functional.cosine_similarity(raw, recon, dim=1, eps=1e-12)
    direction_cosine = torch.nn.functional.cosine_similarity(target_direction, recon_direction, dim=1, eps=1e-12)
    acc["native_cosine_sum"] = float(acc["native_cosine_sum"]) + _scalar(native_cosine.sum())
    acc["direction_cosine_sum"] = float(acc["direction_cosine_sum"]) + _scalar(direction_cosine.sum())
    acc["l0_sum"] = float(acc["l0_sum"]) + float(l0.sum().detach().cpu().item())
    acc["l0_sq_sum"] = float(acc["l0_sq_sum"]) + float((l0 * l0).sum().detach().cpu().item())


def _finish_row(
    *,
    layer: str,
    hidden_size: int,
    d_sae: int,
    acc: dict[str, torch.Tensor | float | int],
) -> dict[str, float | int | str]:
    tokens = int(acc["tokens"])
    elements = int(acc["elements"])
    native_sse = float(acc["native_sse"])
    direction_sse = float(acc["direction_sse"])
    native_baseline_sse = _baseline_sse(
        target_ss=float(acc["native_target_ss"]),
        target_sum=acc["native_target_sum"],
        tokens=tokens,
    )
    direction_baseline_sse = _baseline_sse(
        target_ss=float(acc["direction_target_ss"]),
        target_sum=acc["direction_target_sum"],
        tokens=tokens,
    )
    mean_l0 = float(acc["l0_sum"]) / max(tokens, 1)
    mean_l0_sq = float(acc["l0_sq_sum"]) / max(tokens, 1)
    l0_variance = max(mean_l0_sq - mean_l0 * mean_l0, 0.0)
    return {
        "layer": layer,
        "sae_k": "native",
        "tokens": tokens,
        "hidden_size": hidden_size,
        "d_sae": d_sae,
        "mean_l0": mean_l0,
        "std_l0": l0_variance**0.5,
        "native_mse": native_sse / max(elements, 1),
        "native_baseline_mse": native_baseline_sse / max(elements, 1),
        "native_normalized_mse": native_sse / max(native_baseline_sse, 1e-30),
        "native_explained_variance": 1.0 - native_sse / max(native_baseline_sse, 1e-30),
        "native_mean_cosine": float(acc["native_cosine_sum"]) / max(tokens, 1),
        "direction_mse": direction_sse / max(elements, 1),
        "direction_baseline_mse": direction_baseline_sse / max(elements, 1),
        "direction_normalized_mse": direction_sse / max(direction_baseline_sse, 1e-30),
        "direction_explained_variance": 1.0 - direction_sse / max(direction_baseline_sse, 1e-30),
        "direction_mean_cosine": float(acc["direction_cosine_sum"]) / max(tokens, 1),
    }


def _baseline_sse(*, target_ss: float, target_sum: object, tokens: int) -> float:
    if target_sum is None or tokens <= 0:
        return 0.0
    sum_sq = float((target_sum * target_sum).sum().item())
    return max(target_ss - sum_sq / tokens, 0.0)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _layer_index(layer: str) -> int:
    return int(str(layer).rsplit("_", 1)[-1])


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
