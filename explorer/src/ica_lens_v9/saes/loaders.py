from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Any

import torch

from ..paths import V5_ROOT
from .counterparts import SaeCounterpart


def load_counterpart_sae(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[str, Any]:
    _ensure_saebench_path(counterpart.model_name)
    if counterpart.source == "sae_lens_registry":
        return _load_registry_sae(counterpart=counterpart, layer_index=layer_index, device=device, dtype=dtype)
    if counterpart.source == "custom_checkpoint":
        return _load_custom_checkpoint_sae(counterpart=counterpart, layer_index=layer_index, device=device, dtype=dtype)
    raise ValueError(f"Unsupported SAE counterpart source: {counterpart.source!r}")


def load_counterpart_lightweight_sae(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[str, Any]:
    """Load a minimal local SAE module directly from configured checkpoint tensors."""
    weights_path = _resolve_checkpoint_path(counterpart, layer_index)
    weights = _load_weights(weights_path, checkpoint_format=counterpart.checkpoint_format)
    w_dec = _decoder_weight(weights, hidden_size=counterpart.hidden_size, preferred_key=counterpart.decoder_key).to(torch.float32)
    w_enc = _encoder_weight(weights, hidden_size=counterpart.hidden_size, d_sae=int(w_dec.shape[0])).to(torch.float32)
    b_enc = _vector_weight(weights, names=("b_enc", "encoder.bias"), length=int(w_dec.shape[0]))
    b_dec = _vector_weight(weights, names=("b_dec", "decoder.bias"), length=counterpart.hidden_size)
    threshold = _optional_vector_weight(weights, names=("threshold",), length=int(w_dec.shape[0]))
    if counterpart.source == "sae_lens_registry":
        release = _required(counterpart.release_pattern, "release_pattern")
        sae_id = _counterpart_display_id(counterpart, layer_index)
        name = f"{release}::{sae_id}"
    elif counterpart.source == "custom_checkpoint":
        name = _required(counterpart.release_name_template, "release_name_template").format(layer=layer_index)
    else:
        name = f"{counterpart.repo_id}::{Path(weights_path).name}"
    sae = LightweightCheckpointSAE(
        w_enc=w_enc,
        w_dec=w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        counterpart=counterpart,
        layer_index=layer_index,
        checkpoint_path=str(weights_path),
        device=device,
        dtype=dtype,
    )
    return name, sae


def load_counterpart_decoder(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    hidden_size: int | None = None,
) -> tuple[str, torch.Tensor]:
    """Load only the SAE decoder matrix, avoiding SAE Lens / SAEBench runtime imports."""
    weights_path = _resolve_checkpoint_path(counterpart, layer_index)
    weights = _load_weights(weights_path, checkpoint_format=counterpart.checkpoint_format)
    decoder = _decoder_weight(
        weights,
        hidden_size=int(hidden_size or counterpart.hidden_size),
        preferred_key=counterpart.decoder_key,
    ).to(torch.float32)
    if counterpart.source == "sae_lens_registry":
        release = _required(counterpart.release_pattern, "release_pattern")
        sae_id = _counterpart_display_id(counterpart, layer_index)
        name = f"{release}::{sae_id}"
    elif counterpart.source == "custom_checkpoint":
        name = _required(counterpart.release_name_template, "release_name_template").format(layer=layer_index)
    else:
        name = f"{counterpart.repo_id}::{Path(weights_path).name}"
    return name, decoder.contiguous()


class LightweightCheckpointSAE(torch.nn.Module):
    def __init__(
        self,
        *,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor | None,
        counterpart: SaeCounterpart,
        layer_index: int,
        checkpoint_path: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        decoder_norms = w_dec.norm(dim=1).clamp_min(1e-12)
        w_dec_unit = w_dec / decoder_norms[:, None]
        self.W_enc = torch.nn.Parameter(w_enc.contiguous(), requires_grad=False)
        self.W_dec = torch.nn.Parameter(w_dec_unit.contiguous(), requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc.contiguous(), requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec.contiguous(), requires_grad=False)
        if threshold is not None:
            self.threshold = torch.nn.Parameter(threshold.contiguous(), requires_grad=False)
        else:
            self.threshold = None
        self.register_buffer("decoder_norms", decoder_norms.reshape(1, -1).contiguous())
        self.activation = counterpart.activation
        self.top_k = int(counterpart.top_k) if counterpart.top_k is not None else None
        self.apply_b_dec_to_input = bool(counterpart.apply_b_dec_to_input)
        self.normalize_activations = counterpart.normalize_activations
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = _SimpleConfig(
            model_name=counterpart.sae_model_name,
            d_in=counterpart.hidden_size,
            d_sae=int(w_dec.shape[0]),
            hook_layer=layer_index,
            hook_name=counterpart.hook_name_template.format(layer=layer_index),
            architecture="v9_lightweight_checkpoint_sae",
            activation_fn_str=counterpart.activation,
            checkpoint_path=checkpoint_path,
            checkpoint_format=counterpart.checkpoint_format,
            top_k=self.top_k,
            apply_b_dec_to_input=self.apply_b_dec_to_input,
            normalize_activations=self.normalize_activations,
        )
        self.to(device=self.device, dtype=self.dtype)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.W_enc.dtype)
        if self.normalize_activations == "none":
            return x
        if self.normalize_activations == "layer_norm":
            mean = x.mean(dim=-1, keepdim=True)
            variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            return (x - mean) * torch.rsqrt(variance + 1e-5)
        raise ValueError(f"Unsupported SAE activation preprocessing: {self.normalize_activations!r}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.preprocess_input(x)
        x_centered = x_in - self.b_dec if self.apply_b_dec_to_input else x_in
        acts = x_centered @ self.W_enc + self.b_enc
        if self.threshold is not None or self.activation == "jumprelu":
            if self.threshold is None:
                raise ValueError("JumpReLU SAE is missing threshold weights.")
            acts = torch.where(acts > self.threshold.to(dtype=acts.dtype, device=acts.device), acts, torch.zeros_like(acts))
        elif self.activation == "topk":
            acts = torch.relu(acts)
            if self.top_k is not None and self.top_k < int(acts.shape[-1]):
                values, indices = torch.topk(acts, k=self.top_k, dim=-1)
                filtered = torch.zeros_like(acts)
                filtered.scatter_(-1, indices, values)
                acts = filtered
        elif self.activation == "identity":
            pass
        else:
            acts = torch.relu(acts)
        return acts * self.decoder_norms.to(dtype=acts.dtype, device=acts.device)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts.to(device=self.device, dtype=self.W_dec.dtype) @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _load_registry_sae(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[str, Any]:
    from sae_bench.sae_bench_utils.general_utils import load_and_format_sae
    from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex

    release_pattern = _required(counterpart.release_pattern, "release_pattern")
    id_pattern = _registry_id_pattern(counterpart, layer_index)
    selected = get_saes_from_regex(release_pattern, id_pattern)
    if len(selected) != 1:
        raise ValueError(
            f"Expected one SAE counterpart for {counterpart.model_name} layer {layer_index}, got {selected!r}."
        )
    sae_release, sae_object_or_id = selected[0]
    sae_id, sae, _sparsity = load_and_format_sae(sae_release, sae_object_or_id, device)
    sae = sae.to(device=device, dtype=dtype)
    return f"{sae_release}::{sae_id}", sae


def _load_custom_checkpoint_sae(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[str, Any]:
    weights_path = _resolve_checkpoint_path(counterpart, layer_index)
    weights = _load_weights(weights_path, checkpoint_format=counterpart.checkpoint_format)
    w_dec = _decoder_weight(weights, hidden_size=counterpart.hidden_size, preferred_key=counterpart.decoder_key).to(torch.float32)
    w_enc = _encoder_weight(weights, hidden_size=counterpart.hidden_size, d_sae=int(w_dec.shape[0])).to(torch.float32)
    b_enc = _vector_weight(weights, names=("b_enc", "encoder.bias"), length=int(w_dec.shape[0]))
    b_dec = _vector_weight(weights, names=("b_dec", "decoder.bias"), length=counterpart.hidden_size)
    threshold = _optional_vector_weight(weights, names=("threshold",), length=int(w_dec.shape[0]))
    decoder_norms = w_dec.norm(dim=1).clamp_min(1e-12)
    w_dec = w_dec / decoder_norms[:, None]

    class CheckpointSAELike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W_enc = torch.nn.Parameter(w_enc.contiguous())
            self.W_dec = torch.nn.Parameter(w_dec.contiguous())
            self.b_enc = torch.nn.Parameter(b_enc.contiguous())
            self.b_dec = torch.nn.Parameter(b_dec.contiguous())
            if threshold is not None:
                self.threshold = torch.nn.Parameter(threshold.contiguous(), requires_grad=False)
            else:
                self.threshold = None
            self.register_buffer("decoder_norms", decoder_norms.reshape(1, -1).contiguous())
            self.activation = counterpart.activation
            self.top_k = int(counterpart.top_k) if counterpart.top_k is not None else None
            self.apply_b_dec_to_input = bool(counterpart.apply_b_dec_to_input)
            self.normalize_activations = counterpart.normalize_activations
            self.device = torch.device(device)
            self.dtype = dtype
            self.cfg = _SimpleConfig(
                model_name=counterpart.sae_model_name,
                d_in=counterpart.hidden_size,
                d_sae=int(w_dec.shape[0]),
                hook_layer=layer_index,
                hook_name=counterpart.hook_name_template.format(layer=layer_index),
                architecture="checkpoint_sae_like_post_activation_feature_scaling",
                activation_fn_str=counterpart.activation,
                checkpoint_path=str(weights_path),
                checkpoint_format=counterpart.checkpoint_format,
                top_k=self.top_k,
                apply_b_dec_to_input=self.apply_b_dec_to_input,
                normalize_activations=self.normalize_activations,
            )
            self.to(device=self.device, dtype=self.dtype)

        def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
            x = x.to(device=self.device, dtype=self.W_enc.dtype)
            if self.normalize_activations == "none":
                return x
            if self.normalize_activations == "layer_norm":
                mean = x.mean(dim=-1, keepdim=True)
                variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
                return (x - mean) * torch.rsqrt(variance + 1e-5)
            raise ValueError(f"Unsupported SAE activation preprocessing: {self.normalize_activations!r}")

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            x_in = self.preprocess_input(x)
            x_centered = x_in - self.b_dec if self.apply_b_dec_to_input else x_in
            acts = x_centered @ self.W_enc + self.b_enc
            if self.threshold is not None or self.activation == "jumprelu":
                if self.threshold is None:
                    raise ValueError("JumpReLU SAE is missing threshold weights.")
                acts = torch.where(acts > self.threshold.to(dtype=acts.dtype, device=acts.device), acts, torch.zeros_like(acts))
            elif self.activation == "topk":
                acts = torch.relu(acts)
                if self.top_k is not None and self.top_k < int(acts.shape[-1]):
                    values, indices = torch.topk(acts, k=self.top_k, dim=-1)
                    filtered = torch.zeros_like(acts)
                    filtered.scatter_(-1, indices, values)
                    acts = filtered
            elif self.activation == "identity":
                pass
            else:
                acts = torch.relu(acts)
            return acts * self.decoder_norms.to(dtype=acts.dtype, device=acts.device)

        def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
            return feature_acts.to(dtype=self.W_dec.dtype) @ self.W_dec + self.b_dec

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

    release_name = _required(counterpart.release_name_template, "release_name_template").format(layer=layer_index)
    return release_name, CheckpointSAELike()


@dataclass
class _SimpleConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    architecture: str
    activation_fn_str: str
    checkpoint_path: str
    checkpoint_format: str | None
    top_k: int | None
    apply_b_dec_to_input: bool
    normalize_activations: str


def _registry_id_pattern(counterpart: SaeCounterpart, layer_index: int) -> str:
    if counterpart.layer_checkpoints:
        checkpoint = counterpart.layer_checkpoints[layer_index]
        return checkpoint.removesuffix("/params.npz").removesuffix(".npz")
    return _required(counterpart.id_pattern_template, "id_pattern_template").format(layer=layer_index)


def _counterpart_display_id(counterpart: SaeCounterpart, layer_index: int) -> str:
    if counterpart.layer_checkpoints:
        return counterpart.layer_checkpoints[layer_index].removesuffix("/params.npz").removesuffix(".npz")
    return counterpart.hook_name_template.format(layer=layer_index)


def _resolve_checkpoint_path(counterpart: SaeCounterpart, layer_index: int) -> str:
    from huggingface_hub import hf_hub_download

    if counterpart.layer_checkpoints:
        filename = counterpart.layer_checkpoints[layer_index]
    else:
        filename = _required(counterpart.checkpoint_template, "checkpoint_template").format(layer=layer_index)
    return hf_hub_download(repo_id=counterpart.repo_id, filename=filename)


def _load_weights(path: str, *, checkpoint_format: str) -> dict[str, Any]:
    import numpy as np
    from safetensors.torch import load_file

    fmt = checkpoint_format.lower()
    if not fmt:
        suffix = Path(path).suffix.lower()
        fmt = {".safetensors": "safetensors", ".npz": "npz", ".pt": "torch", ".pth": "torch"}.get(suffix, "")
    if fmt == "safetensors":
        return dict(load_file(path, device="cpu"))
    if fmt == "npz":
        with np.load(path) as arrays:
            return {key: torch.from_numpy(arrays[key]) for key in arrays.files}
    if fmt in {"torch", "pt", "pth"}:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected torch checkpoint {path} to contain a dict, got {type(loaded).__name__}.")
        return _flatten_tensor_dict(loaded)
    raise ValueError(f"Could not determine SAE checkpoint format for {path}.")


def _flatten_tensor_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, torch.Tensor):
            flattened[name] = value
            flattened[str(key)] = value
        elif isinstance(value, dict):
            flattened.update(_flatten_tensor_dict(value, name))
    return flattened


def _decoder_weight(weights: dict[str, Any], *, hidden_size: int, preferred_key: str | None = None) -> torch.Tensor:
    if preferred_key is not None:
        return _orient_decoder(weights[preferred_key], hidden_size=hidden_size)
    for key in ("W_dec", "decoder.weight", "W_dec.weight", "decoder.W_dec"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_decoder(tensor, hidden_size=hidden_size)
    candidates = [tensor for tensor in weights.values() if getattr(tensor, "ndim", None) == 2 and hidden_size in tensor.shape]
    if len(candidates) == 1:
        return _orient_decoder(candidates[0], hidden_size=hidden_size)
    raise KeyError(f"Could not identify SAE decoder weight. Available keys: {sorted(weights)}")


def _encoder_weight(weights: dict[str, Any], *, hidden_size: int, d_sae: int) -> torch.Tensor:
    for key in ("W_enc", "encoder.weight", "W_enc.weight", "encoder.W_enc"):
        tensor = weights.get(key)
        if tensor is not None:
            return _orient_encoder(tensor, hidden_size=hidden_size, d_sae=d_sae)
    return _decoder_weight(weights, hidden_size=hidden_size).T.contiguous()


def _orient_decoder(tensor: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    if int(tensor.shape[1]) == hidden_size:
        return tensor
    if int(tensor.shape[0]) == hidden_size:
        return tensor.T
    raise ValueError(f"Decoder weight shape {tuple(tensor.shape)} does not contain hidden size {hidden_size}.")


def _orient_encoder(tensor: torch.Tensor, *, hidden_size: int, d_sae: int) -> torch.Tensor:
    if tuple(tensor.shape) == (hidden_size, d_sae):
        return tensor
    if tuple(tensor.shape) == (d_sae, hidden_size):
        return tensor.T
    if int(tensor.shape[0]) == hidden_size:
        return tensor
    if int(tensor.shape[1]) == hidden_size:
        return tensor.T
    raise ValueError(f"Encoder weight shape {tuple(tensor.shape)} does not match hidden={hidden_size}, d_sae={d_sae}.")


def _vector_weight(weights: dict[str, Any], *, names: tuple[str, ...], length: int) -> torch.Tensor:
    for name in names:
        tensor = weights.get(name)
        if tensor is not None and getattr(tensor, "ndim", None) == 1 and int(tensor.shape[0]) == length:
            return tensor.to(torch.float32)
    return torch.zeros(length, dtype=torch.float32)


def _optional_vector_weight(weights: dict[str, Any], *, names: tuple[str, ...], length: int) -> torch.Tensor | None:
    for name in names:
        tensor = weights.get(name)
        if tensor is not None and getattr(tensor, "ndim", None) == 1 and int(tensor.shape[0]) == length:
            return tensor.to(torch.float32)
    return None


def _ensure_saebench_path(model_name: str) -> None:
    saebench_root = V5_ROOT / "vendor" / ("SAEBench-qwen35" if model_name == "qwen3_5_2b_base" else "SAEBench")
    if str(saebench_root) not in sys.path:
        sys.path.insert(0, str(saebench_root))


def _required(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required SAE counterpart field: {name}")
    return value
