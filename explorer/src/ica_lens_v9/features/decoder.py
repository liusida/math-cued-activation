from __future__ import annotations

"""Reusable decoder utilities for ICA Lens feature artifacts."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


RECONSTRUCTION_TENSOR_KEYS = {
    "feature_directions",
    "preprocess_mean",
    "decoder",
    "source_component_index",
    "source_sign",
}


@dataclass(frozen=True)
class IcaLensFeatureDecoder:
    feature_directions: torch.Tensor
    preprocess_mean: torch.Tensor
    decoder: torch.Tensor
    source_component_index: torch.Tensor
    source_sign: torch.Tensor
    norm_eps: float = 1e-12
    _last_input_norm: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_nonzero_mask: torch.Tensor | None = field(default=None, init=False, repr=False)
    _last_leading_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    @property
    def n_features(self) -> int:
        return int(self.feature_directions.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.decoder.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.decoder.shape[1])

    def encode(self, activations: torch.Tensor, *, cache_norm: bool = True) -> torch.Tensor:
        normalized = self.normalize(activations, cache_norm=cache_norm)
        return self.feature_values(normalized)

    def decode(self, feature_values: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        normalized_reconstruction = self.reconstruct_normalized_from_feature_values(feature_values)
        if not restore_norm:
            return normalized_reconstruction
        return self.restore_cached_norm(normalized_reconstruction, feature_values)

    def forward(self, activations: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.decode(self.encode(activations), restore_norm=restore_norm)

    def normalize(self, activations: torch.Tensor, *, cache_norm: bool = True) -> torch.Tensor:
        activations = activations.to(device=self.feature_directions.device, dtype=self.feature_directions.dtype)
        norm = torch.linalg.vector_norm(activations, dim=-1, keepdim=True)
        norm_clamped = norm.clamp_min(float(self.norm_eps))
        nonzero_mask = norm > float(self.norm_eps)
        normalized = torch.where(nonzero_mask, activations / norm_clamped, torch.zeros_like(activations))
        if cache_norm:
            object.__setattr__(self, "_last_input_norm", norm_clamped.detach())
            object.__setattr__(self, "_last_nonzero_mask", nonzero_mask.detach())
            object.__setattr__(self, "_last_leading_shape", tuple(activations.shape[:-1]))
        return normalized

    def feature_values(self, normalized_activations: torch.Tensor) -> torch.Tensor:
        normalized_activations = normalized_activations.to(
            device=self.feature_directions.device,
            dtype=self.feature_directions.dtype,
        )
        scores = (normalized_activations - self.preprocess_mean) @ self.feature_directions.T
        return torch.relu(scores)

    def component_scores(self, feature_values: torch.Tensor) -> torch.Tensor:
        leading_shape = tuple(feature_values.shape[:-1])
        flat_values = feature_values.reshape(-1, int(feature_values.shape[-1]))
        signed_scores = torch.zeros(
            (int(flat_values.shape[0]), self.n_components),
            device=feature_values.device,
            dtype=feature_values.dtype,
        )
        comp_idx = self.source_component_index.unsqueeze(0).expand(int(flat_values.shape[0]), -1)
        signed_scores.scatter_add_(1, comp_idx, flat_values * self.source_sign.unsqueeze(0))
        return signed_scores.reshape(*leading_shape, self.n_components)

    def reconstruct_normalized_from_feature_values(self, feature_values: torch.Tensor) -> torch.Tensor:
        return self.preprocess_mean + self.component_scores(feature_values) @ self.decoder

    def reconstruct_normalized_from_component_scores(self, component_scores: torch.Tensor) -> torch.Tensor:
        return self.preprocess_mean + component_scores @ self.decoder

    def reconstruct_from_feature_values(self, feature_values: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_normalized_from_feature_values(feature_values)

    def reconstruct_from_component_scores(self, component_scores: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_normalized_from_component_scores(component_scores)

    def restore_cached_norm(self, normalized_reconstruction: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        if self._last_input_norm is None or self._last_nonzero_mask is None or self._last_leading_shape is None:
            raise RuntimeError("Norm-restoring decode requires encode(x) to be called immediately before decode(features).")
        if tuple(feature_values.shape[:-1]) != self._last_leading_shape:
            raise RuntimeError(
                "Norm-restoring decode got feature activations with leading shape "
                f"{tuple(feature_values.shape[:-1])}, but cached input shape is {self._last_leading_shape}."
            )
        input_norm = self._last_input_norm.to(
            device=normalized_reconstruction.device,
            dtype=normalized_reconstruction.dtype,
        )
        nonzero_mask = self._last_nonzero_mask.to(device=normalized_reconstruction.device)
        restored = normalized_reconstruction * input_norm
        return torch.where(nonzero_mask, restored, torch.zeros_like(restored))


def build_feature_decoder_from_tensors(
    feature_tensors: dict[str, Any],
    *,
    feature_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    norm_eps: float = 1e-12,
) -> IcaLensFeatureDecoder:
    require_feature_decoder_tensors(feature_tensors, feature_path)
    decoder = IcaLensFeatureDecoder(
        feature_directions=feature_tensors["feature_directions"].to(device=device, dtype=dtype),
        preprocess_mean=feature_tensors["preprocess_mean"].to(device=device, dtype=dtype),
        decoder=feature_tensors["decoder"].to(device=device, dtype=dtype),
        source_component_index=feature_tensors["source_component_index"].to(device=device, dtype=torch.long),
        source_sign=feature_tensors["source_sign"].to(device=device, dtype=dtype),
        norm_eps=float(norm_eps),
    )
    validate_feature_decoder(decoder, feature_path)
    return decoder


def require_feature_decoder_tensors(feature_tensors: dict[str, Any], feature_path: Path) -> None:
    missing = sorted(RECONSTRUCTION_TENSOR_KEYS.difference(feature_tensors))
    if missing:
        raise KeyError(
            f"Feature artifact {feature_path} is missing reconstruction tensors {missing}. "
            "Rebuild the ICA Lens feature interface so reconstruction no longer depends on raw ICA artifacts."
        )


def validate_feature_decoder(decoder: IcaLensFeatureDecoder, feature_path: Path) -> None:
    if int(decoder.source_component_index.numel()) != decoder.n_features:
        raise ValueError(f"Malformed feature artifact {feature_path}: source component mapping length does not match features.")
    if int(decoder.source_sign.numel()) != decoder.n_features:
        raise ValueError(f"Malformed feature artifact {feature_path}: source sign mapping length does not match features.")
    if decoder.n_features == 0:
        raise ValueError(f"Malformed feature artifact {feature_path}: no features found.")
    if int(decoder.source_component_index.max().item()) >= decoder.n_components:
        raise ValueError(f"Malformed feature artifact {feature_path}: source component index exceeds decoder rows.")
