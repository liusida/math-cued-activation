from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..features.decoder import build_feature_decoder_from_tensors
from ..features.decoder import RECONSTRUCTION_TENSOR_KEYS
from ..io_utils import load_json
from ..layers import layer_shard_records
from ..paths import V5_ROOT
from ..saes.counterparts import SAE_COUNTERPARTS
from ..saes.counterparts import SaeCounterpart
from ..saes.loaders import load_counterpart_sae
from ..saes.loaders import _decoder_weight, _encoder_weight, _load_weights, _optional_vector_weight, _resolve_checkpoint_path, _vector_weight
from .config import HOOK_NAME_TEMPLATE, SAEBENCH_MODEL_NAMES, layer_index


MATRYOSHKA_REPO_ID = "chanind/gemma-2-2b-batch-topk-matryoshka-saes-w-32k-l0-40"
MATRYOSHKA_VARIANT = "snap"
MATRYOSHKA_HOOK = "blocks.12.hook_resid_post"
MATRYOSHKA_LAYER = 12


def ensure_saebench_imports(model: str) -> None:
    root = V5_ROOT / "vendor" / ("SAEBench-qwen35" if model == "qwen3_5_2b_base" else "SAEBench")
    for path in (root,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


class SimpleConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def custom_config(**kwargs: Any) -> Any:
    core_keys = {"model_name", "d_in", "d_sae", "hook_layer", "hook_name"}
    try:
        from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig

        cfg = CustomSAEConfig(**{key: kwargs[key] for key in core_keys})
        for key, value in kwargs.items():
            if key not in core_keys:
                setattr(cfg, key, value)
        return cfg
    except Exception:
        return SimpleConfig(**kwargs)


def load_ica_lens_sae(
    *,
    feature_interface_dir: Path,
    model: str,
    layer: str,
    device: str,
    dtype: torch.dtype,
    norm_eps: float = 1e-12,
) -> tuple[str, Any, dict[str, object]]:
    feature_path = feature_interface_dir / f"{layer}_features.pt"
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    missing = sorted(RECONSTRUCTION_TENSOR_KEYS.difference(artifact.get("tensors", {})))
    if missing:
        raise RuntimeError(_feature_rebuild_message(feature_path=feature_path, layer=layer, artifact=artifact, missing=missing))
    decoder = build_feature_decoder_from_tensors(
        artifact["tensors"],
        feature_path=feature_path,
        device=torch.device(device),
        dtype=dtype,
        norm_eps=norm_eps,
    )
    sae = IcaLensSAELike(
        decoder=decoder,
        model_name=SAEBENCH_MODEL_NAMES[model],
        hook_layer=layer_index(layer),
        hook_name=HOOK_NAME_TEMPLATE.format(layer=layer_index(layer)),
        feature_path=feature_path,
        device=device,
        dtype=dtype,
    )
    name = f"v9_ica_lens_{model}_{layer}"
    metadata = {"n_saebench_features": int(sae.cfg.d_sae), "feature_artifact": str(feature_path)}
    return name, sae, metadata


def _feature_rebuild_message(*, feature_path: Path, layer: str, artifact: dict[str, Any], missing: list[str]) -> str:
    metadata = artifact.get("metadata", {})
    source_ica_artifact = metadata.get("source_ica_artifact") if isinstance(metadata, dict) else None
    if isinstance(source_ica_artifact, str) and source_ica_artifact:
        ica_run_dir = Path(source_ica_artifact).resolve().parent
        command = (
            "uv run python scripts/build_feature_interface.py "
            f"--ica-run-dir {ica_run_dir} "
            f"--layers {layer} "
            "--force"
        )
    else:
        command = "uv run python scripts/build_feature_interface.py --ica-run-dir <ICA_RUN_DIR> --layers <LAYER> --force"
    return (
        f"Feature artifact {feature_path} is missing reconstruction tensors {missing}. "
        "It was built before ICA Lens decoder tensors were embedded. Rebuild this layer's feature interface with:\n\n"
        f"  {command}\n\n"
        "Then rerun the SAEBench command."
    )


class IcaLensSAELike(torch.nn.Module):
    def __init__(
        self,
        *,
        decoder: Any,
        model_name: str,
        hook_layer: int,
        hook_name: str,
        feature_path: Path,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.feature_decoder = decoder
        w_dec = decoder.decoder[decoder.source_component_index] * decoder.source_sign[:, None]
        w_dec = w_dec / torch.linalg.vector_norm(w_dec, dim=1, keepdim=True).clamp_min(float(decoder.norm_eps))
        self.W_dec = torch.nn.Parameter(w_dec.contiguous(), requires_grad=False)
        self.W_enc = torch.nn.Parameter(decoder.feature_directions.T.contiguous(), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.zeros(decoder.n_features, dtype=dtype, device=decoder.feature_directions.device), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(decoder.hidden_size, dtype=dtype, device=decoder.feature_directions.device), requires_grad=False)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = custom_config(
            model_name=model_name,
            d_in=decoder.hidden_size,
            d_sae=decoder.n_features,
            hook_layer=int(hook_layer),
            hook_name=hook_name,
            architecture="ica_lens_split_origin_relu",
            activation_fn_str="relu",
            dtype=str(dtype).removeprefix("torch."),
            device=str(self.device),
            feature_artifact=str(feature_path),
            norm_restoring_decode=True,
        )
        self.to(device=self.device, dtype=self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_decoder.encode(x.to(device=self.device, dtype=self.dtype))

    def decode(self, feature_acts: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.feature_decoder.decode(
            feature_acts.to(device=self.device, dtype=self.dtype),
            restore_norm=restore_norm,
        )

    def forward(self, x: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.decode(self.encode(x), restore_norm=restore_norm)


def load_selected_sae(
    *,
    method: str,
    model: str,
    layer: str,
    feature_interface_dir: Path,
    output_root: Path,
    activation_manifest_path: Path,
    device: str,
    dtype: torch.dtype,
    force: bool,
    itda_k: int = 40,
    itda_tau: float = 4e-4,
    itda_max_atoms: int = 4096,
    itda_max_train_tokens: int = 1_000_000,
) -> tuple[list[tuple[str, Any]], str, dict[str, object]]:
    ensure_saebench_imports(model)
    if method == "ica_lens":
        name, sae, metadata = load_ica_lens_sae(
            feature_interface_dir=feature_interface_dir,
            model=model,
            layer=layer,
            device=device,
            dtype=dtype,
        )
        return [(name, sae)], "ica_lens", metadata
    if method == "sae_baseline":
        name, sae = load_counterpart_sae(
            counterpart=SAE_COUNTERPARTS[model],
            layer_index=layer_index(layer),
            device=device,
            dtype=dtype,
        )
        return [(name, sae)], "sae_baseline", {"sae_release": name, "n_saebench_features": int(sae.cfg.d_sae)}
    if method == "random_in_ica_lens_structure":
        sae = load_random_ica_lens_feature_sae(
            feature_interface_dir=feature_interface_dir,
            model=model,
            layer=layer,
            device=device,
            dtype=dtype,
        )
        return [(f"v9_random_ica_lens_structure_{model}_{layer}", sae)], "random_in_ica_lens_structure", {
            "n_saebench_features": int(sae.cfg.d_sae),
            "random_seed": int(sae.cfg.random_seed),
            "random_structure": "ica_lens_features",
        }
    if method == "random_in_sae_structure":
        sae = load_random_counterpart_sae(
            counterpart=SAE_COUNTERPARTS[model],
            layer_index=layer_index(layer),
            model=model,
            layer=layer,
            device=device,
            dtype=dtype,
        )
        return [(f"v9_random_sae_structure_{model}_{layer}", sae)], "random_in_sae_structure", {
            "n_saebench_features": int(sae.cfg.d_sae),
            "random_seed": int(sae.cfg.random_seed),
            "random_structure": "sae_counterpart",
            "activation": str(sae.cfg.activation_fn_str),
            "top_k": getattr(sae.cfg, "top_k", None),
        }
    if method == "pca":
        prefix = pca_prefix(output_root, model, layer)
        fit_pca_artifact(
            activation_manifest_path=activation_manifest_path,
            layer=layer,
            output_prefix=prefix,
            n_components=_n_ica_components(feature_interface_dir, layer),
            device=device,
            force=force,
        )
        sae = load_pca_sae(prefix, model=model, layer=layer, device=device, dtype=dtype)
        return [(f"v9_pca_{model}_{layer}", sae)], "pca_two_sign", {"artifact_prefix": str(prefix), "n_saebench_features": int(sae.cfg.d_sae)}
    if method == "itda":
        prefix = itda_prefix(output_root, model, layer, k=itda_k, tau=itda_tau, max_atoms=itda_max_atoms)
        train_itda_artifact(
            activation_manifest_path=activation_manifest_path,
            layer=layer,
            output_prefix=prefix,
            k=itda_k,
            loss_threshold=itda_tau,
            max_atoms=itda_max_atoms,
            max_train_tokens=itda_max_train_tokens,
            device=device,
            force=force,
        )
        sae = load_itda_sae(prefix, model=model, layer=layer, device=device, dtype=dtype, encode_k=itda_k)
        return [(f"v9_itda_{model}_{layer}", sae)], "itda", {"artifact_prefix": str(prefix), "n_saebench_features": int(sae.cfg.d_sae), "n_itda_atoms": int(sae.cfg.itda_n_atoms)}
    if method.startswith("matryoshka_"):
        width = int(method.rsplit("_", 1)[-1])
        checkpoint_dir = download_matryoshka_checkpoint(output_root / "artifacts" / "matryoshka" / "gemma2_2b" / "layer_12")
        sae = load_matryoshka_prefix_sae(checkpoint_dir=checkpoint_dir, width=width, device=device, dtype=dtype)
        return [(f"v9_matryoshka_{width}_gemma2_2b_layer_12", sae)], f"matryoshka_{width}", {"checkpoint_dir": str(checkpoint_dir), "matryoshka_width": width, "n_saebench_features": int(sae.cfg.d_sae)}
    raise ValueError(f"Unsupported comparison method: {method!r}")


def pca_prefix(output_root: Path, model: str, layer: str) -> Path:
    return output_root / "artifacts" / "pca" / model / f"{layer}_pca"


def itda_prefix(output_root: Path, model: str, layer: str, *, k: int, tau: float, max_atoms: int) -> Path:
    tau_text = f"{tau:g}".replace(".", "p").replace("-", "m")
    return output_root / "artifacts" / "itda" / model / f"{layer}_itda_k{k}_tau{tau_text}_atoms{max_atoms}"


def _n_ica_components(feature_interface_dir: Path, layer: str) -> int:
    metadata = load_json(feature_interface_dir / f"{layer}_features.json")
    return int(metadata["n_components"])


def load_random_ica_lens_feature_sae(
    *,
    feature_interface_dir: Path,
    model: str,
    layer: str,
    device: str,
    dtype: torch.dtype,
    norm_eps: float = 1e-12,
) -> Any:
    feature_path = feature_interface_dir / f"{layer}_features.pt"
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    missing = sorted(RECONSTRUCTION_TENSOR_KEYS.difference(artifact.get("tensors", {})))
    if missing:
        raise RuntimeError(_feature_rebuild_message(feature_path=feature_path, layer=layer, artifact=artifact, missing=missing))
    tensors = artifact["tensors"]
    n_features = int(tensors["feature_directions"].shape[0])
    hidden_size = int(tensors["feature_directions"].shape[1])
    seed = _stable_random_seed("random_in_ica_lens_structure", model, layer, str(n_features), str(hidden_size))
    feature_directions = _random_unit_rows(n_features, hidden_size, seed=seed)
    decoder_rows = _random_unit_rows(n_features, hidden_size, seed=seed + 1)
    preprocess_mean = tensors["preprocess_mean"].to(torch.float32).reshape(1, hidden_size)
    return RandomIcaLensFeatureSAELike(
        feature_directions=feature_directions,
        decoder_rows=decoder_rows,
        preprocess_mean=preprocess_mean,
        model_name=SAEBENCH_MODEL_NAMES[model],
        hook_layer=layer_index(layer),
        hook_name=HOOK_NAME_TEMPLATE.format(layer=layer_index(layer)),
        feature_path=feature_path,
        seed=seed,
        device=device,
        dtype=dtype,
        norm_eps=norm_eps,
    )


class RandomIcaLensFeatureSAELike(torch.nn.Module):
    def __init__(
        self,
        *,
        feature_directions: torch.Tensor,
        decoder_rows: torch.Tensor,
        preprocess_mean: torch.Tensor,
        model_name: str,
        hook_layer: int,
        hook_name: str,
        feature_path: Path,
        seed: int,
        device: str,
        dtype: torch.dtype,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if tuple(feature_directions.shape) != tuple(decoder_rows.shape):
            raise ValueError("Random ICA Lens feature directions and decoder rows must have the same shape.")
        n_features, hidden_size = feature_directions.shape
        self.register_buffer("feature_directions", feature_directions.contiguous())
        self.register_buffer("preprocess_mean", preprocess_mean.reshape(1, hidden_size).contiguous())
        self.W_dec = torch.nn.Parameter(decoder_rows.contiguous(), requires_grad=False)
        self.W_enc = torch.nn.Parameter(feature_directions.T.contiguous(), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.zeros(int(n_features)), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(int(hidden_size)), requires_grad=False)
        self.device = torch.device(device)
        self.dtype = dtype
        self.norm_eps = float(norm_eps)
        self._last_input_norm = None
        self._last_nonzero_mask = None
        self.cfg = custom_config(
            model_name=model_name,
            d_in=int(hidden_size),
            d_sae=int(n_features),
            hook_layer=int(hook_layer),
            hook_name=hook_name,
            architecture="random_ica_lens_feature_structure",
            activation_fn_str="relu",
            random_seed=int(seed),
            random_structure="ica_lens_features",
            feature_artifact=str(feature_path),
            norm_restoring_decode=True,
        )
        self.to(device=self.device, dtype=self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        norm_clamped = norm.clamp_min(self.norm_eps)
        nonzero = norm > self.norm_eps
        normalized = torch.where(nonzero, x / norm_clamped, torch.zeros_like(x))
        self._last_input_norm = norm_clamped.detach()
        self._last_nonzero_mask = nonzero.detach()
        scores = (normalized - self.preprocess_mean.to(device=normalized.device, dtype=normalized.dtype)) @ self.feature_directions.T
        return torch.relu(scores)

    def decode(self, feature_acts: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        normalized = feature_acts.to(device=self.device, dtype=self.dtype) @ self.W_dec
        if not restore_norm:
            return normalized
        if self._last_input_norm is None or self._last_nonzero_mask is None:
            raise RuntimeError("Norm-restoring decode requires encode(x) to be called before decode(features).")
        restored = normalized * self._last_input_norm.to(device=normalized.device, dtype=normalized.dtype)
        return torch.where(self._last_nonzero_mask.to(device=normalized.device), restored, torch.zeros_like(restored))

    def forward(self, x: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.decode(self.encode(x), restore_norm=restore_norm)


def load_random_counterpart_sae(
    *,
    counterpart: SaeCounterpart,
    layer_index: int,
    model: str,
    layer: str,
    device: str,
    dtype: torch.dtype,
) -> Any:
    weights_path = _resolve_checkpoint_path(counterpart, layer_index)
    weights = _load_weights(weights_path, checkpoint_format=counterpart.checkpoint_format)
    real_w_dec = _decoder_weight(weights, hidden_size=counterpart.hidden_size, preferred_key=counterpart.decoder_key).to(torch.float32)
    real_w_enc = _encoder_weight(weights, hidden_size=counterpart.hidden_size, d_sae=int(real_w_dec.shape[0])).to(torch.float32)
    b_enc = _vector_weight(weights, names=("b_enc", "encoder.bias"), length=int(real_w_dec.shape[0]))
    b_dec = _vector_weight(weights, names=("b_dec", "decoder.bias"), length=counterpart.hidden_size)
    threshold = _optional_vector_weight(weights, names=("threshold",), length=int(real_w_dec.shape[0]))
    decoder_norms = real_w_dec.norm(dim=1).clamp_min(1e-12)
    seed = _stable_random_seed("random_in_sae_structure", model, layer, str(real_w_dec.shape[0]), str(real_w_dec.shape[1]))
    random_w_dec = _random_unit_rows(int(real_w_dec.shape[0]), int(real_w_dec.shape[1]), seed=seed)
    random_w_enc = _random_like_columns(real_w_enc, seed=seed + 1)
    return RandomCounterpartSAELike(
        w_enc=random_w_enc,
        w_dec=random_w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
        decoder_norms=decoder_norms,
        counterpart=counterpart,
        layer_index=layer_index,
        checkpoint_path=str(weights_path),
        seed=seed,
        device=device,
        dtype=dtype,
    )


class RandomCounterpartSAELike(torch.nn.Module):
    def __init__(
        self,
        *,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor | None,
        decoder_norms: torch.Tensor,
        counterpart: SaeCounterpart,
        layer_index: int,
        checkpoint_path: str,
        seed: int,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.W_enc = torch.nn.Parameter(w_enc.contiguous(), requires_grad=False)
        self.W_dec = torch.nn.Parameter(w_dec.contiguous(), requires_grad=False)
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
        self.cfg = custom_config(
            model_name=counterpart.sae_model_name,
            d_in=counterpart.hidden_size,
            d_sae=int(w_dec.shape[0]),
            hook_layer=layer_index,
            hook_name=counterpart.hook_name_template.format(layer=layer_index),
            architecture="random_sae_counterpart_structure",
            activation_fn_str=counterpart.activation,
            checkpoint_path=checkpoint_path,
            checkpoint_format=counterpart.checkpoint_format,
            top_k=self.top_k,
            apply_b_dec_to_input=self.apply_b_dec_to_input,
            normalize_activations=self.normalize_activations,
            random_seed=int(seed),
            random_structure="sae_counterpart",
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


def _stable_random_seed(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _random_unit_rows(rows: int, cols: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(int(rows), int(cols), generator=generator, dtype=torch.float32)
    return _normalize_rows(matrix)


def _random_like_columns(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random = torch.randn(tuple(reference.shape), generator=generator, dtype=torch.float32)
    norms = torch.linalg.vector_norm(reference.to(torch.float32), dim=0, keepdim=True)
    random = random / torch.linalg.vector_norm(random, dim=0, keepdim=True).clamp_min(1e-12)
    return random * norms


def fit_pca_artifact(
    *,
    activation_manifest_path: Path,
    layer: str,
    output_prefix: Path,
    n_components: int,
    device: str,
    norm_eps: float = 1e-12,
    force: bool = False,
) -> None:
    output_path = output_prefix.with_suffix(".pt")
    metadata_path = output_prefix.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not force:
        return
    started = time.time()
    activation_manifest = load_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent
    dev = torch.device(device)
    total = None
    rows = 0
    for batch in _iter_normalized_layer_batches(activation_dir, activation_manifest, layer, dev, norm_eps):
        total = batch.sum(dim=0) if total is None else total + batch.sum(dim=0)
        rows += int(batch.shape[0])
    if total is None or rows < 2:
        raise RuntimeError(f"Need at least two rows to fit PCA for {layer}.")
    mean = total / rows
    cov = torch.zeros((int(mean.numel()), int(mean.numel())), dtype=torch.float64, device=dev)
    mean64 = mean.to(dtype=torch.float64)
    for batch in _iter_normalized_layer_batches(activation_dir, activation_manifest, layer, dev, norm_eps):
        centered = batch.to(dtype=torch.float64) - mean64
        cov += centered.T @ centered
    cov /= rows - 1
    eigvals, eigvecs = torch.linalg.eigh(cov.cpu())
    order = torch.argsort(eigvals, descending=True)
    components = eigvecs[:, order[:n_components]].T.contiguous().to(torch.float32)
    tensors = {"mean": mean.cpu().to(torch.float32).reshape(1, -1), "components": components}
    metadata = {
        "method": "pca",
        "layer": layer,
        "activation_manifest": str(activation_manifest_path),
        "preprocess": "with_normalization",
        "rows": rows,
        "n_components": int(n_components),
        "norm_eps": float(norm_eps),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": tensors, "metadata": metadata}, output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _iter_normalized_layer_batches(activation_dir: Path, activation_manifest: dict[str, Any], layer: str, device: torch.device, norm_eps: float):
    for shard in layer_shard_records(activation_manifest, layer):
        layer_path = shard["layers"].get(layer)
        if not isinstance(layer_path, str):
            raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}")
        tensor = torch.load(activation_dir / layer_path, map_location="cpu")
        batch = tensor.to(device=device, dtype=torch.float32)
        yield batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)


def load_pca_sae(prefix: Path, *, model: str, layer: str, device: str, dtype: torch.dtype) -> Any:
    artifact = torch.load(prefix.with_suffix(".pt"), map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    components = tensors["components"].to(torch.float32)
    mean = tensors["mean"].to(torch.float32)
    return PCASAELike(components=components, mean=mean, model=model, layer=layer, prefix=prefix, device=device, dtype=dtype)


class PCASAELike(torch.nn.Module):
    def __init__(self, *, components: torch.Tensor, mean: torch.Tensor, model: str, layer: str, prefix: Path, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        n_components, d_in = components.shape
        self.register_buffer("components", components.contiguous())
        self.register_buffer("mean", mean.reshape(1, -1).contiguous())
        self.W_enc = torch.nn.Parameter(components.T.contiguous(), requires_grad=False)
        self.W_dec = torch.nn.Parameter(torch.cat([components, -components], dim=0).contiguous(), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.zeros(2 * n_components), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in), requires_grad=False)
        self.norm_eps = 1e-12
        self.device = torch.device(device)
        self.dtype = dtype
        self._last_input_norm = None
        self._last_nonzero_mask = None
        self.cfg = custom_config(model_name=SAEBENCH_MODEL_NAMES[model], d_in=int(d_in), d_sae=int(2 * n_components), hook_layer=layer_index(layer), hook_name=HOOK_NAME_TEMPLATE.format(layer=layer_index(layer)), architecture="pca_two_sign_sae_like", activation_fn_str="relu", pca_artifact_prefix=str(prefix))
        self.to(device=self.device, dtype=self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.components.device, dtype=self.components.dtype)
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        nonzero = norm > self.norm_eps
        z = torch.where(nonzero, x / norm.clamp_min(self.norm_eps), torch.zeros_like(x))
        self._last_input_norm = norm.clamp_min(self.norm_eps).detach()
        self._last_nonzero_mask = nonzero.detach()
        scores = (z - self.mean) @ self.components.T
        scores = torch.where(nonzero, scores, torch.zeros_like(scores))
        return torch.cat([torch.relu(scores), torch.relu(-scores)], dim=-1)

    def decode(self, feature_acts: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        n_components = int(self.components.shape[0])
        scores = feature_acts[..., :n_components] - feature_acts[..., n_components:]
        z_hat = scores.to(self.components.dtype) @ self.components + self.mean
        if not restore_norm:
            return z_hat
        if self._last_input_norm is None or self._last_nonzero_mask is None:
            return z_hat
        restored = z_hat * self._last_input_norm.to(device=z_hat.device, dtype=z_hat.dtype)
        return torch.where(self._last_nonzero_mask.to(device=z_hat.device), restored, torch.zeros_like(restored))

    def forward(self, x: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.decode(self.encode(x), restore_norm=restore_norm)


def train_itda_artifact(
    *,
    activation_manifest_path: Path,
    layer: str,
    output_prefix: Path,
    k: int,
    loss_threshold: float,
    max_atoms: int,
    max_train_tokens: int,
    device: str,
    force: bool,
) -> None:
    atoms_path = output_prefix.parent / f"{output_prefix.name}_atoms.pt"
    sources_path = output_prefix.parent / f"{output_prefix.name}_atom_sources.pt"
    metadata_path = output_prefix.with_suffix(".json")
    if atoms_path.exists() and sources_path.exists() and metadata_path.exists() and not force:
        return
    started = time.time()
    manifest = load_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent
    parts = []
    rows = 0
    for batch in _iter_normalized_layer_batches(activation_dir, manifest, layer, torch.device("cpu"), 1e-12):
        take = min(int(batch.shape[0]), max_train_tokens - rows)
        if take <= 0:
            break
        parts.append(batch[:take].cpu())
        rows += take
        if rows >= max_train_tokens:
            break
    if not parts:
        raise RuntimeError(f"No rows available to train ITDA for {layer}.")
    x = torch.cat(parts, dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    order = torch.randperm(int(x.shape[0]), generator=generator)
    train_device = torch.device(device)
    if train_device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        atoms, atom_order_indices, train_stats = train_itda(
            activations=x,
            order=order,
            max_atoms=max_atoms,
            k=k,
            loss_threshold=loss_threshold,
            batch_size=1024,
            device=train_device,
        )
    except Exception as exc:
        if train_device.type != "cuda" or not _is_cuda_oom(exc):
            raise
        torch.cuda.empty_cache()
        fallback_device = torch.device("cpu")
        atoms, atom_order_indices, train_stats = train_itda(
            activations=x,
            order=order,
            max_atoms=max_atoms,
            k=k,
            loss_threshold=loss_threshold,
            batch_size=1024,
            device=fallback_device,
        )
        train_stats["fallback_from_device"] = str(train_device)
        train_stats["fallback_reason"] = str(exc).splitlines()[0]
        train_device = fallback_device
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(atoms, atoms_path)
    torch.save(atom_order_indices.cpu(), sources_path)
    metadata = {
        "method": "itda",
        "layer": layer,
        "activation_manifest": str(activation_manifest_path),
        "preprocess": "with_normalization",
        "n_atoms": int(atoms.shape[0]),
        "d_model": int(atoms.shape[1]),
        "k": int(k),
        "loss_threshold": float(loss_threshold),
        "max_atoms": int(max_atoms),
        "rows_sampled": int(rows),
        "artifact_files": {"atoms": str(atoms_path), "atom_sources": str(sources_path)},
        "seed": 0,
        "train_device": str(train_device),
        "train_stats": train_stats,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _is_cuda_oom(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "cuda" in text and ("out of memory" in text or "memoryallocation" in text)


@torch.no_grad()
def train_itda(
    *,
    activations: torch.Tensor,
    order: torch.Tensor,
    max_atoms: int,
    k: int,
    loss_threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    n_train = int(order.numel())
    d_model = int(activations.shape[1])
    init_count = min(max_atoms, d_model, n_train)
    init_indices = order[:init_count]
    atoms = _normalize_rows(activations[init_indices].to(device=device, dtype=torch.float32))
    atom_order_indices = init_indices.clone()
    rows_seen = init_count
    added_after_init = 0
    batches = 0
    last_batch_mean_error: float | None = None
    cursor = init_count
    while cursor < n_train and int(atoms.shape[0]) < max_atoms:
        batch_indices = order[cursor : cursor + batch_size]
        x = activations[batch_indices].to(device=device, dtype=torch.float32)
        recon = _matching_pursuit_reconstruct(atoms, x, k=min(k, int(atoms.shape[0])))
        errors = _normalized_mse(x, recon)
        last_batch_mean_error = float(errors.mean().item())
        remaining = max_atoms - int(atoms.shape[0])
        selected = torch.nonzero(errors > loss_threshold, as_tuple=True)[0]
        if int(selected.numel()) > remaining:
            selected = selected[:remaining]
        if int(selected.numel()) > 0:
            atoms = torch.cat([atoms, _normalize_rows(x[selected])], dim=0)
            atom_order_indices = torch.cat([atom_order_indices, batch_indices[selected.cpu()].cpu()])
            added_after_init += int(selected.numel())
        rows_seen += int(batch_indices.numel())
        batches += 1
        cursor += batch_size
    return atoms.to(device="cpu", dtype=torch.float32), atom_order_indices.cpu(), {
        "rows_seen": int(rows_seen),
        "batches": int(batches),
        "initial_atoms": int(init_count),
        "added_after_init": int(added_after_init),
        "stopped_because_full": bool(int(atoms.shape[0]) >= max_atoms),
        "last_batch_mean_error": last_batch_mean_error,
    }


@torch.no_grad()
def _matching_pursuit_reconstruct(atoms: torch.Tensor, x: torch.Tensor, k: int) -> torch.Tensor:
    residual = x.clone()
    recon = torch.zeros_like(x)
    atoms_t = atoms.T.contiguous()
    rows = torch.arange(int(x.shape[0]), device=x.device)
    for _ in range(k):
        correlations = residual @ atoms_t
        best_atoms = torch.argmax(torch.abs(correlations), dim=1)
        coeffs = correlations[rows, best_atoms]
        update = coeffs[:, None] * atoms[best_atoms]
        recon += update
        residual -= update
    return recon


def _normalized_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    x_norm = torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-9)
    recon_norm = torch.linalg.vector_norm(recon, dim=1, keepdim=True).clamp_min(1e-9)
    return ((x / x_norm) - (recon / recon_norm)).pow(2).mean(dim=1)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-9)


def load_itda_sae(prefix: Path, *, model: str, layer: str, device: str, dtype: torch.dtype, encode_k: int) -> Any:
    atoms = torch.load(prefix.parent / f"{prefix.name}_atoms.pt", map_location="cpu").to(torch.float32)
    return ITDASAELike(atoms=atoms, model=model, layer=layer, prefix=prefix, device=device, dtype=dtype, encode_k=encode_k)


class ITDASAELike(torch.nn.Module):
    def __init__(self, *, atoms: torch.Tensor, model: str, layer: str, prefix: Path, device: str, dtype: torch.dtype, encode_k: int) -> None:
        super().__init__()
        atoms = atoms / torch.linalg.vector_norm(atoms, dim=1, keepdim=True).clamp_min(1e-12)
        n_atoms, d_in = atoms.shape
        self.register_buffer("atoms", atoms.contiguous())
        self.W_dec = torch.nn.Parameter(atoms.contiguous(), requires_grad=False)
        self.W_enc = torch.nn.Parameter(torch.zeros(d_in, n_atoms), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in), requires_grad=False)
        self.encode_k = int(encode_k)
        self.device = torch.device(device)
        self.dtype = dtype
        self.norm_eps = 1e-12
        self._last_input_norm = None
        self._last_nonzero_mask = None
        self.cfg = custom_config(model_name=SAEBENCH_MODEL_NAMES[model], d_in=int(d_in), d_sae=int(n_atoms), hook_layer=layer_index(layer), hook_name=HOOK_NAME_TEMPLATE.format(layer=layer_index(layer)), architecture="itda_matching_pursuit_sae_like", activation_fn_str="identity", itda_n_atoms=int(n_atoms), itda_encode_k=int(encode_k), itda_artifact_prefix=str(prefix))
        self.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = tuple(x.shape)
        flat = x.reshape(-1, shape[-1]).to(device=self.atoms.device, dtype=torch.float32)
        self._last_input_norm = None
        self._last_nonzero_mask = None
        atoms_t = self.atoms.T.contiguous()
        residual = flat.clone()
        coeffs = torch.zeros(flat.shape[0], self.atoms.shape[0], device=flat.device, dtype=flat.dtype)
        rows = torch.arange(flat.shape[0], device=flat.device)
        for _ in range(min(self.encode_k, int(self.atoms.shape[0]))):
            corr = residual @ atoms_t
            best = torch.argmax(torch.abs(corr), dim=1)
            values = corr[rows, best]
            coeffs[rows, best] += values
            residual -= values[:, None] * self.atoms[best]
        return coeffs.reshape(*shape[:-1], -1).to(dtype=self.dtype)

    def decode(self, feature_acts: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        z_hat = feature_acts.to(self.atoms.dtype) @ self.atoms
        if not restore_norm or self._last_input_norm is None or self._last_nonzero_mask is None:
            return z_hat
        restored = z_hat * self._last_input_norm.to(device=z_hat.device, dtype=z_hat.dtype)
        return torch.where(self._last_nonzero_mask.to(device=z_hat.device), restored, torch.zeros_like(restored))

    def forward(self, x: torch.Tensor, *, restore_norm: bool = True) -> torch.Tensor:
        return self.decode(self.encode(x), restore_norm=restore_norm)


def download_matryoshka_checkpoint(output_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    checkpoint_dir = f"{MATRYOSHKA_VARIANT}/{MATRYOSHKA_HOOK}"
    for filename in ("cfg.json", "sae_weights.safetensors", "sparsity.safetensors"):
        hf_hub_download(repo_id=MATRYOSHKA_REPO_ID, filename=f"{checkpoint_dir}/{filename}", local_dir=str(output_root))
    return output_root / checkpoint_dir


def load_matryoshka_prefix_sae(*, checkpoint_dir: Path, width: int, device: str, dtype: torch.dtype) -> Any:
    from safetensors.torch import load_file

    cfg = json.loads((checkpoint_dir / "cfg.json").read_text(encoding="utf-8"))
    state = load_file(str(checkpoint_dir / "sae_weights.safetensors"), device="cpu")

    class MatryoshkaPrefixSAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            d_in = int(cfg["d_in"])
            self.W_enc = torch.nn.Parameter(state["W_enc"][:, :width].contiguous())
            w_dec = state["W_dec"][:width, :].contiguous()
            self.W_dec = torch.nn.Parameter(w_dec / torch.linalg.vector_norm(w_dec, dim=1, keepdim=True).clamp_min(1e-12))
            self.b_enc = torch.nn.Parameter(state["b_enc"][:width].contiguous())
            self.b_dec = torch.nn.Parameter(state["b_dec"].contiguous())
            self.threshold = torch.nn.Parameter(state["threshold"][:width].contiguous(), requires_grad=False)
            self.device = torch.device(device)
            self.dtype = dtype
            self.cfg = custom_config(model_name="gemma-2-2b", d_in=d_in, d_sae=width, hook_layer=MATRYOSHKA_LAYER, hook_name=MATRYOSHKA_HOOK, architecture="matryoshka_jumprelu_prefix", activation_fn_str="relu")
            self.to(device=self.device, dtype=self.dtype)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            pre = x.to(device=self.W_enc.device, dtype=self.W_enc.dtype) @ self.W_enc + self.b_enc
            return torch.relu(pre) * (pre > self.threshold.to(device=pre.device, dtype=pre.dtype))

        def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
            return feature_acts.to(device=self.W_dec.device, dtype=self.W_dec.dtype) @ self.W_dec + self.b_dec

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x))

    return MatryoshkaPrefixSAE()
