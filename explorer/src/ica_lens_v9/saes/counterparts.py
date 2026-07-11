from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Literal

from ..paths import V9_ROOT


SaeSource = Literal["sae_lens_registry", "custom_checkpoint"]


@dataclass(frozen=True)
class SaeCounterpart:
    model_name: str
    sae_model_name: str
    hidden_size: int
    source: SaeSource
    repo_id: str
    hook_name_template: str
    release_pattern: str | None = None
    id_pattern_template: str | None = None
    release_name_template: str | None = None
    checkpoint_template: str | None = None
    checkpoint_format: str = ""
    decoder_key: str | None = None
    activation: str = "relu"
    top_k: int | None = None
    apply_b_dec_to_input: bool = False
    normalize_activations: str = "none"
    layer_checkpoints: dict[int, str] = field(default_factory=dict)
    allow_multiple: bool = False


CONFIG_PATHS = {
    "gpt2": V9_ROOT / "config" / "gpt.toml",
    "gemma2_2b": V9_ROOT / "config" / "gemma.toml",
    "qwen3_5_2b_base": V9_ROOT / "config" / "qwen.toml",
}


def load_sae_counterpart(model_name: str, *, config_path: Path | None = None) -> SaeCounterpart:
    path = config_path or CONFIG_PATHS[model_name]
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    section = dict(cfg.get("sae_counterpart") or {})
    if not section:
        raise KeyError(f"No [sae_counterpart] section found in {path}")
    layer_checkpoints = {
        int(layer): str(checkpoint)
        for layer, checkpoint in dict(section.get("layer_checkpoints") or {}).items()
    }
    top_k = section.get("top_k")
    return SaeCounterpart(
        model_name=model_name,
        sae_model_name=str(section["sae_model_name"]),
        hidden_size=int(section["hidden_size"]),
        source=str(section["source"]),  # type: ignore[arg-type]
        repo_id=str(section["repo_id"]),
        hook_name_template=str(section["hook_name_template"]),
        release_pattern=_optional_str(section.get("release_pattern")),
        id_pattern_template=_optional_str(section.get("id_pattern_template")),
        release_name_template=_optional_str(section.get("release_name_template")),
        checkpoint_template=_optional_str(section.get("checkpoint_template")),
        checkpoint_format=str(section.get("checkpoint_format") or ""),
        decoder_key=_optional_str(section.get("decoder_key")),
        activation=str(section.get("activation") or "relu"),
        top_k=int(top_k) if top_k is not None else None,
        apply_b_dec_to_input=bool(section.get("apply_b_dec_to_input", False)),
        normalize_activations=str(section.get("normalize_activations") or "none"),
        layer_checkpoints=layer_checkpoints,
        allow_multiple=bool(section.get("allow_multiple", False)),
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


SAE_COUNTERPARTS = {model_name: load_sae_counterpart(model_name) for model_name in CONFIG_PATHS}
