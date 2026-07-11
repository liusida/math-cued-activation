from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib
from typing import Any, Mapping, TypeVar


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    tokenizer: str
    revision: str | None
    hidden_size: int
    layer_count: int
    dtype: str = "auto"


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    split: str
    revision: str | None
    id_field: str
    prompt_field: str
    answer_field: str
    sample_size: int
    start_index: int = 0


@dataclass(frozen=True)
class PromptConfig:
    answer_only: bool = False


@dataclass(frozen=True)
class VllmConfig:
    api_url: str
    api_key_file: Path
    server_name: str
    temperature: float
    top_p: float
    max_tokens: int | None
    concurrency: int
    request_timeout: float
    retries: int
    retry_sleep: float
    host: str
    port: int
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float
    cuda_visible_devices: str
    disable_flashinfer_sampler: bool
    startup_timeout: float
    shutdown_timeout: float
    pid_file: Path
    log_file: Path
    enforce_eager: bool = False


@dataclass(frozen=True)
class StorageConfig:
    responses: Path
    activations: Path
    ica: Path
    explorer: Path
    database: Path


@dataclass(frozen=True)
class CaptureConfig:
    layers: tuple[int, ...]
    site: str
    activation_dtype: str
    capture_prompt_activations: bool
    sanity_check_next_token: bool


@dataclass(frozen=True)
class IcaConfig:
    max_rows: int
    seed: int
    max_iter: int
    tolerance: float
    nonlinearity: str
    whitening_solver: str
    device: str


@dataclass(frozen=True)
class EnrichmentConfig:
    max_rows: int
    seed: int
    examples: int
    context_window: int
    chunk_size: int
    write_workers: int
    device: str
    dtype: str


@dataclass(frozen=True)
class ExplorerConfig:
    run_id: str
    display_name: str
    host: str
    port: int


@dataclass(frozen=True)
class PipelineConfig:
    version: int
    path: Path
    model: ModelConfig
    dataset: DatasetConfig
    prompt: PromptConfig
    vllm: VllmConfig
    storage: StorageConfig
    capture: CaptureConfig
    ica: IcaConfig
    enrichment: EnrichmentConfig
    explorer: ExplorerConfig


T = TypeVar("T")


def _section(cls: type[T], raw: Mapping[str, Any], name: str) -> T:
    allowed = {f.name for f in fields(cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in [{name}]: {', '.join(sorted(unknown))}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"invalid [{name}] section: {exc}") from exc


def _path(value: str, *, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    expected = {"version", "model", "dataset", "prompt", "vllm", "storage", "capture", "ica", "enrichment", "explorer"}
    unknown = set(raw) - expected
    if unknown:
        raise ConfigError(f"unknown top-level keys: {', '.join(sorted(unknown))}")
    if raw.get("version") != 1:
        raise ConfigError("only configuration version 1 is supported")
    base = config_path.parent.parent
    sections = {name: raw.get(name) for name in expected - {"version"}}
    missing = sorted(name for name, value in sections.items() if not isinstance(value, dict))
    if missing:
        raise ConfigError(f"missing configuration sections: {', '.join(missing)}")
    model = _section(ModelConfig, sections["model"], "model")
    dataset = _section(DatasetConfig, sections["dataset"], "dataset")
    prompt = _section(PromptConfig, sections["prompt"], "prompt")
    vllm_raw = dict(sections["vllm"])
    vllm_raw["api_key_file"] = _path(vllm_raw["api_key_file"], base=base)
    vllm_raw["pid_file"] = _path(vllm_raw["pid_file"], base=base)
    vllm_raw["log_file"] = _path(vllm_raw["log_file"], base=base)
    vllm = _section(VllmConfig, vllm_raw, "vllm")
    storage_raw = {key: _path(value, base=base) for key, value in sections["storage"].items()}
    storage = _section(StorageConfig, storage_raw, "storage")
    capture_raw = dict(sections["capture"])
    capture_raw["layers"] = tuple(capture_raw.get("layers", ()))
    capture = _section(CaptureConfig, capture_raw, "capture")
    ica = _section(IcaConfig, sections["ica"], "ica")
    enrichment = _section(EnrichmentConfig, sections["enrichment"], "enrichment")
    explorer = _section(ExplorerConfig, sections["explorer"], "explorer")
    _validate(model, dataset, vllm, capture, ica, enrichment, explorer)
    return PipelineConfig(1, config_path, model, dataset, prompt, vllm, storage, capture, ica, enrichment, explorer)


def _validate(model: ModelConfig, dataset: DatasetConfig, vllm: VllmConfig, capture: CaptureConfig,
              ica: IcaConfig, enrichment: EnrichmentConfig, explorer: ExplorerConfig) -> None:
    if not capture.layers:
        raise ConfigError("capture.layers must not be empty")
    if capture.site != "residual-post":
        raise ConfigError("only capture.site = 'residual-post' is supported")
    invalid = [layer for layer in capture.layers if layer < 0 or layer >= model.layer_count]
    if invalid:
        raise ConfigError(f"capture layers outside model range: {invalid}")
    if len(set(capture.layers)) != len(capture.layers):
        raise ConfigError("capture.layers contains duplicates")
    if model.hidden_size <= 0 or dataset.sample_size <= 0 or dataset.start_index < 0:
        raise ConfigError("hidden_size/sample_size must be positive and start_index nonnegative")
    if vllm.temperature < 0 or not 0 < vllm.top_p <= 1 or vllm.concurrency <= 0:
        raise ConfigError("invalid vLLM sampling or concurrency settings")
    if not 1 <= vllm.port <= 65535 or vllm.max_model_len <= 0:
        raise ConfigError("invalid vLLM server port or model context length")
    if not 0 < vllm.gpu_memory_utilization <= 1:
        raise ConfigError("vllm.gpu_memory_utilization must be in (0, 1]")
    expected_url = f"http://{vllm.host}:{vllm.port}/v1/chat/completions"
    if vllm.api_url != expected_url:
        raise ConfigError(f"vllm.api_url must match the configured server: {expected_url}")
    if ica.max_rows <= 0 or ica.max_iter <= 0 or enrichment.max_rows <= 0:
        raise ConfigError("ICA and enrichment row/iteration limits must be positive")
    if explorer.port < 1 or explorer.port > 65535:
        raise ConfigError("explorer.port must be in 1..65535")


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def dataset_slug(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")
