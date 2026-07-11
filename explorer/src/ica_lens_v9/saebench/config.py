from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..paths import V5_ROOT, V9_ROOT


ComparisonTask = Literal["sparse_probe", "tpp"]
Preset = Literal["smoke", "full"]
Method = Literal[
    "ica_lens",
    "sae_baseline",
    "pca",
    "itda",
    "random_in_ica_lens_structure",
    "random_in_sae_structure",
    "matryoshka_128",
    "matryoshka_512",
]

DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "ica_sae_comparison"
DEFAULT_FEATURE_INTERFACE_ROOT = V9_ROOT / "artifacts" / "feature_interfaces"
DEFAULT_METHOD = "split_origin_relu"
DEFAULT_SAEBENCH_ARTIFACTS = Path("/home/liusida/data/ICA-data/saebench")
SAEBENCH_ROOT = V5_ROOT / "vendor" / "SAEBench"
SAEBENCH_PYTHON = SAEBENCH_ROOT / ".venv" / "bin" / "python"
QWEN35_SAEBENCH_ROOT = V5_ROOT / "vendor" / "SAEBench-qwen35"
QWEN35_SAEBENCH_PYTHON = QWEN35_SAEBENCH_ROOT / ".venv" / "bin" / "python"

RUN_NAMES = {
    "gpt2": "gpt2_tok1000000_c768_iter200",
    "gemma2_2b": "gemma2_2b_tok1000000_c2304_iter200",
    "qwen3_5_2b_base": "qwen3_5_2b_base_tok1000000_c2048_iter200",
}
SAEBENCH_MODEL_NAMES = {
    "gpt2": "gpt2-small",
    "gemma2_2b": "gemma-2-2b",
    "qwen3_5_2b_base": "Qwen/Qwen3.5-2B-Base",
}
HOOK_NAME_TEMPLATE = "blocks.{layer}.hook_resid_post"
MODEL_LAYER_SPECS = [
    ("gpt2", "layer_06"),
    ("gpt2", "layer_10"),
    ("gemma2_2b", "layer_12"),
    ("gemma2_2b", "layer_20"),
    ("qwen3_5_2b_base", "layer_12"),
    ("qwen3_5_2b_base", "layer_20"),
]
ALL_METHODS: tuple[Method, ...] = (
    "ica_lens",
    "sae_baseline",
    "pca",
    "itda",
    "random_in_ica_lens_structure",
    "random_in_sae_structure",
    "matryoshka_128",
    "matryoshka_512",
)
CORE_METHODS: tuple[Method, ...] = ("ica_lens", "sae_baseline", "pca", "itda")

SPARSE_DATASETS_FULL = [
    "LabHC/bias_in_bios_class_set1",
    "LabHC/bias_in_bios_class_set2",
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "canrager/amazon_reviews_mcauley_1and5_sentiment",
    "codeparrot/github-code",
    "fancyzhx/ag_news",
    "Helsinki-NLP/europarl",
]
TPP_DATASETS_FULL = ["LabHC/bias_in_bios_class_set1", "canrager/amazon_reviews_mcauley_1and5"]
SMOKE_DATASETS = ["LabHC/bias_in_bios_class_set1", "canrager/amazon_reviews_mcauley_1and5"]


@dataclass(frozen=True)
class RunTarget:
    model: str
    layer: str
    method: Method


@dataclass(frozen=True)
class EvalPreset:
    dataset_names: list[str]
    sparse_k_values: list[int]
    tpp_n_values: list[int]
    sparse_train_size: int
    sparse_test_size: int
    tpp_train_size: int
    tpp_test_size: int
    context_length: int
    save_activations: bool


def preset_settings(task: ComparisonTask, preset: Preset) -> EvalPreset:
    if preset == "smoke":
        return EvalPreset(
            dataset_names=list(SMOKE_DATASETS[:1]),
            sparse_k_values=[5, 10, 20],
            tpp_n_values=[5, 10, 20],
            sparse_train_size=80,
            sparse_test_size=20,
            tpp_train_size=80,
            tpp_test_size=20,
            context_length=128,
            save_activations=False,
        )
    return EvalPreset(
        dataset_names=list(SPARSE_DATASETS_FULL if task == "sparse_probe" else TPP_DATASETS_FULL),
        sparse_k_values=[1, 2, 5, 10, 20, 50, 100],
        tpp_n_values=[2, 5, 10, 20, 50, 100, 500],
        sparse_train_size=4000,
        sparse_test_size=1000,
        tpp_train_size=4000,
        tpp_test_size=1000,
        context_length=128,
        save_activations=True,
    )


def canonical_layer(layer: str | int) -> str:
    if isinstance(layer, str) and layer.startswith("layer_"):
        return f"layer_{int(layer.removeprefix('layer_')):02d}"
    return f"layer_{int(layer):02d}"


def layer_index(layer: str) -> int:
    return int(canonical_layer(layer).removeprefix("layer_"))


def official_targets(
    *,
    models: list[str] | None = None,
    layers: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[RunTarget]:
    selected_models = set(models or sorted(RUN_NAMES))
    selected_layers = {canonical_layer(layer) for layer in layers} if layers else None
    selected_methods = list(methods or CORE_METHODS)
    if "all" in selected_methods:
        selected_methods = list(ALL_METHODS)
    targets: list[RunTarget] = []
    for model, layer in MODEL_LAYER_SPECS:
        if model not in selected_models:
            continue
        if selected_layers is not None and canonical_layer(layer) not in selected_layers:
            continue
        for method in selected_methods:
            if method in {"matryoshka_128", "matryoshka_512"} and (model, layer) != ("gemma2_2b", "layer_12"):
                continue
            targets.append(RunTarget(model=model, layer=layer, method=method))  # type: ignore[arg-type]
    return targets


def feature_interface_dir(model: str, *, feature_interface_root: Path, method: str = DEFAULT_METHOD) -> Path:
    return feature_interface_root / RUN_NAMES[model] / method


def saebench_env(model: str) -> tuple[Path, Path]:
    if model == "qwen3_5_2b_base":
        return QWEN35_SAEBENCH_ROOT, QWEN35_SAEBENCH_PYTHON
    return SAEBENCH_ROOT, SAEBENCH_PYTHON


def slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_")
