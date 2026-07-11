from __future__ import annotations

from pathlib import Path

from .compat import run_script
from .config import PipelineConfig, dataset_slug, model_slug
from .generation import generate_from_config


def _flag(condition: bool, value: str) -> list[str]:
    return [value] if condition else []


def generate(config: PipelineConfig, *, force: bool = False) -> None:
    generate_from_config(config, force=force)


def capture(config: PipelineConfig, *, force: bool = False) -> None:
    for layer in config.capture.layers:
        args = [
            "--model", config.model.id,
            "--sample-size", str(config.dataset.sample_size),
            "--start-index", str(config.dataset.start_index),
            "--capture-layer", str(layer),
            "--activation-dtype", config.capture.activation_dtype,
            "--generated-text-dir", str(config.storage.responses),
            "--activation-dir", str(config.storage.activations),
            *_flag(config.capture.capture_prompt_activations, "--capture-prompt-activations"),
            *_flag(config.capture.sanity_check_next_token, "--sanity-check-next-token"),
        ]
        run_script("src/math_cued_activation/_compat_scripts/capture_imo_activations.py", args)


def checkpoint_path(config: PipelineConfig, layer: int) -> Path:
    return config.storage.ica / f"vibethinker_only_layer{layer}_c{config.model.hidden_size}_iter{config.ica.max_iter}.pt"


def fit(config: PipelineConfig, *, force: bool = False) -> None:
    for layer in config.capture.layers:
        output = checkpoint_path(config, layer)
        if output.exists() and not force:
            print(f"skip existing ICA checkpoint: {output}")
            continue
        args = [
            "--activation-root", str(config.storage.activations),
            "--dataset-slug", dataset_slug(config.dataset.id),
            "--vibethinker-slug", model_slug(config.model.id),
            "--source", "vibethinker",
            "--layer", str(layer),
            "--max-vibethinker-activations", str(config.ica.max_rows),
            "--seed", str(config.ica.seed),
            "--max-iter", str(config.ica.max_iter),
            "--tol", str(config.ica.tolerance),
            "--fun", config.ica.nonlinearity,
            "--whiten-solver", config.ica.whitening_solver,
            "--device", config.ica.device,
            "--output", str(output),
        ]
        run_script("src/math_cued_activation/_compat_scripts/fit_ica_qwen_vibethinker_mixed.py", args)


def register(config: PipelineConfig, *, force: bool = False) -> None:
    for index, layer in enumerate(config.capture.layers):
        run_exists = (config.storage.explorer / config.explorer.run_id).is_dir()
        args = [
            "--checkpoint", str(checkpoint_path(config, layer)),
            "--run-id", config.explorer.run_id,
            "--model-id", config.model.id,
            "--display-name", config.explorer.display_name,
            "--layer", f"layer_{layer}",
            "--db-path", str(config.storage.database),
            "--output-root", str(config.storage.explorer.parent),
        ]
        if run_exists or index > 0:
            args.append("--add-layer")
        if force and index == 0:
            args.append("--force-db")
            args.append("--force-run")
        run_script("explorer/tools/register_math_cued_ica.py", args)


def enrich(config: PipelineConfig, *, force: bool = False) -> None:
    for layer in config.capture.layers:
        common = ["--activation-root", str(config.storage.activations), "--run-id", config.explorer.run_id,
                  "--layer", f"layer_{layer}", "--db-path", str(config.storage.database),
                  "--model-slug", model_slug(config.model.id), "--dataset-slug", dataset_slug(config.dataset.id)]
        run_script("explorer/tools/populate_feature_properties.py", [*common, "--max-rows", str(config.enrichment.max_rows),
                   "--seed", str(config.enrichment.seed), "--device", config.enrichment.device,
                   "--dtype", config.enrichment.dtype])
        render_histograms(config, layer=layer, force=force)
        run_script("explorer/tools/populate_top_samples.py", [*common, "--max-rows", str(config.enrichment.max_rows),
                   "--seed", str(config.enrichment.seed), "--examples", str(config.enrichment.examples),
                   "--context-window", str(config.enrichment.context_window), "--chunk-size", str(config.enrichment.chunk_size),
                   "--write-workers", str(config.enrichment.write_workers), "--device", config.enrichment.device,
                   "--dtype", config.enrichment.dtype, "--tokenizer", config.model.tokenizer])


def render_histograms(config: PipelineConfig, *, layer: int, force: bool) -> None:
    feature_dir = config.storage.explorer / config.explorer.run_id / "feature_interfaces" / "split_origin_relu"
    args = ["--feature-interface-dir", str(feature_dir), "--layers", f"layer_{layer}",
            "--mini-histogram-svgs", *_flag(force, "--force")]
    run_script("explorer/tools/render_histograms.py", args)
