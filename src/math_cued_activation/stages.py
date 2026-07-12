from __future__ import annotations

from pathlib import Path
import re
import sqlite3

from .compat import run_script
from .capture import capture_from_config
from .config import PipelineConfig, dataset_slug, model_slug
from .generation import generate_from_config


def _flag(condition: bool, value: str) -> list[str]:
    return [value] if condition else []


def generate(config: PipelineConfig, *, force: bool = False) -> None:
    generate_from_config(config, force=force)


def capture(config: PipelineConfig, *, force: bool = False) -> None:
    for layer in config.capture.layers:
        capture_from_config(config, layer=layer)


def checkpoint_path(config: PipelineConfig, layer: int) -> Path:
    if config.model.id == "WeiboAI/VibeThinker-3B":
        model_name = "vibethinker_only"
    else:
        model_name = re.sub(r"[^a-z0-9]+", "_", config.model.id.lower()).strip("_")
    return config.storage.ica / f"{model_name}_layer{layer}_c{config.model.hidden_size}_iter{config.ica.max_iter}.pt"


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
        layer_name = f"layer_{layer}"
        if not force and _registered_layer_is_complete(config, layer_name):
            print(f"skip existing Explorer registration: {config.explorer.run_id}/{layer_name}")
            continue
        run_exists = (config.storage.explorer / config.explorer.run_id).is_dir()
        args = [
            "--checkpoint", str(checkpoint_path(config, layer)),
            "--run-id", config.explorer.run_id,
            "--model-id", config.model.id,
            "--display-name", config.explorer.display_name,
            "--layer", layer_name,
            "--db-path", str(config.storage.database),
            "--output-root", str(config.storage.explorer.parent),
        ]
        if run_exists or index > 0:
            args.append("--add-layer")
        if force and index == 0:
            args.append("--force-db")
            args.append("--force-run")
        run_script("explorer/tools/register_math_cued_ica.py", args)


def _registered_layer_is_complete(config: PipelineConfig, layer: str) -> bool:
    if not config.storage.database.is_file():
        return False
    try:
        with sqlite3.connect(config.storage.database) as conn:
            row = conn.execute(
                "SELECT feature_pt_path, n_features FROM layers WHERE run_id = ? AND layer = ?",
                (config.explorer.run_id, layer),
            ).fetchone()
            if row is None:
                return False
            feature_count = conn.execute(
                "SELECT COUNT(*) FROM features WHERE run_id = ? AND layer = ?",
                (config.explorer.run_id, layer),
            ).fetchone()[0]
    except sqlite3.Error as exc:
        raise RuntimeError(f"cannot inspect Explorer database {config.storage.database}: {exc}") from exc
    feature_path, expected_count = row
    return Path(feature_path).is_file() and int(feature_count) == int(expected_count)


def enrich(config: PipelineConfig, *, force: bool = False) -> None:
    for layer in config.capture.layers:
        layer_name = f"layer_{layer}"
        common = ["--activation-root", str(config.storage.activations), "--run-id", config.explorer.run_id,
                  "--layer", layer_name, "--db-path", str(config.storage.database),
                  "--model-slug", model_slug(config.model.id), "--dataset-slug", dataset_slug(config.dataset.id)]
        if not force and _feature_column_complete(config, layer_name, "kurtosis"):
            print(f"skip existing feature properties: {config.explorer.run_id}/{layer_name}")
        else:
            run_script("explorer/tools/populate_feature_properties.py", [*common, "--max-rows", str(config.enrichment.max_rows),
                       "--seed", str(config.enrichment.seed), "--device", config.enrichment.device,
                       "--dtype", config.enrichment.dtype])
        if not force and _feature_paths_complete(config, layer_name, "mini_histogram_svg_path"):
            print(f"skip existing mini histograms: {config.explorer.run_id}/{layer_name}")
        else:
            render_histograms(config, layer=layer, force=force)
        if not force and _top_sample_evidence_complete(config, layer_name):
            print(f"skip existing top-sample evidence: {config.explorer.run_id}/{layer_name}")
        else:
            run_script("explorer/tools/populate_top_samples.py", [*common, "--max-rows", str(config.enrichment.max_rows),
                       "--seed", str(config.enrichment.seed), "--examples", str(config.enrichment.examples),
                       "--context-window", str(config.enrichment.context_window), "--chunk-size", str(config.enrichment.chunk_size),
                       "--write-workers", str(config.enrichment.write_workers), "--device", config.enrichment.device,
                       "--dtype", config.enrichment.dtype, "--tokenizer", config.model.tokenizer])


def _feature_column_complete(config: PipelineConfig, layer: str, column: str) -> bool:
    allowed = {"kurtosis"}
    if column not in allowed or not config.storage.database.is_file():
        return False
    with sqlite3.connect(config.storage.database) as conn:
        total, populated = conn.execute(
            f"SELECT COUNT(*), COUNT({column}) FROM features WHERE run_id = ? AND layer = ?",
            (config.explorer.run_id, layer),
        ).fetchone()
    return int(total) > 0 and int(total) == int(populated)


def _feature_paths_complete(config: PipelineConfig, layer: str, column: str) -> bool:
    allowed = {"mini_histogram_svg_path", "annotation_evidence_path"}
    if column not in allowed or not config.storage.database.is_file():
        return False
    with sqlite3.connect(config.storage.database) as conn:
        rows = conn.execute(
            f"SELECT {column} FROM features WHERE run_id = ? AND layer = ?",
            (config.explorer.run_id, layer),
        ).fetchall()
    return bool(rows) and all(value and Path(value).is_file() for (value,) in rows)


def _top_sample_evidence_complete(config: PipelineConfig, layer: str) -> bool:
    if not _feature_paths_complete(config, layer, "annotation_evidence_path"):
        return False
    with sqlite3.connect(config.storage.database) as conn:
        total, populated = conn.execute(
            "SELECT COUNT(*), COUNT(annotation_evidence_json) FROM features WHERE run_id = ? AND layer = ?",
            (config.explorer.run_id, layer),
        ).fetchone()
    return int(total) > 0 and int(total) == int(populated)


def render_histograms(config: PipelineConfig, *, layer: int, force: bool) -> None:
    feature_dir = config.storage.explorer / config.explorer.run_id / "feature_interfaces" / "split_origin_relu"
    args = ["--feature-interface-dir", str(feature_dir), "--layers", f"layer_{layer}",
            "--mini-histogram-svgs", *_flag(force, "--force")]
    run_script("explorer/tools/render_histograms.py", args)
