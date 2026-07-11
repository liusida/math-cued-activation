from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..io_utils import load_json
from ..layers import layer_index


DEFAULT_METHOD = "split_origin_relu"


def build_feature_index(
    *,
    output: Path,
    ica_root: Path,
    feature_interface_root: Path,
    method: str,
    run_dirs: list[Path],
    force: bool,
) -> None:
    started_at = time.time()
    output = output.resolve()
    if output.exists() and not force:
        raise FileExistsError(f"SQLite index already exists: {output}; pass --force.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    if run_dirs:
        feature_dirs = _feature_dirs_for_runs(run_dirs, feature_interface_root.resolve(), method)
        if len(feature_dirs) != len(run_dirs):
            found = {feature_dir.parent.name for feature_dir in feature_dirs}
            missing = [run_dir.name for run_dir in run_dirs if run_dir.name not in found]
            raise FileNotFoundError(
                f"Missing feature interface directories under {feature_interface_root} for: {', '.join(missing)}"
            )
    else:
        feature_dirs = _discover_feature_dirs(feature_interface_root.resolve(), method)
    if not feature_dirs:
        raise FileNotFoundError(f"No feature interface directories found under {feature_interface_root}")

    with sqlite3.connect(output) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _create_schema(conn)
        indexed_run_dirs: list[Path] = []
        for feature_dir in feature_dirs:
            indexed_run_dirs.append(_index_run(conn, feature_dir=feature_dir, ica_root=ica_root.resolve(), method=method))
        _write_build_info(conn, run_dirs=indexed_run_dirs, elapsed_seconds=time.time() - started_at)
        conn.commit()
        conn.execute("PRAGMA optimize")


def _feature_dirs_for_runs(run_dirs: list[Path], feature_interface_root: Path, method: str) -> list[Path]:
    result: list[Path] = []
    for run_dir in run_dirs:
        feature_dir = feature_interface_root / run_dir.resolve().name / method
        if any(feature_dir.glob("layer_*_features.pt")):
            result.append(feature_dir)
    return result


def _discover_feature_dirs(feature_interface_root: Path, method: str) -> list[Path]:
    result: list[Path] = []
    for feature_dir in sorted(feature_interface_root.glob(f"*/{method}")):
        if any(feature_dir.glob("layer_*_features.pt")):
            result.append(feature_dir)
    return result


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE build_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE model_runs (
            run_id TEXT PRIMARY KEY,
            model_id TEXT,
            model_short_name TEXT,
            display_name TEXT,
            ica_run_dir TEXT NOT NULL,
            feature_interface_dir TEXT NOT NULL,
            method TEXT NOT NULL,
            activation_manifest TEXT,
            token_budget INTEGER,
            n_components INTEGER,
            hidden_size INTEGER,
            max_iter INTEGER,
            norm_eps REAL,
            manifest_json TEXT
        );

        CREATE TABLE layers (
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            layer_index INTEGER,
            rows INTEGER,
            hidden_size INTEGER,
            n_components INTEGER,
            n_features INTEGER,
            alive_count INTEGER,
            dead_count INTEGER,
            dead_kurtosis_threshold REAL,
            feature_pt_path TEXT NOT NULL,
            source_ica_artifact TEXT,
            ranking_csv_path TEXT,
            ranking_plot_path TEXT,
            histogram_csv_path TEXT,
            histogram_png_dir TEXT,
            mini_histogram_svg_dir TEXT,
            kurtosis_summary_json TEXT,
            activation_frequency_summary_json TEXT,
            metadata_json TEXT,
            PRIMARY KEY (run_id, layer),
            FOREIGN KEY (run_id) REFERENCES model_runs(run_id)
        );

        CREATE TABLE features (
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            feature_id INTEGER NOT NULL,
            source_feature_id INTEGER NOT NULL,
            source_component_index INTEGER NOT NULL,
            source_sign INTEGER NOT NULL,
            source_side TEXT NOT NULL,
            dead INTEGER NOT NULL,
            kurtosis REAL NOT NULL,
            excess_kurtosis REAL NOT NULL,
            activation_frequency REAL NOT NULL,
            mean REAL NOT NULL,
            variance REAL NOT NULL,
            max_activation REAL NOT NULL,
            mini_histogram_svg_path TEXT,
            effective_context_mean REAL,
            effective_receptive_field_json TEXT,
            annotation_evidence_path TEXT,
            annotation_evidence_json TEXT,
            annotation_label TEXT,
            annotation_simple_label TEXT,
            manual_label TEXT,
            annotation_description TEXT,
            annotation_reasoning TEXT,
            annotation_confidence TEXT,
            annotation_provider TEXT,
            annotation_model TEXT,
            annotation_path TEXT,
            annotation_raw_response_path TEXT,
            PRIMARY KEY (run_id, layer, feature_id),
            FOREIGN KEY (run_id, layer) REFERENCES layers(run_id, layer)
        );

        CREATE TABLE feature_annotations (
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            feature_id INTEGER NOT NULL,
            provider_label TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            created_at TEXT,
            label TEXT,
            simple_label TEXT,
            description TEXT,
            reasoning TEXT,
            confidence TEXT,
            test_cases_json TEXT,
            annotation_path TEXT,
            raw_response_path TEXT,
            PRIMARY KEY (run_id, layer, feature_id, provider_label),
            FOREIGN KEY (run_id, layer, feature_id) REFERENCES features(run_id, layer, feature_id)
        );

        CREATE TABLE feature_annotation_refinements (
            run_id TEXT NOT NULL,
            layer TEXT NOT NULL,
            feature_id INTEGER NOT NULL,
            provider_label TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            provider TEXT,
            model TEXT,
            created_at TEXT,
            source_annotation_path TEXT,
            tests_path TEXT,
            request_path TEXT,
            raw_response_path TEXT,
            annotation_path TEXT,
            label_before_json TEXT,
            test_results_json TEXT,
            annotation_label TEXT,
            annotation_simple_label TEXT,
            annotation_description TEXT,
            annotation_reasoning TEXT,
            annotation_confidence TEXT,
            annotation_test_cases_json TEXT,
            PRIMARY KEY (run_id, layer, feature_id, provider_label, round_index),
            FOREIGN KEY (run_id, layer, feature_id) REFERENCES features(run_id, layer, feature_id)
        );

        CREATE INDEX idx_layers_run_layer_index ON layers(run_id, layer_index);
        CREATE INDEX idx_features_lookup ON features(run_id, layer, feature_id);
        CREATE INDEX idx_features_kurtosis ON features(run_id, layer, kurtosis DESC);
        CREATE INDEX idx_features_dead ON features(run_id, layer, dead, kurtosis DESC);
        CREATE INDEX idx_feature_annotations_lookup ON feature_annotations(run_id, layer, feature_id);
        CREATE INDEX idx_feature_annotation_refinements_lookup
            ON feature_annotation_refinements(run_id, layer, feature_id, provider_label, round_index);
        """
    )


def _index_run(conn: sqlite3.Connection, *, feature_dir: Path, ica_root: Path, method: str) -> Path:
    feature_manifest_path = feature_dir / "manifest.json"
    feature_manifest = load_json(feature_manifest_path) if feature_manifest_path.is_file() else {}
    source_ica_run_dir = feature_manifest.get("source_ica_run_dir")
    run_dir = Path(str(source_ica_run_dir)).resolve() if source_ica_run_dir else (ica_root / feature_dir.parent.name).resolve()
    run_manifest_path = run_dir / "manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {run_manifest_path}")
    run_manifest = load_json(run_manifest_path)
    feature_paths = sorted(feature_dir.glob("layer_*_features.pt"))
    if not feature_paths:
        raise FileNotFoundError(f"No layer feature artifacts found: {feature_dir}")

    model = run_manifest.get("model", {})
    settings = run_manifest.get("settings", {})
    model_short_name = str(model.get("short_name") or _short_name_from_run_dir(run_dir))
    run_id = _run_id(run_dir)
    conn.execute(
        """
        INSERT INTO model_runs (
            run_id, model_id, model_short_name, display_name, ica_run_dir,
            feature_interface_dir, method, activation_manifest, token_budget,
            n_components, hidden_size, max_iter, norm_eps, manifest_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _maybe_str(model.get("id")),
            model_short_name,
            _display_name(model_short_name),
            str(run_dir),
            str(feature_dir),
            method,
            _maybe_str(run_manifest.get("activation_manifest")),
            _maybe_int(settings.get("token_budget")),
            _maybe_int(settings.get("n_components")),
            _maybe_int(model.get("hidden_size")),
            _maybe_int(settings.get("max_iter")),
            _maybe_float(settings.get("norm_eps")),
            json.dumps(run_manifest, sort_keys=True),
        ),
    )

    for feature_path in tqdm(feature_paths, desc=f"index {run_id}", unit="layer", dynamic_ncols=True):
        _index_layer(conn, run_id=run_id, feature_path=feature_path, feature_dir=feature_dir)
    return run_dir


def _index_layer(conn: sqlite3.Connection, *, run_id: str, feature_path: Path, feature_dir: Path) -> None:
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    metadata = artifact["metadata"]
    layer = str(metadata["layer"])
    layer_index_value = layer_index(layer)
    mini_dir = feature_dir / f"{layer}_mini_histograms"
    histogram_png_dir = feature_dir / f"{layer}_histograms"
    conn.execute(
        """
        INSERT INTO layers (
            run_id, layer, layer_index, rows, hidden_size, n_components, n_features,
            alive_count, dead_count, dead_kurtosis_threshold, feature_pt_path,
            source_ica_artifact, ranking_csv_path, ranking_plot_path, histogram_csv_path,
            histogram_png_dir, mini_histogram_svg_dir, kurtosis_summary_json,
            activation_frequency_summary_json, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            layer,
            layer_index_value,
            _maybe_int(metadata.get("rows")),
            _maybe_int(metadata.get("hidden_size")),
            _maybe_int(metadata.get("n_components")),
            _maybe_int(metadata.get("n_features")),
            _maybe_int(metadata.get("alive_count")),
            _maybe_int(metadata.get("dead_count")),
            _maybe_float(metadata.get("dead_kurtosis_threshold")),
            str(feature_path),
            _maybe_str(metadata.get("source_ica_artifact")),
            str(feature_dir / f"{layer}_ranking.csv"),
            str(feature_dir / f"{layer}_ranking.png"),
            str(feature_dir / f"{layer}_histograms.csv"),
            str(histogram_png_dir) if histogram_png_dir.is_dir() else None,
            str(mini_dir) if mini_dir.is_dir() else None,
            json.dumps(metadata.get("kurtosis_summary", {}), sort_keys=True),
            json.dumps(metadata.get("activation_frequency_summary", {}), sort_keys=True),
            json.dumps(metadata, sort_keys=True),
        ),
    )

    rows = []
    n_features = int(tensors["feature_id"].numel())
    for i in range(n_features):
        feature_id = int(tensors["feature_id"][i].item())
        source_sign = int(tensors["source_sign"][i].item())
        rows.append(
            (
                run_id,
                layer,
                feature_id,
                int(tensors["source_feature_id"][i].item()),
                int(tensors["source_component_index"][i].item()),
                source_sign,
                "positive" if source_sign > 0 else "negative",
                int(bool(tensors["dead"][i].item())),
                float(tensors["kurtosis"][i].item()),
                float(tensors["excess_kurtosis"][i].item()),
                float(tensors["activation_frequency"][i].item()),
                float(tensors["mean"][i].item()),
                float(tensors["variance"][i].item()),
                float(tensors["max"][i].item()),
                str(mini_dir / f"feature_{feature_id:06d}.svg") if mini_dir.is_dir() else None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        )
    conn.executemany(
        """
        INSERT INTO features (
            run_id, layer, feature_id, source_feature_id, source_component_index,
            source_sign, source_side, dead, kurtosis, excess_kurtosis,
            activation_frequency, mean, variance, max_activation, mini_histogram_svg_path,
            effective_context_mean, effective_receptive_field_json, annotation_evidence_path,
            annotation_label, annotation_simple_label, annotation_description,
            annotation_reasoning, annotation_confidence, annotation_provider,
            annotation_model, annotation_path, annotation_raw_response_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _write_build_info(conn: sqlite3.Connection, *, run_dirs: list[Path], elapsed_seconds: float) -> None:
    values = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "schema_version": "2",
        "description": "Minimal v9 feature metadata index. Feature kurtosis is active-mirrored; tensor directions remain in layer feature .pt artifacts. Estimated ERF and annotation-label fields are imported from generated artifacts.",
        "run_dirs": json.dumps([str(path) for path in run_dirs]),
        "elapsed_seconds": f"{elapsed_seconds:.3f}",
    }
    conn.executemany("INSERT INTO build_info (key, value) VALUES (?, ?)", values.items())


def _run_id(run_dir: Path) -> str:
    return run_dir.name


def _short_name_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    for suffix in ("_tok1000000", "_c"):
        if suffix in name:
            return name.split(suffix, 1)[0]
    return name


def _display_name(short_name: str) -> str:
    names = {
        "gpt2-small": "GPT-2",
        "gpt2": "GPT-2",
        "gemma2_2b": "Gemma 2 2B",
        "qwen3_5_2b_base": "Qwen3.5 2B Base",
    }
    return names.get(short_name, short_name)


def _maybe_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)
