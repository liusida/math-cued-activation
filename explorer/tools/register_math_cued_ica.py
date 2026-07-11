from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import types
from pathlib import Path, PosixPath, WindowsPath
from typing import Any

import torch

from ica_lens_v9.features.index import _create_schema


V9_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(
    "/home/liusida/research/Math-Cued-Activation/results/ica/qwen_vibethinker_mixed_layer32_c2048_iter100.pt"
)
DEFAULT_OUTPUT_ROOT = V9_ROOT / "diagnostics" / "math-cued-ica"
DEFAULT_RUN_ID = "math_cued_qwen_vibethinker_mixed_layer32_c2048_iter100"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_LAYER = "layer_32"
METHOD = "split_origin_relu"
MODEL_PRESETS = {
    "qwen": {
        "run_id": "math_cued_qwen_layer32_c2048_iter100",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "display_name": "Math-Cued ICA on Qwen2.5-Coder-3B",
    },
    "vibethinker": {
        "run_id": "math_cued_vibethinker_layer32_c2048_iter100",
        "model_id": "WeiboAI/VibeThinker-3B",
        "display_name": "Math-Cued ICA on VibeThinker-3B",
    },
    "vibethinker_only": {
        "run_id": "math_cued_vibethinker_only_layer32_c2048_iter100",
        "model_id": "WeiboAI/VibeThinker-3B",
        "display_name": "VibeThinker-3B",
        "checkpoint": Path("/home/liusida/research/Math-Cued-Activation/results/ica/vibethinker_only_layer32_c2048_iter100.pt"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrap a Math-Cued ICA checkpoint as a v9 Explorer diagnostic feature index."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS),
        help="Register a known target runtime. Explicit --run-id/--model-id override the preset.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--display-name", default="Math-Cued Qwen/VibeThinker Mixed ICA")
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--force-run", action="store_true", help="Replace this run's artifact folder and DB rows.")
    parser.add_argument("--force-db", action="store_true", help="Replace the shared diagnostic SQLite before registering.")
    parser.add_argument("--force", action="store_true", help="Alias for --force-run --force-db for a clean rebuild.")
    parser.add_argument(
        "--add-layer",
        action="store_true",
        help="Add one new layer to an existing run without replacing its other layers or feature labels.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    output_root = args.output_root.resolve()
    run_id = args.run_id
    model_id = args.model_id
    display_name = args.display_name
    if args.preset:
        preset = MODEL_PRESETS[args.preset]
        if checkpoint == DEFAULT_CHECKPOINT and "checkpoint" in preset:
            checkpoint = preset["checkpoint"]
        if run_id == DEFAULT_RUN_ID:
            run_id = preset["run_id"]
        if model_id == DEFAULT_MODEL_ID:
            model_id = preset["model_id"]
        if display_name == "Math-Cued Qwen/VibeThinker Mixed ICA":
            display_name = preset["display_name"]
    checkpoint = checkpoint.expanduser().resolve()

    db_path = (args.db_path or (output_root / "feature_index.sqlite")).resolve()
    run_root = output_root / "runs" / run_id
    ica_root = run_root / "ica"
    feature_dir = run_root / "feature_interfaces" / METHOD
    ica_path = ica_root / f"{args.layer}_fastica.pt"
    feature_path = feature_dir / f"{args.layer}_features.pt"
    ranking_csv = feature_dir / f"{args.layer}_ranking.csv"
    manifest_path = feature_dir / "manifest.json"

    force_run = bool(args.force or args.force_run)
    force_db = bool(args.force or args.force_db)
    if args.add_layer and (force_run or force_db):
        raise ValueError("--add-layer cannot be combined with --force, --force-run, or --force-db")
    if db_path.exists() and force_db:
        db_path.unlink()
    if run_root.exists() and force_run:
        shutil.rmtree(run_root)
    if args.add_layer and not run_root.is_dir():
        raise FileNotFoundError(f"Existing run directory required for --add-layer: {run_root}")
    if not args.add_layer and run_root.exists():
        raise FileExistsError(f"Output already exists: {run_root}; pass --force-run to replace it.")
    if args.add_layer:
        if not db_path.is_file():
            raise FileNotFoundError(f"Existing feature database required for --add-layer: {db_path}")
        for path in (ica_path, feature_path, ranking_csv):
            if path.exists():
                raise FileExistsError(f"Target layer artifact already exists: {path}")

    obj = _load_math_cued_checkpoint(checkpoint)
    components = _tensor(obj, "components").detach().cpu().to(torch.float32)
    mean = _tensor(obj, "mean").detach().cpu().to(torch.float32).reshape(1, -1)
    whitening = _tensor(obj, "whitening").detach().cpu().to(torch.float32)
    unmixing = _tensor(obj, "unmixing").detach().cpu().to(torch.float32)
    mixing = _tensor(obj, "mixing").detach().cpu().to(torch.float32)
    hidden_size = int(components.shape[1])
    n_components = int(components.shape[0])
    n_features = n_components * 2
    layer_index = _layer_index(args.layer)

    existing_manifest: dict[str, Any] | None = None
    if args.add_layer:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Existing run manifest required for --add-layer: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.layer in existing_manifest.get("layer_summaries", {}):
            raise FileExistsError(f"Layer already exists in run manifest: {run_id}/{args.layer}")
        if int(existing_manifest.get("model", {}).get("hidden_size", -1)) != hidden_size:
            raise ValueError("New layer hidden size does not match the existing run")
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT model_id, hidden_size, n_components FROM model_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            duplicate = conn.execute(
                "SELECT 1 FROM layers WHERE run_id = ? AND layer = ?", (run_id, args.layer)
            ).fetchone()
        if row is None:
            raise KeyError(f"Existing database run required for --add-layer: {run_id}")
        if str(row[0]) != model_id or int(row[1]) != hidden_size or int(row[2]) != n_components:
            raise ValueError("New layer model or dimensions do not match the existing database run")
        if duplicate is not None:
            raise FileExistsError(f"Layer already exists in database: {run_id}/{args.layer}")

    ica_root.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    ica_metadata = {
        "layer": args.layer,
        "artifact_format": "torch_save_tensors_and_metadata",
        "source_checkpoint": str(checkpoint),
        "source_schema": obj.get("schema"),
        "source_config": _jsonable(obj.get("config")),
        "source_data": _jsonable(obj.get("data")),
        "component_id_convention": "Component id is the row index in the Math-Cued FastICA components tensor.",
        "preprocess": "row_normalize_center_whiten",
        "rows": int(_maybe_len(obj.get("labels")) or 0),
        "hidden_size": hidden_size,
        "n_components": n_components,
        "iterations": int(obj.get("n_iter", 0)),
        "norm_eps": 1e-12,
        "diagnostic_note": "Converted for v9 Explorer viewing; per-feature distribution statistics are placeholders.",
    }
    torch.save(
        {
            "tensors": {
                "mean": mean,
                "whitening": whitening,
                "components": components,
                "unmixing": unmixing,
                "directions": unmixing,
                "mixing": mixing,
            },
            "metadata": ica_metadata,
        },
        ica_path,
    )

    feature_ids = torch.arange(n_features, dtype=torch.long)
    source_component_index = torch.arange(n_components, dtype=torch.long).repeat_interleave(2)
    source_sign = torch.tensor([1, -1], dtype=torch.int8).repeat(n_components)
    feature_directions = torch.empty((n_features, hidden_size), dtype=torch.float32)
    feature_directions[0::2] = components
    feature_directions[1::2] = -components
    zeros = torch.zeros(n_features, dtype=torch.float32)
    dead = torch.zeros(n_features, dtype=torch.bool)
    histogram_counts = torch.zeros((n_features, 19), dtype=torch.long)
    histogram_edges = torch.linspace(0, 1, 20, dtype=torch.float32)
    torch.save(
        {
            "tensors": {
                "feature_id": feature_ids,
                "feature_directions": feature_directions,
                "preprocess_mean": mean,
                "decoder": mixing,
                "source_feature_id": feature_ids,
                "source_component_index": source_component_index,
                "source_sign": source_sign,
                "kurtosis": zeros,
                "excess_kurtosis": zeros,
                "dead": dead,
                "activation_frequency": zeros,
                "mean": zeros,
                "variance": zeros,
                "max": zeros,
                "histogram_counts": histogram_counts,
                "histogram_bin_edges_log1p": histogram_edges,
                "histogram_bin_edges": histogram_edges,
            },
            "metadata": {
                "layer": args.layer,
                "method": METHOD,
                "source_ica_artifact": str(ica_path),
                "rows": int(_maybe_len(obj.get("labels")) or 0),
                "hidden_size": hidden_size,
                "n_components": n_components,
                "n_features": n_features,
                "dead_kurtosis_threshold": 0.0,
                "dead_count": 0,
                "alive_count": n_features,
                "feature_id_convention": (
                    "Registration uses temporary component/sign ids; run populate_feature_properties.py "
                    "to sort exposed feature_id by descending active-mirrored raw kurtosis, matching main v9."
                ),
                "source_feature_id_convention": "source_feature_id = 2 * component for positive, 2 * component + 1 for negative.",
                "diagnostic_note": "Distribution statistics are placeholders; Explorer activations are computed live from feature_directions.",
                "ranking_csv": str(ranking_csv),
            },
        },
        feature_path,
    )

    _write_ranking_csv(ranking_csv, n_features)
    layer_summary = {
        "layer": args.layer,
        "layer_index": layer_index,
        "rows": int(_maybe_len(obj.get("labels")) or 0),
        "hidden_size": hidden_size,
        "n_components": n_components,
        "n_features": n_features,
        "alive_count": n_features,
        "dead_count": 0,
        "dead_kurtosis_threshold": 0.0,
        "feature_pt_path": str(feature_path),
        "source_ica_artifact": str(ica_path),
        "ranking_csv": str(ranking_csv),
    }
    fresh_manifest = {
        "artifact": run_id,
        "model": {"id": model_id, "short_name": run_id, "hidden_size": hidden_size},
        "layers": [args.layer],
        "output_dir": str(ica_root),
        "source_checkpoint": str(checkpoint),
        "feature_interface_dir": str(feature_dir),
        "method": METHOD,
        "settings": {"norm_eps": 1e-12, "n_components": n_components},
        "layer_summaries": {args.layer: layer_summary},
    }
    if args.add_layer:
        assert existing_manifest is not None
        manifest = existing_manifest
        manifest.setdefault("layers", []).append(args.layer)
        manifest["layers"] = sorted(set(manifest["layers"]), key=lambda value: (_layer_index(value) is None, _layer_index(value) or -1))
        manifest.setdefault("layer_summaries", {})[args.layer] = layer_summary
        manifest.setdefault("source_checkpoints", {})[args.layer] = str(checkpoint)
    else:
        manifest = fresh_manifest
        manifest["source_checkpoints"] = {args.layer: str(checkpoint)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _write_sqlite(
        db_path=db_path,
        run_id=run_id,
        model_id=model_id,
        display_name=display_name,
        ica_root=ica_root,
        feature_dir=feature_dir,
        manifest=manifest,
        layer=args.layer,
        layer_index=layer_index,
        feature_path=feature_path,
        ica_path=ica_path,
        ranking_csv=ranking_csv,
        n_components=n_components,
        n_features=n_features,
        hidden_size=hidden_size,
        add_layer=bool(args.add_layer),
    )
    print(f"wrote diagnostic Explorer index: {db_path}")
    print("run Explorer with:")
    print(f"  ICA_V9_FEATURE_DB={db_path} uv run uvicorn server.app:app --reload --host 127.0.0.1 --port 8000")


def _load_math_cued_checkpoint(path: Path) -> dict[str, Any]:
    # Older torch/pathlib combinations can pickle pathlib._local.PosixPath.
    if "pathlib._local" not in sys.modules:
        module = types.ModuleType("pathlib._local")
        module.PosixPath = PosixPath
        module.WindowsPath = WindowsPath
        sys.modules["pathlib._local"] = module
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(obj)!r}")
    return obj


def _tensor(obj: dict[str, Any], key: str) -> torch.Tensor:
    value = obj.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Checkpoint missing tensor {key!r}")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return value


def _maybe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def _layer_index(layer: str) -> int | None:
    if layer.startswith("layer_"):
        try:
            return int(layer.removeprefix("layer_"))
        except ValueError:
            return None
    return None


def _write_ranking_csv(path: Path, n_features: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("feature_id,source_component_index,source_sign,kurtosis,dead\n")
        for feature_id in range(n_features):
            component = feature_id // 2
            sign = 1 if feature_id % 2 == 0 else -1
            handle.write(f"{feature_id},{component},{sign},0.0,0\n")


def _write_sqlite(
    *,
    db_path: Path,
    run_id: str,
    model_id: str,
    display_name: str,
    ica_root: Path,
    feature_dir: Path,
    manifest: dict[str, Any],
    layer: str,
    layer_index: int | None,
    feature_path: Path,
    ica_path: Path,
    ranking_csv: Path,
    n_components: int,
    n_features: int,
    hidden_size: int,
    add_layer: bool,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        if not _schema_exists(conn):
            _create_schema(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        if add_layer:
            existing = conn.execute(
                "SELECT model_id, hidden_size, n_components FROM model_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is None:
                raise KeyError(f"Existing database run required for --add-layer: {run_id}")
            if str(existing[0]) != model_id:
                raise ValueError(f"Model id mismatch for existing run: {existing[0]} != {model_id}")
            if int(existing[1]) != hidden_size or int(existing[2]) != n_components:
                raise ValueError("New layer dimensions do not match the existing database run")
            duplicate = conn.execute(
                "SELECT 1 FROM layers WHERE run_id = ? AND layer = ?", (run_id, layer)
            ).fetchone()
            if duplicate is not None:
                raise FileExistsError(f"Layer already exists in database: {run_id}/{layer}")
            conn.execute("UPDATE model_runs SET manifest_json = ? WHERE run_id = ?", (json.dumps(manifest), run_id))
        else:
            conn.execute("DELETE FROM features WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM layers WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM model_runs WHERE run_id = ?", (run_id,))
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
                model_id,
                run_id,
                display_name,
                str(ica_root),
                str(feature_dir),
                METHOD,
                None,
                None,
                n_components,
                hidden_size,
                None,
                1e-12,
                json.dumps(manifest),
            ),
            )
        conn.execute(
            """
            INSERT INTO layers (
                run_id, layer, layer_index, rows, hidden_size, n_components,
                n_features, alive_count, dead_count, dead_kurtosis_threshold,
                feature_pt_path, source_ica_artifact, ranking_csv_path,
                ranking_plot_path, histogram_csv_path, histogram_png_dir,
                mini_histogram_svg_dir, kurtosis_summary_json,
                activation_frequency_summary_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                layer,
                layer_index,
                None,
                hidden_size,
                n_components,
                n_features,
                n_features,
                0,
                0.0,
                str(feature_path),
                str(ica_path),
                str(ranking_csv),
                None,
                None,
                None,
                None,
                json.dumps({"diagnostic": True}),
                json.dumps({"diagnostic": True}),
                json.dumps(manifest["layer_summaries"][layer]),
            ),
        )
        conn.executemany(
            """
            INSERT INTO features (
                run_id, layer, feature_id, source_feature_id,
                source_component_index, source_sign, source_side, dead,
                kurtosis, excess_kurtosis, activation_frequency,
                mean, variance, max_activation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    layer,
                    feature_id,
                    feature_id,
                    feature_id // 2,
                    1 if feature_id % 2 == 0 else -1,
                    "positive" if feature_id % 2 == 0 else "negative",
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                for feature_id in range(n_features)
            ],
        )


def _schema_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'model_runs'").fetchone()
    return row is not None


if __name__ == "__main__":
    main()
