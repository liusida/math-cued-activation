from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
from typing import Any

import torch

from .config import PipelineConfig, dataset_slug, model_slug
from .stages import checkpoint_path


def validate_pipeline(config: PipelineConfig) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    check("config", True, str(config.path))
    response_files = sorted(config.storage.responses.glob("*.json")) if config.storage.responses.is_dir() else []
    check("responses", bool(response_files), f"{len(response_files)} JSON bundles under {config.storage.responses}")
    malformed = 0
    for path in response_files:
        try:
            payload = json.loads(path.read_text())
            tokens = payload.get("tokens", {})
            text = payload.get("text", {})
            if not text.get("generated") or not isinstance(tokens.get("generated_token_ids"), list):
                malformed += 1
        except (OSError, ValueError):
            malformed += 1
    check("response_integrity", malformed == 0, f"{malformed} malformed bundles")

    for layer in config.capture.layers:
        activation_dir = config.storage.activations / dataset_slug(config.dataset.id) / model_slug(config.model.id) / f"layer_{layer:02d}"
        activation_files = sorted(activation_dir.glob("*.pt")) if activation_dir.is_dir() else []
        check(f"activations.layer_{layer}", bool(activation_files), f"{len(activation_files)} bundles under {activation_dir}")
        if activation_files:
            try:
                bundle = torch.load(activation_files[0], map_location="cpu", weights_only=False)
                acts = bundle.get("activations")
                valid = isinstance(acts, torch.Tensor) and acts.ndim == 2 and acts.shape[1] == config.model.hidden_size
                check(f"activation_shape.layer_{layer}", valid, str(getattr(acts, "shape", None)))
            except Exception as exc:
                check(f"activation_shape.layer_{layer}", False, str(exc))
        checkpoint = checkpoint_path(config, layer)
        check(f"ica.layer_{layer}", checkpoint.is_file(), str(checkpoint))

    db = config.storage.database
    check("database", db.is_file(), str(db))
    if db.is_file():
        try:
            with sqlite3.connect(db) as conn:
                layer_rows = conn.execute("SELECT layer, feature_pt_path FROM layers WHERE run_id = ?", (config.explorer.run_id,)).fetchall()
                feature_count = conn.execute("SELECT COUNT(*) FROM features WHERE run_id = ?", (config.explorer.run_id,)).fetchone()[0]
            missing = [path for _, path in layer_rows if not Path(path).is_file()]
            check("explorer_layers", len(layer_rows) == len(config.capture.layers), f"{len(layer_rows)} registered layers")
            check("explorer_features", feature_count > 0, f"{feature_count} features")
            check("explorer_artifacts", not missing, f"{len(missing)} missing feature artifacts")
        except sqlite3.Error as exc:
            check("database_schema", False, str(exc))

    return {"config": str(config.path), "ok": all(item["ok"] for item in checks), "checks": checks}
