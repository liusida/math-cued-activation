from dataclasses import replace
from pathlib import Path
import sqlite3

from math_cued_activation.config import load_config
from math_cued_activation.stages import _registered_layer_is_complete


ROOT = Path(__file__).resolve().parents[1]


def test_registered_layer_requires_db_rows_and_artifact(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_gsm8k.toml")
    artifact = tmp_path / "layer_19_features.pt"
    artifact.write_bytes(b"fixture")
    database = tmp_path / "features.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE layers (run_id TEXT, layer TEXT, feature_pt_path TEXT, n_features INTEGER)")
        conn.execute("CREATE TABLE features (run_id TEXT, layer TEXT, feature_id INTEGER)")
        conn.execute("INSERT INTO layers VALUES (?, ?, ?, ?)", (base.explorer.run_id, "layer_19", str(artifact), 2))
        conn.executemany("INSERT INTO features VALUES (?, ?, ?)", [
            (base.explorer.run_id, "layer_19", 0),
            (base.explorer.run_id, "layer_19", 1),
        ])
    config = replace(base, storage=replace(base.storage, database=database))
    assert _registered_layer_is_complete(config, "layer_19") is True
    artifact.unlink()
    assert _registered_layer_is_complete(config, "layer_19") is False
