from dataclasses import replace
from pathlib import Path
import sqlite3

from math_cued_activation.config import load_config
from math_cued_activation.stages import _feature_column_complete, _feature_paths_complete, _top_sample_evidence_complete


ROOT = Path(__file__).resolve().parents[1]


def test_enrichment_completion_checks_files_and_sqlite(tmp_path: Path) -> None:
    base = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_gsm8k.toml")
    histogram = tmp_path / "feature.svg"
    evidence = tmp_path / "evidence.json"
    histogram.write_text("<svg/>")
    evidence.write_text("{}")
    database = tmp_path / "features.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE features (run_id TEXT, layer TEXT, feature_id INTEGER, kurtosis REAL, mini_histogram_svg_path TEXT, annotation_evidence_path TEXT, annotation_evidence_json TEXT)")
        conn.execute("INSERT INTO features VALUES (?, ?, 0, 3.0, ?, ?, '{}')", (base.explorer.run_id, "layer_19", str(histogram), str(evidence)))
    config = replace(base, storage=replace(base.storage, database=database))
    assert _feature_column_complete(config, "layer_19", "kurtosis")
    assert _feature_paths_complete(config, "layer_19", "mini_histogram_svg_path")
    assert _top_sample_evidence_complete(config, "layer_19")
    evidence.unlink()
    assert not _top_sample_evidence_complete(config, "layer_19")
