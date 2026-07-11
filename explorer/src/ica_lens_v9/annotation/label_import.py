from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..paths import DEFAULT_FEATURE_INDEX, V9_ROOT


DEFAULT_ANNOTATION_ROOT = V9_ROOT / "results" / "auto_annotation" / "annotations"
DEFAULT_REFINEMENT_ROOT = V9_ROOT / "results" / "auto_annotation" / "refinements"
ANNOTATION_RE = re.compile(r"^(?P<label>.+)_annotation\.json$")
REFINEMENT_RE = re.compile(r"^(?P<label>.+)_round(?P<round_index>\d+)_annotation\.json$")
FEATURE_DIR_RE = re.compile(r"^F(?P<feature_id>\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import auto-annotation labels into the feature SQLite index.")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--refinement-root", type=Path, default=DEFAULT_REFINEMENT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--run-id", action="append", default=None, help="Run id to import. Repeatable. Default: all runs.")
    parser.add_argument("--layer", action="append", default=None, help="Layer to import. Repeatable. Default: all layers.")
    parser.add_argument("--provider-label", default=None, help="Import only a file label such as 'mi' or 'mi_pro'.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    count = import_annotation_labels(
        annotation_root=args.annotation_root,
        refinement_root=args.refinement_root,
        db_path=args.db_path,
        run_ids=set(args.run_id) if args.run_id else None,
        layers=set(args.layer) if args.layer else None,
        provider_label=args.provider_label,
    )
    print(f"imported {count} annotation label(s) into {args.db_path}")


def import_annotation_labels(
    *,
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT,
    refinement_root: Path = DEFAULT_REFINEMENT_ROOT,
    db_path: Path = DEFAULT_FEATURE_INDEX,
    run_ids: set[str] | None = None,
    layers: set[str] | None = None,
    provider_label: str | None = None,
) -> int:
    annotation_root = annotation_root.resolve()
    refinement_root = refinement_root.resolve()
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Feature index SQLite database not found: {db_path}")

    rows = []
    for annotation_path in sorted(annotation_root.glob("*/*/F*/*_annotation.json")):
        parsed = _parse_annotation_path(annotation_root=annotation_root, annotation_path=annotation_path)
        if parsed is None:
            continue
        run_id, layer, feature_id, file_label = parsed
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        packet = _read_json(annotation_path)
        annotation = packet.get("annotation")
        if not isinstance(annotation, dict):
            continue
        rows.append(
            (
                run_id,
                layer,
                int(feature_id),
                file_label,
                str(packet.get("provider") or ""),
                str(packet.get("model") or ""),
                str(packet.get("created_at") or ""),
                str(annotation.get("label") or ""),
                str(annotation.get("simple_label") or ""),
                str(annotation.get("description") or ""),
                str(annotation.get("reasoning") or ""),
                str(annotation.get("confidence") or "unclear").lower(),
                json.dumps(annotation.get("test_cases") or [], ensure_ascii=False, sort_keys=True),
                str(annotation_path.resolve()),
                str(annotation_path.with_name(f"{file_label}_raw_response.json").resolve()),
            )
        )
    refinement_rows = _collect_refinement_rows(
        refinement_root=refinement_root,
        run_ids=run_ids,
        layers=layers,
        provider_label=provider_label,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _ensure_annotation_columns(conn)
        _ensure_feature_annotations_table(conn)
        _ensure_refinement_schema(conn)
        _prune_missing_refinements(
            conn,
            run_ids=run_ids,
            layers=layers,
            provider_label=provider_label,
        )
        _delete_stopped_refinements(
            conn,
            refinement_root=refinement_root,
            run_ids=run_ids,
            layers=layers,
            provider_label=provider_label,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO feature_annotations (
                run_id, layer, feature_id, provider_label, provider, model, created_at,
                label, simple_label, description, reasoning, confidence, test_cases_json,
                annotation_path, raw_response_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.executemany(
            """
            UPDATE features
            SET annotation_label = ?,
                annotation_simple_label = ?,
                annotation_description = ?,
                annotation_reasoning = ?,
                annotation_confidence = ?,
                annotation_provider = ?,
                annotation_model = ?,
                annotation_path = ?,
                annotation_raw_response_path = ?
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            [
                (
                    label,
                    simple_label,
                    description,
                    reasoning,
                    confidence,
                    provider,
                    model,
                    annotation_path,
                    raw_response_path,
                    run_id,
                    layer,
                    feature_id,
                )
                for (
                    run_id,
                    layer,
                    feature_id,
                    _provider_label,
                    provider,
                    model,
                    _created_at,
                    label,
                    simple_label,
                    description,
                    reasoning,
                    confidence,
                    _test_cases_json,
                    annotation_path,
                    raw_response_path,
                ) in rows
            ],
        )
        conn.executemany(
            """
            INSERT INTO feature_annotation_refinements (
                run_id, layer, feature_id, provider_label, round_index,
                provider, model, created_at, source_annotation_path,
                tests_path, request_path, raw_response_path, annotation_path,
                label_before_json, test_results_json,
                annotation_label, annotation_simple_label, annotation_description,
                annotation_reasoning, annotation_confidence, annotation_test_cases_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, layer, feature_id, provider_label, round_index) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                created_at = excluded.created_at,
                source_annotation_path = excluded.source_annotation_path,
                tests_path = excluded.tests_path,
                request_path = excluded.request_path,
                raw_response_path = excluded.raw_response_path,
                annotation_path = excluded.annotation_path,
                label_before_json = excluded.label_before_json,
                test_results_json = excluded.test_results_json,
                annotation_label = excluded.annotation_label,
                annotation_simple_label = excluded.annotation_simple_label,
                annotation_description = excluded.annotation_description,
                annotation_reasoning = excluded.annotation_reasoning,
                annotation_confidence = excluded.annotation_confidence,
                annotation_test_cases_json = excluded.annotation_test_cases_json
            """,
            refinement_rows,
        )
        _promote_latest_refinements(
            conn,
            run_ids=run_ids,
            layers=layers,
            provider_label=provider_label,
        )
        conn.commit()
    return len(rows) + len(refinement_rows)


def _prune_missing_refinements(
    conn: sqlite3.Connection,
    *,
    run_ids: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> None:
    try:
        rows = conn.execute(
            """
            SELECT run_id, layer, feature_id, provider_label, round_index, annotation_path
            FROM feature_annotation_refinements
            WHERE COALESCE(annotation_path, '') != ''
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return
    deletes = []
    for row in rows:
        run_id = str(row["run_id"])
        layer = str(row["layer"])
        row_provider_label = str(row["provider_label"])
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and row_provider_label != provider_label:
            continue
        if Path(str(row["annotation_path"])).exists():
            continue
        deletes.append(
            (
                run_id,
                layer,
                int(row["feature_id"]),
                row_provider_label,
                int(row["round_index"]),
            )
        )
    conn.executemany(
        """
        DELETE FROM feature_annotation_refinements
        WHERE run_id = ?
          AND layer = ?
          AND feature_id = ?
          AND provider_label = ?
          AND round_index >= ?
        """,
        deletes,
    )


def _delete_stopped_refinements(
    conn: sqlite3.Connection,
    *,
    refinement_root: Path,
    run_ids: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> None:
    deletes = []
    for stop_path in sorted(refinement_root.glob("*/*/F*/*_round*_stop.json")):
        annotation_path = stop_path.with_name(stop_path.name.replace("_stop.json", "_annotation.json"))
        parsed = _parse_refinement_path(refinement_root=refinement_root, annotation_path=annotation_path)
        if parsed is None:
            continue
        run_id, layer, feature_id, file_label, round_index = parsed
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        deletes.append((run_id, layer, int(feature_id), file_label, int(round_index)))
    conn.executemany(
        """
        DELETE FROM feature_annotation_refinements
        WHERE run_id = ?
          AND layer = ?
          AND feature_id = ?
          AND provider_label = ?
          AND round_index = ?
        """,
        deletes,
    )


def _has_stop_at_or_before(*, annotation_path: Path, file_label: str, round_index: int) -> bool:
    return any(
        annotation_path.with_name(f"{file_label}_round{index:02d}_stop.json").is_file()
        for index in range(1, int(round_index) + 1)
    )


def _collect_refinement_rows(
    *,
    refinement_root: Path,
    run_ids: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> list[tuple[Any, ...]]:
    rows = []
    for annotation_path in sorted(refinement_root.glob("*/*/F*/*_round*_annotation.json")):
        parsed = _parse_refinement_path(refinement_root=refinement_root, annotation_path=annotation_path)
        if parsed is None:
            continue
        run_id, layer, feature_id, file_label, round_index = parsed
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        if _has_stop_at_or_before(annotation_path=annotation_path, file_label=file_label, round_index=int(round_index)):
            continue
        packet = _read_json(annotation_path)
        annotation = packet.get("annotation")
        if not isinstance(annotation, dict):
            continue
        tests_path = Path(str(packet.get("test_results") or annotation_path.with_name(f"{file_label}_round{round_index:02d}_tests.json")))
        request_path = annotation_path.with_name(f"{file_label}_round{round_index:02d}_request_preview.json")
        raw_response_path = annotation_path.with_name(f"{file_label}_round{round_index:02d}_raw_response.json")
        tests_packet = _read_json(tests_path) if tests_path.is_file() else {}
        rows.append(
            (
                run_id,
                layer,
                int(feature_id),
                file_label,
                int(round_index),
                str(packet.get("provider") or ""),
                str(packet.get("model") or ""),
                str(packet.get("created_at") or tests_packet.get("created_at") or ""),
                str(packet.get("source_annotation") or tests_packet.get("source_annotation") or ""),
                str(tests_path.resolve()) if tests_path.is_file() else None,
                str(request_path.resolve()) if request_path.is_file() else None,
                str(raw_response_path.resolve()) if raw_response_path.is_file() else None,
                str(annotation_path.resolve()),
                json.dumps(tests_packet.get("label_before_tests"), ensure_ascii=False, sort_keys=True)
                if "label_before_tests" in tests_packet
                else None,
                json.dumps(tests_packet.get("test_results") or [], ensure_ascii=False, sort_keys=True),
                str(annotation.get("label") or ""),
                str(annotation.get("simple_label") or ""),
                str(annotation.get("description") or ""),
                str(annotation.get("reasoning") or ""),
                str(annotation.get("confidence") or "").lower(),
                json.dumps(annotation.get("test_cases") or [], ensure_ascii=False, sort_keys=True),
            )
        )
    return rows


def _parse_annotation_path(*, annotation_root: Path, annotation_path: Path) -> tuple[str, str, int, str] | None:
    try:
        relative = annotation_path.resolve().relative_to(annotation_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 4:
        return None
    run_id, layer, feature_dir, filename = parts
    feature_match = FEATURE_DIR_RE.match(feature_dir)
    annotation_match = ANNOTATION_RE.match(filename)
    if not feature_match or not annotation_match:
        return None
    return str(run_id), str(layer), int(feature_match.group("feature_id")), str(annotation_match.group("label"))


def _parse_refinement_path(*, refinement_root: Path, annotation_path: Path) -> tuple[str, str, int, str, int] | None:
    try:
        relative = annotation_path.resolve().relative_to(refinement_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 4:
        return None
    run_id, layer, feature_dir, filename = parts
    feature_match = FEATURE_DIR_RE.match(feature_dir)
    refinement_match = REFINEMENT_RE.match(filename)
    if not feature_match or not refinement_match:
        return None
    return (
        str(run_id),
        str(layer),
        int(feature_match.group("feature_id")),
        str(refinement_match.group("label")),
        int(refinement_match.group("round_index")),
    )


def _ensure_annotation_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    columns = {
        "annotation_label": "TEXT",
        "annotation_simple_label": "TEXT",
        "annotation_description": "TEXT",
        "annotation_reasoning": "TEXT",
        "annotation_confidence": "TEXT",
        "annotation_provider": "TEXT",
        "annotation_model": "TEXT",
        "annotation_path": "TEXT",
        "annotation_raw_response_path": "TEXT",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {name} {definition}")


def _ensure_feature_annotations_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feature_annotations (
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
        CREATE INDEX IF NOT EXISTS idx_feature_annotations_lookup
        ON feature_annotations(run_id, layer, feature_id);
        """
    )
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(feature_annotations)").fetchall()}
    if "test_cases_json" not in existing:
        conn.execute("ALTER TABLE feature_annotations ADD COLUMN test_cases_json TEXT")


def _ensure_refinement_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_annotation_refinements (
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
            PRIMARY KEY (run_id, layer, feature_id, provider_label, round_index)
        )
        """
    )
    existing = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(feature_annotation_refinements)").fetchall()
    }
    columns = {
        "provider": "TEXT",
        "model": "TEXT",
        "created_at": "TEXT",
        "source_annotation_path": "TEXT",
        "tests_path": "TEXT",
        "request_path": "TEXT",
        "raw_response_path": "TEXT",
        "annotation_path": "TEXT",
        "label_before_json": "TEXT",
        "test_results_json": "TEXT",
        "annotation_label": "TEXT",
        "annotation_simple_label": "TEXT",
        "annotation_description": "TEXT",
        "annotation_reasoning": "TEXT",
        "annotation_confidence": "TEXT",
        "annotation_test_cases_json": "TEXT",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE feature_annotation_refinements ADD COLUMN {name} {definition}")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_feature_annotation_refinements_lookup
        ON feature_annotation_refinements(run_id, layer, feature_id, provider_label, round_index)
        """
    )


def _promote_latest_refinements(
    conn: sqlite3.Connection,
    *,
    run_ids: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> None:
    try:
        rows = conn.execute(
            """
            SELECT r.*
            FROM feature_annotation_refinements r
            WHERE COALESCE(r.annotation_label, '') != ''
              AND NOT EXISTS (
                  SELECT 1
                  FROM feature_annotation_refinements newer
                  WHERE newer.run_id = r.run_id
                    AND newer.layer = r.layer
                    AND newer.feature_id = r.feature_id
                    AND newer.provider_label = r.provider_label
                    AND COALESCE(newer.annotation_label, '') != ''
                    AND newer.round_index > r.round_index
              )
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return
    updates = []
    for row in rows:
        run_id = str(row["run_id"])
        layer = str(row["layer"])
        row_provider_label = str(row["provider_label"])
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and row_provider_label != provider_label:
            continue
        updates.append(
            (
                str(row["annotation_label"] or ""),
                str(row["annotation_simple_label"] or ""),
                str(row["annotation_description"] or ""),
                str(row["annotation_reasoning"] or ""),
                str(row["annotation_confidence"] or "").lower(),
                f"{row_provider_label}:refinement:{int(row['round_index'])}",
                str(row["model"] or ""),
                str(row["annotation_path"] or ""),
                str(row["raw_response_path"] or ""),
                run_id,
                layer,
                int(row["feature_id"]),
            )
        )
    conn.executemany(
        """
        UPDATE features
        SET annotation_label = ?,
            annotation_simple_label = ?,
            annotation_description = ?,
            annotation_reasoning = ?,
            annotation_confidence = ?,
            annotation_provider = ?,
            annotation_model = ?,
            annotation_path = ?,
            annotation_raw_response_path = ?
        WHERE run_id = ? AND layer = ? AND feature_id = ?
        """,
        updates,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
