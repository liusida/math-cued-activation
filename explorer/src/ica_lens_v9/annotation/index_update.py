from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..paths import DEFAULT_FEATURE_INDEX
from .evidence import (
    COMPACT_EVIDENCE_FILENAME,
    DEFAULT_OUTPUT_ROOT,
    LEGACY_COMPACT_EVIDENCE_FILENAME,
    summarize_effective_receptive_field,
    update_feature_index_with_erf,
)


FEATURE_DIR_RE = re.compile(r"^F(?P<feature_id>\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import feature evidence paths and ERF summaries into SQLite.")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--run-id", action="append", default=None, help="Run id to import. Repeatable. Default: all runs.")
    parser.add_argument("--layer", action="append", default=None, help="Layer to import. Repeatable. Default: all layers.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    count = update_feature_index_from_evidence_root(
        evidence_root=args.evidence_root,
        db_path=args.db_path,
        run_ids=set(args.run_id) if args.run_id else None,
        layers=set(args.layer) if args.layer else None,
    )
    print(f"updated {count} feature evidence row(s) in {args.db_path}")


def update_feature_index_from_evidence_root(
    *,
    evidence_root: Path = DEFAULT_OUTPUT_ROOT,
    db_path: Path = DEFAULT_FEATURE_INDEX,
    run_ids: set[str] | None = None,
    layers: set[str] | None = None,
) -> int:
    evidence_root = evidence_root.resolve()
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Feature index SQLite database not found: {db_path}")
    count = 0
    paths_by_feature: dict[tuple[str, str, int], Path] = {}
    for compact_path in sorted(evidence_root.glob(f"*/layer_*/F*/{LEGACY_COMPACT_EVIDENCE_FILENAME}")):
        parsed = _parse_evidence_path(evidence_root=evidence_root, compact_path=compact_path)
        if parsed is None:
            continue
        paths_by_feature[parsed] = compact_path
    for compact_path in sorted(evidence_root.glob(f"*/layer_*/F*/{COMPACT_EVIDENCE_FILENAME}")):
        parsed = _parse_evidence_path(evidence_root=evidence_root, compact_path=compact_path)
        if parsed is None:
            continue
        paths_by_feature[parsed] = compact_path
    for (run_id, layer, feature_id), compact_path in sorted(paths_by_feature.items()):
        if run_ids is not None and run_id not in run_ids:
            continue
        if layers is not None and layer not in layers:
            continue
        packet = _read_json(compact_path)
        examples = packet.get("erf_examples") or packet.get("examples")
        if not isinstance(examples, list):
            examples = []
        update_feature_index_with_erf(
            db_path=db_path,
            run_id=run_id,
            layer=layer,
            feature_id=feature_id,
            effective_receptive_field=summarize_effective_receptive_field(examples),
            evidence_path=compact_path,
            evidence_payload=packet,
        )
        count += 1
    return count


def _parse_evidence_path(*, evidence_root: Path, compact_path: Path) -> tuple[str, str, int] | None:
    try:
        relative = compact_path.resolve().relative_to(evidence_root.resolve())
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) != 4 or parts[-1] not in {COMPACT_EVIDENCE_FILENAME, LEGACY_COMPACT_EVIDENCE_FILENAME}:
        return None
    run_id, layer, feature_dir, _ = parts
    match = FEATURE_DIR_RE.match(feature_dir)
    if not match:
        return None
    return str(run_id), str(layer), int(match.group("feature_id"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
