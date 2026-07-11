from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_root = Path(args.input_root)
    model_dir = comparison_model_dir(input_root=input_root, target_kind=str(args.target_kind), model=str(args.model), layer=str(args.layer), legacy_layout=bool(args.legacy_layout))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing AutoInterp comparison result directory: {model_dir}")

    output_root = Path(args.output_root) if args.output_root else model_dir / "annotator_exam" / str(args.candidate)
    output_root.mkdir(parents=True, exist_ok=True)

    feature_ids = selected_feature_ids(args, model_dir=model_dir)
    summary_rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        report = build_feature_report(
            model_dir=model_dir,
            feature_id=feature_id,
            candidate=str(args.candidate),
            provider_labels=set(str(label) for label in args.provider_label),
            judge_labels=set(str(label) for label in args.judge_label),
        )
        if report is None:
            continue
        report_path = output_root / f"F{feature_id:06d}.toml"
        report_path.write_text(render_feature_report_toml(report), encoding="utf-8")
        summary_rows.extend(report["annotators"])
        print(f"wrote {report_path}")

    write_summary_csv(output_root / "summary.csv", summary_rows)
    write_manifest(output_root / "manifest.json", args=args, feature_ids=feature_ids, n_rows=len(summary_rows))
    print(f"wrote summary: {output_root / 'summary.csv'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write human-readable TOML annotator exam reports for AutoInterp label-scoring runs.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-kind", choices=("sae_counterpart", "ica"), default="sae_counterpart")
    parser.add_argument("--legacy-layout", action="store_true", help="Read old result roots where model/layer lived directly under input-root.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--candidate", choices=["auto_initial", "auto_latest", "neuronpedia"], default="auto_initial")
    parser.add_argument("--provider-label", action="append", default=[], help="Only include these annotation provider labels.")
    parser.add_argument("--judge-label", action="append", default=[], help="Only include these judge labels.")
    parser.add_argument("--feature-id", type=int, action="append", default=[])
    parser.add_argument("--feature-start", type=int, default=None)
    parser.add_argument("--feature-end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def canonical_layer(layer: str) -> str:
    if layer.startswith("layer_"):
        return layer
    if layer.isdigit():
        return f"layer_{int(layer):02d}"
    return layer


def comparison_model_dir(*, input_root: Path, target_kind: str, model: str, layer: str, legacy_layout: bool) -> Path:
    if legacy_layout:
        return input_root / model / canonical_layer(layer)
    return input_root / target_kind / model / canonical_layer(layer)


def selected_feature_ids(args: argparse.Namespace, *, model_dir: Path) -> list[int]:
    if args.feature_id:
        ids = sorted({int(feature_id) for feature_id in args.feature_id})
    else:
        ids = sorted(discover_feature_ids(model_dir))
    if args.feature_start is not None:
        ids = [feature_id for feature_id in ids if feature_id >= int(args.feature_start)]
    if args.feature_end is not None:
        ids = [feature_id for feature_id in ids if feature_id <= int(args.feature_end)]
    if args.limit is not None:
        ids = ids[: int(args.limit)]
    return ids


def discover_feature_ids(model_dir: Path) -> set[int]:
    feature_ids: set[int] = set()
    for path in model_dir.glob("*_judge_*/F[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if path.is_dir():
            feature_ids.add(int(path.name[1:]))
    return feature_ids


def build_feature_report(
    *,
    model_dir: Path,
    feature_id: int,
    candidate: str,
    provider_labels: set[str],
    judge_labels: set[str],
) -> dict[str, Any] | None:
    feature_name = f"F{feature_id:06d}"
    examples = load_examples(
        model_dir=model_dir,
        feature_name=feature_name,
        candidate=candidate,
        provider_labels=provider_labels,
        judge_labels=judge_labels,
    )
    if examples is None:
        return None

    annotators = []
    correct = [int(example["index"]) for example in examples if bool(example.get("is_active"))]
    for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and "_judge_" in path.name):
        score_path = run_dir / feature_name / f"{candidate}_score.json"
        if not score_path.is_file():
            continue
        packet = read_json(score_path)
        row = packet.get("row") if isinstance(packet, dict) else {}
        candidate_packet = packet.get("candidate") if isinstance(packet, dict) else {}
        if not isinstance(row, dict) or not isinstance(candidate_packet, dict):
            continue
        provider_label, judge_label = split_run_name(run_dir.name)
        if provider_labels and provider_label not in provider_labels:
            continue
        if judge_labels and judge_label not in judge_labels:
            continue
        predicted = [int(value) for value in (row.get("predictions") or [])]
        row_correct = [int(value) for value in (row.get("correct_sequences") or correct)]
        false_positives = sorted(set(predicted) - set(row_correct))
        false_negatives = sorted(set(row_correct) - set(predicted))
        annotator = {
            "feature_id": int(feature_id),
            "provider_label": provider_label,
            "judge_label": judge_label,
            "candidate": candidate,
            "label": str(candidate_packet.get("label") or ""),
            "simple_label": str(candidate_packet.get("simple_label") or ""),
            "description": str(candidate_packet.get("description") or ""),
            "confidence": str(candidate_packet.get("confidence") or ""),
            "source_path": str(candidate_packet.get("source_path") or ""),
            "score": float(row.get("saebench_score") or 0.0),
            "predicted_active": predicted,
            "correct_active": row_correct,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "judge_output_text": str(packet.get("judge_output_text") or ""),
            "judge_cache_status": str(row.get("judge_cache_status") or ""),
            "feedback": feedback_for_annotator(
                score=float(row.get("saebench_score") or 0.0),
                description=str(candidate_packet.get("description") or ""),
                false_positives=false_positives,
                false_negatives=false_negatives,
                examples=examples,
            ),
        }
        annotator["false_positive_examples"] = example_snippets(examples, false_positives)
        annotator["false_negative_examples"] = example_snippets(examples, false_negatives)
        annotators.append(annotator)

    if not annotators:
        return None
    annotators.sort(key=lambda row: (-float(row["score"]), str(row["provider_label"])))
    return {
        "feature": {
            "model": model_dir.parent.name,
            "layer": model_dir.name,
            "feature_id": int(feature_id),
            "feature": feature_name,
            "candidate": candidate,
            "n_examples": len(examples),
            "correct_active": correct,
        },
        "examples": examples,
        "annotators": annotators,
    }


def load_examples(
    *,
    model_dir: Path,
    feature_name: str,
    candidate: str,
    provider_labels: set[str],
    judge_labels: set[str],
) -> list[dict[str, Any]] | None:
    for run_dir in filtered_run_dirs(model_dir=model_dir, provider_labels=provider_labels, judge_labels=judge_labels):
        examples = load_examples_from_request(run_dir=run_dir, feature_name=feature_name, candidate=candidate)
        if examples is not None:
            return examples
    for run_dir in filtered_run_dirs(model_dir=model_dir, provider_labels=provider_labels, judge_labels=judge_labels):
        path = run_dir / feature_name / "saebench_scoring_examples.json"
        try:
            rows = read_json(path)
        except Exception:
            continue
        if isinstance(rows, list):
            return normalize_examples(rows)
    return None


def filtered_run_dirs(*, model_dir: Path, provider_labels: set[str], judge_labels: set[str]) -> list[Path]:
    out = []
    for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and "_judge_" in path.name):
        provider_label, judge_label = split_run_name(run_dir.name)
        if provider_labels and provider_label not in provider_labels:
            continue
        if judge_labels and judge_label not in judge_labels:
            continue
        out.append(run_dir)
    return out


def load_examples_from_request(*, run_dir: Path, feature_name: str, candidate: str) -> list[dict[str, Any]] | None:
    feature_dir = run_dir / feature_name
    score_path = feature_dir / f"{candidate}_score.json"
    request_path = feature_dir / f"{candidate}_request_preview.json"
    if score_path.is_file():
        try:
            score_packet = read_json(score_path)
        except Exception:
            score_packet = None
        if isinstance(score_packet, dict) and isinstance(score_packet.get("scoring_examples"), list):
            return normalize_examples(score_packet["scoring_examples"])
    if not score_path.is_file() or not request_path.is_file():
        return None
    score_packet = read_json(score_path)
    request_packet = read_json(request_path)
    row = score_packet.get("row") if isinstance(score_packet, dict) else {}
    correct = {int(value) for value in row.get("correct_sequences", [])} if isinstance(row, dict) else set()
    messages = request_packet.get("messages") if isinstance(request_packet, dict) else None
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    content = str(messages[-1].get("content") or "")
    rows = parse_numbered_examples(content)
    if not rows:
        return None
    return [
        {
            "index": int(index),
            "is_active": int(index) in correct,
            "max_activation": 0.0,
            "sequence": sequence,
            "marked_sequence": sequence,
            "source": "judge_request",
            "note": "Exact judge-request sequence; max activation and token marks are not stored in this legacy score JSON.",
        }
        for index, sequence in rows
    ]


def parse_numbered_examples(content: str) -> list[tuple[int, str]]:
    marker = "Here are the examples:"
    if marker in content:
        content = content.split(marker, 1)[1]
    rows: list[tuple[int, str]] = []
    current_index: int | None = None
    current_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.lstrip()
        prefix, dot, rest = stripped.partition(".")
        if dot and prefix.isdigit():
            if current_index is not None:
                rows.append((current_index, "\n".join(current_lines).rstrip()))
            current_index = int(prefix)
            current_lines = [rest[1:] if rest.startswith(" ") else rest]
        elif current_index is not None:
            current_lines.append(line)
    if current_index is not None:
        rows.append((current_index, "\n".join(current_lines).rstrip()))
    return rows


def normalize_examples(rows: list[Any]) -> list[dict[str, Any]]:
    examples = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        examples.append(
            {
                "index": int(row.get("index") or index),
                "is_active": bool(row.get("is_active")),
                "max_activation": float(row.get("max_activation") or 0.0),
                "sequence": str(row.get("sequence") or ""),
                "marked_sequence": str(row.get("marked_sequence") or row.get("sequence") or ""),
            }
        )
    examples.sort(key=lambda row: int(row["index"]))
    return examples


def split_run_name(run_name: str) -> tuple[str, str]:
    provider, sep, judge = run_name.partition("_judge_")
    if not sep:
        return run_name, ""
    return provider, judge


def feedback_for_annotator(
    *,
    score: float,
    description: str,
    false_positives: list[int],
    false_negatives: list[int],
    examples: list[dict[str, Any]],
) -> str:
    parts = []
    if score >= 0.95:
        parts.append("Excellent match to the scoring examples.")
    elif score >= 0.90:
        parts.append("Strong label; remaining errors are small or noisy.")
    elif false_negatives and not false_positives:
        parts.append("Likely too narrow: it missed active examples without adding false positives.")
    elif false_positives and not false_negatives:
        parts.append("Likely too broad: it selected inactive examples.")
    elif false_positives and false_negatives:
        parts.append("Boundary is unclear: it has both false positives and false negatives.")
    else:
        parts.append("Weak score; inspect the label and scoring examples.")

    lower_description = description.lower()
    if false_negatives and any(term in lower_description for term in ["leading space", "tokenized", "tokenizer", "token "]):
        parts.append("Tokenizer-facing wording may make a plain-text judge too conservative.")

    low_activation_false_negatives = [
        index
        for index in false_negatives
        if (example := example_by_index(examples, index)) is not None and float(example["max_activation"]) < 2.0
    ]
    if low_activation_false_negatives:
        parts.append(
            "Some missed active examples have very low activation and may be scoring-set noise: "
            + ", ".join(str(index) for index in low_activation_false_negatives)
            + "."
        )
    return " ".join(parts)


def example_by_index(examples: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for example in examples:
        if int(example["index"]) == int(index):
            return example
    return None


def example_snippets(examples: list[dict[str, Any]], indices: list[int]) -> list[str]:
    snippets = []
    for index in indices:
        example = example_by_index(examples, index)
        if example is None:
            continue
        sequence = compact_text(str(example["sequence"]))
        snippets.append(f"{index}: {sequence[:180]}")
    return snippets


def compact_text(text: str) -> str:
    return " ".join(text.replace("\n", "↵").split())


def render_feature_report_toml(report: dict[str, Any]) -> str:
    lines = [
        "# AutoInterp Annotator Exam Report",
        "# Generated from saved label-scoring JSONs.",
        "",
        "[feature]",
    ]
    for key, value in report["feature"].items():
        lines.append(toml_line(key, value))
    lines.append("")

    for example in report["examples"]:
        lines.append("[[examples]]")
        for key in ["index", "is_active", "max_activation", "sequence", "marked_sequence", "source", "note"]:
            if key in example:
                lines.append(toml_line(key, example[key]))
        lines.append("")

    for annotator in report["annotators"]:
        lines.append("[[annotators]]")
        for key in [
            "provider_label",
            "judge_label",
            "candidate",
            "label",
            "simple_label",
            "description",
            "confidence",
            "score",
            "predicted_active",
            "correct_active",
            "false_positives",
            "false_negatives",
            "false_positive_examples",
            "false_negative_examples",
            "feedback",
            "judge_output_text",
            "judge_cache_status",
            "source_path",
        ]:
            lines.append(toml_line(key, annotator[key]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def toml_line(key: str, value: Any) -> str:
    return f"{key} = {toml_value(value)}"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "provider_label",
        "judge_label",
        "candidate",
        "score",
        "label",
        "simple_label",
        "description",
        "false_positives",
        "false_negatives",
        "feedback",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_id", *fields])
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            out["feature_id"] = str(row.get("feature_id") or feature_id_from_source_path(str(row.get("source_path") or "")))
            out["false_positives"] = " ".join(str(value) for value in row.get("false_positives", []))
            out["false_negatives"] = " ".join(str(value) for value in row.get("false_negatives", []))
            writer.writerow(out)


def feature_id_from_source_path(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("F") and part[1:].isdigit():
            return str(int(part[1:]))
    return ""


def write_manifest(path: Path, *, args: argparse.Namespace, feature_ids: list[int], n_rows: int) -> None:
    packet = {
        "input_root": str(Path(args.input_root).resolve()),
        "target_kind": str(args.target_kind),
        "legacy_layout": bool(args.legacy_layout),
        "model": str(args.model),
        "layer": canonical_layer(str(args.layer)),
        "candidate": str(args.candidate),
        "provider_labels": [str(label) for label in args.provider_label],
        "judge_labels": [str(label) for label in args.judge_label],
        "feature_ids": [int(feature_id) for feature_id in feature_ids],
        "n_annotator_rows": int(n_rows),
    }
    path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
