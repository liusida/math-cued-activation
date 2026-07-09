#!/usr/bin/env python3
"""Summarize VibeThinker IMO-AnswerBench answers for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


VIBETHINKER_FOLDER = "WeiboAI__VibeThinker-3B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/imo-answerbench-responses"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation/imo_answer_summary.csv"))
    return parser.parse_args()


def boxed_spans(text: str) -> list[str]:
    key = "\\boxed{"
    out = []
    for match in re.finditer(re.escape(key), text or ""):
        index = match.end()
        depth = 1
        cursor = index
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            out.append(text[index : cursor - 1].strip())
    return out


def strip_outer(text: str) -> str:
    text = str(text or "").strip()
    if text.lower().startswith("final answer:"):
        text = text.split(":", 1)[1].strip()
    changed = True
    while changed:
        changed = False
        value = text.strip()
        boxed = boxed_spans(value)
        if len(boxed) == 1 and value.startswith("\\boxed"):
            text = boxed[0]
            changed = True
            continue
        if value.startswith("$") and value.endswith("$"):
            text = value[1:-1].strip()
            changed = True
            continue
        if value.startswith("\\(") and value.endswith("\\)"):
            text = value[2:-2].strip()
            changed = True
            continue
    return text


def answer_candidates(generated: str) -> list[str]:
    generated = generated or ""
    candidates = boxed_spans(generated)
    final_markers = list(re.finditer(r"final\s+answer\s*:\s*", generated, flags=re.I))
    for index, marker in enumerate(final_markers):
        start = marker.end()
        end = final_markers[index + 1].start() if index + 1 < len(final_markers) else len(generated)
        candidates.append(generated[start:end].strip())
    if not candidates:
        answer_like = re.finditer(
            r"(?:thus|therefore|hence|so|then)?\s*(?:the\s+)?answer\s+(?:is|:)\s*([^\n]+)",
            generated,
            flags=re.I,
        )
        candidates.extend(
            candidate
            for match in answer_like
            if (candidate := match.group(1).strip()) and plausible_fallback(candidate)
        )
    return candidates


def plausible_fallback(candidate: str) -> bool:
    candidate = candidate.strip()
    lowered = f" {candidate.lower()} "
    if len(candidate) > 140:
        return False
    prose_markers = [
        " where ",
        " such that ",
        " satisfies ",
        " because ",
        " need to ",
        " correct ",
        " plausible ",
        " likely ",
    ]
    return not any(marker in lowered for marker in prose_markers)


def extracted_answer(generated: str) -> str:
    candidates = answer_candidates(generated)
    if not candidates:
        return ""
    answer = strip_outer(candidates[-1])
    boxes = boxed_spans(answer)
    if boxes:
        answer = boxes[-1]
    answer = re.sub(r"</?think>", "", answer).strip()
    answer = re.sub(r"^\*+|\*+$", "", answer).strip()
    answer = re.sub(r"^}+", "", answer).strip()
    answer = re.sub(r"\\\]$", "", answer).strip()
    answer = answer.rstrip(".").strip()
    return answer


def table_cell(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    reference_dir = args.root / VIBETHINKER_FOLDER
    problem_files = sorted(reference_dir.glob("*.json"))
    rows = []
    for reference_path in problem_files:
        reference = load_json(reference_path)
        problem = reference.get("problem", {})
        row = {
            "problem_id": problem.get("problem_id", reference_path.stem),
            "correct_answer": table_cell(problem.get("short_answer", "")),
        }
        path = args.root / VIBETHINKER_FOLDER / reference_path.name
        if path.exists():
            payload = load_json(path)
            generated = payload.get("text", {}).get("generated", "")
            row["vibethinker_answer"] = table_cell(extracted_answer(generated))
        else:
            row["vibethinker_answer"] = ""
        rows.append(row)

    fieldnames = [
        "problem_id",
        "correct_answer",
        "vibethinker_answer",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {args.output}")


if __name__ == "__main__":
    main()
