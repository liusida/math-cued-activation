from __future__ import annotations

from typing import Any

from datasets import load_dataset

from ..config import PipelineConfig


def load_rows(config: PipelineConfig) -> list[dict[str, Any]]:
    """Load and normalize selected rows into the artifact-compatible shape."""
    kwargs: dict[str, Any] = {"split": config.dataset.split}
    if config.dataset.revision and config.dataset.revision != "main":
        kwargs["revision"] = config.dataset.revision
    dataset = load_dataset(config.dataset.id, config.dataset.config_name, **kwargs)
    indexed = [(index, dict(row)) for index, row in enumerate(dataset)]
    start = config.dataset.start_index
    stop = start + config.dataset.sample_size
    selected = indexed[start:stop]
    return [_normalize_row(config, index, row) for index, row in selected]


def _value(row: dict[str, Any], field: str, *, index: int) -> Any:
    if field == "_index":
        return index
    if field not in row:
        available = ", ".join(sorted(row))
        raise KeyError(f"dataset field {field!r} is missing; available fields: {available}")
    return row[field]


def _normalize_row(config: PipelineConfig, index: int, row: dict[str, Any]) -> dict[str, Any]:
    problem_id = str(_value(row, config.dataset.id_field, index=index))
    if config.dataset.id_field == "_index":
        prefix = config.dataset.id.split("/")[-1].lower().replace("_", "-")
        problem_id = f"{prefix}-{index:05d}"
    problem = str(_value(row, config.dataset.prompt_field, index=index))
    answer = str(_value(row, config.dataset.answer_field, index=index))
    return {
        "Problem ID": problem_id,
        "Problem": problem,
        "Short Answer": answer,
        "Category": str(row.get("Category") or row.get("category") or "General"),
        "Subcategory": str(row.get("Subcategory") or row.get("subcategory") or ""),
        "Source": str(row.get("Source") or row.get("source") or config.dataset.id),
        "_dataset_index": index,
        "_raw_row": row,
    }


def build_prompt(config: PipelineConfig, row: dict[str, Any]) -> str:
    problem = str(row["Problem"]).strip()
    if config.dataset.id == "OpenEvals/IMO-AnswerBench":
        if config.prompt.answer_only:
            return f"{problem}\n\nProvide only the final answer."
        return f"Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{problem}\n"
    if config.dataset.id == "openai/gsm8k":
        if config.prompt.answer_only:
            return f"Solve the following problem. Return only the final numeric answer.\n\n{problem}"
        return f"Solve the following grade-school math problem. Show your reasoning, then give the final numeric answer.\n\n{problem}"
    if config.prompt.answer_only:
        return f"{problem}\n\nReturn only the final answer."
    return problem
