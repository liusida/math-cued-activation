from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..features.probe import DEFAULT_DB_PATH, connect, load_feature_bundle
from ..model_runtime import hidden_states_for_layer, load_runtime
from ..paths import V9_ROOT
from .auto_annotate import (
    ANNOTATION_RESPONSE_FORMAT,
    ANNOTATION_SCHEMA,
    DEFAULT_ANNOTATION_ROOT,
    DEFAULT_ICL_EXAMPLE_ROOT,
    DEFAULT_MI_TOKEN_PATH,
    SYSTEM_PROMPT,
    apply_openai_reasoning_controls,
    build_annotation_style_guide,
    call_annotation_provider,
    compact_evidence_for_prompt,
    default_base_url_for_provider,
    default_model_for_provider,
    extract_chat_completion_text,
    env_base_url_key,
    load_api_key_for_provider,
    parse_annotation,
    repair_annotation_output,
    resolve_model,
    warn_single_token_test_cases,
    write_json,
)
from .evidence import COMPACT_EVIDENCE_FILENAME, DEFAULT_OUTPUT_ROOT as DEFAULT_EVIDENCE_ROOT, LEGACY_COMPACT_EVIDENCE_FILENAME


DEFAULT_REFINEMENT_ROOT = V9_ROOT / "results" / "auto_annotation" / "refinements"
MARKED_TEST_RESULT_MAX_RANK = 5
REFINEMENT_TEST_INSTRUCTION = {
    "task": "Audit another annotator's annotation using all available evidence: original evidence plus added test results. Decide whether to agree with it or correct it.",
    "output": "Return JSON with keys: reasoning, label, simple_label, description, confidence, test_cases.",
    "label_style": "If you agree, keep the annotator's label. If you disagree, revise it using label_naming_policy from the style guide. Use simple_label for plain English details.",
    "evidence_policy": (
        "Treat original evidence and added test results together as evidence. "
        "The final label must explain the original evidence and should only differ from the annotator's label when the combined evidence supports a better label."
    ),
    "critique_policy": "Criticize the annotator's label only when it fails to explain the combined evidence. Otherwise agree and keep it.",
    "test_result_policy": (
        f"Use marked_result as the measured activation evidence, not as a prediction. Marks [1] through [{MARKED_TEST_RESULT_MAX_RANK}] "
        f"mean this feature ranked in the top {MARKED_TEST_RESULT_MAX_RANK} at that token; unmarked tokens did not strongly activate this feature. "
        "For each marked token, interpret the cause using only that token and its left context, not later right context."
    ),
    "reasoning_style": "Give one concise audit rationale sentence, under 40 words. Do not write step-by-step analysis.",
    "next_test_strategy": (
        "The test_cases you return are new tests for the next refinement round, not a history of tests already run. "
        "Do not repeat or paraphrase prior tests. "
        "Probe unresolved boundaries from the combined evidence, not only the most recent test results; if prior results are sparse or contradictory, broaden diagnostics instead of narrowing around one positive case. "
        "Avoid single-word prompts. Unless the hypothesis is about beginning-of-text or token position, put the suspected activating token later in a short natural context. "
        "For causal language models, the decisive evidence should be at or before the tested token, not after it."
    ),
}


@dataclass(frozen=True)
class RefinementConfig:
    run_id: str
    layer: str
    feature_id: int
    provider_label: str
    source_annotation: Path | None = None
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT
    icl_example_root: Path = DEFAULT_ICL_EXAMPLE_ROOT
    refinement_root: Path = DEFAULT_REFINEMENT_ROOT
    db_path: Path = DEFAULT_DB_PATH
    round_index: int | None = None
    top_k: int = 10
    activation_relative_threshold: float = 0.2
    device: str = "auto"
    dtype: str = "auto"
    provider: str = "ollama"
    model: str = "local-qwen"
    base_url: str | None = None
    api_key_file: Path = DEFAULT_MI_TOKEN_PATH
    max_tokens: int = 1400
    timeout: float = 120.0
    retries: int = 3
    retry_sleep_seconds: float = 15.0
    call_provider: bool = False
    force: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one annotation refinement round from proposed test cases.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--provider-label", default="local-qwen", help="Annotation file label to refine, e.g. local-qwen or mi.")
    parser.add_argument("--source-annotation", type=Path, default=None, help="Annotation JSON to refine. Defaults to prior round if present, otherwise provider-label annotation.")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--icl-example-root", type=Path, default=DEFAULT_ICL_EXAMPLE_ROOT)
    parser.add_argument("--refinement-root", type=Path, default=DEFAULT_REFINEMENT_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--round", dest="round_index", type=int, default=None, help="Refinement round to run. Omit to run the next stale/missing round automatically.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--activation-relative-threshold", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--provider", choices=["mi", "ollama", "openai", "deepseek", "tinker-sdk"], default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_MI_TOKEN_PATH)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=15.0)
    parser.add_argument("--call-provider", action="store_true", help="Call the annotation provider for a corrected label.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    provider = str(args.provider)
    cfg = RefinementConfig(
        run_id=str(args.run_id),
        layer=str(args.layer),
        feature_id=int(args.feature_id),
        provider_label=str(args.provider_label),
        source_annotation=args.source_annotation,
        annotation_root=args.annotation_root,
        evidence_root=args.evidence_root,
        icl_example_root=args.icl_example_root,
        refinement_root=args.refinement_root,
        db_path=args.db_path,
        round_index=int(args.round_index) if args.round_index is not None else None,
        top_k=int(args.top_k),
        activation_relative_threshold=float(args.activation_relative_threshold),
        device=str(args.device),
        dtype=str(args.dtype),
        provider=provider,
        model=resolve_model(str(args.model or default_model_for_provider(provider))),
        base_url=str(
            args.base_url
            or os.environ.get(env_base_url_key(provider))
            or default_base_url_for_provider(provider)
        ),
        api_key_file=args.api_key_file,
        max_tokens=int(args.max_tokens),
        timeout=float(args.timeout),
        retries=int(args.retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
        call_provider=bool(args.call_provider),
        force=bool(args.force),
    )
    outputs = run_refinement(cfg)
    for path in outputs:
        print(path)


def run_refinement(cfg: RefinementConfig) -> list[Path]:
    cfg = resolve_refinement_round(cfg)
    paths = refinement_paths(cfg)
    if paths["tests"].exists() and not cfg.force:
        test_packet = read_json(paths["tests"])
    else:
        source_annotation = annotation_path(cfg)
        annotation_packet = read_json(source_annotation)
        annotation = annotation_packet.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation file has no annotation object: {source_annotation}")
        test_cases = annotation.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            raise ValueError(f"Annotation has no test_cases. Re-run annotation with the updated schema: {source_annotation}")
        test_packet = build_test_packet(cfg=cfg, annotation_packet=annotation_packet, test_cases=test_cases)
        write_json(paths["tests"], test_packet)

    request_payload = build_refinement_request(cfg=cfg, test_packet=test_packet)
    write_json(paths["request"], request_payload)
    write_request_debug_bundle(paths["request_debug_dir"], request_payload)
    outputs = [paths["tests"], paths["request"], paths["request_debug_dir"]]
    if not cfg.call_provider:
        return outputs

    api_key = load_api_key_for_provider(cfg.provider, cfg.api_key_file)
    response_json = call_annotation_provider(
        request_payload=request_payload,
        provider=cfg.provider,
        api_key=api_key,
        base_url=cfg.base_url or default_base_url_for_provider(cfg.provider),
        timeout=float(cfg.timeout),
        retries=int(cfg.retries),
        retry_sleep_seconds=float(cfg.retry_sleep_seconds),
    )
    write_json(paths["raw_response"], response_json)
    output_text = extract_chat_completion_text(response_json)
    repair_response_json = None
    try:
        annotation = parse_annotation(output_text)
    except Exception:
        annotation, repair_response_json = repair_annotation_output(
            malformed_output=output_text,
            original_request_payload=request_payload,
            provider=cfg.provider,
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url or default_base_url_for_provider(cfg.provider),
            timeout=float(cfg.timeout),
            retries=int(cfg.retries),
            retry_sleep_seconds=float(cfg.retry_sleep_seconds),
            max_tokens=int(cfg.max_tokens),
        )
        write_json(paths["repair_raw_response"], repair_response_json)
    response_info = {
        "id": response_json.get("id"),
        "usage": response_json.get("usage"),
    }
    if repair_response_json is not None:
        response_info["repaired_from_raw_response"] = str(paths["raw_response"].resolve())
        response_info["repair_raw_response"] = str(paths["repair_raw_response"].resolve())
        response_info["repair_id"] = repair_response_json.get("id")
        response_info["repair_usage"] = repair_response_json.get("usage")
    warn_single_token_test_cases(
        annotation,
        model_id=model_id_for_run(cfg.run_id, cfg.db_path),
        context=f"{cfg.run_id}/{cfg.layer}/F{int(cfg.feature_id):06d} round {int(cfg.round_index)}",
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": cfg.provider,
        "model": cfg.model,
        "round": int(cfg.round_index),
        "source_annotation": str(annotation_path(cfg).resolve()),
        "test_results": str(paths["tests"].resolve()),
        "feature": f"{cfg.run_id}/{cfg.layer}/F{cfg.feature_id}",
        "annotation": annotation,
        "response": response_info,
    }
    write_json(paths["annotation"], result)
    outputs.extend([paths["raw_response"], paths["annotation"]])
    return outputs


def resolve_refinement_round(cfg: RefinementConfig) -> RefinementConfig:
    if cfg.round_index is not None:
        return cfg
    round_index = select_auto_round(cfg)
    return RefinementConfig(
        run_id=cfg.run_id,
        layer=cfg.layer,
        feature_id=cfg.feature_id,
        provider_label=cfg.provider_label,
        source_annotation=cfg.source_annotation,
        annotation_root=cfg.annotation_root,
        evidence_root=cfg.evidence_root,
        icl_example_root=cfg.icl_example_root,
        refinement_root=cfg.refinement_root,
        db_path=cfg.db_path,
        round_index=round_index,
        top_k=cfg.top_k,
        activation_relative_threshold=cfg.activation_relative_threshold,
        device=cfg.device,
        dtype=cfg.dtype,
        provider=cfg.provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_file=cfg.api_key_file,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        retries=cfg.retries,
        retry_sleep_seconds=cfg.retry_sleep_seconds,
        call_provider=cfg.call_provider,
        force=cfg.force,
    )


def select_auto_round(cfg: RefinementConfig) -> int:
    if cfg.source_annotation is not None:
        return 1
    base_annotation = base_annotation_path(cfg)
    if not base_annotation.is_file():
        raise FileNotFoundError(f"Base annotation JSON not found: {base_annotation}")
    round_index = 1
    while True:
        source = source_annotation_for_round(cfg, round_index)
        annotation = refinement_paths_for_round(cfg, round_index)["annotation"]
        if cfg.force and round_index > 1 and not source.is_file():
            return round_index - 1
        if not annotation.is_file() or file_mtime(annotation) < file_mtime(source):
            return round_index
        round_index += 1

def source_annotation_for_round(cfg: RefinementConfig, round_index: int) -> Path:
    if round_index <= 1:
        return base_annotation_path(cfg)
    previous = refinement_paths_for_round(cfg, round_index - 1)["annotation"]
    if not previous.is_file():
        return previous
    return previous


def base_annotation_path(cfg: RefinementConfig) -> Path:
    return (
        cfg.annotation_root.resolve()
        / cfg.run_id
        / cfg.layer
        / f"F{int(cfg.feature_id):06d}"
        / f"{cfg.provider_label}_annotation.json"
    )


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def build_test_packet(*, cfg: RefinementConfig, annotation_packet: dict[str, Any], test_cases: list[Any]) -> dict[str, Any]:
    model_id = model_id_for_run(cfg.run_id, cfg.db_path)
    results = evaluate_test_cases(
        run_id=cfg.run_id,
        layer=cfg.layer,
        feature_id=int(cfg.feature_id),
        model_id=model_id,
        test_cases=test_cases,
        top_k=int(cfg.top_k),
        threshold=float(cfg.activation_relative_threshold),
        device=cfg.device,
        dtype=cfg.dtype,
        db_path=cfg.db_path,
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round": int(cfg.round_index),
        "feature": f"{cfg.run_id}/{cfg.layer}/F{cfg.feature_id}",
        "source_annotation": str(annotation_path(cfg).resolve()),
        "label_before_tests": annotation_packet.get("annotation"),
        "activation_relative_threshold": float(cfg.activation_relative_threshold),
        "test_results": results,
    }


def evaluate_test_cases(
    *,
    run_id: str,
    layer: str,
    feature_id: int,
    model_id: str,
    test_cases: list[Any],
    top_k: int,
    threshold: float,
    device: str,
    dtype: str,
    db_path: Path,
) -> list[dict[str, Any]]:
    runtime = load_runtime(model_id, device, dtype)
    bundle = load_feature_bundle(run_id, layer, str(db_path))
    feature_directions = bundle.feature_directions.to(runtime.device)
    mean = bundle.mean.to(runtime.device)
    target_feature = int(feature_id)
    rows = []
    for index, case in enumerate(test_cases):
        if not isinstance(case, dict):
            continue
        text = str(case.get("text") or "")
        expected = str(case.get("expected") or "ambiguous").lower()
        reason = str(case.get("reason") or "")
        result = {
            "case_index": index,
            "text": text,
            "expected": expected,
            "reason": reason,
        }
        if not text:
            result["error"] = "missing text"
            rows.append(result)
            continue
        encoded = runtime.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        scored_token_indices = [int(i) for i, (start, end) in enumerate(offsets) if int(end) > int(start)]
        if not scored_token_indices:
            result["error"] = "prompt did not align to any non-special token"
            rows.append(result)
            continue
        token_rows, best_row, max_relative = score_encoded_prompt(
            runtime=runtime,
            encoded=encoded,
            layer=layer,
            mean=mean,
            feature_directions=feature_directions,
            target_feature=target_feature,
            norm_eps=float(bundle.norm_eps),
            top_k=int(top_k),
            scored_token_indices=scored_token_indices,
            model_index_shift=0,
        )
        spacer_token_id = first_token_spacer_token_id(
            runtime.tokenizer,
            offsets=offsets,
            scored_token_indices=scored_token_indices,
            best_row=best_row,
        )
        if spacer_token_id is not None:
            token_rows, best_row, max_relative = score_encoded_prompt(
                runtime=runtime,
                encoded=prepend_hidden_token(encoded, token_id=int(spacer_token_id)),
                layer=layer,
                mean=mean,
                feature_directions=feature_directions,
                target_feature=target_feature,
                norm_eps=float(bundle.norm_eps),
                top_k=int(top_k),
                scored_token_indices=scored_token_indices,
                model_index_shift=1,
            )
            result["hidden_scoring_prefix"] = {
                "token_id": int(spacer_token_id),
                "token_text": runtime.tokenizer.decode([int(spacer_token_id)], clean_up_tokenization_spaces=False),
                "reason": "strongest activating token was the first scored token; hidden spacer reduces first-token position artifacts",
            }
        observed = "activate" if max_relative >= float(threshold) else "not_activate"
        result.update(
            {
                "scoring_mode": "whole_prompt_max",
                "observed": observed,
                "passes_expected": expected == "ambiguous" or expected == observed,
                "max_feature_relative": max_relative,
                "best_token": best_row,
                "scored_tokens": token_rows,
            }
        )
        rows.append(result)
    return rows


def score_encoded_prompt(
    *,
    runtime: Any,
    encoded: Any,
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    target_feature: int,
    norm_eps: float,
    top_k: int,
    scored_token_indices: list[int],
    model_index_shift: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    inputs = {key: value.to(runtime.device) for key, value in encoded.items()}
    model_scored_token_indices = [int(i) + int(model_index_shift) for i in scored_token_indices]
    hidden_states = hidden_states_for_layer(runtime.model, layer, inputs)
    normalized = hidden_states / torch.linalg.vector_norm(hidden_states, dim=1, keepdim=True).clamp_min(float(norm_eps))
    activations = torch.relu((normalized - mean) @ feature_directions.T).detach().cpu()
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    token_rows = []
    for original_token_index, model_token_index in zip(scored_token_indices, model_scored_token_indices, strict=True):
        values = activations[int(model_token_index)]
        feature_value = float(values[int(target_feature)].item())
        top_value = float(torch.max(values).item())
        rank = 1 + int((values > values[int(target_feature)]).sum().item()) if feature_value > 0 else None
        relative = 0.0 if top_value <= 0 else feature_value / top_value
        top_values, top_indices = torch.topk(values, k=min(int(top_k), int(values.numel())))
        token_rows.append(
            {
                "token_index": int(original_token_index),
                "model_token_index": int(model_token_index),
                "token_id": int(input_ids[int(model_token_index)]),
                "token_text": runtime.tokenizer.decode([int(input_ids[int(model_token_index)])], clean_up_tokenization_spaces=False),
                "feature_activation": feature_value,
                "feature_relative": relative,
                "feature_rank": rank,
                "top_activation": top_value,
                "top_features": [
                    {"feature_id": int(feature_id), "activation": float(value)}
                    for feature_id, value in zip(top_indices.tolist(), top_values.tolist(), strict=True)
                ],
            }
        )
    max_relative = max((float(row["feature_relative"]) for row in token_rows), default=0.0)
    best_row = max(token_rows, key=lambda row: float(row["feature_relative"]), default=None)
    return token_rows, best_row, max_relative


def first_token_spacer_token_id(
    tokenizer: Any,
    *,
    offsets: list[list[int]],
    scored_token_indices: list[int],
    best_row: dict[str, Any] | None,
) -> int | None:
    if not scored_token_indices:
        return None
    first_index = int(scored_token_indices[0])
    if not isinstance(best_row, dict) or int(best_row.get("token_index", -1)) != first_index:
        return None
    if first_index < 0 or first_index >= len(offsets):
        return None
    start, end = offsets[first_index]
    if int(start) != 0 or int(end) <= int(start):
        return None
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return int(bos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return int(eos_token_id) if eos_token_id is not None else None


def prepend_hidden_token(encoded: Any, *, token_id: int) -> dict[str, torch.Tensor]:
    input_ids = encoded["input_ids"]
    prefix_ids = torch.full((input_ids.shape[0], 1), int(token_id), dtype=input_ids.dtype, device=input_ids.device)
    updated = {key: value for key, value in encoded.items()}
    updated["input_ids"] = torch.cat([prefix_ids, input_ids], dim=1)
    for key, value in encoded.items():
        if key == "input_ids" or not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2 or value.shape != input_ids.shape:
            continue
        fill_value = 1 if key == "attention_mask" else 0
        prefix = torch.full((value.shape[0], 1), fill_value, dtype=value.dtype, device=value.device)
        updated[key] = torch.cat([prefix, value], dim=1)
    return updated


def build_refinement_request(*, cfg: RefinementConfig, test_packet: dict[str, Any]) -> dict[str, Any]:
    messages = build_refinement_messages(cfg=cfg, test_packet=test_packet)
    if cfg.provider == "ollama":
        return {
            "model": cfg.model,
            "messages": messages,
            "think": False,
            "stream": False,
            "format": "json",
            "options": {"num_predict": int(cfg.max_tokens)},
        }
    payload = {
        "model": cfg.model,
        "max_tokens": int(cfg.max_tokens),
        "messages": messages,
    }
    if cfg.provider == "tinker-sdk":
        return payload
    if cfg.provider == "deepseek":
        payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
        return payload
    apply_openai_reasoning_controls(payload, provider=cfg.provider, model=cfg.model)
    payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ica_lens_feature_annotation_refinement",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            },
    }
    return payload


def build_refinement_messages(*, cfg: RefinementConfig, test_packet: dict[str, Any]) -> list[dict[str, str]]:
    return build_single_packet_refinement_messages(
        original_evidence=load_original_evidence(cfg),
        current_annotation=test_packet.get("label_before_tests"),
        previous_refinements=load_previous_refinement_rounds(cfg),
        current_test_packet=test_packet,
        round_index=int(cfg.round_index),
    )


def build_single_packet_refinement_messages(
    *,
    original_evidence: dict[str, Any],
    current_annotation: Any,
    previous_refinements: list[dict[str, Any]],
    current_test_packet: dict[str, Any],
    round_index: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": stable_json(
                {
                    "instruction": "Use this style guide only for wording and confidence calibration.",
                    "style_guide": build_annotation_style_guide(),
                }
            ),
        },
        {
            "role": "user",
            "content": stable_json(
                {
                    "instruction": REFINEMENT_TEST_INSTRUCTION,
                    "response_format": ANNOTATION_RESPONSE_FORMAT,
                    "refinement_evidence_json": {
                        "original_evidence": compact_evidence_for_prompt(original_evidence),
                        "annotator_annotation": annotation_without_test_cases(current_annotation),
                        "previous_refinement_history": refinement_history_for_prompt(previous_refinements),
                        "additional_test_results": test_packet_for_prompt(current_test_packet),
                    },
                }
            ),
        },
    ]


def annotation_without_test_cases(annotation: Any) -> dict[str, Any]:
    if not isinstance(annotation, dict):
        return {}
    return {
        key: annotation.get(key)
        for key in ("reasoning", "label", "simple_label", "description", "confidence")
        if key in annotation
    }


def refinement_history_for_prompt(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prior in history:
        row: dict[str, Any] = {"round": prior.get("round")}
        if isinstance(prior.get("annotation"), dict):
            annotation = prior["annotation"]
            row["annotation"] = {
                key: annotation.get(key)
                for key in ("reasoning", "label", "simple_label", "description", "confidence")
                if key in annotation
            }
        if isinstance(prior.get("test_packet"), dict):
            row["additional_test_results"] = test_packet_for_prompt(prior["test_packet"])
        rows.append(row)
    return rows


def test_packet_for_prompt(test_packet: dict[str, Any]) -> dict[str, Any]:
    prompt_packet = {
        key: value
        for key, value in test_packet.items()
        if key
        not in {
            "activation_relative_threshold",
            "created_at",
            "feature",
            "label_before_tests",
            "source_annotation",
            "test_results",
        }
    }
    prompt_packet["result_format_note"] = (
        f"marked_result appends [rank] only after scored tokens where this feature ranked in the top {MARKED_TEST_RESULT_MAX_RANK}; "
        "unmarked tokens did not strongly activate this feature."
    )
    prompt_packet["test_results"] = [
        summarize_test_result_for_prompt(result)
        for result in (test_packet.get("test_results") or [])
        if isinstance(result, dict)
    ]
    return prompt_packet


def summarize_test_result_for_prompt(result: dict[str, Any]) -> dict[str, Any]:
    marked_result = marked_text_from_scored_tokens(result)
    row = {
        "case_index": result.get("case_index"),
        "test_text": result.get("text"),
        "what_is_tested": result.get("reason"),
        "marked_result": marked_result,
        "is_feature_activated": marked_result != "",
    }
    if isinstance(result.get("hidden_scoring_prefix"), dict):
        row["scoring_note"] = "A hidden neutral spacer token was added before scoring and omitted from marked_result."
    if result.get("error"):
        row["error"] = result.get("error")
    return row


def marked_text_from_scored_tokens(result: dict[str, Any]) -> str:
    scored_tokens = result.get("scored_tokens")
    if not isinstance(scored_tokens, list) or not scored_tokens:
        return ""
    chunks = []
    has_mark = False
    for token in scored_tokens:
        if not isinstance(token, dict):
            continue
        text = str(token.get("token_text") or "")
        rank = token.get("feature_rank")
        activation = float(token.get("feature_activation") or 0.0)
        if isinstance(rank, int) and rank <= MARKED_TEST_RESULT_MAX_RANK and activation > 0:
            text = f"{text}[{rank}]"
            has_mark = True
        chunks.append(text)
    if not has_mark:
        return ""
    return "".join(chunks)


def round_float(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def write_readable_request(path: Path, request_payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_readable_request(request_payload), encoding="utf-8")


def write_request_debug_bundle(output_dir: Path, request_payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = request_payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    manifest = {
        "model": request_payload.get("model"),
        "message_count": len(messages),
        "request_keys": sorted(str(key) for key in request_payload.keys()),
        "message_files": [],
    }
    write_json(
        output_dir / "request_options.json",
        {key: value for key, value in request_payload.items() if key != "messages"},
    )
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown") if isinstance(message, dict) else "unknown"
        content = str(message.get("content") or "") if isinstance(message, dict) else str(message)
        parsed = parse_json_content(content)
        filename = f"message_{index:02d}_{safe_filename(role)}.json"
        message_packet: dict[str, Any] = {
            "index": index,
            "role": role,
            "content_characters": len(content),
        }
        if parsed is None:
            message_packet["content_text"] = content
        else:
            message_packet["content_json"] = parsed
        write_json(output_dir / filename, message_packet)
        manifest["message_files"].append(
            {
                "index": index,
                "role": role,
                "file": filename,
                "content_characters": len(content),
                "content_type": "json" if parsed is not None else "text",
            }
        )
    write_json(output_dir / "manifest.json", manifest)


def format_readable_request(request_payload: dict[str, Any]) -> str:
    model = str(request_payload.get("model") or "")
    lines = [
        "# Annotation Refinement Request",
        "",
        f"Model: `{model}`" if model else "Model: unknown",
        "",
    ]
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        lines.append("No messages found.")
        return "\n".join(lines).rstrip() + "\n"
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "")
        lines.extend(
            [
                f"## Message {index + 1}: {role}",
                "",
                format_readable_message_content(content),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def format_readable_message_content(content: str) -> str:
    parsed = parse_json_content(content)
    if parsed is None:
        return content
    return json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=False)


def parse_json_content(content: str) -> Any | None:
    try:
        return json.loads(content)
    except Exception:
        return None


def safe_filename(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned or "message"


def evidence_path(cfg: RefinementConfig) -> Path:
    path = (
        cfg.evidence_root.resolve()
        / cfg.run_id
        / cfg.layer
        / f"F{int(cfg.feature_id):06d}"
        / COMPACT_EVIDENCE_FILENAME
    )
    if path.is_file():
        return path
    return path.with_name(LEGACY_COMPACT_EVIDENCE_FILENAME)


def load_original_evidence(cfg: RefinementConfig) -> dict[str, Any]:
    path = evidence_path(cfg)
    if not path.is_file():
        return {"missing": True, "path": str(path)}
    evidence = read_json(path)
    return evidence if isinstance(evidence, dict) else {"invalid": True, "path": str(path)}


def load_initial_annotation(cfg: RefinementConfig) -> dict[str, Any]:
    path = (
        cfg.annotation_root.resolve()
        / cfg.run_id
        / cfg.layer
        / f"F{int(cfg.feature_id):06d}"
        / f"{cfg.provider_label}_annotation.json"
    )
    if not path.is_file():
        return {}
    packet = read_json(path)
    annotation = packet.get("annotation") if isinstance(packet, dict) else None
    return annotation if isinstance(annotation, dict) else {}


def load_previous_refinement_rounds(cfg: RefinementConfig) -> list[dict[str, Any]]:
    history = []
    for round_index in range(1, int(cfg.round_index)):
        paths = refinement_paths_for_round(cfg, round_index)
        row: dict[str, Any] = {"round": round_index}
        if paths["tests"].is_file():
            row["test_packet"] = read_json(paths["tests"])
        if paths["annotation"].is_file():
            packet = read_json(paths["annotation"])
            row["annotation"] = packet.get("annotation") if isinstance(packet, dict) else None
        if row.keys() != {"round"}:
            history.append(row)
    return history


def annotation_path(cfg: RefinementConfig) -> Path:
    if cfg.source_annotation is not None:
        return cfg.source_annotation.resolve()
    return source_annotation_for_round(cfg, int(cfg.round_index or 1))


def refinement_paths(cfg: RefinementConfig) -> dict[str, Path]:
    return refinement_paths_for_round(cfg, int(cfg.round_index))


def refinement_paths_for_round(cfg: RefinementConfig, round_index: int) -> dict[str, Path]:
    output_dir = cfg.refinement_root.resolve() / cfg.run_id / cfg.layer / f"F{int(cfg.feature_id):06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.provider_label}_round{int(round_index):02d}"
    return {
        "tests": output_dir / f"{stem}_tests.json",
        "stop": output_dir / f"{stem}_stop.json",
        "request": output_dir / f"{stem}_request_preview.json",
        "request_debug_dir": output_dir / f"{stem}_request_debug",
        "raw_response": output_dir / f"{stem}_raw_response.json",
        "repair_raw_response": output_dir / f"{stem}_repair_raw_response.json",
        "annotation": output_dir / f"{stem}_annotation.json",
    }


def model_id_for_run(run_id: str, db_path: Path) -> str:
    with connect(db_path) as conn:
        row = conn.execute("SELECT model_id FROM model_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown run id: {run_id}")
    return str(row["model_id"])


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def upsert_refinement_round(
    *,
    cfg: RefinementConfig,
    paths: dict[str, Path],
    test_packet: dict[str, Any],
    annotation_packet: dict[str, Any] | None,
) -> None:
    annotation = annotation_packet.get("annotation") if isinstance(annotation_packet, dict) else None
    if not isinstance(annotation, dict):
        annotation = None
    label_before = test_packet.get("label_before_tests")
    created_at = ""
    if isinstance(annotation_packet, dict):
        created_at = str(annotation_packet.get("created_at") or "")
    if not created_at:
        created_at = str(test_packet.get("created_at") or datetime.now(timezone.utc).isoformat())
    with connect(cfg.db_path) as conn:
        _ensure_refinement_schema(conn)
        conn.execute(
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
            (
                cfg.run_id,
                cfg.layer,
                int(cfg.feature_id),
                cfg.provider_label,
                int(cfg.round_index),
                cfg.provider,
                cfg.model,
                created_at,
                str(annotation_path(cfg).resolve()),
                str(paths["tests"].resolve()),
                str(paths["request"].resolve()),
                str(paths["raw_response"].resolve()) if paths["raw_response"].is_file() else None,
                str(paths["annotation"].resolve()) if paths["annotation"].is_file() else None,
                json.dumps(label_before, ensure_ascii=False, sort_keys=True) if label_before is not None else None,
                json.dumps(test_packet.get("test_results") or [], ensure_ascii=False, sort_keys=True),
                str(annotation.get("label") or "") if annotation else None,
                str(annotation.get("simple_label") or "") if annotation else None,
                str(annotation.get("description") or "") if annotation else None,
                str(annotation.get("reasoning") or "") if annotation else None,
                str(annotation.get("confidence") or "") if annotation else None,
                json.dumps(annotation.get("test_cases") or [], ensure_ascii=False, sort_keys=True) if annotation else None,
            ),
        )
        conn.commit()


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
