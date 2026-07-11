from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..fit.inputs import activation_manifest_path, load_activation_config
from ..io_utils import load_json
from ..layers import layer_index, layer_shard_records
from ..model_runtime import hidden_states_for_layer, load_runtime
from ..paths import DEFAULT_FEATURE_INDEX, V9_ROOT
from ..saes.counterparts import CONFIG_PATHS, SAE_COUNTERPARTS, SaeCounterpart
from ..saes.loaders import load_counterpart_lightweight_sae
from .auto_annotate import (
    ANNOTATION_SCHEMA,
    DEFAULT_MI_BASE_URL,
    DEFAULT_MI_TOKEN_PATH,
    DEFAULT_OLLAMA_BASE_URL,
    apply_openai_reasoning_controls,
    call_annotation_provider,
    default_base_url_for_provider,
    default_model_for_provider,
    extract_chat_completion_text,
    load_api_key_for_provider,
    parse_annotation,
    repair_annotation_output,
    resolve_model,
    warn_single_token_test_cases,
    write_json,
)
from .evidence import (
    COMPACT_EVIDENCE_FILENAME,
    DEBUGGING_FILENAME,
    DEFAULT_SEMANTIC_EXAMPLE_COUNT,
    DEFAULT_SEMANTIC_WINDOW_RADIUS,
    EVIDENCE_FILENAME,
    LEGACY_COMPACT_EVIDENCE_FILENAME,
    MAX_LARGEST_JUMP_REFINEMENT_CALLS,
    REQUESTED_RESPONSE_FORMAT,
    _added_left_text,
    _coarse_context_lengths,
    _compact_effective_receptive_field,
    _debug_effective_receptive_field,
    _filter_samples_by_score_threshold,
    _key_tested_context_lengths,
    _largest_observed_relative_score_jump,
    _load_dataset_texts,
    _load_optional_tensor,
    _marked_rank_text,
    _marked_token_rows,
    _max_replay_context_length,
    _monotonic_bound_satisfied,
    _none_if_negative,
    _read_json,
    _refine_largest_observed_relative_score_jump,
    _replay_prefix_token_ids,
    _round_float,
    _single_token_replay_pad_token_id,
    _single_token_replay_padding_ids,
    _sorted_tested,
)
from .refinement import (
    MARKED_TEST_RESULT_MAX_RANK,
    build_single_packet_refinement_messages,
    default_model_for_provider as _unused_default_model_for_provider,  # noqa: F401
    marked_text_from_scored_tokens,
    parse_json_content,
    safe_filename,
    test_packet_has_marked_activation,
)


DEFAULT_SAE_ROOT = V9_ROOT / "results" / "auto_annotation_sae"
DEFAULT_SAE_EVIDENCE_ROOT = DEFAULT_SAE_ROOT / "evidence"
DEFAULT_SAE_ANNOTATION_ROOT = DEFAULT_SAE_ROOT / "annotations"
DEFAULT_SAE_REFINEMENT_ROOT = DEFAULT_SAE_ROOT / "refinements"
DEFAULT_ACTIVATION_ROOT = Path("/home/liusida/research/ICA-paper/data/activations_v9")
ANNOTATION_FILENAME_RE = __import__("re").compile(r"^(?P<label>.+)_annotation\.json$")
REFINEMENT_FILENAME_RE = __import__("re").compile(r"^(?P<label>.+)_round(?P<round>\d+)_annotation\.json$")
FEATURE_DIR_RE = __import__("re").compile(r"^F(?P<feature_id>\d+)$")


@dataclass(frozen=True)
class SaeEvidenceConfig:
    model: str
    layer: str
    feature_id: int
    output_root: Path = DEFAULT_SAE_EVIDENCE_ROOT
    activation_root: Path = DEFAULT_ACTIVATION_ROOT
    activation_manifest: Path | None = None
    token_budget: int | None = None
    top_k: int = 40
    examples: int = 10
    example_score_threshold_frac: float | None = 0.15
    batch_size: int = 512
    device: str = "cuda"
    dtype: str = "float32"
    force: bool = False


@dataclass(frozen=True)
class SaeRefinementConfig:
    model_name: str
    layer: str
    feature_id: int
    provider_label: str
    source_annotation: Path | None = None
    annotation_root: Path = DEFAULT_SAE_ANNOTATION_ROOT
    evidence_root: Path = DEFAULT_SAE_EVIDENCE_ROOT
    refinement_root: Path = DEFAULT_SAE_REFINEMENT_ROOT
    db_path: Path = DEFAULT_FEATURE_INDEX
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


def build_sae_feature_evidence(cfg: SaeEvidenceConfig) -> Path:
    model_name = _normalize_model_name(cfg.model)
    counterpart = SAE_COUNTERPARTS[model_name]
    layer_i = _require_layer_index(cfg.layer)
    output_dir = cfg.output_root.resolve() / model_name / cfg.layer / f"F{int(cfg.feature_id):06d}"
    evidence_path = output_dir / EVIDENCE_FILENAME
    debugging_path = output_dir / DEBUGGING_FILENAME
    if evidence_path.exists() and debugging_path.exists() and not cfg.force:
        return evidence_path

    activation_cfg = load_activation_config(CONFIG_PATHS[model_name])
    token_budget = int(cfg.token_budget or activation_cfg["dataset"]["token_budget"])
    manifest_path = activation_manifest_path(
        explicit=cfg.activation_manifest,
        activation_root=cfg.activation_root,
        activation_cfg=activation_cfg,
        token_budget=token_budget,
    )
    manifest = load_json(manifest_path)
    activation_dir = manifest_path.parent
    model_id = str((manifest.get("model") or {}).get("id") or activation_cfg["model"]["id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    device = torch.device(cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _torch_dtype(cfg.dtype, manifest)
    sae_name, sae = load_counterpart_lightweight_sae(
        counterpart=counterpart,
        layer_index=layer_i,
        device=str(device),
        dtype=dtype,
    )
    d_sae = int(sae.cfg.d_sae)
    feature_id = int(cfg.feature_id)
    if feature_id < 0 or feature_id >= d_sae:
        raise IndexError(f"SAE feature_id {feature_id} outside [0, {d_sae - 1}]")

    samples = _scan_sae_feature_samples(
        activation_dir=activation_dir,
        activation_manifest=manifest,
        layer=cfg.layer,
        sae=sae,
        feature_id=feature_id,
        top_k=int(cfg.top_k),
        batch_size=int(cfg.batch_size),
        tokenizer=tokenizer,
    )
    filtered = _filter_samples_by_score_threshold(samples, score_threshold_frac=cfg.example_score_threshold_frac)
    selected = _select_distinct_doc_samples(filtered, count=int(cfg.examples))
    _enrich_samples_with_text(selected, manifest=manifest, tokenizer=tokenizer)
    runtime = load_runtime(model_id, str(device), "auto")
    prefix_token_ids = _replay_prefix_token_ids(tokenizer)
    single_token_replay_pad_token_id = _single_token_replay_pad_token_id(tokenizer)
    _add_sae_effective_receptive_field(
        selected,
        runtime=runtime,
        sae=sae,
        layer=cfg.layer,
        feature_id=feature_id,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=device,
    )
    evidence_packet, debugging_packet = _sae_evidence_packets(
        model_name=model_name,
        layer=cfg.layer,
        feature_id=feature_id,
        counterpart=counterpart,
        sae_name=sae_name,
        d_sae=d_sae,
        samples=selected,
        settings={
            "candidate_pool_size": int(cfg.top_k),
            "examples": int(cfg.examples),
            "example_score_threshold_frac": cfg.example_score_threshold_frac,
            "replay_prefix_token_count": len(prefix_token_ids),
            "single_token_replay_pad_token_id": single_token_replay_pad_token_id,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(evidence_path, evidence_packet)
    write_json(debugging_path, debugging_packet)
    return evidence_path


def _scan_sae_feature_samples(
    *,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    sae: Any,
    feature_id: int,
    top_k: int,
    batch_size: int,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    global_offset = 0
    for shard in tqdm(layer_shard_records(activation_manifest, layer), desc=f"scan SAE {layer}", unit="shard", dynamic_ncols=True):
        layer_path = shard["layers"].get(layer)
        if not isinstance(layer_path, str):
            raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
        shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
        input_ids = _load_optional_tensor(activation_dir, shard.get("input_ids"))
        doc_ids = _load_optional_tensor(activation_dir, shard.get("doc_ids"))
        positions = _load_optional_tensor(activation_dir, shard.get("positions"))
        shard_best: list[dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, int(shard_tensor.shape[0]), batch_size):
                batch = shard_tensor[start : start + batch_size].to(device=sae.device, dtype=sae.dtype, non_blocking=True)
                acts = sae.encode(batch).detach()
                feature_values = acts[:, int(feature_id)]
                top_values, top_indices = torch.max(acts, dim=1)
                k = min(int(top_k), int(feature_values.numel()))
                values, indices = torch.topk(feature_values, k=k, largest=True)
                for score, index in zip(values.cpu().tolist(), indices.cpu().tolist(), strict=True):
                    if float(score) <= 0:
                        continue
                    local_index = int(start + int(index))
                    token_id = int(input_ids[local_index]) if input_ids is not None else None
                    top_activation = float(top_values[int(index)].cpu().item())
                    top_feature_id = int(top_indices[int(index)].cpu().item())
                    shard_best.append(
                        {
                            "activation": float(score),
                            "relative_activation": float(score) / top_activation if top_activation > 0 else None,
                            "top_feature_id_at_position": top_feature_id,
                            "top_feature_activation_at_position": top_activation,
                            "shard_index": int(shard["index"]),
                            "local_index": local_index,
                            "global_index": int(global_offset + local_index),
                            "token_id": token_id,
                            "token": tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None,
                            "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False) if token_id is not None else None,
                            "doc_id": int(doc_ids[local_index]) if doc_ids is not None else None,
                            "position": int(positions[local_index]) if positions is not None else None,
                        }
                    )
                del batch, acts
        candidates.extend(shard_best)
        candidates = sorted(candidates, key=lambda row: float(row["activation"]), reverse=True)[:top_k]
        global_offset += int(shard.get("tokens", shard_tensor.shape[0]))
    return candidates


def _select_distinct_doc_samples(samples: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_doc_ids: set[int] = set()
    for sample in samples:
        doc_id = sample.get("doc_id")
        if doc_id is not None and int(doc_id) in seen_doc_ids:
            continue
        selected.append(sample)
        if doc_id is not None:
            seen_doc_ids.add(int(doc_id))
        if len(selected) >= int(count):
            break
    return selected


def _enrich_samples_with_text(samples: list[dict[str, Any]], *, manifest: dict[str, Any], tokenizer: Any) -> None:
    doc_ids = sorted({int(sample["doc_id"]) for sample in samples if sample.get("doc_id") is not None})
    if not doc_ids:
        return
    texts = _load_dataset_texts(manifest=manifest, doc_ids=doc_ids)
    context_length = int((manifest.get("capture") or {}).get("context_length") or 1024)
    for sample in samples:
        doc_id = sample.get("doc_id")
        position = sample.get("position")
        if doc_id is None or position is None:
            continue
        document_text = texts.get(int(doc_id))
        if document_text is None:
            continue
        input_ids = list(tokenizer(document_text, truncation=True, max_length=context_length)["input_ids"])
        pos = int(position)
        if pos < 0 or pos >= len(input_ids):
            continue
        sample["_context_to_target_ids"] = [int(token_id) for token_id in input_ids[: pos + 1]]
        window_left = max(0, pos - DEFAULT_SEMANTIC_WINDOW_RADIUS)
        window_right = min(len(input_ids), pos + DEFAULT_SEMANTIC_WINDOW_RADIUS + 1)
        sample["_semantic_window_ids"] = [int(token_id) for token_id in input_ids[window_left:window_right]]
        sample["_semantic_window_target_offset"] = int(pos - window_left)
        right_hint_ids = input_ids[pos + 1 : min(len(input_ids), pos + 5)]
        right_hint_text = tokenizer.decode(right_hint_ids, clean_up_tokenization_spaces=False)
        if right_hint_text and "\ufffd" not in right_hint_text:
            sample["next_text_hint"] = right_hint_text


def _add_sae_effective_receptive_field(
    samples: list[dict[str, Any]],
    *,
    runtime: Any,
    sae: Any,
    layer: str,
    feature_id: int,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> None:
    for sample in tqdm(samples, desc=f"estimated SAE ERF {layer}", unit="example", dynamic_ncols=True):
        ids = sample.get("_context_to_target_ids")
        if not isinstance(ids, list) or not ids:
            sample["effective_receptive_field"] = {"available": False, "reason": "missing_context_to_target_ids"}
            continue
        full_activation = float(sample.get("activation") or 0.0)
        sample["effective_receptive_field"] = _build_sae_effective_receptive_field(
            runtime=runtime,
            sae=sae,
            ids=[int(token_id) for token_id in ids],
            layer=layer,
            feature_id=feature_id,
            full_activation=full_activation,
            prefix_token_ids=prefix_token_ids,
            single_token_replay_pad_token_id=single_token_replay_pad_token_id,
            device=device,
        )
        _add_sae_semantic_activation_window(
            sample,
            runtime=runtime,
            sae=sae,
            layer=layer,
            feature_id=feature_id,
            prefix_token_ids=prefix_token_ids,
            device=device,
        )


def _build_sae_effective_receptive_field(
    *,
    runtime: Any,
    sae: Any,
    ids: list[int],
    layer: str,
    feature_id: int,
    full_activation: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> dict[str, Any]:
    tested: dict[int, float] = {0: 0.0}
    max_len = min(len(ids), _max_replay_context_length(runtime.model, prefix_token_ids=prefix_token_ids))
    if max_len <= 0:
        return {"available": False, "reason": "no_context_fits_model_window_after_replay_prefix"}

    def relative_score(length: int) -> float:
        length = int(max(1, min(length, max_len)))
        if length not in tested:
            tested[length] = _replay_sae_feature_relative_score(
                runtime=runtime,
                sae=sae,
                input_ids=ids[-length:],
                layer=layer,
                feature_id=feature_id,
                prefix_token_ids=prefix_token_ids,
                single_token_replay_pad_token_id=single_token_replay_pad_token_id,
                device=device,
            )
        return tested[length]

    for length in _coarse_context_lengths(max_len):
        relative_score(length)
        if _monotonic_bound_satisfied(tested, max_length=max_len):
            break
    for _ in range(MAX_LARGEST_JUMP_REFINEMENT_CALLS):
        before = set(tested)
        _refine_largest_observed_relative_score_jump(tested=tested, score_fn=relative_score, max_length=max_len)
        if set(tested) == before:
            break
    rows = _sorted_tested(tested, tokenizer=runtime.tokenizer, ids=ids)
    jump = _largest_observed_relative_score_jump(rows)
    _add_sae_rank_trace_to_largest_jump(
        jump,
        runtime=runtime,
        sae=sae,
        ids=ids,
        layer=layer,
        feature_id=feature_id,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=device,
    )
    return {
        "available": True,
        "definition": "right edge of the largest observed positive relative-score jump in left-context replay",
        "estimated_effective_receptive_field_length": int(jump["to_context_length"]) if isinstance(jump, dict) and jump.get("to_context_length") else None,
        "full_activation": _round_float(full_activation),
        "available_prefix_context_length": len(ids),
        "replay_prefix_length": len(prefix_token_ids),
        "full_prefix_context_length": int(max_len),
        "full_prefix_relative_score": _round_float(relative_score(max_len)),
        "largest_observed_relative_score_jump": jump,
        "tested_context_lengths": rows,
    }


def _replay_sae_feature_relative_score(
    *,
    runtime: Any,
    sae: Any,
    input_ids: list[int],
    layer: str,
    feature_id: int,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> float:
    replay_ids = [*prefix_token_ids, *_single_token_replay_padding_ids(input_ids=input_ids, pad_token_id=single_token_replay_pad_token_id), *input_ids]
    token_tensor = torch.tensor([replay_ids], dtype=torch.long, device=device)
    hidden = hidden_states_for_layer(runtime.model, layer, {"input_ids": token_tensor})[-1].detach()
    with torch.no_grad():
        acts = sae.encode(hidden.reshape(1, -1).to(device=sae.device, dtype=sae.dtype)).squeeze(0)
    top = float(torch.max(acts).detach().cpu().item())
    value = float(acts[int(feature_id)].detach().cpu().item())
    return 0.0 if top <= 0 else value / top


def _replay_sae_feature_token_ranks(
    *,
    runtime: Any,
    sae: Any,
    input_ids: list[int],
    layer: str,
    feature_id: int,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> tuple[list[int | None], list[float]]:
    replay_ids = [*prefix_token_ids, *_single_token_replay_padding_ids(input_ids=input_ids, pad_token_id=single_token_replay_pad_token_id), *input_ids]
    token_tensor = torch.tensor([replay_ids], dtype=torch.long, device=device)
    hidden_start = len(prefix_token_ids) + len(_single_token_replay_padding_ids(input_ids=input_ids, pad_token_id=single_token_replay_pad_token_id))
    hidden = hidden_states_for_layer(runtime.model, layer, {"input_ids": token_tensor})[hidden_start:].detach()
    with torch.no_grad():
        acts = sae.encode(hidden.to(device=sae.device, dtype=sae.dtype)).detach().cpu()
    feature_values = acts[:, int(feature_id)]
    top_values = acts.max(dim=1).values.clamp_min(1e-12)
    ranks = 1 + (acts > feature_values[:, None]).sum(dim=1)
    ranks = torch.where(feature_values > 0, ranks, torch.zeros_like(ranks))
    return [int(rank) if int(rank) > 0 else None for rank in ranks.tolist()], [float(v) for v in (feature_values / top_values).tolist()]


def _add_sae_rank_trace_to_largest_jump(
    jump: dict[str, Any] | None,
    *,
    runtime: Any,
    sae: Any,
    ids: list[int],
    layer: str,
    feature_id: int,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
    max_rank: int = 10,
) -> None:
    if not isinstance(jump, dict):
        return
    to_length = int(jump.get("to_context_length") or 0)
    if to_length <= 0:
        return
    to_ids = ids[-to_length:]
    ranks, rel = _replay_sae_feature_token_ranks(
        runtime=runtime,
        sae=sae,
        input_ids=to_ids,
        layer=layer,
        feature_id=feature_id,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=device,
    )
    jump["feature_rank_trace"] = {
        "max_rank_shown": int(max_rank),
        "marked_replay_context_ending_at_target": _marked_rank_text(
            tokenizer=runtime.tokenizer,
            token_ids=to_ids,
            ranks=ranks,
            max_rank=max_rank,
        ),
    }


def _add_sae_semantic_activation_window(
    sample: dict[str, Any],
    *,
    runtime: Any,
    sae: Any,
    layer: str,
    feature_id: int,
    prefix_token_ids: list[int],
    device: torch.device,
    max_rank: int = 10,
) -> None:
    window_ids = sample.get("_semantic_window_ids")
    target_offset = sample.get("_semantic_window_target_offset")
    if not isinstance(window_ids, list) or target_offset is None:
        return
    ranks, rel = _replay_sae_feature_token_ranks(
        runtime=runtime,
        sae=sae,
        input_ids=[int(token_id) for token_id in window_ids],
        layer=layer,
        feature_id=feature_id,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=None,
        device=device,
    )
    sample["semantic_activation_window"] = {
        "target_offset_in_window": int(target_offset),
        "max_rank_shown": int(max_rank),
        "marked_window": _marked_rank_text(
            tokenizer=runtime.tokenizer,
            token_ids=[int(token_id) for token_id in window_ids],
            ranks=ranks,
            max_rank=max_rank,
            target_offset=int(target_offset),
        ),
        "marked_tokens": _marked_token_rows(
            tokenizer=runtime.tokenizer,
            token_ids=[int(token_id) for token_id in window_ids],
            ranks=ranks,
            relative_scores=rel,
            max_rank=max_rank,
        ),
    }


def _sae_evidence_packets(
    *,
    model_name: str,
    layer: str,
    feature_id: int,
    counterpart: SaeCounterpart,
    sae_name: str,
    d_sae: int,
    samples: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_packet: dict[str, Any] = {
        "feature_family": "sae_counterpart",
        "rank_marker_legend": {
            "syntax": "[n] after a token means this SAE feature ranked n-th among all SAE features at that token.",
            "max_rank_shown": 10,
            "target_marker": "[target] marks the selected target token in semantic_examples.",
            "right_context_note": "Right context is included only for readability; it is not causal evidence for activation.",
        },
        "semantic_examples": [_sae_semantic_sample(index, sample) for index, sample in enumerate(samples[:DEFAULT_SEMANTIC_EXAMPLE_COUNT])],
    }
    evidence_packet.update({
        "effective_receptive_field_legend": {
            "definition": (
                "Estimated effective receptive field (ERF) examples replay the target token with progressively more "
                "left context. Every sudden score jump is important evidence: compare the jump points across all "
                "examples and choose a label that explains their common pattern, not just one example."
            ),
        },
        "erf_examples": [_sae_compact_sample(index, sample) for index, sample in enumerate(samples)],
    })
    debugging_packet = {
        "debugging_type": "sae_feature_evidence_build_debug",
        "evidence_type": "sae_top_activating_examples",
        "feature": f"{model_name}/{layer}/F{feature_id:06d}",
        "feature_metadata": {
            "feature_id": int(feature_id),
            "d_sae": int(d_sae),
            "sae_name": sae_name,
            "sae_model_name": counterpart.sae_model_name,
            "sae_repo_id": counterpart.repo_id,
            "sae_activation": counterpart.activation,
            "sae_top_k": counterpart.top_k,
            "evidence_dead_no_examples": len(samples) == 0,
        },
        "model_facing_evidence_file": EVIDENCE_FILENAME,
        "example_selection_summary": {
            "candidate_pool_size": settings.get("candidate_pool_size"),
            "selected_examples": settings.get("examples"),
            "selection_rule": "Examples are selected from strongest target SAE feature activations, filtered by strength, sorted by activation, and deduplicated to at most one per source document.",
            "score_threshold_fraction_of_best": settings.get("example_score_threshold_frac"),
            "replay_prefix_token_count": settings.get("replay_prefix_token_count"),
            "single_token_replay_pad_token_id": settings.get("single_token_replay_pad_token_id"),
        },
        "builder_settings": settings,
        "legacy_instruction_removed_from_evidence": {
            "annotation_instruction": "Return only valid JSON with keys: reasoning, label, simple_label, description, confidence, test_cases.",
            "requested_response_format": REQUESTED_RESPONSE_FORMAT,
        },
        "full_erf_examples": [_sae_debug_erf_sample(index, sample) for index, sample in enumerate(samples)],
    }
    return evidence_packet, debugging_packet


def _sae_semantic_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": int(index),
        "target_token": sample.get("text") or sample.get("token"),
        "position": sample.get("position"),
        "relative_activation": _round_float(sample.get("relative_activation")),
        "marked_activation_window": sample.get("semantic_activation_window"),
    }


def _sae_compact_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    erf = _compact_effective_receptive_field(sample.get("effective_receptive_field"))
    row = {
        "target_token": sample.get("text") or sample.get("token"),
    }
    if isinstance(erf, dict):
        row.update(erf)
    else:
        row["effective_receptive_field"] = erf
    return row


def _sae_debug_erf_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": int(index),
        "activation": _round_float(sample.get("activation")),
        "relative_activation": _round_float(sample.get("relative_activation")),
        "target_token": sample.get("text") or sample.get("token"),
        "doc_id": sample.get("doc_id"),
        "position": sample.get("position"),
        "effective_receptive_field": _debug_effective_receptive_field(sample.get("effective_receptive_field")),
    }


def evaluate_sae_test_cases(
    *,
    model_name: str,
    layer: str,
    feature_id: int,
    test_cases: list[Any],
    top_k: int,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    counterpart = SAE_COUNTERPARTS[_normalize_model_name(model_name)]
    model_id = str(load_activation_config(CONFIG_PATHS[_normalize_model_name(model_name)])["model"]["id"])
    runtime = load_runtime(model_id, device, dtype)
    layer_i = _require_layer_index(layer)
    _, sae = load_counterpart_lightweight_sae(
        counterpart=counterpart,
        layer_index=layer_i,
        device=str(runtime.device),
        dtype=_runtime_sae_dtype(dtype),
    )
    rows = []
    for index, case in enumerate(test_cases):
        if not isinstance(case, dict):
            continue
        text = str(case.get("text") or "")
        result: dict[str, Any] = {
            "case_index": int(index),
            "text": text,
            "expected": str(case.get("expected") or "ambiguous").lower(),
            "reason": str(case.get("reason") or ""),
        }
        if not text:
            result["error"] = "missing text"
            rows.append(result)
            continue
        encoded = runtime.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        scored_token_indices = [int(i) for i, (start, end) in enumerate(offsets) if int(end) > int(start)]
        if not scored_token_indices:
            result["error"] = "prompt did not align to any non-special token"
            rows.append(result)
            continue
        token_rows, best_row, max_relative = score_sae_encoded_prompt(
            runtime=runtime,
            sae=sae,
            encoded=encoded,
            layer=layer,
            feature_id=int(feature_id),
            top_k=int(top_k),
            scored_token_indices=scored_token_indices,
            model_index_shift=0,
        )
        result.update(
            {
                "scoring_mode": "sae_whole_prompt_max",
                "max_feature_relative": max_relative,
                "best_token": best_row,
                "scored_tokens": token_rows,
            }
        )
        rows.append(result)
    return rows


def score_sae_encoded_prompt(
    *,
    runtime: Any,
    sae: Any,
    encoded: Any,
    layer: str,
    feature_id: int,
    top_k: int,
    scored_token_indices: list[int],
    model_index_shift: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    inputs = {key: value.to(runtime.device) for key, value in encoded.items()}
    model_scored_token_indices = [int(i) + int(model_index_shift) for i in scored_token_indices]
    hidden = hidden_states_for_layer(runtime.model, layer, inputs)
    with torch.no_grad():
        acts = sae.encode(hidden.to(device=sae.device, dtype=sae.dtype)).detach().cpu()
    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    token_rows = []
    for original_token_index, model_token_index in zip(scored_token_indices, model_scored_token_indices, strict=True):
        values = acts[int(model_token_index)]
        feature_value = float(values[int(feature_id)].item())
        top_value = float(torch.max(values).item())
        rank = 1 + int((values > values[int(feature_id)]).sum().item()) if feature_value > 0 else None
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
                    {"feature_id": int(fid), "activation": float(value)}
                    for fid, value in zip(top_indices.tolist(), top_values.tolist(), strict=True)
                ],
            }
        )
    max_relative = max((float(row["feature_relative"]) for row in token_rows), default=0.0)
    best_row = max(token_rows, key=lambda row: float(row["feature_relative"]), default=None)
    return token_rows, best_row, max_relative


def run_sae_refinement(cfg: SaeRefinementConfig) -> list[Path]:
    cfg = resolve_sae_refinement_round(cfg)
    paths = sae_refinement_paths(cfg)
    source_annotation = sae_annotation_path(cfg)
    annotation_packet = _read_json(source_annotation)
    annotation = annotation_packet.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation file has no annotation object: {source_annotation}")
    test_cases = annotation.get("test_cases")
    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError(f"Annotation has no test_cases: {source_annotation}")
    if paths["tests"].exists() and not cfg.force:
        test_packet = _read_json(paths["tests"])
    else:
        test_packet = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "round": int(cfg.round_index),
            "feature": f"{cfg.model_name}/{cfg.layer}/F{int(cfg.feature_id):06d}",
            "source_annotation": str(source_annotation.resolve()),
            "label_before_tests": annotation,
            "test_results": evaluate_sae_test_cases(
                model_name=cfg.model_name,
                layer=cfg.layer,
                feature_id=int(cfg.feature_id),
                test_cases=test_cases,
                top_k=int(cfg.top_k),
                device=cfg.device,
                dtype=cfg.dtype,
            ),
        }
        write_json(paths["tests"], test_packet)
    if not test_packet_has_marked_activation(test_packet):
        stop_packet = stopped_sae_refinement_packet(cfg=cfg, source_annotation=source_annotation, test_packet=test_packet)
        write_json(paths["stop"], stop_packet)
        return [paths["tests"], paths["stop"]]
    request_payload = build_sae_refinement_request(cfg=cfg, test_packet=test_packet)
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
    try:
        refined = parse_annotation(output_text)
    except Exception:
        refined, repair_response_json = repair_annotation_output(
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
    warn_single_token_test_cases(
        refined,
        model_id=str(load_activation_config(CONFIG_PATHS[_normalize_model_name(cfg.model_name)])["model"]["id"]),
        context=f"{cfg.model_name}/{cfg.layer}/F{int(cfg.feature_id):06d} round {int(cfg.round_index)}",
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": cfg.provider,
        "model": cfg.model,
        "round": int(cfg.round_index),
        "source_annotation": str(source_annotation.resolve()),
        "test_results": str(paths["tests"].resolve()),
        "feature": f"{cfg.model_name}/{cfg.layer}/F{int(cfg.feature_id):06d}",
        "annotation": refined,
        "response": {"id": response_json.get("id"), "usage": response_json.get("usage")},
    }
    write_json(paths["annotation"], result)
    outputs.extend([paths["raw_response"], paths["annotation"]])
    return outputs


def stopped_sae_refinement_packet(*, cfg: SaeRefinementConfig, source_annotation: Path, test_packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "stopped",
        "reason": "no_marked_activation",
        "detail": (
            f"No proposed test case put this SAE feature in the top {MARKED_TEST_RESULT_MAX_RANK} at any token. "
            "Keeping the previous annotation stable instead of refining from failed tests."
        ),
        "round": int(cfg.round_index),
        "feature": f"{cfg.model_name}/{cfg.layer}/F{int(cfg.feature_id):06d}",
        "source_annotation": str(source_annotation.resolve()),
        "test_results": str(sae_refinement_paths(cfg)["tests"].resolve()),
        "label_before_tests": test_packet.get("label_before_tests"),
    }


def build_sae_refinement_request(*, cfg: SaeRefinementConfig, test_packet: dict[str, Any]) -> dict[str, Any]:
    evidence = _read_json(sae_evidence_path(cfg))
    messages = build_single_packet_refinement_messages(
        original_evidence=evidence,
        current_annotation=test_packet.get("label_before_tests"),
        previous_refinements=load_previous_sae_refinements(cfg),
        current_test_packet=test_packet,
        round_index=int(cfg.round_index),
    )
    if cfg.provider == "ollama":
        return {"model": cfg.model, "messages": messages, "think": False, "stream": False, "format": "json", "options": {"num_predict": int(cfg.max_tokens)}}
    payload: dict[str, Any] = {"model": cfg.model, "max_tokens": int(cfg.max_tokens), "messages": messages}
    if cfg.provider == "tinker-sdk":
        return payload
    if cfg.provider == "deepseek":
        payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
        return payload
    apply_openai_reasoning_controls(payload, provider=cfg.provider, model=cfg.model)
    payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "sae_feature_annotation_refinement", "strict": True, "schema": ANNOTATION_SCHEMA}}
    return payload


def import_sae_annotation_labels(
    *,
    annotation_root: Path = DEFAULT_SAE_ANNOTATION_ROOT,
    refinement_root: Path = DEFAULT_SAE_REFINEMENT_ROOT,
    db_path: Path = DEFAULT_FEATURE_INDEX,
    models: set[str] | None = None,
    layers: set[str] | None = None,
    provider_label: str | None = None,
) -> int:
    rows = []
    for path in sorted(annotation_root.resolve().glob("*/*/F*/*_annotation.json")):
        parsed = _parse_sae_annotation_path(annotation_root.resolve(), path)
        if parsed is None:
            continue
        model_name, layer, feature_id, file_label = parsed
        if models is not None and model_name not in models:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        packet = _read_json(path)
        annotation = packet.get("annotation")
        if not isinstance(annotation, dict):
            continue
        rows.append(_annotation_db_row(model_name, layer, feature_id, file_label, packet, annotation, path))
    refinement_rows = []
    for path in sorted(refinement_root.resolve().glob("*/*/F*/*_round*_annotation.json")):
        parsed = _parse_sae_refinement_path(refinement_root.resolve(), path)
        if parsed is None:
            continue
        model_name, layer, feature_id, file_label, round_index = parsed
        if models is not None and model_name not in models:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        if has_sae_stop_at_or_before(annotation_path=path, file_label=file_label, round_index=int(round_index)):
            continue
        packet = _read_json(path)
        annotation = packet.get("annotation")
        if not isinstance(annotation, dict):
            continue
        tests_path = path.with_name(f"{file_label}_round{round_index:02d}_tests.json")
        refinement_rows.append(
            (
                model_name,
                layer,
                int(feature_id),
                file_label,
                int(round_index),
                str(packet.get("provider") or ""),
                str(packet.get("model") or ""),
                str(packet.get("created_at") or ""),
                str(packet.get("source_annotation") or ""),
                str(tests_path.resolve()) if tests_path.is_file() else "",
                str(path.with_name(f"{file_label}_round{round_index:02d}_request_preview.json").resolve()),
                str(path.with_name(f"{file_label}_round{round_index:02d}_raw_response.json").resolve()),
                str(path.resolve()),
                json.dumps(annotation, ensure_ascii=False, sort_keys=True),
                str(annotation.get("label") or ""),
                str(annotation.get("simple_label") or ""),
                str(annotation.get("description") or ""),
                str(annotation.get("reasoning") or ""),
                str(annotation.get("confidence") or "unclear").lower(),
                json.dumps(annotation.get("test_cases") or [], ensure_ascii=False, sort_keys=True),
            )
        )
    with sqlite3.connect(db_path.resolve()) as conn:
        ensure_sae_annotation_schema(conn)
        delete_stopped_sae_refinements(
            conn,
            refinement_root=refinement_root.resolve(),
            models=models,
            layers=layers,
            provider_label=provider_label,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO sae_feature_annotations (
                model_name, layer, feature_id, provider_label, provider, provider_model, created_at,
                label, simple_label, description, reasoning, confidence, test_cases_json,
                annotation_path, raw_response_path, is_current
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            rows,
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO sae_feature_annotation_refinements (
                model_name, layer, feature_id, provider_label, round_index,
                provider, provider_model, created_at, source_annotation_path, tests_path,
                request_path, raw_response_path, annotation_path, annotation_json,
                label, simple_label, description, reasoning, confidence, test_cases_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            refinement_rows,
        )
        conn.commit()
    return len(rows) + len(refinement_rows)


def delete_stopped_sae_refinements(
    conn: sqlite3.Connection,
    *,
    refinement_root: Path,
    models: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> None:
    deletes = []
    for stop_path in sorted(refinement_root.glob("*/*/F*/*_round*_stop.json")):
        annotation_path = stop_path.with_name(stop_path.name.replace("_stop.json", "_annotation.json"))
        parsed = _parse_sae_refinement_path(refinement_root, annotation_path)
        if parsed is None:
            continue
        model_name, layer, feature_id, file_label, round_index = parsed
        if models is not None and model_name not in models:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and file_label != provider_label:
            continue
        deletes.append((model_name, layer, int(feature_id), file_label, int(round_index)))
    conn.executemany(
        """
        DELETE FROM sae_feature_annotation_refinements
        WHERE model_name = ?
          AND layer = ?
          AND feature_id = ?
          AND provider_label = ?
          AND round_index >= ?
        """,
        deletes,
    )


def has_sae_stop_at_or_before(*, annotation_path: Path, file_label: str, round_index: int) -> bool:
    return any(
        annotation_path.with_name(f"{file_label}_round{index:02d}_stop.json").is_file()
        for index in range(1, int(round_index) + 1)
    )


def ensure_sae_annotation_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sae_feature_annotations (
            model_name TEXT NOT NULL,
            layer TEXT NOT NULL,
            feature_id INTEGER NOT NULL,
            provider_label TEXT NOT NULL,
            provider TEXT,
            provider_model TEXT,
            created_at TEXT,
            label TEXT,
            simple_label TEXT,
            description TEXT,
            reasoning TEXT,
            confidence TEXT,
            test_cases_json TEXT,
            annotation_path TEXT,
            raw_response_path TEXT,
            is_current INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (model_name, layer, feature_id, provider_label)
        );
        CREATE TABLE IF NOT EXISTS sae_feature_annotation_refinements (
            model_name TEXT NOT NULL,
            layer TEXT NOT NULL,
            feature_id INTEGER NOT NULL,
            provider_label TEXT NOT NULL,
            round_index INTEGER NOT NULL,
            provider TEXT,
            provider_model TEXT,
            created_at TEXT,
            source_annotation_path TEXT,
            tests_path TEXT,
            request_path TEXT,
            raw_response_path TEXT,
            annotation_path TEXT,
            annotation_json TEXT,
            label TEXT,
            simple_label TEXT,
            description TEXT,
            reasoning TEXT,
            confidence TEXT,
            test_cases_json TEXT,
            PRIMARY KEY (model_name, layer, feature_id, provider_label, round_index)
        );
        CREATE INDEX IF NOT EXISTS idx_sae_feature_annotations_lookup
            ON sae_feature_annotations(model_name, layer, feature_id);
        CREATE INDEX IF NOT EXISTS idx_sae_feature_annotation_refinements_lookup
            ON sae_feature_annotation_refinements(model_name, layer, feature_id, provider_label, round_index);
        """
    )


def _promote_latest_sae_refinements(
    conn: sqlite3.Connection,
    *,
    models: set[str] | None,
    layers: set[str] | None,
    provider_label: str | None,
) -> None:
    rows = conn.execute(
        """
        SELECT r.*
        FROM sae_feature_annotation_refinements r
        WHERE r.label != ''
          AND NOT EXISTS (
              SELECT 1
              FROM sae_feature_annotation_refinements newer
              WHERE newer.model_name = r.model_name
                AND newer.layer = r.layer
                AND newer.feature_id = r.feature_id
                AND newer.provider_label = r.provider_label
                AND newer.round_index > r.round_index
                AND newer.label != ''
          )
        """
    ).fetchall()
    for row in rows:
        model_name, layer, row_provider = str(row[0]), str(row[1]), str(row[3])
        if models is not None and model_name not in models:
            continue
        if layers is not None and layer not in layers:
            continue
        if provider_label is not None and row_provider != provider_label:
            continue
        conn.execute(
            """
            INSERT INTO sae_feature_annotations (
                model_name, layer, feature_id, provider_label, provider, provider_model, created_at,
                label, simple_label, description, reasoning, confidence, test_cases_json,
                annotation_path, raw_response_path, is_current
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(model_name, layer, feature_id, provider_label) DO UPDATE SET
                provider = excluded.provider,
                provider_model = excluded.provider_model,
                created_at = excluded.created_at,
                label = excluded.label,
                simple_label = excluded.simple_label,
                description = excluded.description,
                reasoning = excluded.reasoning,
                confidence = excluded.confidence,
                test_cases_json = excluded.test_cases_json,
                annotation_path = excluded.annotation_path,
                raw_response_path = excluded.raw_response_path,
                is_current = 1
            """,
            (
                row["model_name"], row["layer"], int(row["feature_id"]), row["provider_label"],
                row["provider"], row["provider_model"], row["created_at"], row["label"], row["simple_label"],
                row["description"], row["reasoning"], row["confidence"], row["test_cases_json"],
                row["annotation_path"], row["raw_response_path"],
            ) if isinstance(row, sqlite3.Row) else (
                row[0], row[1], int(row[2]), row[3], row[5], row[6], row[7], row[14], row[15], row[16], row[17], row[18], row[19], row[12], row[11]
            ),
        )


def _annotation_db_row(model_name: str, layer: str, feature_id: int, file_label: str, packet: dict[str, Any], annotation: dict[str, Any], path: Path) -> tuple[Any, ...]:
    return (
        model_name,
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
        str(path.resolve()),
        str(path.with_name(f"{file_label}_raw_response.json").resolve()),
    )


def _parse_sae_annotation_path(root: Path, path: Path) -> tuple[str, str, int, str] | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) != 4:
        return None
    model_name, layer, feature_dir, filename = rel.parts
    feature_match = FEATURE_DIR_RE.match(feature_dir)
    file_match = ANNOTATION_FILENAME_RE.match(filename)
    if not feature_match or not file_match:
        return None
    return model_name, layer, int(feature_match.group("feature_id")), file_match.group("label")


def _parse_sae_refinement_path(root: Path, path: Path) -> tuple[str, str, int, str, int] | None:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if len(rel.parts) != 4:
        return None
    model_name, layer, feature_dir, filename = rel.parts
    feature_match = FEATURE_DIR_RE.match(feature_dir)
    file_match = REFINEMENT_FILENAME_RE.match(filename)
    if not feature_match or not file_match:
        return None
    return model_name, layer, int(feature_match.group("feature_id")), file_match.group("label"), int(file_match.group("round"))


def resolve_sae_refinement_round(cfg: SaeRefinementConfig) -> SaeRefinementConfig:
    if cfg.round_index is not None:
        return cfg
    base = sae_base_annotation_path(cfg)
    if not base.is_file():
        raise FileNotFoundError(f"Base SAE annotation JSON not found: {base}")
    stopped_round = first_stopped_sae_refinement_round(cfg)
    if stopped_round is not None:
        return SaeRefinementConfig(**{**cfg.__dict__, "round_index": stopped_round})
    round_index = 1
    while True:
        source = sae_source_annotation_for_round(cfg, round_index)
        annotation = sae_refinement_paths_for_round(cfg, round_index)["annotation"]
        if cfg.force and round_index > 1 and not source.is_file():
            round_index -= 1
            break
        if not annotation.is_file() or annotation.stat().st_mtime < source.stat().st_mtime:
            break
        round_index += 1
    return SaeRefinementConfig(**{**cfg.__dict__, "round_index": round_index})


def first_stopped_sae_refinement_round(cfg: SaeRefinementConfig) -> int | None:
    round_index = 1
    while True:
        stop_path = sae_refinement_paths_for_round(cfg, round_index)["stop"]
        annotation_path = sae_refinement_paths_for_round(cfg, round_index)["annotation"]
        tests_path = sae_refinement_paths_for_round(cfg, round_index)["tests"]
        if stop_path.is_file():
            return round_index
        if not annotation_path.is_file() and not tests_path.is_file():
            return None
        round_index += 1


def sae_base_annotation_path(cfg: SaeRefinementConfig) -> Path:
    return cfg.annotation_root.resolve() / cfg.model_name / cfg.layer / f"F{int(cfg.feature_id):06d}" / f"{cfg.provider_label}_annotation.json"


def sae_source_annotation_for_round(cfg: SaeRefinementConfig, round_index: int) -> Path:
    if cfg.source_annotation is not None:
        return cfg.source_annotation.resolve()
    if int(round_index) <= 1:
        return sae_base_annotation_path(cfg)
    return sae_refinement_paths_for_round(cfg, int(round_index) - 1)["annotation"]


def sae_annotation_path(cfg: SaeRefinementConfig) -> Path:
    if cfg.source_annotation is not None:
        return cfg.source_annotation.resolve()
    return sae_source_annotation_for_round(cfg, int(cfg.round_index or 1))


def sae_evidence_path(cfg: SaeRefinementConfig) -> Path:
    path = cfg.evidence_root.resolve() / cfg.model_name / cfg.layer / f"F{int(cfg.feature_id):06d}" / COMPACT_EVIDENCE_FILENAME
    if path.is_file():
        return path
    return path.with_name(LEGACY_COMPACT_EVIDENCE_FILENAME)


def sae_refinement_paths(cfg: SaeRefinementConfig) -> dict[str, Path]:
    return sae_refinement_paths_for_round(cfg, int(cfg.round_index))


def sae_refinement_paths_for_round(cfg: SaeRefinementConfig, round_index: int) -> dict[str, Path]:
    output_dir = cfg.refinement_root.resolve() / cfg.model_name / cfg.layer / f"F{int(cfg.feature_id):06d}"
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


def load_sae_initial_annotation(cfg: SaeRefinementConfig) -> dict[str, Any]:
    path = sae_base_annotation_path(cfg)
    if not path.is_file():
        return {}
    packet = _read_json(path)
    annotation = packet.get("annotation")
    return annotation if isinstance(annotation, dict) else {}


def load_previous_sae_refinements(cfg: SaeRefinementConfig) -> list[dict[str, Any]]:
    history = []
    for round_index in range(1, int(cfg.round_index)):
        paths = sae_refinement_paths_for_round(cfg, round_index)
        row: dict[str, Any] = {"round": round_index}
        if paths["tests"].is_file():
            row["test_packet"] = _read_json(paths["tests"])
        if paths["annotation"].is_file():
            packet = _read_json(paths["annotation"])
            row["annotation"] = packet.get("annotation") if isinstance(packet, dict) else None
        if row.keys() != {"round"}:
            history.append(row)
    return history


def write_request_debug_bundle(output_dir: Path, request_payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = request_payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    manifest = {"model": request_payload.get("model"), "message_count": len(messages), "request_keys": sorted(str(k) for k in request_payload), "message_files": []}
    write_json(output_dir / "request_options.json", {key: value for key, value in request_payload.items() if key != "messages"})
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown") if isinstance(message, dict) else "unknown"
        content = str(message.get("content") or "") if isinstance(message, dict) else str(message)
        parsed = parse_json_content(content)
        filename = f"message_{index:02d}_{safe_filename(role)}.json"
        write_json(output_dir / filename, {"index": index, "role": role, "content_characters": len(content), "content_json": parsed} if parsed is not None else {"index": index, "role": role, "content_characters": len(content), "content_text": content})
        manifest["message_files"].append({"index": index, "role": role, "file": filename, "content_characters": len(content), "content_type": "json" if parsed is not None else "text"})
    write_json(output_dir / "manifest.json", manifest)


def _lookup_neuronpedia_label(
    *,
    db_path: Path,
    counterpart: SaeCounterpart,
    layer_index: int,
    feature_id: int,
) -> dict[str, Any] | None:
    identity = _neuronpedia_identity(counterpart.repo_id)
    if identity is None or not db_path.is_file():
        return None
    _model_slug, sae_set = identity
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT description, explanation_model_name, neuronpedia_id
                FROM neuronpedia_labels
                WHERE model_name = ? AND sae_set = ? AND layer_index = ? AND feature_id = ?
                """,
                (counterpart.sae_model_name, sae_set, int(layer_index), int(feature_id)),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {"description": str(row[0] or ""), "explanation_model_name": row[1], "neuronpedia_id": row[2]}


def _neuronpedia_identity(counterpart_repo: str) -> tuple[str, str] | None:
    repo = str(counterpart_repo).lower()
    if "gpt2-small-oai-v5-32k-resid-post" in repo:
        return ("gpt2-small", "res_post_32k-oai")
    if "gemma-scope-2b-pt-res" in repo:
        return ("gemma-2-2b", "gemmascope-res-16k")
    if "sae-res-qwen3.5-2b-base-w32k" in repo:
        return ("qwen3.5-2b-pt", "qwenscope-res-32k")
    return None


def _normalize_model_name(model: str) -> str:
    aliases = {"gemma": "gemma2_2b", "qwen": "qwen3_5_2b_base", "gpt": "gpt2"}
    model_name = aliases.get(str(model), str(model))
    if model_name not in SAE_COUNTERPARTS:
        raise KeyError(f"Unsupported SAE model {model!r}; expected one of {sorted(SAE_COUNTERPARTS)}")
    return model_name


def _require_layer_index(layer: str) -> int:
    index = layer_index(layer)
    if index is None or index < 0:
        raise ValueError(f"SAE feature evidence requires a numbered transformer layer, got {layer!r}")
    return int(index)


def _torch_dtype(name: str, manifest: dict[str, Any]) -> torch.dtype:
    if name == "auto":
        name = str((manifest.get("capture") or {}).get("activation_dtype") or "float32")
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _runtime_sae_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.float32)
