from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from ..layers import layer_shard_records
from ..model_runtime import hidden_states_for_layer, load_runtime
from ..paths import DEFAULT_FEATURE_INDEX, V9_ROOT


DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "auto_annotation" / "evidence"
DEFAULT_SAMPLE_CACHE_ROOT = V9_ROOT / "artifacts" / "auto_annotation" / "feature_sample_cache"
EVIDENCE_FILENAME = "evidence.json"
DEBUGGING_FILENAME = "debugging.json"
LEGACY_COMPACT_EVIDENCE_FILENAME = "compact_evidence.json"
COMPACT_EVIDENCE_FILENAME = EVIDENCE_FILENAME
MAX_LARGEST_JUMP_REFINEMENT_CALLS = 64
DEFAULT_EXAMPLE_SCORE_THRESHOLD_FRAC = 0.33
DEFAULT_FEATURE_SAMPLE_CANDIDATE_POOL_SIZE = 40
DEFAULT_SEMANTIC_WINDOW_RADIUS = 16
DEFAULT_SEMANTIC_EXAMPLE_COUNT = 2
DEFAULT_EXAMPLE_ANCHOR_COUNT = 3
DEFAULT_REPLAY_PREFIX_TOKEN_COUNT = 0
DEFAULT_SINGLE_TOKEN_REPLAY_PAD = True
ANNOTATION_INSTRUCTION = (
    "Return only valid JSON with keys: reasoning, label, simple_label, description, confidence, test_cases. "
    "Use semantic_examples first: infer the broadest simple pattern that explains most examples, not just the shortest marked substring. "
    "Use erf_examples to check which left context is causally important; each left_context_ending_at_target ends with the activating target token. "
    "Right context in semantic windows is for readability, not causal evidence. "
    "Use high confidence only when one clear pattern covers most examples; otherwise use medium or low. "
    "In reasoning, give only one concise rationale sentence, under 40 words; do not write step-by-step analysis. "
    "In test_cases, propose exactly 8 compact prompts when possible, organized as two four-case contrast ladders. "
    "Each ladder should start from an evidence-like positive prompt, move away one factor at a time through intermediate prompts, "
    "and end at an expected negative prompt."
)
REQUESTED_RESPONSE_FORMAT = {
    "reasoning": "one concise rationale sentence, under 40 words",
    "label": "very short feature name, ideally 1-3 words",
    "simple_label": "plain easy label for non-native English speakers",
    "description": "one simple sentence describing what activates the feature",
    "confidence": "high, medium, or low",
    "test_cases": [
        {
            "text": "compact prompt to test; positives should preserve important left context from evidence",
            "expected": "activate, not_activate, or ambiguous",
            "reason": "why this case tests the proposed label",
        }
    ],
}


@dataclass(frozen=True)
class FeatureEvidenceConfig:
    feature_interface_dir: Path
    layer: str
    feature_id: int
    output_root: Path = DEFAULT_OUTPUT_ROOT
    sample_cache_root: Path = DEFAULT_SAMPLE_CACHE_ROOT
    db_path: Path = DEFAULT_FEATURE_INDEX
    top_k: int = DEFAULT_FEATURE_SAMPLE_CANDIDATE_POOL_SIZE
    examples: int = 10
    example_score_threshold_frac: float | None = DEFAULT_EXAMPLE_SCORE_THRESHOLD_FRAC
    batch_size: int = 8192
    device: str = "cuda"
    dtype: str = "float32"
    force_rebuild_sample_cache: bool = False
    force: bool = False
    update_index: bool = True


@dataclass(frozen=True)
class TopFeatureScanConfig:
    feature_interface_dir: Path
    layer: str | None = None
    sample_cache_root: Path = DEFAULT_SAMPLE_CACHE_ROOT
    db_path: Path = DEFAULT_FEATURE_INDEX
    top_k: int = DEFAULT_FEATURE_SAMPLE_CANDIDATE_POOL_SIZE
    batch_size: int = 8192
    device: str = "cuda"
    dtype: str = "float32"
    force_rebuild_sample_cache: bool = False
    update_index: bool = True


def build_top_feature_scan(cfg: TopFeatureScanConfig) -> list[Path]:
    feature_interface_dir = cfg.feature_interface_dir.resolve()
    feature_manifest = _read_json(feature_interface_dir / "manifest.json")
    activation_manifest_path = Path(str(feature_manifest["source_activation_manifest"]))
    activation_manifest = _read_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent
    run_id = feature_interface_dir.parent.name
    layer_paths = sorted(feature_interface_dir.glob("layer_*_features.pt"))
    if cfg.layer is not None:
        layer_paths = [feature_interface_dir / f"{cfg.layer}_features.pt"]
    outputs = []
    for layer_path in layer_paths:
        layer = layer_path.name.removesuffix("_features.pt")
        feature_artifact = torch.load(layer_path, map_location="cpu", weights_only=False)
        feature_tensors = feature_artifact["tensors"]
        layer_metadata = feature_artifact["metadata"]
        ica_artifact_path = Path(str(layer_metadata["source_ica_artifact"]))
        ica_artifact = torch.load(ica_artifact_path, map_location="cpu", weights_only=False)
        ica_tensors = ica_artifact["tensors"]
        compute_dtype = _torch_dtype(cfg.dtype, activation_manifest=activation_manifest)
        cache = _load_or_build_feature_sample_cache(
            cache_root=cfg.sample_cache_root,
            run_id=run_id,
            activation_dir=activation_dir,
            activation_manifest=activation_manifest,
            layer=layer,
            mean=ica_tensors["mean"].detach().cpu().to(torch.float32),
            feature_directions=feature_tensors["feature_directions"].detach().cpu().to(torch.float32),
            norm_eps=float(ica_artifact["metadata"].get("norm_eps", 1e-12)),
            top_k=int(cfg.top_k),
            batch_size=int(cfg.batch_size),
            device=torch.device(cfg.device),
            dtype=compute_dtype,
            force_rebuild=bool(cfg.force_rebuild_sample_cache),
        )
        cache_path = _feature_sample_cache_path(cache_root=cfg.sample_cache_root, run_id=run_id, layer=layer, top_k=int(cfg.top_k))
        if cfg.update_index:
            _update_feature_index_with_top1_scan(db_path=cfg.db_path, run_id=run_id, layer=layer, cache=cache)
        outputs.append(cache_path)
    return outputs


def build_feature_evidence(cfg: FeatureEvidenceConfig) -> Path:
    feature_interface_dir = cfg.feature_interface_dir.resolve()
    feature_manifest = _read_json(feature_interface_dir / "manifest.json")
    feature_artifact_path = feature_interface_dir / f"{cfg.layer}_features.pt"
    feature_artifact = torch.load(feature_artifact_path, map_location="cpu", weights_only=False)
    feature_tensors = feature_artifact["tensors"]
    layer_metadata = feature_artifact["metadata"]

    feature_id = int(cfg.feature_id)
    n_features = int(feature_tensors["feature_id"].numel())
    if feature_id < 0 or feature_id >= n_features:
        raise IndexError(f"feature_id {feature_id} outside [0, {n_features - 1}]")

    ica_artifact_path = Path(str(layer_metadata["source_ica_artifact"]))
    ica_artifact = torch.load(ica_artifact_path, map_location="cpu", weights_only=False)
    ica_tensors = ica_artifact["tensors"]
    activation_manifest_path = Path(str(feature_manifest["source_activation_manifest"]))
    activation_manifest = _read_json(activation_manifest_path)
    activation_dir = activation_manifest_path.parent

    run_id = feature_interface_dir.parent.name
    output_dir = cfg.output_root.resolve() / run_id / cfg.layer / f"F{feature_id:06d}"
    evidence_path = output_dir / EVIDENCE_FILENAME
    debugging_path = output_dir / DEBUGGING_FILENAME
    if evidence_path.exists() and debugging_path.exists() and not cfg.force:
        return evidence_path

    model_id = str((activation_manifest.get("model") or {}).get("id") or "")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    compute_dtype = _torch_dtype(cfg.dtype, activation_manifest=activation_manifest)

    sample_cache = _load_or_build_feature_sample_cache(
        cache_root=cfg.sample_cache_root,
        run_id=run_id,
        activation_dir=activation_dir,
        activation_manifest=activation_manifest,
        layer=cfg.layer,
        mean=ica_tensors["mean"].detach().cpu().to(torch.float32),
        feature_directions=feature_tensors["feature_directions"].detach().cpu().to(torch.float32),
        norm_eps=float(ica_artifact["metadata"].get("norm_eps", 1e-12)),
        top_k=int(cfg.top_k),
        batch_size=int(cfg.batch_size),
        device=torch.device(cfg.device),
        dtype=compute_dtype,
        force_rebuild=bool(cfg.force_rebuild_sample_cache),
    )
    samples = _samples_for_feature_from_cache(
        sample_cache,
        feature_id=feature_id,
        tokenizer=tokenizer,
    )
    if not samples:
        raise RuntimeError(f"No cached samples found for {run_id}/{cfg.layer}/F{feature_id}.")

    _add_relative_activation_scores(
        samples,
        activation_dir=activation_dir,
        activation_manifest=activation_manifest,
        layer=cfg.layer,
        mean=ica_tensors["mean"].detach().cpu().to(torch.float32),
        feature_directions=feature_tensors["feature_directions"].detach().cpu().to(torch.float32),
        norm_eps=float(ica_artifact["metadata"].get("norm_eps", 1e-12)),
        device=torch.device(cfg.device),
        dtype=compute_dtype,
    )
    top1_samples = _filter_top1_samples(samples, feature_id=feature_id)
    filtered_top1_samples = _filter_samples_by_score_threshold(
        top1_samples,
        score_threshold_frac=cfg.example_score_threshold_frac,
    )
    _enrich_samples_with_text(
        filtered_top1_samples,
        manifest=activation_manifest,
        tokenizer=tokenizer,
    )
    selected = _select_diverse_samples(
        filtered_top1_samples,
        count=int(cfg.examples),
        seed_parts=(run_id, cfg.layer, str(feature_id)),
    )
    replay_prefix_token_ids = _replay_prefix_token_ids(tokenizer)
    single_token_replay_pad_token_id = _single_token_replay_pad_token_id(tokenizer)
    _add_effective_receptive_field(
        selected,
        model_id=model_id,
        tokenizer=tokenizer,
        layer=cfg.layer,
        mean=ica_tensors["mean"].detach().cpu().to(torch.float32),
        feature_directions=feature_tensors["feature_directions"].detach().cpu().to(torch.float32),
        feature_id=feature_id,
        norm_eps=float(ica_artifact["metadata"].get("norm_eps", 1e-12)),
        prefix_token_ids=replay_prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=torch.device(cfg.device),
        dtype=compute_dtype,
    )

    evidence_packet, debugging_packet = _evidence_packets(
        run_id=run_id,
        layer=cfg.layer,
        feature_id=feature_id,
        feature_tensors=feature_tensors,
        layer_metadata=layer_metadata,
        samples=selected,
        settings={
            "candidate_pool_size": int(cfg.top_k),
            "examples": int(cfg.examples),
            "candidate_count_before_top1_filter": len(samples),
            "top1_candidate_count": len(top1_samples),
            "candidate_count_after_score_threshold": len(filtered_top1_samples),
            "example_score_threshold_frac": cfg.example_score_threshold_frac,
            "example_selection": (
                "Evidence examples are selected only from positions where this feature is the top-1 active feature "
                "at the target token. If example_score_threshold_frac is set, candidates below that fraction of the "
                "best top-1 candidate activation are removed. The strongest examples are kept as anchors; remaining "
                "examples are selected with deterministic stratified sampling over immediate-left token text, target-token "
                "text, capitalization, position bucket, and source document, so the final evidence preserves strength while "
                "covering more modes."
            ),
            "effective_receptive_field_definition": (
                "Estimated effective receptive field length is the right edge of the largest observed positive relative-score jump in left-context replay. "
                "It is the shortest tested context length after which the main observed activation-causing context "
                "has entered the prompt."
            ),
            "effective_receptive_field_search": (
                "Probe powers-of-two left-context lengths up to the full available document prefix, then refine "
                "the largest observed relative-score jump with midpoint probes until the interval is adjacent "
                "or the internal safety budget is reached."
            ),
            "largest_observed_relative_score_jump_note": (
                "largest_observed_relative_score_jump is computed from tested effective-receptive-field replay rows after midpoint "
                "refinement of the largest positive unresolved relative-score-change interval. Refinement keeps splitting the "
                "currently largest positive interval until it is adjacent or the internal safety budget is reached. It is an "
                "estimated boundary rather than an exhaustive search over every possible context length."
            ),
            "replay_relative_score_note": (
                "For each replayed left context, relative_score is this feature's activation divided by the largest feature "
                "activation at the target position in that replay. The top feature in that replay has relative_score = 1.0."
            ),
            "target_position_note": "Each left_context_ending_at_target field ends with the target token.",
            "relative_activation_note": (
                "relative_activation is this feature's activation divided by the largest feature activation "
                "at the same token position; the top feature at that position has relative_activation = 1.0."
            ),
            "right_context_hint_note": (
                "right_context_for_readability_not_causal_evidence is shown only to make the surrounding text pattern readable. "
                "It is future text and not causal evidence for this activation. It is omitted when the tokenizer cannot decode the next tokens cleanly."
            ),
            "beginning_of_sequence_token_used_for_replay": True,
            "replay_prefix_token_count": len(replay_prefix_token_ids),
            "single_token_replay_pad_token_id": single_token_replay_pad_token_id,
            "position_ids_preserved_for_replay": False,
            "tokenization_note": (
                "Replay uses exact suffix token IDs from the original document tokenization as a fresh prompt. "
                "It does not preserve original absolute position IDs, because annotation evidence should be understandable "
                "from the shown prompt alone. "
                "Context length excludes any hidden replay prefix or spacer tokens. The replay prefix is disabled by default. "
                "For context_length=1, one hidden pad token is inserted before the original token when the tokenizer provides a pad token."
            ),
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence_packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    debugging_path.write_text(json.dumps(debugging_packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if cfg.update_index:
        _update_feature_index_with_erf(
            db_path=cfg.db_path,
            run_id=run_id,
            layer=cfg.layer,
            feature_id=feature_id,
            effective_receptive_field=_summary_effective_receptive_field(selected),
            evidence_path=evidence_path,
            evidence_payload=evidence_packet,
        )
    return evidence_path


def _load_or_build_feature_sample_cache(
    *,
    cache_root: Path,
    run_id: str,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    norm_eps: float,
    top_k: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    cache_path = _feature_sample_cache_path(cache_root=cache_root, run_id=run_id, layer=layer, top_k=top_k)
    if cache_path.is_file() and not force_rebuild:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    cache = _build_feature_sample_cache(
        activation_dir=activation_dir,
        activation_manifest=activation_manifest,
        layer=layer,
        mean=mean,
        feature_directions=feature_directions,
        norm_eps=norm_eps,
        top_k=top_k,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    cache["metadata"] = {
        "cache_kind": "v9_feature_top_sample_cache",
        "run_id": run_id,
        "layer": layer,
        "top_k": int(top_k),
        "activation_manifest": str(activation_dir / "manifest.json"),
        "token_count": int(cache.get("token_count") or 0),
        "n_features": int(cache["scores"].shape[1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _feature_sample_cache_path(*, cache_root: Path, run_id: str, layer: str, top_k: int) -> Path:
    return cache_root.resolve() / run_id / layer / f"top1_feature_samples_k{int(top_k)}.pt"


def _build_feature_sample_cache(
    *,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    norm_eps: float,
    top_k: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    mean = mean.to(device=device, dtype=dtype)
    feature_directions = feature_directions.to(device=device, dtype=dtype)
    n_features = int(feature_directions.shape[0])
    scores_state = torch.full((n_features, int(top_k)), -float("inf"), dtype=torch.float32)
    local_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    global_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    shard_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    token_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    doc_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    position_state = torch.full((n_features, int(top_k)), -1, dtype=torch.long)
    top1_counts = torch.zeros(n_features, dtype=torch.long)
    top3_counts = torch.zeros(n_features, dtype=torch.long)

    global_offset = 0
    for shard in tqdm(layer_shard_records(activation_manifest, layer), desc=f"cache {layer}", unit="shard", dynamic_ncols=True):
        layer_path = shard["layers"].get(layer)
        if not isinstance(layer_path, str):
            raise KeyError(f"Layer {layer!r} missing from shard {shard.get('index')}.")
        shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
        input_ids = _load_optional_tensor(activation_dir, shard.get("input_ids"))
        doc_ids = _load_optional_tensor(activation_dir, shard.get("doc_ids"))
        positions = _load_optional_tensor(activation_dir, shard.get("positions"))
        shard_rows = int(shard_tensor.shape[0])
        for start in range(0, shard_rows, batch_size):
            end = min(shard_rows, start + batch_size)
            batch = shard_tensor[start:end].to(device=device, dtype=dtype, non_blocking=True)
            normalized = batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)
            activations = torch.relu((normalized - mean) @ feature_directions.T)
            top_values, top_feature_ids = torch.max(activations, dim=1)
            positive_winners = top_feature_ids[top_values.detach() > 0]
            if positive_winners.numel():
                top1_counts += torch.bincount(positive_winners.detach().to("cpu"), minlength=n_features).to(torch.long)
            top3_k = min(3, int(activations.shape[1]))
            top3_values, top3_feature_ids = torch.topk(activations, k=top3_k, dim=1, largest=True)
            positive_top3 = top3_feature_ids[top3_values.detach() > 0]
            if positive_top3.numel():
                top3_counts += torch.bincount(positive_top3.detach().to("cpu"), minlength=n_features).to(torch.long)
            _update_top1_feature_sample_cache_state(
                scores_state=scores_state,
                local_state=local_state,
                global_state=global_state,
                shard_state=shard_state,
                token_state=token_state,
                doc_state=doc_state,
                position_state=position_state,
                top_values=top_values.detach().to("cpu", dtype=torch.float32),
                top_feature_ids=top_feature_ids.detach().to("cpu", dtype=torch.long),
                start=int(start),
                shard_index=int(shard["index"]),
                global_offset=int(global_offset),
                input_ids=input_ids,
                doc_ids=doc_ids,
                positions=positions,
            )
            del batch, normalized, activations
        global_offset += int(shard.get("tokens", shard_rows))

    return {
        "scores": scores_state.T.contiguous(),
        "feature_ids": torch.arange(n_features, dtype=torch.long),
        "local_indices": local_state.T.contiguous(),
        "global_indices": global_state.T.contiguous(),
        "shard_indices": shard_state.T.contiguous(),
        "token_ids": token_state.T.contiguous(),
        "doc_ids": doc_state.T.contiguous(),
        "positions": position_state.T.contiguous(),
        "top1_counts": top1_counts,
        "top3_counts": top3_counts,
        "token_count": int(global_offset),
    }


def _update_top1_feature_sample_cache_state(
    *,
    scores_state: torch.Tensor,
    local_state: torch.Tensor,
    global_state: torch.Tensor,
    shard_state: torch.Tensor,
    token_state: torch.Tensor,
    doc_state: torch.Tensor,
    position_state: torch.Tensor,
    top_values: torch.Tensor,
    top_feature_ids: torch.Tensor,
    start: int,
    shard_index: int,
    global_offset: int,
    input_ids: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> None:
    n_features, keep_k = scores_state.shape
    for feature_id in torch.unique(top_feature_ids).tolist():
        feature_id = int(feature_id)
        if feature_id < 0 or feature_id >= n_features:
            continue
        mask = top_feature_ids == feature_id
        count = int(mask.sum().item())
        if count <= 0:
            continue
        feature_scores = top_values[mask]
        local_offsets = torch.nonzero(mask, as_tuple=False).flatten().to(torch.long)
        k = min(int(keep_k), int(count))
        values, order = torch.topk(feature_scores, k=k, largest=True)
        local_indices = local_offsets[order] + int(start)
        _update_single_feature_sample_cache_state(
            feature_id=feature_id,
            scores_state=scores_state,
            local_state=local_state,
            global_state=global_state,
            shard_state=shard_state,
            token_state=token_state,
            doc_state=doc_state,
            position_state=position_state,
            new_scores=values,
            new_local_indices=local_indices,
            shard_index=shard_index,
            global_offset=global_offset,
            input_ids=input_ids,
            doc_ids=doc_ids,
            positions=positions,
        )


def _update_single_feature_sample_cache_state(
    *,
    feature_id: int,
    scores_state: torch.Tensor,
    local_state: torch.Tensor,
    global_state: torch.Tensor,
    shard_state: torch.Tensor,
    token_state: torch.Tensor,
    doc_state: torch.Tensor,
    position_state: torch.Tensor,
    new_scores: torch.Tensor,
    new_local_indices: torch.Tensor,
    shard_index: int,
    global_offset: int,
    input_ids: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> None:
    keep_k = int(scores_state.shape[1])
    new_global_indices = new_local_indices + int(global_offset)
    new_shard_indices = torch.full_like(new_local_indices, int(shard_index))
    new_token_ids = _gather_optional_1d(input_ids, new_local_indices)
    new_doc_ids = _gather_optional_1d(doc_ids, new_local_indices)
    new_positions = _gather_optional_1d(positions, new_local_indices)

    combined_scores = torch.cat([scores_state[feature_id], new_scores.to(torch.float32)], dim=0)
    combined_local = torch.cat([local_state[feature_id], new_local_indices.to(torch.long)], dim=0)
    combined_global = torch.cat([global_state[feature_id], new_global_indices.to(torch.long)], dim=0)
    combined_shard = torch.cat([shard_state[feature_id], new_shard_indices.to(torch.long)], dim=0)
    combined_token = torch.cat([token_state[feature_id], new_token_ids.to(torch.long)], dim=0)
    combined_doc = torch.cat([doc_state[feature_id], new_doc_ids.to(torch.long)], dim=0)
    combined_position = torch.cat([position_state[feature_id], new_positions.to(torch.long)], dim=0)

    order = torch.topk(combined_scores, k=keep_k, largest=True).indices
    scores_state[feature_id].copy_(combined_scores[order])
    local_state[feature_id].copy_(combined_local[order])
    global_state[feature_id].copy_(combined_global[order])
    shard_state[feature_id].copy_(combined_shard[order])
    token_state[feature_id].copy_(combined_token[order])
    doc_state[feature_id].copy_(combined_doc[order])
    position_state[feature_id].copy_(combined_position[order])


def _update_feature_sample_cache_state(
    *,
    scores_state: torch.Tensor,
    local_state: torch.Tensor,
    global_state: torch.Tensor,
    shard_state: torch.Tensor,
    token_state: torch.Tensor,
    doc_state: torch.Tensor,
    position_state: torch.Tensor,
    new_scores: torch.Tensor,
    new_local_indices: torch.Tensor,
    shard_index: int,
    global_offset: int,
    input_ids: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
) -> None:
    n_features, keep_k = scores_state.shape
    new_global_indices = new_local_indices + int(global_offset)
    new_shard_indices = torch.full_like(new_local_indices, int(shard_index))
    new_token_ids = _gather_optional_1d(input_ids, new_local_indices)
    new_doc_ids = _gather_optional_1d(doc_ids, new_local_indices)
    new_positions = _gather_optional_1d(positions, new_local_indices)

    combined_scores = torch.cat([scores_state, new_scores], dim=1)
    combined_local = torch.cat([local_state, new_local_indices], dim=1)
    combined_global = torch.cat([global_state, new_global_indices], dim=1)
    combined_shard = torch.cat([shard_state, new_shard_indices], dim=1)
    combined_token = torch.cat([token_state, new_token_ids], dim=1)
    combined_doc = torch.cat([doc_state, new_doc_ids], dim=1)
    combined_position = torch.cat([position_state, new_positions], dim=1)
    order = torch.topk(combined_scores, k=keep_k, dim=1, largest=True).indices
    scores_state.copy_(torch.gather(combined_scores, 1, order))
    local_state.copy_(torch.gather(combined_local, 1, order))
    global_state.copy_(torch.gather(combined_global, 1, order))
    shard_state.copy_(torch.gather(combined_shard, 1, order))
    token_state.copy_(torch.gather(combined_token, 1, order))
    doc_state.copy_(torch.gather(combined_doc, 1, order))
    position_state.copy_(torch.gather(combined_position, 1, order))


def _gather_optional_1d(values: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor:
    if values is None:
        return torch.full_like(indices, -1)
    flat = values.to("cpu", dtype=torch.long)
    safe_indices = indices.clamp_min(0)
    return flat[safe_indices]


def _samples_for_feature_from_cache(cache: dict[str, Any], *, feature_id: int, tokenizer: Any) -> list[dict[str, Any]]:
    scores = cache["scores"]
    if feature_id < 0 or feature_id >= int(scores.shape[1]):
        raise IndexError(f"feature_id {feature_id} outside cached score shape {tuple(scores.shape)}.")
    samples = []
    for rank in range(int(scores.shape[0])):
        score = float(scores[rank, feature_id])
        if not torch.isfinite(torch.tensor(score)) or score <= 0:
            continue
        token_id = int(cache["token_ids"][rank, feature_id])
        samples.append(
            {
                "activation": score,
                "shard_index": int(cache["shard_indices"][rank, feature_id]),
                "local_index": int(cache["local_indices"][rank, feature_id]),
                "global_index": int(cache["global_indices"][rank, feature_id]),
                "token_id": token_id if token_id >= 0 else None,
                "token": tokenizer.convert_ids_to_tokens(token_id) if token_id >= 0 else None,
                "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False) if token_id >= 0 else None,
                "doc_id": _none_if_negative(int(cache["doc_ids"][rank, feature_id])),
                "position": _none_if_negative(int(cache["positions"][rank, feature_id])),
                "source": "feature_sample_cache",
            }
        )
    return samples


def _none_if_negative(value: int) -> int | None:
    return None if int(value) < 0 else int(value)


def _scan_top_feature_samples(
    *,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    mean: torch.Tensor,
    direction: torch.Tensor,
    norm_eps: float,
    top_k: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    mean = mean.to(device=device, dtype=dtype)
    direction = direction.to(device=device, dtype=dtype)
    candidates: list[dict[str, Any]] = []
    global_offset = 0
    for shard in tqdm(layer_shard_records(activation_manifest, layer), desc=f"scan {layer}", unit="shard", dynamic_ncols=True):
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
                batch_cpu = shard_tensor[start : start + batch_size]
                batch = batch_cpu.to(device=device, dtype=dtype, non_blocking=True)
                normalized = batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)
                scores = torch.relu((normalized - mean) @ direction)
                k = min(top_k, int(scores.numel()))
                values, indices = torch.topk(scores, k=k, largest=True)
                for score, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist(), strict=True):
                    local_index = int(start + int(index))
                    token_id = int(input_ids[local_index]) if input_ids is not None else None
                    shard_best.append(
                        {
                            "activation": float(score),
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
                del batch, normalized, scores
        candidates.extend(shard_best)
        candidates = sorted(candidates, key=lambda row: float(row["activation"]), reverse=True)[:top_k]
        global_offset += int(shard.get("tokens", shard_tensor.shape[0]))
    return candidates


def _select_diverse_samples(samples: list[dict[str, Any]], *, count: int, seed_parts: tuple[str, ...]) -> list[dict[str, Any]]:
    if count <= 0 or not samples:
        return []

    sorted_samples = sorted(samples, key=lambda row: float(row.get("activation") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    seen_doc_ids: set[int] = set()
    seen_ids: set[tuple[int, int]] = set()

    def add(sample: dict[str, Any]) -> bool:
        sample_id = (int(sample.get("shard_index", -1)), int(sample.get("local_index", -1)))
        if sample_id in seen_ids:
            return False
        doc_id = sample.get("doc_id")
        if doc_id is not None and int(doc_id) in seen_doc_ids:
            return False
        selected.append(sample)
        seen_ids.add(sample_id)
        if doc_id is not None:
            seen_doc_ids.add(int(doc_id))
        return True

    anchor_count = min(DEFAULT_EXAMPLE_ANCHOR_COUNT, int(count))
    for sample in sorted_samples:
        add(sample)
        if len(selected) >= anchor_count:
            break

    if len(selected) >= count:
        return selected[:count]

    rng = random.Random("|".join(seed_parts))
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sample in sorted_samples:
        sample_id = (int(sample.get("shard_index", -1)), int(sample.get("local_index", -1)))
        if sample_id in seen_ids:
            continue
        buckets.setdefault(_sample_diversity_key(sample), []).append(sample)

    bucket_items = list(buckets.items())
    rng.shuffle(bucket_items)
    bucket_items.sort(key=lambda item: (-float(item[1][0].get("activation") or 0.0), item[0]))
    queues = [queue for _, queue in bucket_items]
    for queue in queues:
        queue.sort(key=lambda row: float(row.get("activation") or 0.0), reverse=True)

    while len(selected) < count and queues:
        progressed = False
        next_queues = []
        for queue in queues:
            while queue:
                sample = queue.pop(0)
                if add(sample):
                    progressed = True
                    break
            if queue:
                next_queues.append(queue)
            if len(selected) >= count:
                break
        if not progressed:
            break
        queues = next_queues

    if len(selected) < count:
        for sample in sorted_samples:
            add(sample)
            if len(selected) >= count:
                break
    return selected[:count]


def _sample_left_key(sample: dict[str, Any]) -> str:
    return str(sample.get("previous_token_text") or "").strip().lower() or "<bos>"


def _sample_diversity_key(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    token_text = str(sample.get("text") or sample.get("token") or "")
    normalized = token_text.strip().lower() or "<blank>"
    stripped = token_text.strip()
    previous_text = _sample_left_key(sample)
    if stripped.isupper() and any(ch.isalpha() for ch in stripped):
        capitalization = "upper"
    elif stripped[:1].isupper():
        capitalization = "capitalized"
    elif stripped.islower() and any(ch.isalpha() for ch in stripped):
        capitalization = "lower"
    elif any(ch.isdigit() for ch in stripped):
        capitalization = "digit"
    else:
        capitalization = "other"
    position = sample.get("position")
    if position is None:
        position_bucket = "unknown_pos"
    else:
        pos = int(position)
        if pos < 8:
            position_bucket = "pos_0_7"
        elif pos < 32:
            position_bucket = "pos_8_31"
        elif pos < 128:
            position_bucket = "pos_32_127"
        else:
            position_bucket = "pos_128_plus"
    return (previous_text, normalized, capitalization, position_bucket)


def _filter_samples_by_score_threshold(
    samples: list[dict[str, Any]],
    *,
    score_threshold_frac: float | None,
) -> list[dict[str, Any]]:
    if score_threshold_frac is None:
        return samples
    if float(score_threshold_frac) < 0:
        raise ValueError("example_score_threshold_frac must be non-negative.")
    if not samples:
        return samples
    best_activation = max(float(sample.get("activation") or 0.0) for sample in samples)
    min_activation = best_activation * float(score_threshold_frac)
    return [sample for sample in samples if float(sample.get("activation") or 0.0) >= min_activation]


def _filter_top1_samples(samples: list[dict[str, Any]], *, feature_id: int) -> list[dict[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("top_feature_id_at_position") is not None
        and int(sample["top_feature_id_at_position"]) == int(feature_id)
    ]


def _add_relative_activation_scores(
    samples: list[dict[str, Any]],
    *,
    activation_dir: Path,
    activation_manifest: dict[str, Any],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    norm_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not samples:
        return
    mean = mean.to(device=device, dtype=dtype)
    feature_directions = feature_directions.to(device=device, dtype=dtype)
    shards = {int(record["index"]): record for record in layer_shard_records(activation_manifest, layer)}
    by_shard: dict[int, list[dict[str, Any]]] = {}
    for example_index, sample in enumerate(samples):
        by_shard.setdefault(int(sample["shard_index"]), []).append(sample)

    with torch.no_grad():
        for shard_index, shard_samples in by_shard.items():
            shard = shards[shard_index]
            layer_path = shard["layers"].get(layer)
            if not isinstance(layer_path, str):
                raise KeyError(f"Layer {layer!r} missing from shard {shard_index}.")
            shard_tensor = torch.load(activation_dir / layer_path, map_location="cpu")
            local_indices = [int(sample["local_index"]) for sample in shard_samples]
            batch = shard_tensor[local_indices].to(device=device, dtype=dtype, non_blocking=True)
            normalized = batch / torch.linalg.vector_norm(batch, dim=1, keepdim=True).clamp_min(norm_eps)
            activations = torch.relu((normalized - mean) @ feature_directions.T)
            top_values, top_indices = torch.max(activations, dim=1)
            for row, top_value, top_index in zip(shard_samples, top_values, top_indices, strict=True):
                top_activation = float(top_value.detach().cpu().item())
                activation = float(row.get("activation") or 0.0)
                row["top_feature_id_at_position"] = int(top_index.detach().cpu().item())
                row["top_feature_activation_at_position"] = top_activation
                row["relative_activation"] = activation / top_activation if top_activation > 0 else None


def _enrich_samples_with_text(
    samples: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    tokenizer: Any,
) -> None:
    doc_ids = sorted({int(sample["doc_id"]) for sample in samples if sample.get("doc_id") is not None})
    if not doc_ids:
        return
    texts = _load_dataset_texts(manifest=manifest, doc_ids=doc_ids)
    context_length = int((manifest.get("capture") or {}).get("context_length") or 1024)
    for example_index, sample in enumerate(samples):
        doc_id = sample.get("doc_id")
        position = sample.get("position")
        if doc_id is None or position is None:
            continue
        document_text = texts.get(int(doc_id))
        if document_text is None:
            continue
        input_ids = list(tokenizer(document_text, truncation=True, max_length=context_length)["input_ids"])
        pos = int(position)
        sample["_context_to_target_ids"] = [int(token_id) for token_id in input_ids[: pos + 1]]
        window_left = max(0, pos - DEFAULT_SEMANTIC_WINDOW_RADIUS)
        window_right = min(len(input_ids), pos + DEFAULT_SEMANTIC_WINDOW_RADIUS + 1)
        sample["_semantic_window_ids"] = [int(token_id) for token_id in input_ids[window_left:window_right]]
        sample["_semantic_window_target_offset"] = int(pos - window_left)
        if pos > 0:
            previous_token_id = int(input_ids[pos - 1])
            sample["previous_token_text"] = tokenizer.decode([previous_token_id], clean_up_tokenization_spaces=False)
        right_hint_ids = input_ids[pos + 1 : min(len(input_ids), pos + 5)]
        right_hint_text = tokenizer.decode(right_hint_ids, clean_up_tokenization_spaces=False)
        if right_hint_text and "\ufffd" not in right_hint_text:
            sample["next_text_hint"] = right_hint_text


def _add_effective_receptive_field(
    samples: list[dict[str, Any]],
    *,
    model_id: str,
    tokenizer: Any,
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not samples:
        return
    runtime = load_runtime(model_id, str(device), "auto")
    model = runtime.model
    tokenizer = runtime.tokenizer
    mean = mean.to(device=device, dtype=torch.float32)
    feature_directions = feature_directions.to(device=device, dtype=torch.float32)
    for sample in tqdm(samples, desc=f"estimated ERF {layer}", unit="example", dynamic_ncols=True):
        ids = sample.get("_context_to_target_ids")
        if not isinstance(ids, list) or not ids:
            sample["effective_receptive_field"] = {"available": False, "reason": "missing_context_to_target_ids"}
            continue
        full_activation = float(sample.get("activation") or 0.0)
        if full_activation <= 0:
            sample["effective_receptive_field"] = {
                "available": False,
                "reason": "nonpositive_full_activation",
                "full_activation": _round_float(full_activation),
            }
            continue
        effective_receptive_field = _build_effective_receptive_field(
            model=model,
            tokenizer=tokenizer,
            ids=[int(token_id) for token_id in ids],
            layer=layer,
            mean=mean,
            feature_directions=feature_directions,
            feature_id=int(feature_id),
            norm_eps=float(norm_eps),
            full_activation=full_activation,
            prefix_token_ids=prefix_token_ids,
            single_token_replay_pad_token_id=single_token_replay_pad_token_id,
            device=device,
        )
        sample["effective_receptive_field"] = effective_receptive_field
        _add_semantic_activation_window(
            sample,
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            mean=mean,
            feature_directions=feature_directions,
            feature_id=int(feature_id),
            norm_eps=float(norm_eps),
            prefix_token_ids=prefix_token_ids,
            device=device,
        )


def _add_semantic_activation_window(
    sample: dict[str, Any],
    *,
    model: Any,
    tokenizer: Any,
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    prefix_token_ids: list[int],
    device: torch.device,
    max_rank: int = 10,
) -> None:
    window_ids = sample.get("_semantic_window_ids")
    target_offset = sample.get("_semantic_window_target_offset")
    if not isinstance(window_ids, list) or target_offset is None:
        return
    if not window_ids:
        return
    target_offset = int(target_offset)
    if target_offset < 0 or target_offset >= len(window_ids):
        return
    token_ranks, relative_scores = _replay_feature_token_ranks(
        model=model,
        input_ids=[int(token_id) for token_id in window_ids],
        layer=layer,
        mean=mean,
        feature_directions=feature_directions,
        feature_id=feature_id,
        norm_eps=norm_eps,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=None,
        device=device,
    )
    sample["semantic_activation_window"] = {
        "target_offset_in_window": target_offset,
        "max_rank_shown": int(max_rank),
        "marked_window": _marked_rank_text(
            tokenizer=tokenizer,
            token_ids=[int(token_id) for token_id in window_ids],
            ranks=token_ranks,
            max_rank=max_rank,
            target_offset=target_offset,
        ),
        "marked_tokens": _marked_token_rows(
            tokenizer=tokenizer,
            token_ids=[int(token_id) for token_id in window_ids],
            ranks=token_ranks,
            relative_scores=relative_scores,
            max_rank=max_rank,
        ),
    }


def _build_effective_receptive_field(
    *,
    model: Any,
    tokenizer: Any,
    ids: list[int],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    full_activation: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> dict[str, Any]:
    tested: dict[int, float] = {0: 0.0}
    available_prefix_len = len(ids)
    max_len = min(available_prefix_len, _max_replay_context_length(model, prefix_token_ids=prefix_token_ids))
    if max_len <= 0:
        return {
            "available": False,
            "reason": "no_context_fits_model_window_after_replay_prefix",
            "available_prefix_context_length": int(available_prefix_len),
            "replay_prefix_length": int(len(prefix_token_ids)),
        }

    def relative_score(length: int) -> float:
        length = int(max(1, min(length, max_len)))
        if length not in tested:
            tested[length] = _replay_feature_relative_score(
                model=model,
                input_ids=ids[-length:],
                layer=layer,
                mean=mean,
                feature_directions=feature_directions,
                feature_id=feature_id,
                norm_eps=norm_eps,
                prefix_token_ids=prefix_token_ids,
                single_token_replay_pad_token_id=single_token_replay_pad_token_id,
                device=device,
            )
        return tested[length]

    for length in _coarse_context_lengths(max_len):
        relative_score(length)
        if _monotonic_bound_satisfied(tested, max_length=max_len):
            break
    _refine_largest_observed_relative_score_jump(
        tested=tested,
        score_fn=relative_score,
        max_length=max_len,
    )
    tested_rows = _sorted_tested(tested, tokenizer=tokenizer, ids=ids)
    full_prefix_relative_score = relative_score(max_len)
    largest_jump = _largest_observed_relative_score_jump(tested_rows)
    _add_rank_trace_to_largest_jump(
        largest_jump,
        model=model,
        tokenizer=tokenizer,
        ids=ids,
        layer=layer,
        mean=mean,
        feature_directions=feature_directions,
        feature_id=feature_id,
        norm_eps=norm_eps,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=device,
    )
    estimated_effective_receptive_field_length = (
        int(largest_jump["to_context_length"]) if isinstance(largest_jump, dict) and largest_jump.get("to_context_length") else None
    )
    return {
        "available": True,
        "definition": "right edge of the largest observed positive relative-score jump in left-context replay",
        "estimated_effective_receptive_field_length": estimated_effective_receptive_field_length,
        "full_activation": _round_float(full_activation),
        "available_prefix_context_length": int(available_prefix_len),
        "replay_prefix_length": int(len(prefix_token_ids)),
        "replay_context_clipped_by_model_window": bool(max_len < available_prefix_len),
        "full_prefix_context_length": int(max_len),
        "full_prefix_relative_score": _round_float(full_prefix_relative_score),
        "largest_observed_relative_score_jump": largest_jump,
        "tested_context_lengths": tested_rows,
    }


def _replay_feature_relative_score(
    *,
    model: Any,
    input_ids: list[int],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> float:
    replay_padding_ids = _single_token_replay_padding_ids(
        input_ids=input_ids,
        pad_token_id=single_token_replay_pad_token_id,
    )
    replay_ids = [*prefix_token_ids, *replay_padding_ids, *input_ids]
    token_tensor = torch.tensor([replay_ids], dtype=torch.long, device=device)
    hidden_states = hidden_states_for_layer(model, layer, {"input_ids": token_tensor})
    hidden = hidden_states[-1].detach().to(torch.float32)
    normalized = hidden / torch.linalg.vector_norm(hidden).clamp_min(norm_eps)
    activations = torch.relu((normalized - mean).reshape(1, -1) @ feature_directions.T).squeeze(0)
    top_activation = torch.max(activations).clamp_min(norm_eps)
    return float((activations[int(feature_id)] / top_activation).detach().cpu().item())


def _add_rank_trace_to_largest_jump(
    largest_jump: dict[str, Any] | None,
    *,
    model: Any,
    tokenizer: Any,
    ids: list[int],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
    max_rank: int = 10,
) -> None:
    if not isinstance(largest_jump, dict):
        return
    from_length = int(largest_jump.get("from_context_length") or 0)
    to_length = int(largest_jump.get("to_context_length") or 0)
    if to_length <= 0 or to_length <= from_length:
        return
    to_ids = ids[-to_length:]
    if not to_ids:
        return
    token_ranks, relative_scores = _replay_feature_token_ranks(
        model=model,
        input_ids=to_ids,
        layer=layer,
        mean=mean,
        feature_directions=feature_directions,
        feature_id=feature_id,
        norm_eps=norm_eps,
        prefix_token_ids=prefix_token_ids,
        single_token_replay_pad_token_id=single_token_replay_pad_token_id,
        device=device,
    )
    largest_jump["feature_rank_trace"] = {
        "max_rank_shown": int(max_rank),
        "marked_replay_context_ending_at_target": _marked_rank_text(
            tokenizer=tokenizer,
            token_ids=to_ids,
            ranks=token_ranks,
            max_rank=max_rank,
        ),
    }


def _replay_feature_token_ranks(
    *,
    model: Any,
    input_ids: list[int],
    layer: str,
    mean: torch.Tensor,
    feature_directions: torch.Tensor,
    feature_id: int,
    norm_eps: float,
    prefix_token_ids: list[int],
    single_token_replay_pad_token_id: int | None,
    device: torch.device,
) -> tuple[list[int | None], list[float]]:
    replay_padding_ids = _single_token_replay_padding_ids(
        input_ids=input_ids,
        pad_token_id=single_token_replay_pad_token_id,
    )
    replay_ids = [*prefix_token_ids, *replay_padding_ids, *input_ids]
    token_tensor = torch.tensor([replay_ids], dtype=torch.long, device=device)
    hidden_start = len(prefix_token_ids) + len(replay_padding_ids)
    hidden = hidden_states_for_layer(model, layer, {"input_ids": token_tensor})[hidden_start:].detach().to(torch.float32)
    normalized = hidden / torch.linalg.vector_norm(hidden, dim=1, keepdim=True).clamp_min(norm_eps)
    activations = torch.relu((normalized - mean) @ feature_directions.T)
    feature_values = activations[:, int(feature_id)]
    top_values = activations.max(dim=1).values.clamp_min(norm_eps)
    ranks = 1 + (activations > feature_values[:, None]).sum(dim=1)
    ranks = torch.where(feature_values > 0, ranks, torch.zeros_like(ranks))
    relative = (feature_values / top_values).detach().cpu().tolist()
    ranks_cpu = ranks.detach().cpu().tolist()
    rank_list = [int(rank) if int(rank) > 0 else None for rank in ranks_cpu]
    return rank_list, [float(value) for value in relative]


def _marked_rank_text(
    *,
    tokenizer: Any,
    token_ids: list[int],
    ranks: list[int | None],
    max_rank: int,
    target_offset: int | None = None,
) -> str:
    pieces: list[str] = []
    for offset, (token_id, rank) in enumerate(zip(token_ids, ranks, strict=False)):
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
        if rank is not None and int(rank) <= int(max_rank):
            text = f"{text}[{int(rank)}]"
        if target_offset is not None and int(offset) == int(target_offset):
            text = f"{text}[target]"
        pieces.append(text)
    return "".join(pieces)


def _marked_token_rows(
    *,
    tokenizer: Any,
    token_ids: list[int],
    ranks: list[int | None],
    relative_scores: list[float],
    max_rank: int,
) -> list[dict[str, Any]]:
    rows = []
    for offset, (token_id, rank, relative_score) in enumerate(zip(token_ids, ranks, relative_scores, strict=False)):
        if rank is None or int(rank) > int(max_rank):
            continue
        rows.append(
            {
                "offset_in_window": int(offset),
                "token": tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False),
                "rank": int(rank),
                "score": _round_float(relative_score),
            }
        )
    return rows


def _sorted_tested(
    tested: dict[int, float],
    *,
    tokenizer: Any,
    ids: list[int],
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    rows = [
        {
            "context_length": 0,
            "relative_score": 0.0,
            "left_context_ending_at_target": "",
        }
    ]
    for length in sorted(tested):
        if int(length) <= 0:
            continue
        if max_length is not None and int(length) > int(max_length):
            continue
        left = max(0, len(ids) - int(length))
        rows.append(
            {
                "context_length": int(length),
                "relative_score": _round_float(tested[length]),
                "left_context_ending_at_target": tokenizer.decode(
                    ids[left:],
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
    return rows


def _coarse_context_lengths(max_len: int) -> list[int]:
    lengths = []
    length = 1
    while length < max_len:
        lengths.append(length)
        length *= 2
    lengths.append(int(max_len))
    return sorted(set(lengths))


def _max_replay_context_length(model: Any, *, prefix_token_ids: list[int]) -> int:
    config = getattr(model, "config", None)
    max_positions = None
    for name in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(config, name, None)
        if value is not None:
            max_positions = int(value)
            break
    if max_positions is None:
        return 10**9
    return max(0, int(max_positions) - len(prefix_token_ids))


def _largest_observed_relative_score_jump(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    best: dict[str, Any] | None = None
    for left, right in zip(rows, rows[1:], strict=False):
        left_length = int(left["context_length"])
        right_length = int(right["context_length"])
        length_delta = right_length - left_length
        if length_delta <= 0:
            continue
        left_score = float(left["relative_score"])
        right_score = float(right["relative_score"])
        delta = right_score - left_score
        if delta <= 0:
            continue
        slope = delta / float(length_delta)
        candidate = {
            "from_context_length": left_length,
            "to_context_length": right_length,
            "from_relative_score": _round_float(left_score),
            "to_relative_score": _round_float(right_score),
            "relative_score_delta": _round_float(delta),
            "absolute_relative_score_delta": _round_float(delta),
            "relative_score_delta_per_context_length": _round_float(slope),
            "direction": "increase",
            "added_left_text": _added_left_text(
                str(left.get("left_context_ending_at_target") or ""),
                str(right.get("left_context_ending_at_target") or ""),
            ),
        }
        if best is None or float(candidate["relative_score_delta"] or 0.0) > float(
            best["relative_score_delta"] or 0.0
        ):
            best = candidate
    return best


def _refine_largest_observed_relative_score_jump(
    *,
    tested: dict[int, float],
    score_fn: Any,
    max_length: int,
) -> None:
    for _ in range(MAX_LARGEST_JUMP_REFINEMENT_CALLS):
        interval = _largest_jump_interval_from_tested(tested, max_length=max_length)
        if interval is None:
            return
        left, right = interval
        if right - left <= 1:
            return
        mid = (left + right) // 2
        if mid in tested:
            return
        score_fn(mid)


def _largest_jump_interval_from_tested(tested: dict[int, float], *, max_length: int) -> tuple[int, int] | None:
    lengths = sorted(length for length in tested if int(length) <= int(max_length))
    if len(lengths) < 2:
        return None
    best_pair: tuple[int, int] | None = None
    best_delta = 0.0
    for left, right in zip(lengths, lengths[1:], strict=False):
        if int(right) <= int(left):
            continue
        if int(right) - int(left) <= 1:
            continue
        positive_delta = max(0.0, float(tested[right]) - float(tested[left]))
        if positive_delta > best_delta:
            best_delta = positive_delta
            best_pair = (int(left), int(right))
    return best_pair


def _monotonic_bound_satisfied(tested: dict[int, float], *, max_length: int) -> bool:
    lengths = sorted(length for length in tested if int(length) <= int(max_length))
    if len(lengths) < 2:
        return False
    best_observed_delta = 0.0
    largest_unresolved_possible_delta = 0.0
    for left, right in zip(lengths, lengths[1:], strict=False):
        if int(right) <= int(left):
            continue
        positive_delta = max(0.0, float(tested[right]) - float(tested[left]))
        best_observed_delta = max(best_observed_delta, positive_delta)
        if int(right) - int(left) > 1:
            largest_unresolved_possible_delta = max(largest_unresolved_possible_delta, positive_delta)

    last_length = int(lengths[-1])
    if last_length < int(max_length):
        # Monotonic + bounded relative score: the unseen tail can add at most
        # the remaining headroom up to the maximum relative score of 1.
        tail_possible_delta = max(0.0, 1.0 - float(tested[last_length]))
        largest_unresolved_possible_delta = max(largest_unresolved_possible_delta, tail_possible_delta)

    return best_observed_delta > 0.0 and largest_unresolved_possible_delta <= best_observed_delta


def _added_left_text(left_context: str, right_context: str) -> str:
    if right_context.endswith(left_context):
        return _tail_text(right_context[: len(right_context) - len(left_context)])
    return _tail_text(right_context)


def _tail_text(text: str, *, max_chars: int = 360) -> str:
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _evidence_packets(
    *,
    run_id: str,
    layer: str,
    feature_id: int,
    feature_tensors: dict[str, torch.Tensor],
    layer_metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_dead = len(samples) == 0 and int(settings.get("top1_candidate_count") or 0) == 0
    evidence_packet = {
        "rank_marker_legend": {
            "syntax": "[n] after a token means this feature ranked n-th among all features at that token.",
            "max_rank_shown": 10,
            "target_marker": "[target] marks the selected target token in semantic_examples.",
        },
    }
    position_hint = _same_position_hint(samples)
    if position_hint is not None:
        evidence_packet["position_hint"] = position_hint
    evidence_packet["semantic_examples"] = [
        _semantic_sample(index, sample)
        for index, sample in enumerate(samples[:DEFAULT_SEMANTIC_EXAMPLE_COUNT])
    ]
    evidence_packet.update({
        "effective_receptive_field_legend": {
            "definition": (
                "Estimated effective receptive field (ERF) examples replay the target token with progressively more "
                "left context. Every sudden score jump is important evidence: compare the jump points across all "
                "examples and choose a label that explains their common pattern, not just one example."
            ),
        },
        "erf_examples": [_compact_sample(index, sample) for index, sample in enumerate(samples)],
    })
    debugging_packet = {
        "debugging_type": "feature_evidence_build_debug",
        "evidence_type": "top_activating_examples",
        "feature": f"{run_id}/{layer}/F{feature_id}",
        "feature_metadata": {
            "feature_id": feature_id,
            "kurtosis": _round_float(feature_tensors["kurtosis"][feature_id]),
            "activation_frequency": _round_float(feature_tensors["activation_frequency"][feature_id]),
            "dead": bool(feature_tensors["dead"][feature_id].item()),
            "evidence_dead_no_top1_examples": evidence_dead,
        },
        "model_facing_evidence_file": EVIDENCE_FILENAME,
        "example_selection_summary": _example_selection_summary(settings),
        "builder_settings": {
            key: value
            for key, value in settings.items()
            if key not in {"example_selection"}
        },
        "builder_notes": {
            "example_selection": settings.get("example_selection"),
            "effective_receptive_field_definition": settings.get("effective_receptive_field_definition"),
            "effective_receptive_field_search": settings.get("effective_receptive_field_search"),
            "largest_observed_relative_score_jump_note": settings.get("largest_observed_relative_score_jump_note"),
            "replay_relative_score_note": settings.get("replay_relative_score_note"),
            "target_position_note": settings.get("target_position_note"),
            "relative_activation_note": settings.get("relative_activation_note"),
            "right_context_hint_note": settings.get("right_context_hint_note"),
            "tokenization_note": settings.get("tokenization_note"),
        },
        "legacy_instruction_removed_from_evidence": {
            "annotation_instruction": ANNOTATION_INSTRUCTION,
            "requested_response_format": REQUESTED_RESPONSE_FORMAT,
        },
        "full_erf_examples": [_debug_erf_sample(index, sample) for index, sample in enumerate(samples)],
    }
    return evidence_packet, debugging_packet


def _example_selection_summary(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_pool_size": settings.get("candidate_pool_size"),
        "selected_examples": settings.get("examples"),
        "selection_rule": (
            "Examples are selected from positions where this feature is top-1 at the target token, "
            "filtered to sufficiently strong activations, then selected with deterministic diversity: keep the strongest "
            "anchor examples and fill the rest across immediate-left token text, target-token text, capitalization, "
            "position bucket, and source document."
        ),
        "score_threshold_fraction_of_best": settings.get("example_score_threshold_frac"),
        "beginning_of_sequence_token_used_for_replay": settings.get("beginning_of_sequence_token_used_for_replay"),
        "replay_prefix_token_count": settings.get("replay_prefix_token_count"),
        "single_token_replay_pad_token_id": settings.get("single_token_replay_pad_token_id"),
        "position_ids_preserved_for_replay": settings.get("position_ids_preserved_for_replay"),
    }


def _same_position_hint(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    positions = [sample.get("position") for sample in samples]
    if len(positions) < 2 or any(position is None for position in positions):
        return None
    unique_positions = {int(position) for position in positions}
    if len(unique_positions) != 1:
        return None
    position = unique_positions.pop()
    return {
        "shared_position": position,
        "interpretation": "All selected examples share this document token position; consider whether this is a position feature.",
    }


def _semantic_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    row = {
        "example_index": index,
        "target_token": sample.get("text") or sample.get("token"),
        "position": sample.get("position"),
        "relative_activation": _round_float(sample.get("relative_activation")),
        "marked_activation_window": _semantic_activation_window(sample),
    }
    right_context_hint = sample.get("next_text_hint")
    if right_context_hint:
        target_token = str(sample.get("text") or sample.get("token") or "")
        readable_span = target_token + str(right_context_hint)
        row["right_context_hint"] = (
            f"Target token `{target_token}` together with a few right tokens becomes "
            f"`{readable_span}...`; right tokens are for readability only and are not causal evidence."
        )
    return row


def _semantic_activation_window(sample: dict[str, Any]) -> dict[str, Any] | None:
    semantic_window = sample.get("semantic_activation_window")
    if isinstance(semantic_window, dict):
        return semantic_window
    erf = sample.get("effective_receptive_field")
    if not isinstance(erf, dict):
        return None
    jump = erf.get("largest_observed_relative_score_jump")
    if not isinstance(jump, dict):
        return None
    trace = jump.get("feature_rank_trace")
    if not isinstance(trace, dict):
        return None
    marked = str(trace.get("marked_replay_context_ending_at_target") or "")
    if not marked:
        return None
    right_context = str(sample.get("next_text_hint") or "")
    marked_window = marked + right_context
    return {
        "marked_left_context_ending_at_target": marked,
        "marked_window_with_right_context": marked_window,
    }


def _compact_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    erf = _compact_effective_receptive_field(sample.get("effective_receptive_field"))
    row = {
        "target_token": sample.get("text") or sample.get("token"),
    }
    if isinstance(erf, dict):
        row.update(erf)
    else:
        row["effective_receptive_field"] = erf
    return row


def _compact_effective_receptive_field(erf: Any) -> Any:
    if not isinstance(erf, dict) or not erf.get("available"):
        return erf
    jump = erf.get("largest_observed_relative_score_jump")
    tested_rows = erf.get("tested_context_lengths")
    trace = jump.get("feature_rank_trace") if isinstance(jump, dict) else None
    return {
        "erf_length": erf.get("estimated_effective_receptive_field_length"),
        "key_left_context": jump.get("added_left_text") if isinstance(jump, dict) else None,
        "marked_replay_context": trace.get("marked_replay_context_ending_at_target") if isinstance(trace, dict) else None,
        "context_tests": _context_tests_for_labeling(tested_rows, jump),
    }


def _context_tests_for_labeling(tested_rows: Any, jump: Any) -> list[dict[str, Any]]:
    if not isinstance(tested_rows, list):
        return []
    wanted: set[int] = set()
    sudden_jump_lengths: set[int] = set()
    if isinstance(jump, dict):
        from_value = jump.get("from_context_length")
        to_value = jump.get("to_context_length")
        if from_value is not None:
            wanted.add(int(from_value))
        if to_value is not None:
            length = int(to_value)
            wanted.add(length)
            sudden_jump_lengths.add(length)
    if tested_rows:
        for row in (tested_rows[0], tested_rows[-1]):
            if isinstance(row, dict) and row.get("context_length") is not None:
                wanted.add(int(row["context_length"]))
    rows = []
    for row in tested_rows:
        if not isinstance(row, dict) or row.get("context_length") is None:
            continue
        if int(row["context_length"]) not in wanted:
            continue
        text = str(row.get("left_context_ending_at_target") or "")
        if not text:
            continue
        output_row = {
            "text": text,
            "score": _round_float(row.get("relative_score")),
        }
        if int(row["context_length"]) in sudden_jump_lengths:
            output_row["sudden_jump"] = True
        rows.append(output_row)
    return rows


def _debug_erf_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": index,
        "activation": _round_float(sample.get("activation")),
        "relative_activation": _round_float(sample.get("relative_activation")),
        "target_token": sample.get("text") or sample.get("token"),
        "doc_id": sample.get("doc_id"),
        "position": sample.get("position"),
        "effective_receptive_field": _debug_effective_receptive_field(sample.get("effective_receptive_field")),
    }


def _debug_effective_receptive_field(erf: Any) -> Any:
    if not isinstance(erf, dict) or not erf.get("available"):
        return erf
    jump = erf.get("largest_observed_relative_score_jump")
    tested_rows = erf.get("tested_context_lengths")
    return {
        "estimated_effective_receptive_field_length": erf.get("estimated_effective_receptive_field_length"),
        "full_prefix_relative_score": erf.get("full_prefix_relative_score"),
        "largest_observed_relative_score_jump": jump,
        "key_tested_context_lengths": _key_tested_context_lengths(tested_rows, jump),
    }


def _key_tested_context_lengths(tested_rows: Any, jump: Any) -> list[dict[str, Any]]:
    if not isinstance(tested_rows, list):
        return []
    wanted: set[int] = set()
    if isinstance(jump, dict):
        for key in ("from_context_length", "to_context_length"):
            value = jump.get(key)
            if value is not None:
                wanted.add(int(value))
    if tested_rows:
        for row in (tested_rows[0], tested_rows[-1]):
            if isinstance(row, dict) and row.get("context_length") is not None:
                wanted.add(int(row["context_length"]))
    return [
        row
        for row in tested_rows
        if isinstance(row, dict) and row.get("context_length") is not None and int(row["context_length"]) in wanted
    ]


def _summary_effective_receptive_field(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    examples = []
    for example_index, sample in enumerate(samples):
        erf = sample.get("effective_receptive_field")
        if not isinstance(erf, dict):
            continue
        if erf.get("available") is False:
            continue
        estimated = erf.get("estimated_effective_receptive_field_length")
        if estimated is not None:
            values.append(int(estimated))
        examples.append(
            {
                "example_index": example_index,
                "doc_id": sample.get("doc_id"),
                "position": sample.get("position"),
                "activation": _round_float(sample.get("activation")),
                "relative_activation": _round_float(sample.get("relative_activation")),
                "estimated_effective_receptive_field_length": estimated,
                "full_prefix_relative_score": erf.get("full_prefix_relative_score"),
                "largest_observed_relative_score_jump": erf.get("largest_observed_relative_score_jump"),
            }
        )
    values.sort()
    if values:
        mid = len(values) // 2
        median = float(values[mid]) if len(values) % 2 else (float(values[mid - 1]) + float(values[mid])) / 2.0
    else:
        median = None
    return {
        "definition": "Estimated effective receptive field length is the right edge of the largest observed positive relative-score jump in left-context replay.",
        "summary_statistic": "median estimated_effective_receptive_field_length over selected evidence examples",
        "effective_context_mean": median,
        "estimated_effective_receptive_field_length_values": values,
        "examples": examples,
    }


def summarize_effective_receptive_field(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-example ERF evidence for feature-index import."""

    return _summary_effective_receptive_field(samples)


def update_feature_index_with_erf(
    *,
    db_path: Path,
    run_id: str,
    layer: str,
    feature_id: int,
    effective_receptive_field: dict[str, Any],
    evidence_path: Path,
    evidence_payload: dict[str, Any] | None = None,
) -> None:
    """Import ERF summary and evidence path into the feature SQLite index."""

    _update_feature_index_with_erf(
        db_path=db_path,
        run_id=run_id,
        layer=layer,
        feature_id=feature_id,
        effective_receptive_field=effective_receptive_field,
        evidence_path=evidence_path,
        evidence_payload=evidence_payload,
    )


def _update_feature_index_with_erf(
    *,
    db_path: Path,
    run_id: str,
    layer: str,
    feature_id: int,
    effective_receptive_field: dict[str, Any],
    evidence_path: Path,
    evidence_payload: dict[str, Any] | None = None,
) -> None:
    db_path = db_path.resolve()
    if not db_path.is_file():
        return
    with sqlite3.connect(db_path) as conn:
        _ensure_feature_erf_columns(conn)
        conn.execute(
            """
            UPDATE features
            SET effective_context_mean = ?,
                effective_receptive_field_json = ?,
                annotation_evidence_path = ?,
                annotation_evidence_json = ?
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (
                effective_receptive_field.get("effective_context_mean"),
                json.dumps(effective_receptive_field, sort_keys=True),
                str(evidence_path.resolve()),
                json.dumps(evidence_payload, sort_keys=True, ensure_ascii=False) if evidence_payload is not None else None,
                run_id,
                layer,
                int(feature_id),
            ),
        )
        conn.commit()


def _update_feature_index_with_top1_scan(*, db_path: Path, run_id: str, layer: str, cache: dict[str, Any]) -> None:
    db_path = db_path.resolve()
    if not db_path.is_file():
        return
    top1_counts = cache.get("top1_counts")
    if top1_counts is None:
        scores = cache["scores"]
        top1_counts = (scores > 0).sum(dim=0).to(torch.long)
    top3_counts = cache.get("top3_counts")
    if top3_counts is None:
        top3_counts = top1_counts
    top1_counts = top1_counts.detach().cpu().to(torch.long)
    top3_counts = top3_counts.detach().cpu().to(torch.long)
    token_count = int((cache.get("metadata") or {}).get("token_count") or 0)
    rows = []
    for feature_id, (top1_count, top3_count) in enumerate(zip(top1_counts.tolist(), top3_counts.tolist(), strict=True)):
        top1_count = int(top1_count)
        top3_count = int(top3_count)
        top1_frequency = (float(top1_count) / float(token_count)) if token_count > 0 else None
        top3_frequency = (float(top3_count) / float(token_count)) if token_count > 0 else None
        rows.append(
            (
                top1_count,
                top1_frequency,
                int(top1_count == 0),
                top3_count,
                top3_frequency,
                int(top3_count == 0),
                run_id,
                layer,
                int(feature_id),
            )
        )
    with sqlite3.connect(db_path) as conn:
        _ensure_feature_top1_scan_columns(conn)
        conn.executemany(
            """
            UPDATE features
            SET top1_count = ?,
                top1_frequency = ?,
                top1_dead = ?,
                top3_count = ?,
                top3_frequency = ?,
                top3_dead = ?
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            rows,
        )
        conn.commit()


def _ensure_feature_erf_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    columns = {
        "effective_context_mean": "REAL",
        "effective_receptive_field_json": "TEXT",
        "annotation_evidence_path": "TEXT",
        "annotation_evidence_json": "TEXT",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {name} {definition}")


def _ensure_feature_top1_scan_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    columns = {
        "top1_count": "INTEGER",
        "top1_frequency": "REAL",
        "top1_dead": "INTEGER",
        "top3_count": "INTEGER",
        "top3_frequency": "REAL",
        "top3_dead": "INTEGER",
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE features ADD COLUMN {name} {definition}")


def _load_dataset_texts(*, manifest: dict[str, Any], doc_ids: list[int]) -> dict[int, str]:
    from datasets import load_dataset

    dataset_info = manifest.get("dataset") or {}
    path = str(dataset_info["path"])
    name = dataset_info.get("name")
    split = str(dataset_info.get("split") or "train")
    text_column = str(dataset_info.get("text_column") or "text")
    dataset = load_dataset(
        path,
        name,
        split=split,
        streaming=False,
        download_mode="reuse_dataset_if_exists",
    )
    return {int(doc_id): str(dataset[int(doc_id)][text_column]) for doc_id in doc_ids}


def _load_optional_tensor(root: Path, relative_path: Any) -> torch.Tensor | None:
    if not isinstance(relative_path, str):
        return None
    path = root / relative_path
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu")


def _replay_prefix_token_ids(tokenizer: Any) -> list[int]:
    if DEFAULT_REPLAY_PREFIX_TOKEN_COUNT <= 0:
        return []
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(tokenizer, "eos_token_id", None)
    if bos_token_id is None:
        raise ValueError("Replay requires a BOS prefix, but tokenizer has neither bos_token_id nor eos_token_id.")
    return [int(bos_token_id)] * DEFAULT_REPLAY_PREFIX_TOKEN_COUNT


def _single_token_replay_pad_token_id(tokenizer: Any) -> int | None:
    if not DEFAULT_SINGLE_TOKEN_REPLAY_PAD:
        return None
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        return int(bos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return int(eos_token_id) if eos_token_id is not None else None


def _single_token_replay_padding_ids(*, input_ids: list[int], pad_token_id: int | None) -> list[int]:
    if pad_token_id is None or len(input_ids) != 1:
        return []
    return [int(pad_token_id)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_float(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    return round(float(value), digits)


def _torch_dtype(name: str, *, activation_manifest: dict[str, Any] | None = None) -> torch.dtype:
    if name == "auto":
        manifest_dtype = str((activation_manifest or {}).get("capture", {}).get("activation_dtype") or "")
        if manifest_dtype in {"float32", "bfloat16", "float16"}:
            name = manifest_dtype
        else:
            name = "float32"
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")
