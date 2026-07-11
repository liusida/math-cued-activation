from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..model_runtime import hidden_states_and_logits_for_layer, load_runtime
from ..paths import DEFAULT_FEATURE_INDEX


DEFAULT_DB_PATH = DEFAULT_FEATURE_INDEX


@dataclass(frozen=True)
class FeatureBundle:
    run_id: str
    layer: str
    feature_pt_path: Path
    source_ica_artifact: Path
    feature_directions: torch.Tensor
    source_component_index: torch.Tensor
    source_sign: torch.Tensor
    kurtosis: torch.Tensor
    dead: torch.Tensor
    activation_frequency: torch.Tensor
    max_activation: torch.Tensor
    mean: torch.Tensor
    norm_eps: float


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_optional_feature_columns(conn)
    return conn


def _ensure_optional_feature_columns(conn: sqlite3.Connection) -> None:
    existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    columns = {
        "effective_context_mean": "REAL",
        "effective_receptive_field_json": "TEXT",
        "annotation_evidence_path": "TEXT",
        "activation_frequency_gt_1": "REAL",
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
    conn.commit()


def list_meta(db_path: Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    with connect(db_path) as conn:
        runs = []
        for run in conn.execute(
            "SELECT run_id, model_id, model_short_name, display_name, method, token_budget, n_components, hidden_size FROM model_runs ORDER BY run_id"
        ):
            layers = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT layer, layer_index, rows, n_features, alive_count, dead_count,
                           dead_kurtosis_threshold
                    FROM layers
                    WHERE run_id = ?
                    ORDER BY layer_index
                    """,
                    (run["run_id"],),
                )
            ]
            runs.append({**dict(run), "layers": layers})
        return {"db_path": str(db_path), "runs": runs}


def get_feature_rows(run_id: str, layer: str, feature_ids: list[int], db_path: Path = DEFAULT_DB_PATH) -> dict[int, sqlite3.Row]:
    if not feature_ids:
        return {}
    placeholders = ",".join("?" for _ in feature_ids)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id IN ({placeholders})
            """,
            (run_id, layer, *feature_ids),
        ).fetchall()
    return {int(row["feature_id"]): row for row in rows}


@lru_cache(maxsize=16)
def load_feature_bundle(run_id: str, layer: str, db_path_str: str = str(DEFAULT_DB_PATH)) -> FeatureBundle:
    db_path = Path(db_path_str)
    with connect(db_path) as conn:
        layer_row = conn.execute(
            """
            SELECT feature_pt_path, source_ica_artifact
            FROM layers
            WHERE run_id = ? AND layer = ?
            """,
            (run_id, layer),
        ).fetchone()
        run_row = conn.execute(
            "SELECT norm_eps FROM model_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    if layer_row is None:
        raise KeyError(f"No layer indexed for {run_id} {layer}")

    feature_pt_path = Path(layer_row["feature_pt_path"])
    feature_artifact = torch.load(feature_pt_path, map_location="cpu", weights_only=False)
    feature_tensors = feature_artifact["tensors"]
    source_ica_artifact = Path(layer_row["source_ica_artifact"])
    ica_artifact = torch.load(source_ica_artifact, map_location="cpu", weights_only=False)
    mean = ica_artifact["tensors"]["mean"].detach().cpu().to(torch.float32)
    norm_eps = float(run_row["norm_eps"] if run_row and run_row["norm_eps"] is not None else 1e-12)
    return FeatureBundle(
        run_id=run_id,
        layer=layer,
        feature_pt_path=feature_pt_path,
        source_ica_artifact=source_ica_artifact,
        feature_directions=feature_tensors["feature_directions"].detach().cpu().to(torch.float32),
        source_component_index=feature_tensors["source_component_index"].detach().cpu(),
        source_sign=feature_tensors["source_sign"].detach().cpu(),
        kurtosis=feature_tensors["kurtosis"].detach().cpu().to(torch.float32),
        dead=feature_tensors["dead"].detach().cpu(),
        activation_frequency=feature_tensors["activation_frequency"].detach().cpu().to(torch.float32),
        max_activation=feature_tensors["max"].detach().cpu().to(torch.float32),
        mean=mean,
        norm_eps=norm_eps,
    )


def probe_text(
    *,
    run_id: str,
    layer: str,
    text: str,
    top_k: int,
    model_id: str,
    device: str,
    dtype: str,
    show_next_token: bool = False,
    show_logit_lens: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    runtime = load_runtime(model_id, device, dtype)
    bundle = load_feature_bundle(run_id, layer, str(db_path))
    inputs = runtime.tokenizer(text, return_tensors="pt", truncation=True)
    inputs = {key: value.to(runtime.device) for key, value in inputs.items()}
    hidden_states, final_logits = hidden_states_and_logits_for_layer(runtime.model, layer, inputs)
    prediction_ids = torch.argmax(final_logits, dim=-1).detach().cpu().tolist() if show_next_token else None
    logit_lens_ids = _greedy_logit_lens_predictions(runtime.model, hidden_states) if show_logit_lens else None
    feature_directions = bundle.feature_directions.to(runtime.device)
    mean = bundle.mean.to(runtime.device)
    normalized = hidden_states / torch.linalg.vector_norm(hidden_states, dim=1, keepdim=True).clamp_min(bundle.norm_eps)
    activations = torch.relu((normalized - mean) @ feature_directions.T)
    k = min(int(top_k), int(activations.shape[1]))
    values, indices = torch.topk(activations, k=k, dim=1)
    values_cpu = values.detach().cpu()
    indices_cpu = indices.detach().cpu()

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    token_texts = [_decode_token(runtime.tokenizer, token_id) for token_id in input_ids]
    token_display_parts = _replacement_token_display_parts(runtime.tokenizer, input_ids, token_texts)
    all_feature_ids = sorted({int(i) for i in indices_cpu.reshape(-1).tolist()})
    feature_rows = get_feature_rows(run_id, layer, all_feature_ids, db_path)

    tokens = []
    for token_index, (token_id, token_text) in enumerate(zip(input_ids, token_texts)):
        top_features = []
        for rank in range(k):
            feature_id = int(indices_cpu[token_index, rank].item())
            row = feature_rows[feature_id]
            top_features.append(
                {
                    "feature_id": feature_id,
                    "rank": rank,
                    "activation": float(values_cpu[token_index, rank].item()),
                    "kurtosis": float(row["kurtosis"]),
                    "dead": _feature_dead(row),
                    "kurtosis_dead": bool(row["dead"]),
                    "top1_count": _optional_int(row["top1_count"]),
                    "top1_frequency": _optional_float(row["top1_frequency"]),
                    "top1_dead": _optional_bool(row["top1_dead"]),
                    "top3_count": _optional_int(row["top3_count"]),
                    "top3_frequency": _optional_float(row["top3_frequency"]),
                    "top3_dead": _optional_bool(row["top3_dead"]),
                    "activation_frequency": float(row["activation_frequency"]),
                    "max_activation": float(row["max_activation"]),
                    "effective_context_mean": None
                    if row["effective_context_mean"] is None
                    else float(row["effective_context_mean"]),
                    "source_component_index": int(row["source_component_index"]),
                    "source_sign": int(row["source_sign"]),
                    "source_side": str(row["source_side"]),
                    "mini_histogram_url": f"/api/mini-histogram/{run_id}/{layer}/{feature_id}",
                }
            )
        token = {
            "token_index": token_index,
            "token_id": int(token_id),
            "text": token_text,
            "top_features": top_features,
        }
        if token_display_parts[token_index] is not None:
            token.update(token_display_parts[token_index])
        if prediction_ids is not None:
            prediction_id = int(prediction_ids[token_index])
            token["prediction"] = {
                "token_id": prediction_id,
                "text": _decode_token(runtime.tokenizer, prediction_id),
                "source": "greedy_model_prediction",
            }
        if logit_lens_ids is not None:
            prediction_id = int(logit_lens_ids[token_index])
            token["logit_lens_prediction"] = {
                "token_id": prediction_id,
                "text": _decode_token(runtime.tokenizer, prediction_id),
                "source": "greedy_logit_lens",
            }
        tokens.append(token)

    return {
        "run_id": run_id,
        "model_id": model_id,
        "layer": layer,
        "top_k": k,
        "tokens": tokens,
    }


def mini_histogram_path(run_id: str, layer: str, feature_id: int, db_path: Path = DEFAULT_DB_PATH) -> Path:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT mini_histogram_svg_path
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, layer, feature_id),
        ).fetchone()
    if row is None or row["mini_histogram_svg_path"] is None:
        raise FileNotFoundError(f"No mini histogram for {run_id} {layer} feature {feature_id}")
    return Path(row["mini_histogram_svg_path"])

def _decode_token(tokenizer: object, token_id: int) -> str:
    text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if text == "":
        return f"<{token_id}>"
    return text


def _replacement_token_display_parts(
    tokenizer: object,
    token_ids: list[int],
    token_texts: list[str],
) -> list[dict[str, Any] | None]:
    """Describe replacement-character tokens that jointly decode to valid text."""
    parts: list[dict[str, Any] | None] = [None] * len(token_ids)
    index = 0
    while index < len(token_ids):
        if "�" not in token_texts[index]:
            index += 1
            continue
        matched_end = None
        decoded_group = ""
        for end in range(index + 2, min(len(token_ids), index + 4) + 1):
            if any("�" not in token_texts[piece] for piece in range(index, end)):
                break
            candidate = tokenizer.decode(token_ids[index:end], clean_up_tokenization_spaces=False)
            if candidate and "�" not in candidate:
                matched_end = end
                isolated_group = "".join(token_texts[index:end])
                decoded_group = _recovered_group_text(candidate, isolated_group)
                break
        if matched_end is None:
            index += 1
            continue
        piece_count = matched_end - index
        for piece_index, token_index in enumerate(range(index, matched_end), start=1):
            raw_piece = tokenizer.convert_ids_to_tokens(int(token_ids[token_index]))
            parts[token_index] = {
                "display_text": decoded_group,
                "display_piece_index": piece_index,
                "display_piece_count": piece_count,
                "raw_token_piece": str(raw_piece),
            }
        index = matched_end
    return parts


def _recovered_group_text(decoded_group: str, isolated_group: str) -> str:
    """Return only the text repaired by joint decoding, excluding stable prefix/suffix text."""
    prefix = 0
    prefix_limit = min(len(decoded_group), len(isolated_group))
    while prefix < prefix_limit and decoded_group[prefix] == isolated_group[prefix]:
        prefix += 1
    suffix = 0
    suffix_limit = min(len(decoded_group) - prefix, len(isolated_group) - prefix)
    while suffix < suffix_limit and decoded_group[-1 - suffix] == isolated_group[-1 - suffix]:
        suffix += 1
    end = len(decoded_group) - suffix
    repaired = decoded_group[prefix:end]
    return repaired or decoded_group


def _greedy_logit_lens_predictions(model: object, hidden_states: torch.Tensor) -> list[int]:
    """Greedy next-token IDs from an intermediate residual stream via the final norm and LM head."""
    final_norm = _final_model_norm(model)
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model has no output embedding / LM head for next-token prediction")
    parameter = next(output_embeddings.parameters(), None)
    target_dtype = parameter.dtype if parameter is not None else hidden_states.dtype
    predictions = []
    with torch.inference_mode():
        for start in range(0, int(hidden_states.shape[0]), 64):
            chunk = hidden_states[start : start + 64].to(dtype=target_dtype)
            logits = output_embeddings(final_norm(chunk))
            predictions.extend(int(token_id) for token_id in torch.argmax(logits, dim=-1).detach().cpu().tolist())
    return predictions


def _final_model_norm(model: object) -> object:
    candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(getattr(model, "transformer", None), "ln_f", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "final_layer_norm", None),
        getattr(getattr(model, "gpt_neox", None), "final_layer_norm", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            return candidate
    raise ValueError(f"Unsupported final normalization layout for {type(model).__name__}")


def _feature_dead(row: sqlite3.Row) -> bool:
    if "top3_dead" in row.keys() and row["top3_dead"] is not None:
        return bool(row["top3_dead"])
    if "top1_dead" in row.keys() and row["top1_dead"] is not None:
        return bool(row["top1_dead"])
    return bool(row["dead"])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
