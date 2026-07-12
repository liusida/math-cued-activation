from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import Future
from collections import Counter
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from ica_lens_v9.model_runtime import hidden_states_for_layer, load_runtime
from ica_lens_v9.features.probe import DEFAULT_DB_PATH, connect, list_meta, mini_histogram_path, probe_text
from ica_lens_v9.features.probe import load_feature_bundle
from ica_lens_v9.layers import layer_index
from ica_lens_v9.saes.counterparts import SAE_COUNTERPARTS
from ica_lens_v9.saes.loaders import load_counterpart_lightweight_sae


SERVER_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"
ASSETS_DIR = SERVER_DIR / "assets"
DB_PATH = Path(os.environ.get("ICA_V9_FEATURE_DB", str(DEFAULT_DB_PATH))).resolve()
NEURONPEDIA_DB_PATH = Path(
    os.environ.get("ICA_V9_NEURONPEDIA_DB", str(SERVER_DIR.parent / "neuronpedia_exports" / "neuronpedia.sqlite"))
).resolve()
_DOCUMENT_ACTIVATION_CACHE_MAX_SIZE = 64
_document_activation_cache: dict[tuple[Any, ...], Future] = {}
_document_activation_cache_lock = Lock()

app = FastAPI(title="v9 ICA Feature Explorer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ProbeRequest(BaseModel):
    run_id: str | None = None
    model_name: str | None = None
    layer: str
    text: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=32)
    keep_models: bool = True
    highlights: list[Any] = Field(default_factory=list)
    device: str = "auto"
    dtype: str = "auto"
    show_next_token: bool = False
    show_logit_lens: bool = False


class SaeProbeRequest(BaseModel):
    model_name: str
    layer: str
    text: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=128)
    keep_models: bool = True
    device: str = "auto"
    dtype: str = "auto"


class ManualLabelRequest(BaseModel):
    model: str
    layer: str
    feature: int = Field(ge=0)
    manual_label: str = Field(default="", max_length=10_000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(ASSETS_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "db_path": str(DB_PATH), "frontend": str(STATIC_DIR / "index.html")}


@app.get("/api/meta")
def meta(model: str | None = None) -> dict[str, Any]:
    try:
        if model:
            return _v5_meta(model)
        return list_meta(DB_PATH)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/probe")
def probe(request: ProbeRequest) -> dict[str, Any]:
    run_id = request.run_id or _run_id_for_model(request.model_name)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id/model: {request.run_id or request.model_name}")
    layers = {layer["layer"] for layer in run["layers"]}
    if request.layer not in layers:
        raise HTTPException(status_code=404, detail=f"Unknown layer for run: {request.layer}")
    try:
        out = probe_text(
            run_id=run_id,
            layer=request.layer,
            text=request.text,
            top_k=request.top_k,
            model_id=run["model_id"],
            device=request.device,
            dtype=request.dtype,
            show_next_token=request.show_next_token,
            show_logit_lens=request.show_logit_lens,
            db_path=DB_PATH,
        )
        if request.model_name is not None:
            return _to_v5_probe_response(out, model_name=request.model_name)
        return out
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/sae-meta")
def sae_meta(model: str) -> dict[str, Any]:
    run_id = _run_id_for_model(model)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    model_name = _model_name_for_run(run_id, run.get("model_short_name"))
    counterpart = SAE_COUNTERPARTS.get(model_name)
    if counterpart is None:
        raise HTTPException(status_code=404, detail=f"No SAE counterpart configured for {model_name}")
    layers = [str(layer["layer"]) for layer in run["layers"]]
    return {
        "model_name": model_name,
        "run_id": run_id,
        "model_id": run["model_id"],
        "layers": layers,
        "sae": {
            "source": counterpart.source,
            "repo_id": counterpart.repo_id,
            "sae_model_name": counterpart.sae_model_name,
            "hook_name_template": counterpart.hook_name_template,
            "width": _counterpart_width(counterpart),
            "hidden_size": counterpart.hidden_size,
            "activation": counterpart.activation,
            "top_k": counterpart.top_k,
            "normalize_activations": counterpart.normalize_activations,
            "apply_b_dec_to_input": counterpart.apply_b_dec_to_input,
            "checkpoint_format": counterpart.checkpoint_format,
        },
    }


@app.post("/api/sae-probe")
def sae_probe(request: SaeProbeRequest) -> dict[str, Any]:
    run_id = _run_id_for_model(request.model_name)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model_name}")
    model_name = _model_name_for_run(run_id, run.get("model_short_name"))
    layers = {str(layer["layer"]) for layer in run["layers"]}
    if request.layer not in layers:
        raise HTTPException(status_code=404, detail=f"Unknown layer for run: {request.layer}")
    counterpart = SAE_COUNTERPARTS.get(model_name)
    if counterpart is None:
        raise HTTPException(status_code=404, detail=f"No SAE counterpart configured for {model_name}")
    index = layer_index(request.layer)
    if index is None or index < 0:
        raise HTTPException(status_code=400, detail=f"SAE probe requires a numbered transformer layer: {request.layer}")
    try:
        runtime = load_runtime(run["model_id"], request.device, request.dtype)
        sae_name, sae = _load_sae_for_probe(
            model_name=model_name,
            layer=request.layer,
            device=str(runtime.device),
            dtype=_server_torch_dtype(request.dtype),
        )
        encoded = runtime.tokenizer(request.text, return_tensors="pt", truncation=True)
        inputs = {key: value.to(runtime.device) for key, value in encoded.items()}
        hidden = hidden_states_for_layer(runtime.model, request.layer, inputs)
        with torch.no_grad():
            acts = sae.encode(hidden.to(device=sae.device, dtype=sae.dtype))
        k = min(int(request.top_k), int(acts.shape[-1]))
        values, indices = torch.topk(acts, k=k, dim=-1)
        values_cpu = values.detach().cpu()
        indices_cpu = indices.detach().cpu()
        input_ids = inputs["input_ids"][0].detach().cpu().tolist()
        token_texts = [_decode_token(runtime.tokenizer, token_id) for token_id in input_ids]
        feature_ids = sorted({int(feature_id) for feature_id in indices_cpu.flatten().tolist()})
        neuronpedia_identity = _neuronpedia_identity(counterpart.repo_id)
        neuronpedia_labels = _neuronpedia_labels(
            sae_model_name=counterpart.sae_model_name,
            neuronpedia_identity=neuronpedia_identity,
            layer_index=int(index),
            feature_ids=feature_ids,
        )
        auto_labels = _sae_auto_annotation_labels(
            model_name=model_name,
            layer=request.layer,
            feature_ids=feature_ids,
        )
        tokens = []
        for token_index, (token_id, token_text) in enumerate(zip(input_ids, token_texts, strict=True)):
            top = []
            for rank in range(k):
                feature_id = int(indices_cpu[token_index, rank].item())
                activation = float(values_cpu[token_index, rank].item())
                top.append(
                    {
                        "feature": feature_id,
                        "component": feature_id,
                        "rank": rank,
                        "activation": activation,
                        "score": activation,
                    }
                )
                label = neuronpedia_labels.get(feature_id)
                if label:
                    top[-1]["neuronpedia_label"] = label
                auto_label = auto_labels.get(feature_id)
                if auto_label:
                    top[-1]["auto_annotation"] = auto_label
                    top[-1]["auto_annotation_label"] = auto_label.get("label")
                    top[-1]["auto_annotation_simple_label"] = auto_label.get("simple_label")
                url = _neuronpedia_url(
                    neuronpedia_identity=neuronpedia_identity,
                    layer_index=int(index),
                    feature_id=feature_id,
                )
                if url:
                    top[-1]["neuronpedia_url"] = url
            tokens.append(
                {
                    "position": int(token_index),
                    "token_index": int(token_index),
                    "token_id": int(token_id),
                    "token": token_text,
                    "token_text": token_text,
                    "top": top,
                }
            )
        return {
            "model_name": model_name,
            "run_id": run_id,
            "model_id": run["model_id"],
            "layer": request.layer,
            "top_k": k,
            "sae_name": sae_name,
            "sae": {
                "activation": counterpart.activation,
                "top_k": counterpart.top_k,
                "normalize_activations": counterpart.normalize_activations,
                "apply_b_dec_to_input": counterpart.apply_b_dec_to_input,
                "d_sae": int(sae.cfg.d_sae),
            },
            "tokens": tokens,
            "truncated": False,
            "max_length": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models")
def models() -> dict[str, Any]:
    payload = list_meta(DB_PATH)
    models_out = []
    preferred_run_id = "math_cued_vibethinker_only_layer32_c2048_iter100"
    runs = sorted(payload["runs"], key=lambda run: (run["run_id"] != preferred_run_id, run["run_id"]))
    for run in runs:
        model_name = _model_name_for_run(run["run_id"], run.get("model_short_name"))
        models_out.append(
            {
                "model_name": model_name,
                "display_name": run["display_name"],
                "model_id": run["model_id"],
                "context_length": 1024,
                "has_examples": False,
                "probe_supported": True,
                "ica_layers": [layer["layer"] for layer in run["layers"]],
                "run_id": run["run_id"],
            }
        )
    return {"models": models_out}


@app.get("/api/layers")
def layers(model: str) -> dict[str, Any]:
    return {"layers": _v5_meta(model)["layers"]}


@app.get("/api/features")
def features(model: str, layer: str | None = None, search: str | None = None) -> dict[str, Any]:
    run_id = _run_id_for_model(model)
    with connect(DB_PATH) as conn:
        _ensure_feature_manual_label_column(conn)
        rows = conn.execute(
            """
            SELECT feature_id, layer, dead, kurtosis, excess_kurtosis,
                   activation_frequency, activation_frequency_gt_1,
                   top1_count, top1_frequency, top1_dead,
                   top3_count, top3_frequency, top3_dead,
                   max_activation, effective_context_mean,
                   source_component_index, source_side,
                   annotation_label, annotation_simple_label, manual_label, annotation_description,
                   annotation_reasoning, annotation_confidence, annotation_provider,
                   annotation_model, annotation_path, annotation_raw_response_path
            FROM features
            WHERE run_id = ? AND (? IS NULL OR layer = ?)
            ORDER BY layer, feature_id
            """,
            (run_id, layer, layer),
        ).fetchall()
    query = (search or "").strip().lower()
    out = []
    for row in rows:
        item = _feature_row(model, row)
        if query and query not in str(item["feature"]).lower() and query not in str(item["layer"]).lower():
            continue
        out.append(item)
    return {"features": out, "components": out}


@app.get("/api/components")
def components(model: str, layer: str | None = None, search: str | None = None) -> dict[str, Any]:
    return features(model=model, layer=layer, search=search)


@app.get("/api/feature-token-stats")
def feature_token_stats(model: str, layer: str, feature: int | None = None, component: int | None = None) -> dict[str, Any]:
    feature_id = feature if feature is not None else component
    if feature_id is None:
        raise HTTPException(status_code=400, detail="feature is required")
    run_id = _run_id_for_model(model)
    with connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT feature_id
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, layer, int(feature_id)),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown feature: {model} {layer} F{feature_id}")
    evidence = _feature_evidence_from_db(run_id=run_id, layer=layer, feature_id=int(feature_id))
    examples = evidence.get("examples") if isinstance(evidence, dict) else []
    counts: Counter[str] = Counter()
    for example in examples if isinstance(examples, list) else []:
        token = _example_token_text(example)
        if token:
            counts[token] += 1
    tokens = [
        {"token": token, "count": count}
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    ]
    return {
        "model": model,
        "layer": layer,
        "feature": feature_id,
        "component": feature_id,
        "source": "annotation_evidence_examples" if evidence is not None else "missing_annotation_evidence",
        "tokens": tokens,
    }


@app.get("/api/component-token-stats")
def component_token_stats(model: str, layer: str, component: int) -> dict[str, Any]:
    return feature_token_stats(model=model, layer=layer, feature=component)


@app.get("/api/feature-neighbors")
def feature_neighbors(model: str, layer: str, feature: int | None = None, component: int | None = None) -> dict[str, Any]:
    feature_id = feature if feature is not None else component
    if feature_id is None:
        raise HTTPException(status_code=400, detail="feature is required")
    run_id = _run_id_for_model(model)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    layers = [str(item["layer"]) for item in run["layers"]]
    if layer not in layers:
        raise HTTPException(status_code=404, detail=f"Unknown layer for run: {layer}")
    feature_interface_dir = _feature_interface_dir(run_id)
    if feature_interface_dir is None:
        return {"model": model, "layer": layer, "feature": feature_id, "component": feature_id, "neighbors": []}

    layer_index = layers.index(layer)
    source_directions = _feature_directions(str(feature_interface_dir), layer)
    feature_id = int(feature_id)
    if feature_id < 0 or feature_id >= int(source_directions.shape[0]):
        raise HTTPException(status_code=404, detail=f"Unknown feature: {model} {layer} F{feature_id}")
    source = source_directions[feature_id]
    neighbors = []
    if layer_index > 0:
        neighbors.append(
            _closest_feature_neighbor(
                model=model,
                run_id=run_id,
                feature_interface_dir=feature_interface_dir,
                source=source,
                direction="prev",
                neighbor_layer=layers[layer_index - 1],
            )
        )
    if layer_index + 1 < len(layers):
        neighbors.append(
            _closest_feature_neighbor(
                model=model,
                run_id=run_id,
                feature_interface_dir=feature_interface_dir,
                source=source,
                direction="next",
                neighbor_layer=layers[layer_index + 1],
            )
        )
    return {"model": model, "layer": layer, "feature": feature_id, "component": feature_id, "neighbors": [item for item in neighbors if item is not None]}


@app.get("/api/component-neighbors")
def component_neighbors(model: str, layer: str, component: int) -> dict[str, Any]:
    return feature_neighbors(model=model, layer=layer, feature=component)


@app.get("/api/annotations/component")
def annotation_component(model: str, layer: str, component: int) -> dict[str, Any]:
    return {
        "model_name": model,
        "layer": layer,
        "component": component,
        "positive_label": "",
        "negative_label": "",
        "summary": "",
        "notes": "",
    }


@app.get("/api/feature")
def feature_detail(
    model: str,
    layer: str,
    feature: int | None = None,
    component: int | None = None,
) -> dict[str, Any]:
    feature_id = feature if feature is not None else component
    if feature_id is None:
        raise HTTPException(status_code=400, detail="feature is required")
    run_id = _run_id_for_model(model)
    with connect(DB_PATH) as conn:
        _ensure_feature_manual_label_column(conn)
        row = conn.execute(
            """
            SELECT feature_id, layer, source_feature_id, source_component_index,
                   source_sign, source_side, dead, kurtosis, excess_kurtosis,
                   activation_frequency, activation_frequency_gt_1, mean, variance,
                   top1_count, top1_frequency, top1_dead,
                   top3_count, top3_frequency, top3_dead,
                   max_activation, effective_context_mean,
                   effective_receptive_field_json, annotation_evidence_path,
                   annotation_label, annotation_simple_label, manual_label, annotation_description,
                   annotation_reasoning, annotation_confidence, annotation_provider,
                   annotation_model, annotation_path, annotation_raw_response_path,
                   mini_histogram_svg_path
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, layer, int(feature_id)),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Unknown feature: {model} {layer} F{feature_id}")
        opposite = conn.execute(
            """
            SELECT feature_id, layer, source_feature_id, source_component_index,
                   source_sign, source_side, dead, kurtosis, excess_kurtosis,
                   activation_frequency, activation_frequency_gt_1, mean, variance,
                   top1_count, top1_frequency, top1_dead,
                   top3_count, top3_frequency, top3_dead,
                   max_activation, effective_context_mean,
                   effective_receptive_field_json, annotation_evidence_path,
                   annotation_label, annotation_simple_label, manual_label, annotation_description,
                   annotation_reasoning, annotation_confidence, annotation_provider,
                   annotation_model, annotation_path, annotation_raw_response_path,
                   mini_histogram_svg_path
            FROM features
            WHERE run_id = ? AND layer = ? AND source_component_index = ? AND source_sign = ?
            """,
            (run_id, layer, int(row["source_component_index"]), -int(row["source_sign"])),
        ).fetchone()

    evidence = _feature_evidence_from_db(run_id=run_id, layer=layer, feature_id=int(feature_id))
    return {
        "model": model,
        "run_id": run_id,
        "layer": layer,
        "feature": _feature_detail_row(model, run_id, row),
        "opposite_feature": _feature_detail_row(model, run_id, opposite) if opposite is not None else None,
        "annotations": _annotation_packets_for_feature(
            run_id=run_id,
            layer=layer,
            feature_id=int(feature_id),
            active_path=row["annotation_path"],
        ),
        "refinements": _refinement_packets_for_feature(
            run_id=run_id,
            layer=layer,
            feature_id=int(feature_id),
        ),
        "evidence_path": row["annotation_evidence_path"],
        "evidence": evidence,
    }


@app.put("/api/feature/manual-label")
def update_feature_manual_label(request: ManualLabelRequest) -> dict[str, Any]:
    run_id = _run_id_for_model(request.model)
    manual_label = request.manual_label.strip()
    with connect(DB_PATH) as conn:
        _ensure_feature_manual_label_column(conn)
        cursor = conn.execute(
            """
            UPDATE features
            SET manual_label = ?
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (manual_label or None, run_id, request.layer, int(request.feature)),
        )
        if cursor.rowcount != 1:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown feature: {request.model} {request.layer} F{request.feature}",
            )
        conn.commit()
    return {
        "model": request.model,
        "run_id": run_id,
        "layer": request.layer,
        "feature": int(request.feature),
        "manual_label": manual_label,
    }


@app.get("/api/sae-feature")
def sae_feature_detail(
    model: str,
    layer: str,
    feature: int,
) -> dict[str, Any]:
    model_name = _normalize_sae_model_name(model)
    feature_id = int(feature)
    layer_i = layer_index(layer)
    if layer_i is None or layer_i < 0:
        raise HTTPException(status_code=400, detail=f"SAE feature detail requires a numbered layer: {layer}")
    if model_name not in SAE_COUNTERPARTS:
        raise HTTPException(status_code=404, detail=f"Unknown SAE model: {model}")
    counterpart = SAE_COUNTERPARTS[model_name]
    neuronpedia_identity = _neuronpedia_identity(counterpart.repo_id)
    neuronpedia_labels = _neuronpedia_labels(
        sae_model_name=counterpart.sae_model_name,
        neuronpedia_identity=neuronpedia_identity,
        layer_index=int(layer_i),
        feature_ids=[feature_id],
    )
    neuronpedia_url = _neuronpedia_url(
        neuronpedia_identity=neuronpedia_identity,
        layer_index=int(layer_i),
        feature_id=feature_id,
    )
    current_annotations = _sae_current_annotation_packets_for_feature(model_name=model_name, layer=layer, feature_id=feature_id)
    refinements = _sae_refinement_packets_for_feature(model_name=model_name, layer=layer, feature_id=feature_id)
    annotations = _sae_annotation_packets_for_feature(model_name=model_name, layer=layer, feature_id=feature_id)
    current = next((item for item in current_annotations if item.get("active")), current_annotations[0] if current_annotations else {})
    evidence_path = _sae_evidence_path(model_name=model_name, layer=layer, feature_id=feature_id)
    evidence = _json_file(evidence_path)
    return {
        "kind": "sae",
        "model": model_name,
        "layer": layer,
        "feature": _sae_feature_detail_row(
            model_name=model_name,
            layer=layer,
            feature_id=feature_id,
            counterpart=counterpart,
            neuronpedia_label=neuronpedia_labels.get(feature_id, ""),
            neuronpedia_url=neuronpedia_url or "",
            current=current,
            evidence_path=evidence_path,
        ),
        "opposite_feature": None,
        "annotations": annotations,
        "refinements": refinements,
        "evidence_path": str(evidence_path) if evidence_path and evidence_path.is_file() else "",
        "evidence": evidence,
    }


@app.get("/api/document")
def document(model: str, doc_id: str, position: int | None = None, layer: str | None = None, feature: int | None = None) -> dict[str, Any]:
    run_id = _run_id_for_model(model)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    manifest_path = _activation_manifest_path(run_id)
    try:
        if manifest_path is not None:
            text = _load_document_text(str(manifest_path), doc_id)
        else:
            text = _load_document_text_from_feature_evidence(run_id=run_id, layer=layer, feature=feature, doc_id=doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    target_span = None
    if position is not None:
        try:
            target_span = _document_token_span(run["model_id"], text, int(position))
        except Exception as exc:
            target_span = {"position": int(position), "error": str(exc)}
    context_length = _capture_context_length(str(manifest_path)) if manifest_path is not None else 4096
    tokenized_prefix = _tokenized_document_prefix(run["model_id"], text, max_length=context_length)
    return {
        "model": model,
        "run_id": run_id,
        "doc_id": str(doc_id),
        "text": text,
        "target_span": target_span,
        "tokenized_prefix": tokenized_prefix,
    }


@app.get("/api/document-activations")
def document_activations(
    model: str,
    layer: str,
    feature: int,
    doc_id: str,
    min_relative: float = 0.2,
    position: int | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    run_id = _run_id_for_model(model)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    manifest_path = _activation_manifest_path(run_id)
    if manifest_path is None:
        source_path = _math_cued_source_path_for_doc(
            run_id=run_id,
            layer=layer,
            feature_id=int(feature),
            doc_id=doc_id,
        )
        if source_path is None or not source_path.is_file():
            raise HTTPException(status_code=404, detail=f"No captured activations for {model} document {doc_id}")
        try:
            payload = _math_cued_document_activation_payload(
                source_path=source_path,
                model_id=run["model_id"],
                run_id=run_id,
                layer=layer,
                feature_id=int(feature),
                min_relative=float(min_relative),
                target_position=position,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"model": model, "run_id": run_id, "layer": layer, "feature": int(feature), "doc_id": doc_id, **payload}
    try:
        numeric_doc_id = int(doc_id)
        payload = _cached_document_activation_payload(
            model_id=run["model_id"],
            run_id=run_id,
            model=model,
            layer=layer,
            feature_id=int(feature),
            doc_id=numeric_doc_id,
            manifest_path=str(manifest_path),
            context_length=_capture_context_length(str(manifest_path)),
            min_relative=float(min_relative),
            target_position=position,
            device=device,
            dtype=dtype,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"model": model, "run_id": run_id, "layer": layer, "feature": int(feature), "doc_id": numeric_doc_id, **payload}


@app.get("/api/document-text")
def document_text(model: str, doc_id: str) -> PlainTextResponse:
    payload = document(model=model, doc_id=doc_id)
    return PlainTextResponse(str(payload["text"]))


@app.get("/annotate")
def annotate() -> FileResponse:
    return FileResponse(STATIC_DIR / "feature.html")


@app.get("/document")
def document_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "document.html")


@app.get("/feature")
def feature_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "feature.html")


@app.get("/sae-feature")
def sae_feature_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "feature.html")


@app.get("/component")
def component_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "component.html")


@app.get("/stats")
def stats_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "stats.html")


@app.get("/sae-explorer")
def sae_explorer_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "sae_explorer.html")


@app.get("/random-components")
def random_components_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "random_components.html")


@app.get("/api/mini-histogram/{run_id}/{layer}/{feature_id}")
def mini_histogram(run_id: str, layer: str, feature_id: int) -> FileResponse:
    try:
        path = mini_histogram_path(run_id, layer, feature_id, DB_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Mini histogram file missing: {path}")
    return FileResponse(path, media_type="image/svg+xml")


def _run_for_id(run_id: str | None) -> dict[str, Any] | None:
    if run_id is None:
        return None
    for run in list_meta(DB_PATH)["runs"]:
        if run["run_id"] == run_id:
            return run
    return None


@lru_cache(maxsize=24)
def _load_sae_for_probe(*, model_name: str, layer: str, device: str, dtype: torch.dtype) -> tuple[str, Any]:
    counterpart = SAE_COUNTERPARTS[model_name]
    index = layer_index(layer)
    if index is None or index < 0:
        raise ValueError(f"SAE probe requires a numbered transformer layer: {layer}")
    return load_counterpart_lightweight_sae(
        counterpart=counterpart,
        layer_index=int(index),
        device=device,
        dtype=dtype,
    )


def _server_torch_dtype(name: str) -> torch.dtype:
    lowered = str(name or "auto").lower()
    if lowered in {"auto", "float32", "fp32"}:
        return torch.float32
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _counterpart_width(counterpart: Any) -> int | None:
    if counterpart.source == "custom_checkpoint":
        if counterpart.top_k == 50 and "W32K" in counterpart.repo_id:
            return 32768
        return None
    text = " ".join(str(value or "") for value in [counterpart.repo_id, counterpart.release_pattern, counterpart.checkpoint_template])
    if "32k" in text.lower():
        return 32768
    if "16k" in text.lower() or any("width_16k" in path for path in counterpart.layer_checkpoints.values()):
        return 16384
    return None


def _neuronpedia_labels(
    *,
    sae_model_name: str,
    neuronpedia_identity: tuple[str, str] | None,
    layer_index: int,
    feature_ids: list[int],
) -> dict[int, str]:
    if not feature_ids or not NEURONPEDIA_DB_PATH.is_file():
        return {}
    if neuronpedia_identity is None:
        return {}
    _model_slug, sae_set = neuronpedia_identity
    placeholders = ",".join("?" for _ in feature_ids)
    query = f"""
        SELECT feature_id, description
        FROM neuronpedia_labels
        WHERE model_name = ?
          AND sae_set = ?
          AND layer_index = ?
          AND feature_id IN ({placeholders})
    """
    try:
        with sqlite3.connect(NEURONPEDIA_DB_PATH) as conn:
            rows = conn.execute(query, (sae_model_name, sae_set, layer_index, *feature_ids)).fetchall()
    except sqlite3.Error:
        return {}
    return {int(feature_id): str(description) for feature_id, description in rows if str(description).strip()}


def _sae_auto_annotation_labels(
    *,
    model_name: str,
    layer: str,
    feature_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not feature_ids or not DB_PATH.is_file():
        return {}
    placeholders = ",".join("?" for _ in feature_ids)
    query = f"""
        WITH latest_refinement AS (
          SELECT r.*
          FROM sae_feature_annotation_refinements r
          WHERE r.model_name = ?
            AND r.layer = ?
            AND r.feature_id IN ({placeholders})
            AND r.label != ''
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
        ),
        current_labels AS (
          SELECT feature_id, provider_label, provider_model, label, simple_label,
                 description, reasoning, confidence, annotation_path, created_at
          FROM latest_refinement
          UNION ALL
          SELECT a.feature_id, a.provider_label, a.provider_model, a.label, a.simple_label,
                 a.description, a.reasoning, a.confidence, a.annotation_path, a.created_at
          FROM sae_feature_annotations a
          WHERE a.model_name = ?
            AND a.layer = ?
            AND a.feature_id IN ({placeholders})
            AND NOT EXISTS (
              SELECT 1
              FROM latest_refinement r
              WHERE r.feature_id = a.feature_id
                AND r.provider_label = a.provider_label
            )
        )
        SELECT feature_id, provider_label, provider_model, label, simple_label,
               description, reasoning, confidence, annotation_path
        FROM current_labels
        ORDER BY
          CASE provider_label
            WHEN 'mi_pro' THEN 0
            WHEN 'mi' THEN 1
            WHEN 'local-qwen' THEN 2
            ELSE 3
          END,
          created_at DESC
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (model_name, layer, *feature_ids, model_name, layer, *feature_ids)).fetchall()
    except sqlite3.Error:
        return {}
    labels: dict[int, dict[str, Any]] = {}
    for row in rows:
        feature_id = int(row["feature_id"])
        if feature_id in labels:
            continue
        label = str(row["label"] or "").strip()
        simple_label = str(row["simple_label"] or "").strip()
        if not label and not simple_label:
            continue
        labels[feature_id] = {
            "provider_label": str(row["provider_label"] or ""),
            "model": str(row["provider_model"] or ""),
            "label": label,
            "simple_label": simple_label,
            "description": str(row["description"] or ""),
            "reasoning": str(row["reasoning"] or ""),
            "confidence": str(row["confidence"] or ""),
            "annotation_path": str(row["annotation_path"] or ""),
        }
    return labels


def _normalize_sae_model_name(model: str) -> str:
    aliases = {"gpt": "gpt2", "gemma": "gemma2_2b", "qwen": "qwen3_5_2b_base"}
    return aliases.get(str(model), str(model))


def _sae_evidence_path(*, model_name: str, layer: str, feature_id: int) -> Path:
    return SERVER_DIR.parent / "results" / "auto_annotation_sae" / "evidence" / model_name / layer / f"F{int(feature_id):06d}" / "compact_evidence.json"


def _json_file(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _neuronpedia_identity(counterpart_repo: str) -> tuple[str, str] | None:
    repo = str(counterpart_repo).lower()
    if "gpt2-small-oai-v5-32k-resid-post" in repo:
        return ("gpt2-small", "res_post_32k-oai")
    if "gemma-scope-2b-pt-res" in repo:
        return ("gemma-2-2b", "gemmascope-res-16k")
    if "sae-res-qwen3.5-2b-base-w32k" in repo:
        return ("qwen3.5-2b-pt", "qwenscope-res-32k")
    return None


def _neuronpedia_url(
    *,
    neuronpedia_identity: tuple[str, str] | None,
    layer_index: int,
    feature_id: int,
) -> str | None:
    if neuronpedia_identity is None:
        return None
    model_slug, sae_set = neuronpedia_identity
    return f"https://www.neuronpedia.org/{model_slug}/{layer_index}-{sae_set}/{feature_id}"


def _decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    if text == "":
        return f"<{int(token_id)}>"
    return text


def _feature_interface_dir(run_id: str) -> Path | None:
    with connect(DB_PATH) as conn:
        row = conn.execute("SELECT feature_interface_dir FROM model_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or not row["feature_interface_dir"]:
        return None
    path = Path(str(row["feature_interface_dir"]))
    if not path.is_absolute():
        path = (SERVER_DIR.parent / path).resolve()
    return path if path.is_dir() else None


@lru_cache(maxsize=96)
def _feature_directions(feature_interface_dir: str, layer: str) -> torch.Tensor:
    path = Path(feature_interface_dir) / f"{layer}_features.pt"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing feature artifact: {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    directions = artifact["tensors"]["feature_directions"].detach().cpu().to(torch.float32)
    return torch.nn.functional.normalize(directions, dim=1, eps=1e-12)


def _closest_feature_neighbor(
    *,
    model: str,
    run_id: str,
    feature_interface_dir: Path,
    source: torch.Tensor,
    direction: str,
    neighbor_layer: str,
) -> dict[str, Any] | None:
    neighbor_directions = _feature_directions(str(feature_interface_dir), neighbor_layer)
    cosine = neighbor_directions @ source.to(torch.float32)
    if cosine.numel() == 0:
        return None
    abs_cosine, index = torch.max(torch.abs(cosine), dim=0)
    feature_id = int(index.item())
    signed_cosine = float(cosine[feature_id].item())
    with connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT feature_id, source_sign, source_side, kurtosis, excess_kurtosis,
                   effective_context_mean
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, neighbor_layer, feature_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "direction": direction,
        "model_name": model,
        "neighbor_layer": neighbor_layer,
        "neighbor_component": feature_id,
        "neighbor_feature": feature_id,
        "neighbor_sign": 1 if signed_cosine >= 0 else -1,
        "cosine": signed_cosine,
        "abs_cosine": float(abs_cosine.item()),
        "positive_label": f"F{feature_id}",
        "negative_label": "",
        "positive_confidence": "",
        "negative_confidence": "",
        "positive_types": [],
        "negative_types": [],
        "source_sign": int(row["source_sign"]),
        "source_side": str(row["source_side"]),
        "kurtosis": float(row["kurtosis"]),
        "excess_kurtosis": float(row["excess_kurtosis"]),
        "effective_context_mean": _optional_float(row["effective_context_mean"]),
    }


def _run_id_for_model(model_name: str | None) -> str:
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name or run_id is required")
    aliases = {
        "gpt2": "gpt2_tok1000000_c768_iter200",
        "gpt2-small": "gpt2_tok1000000_c768_iter200",
        "gemma2_2b": "gemma2_2b_tok1000000_c2304_iter200",
        "qwen3_5_2b_base": "qwen3_5_2b_base_tok1000000_c2048_iter200",
    }
    if model_name in aliases:
        return aliases[model_name]
    for run in list_meta(DB_PATH)["runs"]:
        if model_name in {run["run_id"], run.get("model_short_name"), run.get("display_name")}:
            return str(run["run_id"])
    raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")


def _model_name_for_run(run_id: str, short_name: str | None) -> str:
    if run_id == "gpt2_tok1000000_c768_iter200":
        return "gpt2"
    if run_id == "gemma2_2b_tok1000000_c2304_iter200":
        return "gemma2_2b"
    if run_id == "qwen3_5_2b_base_tok1000000_c2048_iter200":
        return "qwen3_5_2b_base"
    return short_name or run_id


def _v5_meta(model: str) -> dict[str, Any]:
    run_id = _run_id_for_model(model)
    run = _run_for_id(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    layers = [layer["layer"] for layer in run["layers"]]
    return {
        "model_name": model,
        "run_id": run_id,
        "model_id": run["model_id"],
        "display_name": run["display_name"],
        "layers": layers,
        "ica_layers": layers,
        "probe_supported": True,
    }


def _feature_row(model: str, row: sqlite3.Row) -> dict[str, Any]:
    side = str(row["source_side"])
    source_component = int(row["source_component_index"])
    feature_id = int(row["feature_id"])
    label = _annotation_label(row) or f"F{feature_id}"
    confidence = _annotation_confidence(row)
    return {
        "model_name": model,
        "layer": row["layer"],
        "feature": feature_id,
        "component": feature_id,
        "positive_label": label,
        "negative_label": "",
        "positive_confidence": confidence,
        "negative_confidence": "",
        "positive_types": [],
        "negative_types": [],
        "summary": _annotation_description(row) or f"source C{source_component} {side}",
        "notes": _annotation_reasoning(row),
        "annotation_label": _optional_text(row, "annotation_label"),
        "annotation_simple_label": _optional_text(row, "annotation_simple_label"),
        "manual_label": _optional_text(row, "manual_label"),
        "annotation_description": _optional_text(row, "annotation_description"),
        "annotation_reasoning": _optional_text(row, "annotation_reasoning"),
        "annotation_confidence": confidence,
        "annotation_provider": _optional_text(row, "annotation_provider"),
        "annotation_model": _optional_text(row, "annotation_model"),
        "annotation_path": _optional_text(row, "annotation_path"),
        "annotation_raw_response_path": _optional_text(row, "annotation_raw_response_path"),
        "dead": _feature_dead(row),
        "kurtosis_dead": bool(row["dead"]),
        "kurtosis": float(row["kurtosis"]),
        "excess_kurtosis": float(row["excess_kurtosis"]),
        "activation_frequency": float(row["activation_frequency"]),
        "activation_frequency_gt_1": _optional_float(row["activation_frequency_gt_1"]),
        "top1_count": _optional_int(row["top1_count"]),
        "top1_frequency": _optional_float(row["top1_frequency"]),
        "top1_dead": _optional_bool(row["top1_dead"]),
        "top3_count": _optional_int(row["top3_count"]),
        "top3_frequency": _optional_float(row["top3_frequency"]),
        "top3_dead": _optional_bool(row["top3_dead"]),
        "max_activation": float(row["max_activation"]),
        "effective_context_mean": _optional_float(row["effective_context_mean"]),
        "source_component_index": source_component,
        "source_side": side,
        "mini_histogram_url": _mini_histogram_url(_run_id_for_model(model), str(row["layer"]), feature_id),
    }


def _to_v5_probe_response(out: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    feature_ids = sorted(
        {
            int(feature["feature_id"])
            for token in out["tokens"]
            for feature in token["top_features"]
        }
    )
    annotation_lookup = _feature_annotation_lookup(
        run_id=str(out["run_id"]),
        layer=str(out["layer"]),
        feature_ids=feature_ids,
    )
    annotated: dict[int, dict[str, Any]] = {}
    tokens = []
    for token in out["tokens"]:
        top = []
        for feature in token["top_features"]:
            feature_id = int(feature["feature_id"])
            top.append(
                {
                    "feature": feature_id,
                    "component": feature_id,
                    "score": float(feature["activation"]),
                    "activation": float(feature["activation"]),
                    "effective_context_mean": feature.get("effective_context_mean"),
                    "source_component_index": int(feature["source_component_index"]),
                    "source_side": str(feature["source_side"]),
                }
            )
            annotation = annotation_lookup.get(feature_id, {})
            label = str(annotation.get("label") or "").strip() or f"F{feature_id}"
            simple_label = str(annotation.get("simple_label") or "").strip()
            manual_label = str(annotation.get("manual_label") or "").strip()
            confidence = str(annotation.get("confidence") or "").strip()
            annotated[feature_id] = {
                "feature": feature_id,
                "component": feature_id,
                "positive_label": label,
                "positive_simple_label": simple_label,
                "positive_manual_label": manual_label,
                "negative_label": "",
                "negative_simple_label": "",
                "positive_confidence": confidence,
                "negative_confidence": "",
                "positive_types": [],
                "negative_types": [],
                "summary": str(annotation.get("description") or ""),
                "notes": str(annotation.get("reasoning") or ""),
                "excess_kurtosis": float(feature["kurtosis"]) - 3.0,
                "effective_context_mean": feature.get("effective_context_mean"),
                "mini_histogram_url": feature["mini_histogram_url"],
            }
        tokens.append(
            {
                "position": int(token["token_index"]),
                "token": token["text"],
                "token_text": token["text"],
                "token_id": int(token["token_id"]),
                "display_text": token.get("display_text"),
                "display_piece_index": token.get("display_piece_index"),
                "display_piece_count": token.get("display_piece_count"),
                "raw_token_piece": token.get("raw_token_piece"),
                "prediction": (
                    {
                        "token_id": int(token["prediction"]["token_id"]),
                        "token": str(token["prediction"]["text"]),
                        "token_text": str(token["prediction"]["text"]),
                        "source": str(token["prediction"]["source"]),
                    }
                    if token.get("prediction")
                    else None
                ),
                "logit_lens_prediction": (
                    {
                        "token_id": int(token["logit_lens_prediction"]["token_id"]),
                        "token": str(token["logit_lens_prediction"]["text"]),
                        "token_text": str(token["logit_lens_prediction"]["text"]),
                        "source": str(token["logit_lens_prediction"]["source"]),
                    }
                    if token.get("logit_lens_prediction")
                    else None
                ),
                "top": top,
            }
        )
    return {
        "model_name": model_name,
        "run_id": out["run_id"],
        "layer": out["layer"],
        "top_k": out["top_k"],
        "tokens": tokens,
        "annotated_features": list(annotated.values()),
        "annotated_components": list(annotated.values()),
        "truncated": False,
        "max_length": None,
    }


def _feature_annotation_lookup(*, run_id: str, layer: str, feature_ids: list[int]) -> dict[int, dict[str, str]]:
    if not feature_ids:
        return {}
    placeholders = ",".join("?" for _ in feature_ids)
    with connect(DB_PATH) as conn:
        _ensure_feature_manual_label_column(conn)
        rows = conn.execute(
            f"""
            SELECT feature_id, annotation_label, annotation_simple_label, manual_label,
                   annotation_description, annotation_reasoning, annotation_confidence
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id IN ({placeholders})
            """,
            (run_id, layer, *feature_ids),
        ).fetchall()
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        label = _optional_text(row, "annotation_label") or _optional_text(row, "annotation_simple_label")
        simple_label = _optional_text(row, "annotation_simple_label")
        manual_label = _optional_text(row, "manual_label")
        confidence = _annotation_confidence(row)
        if not label and not manual_label:
            continue
        out[int(row["feature_id"])] = {
            "label": label,
            "simple_label": simple_label,
            "manual_label": manual_label,
            "description": _annotation_description(row),
            "reasoning": _annotation_reasoning(row),
            "confidence": confidence,
        }
    return out


def _mini_histogram_url(run_id: str, layer: str, feature_id: int) -> str:
    return f"/api/mini-histogram/{run_id}/{layer}/{feature_id}"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_text(row: sqlite3.Row, key: str) -> str:
    if key not in row.keys() or row[key] is None:
        return ""
    return str(row[key]).strip()


def _ensure_feature_manual_label_column(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(features)").fetchall()}
    if "manual_label" not in columns:
        conn.execute("ALTER TABLE features ADD COLUMN manual_label TEXT")
        conn.commit()


def _annotation_label(row: sqlite3.Row) -> str:
    return _optional_text(row, "annotation_label") or _optional_text(row, "annotation_simple_label")


def _annotation_description(row: sqlite3.Row) -> str:
    return _optional_text(row, "annotation_description")


def _annotation_reasoning(row: sqlite3.Row) -> str:
    return _optional_text(row, "annotation_reasoning")


def _annotation_confidence(row: sqlite3.Row) -> str:
    confidence = _optional_text(row, "annotation_confidence").lower()
    return confidence if confidence in {"high", "medium", "low", "unclear"} else ""


def _feature_dead(row: sqlite3.Row) -> bool:
    if "top3_dead" in row.keys() and row["top3_dead"] is not None:
        return bool(row["top3_dead"])
    if "top1_dead" in row.keys() and row["top1_dead"] is not None:
        return bool(row["top1_dead"])
    return bool(row["dead"])


def _feature_detail_row(model: str, run_id: str, row: sqlite3.Row) -> dict[str, Any]:
    feature_id = int(row["feature_id"])
    confidence = _annotation_confidence(row)
    erf_json = None
    if row["effective_receptive_field_json"]:
        try:
            erf_json = json.loads(row["effective_receptive_field_json"])
        except json.JSONDecodeError:
            erf_json = None
    return {
        "model_name": model,
        "run_id": run_id,
        "layer": row["layer"],
        "feature_id": feature_id,
        "feature_label": f"F{feature_id:06d}",
        "annotation_label": _optional_text(row, "annotation_label"),
        "annotation_simple_label": _optional_text(row, "annotation_simple_label"),
        "manual_label": _optional_text(row, "manual_label"),
        "annotation_description": _optional_text(row, "annotation_description"),
        "annotation_reasoning": _optional_text(row, "annotation_reasoning"),
        "annotation_confidence": confidence,
        "annotation_provider": _optional_text(row, "annotation_provider"),
        "annotation_model": _optional_text(row, "annotation_model"),
        "annotation_path": _optional_text(row, "annotation_path"),
        "annotation_raw_response_path": _optional_text(row, "annotation_raw_response_path"),
        "source_feature_id": int(row["source_feature_id"]),
        "source_component_index": int(row["source_component_index"]),
        "source_sign": int(row["source_sign"]),
        "source_side": str(row["source_side"]),
        "dead": _feature_dead(row),
        "kurtosis_dead": bool(row["dead"]),
        "kurtosis": float(row["kurtosis"]),
        "excess_kurtosis": float(row["excess_kurtosis"]),
        "activation_frequency": float(row["activation_frequency"]),
        "activation_frequency_gt_1": _optional_float(row["activation_frequency_gt_1"]),
        "top1_count": _optional_int(row["top1_count"]),
        "top1_frequency": _optional_float(row["top1_frequency"]),
        "top1_dead": _optional_bool(row["top1_dead"]),
        "top3_count": _optional_int(row["top3_count"]),
        "top3_frequency": _optional_float(row["top3_frequency"]),
        "top3_dead": _optional_bool(row["top3_dead"]),
        "mean": float(row["mean"]),
        "variance": float(row["variance"]),
        "max_activation": float(row["max_activation"]),
        "effective_context_mean": _optional_float(row["effective_context_mean"]),
        "effective_receptive_field": erf_json,
        "annotation_evidence_path": row["annotation_evidence_path"],
        "mini_histogram_url": _mini_histogram_url(run_id, str(row["layer"]), feature_id),
    }


def _sae_feature_detail_row(
    *,
    model_name: str,
    layer: str,
    feature_id: int,
    counterpart: Any,
    neuronpedia_label: str,
    neuronpedia_url: str,
    current: dict[str, Any],
    evidence_path: Path | None,
) -> dict[str, Any]:
    label = str(current.get("label") or "").strip()
    simple_label = str(current.get("simple_label") or "").strip()
    return {
        "model_name": model_name,
        "layer": layer,
        "feature_id": int(feature_id),
        "feature_label": f"F{int(feature_id):06d}",
        "source_kind": "SAE counterpart",
        "sae_model_name": str(counterpart.sae_model_name),
        "sae_repo_id": str(counterpart.repo_id),
        "d_sae": _counterpart_width(counterpart),
        "sae_activation": str(counterpart.activation),
        "sae_top_k": int(counterpart.top_k) if counterpart.top_k is not None else None,
        "neuronpedia_label": str(neuronpedia_label or "").strip(),
        "neuronpedia_url": str(neuronpedia_url or "").strip(),
        "annotation_label": label or simple_label,
        "annotation_simple_label": simple_label,
        "annotation_description": str(current.get("description") or "").strip(),
        "annotation_reasoning": str(current.get("reasoning") or "").strip(),
        "annotation_confidence": str(current.get("confidence") or "").strip().lower(),
        "annotation_provider": str(current.get("provider_label") or current.get("provider") or "").strip(),
        "annotation_model": str(current.get("model") or "").strip(),
        "annotation_path": str(current.get("annotation_path") or "").strip(),
        "annotation_raw_response_path": str(current.get("raw_response_path") or "").strip(),
        "annotation_evidence_path": str(evidence_path) if evidence_path and evidence_path.is_file() else "",
    }


def _feature_evidence_from_db(*, run_id: str, layer: str, feature_id: int) -> dict[str, Any] | None:
    try:
        with connect(DB_PATH) as conn:
            row = conn.execute(
                """
                SELECT annotation_evidence_json
                FROM features
                WHERE run_id = ? AND layer = ? AND feature_id = ?
                """,
                (run_id, layer, int(feature_id)),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None or not row["annotation_evidence_json"]:
        return None
    try:
        evidence = json.loads(str(row["annotation_evidence_json"]))
    except json.JSONDecodeError:
        return None
    if isinstance(evidence, dict) and "examples" not in evidence and isinstance(evidence.get("erf_examples"), list):
        # Backward-compatible alias for pages/routes that still expect the old
        # compact evidence field name.
        evidence["examples"] = evidence["erf_examples"]
    return evidence if isinstance(evidence, dict) else None


def _annotation_packets_for_feature(
    *,
    run_id: str,
    layer: str,
    feature_id: int,
    active_path: Any,
) -> list[dict[str, Any]]:
    active = str(active_path or "").strip()
    try:
        with connect(DB_PATH) as conn:
            rows = _feature_annotation_rows(conn, run_id=run_id, layer=layer, feature_id=feature_id)
    except sqlite3.OperationalError:
        return []
    packets = [
        {
            "provider_label": _optional_text(row, "provider_label"),
            "provider": _optional_text(row, "provider"),
            "model": _optional_text(row, "model"),
            "created_at": _optional_text(row, "created_at"),
            "label": _optional_text(row, "label"),
            "simple_label": _optional_text(row, "simple_label"),
            "description": _optional_text(row, "description"),
            "reasoning": _optional_text(row, "reasoning"),
            "confidence": _optional_text(row, "confidence").lower(),
            "test_cases": _annotation_test_cases(row["test_cases_json"]),
            "annotation_path": _optional_text(row, "annotation_path"),
            "raw_response_path": _optional_text(row, "raw_response_path"),
            "active": bool(active) and _optional_text(row, "annotation_path") == active,
        }
        for row in rows
    ]
    packets.sort(key=lambda item: (not bool(item.get("active")), str(item.get("provider_label") or "")))
    return packets


def _refinement_packets_for_feature(*, run_id: str, layer: str, feature_id: int) -> list[dict[str, Any]]:
    try:
        with connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT provider_label, round_index, provider, model, created_at,
                       source_annotation_path, tests_path, request_path,
                       raw_response_path, annotation_path,
                       label_before_json, test_results_json,
                       annotation_label, annotation_simple_label,
                       annotation_description, annotation_reasoning,
                       annotation_confidence, annotation_test_cases_json
                FROM feature_annotation_refinements
                WHERE run_id = ? AND layer = ? AND feature_id = ?
                ORDER BY provider_label, round_index
                """,
                (run_id, layer, int(feature_id)),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    packets = []
    for row in rows:
        packets.append(
            {
                "provider_label": _optional_text(row, "provider_label"),
                "round_index": _optional_int(row["round_index"]),
                "provider": _optional_text(row, "provider"),
                "model": _optional_text(row, "model"),
                "created_at": _optional_text(row, "created_at"),
                "source_annotation_path": _optional_text(row, "source_annotation_path"),
                "tests_path": _optional_text(row, "tests_path"),
                "request_path": _optional_text(row, "request_path"),
                "raw_response_path": _optional_text(row, "raw_response_path"),
                "annotation_path": _optional_text(row, "annotation_path"),
                "label_before": _json_object(row["label_before_json"]),
                "test_results": _json_list(row["test_results_json"]),
                "annotation": {
                    "label": _optional_text(row, "annotation_label"),
                    "simple_label": _optional_text(row, "annotation_simple_label"),
                    "description": _optional_text(row, "annotation_description"),
                    "reasoning": _optional_text(row, "annotation_reasoning"),
                    "confidence": _optional_text(row, "annotation_confidence").lower(),
                    "test_cases": _annotation_test_cases(row["annotation_test_cases_json"]),
                },
            }
        )
    return packets


def _sae_annotation_packets_for_feature(*, model_name: str, layer: str, feature_id: int) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT provider_label, provider, provider_model, created_at,
                       label, simple_label, description, reasoning, confidence,
                       test_cases_json, annotation_path, raw_response_path,
                       NOT EXISTS (
                         SELECT 1
                         FROM sae_feature_annotation_refinements r
                         WHERE r.model_name = sae_feature_annotations.model_name
                           AND r.layer = sae_feature_annotations.layer
                           AND r.feature_id = sae_feature_annotations.feature_id
                           AND r.provider_label = sae_feature_annotations.provider_label
                           AND r.label != ''
                       ) AS is_active_initial
                FROM sae_feature_annotations
                WHERE model_name = ? AND layer = ? AND feature_id = ?
                ORDER BY CASE provider_label
                    WHEN 'mi_pro' THEN 0
                    WHEN 'mi' THEN 1
                    WHEN 'local-qwen' THEN 2
                    ELSE 3
                  END,
                  created_at DESC
                """,
                (model_name, layer, int(feature_id)),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    packets = [
        {
            "provider_label": _optional_text(row, "provider_label"),
            "provider": _optional_text(row, "provider"),
            "model": _optional_text(row, "provider_model"),
            "created_at": _optional_text(row, "created_at"),
            "label": _optional_text(row, "label"),
            "simple_label": _optional_text(row, "simple_label"),
            "description": _optional_text(row, "description"),
            "reasoning": _optional_text(row, "reasoning"),
            "confidence": _optional_text(row, "confidence").lower(),
            "test_cases": _annotation_test_cases(row["test_cases_json"]),
            "annotation_path": _optional_text(row, "annotation_path"),
            "raw_response_path": _optional_text(row, "raw_response_path"),
            "active": bool(row["is_active_initial"]),
        }
        for row in rows
    ]
    packets.sort(key=lambda item: (not bool(item.get("active")), str(item.get("provider_label") or "")))
    return packets


def _sae_current_annotation_packets_for_feature(*, model_name: str, layer: str, feature_id: int) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                WITH latest_refinement AS (
                  SELECT r.*
                  FROM sae_feature_annotation_refinements r
                  WHERE r.model_name = ?
                    AND r.layer = ?
                    AND r.feature_id = ?
                    AND r.label != ''
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
                ),
                current_labels AS (
                  SELECT provider_label, provider, provider_model, created_at,
                         label, simple_label, description, reasoning, confidence,
                         test_cases_json, annotation_path, raw_response_path
                  FROM latest_refinement
                  UNION ALL
                  SELECT a.provider_label, a.provider, a.provider_model, a.created_at,
                         a.label, a.simple_label, a.description, a.reasoning, a.confidence,
                         a.test_cases_json, a.annotation_path, a.raw_response_path
                  FROM sae_feature_annotations a
                  WHERE a.model_name = ?
                    AND a.layer = ?
                    AND a.feature_id = ?
                    AND NOT EXISTS (
                      SELECT 1
                      FROM latest_refinement r
                      WHERE r.provider_label = a.provider_label
                    )
                )
                SELECT *
                FROM current_labels
                ORDER BY CASE provider_label
                    WHEN 'mi_pro' THEN 0
                    WHEN 'mi' THEN 1
                    WHEN 'local-qwen' THEN 2
                    ELSE 3
                  END,
                  created_at DESC
                """,
                (model_name, layer, int(feature_id), model_name, layer, int(feature_id)),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {
            "provider_label": _optional_text(row, "provider_label"),
            "provider": _optional_text(row, "provider"),
            "model": _optional_text(row, "provider_model"),
            "created_at": _optional_text(row, "created_at"),
            "label": _optional_text(row, "label"),
            "simple_label": _optional_text(row, "simple_label"),
            "description": _optional_text(row, "description"),
            "reasoning": _optional_text(row, "reasoning"),
            "confidence": _optional_text(row, "confidence").lower(),
            "test_cases": _annotation_test_cases(row["test_cases_json"]),
            "annotation_path": _optional_text(row, "annotation_path"),
            "raw_response_path": _optional_text(row, "raw_response_path"),
            "active": True,
        }
        for row in rows
    ]


def _sae_refinement_packets_for_feature(*, model_name: str, layer: str, feature_id: int) -> list[dict[str, Any]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT provider_label, round_index, provider, provider_model, created_at,
                       source_annotation_path, tests_path, request_path,
                       raw_response_path, annotation_path, annotation_json,
                       label, simple_label, description, reasoning, confidence, test_cases_json
                FROM sae_feature_annotation_refinements
                WHERE model_name = ? AND layer = ? AND feature_id = ?
                ORDER BY provider_label, round_index
                """,
                (model_name, layer, int(feature_id)),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    packets = []
    for row in rows:
        label_before = _sae_label_before_from_tests_path(_optional_text(row, "tests_path"))
        packets.append(
            {
                "provider_label": _optional_text(row, "provider_label"),
                "round_index": _optional_int(row["round_index"]),
                "provider": _optional_text(row, "provider"),
                "model": _optional_text(row, "provider_model"),
                "created_at": _optional_text(row, "created_at"),
                "source_annotation_path": _optional_text(row, "source_annotation_path"),
                "tests_path": _optional_text(row, "tests_path"),
                "request_path": _optional_text(row, "request_path"),
                "raw_response_path": _optional_text(row, "raw_response_path"),
                "annotation_path": _optional_text(row, "annotation_path"),
                "label_before": label_before,
                "test_results": _sae_test_results_from_tests_path(_optional_text(row, "tests_path")),
                "annotation": {
                    "label": _optional_text(row, "label"),
                    "simple_label": _optional_text(row, "simple_label"),
                    "description": _optional_text(row, "description"),
                    "reasoning": _optional_text(row, "reasoning"),
                    "confidence": _optional_text(row, "confidence").lower(),
                    "test_cases": _annotation_test_cases(row["test_cases_json"]),
                },
            }
        )
    return packets


def _sae_label_before_from_tests_path(path_text: str) -> dict[str, Any]:
    packet = _json_file(Path(path_text)) if path_text else None
    label_before = packet.get("label_before_tests") if isinstance(packet, dict) else None
    return label_before if isinstance(label_before, dict) else {}


def _sae_test_results_from_tests_path(path_text: str) -> list[Any]:
    packet = _json_file(Path(path_text)) if path_text else None
    test_results = packet.get("test_results") if isinstance(packet, dict) else None
    return test_results if isinstance(test_results, list) else []


def _feature_annotation_rows(conn: sqlite3.Connection, *, run_id: str, layer: str, feature_id: int) -> list[sqlite3.Row]:
    try:
        return conn.execute(
            """
            SELECT provider_label, provider, model, created_at,
                   label, simple_label, description, reasoning, confidence,
                   test_cases_json, annotation_path, raw_response_path
            FROM feature_annotations
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            ORDER BY provider_label
            """,
            (run_id, layer, int(feature_id)),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "test_cases_json" not in str(exc):
            raise
        return conn.execute(
            """
            SELECT provider_label, provider, model, created_at,
                   label, simple_label, description, reasoning, confidence,
                   '' AS test_cases_json, annotation_path, raw_response_path
            FROM feature_annotations
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            ORDER BY provider_label
            """,
            (run_id, layer, int(feature_id)),
        ).fetchall()


def _annotation_test_cases(value: Any) -> list[dict[str, str]]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        row = {
            "text": str(item.get("text") or "").strip(),
            "target": str(item.get("target") or "").strip(),
            "expected": str(item.get("expected") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
        }
        if row["text"]:
            out.append(row)
    return out


def _json_object(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _example_token_text(example: Any) -> str:
    if not isinstance(example, dict):
        return ""
    target_token = example.get("target_token")
    if target_token is not None:
        return str(target_token)
    text = example.get("text")
    if text is not None:
        return str(text)
    token = example.get("token")
    if token is not None:
        return str(token).replace("Ġ", " ")
    return ""


def _activation_manifest_path(run_id: str) -> Path | None:
    with connect(DB_PATH) as conn:
        row = conn.execute("SELECT activation_manifest FROM model_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or not row["activation_manifest"]:
        return None
    path = Path(str(row["activation_manifest"]))
    if not path.is_absolute():
        path = (SERVER_DIR.parent / path).resolve()
    return path


@lru_cache(maxsize=8)
def _document_tokenizer(model_id: str) -> Any:
    return AutoTokenizer.from_pretrained(model_id)


@lru_cache(maxsize=64)
def _capture_context_length(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    value = (manifest.get("capture") or {}).get("context_length")
    if value is None:
        return 1024
    length = int(value)
    if length <= 0:
        raise ValueError(f"Invalid capture context_length in {manifest_path}: {value}")
    return length


def _document_token_span(model_id: str, text: str, position: int) -> dict[str, Any] | None:
    if position < 0:
        return None
    tokenizer = _document_tokenizer(model_id)
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max(1024, position + 1),
    )
    offsets = encoded.get("offset_mapping") or []
    input_ids = encoded.get("input_ids") or []
    if position >= len(offsets):
        return {
            "position": int(position),
            "available": False,
            "reason": f"position outside tokenized document ({len(offsets)} tokens)",
        }
    start, end = offsets[position]
    start = int(start)
    end = int(end)
    token_id = int(input_ids[position]) if position < len(input_ids) else None
    token_text = tokenizer.decode([token_id]) if token_id is not None else ""
    if end <= start:
        return {
            "position": int(position),
            "available": False,
            "token_id": token_id,
            "token_text": token_text,
            "start": start,
            "end": end,
            "reason": "token has no document character span",
        }
    return {
        "position": int(position),
        "available": True,
        "token_id": token_id,
        "token_text": token_text,
        "start": start,
        "end": end,
    }


def _tokenized_document_prefix(model_id: str, text: str, *, max_length: int) -> dict[str, Any]:
    tokenizer = _document_tokenizer(model_id)
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.get("offset_mapping") or []
    input_ids = encoded.get("input_ids") or []
    prefix_end = 0
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end > start:
            prefix_end = max(prefix_end, end)
    return {
        "token_count": len(input_ids),
        "prefix_end": int(prefix_end),
        "truncated": int(prefix_end) < len(text),
        "max_length": int(max_length),
    }


def _feature_activation_spans(
    *,
    model_id: str,
    run_id: str,
    layer: str,
    feature_id: int,
    text: str,
    context_length: int,
    min_relative: float,
    target_position: int | None,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    runtime = load_runtime(model_id, device, dtype)
    bundle = load_feature_bundle(run_id, layer, str(DB_PATH))
    if feature_id < 0 or feature_id >= int(bundle.feature_directions.shape[0]):
        raise ValueError(f"feature {feature_id} is outside 0..{int(bundle.feature_directions.shape[0]) - 1}")
    max_length = int(context_length)
    encoded = runtime.tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")[0].detach().cpu().tolist()
    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    inputs = {key: value.to(runtime.device) for key, value in encoded.items()}
    hidden = hidden_states_for_layer(runtime.model, layer, inputs)
    direction = bundle.feature_directions[int(feature_id)].to(runtime.device)
    mean = bundle.mean.to(runtime.device)
    normalized = hidden / torch.linalg.vector_norm(hidden, dim=1, keepdim=True).clamp_min(bundle.norm_eps)
    activations = torch.relu((normalized - mean) @ direction).detach().cpu()
    max_activation = float(torch.max(activations).item()) if int(activations.numel()) else 0.0
    threshold = max(0.0, min(1.0, float(min_relative))) * max_activation
    spans = []
    for token_index, ((start, end), token_id, activation) in enumerate(zip(offsets, input_ids, activations.tolist(), strict=True)):
        start = int(start)
        end = int(end)
        activation = float(activation)
        if end <= start:
            continue
        is_target = target_position is not None and int(token_index) == int(target_position)
        if activation < threshold and not is_target:
            continue
        spans.append(
            {
                "token_index": int(token_index),
                "token_id": int(token_id),
                "token_text": runtime.tokenizer.decode([int(token_id)]),
                "start": start,
                "end": end,
                "activation": activation,
                "relative": 0.0 if max_activation <= 0 else activation / max_activation,
                "target": is_target,
            }
        )
    return {
        "text": text,
        "max_activation": max_activation,
        "min_relative": max(0.0, min(1.0, float(min_relative))),
        "spans": spans,
        "token_count": len(input_ids),
        "activation_prefix_end": max((int(end) for start, end in offsets if int(end) > int(start)), default=0),
        "max_length": max_length,
        "truncated": True,
    }


def _math_cued_document_activation_payload(
    *,
    source_path: Path,
    model_id: str,
    run_id: str,
    layer: str,
    feature_id: int,
    min_relative: float,
    target_position: int | None,
) -> dict[str, Any]:
    payload = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("activations"), torch.Tensor):
        raise ValueError(f"No captured activation tensor in {source_path}")
    token_data = payload.get("tokens") if isinstance(payload.get("tokens"), dict) else {}
    raw_token_ids = token_data.get("captured_token_ids") or token_data.get("sequence_token_ids") or []
    token_ids = [int(token_id) for token_id in raw_token_ids]
    activations = payload["activations"]
    if activations.ndim != 2 or int(activations.shape[0]) != len(token_ids):
        raise ValueError(
            f"Activation/token alignment mismatch in {source_path}: "
            f"{tuple(activations.shape)} versus {len(token_ids)} token ids"
        )

    bundle = load_feature_bundle(run_id, layer, str(DB_PATH))
    if feature_id < 0 or feature_id >= int(bundle.feature_directions.shape[0]):
        raise ValueError(f"feature {feature_id} is outside 0..{int(bundle.feature_directions.shape[0]) - 1}")
    direction = bundle.feature_directions[feature_id].detach().cpu().to(torch.float32)
    mean = bundle.mean.detach().cpu().to(torch.float32)
    if mean.ndim > 1:
        mean = mean.reshape(-1)
    centered_direction_bias = torch.dot(mean, direction)
    score_chunks = []
    for start in range(0, int(activations.shape[0]), 4096):
        chunk = activations[start : start + 4096].to(torch.float32)
        normalized = chunk / torch.linalg.vector_norm(chunk, dim=1, keepdim=True).clamp_min(bundle.norm_eps)
        score_chunks.append(torch.relu(normalized @ direction - centered_direction_bias))
    scores = torch.cat(score_chunks) if score_chunks else torch.empty(0, dtype=torch.float32)

    tokenizer = _document_tokenizer(model_id)
    text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    aligned_ids = [int(token_id) for token_id in encoded.get("input_ids") or []]
    offsets = encoded.get("offset_mapping") or []
    if aligned_ids != token_ids or len(offsets) != len(token_ids):
        raise ValueError(f"Decoded document does not round-trip to captured token ids for {source_path}")

    max_activation = float(scores.max().item()) if int(scores.numel()) else 0.0
    min_relative = max(0.0, min(1.0, float(min_relative)))
    threshold = min_relative * max_activation
    spans = []
    for token_index, ((start, end), token_id, activation) in enumerate(
        zip(offsets, token_ids, scores.tolist(), strict=True)
    ):
        start = int(start)
        end = int(end)
        activation = float(activation)
        if end <= start:
            continue
        is_target = target_position is not None and token_index == int(target_position)
        if activation < threshold and not is_target:
            continue
        spans.append(
            {
                "token_index": token_index,
                "token_id": token_id,
                "token_text": text[start:end],
                "start": start,
                "end": end,
                "activation": activation,
                "relative": 0.0 if max_activation <= 0 else activation / max_activation,
                "target": is_target,
            }
        )
    return {
        "text": text,
        "max_activation": max_activation,
        "min_relative": min_relative,
        "spans": spans,
        "token_count": len(token_ids),
        "activation_prefix_end": len(text),
        "max_length": len(token_ids),
        "truncated": False,
        "activation_source": "captured_document_bundle",
    }


def _cached_document_activation_payload(
    *,
    model_id: str,
    run_id: str,
    model: str,
    layer: str,
    feature_id: int,
    doc_id: int,
    manifest_path: str,
    context_length: int,
    min_relative: float,
    target_position: int | None,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    key = (
        model,
        run_id,
        layer,
        int(feature_id),
        int(doc_id),
        int(context_length),
        round(float(min_relative), 6),
        None if target_position is None else int(target_position),
        device,
        dtype,
    )
    with _document_activation_cache_lock:
        future = _document_activation_cache.get(key)
        if future is None:
            future = Future()
            _document_activation_cache[key] = future
            owner = True
        else:
            owner = False

    if not owner:
        return future.result()

    try:
        text = _load_document_text(manifest_path, int(doc_id))
        payload = _feature_activation_spans(
            model_id=model_id,
            run_id=run_id,
            layer=layer,
            feature_id=int(feature_id),
            text=text,
            context_length=int(context_length),
            min_relative=float(min_relative),
            target_position=target_position,
            device=device,
            dtype=dtype,
        )
    except BaseException as exc:
        future.set_exception(exc)
        with _document_activation_cache_lock:
            if _document_activation_cache.get(key) is future:
                _document_activation_cache.pop(key, None)
        raise

    future.set_result(payload)
    with _document_activation_cache_lock:
        while len(_document_activation_cache) > _DOCUMENT_ACTIVATION_CACHE_MAX_SIZE:
            old_key = next(iter(_document_activation_cache))
            if old_key == key:
                break
            _document_activation_cache.pop(old_key, None)
    return payload


@lru_cache(maxsize=256)
def _load_document_text(manifest_path: str, doc_id: str) -> str:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
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
    return str(dataset[int(doc_id)][text_column])


@lru_cache(maxsize=256)
def _load_document_text_from_feature_evidence(*, run_id: str, layer: str | None, feature: int | None, doc_id: str) -> str:
    if not layer or feature is None:
        raise FileNotFoundError(f"No activation manifest for {run_id}; layer and feature are required for diagnostic document lookup.")
    math_cued_path = _math_cued_source_path_for_doc(run_id=run_id, layer=layer, doc_id=doc_id)
    if math_cued_path is not None and math_cued_path.is_file():
        return _load_math_cued_source_text(str(math_cued_path))
    with connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT annotation_evidence_json
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, layer, int(feature)),
        ).fetchone()
    if row is None or not row["annotation_evidence_json"]:
        raise FileNotFoundError(f"No feature evidence for {run_id}/{layer}/F{int(feature):06d}.")
    evidence = json.loads(str(row["annotation_evidence_json"]))
    source_path: Path | None = None
    for example in evidence.get("examples") or []:
        if str(example.get("doc_id")) == str(doc_id) and example.get("source_path"):
            source_path = Path(str(example["source_path"]))
            break
    if source_path is None:
        raise FileNotFoundError(f"No source_path for doc_id {doc_id!r} in {run_id}/{layer}/F{int(feature):06d}.")
    return _load_math_cued_source_text(str(source_path))


def _math_cued_source_path_for_doc(*, run_id: str, layer: str, feature_id: int, doc_id: str) -> Path | None:
    """Resolve a captured document from feature evidence, without path maps."""
    with connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT annotation_evidence_json
            FROM features
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            (run_id, layer, int(feature_id)),
        ).fetchone()
    if row is None or not row["annotation_evidence_json"]:
        return None
    try:
        evidence = json.loads(str(row["annotation_evidence_json"]))
    except json.JSONDecodeError:
        return None
    for example in evidence.get("examples") or []:
        if str(example.get("doc_id")) == str(doc_id) and example.get("source_path"):
            return Path(str(example["source_path"]))
    return None


@lru_cache(maxsize=256)
def _load_math_cued_source_text(source_path: str) -> str:
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source document file missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected source document payload in {path}")
    tokens = payload.get("tokens")
    generation = payload.get("generation")
    if isinstance(tokens, dict) and isinstance(generation, dict) and tokens.get("sequence_token_ids") and generation.get("model"):
        tokenizer = _document_tokenizer(str(generation["model"]))
        return str(tokenizer.decode(list(tokens["sequence_token_ids"])))
    text = payload.get("text")
    if isinstance(text, dict):
        prompt = str(text.get("prompt") or "")
        generated = str(text.get("generated") or "")
        if prompt or generated:
            return prompt + generated
    problem = payload.get("problem")
    if isinstance(problem, dict) and problem.get("problem"):
        return str(problem["problem"])
    raise ValueError(f"No readable text found in {path}")
