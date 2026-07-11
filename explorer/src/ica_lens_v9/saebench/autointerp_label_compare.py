from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ..annotation.auto_annotate import (
    DEFAULT_DEEPSEEK_TOKEN_PATH,
    DEFAULT_MI_TOKEN_PATH,
    DEFAULT_OPENAI_TOKEN_PATH,
    DEFAULT_TINKER_TOKEN_PATH,
    apply_openai_reasoning_controls,
    call_annotation_provider,
    default_base_url_for_provider,
    default_model_for_provider,
    extract_chat_completion_text,
    load_api_key_for_provider,
    read_json,
    resolve_model,
    safe_filename,
    write_json,
    write_request_debug_bundle,
)
from .adapters import load_selected_sae
from .config import (
    DEFAULT_FEATURE_INTERFACE_ROOT,
    DEFAULT_METHOD,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAEBENCH_ARTIFACTS,
    RUN_NAMES,
    SAEBENCH_MODEL_NAMES,
    RunTarget,
    feature_interface_dir,
    layer_index,
    saebench_env,
)
from .runner_utils import setup_saebench_runtime, str_to_dtype
from ..saes.counterparts import SAE_COUNTERPARTS


DEFAULT_AUTO_ANNOTATION_SAE_ROOT = Path(__file__).resolve().parents[3] / "results" / "auto_annotation_sae"
DEFAULT_AUTO_ANNOTATION_ICA_ROOT = Path(__file__).resolve().parents[3] / "results" / "auto_annotation"
DEFAULT_NEURONPEDIA_DB = Path(__file__).resolve().parents[3] / "neuronpedia_exports" / "neuronpedia.sqlite"
DEFAULT_LABEL_SCORING_ROOT = Path(__file__).resolve().parents[3] / "results" / "auto_interp_label_scoring"
BINARY_JUDGE_PROMPT_VERSION = "binary_tokenized_reason_example_index_v1"
CHOICE_JUDGE_PROMPT_VERSION = "saebench_choice_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAE label candidates with SAEBench AutoInterp scoring examples and scoring logic."
    )
    parser.add_argument("--target-kind", choices=["sae_counterpart", "ica"], default="sae_counterpart")
    parser.add_argument("--model", required=True, choices=list(SAEBENCH_MODEL_NAMES))
    parser.add_argument("--layer", required=True)
    parser.add_argument("--feature-id", type=int, action="append", default=[])
    parser.add_argument("--feature-start", type=int)
    parser.add_argument("--feature-end", type=int)
    parser.add_argument("--provider-label", default="mi", help="Auto-annotation provider label to compare.")
    parser.add_argument("--judge-provider", choices=["mi", "ollama", "openai", "deepseek", "tinker-sdk", "saebench_openai"], default="mi")
    parser.add_argument("--judge-model", default=None, help="Judge model alias/name. SAEBench OpenAI uses its built-in gpt-4o-mini path.")
    parser.add_argument(
        "--judge-mode",
        choices=["choice", "binary"],
        default="choice",
        help="choice uses the original SAEBench index-selection prompt; binary asks for an independent active/inactive decision for each example.",
    )
    parser.add_argument("--api-key-file", type=Path, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_LABEL_SCORING_ROOT)
    parser.add_argument("--saebench-artifacts-path", type=Path, default=DEFAULT_SAEBENCH_ARTIFACTS / "autointerp_label_compare")
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--feature-interface-method", default=DEFAULT_METHOD)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_AUTO_ANNOTATION_SAE_ROOT / "evidence")
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_AUTO_ANNOTATION_SAE_ROOT / "annotations")
    parser.add_argument("--refinement-root", type=Path, default=DEFAULT_AUTO_ANNOTATION_SAE_ROOT / "refinements")
    parser.add_argument("--neuronpedia-db-path", type=Path, default=DEFAULT_NEURONPEDIA_DB)
    parser.add_argument("--dataset-name", default="monology/pile-uncopyrighted")
    parser.add_argument("--total-tokens", type=int, default=2_000_000)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--buffer", type=int, default=10)
    parser.add_argument(
        "--example-context",
        choices=["causal_left", "saebench_window"],
        default="causal_left",
        help=(
            "Context shown to judges. causal_left shows all tokens from the start of the scored sequence through "
            "+buffer right context; saebench_window preserves SAEBench's ±buffer display window."
        ),
    )
    parser.add_argument("--act-threshold-frac", type=float, default=0.33)
    parser.add_argument("--n-top-scoring", type=int, default=2)
    parser.add_argument("--n-random-scoring", type=int, default=10)
    parser.add_argument("--n-iw-scoring", type=int, default=2)
    parser.add_argument("--n-top-generation", type=int, default=10, help="Kept for SAEBench gather_data; generation is not called.")
    parser.add_argument("--n-iw-generation", type=int, default=5, help="Kept for SAEBench gather_data; generation is not called.")
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--llm-dtype", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--judge-parallelism",
        type=int,
        default=0,
        help="Concurrent binary judge calls per feature. 0 means all examples in parallel.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--force-judge", action="store_true", help="Ignore the cross-run judge cache and call the judge again.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    feature_ids = selected_feature_ids(args)
    if args.worker:
        run_worker(args, feature_ids=feature_ids)
        return
    if args.dry_run:
        root, python = saebench_env(str(args.model))
        print(
            json.dumps(
                {
                    "model": args.model,
                    "layer": args.layer,
                    "target_kind": args.target_kind,
                    "feature_ids": feature_ids,
                    "saebench_root": str(root),
                    "saebench_python": str(python),
                    "output_root": str(args.output_root.resolve()),
                    "resolved_output_dir": str(result_output_dir(args).resolve()),
                    "saebench_artifacts_path": str(args.saebench_artifacts_path.resolve()),
                },
                indent=2,
            )
        )
        return
    root, python = saebench_env(str(args.model))
    if not python.is_file():
        raise FileNotFoundError(f"Missing SAEBench interpreter: {python}. Run `bash scripts/setup_saebench_envs.sh` from v5 first.")
    command = [str(python), str(Path(__file__).resolve().parents[3] / "scripts" / "compare_sae_label_autointerp.py"), "--worker", *cli_args]
    command = [part for part in command if part != "--dry-run"]
    print(f"SAEBench AutoInterp label comparison via {python}")
    print(f"features: {', '.join(str(fid) for fid in feature_ids)}")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])


def run_worker(args: argparse.Namespace, *, feature_ids: list[int]) -> None:
    device = setup_saebench_runtime(str(args.model))
    from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
    from sae_bench.evals.autointerp.main import AutoInterp
    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
    from transformer_lens import HookedTransformer

    model_key = str(args.model)
    layer = canonical_layer(str(args.layer))
    dtype_name = str(args.llm_dtype or activation_collection.LLM_NAME_TO_DTYPE.get(SAEBENCH_MODEL_NAMES[model_key], "float32"))
    dtype = str_to_dtype(dtype_name)
    output_dir = result_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_spec = run_spec_payload(args=args, feature_ids=feature_ids)
    run_hash = stable_hash(run_spec)

    config = AutoInterpEvalConfig(model_name=SAEBENCH_MODEL_NAMES[model_key], n_latents=None, override_latents=feature_ids)
    config.dataset_name = str(args.dataset_name)
    config.total_tokens = int(args.total_tokens)
    config.llm_context_size = int(args.context_size)
    config.buffer = int(args.buffer)
    config.act_threshold_frac = float(args.act_threshold_frac)
    config.n_top_ex_for_generation = int(args.n_top_generation)
    config.n_iw_sampled_ex_for_generation = int(args.n_iw_generation)
    config.n_top_ex_for_scoring = int(args.n_top_scoring)
    config.n_random_ex_for_scoring = int(args.n_random_scoring)
    config.n_iw_sampled_ex_for_scoring = int(args.n_iw_scoring)
    config.llm_batch_size = int(args.llm_batch_size or activation_collection.LLM_NAME_TO_BATCH_SIZE.get(config.model_name, 1))
    config.llm_dtype = dtype_name
    config.random_seed = int(args.random_seed)
    torch.manual_seed(int(args.random_seed))

    model = HookedTransformer.from_pretrained_no_processing(config.model_name, device=device, dtype=dtype)
    if args.target_kind == "ica":
        selected_saes, _method_name, metadata = load_selected_sae(
            method="ica_lens",
            model=model_key,
            layer=layer,
            feature_interface_dir=feature_interface_dir(
                model_key,
                feature_interface_root=Path(args.feature_interface_root),
                method=str(args.feature_interface_method),
            ),
            output_root=Path(args.output_root),
            activation_manifest_path=Path(args.output_root) / "unused_manifest.json",
            device=device,
            dtype=dtype,
            force=bool(args.force_rerun),
        )
    else:
        selected_saes, _method_name, metadata = load_selected_sae(
            method="sae_baseline",
            model=model_key,
            layer=layer,
            feature_interface_dir=args.feature_interface_root / "unused",
            output_root=args.output_root,
            activation_manifest_path=args.output_root / "unused_manifest.json",
            device=device,
            dtype=dtype,
            force=bool(args.force_rerun),
        )
    sae_name, sae = selected_saes[0]

    tokenized_dataset = load_or_tokenize_dataset(
        config=config,
        model=model,
        artifacts_path=args.saebench_artifacts_path,
        device=device,
    )
    sparsity = torch.zeros(int(sae.cfg.d_sae), device=device)
    autointerp = AutoInterp(config, model, sae, tokenized_dataset, sparsity, device, api_key=load_saebench_openai_key(args) if args.judge_provider == "saebench_openai" else "")
    _generation_examples, scoring_examples = load_or_gather_scoring_examples(
        args=args,
        autointerp=autointerp,
        feature_ids=feature_ids,
        model=model,
    )

    all_rows: list[dict[str, Any]] = []
    for feature_id in tqdm(feature_ids, desc="SAEBench label scoring", unit="feature", dynamic_ncols=True):
        rows = score_one_feature(
            args=args,
            autointerp=autointerp,
            scoring_examples=scoring_examples,
            output_dir=output_dir,
            feature_id=int(feature_id),
            sae_name=str(sae_name),
            metadata=metadata,
            run_spec=run_spec,
            run_spec_hash=run_hash,
        )
        all_rows.extend(rows)
        write_score_tables(output_dir=output_dir, rows=all_rows)
    write_json(
        output_dir / "summary.json",
        {
            "created_at_unix": time.time(),
            "model": model_key,
            "layer": layer,
            "target_kind": str(args.target_kind),
            "feature_ids": feature_ids,
            "sae_name": str(sae_name),
            "saebench_config": config.__dict__,
            "metadata": metadata,
            "run_spec": run_spec,
            "run_spec_hash": run_hash,
            "candidate_accuracy_mean": summarize_accuracy(all_rows),
        },
    )


def score_one_feature(
    *,
    args: argparse.Namespace,
    autointerp: Any,
    scoring_examples: dict[int, Any],
    output_dir: Path,
    feature_id: int,
    sae_name: str,
    metadata: dict[str, object],
    run_spec: dict[str, Any],
    run_spec_hash: str,
) -> list[dict[str, Any]]:
    examples = scoring_examples.get(int(feature_id))
    feature_dir = output_dir / f"F{feature_id:06d}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    candidates = collect_candidates(args=args, feature_id=feature_id)
    write_json(feature_dir / "candidate_labels.json", candidates)
    if examples is None:
        write_json(feature_dir / "error.json", {"error": "SAEBench found no scoring examples for this feature."})
        return []
    write_json(feature_dir / "saebench_scoring_examples.json", serialize_examples(examples))
    examples_hash = scoring_examples_hash(examples)
    examples_debug = serialize_examples(examples)
    examples_raw = serialize_examples_with_raw(examples)
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_hash_value = candidate_hash(candidate)
        result_path = feature_dir / f"{safe_filename(candidate_id)}_score.json"
        if result_path.exists() and not args.force_rerun:
            result = read_json(result_path)
            if (
                result.get("run_spec_hash") == run_spec_hash
                and result.get("candidate_hash") == candidate_hash_value
                and result.get("scoring_examples_hash") == examples_hash
            ):
                rows.append(result["row"])
                continue
        if candidate.get("invalid"):
            correct = [i for i, ex in enumerate(examples, start=1) if ex.is_active]
            row = build_zero_score_row(
                args=args,
                feature_id=feature_id,
                candidate=candidate,
                examples=examples,
                correct=correct,
                sae_name=sae_name,
                metadata=metadata,
            )
            write_json(
                result_path,
                {
                    "candidate": candidate,
                    "row": row,
                    "run_spec": run_spec,
                    "run_spec_hash": run_spec_hash,
                    "candidate_hash": candidate_hash_value,
                    "scoring_examples_hash": examples_hash,
                    "scoring_examples": examples_debug,
                    "scoring_examples_raw": examples_raw,
                    "correct_sequences": correct,
                    "judge_request": None,
                    "judge_output_text": "",
                    "raw_response": None,
                    "saebench_prompt_note": "Candidate label was missing or malformed; assigned score 0 without judge call.",
                },
            )
            rows.append(row)
            continue
        request_payload = build_judge_request_payload(args=args, candidate=candidate, examples=examples, autointerp=autointerp)
        request_path = feature_dir / f"{safe_filename(candidate_id)}_request_preview.json"
        write_json(request_path, request_payload)
        write_judge_request_debug_bundle(feature_dir / f"{safe_filename(candidate_id)}_request_debug", request_payload)
        correct = [i for i, ex in enumerate(examples, start=1) if ex.is_active]
        cache_path = judge_cache_path(args=args, candidate=candidate, feature_id=feature_id, examples=examples)
        cached = None if args.force_judge else read_cached_judge_result(cache_path)
        cache_status = "global_cache" if cached is not None else ""
        if cached is None and not args.force_judge:
            cached = find_equivalent_judge_result(args=args, candidate=candidate, feature_id=feature_id, examples=examples)
            cache_status = "existing_run" if cached is not None else ""
        if cached is not None:
            raw_text = str(cached.get("judge_output_text") or "")
            raw_response = cached.get("raw_response")
            predictions = cached.get("predictions")
            binary_decisions = cached.get("binary_decisions")
            score = float(cached.get("saebench_score") or 0.0)
        else:
            if args.judge_mode == "binary":
                raw_text, raw_response, predictions, binary_decisions = call_binary_judge_individually(
                    args=args,
                    request_payload=request_payload,
                    autointerp=autointerp,
                    examples=examples,
                )
            else:
                raw_text, raw_response = call_judge(args=args, request_payload=request_payload, autointerp=autointerp)
                predictions, binary_decisions = parse_judge_predictions(args=args, raw_text=raw_text, autointerp=autointerp, examples=examples)
            score = 0.0 if predictions is None else float(autointerp.score_predictions(predictions, examples))
            write_cached_judge_result(
                cache_path=cache_path,
                args=args,
                candidate=candidate,
                feature_id=feature_id,
                examples=examples,
                raw_text=raw_text,
                raw_response=raw_response,
                predictions=predictions,
                binary_decisions=binary_decisions,
                score=score,
            )
            cache_status = "miss"
        row = {
            "model": str(args.model),
            "layer": canonical_layer(str(args.layer)),
            "target_kind": str(args.target_kind),
            "feature_id": int(feature_id),
            "candidate_id": candidate_id,
            "candidate_source": candidate.get("source"),
            "label": candidate.get("label"),
            "simple_label": candidate.get("simple_label"),
            "description": candidate.get("description"),
            "saebench_score": score,
            "predictions": predictions,
            "binary_decisions": binary_decisions,
            "correct_sequences": correct,
            "n_examples": len(examples),
            "judge_mode": str(args.judge_mode),
            "judge_prompt_version": judge_prompt_version(args),
            "judge_cache_status": cache_status,
            "sae_name": sae_name,
            "run_spec_hash": run_spec_hash,
            "candidate_hash": candidate_hash_value,
            "scoring_examples_hash": examples_hash,
            **{f"metadata_{key}": value for key, value in metadata.items()},
        }
        result = {
            "candidate": candidate,
            "row": row,
            "run_spec": run_spec,
            "run_spec_hash": run_spec_hash,
            "candidate_hash": candidate_hash_value,
            "scoring_examples_hash": examples_hash,
            "scoring_examples": examples_debug,
            "scoring_examples_raw": examples_raw,
            "correct_sequences": correct,
            "judge_request": request_payload,
            "judge_output_text": raw_text,
            "raw_response": raw_response,
            "binary_decisions": binary_decisions,
            "saebench_prompt_note": (
                "Prompt, prediction parsing, and score use SAEBench AutoInterp."
                if args.judge_mode == "choice"
                else "Examples come from SAEBench AutoInterp; judge prompt asks independent binary active/inactive decisions."
            ),
        }
        write_json(result_path, result)
        rows.append(row)
    return rows


def build_zero_score_row(
    *,
    args: argparse.Namespace,
    feature_id: int,
    candidate: dict[str, Any],
    examples: Any,
    correct: list[int],
    sae_name: str,
    metadata: dict[str, object],
) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "layer": canonical_layer(str(args.layer)),
        "target_kind": str(args.target_kind),
        "feature_id": int(feature_id),
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_source": candidate.get("source"),
        "label": candidate.get("label"),
        "simple_label": candidate.get("simple_label"),
        "description": candidate.get("description"),
        "saebench_score": 0.0,
        "predictions": None,
        "binary_decisions": None,
        "correct_sequences": correct,
        "n_examples": len(examples),
        "judge_mode": str(args.judge_mode),
        "sae_name": sae_name,
        **{f"metadata_{key}": value for key, value in metadata.items()},
    }


def judge_label(args: argparse.Namespace) -> str:
    if args.judge_provider == "saebench_openai":
        return "saebench_openai"
    return safe_filename(resolve_model(str(args.judge_model or default_model_for_provider(str(args.judge_provider)))))


def run_spec_payload(*, args: argparse.Namespace, feature_ids: list[int]) -> dict[str, Any]:
    return {
        "target_kind": str(args.target_kind),
        "model": str(args.model),
        "layer": canonical_layer(str(args.layer)),
        "feature_ids": [int(feature_id) for feature_id in feature_ids],
        "annotation_provider_label": str(args.provider_label),
        "judge_provider": str(args.judge_provider),
        "judge_model": judge_label(args),
        "judge_mode": str(args.judge_mode),
        "judge_prompt_version": judge_prompt_version(args),
        "judge_parallelism": int(args.judge_parallelism),
        "scoring_settings": scoring_examples_cache_settings(args),
        "output_schema": "auto_interp_label_scoring.v2",
    }


def candidate_hash(candidate: dict[str, Any]) -> str:
    payload = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source": str(candidate.get("source") or ""),
        "label": str(candidate.get("label") or ""),
        "simple_label": str(candidate.get("simple_label") or ""),
        "description": str(candidate.get("description") or ""),
        "explanation": str(candidate.get("explanation") or ""),
        "source_path": str(candidate.get("source_path") or ""),
        "invalid": bool(candidate.get("invalid", False)),
        "invalid_reason": str(candidate.get("invalid_reason") or ""),
    }
    return stable_hash(payload)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def judge_cache_path(*, args: argparse.Namespace, candidate: dict[str, Any], feature_id: int, examples: Any) -> Path:
    key = judge_cache_key(args=args, candidate=candidate, feature_id=feature_id, examples=examples)
    return (
        Path(args.output_root)
        / "_judge_cache"
        / str(args.target_kind)
        / str(args.model)
        / canonical_layer(str(args.layer))
        / judge_label(args)
        / f"F{int(feature_id):06d}"
        / f"{key}.json"
    )


def judge_cache_key(*, args: argparse.Namespace, candidate: dict[str, Any], feature_id: int, examples: Any) -> str:
    payload = {
        "model": str(args.model),
        "layer": canonical_layer(str(args.layer)),
        "target_kind": str(args.target_kind),
        "feature_id": int(feature_id),
        "judge_provider": str(args.judge_provider),
        "judge_model": judge_label(args),
        "judge_mode": str(args.judge_mode),
        "judge_prompt_version": judge_prompt_version(args),
        "dataset_name": str(args.dataset_name),
        "total_tokens": int(args.total_tokens),
        "context_size": int(args.context_size),
        "buffer": int(args.buffer),
        "act_threshold_frac": float(args.act_threshold_frac),
        "n_top_scoring": int(args.n_top_scoring),
        "n_random_scoring": int(args.n_random_scoring),
        "n_iw_scoring": int(args.n_iw_scoring),
        "random_seed": int(args.random_seed),
        "explanation": str(candidate.get("explanation") or ""),
        "examples": serialize_examples_with_raw(examples),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def judge_prompt_version(args: argparse.Namespace) -> str:
    if str(args.judge_mode) == "binary":
        return BINARY_JUDGE_PROMPT_VERSION
    return CHOICE_JUDGE_PROMPT_VERSION


def read_cached_judge_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def write_cached_judge_result(
    *,
    cache_path: Path,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    feature_id: int,
    examples: Any,
    raw_text: str,
    raw_response: Any,
    predictions: Any,
    binary_decisions: Any,
    score: float,
) -> None:
    write_json(
        cache_path,
        {
            "created_at_unix": time.time(),
            "cache_key": cache_path.stem,
            "model": str(args.model),
            "layer": canonical_layer(str(args.layer)),
            "target_kind": str(args.target_kind),
            "feature_id": int(feature_id),
            "judge_provider": str(args.judge_provider),
            "judge_model": judge_label(args),
            "judge_mode": str(args.judge_mode),
            "judge_prompt_version": judge_prompt_version(args),
            "candidate_explanation": str(candidate.get("explanation") or ""),
            "examples_hash": scoring_examples_hash(examples),
            "saebench_score": float(score),
            "predictions": predictions,
            "binary_decisions": binary_decisions,
            "judge_output_text": raw_text,
            "raw_response": raw_response,
        },
    )


def find_equivalent_judge_result(
    *,
    args: argparse.Namespace,
    candidate: dict[str, Any],
    feature_id: int,
    examples: Any,
) -> dict[str, Any] | None:
    base = Path(args.output_root) / str(args.target_kind) / str(args.model) / canonical_layer(str(args.layer))
    current_run = result_output_dir(args)
    wanted_explanation = str(candidate.get("explanation") or "")
    wanted_examples_hash = scoring_examples_hash(examples)
    suffix = "" if str(args.judge_mode) == "choice" else f"_{safe_filename(str(args.judge_mode))}"
    for path in sorted(base.glob(f"*_judge_{judge_label(args)}{suffix}/F{int(feature_id):06d}/*_score.json")):
        if current_run in path.parents:
            continue
        try:
            packet = read_json(path)
        except Exception:
            continue
        old_candidate = packet.get("candidate") if isinstance(packet, dict) else None
        old_row = packet.get("row") if isinstance(packet, dict) else None
        if not isinstance(old_candidate, dict) or not isinstance(old_row, dict):
            continue
        old_run_spec = packet.get("run_spec") if isinstance(packet, dict) else {}
        old_judge_mode = old_run_spec.get("judge_mode") if isinstance(old_run_spec, dict) else old_row.get("judge_mode")
        if str(old_judge_mode or "choice") != str(args.judge_mode):
            continue
        old_prompt_version = old_run_spec.get("judge_prompt_version") if isinstance(old_run_spec, dict) else old_row.get("judge_prompt_version")
        if str(args.judge_mode) == "binary" and str(old_prompt_version or "") != judge_prompt_version(args):
            continue
        if str(args.judge_mode) != "binary" and old_prompt_version is not None and str(old_prompt_version) != judge_prompt_version(args):
            continue
        if str(old_candidate.get("explanation") or "") != wanted_explanation:
            continue
        if packet.get("scoring_examples_hash") != wanted_examples_hash:
            continue
        if packet.get("candidate_hash") != candidate_hash(candidate):
            continue
        return {
            "saebench_score": float(old_row.get("saebench_score") or 0.0),
            "predictions": old_row.get("predictions"),
            "binary_decisions": packet.get("binary_decisions") or old_row.get("binary_decisions"),
            "judge_output_text": packet.get("judge_output_text") or "",
            "raw_response": packet.get("raw_response"),
        }
    return None


def build_judge_request_payload(*, args: argparse.Namespace, candidate: dict[str, Any], examples: Any, autointerp: Any) -> dict[str, Any]:
    if args.judge_mode == "choice":
        messages = autointerp.get_scoring_prompts(explanation=str(candidate["explanation"]), scoring_examples=examples)
        return build_request_payload(args=args, messages=messages, autointerp=autointerp)
    requests = []
    for index, example in enumerate(examples, start=1):
        messages = binary_judge_messages_for_example(
            explanation=str(candidate["explanation"]),
            index=int(index),
            text=example.to_str(mark_toks=False),
            tokenized_text=example_tokenized_text(example),
        )
        requests.append(
            {
                "index": int(index),
                "request": build_request_payload(args=args, messages=messages, autointerp=autointerp),
            }
        )
    return {
        "judge_mode": "binary",
        "execution": "one_request_per_example",
        "requests": requests,
    }


def write_judge_request_debug_bundle(output_dir: Path, request_payload: dict[str, Any]) -> None:
    if request_payload.get("judge_mode") != "binary":
        write_request_debug_bundle(output_dir, request_payload)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = request_payload.get("requests")
    requests = requests if isinstance(requests, list) else []
    manifest: dict[str, Any] = {
        "judge_mode": "binary",
        "execution": "one_request_per_example",
        "request_count": len(requests),
        "request_dirs": [],
    }
    write_json(
        output_dir / "request_options.json",
        {key: value for key, value in request_payload.items() if key != "requests"},
    )
    for position, item in enumerate(requests, start=1):
        index = int(item.get("index") or position) if isinstance(item, dict) else position
        request = item.get("request") if isinstance(item, dict) else None
        request = request if isinstance(request, dict) else {}
        dirname = f"request_{position:02d}_example_{index:02d}"
        write_request_debug_bundle(output_dir / dirname, request)
        manifest["request_dirs"].append(
            {
                "position": position,
                "example_index": index,
                "dir": dirname,
            }
        )
    write_json(output_dir / "manifest.json", manifest)


def example_tokenized_text(example: Any) -> list[str]:
    tokens = getattr(example, "str_toks", None)
    if not isinstance(tokens, list):
        return []
    return [str(token).replace("�", "") for token in tokens]


def binary_judge_messages_for_example(*, explanation: str, index: int, text: str, tokenized_text: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You judge whether one feature explanation predicts activation on one text example. Return only valid JSON.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": (
                        "Decide whether the target LLM feature described by explanation would activate on any token "
                        "in this single example. Use tokenized_text as the target LLM tokenization; do not rely on "
                        "your own tokenization. Return JSON with keys example_index, reason, and active. "
                        "If you mention token positions, put them in reason; example_index must equal the example index."
                    ),
                    "response_format": {"example_index": int(index), "reason": "short decision rationale", "active": True},
                    "explanation": explanation,
                    "example": {"index": int(index), "text": text, "tokenized_text": tokenized_text},
                },
                ensure_ascii=False,
            ),
        },
    ]


def parse_judge_predictions(*, args: argparse.Namespace, raw_text: str, autointerp: Any, examples: Any) -> tuple[Any, list[dict[str, Any]] | None]:
    if args.judge_mode == "choice":
        return autointerp.parse_predictions(raw_text), None
    decisions = parse_binary_decisions(raw_text=raw_text, n_examples=len(examples))
    if decisions is None:
        return None, None
    predictions = [row["index"] for row in decisions if bool(row["active"])]
    return predictions, decisions


def call_binary_judge_individually(
    *,
    args: argparse.Namespace,
    request_payload: dict[str, Any],
    autointerp: Any,
    examples: Any,
) -> tuple[str, dict[str, Any], Any, list[dict[str, Any]] | None]:
    requests = request_payload.get("requests") if isinstance(request_payload, dict) else None
    if not isinstance(requests, list):
        return "", {"error": "binary request payload missing requests list"}, None, None
    tasks: list[tuple[int, dict[str, Any]]] = []
    for item in requests:
        if not isinstance(item, dict) or not isinstance(item.get("request"), dict):
            return "", {"error": "invalid binary request item", "item": item}, None, None
        tasks.append((int(item.get("index")), item["request"]))
    max_workers = int(args.judge_parallelism)
    if max_workers <= 0:
        max_workers = max(1, len(tasks))
    max_workers = max(1, min(max_workers, len(tasks)))
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(call_one_binary_judge, args=args, autointerp=autointerp, expected_index=expected_index, request=request): expected_index
            for expected_index, request in tasks
        }
        for future in as_completed(futures):
            responses.append(future.result())
    responses.sort(key=lambda row: int(row["index"]))
    decisions = []
    for row in responses:
        decision = row.get("parsed_decision")
        if decision is None:
            raw_joined = "\n".join(str(item.get("judge_output_text") or "") for item in responses)
            return raw_joined, {"responses": responses, "error": f"Could not parse binary decision for index {row.get('index')}."}, None, None
        decisions.append(decision)
    decisions.sort(key=lambda row: int(row["index"]))
    predictions = [row["index"] for row in decisions if bool(row["active"])]
    raw_joined = "\n".join(str(row.get("judge_output_text") or "") for row in responses)
    return raw_joined, {"responses": responses, "parallelism": max_workers}, predictions, decisions


def call_one_binary_judge(
    *,
    args: argparse.Namespace,
    autointerp: Any,
    expected_index: int,
    request: dict[str, Any],
) -> dict[str, Any]:
    try:
        raw_text, raw_response = call_judge(args=args, request_payload=request, autointerp=autointerp)
        decision = parse_single_binary_decision(raw_text=raw_text, expected_index=expected_index)
        return {
            "index": int(expected_index),
            "judge_output_text": raw_text,
            "raw_response": raw_response,
            "parsed_decision": decision,
        }
    except Exception as exc:
        return {
            "index": int(expected_index),
            "judge_output_text": "",
            "raw_response": None,
            "parsed_decision": None,
            "error": str(exc),
        }


def parse_single_binary_decision(*, raw_text: str, expected_index: int) -> dict[str, Any] | None:
    parsed = parse_jsonish(raw_text)
    if not isinstance(parsed, dict):
        return None
    if "decisions" in parsed:
        decisions = parsed.get("decisions")
        if isinstance(decisions, list) and len(decisions) == 1 and isinstance(decisions[0], dict):
            parsed = decisions[0]
    raw_index = parsed.get("example_index")
    if raw_index is None:
        raw_index = parsed.get("index")
    try:
        index = int(raw_index)
    except Exception:
        return None
    if index != int(expected_index):
        return None
    active = parsed.get("active")
    if not isinstance(active, bool):
        return None
    return {"index": index, "reason": str(parsed.get("reason") or ""), "active": bool(active)}


def parse_binary_decisions(*, raw_text: str, n_examples: int) -> list[dict[str, Any]] | None:
    parsed = parse_jsonish(raw_text)
    if not isinstance(parsed, dict):
        return None
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        return None
    by_index: dict[int, dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            return None
        try:
            index = int(item.get("index"))
        except Exception:
            return None
        if index < 1 or index > int(n_examples) or index in by_index:
            return None
        active = item.get("active")
        if not isinstance(active, bool):
            return None
        by_index[index] = {"index": index, "active": bool(active)}
    wanted = set(range(1, int(n_examples) + 1))
    if set(by_index) != wanted:
        return None
    return [by_index[index] for index in sorted(by_index)]


def parse_jsonish(text: str) -> Any:
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def build_request_payload(*, args: argparse.Namespace, messages: list[dict[str, str]], autointerp: Any) -> dict[str, Any]:
    if args.judge_provider == "saebench_openai":
        max_tokens = int(autointerp.cfg.max_tokens_in_prediction)
        if args.judge_mode == "binary":
            max_tokens = max(max_tokens, 1200)
        return {"provider": "saebench_openai", "model": "gpt-4o-mini", "messages": messages, "max_tokens": max_tokens}
    judge_model = resolve_model(str(args.judge_model or default_model_for_provider(str(args.judge_provider))))
    if args.judge_provider == "ollama":
        return {"model": judge_model, "messages": messages, "think": False, "stream": False, "options": {"num_predict": int(args.max_tokens or autointerp.cfg.max_tokens_in_prediction)}}
    max_tokens = int(args.max_tokens or autointerp.cfg.max_tokens_in_prediction)
    if args.judge_mode == "binary":
        max_tokens = max(max_tokens, 1200)
    payload = {"model": judge_model, "max_tokens": max_tokens, "messages": messages}
    if args.judge_provider == "mi":
        payload["thinking"] = {"type": "disabled"}
    if args.judge_provider in {"openai", "deepseek"} and args.judge_mode == "binary":
        payload["response_format"] = {"type": "json_object"}
    apply_openai_reasoning_controls(payload, provider=str(args.judge_provider), model=judge_model)
    return payload


def call_judge(*, args: argparse.Namespace, request_payload: dict[str, Any], autointerp: Any) -> tuple[str, dict[str, Any]]:
    if args.judge_provider == "saebench_openai":
        responses, logs = autointerp.get_api_response(request_payload["messages"], int(request_payload["max_tokens"]))
        return responses[0], {"logs": logs}
    api_key = "" if args.judge_provider == "ollama" else load_api_key_for_provider(
        str(args.judge_provider),
        judge_api_key_file(args),
    )
    response = call_annotation_provider(
        request_payload=request_payload,
        provider=str(args.judge_provider),
        api_key=api_key,
        base_url=str(args.base_url or default_base_url_for_provider(str(args.judge_provider))),
        timeout=float(args.timeout),
        retries=int(args.retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
    )
    return extract_chat_completion_text(response), response


def judge_api_key_file(args: argparse.Namespace) -> Path:
    if args.api_key_file is not None:
        return Path(args.api_key_file)
    if args.judge_provider == "openai":
        return DEFAULT_OPENAI_TOKEN_PATH
    if args.judge_provider == "deepseek":
        return DEFAULT_DEEPSEEK_TOKEN_PATH
    if args.judge_provider == "tinker-sdk":
        return DEFAULT_TINKER_TOKEN_PATH
    return DEFAULT_MI_TOKEN_PATH


def load_or_tokenize_dataset(*, config: Any, model: Any, artifacts_path: Path, device: str) -> torch.Tensor:
    import sae_bench.sae_bench_utils.dataset_utils as dataset_utils

    folder = artifacts_path / "autointerp"
    folder.mkdir(parents=True, exist_ok=True)
    tokens_path = folder / f"{safe_filename(config.model_name)}_{config.total_tokens}_tokens_{config.llm_context_size}_ctx.pt"
    if tokens_path.is_file():
        return torch.load(tokens_path).to(device)
    tokens = dataset_utils.load_and_tokenize_dataset(
        config.dataset_name,
        config.llm_context_size,
        config.total_tokens,
        model.tokenizer,
    ).to(device)
    torch.save(tokens, tokens_path)
    return tokens


def gather_causal_left_scoring_examples(*, autointerp: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    """Gather AutoInterp examples, but show judges causal-left context.

    SAEBench computes activations on the full context, then displays a ±buffer
    local window. For causal LLM residual features, omitted left context can be
    causal. This mirrors SAEBench's sampling while replacing only the displayed
    scoring window with tokens from sequence start through +buffer right context.
    """

    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    from sae_bench.evals.autointerp.main import Example, Examples
    from sae_bench.sae_bench_utils.indexing_utils import get_iw_sample_indices, get_k_largest_indices, index_with_buffer

    cfg = autointerp.cfg
    dataset_size, seq_len = autointerp.tokenized_dataset.shape
    acts = activation_collection.collect_sae_activations(
        autointerp.tokenized_dataset,
        autointerp.model,
        autointerp.sae,
        cfg.llm_batch_size,
        autointerp.sae.cfg.hook_layer,
        autointerp.sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
        selected_latents=autointerp.latents,
        activation_dtype=torch.bfloat16,
    )

    generation_examples: dict[int, Any] = {}
    scoring_examples: dict[int, Any] = {}

    def causal_slice(tensor: torch.Tensor, indices: torch.Tensor, *, fill_zero: bool = False) -> list[torch.Tensor]:
        rows, cols = indices.unbind(dim=-1)
        out = []
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            end = min(int(seq_len), int(col) + int(cfg.buffer) + 1)
            if fill_zero:
                out.append(torch.zeros(end, dtype=torch.float32, device=tensor.device))
            else:
                out.append(tensor[int(row), :end])
        return out

    def make_examples(
        toks_list: list[torch.Tensor],
        acts_list: list[torch.Tensor],
        indices: torch.Tensor,
        *,
        act_threshold: float,
        context_mode: str,
    ) -> list[Any]:
        examples = []
        for toks, values, (row, col) in zip(toks_list, acts_list, indices.tolist(), strict=True):
            example = Example(
                toks=[int(tok) for tok in toks.tolist()],
                acts=[float(act) for act in values.tolist()],
                act_threshold=float(act_threshold),
                model=autointerp.model,
            )
            example.metadata = {
                "context_mode": context_mode,
                "source_row": int(row),
                "target_token_index": int(col),
                "display_token_start": 0,
                "display_token_end": int(len(toks) - 1),
                "right_context_tokens": int(min(int(cfg.buffer), int(len(toks) - int(col) - 1))),
            }
            examples.append(example)
        return examples

    for feature_position, latent in tqdm(
        enumerate(autointerp.latents),
        desc="Collecting causal-left examples for LLM judge",
        total=len(autointerp.latents),
    ):
        rand_indices = torch.stack(
            [
                torch.randint(0, dataset_size, (cfg.n_random_ex_for_scoring,), device=acts.device),
                torch.randint(cfg.buffer, seq_len - cfg.buffer, (cfg.n_random_ex_for_scoring,), device=acts.device),
            ],
            dim=-1,
        )

        feature_acts = acts[..., feature_position]
        top_indices = get_k_largest_indices(
            feature_acts,
            k=cfg.n_top_ex,
            buffer=cfg.buffer,
            no_overlap=cfg.no_overlap,
        )
        top_values_for_threshold = index_with_buffer(feature_acts, top_indices, buffer=cfg.buffer)
        act_threshold = float(cfg.act_threshold_frac * top_values_for_threshold.max().item())

        threshold = top_values_for_threshold[:, cfg.buffer].min().item()
        acts_thresholded = torch.where(feature_acts >= threshold, 0.0, feature_acts)
        if acts_thresholded[:, cfg.buffer : -cfg.buffer].max() < 1e-6:
            continue
        iw_indices = get_iw_sample_indices(
            acts_thresholded,
            k=cfg.n_iw_sampled_ex,
            buffer=cfg.buffer,
        )

        top_toks = causal_slice(autointerp.tokenized_dataset, top_indices)
        top_values = causal_slice(feature_acts, top_indices)
        iw_toks = causal_slice(autointerp.tokenized_dataset, iw_indices)
        iw_values = causal_slice(feature_acts, iw_indices)
        rand_toks = causal_slice(autointerp.tokenized_dataset, rand_indices)
        rand_values = causal_slice(feature_acts, rand_indices, fill_zero=True)

        top_split = torch.randperm(cfg.n_top_ex, device=acts.device)
        top_gen_indices = top_split[: cfg.n_top_ex_for_generation]
        top_scoring_indices = top_split[cfg.n_top_ex_for_generation :]
        iw_split = torch.randperm(cfg.n_iw_sampled_ex, device=acts.device)
        iw_gen_indices = iw_split[: cfg.n_iw_sampled_ex_for_generation]
        iw_scoring_indices = iw_split[cfg.n_iw_sampled_ex_for_generation :]

        def select(items: list[torch.Tensor], positions: torch.Tensor) -> list[torch.Tensor]:
            return [items[int(index)] for index in positions.tolist()]

        generation_examples[int(latent)] = Examples(
            make_examples(
                select(top_toks, top_gen_indices),
                select(top_values, top_gen_indices),
                top_indices[top_gen_indices],
                act_threshold=act_threshold,
                context_mode="causal_left_generation_top",
            )
            + make_examples(
                select(iw_toks, iw_gen_indices),
                select(iw_values, iw_gen_indices),
                iw_indices[iw_gen_indices],
                act_threshold=act_threshold,
                context_mode="causal_left_generation_iw",
            ),
        )
        scoring_examples[int(latent)] = Examples(
            make_examples(
                select(top_toks, top_scoring_indices),
                select(top_values, top_scoring_indices),
                top_indices[top_scoring_indices],
                act_threshold=act_threshold,
                context_mode="causal_left_scoring_top",
            )
            + make_examples(
                select(iw_toks, iw_scoring_indices),
                select(iw_values, iw_scoring_indices),
                iw_indices[iw_scoring_indices],
                act_threshold=act_threshold,
                context_mode="causal_left_scoring_iw",
            )
            + make_examples(
                rand_toks,
                rand_values,
                rand_indices,
                act_threshold=act_threshold,
                context_mode="causal_left_scoring_random",
            ),
            shuffle=True,
        )

    return generation_examples, scoring_examples


def load_or_gather_scoring_examples(*, args: argparse.Namespace, autointerp: Any, feature_ids: list[int], model: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    cache_path = scoring_examples_cache_path(args=args, feature_ids=feature_ids)
    if cache_path.is_file() and not args.force_rerun:
        packet = read_json(cache_path)
        print(f"reuse SAEBench scoring examples: {cache_path}")
        return deserialize_examples_packet(packet, model=model)
    if str(args.example_context) == "saebench_window":
        generation_examples, scoring_examples = autointerp.gather_data()
    else:
        generation_examples, scoring_examples = gather_causal_left_scoring_examples(autointerp=autointerp)
    write_json(
        cache_path,
        {
            "created_at_unix": time.time(),
            "cache_key": cache_path.stem,
            "model": str(args.model),
            "layer": canonical_layer(str(args.layer)),
            "target_kind": str(args.target_kind),
            "feature_ids": [int(fid) for fid in feature_ids],
            "settings": scoring_examples_cache_settings(args),
            "generation_examples": serialize_examples_dict(generation_examples),
            "scoring_examples": serialize_examples_dict(scoring_examples),
        },
    )
    print(f"wrote SAEBench scoring examples: {cache_path}")
    return generation_examples, scoring_examples


def scoring_examples_cache_path(*, args: argparse.Namespace, feature_ids: list[int]) -> Path:
    key = scoring_examples_cache_key(args=args, feature_ids=feature_ids)
    return (
        Path(args.saebench_artifacts_path)
        / "autointerp_label_compare"
        / "scoring_examples"
        / str(args.target_kind)
        / str(args.model)
        / canonical_layer(str(args.layer))
        / f"{key}.json"
    )


def scoring_examples_cache_key(*, args: argparse.Namespace, feature_ids: list[int]) -> str:
    payload = {
        "model": str(args.model),
        "layer": canonical_layer(str(args.layer)),
        "target_kind": str(args.target_kind),
        "feature_ids": [int(fid) for fid in feature_ids],
        "settings": scoring_examples_cache_settings(args),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def scoring_examples_cache_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_name": str(args.dataset_name),
        "total_tokens": int(args.total_tokens),
        "context_size": int(args.context_size),
        "buffer": int(args.buffer),
        "example_context": str(args.example_context),
        "act_threshold_frac": float(args.act_threshold_frac),
        "n_top_scoring": int(args.n_top_scoring),
        "n_random_scoring": int(args.n_random_scoring),
        "n_iw_scoring": int(args.n_iw_scoring),
        "n_top_generation": int(args.n_top_generation),
        "n_iw_generation": int(args.n_iw_generation),
        "random_seed": int(args.random_seed),
    }


def serialize_examples_dict(examples_by_feature: dict[int, Any]) -> dict[str, list[dict[str, Any]]]:
    return {str(int(feature_id)): serialize_examples_with_raw(example_group) for feature_id, example_group in sorted(examples_by_feature.items())}


def serialize_examples_with_raw(examples: Any) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        row = {
            "toks": [int(tok) for tok in example.toks],
            "acts": [float(act) for act in example.acts],
            "act_threshold": float(example.act_threshold),
        }
        metadata = getattr(example, "metadata", None)
        if isinstance(metadata, dict):
            row["metadata"] = metadata
        rows.append(row)
    return rows


def scoring_examples_hash(examples: Any) -> str:
    blob = json.dumps(
        serialize_examples_with_raw(examples),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def deserialize_examples_packet(packet: dict[str, Any], *, model: Any) -> tuple[dict[int, Any], dict[int, Any]]:
    from sae_bench.evals.autointerp.main import Example, Examples

    def build_examples(rows: Any) -> list[Any]:
        examples = []
        for row in rows if isinstance(rows, list) else []:
            example = Example(
                toks=[int(tok) for tok in row.get("toks", [])],
                acts=[float(act) for act in row.get("acts", [])],
                act_threshold=float(row.get("act_threshold", 0.0)),
                model=model,
            )
            if isinstance(row.get("metadata"), dict):
                example.metadata = row["metadata"]
            examples.append(example)
        return examples

    def examples_preserve_order(examples: list[Any]) -> Any:
        group = Examples([], shuffle=False)
        group.examples = examples
        return group

    def convert_generation(groups: Any) -> dict[int, Any]:
        out = {}
        if not isinstance(groups, dict):
            return out
        for feature_id, rows in groups.items():
            out[int(feature_id)] = Examples(build_examples(rows), shuffle=False)
        return out

    def convert_scoring(groups: Any) -> dict[int, Any]:
        out = {}
        if not isinstance(groups, dict):
            return out
        for feature_id, rows in groups.items():
            out[int(feature_id)] = examples_preserve_order(build_examples(rows))
        return out

    return convert_generation(packet.get("generation_examples")), convert_scoring(packet.get("scoring_examples"))


def load_saebench_openai_key(args: argparse.Namespace) -> str:
    root, _python = saebench_env(str(args.model))
    path = root / "openai_api_key.txt"
    if not path.is_file():
        raise FileNotFoundError(f"SAEBench OpenAI judge expects API key file: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"SAEBench OpenAI API key file is empty: {path}")
    return key


def collect_candidates(*, args: argparse.Namespace, feature_id: int) -> list[dict[str, Any]]:
    candidates = []
    feature_dir = f"F{feature_id:06d}"
    if args.target_kind == "sae_counterpart":
        neuronpedia = lookup_neuronpedia_label(args=args, feature_id=feature_id)
        if neuronpedia is not None and str(neuronpedia.get("description") or "").strip():
            desc = str(neuronpedia.get("description") or "").strip()
            candidates.append(
                {
                    "candidate_id": "neuronpedia",
                    "source": "neuronpedia",
                    "label": "",
                    "simple_label": "",
                    "description": desc,
                    "explanation": desc,
                    "source_path": str(args.neuronpedia_db_path),
                    "metadata": {key: value for key, value in neuronpedia.items() if key != "description"},
                }
            )
        else:
            candidates.append(missing_candidate("neuronpedia", args.neuronpedia_db_path))
    initial_path = annotation_path(args=args, feature_id=feature_id)
    candidates.append(candidate_from_annotation_path("auto_initial", initial_path))
    latest_path = latest_refinement_path(args=args, feature_id=feature_id)
    if latest_path is not None and latest_path != initial_path:
        candidates.append(candidate_from_annotation_path("auto_latest", latest_path))
    return candidates


def annotation_path(*, args: argparse.Namespace, feature_id: int) -> Path:
    model_folder = str(args.model) if args.target_kind == "sae_counterpart" else RUN_NAMES[str(args.model)]
    return (
        args.annotation_root
        / model_folder
        / canonical_layer(str(args.layer))
        / f"F{feature_id:06d}"
        / f"{args.provider_label}_annotation.json"
    )


def lookup_neuronpedia_label(*, args: argparse.Namespace, feature_id: int) -> dict[str, Any] | None:
    db_path = Path(args.neuronpedia_db_path)
    counterpart = SAE_COUNTERPARTS[str(args.model)]
    identity = neuronpedia_identity(counterpart.repo_id)
    if identity is None or not db_path.is_file():
        return None
    _model_slug, sae_set = identity
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT description, explanation_model_name, neuronpedia_id, type_name, author_id, source_path
                FROM neuronpedia_labels
                WHERE model_name = ? AND sae_set = ? AND layer_index = ? AND feature_id = ?
                """,
                (counterpart.sae_model_name, sae_set, layer_index(str(args.layer)), int(feature_id)),
            ).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return {
        "description": str(row[0] or ""),
        "explanation_model_name": row[1],
        "neuronpedia_id": row[2],
        "type_name": row[3],
        "author_id": row[4],
        "source_path": row[5],
    }


def neuronpedia_identity(counterpart_repo: str) -> tuple[str, str] | None:
    repo = str(counterpart_repo).lower()
    if "gpt2-small-oai-v5-32k-resid-post" in repo:
        return ("gpt2-small", "res_post_32k-oai")
    if "gemma-scope-2b-pt-res" in repo:
        return ("gemma-2-2b", "gemmascope-res-16k")
    if "sae-res-qwen3.5-2b-base-w32k" in repo:
        return ("qwen3.5-2b-pt", "qwenscope-res-32k")
    return None


def candidate_from_annotation(candidate_id: str, annotation: dict[str, Any], path: Path) -> dict[str, Any]:
    label = str(annotation.get("label") or "").strip()
    simple = str(annotation.get("simple_label") or "").strip()
    desc = str(annotation.get("description") or "").strip()
    pieces = [piece for piece in [label, simple if simple and simple != label else "", desc] if piece]
    return {
        "candidate_id": candidate_id,
        "source": candidate_id,
        "label": label,
        "simple_label": simple,
        "description": desc,
        "explanation": ". ".join(pieces),
        "source_path": str(path),
        "confidence": annotation.get("confidence"),
    }


def candidate_from_annotation_path(candidate_id: str, path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return missing_candidate(candidate_id, path)
    try:
        annotation = read_json(path).get("annotation")
    except Exception as exc:
        return missing_candidate(candidate_id, path, reason=f"Could not read annotation JSON: {exc}")
    if not isinstance(annotation, dict):
        return missing_candidate(candidate_id, path, reason="Annotation file has no annotation object.")
    candidate = candidate_from_annotation(candidate_id, annotation, path)
    if not str(candidate.get("explanation") or "").strip():
        return missing_candidate(candidate_id, path, reason="Annotation has no usable label text.")
    return candidate


def missing_candidate(candidate_id: str, path: Path | None, reason: str = "Label candidate is missing.") -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source": candidate_id,
        "label": "",
        "simple_label": "",
        "description": "",
        "explanation": "",
        "source_path": str(path) if path is not None else "",
        "invalid": True,
        "invalid_reason": reason,
    }


def latest_refinement_path(*, args: argparse.Namespace, feature_id: int) -> Path | None:
    model_folder = str(args.model) if args.target_kind == "sae_counterpart" else RUN_NAMES[str(args.model)]
    folder = args.refinement_root / model_folder / canonical_layer(str(args.layer)) / f"F{feature_id:06d}"
    first_stop = first_stopped_refinement_round(folder=folder, provider_label=str(args.provider_label))
    paths = [
        path
        for path in sorted(folder.glob(f"{args.provider_label}_round*_annotation.json"))
        if first_stop is None or refinement_round_index(path) < first_stop
    ]
    if not paths:
        return None
    return max(paths, key=refinement_round_index)


def first_stopped_refinement_round(*, folder: Path, provider_label: str) -> int | None:
    stop_rounds = [
        refinement_round_index(path)
        for path in folder.glob(f"{provider_label}_round*_stop.json")
    ]
    return min(stop_rounds) if stop_rounds else None


def refinement_round_index(path: Path) -> int:
    return int(path.name.split("_round", 1)[1].split("_", 1)[0])


def serialize_examples(examples: Any) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples, start=1):
        row = {
            "index": index,
            "is_active": bool(example.is_active),
            "max_activation": float(max(example.acts)),
            "sequence": example.to_str(mark_toks=False),
            "marked_sequence": example.to_str(mark_toks=True),
        }
        metadata = getattr(example, "metadata", None)
        if isinstance(metadata, dict):
            row["metadata"] = metadata
        rows.append(row)
    return rows


def write_score_tables(*, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "scores.json", rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        score = row.get("saebench_score")
        if score is not None:
            grouped.setdefault(str(row.get("candidate_id")), []).append(float(score))
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items()) if values}


def result_output_dir(args: argparse.Namespace) -> Path:
    judge = "saebench_openai" if args.judge_provider == "saebench_openai" else safe_filename(resolve_model(str(args.judge_model or default_model_for_provider(str(args.judge_provider)))))
    suffix = "" if str(args.judge_mode) == "choice" else f"_{safe_filename(str(args.judge_mode))}"
    return args.output_root / str(args.target_kind) / str(args.model) / canonical_layer(str(args.layer)) / f"{args.provider_label}_judge_{judge}{suffix}"


def selected_feature_ids(args: argparse.Namespace) -> list[int]:
    feature_ids = list(args.feature_id or [])
    if args.feature_start is not None or args.feature_end is not None:
        if args.feature_start is None or args.feature_end is None:
            raise SystemExit("--feature-start and --feature-end must be provided together.")
        feature_ids.extend(range(int(args.feature_start), int(args.feature_end) + 1))
    feature_ids = sorted(set(int(item) for item in feature_ids))
    if not feature_ids:
        raise SystemExit("Provide --feature-id or --feature-start/--feature-end.")
    return feature_ids


def canonical_layer(layer: str) -> str:
    return f"layer_{layer_index(layer):02d}"


if __name__ == "__main__":
    main()
