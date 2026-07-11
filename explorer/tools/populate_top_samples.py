from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sqlite3
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer


V9_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = V9_ROOT / "diagnostics" / "math-cued-ica" / "feature_index.sqlite"
DEFAULT_ACTIVATION_ROOT = Path("/home/liusida/data/ICA-data/math-cued-activation")
DATASET_SLUG = "OpenEvals__IMO-AnswerBench"
RUNS = {
    "math_cued_qwen_layer32_c2048_iter100": {
        "model_slug": "Qwen__Qwen2.5-Coder-3B-Instruct",
        "tokenizer": "Qwen/Qwen2.5-Coder-3B-Instruct",
    },
    "math_cued_vibethinker_layer32_c2048_iter100": {
        "model_slug": "WeiboAI__VibeThinker-3B",
        "tokenizer": "WeiboAI/VibeThinker-3B",
    },
    "math_cued_vibethinker_only_layer32_c2048_iter100": {
        "model_slug": "WeiboAI__VibeThinker-3B",
        "tokenizer": "WeiboAI/VibeThinker-3B",
    },
}


@dataclass(frozen=True)
class ActivationFile:
    path: Path
    rows: int
    hidden_size: int


@dataclass(frozen=True)
class RowRef:
    path: Path
    local_row: int
    global_row: int


def _run_model(db_path: Path, run_id: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT model_id FROM model_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"No run row for {run_id} in {db_path}")
    return str(row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate Math-Cued diagnostic top samples for Explorer feature pages.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--run-id", action="append", help="Run to update; repeatable.")
    parser.add_argument("--model-slug", help="Activation directory model slug for custom runs.")
    parser.add_argument("--tokenizer", help="Tokenizer ID for custom runs.")
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--layer", default="layer_32")
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--write-workers", type=int, default=1, help="Parallel CPU workers for per-feature evidence JSON writing.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()

    device = _device(args.device)
    dtype = _dtype(args.dtype)
    for run_id in args.run_id or list(RUNS):
        cfg = RUNS.get(run_id, {})
        print(f"[{run_id}] populate top samples")
        _populate_run(
            db_path=args.db_path.resolve(),
            activation_root=args.activation_root.expanduser(),
            run_id=run_id,
            model_slug=args.model_slug or cfg.get("model_slug") or _run_model(args.db_path, run_id).replace("/", "__"),
            tokenizer_name=args.tokenizer or cfg.get("tokenizer") or _run_model(args.db_path, run_id),
            dataset_slug=args.dataset_slug,
            layer=args.layer,
            max_rows=int(args.max_rows),
            seed=int(args.seed),
            examples=int(args.examples),
            context_window=int(args.context_window),
            chunk_size=int(args.chunk_size),
            write_workers=max(1, min(16, int(args.write_workers))),
            device=device,
            dtype=dtype,
        )


def _populate_run(
    *,
    db_path: Path,
    activation_root: Path,
    run_id: str,
    model_slug: str,
    tokenizer_name: str,
    dataset_slug: str,
    layer: str,
    max_rows: int,
    seed: int,
    examples: int,
    context_window: int,
    chunk_size: int,
    write_workers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT feature_pt_path, source_ica_artifact FROM layers WHERE run_id = ? AND layer = ?",
            (run_id, layer),
        ).fetchone()
    if row is None:
        raise KeyError(f"No layer row for {run_id}/{layer}")
    feature_path = Path(row["feature_pt_path"])
    feature_artifact = torch.load(feature_path, map_location="cpu", weights_only=False)["tensors"]
    feature_directions = feature_artifact["feature_directions"].detach().to(device=device, dtype=dtype)
    mean = feature_artifact["preprocess_mean"].detach().to(device=device, dtype=dtype)
    if mean.ndim == 2:
        mean = mean[0]
    n_features = int(feature_directions.shape[0])

    files = _activation_files(activation_root / dataset_slug / model_slug / layer)
    total_rows = sum(file.rows for file in files)
    selected_rows = _choose_global_rows(total_rows, min(max_rows, total_rows), random.Random(seed))
    file_rows = _rows_by_file(files, selected_rows)
    print(f"  selected {len(selected_rows):,} / {total_rows:,} rows from {model_slug}")

    top_values = torch.full((n_features, examples), -torch.inf, dtype=torch.float32, device=device)
    top_ref_ids = torch.full((n_features, examples), -1, dtype=torch.long, device=device)
    refs: list[RowRef] = []
    path_contexts: dict[Path, tuple[list[int], dict[str, Any]]] = {}

    for path, rows, global_rows, chunk, token_ids, problem in _iter_selected_chunks(file_rows, chunk_size=chunk_size, device=device, dtype=dtype):
        if path not in path_contexts:
            path_contexts[path] = (token_ids, problem)
        start_ref = len(refs)
        refs.extend(RowRef(path=path, local_row=int(local), global_row=int(global_row)) for local, global_row in zip(rows, global_rows, strict=True))
        ref_ids = torch.arange(start_ref, start_ref + len(rows), dtype=torch.long, device=device)
        acts = torch.relu((chunk - mean) @ feature_directions.T).float()
        chunk_values, chunk_indices = torch.topk(acts, k=min(examples, acts.shape[0]), dim=0)
        chunk_values = chunk_values.T.contiguous()
        chunk_ref_ids = ref_ids[chunk_indices.T.contiguous()]
        merged_values = torch.cat([top_values, chunk_values], dim=1)
        merged_ref_ids = torch.cat([top_ref_ids, chunk_ref_ids], dim=1)
        values, order = torch.topk(merged_values, k=examples, dim=1)
        top_values = values
        top_ref_ids = torch.gather(merged_ref_ids, 1, order)

    evidence_root = feature_path.parent / f"{layer}_top_samples"
    evidence_root.mkdir(parents=True, exist_ok=True)
    values_cpu = top_values.cpu()
    refs_cpu = top_ref_ids.cpu()

    feature_refs_by_feature: list[list[tuple[int, RowRef, float]]] = []
    build_refs_by_path: dict[Path, list[ExampleBuildRef]] = defaultdict(list)
    for feature_id in range(n_features):
        feature_refs = []
        for rank in range(examples):
            ref_id = int(refs_cpu[feature_id, rank].item())
            if ref_id < 0:
                continue
            ref = refs[ref_id]
            activation = float(values_cpu[feature_id, rank].item())
            feature_refs.append((rank, ref, activation))
            build_refs_by_path[ref.path].append(
                ExampleBuildRef(
                    feature_id=feature_id,
                    rank=rank,
                    local_row=ref.local_row,
                    global_row=ref.global_row,
                    activation=activation,
                )
            )
        feature_refs_by_feature.append(feature_refs)

    example_rows_by_feature: list[list[dict[str, Any] | None]] = [[None] * examples for _ in range(n_features)]
    path_tasks = [
        PathExampleBuildTask(
            path=path,
            refs=refs_for_path,
            token_ids=path_contexts[path][0],
            problem=path_contexts[path][1],
            context_window=context_window,
        )
        for path, refs_for_path in build_refs_by_path.items()
    ]
    if write_workers == 1:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        for built_examples in tqdm(
            (_build_examples_for_path(task, tokenizer=tokenizer) for task in path_tasks),
            total=len(path_tasks),
            desc="build evidence examples",
            unit="file",
        ):
            _store_built_examples(example_rows_by_feature, built_examples)
    else:
        print(f"  building evidence examples with {write_workers} CPU worker(s)")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=write_workers,
            initializer=_init_write_worker,
            initargs=(tokenizer_name,),
        ) as executor:
            futures = [executor.submit(_build_examples_for_path, task) for task in path_tasks]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="build evidence examples",
                unit="file",
            ):
                _store_built_examples(example_rows_by_feature, future.result())

    updates = [
        _write_feature_evidence(
            evidence_root=evidence_root,
            run_id=run_id,
            layer=layer,
            feature_id=feature_id,
            selected_rows=len(selected_rows),
            total_rows=total_rows,
            seed=seed,
            max_rows=max_rows,
            examples=examples,
            rows=[row for row in example_rows_by_feature[feature_id] if row is not None],
        )
        for feature_id in tqdm(range(n_features), desc="write evidence", unit="feature")
    ]

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            UPDATE features
            SET annotation_evidence_path = ?, annotation_evidence_json = ?
            WHERE run_id = ? AND layer = ? AND feature_id = ?
            """,
            updates,
        )
        conn.commit()
    print(f"  wrote top samples under {evidence_root}")


@dataclass(frozen=True)
class ExampleBuildRef:
    feature_id: int
    rank: int
    local_row: int
    global_row: int
    activation: float


@dataclass(frozen=True)
class PathExampleBuildTask:
    path: Path
    refs: list[ExampleBuildRef]
    token_ids: list[int]
    problem: dict[str, Any]
    context_window: int


_WORKER_TOKENIZER: Any | None = None


def _init_write_worker(tokenizer_name: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def _build_examples_for_path(task: PathExampleBuildTask, tokenizer: Any | None = None) -> list[tuple[int, int, dict[str, Any]]]:
    tokenizer = tokenizer or _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("Evidence writer tokenizer was not initialized")
    return [
        (
            ref.feature_id,
            ref.rank,
            _example_from_ref(
                tokenizer=tokenizer,
                token_ids=task.token_ids,
                problem=task.problem,
                source_path=task.path,
                local_row=ref.local_row,
                global_row=ref.global_row,
                feature_id=ref.feature_id,
                rank=ref.rank,
                activation=ref.activation,
                context_window=task.context_window,
            ),
        )
        for ref in task.refs
    ]


def _store_built_examples(
    example_rows_by_feature: list[list[dict[str, Any] | None]],
    built_examples: list[tuple[int, int, dict[str, Any]]],
) -> None:
    for feature_id, rank, example in built_examples:
        example_rows_by_feature[feature_id][rank] = example


def _write_feature_evidence(
    *,
    evidence_root: Path,
    run_id: str,
    layer: str,
    feature_id: int,
    selected_rows: int,
    total_rows: int,
    seed: int,
    max_rows: int,
    examples: int,
    rows: list[dict[str, Any]],
) -> tuple[str, str, str, str, int]:
    rows.sort(key=lambda row: int(row["rank"]))
    packet = {
        "evidence_type": "math_cued_top_activating_examples",
        "run_id": run_id,
        "layer": layer,
        "feature_id": feature_id,
        "selection": {
            "source": "top activations over selected Math-Cued rows",
            "selected_rows": selected_rows,
            "available_rows": total_rows,
            "seed": seed,
            "max_rows": max_rows,
            "examples": examples,
        },
        "examples": rows,
    }
    feature_dir = evidence_root / f"F{feature_id:06d}"
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / "evidence.json"
    text = json.dumps(packet, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return (str(path), text, run_id, layer, feature_id)


def _example_from_ref(
    *,
    tokenizer: Any,
    token_ids: list[int],
    problem: dict[str, Any],
    source_path: Path,
    local_row: int,
    global_row: int,
    feature_id: int,
    rank: int,
    activation: float,
    context_window: int,
) -> dict[str, Any]:
    left = max(0, local_row - context_window)
    right = min(len(token_ids), local_row + context_window + 1)
    context_ids = token_ids[left:right]
    target_id = token_ids[local_row] if 0 <= local_row < len(token_ids) else None
    target_offset = local_row - left
    context_text, target_char_start, target_char_end = _decoded_context_with_target(
        tokenizer,
        context_ids,
        target_offset=target_offset,
    )
    marked_context = (
        context_text[:target_char_start]
        + "[target]"
        + context_text[target_char_start:target_char_end]
        + "[/target]"
        + context_text[target_char_end:]
    )
    return {
        "example_index": rank,
        "rank": rank + 1,
        "feature_id": feature_id,
        "doc_id": problem.get("problem_id") or source_path.stem,
        "position": local_row,
        "global_row": global_row,
        "activation": activation,
        "relative_activation": 1.0,
        "target_token": context_text[target_char_start:target_char_end],
        "context_text": context_text,
        "target_char_start": target_char_start,
        "target_char_end": target_char_end,
        "left_context_ending_at_target": context_text[:target_char_end],
        "right_context_for_readability_not_causal_evidence": context_text[target_char_end:],
        "marked_context": marked_context,
        "effective_receptive_field": {
            "estimated_effective_receptive_field_length": None,
            "largest_observed_relative_score_jump": {
                "from_relative_score": None,
                "to_relative_score": None,
                "from_context_length": None,
                "to_context_length": None,
            },
        },
        "source_path": str(source_path),
    }


def _context_from_bundle(bundle: dict[str, Any]) -> tuple[list[int], dict[str, Any]]:
    tokens = bundle.get("tokens", {}) if isinstance(bundle.get("tokens"), dict) else {}
    token_ids = tokens.get("captured_token_ids") or tokens.get("sequence_token_ids") or []
    if isinstance(token_ids, torch.Tensor):
        token_ids = [int(x) for x in token_ids.detach().cpu().tolist()]
    else:
        token_ids = [int(x) for x in token_ids]
    problem = bundle.get("problem", {}) if isinstance(bundle.get("problem"), dict) else {}
    return token_ids, dict(problem)


def _token_texts(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [_decode_token(tokenizer, token_id) for token_id in token_ids]


def _decoded_context_with_target(
    tokenizer: Any,
    token_ids: list[int],
    *,
    target_offset: int,
) -> tuple[str, int, int]:
    """Decode once, then locate the visible span affected by the target token.

    A model token may contain only part of a UTF-8 character, so decoding the
    target token or every context token independently is not text-safe. The
    stable prefix before the target and stable suffix after it delimit the
    complete visible character span associated with the target token.
    """
    context_text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    before_text = tokenizer.decode(token_ids[:target_offset], clean_up_tokenization_spaces=False)
    after_text = tokenizer.decode(token_ids[target_offset + 1 :], clean_up_tokenization_spaces=False)

    target_char_start = 0
    prefix_limit = min(len(context_text), len(before_text))
    while target_char_start < prefix_limit and context_text[target_char_start] == before_text[target_char_start]:
        target_char_start += 1

    suffix_length = 0
    suffix_limit = min(len(context_text) - target_char_start, len(after_text))
    while suffix_length < suffix_limit and context_text[-1 - suffix_length] == after_text[-1 - suffix_length]:
        suffix_length += 1
    target_char_end = len(context_text) - suffix_length
    if target_char_end < target_char_start:
        target_char_end = target_char_start
    return context_text, target_char_start, target_char_end


def _decode_token(tokenizer: Any, token_id: int | None) -> str:
    if token_id is None:
        return ""
    return tokenizer.decode([int(token_id)])


def _activation_files(layer_dir: Path) -> list[ActivationFile]:
    files: list[ActivationFile] = []
    for path in sorted(layer_dir.glob("*.pt")):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        acts = bundle.get("activations")
        if isinstance(acts, torch.Tensor) and acts.ndim == 2:
            files.append(ActivationFile(path=path, rows=int(acts.shape[0]), hidden_size=int(acts.shape[1])))
    if not files:
        raise FileNotFoundError(f"No activation bundles under {layer_dir}")
    return files


def _choose_global_rows(total_rows: int, max_rows: int, rng: random.Random) -> list[int]:
    if max_rows >= total_rows:
        return list(range(total_rows))
    return sorted(rng.sample(range(total_rows), max_rows))


def _rows_by_file(files: list[ActivationFile], selected_rows: list[int]) -> dict[Path, tuple[list[int], list[int]]]:
    out: dict[Path, tuple[list[int], list[int]]] = {}
    cursor = 0
    selected_cursor = 0
    for file in files:
        start = cursor
        end = cursor + file.rows
        local: list[int] = []
        global_rows: list[int] = []
        while selected_cursor < len(selected_rows) and selected_rows[selected_cursor] < end:
            global_row = selected_rows[selected_cursor]
            if global_row >= start:
                local.append(global_row - start)
                global_rows.append(global_row)
            selected_cursor += 1
        if local:
            out[file.path] = (local, global_rows)
        cursor = end
    return out


def _iter_selected_chunks(
    file_rows: dict[Path, tuple[list[int], list[int]]],
    *,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    for path, (rows, global_rows) in tqdm(file_rows.items(), desc="activation files", unit="file"):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        acts = bundle["activations"]
        token_ids, problem = _context_from_bundle(bundle)
        for start in range(0, len(rows), chunk_size):
            local_rows = rows[start : start + chunk_size]
            local_global = global_rows[start : start + chunk_size]
            idx = torch.tensor(local_rows, dtype=torch.long)
            chunk = acts.index_select(0, idx).to(device=device, dtype=dtype)
            yield path, local_rows, local_global, F.normalize(chunk, p=2, dim=1, eps=1e-12), token_ids, problem


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


if __name__ == "__main__":
    main()
