from __future__ import annotations

import argparse
import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ica_lens_v9.features.exports import write_feature_exports_from_artifact


V9_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = V9_ROOT / "diagnostics" / "math-cued-ica" / "feature_index.sqlite"
DEFAULT_ACTIVATION_ROOT = Path("/home/liusida/data/ICA-data/math-cued-activation")
DATASET_SLUG = "OpenEvals__IMO-AnswerBench"
RUN_TO_MODEL_SLUG = {
    "math_cued_qwen_layer32_c2048_iter100": "Qwen__Qwen2.5-Coder-3B-Instruct",
    "math_cued_vibethinker_layer32_c2048_iter100": "WeiboAI__VibeThinker-3B",
    "math_cued_vibethinker_only_layer32_c2048_iter100": "WeiboAI__VibeThinker-3B",
}
RUN_TO_ORDER_GROUP = {
    "math_cued_qwen_layer32_c2048_iter100": "mixed_qwen_vibethinker",
    "math_cued_vibethinker_layer32_c2048_iter100": "mixed_qwen_vibethinker",
    "math_cued_vibethinker_only_layer32_c2048_iter100": "vibethinker_only",
}


@dataclass(frozen=True)
class ActivationFile:
    path: Path
    rows: int
    hidden_size: int


@dataclass
class RunStats:
    run_id: str
    model_slug: str
    feature_path: Path
    feature_artifact: dict
    values: dict[str, torch.Tensor]
    histogram_counts: torch.Tensor
    hist_edges: torch.Tensor
    selected_count: int
    total_rows: int
    max_rows: int
    seed: int
    activation_root: Path


def _model_slug_from_db(db_path: Path, run_id: str) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT model_id FROM model_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise KeyError(f"No run row for {run_id} in {db_path}")
    return str(row[0]).replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate diagnostic Math-Cued ICA Explorer feature properties.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--run-id", action="append", help="Run to update; repeatable.")
    parser.add_argument("--model-slug", help="Activation directory model slug for custom runs.")
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--layer", default="layer_32")
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    args = parser.parse_args()

    db_path = args.db_path.resolve()
    run_ids = args.run_id or list(RUN_TO_MODEL_SLUG)
    device = _device(args.device)
    dtype = _dtype(args.dtype)

    run_stats: list[RunStats] = []
    for run_id in run_ids:
        print(f"[{run_id}] populate properties")
        run_stats.append(_compute_run_stats(
            db_path=db_path,
            activation_root=args.activation_root.expanduser(),
            run_id=run_id,
            model_slug=args.model_slug or RUN_TO_MODEL_SLUG.get(run_id) or _model_slug_from_db(db_path, run_id),
            dataset_slug=args.dataset_slug,
            layer=args.layer,
            max_rows=args.max_rows,
            seed=args.seed,
            chunk_size=args.chunk_size,
            device=device,
            dtype=dtype,
        ))

    stats_by_group: dict[str, list[RunStats]] = {}
    for stats in run_stats:
        stats_by_group.setdefault(RUN_TO_ORDER_GROUP.get(stats.run_id, stats.run_id), []).append(stats)
    for group_name, group_stats in stats_by_group.items():
        print(f"[{group_name}] compute shared feature order for {len(group_stats)} run(s)")
        shared_source_order = _joint_source_order(group_stats)
        for stats in group_stats:
            _write_run_with_shared_order(stats=stats, db_path=db_path, layer=args.layer, source_order=shared_source_order)


def _compute_run_stats(
    *,
    db_path: Path,
    activation_root: Path,
    run_id: str,
    model_slug: str,
    dataset_slug: str,
    layer: str,
    max_rows: int,
    seed: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> RunStats:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT feature_pt_path FROM layers WHERE run_id = ? AND layer = ?",
            (run_id, layer),
        ).fetchone()
    if row is None:
        raise KeyError(f"No layer row for {run_id}/{layer} in {db_path}")

    feature_path = Path(row["feature_pt_path"])
    feature_artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    feature_tensors = feature_artifact["tensors"]
    feature_directions = feature_tensors["feature_directions"].detach().to(device=device, dtype=dtype)
    mean = feature_tensors["preprocess_mean"].detach().to(device=device, dtype=dtype)
    if mean.ndim == 2:
        mean = mean[0]
    n_features = int(feature_directions.shape[0])

    files = _activation_files(activation_root / dataset_slug / model_slug / layer)
    total_rows = sum(file.rows for file in files)
    selected_rows = _choose_global_rows(total_rows, min(max_rows, total_rows), random.Random(seed))
    file_rows = _rows_by_file(files, selected_rows)
    selected_count = len(selected_rows)
    print(f"  selected {selected_count:,} / {total_rows:,} rows from {model_slug}")

    stats = _new_stats(n_features, device=device)
    for acts in _iter_selected_chunks(file_rows, chunk_size=chunk_size, device=device, dtype=dtype):
        feature_acts = torch.relu((acts - mean) @ feature_directions.T)
        _accumulate_feature_stats(stats, feature_acts)

    values = _finalize_stats(stats, selected_count)
    hist_edges = _histogram_edges(values["max"])
    histogram_counts = torch.zeros((n_features, hist_edges.numel() - 1), dtype=torch.long)
    hist_edges_device = hist_edges.to(device=device)
    for acts in _iter_selected_chunks(file_rows, chunk_size=chunk_size, device=device, dtype=dtype):
        feature_acts = torch.relu((acts - mean) @ feature_directions.T)
        _accumulate_feature_histograms(histogram_counts, feature_acts, hist_edges_device)

    return RunStats(
        run_id=run_id,
        model_slug=model_slug,
        feature_path=feature_path,
        feature_artifact=feature_artifact,
        values=values,
        histogram_counts=histogram_counts,
        hist_edges=hist_edges,
        selected_count=selected_count,
        total_rows=total_rows,
        max_rows=max_rows,
        seed=seed,
        activation_root=activation_root,
    )


def _write_run_with_shared_order(
    *,
    stats: RunStats,
    db_path: Path,
    layer: str,
    source_order: torch.Tensor,
) -> None:
    feature_artifact = stats.feature_artifact
    values, histogram_counts = _apply_feature_order(
        feature_artifact=feature_artifact,
        values=stats.values,
        histogram_counts=stats.histogram_counts,
        hist_edges=stats.hist_edges,
        source_order=source_order,
    )

    metadata = feature_artifact.setdefault("metadata", {})
    metadata["rows"] = stats.selected_count
    metadata["dead_count"] = int(values["dead"].sum().item())
    metadata["alive_count"] = int((~values["dead"]).sum().item())
    metadata["kurtosis_summary"] = _summary(values["kurtosis"])
    metadata["excess_kurtosis_summary"] = _summary(values["excess_kurtosis"])
    metadata["activation_frequency_summary"] = _summary(values["activation_frequency"])
    metadata["feature_id_convention"] = (
        "feature_id is a shared exposed-feature index across Math-Cued model views, "
        "ordered by descending joint active-mirrored raw kurtosis"
    )
    metadata["source_feature_id_convention"] = (
        "source_feature_id = 2 * source_component_index for positive side, "
        "2 * source_component_index + 1 for negative side"
    )
    metadata["diagnostic_note"] = (
        "Feature properties populated from selected Math-Cued activation rows for this Explorer runtime."
    )
    metadata["property_source"] = {
        "activation_root": str(stats.activation_root),
        "dataset_slug": DATASET_SLUG,
        "model_slug": stats.model_slug,
        "selected_rows": stats.selected_count,
        "available_rows": stats.total_rows,
        "seed": stats.seed,
        "max_rows": stats.max_rows,
    }
    metadata["joint_feature_order"] = {
        "score": "max active-mirrored raw kurtosis across selected Math-Cued model views",
        "source_feature_order": [int(x) for x in source_order.tolist()],
    }
    torch.save(feature_artifact, stats.feature_path)
    write_feature_exports_from_artifact(
        stats.feature_path,
        ranking_csv_path=stats.feature_path.parent / f"{layer}_ranking.csv",
        histogram_csv_path=stats.feature_path.parent / f"{layer}_histograms.csv",
    )

    _update_sqlite(
        db_path=db_path,
        run_id=stats.run_id,
        layer=layer,
        feature_dir=stats.feature_path.parent,
        tensors=feature_artifact["tensors"],
        selected_count=stats.selected_count,
        metadata=metadata,
    )
    print(f"  updated {stats.feature_path}")


def _activation_files(layer_dir: Path) -> list[ActivationFile]:
    files: list[ActivationFile] = []
    for path in sorted(layer_dir.glob("*.pt")):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        acts = bundle.get("activations")
        if not isinstance(acts, torch.Tensor) or acts.ndim != 2:
            continue
        files.append(ActivationFile(path=path, rows=int(acts.shape[0]), hidden_size=int(acts.shape[1])))
    if not files:
        raise FileNotFoundError(f"No activation bundles under {layer_dir}")
    hidden_sizes = {file.hidden_size for file in files}
    if len(hidden_sizes) != 1:
        raise ValueError(f"Hidden size mismatch under {layer_dir}: {sorted(hidden_sizes)}")
    return files


def _choose_global_rows(total_rows: int, max_rows: int, rng: random.Random) -> list[int]:
    if max_rows >= total_rows:
        return list(range(total_rows))
    return sorted(rng.sample(range(total_rows), max_rows))


def _rows_by_file(files: list[ActivationFile], selected_rows: list[int]) -> dict[Path, list[int]]:
    out: dict[Path, list[int]] = {}
    cursor = 0
    selected_cursor = 0
    for file in files:
        start = cursor
        end = cursor + file.rows
        local: list[int] = []
        while selected_cursor < len(selected_rows) and selected_rows[selected_cursor] < end:
            global_row = selected_rows[selected_cursor]
            if global_row >= start:
                local.append(global_row - start)
            selected_cursor += 1
        if local:
            out[file.path] = local
        cursor = end
    return out


def _iter_selected_chunks(
    file_rows: dict[Path, list[int]],
    *,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    iterator = tqdm(file_rows.items(), desc="activation files", unit="file")
    for path, rows in iterator:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        acts = bundle["activations"]
        for start in range(0, len(rows), chunk_size):
            idx = torch.tensor(rows[start : start + chunk_size], dtype=torch.long)
            chunk = acts.index_select(0, idx).to(device=device, dtype=dtype)
            yield F.normalize(chunk, p=2, dim=1, eps=1e-12)


def _new_stats(n_features: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "count_active": torch.zeros(n_features, device=device, dtype=torch.float64),
        "sum": torch.zeros(n_features, device=device, dtype=torch.float64),
        "sum2": torch.zeros(n_features, device=device, dtype=torch.float64),
        "sum4_active": torch.zeros(n_features, device=device, dtype=torch.float64),
        "max": torch.zeros(n_features, device=device, dtype=torch.float32),
    }


def _accumulate_side_stats(stats: dict[str, torch.Tensor], scores: torch.Tensor) -> None:
    pos = torch.relu(scores)
    neg = torch.relu(-scores)
    for offset, acts in [(0, pos), (1, neg)]:
        active = acts > 0
        acts64 = acts.to(torch.float64)
        stats["count_active"][offset::2] += active.sum(dim=0).to(torch.float64)
        stats["sum"][offset::2] += acts64.sum(dim=0)
        stats["sum2"][offset::2] += (acts64 * acts64).sum(dim=0)
        stats["sum4_active"][offset::2] += (acts64 * acts64 * acts64 * acts64).sum(dim=0)
        stats["max"][offset::2] = torch.maximum(stats["max"][offset::2], acts.max(dim=0).values.detach().to(torch.float32))


def _accumulate_feature_stats(stats: dict[str, torch.Tensor], acts: torch.Tensor) -> None:
    active = acts > 0
    acts64 = acts.to(torch.float64)
    stats["count_active"] += active.sum(dim=0).to(torch.float64)
    stats["sum"] += acts64.sum(dim=0)
    stats["sum2"] += (acts64 * acts64).sum(dim=0)
    stats["sum4_active"] += (acts64 * acts64 * acts64 * acts64).sum(dim=0)
    stats["max"] = torch.maximum(stats["max"], acts.max(dim=0).values.detach().to(torch.float32))


def _finalize_stats(stats: dict[str, torch.Tensor], selected_count: int) -> dict[str, torch.Tensor]:
    count_active = stats["count_active"].cpu()
    sums = stats["sum"].cpu()
    sum2 = stats["sum2"].cpu()
    sum4 = stats["sum4_active"].cpu()
    mean = sums / float(selected_count)
    second = sum2 / float(selected_count)
    variance = torch.clamp(second - mean * mean, min=0.0)
    active_second = torch.where(count_active > 0, sum2 / count_active.clamp_min(1), torch.zeros_like(sum2))
    active_fourth = torch.where(count_active > 0, sum4 / count_active.clamp_min(1), torch.zeros_like(sum4))
    kurtosis = torch.where(active_second > 0, active_fourth / (active_second * active_second), torch.zeros_like(active_second))
    activation_frequency = count_active / float(selected_count)
    dead = count_active == 0
    return {
        "mean": mean.to(torch.float32),
        "variance": variance.to(torch.float32),
        "kurtosis": kurtosis.to(torch.float32),
        "activation_frequency": activation_frequency.to(torch.float32),
        "max": stats["max"].cpu().to(torch.float32),
        "dead": dead.cpu(),
    }


def _histogram_edges(max_values: torch.Tensor) -> torch.Tensor:
    upper = float(max_values.max().item())
    if upper <= 0:
        upper = 1.0
    return torch.expm1(torch.linspace(0.0, torch.log1p(torch.tensor(upper)).item(), 20, dtype=torch.float32))


def _accumulate_histograms(histogram_counts: torch.Tensor, scores: torch.Tensor, edges: torch.Tensor) -> None:
    pos = torch.relu(scores)
    neg = torch.relu(-scores)
    edges_log1p = torch.log1p(edges)
    for offset, acts in [(0, pos), (1, neg)]:
        counts = _log1p_histogram_counts(acts, edges_log1p)
        histogram_counts[offset::2] += counts


def _accumulate_feature_histograms(histogram_counts: torch.Tensor, acts: torch.Tensor, edges: torch.Tensor) -> None:
    histogram_counts += _log1p_histogram_counts(acts, torch.log1p(edges))


def _log1p_histogram_counts(values: torch.Tensor, edges_log1p: torch.Tensor) -> torch.Tensor:
    n_features = int(values.shape[1])
    n_bins = int(edges_log1p.numel() - 1)
    edges_device = edges_log1p.to(device=values.device, dtype=torch.float64)
    log_values = torch.log1p(values.to(torch.float64))
    bin_idx = torch.bucketize(log_values, edges_device, right=False) - 1
    bin_idx = bin_idx.clamp_(0, n_bins - 1).to(torch.long)
    offsets = (torch.arange(n_features, device=values.device, dtype=torch.long) * n_bins).view(1, -1)
    flat = (bin_idx + offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=n_features * n_bins).reshape(n_features, n_bins)
    return counts.detach().cpu().to(torch.long)


def _joint_source_order(run_stats: list[RunStats]) -> torch.Tensor:
    if not run_stats:
        raise ValueError("No runs to rank")
    scores_by_source: dict[int, float] = {}
    all_sources: set[int] = set()
    for stats in run_stats:
        tensors = stats.feature_artifact["tensors"]
        source_ids = tensors.get("source_feature_id")
        if not isinstance(source_ids, torch.Tensor):
            raise KeyError(f"Missing source_feature_id in {stats.feature_path}")
        if int(source_ids.numel()) != int(stats.values["kurtosis"].numel()):
            raise ValueError(f"source_feature_id length mismatch in {stats.feature_path}")
        for row_index, source_id_tensor in enumerate(source_ids.detach().cpu()):
            source_id = int(source_id_tensor.item())
            all_sources.add(source_id)
            score = float(stats.values["kurtosis"][row_index].item())
            scores_by_source[source_id] = max(scores_by_source.get(source_id, float("-inf")), score)
    expected = set(range(len(all_sources)))
    if all_sources != expected:
        missing = sorted(expected - all_sources)[:10]
        extra = sorted(all_sources - expected)[:10]
        raise ValueError(f"source_feature_id values are not contiguous 0..N-1; missing={missing}, extra={extra}")
    ordered = sorted(all_sources, key=lambda source_id: (-scores_by_source[source_id], source_id))
    return torch.tensor(ordered, dtype=torch.long)


def _apply_feature_order(
    *,
    feature_artifact: dict,
    values: dict[str, torch.Tensor],
    histogram_counts: torch.Tensor,
    hist_edges: torch.Tensor,
    source_order: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    tensors = feature_artifact["tensors"]
    n_features = int(values["kurtosis"].numel())
    source_ids = tensors.get("source_feature_id")
    if not isinstance(source_ids, torch.Tensor):
        raise KeyError("feature artifact is missing source_feature_id")
    if int(source_order.numel()) != n_features:
        raise ValueError(f"Shared source order has {source_order.numel()} entries, expected {n_features}")
    row_by_source = {int(source_id.item()): row_index for row_index, source_id in enumerate(source_ids.detach().cpu())}
    try:
        order = torch.tensor([row_by_source[int(source_id.item())] for source_id in source_order], dtype=torch.long)
    except KeyError as exc:
        raise ValueError(f"Feature artifact is missing source_feature_id {exc.args[0]}") from exc
    for key, value in list(tensors.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and int(value.shape[0]) == n_features:
            tensors[key] = value.index_select(0, order)
    sorted_values = {key: value.index_select(0, order) for key, value in values.items()}
    sorted_values["excess_kurtosis"] = torch.clamp(sorted_values["kurtosis"] - 3.0, min=0.0)
    tensors["feature_id"] = torch.arange(n_features, dtype=torch.long)
    tensors["kurtosis"] = sorted_values["kurtosis"]
    tensors["excess_kurtosis"] = sorted_values["excess_kurtosis"]
    tensors["dead"] = sorted_values["dead"]
    tensors["activation_frequency"] = sorted_values["activation_frequency"]
    tensors["mean"] = sorted_values["mean"]
    tensors["variance"] = sorted_values["variance"]
    tensors["max"] = sorted_values["max"]
    tensors["histogram_counts"] = histogram_counts.index_select(0, order)
    tensors["histogram_bin_edges"] = hist_edges
    tensors["histogram_bin_edges_log1p"] = torch.log1p(hist_edges)
    return sorted_values, tensors["histogram_counts"]


def _update_sqlite(
    *,
    db_path: Path,
    run_id: str,
    layer: str,
    feature_dir: Path,
    tensors: dict[str, torch.Tensor],
    selected_count: int,
    metadata: dict,
) -> None:
    n_features = int(tensors["feature_id"].numel())
    mini_dir = feature_dir / f"{layer}_mini_histograms"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE layers
            SET rows = ?, alive_count = ?, dead_count = ?,
                ranking_csv_path = ?, histogram_csv_path = ?, mini_histogram_svg_dir = ?,
                kurtosis_summary_json = ?, activation_frequency_summary_json = ?,
                metadata_json = ?
            WHERE run_id = ? AND layer = ?
            """,
            (
                selected_count,
                int((~tensors["dead"]).sum().item()),
                int(tensors["dead"].sum().item()),
                str(feature_dir / f"{layer}_ranking.csv"),
                str(feature_dir / f"{layer}_histograms.csv"),
                str(mini_dir),
                json.dumps(_summary(tensors["kurtosis"])),
                json.dumps(_summary(tensors["activation_frequency"])),
                json.dumps(metadata),
                run_id,
                layer,
            ),
        )
        rows = []
        for feature_id in range(n_features):
            source_sign = int(tensors["source_sign"][feature_id].item())
            rows.append(
                (
                    run_id,
                    layer,
                    feature_id,
                    int(tensors["source_feature_id"][feature_id].item()),
                    int(tensors["source_component_index"][feature_id].item()),
                    source_sign,
                    "positive" if source_sign > 0 else "negative",
                    int(tensors["dead"][feature_id].item()),
                    float(tensors["kurtosis"][feature_id].item()),
                    float(tensors["excess_kurtosis"][feature_id].item()),
                    float(tensors["activation_frequency"][feature_id].item()),
                    float(tensors["mean"][feature_id].item()),
                    float(tensors["variance"][feature_id].item()),
                    float(tensors["max"][feature_id].item()),
                    str(mini_dir / f"feature_{feature_id:06d}.svg"),
                )
            )
        conn.execute("DELETE FROM features WHERE run_id = ? AND layer = ?", (run_id, layer))
        conn.executemany(
            """
            INSERT INTO features (
                run_id, layer, feature_id, source_feature_id,
                source_component_index, source_sign, source_side, dead,
                kurtosis, excess_kurtosis, activation_frequency,
                mean, variance, max_activation, mini_histogram_svg_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()


def _summary(tensor: torch.Tensor) -> dict[str, float | int]:
    x = tensor.detach().cpu().float()
    if x.numel() == 0:
        return {"n": 0}
    return {
        "n": int(x.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "median": float(x.median().item()),
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


if __name__ == "__main__":
    main()
