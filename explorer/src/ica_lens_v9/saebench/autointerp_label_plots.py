from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import V9_ROOT


DEFAULT_INPUT_ROOT = V9_ROOT / "results" / "auto_interp_label_scoring"
DEFAULT_FIGURE_ROOT = V9_ROOT / "figures" / "auto_interp_label_scoring"
DEFAULT_FORMATS = ("png", "pdf")
CANDIDATE_ORDER = ("neuronpedia", "auto_initial", "auto_latest")
CANDIDATE_LABELS = {
    "neuronpedia": "Neuronpedia",
    "auto_initial": "Auto initial",
    "auto_latest": "Auto latest",
}
CANDIDATE_COLORS = {
    "neuronpedia": "#B45F4D",
    "auto_initial": "#3D5F99",
    "auto_latest": "#5B8C6A",
}
MODEL_LABELS = {
    "gpt2": "GPT-2 Small",
    "gemma2_2b": "Gemma 2 2B",
    "qwen3_5_2b_base": "Qwen 3.5 2B Base",
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    figure_root = args.figure_root / args.target_kind if args.figure_root == DEFAULT_FIGURE_ROOT else args.figure_root
    if args.all_runs:
        manifest = plot_all_auto_interp_label_comparisons(
            input_root=args.input_root,
            figure_root=figure_root,
            target_kind=args.target_kind,
            legacy_layout=bool(args.legacy_layout),
            model=args.model,
            layer=args.layer,
            formats=tuple(args.formats),
            force=bool(args.force),
        )
    else:
        manifest = plot_auto_interp_label_comparison(
            input_root=args.input_root,
            figure_root=figure_root,
            target_kind=args.target_kind,
            legacy_layout=bool(args.legacy_layout),
            model=args.model,
            layer=args.layer,
            run_name=args.run_name,
            formats=tuple(args.formats),
            force=bool(args.force),
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAEBench AutoInterp label-comparison scores.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--target-kind", choices=("sae_counterpart", "ica"), default="sae_counterpart")
    parser.add_argument("--legacy-layout", action="store_true", help="Read old result roots where model/layer lived directly under input-root.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--layer", default="layer_06")
    parser.add_argument("--run-name", default=None, help="Run folder under model/layer, e.g. mi_judge_saebench_openai.")
    parser.add_argument("--all-runs", action="store_true", help="Plot all completed run folders for this model/layer.")
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=("png", "pdf", "svg"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def plot_auto_interp_label_comparison(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    target_kind: str = "sae_counterpart",
    legacy_layout: bool = False,
    model: str = "gpt2",
    layer: str = "layer_06",
    run_name: str | None = None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    force: bool = False,
) -> dict[str, Any]:
    score_path = resolve_score_path(input_root=input_root, target_kind=target_kind, model=model, layer=layer, run_name=run_name, legacy_layout=legacy_layout)
    rows = load_score_rows(score_path)
    if not rows:
        raise RuntimeError(f"No score rows found in {score_path}")
    figure_root.mkdir(parents=True, exist_ok=True)
    stem = f"{target_kind}_{model}_{layer}_{score_path.parent.name}_auto_interp_label_scoring"
    outputs = []
    outputs.extend(plot_mean_scores(rows=rows, figure_root=figure_root, stem=stem, formats=formats, force=force))
    outputs.extend(plot_feature_scores(rows=rows, figure_root=figure_root, stem=stem, formats=formats, force=force))
    caption_path = figure_root / f"{stem}_caption.txt"
    caption_path.write_text(build_caption(rows=rows, model=model, layer=layer, run_name=score_path.parent.name), encoding="utf-8")
    manifest = {
        "input_scores": str(score_path),
        "figure_root": str(figure_root),
        "target_kind": target_kind,
        "legacy_layout": legacy_layout,
        "model": model,
        "layer": layer,
        "run_name": score_path.parent.name,
        "n_rows": len(rows),
        "n_features": len({int(row["feature_id"]) for row in rows}),
        "outputs": outputs,
        "caption": str(caption_path),
        "candidate_summary": summarize_candidates(rows),
    }
    (figure_root / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def plot_all_auto_interp_label_comparisons(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    target_kind: str = "sae_counterpart",
    legacy_layout: bool = False,
    model: str = "gpt2",
    layer: str = "layer_06",
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    force: bool = False,
) -> dict[str, Any]:
    run_rows = load_all_run_rows(input_root=input_root, target_kind=target_kind, model=model, layer=layer, legacy_layout=legacy_layout)
    if not run_rows:
        raise RuntimeError(f"No completed score runs found under {comparison_base_dir(input_root=input_root, target_kind=target_kind, model=model, layer=layer, legacy_layout=legacy_layout)}")
    figure_root.mkdir(parents=True, exist_ok=True)
    stem = f"{target_kind}_{model}_{layer}_all_runs_auto_interp_label_scoring"
    outputs = []
    outputs.extend(plot_all_run_mean_scores(run_rows=run_rows, figure_root=figure_root, stem=stem, formats=formats, force=force))
    outputs.extend(plot_all_run_initial_feature_scores(run_rows=run_rows, figure_root=figure_root, stem=stem, formats=formats, force=force))
    caption_path = figure_root / f"{stem}_caption.txt"
    caption_path.write_text(build_all_runs_caption(run_rows=run_rows, model=model, layer=layer), encoding="utf-8")
    manifest = {
        "figure_root": str(figure_root),
        "target_kind": target_kind,
        "legacy_layout": legacy_layout,
        "model": model,
        "layer": layer,
        "run_names": [item["run_name"] for item in run_rows],
        "outputs": outputs,
        "caption": str(caption_path),
        "candidate_summary_by_run": {
            item["run_name"]: summarize_candidates(item["rows"])
            for item in run_rows
        },
    }
    (figure_root / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def resolve_score_path(*, input_root: Path, target_kind: str, model: str, layer: str, run_name: str | None, legacy_layout: bool) -> Path:
    base = comparison_base_dir(input_root=input_root, target_kind=target_kind, model=model, layer=layer, legacy_layout=legacy_layout)
    if run_name:
        path = base / run_name / "scores.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(base.glob("*/scores.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No scores.csv under {base}")
    return candidates[-1]


def load_all_run_rows(*, input_root: Path, target_kind: str, model: str, layer: str, legacy_layout: bool) -> list[dict[str, Any]]:
    base = comparison_base_dir(input_root=input_root, target_kind=target_kind, model=model, layer=layer, legacy_layout=legacy_layout)
    run_rows = []
    for score_path in sorted(base.glob("*/scores.csv")):
        if score_path.parent.name.startswith("_"):
            continue
        rows = load_score_rows(score_path)
        if rows:
            run_rows.append({"run_name": score_path.parent.name, "score_path": str(score_path), "rows": rows})
    return run_rows


def comparison_base_dir(*, input_root: Path, target_kind: str, model: str, layer: str, legacy_layout: bool) -> Path:
    if legacy_layout:
        return input_root / model / layer
    return input_root / target_kind / model / layer


def load_score_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["feature_id"] = int(row["feature_id"])
        row["saebench_score"] = float(row["saebench_score"]) if row.get("saebench_score") else float("nan")
    return rows


def plot_mean_scores(*, rows: list[dict[str, Any]], figure_root: Path, stem: str, formats: tuple[str, ...], force: bool) -> list[dict[str, str]]:
    summary = summarize_candidates(rows)
    baseline = null_empty_prediction_baseline(rows)
    labels = [candidate for candidate in CANDIDATE_ORDER if candidate in summary]
    means = [summary[candidate]["mean"] for candidate in labels]
    sems = [summary[candidate]["sem"] for candidate in labels]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.bar(
        [CANDIDATE_LABELS.get(label, label) for label in labels],
        means,
        yerr=sems,
        color=[CANDIDATE_COLORS.get(label, "#777777") for label in labels],
        capsize=4,
        edgecolor="#222222",
        linewidth=0.8,
    )
    set_score_ylim(ax, rows=rows, baseline=baseline)
    ax.set_ylabel("SAEBench AutoInterp score")
    ax.set_title("AutoInterp Label Comparison")
    if baseline is not None:
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(
            0.99,
            baseline + 0.01,
            f"empty-prediction baseline {baseline:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save_figure(fig, figure_root=figure_root, stem=f"{stem}_mean_scores", formats=formats, force=force)


def plot_all_run_mean_scores(*, run_rows: list[dict[str, Any]], figure_root: Path, stem: str, formats: tuple[str, ...], force: bool) -> list[dict[str, str]]:
    judge_groups = group_runs_by_judge(run_rows)
    all_rows = [row for item in run_rows for row in item["rows"]]
    baselines = [null_empty_prediction_baseline(item["rows"]) for item in run_rows]
    valid_baselines = [value for value in baselines if value is not None]
    baseline = sum(valid_baselines) / len(valid_baselines) if valid_baselines else None
    neuronpedia_baseline = shared_neuronpedia_baseline(run_rows)
    labels = [
        candidate
        for candidate in CANDIDATE_ORDER
        if candidate != "neuronpedia"
        and any(candidate in summarize_candidates(item["rows"]) for item in run_rows)
    ]
    width = min(0.24, 0.8 / max(1, len(labels)))
    n_panels = len(judge_groups)
    if n_panels == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    else:
        fig, axes = plt.subplots(1, 1, figsize=(max(7.0, min(16.0, len(run_rows) * 1.25)), 4.4))
        axes = [axes]
    if not isinstance(axes, (list, tuple)):
        axes = list(axes)

    handles = None
    legend_labels = []
    for axis_index, (ax, (judge_key, items)) in enumerate(zip(axes, judge_groups.items())):
        summaries = [(item["run_name"], summarize_candidates(item["rows"])) for item in items]
        run_labels = [short_run_label(run_name) for run_name, _summary in summaries]
        xs = list(range(len(summaries)))
        for offset_index, candidate in enumerate(labels):
            offset = (offset_index - (len(labels) - 1) / 2) * width
            means = [summary.get(candidate, {}).get("mean", float("nan")) for _run, summary in summaries]
            sems = [summary.get(candidate, {}).get("sem", 0.0) for _run, summary in summaries]
            ax.bar(
                [x + offset for x in xs],
                means,
                yerr=sems,
                width=width,
                label=CANDIDATE_LABELS.get(candidate, candidate),
                color=CANDIDATE_COLORS.get(candidate, "#777777"),
                capsize=3,
                edgecolor="#222222",
                linewidth=0.6,
            )
        set_score_ylim(
            ax,
            rows=all_rows,
            baseline=baseline,
            reference_values=[value for value in [neuronpedia_baseline] if value is not None],
        )
        if axis_index == 0:
            ax.set_ylabel("SAEBench AutoInterp score")
        ax.set_xlabel("auto-label source")
        ax.set_xticks(xs, run_labels, rotation=25, ha="right")
        ax.set_title(judge_display_label(judge_key))
        if neuronpedia_baseline is not None:
            ax.axhline(neuronpedia_baseline, color=CANDIDATE_COLORS["neuronpedia"], linestyle="--", linewidth=1.2, alpha=0.9, label="Neuronpedia")
            ax.text(
                0.01,
                neuronpedia_baseline + 0.005,
                f"Neuronpedia {neuronpedia_baseline:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=8,
                color=CANDIDATE_COLORS["neuronpedia"],
            )
        if baseline is not None:
            ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.text(
                0.99,
                baseline + 0.01,
                f"empty baseline {baseline:.3f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
    fig.suptitle("AutoInterp Label Comparison Across Judges" if n_panels == 2 else "AutoInterp Label Comparison Across Runs", y=1.02)
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=min(3, len(labels) + 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, figure_root=figure_root, stem=f"{stem}_mean_scores", formats=formats, force=force)


def plot_feature_scores(*, rows: list[dict[str, Any]], figure_root: Path, stem: str, formats: tuple[str, ...], force: bool) -> list[dict[str, str]]:
    by_feature: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_feature[int(row["feature_id"])][str(row["candidate_id"])] = float(row["saebench_score"])
    feature_ids = sorted(by_feature)
    fig, ax = plt.subplots(figsize=(max(7.0, min(14.0, len(feature_ids) * 0.18)), 4.2))
    x_positions = {feature_id: index for index, feature_id in enumerate(feature_ids)}
    for feature_id in feature_ids:
        x = x_positions[feature_id]
        values = by_feature[feature_id]
        ordered = [candidate for candidate in CANDIDATE_ORDER if candidate in values]
        if len(ordered) >= 2:
            ax.plot([x] * len(ordered), [values[c] for c in ordered], color="#CFCFCF", linewidth=0.8, zorder=1)
    for candidate in CANDIDATE_ORDER:
        xs = [x_positions[feature_id] for feature_id in feature_ids if candidate in by_feature[feature_id]]
        ys = [by_feature[feature_id][candidate] for feature_id in feature_ids if candidate in by_feature[feature_id]]
        if xs:
            ax.scatter(xs, ys, s=24, label=CANDIDATE_LABELS.get(candidate, candidate), color=CANDIDATE_COLORS.get(candidate), alpha=0.9, zorder=2)
    set_score_ylim(ax, rows=rows, baseline=null_empty_prediction_baseline(rows))
    ax.set_ylabel("SAEBench AutoInterp score")
    ax.set_xlabel("sampled SAE feature")
    tick_stride = max(1, len(feature_ids) // 20)
    tick_ids = feature_ids[::tick_stride]
    ax.set_xticks([x_positions[feature_id] for feature_id in tick_ids], [str(feature_id) for feature_id in tick_ids], rotation=45, ha="right")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=3, frameon=False)
    ax.set_title("Per-Feature Label Scores")
    fig.tight_layout()
    return save_figure(fig, figure_root=figure_root, stem=f"{stem}_per_feature_scores", formats=formats, force=force)


def plot_all_run_initial_feature_scores(*, run_rows: list[dict[str, Any]], figure_root: Path, stem: str, formats: tuple[str, ...], force: bool) -> list[dict[str, str]]:
    candidate_ids = [
        candidate
        for candidate in ("auto_initial", "auto_latest")
        if any(str(row.get("candidate_id")) == candidate for item in run_rows for row in item["rows"])
    ]
    rows_by_run: list[tuple[str, dict[str, dict[int, float]]]] = []
    feature_ids: set[int] = set()
    all_rows = []
    for item in run_rows:
        run_name = str(item["run_name"])
        values_by_candidate: dict[str, dict[int, float]] = {candidate: {} for candidate in candidate_ids}
        for row in item["rows"]:
            candidate = str(row.get("candidate_id"))
            if candidate not in values_by_candidate:
                continue
            feature_id = int(row["feature_id"])
            values_by_candidate[candidate][feature_id] = float(row["saebench_score"])
            feature_ids.add(feature_id)
            all_rows.append(row)
        if any(values_by_candidate.values()):
            rows_by_run.append((run_name, values_by_candidate))
    if not rows_by_run or not feature_ids or not candidate_ids:
        return []

    ordered_features = sorted(feature_ids)
    x_positions = {feature_id: index for index, feature_id in enumerate(ordered_features)}
    bar_series_count = max(1, len(rows_by_run) * len(candidate_ids))
    width = min(0.09, 0.84 / bar_series_count)
    fig_width = max(9.0, min(20.0, len(ordered_features) * 0.65))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    colors = provider_colors(len(rows_by_run))
    for run_index, (run_name, values_by_candidate) in enumerate(rows_by_run):
        for candidate_index, candidate in enumerate(candidate_ids):
            series_index = run_index * len(candidate_ids) + candidate_index
            offset = (series_index - (bar_series_count - 1) / 2) * width
            xs = [x_positions[feature_id] + offset for feature_id in ordered_features]
            values = values_by_candidate.get(candidate, {})
            ys = [values.get(feature_id, float("nan")) for feature_id in ordered_features]
            label = short_run_label(run_name)
            if len(candidate_ids) > 1:
                label = f"{label} {CANDIDATE_LABELS.get(candidate, candidate).replace('Auto ', '')}"
            ax.bar(
                xs,
                ys,
                width=width,
                label=label,
                color=colors[run_index],
                alpha=0.95 if candidate == "auto_initial" else 0.55,
                hatch="" if candidate == "auto_initial" else "//",
                edgecolor="#222222",
                linewidth=0.35,
            )
    baseline = null_empty_prediction_baseline(all_rows)
    set_score_ylim(ax, rows=all_rows, baseline=baseline)
    if baseline is not None:
        ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.text(
            0.99,
            baseline + 0.01,
            f"empty baseline {baseline:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
    ax.set_ylabel("SAEBench AutoInterp score")
    ax.set_xlabel("feature id")
    title = "Annotation Scores by Feature" if "auto_latest" in candidate_ids else "Initial Annotation Scores by Feature"
    ax.set_title(title)
    ax.set_xticks(
        [x_positions[feature_id] for feature_id in ordered_features],
        [str(feature_id) for feature_id in ordered_features],
        rotation=45,
        ha="right",
    )
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=min(4, bar_series_count), frameon=False)
    fig.tight_layout()
    output_stem = f"{stem}_scores_by_feature" if "auto_latest" in candidate_ids else f"{stem}_initial_scores_by_feature"
    return save_figure(fig, figure_root=figure_root, stem=output_stem, formats=formats, force=force)


def provider_colors(count: int) -> list[str]:
    base = [
        "#3D5F99",
        "#5B8C6A",
        "#B45F4D",
        "#7B5BA7",
        "#D69C3C",
        "#4B9DA6",
        "#9A6B45",
        "#777777",
    ]
    if count <= len(base):
        return base[:count]
    repeats = (count + len(base) - 1) // len(base)
    return (base * repeats)[:count]


def group_runs_by_judge(run_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in run_rows:
        run_name = str(item["run_name"])
        judge = run_name.split("_judge_", 1)[1] if "_judge_" in run_name else "unknown"
        grouped[judge].append(item)
    return dict(sorted(grouped.items(), key=lambda item: judge_sort_key(item[0])))


def judge_sort_key(judge: str) -> tuple[int, str]:
    if judge == "saebench_openai":
        return (0, judge)
    if judge == "gpt_5_5_2026_04_23":
        return (1, judge)
    return (2, judge)


def judge_display_label(judge: str) -> str:
    if judge == "saebench_openai":
        return "Judge: GPT-4o-mini (SAEBench default)"
    if judge == "gpt_5_5_2026_04_23":
        return "Judge: GPT-5.5"
    return "Judge: " + judge.replace("_", "-")


def short_run_label(run_name: str) -> str:
    label = run_name
    if "_judge_" in label:
        label = label.split("_judge_", 1)[0]
    return label.replace("_", "-")


def save_figure(fig: Any, *, figure_root: Path, stem: str, formats: tuple[str, ...], force: bool) -> list[dict[str, str]]:
    outputs = []
    for fmt in formats:
        path = figure_root / f"{stem}.{fmt}"
        if path.exists() and not force:
            outputs.append({"format": fmt, "path": str(path), "status": "exists"})
            continue
        fig.savefig(path, dpi=220 if fmt == "png" else None, bbox_inches="tight")
        outputs.append({"format": fmt, "path": str(path), "status": "written"})
    plt.close(fig)
    return outputs


def set_score_ylim(
    ax: Any,
    *,
    rows: list[dict[str, Any]],
    baseline: float | None,
    reference_values: list[float] | None = None,
) -> None:
    values = [
        float(row["saebench_score"])
        for row in rows
        if row.get("saebench_score") not in (None, "")
    ]
    if baseline is not None:
        values.append(float(baseline))
    for value in reference_values or []:
        values.append(float(value))
    if not values:
        ax.set_ylim(0, 1.02)
        return
    if baseline is not None:
        low = max(0.0, float(baseline) - 0.025)
    else:
        low = max(0.0, min(values) - 0.035)
    high = min(1.02, max(1.0, max(values) + 0.035))
    ax.set_ylim(low, high)


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(float(row["saebench_score"]))
    out = {}
    for candidate, values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        sem = (variance ** 0.5) / (len(values) ** 0.5)
        out[candidate] = {"n": len(values), "mean": mean, "sem": sem}
    return out


def shared_neuronpedia_baseline(run_rows: list[dict[str, Any]]) -> float | None:
    preferred = []
    fallback = []
    for item in run_rows:
        summary = summarize_candidates(item["rows"])
        if "neuronpedia" in summary:
            statuses = {
                str(row.get("judge_cache_status") or "")
                for row in item["rows"]
                if str(row.get("candidate_id")) == "neuronpedia"
            }
            value = float(summary["neuronpedia"]["mean"])
            if statuses & {"global_cache", "miss", "existing_run"}:
                preferred.append(value)
            else:
                fallback.append(value)
    values = preferred or fallback
    if values:
        return sum(values) / len(values)
    return None


def null_empty_prediction_baseline(rows: list[dict[str, Any]]) -> float | None:
    by_feature: dict[int, float] = {}
    for row in rows:
        feature_id = int(row["feature_id"])
        correct = parse_sequence_list(row.get("correct_sequences"))
        n_examples = int(float(row.get("n_examples") or 0))
        if n_examples <= 0:
            continue
        by_feature[feature_id] = (n_examples - len(correct)) / n_examples
    if not by_feature:
        return None
    return sum(by_feature.values()) / len(by_feature)


def parse_sequence_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def build_caption(*, rows: list[dict[str, Any]], model: str, layer: str, run_name: str) -> str:
    summary = summarize_candidates(rows)
    baseline = null_empty_prediction_baseline(rows)
    parts = [
        f"SAEBench AutoInterp label-comparison scores for {MODEL_LABELS.get(model, model)} {layer} ({run_name}).",
        f"The plot compares Neuronpedia descriptions with initial and latest auto-annotation labels on {len({int(row['feature_id']) for row in rows})} SAE features.",
        "Scores are the fraction of SAEBench held-out scoring sequences whose active/inactive status was correctly predicted from the candidate explanation.",
    ]
    if baseline is not None:
        parts.append(f"The dashed line at {baseline:.3f} is the null baseline from predicting no active sequences.")
    for candidate in CANDIDATE_ORDER:
        if candidate in summary:
            parts.append(f"{CANDIDATE_LABELS.get(candidate, candidate)} mean={summary[candidate]['mean']:.3f}.")
    return " ".join(parts) + "\n"


def build_all_runs_caption(*, run_rows: list[dict[str, Any]], model: str, layer: str) -> str:
    neuronpedia_baseline = shared_neuronpedia_baseline(run_rows)
    baselines = [null_empty_prediction_baseline(item["rows"]) for item in run_rows]
    valid_baselines = [value for value in baselines if value is not None]
    baseline = sum(valid_baselines) / len(valid_baselines) if valid_baselines else None
    parts = [
        f"SAEBench AutoInterp label-comparison scores for {MODEL_LABELS.get(model, model)} {layer} across {len(run_rows)} auto-label runs.",
        "Each run compares Neuronpedia descriptions with the initial and latest labels from that auto-annotation source.",
        "Scores are cached by judge, explanation text, benchmark settings, and scoring examples, so unchanged labels are not re-submitted to the remote judge.",
    ]
    if baseline is not None:
        parts.append(f"The dashed line at {baseline:.3f} is the null baseline from predicting no active sequences.")
    if neuronpedia_baseline is not None:
        parts.append(f"The red dashed line at {neuronpedia_baseline:.3f} is the shared Neuronpedia reference score.")
    for item in run_rows:
        summary = summarize_candidates(item["rows"])
        values = []
        for candidate in CANDIDATE_ORDER:
            if candidate in summary:
                values.append(f"{CANDIDATE_LABELS.get(candidate, candidate)}={summary[candidate]['mean']:.3f}")
        parts.append(f"{short_run_label(item['run_name'])}: {', '.join(values)}.")
    return " ".join(parts) + "\n"


if __name__ == "__main__":
    main()
