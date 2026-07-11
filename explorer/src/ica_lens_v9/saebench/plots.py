from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import V9_ROOT
from .config import DEFAULT_OUTPUT_ROOT


DEFAULT_SPARSE_FIGURE_ROOT = V9_ROOT / "figures" / "ica_sae_comparison" / "saebench_sparse_probing"
DEFAULT_TPP_FIGURE_ROOT = V9_ROOT / "figures" / "ica_sae_comparison" / "saebench_tpp"
DEFAULT_FIGURE_ROOT = DEFAULT_SPARSE_FIGURE_ROOT
DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_K_VALUES = (1, 2, 5, 10, 20, 50, 100)

PANEL_SPECS = (
    ("gpt2", "layer_06"),
    ("gemma2_2b", "layer_12"),
    ("qwen3_5_2b_base", "layer_12"),
    ("gpt2", "layer_10"),
    ("gemma2_2b", "layer_20"),
    ("qwen3_5_2b_base", "layer_20"),
)
LAYER_SPECS_BY_MODEL = {
    "gpt2": ("layer_06", "layer_10"),
    "gemma2_2b": ("layer_12", "layer_20"),
    "qwen3_5_2b_base": ("layer_12", "layer_20"),
}
MODEL_LABELS = {
    "gpt2": "GPT-2 Small",
    "gemma2_2b": "Gemma 2 2B",
    "qwen3_5_2b_base": "Qwen 3.5 2B Base",
}
METHOD_LABELS = {
    "ica_lens": "ICA Lens",
    "ica_two_sign": "ICA Lens",
    "sae_baseline": "SAE",
    "pca": "PCA",
    "pca_two_sign": "PCA",
    "itda": "ITDA",
    "itda_two_sign": "ITDA",
    "random_in_ica_lens_structure": "Random ICA-feature",
    "random_in_sae_structure": "Random SAE-structure",
    "random_ica_width": "Random (ICA width)",
    "random_sae_width": "Random (SAE width)",
    "matryoshka_128": "SAE-Matryoshka-128",
    "matryoshka_512": "SAE-Matryoshka-512",
}
METHOD_COLORS = {
    "ica_lens": "#3D5F99",
    "ica_two_sign": "#3D5F99",
    "sae_baseline": "#B45F4D",
    "pca": "#5B8C6A",
    "pca_two_sign": "#5B8C6A",
    "itda": "#6F6F6F",
    "itda_two_sign": "#6F6F6F",
    "random_in_ica_lens_structure": "#A8A8A8",
    "random_in_sae_structure": "#4C4C4C",
    "random_ica_width": "#A8A8A8",
    "random_sae_width": "#4C4C4C",
    "matryoshka_128": "#8B6BBE",
    "matryoshka_512": "#D49A2A",
}
SPARSE_PROBE_MARKERS = {
    "ica_lens": "^",
    "ica_two_sign": "^",
    "pca": "^",
    "pca_two_sign": "^",
    "itda": "s",
    "itda_two_sign": "s",
    "random_in_ica_lens_structure": "x",
    "random_in_sae_structure": "X",
    "random_ica_width": "x",
    "random_sae_width": "X",
    "sae_baseline": "o",
    "matryoshka_128": "o",
    "matryoshka_512": "o",
}
CORE_METHOD_ORDER = (
    "ica_lens",
    "sae_baseline",
    "random_in_ica_lens_structure",
    "random_in_sae_structure",
    "itda",
    "itda_two_sign",
    "pca",
    "pca_two_sign",
    "random_ica_width",
    "random_sae_width",
)
GEMMA_METHOD_ORDER = (
    "ica_lens",
    "sae_baseline",
    "random_in_ica_lens_structure",
    "random_in_sae_structure",
    "itda",
    "itda_two_sign",
    "matryoshka_512",
    "matryoshka_128",
    "pca",
    "pca_two_sign",
    "random_ica_width",
    "random_sae_width",
)

METADATA_KEYS = {
    "task",
    "model_name",
    "layer",
    "method",
    "elapsed_seconds",
    "n_saebench_features",
    "feature_artifact",
    "sae_release",
    "artifact_prefix",
    "checkpoint_dir",
    "matryoshka_width",
    "n_itda_atoms",
    "random_seed",
    "random_structure",
    "activation",
    "top_k",
}
SPARSE_HINTS = ("accuracy", "acc", "f1")
TPP_HINTS = ("tpp", "total_metric", "intended", "unintended")
TPP_METRIC_LABELS = {
    "total_metric": "total score",
    "intended_diff_only": "intended",
    "unintended_diff_only": "unintended",
}
TPP_METHOD_ORDER = (
    "ica_lens",
    "sae_baseline",
    "random_in_ica_lens_structure",
    "random_in_sae_structure",
    "itda",
    "itda_two_sign",
    "pca_two_sign",
    "pca",
    "matryoshka_512",
    "matryoshka_128",
)


def _add_common_args(parser: argparse.ArgumentParser, *, figure_root: Path = DEFAULT_FIGURE_ROOT) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=figure_root)
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), choices=("png", "pdf", "svg"))
    parser.add_argument("--force", action="store_true")


def parse_sparse_probe_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v9 SAEBench sparse probing outputs.")
    _add_common_args(parser, figure_root=DEFAULT_SPARSE_FIGURE_ROOT)
    parser.add_argument("--methods", nargs="+", help="Optional methods to plot, e.g. ica_lens sae_baseline pca.")
    return parser.parse_args(argv)


def parse_tpp_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v9 SAEBench TPP outputs.")
    _add_common_args(parser, figure_root=DEFAULT_TPP_FIGURE_ROOT)
    return parser.parse_args(argv)


def sparse_probe_main(argv: list[str] | None = None) -> None:
    args = parse_sparse_probe_args(argv)
    manifest = plot_sparse_probe_comparison(
        output_root=args.output_root,
        figure_root=args.figure_root,
        formats=tuple(args.formats),
        methods=tuple(args.methods) if args.methods else None,
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def tpp_main(argv: list[str] | None = None) -> None:
    args = parse_tpp_args(argv)
    manifest = plot_tpp_comparison(
        output_root=args.output_root,
        figure_root=args.figure_root,
        formats=tuple(args.formats),
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def plot_sparse_probe_comparison(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    figure_root: Path = DEFAULT_SPARSE_FIGURE_ROOT,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    methods: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    figure_dir = figure_root
    summary_dir = output_root / "summary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, str]] = []

    sparse_rows = _load_task_rows(output_root, "sparse_probe")
    if sparse_rows:
        outputs.extend(_plot_sparse_probe(sparse_rows, figure_dir=figure_dir, formats=formats, force=force, methods=methods))

    manifest = {
        "output_root": str(output_root),
        "figure_root": str(figure_root),
        "methods": list(methods) if methods else None,
        "figures": outputs,
    }
    (summary_dir / "saebench_sparse_probe_plots.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def plot_tpp_comparison(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    figure_root: Path = DEFAULT_TPP_FIGURE_ROOT,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    force: bool = False,
) -> dict[str, Any]:
    figure_dir = figure_root
    summary_dir = output_root / "summary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, str]] = []

    tpp_rows = _load_task_rows(output_root, "tpp")
    if tpp_rows:
        outputs.extend(_plot_tpp_or_generic(tpp_rows, figure_dir=figure_dir, formats=formats, force=force))

    manifest = {
        "output_root": str(output_root),
        "figure_root": str(figure_root),
        "figures": outputs,
    }
    (summary_dir / "saebench_tpp_plots.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_task_rows(output_root: Path, task: str) -> list[dict[str, str]]:
    path = output_root / task / "summary" / f"{task}_long.csv"
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_sparse_probe(
    rows: list[dict[str, str]],
    *,
    figure_dir: Path,
    formats: tuple[str, ...],
    force: bool,
    methods: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    rows = _canonical_sparse_rows(rows)
    if methods:
        allowed = {_canonical_method_name(method) for method in methods}
        rows = [row for row in rows if row.get("method") in allowed]
    outputs: list[dict[str, str]] = []
    plot_specs: list[tuple[str, Any]] = [
        ("sparse_probe_baselines_all_layers", _plot_sparse_all_layers),
        ("sparse_probe_baselines_by_model", _plot_sparse_by_model),
    ]
    if any(
        row.get("model_name") == "gemma2_2b"
        and row.get("layer") == "layer_12"
        and row.get("method") in {"itda", "itda_two_sign", "matryoshka_128", "matryoshka_512"}
        for row in rows
    ):
        plot_specs.append(("gemma2_layer12_probe_with_matryoshka_and_itda", _plot_sparse_gemma_focus))
    for stem, draw in plot_specs:
        paths = [figure_dir / f"{stem}.{fmt}" for fmt in formats]
        caption_path = figure_dir / f"{stem}_caption.txt"
        if force or any(not path.exists() for path in paths):
            draw(rows, paths)
        _write_caption(caption_path, _sparse_caption(stem))
        outputs.extend({"task": "sparse_probe", "figure": str(path), "style": "v5"} for path in paths)
        outputs.append({"task": "sparse_probe", "caption": str(caption_path), "style": "v5"})
    return outputs


def _canonical_sparse_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        canonical = dict(row)
        canonical["method"] = _canonical_method_name(canonical.get("method", ""))
        out.append(canonical)
    return out


def _canonical_method_name(method: str) -> str:
    if method == "pca":
        return "pca_two_sign"
    return method


def _plot_sparse_all_layers(rows: list[dict[str, str]], paths: list[Path]) -> None:
    plt.rcParams.update(_style_rcparams())
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey=True, squeeze=False)
    for ax, (model, layer) in zip(axes.ravel(), PANEL_SPECS, strict=True):
        _draw_sparse_panel(
            ax,
            rows,
            model=model,
            layer=layer,
            methods=_available_methods(rows, CORE_METHOD_ORDER),
            title=f"{MODEL_LABELS.get(model, model)} {layer.replace('_', ' ')}",
        )
    handles = _legend_handles(_available_methods(rows, CORE_METHOD_ORDER))
    if handles:
        fig.legend(handles=handles, frameon=False, loc="upper center", ncols=min(4, len(handles)))
    fig.supxlabel(r"top-$k$ features used by probe", y=0.03)
    fig.supylabel("probe accuracy", x=0.01)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.13, top=0.86, wspace=0.16, hspace=0.35)
    _save_figure(fig, paths)


def _plot_sparse_by_model(rows: list[dict[str, str]], paths: list[Path]) -> None:
    plt.rcParams.update(_style_rcparams())
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55), sharey=True, squeeze=False)
    methods = _available_methods(rows, CORE_METHOD_ORDER)
    for ax, model in zip(axes.ravel(), ("gpt2", "gemma2_2b", "qwen3_5_2b_base"), strict=True):
        for method in methods:
            xs, ys = _mean_series(rows, model=model, layers=LAYER_SPECS_BY_MODEL[model], method=method)
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method],
                marker=SPARSE_PROBE_MARKERS[method],
                linewidth=1.35,
                markersize=2.4,
                label=METHOD_LABELS[method],
            )
        _style_sparse_axis(ax, title=MODEL_LABELS.get(model, model))
    handles = _legend_handles(methods)
    if handles:
        fig.legend(handles=handles, frameon=False, loc="upper center", ncols=min(4, len(handles)))
    fig.supxlabel(r"top-$k$ features used by probe", y=0.04)
    fig.supylabel("mean probe accuracy", x=0.01)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.24, top=0.76, wspace=0.16)
    _save_figure(fig, paths)


def _plot_sparse_gemma_focus(rows: list[dict[str, str]], paths: list[Path]) -> None:
    plt.rcParams.update(_style_rcparams())
    fig, ax = plt.subplots(1, 1, figsize=(4.75, 2.65))
    methods = _available_methods(rows, GEMMA_METHOD_ORDER)
    _draw_sparse_panel(
        ax,
        rows,
        model="gemma2_2b",
        layer="layer_12",
        methods=methods,
        title="Gemma 2 2B layer 12",
    )
    ax.set_ylabel("probe accuracy")
    ax.set_xlabel(r"top-$k$ features used by probe")
    if methods:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.subplots_adjust(left=0.12, right=0.66, bottom=0.20, top=0.93)
    _save_figure(fig, paths)


def _draw_sparse_panel(
    ax: Any,
    rows: list[dict[str, str]],
    *,
    model: str,
    layer: str,
    methods: tuple[str, ...],
    title: str,
) -> None:
    for method in methods:
        row = next(
            (
                candidate
                for candidate in rows
                if candidate.get("model_name") == model
                and candidate.get("layer") == layer
                and candidate.get("method") == method
            ),
            None,
        )
        if row is None:
            continue
        xs, ys = _sparse_series(row)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=METHOD_COLORS[method],
            marker=SPARSE_PROBE_MARKERS[method],
            linewidth=1.35,
            markersize=2.4,
            linestyle="-",
            label=METHOD_LABELS[method],
        )
    _style_sparse_axis(ax, title=title)


def _style_sparse_axis(ax: Any, *, title: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    ax.axvline(20, color="#c9ced6", linewidth=0.65, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks(list(DEFAULT_K_VALUES))
    ax.set_xticklabels([str(k) for k in DEFAULT_K_VALUES], rotation=45, ha="right")
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _sparse_series(row: dict[str, str]) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for k in DEFAULT_K_VALUES:
        value = _first_float(
            row.get(f"sae_sae_top_{k}_test_accuracy"),
            row.get(f"sae_top_{k}_test_accuracy"),
            row.get(f"top_{k}_test_accuracy"),
        )
        if value is not None:
            xs.append(k)
            ys.append(value)
    return xs, ys


def _mean_series(
    rows: list[dict[str, str]],
    *,
    model: str,
    layers: tuple[str, ...],
    method: str,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    selected_rows = [
        row
        for row in rows
        if row.get("model_name") == model and row.get("layer") in layers and row.get("method") == method
    ]
    for k in DEFAULT_K_VALUES:
        values = [
            value
            for row in selected_rows
            if (value := _first_float(row.get(f"sae_sae_top_{k}_test_accuracy"), row.get(f"sae_top_{k}_test_accuracy"))) is not None
        ]
        if values:
            xs.append(k)
            ys.append(sum(values) / len(values))
    return xs, ys


def _available_methods(rows: list[dict[str, str]], order: tuple[str, ...]) -> tuple[str, ...]:
    present = {row.get("method", "") for row in rows}
    return tuple(method for method in order if method in present)


def _legend_handles(methods: tuple[str, ...]) -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=SPARSE_PROBE_MARKERS[method],
            linewidth=1.35,
            markersize=2.4,
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def _plot_tpp_or_generic(
    rows: list[dict[str, str]],
    *,
    figure_dir: Path,
    formats: tuple[str, ...],
    force: bool,
) -> list[dict[str, str]]:
    series_rows = _tpp_series_rows(rows)
    if not series_rows:
        return [{"task": "tpp", "status": "no_tpp_threshold_metrics"}]
    stem = "saebench_tpp_intended_unintended"
    paths = [figure_dir / f"{stem}.{fmt}" for fmt in formats]
    caption_path = figure_dir / f"{stem}_caption.txt"
    if force or any(not path.exists() for path in paths):
        _plot_tpp_curves(series_rows, paths=paths)
    _write_caption(caption_path, _tpp_caption())
    outputs = [{"task": "tpp", "figure": str(path), "style": "v5_curves"} for path in paths]
    outputs.append({"task": "tpp", "caption": str(caption_path), "style": "v5_curves"})
    return outputs


def _tpp_series_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prefix = "tpp_metrics_tpp_threshold_"
    suffixes = tuple(TPP_METRIC_LABELS)
    for row in rows:
        method = _canonical_method_name(row.get("method", ""))
        model = row.get("model_name", "")
        layer = row.get("layer", "")
        if not method or not model or not layer:
            continue
        for key, value in row.items():
            if not key.startswith(prefix):
                continue
            parsed_value = _to_float(value)
            if parsed_value is None:
                continue
            rest = key.removeprefix(prefix)
            threshold_text, metric_kind = _split_tpp_threshold_metric(rest, suffixes=suffixes)
            if threshold_text is None or metric_kind is None:
                continue
            out.append(
                {
                    "model_name": model,
                    "layer": layer,
                    "method": method,
                    "threshold": int(threshold_text),
                    "metric_kind": metric_kind,
                    "value": parsed_value,
                }
            )
    return out


def _split_tpp_threshold_metric(rest: str, *, suffixes: tuple[str, ...]) -> tuple[str | None, str | None]:
    for suffix in suffixes:
        marker = f"_{suffix}"
        if rest.endswith(marker):
            threshold_text = rest[: -len(marker)]
            if threshold_text.isdigit():
                return threshold_text, suffix
    return None, None


def _plot_tpp_curves(rows: list[dict[str, Any]], *, paths: list[Path]) -> None:
    models = _ordered_present((row["model_name"] for row in rows), order=tuple(MODEL_LABELS))
    if not models:
        return
    plt.rcParams.update(_style_rcparams())
    fig = plt.figure(figsize=(7.0, 3.85))
    outer = fig.add_gridspec(1, len(models), wspace=0.18)

    total_max_y = _metric_upper(rows, metric_kind="total_metric", models=models)
    effect_max_y = _metric_upper(rows, metric_kind=("intended_diff_only", "unintended_diff_only"), models=models)
    total_axes: list[Any] = []
    effect_axes: list[Any] = []
    for col, model in enumerate(models):
        column = outer[0, col].subgridspec(2, 1, height_ratios=[1.0, 0.58], hspace=0.28)
        total_ax = fig.add_subplot(column[0, 0])
        bottom = column[1, 0].subgridspec(1, 2, wspace=0.20)
        intended_ax = fig.add_subplot(bottom[0, 0])
        unintended_ax = fig.add_subplot(bottom[0, 1], sharey=intended_ax)
        total_axes.append(total_ax)
        effect_axes.extend([intended_ax, unintended_ax])

        _draw_tpp_axis(
            total_ax,
            rows=rows,
            model=model,
            metric_kind="total_metric",
            title=MODEL_LABELS.get(model, model),
            compact_ticks=False,
            show_method_total=True,
        )
        _draw_tpp_axis(
            intended_ax,
            rows=rows,
            model=model,
            metric_kind="intended_diff_only",
            title="intended",
            compact_ticks=True,
        )
        _draw_tpp_axis(
            unintended_ax,
            rows=rows,
            model=model,
            metric_kind="unintended_diff_only",
            title="unintended",
            compact_ticks=True,
        )

        total_ax.set_ylim(min(-0.01, total_ax.get_ylim()[0]), total_max_y)
        intended_ax.set_ylim(min(-0.01, intended_ax.get_ylim()[0]), effect_max_y)
        unintended_ax.set_ylim(min(-0.01, unintended_ax.get_ylim()[0]), effect_max_y)
        if col == 0:
            total_ax.set_ylabel("TPP total")
            intended_ax.set_ylabel("TPP effect")
        else:
            total_ax.set_yticklabels([])
            intended_ax.set_yticklabels([])
            unintended_ax.set_yticklabels([])
        total_ax.tick_params(axis="x", labelbottom=False)
        unintended_ax.tick_params(axis="y", labelleft=False)

    handles, labels = total_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncols=min(4, len(handles)),
            handlelength=1.8,
            columnspacing=1.0,
        )
    fig.supxlabel(r"top-$N$ ablated features", y=0.035)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.82)
    _save_figure(fig, paths)


def _draw_tpp_axis(
    ax: Any,
    *,
    rows: list[dict[str, Any]],
    model: str,
    metric_kind: str,
    title: str,
    compact_ticks: bool,
    show_method_total: bool = False,
) -> None:
    model_rows = [row for row in rows if row["model_name"] == model and row["metric_kind"] == metric_kind]
    thresholds = sorted({int(row["threshold"]) for row in model_rows})
    methods = _available_tpp_methods(model_rows)
    for method in methods:
        xs, means, p25s, p75s = _tpp_aggregate_series(model_rows, method=method, thresholds=thresholds)
        if not xs:
            continue
        color = METHOD_COLORS.get(method, "#3D5F99")
        marker = SPARSE_PROBE_MARKERS.get(method, "o")
        ax.fill_between(xs, p25s, p75s, color=color, alpha=0.12, linewidth=0, zorder=2)
        label = METHOD_LABELS.get(method, method)
        if show_method_total:
            label = f"{label} total"
        ax.plot(
            xs,
            means,
            color=color,
            marker=marker,
            linewidth=1.25 if metric_kind == "total_metric" else 1.0,
            markersize=2.5 if metric_kind == "total_metric" else 2.0,
            alpha=0.94,
            label=label,
            zorder=3,
        )
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    _style_tpp_axis(ax, thresholds=thresholds, compact_ticks=compact_ticks)


def _style_tpp_axis(ax: Any, *, thresholds: list[int], compact_ticks: bool) -> None:
    ax.axhline(0.0, color="#9aa3ad", linewidth=0.55, zorder=1)
    if 20 in thresholds:
        ax.axvline(20, color="#c9ced6", linewidth=0.65, linestyle=":", zorder=1)
    ax.set_xscale("log")
    shown_thresholds = [2, 20, 500] if compact_ticks else thresholds
    shown_thresholds = [threshold for threshold in shown_thresholds if threshold in thresholds]
    ax.set_xticks(shown_thresholds)
    ax.set_xticklabels(
        [str(threshold) for threshold in shown_thresholds],
        rotation=45 if compact_ticks else 0,
        ha="right" if compact_ticks else "center",
    )
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _available_tpp_methods(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    present = {row["method"] for row in rows}
    ordered = [method for method in TPP_METHOD_ORDER if method in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _tpp_aggregate_series(
    rows: list[dict[str, Any]],
    *,
    method: str,
    thresholds: list[int],
) -> tuple[list[int], list[float], list[float], list[float]]:
    xs: list[int] = []
    means: list[float] = []
    p25s: list[float] = []
    p75s: list[float] = []
    for threshold in thresholds:
        values = [
            float(row["value"])
            for row in rows
            if row["method"] == method and int(row["threshold"]) == threshold
        ]
        if not values:
            continue
        xs.append(threshold)
        means.append(sum(values) / len(values))
        p25s.append(_percentile(values, 25.0))
        p75s.append(_percentile(values, 75.0))
    return xs, means, p25s, p75s


def _metric_upper(rows: list[dict[str, Any]], *, metric_kind: str | tuple[str, ...], models: tuple[str, ...]) -> float:
    kinds = {metric_kind} if isinstance(metric_kind, str) else set(metric_kind)
    values = [
        float(row["value"])
        for row in rows
        if row["model_name"] in models and row["metric_kind"] in kinds
    ]
    if not values:
        return 1.0
    return max(0.02, max(values) * 1.15)


def _ordered_present(values: Any, *, order: tuple[str, ...]) -> tuple[str, ...]:
    present = set(values)
    ordered = [value for value in order if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return tuple(ordered)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _choose_metric(rows: list[dict[str, str]], *, hints: tuple[str, ...]) -> str | None:
    metric_keys = _numeric_metric_keys(rows)
    if not metric_keys:
        return None
    lowered = {key: key.lower() for key in metric_keys}
    for hint in hints:
        matches = [key for key, lower in lowered.items() if hint in lower]
        if matches:
            return sorted(matches, key=lambda key: (len(key), key))[0]
    return metric_keys[0]


def _numeric_metric_keys(rows: list[dict[str, str]]) -> list[str]:
    keys = sorted({key for row in rows for key in row if key not in METADATA_KEYS})
    out: list[str] = []
    for key in keys:
        values = [row.get(key, "") for row in rows]
        if any(_to_float(value) is not None for value in values):
            out.append(key)
    return out


def _plot_metric(rows: list[dict[str, str]], *, metric: str, title: str, paths: list[Path]) -> None:
    items = []
    for row in rows:
        value = _to_float(row.get(metric, ""))
        if value is None:
            continue
        method = row.get("method", "?")
        method_label = METHOD_LABELS.get(method, method)
        label = f"{row.get('model_name', '?')} {row.get('layer', '?')} {method_label}"
        items.append((label, value, method))
    items.sort(key=lambda item: item[0])
    if not items:
        return

    plt.rcParams.update(_style_rcparams())
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    colors = [METHOD_COLORS.get(item[2], "#3D5F99") for item in items]
    width = max(4.8, min(12.0, 0.28 * len(labels)))
    fig, ax = plt.subplots(figsize=(width, 3.0))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6.8)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, paths)


def _save_figure(fig: Any, paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": 220} if path.suffix.lower() != ".pdf" else {}
        fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)


def _sparse_caption(stem: str) -> str:
    if stem == "sparse_probe_baselines_all_layers":
        return (
            "Sparse probing comparison across the two evaluated layers for each model. "
            "For each method, linear probes are trained using the top-k active features from ICA Lens features, "
            "counterpart SAEs, PCA two-sign components, or ITDA atoms. Higher accuracy indicates that the feature "
            "basis preserves more linearly accessible task information at a fixed feature budget."
        )
    if stem == "sparse_probe_baselines_by_model":
        return (
            "Mean sparse probing accuracy by model, averaged across the two evaluated layers. "
            "Curves compare ICA Lens features with counterpart SAEs, PCA two-sign components, and ITDA atoms as the "
            "number of top-k features available to the probe increases."
        )
    if stem == "gemma2_layer12_probe_with_matryoshka_and_itda":
        return (
            "Sparse probing comparison for Gemma 2 2B layer 12, including ICA Lens features, the counterpart SAE, "
            "PCA two-sign components, ITDA atoms, and available Matryoshka SAE baselines. Accuracy is reported as a "
            "function of the top-k features used by the probe."
        )
    return (
        "Sparse probing comparison of ICA Lens features and baseline feature bases. Accuracy is reported as a "
        "function of the top-k features used by the probe."
    )


def _tpp_caption() -> str:
    return (
        "SAEBench targeted probe perturbation comparison. The top row shows the total TPP score, and the bottom row "
        "decomposes the same interventions into intended and unintended effects. Curves show means across evaluated "
        "layers for each model, with shaded interquartile bands when multiple layers are available."
    )


def _write_caption(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _first_float(*values: object) -> float | None:
    for value in values:
        parsed = _to_float(value)
        if parsed is not None:
            return parsed
    return None


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _style_rcparams() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.75,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


if __name__ == "__main__":
    main()
