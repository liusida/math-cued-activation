#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..paths import V9_ROOT


DEFAULT_RESULTS_ROOT = V9_ROOT / "results" / "reconstruction_error"
DEFAULT_AGGREGATE_KS = ["10", "50", "100", "200", "all"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ICA feature reconstruction-error curves.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--per-layer", action="store_true", help="Also write one plot per layer CSV.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_reconstruction_error_results(
        results_root=args.results_root,
        models=[str(model) for model in args.models] if args.models is not None else None,
        per_layer=bool(args.per_layer),
        force=bool(args.force),
    )


def plot_reconstruction_error_results(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    models: list[str] | None = None,
    per_layer: bool = False,
    force: bool = False,
) -> list[Path]:
    results_root = results_root.resolve()
    model_dirs = [results_root / model for model in models] if models is not None else sorted(p for p in results_root.iterdir() if p.is_dir())
    outputs: list[Path] = []
    plotted_model_dirs: list[Path] = []
    for model_dir in model_dirs:
        csv_paths = sorted(model_dir.glob("layer_*.csv"))
        if not csv_paths:
            continue
        plotted_model_dirs.append(model_dir)
        if not per_layer:
            continue
        for csv_path in csv_paths:
            rows = _read_rows(csv_path)
            output_path = csv_path.with_suffix(".png")
            if output_path.exists() and not force:
                raise FileExistsError(f"Plot already exists: {output_path}; pass --force.")
            _plot_layer(rows, title=f"{model_dir.name} {csv_path.stem}", output_path=output_path)
            outputs.append(output_path)
    outputs.extend(_plot_aggregate_results(plotted_model_dirs, results_root=results_root, force=force))
    print(f"wrote {len(outputs)} reconstruction-error plots")
    return outputs


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _plot_layer(rows: list[dict[str, str]], *, title: str, output_path: Path) -> None:
    all_row = next(row for row in rows if row["k"] == "all")
    k_rows = [row for row in rows if row["k"] != "all"]
    xs = [int(row["k"]) for row in k_rows]
    plot_xs = list(range(len(xs)))
    full_x = len(xs) + 0.75
    normalized_mse = [float(row["normalized_mse"]) for row in k_rows]
    cosine = [float(row["mean_cosine"]) for row in k_rows]
    all_normalized_mse = float(all_row["normalized_mse"])
    all_cosine = float(all_row["mean_cosine"])

    plt.rcParams.update(_style_rcparams(font_family="Times New Roman", font_size=8.0))
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), dpi=180)
    ax_mse, ax_cos = axes

    ica_blue = "#3B5B92"
    contrast = "#B45F4D"
    reference = "0.45"

    ax_mse.plot(plot_xs, normalized_mse, marker="o", markersize=2.8, color=ica_blue, linewidth=1.15)
    ax_mse.plot(full_x, all_normalized_mse, marker="D", markersize=3.2, color=ica_blue)
    ax_mse.axhline(all_normalized_mse, color=reference, linestyle=":", linewidth=0.65)
    ax_mse.set_title("(a) Reconstruction error", loc="left", fontweight="semibold")
    ax_mse.set_xlabel("active features")
    ax_mse.set_ylabel("normalized MSE")

    ax_cos.plot(plot_xs, cosine, marker="o", markersize=2.8, color=contrast, linewidth=1.15)
    ax_cos.plot(full_x, all_cosine, marker="D", markersize=3.2, color=contrast)
    ax_cos.axhline(all_cosine, color=reference, linestyle=":", linewidth=0.65)
    ax_cos.set_title("(b) Reconstruction cosine", loc="left", fontweight="semibold")
    ax_cos.set_xlabel("active features")
    ax_cos.set_ylabel("mean cosine")
    ax_cos.set_ylim(0.65, 1.01)

    for ax in axes:
        ax.set_xticks([*plot_xs, full_x])
        ax.set_xticklabels([*(str(x) for x in xs), "full"])
        ax.set_xlim(-0.35, full_x + 0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.90", linewidth=0.45)
        ax.set_axisbelow(True)

    fig.suptitle(title, x=0.055, y=0.995, ha="left", fontweight="semibold")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.23, top=0.78, wspace=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_aggregate_results(model_dirs: list[Path], *, results_root: Path, force: bool) -> list[Path]:
    if not model_dirs:
        return []
    outputs = [
        results_root / "aggregate_normalized_mse.png",
        results_root / "aggregate_mean_cosine.png",
        results_root / "overall_normalized_mse.png",
        results_root / "overall_mean_cosine.png",
    ]
    for path in outputs:
        if path.exists() and not force:
            raise FileExistsError(f"Plot already exists: {path}; pass --force.")
    _plot_aggregate_metric(
        model_dirs,
        metric="normalized_mse",
        ylabel="normalized MSE",
        title="Reconstruction Error Across Layers",
        output_path=outputs[0],
    )
    _plot_aggregate_metric(
        model_dirs,
        metric="mean_cosine",
        ylabel="mean cosine",
        title="Reconstruction Cosine Across Layers",
        output_path=outputs[1],
    )
    _plot_overall_metric(
        model_dirs,
        metric="normalized_mse",
        ylabel="normalized MSE",
        title="Mean Reconstruction Error Across Layers",
        output_path=outputs[2],
    )
    _plot_overall_metric(
        model_dirs,
        metric="mean_cosine",
        ylabel="mean cosine",
        title="Mean Reconstruction Cosine Across Layers",
        output_path=outputs[3],
    )
    return outputs


def _plot_aggregate_metric(model_dirs: list[Path], *, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    plt.rcParams.update(_style_rcparams(font_family="Times New Roman", font_size=8.0))
    fig, axes = plt.subplots(len(model_dirs), 1, figsize=(6.9, 1.75 * len(model_dirs)), dpi=180, sharex=False)
    if len(model_dirs) == 1:
        axes = [axes]

    colors = {
        "10": "#8DA0CB",
        "50": "#5B7DB2",
        "100": "#3B5B92",
        "200": "#23395D",
        "all": "#B45F4D",
    }
    markers = {"all": "D"}
    for ax, model_dir in zip(axes, model_dirs):
        layer_rows = _read_model_layer_rows(model_dir)
        layer_indices = [item[0] for item in layer_rows]
        by_k: dict[str, list[float]] = {k: [] for k in DEFAULT_AGGREGATE_KS}
        available_ks = {row["k"] for _, rows in layer_rows for row in rows}
        selected_ks = [k for k in DEFAULT_AGGREGATE_KS if k in available_ks]
        for _, rows in layer_rows:
            row_by_k = {row["k"]: row for row in rows}
            for k in selected_ks:
                by_k[k].append(float(row_by_k[k][metric]))

        for k in selected_ks:
            label = "full" if k == "all" else f"k={k}"
            ax.plot(
                layer_indices,
                by_k[k],
                marker=markers.get(k, "o"),
                markersize=2.8,
                linewidth=1.05,
                color=colors.get(k, "0.35"),
                label=label,
            )
        ax.set_title(_display_model_name(model_dir.name), loc="left", fontweight="semibold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.90", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([str(i) for i in layer_indices])
        if metric == "mean_cosine":
            ax.set_ylim(0.65, 1.01)
        if ax is axes[0]:
            ax.legend(frameon=False, ncol=min(len(selected_ks), 5), loc="best")
    axes[-1].set_xlabel("layer index")
    fig.suptitle(title, x=0.055, y=0.995, ha="left", fontweight="semibold")
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.09, top=0.92, hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_overall_metric(model_dirs: list[Path], *, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    plt.rcParams.update(_style_rcparams(font_family="Times New Roman", font_size=8.0))
    fig, ax = plt.subplots(1, 1, figsize=(4.65, 2.75), dpi=180)
    colors = {
        "gpt2": "#3B5B92",
        "gemma2_2b": "#6D8F63",
        "qwen3_5_2b_base": "#B45F4D",
    }
    labels = [("full" if k == "all" else k) for k in DEFAULT_AGGREGATE_KS]
    xs = np.arange(len(DEFAULT_AGGREGATE_KS), dtype=np.float64)

    for model_dir in model_dirs:
        layer_rows = _read_model_layer_rows(model_dir)
        rows_by_layer = [{row["k"]: row for row in rows} for _, rows in layer_rows]
        selected_ks = [k for k in DEFAULT_AGGREGATE_KS if all(k in row_by_k for row_by_k in rows_by_layer)]
        selected_indices = [DEFAULT_AGGREGATE_KS.index(k) for k in selected_ks]
        selected_xs = xs[selected_indices]
        values = np.asarray(
            [[float(row_by_k[k][metric]) for k in selected_ks] for row_by_k in rows_by_layer],
            dtype=np.float64,
        )
        mean = values.mean(axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        color = colors.get(model_dir.name, "0.35")
        ax.fill_between(selected_xs, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            selected_xs,
            mean,
            marker="o",
            markersize=3.0,
            linewidth=1.25,
            color=color,
            label=_display_model_name(model_dir.name),
        )

    ax.set_title(title, loc="left", fontweight="semibold")
    ax.set_xlabel("active features")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    if metric == "mean_cosine":
        ax.set_ylim(0.65, 1.01)
    ax.grid(axis="y", color="0.90", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.14, right=0.995, bottom=0.18, top=0.88)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _read_model_layer_rows(model_dir: Path) -> list[tuple[int, list[dict[str, str]]]]:
    out: list[tuple[int, list[dict[str, str]]]] = []
    for csv_path in sorted(model_dir.glob("layer_*.csv")):
        out.append((_layer_index(csv_path.stem), _read_rows(csv_path)))
    return sorted(out, key=lambda item: item[0])


def _layer_index(layer_name: str) -> int:
    match = re.search(r"(\d+)$", layer_name)
    if match is None:
        return 0
    return int(match.group(1))


def _display_model_name(model_dir_name: str) -> str:
    labels = {
        "gpt2": "GPT-2",
        "gemma2_2b": "Gemma 2 2B",
        "qwen3_5_2b_base": "Qwen3.5 2B Base",
    }
    return labels.get(model_dir_name, model_dir_name)


def _style_rcparams(*, font_family: str, font_size: float) -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size * 0.88,
        "ytick.labelsize": font_size * 0.88,
        "legend.fontsize": font_size * 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }


if __name__ == "__main__":
    main()
