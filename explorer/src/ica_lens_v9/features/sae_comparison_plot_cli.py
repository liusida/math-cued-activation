#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from matplotlib.colors import to_rgba
import numpy as np
import torch

from ..paths import V9_ROOT
from .reconstruction import DEFAULT_ACTIVATION_THRESHOLDS


DEFAULT_RECONSTRUCTION_ROOT = V9_ROOT / "results" / "ica_sae_comparison" / "reconstruction"
DEFAULT_ICA_RESULTS_ROOT = DEFAULT_RECONSTRUCTION_ROOT / "ica_lens"
DEFAULT_SAE_RESULTS_ROOT = DEFAULT_RECONSTRUCTION_ROOT / "sae_counterpart"
DEFAULT_OUTPUT_ROOT = V9_ROOT / "figures" / "ica_sae_comparison" / "reconstruction"
DEFAULT_MODELS = ["gpt2", "gemma2_2b", "qwen3_5_2b_base"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full ICA and SAE reconstruction error against measured active feature count.")
    parser.add_argument("--ica-results-root", type=Path, default=DEFAULT_ICA_RESULTS_ROOT)
    parser.add_argument("--sae-results-root", type=Path, default=DEFAULT_SAE_RESULTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=("pdf", "png", "svg"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_sae_ica_reconstruction_comparison(
        ica_results_root=args.ica_results_root,
        sae_results_root=args.sae_results_root,
        output_root=args.output_root,
        models=[str(model) for model in args.models],
        formats=[str(fmt) for fmt in args.formats],
        force=bool(args.force),
    )


def plot_sae_ica_reconstruction_comparison(
    *,
    ica_results_root: Path = DEFAULT_ICA_RESULTS_ROOT,
    sae_results_root: Path = DEFAULT_SAE_RESULTS_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    models: list[str] = DEFAULT_MODELS,
    formats: list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    output_root = output_root.resolve()
    formats = list(formats or ["png"])
    mse_stem = "active_features_vs_direction_normalized_mse"
    cosine_stem = "active_features_vs_direction_mean_cosine"
    outputs = [
        *(output_root / f"{mse_stem}.{fmt}" for fmt in formats),
        *(output_root / f"{cosine_stem}.{fmt}" for fmt in formats),
    ]
    for path in outputs:
        if path.exists() and not force:
            raise FileExistsError(f"Plot already exists: {path}; pass --force.")

    summaries = [
        _load_model_summary(ica_results_root / model, sae_results_root / model)
        for model in models
        if (ica_results_root / model).is_dir() and (sae_results_root / model).is_dir()
    ]
    if not summaries:
        raise FileNotFoundError("No overlapping ICA/SAE reconstruction results found.")

    _plot_metric(
        summaries,
        ica_metric="normalized_mse",
        sae_metric="direction_normalized_mse",
        ylabel="direction normalized MSE",
        title="ICA vs SAE Reconstruction Error",
        output_paths=[output_root / f"{mse_stem}.{fmt}" for fmt in formats],
    )
    _plot_metric(
        summaries,
        ica_metric="mean_cosine",
        sae_metric="direction_mean_cosine",
        ylabel="direction mean cosine",
        title="ICA vs SAE Reconstruction Cosine",
        output_paths=[output_root / f"{cosine_stem}.{fmt}" for fmt in formats],
    )
    caption_paths = [
        output_root / f"{mse_stem}_caption.txt",
        output_root / f"{cosine_stem}_caption.txt",
    ]
    _write_caption(caption_paths[0], _mse_caption())
    _write_caption(caption_paths[1], _cosine_caption())
    all_outputs = [*outputs, *caption_paths]
    print(f"wrote {len(all_outputs)} measured active-feature comparison outputs")
    return all_outputs


def _load_model_summary(ica_model_dir: Path, sae_model_dir: Path) -> dict[str, object]:
    common_layers = sorted(
        {path.stem for path in ica_model_dir.glob("layer_*.csv")}
        & {path.stem for path in sae_model_dir.glob("layer_*.csv")},
        key=_layer_index,
    )
    if not common_layers:
        raise FileNotFoundError(f"No common layer CSVs for {ica_model_dir.name}.")

    ica_rows_by_layer = {layer: _read_rows(ica_model_dir / f"{layer}.csv") for layer in common_layers}
    sae_rows_by_layer = {layer: _read_rows(sae_model_dir / f"{layer}.csv") for layer in common_layers}

    ica_full_rows = [_row_by_key(ica_rows_by_layer[layer], "k", "all") for layer in common_layers]
    fallback_full_active = _load_ica_active_counts(ica_model_dir, common_layers, ica_full_rows)
    ica_curve = _load_ica_curve(ica_rows_by_layer, common_layers, fallback_full_active)

    sae_rows = [sae_rows_by_layer[layer][0] for layer in common_layers]
    sae_active = np.asarray([float(row["mean_l0"]) for row in sae_rows], dtype=np.float64)
    sae_normalized_mse = np.asarray([float(row["direction_normalized_mse"]) for row in sae_rows], dtype=np.float64)
    sae_mean_cosine = np.asarray([float(row["direction_mean_cosine"]) for row in sae_rows], dtype=np.float64)

    return {
        "model": ica_model_dir.name,
        "layers": common_layers,
        "ica": ica_curve,
        "sae": {
            "active_features": sae_active,
            "direction_normalized_mse": sae_normalized_mse,
            "direction_mean_cosine": sae_mean_cosine,
        },
    }


def _load_ica_curve(
    ica_rows_by_layer: dict[str, list[dict[str, str]]],
    layers: list[str],
    fallback_full_active: np.ndarray,
) -> dict[str, np.ndarray]:
    allowed_threshold_keys = {_threshold_key(threshold) for threshold in DEFAULT_ACTIVATION_THRESHOLDS}
    keys = sorted(
        {
            row["k"]
            for rows in ica_rows_by_layer.values()
            for row in rows
            if row["k"] == "all" or row["k"] in allowed_threshold_keys
        },
        key=_ica_curve_key_sort,
    )
    points: list[dict[str, float]] = []
    for key in keys:
        active_values: list[float] = []
        normalized_mse_values: list[float] = []
        mean_cosine_values: list[float] = []
        for layer_index, layer in enumerate(layers):
            try:
                row = _row_by_key(ica_rows_by_layer[layer], "k", key)
            except KeyError:
                continue
            if row.get("mean_active_features") not in (None, ""):
                active_values.append(float(row["mean_active_features"]))
            elif key == "all":
                active_values.append(float(fallback_full_active[layer_index]))
            else:
                continue
            normalized_mse_values.append(float(row["normalized_mse"]))
            mean_cosine_values.append(float(row["mean_cosine"]))
        if len(active_values) != len(layers):
            continue
        active = np.asarray(active_values, dtype=np.float64)
        normalized_mse = np.asarray(normalized_mse_values, dtype=np.float64)
        mean_cosine = np.asarray(mean_cosine_values, dtype=np.float64)
        points.append(
            {
                "label": _ica_curve_key_label(key),
                "active_features": float(active.mean()),
                "active_features_q25": float(np.quantile(active, 0.25)),
                "active_features_q75": float(np.quantile(active, 0.75)),
                "normalized_mse": float(normalized_mse.mean()),
                "normalized_mse_q25": float(np.quantile(normalized_mse, 0.25)),
                "normalized_mse_q75": float(np.quantile(normalized_mse, 0.75)),
                "mean_cosine": float(mean_cosine.mean()),
                "mean_cosine_q25": float(np.quantile(mean_cosine, 0.25)),
                "mean_cosine_q75": float(np.quantile(mean_cosine, 0.75)),
            }
        )
    points.sort(key=lambda point: point["active_features"])
    return {
        "labels": np.asarray([point["label"] for point in points], dtype=object),
        "active_features": np.asarray([point["active_features"] for point in points], dtype=np.float64),
        "active_features_q25": np.asarray([point["active_features_q25"] for point in points], dtype=np.float64),
        "active_features_q75": np.asarray([point["active_features_q75"] for point in points], dtype=np.float64),
        "normalized_mse": np.asarray([point["normalized_mse"] for point in points], dtype=np.float64),
        "normalized_mse_q25": np.asarray([point["normalized_mse_q25"] for point in points], dtype=np.float64),
        "normalized_mse_q75": np.asarray([point["normalized_mse_q75"] for point in points], dtype=np.float64),
        "mean_cosine": np.asarray([point["mean_cosine"] for point in points], dtype=np.float64),
        "mean_cosine_q25": np.asarray([point["mean_cosine_q25"] for point in points], dtype=np.float64),
        "mean_cosine_q75": np.asarray([point["mean_cosine_q75"] for point in points], dtype=np.float64),
    }


def _ica_curve_key_sort(key: str) -> float:
    if key == "all":
        return float("inf")
    if key.startswith("threshold_"):
        return float(key.removeprefix("threshold_").replace("p", "."))
    return float("inf")


def _threshold_key(threshold: float) -> str:
    text = f"{threshold:.8g}".replace(".", "p")
    return f"threshold_{text}"


def _ica_curve_key_label(key: str) -> str:
    if key == "all":
        return "full"
    if key.startswith("threshold_"):
        text = key.removeprefix("threshold_").replace("p", ".")
        if text.startswith("0."):
            text = text[1:]
        return f">{text}"
    return key


def _load_ica_active_counts(
    ica_model_dir: Path,
    layers: list[str],
    ica_full_rows: list[dict[str, str]],
) -> np.ndarray:
    row_counts: list[float] = []
    for row in ica_full_rows:
        value = row.get("mean_active_features")
        if value not in (None, ""):
            row_counts.append(float(value))
    if len(row_counts) == len(layers):
        return np.asarray(row_counts, dtype=np.float64)

    manifest_path = ica_model_dir / "manifest.json"
    if not manifest_path.is_file():
        return np.asarray([float(row["hidden_size"]) for row in ica_full_rows], dtype=np.float64)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_dir_value = manifest.get("source_feature_interface_dir")
    if not isinstance(feature_dir_value, str):
        return np.asarray([float(row["hidden_size"]) for row in ica_full_rows], dtype=np.float64)

    feature_dir = Path(feature_dir_value)
    counts: list[float] = []
    for layer, row in zip(layers, ica_full_rows):
        feature_path = feature_dir / f"{layer}_features.pt"
        if not feature_path.is_file():
            counts.append(float(row["hidden_size"]))
            continue
        artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
        activation_frequency = artifact["tensors"]["activation_frequency"].detach().cpu().to(torch.float64)
        counts.append(float(activation_frequency.sum().item()))
    return np.asarray(counts, dtype=np.float64)


def _plot_metric(
    summaries: list[dict[str, object]],
    *,
    ica_metric: str,
    sae_metric: str,
    ylabel: str,
    title: str,
    output_paths: list[Path],
) -> None:
    plt.rcParams.update(_style_rcparams(font_family="Times New Roman", font_size=8.0))
    fig, axes = plt.subplots(1, len(summaries), figsize=(6.9, 2.55), dpi=180, sharey=True)
    if len(summaries) == 1:
        axes = [axes]

    colors = {"ica": "#3B5B92", "sae": "#B45F4D"}
    markers = {"ica": "D", "sae": "o"}
    for ax, summary in zip(axes, summaries):
        model = str(summary["model"])
        ica = summary["ica"]  # type: ignore[assignment]
        sae = summary["sae"]  # type: ignore[assignment]
        _draw_ica_curve(
            ax,
            active=np.asarray(ica["active_features"], dtype=np.float64),  # type: ignore[index]
            values=np.asarray(ica[ica_metric], dtype=np.float64),  # type: ignore[index]
            value_q25=np.asarray(ica[f"{ica_metric}_q25"], dtype=np.float64),  # type: ignore[index]
            value_q75=np.asarray(ica[f"{ica_metric}_q75"], dtype=np.float64),  # type: ignore[index]
            labels=np.asarray(ica["labels"], dtype=object),  # type: ignore[index]
            color=colors["ica"],
            marker=markers["ica"],
        )
        _draw_method(
            ax,
            active=np.asarray(sae["active_features"], dtype=np.float64),  # type: ignore[index]
            values=np.asarray(sae[sae_metric], dtype=np.float64),  # type: ignore[index]
            color=colors["sae"],
            marker=markers["sae"],
            label="SAE",
        )

        layer_count = len(summary["layers"])
        layer_word = "layer" if layer_count == 1 else "layers"
        ax.set_title(f"{_display_model_name(model)} ({layer_count} {layer_word})", loc="left", fontweight="bold")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(axis="y", color="0.90", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ica_metric == "mean_cosine":
            ax.set_ylim(0.65, 1.01)
        ica_active = np.asarray(ica["active_features"], dtype=np.float64)  # type: ignore[index]
        sae_active = np.asarray(sae["active_features"], dtype=np.float64)  # type: ignore[index]
        xticks = _comparison_xticks(ica_active=ica_active, sae_active=sae_active)
        ax.set_xticks(xticks)
        ax.set_xticklabels([_format_tick(tick) for tick in xticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        x_values = np.concatenate([ica_active, sae_active])
        ax.set_xlim(float(x_values.min()) * 0.65, float(x_values.max()) * 1.35)

    handles = [
        *_ica_threshold_legend_handles(summaries=summaries, color=colors["ica"], marker=markers["ica"]),
        Line2D(
            [0],
            [0],
            color=colors["sae"],
            marker=markers["sae"],
            markerfacecolor=colors["sae"],
            markeredgecolor=colors["sae"],
            linewidth=0,
            markersize=4.5,
            label="SAE",
        ),
    ]
    labels = [handle.get_label() for handle in handles]
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=min(len(handles), 7),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        handlelength=1.6,
        columnspacing=0.85,
    )
    fig.supxlabel("measured active features", y=0.045)
    fig.suptitle(title, x=0.055, y=1.035, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.75, wspace=0.16)
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    for output_path in output_paths:
        save_kwargs = {"dpi": 180} if output_path.suffix.lower() != ".pdf" else {}
        fig.savefig(output_path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)


def _draw_ica_curve(
    ax: object,
    *,
    active: np.ndarray,
    values: np.ndarray,
    value_q25: np.ndarray,
    value_q75: np.ndarray,
    labels: np.ndarray,
    color: str,
    marker: str,
) -> None:
    if active.size == 0:
        return
    order = np.argsort(active)
    active = active[order]
    values = values[order]
    value_q25 = value_q25[order]
    value_q75 = value_q75[order]
    labels = labels[order]
    ax.fill_between(active, value_q25, value_q75, color=color, alpha=0.14, linewidth=0)
    ax.plot(active, values, color=color, linewidth=1.15, linestyle=":")
    fill_alphas = np.linspace(0.0, 1.0, int(active.size))
    for x, y, fill_alpha in zip(active, values, fill_alphas):
        facecolor = "white" if fill_alpha <= 0.0 else to_rgba(color, float(fill_alpha))
        ax.plot(
            [float(x)],
            [float(y)],
            marker=marker,
            markersize=3.6,
            linestyle="",
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=0.9,
        )


def _ica_threshold_legend_handles(*, summaries: list[dict[str, object]], color: str, marker: str) -> list[Line2D]:
    labels: list[str] = []
    for summary in summaries:
        ica = summary["ica"]  # type: ignore[assignment]
        for label in np.asarray(ica["labels"], dtype=object):  # type: ignore[index]
            text = str(label)
            if text.startswith(">") and text not in labels:
                labels.append(text)
    labels = sorted(labels, key=_threshold_label_sort, reverse=True)
    labels.append("full")
    if not labels:
        labels = ["full"]
    fill_alphas = np.linspace(0.0, 1.0, len(labels))
    handles: list[Line2D] = []
    for index, (label, fill_alpha) in enumerate(zip(labels, fill_alphas)):
        display_label = f"ICA {label}" if index == 0 else label
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                markerfacecolor="white" if fill_alpha <= 0.0 else to_rgba(color, float(fill_alpha)),
                markeredgecolor=color,
                markeredgewidth=0.9,
                linewidth=0,
                markersize=4.5,
                label=display_label,
            )
        )
    return handles


def _threshold_label_sort(label: str) -> float:
    return float(label.removeprefix(">"))


def _draw_method(
    ax: object,
    *,
    active: np.ndarray,
    values: np.ndarray,
    color: str,
    marker: str,
    label: str,
) -> None:
    mean_x = float(active.mean())
    mean_y = float(values.mean())
    x_q25, x_q75 = np.quantile(active, [0.25, 0.75])
    y_q25, y_q75 = np.quantile(values, [0.25, 0.75])
    ax.errorbar(
        [mean_x],
        [mean_y],
        yerr=[[max(mean_y - y_q25, 0.0)], [max(y_q75 - mean_y, 0.0)]],
        fmt=marker,
        markersize=4.0,
        color=color,
        ecolor=color,
        elinewidth=0.9,
        capsize=2.0,
        label=label,
    )


def _comparison_xticks(*, ica_active: np.ndarray, sae_active: np.ndarray) -> list[float]:
    candidates = [
        float(sae_active.mean()),
        float(np.min(ica_active)),
        float(np.max(ica_active)),
    ]
    candidates = sorted(value for value in candidates if value > 0)
    ticks: list[float] = []
    for value in candidates:
        if not ticks:
            ticks.append(value)
            continue
        previous = ticks[-1]
        if np.log10(value) - np.log10(previous) >= 0.08:
            ticks.append(value)
    return ticks


def _format_tick(value: float) -> str:
    if value >= 1000:
        if value % 1000 == 0:
            return f"{int(value / 1000)}k"
        return f"{value / 1000:.1f}k"
    return f"{int(value)}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_by_key(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(value)


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


def _mse_caption() -> str:
    return (
        "Direction-space reconstruction error for ICA Lens features and matched public SAE counterparts. "
        "ICA points trace reconstruction using measured active split-origin ICA Lens feature budgets, while SAE points show the matched SAE reconstruction at its measured average active-feature count. "
        "Errors are normalized by the baseline error of predicting the mean normalized activation direction."
    )


def _cosine_caption() -> str:
    return (
        "Direction-space reconstruction cosine for ICA Lens features and matched public SAE counterparts. "
        "For both methods, reconstructions are compared to row-normalized activation directions so the metric follows the v9 ICA fitting space rather than raw activation norm. "
        "ICA points vary the active-feature budget; SAE points use the native matched SAE activation pattern."
    )


def _write_caption(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
