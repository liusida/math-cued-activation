from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from ..paths import V9_ROOT


DEFAULT_INPUT_ROOT = V9_ROOT / "results" / "ica_sae_comparison" / "overlap"
DEFAULT_OUTPUT_DIR = V9_ROOT / "figures" / "ica_sae_comparison" / "overlap"
DEFAULT_MODELS = ("gpt2", "gemma2_2b", "qwen3_5_2b_base")
MODEL_LABELS = {
    "gpt2": "GPT-2 Small",
    "gemma2_2b": "Gemma 2 2B",
    "qwen3_5_2b_base": "Qwen 3.5 2B Base",
}
MODEL_COLORS = {
    "gpt2": "#4C78A8",
    "gemma2_2b": "#59A14F",
    "qwen3_5_2b_base": "#B279A2",
}


def main() -> None:
    args = parse_args()
    rows_by_model = {
        model: _load_direction_rows(args.input_root.resolve(), args.basis, model)
        for model in args.models
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_stem = args.distribution_stem or f"all_models_{args.basis}_sae_nearest_cosine_distribution"
    layer_stem = args.layer_stem or f"all_models_{args.basis}_sae_nearest_cosine_by_layer"
    value_key = _default_value_key(args.basis)
    dist_paths = [output_dir / f"{dist_stem}.{fmt}" for fmt in args.formats]
    layer_paths = [output_dir / f"{layer_stem}.{fmt}" for fmt in args.formats]
    plot_distribution(
        rows_by_model=rows_by_model,
        value_key=value_key,
        paths=dist_paths,
        font_family=args.font_family,
        font_size=float(args.font_size),
        width_in=float(args.width_in),
        height_in=float(args.height_in),
        dpi=int(args.dpi),
    )
    plot_layers(
        rows_by_model=rows_by_model,
        value_key=value_key,
        paths=layer_paths,
        font_family=args.font_family,
        font_size=float(args.font_size),
        width_in=float(args.width_in),
        height_in=float(args.height_in),
        dpi=int(args.dpi),
    )
    caption_paths = [
        output_dir / f"{dist_stem}_caption.txt",
        output_dir / f"{layer_stem}_caption.txt",
    ]
    _write_caption(caption_paths[0], _distribution_caption(args.basis))
    _write_caption(caption_paths[1], _layer_caption(args.basis))
    for path in [*dist_paths, *layer_paths]:
        print(f"wrote {path}")
    for path in caption_paths:
        print(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v9 ICA/SAE nearest-cosine overlap.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basis", choices=("ica_components", "ica_lens_features"), default="ica_components")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=("pdf", "png", "svg"))
    parser.add_argument("--distribution-stem", default=None)
    parser.add_argument("--layer-stem", default=None)
    parser.add_argument("--font-family", default="Times New Roman")
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--width-in", type=float, default=6.9)
    parser.add_argument("--height-in", type=float, default=2.55)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def plot_distribution(
    *,
    rows_by_model: dict[str, list[dict[str, Any]]],
    value_key: str,
    paths: list[Path],
    font_family: str,
    font_size: float,
    width_in: float,
    height_in: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(_style_rcparams(font_family=font_family, font_size=font_size))
    fig, axes = plt.subplots(
        1,
        len(rows_by_model),
        figsize=(width_in, max(2.05, height_in * 0.82)),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)
    bins = np.linspace(0.0, 1.0, 41)
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_y = 0.0
    for model, rows in rows_by_model.items():
        values = np.asarray([row[value_key] for row in rows if row["rank"] == 1], dtype=float)
        weights = np.full_like(values, 100.0 / max(1, len(values)))
        counts, _ = np.histogram(values, bins=bins, weights=weights)
        histograms[model] = (values, counts)
        max_y = max(max_y, float(np.max(counts)))

    for ax, model in zip(axes, rows_by_model):
        values, _counts = histograms[model]
        color = MODEL_COLORS.get(model, "#555555")
        ax.hist(
            values,
            bins=bins,
            weights=np.full_like(values, 100.0 / max(1, len(values))),
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(float(np.median(values)), color="#222222", linewidth=0.8, alpha=0.85)
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="semibold", fontsize=font_size)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max_y * 1.12)
        _clean_axis(ax)
    axes[0].set_ylabel("directions (%)")
    fig.supxlabel(_x_label(value_key), y=0.045, fontsize=font_size)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.82, wspace=0.18)
    _save(fig, paths, dpi=dpi)


def plot_layers(
    *,
    rows_by_model: dict[str, list[dict[str, Any]]],
    value_key: str,
    paths: list[Path],
    font_family: str,
    font_size: float,
    width_in: float,
    height_in: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(_style_rcparams(font_family=font_family, font_size=font_size))
    fig, axes = plt.subplots(
        1,
        len(rows_by_model),
        figsize=(width_in, max(2.1, height_in * 0.88)),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)
    series: dict[str, dict[str, Any]] = {}
    global_top = 0.0
    for model, rows in rows_by_model.items():
        by_layer = _values_by_layer([row for row in rows if row["rank"] == 1], value_key=value_key)
        layers = sorted(by_layer, key=_layer_sort_key)
        xs = np.arange(len(layers), dtype=float)
        med = np.asarray([np.median(by_layer[layer]) for layer in layers], dtype=float)
        q25 = np.asarray([np.percentile(by_layer[layer], 25) for layer in layers], dtype=float)
        q75 = np.asarray([np.percentile(by_layer[layer], 75) for layer in layers], dtype=float)
        global_top = max(global_top, float(np.max(q75)))
        series[model] = {"layers": layers, "xs": xs, "med": med, "q25": q25, "q75": q75}
    y_top = min(1.0, max(0.65, global_top + 0.07))
    for ax, model in zip(axes, rows_by_model):
        data = series[model]
        color = MODEL_COLORS.get(model, "#555555")
        ax.fill_between(data["xs"], data["q25"], data["q75"], color=color, alpha=0.20, linewidth=0)
        ax.plot(data["xs"], data["med"], color=color, linewidth=1.35, marker="o", markersize=2.5)
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="semibold", fontsize=font_size)
        ax.set_ylim(0.0, y_top)
        ax.set_xlim(float(data["xs"][0]) - 0.5, float(data["xs"][-1]) + 0.5)
        tick_idx = _layer_tick_indices(len(data["layers"]))
        ax.set_xticks(data["xs"][tick_idx])
        ax.set_xticklabels([_layer_tick(str(data["layers"][int(i)])) for i in tick_idx])
        _clean_axis(ax)
    axes[0].set_ylabel(_x_label(value_key))
    fig.supxlabel("layer", y=0.045, fontsize=font_size)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.27, top=0.82, wspace=0.18)
    _save(fig, paths, dpi=dpi)


def _load_direction_rows(input_root: Path, basis: str, model: str) -> list[dict[str, Any]]:
    kind = "features" if basis == "ica_lens_features" else "components"
    path = input_root / basis / model / f"{model}_{kind}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No overlap rows found in {path}")
    return [
        {
            "model": row["model"],
            "layer": row["layer"],
            "basis": row["basis"],
            "direction_index": int(row["direction_index"]),
            "rank": int(row["rank"]),
            "nearest_sae_feature": int(row["nearest_sae_feature"]),
            "cosine": float(row["cosine"]),
            "abs_cosine": float(row["abs_cosine"]),
        }
        for row in rows
    ]


def _values_by_layer(rows: list[dict[str, Any]], *, value_key: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in rows:
        out.setdefault(str(row["layer"]), []).append(float(row[value_key]))
    return out


def _layer_sort_key(layer: str) -> tuple[int, int | str]:
    if layer.startswith("layer_"):
        return (0, int(layer.removeprefix("layer_")))
    return (1, layer)


def _layer_tick(layer: str) -> str:
    if layer.startswith("layer_"):
        return str(int(layer.removeprefix("layer_")))
    return layer


def _layer_tick_indices(n_layers: int) -> Any:
    import numpy as np

    if n_layers <= 13:
        return np.arange(n_layers)
    step = 5 if n_layers > 20 else 3
    ticks = list(range(0, n_layers, step))
    if ticks[-1] != n_layers - 1:
        ticks.append(n_layers - 1)
    return np.asarray(ticks, dtype=int)


def _clean_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.45)
    ax.set_axisbelow(True)


def _default_value_key(basis: str) -> str:
    return "cosine" if basis == "ica_lens_features" else "abs_cosine"


def _x_label(value_key: str) -> str:
    return "nearest SAE cosine" if value_key == "cosine" else "nearest absolute SAE cosine"


def _distribution_caption(basis: str) -> str:
    if basis == "ica_lens_features":
        return (
            "Nearest-SAE overlap for signed ICA Lens features across models. "
            "For each exposed positive or negative ICA Lens feature, we find the public SAE decoder direction "
            "with maximum signed cosine at the same model and layer. "
            "The distribution shows how closely each signed ICA feature side is covered by a same-direction SAE feature."
        )
    return (
        "Nearest-SAE overlap for ICA components across models. "
        "For each ICA component, we find the public SAE decoder direction with maximum absolute cosine at the same model and layer. "
        "Because ICA component signs are arbitrary, the absolute cosine treats each component as an undirected axis."
    )


def _layer_caption(basis: str) -> str:
    if basis == "ica_lens_features":
        return (
            "Layer-wise nearest-SAE overlap for signed ICA Lens features across models. "
            "For each layer, points show the median maximum signed cosine between exposed ICA Lens feature directions and public SAE decoder directions, "
            "and shaded bands show interquartile ranges."
        )
    return (
        "Layer-wise nearest-SAE overlap for ICA components across models. "
        "For each layer, points show the median maximum absolute cosine between ICA component axes and public SAE decoder directions, "
        "and shaded bands show interquartile ranges."
    )


def _write_caption(path: Path, text: str) -> None:
    path.write_text(text + "\n", encoding="utf-8")


def _save(fig: Any, paths: list[Path], *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    for path in paths:
        save_kwargs = {"dpi": dpi} if path.suffix.lower() != ".pdf" else {}
        fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)


def _style_rcparams(*, font_family: str, font_size: float) -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif"],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size * 0.88,
        "ytick.labelsize": font_size * 0.88,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


if __name__ == "__main__":
    main()
