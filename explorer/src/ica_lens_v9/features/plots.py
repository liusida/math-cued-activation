from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..io_utils import load_json
from ..torch_utils import float_item


def plot_layer_feature_interface(
    *,
    feature_interface_dir: Path,
    layer: str,
    ranking_plot: bool,
    mini_histogram_svgs: bool,
    full_histogram_pngs: bool,
    force: bool,
) -> dict[str, Any]:
    pt_path = feature_interface_dir / f"{layer}_features.pt"
    json_path = feature_interface_dir / f"{layer}_features.json"
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing feature artifact: {pt_path}")
    artifact = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    metadata = dict(artifact["metadata"])
    model_display_name = _model_display_name_from_feature_metadata(metadata)

    ranking_plot_path = feature_interface_dir / f"{layer}_ranking.png"
    histogram_dir = feature_interface_dir / f"{layer}_histograms"
    mini_histogram_dir = feature_interface_dir / f"{layer}_mini_histograms"

    guarded_paths: list[Path] = []
    guarded_dirs: list[Path] = []
    if ranking_plot:
        guarded_paths.append(ranking_plot_path)
    if full_histogram_pngs:
        guarded_dirs.append(histogram_dir)
    if mini_histogram_svgs:
        guarded_dirs.append(mini_histogram_dir)
    if not force and (any(path.exists() for path in guarded_paths) or any(path.exists() for path in guarded_dirs)):
        raise FileExistsError(f"Plot outputs already exist for {layer}; pass --force.")
    if force:
        for directory in guarded_dirs:
            if directory.exists():
                shutil.rmtree(directory)

    if ranking_plot:
        _write_ranking_plot(
            ranking_plot_path,
            tensors=tensors,
            layer=layer,
            dead_kurtosis_threshold=float(metadata["dead_kurtosis_threshold"]),
        )
        metadata["ranking_plot"] = str(ranking_plot_path)
    if full_histogram_pngs:
        _write_histogram_pngs(
            histogram_dir,
            tensors=tensors,
            layer=layer,
            model_display_name=model_display_name,
        )
        metadata["histogram_png_dir"] = str(histogram_dir)
    if mini_histogram_svgs:
        _write_mini_histogram_svgs(mini_histogram_dir, tensors=tensors, layer=layer)
        metadata["mini_histogram_svg_dir"] = str(mini_histogram_dir)

    histogram_metadata = dict(metadata.get("histogram", {}))
    if full_histogram_pngs:
        histogram_metadata["png_style"] = (
            "token counts, log1p-spaced x positions with raw feature magnitude tick labels, "
            "gray relu(N(0,1)) reference, y-axis cap at 500 with empty broken-axis indicator"
        )
    if mini_histogram_svgs:
        histogram_metadata["mini_svg_style"] = (
            "32x32 SVG, white background, no axes/text, observed blue bins with gray relu(N(0,1)) reference overlaid"
        )
    metadata["histogram"] = histogram_metadata

    torch.save({"tensors": tensors, "metadata": metadata}, pt_path)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def plot_layer_feature_ranking(
    *,
    feature_interface_dir: Path,
    layer: str,
    output_path: Path | None = None,
    mark_component_count: bool = True,
    force: bool = False,
) -> Path:
    pt_path = feature_interface_dir / f"{layer}_features.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"Missing feature artifact: {pt_path}")
    artifact = torch.load(pt_path, map_location="cpu", weights_only=False)
    tensors = artifact["tensors"]
    metadata = artifact["metadata"]
    path = output_path or (feature_interface_dir / f"{layer}_ranking.png")
    if path.exists() and not force:
        raise FileExistsError(f"Ranking plot already exists: {path}; pass --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_ranking_plot(
        path,
        tensors=tensors,
        layer=layer,
        dead_kurtosis_threshold=float(metadata["dead_kurtosis_threshold"]),
        mark_component_count=mark_component_count,
    )
    return path


def _write_ranking_plot(
    path: Path,
    *,
    tensors: dict[str, torch.Tensor],
    layer: str,
    dead_kurtosis_threshold: float,
    mark_component_count: bool = False,
) -> None:
    plt = _pyplot()
    ranked_kurtosis = tensors["kurtosis"].detach().cpu().to(torch.float64)
    ranks = torch.arange(ranked_kurtosis.numel())
    dead_count = int((ranked_kurtosis < float(dead_kurtosis_threshold)).sum().item())

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.plot(ranks.numpy(), ranked_kurtosis.numpy(), color="#2563eb", linewidth=1.5)
    ax.axhline(
        float(dead_kurtosis_threshold),
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label=f"dead threshold = {dead_kurtosis_threshold:g}",
    )
    if dead_count:
        first_dead_rank = int((ranked_kurtosis < float(dead_kurtosis_threshold)).nonzero()[0].item())
        ax.axvspan(
            first_dead_rank,
            ranked_kurtosis.numel() - 1,
            color="#fee2e2",
            alpha=0.45,
            label=f"dead features = {dead_count}",
        )
    if mark_component_count:
        n_components = ranked_kurtosis.numel() // 2
        ax.axvline(
            n_components,
            color="#64748b",
            linestyle=":",
            linewidth=1.2,
            label=f"component count d = {n_components}",
        )
    ax.set_title(f"{layer} split-origin feature active-mirrored kurtosis ranking")
    ax.set_xlabel("feature_id (sorted by active-mirrored raw kurtosis)")
    ax.set_ylabel("active-mirrored raw kurtosis")
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_histogram_pngs(path: Path, *, tensors: dict[str, torch.Tensor], layer: str, model_display_name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    counts = tensors["histogram_counts"].detach().cpu().to(torch.float64)
    edges_log1p = tensors["histogram_bin_edges_log1p"].detach().cpu().to(torch.float64)
    edges = tensors["histogram_bin_edges"].detach().cpu().to(torch.float64)
    total_rows = max(1.0, float(counts[0].sum().item()))
    gaussian_counts = _relu_gaussian_reference_counts(edges, total_rows)
    for feature_id in tqdm(range(int(counts.shape[0])), desc=f"plot histograms {layer}", dynamic_ncols=True):
        _write_one_histogram_png(
            path / f"feature_{feature_id:06d}.png",
            counts=counts[feature_id],
            gaussian_counts=gaussian_counts,
            edges_log1p=edges_log1p,
            layer=layer,
            model_display_name=model_display_name,
            feature_id=feature_id,
            kurtosis=float_item(tensors["kurtosis"][feature_id]),
            activation_frequency=float_item(tensors["activation_frequency"][feature_id]),
            max_value=float_item(tensors["max"][feature_id]),
        )


def _write_mini_histogram_svgs(path: Path, *, tensors: dict[str, torch.Tensor], layer: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    counts = tensors["histogram_counts"].detach().cpu().to(torch.float64)
    edges_log1p = tensors["histogram_bin_edges_log1p"].detach().cpu().to(torch.float64)
    edges = tensors["histogram_bin_edges"].detach().cpu().to(torch.float64)
    total_rows = max(1.0, float(counts[0].sum().item()))
    gaussian_counts = _relu_gaussian_reference_counts(edges, total_rows)
    for feature_id in tqdm(range(int(counts.shape[0])), desc=f"plot mini histograms {layer}", dynamic_ncols=True):
        _write_one_mini_histogram_svg(
            path / f"feature_{feature_id:06d}.svg",
            counts=counts[feature_id],
            gaussian_counts=gaussian_counts,
            edges_log1p=edges_log1p,
        )


def _write_one_mini_histogram_svg(
    path: Path,
    *,
    counts: torch.Tensor,
    gaussian_counts: torch.Tensor,
    edges_log1p: torch.Tensor,
) -> None:
    size = 32.0
    pad = 1.0
    y_cap = 500.0
    x0 = float(edges_log1p[0].item())
    x1 = float(edges_log1p[-1].item())
    plot_w = size - 2.0 * pad
    plot_h = size - 2.0 * pad

    def sx(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * plot_w

    def sy(value: float) -> float:
        clipped = min(max(value, 0.0), y_cap)
        return pad + plot_h * (1.0 - clipped / y_cap)

    def rects(values: torch.Tensor, *, fill: str, opacity: float) -> list[str]:
        rows: list[str] = []
        for bin_index, value in enumerate(values.tolist()):
            left = sx(float(edges_log1p[bin_index].item()))
            right = sx(float(edges_log1p[bin_index + 1].item()))
            top = sy(float(value))
            height = pad + plot_h - top
            if height <= 0.02:
                continue
            rows.append(
                f'<rect x="{left:.3f}" y="{top:.3f}" width="{max(0.15, right - left):.3f}" height="{height:.3f}" fill="{fill}" opacity="{opacity:.2f}"/>'
            )
        return rows

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32" role="img" aria-label="miniature feature histogram" shape-rendering="crispEdges">',
        '<rect width="32" height="32" fill="white"/>',
    ]
    lines.extend(rects(counts, fill="#1f77b4", opacity=0.88))
    lines.extend(rects(gaussian_counts, fill="#e5e7eb", opacity=0.42))
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _relu_gaussian_reference_counts(edges: torch.Tensor, total_rows: float) -> torch.Tensor:
    normal_cdf = 0.5 * (1.0 + torch.erf(edges / math.sqrt(2.0)))
    return (normal_cdf[1:] - normal_cdf[:-1]) * float(total_rows)


def _write_one_histogram_png(
    path: Path,
    *,
    counts: torch.Tensor,
    gaussian_counts: torch.Tensor,
    edges_log1p: torch.Tensor,
    layer: str,
    model_display_name: str,
    feature_id: int,
    kurtosis: float,
    activation_frequency: float,
    max_value: float,
) -> None:
    plt = _pyplot()
    y_cap = 500.0
    high = max(float(counts.max().item()), float(gaussian_counts.max().item()))
    use_break = high > y_cap
    if use_break:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(5.8, 3.95),
            dpi=150,
            gridspec_kw={"height_ratios": [0.28, 3.0], "hspace": 0.045},
        )
        _draw_feature_histogram_axis(ax_bottom, counts, gaussian_counts, edges_log1p, y_cap=y_cap)
        ax_top.set_ylim(max(y_cap + 1.0, high * 0.985), high * 1.01)
        ax_bottom.set_ylim(0, y_cap)
        ax_top.spines.bottom.set_visible(False)
        ax_bottom.spines.top.set_visible(False)
        ax_top.tick_params(labeltop=False, bottom=False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax_bottom.set_yticks([0, 100, 200, 300, 400])
        ax_bottom.xaxis.tick_bottom()
        ax = ax_bottom
    else:
        fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=150)
        _draw_feature_histogram_axis(ax, counts, gaussian_counts, edges_log1p, y_cap=max(y_cap, high * 1.08))
        ax.set_ylim(0, max(y_cap, high * 1.08))

    _format_feature_histogram_axis(ax, edges_log1p)
    layer_label = layer.removeprefix("layer_")
    title_axis = ax_top if use_break else ax
    title_axis.set_title(f"{model_display_name} Layer {layer_label} Feature {feature_id}", fontsize=11, pad=6)
    ax.text(
        0.98,
        0.94,
        f"k_mirror={kurtosis:.3g}\nactive={activation_frequency * 100:.1f}%\nmax={max_value:.2g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _draw_feature_histogram_axis(
    ax: Any,
    counts: torch.Tensor,
    gaussian_counts: torch.Tensor,
    edges_log1p: torch.Tensor,
    *,
    y_cap: float,
) -> None:
    x = edges_log1p[:-1].numpy()
    widths = torch.diff(edges_log1p).numpy()
    ax.bar(
        x,
        counts.numpy(),
        width=widths,
        align="edge",
        color="#1f77b4",
        edgecolor="#0f3f66",
        linewidth=0.15,
        alpha=0.88,
    )
    ax.stairs(
        gaussian_counts.clamp(max=float(y_cap)).numpy(),
        edges_log1p.numpy(),
        fill=True,
        facecolor="#e5e7eb",
        edgecolor="#6b7280",
        alpha=0.42,
        linestyle="--",
        linewidth=1.25,
    )


def _format_feature_histogram_axis(ax: Any, edges_log1p: torch.Tensor) -> None:
    right_edge = float(edges_log1p[-1].item())
    tick_values = [0, 1, 2, 5, 10, 20, 30, 50, 100]
    tick_positions = [math.log1p(v) for v in tick_values if math.log1p(v) <= right_edge + 1e-9]
    tick_labels = [str(v) for v in tick_values if math.log1p(v) <= right_edge + 1e-9]
    ax.set_xlim(0, right_edge)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("feature magnitude (log1p-spaced)", fontsize=9)
    ax.set_ylabel("token count", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)


def _model_display_name_from_feature_metadata(metadata: dict[str, Any]) -> str:
    source_artifact = Path(str(metadata.get("source_ica_artifact", "")))
    manifest_path = source_artifact.parent / "manifest.json"
    if manifest_path.is_file():
        try:
            return _model_display_name(load_json(manifest_path))
        except Exception:
            pass
    return source_artifact.parent.name or "Model"


def _model_display_name(manifest: dict[str, Any]) -> str:
    model = manifest.get("model", {})
    short_name = str(model.get("short_name") or model.get("id") or "model")
    normalized = short_name.lower().replace("_", "-")
    if normalized in {"gpt2", "gpt2-small"} or normalized.endswith("/gpt2"):
        return "GPT-2"
    if "gemma" in normalized:
        return "Gemma"
    if "qwen" in normalized:
        return "Qwen"
    return short_name


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
