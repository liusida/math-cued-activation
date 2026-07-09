#!/usr/bin/env python3
"""Plot pairwise cosine similarity of sampled corresponding model weights."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from run_vibethinker import load_model_and_tokenizer


DEFAULT_MODELS = [
    ("Qwen Coder Base", "Qwen/Qwen2.5-Coder-3B"),
    ("Qwen Coder Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("VibeThinker", "WeiboAI/VibeThinker-3B"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-file",
        type=Path,
        help="Text file with one model per line. Formats: model_id, label=model_id, or label<TAB>model_id.",
    )
    parser.add_argument("--model", action="append", help="Extra model id. Can be repeated.")
    parser.add_argument("--reference-model", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument(
        "--layers",
        nargs="+",
        help='Layers to compare, e.g. "--layers 4 32" or "--layers all". Defaults to 4 and 32 for capture, all JSON layers for --from-json.',
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of layer weight coordinates to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/visualizations/qwen_finetune_pairwise_weight_cosine.png"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--from-json", type=Path, help="Regenerate plot from a previous JSON output.")
    parser.add_argument("--fig-width", type=float)
    parser.add_argument("--fig-height", type=float)
    parser.add_argument("--plot-cols", type=int)
    parser.add_argument("--order", choices=["cluster", "input"], default="cluster")
    parser.add_argument("--annot-decimals", type=int, default=1)
    parser.add_argument("--highlight-label", default="Qwen Coder Base")
    return parser.parse_args()


def resolve_layers(args: argparse.Namespace, available_layers: list[int] | None = None) -> list[int]:
    if args.layers is None:
        return available_layers if available_layers is not None else [4, 32]
    if len(args.layers) == 1 and args.layers[0].lower() == "all":
        if available_layers is not None:
            return available_layers
        config = AutoConfig.from_pretrained(args.reference_model, trust_remote_code=True)
        return list(range(int(config.num_hidden_layers)))
    layers = []
    for raw_layer in args.layers:
        if raw_layer.lower() == "all":
            raise SystemExit('Use either "--layers all" or explicit layer numbers, not both.')
        layers.append(int(raw_layer))
    return layers


def short_label(model_id: str) -> str:
    last = model_id.split("/")[-1]
    return last.replace("Qwen2.5-Coder-", "Qwen-").replace("-3B", "")


def parse_model_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "\t" in line:
        label, model_id = line.split("\t", 1)
        return label.strip(), model_id.strip()
    if "=" in line:
        label, model_id = line.split("=", 1)
        return label.strip(), model_id.strip()
    return short_label(line), line


def load_model_list(args: argparse.Namespace) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    if args.models_file is not None:
        for line in args.models_file.expanduser().read_text().splitlines():
            parsed = parse_model_line(line)
            if parsed is not None:
                models.append(parsed)
    else:
        models.extend(DEFAULT_MODELS)
    if args.model:
        for model_id in args.model:
            models.append((short_label(model_id), model_id))

    seen = set()
    unique = []
    for label, model_id in models:
        if model_id in seen:
            continue
        seen.add(model_id)
        unique.append((label, model_id))
    if len(unique) < 2:
        raise SystemExit("Need at least two models.")
    return unique


def layer_named_parameters(model, layer: int) -> list[tuple[str, torch.nn.Parameter]]:
    module = model.get_submodule(f"model.layers.{layer}")
    return [(name, param) for name, param in module.named_parameters(recurse=True)]


def layer_param_signature(model, layer: int) -> list[tuple[str, tuple[int, ...], int]]:
    return [
        (name, tuple(param.shape), int(param.numel()))
        for name, param in layer_named_parameters(model, layer)
    ]


def sample_indices(total: int, sample_size: int, seed: int) -> torch.Tensor:
    if total <= 0:
        raise RuntimeError("Cannot sample an empty layer.")
    count = min(sample_size, total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(total, generator=generator)[:count].sort().values


def extract_sampled_layer_vector(
    model,
    layer: int,
    signature: list[tuple[str, tuple[int, ...], int]],
    indices: torch.Tensor,
) -> torch.Tensor:
    params = dict(layer_named_parameters(model, layer))
    chunks = []
    offset = 0
    cursor = 0
    for name, shape, numel in signature:
        param = params.get(name)
        if param is None:
            raise RuntimeError(f"Layer {layer} missing parameter {name!r}.")
        if tuple(param.shape) != shape:
            raise RuntimeError(f"Layer {layer} parameter {name!r} shape mismatch: {tuple(param.shape)} != {shape}")
        next_offset = offset + numel
        start = cursor
        while cursor < len(indices) and int(indices[cursor]) < next_offset:
            cursor += 1
        if cursor > start:
            local_indices = indices[start:cursor] - offset
            flat = param.detach().reshape(-1).cpu().to(torch.float32)
            chunks.append(flat[local_indices])
        offset = next_offset
    if not chunks:
        raise RuntimeError(f"No sampled values extracted for layer {layer}.")
    return F.normalize(torch.cat(chunks), p=2, dim=0, eps=1e-12)


def capture_weight_samples(
    models: list[tuple[str, str]],
    layers: list[int],
    args: argparse.Namespace,
) -> dict[str, dict[int, torch.Tensor]]:
    samples: dict[str, dict[int, torch.Tensor]] = {}
    signatures: dict[int, list[tuple[str, tuple[int, ...], int]]] = {}
    indices_by_layer: dict[int, torch.Tensor] = {}

    for model_index, (label, model_id) in enumerate(models):
        print(f"Loading weights: {label}: {model_id}", flush=True)
        load_args = argparse.Namespace(model=model_id, device=args.device, dtype=args.dtype)
        model, _tokenizer = load_model_and_tokenizer(load_args)

        if model_index == 0:
            for layer in layers:
                signature = layer_param_signature(model, layer)
                signatures[layer] = signature
                total = sum(numel for _name, _shape, numel in signature)
                indices_by_layer[layer] = sample_indices(total, args.sample_size, args.seed + layer)
                print(
                    f"Layer {layer}: sampled {len(indices_by_layer[layer])}/{total} coordinates",
                    flush=True,
                )

        samples[label] = {
            layer: extract_sampled_layer_vector(model, layer, signatures[layer], indices_by_layer[layer])
            for layer in layers
        }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return samples


def pairwise_cosines(labels: list[str], samples: dict[str, dict[int, torch.Tensor]], layer: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in labels:
        row = []
        left_x = samples[left][layer]
        for right in labels:
            right_x = samples[right][layer]
            row.append(float(torch.dot(left_x, right_x).item()))
        matrix.append(row)
    return matrix


def mean_matrix_for_order(matrices: dict[int, list[list[float]]], layers: list[int]) -> list[list[float]]:
    n = len(next(iter(matrices.values())))
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for layer in layers:
        matrix = matrices[layer]
        for i in range(n):
            for j in range(n):
                out[i][j] += float(matrix[i][j])
    denom = float(len(layers))
    return [[value / denom for value in row] for row in out]


def clustered_order(matrices: dict[int, list[list[float]]], layers: list[int]) -> list[int]:
    n = len(next(iter(matrices.values())))
    if n <= 2:
        return list(range(n))
    try:
        import numpy as np
        from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
    except ModuleNotFoundError:
        return list(range(n))

    matrix = np.array(mean_matrix_for_order(matrices, layers), dtype=float)
    distance = np.clip(1.0 - matrix, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    ordered_linkage = optimal_leaf_ordering(linkage_matrix, condensed)
    return [int(index) for index in leaves_list(ordered_linkage)]


def apply_order(
    labels: list[str],
    matrices: dict[int, list[list[float]]],
    layers: list[int],
    args: argparse.Namespace,
) -> tuple[list[str], dict[int, list[list[float]]]]:
    if args.order == "input":
        return labels, matrices
    order = clustered_order(matrices, layers)
    return [labels[index] for index in order], {
        layer: [[float(matrix[i][j]) for j in order] for i in order]
        for layer, matrix in matrices.items()
    }


def plot_heatmaps(labels: list[str], matrices: dict[int, list[list[float]]], sample_size: int, args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing plotting dependency. Run: uv pip install matplotlib") from exc

    labels, matrices = apply_order(labels, matrices, args.layers, args)
    n_layers = len(args.layers)
    plot_cols = args.plot_cols or min(n_layers, 6)
    plot_rows = math.ceil(n_layers / plot_cols)
    if n_layers == 1:
        fig_width = args.fig_width or 8.8
        fig_height = args.fig_height or 7.2
    else:
        fig_width = args.fig_width or max(7.0, 3.05 * plot_cols + 1.4)
        fig_height = args.fig_height or max(5.8, 2.85 * plot_rows + 1.8)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
        }
    )
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(fig_width, fig_height), constrained_layout=False)
    axes_flat = np.array(axes, dtype=object).reshape(-1)
    image = None
    for panel_index, (axis, layer) in enumerate(zip(axes_flat, args.layers)):
        matrix = np.array(matrices[layer], dtype=float)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Greens", aspect="equal")
        axis.set_title(f"Layer {layer}", pad=12, fontweight="semibold")
        row_index = panel_index // plot_cols
        col_index = panel_index % plot_cols
        show_x_labels = n_layers <= plot_cols or row_index == plot_rows - 1
        show_y_labels = col_index == 0
        axis.set_xticks(range(len(labels)), labels=labels if show_x_labels else [], rotation=42, ha="right", rotation_mode="anchor")
        axis.set_yticks(range(len(labels)), labels=labels if show_y_labels else [])
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)

        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            if tick.get_text() == args.highlight_label:
                tick.set_fontweight("bold")
                tick.set_color("#1d4ed8")
        if args.highlight_label in labels:
            base_index = labels.index(args.highlight_label)
            axis.axhline(base_index - 0.5, color="#1d4ed8", linewidth=2.0)
            axis.axhline(base_index + 0.5, color="#1d4ed8", linewidth=2.0)
            axis.axvline(base_index - 0.5, color="#1d4ed8", linewidth=2.0)
            axis.axvline(base_index + 0.5, color="#1d4ed8", linewidth=2.0)

        for spine in axis.spines.values():
            spine.set_visible(False)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                text_color = "white" if value < 0.45 or value > 0.72 else "#111827"
                fontweight = "semibold" if i == j else "normal"
                axis.text(
                    j,
                    i,
                    f"{value:.{args.annot_decimals}f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight=fontweight,
                )
    for axis in axes_flat[n_layers:]:
        axis.set_axis_off()

    fig.suptitle("Pairwise Weight Similarity Across Qwen-Derived Models", fontsize=14, fontweight="semibold", y=0.98 if n_layers > 1 else 0.96)
    fig.text(
        0.5,
        0.925 if n_layers > 1 else 0.89,
        f"Cosine similarity of {sample_size} shared sampled layer-weight coordinates; rows/columns ordered by average similarity",
        ha="center",
        va="center",
        fontsize=10,
        color="#4b5563",
    )
    if n_layers == 1:
        fig.subplots_adjust(left=0.23, right=0.80, top=0.76, bottom=0.23)
    else:
        bottom = 0.21 if n_layers <= plot_cols else 0.10
        fig.subplots_adjust(left=0.11, right=0.91, top=0.88, bottom=bottom, wspace=0.18, hspace=0.36)
    if image is None:
        raise RuntimeError("No layers to plot.")
    cbar = fig.colorbar(image, ax=axes_flat.tolist(), location="right", fraction=0.025, pad=0.025)
    cbar.set_label("Weight cosine", rotation=270, labelpad=14)
    cbar.outline.set_visible(False)

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.expanduser(), bbox_inches="tight")
    print(f"Saved plot: {args.output.expanduser()}")


def main() -> None:
    args = parse_args()
    if args.from_json is not None:
        data = json.loads(args.from_json.expanduser().read_text())
        labels = [item["label"] for item in data["models"]]
        available_layers = [int(layer) for layer in data["layers"]]
        args.layers = resolve_layers(args, available_layers)
        matrices = {int(layer): matrix for layer, matrix in data["matrices"].items() if int(layer) in args.layers}
        plot_heatmaps(labels, matrices, int(data["sample_size"]), args)
        return

    args.layers = resolve_layers(args)
    models = load_model_list(args)
    labels = [label for label, _model_id in models]
    samples = capture_weight_samples(models, args.layers, args)
    matrices = {layer: pairwise_cosines(labels, samples, layer) for layer in args.layers}

    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    json_output.expanduser().write_text(
        json.dumps(
            {
                "models": [{"label": label, "model_id": model_id} for label, model_id in models],
                "reference_model": args.reference_model,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "layers": args.layers,
                "matrices": {str(layer): matrices[layer] for layer in args.layers},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved data: {json_output.expanduser()}")
    plot_heatmaps(labels, matrices, args.sample_size, args)


if __name__ == "__main__":
    main()
