#!/usr/bin/env python3
"""Plot pairwise activation cosine similarity heatmaps for several models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from run_vibethinker import load_model_and_tokenizer


DEFAULT_MODELS = [
    ("Qwen Coder Base", "Qwen/Qwen2.5-Coder-3B"),
    ("Qwen Coder Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ("VibeThinker", "WeiboAI/VibeThinker-3B"),
]

DEFAULT_PROMPT = (
    "Dataset: OpenEvals/IMO-AnswerBench\n"
    "Problem ID: imo-bench-algebra-026\n"
    "Category: Algebra\n"
    "Problem:\n"
    "Let n be a positive integer. Consider integer pairs (a, b) with "
    "1 <= a, b <= n, and study the residues of ab modulo n + 1. "
    "Explain the structure carefully and reason through the pattern."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-file",
        type=Path,
        help=(
            "Text file with one model per line. Formats: model_id, label=model_id, "
            "or label<TAB>model_id. Blank lines and # comments are ignored."
        ),
    )
    parser.add_argument("--model", action="append", help="Extra model id. Can be repeated.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--reference-tokenizer", default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument(
        "--layers",
        nargs="+",
        help='Layers to capture/plot, e.g. "--layers 4 32" or "--layers all". Defaults to 4 and 32 for capture, all JSON layers for --from-json.',
    )
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/visualizations/pairwise_activation_cosine.png"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Regenerate the plot from a previous JSON output without recapturing activations.",
    )
    parser.add_argument("--fig-width", type=float)
    parser.add_argument("--fig-height", type=float)
    parser.add_argument("--plot-cols", type=int, help="Number of subplot columns for multi-layer plots.")
    parser.add_argument(
        "--order",
        choices=["cluster", "input"],
        default="cluster",
        help="Order heatmap rows/columns by similarity clustering or preserve input order.",
    )
    parser.add_argument("--annot-decimals", type=int, default=1)
    return parser.parse_args()


def resolve_layers(args: argparse.Namespace, available_layers: list[int] | None = None) -> list[int]:
    if args.layers is None:
        return available_layers if available_layers is not None else [4, 32]
    if len(args.layers) == 1 and args.layers[0].lower() == "all":
        if available_layers is not None:
            return available_layers
        config = AutoConfig.from_pretrained(args.reference_tokenizer, trust_remote_code=True)
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


def prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.expanduser().read_text()
    return args.prompt


def prompt_metadata(args: argparse.Namespace, text: str) -> dict[str, str | None]:
    return {
        "source": str(args.prompt_file.expanduser()) if args.prompt_file is not None else "argument/default",
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def reference_token_ids(args: argparse.Namespace) -> tuple[list[int], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.reference_tokenizer, trust_remote_code=True)
    kwargs = {"add_special_tokens": False}
    if args.max_tokens > 0:
        kwargs.update({"truncation": True, "max_length": args.max_tokens})
    ids = [int(token_id) for token_id in tokenizer(prompt_text(args), **kwargs)["input_ids"]]
    token_texts = [
        tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
        for token_id in ids
    ]
    return ids, token_texts


def capture_layers(model, token_ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    captures: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise RuntimeError(f"Expected [1, seq, hidden], got {tuple(hidden.shape)}")
            captures[layer].append(hidden.squeeze(0).detach().cpu().to(torch.float32))

        return hook

    try:
        for layer in layers:
            handles.append(model.get_submodule(f"model.layers.{layer}").register_forward_hook(make_hook(layer)))
        input_ids = torch.tensor(token_ids, dtype=torch.long).reshape(1, -1).to(model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    out = {}
    for layer, chunks in captures.items():
        if not chunks:
            raise RuntimeError(f"No activations captured for layer {layer}.")
        out[layer] = F.normalize(torch.cat(chunks, dim=0), p=2, dim=1, eps=1e-12)
    return out


def capture_all_models(
    models: list[tuple[str, str]],
    token_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, dict[int, torch.Tensor]]:
    all_activations = {}
    for label, model_id in models:
        print(f"Capturing {label}: {model_id}", flush=True)
        load_args = argparse.Namespace(model=model_id, device=args.device, dtype=args.dtype)
        model, _tokenizer = load_model_and_tokenizer(load_args)
        all_activations[label] = capture_layers(model, token_ids, args.layers)
    return all_activations


def pairwise_mean_cosines(labels: list[str], activations: dict[str, dict[int, torch.Tensor]], layer: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in labels:
        row = []
        left_x = activations[left][layer]
        for right in labels:
            right_x = activations[right][layer]
            if left_x.shape != right_x.shape:
                row.append(float("nan"))
            else:
                row.append(float((left_x * right_x).sum(dim=1).mean().item()))
        matrix.append(row)
    return matrix


def mean_matrix_for_order(matrices: dict[int, list[list[float]]], layers: list[int]) -> "list[list[float]]":
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
    ordered_labels = [labels[index] for index in order]
    ordered_matrices = {}
    for layer, matrix in matrices.items():
        ordered_matrices[layer] = [
            [float(matrix[i][j]) for j in order]
            for i in order
        ]
    return ordered_labels, ordered_matrices


def plot_heatmaps(
    labels: list[str],
    matrices: dict[int, list[list[float]]],
    token_count: int,
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is not installed. Run: uv pip install matplotlib") from exc

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
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
        }
    )
    fig, axes = plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
    )
    axes_flat = np.array(axes, dtype=object).reshape(-1)
    image = None

    for panel_index, (axis, layer) in enumerate(zip(axes_flat, args.layers)):
        matrix = np.array(matrices[layer], dtype=float)
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        axis.set_title(f"Layer {layer}", pad=12, fontweight="semibold")
        row_index = panel_index // plot_cols
        col_index = panel_index % plot_cols
        show_x_labels = n_layers <= plot_cols or row_index == plot_rows - 1
        show_y_labels = col_index == 0
        axis.set_xticks(
            range(len(labels)),
            labels=labels if show_x_labels else [],
            rotation=42,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(range(len(labels)), labels=labels if show_y_labels else [])
        axis.tick_params(axis="both", length=0)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            if tick.get_text() == "Qwen Coder Base":
                tick.set_fontweight("bold")
                tick.set_color("#1d4ed8")
        if "Qwen Coder Base" in labels:
            base_index = labels.index("Qwen Coder Base")
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

    fig.suptitle(
        "Pairwise Activation Similarity Across Qwen-Derived Models",
        fontsize=14,
        fontweight="semibold",
        y=0.98 if n_layers > 1 else 0.96,
    )
    fig.text(
        0.5,
        0.925 if n_layers > 1 else 0.89,
        f"Mean cosine similarity over {token_count} shared tokens; rows/columns ordered by average similarity",
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
    cbar.set_label("Mean cosine", rotation=270, labelpad=14)
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
        matrices = {
            int(layer): matrix
            for layer, matrix in data["matrices"].items()
            if int(layer) in args.layers
        }
        plot_heatmaps(labels, matrices, int(data["token_count"]), args)
        return

    models = load_model_list(args)
    labels = [label for label, _model_id in models]
    args.layers = resolve_layers(args)
    prompt = prompt_text(args)
    token_ids, _token_texts = reference_token_ids(args)
    print(f"Reference tokenizer: {args.reference_tokenizer}")
    print(f"Token count: {len(token_ids)}")
    activations = capture_all_models(models, token_ids, args)
    matrices = {
        layer: pairwise_mean_cosines(labels, activations, layer)
        for layer in args.layers
    }

    json_output = args.json_output or args.output.with_suffix(".json")
    json_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    json_output.expanduser().write_text(
        json.dumps(
            {
                "models": [{"label": label, "model_id": model_id} for label, model_id in models],
                "reference_tokenizer": args.reference_tokenizer,
                "prompt": prompt_metadata(args, prompt),
                "token_count": len(token_ids),
                "layers": args.layers,
                "matrices": {str(layer): matrices[layer] for layer in args.layers},
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved data: {json_output.expanduser()}")
    plot_heatmaps(labels, matrices, len(token_ids), args)


if __name__ == "__main__":
    main()
