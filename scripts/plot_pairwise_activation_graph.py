#!/usr/bin/env python3
"""Plot a NetworkX graph from pairwise activation cosine JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/visualizations/qwen_finetune_pairwise_activation_cosine.json"),
        help="JSON output from plot_pairwise_activation_cosine.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visualizations/qwen_finetune_pairwise_activation_graph.png"),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Layers to plot. Defaults to all layers in the JSON.",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.55,
        help="Drop edges below this cosine. Use 0.0 to keep all pairwise edges.",
    )
    parser.add_argument(
        "--edge-label-min",
        type=float,
        default=1.01,
        help="Only annotate edges at or above this cosine. Use 0.0 to label all shown edges.",
    )
    parser.add_argument("--fig-width", type=float, default=14.5)
    parser.add_argument("--fig-height", type=float, default=6.2)
    parser.add_argument("--plot-cols", type=int, help="Number of subplot columns for multi-layer graph plots.")
    parser.add_argument(
        "--shared-layout",
        action="store_true",
        help="Use one mean-distance layout for all layers. Default uses each layer's own cosine-distance layout.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["index", "name"],
        default="index",
        help="Use compact numbered node labels with a legend, or draw model names inside each panel.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_data(path: Path) -> dict:
    return json.loads(path.expanduser().read_text())


def mean_matrix(matrices: dict[int, list[list[float]]], layers: list[int]) -> list[list[float]]:
    n = len(next(iter(matrices.values())))
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for layer in layers:
        matrix = matrices[layer]
        for i in range(n):
            for j in range(n):
                out[i][j] += float(matrix[i][j])
    denom = float(len(layers))
    return [[value / denom for value in row] for row in out]


def label_color(label: str) -> str:
    lowered = label.lower()
    if "vibe" in lowered:
        return "#ef4444"
    if "base" in lowered:
        return "#2563eb"
    if "instruct" in lowered:
        return "#16a34a"
    return "#8b5cf6"


def short_label(label: str) -> str:
    aliases = {
        "Qwen Coder Base": "Base",
        "Qwen Coder Instruct": "Instruct",
        "VibeThinker": "VibeThinker",
        "Fasoo Spring": "Fasoo",
        "Security Qwen": "Security",
        "VeriReason RTL": "VeriReason",
        "Schema Aware": "Schema",
        "Git Commit": "Git Commit",
    }
    return aliases.get(label, label)


def expand_axis_limits(axis, pos: dict[str, tuple[float, float]], margin: float = 0.42) -> None:
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x_span = max(xs) - min(xs) or 1.0
    y_span = max(ys) - min(ys) or 1.0
    axis.set_xlim(min(xs) - margin * x_span, max(xs) + margin * x_span)
    axis.set_ylim(min(ys) - margin * y_span, max(ys) + margin * y_span)


def build_graph(labels: list[str], matrix: list[list[float]], min_edge: float):
    import networkx as nx

    graph = nx.Graph()
    for label in labels:
        graph.add_node(label)
    for i, left in enumerate(labels):
        for j in range(i + 1, len(labels)):
            value = float(matrix[i][j])
            if value >= min_edge:
                graph.add_edge(left, labels[j], weight=value)
    return graph


def cosine_distance(cosine: float) -> float:
    # Chord distance between L2-normalized vectors: ||x - y|| = sqrt(2 - 2 cos).
    return max(0.03, (max(0.0, 2.0 - 2.0 * cosine)) ** 0.5)


def distance_dict(labels: list[str], matrix: list[list[float]]) -> dict[str, dict[str, float]]:
    distances: dict[str, dict[str, float]] = {}
    for i, left in enumerate(labels):
        distances[left] = {}
        for j, right in enumerate(labels):
            distances[left][right] = 0.0 if i == j else cosine_distance(float(matrix[i][j]))
    return distances


def layout_from_matrix(labels: list[str], matrix: list[list[float]], seed: int):
    import networkx as nx

    graph = build_graph(labels, matrix, min_edge=0.0)
    distances = distance_dict(labels, matrix)
    try:
        return nx.kamada_kawai_layout(graph, dist=distances)
    except ModuleNotFoundError:
        for left, right, attrs in graph.edges(data=True):
            d = distances[left][right]
            attrs["layout_weight"] = 1.0 / max(d * d, 1e-6)
        return nx.spring_layout(graph, weight="layout_weight", seed=seed, iterations=1200)


def plot_graphs(data: dict, args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing plotting dependency. Run: uv pip install matplotlib networkx") from exc

    labels = [item["label"] for item in data["models"]]
    available_layers = [int(layer) for layer in data["layers"]]
    layers = args.layers or available_layers
    matrices = {
        int(layer): matrix
        for layer, matrix in data["matrices"].items()
        if int(layer) in layers
    }
    missing = sorted(set(layers) - set(matrices))
    if missing:
        raise SystemExit(f"Layer(s) not found in JSON: {missing}")

    shared_pos = None
    if args.shared_layout:
        shared_pos = layout_from_matrix(labels, mean_matrix(matrices, layers), args.seed)

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 12,
            "font.family": "DejaVu Sans",
        }
    )
    plot_cols = args.plot_cols or min(len(layers), 4)
    plot_rows = int(np.ceil(len(layers) / plot_cols))
    if args.fig_width == 14.5 and args.fig_height == 6.2 and len(layers) > 2:
        fig_size = (3.8 * plot_cols + 1.4, 3.7 * plot_rows + 1.5)
    else:
        fig_size = (args.fig_width, args.fig_height)
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=fig_size)
    axes_flat = np.array(axes, dtype=object).reshape(-1)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    node_colors = [label_color(label) for label in labels]
    node_label_map = {
        label: str(index + 1) if args.label_mode == "index" else short_label(label)
        for index, label in enumerate(labels)
    }

    for axis, layer in zip(axes_flat, layers):
        matrix = matrices[layer]
        graph = build_graph(labels, matrix, args.min_edge)
        pos = shared_pos or layout_from_matrix(labels, matrix, args.seed + layer)
        edge_values = [float(graph[u][v]["weight"]) for u, v in graph.edges()]
        edge_widths = [0.8 + 4.5 * max(0.0, value - args.min_edge) / max(1e-6, 1.0 - args.min_edge) for value in edge_values]
        edge_colors = [cmap(norm(value)) for value in edge_values]

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=axis,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.68,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=axis,
            node_size=620 if args.label_mode == "index" else 900,
            node_color=node_colors,
            edgecolors="white",
            linewidths=2.0,
        )
        label_pos = pos if args.label_mode == "index" else {node: (xy[0], xy[1] + 0.035) for node, xy in pos.items()}
        nx.draw_networkx_labels(
            graph,
            label_pos,
            ax=axis,
            labels={label: node_label_map[label] for label in graph.nodes()},
            font_size=8.0,
            font_weight="bold",
            font_color="white" if args.label_mode == "index" else "#111827",
            bbox=None
            if args.label_mode == "index"
            else {"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
        )
        edge_labels = {
            (u, v): f"{attrs['weight']:.3f}"
            for u, v, attrs in graph.edges(data=True)
            if float(attrs["weight"]) >= args.edge_label_min
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                ax=axis,
                font_size=7.5,
                font_color="#374151",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.78},
            )
        axis.set_title(f"Layer {layer}", fontweight="semibold")
        axis.set_axis_off()
        expand_axis_limits(axis, pos, margin=0.55)
    for axis in axes_flat[len(layers):]:
        axis.set_axis_off()

    token_count = data.get("token_count", "?")
    if args.label_mode == "index":
        legend_text = "   ".join(
            f"{index + 1}: {short_label(label)}"
            for index, label in enumerate(labels)
        )
        fig.text(
            0.5,
            0.885,
            legend_text,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#374151",
        )
    fig.suptitle(
        "Pairwise Activation Cosine Graph",
        fontsize=15,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        f"Node distances use sqrt(2 - 2 cos) from mean cosine over {token_count} shared tokens; edges below {args.min_edge:.2f} hidden",
        ha="center",
        va="center",
        fontsize=10,
        color="#4b5563",
    )

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=axes_flat.tolist(), location="right", fraction=0.035, pad=0.02)
    cbar.set_label("Mean cosine", rotation=270, labelpad=14)
    cbar.outline.set_visible(False)

    top = 0.82 if args.label_mode == "index" else 0.84
    fig.subplots_adjust(left=0.04, right=0.88, top=top, bottom=0.05, wspace=0.22, hspace=0.30)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.expanduser(), bbox_inches="tight")
    print(f"Saved graph: {args.output.expanduser()}")


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    plot_graphs(data, args)


if __name__ == "__main__":
    main()
