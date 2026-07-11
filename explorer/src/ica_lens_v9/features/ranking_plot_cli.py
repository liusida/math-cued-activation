from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import V9_ROOT
from .plots import plot_layer_feature_ranking


DEFAULT_FEATURE_INTERFACE_DIR = (
    V9_ROOT / "artifacts" / "feature_interfaces" / "gpt2_tok1000000_c768_iter200" / "split_origin_relu"
)
DEFAULT_FIGURE_ROOT = V9_ROOT / "figures"
DEFAULT_CATEGORY = "feature_rankings"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render only ranking plots from existing ICA Lens feature artifacts."
    )
    parser.add_argument("--feature-interface-dir", type=Path, default=DEFAULT_FEATURE_INTERFACE_DIR)
    parser.add_argument("--layers", nargs="*", default=None, help="Layers to plot. Default: all layers in manifest.")
    parser.add_argument("--figure-root", type=Path, default=DEFAULT_FIGURE_ROOT)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--output-suffix", default="ranking", help="Write layer_XX_<suffix>.png.")
    parser.add_argument("--mark-component-count", dest="mark_component_count", action="store_true", default=True)
    parser.add_argument("--no-mark-component-count", dest="mark_component_count", action="store_false")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_feature_rankings_run(
        feature_interface_dir=args.feature_interface_dir,
        layers=[str(layer) for layer in args.layers] if args.layers is not None else None,
        figure_root=args.figure_root,
        category=str(args.category),
        output_suffix=str(args.output_suffix),
        mark_component_count=bool(args.mark_component_count),
        force=bool(args.force),
    )


def plot_feature_rankings_run(
    *,
    feature_interface_dir: Path = DEFAULT_FEATURE_INTERFACE_DIR,
    layers: list[str] | None = None,
    figure_root: Path = DEFAULT_FIGURE_ROOT,
    category: str = DEFAULT_CATEGORY,
    output_suffix: str = "ranking",
    mark_component_count: bool = True,
    force: bool = False,
) -> list[Path]:
    feature_interface_dir = feature_interface_dir.resolve()
    if layers is None:
        manifest = json.loads((feature_interface_dir / "manifest.json").read_text(encoding="utf-8"))
        layers = [str(layer) for layer in manifest["layers"]]

    run_name = feature_interface_dir.parent.name
    method = feature_interface_dir.name
    output_dir = figure_root.resolve() / category / run_name / method
    output_paths: list[Path] = []
    for layer in [str(layer) for layer in layers]:
        output_path = output_dir / f"{layer}_{output_suffix}.png"
        output_paths.append(
            plot_layer_feature_ranking(
                feature_interface_dir=feature_interface_dir,
                layer=layer,
                output_path=output_path,
                mark_component_count=bool(mark_component_count),
                force=bool(force),
            )
        )
    print(f"wrote {len(output_paths)} ranking plots: {output_dir}")
    return output_paths
