#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..paths import V9_ROOT
from .plots import plot_layer_feature_interface


DEFAULT_FEATURE_INTERFACE_DIR = Path(
    V9_ROOT / "artifacts" / "feature_interfaces" / "gpt2_tok1000000_c768_iter200" / "split_origin_relu"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render plots for an existing split-origin feature interface."
    )
    parser.add_argument("--feature-interface-dir", type=Path, default=DEFAULT_FEATURE_INTERFACE_DIR)
    parser.add_argument("--layers", nargs="*", default=None, help="Layers to plot. Default: all layers in the feature-interface manifest.")
    parser.add_argument("--ranking-plot", dest="ranking_plot", action="store_true", default=True)
    parser.add_argument("--no-ranking-plot", dest="ranking_plot", action="store_false")
    parser.add_argument("--mini-histogram-svgs", dest="mini_histogram_svgs", action="store_true", default=True)
    parser.add_argument("--no-mini-histogram-svgs", dest="mini_histogram_svgs", action="store_false")
    parser.add_argument("--full-histogram-pngs", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plot_feature_interface_run(
        feature_interface_dir=args.feature_interface_dir,
        layers=[str(layer) for layer in args.layers] if args.layers is not None else None,
        ranking_plot=bool(args.ranking_plot),
        mini_histogram_svgs=bool(args.mini_histogram_svgs),
        full_histogram_pngs=bool(args.full_histogram_pngs),
        force=bool(args.force),
    )


def plot_feature_interface_run(
    *,
    feature_interface_dir: Path = DEFAULT_FEATURE_INTERFACE_DIR,
    layers: list[str] | None = None,
    ranking_plot: bool = True,
    mini_histogram_svgs: bool = True,
    full_histogram_pngs: bool = False,
    force: bool = False,
) -> Path:
    feature_interface_dir = feature_interface_dir.resolve()
    if layers is None:
        manifest = json.loads((feature_interface_dir / "manifest.json").read_text(encoding="utf-8"))
        layers = [str(layer) for layer in manifest["layers"]]
    for layer in [str(layer) for layer in layers]:
        plot_layer_feature_interface(
            feature_interface_dir=feature_interface_dir,
            layer=layer,
            ranking_plot=bool(ranking_plot),
            mini_histogram_svgs=bool(mini_histogram_svgs),
            full_histogram_pngs=bool(full_histogram_pngs),
            force=bool(force),
        )
    print(f"wrote feature interface plots: {feature_interface_dir}")
    return feature_interface_dir


if __name__ == "__main__":
    main()
