from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ..paths import V9_ROOT
from .evidence import DEFAULT_OUTPUT_ROOT, DEFAULT_SAMPLE_CACHE_ROOT, FeatureEvidenceConfig, build_feature_evidence


DEFAULT_FEATURE_INTERFACE_DIR = (
    V9_ROOT / "artifacts" / "feature_interfaces" / "gpt2_tok1000000_c768_iter200" / "split_origin_relu"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one minimal feature evidence JSON for future auto-annotation.")
    parser.add_argument("--feature-interface-dir", type=Path, default=DEFAULT_FEATURE_INTERFACE_DIR)
    parser.add_argument("--layer", default="layer_00")
    parser.add_argument("--feature-id", type=int, default=0)
    parser.add_argument("--all-features", action="store_true", help="Build evidence for every feature in the selected layer.")
    parser.add_argument("--limit", type=int, default=None, help="With --all-features, build only the first N feature ids.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-cache-root", type=Path, default=DEFAULT_SAMPLE_CACHE_ROOT)
    parser.add_argument("--db-path", type=Path, default=V9_ROOT / "artifacts" / "feature_index.sqlite")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--force-rebuild-sample-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-update-index",
        action="store_true",
        help="Only write evidence JSONs. Leave SQLite evidence-path/ERF import to the feature-index workflow.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    feature_ids = _feature_ids(args) if args.all_features else [int(args.feature_id)]
    outputs: list[Path] = []
    for index, feature_id in enumerate(tqdm(feature_ids, desc=f"evidence {args.layer}", unit="feature", dynamic_ncols=True)):
        output = build_feature_evidence(
            FeatureEvidenceConfig(
                feature_interface_dir=args.feature_interface_dir,
                layer=str(args.layer),
                feature_id=int(feature_id),
                output_root=args.output_root,
                sample_cache_root=args.sample_cache_root,
                db_path=args.db_path,
                examples=int(args.examples),
                batch_size=int(args.batch_size),
                device=str(args.device),
                dtype=str(args.dtype),
                force_rebuild_sample_cache=bool(args.force_rebuild_sample_cache and index == 0),
                force=bool(args.force),
                update_index=not bool(args.no_update_index),
            )
        )
        outputs.append(output)
        print(output)

    if args.all_features:
        print(f"built {len(outputs)} evidence files")


def _feature_ids(args: argparse.Namespace) -> list[int]:
    artifact_path = Path(args.feature_interface_dir) / f"{args.layer}_features.pt"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    n_features = int(artifact["tensors"]["feature_id"].numel())
    if args.limit is not None:
        limit = int(args.limit)
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        n_features = min(n_features, limit)
    return list(range(n_features))
