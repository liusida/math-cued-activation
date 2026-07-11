from __future__ import annotations

import argparse
from pathlib import Path

from ...annotation.evidence import DEFAULT_SAMPLE_CACHE_ROOT
from ...paths import DEFAULT_FEATURE_INDEX, V9_ROOT
from .top_scan import TopFeatureScanConfig, build_top_feature_scan


DEFAULT_FEATURE_INTERFACE_DIR = (
    V9_ROOT / "artifacts" / "feature_interfaces" / "gpt2_tok1000000_c768_iter200" / "split_origin_relu"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-feature sample cache, counts, and practical deadness.")
    parser.add_argument("--feature-interface-dir", type=Path, default=DEFAULT_FEATURE_INTERFACE_DIR)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--sample-cache-root", type=Path, default=DEFAULT_SAMPLE_CACHE_ROOT)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--force-rebuild-sample-cache", action="store_true")
    parser.add_argument(
        "--no-update-index",
        action="store_true",
        help="Only write/read top-feature cache artifacts; do not update SQLite counts/dead flags.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = build_top_feature_scan(
        TopFeatureScanConfig(
            feature_interface_dir=args.feature_interface_dir,
            layer=args.layer,
            sample_cache_root=args.sample_cache_root,
            db_path=args.db_path,
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            device=str(args.device),
            dtype=str(args.dtype),
            force_rebuild_sample_cache=bool(args.force_rebuild_sample_cache),
            update_index=not bool(args.no_update_index),
        )
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
