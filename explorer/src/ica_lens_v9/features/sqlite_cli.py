#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import V9_ROOT
from .index import DEFAULT_METHOD, build_feature_index


DEFAULT_ICA_ROOT = V9_ROOT / "artifacts" / "ica"
DEFAULT_FEATURE_INTERFACE_ROOT = V9_ROOT / "artifacts" / "feature_interfaces"
DEFAULT_OUTPUT = V9_ROOT / "artifacts" / "feature_index.sqlite"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimal SQLite index for v9 feature interfaces."
    )
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Specific ICA run directory to index. May be passed more than once.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_feature_sqlite_run(
        ica_root=args.ica_root,
        feature_interface_root=args.feature_interface_root,
        output=args.output,
        method=str(args.method),
        run_dirs=[path for path in args.run_dir],
        force=bool(args.force),
    )


def build_feature_sqlite_run(
    *,
    ica_root: Path = DEFAULT_ICA_ROOT,
    feature_interface_root: Path = DEFAULT_FEATURE_INTERFACE_ROOT,
    output: Path = DEFAULT_OUTPUT,
    method: str = DEFAULT_METHOD,
    run_dirs: list[Path] | None = None,
    force: bool = False,
) -> Path:
    output = output.resolve()
    build_feature_index(
        output=output,
        ica_root=ica_root.resolve(),
        feature_interface_root=feature_interface_root.resolve(),
        method=str(method),
        run_dirs=[path.resolve() for path in (run_dirs or [])],
        force=bool(force),
    )
    print(f"wrote SQLite feature index: {output}")
    return output


if __name__ == "__main__":
    main()
