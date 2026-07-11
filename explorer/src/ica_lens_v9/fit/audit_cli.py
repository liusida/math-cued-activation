#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from .audit import DEFAULT_ICA_ROOT, DEFAULT_OUTPUT_ROOT, audit_fastica_runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v9 FastICA fit histories for suspicious runs.")
    parser.add_argument("--ica-root", type=Path, default=DEFAULT_ICA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runs", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    audit_fastica_runs(
        ica_root=args.ica_root,
        output_root=args.output_root,
        run_names=[str(run) for run in args.runs] if args.runs is not None else None,
    )


if __name__ == "__main__":
    main()
