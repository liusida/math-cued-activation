from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig, load_config


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True, help="Versioned TOML pipeline configuration.")
    parser.add_argument("--force", action="store_true", help="Replace outputs owned by this stage.")
    return parser


def parse_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> tuple[argparse.Namespace, PipelineConfig]:
    args = parser.parse_args(argv)
    return args, load_config(args.config)
