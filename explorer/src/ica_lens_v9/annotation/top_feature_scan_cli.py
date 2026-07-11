"""Compatibility wrapper for the feature-measurement top-scan CLI."""

from __future__ import annotations

from ..features.measurements.top_scan_cli import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
