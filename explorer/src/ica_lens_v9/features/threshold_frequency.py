"""Compatibility wrapper for feature frequency measurements."""

from __future__ import annotations

from .measurements.frequencies import compute_layer_threshold_frequency, compute_threshold_frequency_run, main

__all__ = ["compute_layer_threshold_frequency", "compute_threshold_frequency_run", "main"]


if __name__ == "__main__":
    main()
