"""Top-feature scan measurement entry points.

The implementation is still shared with the evidence builder for now because
evidence consumes the same cache format. This module is the public owner for
the measurement job: callers should import from here rather than annotation.
"""

from __future__ import annotations

from ...annotation.evidence import TopFeatureScanConfig, build_top_feature_scan

__all__ = ["TopFeatureScanConfig", "build_top_feature_scan"]
