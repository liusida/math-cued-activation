from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
V9_ROOT = SRC_ROOT.parent
REPO_ROOT = V9_ROOT.parent
V5_ROOT = REPO_ROOT / "v5"
V5_SRC = V5_ROOT / "src"
VENDOR_FASTICA_SRC = V5_ROOT / "vendor" / "FastICA_torch" / "src"

DEFAULT_FEATURE_INDEX = V9_ROOT / "artifacts" / "feature_index.sqlite"
