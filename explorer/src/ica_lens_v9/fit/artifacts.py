from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from ..paths import V5_ROOT


def save_fastica_artifact(path: Path, artifact: dict[str, Any], metadata: dict[str, Any]) -> None:
    tensors = {key: value for key, value in artifact.items() if isinstance(value, torch.Tensor)}
    summary = {key: value for key, value in artifact.items() if not isinstance(value, torch.Tensor)}
    full_metadata = {**metadata, **summary}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": tensors, "metadata": full_metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(full_metadata, indent=2) + "\n", encoding="utf-8")


def write_metric_history_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def vendor_report() -> dict[str, Any]:
    return {
        "path": str((V5_ROOT / "vendor" / "FastICA_torch").resolve()),
        "commit": _git_commit(V5_ROOT / "vendor" / "FastICA_torch"),
        "package_version": _package_version("fastica_torch"),
    }


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
