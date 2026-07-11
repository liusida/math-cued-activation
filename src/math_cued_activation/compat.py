"""Compatibility adapters for the proven pre-refactor stage implementations.

These adapters preserve legacy on-disk formats while the public interface and
configuration are centralized. They deliberately invoke source files from the
repository, not archived entrypoints.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import runpy
import sys
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _argv(script: Path, arguments: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = [str(script), *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def run_script(relative_path: str, arguments: list[str]) -> None:
    script = ROOT / relative_path
    if not script.is_file():
        raise RuntimeError(f"required compatibility implementation is missing: {script}")
    additions = [str(script.parent), str(ROOT / "explorer" / "src")]
    added = [item for item in additions if item not in sys.path]
    for item in reversed(added):
        sys.path.insert(0, item)
    try:
        with _argv(script, arguments):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        for item in added:
            sys.path.remove(item)
