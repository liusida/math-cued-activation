from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))
