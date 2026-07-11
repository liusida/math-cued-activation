from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io_utils import load_toml


MISSING = object()


def load_config(path: Path) -> dict[str, Any]:
    return load_toml(path)


def get_nested(mapping: dict[str, Any], *keys: str, default: Any = MISSING) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            if default is not MISSING:
                return default
            raise ValueError(f"Missing required config value: {'.'.join(keys)}")
        value = value[key]
    return value


def optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def slug_model_id(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
