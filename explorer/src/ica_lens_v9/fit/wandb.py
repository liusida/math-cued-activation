from __future__ import annotations

from pathlib import Path
from typing import Any


def init_wandb(
    *,
    enabled: bool | None,
    token_file: Path,
    entity: str,
    project: str,
    mode: str,
    run_name: str,
    run_dir: Path,
    config: dict[str, Any],
) -> Any | None:
    should_enable = bool(enabled)
    if not should_enable or mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested, but wandb is not installed. "
            "Install it in v9 with: uv pip install --python v9/.venv/bin/python wandb"
        ) from exc
    if not token_file.is_file() and mode == "online":
        raise FileNotFoundError(f"Missing W&B token file: {token_file}")
    if token_file.is_file() and mode == "online":
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"W&B token file is empty: {token_file}")
        wandb.login(key=token, relogin=True)
    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        dir=str(run_dir),
        config=config,
        mode=mode,
    )
    return run
