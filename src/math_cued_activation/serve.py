from __future__ import annotations

import os
from pathlib import Path
import sys

from .config import PipelineConfig


def serve(config: PipelineConfig, *, reload: bool = False) -> None:
    root = Path(__file__).resolve().parents[2]
    explorer_src = root / "explorer" / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(explorer_src) not in sys.path:
        sys.path.insert(0, str(explorer_src))
    os.environ["ICA_V9_FEATURE_DB"] = str(config.storage.database)
    os.environ["MATH_CUED_CONFIG"] = str(config.path)
    import uvicorn
    uvicorn.run("explorer.server.app:app", host=config.explorer.host, port=config.explorer.port, reload=reload)
