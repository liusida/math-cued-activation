from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("start_vllm", "generate", "stop_vllm", "capture", "fit", "register", "enrich", "validate", "serve")


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help_without_install(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
