from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPLORER_SRC = ROOT / "explorer" / "src"
for source_root in (SRC, EXPLORER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
