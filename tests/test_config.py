from pathlib import Path

import pytest

from math_cued_activation.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_reference_config() -> None:
    config = load_config(ROOT / "configs" / "vibethinker_imo.toml")
    assert config.model.id == "WeiboAI/VibeThinker-3B"
    assert config.capture.layers == (20, 32)
    assert config.storage.responses == ROOT / "outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "vibethinker_imo.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(text.replace("version = 1", "version = 1\nunknown = true"))
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config(path)


def test_invalid_layer_is_rejected(tmp_path: Path) -> None:
    text = (ROOT / "configs" / "vibethinker_imo.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(text.replace("layers = [20, 32]", "layers = [99]"))
    with pytest.raises(ConfigError, match="outside model range"):
        load_config(path)
