from pathlib import Path

from math_cued_activation.capture import pipeline
from math_cued_activation.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_gsm8k_capture_config_maps_to_pipeline(monkeypatch) -> None:
    config = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_gsm8k.toml")
    captured = {}

    def fake_run(args) -> None:
        captured.update(vars(args))

    monkeypatch.setattr(pipeline, "run_capture", fake_run)
    pipeline.capture_from_config(config, layer=19)

    assert captured["model"] == config.model.id
    assert captured["capture_layer"] == 19
    assert captured["generated_text_dir"] == config.storage.responses
    assert captured["activation_dir"] == config.storage.activations
    assert captured["pipeline_config"].dataset.id == "openai/gsm8k"
