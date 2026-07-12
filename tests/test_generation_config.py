from pathlib import Path

from math_cued_activation.config import load_config
from math_cued_activation.generation import vllm
from math_cued_activation.stages import checkpoint_path


ROOT = Path(__file__).resolve().parents[1]


def test_generation_config_maps_to_backend(monkeypatch) -> None:
    config = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_imo.toml")
    captured = {}

    def fake_run(args) -> None:
        captured.update(vars(args))

    monkeypatch.setattr(vllm, "run_generation", fake_run)
    vllm.generate_from_config(config, force=True)

    assert captured["model"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert captured["tokenizer_model"] == config.model.tokenizer
    assert captured["api_url"] == config.vllm.api_url
    assert captured["context_window"] == 32768
    assert captured["sample_size"] == 5
    assert captured["generated_text_dir"] == config.storage.responses
    assert captured["rerun_existing"] is True
    assert captured["max_new_tokens"] is None


def test_qwen_smoke_uses_eager_vllm() -> None:
    config = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_imo.toml")
    assert config.vllm.enforce_eager is True
    assert config.vllm.gpu_memory_utilization == 0.75
    assert config.vllm.max_model_len == 32768
    assert checkpoint_path(config, 19).name == "qwen_qwen2_5_coder_3b_instruct_layer19_c2048_iter10.pt"


def test_vibethinker_keeps_legacy_checkpoint_name() -> None:
    config = load_config(ROOT / "configs" / "vibethinker_imo.toml")
    assert checkpoint_path(config, 20).name == "vibethinker_only_layer20_c2048_iter100.pt"
