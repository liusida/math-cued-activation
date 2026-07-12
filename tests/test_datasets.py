from pathlib import Path

from math_cued_activation.config import load_config
from math_cued_activation.datasets.registry import _normalize_row, build_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_gsm8k_row_normalization_and_prompt() -> None:
    config = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_gsm8k.toml")
    assert config.dataset.config_name == "main"
    row = _normalize_row(config, 7, {"question": "What is 2 + 3?", "answer": "work\n#### 5"})
    assert row["Problem ID"] == "gsm8k-00007"
    assert row["Problem"] == "What is 2 + 3?"
    assert row["Short Answer"] == "work\n#### 5"
    prompt = build_prompt(config, row)
    assert "grade-school math problem" in prompt
    assert "What is 2 + 3?" in prompt


def test_imo_row_uses_configured_fields() -> None:
    config = load_config(ROOT / "configs" / "smoke_qwen25_coder_3b_imo.toml")
    row = _normalize_row(config, 2, {
        "Problem ID": "imo-example",
        "Problem": "Prove it.",
        "Short Answer": "Done",
        "Category": "Algebra",
    })
    assert row["Problem ID"] == "imo-example"
    assert row["Category"] == "Algebra"
    assert "\\boxed{}" in build_prompt(config, row)
