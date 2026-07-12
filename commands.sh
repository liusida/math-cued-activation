#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/smoke_qwen25_coder_3b_gsm8k.toml"

# Generation uses the background vLLM server. The trap releases GPU memory if
# generation fails or this script is interrupted.
uv run python scripts/start_vllm.py --config "$CONFIG"
trap 'uv run python scripts/stop_vllm.py --config "$CONFIG" || true' EXIT
uv run python scripts/generate.py --config "$CONFIG"
uv run python scripts/stop_vllm.py --config "$CONFIG"
trap - EXIT

# Transformers replay and the downstream ICA/Explorer pipeline.
uv run python scripts/capture.py --config "$CONFIG"
uv run python scripts/fit.py --config "$CONFIG"
uv run python scripts/register.py --config "$CONFIG"
uv run python scripts/enrich.py --config "$CONFIG"
uv run python scripts/validate.py --config "$CONFIG"
uv run python scripts/serve.py --config "$CONFIG"
