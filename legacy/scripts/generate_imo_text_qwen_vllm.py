#!/usr/bin/env python3
"""Generate IMO-AnswerBench responses from Qwen2.5-Coder through local vLLM."""

from __future__ import annotations

import os
import sys

from generate_imo_text_vllm import main


DEFAULTS = {
    "--api-url": os.environ.get("QWEN_VLLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions"),
    "--model": os.environ.get("QWEN_VLLM_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    "--server-name": os.environ.get("QWEN_VLLM_SERVER_NAME", "qwen-local"),
    "--sample-size": os.environ.get("QWEN_IMO_SAMPLE_SIZE", "400"),
    "--start-index": os.environ.get("QWEN_IMO_START_INDEX", "0"),
    "--concurrency": os.environ.get("QWEN_VLLM_CONCURRENCY", "96"),
    "--context-window": os.environ.get("QWEN_VLLM_CONTEXT_WINDOW", "32768"),
    "--generated-text-dir": os.environ.get(
        "QWEN_IMO_OUTPUT_DIR",
        "outputs/imo-answerbench-responses/Qwen__Qwen2.5-Coder-3B-Instruct",
    ),
    "--tokenizer-model": os.environ.get("QWEN_TOKENIZER_MODEL", "Qwen/Qwen2.5-Coder-3B-Instruct"),
}


def has_arg(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def add_default_args() -> None:
    for flag, value in DEFAULTS.items():
        if not has_arg(flag):
            sys.argv.extend([flag, value])


if __name__ == "__main__":
    add_default_args()
    main()
