#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.vllm_server import start_vllm

if __name__ == "__main__":
    args, config = parse_config(config_parser("Start the configured vLLM server and wait until ready."))
    start_vllm(config)
