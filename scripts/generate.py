#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.stages import generate

if __name__ == "__main__":
    args, config = parse_config(config_parser("Generate and freeze responses with vLLM."))
    generate(config, force=args.force)
