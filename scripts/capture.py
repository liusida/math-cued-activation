#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.stages import capture

if __name__ == "__main__":
    args, config = parse_config(config_parser("Replay saved tokens and capture residual-post activations."))
    capture(config, force=args.force)
