#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.stages import enrich

if __name__ == "__main__":
    args, config = parse_config(config_parser("Finalize feature IDs and build Explorer evidence."))
    enrich(config, force=args.force)
