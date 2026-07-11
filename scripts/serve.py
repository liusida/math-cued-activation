#!/usr/bin/env python3
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.serve import serve

if __name__ == "__main__":
    parser = config_parser("Serve the integrated v9 Explorer.")
    parser.add_argument("--reload", action="store_true")
    args, config = parse_config(parser)
    serve(config, reload=args.reload)
