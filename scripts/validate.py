#!/usr/bin/env python3
import json
import sys
import _bootstrap  # noqa: F401
from math_cued_activation.cli import config_parser, parse_config
from math_cued_activation.validation import validate_pipeline

if __name__ == "__main__":
    parser = config_parser("Validate pipeline artifacts and Explorer consistency.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args, config = parse_config(parser)
    report = validate_pipeline(config)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(rendered + "\n")
    raise SystemExit(0 if report["ok"] else 1)
