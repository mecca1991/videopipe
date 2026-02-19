#!/usr/bin/env python3
"""Inference entrypoint for the object detection pipeline.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --config custom.yaml
    python infer.py --image path/to/image.jpg --override model.name=faster_rcnn
"""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from visionpipe.config import load_config
from visionpipe.inference.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run object detection inference on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides in key=value format (repeatable)",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.image).is_file():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path=args.config)
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, overrides)

    predictor = Predictor(cfg)
    result = predictor.predict(args.image)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
