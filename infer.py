#!/usr/bin/env python3
"""Inference entrypoint for the object detection pipeline.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --config custom.yaml
    python infer.py --image path/to/image.jpg --override model.name=faster_rcnn
"""
import argparse
import json

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

    overrides = {}
    for item in args.override:
        key, value = item.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        overrides[key] = value

    cfg = load_config(config_path=args.config, overrides=overrides or None)
    predictor = Predictor(cfg)
    result = predictor.predict(args.image)

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
