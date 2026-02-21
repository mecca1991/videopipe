#!/usr/bin/env python3
"""Training entrypoint for the object detection pipeline.

Usage:
    python train.py                              # Train with config.yaml defaults
    python train.py --config custom.yaml         # Use custom config
    python train.py --override model.name=faster_rcnn --override training.max_epochs=4
"""

import argparse

import lightning as L
from omegaconf import OmegaConf

from visionpipe.config import load_config
from visionpipe.data.dataloader import create_dataloaders
from visionpipe.training.lightning_module import DetectionModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train an object detection model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides in key=value format (repeatable)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(config_path=args.config)
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, overrides)

    print(f"Training model: {cfg.model.name}")
    print(f"Max epochs: {cfg.training.max_epochs}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    train_loader, val_loader = create_dataloaders(cfg)
    module = DetectionModule(cfg)

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        default_root_dir=cfg.training.checkpoint_dir,
        accelerator="auto",
        log_every_n_steps=10,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Training complete. Checkpoints saved to {cfg.training.checkpoint_dir}")


if __name__ == "__main__":
    main()
