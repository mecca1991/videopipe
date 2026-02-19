import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from visionpipe.data.dataset import COCODetectionDataset


def _collate_fn(batch):
    """Custom collate: images stack into a tensor, targets stay as a list of dicts."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def create_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = COCODetectionDataset(cfg, split="train")
    val_dataset = COCODetectionDataset(cfg, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=_collate_fn,
    )

    return train_loader, val_loader
