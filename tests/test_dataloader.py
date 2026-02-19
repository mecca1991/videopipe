import torch
from omegaconf import OmegaConf


def _make_cfg(coco_fixture):
    return OmegaConf.create(
        {
            "data": {
                "dataset_root": str(coco_fixture["root"]),
                "train_annotation": "annotations/instances_train2017.json",
                "val_annotation": "annotations/instances_train2017.json",
                "train_images": "train2017",
                "val_images": "train2017",
                "batch_size": 2,
                "num_workers": 0,
            },
            "augmentation": {"enabled": False},
        }
    )


def test_create_dataloaders_returns_train_and_val(coco_fixture):
    from visionpipe.data.dataloader import create_dataloaders

    cfg = _make_cfg(coco_fixture)
    train_dl, val_dl = create_dataloaders(cfg)
    assert train_dl is not None
    assert val_dl is not None


def test_train_dataloader_yields_batches(coco_fixture):
    from visionpipe.data.dataloader import create_dataloaders

    cfg = _make_cfg(coco_fixture)
    train_dl, _ = create_dataloaders(cfg)
    batch = next(iter(train_dl))
    images, targets = batch
    assert isinstance(images, torch.Tensor)
    assert images.shape[0] <= cfg.data.batch_size
    assert isinstance(targets, list)
    assert "boxes" in targets[0]
