import torch
import pytest
from omegaconf import OmegaConf


def _make_cfg(coco_fixture, augmentation_enabled=False):
    return OmegaConf.create({
        "data": {
            "dataset_root": str(coco_fixture["root"]),
            "train_annotation": "annotations/instances_train2017.json",
            "train_images": "train2017",
            "batch_size": 2,
            "num_workers": 0,
        },
        "augmentation": {"enabled": augmentation_enabled},
    })


def test_dataset_length(coco_fixture):
    from visionpipe.data.dataset import COCODetectionDataset
    cfg = _make_cfg(coco_fixture)
    ds = COCODetectionDataset(cfg, split="train")
    assert len(ds) == coco_fixture["num_images"]


def test_dataset_getitem_returns_image_and_target(coco_fixture):
    from visionpipe.data.dataset import COCODetectionDataset
    cfg = _make_cfg(coco_fixture)
    ds = COCODetectionDataset(cfg, split="train")
    image, target = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.ndim == 3
    assert "boxes" in target
    assert "labels" in target
    assert isinstance(target["boxes"], torch.Tensor)
    assert isinstance(target["labels"], torch.Tensor)


def test_dataset_bbox_format_is_xyxy(coco_fixture):
    from visionpipe.data.dataset import COCODetectionDataset
    cfg = _make_cfg(coco_fixture)
    ds = COCODetectionDataset(cfg, split="train")
    _, target = ds[0]
    boxes = target["boxes"]
    assert boxes.shape[1] == 4
    assert (boxes[:, 2] > boxes[:, 0]).all()
    assert (boxes[:, 3] > boxes[:, 1]).all()


def test_dataset_image_with_no_annotations(coco_fixture):
    from visionpipe.data.dataset import COCODetectionDataset
    cfg = _make_cfg(coco_fixture)
    ds = COCODetectionDataset(cfg, split="train")
    image, target = ds[2]
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
