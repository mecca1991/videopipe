import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def training_cfg():
    return OmegaConf.create({
        "model": {"name": "faster_rcnn", "num_classes": 80, "pretrained": True},
        "training": {
            "max_epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "checkpoint_dir": "./checkpoints",
        },
        "inference": {
            "confidence_threshold": 0.5,
            "nms_iou_threshold": 0.45,
        },
    })


def test_lightning_module_instantiates(training_cfg):
    from visionpipe.training.lightning_module import DetectionModule
    module = DetectionModule(training_cfg)
    assert module is not None


def test_lightning_module_configure_optimizers(training_cfg):
    from visionpipe.training.lightning_module import DetectionModule
    module = DetectionModule(training_cfg)
    result = module.configure_optimizers()
    assert result is not None


def test_lightning_module_training_step_returns_loss(training_cfg):
    from visionpipe.training.lightning_module import DetectionModule
    module = DetectionModule(training_cfg)

    images = torch.rand(2, 3, 300, 300)
    targets = [
        {"boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[30.0, 30.0, 100.0, 100.0]]), "labels": torch.tensor([2])},
    ]
    batch = (images, targets)

    loss = module.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0
