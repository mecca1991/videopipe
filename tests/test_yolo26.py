import pytest
import torch
from omegaconf import OmegaConf

import visionpipe.models.yolo26  # noqa: F401  # triggers @register_model
from visionpipe.models.registry import MODEL_REGISTRY
from visionpipe.models.yolo26 import YOLO26Detector


@pytest.fixture
def yolo_cfg():
    return OmegaConf.create(
        {
            "model": {"name": "yolo26", "num_classes": 80, "pretrained": True},
            "inference": {"confidence_threshold": 0.5, "nms_iou_threshold": 0.45},
        }
    )


def test_yolo26_is_registered():
    assert "yolo26" in MODEL_REGISTRY


def test_yolo26_instantiates(yolo_cfg):
    model = YOLO26Detector(yolo_cfg)
    assert model is not None


def test_yolo26_postprocess_returns_list(yolo_cfg):
    model = YOLO26Detector(yolo_cfg)
    images = torch.rand(1, 3, 640, 640)
    predictions = model.forward(images)
    detections = model.postprocess(predictions, conf_threshold=0.25, iou_threshold=0.45)
    assert isinstance(detections, list)
    assert isinstance(detections[0], list)
