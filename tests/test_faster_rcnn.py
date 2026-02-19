import pytest
import torch
from omegaconf import OmegaConf

import visionpipe.models.faster_rcnn  # noqa: F401  # triggers @register_model
from visionpipe.models.faster_rcnn import FasterRCNNDetector


@pytest.fixture
def frcnn_cfg():
    return OmegaConf.create({
        "model": {"name": "faster_rcnn", "num_classes": 80, "pretrained": True},
        "inference": {"confidence_threshold": 0.5, "nms_iou_threshold": 0.45},
    })


def test_faster_rcnn_is_registered():
    from visionpipe.models.registry import MODEL_REGISTRY
    assert "faster_rcnn" in MODEL_REGISTRY


def test_faster_rcnn_instantiates(frcnn_cfg):
    model = FasterRCNNDetector(frcnn_cfg)
    assert model is not None


def test_faster_rcnn_forward_returns_predictions(frcnn_cfg):
    model = FasterRCNNDetector(frcnn_cfg)
    images = torch.rand(2, 3, 480, 640)
    predictions = model.forward(images)
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert "boxes" in predictions[0]
    assert "scores" in predictions[0]
    assert "labels" in predictions[0]


def test_faster_rcnn_postprocess_filters_by_confidence(frcnn_cfg):
    model = FasterRCNNDetector(frcnn_cfg)
    images = torch.rand(1, 3, 480, 640)
    predictions = model.forward(images)
    detections = model.postprocess(predictions, conf_threshold=0.5, iou_threshold=0.45)
    assert isinstance(detections, list)
    assert isinstance(detections[0], list)
    for det in detections[0]:
        assert det["confidence"] >= 0.5
