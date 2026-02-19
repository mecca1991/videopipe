import pytest
import torch
from visionpipe.models.base import AbstractDetector
from visionpipe.models.registry import register_model, build_model, MODEL_REGISTRY


class _DummyDetector(AbstractDetector):
    """Minimal concrete implementation for testing the registry."""

    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, images):
        return {"dummy": images}

    def compute_loss(self, predictions, targets):
        return torch.tensor(0.0)

    def postprocess(self, predictions, conf_threshold, iou_threshold):
        return []


def test_register_and_lookup():
    register_model("dummy")(_DummyDetector)
    assert "dummy" in MODEL_REGISTRY
    assert MODEL_REGISTRY["dummy"] is _DummyDetector


def test_build_model_from_config():
    from omegaconf import OmegaConf

    register_model("dummy")(_DummyDetector)
    cfg = OmegaConf.create({"model": {"name": "dummy", "num_classes": 10, "pretrained": False}})
    model = build_model(cfg)
    assert isinstance(model, _DummyDetector)
    assert model.cfg == cfg


def test_build_model_unknown_raises():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": {"name": "nonexistent"}})
    with pytest.raises(KeyError, match="nonexistent"):
        build_model(cfg)


def test_abstract_detector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractDetector()
