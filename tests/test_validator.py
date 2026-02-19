# tests/test_validator.py
import pytest
from omegaconf import OmegaConf


def _make_cfg(center_tolerance=0.3, min_visible=0.9):
    return OmegaConf.create({
        "inference": {
            "validation": {
                "enabled": True,
                "center_tolerance": center_tolerance,
                "min_visible_fraction": min_visible,
            }
        }
    })


def test_centered_fully_visible_object_is_valid():
    from visionpipe.inference.validator import FramingValidator
    v = FramingValidator(_make_cfg())
    detection = {"bbox": {"x": 220, "y": 140, "width": 200, "height": 200}}
    assert v.validate(detection, image_width=640, image_height=480) is True


def test_object_far_from_center_is_invalid():
    from visionpipe.inference.validator import FramingValidator
    v = FramingValidator(_make_cfg(center_tolerance=0.1))
    detection = {"bbox": {"x": 10, "y": 10, "width": 50, "height": 50}}
    assert v.validate(detection, image_width=640, image_height=480) is False


def test_object_partially_outside_frame_is_invalid():
    from visionpipe.inference.validator import FramingValidator
    v = FramingValidator(_make_cfg(center_tolerance=0.5, min_visible=0.9))
    detection = {"bbox": {"x": 550, "y": 200, "width": 100, "height": 80}}
    assert v.validate(detection, image_width=640, image_height=480) is True

    detection2 = {"bbox": {"x": 560, "y": 200, "width": 100, "height": 80}}
    assert v.validate(detection2, image_width=640, image_height=480) is False


def test_validation_disabled_always_returns_true():
    from visionpipe.inference.validator import FramingValidator
    cfg = OmegaConf.create({
        "inference": {"validation": {"enabled": False, "center_tolerance": 0.0, "min_visible_fraction": 1.0}}
    })
    v = FramingValidator(cfg)
    detection = {"bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}
    assert v.validate(detection, image_width=640, image_height=480) is True
