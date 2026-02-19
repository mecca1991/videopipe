# tests/test_predictor.py
import json
import pytest
import numpy as np
from PIL import Image
from omegaconf import OmegaConf


@pytest.fixture
def predictor_cfg():
    return OmegaConf.create({
        "model": {"name": "faster_rcnn", "num_classes": 80, "pretrained": True},
        "inference": {
            "confidence_threshold": 0.5,
            "nms_iou_threshold": 0.45,
            "validation": {
                "enabled": True,
                "center_tolerance": 0.3,
                "min_visible_fraction": 0.9,
            },
        },
    })


def test_predictor_from_numpy_array(predictor_cfg):
    from visionpipe.inference.predictor import Predictor
    predictor = Predictor(predictor_cfg)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = predictor.predict(image)
    assert "image_size" in result
    assert result["image_size"]["width"] == 640
    assert result["image_size"]["height"] == 480
    assert "detections" in result
    assert isinstance(result["detections"], list)


def test_predictor_from_file_path(predictor_cfg, tmp_path):
    from visionpipe.inference.predictor import Predictor
    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    predictor = Predictor(predictor_cfg)
    result = predictor.predict(str(img_path))
    assert result["image_path"] == str(img_path)
    assert "detections" in result


def test_predictor_detections_have_valid_framing_field(predictor_cfg):
    from visionpipe.inference.predictor import Predictor
    predictor = Predictor(predictor_cfg)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = predictor.predict(image)
    for det in result["detections"]:
        assert "valid_framing" in det
        assert isinstance(det["valid_framing"], bool)


def test_predictor_output_is_json_serializable(predictor_cfg):
    from visionpipe.inference.predictor import Predictor
    predictor = Predictor(predictor_cfg)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = predictor.predict(image)
    json.dumps(result)
