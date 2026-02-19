# visionpipe/inference/predictor.py
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import DictConfig

import visionpipe.models.yolo26  # noqa: F401  # triggers @register_model
import visionpipe.models.faster_rcnn  # noqa: F401  # triggers @register_model
from visionpipe.models.registry import build_model
from visionpipe.inference.validator import FramingValidator


class Predictor:
    def __init__(self, cfg: DictConfig, checkpoint_path: str | None = None):
        self.cfg = cfg

        self.model = build_model(cfg)
        self.validator = FramingValidator(cfg)

    def predict(self, image) -> dict:
        image_path = None

        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = np.array(Image.open(image).convert("RGB"))

        height, width = image.shape[:2]

        predictions = self.model.forward(image)
        detections = self.model.postprocess(
            predictions,
            conf_threshold=self.cfg.inference.confidence_threshold,
            iou_threshold=self.cfg.inference.nms_iou_threshold,
        )

        image_detections = detections[0] if detections else []

        for det in image_detections:
            det["valid_framing"] = self.validator.validate(det, width, height)

        result = {
            "image_size": {"width": width, "height": height},
            "detections": image_detections,
        }
        if image_path:
            result["image_path"] = image_path

        return result
