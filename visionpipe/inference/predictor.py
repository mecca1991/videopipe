# visionpipe/inference/predictor.py
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig

from visionpipe.models.registry import build_model
from visionpipe.inference.validator import FramingValidator


class Predictor:
    def __init__(self, cfg: DictConfig, checkpoint_path: str | None = None):
        self.cfg = cfg

        # Import model modules to trigger registration
        import visionpipe.models.yolo26  # noqa: F401
        import visionpipe.models.faster_rcnn  # noqa: F401

        self.model = build_model(cfg)
        self.validator = FramingValidator(cfg)

    def predict(self, image) -> dict:
        image_path = None

        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = np.array(Image.open(image).convert("RGB"))

        height, width = image.shape[:2]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        predictions = self.model.forward(image_tensor)
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
