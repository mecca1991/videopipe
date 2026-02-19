# visionpipe/inference/predictor.py
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

import visionpipe.models.faster_rcnn  # noqa: F401  # triggers @register_model
import visionpipe.models.yolo26  # noqa: F401  # triggers @register_model
from visionpipe.inference.validator import FramingValidator
from visionpipe.models.registry import build_model


class Predictor:
    def __init__(self, cfg: DictConfig, checkpoint_path: str | None = None):
        self.cfg = cfg

        self.model = build_model(cfg)

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        self.validator = FramingValidator(cfg)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        import torch

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in checkpoint:
            # Lightning checkpoint — strip "model." prefix from keys
            state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                key = k.removeprefix("model.")
                state_dict[key] = v
            self.model.model.load_state_dict(state_dict)
        else:
            self.model.model.load_state_dict(checkpoint)

    def predict(self, image) -> dict:
        image_path = None

        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = np.array(Image.open(image).convert("RGB"))

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array or file path, got {type(image)}")

        height, width = image.shape[:2]

        conf = self.cfg.inference.confidence_threshold
        iou = self.cfg.inference.nms_iou_threshold

        predictions = self.model.forward(image, conf=conf, iou=iou)
        detections = self.model.postprocess(
            predictions,
            conf_threshold=conf,
            iou_threshold=iou,
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
