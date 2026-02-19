import torch
from torch import Tensor
from ultralytics import YOLO
from omegaconf import DictConfig

from visionpipe.models.base import AbstractDetector
from visionpipe.models.registry import register_model


@register_model("yolo26")
class YOLO26Detector(AbstractDetector):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        model_variant = "yolo26n.pt"  # nano variant for speed
        self.model = YOLO(model_variant)

    def forward(self, images) -> list:
        """Run YOLO26 inference.

        Args:
            images: numpy array (H, W, C) or tensor [B, C, H, W].
                    Ultralytics handles resizing and preprocessing internally.
        """
        results = self.model(images, verbose=False)
        return results

    def compute_loss(self, predictions, targets) -> Tensor:
        raise NotImplementedError(
            "YOLO26 loss is computed internally by the ultralytics trainer."
        )

    def postprocess(
        self,
        predictions: list,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[list[dict]]:
        all_detections = []
        for result in predictions:
            image_detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i])
                    cls_name = result.names.get(cls_id, str(cls_id))
                    image_detections.append({
                        "class_label": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x": round(x1, 1),
                            "y": round(y1, 1),
                            "width": round(x2 - x1, 1),
                            "height": round(y2 - y1, 1),
                        },
                    })
            all_detections.append(image_detections)
        return all_detections
