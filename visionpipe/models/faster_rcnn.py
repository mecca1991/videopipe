import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import nms

from visionpipe.models.base import AbstractDetector
from visionpipe.models.registry import register_model

COCO_CLASSES = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]


@register_model("faster_rcnn")
class FasterRCNNDetector(AbstractDetector):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        if cfg.model.pretrained:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        else:
            weights = None
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        if cfg.model.num_classes != 91:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, cfg.model.num_classes)
        self.model.eval()

    def forward(self, images) -> list[dict]:
        """Run Faster R-CNN inference.

        Args:
            images: numpy array (H, W, C) uint8 or tensor [B, C, H, W] float 0-1.
        """
        if isinstance(images, np.ndarray):
            tensor = torch.from_numpy(images).permute(2, 0, 1).float() / 255.0
            image_list = [tensor]
        elif isinstance(images, torch.Tensor):
            image_list = list(images.unbind(0)) if images.ndim == 4 else [images]
        else:
            raise TypeError(f"Expected numpy array or torch.Tensor, got {type(images)}")

        with torch.no_grad():
            predictions = self.model(image_list)
        return predictions

    def compute_loss(self, predictions: dict, targets: dict) -> Tensor:
        raise NotImplementedError(
            "Faster R-CNN loss is computed during forward pass in training mode. "
            "See lightning_module.py for the training integration."
        )

    def postprocess(self, predictions: list[dict], conf_threshold: float, iou_threshold: float) -> list[list[dict]]:
        all_detections = []
        for pred in predictions:
            boxes = pred["boxes"]
            scores = pred["scores"]
            labels = pred["labels"]

            keep = scores >= conf_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) > 0:
                nms_keep = nms(boxes, scores, iou_threshold)
                boxes = boxes[nms_keep]
                scores = scores[nms_keep]
                labels = labels[nms_keep]

            image_detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].tolist()
                cls_id = int(labels[i])
                cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
                image_detections.append(
                    {
                        "class_label": cls_name,
                        "confidence": round(float(scores[i]), 4),
                        "bbox": {
                            "x": round(x1, 1),
                            "y": round(y1, 1),
                            "width": round(x2 - x1, 1),
                            "height": round(y2 - y1, 1),
                        },
                    }
                )
            all_detections.append(image_detections)
        return all_detections
