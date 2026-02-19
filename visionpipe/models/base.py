from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class AbstractDetector(ABC):
    """Interface that all detection models must implement."""

    @abstractmethod
    def forward(self, images: Any, **kwargs: Any) -> Any:
        """Run detection on images. Accepts numpy array (H,W,C) or tensor [B,C,H,W]."""

    @abstractmethod
    def compute_loss(self, predictions: Any, targets: Any) -> Tensor:
        """Calculate training loss from predictions and ground truth."""

    @abstractmethod
    def postprocess(
        self,
        predictions: Any,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[list[dict]]:
        """Filter predictions and return structured detections."""
