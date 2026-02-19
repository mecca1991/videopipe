from abc import ABC, abstractmethod

from torch import Tensor


class AbstractDetector(ABC):
    """Interface that all detection models must implement."""

    @abstractmethod
    def forward(self, images: Tensor) -> dict:
        """Run detection on a batch of images. Returns raw predictions."""

    @abstractmethod
    def compute_loss(self, predictions: dict, targets: dict) -> Tensor:
        """Calculate training loss from predictions and ground truth."""

    @abstractmethod
    def postprocess(
        self,
        predictions: dict,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[dict]:
        """Filter predictions and return structured detections."""
