# visionpipe/inference/validator.py
from omegaconf import DictConfig


class FramingValidator:
    def __init__(self, cfg: DictConfig):
        val_cfg = cfg.inference.validation
        self.enabled = val_cfg.enabled
        self.center_tolerance = val_cfg.center_tolerance
        self.min_visible_fraction = val_cfg.min_visible_fraction

    def validate(self, detection: dict, image_width: int, image_height: int) -> bool:
        if not self.enabled:
            return True

        bbox = detection["bbox"]
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        centroid_x = bx + bw / 2
        centroid_y = by + bh / 2
        center_x = image_width / 2
        center_y = image_height / 2

        dx = abs(centroid_x - center_x) / image_width
        dy = abs(centroid_y - center_y) / image_height

        if dx > self.center_tolerance or dy > self.center_tolerance:
            return False

        visible_x1 = max(bx, 0)
        visible_y1 = max(by, 0)
        visible_x2 = min(bx + bw, image_width)
        visible_y2 = min(by + bh, image_height)

        visible_area = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
        total_area = bw * bh

        if total_area <= 1e-6:
            return False

        if visible_area / total_area < self.min_visible_fraction:
            return False

        return True
