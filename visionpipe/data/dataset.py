import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from omegaconf import DictConfig

from visionpipe.data.augmentations import build_transforms


class COCODetectionDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.cfg = cfg
        root = Path(cfg.data.dataset_root)

        annotation_key = "train_annotation" if split == "train" else "val_annotation"
        images_key = "train_images" if split == "train" else "val_images"

        annotation_path = root / cfg.data[annotation_key]
        self.images_dir = root / cfg.data[images_key]

        with open(annotation_path) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        self.img_to_anns: dict[int, list[dict]] = {}
        for img in self.images:
            self.img_to_anns[img["id"]] = []
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 0:
                self.img_to_anns[ann["image_id"]].append(ann)

        is_train = split == "train"
        self.transform = build_transforms(cfg, is_train=is_train)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]
        image = np.array(Image.open(img_path).convert("RGB"))

        anns = self.img_to_anns[img_info["id"]]

        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        labels = transformed["labels"]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if len(bboxes) > 0:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return image_tensor, target
