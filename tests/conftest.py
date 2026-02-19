import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture
def coco_fixture(tmp_path):
    """Create a minimal COCO-format dataset for testing."""
    images_dir = tmp_path / "train2017"
    images_dir.mkdir()
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    for i in range(1, 4):
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(images_dir / f"{i:012d}.jpg")

    annotations = {
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480}
            for i in range(1, 4)
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2,
             "bbox": [300, 200, 50, 80], "area": 4000, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1,
             "bbox": [150, 150, 200, 200], "area": 40000, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "cat", "supercategory": "animal"},
            {"id": 2, "name": "dog", "supercategory": "animal"},
        ],
    }
    annotation_path = annotations_dir / "instances_train2017.json"
    annotation_path.write_text(json.dumps(annotations))

    return {
        "root": tmp_path,
        "annotation_path": annotation_path,
        "images_dir": images_dir,
        "num_images": 3,
    }
