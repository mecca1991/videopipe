import numpy as np
from omegaconf import OmegaConf


def test_build_train_transforms_returns_callable():
    from visionpipe.data.augmentations import build_transforms

    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "rotation_limit": 15,
                "horizontal_flip_prob": 0.5,
            }
        }
    )
    transform = build_transforms(cfg, is_train=True)
    assert callable(transform)


def test_transforms_preserve_bbox_count():
    from visionpipe.data.augmentations import build_transforms

    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "rotation_limit": 15,
                "horizontal_flip_prob": 0.5,
            }
        }
    )
    transform = build_transforms(cfg, is_train=True)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200], [300, 300, 400, 400]]
    labels = [1, 2]

    result = transform(image=image, bboxes=bboxes, labels=labels)
    assert "image" in result
    assert "bboxes" in result
    assert len(result["bboxes"]) <= len(bboxes)


def test_val_transforms_only_resize():
    from visionpipe.data.augmentations import build_transforms

    cfg = OmegaConf.create(
        {
            "augmentation": {
                "enabled": True,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "rotation_limit": 15,
                "horizontal_flip_prob": 0.5,
            }
        }
    )
    transform = build_transforms(cfg, is_train=False)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200]]
    labels = [1]

    result = transform(image=image, bboxes=bboxes, labels=labels)
    assert len(result["bboxes"]) == 1


def test_disabled_augmentation():
    from visionpipe.data.augmentations import build_transforms

    cfg = OmegaConf.create({"augmentation": {"enabled": False}})
    transform = build_transforms(cfg, is_train=True)
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 200]]
    labels = [1]

    result = transform(image=image, bboxes=bboxes, labels=labels)
    assert len(result["bboxes"]) == 1
