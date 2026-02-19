import albumentations as A
from omegaconf import DictConfig


def build_transforms(cfg: DictConfig, is_train: bool) -> A.Compose:
    """
    Build the augmentation pipeline based on the configuration and training mode.

    Args:
        cfg (DictConfig): The configuration object.
        is_train (bool): Whether the pipeline is for training or not.

    Returns:
        A.Compose: The composed augmentation pipeline.
    """
    bbox_params = A.BboxParams(
        format="pascal_voc",
        label_fields=["labels"],
        min_visibility=0.3,
    )

    image_size = cfg.data.get("image_size") if "data" in cfg else None
    aug_cfg = cfg.augmentation

    base_transforms = []
    if image_size:
        base_transforms.append(A.Resize(height=image_size, width=image_size))

    if not aug_cfg.enabled or not is_train:
        return A.Compose(base_transforms, bbox_params=bbox_params)

    transforms = base_transforms + [
        A.RandomBrightnessContrast(
            brightness_limit=aug_cfg.brightness_limit,
            contrast_limit=aug_cfg.contrast_limit,
            p=0.5,
        ),
        A.Rotate(limit=aug_cfg.rotation_limit, p=0.5),
        A.HorizontalFlip(p=aug_cfg.horizontal_flip_prob),
    ]

    return A.Compose(transforms, bbox_params=bbox_params)
