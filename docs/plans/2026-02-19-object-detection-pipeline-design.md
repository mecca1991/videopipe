# Object Detection Pipeline Design

## Overview

A modular, end-to-end object detection pipeline in Python with two subsystems: a Training Pipeline for model tuning on COCO dataset, and an Inference Pipeline for evaluating images and returning structured JSON detection data.

## Decisions

- **Use case:** Learning/portfolio project
- **Dataset:** COCO (80 object classes)
- **Models:** YOLO26 (single-stage, NMS-free) + Faster R-CNN (two-stage) behind an abstract registry
- **Training framework:** PyTorch Lightning
- **Config:** OmegaConf with a single YAML file, zero hardcoding
- **Augmentation:** Albumentations (handles bounding box transforms automatically)
- **Inference output:** JSON with all detections, each annotated with a `valid_framing` boolean flag
- **Package management:** uv

## Project Structure

```
visionpipe/
├── config.yaml                  # Single source of truth for all parameters
├── train.py                     # CLI entrypoint: python train.py
├── infer.py                     # CLI entrypoint: python infer.py --image path
├── visionpipe/
│   ├── __init__.py
│   ├── config.py                # Loads & validates YAML via OmegaConf
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset for COCO format
│   │   ├── dataloader.py        # DataLoader factory with train/val splits
│   │   └── augmentations.py     # Albumentations transforms (config-driven)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # AbstractDetector interface
│   │   ├── registry.py          # String -> class mapping (factory pattern)
│   │   ├── yolo26.py            # YOLO26 wrapper (ultralytics)
│   │   └── faster_rcnn.py       # Faster R-CNN wrapper (torchvision)
│   ├── training/
│   │   ├── __init__.py
│   │   └── lightning_module.py  # Generic training loop for any detector
│   └── inference/
│       ├── __init__.py
│       ├── predictor.py         # Runs model on single image, applies NMS
│       └── validator.py         # Framing validation logic
└── tests/
    ├── test_config.py
    ├── test_dataset.py
    ├── test_registry.py
    ├── test_validator.py
    └── test_inference.py
```

## Configuration Schema

All tunable parameters live in `config.yaml`:

```yaml
model:
  name: "yolo26"            # Swapped via registry ("yolo26" or "faster_rcnn")
  num_classes: 80
  pretrained: true

data:
  dataset_root: "./data/coco"
  annotation_file: "instances_train2017.json"
  batch_size: 16
  num_workers: 4
  train_split: 0.8

augmentation:
  enabled: true
  brightness_limit: 0.2
  contrast_limit: 0.2
  rotation_limit: 15
  horizontal_flip_prob: 0.5

training:
  max_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
  checkpoint_dir: "./checkpoints"

inference:
  confidence_threshold: 0.5
  nms_iou_threshold: 0.45
  validation:
    enabled: true
    center_tolerance: 0.3
    min_visible_fraction: 0.9
```

## Component Design

### 1. Config (`visionpipe/config.py`)

Loads `config.yaml` via OmegaConf. Provides typed dot-access (`cfg.model.name`). Validates required fields at startup.

### 2. Data Pipeline (`visionpipe/data/`)

**`dataset.py`** — PyTorch `Dataset` that:
- Parses COCO JSON to build an `image_id -> [annotations]` index
- On `__getitem__(idx)`: loads image, fetches bounding boxes + labels, applies augmentations, returns tensors

**`augmentations.py`** — Builds an Albumentations `Compose` pipeline from config. Albumentations automatically transforms bounding box coordinates when images are rotated/flipped/cropped.

**`dataloader.py`** — Factory function that:
- Creates train DataLoader (shuffled, augmented)
- Creates val DataLoader (no shuffle, no augmentation)
- Splits dataset by `train_split` ratio

### 3. Model Layer (`visionpipe/models/`)

**`base.py`** — `AbstractDetector` with three required methods:
- `forward(images)` — run detection, return raw predictions
- `compute_loss(predictions, targets)` — calculate training loss
- `postprocess(predictions, conf_threshold, iou_threshold)` — apply filtering, return structured detections

**`registry.py`** — Maps config string to class. `@register_model("yolo26")` decorator + `build_model(cfg)` factory function.

**`yolo26.py`** — Wraps `ultralytics.YOLO` with YOLO26 weights. NMS-free inference (one-to-one detection head). Adapts ultralytics output format to our `AbstractDetector` interface.

**`faster_rcnn.py`** — Wraps `torchvision.models.detection.fasterrcnn_resnet50_fpn`. Requires explicit NMS via `torchvision.ops.nms()` in `postprocess()`.

### 4. Training (`visionpipe/training/lightning_module.py`)

`DetectionModule(LightningModule)` that:
- Accepts any `AbstractDetector` via the registry
- `training_step`: forward pass -> compute loss -> log
- `validation_step`: forward -> loss -> postprocess -> compute mAP (via torchmetrics)
- `configure_optimizers`: reads optimizer/scheduler from config
- Lightning handles: GPU management, checkpointing, early stopping, logging

### 5. Inference (`visionpipe/inference/`)

**`predictor.py`** — `Predictor` class that:
- Loads trained model from checkpoint
- Accepts image path or numpy array
- Preprocesses, runs forward pass, calls `postprocess()`
- Applies framing validation to each detection
- Returns structured JSON

**`validator.py`** — `FramingValidator` that checks each detection:
- Calculates bounding box centroid
- Checks if centroid is within `center_tolerance` of image center
- Checks if bounding box is at least `min_visible_fraction` within the frame
- Returns `True`/`False` per detection

### 6. JSON Output Format

```json
{
  "image_path": "test.jpg",
  "image_size": {"width": 640, "height": 480},
  "detections": [
    {
      "class_label": "cat",
      "confidence": 0.92,
      "bbox": {"x": 120, "y": 80, "width": 200, "height": 180},
      "valid_framing": true
    }
  ]
}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | Core deep learning framework + Faster R-CNN |
| `ultralytics` | YOLO26 model |
| `pytorch-lightning` | Training loop abstraction |
| `albumentations` | Image augmentation with bbox support |
| `omegaconf` | YAML config parsing |
| `pycocotools` | COCO annotation parsing |
| `torchmetrics` | mAP calculation |
| `Pillow` | Image loading |

## Testing Strategy

- Unit tests for config loading and validation
- Unit tests for dataset (single image + annotation round-trip)
- Unit tests for model registry (registration and lookup)
- Unit tests for framing validator (centroid math with known inputs)
- Integration test for inference (pretrained model -> test image -> validate JSON schema)
- Training tested via single-batch `training_step` call (full training too slow for CI)

## Out of Scope

- No web API / REST endpoint (CLI scripts only per spec)
- No model export to ONNX/TensorRT
- No custom training dashboard (Lightning TensorBoard logging is sufficient)
- No distributed/multi-GPU training configuration
