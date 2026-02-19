# VisionPipe

A modular, end-to-end object detection pipeline built with PyTorch. Supports swappable model backbones through a registry pattern, config-driven training via PyTorch Lightning, and structured JSON inference with framing validation.

## Architecture

The pipeline is split into two subsystems:

**Training** — Fine-tune detection models on COCO with configurable augmentation, automatic checkpointing, and mAP evaluation. Swap between YOLO26 and Faster R-CNN by changing one line in the config.

**Inference** — Feed an image in, get structured JSON out. Each detection includes bounding box coordinates, class label, confidence score, and a `valid_framing` flag indicating whether the object is centered and fully visible in the frame.

### How the model abstraction works

All detection models implement an `AbstractDetector` interface with three methods: `forward()`, `compute_loss()`, and `postprocess()`. A registry maps config strings to concrete classes, so the training loop and inference pipeline are model-agnostic.

```
config.yaml                    Registry                     Training Loop
model.name: "yolo26"  ──>  MODEL_REGISTRY["yolo26"]  ──>  DetectionModule(model)
model.name: "faster_rcnn"  ──>  MODEL_REGISTRY["faster_rcnn"]  ──>  same training loop
```

## Project Structure

```
visionpipe/
├── config.yaml                     # All tunable parameters (zero hardcoding)
├── train.py                        # Training entrypoint
├── infer.py                        # Inference entrypoint
├── visionpipe/
│   ├── config.py                   # YAML config loader (OmegaConf)
│   ├── data/
│   │   ├── dataset.py              # COCO-format PyTorch Dataset
│   │   ├── dataloader.py           # Train/val DataLoader factory
│   │   └── augmentations.py        # Albumentations pipeline (config-driven)
│   ├── models/
│   │   ├── base.py                 # AbstractDetector interface
│   │   ├── registry.py             # Model factory + registration decorator
│   │   ├── yolo26.py               # YOLO26 (Ultralytics) — single-stage, NMS-free
│   │   └── faster_rcnn.py          # Faster R-CNN (torchvision) — two-stage
│   ├── training/
│   │   └── lightning_module.py     # Model-agnostic PyTorch Lightning module
│   └── inference/
│       ├── predictor.py            # Image → detections
│       └── validator.py            # Framing validation (centering + visibility)
└── tests/
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- GPU recommended for training (CPU works for inference)

### Installation

```bash
git clone https://github.com/<your-username>/visionpipe.git
cd visionpipe
uv sync
```

### Configuration

All parameters live in `config.yaml`. Key sections:

```yaml
model:
  name: "yolo26"          # or "faster_rcnn"
  num_classes: 80
  pretrained: true

training:
  max_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"

inference:
  confidence_threshold: 0.5
  nms_iou_threshold: 0.45
  validation:
    center_tolerance: 0.3
    min_visible_fraction: 0.9
```

See [`config.yaml`](config.yaml) for the full schema with all augmentation, data, and training parameters.

### Training

```bash
# Download COCO dataset to ./data/coco (see instructions below)

# Train with default config
python train.py

# Train with a different model
python train.py model.name=faster_rcnn
```

Training logs are written to TensorBoard. Checkpoints are saved to `./checkpoints/`.

### Inference

```bash
python infer.py --image path/to/image.jpg
```

Returns JSON with detections:

```json
{
  "image_path": "image.jpg",
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

### Dataset Setup

Download the COCO 2017 dataset:

```bash
mkdir -p data/coco
cd data/coco

# Training images (18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Validation images (1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

## Models

| Model | Type | NMS | Strengths |
|-------|------|-----|-----------|
| **YOLO26** | Single-stage | Built-in (NMS-free) | Fast inference, edge-optimized, strong small-object detection |
| **Faster R-CNN** | Two-stage | Explicit (torchvision) | Higher accuracy on complex scenes, better with overlapping objects |

Adding a new model requires implementing the `AbstractDetector` interface and registering it:

```python
@register_model("my_model")
class MyDetector(AbstractDetector):
    def forward(self, images): ...
    def compute_loss(self, predictions, targets): ...
    def postprocess(self, predictions, conf_threshold, iou_threshold): ...
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Deep learning framework | PyTorch |
| Training abstraction | PyTorch Lightning |
| Object detection (single-stage) | YOLO26 via Ultralytics |
| Object detection (two-stage) | Faster R-CNN via torchvision |
| Data augmentation | Albumentations |
| Configuration | OmegaConf + YAML |
| Dataset format | COCO 2017 |
| Evaluation metric | mAP via torchmetrics |
| Package management | uv |

## License

MIT
