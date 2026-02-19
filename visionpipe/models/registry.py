from omegaconf import DictConfig
from visionpipe.models.base import AbstractDetector

MODEL_REGISTRY: dict[str, type[AbstractDetector]] = {}


def register_model(name: str):
    def decorator(cls: type[AbstractDetector]) -> type[AbstractDetector]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(cfg: DictConfig) -> AbstractDetector:
    name = cfg.model.name
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) or "(none)"
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](cfg)
