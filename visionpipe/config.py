# visionpipe/config.py
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(
    config_path: str | None = None,
    overrides: dict | None = None,
) -> DictConfig:
    """
    Load the configuration from a YAML file and apply any overrides.
    Args:
        config_path (str | None): Path to the YAML configuration file. If None, uses the default path.
        overrides (dict | None): A dictionary of configuration keys and values to override.
    Returns:
        DictConfig: The loaded and updated configuration.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    cfg = OmegaConf.load(path)

    if overrides:
        for key, value in overrides.items():
            OmegaConf.update(cfg, key, value)

    return cfg
