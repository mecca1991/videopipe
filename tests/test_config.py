# tests/test_config.py


def test_load_config_returns_expected_keys():
    from visionpipe.config import load_config

    cfg = load_config()
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "data")
    assert hasattr(cfg, "augmentation")
    assert hasattr(cfg, "training")
    assert hasattr(cfg, "inference")


def test_load_config_model_defaults():
    from visionpipe.config import load_config

    cfg = load_config()
    assert cfg.model.name == "yolo26"
    assert cfg.model.num_classes == 91
    assert cfg.model.pretrained is True


def test_load_config_with_overrides():
    from visionpipe.config import load_config

    cfg = load_config(overrides={"model.name": "faster_rcnn", "training.max_epochs": 10})
    assert cfg.model.name == "faster_rcnn"
    assert cfg.training.max_epochs == 10


def test_load_config_from_custom_path(tmp_path):
    from visionpipe.config import load_config

    custom = tmp_path / "custom.yaml"
    custom.write_text("model:\n  name: faster_rcnn\n  num_classes: 20\n  pretrained: false\n")
    cfg = load_config(config_path=str(custom))
    assert cfg.model.name == "faster_rcnn"
    assert cfg.model.num_classes == 20
