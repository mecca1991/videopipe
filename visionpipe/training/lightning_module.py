import lightning as L
import torch
from omegaconf import DictConfig

import visionpipe.models.faster_rcnn  # noqa: F401  # triggers @register_model
import visionpipe.models.yolo26  # noqa: F401  # triggers @register_model
from visionpipe.models.registry import build_model


class DetectionModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.detector = build_model(cfg)
        self.model = self.detector.model  # underlying nn.Module for Lightning parameter tracking

    def training_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        image_list = [images[i] for i in range(images.shape[0])]
        loss_dict = self.model(image_list, targets)
        total_loss = sum(loss_dict.values())

        for name, value in loss_dict.items():
            self.log(f"train_{name}", value, prog_bar=False)
        self.log("train_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        self.model.train()
        image_list = [images[i] for i in range(images.shape[0])]
        loss_dict = self.model(image_list, targets)
        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        cfg = self.cfg.training

        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=cfg.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
            return [optimizer], [scheduler]

        return optimizer
