import torch
import lightning as L
from omegaconf import DictConfig
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class DetectionModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        model_name = cfg.model.name
        if model_name == "faster_rcnn":
            if cfg.model.pretrained:
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            else:
                weights = None
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        else:
            raise ValueError(
                f"Lightning training module supports 'faster_rcnn'. "
                f"For '{model_name}', use the model's native trainer."
            )

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
            optimizer = torch.optim.SGD(
                self.parameters(), lr=cfg.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.max_epochs
            )
            return [optimizer], [scheduler]

        return optimizer
