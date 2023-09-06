import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
import pytorch_lightning as pl

from channel_attention.utils.interaug import interaug
from channel_attention.utils.lr_scheduler import linear_warmup_cosine_decay


class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,
            **kwargs
    ):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    def shared_step(self, batch, batch_idx, mode: str = "train"):
        if mode == "train" and self.hparams.get("interaug", False):
            x, y = interaug(batch)
        else:
            x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)