from typing import Any

import pyrootutils
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy, F1Score, Metric

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


from src.model.net.custom_conv_net import CustomConvNet  # noqa: E402
from src.model.net.net_model import Net  # noqa: E402


class LungColonCancerClassifier(pl.LightningModule):
    def __init__(
        self,
        net: Net,
        class_names: list[str],
        criterion: torch.nn.Module,
        accuracy: Metric,
        f1_score: Metric,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer  # torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_classes = len(class_names)
        self.accuracy = accuracy  # Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = f1_score  # F1Score(task="multiclass", num_classes=self.num_classes)
        self.criterion = criterion()  # torch.nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def _common_step(self, batch, batch_idx) -> tuple[torch.Tensor, float, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss: torch.Tensor = self.criterion(y_hat, y)
        score = self.accuracy(y_hat, y)
        return loss, score, y_hat

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, score, y_hat = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", score, prog_bar=True)
        return loss

    # def on_train_epoch_end(self, outputs) -> None:
    #     avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     self.log("avg_train_loss", avg_loss, prog_bar=True)
    #     self.log("train_acc_epoch", self.accuracy.compute(), prog_bar=True)
    #     self.accuracy.reset()
    #     self.log("train_f1_epoch", self.f1_score.compute(), prog_bar=True)
    #     self.f1_score.reset()
    #     return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, score, y_hat = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", score, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, score, y_hat = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", score, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            # TODO: add support for schedulers
            scheduler = self.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
