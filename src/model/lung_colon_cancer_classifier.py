import os
from typing import Any

import lightning as pl
import pyrootutils
import torch
import torch.nn.functional as F
from fastapi import params
from torchmetrics import Accuracy, F1Score, Metric

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from src.model.net.custom_conv_net import CustomConvNet  # noqa: E402
from src.model.net.net_model import Net  # noqa: E402


# TODO: take  out optimizer and scheduler from the constructor
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
        self.model = net
        self.optimizer = optimizer  # torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_classes = len(class_names)
        self.accuracy = accuracy  # Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = f1_score  # F1Score(task="multiclass", num_classes=self.num_classes)
        self.criterion = criterion  # torch.nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

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
        return {"optimizer": optimizer}


if __name__ == "__main__":
    class_names = [
        "colon-adenocarcinoma",
        "colon-benign-tissue",
        "lung-adenocarcinoma",
        "lung-benign-tissue",
        "lung-squamous-cell-carcinoma",
    ]
    net = CustomConvNet(
        input_shape=(3, 320, 320),
        conv_layers=5,
        dropout_rate=0.1,
        num_classes=5,
        num_hidden_layers=5,
    )
    # # Create the optimizer
    # optimizer = torch.optim.AdamW

    # Create the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = LungColonCancerClassifier(
        net=net,
        class_names=class_names,
        criterion=torch.nn.CrossEntropyLoss(),
        accuracy=Accuracy(task="multiclass", num_classes=len(class_names)),
        f1_score=F1Score(task="multiclass", num_classes=len(class_names)),
        optimizer=torch.optim.AdamW,
        scheduler=None,  # torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    )
