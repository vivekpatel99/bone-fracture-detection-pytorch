import lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from hp_finding.tunable_model import Net


# TODO: take  out optimizer and scheduler from the constructor
class LungColonClassifier(pl.LightningModule):
    def __init__(
        self,
        model: Net,
        class_names: list[str],
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_classes = len(class_names)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

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
        return self.optimizer  # (self.model.parameters()), self.scheduler
