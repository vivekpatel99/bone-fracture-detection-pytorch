from typing import Any

import pyrootutils
import pytorch_lightning as pl
import seaborn as sns
import torch
from matplotlib import figure
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MaxMetric, MeanMetric

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


class LungColonCancerClassifier(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        class_names: list[str],
        is_compile: bool,
        optimizer: torch.optim.Optimizer,
        lr: float | None = None,
        scheduler: Any | None = None,
    ):
        super().__init__()
        self.net = net
        # self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.is_compile = is_compile
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_f1_sc = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.valid_f1_sc = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.test_f1_sc = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")

        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

        # testing metrics
        self.test_targets = []
        self.test_preds = []
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)  # , num_labels=class_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def _common_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_acc.reset()
        self.valid_acc.reset()
        self.test_acc.reset()
        self.train_f1_sc.reset()
        self.valid_f1_sc.reset()
        self.test_f1_sc.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, preds, targets = self._common_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_f1_sc(preds, targets)
        self.log("train/f1_score", self.train_f1_sc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self._common_step(batch)
        self.valid_loss(loss)
        self.valid_acc(preds, targets)
        self.log("val/loss", self.valid_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.valid_f1_sc(preds, targets)
        self.log("val/f1_score", self.valid_f1_sc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # update best validation accuracy
        self.val_acc_best(self.valid_acc.compute())
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        self.val_f1_best(self.valid_f1_sc.compute())
        self.log("val/f1_score_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        loss, preds, targets = self._common_step(batch)
        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_preds.append(preds)
        self.test_targets.append(targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_f1_sc(preds, targets)
        self.log("test/f1_score", self.test_f1_sc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        targets = torch.cat(self.test_targets)
        preds = torch.cat(self.test_preds)
        # Confusion Matrix
        log.info("Computing confusion matrix...")
        cm = self.test_cm(preds, targets)
        self._log_confusion_matrix(cm)
        # Classification Report
        log.info("Computing classification report...")
        report = classification_report(
            targets.cpu(),
            preds.cpu(),
            output_dict=True,
            target_names=self.class_names,
            labels=list(range(self.num_classes)),
        )

        self._log_classification_report(report)

    def compile_model(self):
        log.info("Compiling model...")
        self.net = torch.compile(self.net)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.is_compile and stage == "fit":
            self.compile_model()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.lr or self.hparams.lr)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer)
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

    def _log_confusion_matrix(self, cm) -> None:
        fig, ax = plt.subplots(figsize=(10, 7))
        cm_np = cm.cpu().numpy()
        # Plot with Predicted labels on the Y-axis and True labels on the X-axis
        # This requires transposing the confusion matrix
        sns.heatmap(
            cm_np.T,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=self.class_names,  # Columns of cm_np.T are True Labels
            yticklabels=self.class_names,
        )  # Rows of cm_np.T are Predicted Labels
        ax.set_xlabel("True labels")  # X-axis now represents True labels
        ax.set_ylabel("Predicted labels")  # Y-axis now represents Predicted labels
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        self.logger.experiment.log_figure(run_id=self.logger.run_id, figure=fig, artifact_file="evaluation/confusion_matrix.png")
        plt.close(fig)

    def _log_classification_report(self, report) -> None:
        self.logger.experiment.log_dict(run_id=self.logger.run_id, dictionary=report, artifact_file="classification_report.json")
