#  uv run  -m hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  python hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers

from pathlib import Path
from typing import Any

import hydra
import mlflow
import optuna
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import Logger
from sklearn import metrics
from torchmetrics import Accuracy, F1Score

from utils.utils import extras

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from hp_finding.tunable_model import Net  # noqa: E402
from src import utils  # noqa: E402
from src.datamodule.lung_colon_cancer_datamodule import LungColonCancerDataModule  # noqa: E402
from src.model.lung_colon_cancer_classifier import LungColonCancerClassifier  # noqa: E402
from src.model.net.custom_conv_net import CustomConvNet  # noqa: E402

log = utils.get_pylogger(__name__)
#  Register a resolver for torch dtypes
OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))
torch.set_float32_matmul_precision("medium")


def train(cfg: DictConfig) -> dict[str, Any]:
    # Load the configuration
    log.info(f"Configuration: {cfg}")
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)
    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[1].size)

    # Create the model
    net = hydra.utils.instantiate(
        cfg.model.net,
        input_shape=input_shape,
        conv_layers=cfg.model.net.conv_layers,
        dropout_rate=cfg.model.net.dropout_rate,
        num_classes=cfg.model.net.num_classes,
        num_hidden_layers=cfg.model.net.num_hidden_layers,
    )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, net=net)

    # data_module: pl.LightningDataModule = hydra.utils.instantiate(
    #     cfg.datamodule, subset_size=cfg.datamodule.subset_size, batch_size=cfg.datamodule.batch_size
    # )
    data_module = LungColonCancerDataModule(
        train_processed_dir=str(Path(root) / cfg.paths.train_processed_dir),
        valid_processed_dir=str(Path(root) / cfg.paths.valid_processed_dir),
        test_processed_dir=str(Path(root) / cfg.paths.test_processed_dir),
        augmentations=cfg.datamodule.augmentations,
        valid_transforms=cfg.datamodule.valid_transforms,
        num_workers=cfg.datamodule.num_workers,
        pin_memory=cfg.datamodule.pin_memory,
        persistent_workers=cfg.datamodule.persistent_workers,
        batch_size=cfg.datamodule.batch_size,
        subset_size=0.1,
    )
    trainer = pl.Trainer(
        # logger=True,
        accelerator="gpu",
        devices=1,
        # callbacks=callbacks,
        max_epochs=10,
        enable_progress_bar=True,
        precision=32,
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=data_module)
    train_metrics = trainer.callback_metrics
    return train_metrics["train_acc"].item()  # , train_metrics["train_loss"].item()


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    # train the model
    metric_dict, _ = train(cfg)


if __name__ == "__main__":
    main()
