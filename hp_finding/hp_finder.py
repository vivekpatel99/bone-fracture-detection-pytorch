#  uv run  -m hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  python hp_finding/hp_finder.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers

import os
from pathlib import Path
from typing import Any

import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
if os.getenv("DATA_ROOT") is None:
    os.environ["DATA_ROOT"] = f"{root}/datasets"

from src import utils  # noqa: E402
from src.datamodule.lung_colon_cancer_datamodule import LungColonCancerDataModule  # noqa: E402
from src.utils.utils import extras, get_metric_value  # noqa: E402

log = utils.get_pylogger(__name__)
#  Register a resolver for torch dtypes
OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))
torch.set_float32_matmul_precision("medium")


def net_hp_finder(cfg: DictConfig) -> dict[str, Any]:
    # Load the configuration
    log.info(f"Configuration: {cfg}")
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating Net with experiment config <{cfg.model.net._target_}>")
    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[1].size)
    net = hydra.utils.instantiate(
        cfg.model.net,
        input_shape=input_shape,
        conv_layers=cfg.model.net.conv_layers,
        dropout_rate=cfg.model.net.dropout_rate,
        num_classes=cfg.model.net.num_classes,
        num_hidden_layers=cfg.model.net.num_hidden_layers,
    )
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, net=net)
    log.info(f"Instantiating data_module <{cfg.datamodule._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, subset_size=cfg.datamodule.subset_size, batch_size=cfg.datamodule.batch_size
    )
    log.info("Instantiating loggers...")
    # logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    # trainer = pl.Trainer(
    #     # logger=True,
    #     accelerator="gpu",
    #     devices=1,
    #     # callbacks=callbacks,
    #     max_epochs=10,
    #     enable_progress_bar=True,
    #     precision=32,
    #     log_every_n_steps=1,
    # )
    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }
    # if logger:
    #     log.info("Logging hyperparameters!")
    #     log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model, datamodule=data_module)

    train_metrics = trainer.callback_metrics
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict  # , object_dict


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)
    metric_dict = net_hp_finder(cfg)
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
