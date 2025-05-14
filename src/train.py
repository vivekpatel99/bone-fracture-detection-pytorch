#  run Experiments - uv run  src/train.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  run Experiments - python src/train.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  cd logs/mlflow/ && mlflow ui
# uv run --env-file .env src/train.py

import json
import os
from pathlib import Path
from typing import Any, List

import dagshub
import hydra
import pyrootutils
import pytorch_lightning as pl
import torch
from lightning.pytorch.tuner import Tuner
from omegaconf import DictConfig, OmegaConf

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402
from src.entity.best_cnn_model import BestCNNModel, ModelConfig  # noqa: E402
from src.utils.instantiators import instantiate_callbacks  # noqa: E402
from src.utils.utils import extras, get_metric_value  # noqa: E402

torch.set_float32_matmul_precision("medium")
log = utils.get_pylogger(__name__)


def train(cfg: DictConfig) -> dict[str, Any]:
    log.debug(f"Configuration: {cfg}")

    # ---- Testing ----
    cfg.trainer.max_epochs = 2
    cfg.datamodule.subset_size = 0.1

    dagshub.init(repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"), repo_name=os.environ.get("DAGSHUB_REPO_NAME"), mlflow=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    ml_logger = hydra.utils.instantiate(cfg.logger, tracking_uri=os.environ.get("DAGSHUB_TRACKING_URI"))

    log.info(f"Instantiating Net with experiment config <{cfg.model.net._target_}>")
    params = {
        "subset_size": cfg.datamodule.subset_size,
        "batch_size": cfg.datamodule.batch_size,
        "conv_layers": cfg.model.net.get("conv_layers"),
        "dropout_rate": cfg.model.net.get("dropout_rate"),
        "num_classes": cfg.model.net.get("num_classes"),
        "num_hidden_layers": cfg.model.net.get("num_hidden_layers"),
    }
    ml_logger.log_hyperparams(params)

    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[0].size)
    # Add/update input_shape in the net's configuration (cfg.model.net).
    # This allows Hydra to use it when instantiating the net as part of the model.
    OmegaConf.update(cfg.model.net, "input_shape", input_shape, merge=True)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating data_module <{cfg.datamodule._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, subset_size=cfg.datamodule.subset_size, batch_size=cfg.datamodule.batch_size
    )
    log.info("Instantiating callbacks...")
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=ml_logger, callbacks=callbacks)
    tuner = Tuner(trainer)

    if cfg.get("batch_size_finder"):
        log.info("Starting batch size finder!")
        tuner.scale_batch_size(model=model, datamodule=data_module, mode="binsearch")
        log.info("new batch_size: {}".format(tuner.results))

    if cfg.get("lr_finder"):
        log.info("Starting lr finder!")
        lr_finder = tuner.lr_find(model=model, datamodule=data_module)
        new_lr = lr_finder.suggestion()
        log.info("new lr: {}".format(new_lr))
        model.lr = new_lr

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
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    log.info("Saving Model configuration!")
    model_config = ModelConfig(
        input_shape=input_shape,
        batch_size=cfg.datamodule.batch_size,
        conv_layers=cfg.model.net.get("conv_layers"),
        dropout_rate=cfg.model.net.get("dropout_rate"),
        num_classes=cfg.model.net.get("num_classes"),
        num_hidden_layers=cfg.model.net.get("num_hidden_layers"),
    )
    best_model = BestCNNModel(
        check_point_path=ckpt_path,
        f1_score=test_metrics["test/f1_score"],
        accuracy=test_metrics["test/acc"],
        loss=test_metrics["test/loss"],
        params=model_config,
    )
    json_schme = best_model.model_dump()

    best_model_json_path = Path(cfg.paths.best_model_path)
    best_model_json_path.mkdir(parents=True, exist_ok=True)
    best_model_json_file = best_model_json_path / cfg.paths.best_model_json_name
    with open(best_model_json_file, "w") as f:
        json.dump(json_schme, f, indent=4)
    log.info(f"Best model saved at: {best_model_json_file}")
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric_dict = train(cfg)
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
