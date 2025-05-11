#  uv run  src/train.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  python src/train.py hparams_search=cnn_layers_search_optuna experiment=find_cnn_layers
#  mlflow ui --backend-store-uri logs/mlflow/mlruns/
from typing import Any, List

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
from src import utils  # noqa: E402
from src.utils.instantiators import instantiate_callbacks  # noqa: E402
from src.utils.utils import extras, get_metric_value  # noqa: E402

log = utils.get_pylogger(__name__)


def net_hp_finder(cfg: DictConfig) -> dict[str, Any]:
    log.debug(f"Configuration: {cfg}")
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating loggers...")
    ml_logger = hydra.utils.instantiate(cfg.logger)

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

    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[1].size)
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
    # # TODO: lr finder
    # if cfg.get("lr_finder"):
    #     log.info("Starting lr finder!")
    #     lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)

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

    return metric_dict


@hydra.main(version_base="1.3", config_path=str(root / "configs"), config_name="train.yaml")
def main(cfg: DictConfig):
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric_dict = net_hp_finder(cfg)
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    #  Register a resolver for torch dtypes
    # OmegaConf.register_new_resolver("torch_dtype", lambda name: getattr(torch, name))
    torch.set_float32_matmul_precision("medium")
    main()
