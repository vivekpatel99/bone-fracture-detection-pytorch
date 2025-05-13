import os
from pathlib import Path
from typing import Any

import dagshub
import hydra
import pyrootutils  # pyright: ignore[reportMissingImports]
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

torch.set_float32_matmul_precision("medium")
log = utils.get_pylogger(__name__)


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    assert cfg.ckpt_path, "Checkpoint path `cfg.ckpt_path` must be provided in the configuration."

    # Resolve the checkpoint path to an absolute path.
    # Assumes cfg.ckpt_path is relative to cfg.paths.root_dir if not already absolute.
    resolved_ckpt_path = Path(cfg.ckpt_path).resolve() if Path(cfg.ckpt_path).is_absolute() else (Path(cfg.paths.root_dir) / cfg.ckpt_path).resolve()

    if not resolved_ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found or is not a file: {resolved_ckpt_path}")

    dagshub.init(repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"), repo_name=os.environ.get("DAGSHUB_REPO_NAME"), mlflow=True)

    log.info("Instantiating loggers...")
    ml_logger = hydra.utils.instantiate(cfg.logger, tracking_uri=os.environ.get("DAGSHUB_TRACKING_URI"))

    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[1].size)
    OmegaConf.update(cfg.model.net, "input_shape", input_shape, merge=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model.compile_model()  # (must match the state before loading the checkpoint)

    model.eval()

    log.info(f"Instantiating data_module <{cfg.datamodule._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=ml_logger)
    object_dict = {
        "cfg": cfg,
        "datamodule": data_module,
        "model": model,
        "logger": ml_logger,
        "trainer": trainer,
    }

    if ml_logger:
        log.info("Logging hyperparameters!")
        ml_logger.log_hyperparams(object_dict)

    log.info("Starting testing with pre-loaded model state.")
    trainer.test(model=model, datamodule=data_module, ckpt_path=resolved_ckpt_path)  # ckpt_path is None as we've loaded it manually

    return trainer.callback_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric = evaluate(cfg)
    print(metric)


if __name__ == "__main__":
    main()
