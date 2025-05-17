import json
import os
from pathlib import Path
from threading import local
from typing import Any

import dagshub
import hydra
import pyrootutils  # pyright: ignore[reportMissingImports]
import pytorch_lightning as pl
import torch
from joblib import PrintTime
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.mlflow import MLFlowLogger

from entity.s3_classifier import S3Classifier

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402
from src.entity.aws_secrets import AwsSecrets  # noqa: E402
from src.entity.best_cnn_model import BestCNNModel, ModelConfig  # noqa: E402
from src.utils.utils import extras  # noqa: E402

torch.set_float32_matmul_precision("medium")
log = utils.get_pylogger(__name__)


def evaluate_model(logger: MLFlowLogger, cfg: DictConfig, check_point_path: str) -> dict[str, Any]:
    """
    Evaluate the model with the given configuration.

    Args:
    - cfg (DictConfig): The Hydra configuration dictionary.

    Returns:
    - dict[str, Any]: The evaluation metrics.
    """
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model.compile_model()  # (must match the state before loading the checkpoint)
    model.eval()

    log.info(f"Instantiating data_module <{cfg.datamodule._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("Starting testing with pre-loaded model state.")
    trainer.test(model=model, datamodule=data_module, ckpt_path=check_point_path)  # ckpt_path is None as we've loaded it manually

    return trainer.callback_metrics


def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Evaluate the model with the given configuration.

    Args:
    - cfg (DictConfig): The Hydra configuration dictionary.

    Returns:
    - dict[str, Any]: The evaluation metrics.

    Raises:
    - FileNotFoundError: If the checkpoint file is not found or is not a file.
    """
    dagshub.init(repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"), repo_name=os.environ.get("DAGSHUB_REPO_NAME"), mlflow=True)

    log.info("Instantiating loggers...")
    ml_logger = hydra.utils.instantiate(cfg.logger, tracking_uri=os.environ.get("DAGSHUB_TRACKING_URI"))
    input_shape = [3] + hydra.utils.instantiate(cfg.data.train_preprocess_transforms[0].size)
    cfg.model.net.input_shape = input_shape

    # --- get local model ---
    best_model_json_path = Path(cfg.paths.best_model_path)
    best_model_json_file = best_model_json_path / cfg.paths.best_model_json_name
    assert best_model_json_file.is_file(), "Best model json does not exist"

    with open(str(best_model_json_file)) as f:
        local_best_model = BestCNNModel.model_validate(json.load(f))

    local_model_metrics = evaluate_model(logger=ml_logger, cfg=cfg, check_point_path=local_best_model.check_point_path)

    # --- get aws model ---
    secrets = AwsSecrets()
    s3_classifier = S3Classifier(
        bucket_name=secrets.bucket_name, cloud_model_key=cfg.paths.cloud_model_key, cloud_model_save_path=cfg.paths.cloud_model_save_path
    )
    is_model_present = s3_classifier.is_model_present(model_name=cfg.paths.cloud_model_key)
    if is_model_present:
        log.info(f"cloud model {cfg.paths.cloud_model_key} is available")

        s3_classifier.fetch_model_weights()

        dwnloaded_ckpt_path = Path(cfg.paths.cloud_model_save_path)
        assert dwnloaded_ckpt_path.is_file(), f"checkpoint file is not downloaded at {dwnloaded_ckpt_path}"
        cloud_model_metrics = evaluate_model(logger=ml_logger, cfg=cfg, check_point_path=cfg.paths.cloud_model_save_path)

        log.info(f"local model metrics: {local_model_metrics}")
        local_f1_score = local_model_metrics["test/f1_score"].item()
        cloud_f1_score = cloud_model_metrics["test/f1_score"].item()
        f1_score_diff = local_f1_score - cloud_f1_score
        log.info(f"local model f1_score: {local_f1_score}")
        log.info(f"cloud model f1_score: {cloud_f1_score}")
        log.info(f"f1_score_diff: {f1_score_diff}")

    else:
        log.info("cloud model is not available")
        f1_score_diff = 0
        cloud_model_metrics = {}

    if (f1_score_diff > cfg.model_perf_diff_threshold) or not is_model_present:
        # local model is better push to aws
        log.info("local model is better than cloud model")
        s3_classifier.upload_model(from_file=local_best_model.check_point_path)
        log.info("local model pushed to aws")

    return local_model_metrics, cloud_model_metrics


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
