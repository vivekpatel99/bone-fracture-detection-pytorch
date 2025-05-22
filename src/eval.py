import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dagshub
import hydra
import pyrootutils  # pyright: ignore[reportMissingImports]
import torch
from omegaconf import DictConfig
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
from src.entity.best_cnn_model import BestCNNModel  # noqa: E402
from src.utils.utils import extras  # noqa: E402

if TYPE_CHECKING:
    import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")
log = utils.get_pylogger(__name__)


def evaluate_model(logger: MLFlowLogger, cfg: DictConfig, check_point_path: str) -> dict[str, Any]:
    """
    Evaluate a model with the given configuration.

    Args:
        logger (MLFlowLogger): The MLFlow logger to use for logging.
        cfg (DictConfig): The Hydra configuration dictionary containing all evaluation parameters.
        check_point_path (str): The path to the model checkpoint to load.

    Returns:
        dict[str, Any]: The evaluation metrics.

    """
    log.info("Instantiating model <%s>", cfg.model._target_)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    model.compile_model()  # (must match the state before loading the checkpoint)
    model.eval()

    log.info("Instantiating data_module <%s>", cfg.datamodule._target_)
    data_module: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    log.info("Starting testing with pre-loaded model state.")
    trainer.test(
        model=model,
        datamodule=data_module,
        ckpt_path=check_point_path,
    )  # ckpt_path is None as we've loaded it manually

    return trainer.callback_metrics


def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: D417
    """
    Evaluate the model with the given configuration.

    Args:
    - cfg (DictConfig): The Hydra configuration dictionary containing all evaluation parameters.

    Returns:
    - dict[str, Any]: The evaluation metrics.

    Raises:
    - FileNotFoundError: If the checkpoint file is not found or is not a file.

    """
    dagshub.init(
        repo_owner=os.environ.get("DAGSHUB_REPO_OWNER"),
        repo_name=os.environ.get("DAGSHUB_REPO_NAME"),
        mlflow=True,
    )

    log.info("Instantiating loggers...")
    ml_logger = hydra.utils.instantiate(cfg.logger, tracking_uri=os.environ.get("DAGSHUB_TRACKING_URI"))
    input_shape = [3, *hydra.utils.instantiate(cfg.data.train_preprocess_transforms[0].size)]
    cfg.model.net.input_shape = input_shape

    # --- get local model ---
    best_model_json_path = Path(cfg.paths.best_model_path)
    best_model_json_file = best_model_json_path / cfg.paths.best_model_json_name
    if not best_model_json_file.is_file():
        msg = "Best model json does not exist"
        raise FileNotFoundError(msg)

    with best_model_json_file.open("r") as f:
        local_best_model = BestCNNModel.model_validate(json.load(f))

    local_model_metrics = evaluate_model(logger=ml_logger, cfg=cfg, check_point_path=local_best_model.check_point_path)

    # --- get aws model ---
    secrets = AwsSecrets()
    s3_classifier = S3Classifier(
        bucket_name=secrets.bucket_name,
        cloud_model_key=cfg.paths.cloud_model_key,
        cloud_model_save_path=cfg.paths.cloud_model_save_path,
    )
    is_model_present = s3_classifier.is_model_present(model_name=cfg.paths.cloud_model_key)
    if is_model_present:
        log.info("cloud model %s is available", cfg.paths.cloud_model_key)

        s3_classifier.fetch_model_weights()

        dwnloaded_ckpt_path = Path(cfg.paths.cloud_model_save_path)
        if not dwnloaded_ckpt_path.is_file():
            mgs = f"checkpoint file is not downloaded at {dwnloaded_ckpt_path}"
            raise FileNotFoundError(mgs)
        cloud_model_metrics = evaluate_model(
            logger=ml_logger,
            cfg=cfg,
            check_point_path=cfg.paths.cloud_model_save_path,
        )

        log.info("local model metrics: %s", local_model_metrics)
        local_f1_score = local_model_metrics["test/f1_score"].item()
        cloud_f1_score = cloud_model_metrics["test/f1_score"].item()
        f1_score_diff = local_f1_score - cloud_f1_score
        log.info("local model f1_score: %s", local_f1_score)
        log.info("cloud model f1_score: %s", cloud_f1_score)
        log.info("f1_score_diff: %s", f1_score_diff)

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
    """
    Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    metric = evaluate(cfg)
    print(metric)


if __name__ == "__main__":
    main()
