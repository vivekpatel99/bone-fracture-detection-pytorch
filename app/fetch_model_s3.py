from pathlib import Path

import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402
from src.entity.aws_secrets import AwsSecrets  # noqa: E402
from src.entity.s3_classifier import S3Classifier  # noqa: E402

log = utils.get_pylogger(__name__)


def fetch_model_s3(cfg: DictConfig) -> Path:
    log.info("Fetching model from S3...")
    # --- get aws model ---
    dwnloaded_ckpt_path = Path(cfg.paths.cloud_model_save_path)

    result_dir = dwnloaded_ckpt_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"result_dir create: {result_dir}")

    secrets = AwsSecrets()
    s3_classifier = S3Classifier(
        bucket_name=secrets.bucket_name, cloud_model_key=cfg.paths.cloud_model_key, cloud_model_save_path=cfg.paths.cloud_model_save_path
    )
    is_model_present = s3_classifier.is_model_present(model_name=cfg.paths.cloud_model_key)
    if is_model_present:
        log.info(f"cloud model {cfg.paths.cloud_model_key} is available")

        s3_classifier.fetch_model_weights()

        assert dwnloaded_ckpt_path.is_file(), f"checkpoint file is not downloaded at {dwnloaded_ckpt_path}"
        return dwnloaded_ckpt_path
    else:
        log.error(f"cloud model {cfg.paths.cloud_model_key} is not available")
        raise FileNotFoundError
