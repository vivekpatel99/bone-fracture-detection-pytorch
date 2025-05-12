import os
from typing import Any, List

import dagshub
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

torch.set_float32_matmul_precision("medium")
log = utils.get_pylogger(__name__)


def evaluate(cfg: DictConfig) -> dict[str, float]: ...


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None: ...


if __name__ == "__main__":
    main()
