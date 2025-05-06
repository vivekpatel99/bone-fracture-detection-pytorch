import os
from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from src import utils  # noqa: E402

log = utils.get_pylogger(__name__)


@hydra.main(config_path=str(root / "configs"), config_name="train", version_base="1.3")
def data_preprocessor(cfg: DictConfig) -> None:
    # DATASET_DIR = Path(root) / cfg.data.dataset_dir / cfg.data.dataset_name
    ...


if __name__ == "__main__":
    data_preprocessor()
