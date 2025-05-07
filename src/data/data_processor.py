from pathlib import Path

import hydra
import pyrootutils
from omegaconf import DictConfig
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

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
    """
    Script to preprocess the data. The script takes the following steps
    - Removes the existing preprocessed data
    - Instantiates the preprocessing transforms according to the config
    - Creates the dataset using the root path and the instantiated transforms
    - Saves the preprocessed dataset to the appropriate directory
    """
    # --- Train dataset ---
    train_processed_dir = Path(root) / cfg.paths.train_processed_dir
    utils.remove_dir(train_processed_dir)
    log.info("Old train preprocessed dir removed")
    log.info(f"Instantiating train transforms <{cfg.data.train_preprocess_transforms._target_}>")
    train_transforms = hydra.utils.instantiate(cfg.data.train_preprocess_transforms)
    train_processed_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ImageFolder(
        root=str(Path(root) / cfg.paths.train_raw_dir),
        transform=v2.Compose(train_transforms),
        target_transform=None,
        is_valid_file=None,
    )
    log.info("Train dataset created")
    utils.save_images_from_datasets(train_processed_dir, train_dataset)
    log.info("Processed train dataset saved")

    # --- Validation dataset ---
    valid_processed_dir = Path(root) / cfg.paths.valid_processed_dir
    utils.remove_dir(valid_processed_dir)
    log.info("Old valid preprocessed dir removed")
    log.info(f"Instantiating valiidation transforms <{cfg.data.valid_preprocess_transforms._target_}>")
    valid_transforms = hydra.utils.instantiate(cfg.data.valid_preprocess_transforms)
    valid_processed_dir.mkdir(parents=True, exist_ok=True)
    valid_dataset = ImageFolder(
        root=str(Path(root) / cfg.paths.valid_raw_dir),
        transform=v2.Compose(valid_transforms),
        target_transform=None,
        is_valid_file=None,
    )
    log.info("Validation dataset created")
    utils.save_images_from_datasets(valid_processed_dir, valid_dataset)
    log.info("Processed valid dataset saved")

    # --- Test dataset ---
    test_processed_dir = Path(root) / cfg.paths.test_processed_dir
    utils.remove_dir(test_processed_dir)
    log.info("Old test preprocessed dir removed")

    test_processed_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = ImageFolder(
        root=str(Path(root) / cfg.paths.test_raw_dir),
        transform=v2.Compose(valid_transforms),
        target_transform=None,
        is_valid_file=None,
    )
    log.info("Test dataset created")
    utils.save_images_from_datasets(test_processed_dir, test_dataset)
    log.info("Processed test dataset saved")


if __name__ == "__main__":
    data_preprocessor()
