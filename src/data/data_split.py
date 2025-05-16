from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
import pyrootutils
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from src import utils  # noqa: E402
from src.utils.download_kaggel_ds import download_kaggle_dataset, flatten_dataset_dir  # noqa: E402
from src.utils.utils import create_dataset_dir, remove_dir  # noqa: E402

log = utils.get_pylogger(__name__)


def get_dataset_df(dataset_dir: Path) -> pd.DataFrame:
    """
    Get the dataset dataframe from the dataset directory.
    """
    log.info("get_dataset_df running")
    datasets = defaultdict(list)
    for class_dir_path in list(dataset_dir.iterdir()):
        for file_path in class_dir_path.iterdir():
            datasets[class_dir_path.name].append(file_path)

    return pd.DataFrame(datasets)


@hydra.main(config_path=str(root / "configs"), config_name="train", version_base="1.3")
def data_spliter(cfg: DictConfig) -> None:
    DATASET_DIR = Path(root) / cfg.data.dataset_dir
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Dataset directory: {DATASET_DIR}")
    download_kaggle_dataset(dowload_dataset_name=cfg.data.dataset_download_name, download_dir=DATASET_DIR)
    flatten_dataset_dir(dataset_dir=DATASET_DIR)

    TRAIN_IMAGE_DIR = Path(root) / cfg.paths.train_raw_dir
    remove_dir(TRAIN_IMAGE_DIR)

    VALID_IMAGE_DIR = Path(root) / cfg.paths.valid_raw_dir
    remove_dir(VALID_IMAGE_DIR)

    TEST_DIR = Path(root) / cfg.paths.test_raw_dir
    remove_dir(TEST_DIR)

    VALID_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    dataset_pth = DATASET_DIR / cfg.data.datasets_name
    assert dataset_pth.exists(), "Dataset does not exist"
    dataset_df = get_dataset_df(DATASET_DIR / cfg.data.datasets_name)

    log.info(f"Dataset shape: {dataset_df.shape}")
    log.info(f"split dataset into train, validation: {cfg.data.valid_ds_size}, and test: {cfg.data.test_ds_size} sets")
    rest_ds, test_ds = train_test_split(
        dataset_df,
        test_size=cfg.data.test_ds_size,
        random_state=42,
        shuffle=True,
    )
    log.info(f"Rest dataset shape: {rest_ds.shape}, Test dataset shape: {test_ds.shape}")
    train_ds, val_ds = train_test_split(
        rest_ds,
        test_size=cfg.data.valid_ds_size,
        random_state=42,
        shuffle=True,
    )
    log.info(f"Train dataset shape: {train_ds.shape}, Validation dataset shape: {val_ds.shape}")

    train_raw_dir = Path(root) / cfg.paths.train_raw_dir
    create_dataset_dir(dataset_split_dir=train_raw_dir, dataset_df=train_ds)  # dataset_split_dir: Path, dataset_df: pd.DataFrame)
    log.info(f"Train dataset split completed {train_raw_dir}")

    valid_raw_dir = Path(root) / cfg.paths.valid_raw_dir
    create_dataset_dir(dataset_split_dir=valid_raw_dir, dataset_df=val_ds)
    log.info(f"Validation dataset split completed {valid_raw_dir}")

    test_raw_dir = Path(root) / cfg.paths.test_raw_dir
    create_dataset_dir(dataset_split_dir=test_raw_dir, dataset_df=test_ds)
    log.info(f"Test dataset split completed {test_raw_dir}")

    log.info("Data split completed!!")


if __name__ == "__main__":
    data_spliter()
