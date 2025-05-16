import os
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from src import utils

log = utils.get_pylogger(__name__)


def download_kaggle_dataset(dowload_dataset_name: str, download_dir: Path):
    """
    Downloads a Kaggle dataset into the specified directory.

    Args:
        dataset_name (str): The Kaggle dataset identifier, e.g., 'andrewmvd/lung-and-colon-cancer-histopathological-images'
        download_dir (str): The directory where the dataset will be downloaded and extracted.
    """
    log.info("Downloading Kaggle dataset... ")
    # Ensure the download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download and unzip the dataset
    api.dataset_download_files(dowload_dataset_name, path=str(download_dir), unzip=True)
    log.info(f"Dataset '{dowload_dataset_name}' downloaded and extracted to '{download_dir}'.")


def move_class_dir_to_root(dataset_dir: Path) -> Path:
    """
    Move all class directories to the root of the dataset directory.

    For example, if the dataset directory has the following structure:
        dataset_dir/general_class_name/
            class1/
                class1_image1.jpg
                class1_image2.jpg
            class2/
                class2_image1.jpg
                class2_image2.jpg
    after calling this function, the directory structure will be:
        dataset_dir/
            class1/
                class1_image1.jpg
                class1_image2.jpg
            class2/
                class2_image1.jpg
                class2_image2.jpg
    This is useful for creating a dataset directory with a flat structure.
    """
    log.info("Moving class directories to root")
    new_root_dir = Path()
    for main_class_path in dataset_dir.iterdir():
        new_root_dir = dataset_dir / main_class_path.name
        for class_path in main_class_path.iterdir():
            for class_name in class_path.iterdir():
                new_image_dir = new_root_dir / class_name.name
                new_image_dir.mkdir(exist_ok=True)
                for current_image_path in class_name.iterdir():
                    new_image_path = new_image_dir / current_image_path.name
                    shutil.move(current_image_path, new_image_path)
    log.info("Class directories moved to root")
    return new_root_dir


def remove_empty_dirs(root_dir) -> None:
    """
    Remove all empty directories under root_dir (including nested ones).
    """
    log.info("Removing empty directories")
    for path in sorted(Path(root_dir).rglob("*"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            log.info(f"Deleting empty directory: {path}")
            path.rmdir()
    log.info("Empty directories removed")


def flatten_dataset_dir(dataset_dir: Path) -> Path:
    new_ds_path = move_class_dir_to_root(dataset_dir)
    remove_empty_dirs(dataset_dir)
    return new_ds_path
