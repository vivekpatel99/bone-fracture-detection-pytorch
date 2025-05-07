import collections
import os
import shutil
from pathlib import Path

import pandas as pd
from torchvision import io
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src import utils

log = utils.get_pylogger(__name__)


def create_dataset_dir(dataset_split_dir: Path, dataset_df: pd.DataFrame) -> None:
    dataset_split_dir.mkdir(parents=True, exist_ok=True)
    for dir_name in dataset_df.columns:
        os.makedirs(dataset_split_dir / dir_name, exist_ok=True)
        for file_path in dataset_df[dir_name]:
            file_name = file_path.name
            new_file_path = dataset_split_dir / dir_name / file_name
            if not new_file_path.exists():
                os.symlink(file_path, new_file_path)


def remove_dir(dir_path: Path) -> None:
    if dir_path.exists():
        log.info(f"Removing directory: {dir_path}")
        try:
            shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error: {e.strerror}")
    else:
        log.info(f"Directory does not exist: {dir_path}")


def save_images_from_datasets(img_save_path: Path, dataset: ImageFolder):
    class_image_counters = collections.defaultdict(int)

    img_save_path.mkdir(parents=True, exist_ok=True)
    for image_data_tuple in tqdm(iterable=dataset, desc="Saving images", unit="image"):
        img = image_data_tuple[0]
        label = dataset.classes[image_data_tuple[1]]
        log.debug(f"Label: {label}, Image size: {img.size()[-2:]}")

        current_count = class_image_counters[label]

        img_path = img_save_path / label
        log.debug(f"Image path: {img_path}")
        img_path.mkdir(parents=True, exist_ok=True)
        io.write_jpeg(input=img, filename=str(img_path / f"{label}_{current_count}.jpeg"), quality=100)
        class_image_counters[label] += 1
